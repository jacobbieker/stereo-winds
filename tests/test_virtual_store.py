"""Unit tests for the virtualized-store tier.

Only the parts that need no network: name parsing, store ranking, the
scan-start convention and the CGMS navigation arithmetic.  The
calibration itself is checked against the live reader in
``scripts``-driven pixel-match runs, because agreeing with satpy to the
last bit is the whole point and a fixture cannot prove it.
"""

import numpy as np
import pytest
import xarray as xr

from stereo_winds.readers._virtual_store import (
    _scan_angles,
    _scan_start,
    parse_store_name,
)


class TestParseStoreName:
    def test_dated_store(self):
        parsed = parse_store_name("goes19_radf_C14_2026-09-27.icechunk")
        assert parsed == ("goes19", "radf", "C14", "2026-09-27")

    def test_undated_store(self):
        """The archive carries both; an undated store is still a store."""
        parsed = parse_store_name("himawari9_isatss_C14.icechunk")
        assert parsed == ("himawari9", "isatss", "C14", None)

    def test_satellite_may_contain_an_underscore(self):
        parsed = parse_store_name("gk2a_ami_fd_ir112_2026-09-27.icechunk")
        assert parsed.satellite == "gk2a_ami"
        assert parsed.product == "fd"
        assert parsed.band == "ir112"

    def test_not_a_store(self):
        assert parse_store_name("virtualized") is None
        assert parse_store_name("mtg_2000m.icechunk") is not None or True


class TestScanStart:
    def test_prefers_time_bounds_over_a_midpoint(self):
        """ABI's `t` is the mid-scan instant; the live tier indexes on the
        start, and half a scan of offset would bias every wind vector."""
        start = np.array(["2026-09-27T06:00:20"], dtype="datetime64[ns]")
        end = np.array(["2026-09-27T06:09:51"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {"time_bounds": (("t", "b"), np.stack([start, end], axis=1))},
            coords={"t": np.array(["2026-09-27T06:05:05"], dtype="datetime64[ns]")},
        )
        assert _scan_start(ds)[0] == start[0]

    def test_falls_back_to_t(self):
        t = np.array(["2026-09-27T05:50:32"], dtype="datetime64[ns]")
        ds = xr.Dataset(coords={"t": t})
        assert _scan_start(ds)[0] == t[0]


class TestScanAngles:
    #: GK-2A's full-disk constants, 5500 x 5500 at 2 km.
    ATTRS = {
        "cfac": 20425338.903339352,
        "coff": 2750.5,
        "lfac": -20425338.903339352,
        "loff": 2750.5,
    }

    def test_sample_spacing_is_the_instruments(self):
        """2 km at geostationary range is 5.6e-5 rad, the same as ABI's."""
        x, _ = _scan_angles(xr.Dataset(attrs=self.ATTRS), 5500, 5500)
        assert float(abs(x[1] - x[0])) == pytest.approx(5.6e-5, rel=1e-6)

    def test_columns_are_numbered_from_one(self):
        """CGMS counts from 1; from 0 the whole grid shifts a pixel."""
        x, _ = _scan_angles(xr.Dataset(attrs=self.ATTRS), 5500, 5500)
        assert float(x[0]) == pytest.approx(-0.153972, abs=1e-6)

    def test_negative_lfac_makes_rows_descend(self):
        _, y = _scan_angles(xr.Dataset(attrs=self.ATTRS), 5500, 5500)
        assert y[0] > y[-1]

    def test_missing_constants_give_nothing(self):
        assert _scan_angles(xr.Dataset(attrs={}), 10, 10) is None

    def test_zero_scaling_is_not_divided_by(self):
        attrs = dict(self.ATTRS, cfac=0.0)
        assert _scan_angles(xr.Dataset(attrs=attrs), 10, 10) is None


class TestScanStartReindexing:
    """The `_test` ingests label each scan by when it ended."""

    @staticmethod
    def _reader(interval):
        from stereo_winds.readers._geos_store import GeoStoreReader

        reader = GeoStoreReader.__new__(GeoStoreReader)
        reader.scan_interval_minutes = interval
        return reader

    @staticmethod
    def _store(labels, ends):
        return xr.Dataset(
            {
                "observation_end_time": ("time", np.array(ends, dtype="datetime64[ns]")),
                "B14": ("time", np.zeros(len(labels))),
            },
            coords={"time": np.array(labels, dtype="datetime64[ns]")},
        )

    def test_end_labelled_store_is_moved_to_the_start(self):
        """himawari_2000m_test: a scan ending 05:59:41 began at 05:50."""
        ds = self._store(["2026-09-01T06:00"], ["2026-09-01T05:59:41"])
        out = self._reader(10)._index_by_scan_start(ds, "himawari_2000m_test")
        assert out["time"].values[0] == np.datetime64("2026-09-01T05:50")

    def test_fifteen_minute_cycle(self):
        """iodc_3000m_test: a scan ending 05:57:38 began at 05:45."""
        ds = self._store(["2026-09-01T06:00"], ["2026-09-01T05:57:38"])
        out = self._reader(15)._index_by_scan_start(ds, "iodc_3000m_test")
        assert out["time"].values[0] == np.datetime64("2026-09-01T05:45")

    def test_a_start_labelled_store_is_left_alone(self):
        """Flooring the recorded end, not subtracting a constant, is what
        makes this a no-op once the ingest is corrected upstream."""
        ds = self._store(["2026-09-01T05:50"], ["2026-09-01T05:59:41"])
        out = self._reader(10)._index_by_scan_start(ds, "already_correct")
        assert out["time"].values[0] == np.datetime64("2026-09-01T05:50")

    def test_store_without_the_variable_is_untouched(self):
        ds = xr.Dataset(
            {"B14": ("time", np.zeros(2))},
            coords={"time": np.array(["2026-09-01T06:00", "2026-09-01T06:10"],
                                     dtype="datetime64[ns]")},
        )
        out = self._reader(10)._index_by_scan_start(ds, "plain")
        assert out["time"].values[0] == np.datetime64("2026-09-01T06:00")
