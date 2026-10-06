"""Tests for the standalone GOES reader (virtualized tier first, public S3 second)."""

import datetime as dt

import numpy as np
import pytest

from stereo_winds.readers.goes import GOES, PRODUCT_TIMESTEPS


class FakeFS:
    """Records glob patterns; returns a single fake match."""

    def __init__(self):
        self.patterns = []

    def glob(self, pattern):
        self.patterns.append(pattern)
        return ["fake-key.nc"]


class TestSnapTime:
    def test_radf_10min(self):
        g = GOES(satellite="goes19", product="ABI-L1b-RadF")
        t = g._snap_time(dt.datetime(2026, 8, 11, 12, 34, 56))
        assert t == dt.datetime(2026, 8, 11, 12, 30)

    def test_radc_5min(self):
        g = GOES(satellite="goes19", product="ABI-L1b-RadC")
        assert g.step == 5
        assert (g._snap_time(dt.datetime(2026, 8, 11, 12, 34))
                == dt.datetime(2026, 8, 11, 12, 30))
        assert (g._snap_time(dt.datetime(2026, 8, 11, 12, 36))
                == dt.datetime(2026, 8, 11, 12, 35))

    def test_radm_1min(self):
        g = GOES(satellite="goes19", product="ABI-L1b-RadM")
        t = g._snap_time(dt.datetime(2026, 8, 11, 12, 34, 56))
        assert t == dt.datetime(2026, 8, 11, 12, 34)

    def test_product_timesteps(self):
        assert PRODUCT_TIMESTEPS["ABI-L1b-RadF"] == 10
        assert PRODUCT_TIMESTEPS["ABI-L1b-RadC"] == 5
        assert PRODUCT_TIMESTEPS["ABI-L1b-RadM"] == 1


class TestFindKey:
    """S3 glob pattern construction (RadC scans start +1 min past nominal)."""

    def _pattern(self, satellite, product, t, band="C14"):
        g = GOES(satellite=satellite, product=product, bands=[band])
        g._fs = FakeFS()
        g._find_key(t, band)
        assert len(g._fs.patterns) == 1
        return g._fs.patterns[0]

    def test_radf_pattern(self):
        p = self._pattern("goes19", "ABI-L1b-RadF",
                          dt.datetime(2026, 8, 11, 20, 0))
        assert p.startswith("noaa-goes19/ABI-L1b-RadF/2026/223/20/")
        assert "_G19_s20262232000" in p

    def test_radc_pattern_plus_one_minute(self):
        p = self._pattern("goes18", "ABI-L1b-RadC",
                          dt.datetime(2026, 8, 11, 20, 5))
        assert p.startswith("noaa-goes18/ABI-L1b-RadC/2026/223/20/")
        # nominal 20:05 slot -> scan start 20:06
        assert "_G18_s20262232006" in p

    def test_radc_snaps_then_offsets(self):
        # 20:07 snaps to the 20:05 slot, whose scan starts at 20:06
        p = self._pattern("goes19", "ABI-L1b-RadC",
                          dt.datetime(2026, 8, 11, 20, 7))
        assert "_G19_s20262232006" in p

    def test_no_match_raises(self):
        g = GOES(satellite="goes19", product="ABI-L1b-RadC", bands=["C14"])

        class EmptyFS:
            def glob(self, pattern):
                return []

        g._fs = EmptyFS()
        with pytest.raises(FileNotFoundError):
            g._find_key(dt.datetime(2026, 8, 11, 20, 5), "C14")


class TestRadToBT:
    def test_planck_inversion(self):
        """BT conversion inverts the forward Planck radiance calculation."""
        import numpy as np
        from stereo_winds.data_loading import _rad_to_bt

        # GOES-19 C14 planck constants (from a real L1b file)
        attrs = {"planck_fk1": 8510.22, "planck_fk2": 1286.67,
                 "planck_bc1": 0.18516, "planck_bc2": 0.99938}
        bt_true = np.array([[220.0, 250.0], [280.0, 300.0]])
        # forward: BT -> effective T -> radiance
        t_eff = attrs["planck_bc1"] + attrs["planck_bc2"] * bt_true
        rad = attrs["planck_fk1"] / (np.exp(attrs["planck_fk2"] / t_eff) - 1.0)
        bt = _rad_to_bt(rad.astype(np.float32), attrs)
        np.testing.assert_allclose(bt, bt_true, atol=1e-3)

    def test_missing_planck_raises(self):
        import numpy as np
        import pytest
        from stereo_winds.data_loading import _rad_to_bt

        with pytest.raises(ValueError, match="Planck"):
            _rad_to_bt(np.ones((2, 2), np.float32), {})


def _abi_store(starts, ny=4, nx=5):
    """A tiny ABI-shaped virtualized store: ``t`` is the mid-scan
    instant and ``time_bounds`` the real start, as in the archive."""
    import xarray as xr

    starts = np.asarray(starts, dtype="datetime64[ns]")
    nt = starts.size
    rad = np.arange(nt * ny * nx, dtype=np.float32).reshape(nt, ny, nx)
    planck = {
        name: ("t", np.full(nt, value))
        for name, value in (
            ("planck_fk1", 8510.22),
            ("planck_fk2", 1286.67),
            ("planck_bc1", 0.18516),
            ("planck_bc2", 0.99938),
        )
    }
    bounds = np.stack([starts, starts + np.timedelta64(10, "m")], axis=1)
    return xr.Dataset(
        {
            "Rad": (("t", "y", "x"), rad),
            "time_bounds": (("t", "number_of_time_bounds"), bounds),
            "goes_imager_projection": (
                (),
                0,
                {
                    "perspective_point_height": 35786023.0,
                    "longitude_of_projection_origin": -75.2,
                    "sweep_angle_axis": "x",
                },
            ),
            **planck,
        },
        coords={
            "t": starts + np.timedelta64(5, "m"),
            "x": ("x", np.linspace(-0.1, 0.1, nx)),
            "y": ("y", np.linspace(0.1, -0.1, ny)),
        },
    )


def _starts(*times):
    return np.array(times, dtype="datetime64[ns]")


@pytest.fixture(autouse=True)
def _fresh_virtual_cache():
    from stereo_winds.readers.goes import clear_virtual_cache

    clear_virtual_cache()
    yield
    clear_virtual_cache()


PREFIX = "geo/virtualized/goes19_radf_C14.icechunk"
T = dt.datetime(2026, 9, 1, 6, 0)


def _virtual(monkeypatch, store, prefixes=(PREFIX,), opens=None):
    """Stand in for source.coop: list ``prefixes``, open ``store`` (or
    raise it).  Returns what ``virtual_store_for`` was asked."""
    from stereo_winds.readers import _virtual_store as virtual

    seen = {}

    def _for(bucket, endpoint, satellite, band, when=None):
        seen["when"] = when
        return list(prefixes)

    def _open(bucket, endpoint, prefix):
        if opens is not None:
            opens.append(prefix)
        if isinstance(store, Exception):
            raise store
        return store

    monkeypatch.setattr(virtual, "virtual_store_for", _for)
    monkeypatch.setattr(virtual, "open_virtual_dataset", _open)
    return seen


@pytest.fixture
def reader(tmp_path):
    """A GOES reader whose S3 path serves a one-scan file from the local
    cache and records each time it is asked."""
    raw = _abi_store(_starts("2026-09-01T06:00:20")).isel(t=0)
    raw = raw.drop_vars(["time_bounds", "t"])
    (tmp_path / "goes19").mkdir()
    raw.to_netcdf(tmp_path / "goes19" / "fake.nc")

    def make():
        g = GOES(satellite="goes19", bands=["C14"], cache_dir=tmp_path, cache_retention=None)
        g.s3_calls = []

        def _find_key(t, band):
            g.s3_calls.append((t, band))
            return "noaa-goes19/ABI-L1b-RadF/fake.nc"

        g._find_key = _find_key
        return g

    return make


class TestVirtualFirst:
    """The virtualized tier is asked first; the public bucket answers
    whatever it cannot."""

    def test_virtual_serves_without_touching_s3(self, monkeypatch, reader):
        store = _abi_store(_starts("2026-09-01T05:50:20", "2026-09-01T06:00:20"))
        seen = _virtual(monkeypatch, store)
        g = reader()
        ds = g.data_at_time(T)
        assert g.s3_calls == []
        # The scan that started in the slot, oriented south->north.
        np.testing.assert_array_equal(ds["Rad"].values[0, 0], store["Rad"].values[1, ::-1])
        assert ds["Rad"].attrs["planck_fk1"] == pytest.approx(8510.22)
        # Asked for the stores whose era covers the time.
        assert seen["when"] == T

    def test_virtual_scene_carries_its_own_scan_times(self, monkeypatch, reader):
        """The per-pixel scan-time model reads these; the store's global
        attributes describe the whole store, not this scan."""
        from stereo_winds.data_loading import _scene_time_bounds

        store = _abi_store(_starts("2026-09-01T05:50:20", "2026-09-01T06:00:20"))
        store.attrs["time_coverage_start"] = "2026-07-10T00:00:20.5Z"
        _virtual(monkeypatch, store)
        start, end = _scene_time_bounds(reader().data_at_time(T))
        assert start.replace(tzinfo=None) == dt.datetime(2026, 9, 1, 6, 0, 20)
        assert end.replace(tzinfo=None) == dt.datetime(2026, 9, 1, 6, 10, 20)

    def test_no_virtual_store_falls_back_to_s3(self, monkeypatch, reader):
        _virtual(monkeypatch, None, prefixes=())
        g = reader()
        ds = g.data_at_time(T)
        assert len(g.s3_calls) == 1
        assert ds["Rad"].dims == ("time", "band", "y", "x")

    def test_a_scan_outside_the_tolerance_is_refused(self, monkeypatch, reader):
        """A neighbouring scan is not a substitute: the retrieval takes
        the interval between frames as its baseline."""
        _virtual(monkeypatch, _abi_store(_starts("2026-09-01T09:00:20")))
        g = reader()
        g.data_at_time(T)
        assert len(g.s3_calls) == 1

    def test_virtual_open_raising_falls_back_to_s3(self, monkeypatch, reader):
        _virtual(monkeypatch, OSError("source.coop unreachable"))
        g = reader()
        ds = g.data_at_time(T)
        assert len(g.s3_calls) == 1
        assert ds["Rad"].shape[-2:] == (4, 5)

    def test_virtual_listing_raising_falls_back_to_s3(self, monkeypatch, reader):
        from stereo_winds.readers import _virtual_store as virtual

        def _explode(*a, **k):
            raise OSError("source.coop unreachable")

        monkeypatch.setattr(virtual, "virtual_store_for", _explode)
        g = reader()
        g.data_at_time(T)
        assert len(g.s3_calls) == 1

    def test_s3_error_survives_when_neither_has_the_scan(self, monkeypatch):
        """'No file on S3' is the useful message, not whatever the
        virtualized tier failed with."""
        _virtual(monkeypatch, OSError("source.coop unreachable"))
        g = GOES(satellite="goes19", bands=["C14"])

        def _boom(t, band):
            raise FileNotFoundError("No ABI file on S3: (simulated)")

        g._find_key = _boom
        with pytest.raises(FileNotFoundError, match="No ABI file on S3"):
            g.data_at_time(T)

    def test_an_opened_store_is_reused(self, monkeypatch, reader):
        """Across times and across readers: the time axis is read once."""
        store = _abi_store(_starts("2026-09-01T06:00:20", "2026-09-01T06:10:20"))
        opens = []
        _virtual(monkeypatch, store, opens=opens)
        first, second = reader(), reader()
        first.data_at_time(T)
        first.data_at_time(T + dt.timedelta(minutes=10))
        second.data_at_time(T)
        assert opens == [PREFIX]
        assert first.s3_calls == [] and second.s3_calls == []

    def test_a_stale_store_is_reopened_for_a_time_past_its_end(self, monkeypatch, reader):
        """The undated store is still appended to; a long-running process
        must see the new scans rather than going to S3 for them forever."""
        from stereo_winds.readers import goes as goes_mod

        later = T + dt.timedelta(minutes=10)
        opens = []
        _virtual(monkeypatch, _abi_store(_starts("2026-09-01T06:00:20")), opens=opens)
        g = reader()
        g.data_at_time(T)
        # Fresh handle: a time past its end goes to S3 without a reopen.
        g.data_at_time(later)
        assert opens == [PREFIX] and len(g.s3_calls) == 1

        grown = _abi_store(_starts("2026-09-01T06:00:20", "2026-09-01T06:10:20"))
        _virtual(monkeypatch, grown, opens=opens)
        monkeypatch.setattr(goes_mod, "VIRTUAL_REFRESH_SECONDS", -1.0)
        g.data_at_time(later)
        assert opens == [PREFIX, PREFIX] and len(g.s3_calls) == 1

    def test_only_stores_of_the_readers_product(self, monkeypatch):
        """A CONUS request must not be answered from a full-disk store
        whose scan happens to start inside the tolerance."""
        opens = []
        _virtual(monkeypatch, _abi_store(_starts("2026-09-01T06:00:20")), opens=opens)
        g = GOES(satellite="goes19", product="ABI-L1b-RadC", bands=["C14"])
        assert g._virtual_scene(T, "C14") is None
        assert opens == []


class TestVirtualTimes:
    def test_scan_starts_within_the_window(self, monkeypatch):
        starts = _starts("2026-09-01T05:50:20", "2026-09-01T06:00:20", "2026-09-01T06:10:20")
        _virtual(monkeypatch, _abi_store(starts))
        g = GOES(satellite="goes19", bands=["C14"])
        got = g.virtual_times("C14", T, T + dt.timedelta(minutes=5))
        np.testing.assert_array_equal(got, starts[1:2])

    def test_unreachable_tier_is_empty(self, monkeypatch):
        from stereo_winds.readers import _virtual_store as virtual

        def _explode(*a, **k):
            raise OSError("down")

        monkeypatch.setattr(virtual, "virtual_store_for", _explode)
        g = GOES(satellite="goes19", bands=["C14"])
        assert g.virtual_times("C14", T, T).size == 0


class TestGoesAvailability:
    """A time held only by the virtualized tier still counts as available."""

    def test_union_of_virtual_and_s3(self, monkeypatch):
        import importlib.util
        import sys
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "scripts" / "infer_student_global_ring.py"
        spec = importlib.util.spec_from_file_location("infer_student_global_ring", path)
        ring = sys.modules.get(spec.name)
        if ring is None:
            ring = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = ring
            spec.loader.exec_module(ring)

        virtual_only, both, s3_only = _starts(
            "2026-09-01T06:00:20", "2026-09-01T06:10:20", "2026-09-01T06:20:20"
        )
        monkeypatch.setattr(
            ring, "_goes_virtual_times", lambda *a, **k: np.array([virtual_only, both])
        )
        monkeypatch.setattr(ring, "_goes_available_times", lambda *a, **k: np.array([both, s3_only]))
        got = ring.satellite_available_times("goes19", "C14", T, T + dt.timedelta(hours=1))
        np.testing.assert_array_equal(got, [virtual_only, both, s3_only])


@pytest.mark.network
class TestLiveVirtualFirst:
    """Against source.coop and the NOAA bucket, anonymously."""

    def test_virtual_then_s3_fallback(self, tmp_path, caplog):
        import logging
        import time

        from stereo_winds.readers import goes as goes_mod

        caplog.set_level(logging.INFO, logger=goes_mod.__name__)
        g = GOES(satellite="goes19", bands=["C13"], cache_dir=tmp_path)

        # Inside the live store's range.
        t_virtual = dt.datetime(2026, 9, 15, 12, 0)
        t0 = time.perf_counter()
        a = g.data_at_time(t_virtual)
        cold = time.perf_counter() - t0
        assert "read from virtualized store" in caplog.text
        assert "read from S3" not in caplog.text

        t0 = time.perf_counter()
        g.data_at_time(t_virtual + dt.timedelta(minutes=10))
        warm = time.perf_counter() - t0

        # Same time with the virtualized lookup forced to fail.
        caplog.clear()
        goes_mod.clear_virtual_cache()
        original = g._virtual_prefixes
        g._virtual_prefixes = lambda *a, **k: (_ for _ in ()).throw(OSError("forced"))
        try:
            t0 = time.perf_counter()
            b = g.data_at_time(t_virtual)
            s3 = time.perf_counter() - t0
        finally:
            g._virtual_prefixes = original
        assert "read from S3" in caplog.text
        assert a["Rad"].shape == b["Rad"].shape
        # The store keeps the scan angles as float32; sub-metre apart.
        np.testing.assert_allclose(a["x"].values, b["x"].values, atol=1.0)
        np.testing.assert_allclose(a["y"].values, b["y"].values, atol=1.0)
        np.testing.assert_array_equal(
            np.isnan(a["Rad"].values), np.isnan(b["Rad"].values)
        )
        from stereo_winds.data_loading import _scene_time_bounds

        for got, want in zip(_scene_time_bounds(a), _scene_time_bounds(b)):
            assert abs(got - want) < dt.timedelta(seconds=1)
        # Same scan; the two paths unpack the counts in different precision.
        np.testing.assert_allclose(a["Rad"].values, b["Rad"].values, rtol=1e-5, equal_nan=True)
        # A slot past the end of the live store: S3 answers it.
        caplog.clear()
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        last = g.virtual_times("C13", t_virtual, now).max()
        t_past = last.astype("datetime64[m]").astype(dt.datetime) + dt.timedelta(minutes=30)
        if t_past < now - dt.timedelta(minutes=30):
            t0 = time.perf_counter()
            c = g.data_at_time(t_past)
            past = time.perf_counter() - t0
            assert "read from S3" in caplog.text
            assert c["Rad"].shape == a["Rad"].shape
            print(f"\npast the live store ({t_past}): S3 {past:.1f}s")
        print(f"\nGOES-19 C13 virtual cold {cold:.1f}s, warm {warm:.1f}s, S3 {s3:.1f}s")
