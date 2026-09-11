"""Tests for the MSG SEVIRI icechunk reader."""

import datetime as dt

import numpy as np
import pytest
import xarray as xr

from stereo_winds.config import SATELLITE_CONFIGS
from stereo_winds.readers._geos_meta import scene_ellipsoid
from stereo_winds.readers._satpy_s3 import SceneNotInStore
from stereo_winds.readers.msg import (
    ABI_TO_SEVIRI,
    ABI_WITHOUT_SEVIRI,
    MSG,
    SCAN_INTERVAL_MINUTES,
    _BAND_RESOLUTION,
    _resolve_band,
)


# ── Offline unit tests (no network) ──────────────────────────────────


class TestResolveBand:
    def test_native_seviri_band(self):
        assert _resolve_band("IR_108") == "IR_108"
        assert _resolve_band("WV_062") == "WV_062"

    def test_abi_to_seviri_translation(self):
        assert _resolve_band("C14") == "IR_108"
        assert _resolve_band("C08") == "WV_062"
        assert _resolve_band("C16") == "IR_134"

    def test_every_mapped_abi_band_resolves(self):
        for abi, seviri in ABI_TO_SEVIRI.items():
            assert _resolve_band(abi) == seviri
            assert seviri in _BAND_RESOLUTION, f"{abi} -> unknown {seviri}"

    def test_bands_seviri_does_not_have(self):
        """SEVIRI has no 1.4 µm cirrus or 2.2 µm channel."""
        for band in ABI_WITHOUT_SEVIRI:
            with pytest.raises(ValueError, match="no SEVIRI counterpart"):
                _resolve_band(band)

    def test_unknown_band_raises(self):
        with pytest.raises(ValueError, match="Unknown band"):
            _resolve_band("C99")


class TestBandMapping:
    """Spot-check the mapping against the published band centres."""

    def test_water_vapour_channels(self):
        # C08 6.19 -> WV_062 6.25; C09 6.95 and C10 7.34 -> WV_073 7.35
        assert ABI_TO_SEVIRI["C08"] == "WV_062"
        assert ABI_TO_SEVIRI["C09"] == "WV_073"
        assert ABI_TO_SEVIRI["C10"] == "WV_073"

    def test_window_channels(self):
        # C13 10.35 and C14 11.2 both sit nearest IR_108; C15 -> IR_120
        assert ABI_TO_SEVIRI["C13"] == "IR_108"
        assert ABI_TO_SEVIRI["C14"] == "IR_108"
        assert ABI_TO_SEVIRI["C15"] == "IR_120"

    def test_fewer_channels_means_collisions(self):
        """Eleven SEVIRI channels cannot separate sixteen ABI bands."""
        assert len(set(ABI_TO_SEVIRI.values())) < len(ABI_TO_SEVIRI)

    def test_every_student_band_is_available(self):
        """The default flow/rad bands must all map, or MSG is useless here."""
        from stereo_winds.student_dataset import (
            DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS,
        )
        for band in set(DEFAULT_FLOW_BANDS) | set(DEFAULT_RAD_BANDS):
            assert band in ABI_TO_SEVIRI, f"{band} has no SEVIRI equivalent"


class TestConstruction:
    def test_defaults(self):
        src = MSG()
        assert src.satellite == "msg-iodc"
        assert src.bands == ["IR_108"]

    def test_abi_band_translated(self):
        assert MSG(bands=["C14"]).bands == ["IR_108"]

    def test_unknown_satellite_raises(self):
        with pytest.raises(ValueError, match="Unknown satellite"):
            MSG(satellite="msg-0deg")

    def test_store_prefix(self):
        assert MSG()._store_prefix("3000m") == "geo/iodc_3000m_test.icechunk"

    def test_repr(self):
        assert "msg-iodc" in repr(MSG())


class TestCadence:
    def test_seviri_repeats_every_15_minutes(self):
        assert SCAN_INTERVAL_MINUTES == 15

    def test_ring_script_uses_it(self):
        import importlib.util
        import sys
        from pathlib import Path

        base = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "ring_msg", base / "scripts" / "infer_student_global_ring.py")
        ring = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ring
        spec.loader.exec_module(ring)
        assert ring.scan_interval("msg-iodc") == 15
        assert ring.scan_interval("goes19") == 10
        assert "msg-iodc" in ring.RING_SATELLITES
        # C04/C06 must be reported unavailable so they are zero-filled.
        assert ring._band_available("msg-iodc", "C14")
        assert not ring._band_available("msg-iodc", "C04")
        assert not ring._band_available("msg-iodc", "C06")


class TestSelectTime:
    def _store(self, times):
        return xr.Dataset(
            {"IR_108": (("time", "y", "x"),
                        np.zeros((len(times), 2, 2), np.float32))},
            coords={"time": [np.datetime64(t, "ns") for t in times]},
        )

    def test_nearby_scan_returned(self):
        ds = self._store([dt.datetime(2025, 7, 15, 12, 15)])
        snap = MSG()._select_time(ds, dt.datetime(2025, 7, 15, 12, 20))
        assert snap["time"].values == np.datetime64("2025-07-15T12:15", "ns")

    def test_distant_scan_raises(self):
        ds = self._store([dt.datetime(2025, 7, 15, 12, 15)])
        with pytest.raises(SceneNotInStore):
            MSG()._select_time(ds, dt.datetime(2026, 8, 1, 0, 0))


class TestExtractRadiance:
    def test_band_as_variable(self):
        data = np.arange(4, dtype=np.float32).reshape(2, 2)
        snap = xr.Dataset({"IR_108": (("y", "x"), data)})
        assert np.array_equal(MSG()._extract_radiance(snap, "IR_108"), data)

    def test_missing_band_raises(self):
        snap = xr.Dataset({"WV_062": (("y", "x"), np.zeros((2, 2), np.float32))})
        with pytest.raises(KeyError):
            MSG()._extract_radiance(snap, "IR_108")


class TestBuildCoords:
    def test_uses_store_coords(self):
        x = np.linspace(-5.5e6, 5.5e6, 4)
        ds = xr.Dataset(coords={"x_geostationary": ("x_geostationary", x)})
        assert np.allclose(MSG()._get_coord(ds, "x", 4), x)

    def test_descending_axes_are_normalised(self):
        x = np.linspace(5.5e6, -5.5e6, 3)
        y = np.linspace(5.5e6, -5.5e6, 3)
        ds = xr.Dataset(coords={"x_geostationary": ("x_geostationary", x),
                                "y_geostationary": ("y_geostationary", y)})
        rad = np.arange(9, dtype=np.float32).reshape(3, 3)
        x_m, y_m, out = MSG()._build_coords(ds, rad)
        assert x_m[0] < x_m[-1] and y_m[0] < y_m[-1]
        assert np.array_equal(out, rad[::-1, ::-1])

    def test_radian_coords_converted_to_metres(self):
        rad_coords = np.linspace(-0.15, 0.15, 3)
        ds = xr.Dataset(coords={"x_geostationary": ("x_geostationary", rad_coords),
                                "y_geostationary": ("y_geostationary", rad_coords)})
        x_m, _, _ = MSG()._build_coords(
            ds, np.zeros((3, 3), np.float32), sat_height=35785831.0)
        assert abs(x_m[-1]) > 1e6


class TestEllipsoid:
    def test_read_from_area_definition(self):
        area = ("{'msg_seviri_iodc_3km': {'projection': {'proj': 'geos', "
                "'lon_0': 45.5, 'h': 35785831, 'a': 6378169, "
                "'rf': 295.488065897014}}}")
        a, b = scene_ellipsoid(xr.Dataset(attrs={"area": area}),
                               fallback_semi_major=1.0, fallback_semi_minor=2.0)
        assert a == pytest.approx(6378169.0)
        assert b == pytest.approx(6356583.8, abs=0.1)

    def test_differs_from_grs80(self):
        """The reason this is read at all: MSG is not on GRS80."""
        area = "{'a': {'projection': {'a': 6378169, 'rf': 295.488065897014}}}"
        _, b = scene_ellipsoid(xr.Dataset(attrs={"area": area}),
                               fallback_semi_major=1.0, fallback_semi_minor=2.0)
        assert abs(b - 6356752.31414) > 100.0

    def test_explicit_semi_minor_wins(self):
        area = "{'a': {'projection': {'a': 6378137, 'b': 6356752.0}}}"
        a, b = scene_ellipsoid(xr.Dataset(attrs={"area": area}),
                               fallback_semi_major=1.0, fallback_semi_minor=2.0)
        assert (a, b) == pytest.approx((6378137.0, 6356752.0))

    def test_fallback_when_absent(self):
        a, b = scene_ellipsoid(xr.Dataset(), fallback_semi_major=1.0,
                               fallback_semi_minor=2.0)
        assert (a, b) == (1.0, 2.0)


class TestConfigPreset:
    def test_registered(self):
        assert "msg-iodc" in SATELLITE_CONFIGS

    def test_projection_matches_the_service(self):
        cfg = SATELLITE_CONFIGS["msg-iodc"]
        assert cfg.sub_lon_deg == pytest.approx(45.5)
        assert cfg.sweep == "y"          # only GOES ABI sweeps x
        assert (cfg.n_rows, cfg.n_cols) == (3712, 3712)

    def test_uses_the_msg_ellipsoid(self):
        cfg = SATELLITE_CONFIGS["msg-iodc"]
        assert cfg.semi_major_m == pytest.approx(6378169.0)
        assert cfg.semi_minor_m == pytest.approx(6356583.8, abs=0.1)

    def test_pixel_scale_is_3km(self):
        cfg = SATELLITE_CONFIGS["msg-iodc"]
        assert cfg.scale_x * cfg.satellite_height_m == pytest.approx(3000.4, abs=1)


# ── Smoke tests (require network access to source.coop) ─────────────


@pytest.mark.network
class TestMSGSmoke:
    """Live reads from the IODC icechunk store at source.coop."""

    def test_load_ir108(self):
        ds = MSG(bands=["IR_108"]).data_at_time(dt.datetime(2025, 7, 15, 12, 0))
        rad = ds["Rad"]
        assert rad.dims == ("time", "band", "y", "x")
        assert rad.shape == (1, 1, 3712, 3712)
        assert ds["y"].values[-1] > ds["y"].values[0]
        assert ds["x"].values[-1] > ds["x"].values[0]
        assert ds.attrs["sweep_angle_axis"] == "y"

    def test_projection_read_from_store(self):
        ds = MSG(bands=["C14"]).data_at_time(dt.datetime(2025, 7, 15, 12, 0))
        orb = ds["Rad"].attrs["orbital_parameters"]
        assert orb["projection_longitude"] == pytest.approx(45.5, abs=0.1)
        assert orb["projection_altitude"] == pytest.approx(35785831, rel=1e-4)
        assert ds.attrs["ellipsoid"]["semi_minor_m"] == pytest.approx(
            6356583.8, abs=1.0)

    def test_data_has_valid_values(self):
        ds = MSG(bands=["C14"]).data_at_time(dt.datetime(2025, 7, 15, 12, 0))
        data = ds["Rad"].values[0, 0]
        assert np.isfinite(data).sum() > 0
