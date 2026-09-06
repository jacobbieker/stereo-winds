"""Tests for the Himawari AHI icechunk reader."""

import datetime as dt

import numpy as np
import pytest
import xarray as xr

from stereo_winds.readers.himawari import (
    Himawari,
    _ABI_TO_AHI,
    _BAND_RESOLUTION,
    _resolve_band,
)


# ── Offline unit tests (no network) ──────────────────────────────────


class TestResolveBand:
    def test_native_ahi_band(self):
        assert _resolve_band("B14") == "B14"
        assert _resolve_band("B01") == "B01"

    def test_abi_to_ahi_translation(self):
        assert _resolve_band("C14") == "B14"
        assert _resolve_band("C08") == "B08"

    def test_all_abi_bands_resolve(self):
        for abi, ahi in _ABI_TO_AHI.items():
            assert _resolve_band(abi) == ahi

    def test_unknown_band_raises(self):
        with pytest.raises(ValueError, match="Unknown band"):
            _resolve_band("X99")


class TestBandResolution:
    def test_vis_500m(self):
        for b in ("B01", "B02", "B03"):
            assert _BAND_RESOLUTION[b] == "500m"

    def test_vis_1km(self):
        assert _BAND_RESOLUTION["B04"] == "1000m"

    def test_ir_2km(self):
        for b in ("B07", "B08", "B14", "B16"):
            assert _BAND_RESOLUTION[b] == "2000m"

    def test_all_bands_covered(self):
        assert len(_BAND_RESOLUTION) == 16


class TestConstructor:
    def test_defaults(self):
        h = Himawari()
        assert h.satellite == "himawari9"
        assert h.bands == ["B14"]

    def test_custom_satellite(self):
        h = Himawari(satellite="himawari8")
        assert h.satellite == "himawari8"

    def test_abi_band_translated(self):
        h = Himawari(bands=["C14"])
        assert h.bands == ["B14"]

    def test_unknown_satellite_raises(self):
        with pytest.raises(ValueError, match="Unknown satellite"):
            Himawari(satellite="goes19")

    def test_store_prefix(self):
        h = Himawari()
        assert h._store_prefix("2000m") == "geo/himawari_2000m.icechunk"
        assert h._store_prefix("500m") == "geo/himawari_500m.icechunk"

    def test_repr(self):
        h = Himawari(satellite="himawari9", bands=["B14"])
        assert "himawari9" in repr(h)
        assert "B14" in repr(h)


class TestCoordSynthesis:
    """Verify fallback coordinate generation when the store lacks coords."""

    def test_synthesised_coords_symmetric(self):
        h = Himawari()
        x = h._get_coord(xr.Dataset(), "x", 5500)
        y = h._get_coord(xr.Dataset(), "y", 5500)
        # Symmetric about zero
        np.testing.assert_allclose(x[0], -x[-1], atol=1e-10)
        np.testing.assert_allclose(y[0], -y[-1], atol=1e-10)

    def test_synthesised_scale_2km(self):
        h = Himawari()
        x = h._get_coord(xr.Dataset(), "x", 5500)
        dx = float(x[1] - x[0])
        np.testing.assert_allclose(dx, 5.6e-05, rtol=1e-6)

    def test_synthesised_scale_1km(self):
        h = Himawari()
        x = h._get_coord(xr.Dataset(), "x", 11000)
        dx = float(x[1] - x[0])
        np.testing.assert_allclose(dx, 2.8e-05, rtol=1e-6)

    def test_synthesised_scale_500m(self):
        h = Himawari()
        x = h._get_coord(xr.Dataset(), "x", 22000)
        dx = float(x[1] - x[0])
        np.testing.assert_allclose(dx, 1.4e-05, rtol=1e-6)

    def test_uses_store_coords_when_present(self):
        h = Himawari()
        x_expected = np.linspace(-0.15, 0.15, 5500)
        ds = xr.Dataset(coords={"x": ("x", x_expected)})
        x = h._get_coord(ds, "x", 5500)
        np.testing.assert_array_equal(x, x_expected)


class TestExtractRadiance:
    """Radiance extraction from various store layouts."""

    def test_band_as_variable(self):
        h = Himawari()
        data = np.random.rand(100, 100).astype(np.float32)
        snap = xr.Dataset({"B14": (("y", "x"), data)})
        result = h._extract_radiance(snap, "B14")
        np.testing.assert_array_equal(result, data)

    def test_rad_variable_2d(self):
        h = Himawari()
        data = np.random.rand(100, 100).astype(np.float32)
        snap = xr.Dataset({"Rad": (("y", "x"), data)})
        result = h._extract_radiance(snap, "B14")
        np.testing.assert_array_equal(result, data)

    def test_fallback_first_spatial_var(self):
        h = Himawari()
        data = np.random.rand(100, 100).astype(np.float32)
        snap = xr.Dataset({"brightness": (("y", "x"), data)})
        result = h._extract_radiance(snap, "B14")
        np.testing.assert_array_equal(result, data)

    def test_missing_band_raises(self):
        h = Himawari()
        snap = xr.Dataset({"scalar": 42.0})
        with pytest.raises(KeyError, match="Cannot find radiance"):
            h._extract_radiance(snap, "B14")


class TestBuildCoords:
    def test_flips_descending_y(self):
        h = Himawari()
        rad = np.arange(6).reshape(3, 2).astype(np.float32)
        # y descending (north->south) in radians
        y = np.array([0.01, 0.0, -0.01])
        x = np.array([-0.01, 0.01])
        ds = xr.Dataset(coords={"x": ("x", x), "y": ("y", y)})
        x_m, y_m, rad_out = h._build_coords(ds, rad)
        assert y_m[0] < y_m[-1], "y should be ascending after flip"
        # Rad should be flipped vertically
        np.testing.assert_array_equal(rad_out[0], rad[2])

    def test_radian_to_meter_conversion(self):
        h = Himawari()
        rad = np.ones((3, 3), dtype=np.float32)
        x = np.array([-0.01, 0.0, 0.01])
        y = np.array([-0.01, 0.0, 0.01])
        ds = xr.Dataset(coords={"x": ("x", x), "y": ("y", y)})
        x_m, y_m, _ = h._build_coords(ds, rad)
        # Should be scaled by satellite height
        from stereo_winds.readers.himawari import _SAT_HEIGHT
        np.testing.assert_allclose(x_m[2], 0.01 * _SAT_HEIGHT)


# ── Smoke tests (require network access to source.coop) ─────────────


@pytest.mark.network
class TestHimawariSmoke:
    """Live smoke tests reading from icechunk stores at source.coop."""

    def test_load_b14_2km(self):
        h = Himawari(satellite="himawari9", bands=["B14"])
        ds = h.data_at_time(dt.datetime(2024, 1, 15, 3, 0))
        rad = ds["Rad"]
        assert rad.dims == ("time", "band", "y", "x")
        assert rad.shape[0] == 1 and rad.shape[1] == 1
        assert rad.dtype == np.float32
        # y should be ascending
        y = ds["y"].values
        assert y[-1] > y[0]
        # orbital_parameters present
        orb = rad.attrs["orbital_parameters"]
        assert "projection_altitude" in orb
        assert "satellite_nominal_longitude" in orb
        assert orb["satellite_nominal_longitude"] == pytest.approx(140.7)
        # sweep axis
        assert ds.attrs["sweep_angle_axis"] == "y"

    def test_abi_band_name_accepted(self):
        h = Himawari(satellite="himawari9", bands=["C14"])
        ds = h.data_at_time(dt.datetime(2024, 1, 15, 3, 0))
        assert ds["Rad"].shape[2] > 0

    def test_data_has_valid_values(self):
        h = Himawari(satellite="himawari9", bands=["B14"])
        ds = h.data_at_time(dt.datetime(2024, 1, 15, 3, 0))
        data = ds["Rad"].values[0, 0]
        # Should have some non-NaN values
        assert np.isfinite(data).sum() > 0
