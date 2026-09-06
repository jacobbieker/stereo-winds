"""Tests for the GK-2A AMI icechunk reader."""

import datetime as dt

import numpy as np
import pytest
import xarray as xr

from stereo_winds.readers.gk2a import (
    GK2A,
    _ABI_TO_AMI,
    _BAND_RESOLUTION,
    _resolve_band,
)


# ── Offline unit tests (no network) ──────────────────────────────────


class TestResolveBand:
    def test_native_ami_band(self):
        assert _resolve_band("IR112") == "IR112"
        assert _resolve_band("VI006") == "VI006"

    def test_abi_to_ami_translation(self):
        assert _resolve_band("C14") == "IR112"
        assert _resolve_band("C08") == "WV063"

    def test_all_abi_bands_resolve(self):
        for abi, ami in _ABI_TO_AMI.items():
            assert _resolve_band(abi) == ami

    def test_unknown_band_raises(self):
        with pytest.raises(ValueError, match="Unknown band"):
            _resolve_band("X99")

    def test_c06_not_mapped(self):
        """C06 (2.25 um) has no AMI equivalent."""
        with pytest.raises(ValueError, match="Unknown band"):
            _resolve_band("C06")


class TestBandResolution:
    def test_vis_500m(self):
        for b in ("VI004", "VI005", "VI006"):
            assert _BAND_RESOLUTION[b] == "500m"

    def test_vis_1km(self):
        for b in ("VI008", "NR013", "NR016"):
            assert _BAND_RESOLUTION[b] == "1000m"

    def test_ir_2km(self):
        for b in ("SW038", "WV063", "IR087", "IR112", "IR133"):
            assert _BAND_RESOLUTION[b] == "2000m"

    def test_all_bands_covered(self):
        assert len(_BAND_RESOLUTION) == 16


class TestConstructor:
    def test_defaults(self):
        g = GK2A()
        assert g.satellite == "gk2a"
        assert g.bands == ["IR112"]

    def test_abi_band_translated(self):
        g = GK2A(bands=["C14"])
        assert g.bands == ["IR112"]

    def test_unknown_satellite_raises(self):
        with pytest.raises(ValueError, match="Unknown satellite"):
            GK2A(satellite="goes19")

    def test_store_prefix(self):
        g = GK2A()
        assert g._store_prefix("2000m") == "geo/gk2a_2000m.icechunk"
        assert g._store_prefix("500m") == "geo/gk2a_500m.icechunk"

    def test_repr(self):
        g = GK2A(bands=["IR112"])
        assert "gk2a" in repr(g)
        assert "IR112" in repr(g)


class TestCoordSynthesis:
    def test_synthesised_coords_symmetric(self):
        g = GK2A()
        x = g._get_coord(xr.Dataset(), "x", 5500)
        y = g._get_coord(xr.Dataset(), "y", 5500)
        np.testing.assert_allclose(x[0], -x[-1], atol=1e-10)
        np.testing.assert_allclose(y[0], -y[-1], atol=1e-10)

    def test_synthesised_scale_2km(self):
        g = GK2A()
        x = g._get_coord(xr.Dataset(), "x", 5500)
        dx = float(x[1] - x[0])
        np.testing.assert_allclose(dx, 5.6e-05, rtol=1e-6)

    def test_synthesised_scale_1km(self):
        g = GK2A()
        x = g._get_coord(xr.Dataset(), "x", 11000)
        dx = float(x[1] - x[0])
        np.testing.assert_allclose(dx, 2.8e-05, rtol=1e-6)

    def test_uses_store_coords_when_present(self):
        g = GK2A()
        x_expected = np.linspace(-0.15, 0.15, 5500)
        ds = xr.Dataset(coords={"x": ("x", x_expected)})
        x = g._get_coord(ds, "x", 5500)
        np.testing.assert_array_equal(x, x_expected)


class TestExtractRadiance:
    def test_band_as_variable(self):
        g = GK2A()
        data = np.random.rand(100, 100).astype(np.float32)
        snap = xr.Dataset({"IR112": (("y", "x"), data)})
        result = g._extract_radiance(snap, "IR112")
        np.testing.assert_array_equal(result, data)

    def test_rad_variable_2d(self):
        g = GK2A()
        data = np.random.rand(100, 100).astype(np.float32)
        snap = xr.Dataset({"Rad": (("y", "x"), data)})
        result = g._extract_radiance(snap, "IR112")
        np.testing.assert_array_equal(result, data)

    def test_missing_band_raises(self):
        g = GK2A()
        snap = xr.Dataset({"scalar": 42.0})
        with pytest.raises(KeyError, match="Cannot find radiance"):
            g._extract_radiance(snap, "IR112")


class TestBuildCoords:
    def test_flips_descending_y(self):
        g = GK2A()
        rad = np.arange(6).reshape(3, 2).astype(np.float32)
        y = np.array([0.01, 0.0, -0.01])
        x = np.array([-0.01, 0.01])
        ds = xr.Dataset(coords={"x": ("x", x), "y": ("y", y)})
        x_m, y_m, rad_out = g._build_coords(ds, rad)
        assert y_m[0] < y_m[-1]
        np.testing.assert_array_equal(rad_out[0], rad[2])

    def test_radian_to_meter_conversion(self):
        g = GK2A()
        rad = np.ones((3, 3), dtype=np.float32)
        x = np.array([-0.01, 0.0, 0.01])
        y = np.array([-0.01, 0.0, 0.01])
        ds = xr.Dataset(coords={"x": ("x", x), "y": ("y", y)})
        x_m, y_m, _ = g._build_coords(ds, rad)
        from stereo_winds.readers.gk2a import _SAT_HEIGHT
        np.testing.assert_allclose(x_m[2], 0.01 * _SAT_HEIGHT)


class TestGK2AConfig:
    """Verify GK2A_CONFIG is registered in the satellite configs."""

    def test_config_exists(self):
        from stereo_winds.config import SATELLITE_CONFIGS
        assert "gk2a" in SATELLITE_CONFIGS

    def test_config_values(self):
        from stereo_winds.config import GK2A_CONFIG
        assert GK2A_CONFIG.sub_lon_deg == pytest.approx(128.2)
        assert GK2A_CONFIG.sweep == "y"
        assert GK2A_CONFIG.n_rows == 5500
        assert GK2A_CONFIG.n_cols == 5500


# ── Smoke tests (require network access to source.coop) ─────────────


@pytest.mark.network
class TestGK2ASmoke:
    """Live smoke tests reading from icechunk stores at source.coop."""

    def test_load_ir112_2km(self):
        g = GK2A(bands=["IR112"])
        ds = g.data_at_time(dt.datetime(2024, 1, 15, 3, 0))
        rad = ds["Rad"]
        assert rad.dims == ("time", "band", "y", "x")
        assert rad.shape[0] == 1 and rad.shape[1] == 1
        assert rad.dtype == np.float32
        y = ds["y"].values
        assert y[-1] > y[0]
        orb = rad.attrs["orbital_parameters"]
        assert "projection_altitude" in orb
        assert orb["satellite_nominal_longitude"] == pytest.approx(128.2)
        assert ds.attrs["sweep_angle_axis"] == "y"

    def test_abi_band_name_accepted(self):
        g = GK2A(bands=["C14"])
        ds = g.data_at_time(dt.datetime(2024, 1, 15, 3, 0))
        assert ds["Rad"].shape[2] > 0

    def test_data_has_valid_values(self):
        g = GK2A(bands=["IR112"])
        ds = g.data_at_time(dt.datetime(2024, 1, 15, 3, 0))
        data = ds["Rad"].values[0, 0]
        assert np.isfinite(data).sum() > 0
