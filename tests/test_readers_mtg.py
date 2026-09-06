"""Tests for the MTG FCI icechunk reader."""

import datetime as dt

import numpy as np
import pytest
import xarray as xr

from stereo_winds.config import ABI_TO_FCI_BAND
from stereo_winds.readers.mtg import (
    MTG,
    _BAND_RESOLUTION,
    _resolve_band,
)


# ── Offline unit tests (no network) ──────────────────────────────────


class TestResolveBand:
    def test_native_fci_band(self):
        assert _resolve_band("ir_105") == "ir_105"
        assert _resolve_band("vis_06") == "vis_06"

    def test_abi_to_fci_translation(self):
        assert _resolve_band("C13") == "ir_105"
        assert _resolve_band("C08") == "wv_63"

    def test_all_abi_bands_resolve(self):
        for abi, fci in ABI_TO_FCI_BAND.items():
            assert _resolve_band(abi) == fci

    def test_unknown_band_raises(self):
        with pytest.raises(ValueError, match="Unknown band"):
            _resolve_band("X99")

    def test_c09_not_mapped(self):
        """C09 (6.9 um) has no FCI equivalent."""
        with pytest.raises(ValueError, match="Unknown band"):
            _resolve_band("C09")

    def test_c14_not_mapped(self):
        """C14 (11.2 um) has no FCI equivalent."""
        with pytest.raises(ValueError, match="Unknown band"):
            _resolve_band("C14")


class TestBandResolution:
    def test_vis_500m(self):
        for b in ("vis_04", "vis_05", "vis_06", "vis_08", "vis_09"):
            assert _BAND_RESOLUTION[b] == "500m"

    def test_nir_1km(self):
        for b in ("nir_13", "nir_16", "nir_22"):
            assert _BAND_RESOLUTION[b] == "1000m"

    def test_ir_2km(self):
        for b in ("ir_38", "wv_63", "wv_73", "ir_87", "ir_105", "ir_133"):
            assert _BAND_RESOLUTION[b] == "2000m"

    def test_all_bands_covered(self):
        assert len(_BAND_RESOLUTION) == 16


class TestConstructor:
    def test_defaults(self):
        m = MTG()
        assert m.satellite == "mtg-i1"
        assert m.bands == ["ir_105"]

    def test_abi_band_translated(self):
        m = MTG(bands=["C13"])
        assert m.bands == ["ir_105"]

    def test_unknown_satellite_raises(self):
        with pytest.raises(ValueError, match="Unknown satellite"):
            MTG(satellite="goes19")

    def test_store_prefix(self):
        m = MTG()
        assert m._store_prefix("2000m") == "geo/mtg_2000m.icechunk"
        assert m._store_prefix("500m") == "geo/mtg_500m.icechunk"

    def test_repr(self):
        m = MTG(bands=["ir_105"])
        assert "mtg-i1" in repr(m)
        assert "ir_105" in repr(m)


class TestCoordSynthesis:
    def test_synthesised_coords_symmetric(self):
        m = MTG()
        x = m._get_coord(xr.Dataset(), "x", 5568)
        y = m._get_coord(xr.Dataset(), "y", 5568)
        np.testing.assert_allclose(x[0], -x[-1], atol=1e-10)
        np.testing.assert_allclose(y[0], -y[-1], atol=1e-10)

    def test_synthesised_scale_2km(self):
        m = MTG()
        x = m._get_coord(xr.Dataset(), "x", 5568)
        dx = float(x[1] - x[0])
        np.testing.assert_allclose(dx, 5.58871e-05, rtol=1e-4)

    def test_synthesised_scale_1km(self):
        m = MTG()
        x = m._get_coord(xr.Dataset(), "x", 11136)
        dx = float(x[1] - x[0])
        np.testing.assert_allclose(dx, 2.79436e-05, rtol=1e-4)

    def test_synthesised_scale_500m(self):
        m = MTG()
        x = m._get_coord(xr.Dataset(), "x", 22272)
        dx = float(x[1] - x[0])
        np.testing.assert_allclose(dx, 1.39718e-05, rtol=1e-4)

    def test_uses_store_coords_when_present(self):
        m = MTG()
        x_expected = np.linspace(-0.15, 0.15, 5568)
        ds = xr.Dataset(coords={"x": ("x", x_expected)})
        x = m._get_coord(ds, "x", 5568)
        np.testing.assert_array_equal(x, x_expected)


class TestExtractRadiance:
    def test_band_as_variable(self):
        m = MTG()
        data = np.random.rand(100, 100).astype(np.float32)
        snap = xr.Dataset({"ir_105": (("y", "x"), data)})
        result = m._extract_radiance(snap, "ir_105")
        np.testing.assert_array_equal(result, data)

    def test_rad_variable_2d(self):
        m = MTG()
        data = np.random.rand(100, 100).astype(np.float32)
        snap = xr.Dataset({"Rad": (("y", "x"), data)})
        result = m._extract_radiance(snap, "ir_105")
        np.testing.assert_array_equal(result, data)

    def test_effective_radiance_variable(self):
        m = MTG()
        data = np.random.rand(100, 100).astype(np.float32)
        snap = xr.Dataset({"effective_radiance": (("y", "x"), data)})
        result = m._extract_radiance(snap, "ir_105")
        np.testing.assert_array_equal(result, data)

    def test_missing_band_raises(self):
        m = MTG()
        snap = xr.Dataset({"scalar": 42.0})
        with pytest.raises(KeyError, match="Cannot find radiance"):
            m._extract_radiance(snap, "ir_105")


class TestBuildCoords:
    def test_flips_descending_y(self):
        m = MTG()
        rad = np.arange(6).reshape(3, 2).astype(np.float32)
        y = np.array([0.01, 0.0, -0.01])
        x = np.array([-0.01, 0.01])
        ds = xr.Dataset(coords={"x": ("x", x), "y": ("y", y)})
        x_m, y_m, rad_out = m._build_coords(ds, rad)
        assert y_m[0] < y_m[-1]
        np.testing.assert_array_equal(rad_out[0], rad[2])

    def test_radian_to_meter_conversion(self):
        m = MTG()
        rad = np.ones((3, 3), dtype=np.float32)
        x = np.array([-0.01, 0.0, 0.01])
        y = np.array([-0.01, 0.0, 0.01])
        ds = xr.Dataset(coords={"x": ("x", x), "y": ("y", y)})
        x_m, y_m, _ = m._build_coords(ds, rad)
        from stereo_winds.readers.mtg import _SAT_HEIGHT
        np.testing.assert_allclose(x_m[2], 0.01 * _SAT_HEIGHT["mtg-i1"])


# ── Smoke tests (require network access to source.coop) ─────────────


@pytest.mark.network
class TestMTGSmoke:
    """Live smoke tests reading from icechunk stores at source.coop."""

    def test_load_ir105_2km(self):
        m = MTG(satellite="mtg-i1", bands=["ir_105"])
        ds = m.data_at_time(dt.datetime(2024, 6, 15, 12, 0))
        rad = ds["Rad"]
        assert rad.dims == ("time", "band", "y", "x")
        assert rad.shape[0] == 1 and rad.shape[1] == 1
        assert rad.dtype == np.float32
        y = ds["y"].values
        assert y[-1] > y[0]
        orb = rad.attrs["orbital_parameters"]
        assert "projection_altitude" in orb
        assert orb["satellite_nominal_longitude"] == pytest.approx(0.0)
        assert ds.attrs["sweep_angle_axis"] == "y"

    def test_abi_band_name_accepted(self):
        m = MTG(bands=["C13"])
        ds = m.data_at_time(dt.datetime(2024, 6, 15, 12, 0))
        assert ds["Rad"].shape[2] > 0

    def test_data_has_valid_values(self):
        m = MTG(bands=["ir_105"])
        ds = m.data_at_time(dt.datetime(2024, 6, 15, 12, 0))
        data = ds["Rad"].values[0, 0]
        assert np.isfinite(data).sum() > 0
