"""Tests for per-scene geostationary projection metadata parsing.

The icechunk stores describe their projection in two different layouts,
and navigation depends on reading the sub-satellite longitude actually
recorded for the timestamp rather than a hardcoded nominal value.
"""

import numpy as np
import pytest
import xarray as xr

from stereo_winds.readers._geos_meta import (
    area_projection,
    scene_orbital_parameters,
)

# Verbatim shapes taken from the source.coop stores.
PACKED_ORBITAL = (
    "{'projection_longitude': '140.7', 'projection_latitude': '0.0', "
    "'projection_altitude': '35785863.0', "
    "'satellite_actual_longitude': '140.74429999999998', "
    "'satellite_actual_latitude': '0.01', "
    "'satellite_actual_altitude': '35789640.0', "
    "'nadir_longitude': '140.69480856431218', "
    "'nadir_latitude': '-0.1456086266782693'}"
)
AREA = (
    "{'FLDK': {'description': 'AHI FLDK area', 'projection': {'proj': 'geos', "
    "'lon_0': 140.7, 'h': 35785863, 'x_0': 0, 'y_0': 0, 'a': 6378137, "
    "'rf': 298.257024882273, 'no_defs': None, 'type': 'crs'}, "
    "'shape': {'height': 5500, 'width': 5500}}}"
)

NOMINAL = {"fallback_sub_lon": 999.0, "fallback_height": 1.0}


def _scalar_var(value):
    return xr.DataArray(np.array(value))


class TestAreaProjection:
    def test_keyed_by_area_id(self):
        proj = area_projection(AREA)
        assert proj["lon_0"] == pytest.approx(140.7)
        assert proj["h"] == pytest.approx(35785863)

    def test_unkeyed_area(self):
        proj = area_projection({"projection": {"lon_0": -75.2}})
        assert proj["lon_0"] == pytest.approx(-75.2)

    def test_garbage_returns_none(self):
        assert area_projection("not a dict") is None
        assert area_projection(None) is None
        assert area_projection({"no": "projection"}) is None


class TestPackedLayout:
    """AHI store: one stringified dict per timestep."""

    def test_reads_projection_longitude(self):
        snap = xr.Dataset({"orbital_parameters": _scalar_var(PACKED_ORBITAL)})
        orb = scene_orbital_parameters(snap, **NOMINAL)
        assert orb["projection_longitude"] == pytest.approx(140.7)
        assert orb["satellite_nominal_longitude"] == pytest.approx(140.7)
        assert orb["projection_altitude"] == pytest.approx(35785863.0)

    def test_passes_through_actual_position(self):
        snap = xr.Dataset({"orbital_parameters": _scalar_var(PACKED_ORBITAL)})
        orb = scene_orbital_parameters(snap, **NOMINAL)
        assert orb["satellite_actual_longitude"] == pytest.approx(140.7443)
        assert orb["nadir_latitude"] == pytest.approx(-0.14560862)

    def test_string_values_become_floats(self):
        snap = xr.Dataset({"orbital_parameters": _scalar_var(PACKED_ORBITAL)})
        orb = scene_orbital_parameters(snap, **NOMINAL)
        assert all(isinstance(v, float) for v in orb.values())

    def test_from_band_attribute(self):
        snap = xr.Dataset({"B14": xr.DataArray(np.zeros((2, 2)))})
        snap["B14"].attrs["orbital_parameters"] = PACKED_ORBITAL
        orb = scene_orbital_parameters(snap, "B14", **NOMINAL)
        assert orb["projection_longitude"] == pytest.approx(140.7)


class TestFlattenedLayout:
    """AMI/FCI stores: one variable per field, area as a dataset attribute."""

    def test_reads_per_field_variables(self):
        snap = xr.Dataset({
            "projection_longitude": _scalar_var(128.2),
            "projection_altitude": _scalar_var(35785863.0),
            "satellite_actual_longitude": _scalar_var(128.31),
        })
        orb = scene_orbital_parameters(snap, **NOMINAL)
        assert orb["projection_longitude"] == pytest.approx(128.2)
        assert orb["projection_altitude"] == pytest.approx(35785863.0)
        assert orb["satellite_actual_longitude"] == pytest.approx(128.31)

    def test_area_attribute_supplies_missing_fields(self):
        snap = xr.Dataset(attrs={"area": AREA})
        orb = scene_orbital_parameters(snap, **NOMINAL)
        assert orb["projection_longitude"] == pytest.approx(140.7)
        assert orb["projection_altitude"] == pytest.approx(35785863)

    def test_per_field_variables_win_over_area(self):
        snap = xr.Dataset({"projection_longitude": _scalar_var(41.5)},
                          attrs={"area": AREA})
        orb = scene_orbital_parameters(snap, **NOMINAL)
        assert orb["projection_longitude"] == pytest.approx(41.5)


class TestFallback:
    def test_empty_store_uses_nominal(self, caplog):
        orb = scene_orbital_parameters(
            xr.Dataset(), fallback_sub_lon=-75.2, fallback_height=35786023.0,
        )
        assert orb["projection_longitude"] == pytest.approx(-75.2)
        assert orb["projection_altitude"] == pytest.approx(35786023.0)

    def test_fallback_is_warned_about(self, caplog):
        with caplog.at_level("WARNING"):
            scene_orbital_parameters(
                xr.Dataset(), fallback_sub_lon=-75.2, fallback_height=1.0,
            )
        assert "nominal" in caplog.text

    def test_reading_the_store_does_not_warn(self, caplog):
        snap = xr.Dataset({"orbital_parameters": _scalar_var(PACKED_ORBITAL)})
        with caplog.at_level("WARNING"):
            scene_orbital_parameters(snap, **NOMINAL)
        assert "nominal" not in caplog.text

    def test_unparseable_metadata_falls_back(self):
        snap = xr.Dataset({"orbital_parameters": _scalar_var("<<garbage>>")})
        orb = scene_orbital_parameters(
            snap, fallback_sub_lon=0.0, fallback_height=35786400.0,
        )
        assert orb["projection_longitude"] == pytest.approx(0.0)


class TestDriftWarning:
    def test_large_drift_is_flagged(self, caplog):
        snap = xr.Dataset({
            "projection_longitude": _scalar_var(-75.0),
            "satellite_actual_longitude": _scalar_var(-71.0),
        })
        with caplog.at_level("WARNING"):
            scene_orbital_parameters(snap, **NOMINAL)
        assert "from the projection origin" in caplog.text

    def test_station_keeping_drift_is_quiet(self, caplog):
        snap = xr.Dataset({
            "projection_longitude": _scalar_var(140.7),
            "satellite_actual_longitude": _scalar_var(140.744),
        })
        with caplog.at_level("WARNING"):
            scene_orbital_parameters(snap, **NOMINAL)
        assert "from the projection origin" not in caplog.text

    def test_dateline_straddle_is_not_drift(self, caplog):
        snap = xr.Dataset({
            "projection_longitude": _scalar_var(179.95),
            "satellite_actual_longitude": _scalar_var(-179.95),
        })
        with caplog.at_level("WARNING"):
            scene_orbital_parameters(snap, **NOMINAL)
        assert "from the projection origin" not in caplog.text
