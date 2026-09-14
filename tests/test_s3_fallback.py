"""Tests for the public-S3 + satpy fallback in the AHI and AMI readers.

Offline: S3 listing, downloading and satpy loading are all stubbed, so
these cover the decision to fall back, the key construction, and the
orientation of the converted scene.
"""

import datetime as dt
import sys
import tempfile
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from stereo_winds.readers._satpy_s3 import (
    SceneNotInStore,
    load_scene_array,
    scene_to_rad,
)
from stereo_winds.readers.gk2a import GK2A
from stereo_winds.readers.himawari import Himawari

T = dt.datetime(2026, 8, 1, 4, 3, 17)


# ── Store-miss detection ──────────────────────────────────────────────

def _store(times):
    return xr.Dataset(
        {"B14": (("time", "y", "x"), np.zeros((len(times), 2, 2), np.float32))},
        coords={"time": [np.datetime64(t, "ns") for t in times]},
    )


class TestSelectTime:
    def test_nearby_scan_is_returned(self):
        ds = _store([dt.datetime(2026, 8, 1, 4, 0)])
        snap = Himawari()._select_time(ds, dt.datetime(2026, 8, 1, 4, 3))
        assert snap["time"].values == np.datetime64("2026-08-01T04:00", "ns")

    def test_distant_scan_raises(self):
        """Regression: nearest-neighbour alone happily returns a scan months away."""
        ds = _store([dt.datetime(2025, 2, 1, 23, 30)])
        with pytest.raises(SceneNotInStore):
            Himawari()._select_time(ds, dt.datetime(2026, 8, 1, 0, 0))

    def test_tolerance_can_be_disabled(self):
        ds = _store([dt.datetime(2025, 2, 1, 23, 30)])
        snap = Himawari()._select_time(ds, dt.datetime(2026, 8, 1), tolerance=None)
        assert snap["time"].values == np.datetime64("2025-02-01T23:30", "ns")

    def test_gk2a_uses_the_same_rule(self):
        ds = _store([dt.datetime(2025, 1, 10, 4, 0)])
        with pytest.raises(SceneNotInStore):
            GK2A()._select_time(ds, dt.datetime(2026, 8, 1, 0, 0))


# ── Fallback dispatch ─────────────────────────────────────────────────

class TestFallbackDispatch:
    def _wire(self, monkeypatch, reader, store_exc):
        seen = {}

        def miss(self, t, band):
            raise store_exc

        def s3(self, t, band):
            seen["called"] = (t, band)
            return xr.Dataset(attrs={"source": "public S3 L1b via satpy"})

        monkeypatch.setattr(type(reader), "_icechunk_data_at_time", miss)
        monkeypatch.setattr(type(reader), "_s3_data_at_time", s3)
        return seen

    def test_store_miss_falls_back(self, monkeypatch):
        r = Himawari(satellite="himawari9", bands=["C14"])
        seen = self._wire(monkeypatch, r, SceneNotInStore("nope"))
        out = r.data_at_time(T)
        assert out.attrs["source"] == "public S3 L1b via satpy"
        assert seen["called"] == (T, "B14")

    def test_store_error_falls_back(self, monkeypatch):
        """A store that will not open is as unusable as one that lacks the time."""
        r = GK2A(bands=["C14"])
        seen = self._wire(monkeypatch, r, RuntimeError("store unreachable"))
        r.data_at_time(T)
        assert seen["called"][1] == "IR112"

    def test_fallback_can_be_switched_off(self, monkeypatch):
        r = Himawari(satellite="himawari9", bands=["C14"], allow_s3_fallback=False)
        self._wire(monkeypatch, r, SceneNotInStore("nope"))
        with pytest.raises(SceneNotInStore):
            r.data_at_time(T)

    def test_store_hit_does_not_touch_s3(self, monkeypatch):
        r = Himawari(satellite="himawari9", bands=["C14"])
        monkeypatch.setattr(
            type(r), "_icechunk_data_at_time",
            lambda self, t, band: xr.Dataset(attrs={"source": "icechunk"}))
        monkeypatch.setattr(
            type(r), "_s3_data_at_time",
            lambda self, t, band: pytest.fail("S3 must not be used"))
        assert r.data_at_time(T).attrs["source"] == "icechunk"


# ── S3 key construction ───────────────────────────────────────────────

class _FakeFS:
    def __init__(self, keys=()):
        self.keys = list(keys)
        self.patterns: list[str] = []

    def glob(self, pattern):
        self.patterns.append(pattern)
        return list(self.keys)


class TestS3Keys:
    def test_slot_is_floored_to_the_scan_cadence(self):
        assert Himawari()._snap_slot(T) == dt.datetime(2026, 8, 1, 4, 0)
        assert GK2A()._snap_slot(T) == dt.datetime(2026, 8, 1, 4, 0)

    def test_himawari_pattern(self):
        r = Himawari(satellite="himawari9", bands=["C14"])
        r._fs = _FakeFS()
        r._s3_keys(dt.datetime(2026, 8, 1, 4, 0), "B14")
        assert r._fs.patterns == [
            "noaa-himawari9/AHI-L1b-FLDK/2026/08/01/0400/"
            "HS_H09_20260801_0400_B14_FLDK_R*_S*.DAT*"
        ]

    def test_himawari8_uses_its_own_bucket(self):
        r = Himawari(satellite="himawari8", bands=["C14"])
        r._fs = _FakeFS()
        r._s3_keys(dt.datetime(2026, 8, 1, 4, 0), "B14")
        assert r._fs.patterns[0].startswith("noaa-himawari8/")
        assert "HS_H08_" in r._fs.patterns[0]

    def test_gk2a_pattern_globs_the_resolution_tag(self):
        r = GK2A(bands=["C14"])
        r._fs = _FakeFS()
        r._s3_keys(dt.datetime(2026, 8, 1, 4, 0), "IR112")
        assert r._fs.patterns == [
            "noaa-gk2a-pds/AMI/L1B/FD/202608/01/04/"
            "gk2a_ami_le1b_ir112_fd*ge_202608010400.nc"
        ]

    def test_missing_files_raise(self):
        r = Himawari(satellite="himawari9", bands=["C14"])
        r._fs = _FakeFS([])
        with pytest.raises(FileNotFoundError, match="No AHI HSD files"):
            r._s3_data_at_time(T, "B14")

    def test_partial_segment_set_raises(self):
        """Nine of ten segments would decode to a disk with a missing stripe."""
        r = Himawari(satellite="himawari9", bands=["C14"])
        r._fs = _FakeFS([f"k{i}" for i in range(9)])
        with pytest.raises(FileNotFoundError, match="Expected 10 HSD segments"):
            r._s3_data_at_time(T, "B14")


# ── satpy scene conversion ────────────────────────────────────────────

def _fake_satpy_da(values, extent, sub_lon=140.7, height=35785863.0):
    """Mimic a satpy DataArray: row 0 north, area_extent in metres."""
    area = types.SimpleNamespace(
        area_extent=extent,
        crs=types.SimpleNamespace(
            to_dict=lambda: {"proj": "geos", "lon_0": sub_lon, "h": height}),
    )
    return xr.DataArray(
        values, dims=("y", "x"),
        attrs={
            "area": area,
            "orbital_parameters": {
                "projection_longitude": sub_lon,
                "projection_altitude": height,
                "satellite_actual_longitude": sub_lon + 0.04,
            },
            "start_time": dt.datetime(2026, 8, 1, 4, 0),
            "end_time": dt.datetime(2026, 8, 1, 4, 9),
        },
    )


class TestSceneToRad:
    def _convert(self, values, extent):
        da = _fake_satpy_da(values, extent)
        return scene_to_rad(da, "B14", sweep="y", fallback_sub_lon=0.0,
                            fallback_height=1.0)

    def test_shape_and_dims(self):
        ds = self._convert(np.arange(12, dtype=np.float32).reshape(3, 4),
                           (-4000.0, -3000.0, 4000.0, 3000.0))
        assert ds["Rad"].dims == ("time", "band", "y", "x")
        assert ds["Rad"].shape == (1, 1, 3, 4)

    def test_axes_are_ascending(self):
        ds = self._convert(np.zeros((3, 4), np.float32),
                           (-4000.0, -3000.0, 4000.0, 3000.0))
        assert ds.x.values[0] < ds.x.values[-1]
        assert ds.y.values[0] < ds.y.values[-1]

    def test_rows_flip_with_the_y_axis(self):
        """satpy hands back row 0 = north; we return south-first."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
        ds = self._convert(values, (-2000.0, -2000.0, 2000.0, 2000.0))
        assert np.array_equal(ds["Rad"].values[0, 0], values[::-1])

    def test_cell_centres_are_half_a_pixel_inside_the_extent(self):
        ds = self._convert(np.zeros((2, 2), np.float32),
                           (-2000.0, -2000.0, 2000.0, 2000.0))
        assert np.allclose(ds.x.values, [-1000.0, 1000.0])
        assert np.allclose(ds.y.values, [-1000.0, 1000.0])

    def test_projection_metadata_is_carried_through(self):
        ds = self._convert(np.zeros((2, 2), np.float32),
                           (-2000.0, -2000.0, 2000.0, 2000.0))
        orb = ds["Rad"].attrs["orbital_parameters"]
        assert orb["projection_longitude"] == pytest.approx(140.7)
        assert orb["projection_altitude"] == pytest.approx(35785863.0)
        assert orb["satellite_actual_longitude"] == pytest.approx(140.74)

    def test_source_is_labelled(self):
        ds = self._convert(np.zeros((2, 2), np.float32),
                           (-2000.0, -2000.0, 2000.0, 2000.0))
        assert ds.attrs["source"] == "public S3 L1b via satpy"
        assert ds.attrs["sweep_angle_axis"] == "y"

    def test_inverted_extent_is_normalised(self):
        """Some area definitions carry y inverted; data must follow."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
        ds = self._convert(values, (2000.0, 2000.0, -2000.0, -2000.0))
        assert ds.x.values[0] < ds.x.values[-1]
        assert ds.y.values[0] < ds.y.values[-1]


# ── satpy scratch-file cleanup ────────────────────────────────────────

class _FakeConfig:
    """Stand-in for satpy.config: a get/set pair with a context manager."""

    def __init__(self):
        self._values = {"tmp_dir": tempfile.gettempdir()}

    def get(self, key, default=None):
        return self._values.get(key, default)

    def set(self, **kwargs):
        config = self

        class _Ctx:
            def __enter__(self_inner):
                self_inner.old = dict(config._values)
                config._values.update(kwargs)
                return config

            def __exit__(self_inner, *exc):
                config._values = self_inner.old
                return False

        return _Ctx()


def _install_fake_satpy(monkeypatch, record):
    """Inject a satpy whose Scene writes a file into the configured tmp_dir.

    That is what ``ahi_hsd`` does for every bz2 segment, and what used to
    pile up in /tmp.
    """
    config = _FakeConfig()

    class FakeScene:
        def __init__(self, reader, filenames):
            record["reader"] = reader
            # Simulate decompression into satpy's configured scratch area.
            tmp_dir = Path(config.get("tmp_dir"))
            for i, _ in enumerate(filenames):
                path = tmp_dir / f"segment-{i}.DAT"
                path.write_bytes(b"x" * 16)
                record.setdefault("written", []).append(path)

        def load(self, bands):
            record["loaded"] = list(bands)

        def __getitem__(self, band):
            data = xr.DataArray(np.ones((2, 2), np.float32),
                                dims=("y", "x"), attrs={"band": band})

            class Lazy(xr.DataArray):
                __slots__ = ()

                def compute(self_inner, **kw):
                    record["computed"] = True
                    return data

            return Lazy(data)

    satpy = types.ModuleType("satpy")
    satpy.config = config
    satpy.Scene = FakeScene
    monkeypatch.setitem(sys.modules, "satpy", satpy)
    return record


class TestScratchCleanup:
    def test_scratch_goes_to_the_given_directory_not_tmp(self, tmp_path, monkeypatch):
        record = _install_fake_satpy(monkeypatch, {})
        scratch = tmp_path / "cache"
        load_scene_array("ahi_hsd", [Path("a.bz2"), Path("b.bz2")], "B14",
                         scratch_dir=scratch)
        written = record["written"]
        assert written, "fake satpy wrote nothing"
        for path in written:
            assert scratch in path.parents, f"{path} escaped the scratch dir"

    def test_scratch_directory_is_removed(self, tmp_path, monkeypatch):
        """Regression: satpy only unlinks these when handlers are collected."""
        record = _install_fake_satpy(monkeypatch, {})
        scratch = tmp_path / "cache"
        load_scene_array("ahi_hsd", [Path("a.bz2")], "B14", scratch_dir=scratch)
        assert not list(scratch.glob("satpy-scratch-*"))
        assert not any(p.exists() for p in record["written"])

    def test_data_is_computed_before_cleanup(self, tmp_path, monkeypatch):
        """A lazy array would reference files we are about to delete."""
        record = _install_fake_satpy(monkeypatch, {})
        out = load_scene_array("ahi_hsd", [Path("a.bz2")], "B14",
                               scratch_dir=tmp_path / "cache")
        assert record.get("computed") is True
        assert isinstance(out.values, np.ndarray)

    def test_cleanup_happens_even_on_failure(self, tmp_path, monkeypatch):
        record = _install_fake_satpy(monkeypatch, {})
        satpy = sys.modules["satpy"]

        class Boom(satpy.Scene):
            def load(self, bands):
                raise RuntimeError("decode failed")

        satpy.Scene = Boom
        scratch = tmp_path / "cache"
        with pytest.raises(RuntimeError, match="decode failed"):
            load_scene_array("ahi_hsd", [Path("a.bz2")], "B14",
                             scratch_dir=scratch)
        assert not list(scratch.glob("satpy-scratch-*"))

    def test_each_load_uses_a_fresh_directory(self, tmp_path, monkeypatch):
        """Concurrent or repeated loads must not share a scratch path."""
        record = _install_fake_satpy(monkeypatch, {})
        seen = set()
        for _ in range(3):
            record["written"] = []
            load_scene_array("ahi_hsd", [Path("a.bz2")], "B14",
                             scratch_dir=tmp_path / "cache")
            seen.add(record["written"][0].parent)
        assert len(seen) == 3


# ── Warning hygiene ───────────────────────────────────────────────────

class TestNoSpuriousWarnings:
    """Every scene load used to emit the same three warnings."""

    def test_crs_read_without_the_proj4_warning(self):
        """CRS.to_dict() round-trips via PROJ and warns on every scene."""
        from pyproj import CRS

        from stereo_winds.readers._satpy_s3 import _crs_parameters

        area = types.SimpleNamespace(crs=CRS.from_dict({
            "proj": "geos", "lon_0": 140.7, "h": 35785863.0,
            "a": 6378137.0, "rf": 298.257024882273,
        }))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            params = _crs_parameters(area)
        assert params["proj"] == "geos"
        assert params["lon_0"] == pytest.approx(140.7)
        assert params["h"] == pytest.approx(35785863.0)
        assert params["a"] == pytest.approx(6378137.0)

    def test_crs_parameters_feed_the_metadata_reader(self):
        """The proj-style keys must be the ones area_projection reads."""
        from pyproj import CRS

        from stereo_winds.readers._geos_meta import scene_orbital_parameters
        from stereo_winds.readers._satpy_s3 import _crs_parameters

        area = types.SimpleNamespace(crs=CRS.from_dict({
            "proj": "geos", "lon_0": 45.5, "h": 35785831.0, "a": 6378169.0,
            "rf": 295.488065897014,
        }))
        meta = xr.Dataset(attrs={"area": {"projection": _crs_parameters(area)}})
        orb = scene_orbital_parameters(meta, fallback_sub_lon=0.0,
                                       fallback_height=1.0)
        assert orb["projection_longitude"] == pytest.approx(45.5)
        assert orb["projection_altitude"] == pytest.approx(35785831.0)

    def test_calibration_nan_warning_is_suppressed(self):
        from stereo_winds.readers._satpy_s3 import _quiet_nan_arithmetic

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with _quiet_nan_arithmetic():
                with np.errstate(invalid="warn"):
                    np.log(np.array([-1.0, np.nan]))
        assert not [w for w in caught
                    if "invalid value encountered in log" in str(w.message)]

    def test_other_warnings_still_get_through(self):
        """The filter must be narrow — it runs around real data loading."""
        from stereo_winds.readers._satpy_s3 import _quiet_nan_arithmetic

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with _quiet_nan_arithmetic():
                warnings.warn("something worth seeing", UserWarning)
        assert len(caught) == 1
