"""Tests for the GK-2A AMI icechunk reader."""

import datetime as dt
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from stereo_winds.readers import _geos_store
from stereo_winds.readers._satpy_s3 import SceneNotInStore
from stereo_winds.readers.gk2a import (
    GK2A,
    _ABI_TO_AMI,
    _BAND_RESOLUTION,
    _resolve_band,
)
from stereo_winds.readers.himawari import Himawari

VIRTUAL_IR112 = "geo/virtualized/gk2a_ami_fd_ir112.icechunk"
VIRTUAL_IR112_2025 = "geo/virtualized/gk2a_ami_fd_ir112_2025-12-31.icechunk"


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
        assert g._store_prefix("2000m") == "geo/gk2a_2000m_test.icechunk"
        assert g._store_prefix("500m") == "geo/gk2a_500m_test.icechunk"

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


@pytest.fixture
def bucket(monkeypatch):
    """Stand in for both the geo/ listing and the geo/virtualized one."""
    from stereo_winds.readers import _virtual_store as virtual

    geo = [
        "gk2a_2000m_test.icechunk",
        "gk2a_2000m.icechunk",
        "himawari_2000m_test.icechunk",
        "himawari_2000m.icechunk",
    ]
    virtualized = [
        "gk2a_ami_fd_ir112.icechunk",
        "gk2a_ami_fd_ir112_2025-12-31.icechunk",
        "gk2a_ami_fd_ir105.icechunk",
        "ahi_h9_fd_b14.icechunk",
    ]
    monkeypatch.setattr(_geos_store, "_BUCKET_LISTINGS", {})
    monkeypatch.setattr(_geos_store, "_STORE_CONTENTS", {})
    monkeypatch.setattr(
        _geos_store.GeoStoreReader, "_bucket_stores", classmethod(lambda cls: geo)
    )
    monkeypatch.setattr(virtual, "list_virtual_stores", lambda *a, **k: virtualized)
    return geo, virtualized


class TestVirtualOnlyCandidates:
    def test_gk2a_reads_only_the_virtual_stores(self, bucket):
        order = GK2A(bands=["IR112"])._candidate_stores("IR112")
        assert order == [VIRTUAL_IR112, VIRTUAL_IR112_2025]
        assert not any(p.startswith("geo/gk2a_") for p in order)

    def test_virtual_candidates_are_pruned_by_time(self, bucket):
        order = GK2A(bands=["IR112"])._candidate_stores("IR112", dt.datetime(2026, 8, 1))
        assert order == [VIRTUAL_IR112]

    def test_discovery_off_still_reads_virtual(self, bucket, monkeypatch):
        monkeypatch.setattr(GK2A, "store_discovery_prefix", "")
        order = GK2A(bands=["IR112"])._candidate_stores("IR112")
        assert order == [VIRTUAL_IR112, VIRTUAL_IR112_2025]

    def test_other_readers_are_unchanged(self, bucket):
        assert Himawari.virtual_only is False
        order = Himawari(bands=["C14"])._candidate_stores("B14")
        assert order[0] == "geo/himawari_2000m_test.icechunk"
        assert "geo/himawari_2000m.icechunk" in order

    def test_other_readers_keep_named_store_only_without_discovery(
        self, bucket, monkeypatch
    ):
        monkeypatch.setattr(Himawari, "store_discovery_prefix", "")
        assert Himawari(bands=["C14"])._candidate_stores("B14") == [
            "geo/himawari_2000m_test.icechunk"
        ]


def _s3_scene():
    return xr.Dataset({"Rad": (("time", "band", "y", "x"), np.zeros((1, 1, 2, 2)))})


class TestVirtualThenS3:
    """A virtual miss or failure is served from noaa-gk2a-pds via satpy."""

    @pytest.fixture
    def s3_calls(self, monkeypatch):
        calls = []

        def fake_s3(self, t, band):
            calls.append((t, band))
            out = _s3_scene()
            out.attrs["source"] = "public S3 L1b via satpy"
            return out

        monkeypatch.setattr(GK2A, "_s3_data_at_time", fake_s3)
        monkeypatch.setattr(GK2A, "prune_download_cache", lambda self, t: None)
        return calls

    def test_virtual_miss_falls_back_to_s3(self, bucket, monkeypatch, s3_calls):
        when = dt.datetime(2026, 8, 1, 3, 0)
        # The only store left after pruning ends a day before the request.
        _geos_store._STORE_CONTENTS[VIRTUAL_IR112] = (
            frozenset({"IR112"}),
            np.datetime64("2026-01-01", "ns"),
            np.datetime64("2026-07-31", "ns"),
        )
        opened = []
        monkeypatch.setattr(GK2A, "_open_dataset_at", lambda self, p: opened.append(p))
        ds = GK2A(bands=["IR112"]).data_at_time(when)
        assert s3_calls == [(when, "IR112")]
        assert ds.attrs["source"] == "public S3 L1b via satpy"
        assert opened == []  # nothing in geo/gk2a_* was even opened

    def test_virtual_error_falls_back_to_s3(self, bucket, monkeypatch, s3_calls):
        def boom(self, prefix):
            raise OSError(f"ranged GET failed for {prefix}")

        monkeypatch.setattr(GK2A, "_open_dataset_at", boom)
        # Store contents are known, so selection succeeds and the open fails.
        _geos_store._STORE_CONTENTS[VIRTUAL_IR112] = (
            frozenset({"IR112"}),
            np.datetime64("2026-01-01", "ns"),
            np.datetime64("2026-12-31", "ns"),
        )
        when = dt.datetime(2026, 8, 1, 3, 0)
        ds = GK2A(bands=["IR112"]).data_at_time(when)
        assert s3_calls == [(when, "IR112")]
        assert ds.attrs["source"] == "public S3 L1b via satpy"

    def test_virtual_hit_does_not_touch_s3(self, bucket, monkeypatch, s3_calls):
        def served(self, t, band):
            out = _s3_scene()
            out.attrs["store"] = self._select_store(band, t)
            return out

        _geos_store._STORE_CONTENTS[VIRTUAL_IR112] = (
            frozenset({"IR112"}),
            np.datetime64("2026-01-01", "ns"),
            np.datetime64("2026-12-31", "ns"),
        )
        monkeypatch.setattr(GK2A, "_icechunk_data_at_time", served)
        ds = GK2A(bands=["IR112"]).data_at_time(dt.datetime(2026, 8, 1, 3, 0))
        assert ds.attrs["store"] == VIRTUAL_IR112
        assert s3_calls == []

    def test_no_fallback_surfaces_the_miss(self, bucket):
        _geos_store._STORE_CONTENTS[VIRTUAL_IR112] = (
            frozenset({"IR112"}),
            np.datetime64("2026-01-01", "ns"),
            np.datetime64("2026-07-31", "ns"),
        )
        g = GK2A(bands=["IR112"], allow_s3_fallback=False)
        with pytest.raises(SceneNotInStore):
            g.data_at_time(dt.datetime(2026, 8, 1, 3, 0))


def _load_ring():
    name = "infer_student_global_ring"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestAvailability:
    def test_union_of_virtual_and_s3_without_test_stores(self, bucket, monkeypatch):
        ring = _load_ring()
        start = dt.datetime(2026, 8, 1, 0, 0)
        end = dt.datetime(2026, 8, 1, 1, 0)
        virtual_times = np.array(
            ["2026-08-01T00:00", "2026-08-01T00:10", "2026-08-01T00:20"],
            dtype="datetime64[ns]",
        )
        s3_times = np.array(
            ["2026-08-01T00:20", "2026-08-01T00:30", "2026-08-01T00:40"],
            dtype="datetime64[ns]",
        )
        _geos_store._STORE_CONTENTS[VIRTUAL_IR112] = (
            frozenset({"IR112"}),
            virtual_times.min(),
            virtual_times.max(),
        )
        opened = []

        def open_at(self, prefix):
            opened.append(prefix)
            return xr.Dataset(coords={"time": virtual_times})

        monkeypatch.setattr(GK2A, "_open_dataset_at", open_at)
        monkeypatch.setattr(ring, "_s3_l1b_times", lambda *a, **k: s3_times)

        times = ring.satellite_available_times("gk2a", "C14", start, end)
        expected = np.unique(np.concatenate([virtual_times, s3_times]))
        np.testing.assert_array_equal(times, expected)
        assert opened == [VIRTUAL_IR112]


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


@pytest.mark.network
class TestGK2AVirtualThenS3Live:
    """Anonymous reads: the live virtual store first, then noaa-gk2a-pds."""

    def setup_method(self):
        _geos_store.clear_store_cache()

    def _inside_virtual(self, g):
        prefix = g._candidate_stores("IR105")[0]
        assert prefix.startswith("geo/virtualized/gk2a_ami_fd_ir105")
        times = g._open_dataset_at(prefix)["time"].values
        return prefix, times[len(times) // 2].astype("datetime64[s]").item()

    def test_reads_ir105_from_virtual_store(self):
        g = GK2A(bands=["IR105"])
        prefix, when = self._inside_virtual(g)
        t0 = time.perf_counter()
        ds = g.data_at_time(when)
        print(f"virtual {prefix} at {when}: {time.perf_counter() - t0:.1f}s")
        assert ds.attrs["source"] == "icechunk"
        assert ds.attrs["store"].startswith("geo/virtualized/gk2a_ami_fd_ir105")
        assert np.isfinite(ds["Rad"].values).sum() > 0

    def test_falls_back_to_noaa_bucket(self, monkeypatch, tmp_path):
        g = GK2A(bands=["IR105"], cache_dir=str(tmp_path))
        _, when = self._inside_virtual(g)

        def fail(self, prefix):
            raise OSError("virtual lookup forced to fail")

        monkeypatch.setattr(GK2A, "_open_dataset_at", fail)
        t0 = time.perf_counter()
        ds = g.data_at_time(when)
        print(f"satpy noaa-gk2a-pds at {when}: {time.perf_counter() - t0:.1f}s")
        assert ds.attrs["source"] == "public S3 L1b via satpy"
        assert np.isfinite(ds["Rad"].values).sum() > 0
