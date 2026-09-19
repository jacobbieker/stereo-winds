"""Tests for the global-ring adapter.

Entirely offline: the script module is loaded (which only defines
functions), but nothing that touches the network, a GPU or a checkpoint is
ever called.
"""

import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import xarray as xr

from operational.adapters import ring

T0 = datetime(2026, 8, 1, 0, 0)

#: Names the adapter promises to re-export.
REEXPORTS = (
    "infer_satellite",
    "GlobalMosaic",
    "quality_attrs",
    "satellite_available_times",
    "availability_band",
    "scan_interval",
    "time_tag",
    "sat_nc_path",
    "global_nc_path",
    "filter_to_common_times",
    "RING_SATELLITES",
    "OUTPUT_VARS",
    "DT_MINUTES",
    "SCAN_INTERVAL_MINUTES",
)


class TestScriptLocation:
    """The script is found from the package, not from the working dir."""

    def test_repo_root_is_two_levels_above_the_adapter(self):
        assert ring.REPO_ROOT == Path(ring.__file__).resolve().parents[2]

    def test_script_exists(self):
        assert ring.RING_SCRIPT.is_file()
        assert ring.RING_SCRIPT.name == "infer_student_global_ring.py"

    def test_path_survives_a_cwd_change(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert ring.RING_SCRIPT.is_absolute()
        assert ring.load_ring() is sys.modules[ring.RING_MODULE_NAME]


class TestLoadRing:
    """Loading is cached, thread-safe and collision-free."""

    def test_returns_a_module(self):
        assert isinstance(ring.load_ring(), ModuleType)

    def test_registered_under_a_distinct_sys_modules_name(self):
        assert ring.RING_MODULE_NAME == "operational_ring"
        assert sys.modules[ring.RING_MODULE_NAME] is ring.load_ring()

    def test_does_not_collide_with_the_names_other_tests_use(self):
        for other in ("infer_student_global_ring", "ring_prefetch",
                      "write_mosaics", "ring_msg"):
            assert other != ring.RING_MODULE_NAME
            assert sys.modules.get(other) is not ring.load_ring()

    def test_repeated_calls_return_the_identical_module(self):
        assert ring.load_ring() is ring.load_ring()

    def test_warm_cache_does_not_re_execute(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            ring, "_exec_ring_module", lambda: calls.append(1),
        )
        ring.load_ring()
        ring.load_ring()
        assert calls == []

    def test_cold_cache_executes_exactly_once(self, monkeypatch):
        calls = []
        sentinel = ModuleType("sentinel_ring")

        def fake_exec():
            calls.append(1)
            return sentinel

        monkeypatch.setattr(ring, "_exec_ring_module", fake_exec)
        monkeypatch.setattr(ring, "_RING_MODULE", None)
        assert ring.load_ring() is sentinel
        assert ring.load_ring() is sentinel
        assert ring.load_ring() is sentinel
        assert calls == [1]

    def test_concurrent_loads_execute_once(self, monkeypatch):
        calls = []
        sentinel = ModuleType("sentinel_ring")

        def slow_exec():
            calls.append(1)
            time.sleep(0.05)
            return sentinel

        monkeypatch.setattr(ring, "_exec_ring_module", slow_exec)
        monkeypatch.setattr(ring, "_RING_MODULE", None)

        seen: list[ModuleType] = []
        start = threading.Barrier(8)

        def worker():
            start.wait()
            seen.append(ring.load_ring())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert calls == [1]
        assert seen == [sentinel] * 8

    def test_reuses_an_already_executed_sys_modules_entry(self, monkeypatch):
        loaded = ring.load_ring()
        monkeypatch.setattr(ring, "_RING_MODULE", None)
        # A cold adapter cache but a warm sys.modules entry must not
        # re-execute the script.
        assert ring._exec_ring_module() is loaded

    def test_missing_script_raises_a_helpful_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ring, "RING_SCRIPT", tmp_path / "gone.py")
        monkeypatch.setattr(ring, "_RING_MODULE", None)
        with pytest.raises(FileNotFoundError, match="gone.py"):
            ring.load_ring()

    def test_failed_exec_leaves_no_half_built_module(self, tmp_path,
                                                     monkeypatch):
        broken = tmp_path / "broken_ring.py"
        broken.write_text("raise RuntimeError('boom')\n")
        monkeypatch.setattr(ring, "RING_SCRIPT", broken)
        monkeypatch.setattr(ring, "RING_MODULE_NAME", "operational_ring_broken")
        monkeypatch.setattr(ring, "_RING_MODULE", None)
        with pytest.raises(RuntimeError, match="boom"):
            ring.load_ring()
        assert "operational_ring_broken" not in sys.modules


class TestReexports:
    """Every promised name is present, and is the script's own object."""

    @pytest.mark.parametrize("name", REEXPORTS)
    def test_present_on_the_adapter(self, name):
        assert hasattr(ring, name), f"{name} missing from the adapter"

    @pytest.mark.parametrize("name", REEXPORTS)
    def test_is_the_script_object(self, name):
        assert getattr(ring, name) is getattr(ring.load_ring(), name)

    @pytest.mark.parametrize("name", REEXPORTS)
    def test_listed_in_dunder_all(self, name):
        assert name in ring.__all__

    @pytest.mark.parametrize("name", [
        "infer_satellite", "quality_attrs", "satellite_available_times",
        "availability_band", "scan_interval", "time_tag", "sat_nc_path",
        "global_nc_path", "filter_to_common_times",
    ])
    def test_functions_are_callable(self, name):
        assert callable(getattr(ring, name))

    def test_global_mosaic_is_a_class_with_the_expected_api(self):
        assert isinstance(ring.GlobalMosaic, type)
        for method in ("add", "to_dataset"):
            assert callable(getattr(ring.GlobalMosaic, method))

    def test_constant_types(self):
        assert isinstance(ring.RING_SATELLITES, list)
        assert isinstance(ring.OUTPUT_VARS, list)
        assert isinstance(ring.DT_MINUTES, int)
        assert isinstance(ring.SCAN_INTERVAL_MINUTES, dict)


class TestConstants:
    """The constants the operational assets fan out over."""

    def test_ring_satellites_contents(self):
        assert ring.RING_SATELLITES == [
            "goes18", "goes19", "mtg-i1", "msg-iodc", "gk2a", "himawari9",
        ]

    def test_output_vars_contents(self):
        assert ring.OUTPUT_VARS == [
            "u_wind", "v_wind", "cloud_top_height",
            "quality_flag", "sigma_u", "sigma_v", "sigma_h",
        ]

    def test_dt_minutes(self):
        assert ring.DT_MINUTES == 10

    def test_scan_interval_overrides_only_seviri(self):
        assert ring.SCAN_INTERVAL_MINUTES == {"msg-iodc": 15}


class TestPureHelpers:
    """The offline helpers behave as the operational code expects."""

    def test_scan_interval(self):
        assert ring.scan_interval("msg-iodc") == 15
        for sat in ring.RING_SATELLITES:
            if sat != "msg-iodc":
                assert ring.scan_interval(sat) == ring.DT_MINUTES

    def test_time_tag(self):
        assert ring.time_tag(T0) == "20260801T0000"
        assert ring.time_tag(datetime(2026, 12, 31, 23, 50)) == "20261231T2350"

    def test_nc_paths_are_deterministic_and_distinct(self, tmp_path):
        sat = ring.sat_nc_path(tmp_path, "goes19", T0)
        glob = ring.global_nc_path(tmp_path, T0)
        assert isinstance(sat, Path) and isinstance(glob, Path)
        assert sat != glob
        assert sat == ring.sat_nc_path(tmp_path, "goes19", T0)
        assert "goes19" in sat.name
        assert ring.time_tag(T0) in sat.name
        assert ring.time_tag(T0) in glob.name
        assert tmp_path in sat.parents and tmp_path in glob.parents

    def test_global_mosaic_constructs_and_accumulates(self):
        mosaic = ring.GlobalMosaic(resolution_m=200_000.0)
        assert mosaic.add("goes18", _scene("goes18", zenith=10.0)) > 0
        ds = mosaic.to_dataset()
        for var in ring.OUTPUT_VARS:
            assert var in ds


class TestMissingNameGuard:
    """A name vanishing upstream fails loudly, not as a late AttributeError."""

    def test_raises_attribute_error_naming_the_symbol_and_script(self):
        empty = ModuleType("empty_ring")
        with pytest.raises(AttributeError) as exc:
            ring._require(empty, "infer_satellite")
        message = str(exc.value)
        assert "infer_satellite" in message
        assert str(ring.RING_SCRIPT) in message
        assert "operational/adapters/ring.py" in message

    def test_returns_the_attribute_when_present(self):
        stub = ModuleType("stub_ring")
        stub.DT_MINUTES = 7
        assert ring._require(stub, "DT_MINUTES") == 7


def _scene(sat_id: str, zenith: float, ny: int = 16, nx: int = 16):
    """A tiny synthetic AMV scene, shaped like ``infer_satellite`` output."""
    lat, lon = np.meshgrid(
        np.linspace(-20, 20, ny), np.linspace(-30, 30, nx), indexing="ij",
    )
    data = {k: np.full((ny, nx), 1.0, np.float32) for k in ring.OUTPUT_VARS}
    data["quality_flag"] = np.full((ny, nx), 2.0, np.float32)
    return xr.Dataset(
        {k: (("y", "x"), data[k]) for k in ring.OUTPUT_VARS},
        coords={
            "latitude": (("y", "x"), lat.astype(np.float32)),
            "longitude": (("y", "x"), lon.astype(np.float32)),
            "zenith_angle": (
                ("y", "x"), np.full((ny, nx), zenith, np.float32),
            ),
        },
        attrs={"satellite_id": sat_id, "time": str(T0)},
    )
