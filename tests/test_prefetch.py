"""Tests for the scene cache and prefetcher that keep the GPU fed."""

import gc
import importlib.util
import sys
import threading
import time
import weakref
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

BASE = Path(__file__).resolve().parent.parent


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "ring_prefetch", BASE / "scripts" / "infer_student_global_ring.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ring = _load_script()
T0 = datetime(2026, 8, 1, 12, 0)
MB = 2**20


def _scene(nbytes=MB):
    return (np.zeros(nbytes // 4, dtype=np.float32), "cfg")


def _fake_scene(sat_id, t0, n=6):
    """A tiny per-satellite AMV dataset, as infer_satellite would return."""
    lat, lon = np.meshgrid(np.linspace(-10, 10, n), np.linspace(-10, 10, n),
                           indexing="ij")
    data = {v: np.full((n, n), 1.0, np.float32) for v in ring.OUTPUT_VARS}
    data["quality_flag"] = np.full((n, n), 2.0, np.float32)
    return xr.Dataset(
        {v: (("y", "x"), data[v]) for v in ring.OUTPUT_VARS},
        coords={"latitude": (("y", "x"), lat.astype(np.float32)),
                "longitude": (("y", "x"), lon.astype(np.float32)),
                "zenith_angle": (("y", "x"), np.full((n, n), 10.0, np.float32))},
        attrs={"satellite_id": sat_id, "time": str(t0)})


class TestSceneCache:
    def test_hit_and_miss(self):
        cache = ring.SceneCache(10 * MB)
        key = ("goes19", "C14", T0)
        assert cache.get(key) is None
        cache.put(key, _scene())
        assert cache.get(key) is not None
        assert (cache.hits, cache.misses) == (1, 1)

    def test_evicts_least_recently_used(self):
        cache = ring.SceneCache(2 * MB)
        a, b, c = [("s", "C14", T0 + timedelta(minutes=i)) for i in range(3)]
        cache.put(a, _scene())
        cache.put(b, _scene())
        cache.get(a)                 # a is now the most recent
        cache.put(c, _scene())       # evicts b
        assert cache.get(a) is not None
        assert cache.get(b) is None
        assert cache.get(c) is not None

    def test_respects_the_byte_budget(self):
        cache = ring.SceneCache(4 * MB)
        for i in range(20):
            cache.put(("s", "C14", T0 + timedelta(minutes=i)), _scene())
        assert cache.nbytes <= 4 * MB

    def test_oversized_scene_is_not_cached(self):
        cache = ring.SceneCache(MB // 2)
        cache.put(("s", "C14", T0), _scene(MB))
        assert cache.nbytes == 0

    def test_zero_budget_disables_caching(self):
        cache = ring.SceneCache(0)
        cache.put(("s", "C14", T0), _scene())
        assert cache.get(("s", "C14", T0)) is None

    def test_is_thread_safe(self):
        cache = ring.SceneCache(64 * MB)

        def hammer(n):
            for i in range(50):
                cache.put((f"s{n}", "C14", T0 + timedelta(minutes=i)), _scene())
                cache.get((f"s{n}", "C14", T0))

        threads = [threading.Thread(target=hammer, args=(n,)) for n in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert cache.nbytes <= 64 * MB


class TestScenePrefetcher:
    @pytest.fixture
    def loads(self, monkeypatch):
        calls = []
        lock = threading.Lock()

        def fake_load(sat_id, band, t):
            with lock:
                calls.append((sat_id, band, t))
            time.sleep(0.05)            # stand in for the object-store wait
            return _scene()

        monkeypatch.setattr(ring, "_load_scene", fake_load)
        return calls

    def test_get_returns_the_scene(self, loads):
        p = ring.ScenePrefetcher(ring.SceneCache(64 * MB), max_workers=2)
        data, cfg = p.get("goes19", "C14", T0)
        assert isinstance(data, np.ndarray) and cfg == "cfg"
        p.shutdown()

    def test_each_scene_is_loaded_once(self, loads):
        p = ring.ScenePrefetcher(ring.SceneCache(64 * MB), max_workers=4)
        key = ("goes19", "C14", T0)
        p.submit([key, key, key])
        p.get(*key)
        p.get(*key)                     # second read comes from the cache
        assert len(loads) == 1
        p.shutdown()

    def test_prefetch_overlaps_the_wait(self, loads):
        """Eight scenes, four workers: wall clock must beat loading serially."""
        keys = [("goes19", f"C{i:02d}", T0) for i in range(8)]
        p = ring.ScenePrefetcher(ring.SceneCache(64 * MB), max_workers=4)
        p.submit(keys)
        start = time.perf_counter()
        for key in keys:
            p.get(*key)
        elapsed = time.perf_counter() - start
        assert len(loads) == 8
        assert elapsed < 8 * 0.05, f"no overlap: {elapsed:.2f}s"
        p.shutdown()

    def test_inline_when_not_prefetched(self, loads):
        p = ring.ScenePrefetcher(ring.SceneCache(64 * MB), max_workers=2)
        p.get("goes19", "C14", T0)
        assert len(loads) == 1
        p.shutdown()

    def test_loader_errors_reach_the_caller(self, monkeypatch):
        def boom(sat_id, band, t):
            raise RuntimeError("store unreachable")

        monkeypatch.setattr(ring, "_load_scene", boom)
        p = ring.ScenePrefetcher(ring.SceneCache(MB), max_workers=2)
        p.submit([("goes19", "C14", T0)])
        with pytest.raises(RuntimeError, match="store unreachable"):
            p.get("goes19", "C14", T0)
        p.shutdown()


class TestSceneRequests:
    def test_covers_the_triplet_for_every_flow_band(self):
        reqs = ring.scene_requests("goes19", T0, ["C08", "C10"], [])
        assert len(reqs) == 6
        assert {t for _, _, t in reqs} == {
            T0 - timedelta(minutes=10), T0, T0 + timedelta(minutes=10)}

    def test_rad_bands_only_need_t0(self):
        reqs = ring.scene_requests("goes19", T0, [], ["C07", "C08"])
        assert [(b, t) for _, b, t in reqs] == [("C07", T0), ("C08", T0)]

    def test_no_duplicates_between_flow_and_rad(self):
        reqs = ring.scene_requests("goes19", T0, ["C08"], ["C07", "C08"])
        assert len(reqs) == len(set(reqs))
        assert len(reqs) == 4          # C08 triplet + C07 at t0

    def test_uses_the_satellites_own_cadence(self):
        reqs = ring.scene_requests("msg-iodc", T0, ["C08"], [])
        assert {t for _, _, t in reqs} == {
            T0 - timedelta(minutes=15), T0, T0 + timedelta(minutes=15)}

    def test_skips_bands_the_satellite_lacks(self):
        reqs = ring.scene_requests("msg-iodc", T0, ["C04"], ["C06"])
        assert reqs == []

    def test_matches_what_the_builder_asks_for(self, monkeypatch):
        """The request list must not drift from the real load order."""
        asked = []

        def fake_load(sat_id, band, t):
            asked.append((sat_id, band, t))
            from dataclasses import replace
            cfg = replace(ring.SATELLITE_CONFIGS["goes19"], n_rows=8, n_cols=8)
            return np.zeros((8, 8), np.float32), cfg

        class FakeDisp:
            def _run_pair(self, a, b):
                return np.zeros((2, 8, 8), np.float32)

        monkeypatch.setattr(ring, "_load_scene", fake_load)
        flow, rad = ["C08", "C10"], ["C07", "C08"]
        ring._build_input_stack("goes19", T0, FakeDisp(), flow, rad)
        assert set(asked) == set(ring.scene_requests("goes19", T0, flow, rad))


class TestMemorySizing:
    def test_cache_is_sized_below_free_memory(self):
        """Under what is free, and under a share of the box, whichever is less."""
        from stereo_winds.readers._cache import (
            MAX_CACHE_FRACTION,
            available_memory_bytes,
            default_scene_cache_bytes,
            total_memory_bytes,
        )
        available = available_memory_bytes()
        total = total_memory_bytes()
        assert available and total
        limit = default_scene_cache_bytes(reserve=2**30)
        assert limit < available, "cache must stay under what is free"
        assert limit <= max(2**30, int(total * MAX_CACHE_FRACTION))

    def test_explicit_override(self, monkeypatch):
        from stereo_winds.readers._cache import default_scene_cache_bytes
        monkeypatch.setenv("STEREO_WINDS_SCENE_CACHE_GB", "3")
        assert default_scene_cache_bytes() == 3 * 2**30

    def test_never_goes_negative_on_a_full_box(self, monkeypatch):
        import stereo_winds.readers._cache as cache_mod
        monkeypatch.delenv("STEREO_WINDS_SCENE_CACHE_GB", raising=False)
        monkeypatch.setattr(cache_mod, "available_memory_bytes",
                            lambda: 2 * 2**30)
        assert cache_mod.default_scene_cache_bytes(reserve=12 * 2**30) == 2**30


# ── Memory safety ─────────────────────────────────────────────────────

class TestCacheUsefulness:
    """A cache that cannot be hit must not be allowed to fill memory."""

    RING = ["goes18", "goes19", "mtg-i1", "msg-iodc", "gk2a", "himawari9"]

    @pytest.mark.parametrize("step, expected", [
        (10, True),    # t+10 of one timestamp is t0 of the next
        (20, True),    # exactly twice the ABI scan interval
        (30, True),    # twice SEVIRI's 15 min
        (31, False),
        (60, False),
        (360, False),  # the six-hourly run that was being OOM-killed
    ])
    def test_reuse_follows_the_scan_interval(self, step, expected):
        assert ring.scene_cache_is_useful(step, self.RING) is expected

    def test_a_single_satellite_uses_its_own_cadence(self):
        assert ring.scene_cache_is_useful(30, ["msg-iodc"]) is True
        assert ring.scene_cache_is_useful(30, ["goes19"]) is False


class TestCacheSizing:
    def test_never_exceeds_the_share_of_total_ram(self, monkeypatch):
        import stereo_winds.readers._cache as cache_mod
        monkeypatch.delenv("STEREO_WINDS_SCENE_CACHE_GB", raising=False)
        # A box that looks almost entirely free.
        monkeypatch.setattr(cache_mod, "total_memory_bytes", lambda: 64 * 2**30)
        monkeypatch.setattr(cache_mod, "available_memory_bytes",
                            lambda: 62 * 2**30)
        limit = cache_mod.default_scene_cache_bytes(reserve=2 * 2**30)
        assert limit <= 64 * 2**30 * cache_mod.MAX_CACHE_FRACTION
        assert limit < 62 * 2**30 - 2 * 2**30, "the fraction cap did not apply"

    def test_reserve_still_applies_on_a_busy_box(self, monkeypatch):
        import stereo_winds.readers._cache as cache_mod
        monkeypatch.delenv("STEREO_WINDS_SCENE_CACHE_GB", raising=False)
        monkeypatch.setattr(cache_mod, "total_memory_bytes", lambda: 64 * 2**30)
        monkeypatch.setattr(cache_mod, "available_memory_bytes",
                            lambda: 10 * 2**30)
        assert cache_mod.default_scene_cache_bytes(reserve=8 * 2**30) == 2 * 2**30


class TestMemoryPressure:
    def test_shrinks_when_memory_runs_low(self, monkeypatch):
        cache = ring.SceneCache(16 * MB)
        cache._PRESSURE_CHECK_EVERY = 1
        cache._SHRINK_FLOOR = MB
        monkeypatch.setattr(ring, "available_memory_bytes",
                            lambda: ring.MEMORY_PRESSURE_FLOOR // 2)
        for i in range(4):
            cache.put(("s", "C14", T0 + timedelta(minutes=i)), _scene())
        assert cache.limit < cache.max_bytes
        assert cache.shrinks > 0

    def test_stays_within_the_lowered_limit(self, monkeypatch):
        cache = ring.SceneCache(64 * MB)
        cache._PRESSURE_CHECK_EVERY = 1
        cache._SHRINK_FLOOR = MB
        monkeypatch.setattr(ring, "available_memory_bytes",
                            lambda: ring.MEMORY_PRESSURE_FLOOR // 2)
        for i in range(40):
            cache.put(("s", "C14", T0 + timedelta(minutes=i)), _scene())
        assert cache.nbytes <= cache.limit

    def test_does_not_shrink_when_memory_is_fine(self, monkeypatch):
        cache = ring.SceneCache(16 * MB)
        cache._PRESSURE_CHECK_EVERY = 1
        monkeypatch.setattr(ring, "available_memory_bytes",
                            lambda: 100 * 2**30)
        for i in range(4):
            cache.put(("s", "C14", T0 + timedelta(minutes=i)), _scene())
        assert cache.limit == cache.max_bytes and cache.shrinks == 0

    def test_never_shrinks_below_the_floor(self, monkeypatch):
        cache = ring.SceneCache(8 * 2**30)
        cache._PRESSURE_CHECK_EVERY = 1
        monkeypatch.setattr(ring, "available_memory_bytes", lambda: 0)
        for i in range(40):
            cache.put(("s", "C14", T0 + timedelta(minutes=i)), _scene())
        assert cache.limit >= 2**30

    def test_recovers_when_memory_frees_up(self, monkeypatch):
        cache = ring.SceneCache(16 * MB)
        cache._PRESSURE_CHECK_EVERY = 1
        cache._SHRINK_FLOOR = MB
        monkeypatch.setattr(ring, "available_memory_bytes",
                            lambda: ring.MEMORY_PRESSURE_FLOOR // 2)
        cache.put(("s", "C14", T0), _scene())
        lowered = cache.limit
        monkeypatch.setattr(ring, "available_memory_bytes",
                            lambda: 100 * 2**30)
        for i in range(6):
            cache.put(("s", "C14", T0 + timedelta(minutes=i + 1)), _scene())
        assert cache.limit > lowered


# ── Prefetch retention ────────────────────────────────────────────────

class TestPrefetchRelease:
    """Queued-but-uncollected scenes must not be held for the whole run."""

    @pytest.fixture
    def tracked(self, monkeypatch):
        refs = []

        def fake_load(sat_id, band, t):
            arr = np.zeros(1024, dtype=np.float32)
            refs.append(weakref.ref(arr))
            return (arr, "cfg")

        monkeypatch.setattr(ring, "_load_scene", fake_load)
        return refs

    def _settle(self, pre):
        for key in list(pre._inflight):
            pre._inflight[key].exception()   # wait without consuming
        gc.collect()

    def test_reset_releases_unread_scenes(self, tracked):
        """Regression: a skipped satellite left ~2.4 GB per timestamp behind."""
        pre = ring.ScenePrefetcher(ring.SceneCache(0), max_workers=2)
        pre.submit([("goes19", f"C{i:02d}", T0) for i in range(8)])
        self._settle(pre)
        assert any(r() is not None for r in tracked), "nothing was held to release"

        pre.reset()
        gc.collect()
        assert pre.inflight == 0
        assert not [r for r in tracked if r() is not None]
        pre.shutdown()

    def test_inflight_is_bounded(self, tracked):
        pre = ring.ScenePrefetcher(ring.SceneCache(0), max_workers=2)
        pre._MAX_INFLIGHT = 8
        for step in range(5):
            when = T0 + timedelta(minutes=360 * step)
            pre.submit([("goes19", f"C{i:02d}", when) for i in range(8)])
        assert pre.inflight <= 8
        assert pre.dropped > 0
        pre.shutdown()

    def test_reset_keeps_the_cache(self, tracked):
        cache = ring.SceneCache(64 * MB)
        pre = ring.ScenePrefetcher(cache, max_workers=2)
        key = ("goes19", "C14", T0)
        pre.get(*key)                        # loaded and cached
        pre.reset()
        assert cache.get(key) is not None, "reset must not empty the cache"
        pre.shutdown()

    def test_reset_is_safe_when_idle(self):
        pre = ring.ScenePrefetcher(ring.SceneCache(MB), max_workers=2)
        assert pre.reset() == 0
        pre.shutdown()


class TestSkipAwarePrefetch:
    """Do not queue reads for a satellite whose output already exists."""

    def test_existing_output_is_not_prefetched(self, tmp_path):
        path = ring.sat_nc_path(tmp_path, "goes19", T0)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")
        assert not ring._needs_inference("goes19", T0, tmp_path,
                                         skip_existing=True, write_netcdf=True)

    def test_missing_output_is_prefetched(self, tmp_path):
        assert ring._needs_inference("goes19", T0, tmp_path,
                                     skip_existing=True, write_netcdf=True)

    def test_without_skip_existing_everything_is_prefetched(self, tmp_path):
        path = ring.sat_nc_path(tmp_path, "goes19", T0)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")
        assert ring._needs_inference("goes19", T0, tmp_path,
                                     skip_existing=False, write_netcdf=True)

    def test_process_time_releases_across_timestamps(self, tmp_path, monkeypatch):
        """Ten timestamps, one satellite skipped each: nothing may accumulate."""
        refs = []

        def fake_load(sat_id, band, t):
            arr = np.zeros(1024, dtype=np.float32)
            refs.append(weakref.ref(arr))
            return (arr, "cfg")

        monkeypatch.setattr(ring, "_load_scene", fake_load)
        monkeypatch.setattr(
            ring, "infer_satellite",
            lambda sat_id, t0, *a, **k: _fake_scene(sat_id, t0))

        pre = ring.ScenePrefetcher(ring.SceneCache(0), max_workers=2)
        sats = ["goes18", "goes19"]
        for step in range(10):
            t = T0 + timedelta(minutes=360 * step)
            # goes18's output already exists -> it is skipped, and its
            # prefetched scenes are never collected.
            path = ring.sat_nc_path(tmp_path, "goes18", t)
            path.parent.mkdir(parents=True, exist_ok=True)
            _fake_scene("goes18", t).to_netcdf(path)
            ring.process_time(t, sats, None, None, ["C14"], ["C14"], tmp_path,
                              resolution_m=200_000.0, skip_existing=True,
                              prefetcher=pre)
        gc.collect()
        assert pre.inflight == 0
        assert not [r for r in refs if r() is not None], "scenes accumulated"
        pre.shutdown()
