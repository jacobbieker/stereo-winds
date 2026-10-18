"""Preallocated per-satellite stores and their parallel slot filling (CPU, local icechunk)."""

import importlib.util
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

icechunk = pytest.importorskip("icechunk")

from stereo_winds import backfill_store as bs  # noqa: E402
from stereo_winds.icechunk_output import open_icechunk_repo  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
T0 = datetime(2025, 5, 1, 0, 0)
NY, NX = 8, 6


def _scene(t0: datetime, value: float, sat_id: str = "goes18") -> xr.Dataset:
    lat, lon = np.meshgrid(np.linspace(-20, 20, NY), np.linspace(-30, 30, NX), indexing="ij")
    data = {k: np.full((NY, NX), value, np.float32) for k in bs.DEFAULT_FIELDS}
    data["quality_flag"] = np.full((NY, NX), 2.0, np.float32)
    return xr.Dataset(
        {k: (("y", "x"), v) for k, v in data.items()},
        coords={
            "latitude": (("y", "x"), lat.astype(np.float32)),
            "longitude": (("y", "x"), lon.astype(np.float32)),
            "zenith_angle": (("y", "x"), np.full((NY, NX), 30.0, np.float32)),
        },
        attrs={
            "satellite_id": sat_id,
            "time": str(t0),
            "sub_satellite_longitude": -137.0 + value * 1e-3,
            "quality_degraded": 0,
            "quality_note": "all requested bands available",
            "n_bands_missing": 0,
        },
    )


def _repo(tmp_path, name="amv_goes18.icechunk"):
    return open_icechunk_repo(str(tmp_path / name))


def _prealloc(repo, hours=6, chunk=4):
    return bs.preallocate_satellite_store(
        repo, "goes18", T0, T0 + timedelta(hours=hours - 1), 60, chunk=chunk, shape=(NY, NX)
    )


def _open(repo):
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False)


class TestPreallocate:
    def test_lays_out_empty_slots(self, tmp_path):
        repo = _repo(tmp_path)
        assert _prealloc(repo) == "created"
        ds = _open(repo)
        assert dict(ds.sizes) == {"time": 6, "y": NY, "x": NX}
        assert ds.time.values[0] == np.datetime64(T0, "ns")
        assert np.all(np.diff(ds.time.values) == np.timedelta64(1, "h"))
        assert ds.attrs[bs.LAYOUT_ATTR] == bs.LAYOUT
        assert (ds[bs.FILLED_VAR].values == 0).all()
        assert np.isnan(ds.u_wind.values).all()
        assert np.isnan(ds.latitude.values).all()
        assert list(ds.scan_time.values) == [""] * 6
        # One slot per chunk, the 1-D per-scan variables included.
        assert ds.u_wind.encoding["chunks"][0] == 1
        assert ds.scan_time.encoding["chunks"] == (1,)
        assert ds[bs.FILLED_VAR].encoding["chunks"] == (1,)

    def test_writes_no_field_chunks(self, tmp_path):
        repo = _repo(tmp_path)
        _prealloc(repo)
        session = repo.readonly_session("main")
        import zarr

        group = zarr.open_group(session.store, mode="r")
        assert group["u_wind"].nchunks_initialized == 0

    def test_is_idempotent(self, tmp_path):
        repo = _repo(tmp_path)
        _prealloc(repo)
        assert _prealloc(repo) == "exists"

    def test_refuses_a_different_layout(self, tmp_path):
        repo = _repo(tmp_path)
        _prealloc(repo, hours=6)
        with pytest.raises(ValueError, match="refusing"):
            _prealloc(repo, hours=7)

    def test_default_shape_is_the_native_grid(self):
        assert bs._grid_shape("goes18") == (5424, 5424)
        assert bs._grid_shape("mtg-i1") == (5568, 5568)


class TestFillSlot:
    def test_fills_its_slot_only(self, tmp_path):
        repo = _repo(tmp_path)
        _prealloc(repo)
        t = T0 + timedelta(hours=2)
        assert bs.fill_slot(repo, _scene(t, 5.0), t) == "filled"
        ds = _open(repo)
        assert ds[bs.FILLED_VAR].values.tolist() == [0, 0, 1, 0, 0, 0]
        assert (ds.u_wind.isel(time=2).values == 5.0).all()
        assert np.isnan(ds.u_wind.isel(time=[0, 1, 3, 4, 5]).values).all()
        assert ds.scan_time.values[2] == str(t)
        assert ds.scan_time.values[1] == ""
        assert ds.quality_note.values[2] == "all requested bands available"
        assert ds.sub_satellite_longitude.values[2] == pytest.approx(-137.0 + 5e-3)
        assert np.isnan(ds.sub_satellite_longitude.values[0])
        # The static grid arrived with the first scan.
        np.testing.assert_allclose(ds.latitude.values, _scene(t, 0).latitude.values)
        assert (ds.zenith_angle.values == 30.0).all()

    def test_refill_is_skipped(self, tmp_path):
        repo = _repo(tmp_path)
        _prealloc(repo)
        assert bs.fill_slot(repo, _scene(T0, 1.0), T0) == "filled"
        before = repo.lookup_branch("main")
        assert bs.fill_slot(repo, _scene(T0, 9.0), T0) == "skipped"
        assert repo.lookup_branch("main") == before  # no new commit
        assert (_open(repo).u_wind.isel(time=0).values == 1.0).all()

    def test_replace_overwrites(self, tmp_path):
        repo = _repo(tmp_path)
        _prealloc(repo)
        bs.fill_slot(repo, _scene(T0, 1.0), T0)
        assert bs.fill_slot(repo, _scene(T0, 9.0), T0, replace=True) == "replaced"
        assert (_open(repo).u_wind.isel(time=0).values == 9.0).all()

    def test_time_outside_the_axis_raises(self, tmp_path):
        repo = _repo(tmp_path)
        _prealloc(repo)
        t = T0 + timedelta(days=3)
        with pytest.raises(KeyError):
            bs.fill_slot(repo, _scene(t, 1.0), t)

    def test_needs_a_preallocated_store(self, tmp_path):
        from stereo_winds.icechunk_output import write_satellite_to_icechunk

        repo = _repo(tmp_path)
        write_satellite_to_icechunk(repo, _scene(T0, 1.0), T0)
        with pytest.raises(ValueError, match="preallocate"):
            bs.fill_slot(repo, _scene(T0, 1.0), T0)

    def test_wrong_grid_raises(self, tmp_path):
        repo = _repo(tmp_path)
        _prealloc(repo)
        small = _scene(T0, 1.0).isel(y=slice(0, 4))
        with pytest.raises(ValueError, match="slots are"):
            bs.fill_slot(repo, small, T0)


class _BarrierRepo:
    """A repository whose first ``writable_session`` per thread waits for all
    the others, so every writer starts from the same snapshot and every
    commit after the first has to rebase.

    Commits themselves are serialised: icechunk's local-filesystem storage
    has no conditional writes, so two commits landing at the same instant
    can lose one (it says so in a warning).  On S3 that race is settled by
    conditional PUTs; what is under test here is the rebase-and-retry
    logic, which still has to cope with every other writer's commit."""

    def __init__(self, repo, n):
        self._repo = repo
        self._barrier = threading.Barrier(n)
        self._local = threading.local()
        self.commits = 0
        self._lock = threading.Lock()

    def writable_session(self, branch):
        if not getattr(self._local, "waited", False):
            self._local.waited = True
            self._barrier.wait(timeout=60)
        session = self._repo.writable_session(branch)
        outer = self
        commit = session.commit

        def counted(*a, **k):
            with outer._lock:
                out = commit(*a, **k)
                outer.commits += 1
            return out

        session.commit = counted
        return session

    def __getattr__(self, name):
        return getattr(self._repo, name)


class TestConcurrentFill:
    def test_disjoint_slots_from_many_writers(self, tmp_path):
        repo = _repo(tmp_path)
        _prealloc(repo, hours=8)
        slots = [0, 1, 2, 4, 5, 7]  # 3 and 6 stay empty, as missing scans would
        shared = _BarrierRepo(repo, len(slots))

        def work(i):
            t = T0 + timedelta(hours=i)
            return bs.fill_slot(shared, _scene(t, float(i + 1)), t, backoff_s=0.01)

        with ThreadPoolExecutor(len(slots)) as pool:
            outcomes = list(pool.map(work, slots))
        assert outcomes == ["filled"] * len(slots)
        assert shared.commits == len(slots)

        ds = _open(repo)
        filled = ds[bs.FILLED_VAR].values
        assert filled.tolist() == [1, 1, 1, 0, 1, 1, 0, 1]
        for i in range(8):
            u = ds.u_wind.isel(time=i).values
            if i in slots:
                assert (u == i + 1).all(), i
                assert ds.scan_time.values[i] == str(T0 + timedelta(hours=i))
            else:
                assert np.isnan(u).all()
                assert ds.scan_time.values[i] == ""
        # The grid every writer tried to write is there once and intact.
        np.testing.assert_allclose(ds.latitude.values, _scene(T0, 0).latitude.values)

    def test_idempotent_retry_after_partial_run(self, tmp_path):
        repo = _repo(tmp_path)
        _prealloc(repo, hours=4)
        for i in (0, 1):
            t = T0 + timedelta(hours=i)
            bs.fill_slot(repo, _scene(t, 1.0), t)
        # The retried job runs its whole range again.
        outcomes = [
            bs.fill_slot(repo, _scene(T0 + timedelta(hours=i), 2.0), T0 + timedelta(hours=i))
            for i in range(4)
        ]
        assert outcomes == ["skipped", "skipped", "filled", "filled"]
        u = _open(repo).u_wind.values[:, 0, 0]
        assert u.tolist() == [1.0, 1.0, 2.0, 2.0]


# ── The ring script's --fill-slots path ───────────────────────────────


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "infer_student_global_ring_bf", BASE / "scripts" / "infer_student_global_ring.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ring():
    return _load_script()


@pytest.fixture
def stub_infer(ring, monkeypatch):
    calls = []

    def fake(sat_id, t0, *a, **k):
        calls.append((sat_id, t0))
        return _scene(t0, 3.0, sat_id)

    monkeypatch.setattr(ring, "infer_satellite", fake)
    return calls


def _run(ring, t, out, stores, **kw):
    return ring.process_time(
        t, ["goes18"], None, None, ["C14"], ["C14"], out,
        skip_global=True, write_netcdf=False, sat_stores=stores, **kw,
    )  # fmt: skip


class TestScriptFillSlots:
    def _stores(self, ring, tmp_path, hours=3):
        stores = ring.SatelliteStores(str(tmp_path / "amv"), fill_slots=True)
        bs.preallocate_satellite_store(
            stores.repo("goes18"), "goes18", T0, T0 + timedelta(hours=hours - 1), 60,
            shape=(NY, NX), chunk=4,
        )  # fmt: skip
        return stores

    def test_fills_and_skips_filled_slots_before_inference(self, ring, tmp_path, stub_infer):
        stores = self._stores(ring, tmp_path)
        assert _run(ring, T0, tmp_path / "out", stores) is True
        assert stub_infer == [("goes18", T0)]
        # A fresh run (a retried job) does not retrieve it again.
        stores = ring.SatelliteStores(str(tmp_path / "amv"), fill_slots=True)
        assert _run(ring, T0, tmp_path / "out", stores) is True
        t1 = T0 + timedelta(hours=1)
        assert _run(ring, t1, tmp_path / "out", stores) is True
        assert stub_infer == [("goes18", T0), ("goes18", t1)]
        ds = _open(stores.repo("goes18"))
        assert ds[bs.FILLED_VAR].values.tolist() == [1, 1, 0]
        assert not stores.publish_failures

    def test_skip_existing_does_not_mistake_empty_slots_for_scans(
        self, ring, tmp_path, stub_infer
    ):
        """Every slot is on the time axis from the start; --skip-existing's
        per-satellite check must not read that as "already published"."""
        stores = self._stores(ring, tmp_path)
        assert _run(ring, T0, tmp_path / "out", stores, skip_existing=True) is True
        assert stub_infer == [("goes18", T0)]
        assert _open(stores.repo("goes18"))[bs.FILLED_VAR].values.tolist() == [1, 0, 0]

    def test_a_time_outside_the_store_is_reported_not_retrieved(self, ring, tmp_path, stub_infer):
        stores = self._stores(ring, tmp_path)
        t = T0 + timedelta(days=2)
        assert _run(ring, t, tmp_path / "out", stores) is False
        assert stub_infer == []
        assert stores.publish_failures == [f"goes18@{ring.time_tag(t)}"]

    def test_a_failed_write_is_reported(self, ring, tmp_path, stub_infer, monkeypatch):
        stores = self._stores(ring, tmp_path)

        def boom(*a, **k):
            raise RuntimeError("S3 down")

        monkeypatch.setattr(ring, "fill_slot", boom)
        _run(ring, T0, tmp_path / "out", stores)
        assert stores.publish_failures == [f"goes18@{ring.time_tag(T0)}"]

    def test_append_mode_unchanged(self, ring, tmp_path, stub_infer):
        stores = ring.SatelliteStores(str(tmp_path / "amv"))
        _run(ring, T0, tmp_path / "out", stores)
        ds = _open(stores.repo("goes18"))
        assert ds.sizes["time"] == 1
        assert bs.FILLED_VAR not in ds
