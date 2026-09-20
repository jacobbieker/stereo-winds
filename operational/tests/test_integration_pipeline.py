"""End-to-end test of the operational graph on synthetic scenes.

Everything below the science is the real thing: the real Dagster assets,
the real mosaic, the real icechunk publication into a real store on
disk.  Only ``infer_satellite`` — the one step that would need a GPU,
checkpoints and the network — is replaced, by a generator that returns
datasets with the schema the real one produces.

The store the test writes is then read back the way a consumer would,
which is where the interesting failures live: a time axis re-encoded on
append, source codes that mean a different satellite at each timestep,
or quality attributes that never made it out of the mosaic.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from dagster import AssetsDefinition, materialize

from operational.adapters import ring
from operational.assets import amv_assets, mosaic_assets
from operational.core import amv as core_amv
from operational.core import publish as core_publish
from operational.core.partitions import time_for
from operational.resources import (
    IcechunkStoreResource,
    ModelResource,
    PathsResource,
    RunSettingsResource,
)
from operational.tests.conftest import (
    AMV_VARS,
    HALF_WIDTH_DEG,
    SUB_LON_DEG,
    synthetic_scene,
    synthetic_wind,
)

pytestmark = pytest.mark.integration

#: Satellites the test runs.  gk2a and himawari9 overlap, which is the
#: contested region the mosaic has to resolve.
SATELLITES = ["goes18", "goes19", "gk2a", "himawari9"]

#: Viewing zenith per satellite.  gk2a sees the overlap from closer to
#: its sub-point, so it must win every cell the two share.
ZENITH = {"goes18": 10.0, "goes19": 12.0, "gk2a": 5.0, "himawari9": 30.0}

#: goes18 loses three of eight bands — over the quarter that counts as
#: degraded, so the flag has to travel all the way into the store.
DEGRADED_SAT = "goes18"
DEGRADED_BANDS = ("C08", "C09", "C10")

#: A point inside the gk2a / himawari9 overlap.
OVERLAP_LAT = 0.0
OVERLAP_LON = 135.0

#: Coarse grid: ~100 x 200 cells, which keeps the whole test in seconds.
RESOLUTION_M = 200_000.0

SCENE_SHAPE = (64, 64)


def _amv_assets_for(sat_ids: list[str]) -> list[AssetsDefinition]:
    """The built per-satellite assets, in the order given."""
    by_sat = getattr(amv_assets, "AMV_ASSETS_BY_SAT", None)
    if by_sat is not None:
        missing = [s for s in sat_ids if s not in by_sat]
        if missing:
            raise AssertionError(
                f"No AMV asset was built for {missing}; the operational "
                f"config must cover every satellite under test"
            )
        return [by_sat[s] for s in sat_ids]
    partitions_def = getattr(amv_assets, "AMV_PARTITIONS_DEF", None)
    return [amv_assets.build_amv_asset(s, partitions_def) for s in sat_ids]


def _fake_infer(calls: list[tuple[str, datetime]]):
    """A stand-in for ``infer_satellite`` that records what it was asked.

    Accepts the real signature so the production call site is exercised
    unchanged; extra keywords are tolerated for the same reason.
    """

    def infer_satellite(
        sat_id, t0, model, disp, flow_bands, rad_bands, device="cuda", row_strip=1024, **kwargs
    ):
        calls.append((sat_id, t0))
        ny, nx = SCENE_SHAPE
        ds = synthetic_scene(
            sat_id,
            t0,
            ny=ny,
            nx=nx,
            zenith=ZENITH.get(sat_id, 10.0),
            bands_missing=DEGRADED_BANDS if sat_id == DEGRADED_SAT else (),
        )
        # The shared fixture varies the winds smoothly across the disk.
        # These tests need one constant value per satellite instead, so
        # a mosaic cell can be traced back to the satellite that won it.
        u, v = synthetic_wind(sat_id)
        ds["u_wind"].values[...] = u
        ds["v_wind"].values[...] = v
        return ds

    return infer_satellite


def _partition_keys(partitions_def, n: int) -> list[str]:
    """``n`` consecutive partition keys whose windows have closed."""
    keys = partitions_def.get_partition_keys()
    assert len(keys) > n, (
        f"The partitions definition offers only {len(keys)} key(s); " f"the test needs {n + 1}"
    )
    # The last window may still be open at wall-clock time; step back off it.
    return keys[-(n + 1) : -1]


def _open_store(uri: str) -> xr.Dataset:
    """Read the published store the way a consumer would."""
    repo = core_publish.open_store(uri)
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False)


def _commit_messages(uri: str, branch: str = "main") -> list[str]:
    """Commit messages on ``branch``, newest first."""
    repo = core_publish.open_store(uri)
    return [snapshot.message for snapshot in repo.ancestry(branch=branch)]


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Run the whole graph over two partitions, once for all the tests."""
    pytest.importorskip("icechunk")
    from _pytest.monkeypatch import MonkeyPatch

    tmp_path = tmp_path_factory.mktemp("operational_e2e")
    out_dir = tmp_path / "output"
    store_uri = str(tmp_path / "amv.icechunk")

    calls: list[tuple[str, datetime]] = []
    fake = _fake_infer(calls)
    patcher = MonkeyPatch()
    # The science is the only seam: patch it where the retrieval reaches
    # for it, and where a direct import would have bound it.
    patcher.setattr(ring, "infer_satellite", fake)
    if hasattr(core_amv, "infer_satellite"):
        patcher.setattr(core_amv, "infer_satellite", fake, raising=False)
    # The asset resolves the model before it calls the retrieval, so
    # patching only the retrieval still demands a real checkpoint.
    # The fake ignores the model it is handed, so None is enough.
    patcher.setattr(ModelResource, "model", lambda self: None)
    patcher.setattr(ModelResource, "disparity", lambda self: None)

    try:
        assets = [
            *_amv_assets_for(SATELLITES),
            mosaic_assets.global_mosaic,
            mosaic_assets.published_mosaic,
        ]
        partitions_def = mosaic_assets.published_mosaic.partitions_def
        assert partitions_def is not None, "published_mosaic is not partitioned"
        keys = _partition_keys(partitions_def, 2)

        resources = {
            "paths": PathsResource(output_dir=str(out_dir)),
            "store": IcechunkStoreResource(
                store_uri=store_uri,
                branch="main",
                chunk=1024,
            ),
            "model": ModelResource(student_ckpt="", raft_ckpt="", device="cpu"),
            "run_settings": RunSettingsResource(
                satellites=list(SATELLITES),
                resolution_m=RESOLUTION_M,
                row_strip=64,
            ),
        }

        # Creating the store is itself a commit, so the baseline is taken
        # from the empty store rather than assumed to be zero.
        commits_before = len(_commit_messages(store_uri))
        commit_counts = []
        for key in keys:
            result = materialize(assets, partition_key=key, resources=resources)
            assert result.success, f"Materialization failed for {key}"
            commit_counts.append(len(_commit_messages(store_uri)))
    finally:
        patcher.undo()

    return {
        "out_dir": out_dir,
        "store_uri": store_uri,
        "keys": keys,
        "times": [time_for(k) for k in keys],
        "calls": calls,
        "commits_before": commits_before,
        "commit_counts": commit_counts,
    }


@pytest.mark.integration
class TestFullPipeline:
    """Materialize the real graph over two partitions and read the store."""

    def test_every_satellite_was_retrieved_per_partition(self, pipeline_run):
        """Each satellite ran once per partition, and only once."""
        expected = {(sat, t) for t in pipeline_run["times"] for sat in SATELLITES}
        assert set(pipeline_run["calls"]) == expected
        assert len(pipeline_run["calls"]) == len(expected)

    def test_per_satellite_netcdfs_written(self, pipeline_run):
        """Every retrieval landed at its canonical per-day path."""
        out_dir = pipeline_run["out_dir"]
        for t0 in pipeline_run["times"]:
            for sat in SATELLITES:
                path = Path(ring.sat_nc_path(out_dir, sat, t0))
                assert path.exists(), f"missing {path}"
                with xr.open_dataset(path) as ds:
                    assert set(AMV_VARS) <= set(ds.data_vars)
                    assert ds.attrs["satellite_id"] == sat

    def test_mosaic_netcdf_written(self, pipeline_run):
        """The merged mosaic landed at its canonical per-day path."""
        for t0 in pipeline_run["times"]:
            path = Path(ring.global_nc_path(pipeline_run["out_dir"], t0))
            assert path.exists(), f"missing {path}"

    def test_store_time_axis(self, pipeline_run):
        """One timestep per published partition, ordered, unique, intact."""
        times = pipeline_run["times"]
        with _open_store(pipeline_run["store_uri"]) as ds:
            stored = ds["time"].values
            assert ds.sizes["time"] == len(times)
            assert len(np.unique(stored)) == len(times)
            assert list(stored) == sorted(stored)
            # An append that re-encodes time against the first write's
            # units silently shifts every later timestamp; the second one
            # must be exactly what was published.
            for i, t0 in enumerate(times):
                assert stored[i] == np.datetime64(
                    t0, "ns"
                ), f"timestep {i} is {stored[i]}, published {t0}"

    def test_store_variables(self, pipeline_run):
        """The AMV variables and the source codes survive publication."""
        with _open_store(pipeline_run["store_uri"]) as ds:
            assert set(AMV_VARS) <= set(ds.data_vars)
            assert "source_satellite_index" in ds
            source = ds["source_satellite_index"]
            # int8 keeps a global grid affordable; a fill value would
            # promote it to float on read.
            assert source.dtype == np.int8, f"source_satellite_index is {source.dtype}, not int8"
            for name in AMV_VARS:
                assert ds[name].dims == ("time", "latitude", "longitude")

    def test_one_commit_per_timestep(self, pipeline_run):
        """Each published partition adds exactly one commit, named for it."""
        counts = pipeline_run["commit_counts"]
        before = pipeline_run["commits_before"]
        previous = before
        for count in counts:
            assert count == previous + 1, (
                f"commit count went {previous} -> {count}; expected one " f"commit per timestep"
            )
            previous = count

        messages = _commit_messages(pipeline_run["store_uri"])
        for t0 in pipeline_run["times"]:
            tag = ring.time_tag(t0)
            assert any(
                tag in message for message in messages
            ), f"no commit message mentions {tag}: {messages}"

    def test_mosaic_values_follow_the_scenes(self, pipeline_run):
        """The mosaic holds what the synthetic scenes imply."""
        with _open_store(pipeline_run["store_uri"]) as ds:
            first = ds.isel(time=0)
            u = first["u_wind"].values
            assert np.isfinite(u).any(), "the mosaic is entirely empty"

            # Every filled cell carries one of the satellites' winds.
            expected = {synthetic_wind(sat)[0] for sat in SATELLITES}
            found = set(np.unique(u[np.isfinite(u)]).astype(np.float32))
            assert found <= {
                np.float32(v) for v in expected
            }, f"unexpected wind values in the mosaic: {found}"

            # Coverage is a patch per satellite, not the whole globe.
            assert np.isfinite(u).mean() < 0.5

            # The point is only a test of the merge rule if both
            # satellites actually see it.
            for sat in ("gk2a", "himawari9"):
                assert (
                    abs(OVERLAP_LON - SUB_LON_DEG[sat]) < HALF_WIDTH_DEG
                ), f"{sat} does not cover the supposed overlap point"

            names = str(ds["source_satellite_index"].attrs["flag_meanings"]).split()
            source = first["source_satellite_index"].values
            won = {name: int(np.count_nonzero(source == code)) for code, name in enumerate(names)}
            # Losing the contested cells must not cost a satellite the
            # rest of its disk — otherwise the overlap check below would
            # pass on a mosaic that simply dropped himawari9.
            for sat in SATELLITES:
                assert won.get(sat, 0) > 0, f"{sat} won no cells: {won}"

            cell = first.sel(latitude=OVERLAP_LAT, longitude=OVERLAP_LON, method="nearest")
            assert float(cell["u_wind"]) == pytest.approx(
                synthetic_wind("gk2a")[0]
            ), "the overlap did not go to the smallest zenith angle"
            assert names[int(cell["source_satellite_index"])] == "gk2a"

    def test_source_codes_mean_the_same_satellite_at_every_timestep(self, pipeline_run):
        """One vocabulary for the store, so codes are comparable in time."""
        with _open_store(pipeline_run["store_uri"]) as ds:
            names = str(ds["source_satellite_index"].attrs["flag_meanings"]).split()
            assert set(names) == set(SATELLITES)
            for step in range(ds.sizes["time"]):
                cell = ds.isel(time=step).sel(
                    latitude=OVERLAP_LAT, longitude=OVERLAP_LON, method="nearest"
                )
                assert names[int(cell["source_satellite_index"])] == "gk2a"

    def test_quality_attributes_reach_the_store(self, pipeline_run):
        """A degraded contributor is still flagged after publication.

        Quality rides on the time axis rather than on the group, so
        every timestep keeps its own: as group attributes these were
        rewritten by each append, and the whole series ended up
        described by whichever mosaic happened to be published last.
        """
        with _open_store(pipeline_run["store_uri"]) as ds:
            assert (
                "quality_degraded" not in ds.attrs
            ), "quality belongs on the time axis, not the group"
            assert ds["quality_degraded"].dims == ("time",)

            for step in range(ds.sizes["time"]):
                assert int(ds["quality_degraded"].values[step]) == 1
                note = str(ds["quality_note"].values[step])
                assert DEGRADED_SAT in note
                assert "DEGRADED" in note

            satellites = str(ds.attrs["satellites"])
            for sat in SATELLITES:
                assert sat in satellites

    def test_each_timestep_records_the_satellites_behind_it(self, pipeline_run):
        """What a publish needs to decide whether a resume improves on it."""
        with _open_store(pipeline_run["store_uri"]) as ds:
            assert ds["satellites_contributing"].dims == ("time",)
            for step in range(ds.sizes["time"]):
                contributing = {
                    part
                    for part in str(ds["satellites_contributing"].values[step]).split(",")
                    if part
                }
                assert contributing == set(SATELLITES)
