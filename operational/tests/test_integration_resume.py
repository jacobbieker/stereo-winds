"""Integration test: one satellite failing must not cost the rest.

This is the reason the operational pipeline has an asset *per satellite*
rather than one asset for the whole ring.  When GK-2A is unavailable at
12:00, GOES-18, GOES-19 and Himawari-9 must still land, the mosaic must
still build from whoever turned up and say who is absent, and re-running
GK-2A alone must fix the timestamp without re-running — or re-paying for
— any of the others.

Everything here is synthetic: 64x64 scenes standing in for full disks, a
200 km mosaic, and a real local icechunk store.  No network, no GPU, no
checkpoints.
"""

from __future__ import annotations

import ast
import re
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("icechunk")
pytest.importorskip("dagster")

from dagster import (  # noqa: E402
    FilesystemIOManager,
    RetryPolicy,
    materialize,
)

from operational.adapters import ring  # noqa: E402
from operational.assets.amv_assets import (  # noqa: E402
    AMV_ASSETS,
    AMV_ASSETS_BY_SAT,
    AMV_PARTITIONS_DEF,
    build_amv_assets,
)
from operational.assets.mosaic_assets import (  # noqa: E402
    global_mosaic,
    published_mosaic,
)
from operational.config import DEFAULT_CONFIG  # noqa: E402
from operational.core.partitions import time_for  # noqa: E402
from operational.core.publish import open_store  # noqa: E402
from operational.resources import (  # noqa: E402
    IcechunkStoreResource,
    ModelResource,
    PathsResource,
    RunSettingsResource,
)
from operational.tests.conftest import SUB_LON_DEG, synthetic_scene  # noqa: E402

pytestmark = pytest.mark.integration

#: Asset keys, so every selection passed to ``materialize`` is homogeneous.
MOSAIC_KEY = global_mosaic.key
PUBLISH_KEY = published_mosaic.key

#: A coarse mosaic keeps the global grid at 101 x 201 cells.
TEST_RESOLUTION_M = 200_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sat_names(value) -> list[str]:
    """Normalise a ``satellites``-style attribute to a list of names.

    It can arrive as a real list, as a comma-separated string, or as a
    list's ``repr`` after a NetCDF round trip.  Dagster metadata writes
    "(none)" rather than an empty string, since an empty metadata value
    renders as a blank cell in the UI.
    """
    if value is None:
        return []
    if str(value).strip() == "(none)":
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v) for v in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except ValueError, SyntaxError:
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(v) for v in parsed]
    if not text:
        return []
    return [s for s in text.replace(",", " ").split() if s]


def _amv_def(sat_id: str):
    """The retrieval asset for one satellite, however the map is keyed.

    ``AMV_ASSETS_BY_SAT`` is keyed by satellite id, but the asset name
    slugifies punctuation (``mtg-i1`` -> ``amv_mtg_i1``), so fall back to
    matching on that if a lookup by id misses.
    """
    if sat_id in AMV_ASSETS_BY_SAT:
        return AMV_ASSETS_BY_SAT[sat_id]
    slug = "amv_" + re.sub(r"[^0-9a-zA-Z]+", "_", sat_id).strip("_").lower()
    for asset_def in AMV_ASSETS:
        if asset_def.op.name == slug:
            return asset_def
    raise KeyError(f"no AMV asset for {sat_id!r}")


def _amv_key(sat_id: str):
    """Asset key of one satellite's retrieval asset."""
    return _amv_def(sat_id).key


def _amv_node(sat_id: str) -> str:
    """Node (op) name of one satellite's retrieval asset."""
    return _amv_def(sat_id).op.name


def _has_asset(sat_id: str) -> bool:
    """Whether the pipeline actually ships a retrieval asset for ``sat_id``."""
    try:
        _amv_def(sat_id)
    except KeyError:
        return False
    return True


def _fingerprint(path) -> tuple:
    """Size and mtime, so a rewritten file is distinguishable from a kept one."""
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


def _metadata(result, node_name: str) -> dict:
    """Unwrapped metadata of the single materialization ``node_name`` made."""
    materializations = result.asset_materializations_for_node(node_name)
    assert materializations, f"{node_name} emitted no materialization"
    return {k: v.value for k, v in materializations[0].metadata.items()}


def _failed_steps(result) -> set[str]:
    return {
        event.step_key for event in result.all_events if event.event_type_value == "STEP_FAILURE"
    }


def _materialized_keys(result) -> set:
    return {event.asset_key for event in result.get_asset_materialization_events()}


def _first_keys(n: int = 2) -> list[str]:
    """The first ``n`` partition keys, whatever the configured cadence."""
    first = AMV_PARTITIONS_DEF.get_first_partition_key()
    assert first is not None, "the operational partitions have no start"
    horizon = time_for(first) + timedelta(days=2)
    keys = AMV_PARTITIONS_DEF.get_partition_keys(current_time=horizon)
    assert len(keys) >= n, f"only {len(keys)} partition keys available"
    return keys[:n]


#: The satellites this test drives, taken from the shipped config so the
#: test follows the ring rather than a private list of its own.
SATS = [s for s in DEFAULT_CONFIG.satellites if s in SUB_LON_DEG and _has_asset(s)][:4]

#: Rebuilt without retries.  The shipped assets retry twice with a 30 s
#: exponential backoff, which is right in production but would make the
#: deliberate failure below take minutes of sleeping and turn "attempted
#: once" into "attempted three times".  Retry behaviour itself is
#: covered by the AMV asset's own tests; this file is about containment
#: and resume.
AMV_ASSETS_BY_SAT = build_amv_assets(
    tuple(SATS),
    AMV_PARTITIONS_DEF,
    retry_policy=RetryPolicy(max_retries=0),
)
AMV_ASSETS = list(AMV_ASSETS_BY_SAT.values())


class RingStub:
    """Stands in for the student model, and can be told to fail.

    Counts calls per satellite so the test can prove that resuming one
    satellite does not silently re-run the others.
    """

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.failing: set[str] = set()

    def infer_satellite(
        self,
        sat_id,
        t0,
        model,
        disp,
        flow_bands,
        rad_bands,
        device="cpu",
        row_strip=1024,
        prefetcher=None,
    ):
        self.calls[sat_id] = self.calls.get(sat_id, 0) + 1
        if sat_id in self.failing:
            raise RuntimeError(f"synthetic outage: {sat_id} has no imagery at {t0}")
        return synthetic_scene(sat_id, t0)


def _read_store(uri: str, branch: str = "main") -> xr.Dataset:
    """Load the whole store into memory, so later commits cannot race it."""
    repo = open_store(uri)
    session = repo.readonly_session(branch)
    with xr.open_zarr(session.store, consolidated=False) as ds:
        return ds.load()


def _sources_at(ds: xr.Dataset, index: int) -> set[str]:
    """Which satellites actually won cells in the store's ``index`` step."""
    var = ds["source_satellite_index"]
    vocabulary = str(var.attrs.get("flag_meanings", "")).split()
    codes = np.asarray(var.isel(time=index).values)
    return {vocabulary[c] for c in np.unique(codes) if c >= 0}


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


class TestFailureAndResume:
    """A single satellite's failure is contained, and cheap to repair."""

    @pytest.fixture
    def stub(self, monkeypatch) -> RingStub:
        """Replace upstream inference with a countable, failable stand-in."""
        stub = RingStub()
        monkeypatch.setattr(ring, "infer_satellite", stub.infer_satellite)
        # Defensive: if the retrieval step ever binds the name directly
        # rather than going through the adapter module, patch that too.
        import operational.core.amv as amv_module

        if hasattr(amv_module, "infer_satellite"):
            monkeypatch.setattr(amv_module, "infer_satellite", stub.infer_satellite, raising=False)
        # The asset resolves the model before it calls the retrieval,
        # so the stub alone still demands a real checkpoint.
        monkeypatch.setattr(ModelResource, "model", lambda self: None)
        monkeypatch.setattr(ModelResource, "disparity", lambda self: None)
        return stub

    @pytest.fixture
    def harness(self, tmp_path, tmp_store_uri):
        """Assets, resources and paths wired onto throwaway storage."""
        output_dir = tmp_path / "output"
        paths = PathsResource(output_dir=str(output_dir))
        resources = {
            "paths": paths,
            "model": ModelResource(device="cpu"),
            "run_settings": RunSettingsResource(
                satellites=list(SATS),
                resolution_m=TEST_RESOLUTION_M,
                skip_existing=True,
            ),
            "store": IcechunkStoreResource(store_uri=tmp_store_uri),
            # A stable IO-manager root, shared by every run() below.
            # published_mosaic loads global_mosaic's output as an input,
            # so with the default (a fresh temp dir per materialize call)
            # publishing on its own cannot find what the mosaic step
            # wrote in an earlier run -- which is exactly the resume
            # sequence this test drives.
            "io_manager": FilesystemIOManager(base_dir=str(tmp_path / "io")),
        }
        assets = list(AMV_ASSETS) + [global_mosaic, published_mosaic]

        def run(selection, partition_key, raise_on_error=True):
            return materialize(
                assets,
                selection=selection,
                partition_key=partition_key,
                resources=resources,
                raise_on_error=raise_on_error,
            )

        return {
            "run": run,
            "paths": paths,
            "store_uri": tmp_store_uri,
            "amv_selection": [_amv_key(s) for s in SATS],
        }

    def test_one_satellite_fails_then_resumes_alone(self, stub, harness):
        assert len(SATS) >= 3, f"this test needs at least three satellites, got {SATS}"

        run = harness["run"]
        paths = harness["paths"]
        store_uri = harness["store_uri"]
        amv_selection = harness["amv_selection"]

        key0, key1 = _first_keys(2)
        t0, t1 = time_for(key0), time_for(key1)

        fail_sat = SATS[-1]
        healthy = [s for s in SATS if s != fail_sat]

        # -- 1. A single satellite's failure is contained -----------------
        stub.failing = {fail_sat}
        result = run(amv_selection, key0, raise_on_error=False)

        assert result.success is False, "a failing satellite must fail its own asset and so the run"
        assert _failed_steps(result) == {_amv_node(fail_sat)}, f"only {fail_sat} should have failed"
        assert _materialized_keys(result) == {
            _amv_key(s) for s in healthy
        }, "every healthy satellite should still have materialized"

        for sat_id in healthy:
            path = paths.sat_path(sat_id, t0)
            assert path.exists(), f"{sat_id} left no retrieval at {path}"
            metadata = _metadata(result, _amv_node(sat_id))
            assert metadata["reused"] is False
            # The AMV asset reports band coverage; cell counts are the
            # mosaic's metric, not this step's.
            assert metadata["n_bands_missing"] == 0
        assert not paths.sat_path(
            fail_sat, t0
        ).exists(), "a failed retrieval must not leave a file behind"

        # Every satellite was attempted exactly once.
        assert stub.calls == {s: 1 for s in SATS}

        # -- 2. The mosaic still builds from the rest ---------------------
        result = run([MOSAIC_KEY], key0)
        assert result.success

        metadata = _metadata(result, "global_mosaic")
        assert _sat_names(metadata["contributing_satellites"]) == healthy
        assert _sat_names(metadata["missing_satellites"]) == [fail_sat]
        assert fail_sat in metadata["quality_note"]

        mosaic_path = paths.mosaic_path(t0)
        assert mosaic_path.exists()
        with xr.open_dataset(mosaic_path) as ds_mosaic:
            assert _sat_names(ds_mosaic.attrs["satellites"]) == healthy
            assert _sat_names(ds_mosaic.attrs["satellites_missing"]) == [fail_sat]
            assert fail_sat in str(ds_mosaic.attrs["quality_note"])
            assert int(ds_mosaic.attrs["quality_degraded"]) == 1
            codes = np.asarray(ds_mosaic["source_satellite_index"].values)
            meanings = str(ds_mosaic["source_satellite_index"].attrs["flag_meanings"]).split()
            contributed = {meanings[c] for c in np.unique(codes) if c >= 0}
            assert contributed == set(
                healthy
            ), "the mosaic must carry exactly the healthy satellites"
            assert np.isfinite(
                ds_mosaic["u_wind"].values
            ).any(), "the degraded mosaic must still hold real winds"

        # -- 3. Resume is cheap and surgical ------------------------------
        stub.failing = set()
        before = dict(stub.calls)
        untouched = {s: _fingerprint(paths.sat_path(s, t0)) for s in healthy}

        result = run([_amv_key(fail_sat)], key0)
        assert result.success
        assert paths.sat_path(fail_sat, t0).exists()

        metadata = _metadata(result, _amv_node(fail_sat))
        assert metadata["reused"] is False, "the previously failed satellite had nothing to reuse"
        assert metadata["n_bands_missing"] == 0

        assert stub.calls[fail_sat] == before[fail_sat] + 1
        for sat_id in healthy:
            assert (
                stub.calls[sat_id] == before[sat_id]
            ), f"{sat_id} was recomputed during the resume of {fail_sat}"
            assert _fingerprint(paths.sat_path(sat_id, t0)) == untouched[sat_id], (
                f"{sat_id}'s retrieval was rewritten by the resume of " f"{fail_sat}"
            )

        # Re-running the whole ring for this timestamp now costs nothing:
        # every satellite is reused from disk, untouched.
        before = dict(stub.calls)
        untouched = {s: _fingerprint(paths.sat_path(s, t0)) for s in SATS}
        result = run(amv_selection, key0)
        assert result.success
        assert (
            stub.calls == before
        ), "a re-run of an already-complete timestamp must not infer again"
        for sat_id in SATS:
            assert _metadata(result, _amv_node(sat_id))["reused"] is True
            assert _fingerprint(paths.sat_path(sat_id, t0)) == untouched[sat_id]

        # The repaired timestamp mosaics and publishes with the full ring.
        result = run([MOSAIC_KEY, PUBLISH_KEY], key0)
        assert result.success

        metadata = _metadata(result, "global_mosaic")
        assert _sat_names(metadata["contributing_satellites"]) == SATS
        assert _sat_names(metadata["missing_satellites"]) == []

        metadata = _metadata(result, "published_mosaic")
        assert metadata["written"] is True

        stored = _read_store(store_uri)
        assert stored.sizes["time"] == 1
        assert _sources_at(stored, 0) == set(
            SATS
        ), "the store must reflect the complete satellite set after resume"

        # -- 4. Publishing stays idempotent across the resume -------------
        result = run([PUBLISH_KEY], key0)
        assert result.success
        metadata = _metadata(result, "published_mosaic")
        assert metadata["written"] is False
        assert metadata["skipped_reason"], "a skipped publish must say why it was skipped"

        stored = _read_store(store_uri)
        assert (
            stored.sizes["time"] == 1
        ), "re-publishing must not append the timestamp a second time"

        # A second, entirely healthy timestamp still appends normally.
        result = run(amv_selection + [MOSAIC_KEY, PUBLISH_KEY], key1)
        assert result.success
        assert _metadata(result, "published_mosaic")["written"] is True

        # Re-publishing both timestamps changes nothing.
        for key in (key0, key1):
            result = run([PUBLISH_KEY], key)
            assert result.success
            assert _metadata(result, "published_mosaic")["written"] is False

        stored = _read_store(store_uri)
        times = pd.to_datetime(np.asarray(stored["time"].values))
        assert stored.sizes["time"] == 2
        assert len(set(times)) == 2, f"duplicate timestamps in the store: {times}"
        assert sorted(times) == [pd.Timestamp(t0), pd.Timestamp(t1)]
        for index in range(stored.sizes["time"]):
            assert _sources_at(stored, index) == set(SATS)
