"""Tests for the per-satellite AMV assets.

Offline throughout: :func:`operational.core.amv.run_satellite_amv` is
monkeypatched, so no imagery, checkpoint or GPU is touched.  The point
under test is not the retrieval but the orchestration around it --
partition decoding, metadata, and above all that one satellite's failure
leaves the rest of the ring alone.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from dagster import (
    AssetsDefinition,
    DagsterInstance,
    ExecuteInProcessResult,
    RetryPolicy,
    materialize,
)

from operational.assets import amv_assets
from operational.assets.amv_assets import (
    AMV_ASSETS,
    AMV_ASSETS_BY_SAT,
    AMV_GROUP,
    DEFAULT_RETRY_POLICY,
    amv_asset_name,
    build_amv_asset,
    build_amv_assets,
)
from operational.core.amv import AmvResult
from operational.tests.conftest import synthetic_scene
from operational.core.partitions import build_partitions_def, key_for
from operational.resources import ModelResource, PathsResource, RunSettingsResource

SATELLITES = ("goes18", "goes19", "himawari9", "gk2a")
T0 = datetime(2026, 8, 1, 12, 0)
PARTITION_KEY = key_for(T0)

# Retries are what the assets do in production; in a test they are just
# latency, so the built-under-test assets use an immediate single retry.
FAST_RETRY = RetryPolicy(max_retries=1)


class FakeModelResource(ModelResource):
    """A :class:`ModelResource` that hands out sentinels, not checkpoints."""

    def model(self) -> str:
        return "fake-model"

    def disparity(self) -> str:
        return "fake-disparity"


class ExplodingModelResource(ModelResource):
    """A :class:`ModelResource` that fails if a checkpoint is resolved."""

    def model(self) -> str:
        raise AssertionError("student checkpoint loaded for a reused partition")

    def disparity(self) -> str:
        raise AssertionError("RAFT checkpoint loaded for a reused partition")


@pytest.fixture
def partitions_def():
    """Hourly partitions covering the test timestamp."""
    return build_partitions_def(datetime(2026, 8, 1), 60)


@pytest.fixture
def resources(tmp_path: Path) -> dict[str, object]:
    """Resource set pointing every output at ``tmp_path``."""
    return {
        "paths": PathsResource(output_dir=str(tmp_path / "out")),
        "model": FakeModelResource(
            student_ckpt="student.ckpt", raft_ckpt="raft.ckpt", device="cpu",
        ),
        "run_settings": RunSettingsResource(satellites=list(SATELLITES)),
    }


@pytest.fixture
def assets(partitions_def) -> dict[str, AssetsDefinition]:
    """One asset per satellite, built with a fast retry policy."""
    return build_amv_assets(
        SATELLITES, partitions_def, retry_policy=FAST_RETRY,
    )


def fake_result(sat_id: str, t0: datetime, out_dir: Path, **kwargs) -> AmvResult:
    """An :class:`AmvResult` standing in for a real retrieval."""
    path = Path(out_dir) / t0.strftime("%Y%m%d") / (
        f"student_amv_{sat_id}_{t0:%Y%m%dT%H%M}.nc"
    )
    fields = dict(
        dataset=synthetic_scene(sat_id, t0),
        reused=False,
        n_bands_missing=0,
        bands_missing=(),
        quality_degraded=False,
    )
    fields.update(kwargs)
    return AmvResult(sat_id=sat_id, timestamp=t0, path=path, **fields)


def record_calls(monkeypatch, *, raises: dict[str, Exception] | None = None,
                 result_kwargs: dict | None = None) -> list[dict]:
    """Patch ``run_satellite_amv`` and capture how it was called.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to install the stand-in.
    raises : dict, optional
        Satellite id -> exception to raise for that satellite.
    result_kwargs : dict, optional
        Extra fields for the synthetic :class:`AmvResult`.

    Returns
    -------
    list of dict
        One entry per call, in call order.
    """
    calls: list[dict] = []
    failures = raises or {}

    def _fake(sat_id, t0, model, disp, flow_bands, rad_bands, output_dir,
              *, device="cpu", row_strip=1024, skip_existing=True):
        calls.append({
            "sat_id": sat_id, "t0": t0, "model": model, "disp": disp,
            "flow_bands": flow_bands, "rad_bands": rad_bands,
            "output_dir": output_dir, "device": device,
            "row_strip": row_strip, "skip_existing": skip_existing,
        })
        if sat_id in failures:
            raise failures[sat_id]
        return fake_result(sat_id, t0, output_dir, **(result_kwargs or {}))

    monkeypatch.setattr(amv_assets, "run_satellite_amv", _fake)
    return calls


def materialized_keys(result: ExecuteInProcessResult) -> set[str]:
    """Asset keys that produced a materialization in this run."""
    return {
        event.event_specific_data.materialization.asset_key.to_user_string()
        for event in result.get_asset_materialization_events()
    }


def metadata_for(result: ExecuteInProcessResult, sat_id: str) -> dict:
    """Materialization metadata of one satellite's asset, unwrapped."""
    entries = result.asset_materializations_for_node(amv_asset_name(sat_id))
    assert len(entries) == 1, f"expected one materialization for {sat_id}"
    return {k: v.value for k, v in entries[0].metadata.items()}


class TestBuildAmvAsset:
    """Shape of the assets the factory produces."""

    def test_one_asset_per_satellite_with_expected_keys(self, assets):
        assert list(assets) == list(SATELLITES)
        keys = {a.key.to_user_string() for a in assets.values()}
        assert keys == {f"amv_{s}" for s in SATELLITES}

    def test_asset_name_slugifies_punctuation(self):
        assert amv_asset_name("mtg-i1") == "amv_mtg_i1"
        assert amv_asset_name("msg-iodc") == "amv_msg_iodc"

    def test_grouped_and_described_by_satellite(self, assets):
        for sat_id, asset_def in assets.items():
            key = asset_def.key
            assert asset_def.group_names_by_key[key] == AMV_GROUP
            assert sat_id in asset_def.descriptions_by_key[key]

    def test_partitioned(self, assets, partitions_def):
        for asset_def in assets.values():
            assert asset_def.partitions_def == partitions_def

    def test_retry_policy_attached(self, assets):
        for asset_def in assets.values():
            assert asset_def.op.retry_policy == FAST_RETRY

    def test_default_retry_policy_allows_resume(self):
        assert DEFAULT_RETRY_POLICY.max_retries >= 2
        assert DEFAULT_RETRY_POLICY.delay
        for asset_def in AMV_ASSETS:
            assert asset_def.op.retry_policy == DEFAULT_RETRY_POLICY

    def test_module_level_default_set(self):
        # The module-level set follows the configured ring, which is not
        # the short list these tests build their own assets from.
        assert list(AMV_ASSETS_BY_SAT) == list(amv_assets.DEFAULT_CONFIG.satellites)
        assert AMV_ASSETS == list(AMV_ASSETS_BY_SAT.values())
        assert len({a.key for a in AMV_ASSETS}) == len(AMV_ASSETS)

    def test_custom_name_and_prefix(self, partitions_def):
        asset_def = build_amv_asset(
            "goes19", partitions_def, name="custom", key_prefix="ops",
        )
        assert asset_def.key.path == ["ops", "custom"]


class TestMaterialize:
    """A single satellite materializing for a single partition."""

    def test_succeeds_and_decodes_the_partition_key(
        self, assets, resources, monkeypatch,
    ):
        calls = record_calls(monkeypatch)
        result = materialize(
            [assets["goes19"]], partition_key=PARTITION_KEY, resources=resources,
        )
        assert result.success
        assert len(calls) == 1
        assert calls[0]["sat_id"] == "goes19"
        assert calls[0]["t0"] == T0

    def test_passes_resources_through(self, assets, resources, monkeypatch):
        calls = record_calls(monkeypatch)
        materialize(
            [assets["gk2a"]], partition_key=PARTITION_KEY, resources=resources,
        )
        call = calls[0]
        assert call["model"] == "fake-model"
        assert call["disp"] == "fake-disparity"
        assert call["output_dir"] == resources["paths"].output_dir
        assert call["device"] == "cpu"
        assert call["flow_bands"] == resources["run_settings"].flow_bands
        assert call["rad_bands"] == resources["run_settings"].rad_bands
        assert call["skip_existing"] is True

    def test_skip_existing_is_operator_controlled(
        self, assets, resources, monkeypatch,
    ):
        calls = record_calls(monkeypatch)
        resources["run_settings"] = RunSettingsResource(skip_existing=False)
        materialize(
            [assets["gk2a"]], partition_key=PARTITION_KEY, resources=resources,
        )
        assert calls[0]["skip_existing"] is False

    def test_metadata_reports_the_retrieval(
        self, assets, resources, monkeypatch,
    ):
        record_calls(monkeypatch, result_kwargs={
            "n_bands_missing": 3,
            "bands_missing": ("C09", "C12", "C14"),
            "quality_degraded": True,
        })
        result = materialize(
            [assets["goes18"]], partition_key=PARTITION_KEY, resources=resources,
        )
        meta = metadata_for(result, "goes18")
        assert meta["satellite"] == "goes18"
        assert meta["partition"] == PARTITION_KEY
        assert meta["timestamp"] == T0.isoformat()
        assert meta["output_path"].endswith("student_amv_goes18_20260801T1200.nc")
        assert meta["reused"] is False
        assert meta["n_bands_missing"] == 3
        assert meta["bands_missing"] == "C09, C12, C14"
        assert meta["quality_degraded"] is True

    def test_metadata_for_a_complete_retrieval(
        self, assets, resources, monkeypatch,
    ):
        record_calls(monkeypatch)
        result = materialize(
            [assets["goes18"]], partition_key=PARTITION_KEY, resources=resources,
        )
        meta = metadata_for(result, "goes18")
        assert meta["n_bands_missing"] == 0
        assert meta["bands_missing"] == "none"
        assert meta["quality_degraded"] is False
        assert meta["status"] == "computed"

    def test_reuse_is_visible_in_metadata(
        self, assets, resources, monkeypatch,
    ):
        record_calls(monkeypatch, result_kwargs={"reused": True})
        result = materialize(
            [assets["himawari9"]], partition_key=PARTITION_KEY,
            resources=resources,
        )
        meta = metadata_for(result, "himawari9")
        assert meta["reused"] is True
        assert meta["status"] == "reused existing output"

    def test_output_value_is_the_path(self, assets, resources, monkeypatch):
        record_calls(monkeypatch)
        result = materialize(
            [assets["goes19"]], partition_key=PARTITION_KEY, resources=resources,
        )
        assert result.output_for_node(amv_asset_name("goes19")).endswith(
            "student_amv_goes19_20260801T1200.nc"
        )

    def test_different_partitions_reach_different_timestamps(
        self, assets, resources, monkeypatch,
    ):
        calls = record_calls(monkeypatch)
        for hour in (0, 5):
            t = datetime(2026, 8, 1, hour, 0)
            materialize(
                [assets["goes18"]], partition_key=key_for(t), resources=resources,
            )
        assert [c["t0"] for c in calls] == [
            datetime(2026, 8, 1, 0, 0), datetime(2026, 8, 1, 5, 0),
        ]


class TestIndependentFailure:
    """The reason there is one asset per satellite."""

    def test_one_failure_does_not_stop_the_others(
        self, assets, resources, monkeypatch,
    ):
        record_calls(monkeypatch, raises={"gk2a": RuntimeError("no imagery")})
        result = materialize(
            list(assets.values()),
            partition_key=PARTITION_KEY,
            resources=resources,
            raise_on_error=False,
        )
        assert not result.success
        assert materialized_keys(result) == {
            f"amv_{s}" for s in SATELLITES if s != "gk2a"
        }
        failed = {e.step_key for e in result.get_step_failure_events()}
        assert failed == {amv_asset_name("gk2a")}

    def test_every_satellite_is_attempted(
        self, assets, resources, monkeypatch,
    ):
        calls = record_calls(
            monkeypatch, raises={"goes18": RuntimeError("scan late")},
        )
        materialize(
            list(assets.values()),
            partition_key=PARTITION_KEY,
            resources=resources,
            raise_on_error=False,
        )
        attempted = {c["sat_id"] for c in calls}
        assert attempted == set(SATELLITES)

    def test_failed_asset_retries_before_giving_up(
        self, assets, resources, monkeypatch,
    ):
        calls = record_calls(monkeypatch, raises={"gk2a": RuntimeError("boom")})
        materialize(
            [assets["gk2a"]],
            partition_key=PARTITION_KEY,
            resources=resources,
            raise_on_error=False,
        )
        # One initial attempt plus FAST_RETRY.max_retries retries.
        assert len(calls) == 1 + FAST_RETRY.max_retries

    def test_failed_satellite_resumes_on_its_own(
        self, assets, resources, monkeypatch,
    ):
        instance = DagsterInstance.ephemeral()
        record_calls(monkeypatch, raises={"gk2a": RuntimeError("no imagery")})
        first = materialize(
            list(assets.values()),
            partition_key=PARTITION_KEY,
            resources=resources,
            instance=instance,
            raise_on_error=False,
        )
        assert not first.success

        # The repair: re-materialize the one asset, for the one partition.
        calls = record_calls(monkeypatch)
        second = materialize(
            [assets["gk2a"]],
            partition_key=PARTITION_KEY,
            resources=resources,
            instance=instance,
        )
        assert second.success
        assert [c["sat_id"] for c in calls] == ["gk2a"]
        assert materialized_keys(second) == {"amv_gk2a"}


class TestIdempotence:
    """A partition already on disk costs a listing, not a forward pass."""

    def test_second_materialization_reuses_the_existing_file(
        self, assets, resources, monkeypatch, tmp_path,
    ):
        seen: list[bool] = []

        def _fake(sat_id, t0, model, disp, flow_bands, rad_bands, output_dir,
                  *, device="cpu", row_strip=1024, skip_existing=True):
            path = Path(output_dir) / t0.strftime("%Y%m%d") / (
                f"student_amv_{sat_id}_{t0:%Y%m%dT%H%M}.nc"
            )
            reused = skip_existing and path.exists()
            seen.append(reused)
            if not reused:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"netcdf")
            return AmvResult(
                sat_id=sat_id, timestamp=t0, path=path, reused=reused,
                dataset=synthetic_scene(sat_id, t0), n_bands_missing=0,
                bands_missing=(), quality_degraded=False,
            )

        monkeypatch.setattr(amv_assets, "run_satellite_amv", _fake)
        for _ in range(2):
            result = materialize(
                [assets["goes19"]], partition_key=PARTITION_KEY,
                resources=resources,
            )
            assert result.success
        assert seen == [False, True]
        assert metadata_for(result, "goes19")["reused"] is True

    def test_reused_partition_loads_no_checkpoints(
        self, assets, resources, monkeypatch,
    ):
        calls = record_calls(monkeypatch, result_kwargs={"reused": True})
        out_dir = Path(resources["paths"].output_dir)
        existing = out_dir / T0.strftime("%Y%m%d") / (
            f"student_amv_goes18_{T0:%Y%m%dT%H%M}.nc"
        )
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_bytes(b"netcdf")

        # Resolving either checkpoint would blow this resource up.
        resources["model"] = ExplodingModelResource(
            student_ckpt="student.ckpt", raft_ckpt="raft.ckpt",
        )
        result = materialize(
            [assets["goes18"]], partition_key=PARTITION_KEY, resources=resources,
        )
        assert result.success
        assert calls[0]["model"] is None and calls[0]["disp"] is None
        assert metadata_for(result, "goes18")["reused"] is True

    def test_missing_partition_does_load_checkpoints(
        self, assets, resources, monkeypatch,
    ):
        record_calls(monkeypatch)
        resources["model"] = ExplodingModelResource(
            student_ckpt="student.ckpt", raft_ckpt="raft.ckpt",
        )
        result = materialize(
            [assets["goes18"]], partition_key=PARTITION_KEY, resources=resources,
            raise_on_error=False,
        )
        assert not result.success
