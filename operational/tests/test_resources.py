"""Tests for the Dagster resources.

The point of these resources is that they are inert until asked, so most
of what is asserted here is an absence: no directory appears, no
checkpoint is read, no socket is opened.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from dagster import asset, materialize

from operational.config import (
    DEFAULT_FLOW_BANDS,
    DEFAULT_RAD_BANDS,
    OperationalConfig,
)
from operational.resources import (
    IcechunkStoreResource,
    ModelResource,
    PathsResource,
    RunSettingsResource,
    as_naive_utc,
)

T0 = datetime(2024, 1, 15, 12, 0)


def _tree(root: Path) -> list[Path]:
    """Every path under ``root``, for before/after comparisons."""
    return sorted(root.rglob("*"))


class TestImportLight:
    """The module must not drag heavy dependencies in at import time."""

    def test_no_heavy_imports_at_module_scope(self):
        import ast

        src = Path(__file__).resolve().parents[1] / "resources.py"
        tree = ast.parse(src.read_text())
        imported: set[str] = set()
        for node in tree.body:  # module scope only, not nested bodies
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")

        forbidden = {
            "torch",
            "icechunk",
            "operational.adapters.ring",
            "stereo_winds.icechunk_output",
            "stereo_winds.student_zeus_model",
            "stereo_winds.disparity",
            # Pulls torch in transitively, which is the whole point.
            "stereo_winds.student_dataset",
        }
        assert not (imported & forbidden), (
            f"{sorted(imported & forbidden)} must be imported inside a "
            f"method, not at module scope"
        )

    def test_fresh_interpreter_import_pulls_in_nothing_heavy(self):
        """Import the module in a clean interpreter and inspect sys.modules.

        This is the check that matters: the static scan above cannot see
        what a dependency imports.  torch is on the list because
        ``operational.config`` used to reach the band constants through
        ``stereo_winds.student_dataset``, which imports torch at module
        scope — that made every Dagster definition pay a multi-second
        torch import and fail outright on a torch-less machine.
        """
        import subprocess

        repo_root = Path(__file__).resolve().parents[2]
        code = (
            "import sys; import operational.resources; "
            "print(','.join(m for m in ('torch', 'icechunk', 'satpy', "
            "'operational_ring') if m in sys.modules))"
        )
        env = {**os.environ, "PYTHONPATH": str(repo_root)}
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(repo_root),
        )
        assert out.returncode == 0, out.stderr
        assert (
            out.stdout.strip() == ""
        ), f"importing operational.resources pulled in {out.stdout.strip()}"


class TestPathsResource:
    def test_constructs_without_side_effects(self, tmp_path):
        root = tmp_path / "out"
        res = PathsResource(output_dir=str(root))
        assert res.root == root
        assert not root.exists()

    def test_paths_are_canonical(self, tmp_path):
        from operational.adapters.ring import global_nc_path, sat_nc_path

        res = PathsResource(output_dir=str(tmp_path))
        assert res.sat_path("goes18", T0) == sat_nc_path(tmp_path, "goes18", T0)
        assert res.mosaic_path(T0) == global_nc_path(tmp_path, T0)
        # Layout: per-day directory, canonical time tag.
        assert res.sat_path("goes18", T0).name == ("student_amv_goes18_20240115T1200.nc")
        assert res.mosaic_path(T0).name == "student_amv_global_20240115T1200.nc"
        assert res.mosaic_path(T0).parent.name == "20240115"

    def test_paths_do_not_create_directories_by_default(self, tmp_path):
        res = PathsResource(output_dir=str(tmp_path / "out"))
        before = _tree(tmp_path)
        res.sat_path("goes19", T0)
        res.mosaic_path(T0)
        res.day_dir(T0)
        assert _tree(tmp_path) == before

    def test_create_makes_the_day_directory_only(self, tmp_path):
        res = PathsResource(output_dir=str(tmp_path / "out"))
        p = res.sat_path("goes19", T0, create=True)
        assert p.parent.is_dir()
        assert not p.exists()

    def test_mosaic_path_create_and_ensure_root(self, tmp_path):
        res = PathsResource(output_dir=str(tmp_path / "out"))
        m = res.mosaic_path(T0, create=True)
        assert m.parent.is_dir()
        assert res.ensure_root() == tmp_path / "out"
        assert (tmp_path / "out").is_dir()

    def test_day_dir_create(self, tmp_path):
        res = PathsResource(output_dir=str(tmp_path / "out"))
        d = res.day_dir(T0, create=True)
        assert d.is_dir() and d.name == "20240115"

    def test_default_output_dir_matches_config(self):
        assert PathsResource().output_dir == str(OperationalConfig().output_dir)


class TestIcechunkStoreResource:
    def test_constructs_without_side_effects(self, tmp_store_uri, tmp_path):
        before = _tree(tmp_path)
        res = IcechunkStoreResource(store_uri=tmp_store_uri)
        assert res.branch == "main"
        assert res.chunk == 1024
        assert not res.is_s3
        assert _tree(tmp_path) == before
        assert not Path(tmp_store_uri).exists()

    def test_s3_uri_constructs_without_network(self, monkeypatch):
        # Make any socket use raise, so "without network" is enforced
        # rather than merely claimed in the test's name.
        import socket

        def no_network(*a, **k):  # pragma: no cover - must not run
            raise AssertionError("constructing the resource opened a socket")

        monkeypatch.setattr(socket, "socket", no_network)
        monkeypatch.setattr(socket, "create_connection", no_network)

        res = IcechunkStoreResource(
            store_uri="s3://some-bucket/winds/prefix",
            region="us-east-1",
            endpoint_url="https://example.invalid",
            anonymous=True,
            force_path_style=True,
        )
        assert res.is_s3
        assert res.region == "us-east-1"
        assert res.endpoint_url == "https://example.invalid"
        assert res.anonymous is True
        assert res.force_path_style is True

    def test_repo_opens_real_local_store_and_caches(self, tmp_store_uri):
        res = IcechunkStoreResource(store_uri=tmp_store_uri)
        repo = res.repo()
        assert repo is not None
        assert Path(tmp_store_uri).is_dir()
        assert res.repo() is repo

    def test_existing_times_is_empty_for_a_new_store(self, tmp_store_uri):
        res = IcechunkStoreResource(store_uri=tmp_store_uri)
        assert res.existing_times() == set()

    def test_set_repo_bypasses_open(self, monkeypatch):
        sentinel = object()
        res = IcechunkStoreResource(store_uri="s3://nowhere/at-all")
        res.set_repo(sentinel)

        import stereo_winds.icechunk_output as ico

        def boom(*a, **k):  # pragma: no cover - must not run
            raise AssertionError("repo() opened the store despite the cache")

        monkeypatch.setattr(ico, "open_icechunk_repo", boom)
        assert res.repo() is sentinel

    def test_unsupported_uri_scheme_raises(self):
        res = IcechunkStoreResource(store_uri="gs://bucket/prefix")
        with pytest.raises(ValueError, match="Unsupported icechunk store URI"):
            res.repo()


class TestModelResource:
    def test_constructs_with_missing_checkpoints(self, tmp_path):
        missing_student = tmp_path / "nope-student.ckpt"
        missing_raft = tmp_path / "nope-raft.ckpt"
        before = _tree(tmp_path)
        res = ModelResource(
            student_ckpt=str(missing_student),
            raft_ckpt=str(missing_raft),
            device="cpu",
        )
        assert res.device == "cpu"
        assert res.row_strip == 1024
        assert res.available() is False
        # No file was read and nothing was written.
        assert _tree(tmp_path) == before

    def test_available_is_true_when_both_paths_exist(self, tmp_path):
        s = tmp_path / "student.ckpt"
        r = tmp_path / "raft.ckpt"
        s.write_bytes(b"")
        r.write_bytes(b"")
        res = ModelResource(student_ckpt=str(s), raft_ckpt=str(r))
        assert res.available() is True

    def test_available_is_false_when_unconfigured(self):
        assert ModelResource().available() is False

    def test_model_raises_for_missing_checkpoint(self, tmp_path):
        res = ModelResource(student_ckpt=str(tmp_path / "gone.ckpt"))
        with pytest.raises(FileNotFoundError, match="student_ckpt"):
            res.model()

    def test_disparity_raises_for_missing_checkpoint(self, tmp_path):
        res = ModelResource(raft_ckpt=str(tmp_path / "gone.ckpt"))
        with pytest.raises(FileNotFoundError, match="raft_ckpt"):
            res.disparity()

    def test_unconfigured_paths_raise_value_error(self):
        res = ModelResource()
        with pytest.raises(ValueError, match="student_ckpt"):
            res.model()
        with pytest.raises(ValueError, match="raft_ckpt"):
            res.disparity()

    def test_injected_fakes_are_returned_without_loading(self, tmp_path):
        model, disp = object(), object()
        res = ModelResource(
            student_ckpt=str(tmp_path / "missing.ckpt"),
            raft_ckpt=str(tmp_path / "missing.ckpt"),
        )
        res.set_model(model)
        res.set_disparity(disp)
        # Would raise FileNotFoundError if the lazy load ran.
        assert res.model() is model
        assert res.disparity() is disp

    def test_model_is_loaded_once_and_cached(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "student.ckpt"
        ckpt.write_bytes(b"")
        calls = []
        loaded = object()

        class FakeStudent:
            @staticmethod
            def load_from_checkpoint(path, map_location):
                calls.append((path, map_location))

                class _M:
                    def eval(self_inner):
                        return loaded

                return _M()

        import stereo_winds.student_zeus_model as szm

        monkeypatch.setattr(szm, "StudentWindsModel", FakeStudent)
        res = ModelResource(student_ckpt=str(ckpt), device="cpu")
        assert res.model() is loaded
        assert res.model() is loaded
        assert calls == [(str(ckpt), "cpu")]

    def test_disparity_is_loaded_once_and_cached(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "raft.ckpt"
        ckpt.write_bytes(b"")
        calls = []

        class FakeDisparity:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        import stereo_winds.disparity as sd

        monkeypatch.setattr(sd, "StereoDisparity", FakeDisparity)
        res = ModelResource(raft_ckpt=str(ckpt), device="cpu")
        first = res.disparity()
        assert res.disparity() is first
        assert len(calls) == 1
        assert calls[0]["model_ckpt_path"] == str(ckpt)
        assert calls[0]["tile_size"] == 512
        assert calls[0]["overlap"] == 128
        assert calls[0]["batch_size"] == 8
        assert calls[0]["device"] == "cpu"


class TestRunSettingsResource:
    def test_defaults_match_operational_config(self):
        cfg = OperationalConfig()
        res = RunSettingsResource()
        assert tuple(res.satellites) == cfg.satellites
        assert tuple(res.flow_bands) == cfg.flow_bands
        assert tuple(res.rad_bands) == cfg.rad_bands
        assert res.cadence_minutes == cfg.cadence_minutes
        assert res.availability_tolerance_minutes == (cfg.availability_tolerance_minutes)
        assert res.resolution_m == cfg.resolution_m
        assert res.skip_existing is True

    def test_overrides_are_honoured(self):
        res = RunSettingsResource(
            satellites=["goes19"],
            cadence_minutes=10,
            skip_existing=False,
        )
        assert res.satellites == ["goes19"]
        assert res.cadence_minutes == 10
        assert res.skip_existing is False

    def test_to_config_round_trips(self, tmp_path):
        res = RunSettingsResource(satellites=["gk2a"], resolution_m=5000.0)
        cfg = res.to_config(output_dir=tmp_path, store_uri="s3://b/p")
        assert isinstance(cfg, OperationalConfig)
        assert cfg.satellites == ("gk2a",)
        assert cfg.resolution_m == 5000.0
        assert cfg.output_dir == tmp_path
        assert cfg.store_uri == "s3://b/p"

    def test_construction_has_no_side_effects(self, tmp_path, monkeypatch):
        """Construct with tmp_path as cwd so a stray write lands there."""
        monkeypatch.chdir(tmp_path)
        RunSettingsResource()
        assert _tree(tmp_path) == []


class TestResourcesInsideDagster:
    """The resources must work through Dagster's own machinery."""

    def test_materialize_with_all_resources(self, tmp_path, tmp_store_uri):
        @asset
        def probe(
            paths: PathsResource,
            store: IcechunkStoreResource,
            model: ModelResource,
            settings: RunSettingsResource,
        ) -> dict:
            repo = store.repo()
            return {
                # Returned rather than asserted: a bare `assert` in an
                # asset body is stripped under `python -O`.
                "repo_cached": store.repo() is repo,
                "mosaic": str(paths.mosaic_path(T0, create=True)),
                "sat": str(paths.sat_path(settings.satellites[0], T0)),
                "available": model.available(),
                "branch": store.branch,
                "n_sats": len(settings.satellites),
            }

        result = materialize(
            [probe],
            resources={
                "paths": PathsResource(output_dir=str(tmp_path / "out")),
                "store": IcechunkStoreResource(store_uri=tmp_store_uri),
                "model": ModelResource(
                    student_ckpt=str(tmp_path / "absent.ckpt"),
                    raft_ckpt=str(tmp_path / "absent.ckpt"),
                ),
                "settings": RunSettingsResource(satellites=["goes18", "gk2a"]),
            },
        )
        assert result.success
        out = result.output_for_node("probe")
        assert out["mosaic"].endswith("student_amv_global_20240115T1200.nc")
        assert out["sat"].endswith("student_amv_goes18_20240115T1200.nc")
        assert out["available"] is False
        assert out["branch"] == "main"
        assert out["n_sats"] == 2
        assert out["repo_cached"] is True
        assert Path(tmp_store_uri).is_dir()
        assert (tmp_path / "out" / "20240115").is_dir()

    def test_asset_can_use_a_stand_in_for_the_model(self):
        """The documented way to keep a checkpoint out of an asset test."""

        @asset
        def uses_model(model: ModelResource) -> str:
            return model.model()

        class FakeModel:
            def model(self):
                return "fake-model"

            def disparity(self):
                return "fake-disparity"

            def available(self):
                return True

        result = materialize([uses_model], resources={"model": FakeModel()})
        assert result.success
        assert result.output_for_node("uses_model") == "fake-model"

    def test_private_attr_injection_does_not_survive_materialize(self, tmp_path):
        """Guards the caveat the docstring warns about.

        Dagster rebuilds the resource from its config fields, so a
        pre-materialize ``set_model`` is dropped and the lazy load runs —
        which is why a stand-in object is the documented route.
        """

        @asset
        def uses_model(model: ModelResource) -> str:
            return model.model()

        res = ModelResource(student_ckpt=str(tmp_path / "absent.ckpt"))
        res.set_model("fake-model")

        result = materialize(
            [uses_model],
            resources={"model": res},
            raise_on_error=False,
        )
        assert not result.success
        # It must fail for the documented reason — the lazy load running
        # against a checkpoint that is not there — not for some unrelated
        # config or schema error that would keep this test green while
        # the behaviour it pins changed underneath it.
        failures = result.filter_events(lambda e: e.event_type_value == "STEP_FAILURE")
        assert failures
        info = failures[0].step_failure_data.error
        assert "FileNotFoundError" in str(info)
        assert "student_ckpt" in str(info)


class TestBandConstants:
    """The inlined band lists must not drift from the research code."""

    def test_band_constants_match_upstream(self):
        # Imported here, not at module scope: this is the one place the
        # tests accept the torch cost that `operational.config` avoids.
        from stereo_winds.student_dataset import (
            DEFAULT_FLOW_BANDS as UPSTREAM_FLOW,
            DEFAULT_RAD_BANDS as UPSTREAM_RAD,
        )

        assert DEFAULT_FLOW_BANDS == tuple(UPSTREAM_FLOW)
        assert DEFAULT_RAD_BANDS == tuple(UPSTREAM_RAD)

    def test_config_defaults_use_them(self):
        cfg = OperationalConfig()
        assert cfg.flow_bands == DEFAULT_FLOW_BANDS
        assert cfg.rad_bands == DEFAULT_RAD_BANDS


class TestNaiveUtc:
    """Dagster hands out tz-aware times; everything downstream is naive."""

    def test_naive_input_is_unchanged(self):
        assert as_naive_utc(T0) is T0

    def test_aware_utc_loses_only_the_tzinfo(self):
        aware = T0.replace(tzinfo=timezone.utc)
        assert as_naive_utc(aware) == T0
        assert as_naive_utc(aware).tzinfo is None

    def test_offset_is_converted_not_truncated(self):
        # 07:00 at UTC-5 is 12:00 UTC.
        aware = datetime(
            2024,
            1,
            15,
            7,
            0,
            tzinfo=timezone(timedelta(hours=-5)),
        )
        assert as_naive_utc(aware) == T0

    def test_membership_would_fail_without_normalising(self):
        """The trap this helper exists to remove."""
        stored = {T0}
        aware = T0.replace(tzinfo=timezone.utc)
        assert aware not in stored  # the bug
        assert as_naive_utc(aware) in stored  # the fix

    def test_has_time_accepts_aware_and_naive(self, monkeypatch):
        # A Pythonic resource is frozen, so stub the upstream lookup and
        # hand the resource a repo it never has to open.
        import stereo_winds.icechunk_output as ico

        monkeypatch.setattr(ico, "icechunk_existing_times", lambda repo, br: {T0})
        res = IcechunkStoreResource(store_uri="s3://nowhere/at-all")
        res.set_repo(object())

        assert res.has_time(T0)
        assert res.has_time(T0.replace(tzinfo=timezone.utc))
        assert res.has_time(datetime(2024, 1, 15, 7, 0, tzinfo=timezone(timedelta(hours=-5))))
        assert not res.has_time(datetime(2024, 1, 15, 13, 0))

    def test_paths_are_tz_insensitive(self, tmp_path):
        res = PathsResource(output_dir=str(tmp_path))
        assert res.mosaic_path(T0) == res.mosaic_path(T0.replace(tzinfo=timezone.utc))


class TestToConfigWiring:
    """`to_config` must not silently substitute its own defaults."""

    def test_device_and_row_strip_come_from_the_model_resource(self, tmp_path):
        settings = RunSettingsResource()
        model = ModelResource(device="cuda", row_strip=256)
        cfg = settings.to_config(model=model)
        assert cfg.device == "cuda"
        assert cfg.row_strip == 256

    def test_paths_and_store_are_picked_up(self, tmp_path):
        settings = RunSettingsResource()
        cfg = settings.to_config(
            paths=PathsResource(output_dir=str(tmp_path / "out")),
            store=IcechunkStoreResource(store_uri="s3://b/p"),
        )
        assert cfg.output_dir == tmp_path / "out"
        assert cfg.store_uri == "s3://b/p"

    def test_explicit_overrides_beat_the_siblings(self):
        settings = RunSettingsResource()
        cfg = settings.to_config(
            model=ModelResource(device="cuda"),
            device="cpu",
        )
        assert cfg.device == "cpu"

    def test_without_siblings_the_defaults_are_documented_ones(self):
        cfg = RunSettingsResource().to_config()
        assert cfg.device == OperationalConfig().device
        assert cfg.row_strip == OperationalConfig().row_strip
