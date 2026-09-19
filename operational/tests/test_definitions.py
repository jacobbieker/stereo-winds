"""Tests for the operational jobs, backstop schedule and code location.

Everything here is offline: no checkpoint, no GPU, no network, no
icechunk store.  That is not incidental — a code location that cannot be
loaded on a bare machine cannot be loaded by ``dagster dev`` either, so
the laziness of the resources is itself one of the things under test.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest
from dagster import (
    AssetKey,
    DagsterInstance,
    DefaultScheduleStatus,
    Definitions,
    EnvVar,
    RetryPolicy,
    StaticPartitionsDefinition,
    build_schedule_context,
    materialize,
)

from operational.config import OperationalConfig
from operational.core.partitions import build_partitions_def
from operational.definitions import (
    CHECKPOINT_ENV_VAR,
    DEFAULT_CHECKPOINT,
    build_definitions,
    check_mosaic_inputs,
    check_satellite_agreement,
    default_resources,
    defs,
    discover_sensors,
    resolve_amv_assets,
    resolve_partitions_def,
)
from operational.jobs import (
    BACKSTOP_MINUTE_OF_HOUR,
    BACKSTOP_SCHEDULE_NAME,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_RETRY_POLICY,
    FULL_JOB_NAME,
    MAX_CONCURRENT_ENV_VAR,
    RUN_TAGS,
    build_backstop_schedule,
    build_full_job,
    build_satellite_job,
    max_concurrent_from_env,
    modest_executor,
    build_satellite_jobs,
    satellite_job_name,
    supports_minute_of_hour,
)

#: Repository root, so the subprocess tests can import the package the
#: same way ``dagster dev`` would.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The satellites the code location was actually built for.  Read from
#: the environment, not from the dataclass defaults, so a developer with
#: ``STEREO_WINDS_OP_SATELLITES`` exported does not fail a dozen tests
#: for reasons unrelated to the code.
DEFAULT_SATELLITES = OperationalConfig.from_env().satellites

#: Every asset key the code location is expected to expose.
EXPECTED_ASSET_KEYS = {
    *(AssetKey(f"amv_{sat_id}") for sat_id in DEFAULT_SATELLITES),
    AssetKey("global_mosaic"),
    AssetKey("published_mosaic"),
}

#: The cadence the asset modules were imported at.
BUILT_CADENCE_MINUTES = OperationalConfig.from_env().cadence_minutes

#: A partition key in the ``%Y-%m-%d-%H:%M`` format the pipeline uses.
A_PARTITION_KEY = "2026-02-01-06:00"


def _selected_keys(job_def) -> set[AssetKey]:
    """Asset keys a resolved job will execute."""
    return set(job_def.asset_layer.executable_asset_keys)


def _run_module_script(
    script: str, env_overrides: dict[str, str]
) -> subprocess.CompletedProcess:
    """Import the code location in a fresh interpreter and run ``script``.

    A subprocess is the only honest way to test settings the asset
    modules read at *their* import: once they are in ``sys.modules``, the
    graph they built is fixed for the life of the process.

    Parameters
    ----------
    script : str
        Python source, dedented before execution.
    env_overrides : dict of str
        Environment variables layered over the current environment, which
        is first stripped of every ``STEREO_WINDS_OP_*`` setting so a
        stray one in the caller's shell cannot change the result.

    Returns
    -------
    subprocess.CompletedProcess
        With ``stdout`` and ``stderr`` captured as text.
    """
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("STEREO_WINDS_OP_")
    }
    env.update(env_overrides)
    inherited = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        os.pathsep.join([str(REPO_ROOT), inherited]) if inherited else str(REPO_ROOT)
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=300,
    )


class TestDefinitionsLoad:
    """The module-level ``defs`` is what ``dagster dev`` loads."""

    def test_is_a_definitions(self):
        assert isinstance(defs, Definitions)

    def test_validates_loadable(self):
        # Raises if anything in the code location is inconsistent: a
        # sensor targeting a job that does not exist, a resource key an
        # asset requires but nothing provides, mismatched partitions.
        Definitions.validate_loadable(defs)

    def test_expected_asset_keys_are_present(self):
        assert set(defs.resolve_asset_graph().get_all_asset_keys()) == EXPECTED_ASSET_KEYS

    def test_every_asset_is_partitioned_the_same_way(self):
        partitions = {spec.partitions_def for spec in defs.resolve_all_asset_specs()}
        assert len(partitions) == 1
        assert partitions.pop() is not None

    def test_partition_keys_use_the_agreed_format(self):
        spec = next(iter(defs.resolve_all_asset_specs()))
        keys = spec.partitions_def.get_partition_keys(
            current_time=datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc)
        )
        assert keys[0] == "2026-01-01-00:00"

    def test_mosaic_depends_on_every_satellite(self):
        graph = defs.resolve_asset_graph()
        mosaic = graph.get(AssetKey("global_mosaic"))
        assert mosaic.parent_keys == {
            AssetKey(f"amv_{sat_id}") for sat_id in DEFAULT_SATELLITES
        }

    def test_publish_depends_on_the_mosaic(self):
        graph = defs.resolve_asset_graph()
        published = graph.get(AssetKey("published_mosaic"))
        assert published.parent_keys == {AssetKey("global_mosaic")}


class TestFullJob:
    """The job the sensor and the backstop schedule both launch."""

    def test_is_registered(self):
        assert FULL_JOB_NAME in {job.name for job in defs.jobs}

    def test_selects_the_whole_graph(self):
        job_def = defs.resolve_job_def(FULL_JOB_NAME)
        assert _selected_keys(job_def) == EXPECTED_ASSET_KEYS

    def test_is_partitioned(self):
        job_def = defs.resolve_job_def(FULL_JOB_NAME)
        assert job_def.partitions_def is not None

    def test_carries_the_operational_run_tags(self):
        job_def = defs.resolve_job_def(FULL_JOB_NAME)
        for key, value in RUN_TAGS.items():
            assert job_def.tags[key] == value

    def test_carries_the_retry_policy(self):
        job_def = defs.resolve_job_def(FULL_JOB_NAME)
        assert job_def.op_retry_policy == DEFAULT_RETRY_POLICY

    def test_runs_steps_in_separate_processes(self):
        # Process-per-step is what lets the OS reclaim a full disk's
        # worth of memory between satellites.
        job_def = defs.resolve_job_def(FULL_JOB_NAME)
        assert job_def.executor_def.name == "multiprocess"

    def test_accepts_a_narrower_selection(self):
        job = build_full_job(name="probe_job", selection=[AssetKey("global_mosaic")])
        probe = Definitions(assets=defs.assets, jobs=[job])
        assert _selected_keys(probe.resolve_job_def("probe_job")) == {
            AssetKey("global_mosaic")
        }


class TestSatelliteJobs:
    """One narrow re-run job per satellite."""

    def test_one_job_per_satellite(self):
        names = {job.name for job in defs.jobs}
        for sat_id in DEFAULT_SATELLITES:
            assert satellite_job_name(sat_id) in names

    @pytest.mark.parametrize("sat_id", DEFAULT_SATELLITES)
    def test_selects_only_that_satellite(self, sat_id):
        job_def = defs.resolve_job_def(satellite_job_name(sat_id))
        assert _selected_keys(job_def) == {AssetKey(f"amv_{sat_id}")}

    def test_can_opt_into_downstream(self):
        job = build_satellite_job(
            "goes18", [AssetKey("amv_goes18")], include_downstream=True
        )
        probe = Definitions(assets=defs.assets, jobs=[job])
        selected = _selected_keys(probe.resolve_job_def(satellite_job_name("goes18")))
        assert selected == {
            AssetKey("amv_goes18"),
            AssetKey("global_mosaic"),
            AssetKey("published_mosaic"),
        }

    def test_rejects_an_empty_key_set(self):
        with pytest.raises(ValueError, match="no asset keys"):
            build_satellite_job("goes18", [])

    @pytest.mark.parametrize(
        "sat_id, expected",
        [
            ("goes18", "operational_amv_goes18_job"),
            ("mtg-i1", "operational_amv_mtg_i1_job"),
            ("msg-iodc", "operational_amv_msg_iodc_job"),
        ],
    )
    def test_job_names_are_identifiers(self, sat_id, expected):
        # Dagster rejects punctuation in a job name at repository
        # resolution, which fails the whole code location and not just
        # the offending job.
        assert satellite_job_name(sat_id) == expected

    def test_rejects_a_satellite_id_with_no_usable_name(self):
        with pytest.raises(ValueError, match="no usable job name"):
            satellite_job_name("--")

    def test_colliding_slugs_are_rejected(self):
        # Two ids that slug the same would make Dagster reject the whole
        # repository for a duplicate job name, naming neither satellite.
        with pytest.raises(ValueError, match="both give the job name"):
            build_satellite_jobs(
                {
                    "mtg-i1": [AssetKey("amv_mtg_i1")],
                    "mtg_i1": [AssetKey("amv_mtg_i1_b")],
                }
            )

    def test_distinct_slugs_are_accepted(self):
        jobs = build_satellite_jobs(
            {
                "mtg-i1": [AssetKey("amv_mtg_i1")],
                "msg-iodc": [AssetKey("amv_msg_iodc")],
            }
        )
        assert [job.name for job in jobs] == [
            "operational_amv_mtg_i1_job",
            "operational_amv_msg_iodc_job",
        ]

    def test_the_whole_ring_loads(self):
        # Two of the six ring satellites carry a hyphen; the default
        # four do not, so only the full ring exercises the slugging.
        result = _run_module_script(
            """
            from dagster import Definitions

            from operational.definitions import defs

            Definitions.validate_loadable(defs)
            print(sorted(job.name for job in defs.jobs))
            """,
            {
                "STEREO_WINDS_OP_SATELLITES": (
                    "goes18,goes19,mtg-i1,msg-iodc,gk2a,himawari9"
                )
            },
        )
        assert result.returncode == 0, result.stderr
        assert "operational_amv_mtg_i1_job" in result.stdout
        assert "operational_amv_msg_iodc_job" in result.stdout

    def test_shares_the_operational_defaults(self):
        job_def = defs.resolve_job_def(satellite_job_name("goes18"))
        assert job_def.op_retry_policy == DEFAULT_RETRY_POLICY
        assert job_def.executor_def.name == "multiprocess"


class TestConcurrencyLimit:
    """Four full disks at once is what OOM-killed the CLI; guard it."""

    def test_default_is_one(self):
        assert DEFAULT_MAX_CONCURRENT == 1

    def test_unset_environment_uses_the_default(self):
        assert max_concurrent_from_env({}) == DEFAULT_MAX_CONCURRENT

    def test_blank_environment_uses_the_default(self):
        assert max_concurrent_from_env({MAX_CONCURRENT_ENV_VAR: "  "}) == (
            DEFAULT_MAX_CONCURRENT
        )

    def test_environment_override_takes_effect(self):
        assert max_concurrent_from_env({MAX_CONCURRENT_ENV_VAR: "3"}) == 3

    @pytest.mark.parametrize("raw", ["lots", "0", "-2", "1.5"])
    def test_bad_values_fall_back_instead_of_failing_the_load(self, raw):
        # A typo in a deployment's environment must not take the whole
        # code location down.
        assert max_concurrent_from_env({MAX_CONCURRENT_ENV_VAR: raw}) == (
            DEFAULT_MAX_CONCURRENT
        )

    def test_reads_os_environ_by_default(self, monkeypatch):
        monkeypatch.setenv(MAX_CONCURRENT_ENV_VAR, "4")
        assert max_concurrent_from_env() == 4

    def test_executor_rejects_a_nonsense_limit(self):
        with pytest.raises(ValueError, match=">= 1"):
            modest_executor(0)

    def test_executor_is_multiprocess(self):
        assert modest_executor(2).name == "multiprocess"


class TestBackstopSchedule:
    """Belt and braces behind the availability sensor."""

    def test_is_registered(self):
        assert BACKSTOP_SCHEDULE_NAME in {schedule.name for schedule in defs.schedules}

    def test_targets_the_full_job(self):
        schedule = defs.resolve_schedule_def(BACKSTOP_SCHEDULE_NAME)
        assert schedule.job_name == FULL_JOB_NAME

    def test_is_running_by_default(self):
        # A backstop somebody has to remember to switch on is not one.
        schedule = defs.resolve_schedule_def(BACKSTOP_SCHEDULE_NAME)
        assert schedule.default_status == DefaultScheduleStatus.RUNNING

    def test_fires_once_per_hour_off_the_hour_boundary(self):
        schedule = defs.resolve_schedule_def(BACKSTOP_SCHEDULE_NAME)
        assert schedule.cron_schedule == f"{BACKSTOP_MINUTE_OF_HOUR} * * * *"

    @pytest.mark.parametrize(
        "cadence_minutes, expected",
        [(60, True), (1440, True), (10, False), (30, False), (120, False), (360, False)],
    )
    def test_which_cadences_accept_an_hour_offset(self, cadence_minutes, expected):
        # Dagster only lets an offset be applied to a plain hourly or
        # daily partition cron; "*/10 * * * *" and "0 0,6,12,18 * * *"
        # raise while the code location is loading.
        partitions = build_partitions_def(datetime(2026, 1, 1), cadence_minutes)
        assert supports_minute_of_hour(partitions) is expected

    def test_unpartitioned_input_takes_no_offset(self):
        assert supports_minute_of_hour(StaticPartitionsDefinition(["a"])) is False

    def test_an_unusable_grid_leaves_the_boundary_cron(self):
        # Partitions that take no offset must leave the schedule firing
        # on the partition boundary; passing the offset through would
        # make the code location unloadable for the sake of a firing
        # time.
        partitions = build_partitions_def(datetime(2026, 1, 1), 360)
        job = build_full_job(name="probe_job")
        schedule = build_backstop_schedule(job, partitions, name="probe_schedule")
        probe = Definitions(assets=defs.assets, jobs=[job], schedules=[schedule])
        # The job itself is hourly, so the boundary cron is the hourly one.
        assert probe.resolve_schedule_def("probe_schedule").cron_schedule == (
            "0 * * * *"
        )

    def test_the_shipped_schedule_matches_the_shipped_partitions(self):
        # The offset gate reads the partitions it is handed, so the code
        # location must hand it the grid the job really resolves to.
        partitions = defs.resolve_all_asset_specs()[0].partitions_def
        schedule = defs.resolve_schedule_def(BACKSTOP_SCHEDULE_NAME)
        minute = schedule.cron_schedule.split()[0]
        if supports_minute_of_hour(partitions):
            assert minute == str(BACKSTOP_MINUTE_OF_HOUR)
        else:
            assert minute == partitions.cron_schedule.split()[0]

    def test_tick_requests_the_most_recently_closed_partition(self):
        schedule = defs.resolve_schedule_def(BACKSTOP_SCHEDULE_NAME)
        with DagsterInstance.ephemeral() as instance:
            context = build_schedule_context(
                instance=instance,
                scheduled_execution_time=datetime(
                    2026, 2, 1, 6, BACKSTOP_MINUTE_OF_HOUR, tzinfo=timezone.utc
                ),
            )
            requests = schedule.evaluate_tick(context).run_requests
        assert [request.partition_key for request in requests] == ["2026-02-01-05:00"]

    def test_tick_tags_the_run_as_schedule_triggered(self):
        schedule = defs.resolve_schedule_def(BACKSTOP_SCHEDULE_NAME)
        with DagsterInstance.ephemeral() as instance:
            context = build_schedule_context(
                instance=instance,
                scheduled_execution_time=datetime(
                    2026, 2, 1, 6, BACKSTOP_MINUTE_OF_HOUR, tzinfo=timezone.utc
                ),
            )
            request = schedule.evaluate_tick(context).run_requests[0]
        assert request.tags["stereo_winds/trigger"] == "backstop-schedule"
        assert request.tags["stereo_winds/pipeline"] == RUN_TAGS[
            "stereo_winds/pipeline"
        ]

    def test_tick_uses_the_partition_as_its_run_key(self):
        # Which is what stops one schedule from launching the same
        # partition twice if a tick is retried.
        schedule = defs.resolve_schedule_def(BACKSTOP_SCHEDULE_NAME)
        with DagsterInstance.ephemeral() as instance:
            context = build_schedule_context(
                instance=instance,
                scheduled_execution_time=datetime(
                    2026, 2, 1, 6, BACKSTOP_MINUTE_OF_HOUR, tzinfo=timezone.utc
                ),
            )
            request = schedule.evaluate_tick(context).run_requests[0]
        assert request.run_key == request.partition_key


class TestSensors:
    """The availability sensor is registered and points at a real job."""

    def test_a_sensor_is_registered(self):
        assert defs.sensors, "no sensor registered in the code location"

    def test_sensor_targets_a_job_that_exists(self):
        job_names = {job.name for job in defs.jobs}
        for sensor in defs.sensors:
            assert sensor.job_name in job_names

    def test_expected_sensor_name(self):
        from operational.sensors import AVAILABILITY_SENSOR_NAME

        assert AVAILABILITY_SENSOR_NAME in {sensor.name for sensor in defs.sensors}

    def test_discovery_deduplicates_aliases(self):
        # A sensor re-exported under a second name must not be
        # registered twice, which Dagster would reject.
        namespace = type(sys)("probe")
        namespace.first = defs.sensors[0]
        namespace.second = defs.sensors[0]
        assert len(discover_sensors(namespace)) == 1

    def test_discovery_ignores_modules_without_sensors(self):
        namespace = type(sys)("probe")
        namespace.not_a_sensor = 42
        assert discover_sensors(namespace) == []


class TestAmvAssetResolution:
    """Where the per-satellite layer of the graph comes from."""

    def test_prefers_the_modules_prebuilt_assets(self):
        resolved = resolve_amv_assets(
            defs.resolve_all_asset_specs()[0].partitions_def, OperationalConfig()
        )
        assert set(resolved) == set(DEFAULT_SATELLITES)

    def test_falls_back_to_the_factory(self):
        # A module offering only the factory must still yield a graph,
        # built from the configured satellite list.
        from operational.assets import amv_assets as real

        stub = type(sys)("amv_stub")
        stub.build_amv_asset = real.build_amv_asset
        partitions = build_partitions_def(datetime(2026, 1, 1), 60)
        resolved = resolve_amv_assets(
            partitions, OperationalConfig(satellites=("goes19", "mtg-i1")), stub
        )
        assert set(resolved) == {"goes19", "mtg-i1"}

    def test_fallback_slugifies_punctuated_ids(self):
        # Asset names must be identifiers; satellite ids are not.
        from operational.assets import amv_assets as real

        stub = type(sys)("amv_stub")
        stub.build_amv_asset = real.build_amv_asset
        partitions = build_partitions_def(datetime(2026, 1, 1), 60)
        resolved = resolve_amv_assets(
            partitions, OperationalConfig(satellites=("mtg-i1",)), stub
        )
        assert AssetKey("amv_mtg_i1") in resolved["mtg-i1"].keys

    def test_empty_prebuilt_mapping_falls_back(self):
        from operational.assets import amv_assets as real

        stub = type(sys)("amv_stub")
        stub.AMV_ASSETS_BY_SAT = {}
        stub.build_amv_asset = real.build_amv_asset
        partitions = build_partitions_def(datetime(2026, 1, 1), 60)
        resolved = resolve_amv_assets(
            partitions, OperationalConfig(satellites=("goes19",)), stub
        )
        assert set(resolved) == {"goes19"}


class TestResources:
    """Lazy by contract: constructing one must not touch anything."""

    def test_expected_resource_keys(self):
        assert set(defs.resources) == {"paths", "store", "model", "run_settings"}

    def test_defaults_construct_without_any_files(self, tmp_path):
        missing = tmp_path / "definitely" / "not" / "here"
        assert not missing.exists()
        resources = default_resources(
            OperationalConfig(
                output_dir=missing / "out",
                store_uri=str(missing / "store.icechunk"),
            )
        )
        assert set(resources) == {"paths", "store", "model", "run_settings"}
        # Nothing was created on the way.
        assert not missing.exists()

    def test_run_settings_mirror_the_config(self):
        config = OperationalConfig(satellites=("goes19",), cadence_minutes=60)
        settings = default_resources(config, env={})["run_settings"]
        assert settings.satellites == ["goes19"]
        assert settings.cadence_minutes == 60

    def test_unset_environment_uses_literal_defaults(self):
        config = OperationalConfig(store_uri="local/store.icechunk", device="cpu")
        resources = default_resources(config, env={})
        assert resources["store"].store_uri == "local/store.icechunk"
        assert resources["model"].checkpoint_path == DEFAULT_CHECKPOINT
        assert resources["model"].device == "cpu"

    def test_process_environment_is_bound_late(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STEREO_WINDS_OP_STORE_URI", "s3://bucket/prefix")
        monkeypatch.setenv("STEREO_WINDS_OP_OUTPUT_DIR", str(tmp_path))
        monkeypatch.setenv(CHECKPOINT_ENV_VAR, str(tmp_path / "student.ckpt"))
        monkeypatch.setenv("STEREO_WINDS_OP_DEVICE", "cuda")
        resources = default_resources(OperationalConfig())
        # EnvVar, not the current value: the variable is re-read when a
        # run launches, so repointing it needs no redeploy.
        assert isinstance(resources["store"].store_uri, EnvVar)
        assert resources["store"].store_uri.env_var_name == "STEREO_WINDS_OP_STORE_URI"
        assert resources["model"].checkpoint_path.env_var_name == CHECKPOINT_ENV_VAR
        assert resources["model"].device.env_var_name == "STEREO_WINDS_OP_DEVICE"
        assert resources["paths"].output_dir.env_var_name == (
            "STEREO_WINDS_OP_OUTPUT_DIR"
        )

    def test_env_var_resolves_to_the_real_value(self, monkeypatch):
        monkeypatch.setenv("STEREO_WINDS_OP_STORE_URI", "s3://bucket/prefix")
        resources = default_resources(OperationalConfig())
        assert resources["store"].store_uri.get_value() == "s3://bucket/prefix"

    def test_an_explicit_mapping_is_bound_eagerly(self):
        # An EnvVar would be resolved against os.environ at run launch,
        # not against this mapping, and the run would die on a variable
        # that was never set.
        resources = default_resources(
            OperationalConfig(),
            env={"STEREO_WINDS_OP_STORE_URI": "s3://bucket/prefix"},
        )
        assert resources["store"].store_uri == "s3://bucket/prefix"
        assert not isinstance(resources["store"].store_uri, EnvVar)

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_variable_counts_as_unset(self, monkeypatch, blank):
        # An empty value in a unit file must fall back, not bind the
        # resource to "".
        monkeypatch.setenv("STEREO_WINDS_OP_DEVICE", blank)
        resources = default_resources(OperationalConfig(device="cpu"))
        assert resources["model"].device == "cpu"

    def test_run_settings_follow_the_built_graph(self):
        # No step may be told about a satellite that has no asset.
        settings = defs.resources["run_settings"]
        assert settings.satellites == list(DEFAULT_SATELLITES)

    def test_satellites_override_wins_over_the_config(self):
        resources = default_resources(
            OperationalConfig(satellites=("goes18", "goes19")),
            env={},
            satellites=["goes19"],
        )
        assert resources["run_settings"].satellites == ["goes19"]


class TestEnvironmentShapesTheGraph:
    """Structural settings are read eagerly, at code-location load."""

    def test_satellite_list_decides_the_assets(self):
        # The asset modules build the graph at their own import, so the
        # satellite list reaches the code location through them.
        result = _run_module_script(
            """
            from operational.definitions import defs
            from operational.jobs import FULL_JOB_NAME

            job = defs.resolve_job_def(FULL_JOB_NAME)
            print(sorted(k.to_user_string()
                         for k in job.asset_layer.executable_asset_keys))
            """,
            {"STEREO_WINDS_OP_SATELLITES": "goes19,himawari9"},
        )
        assert result.returncode == 0, result.stderr
        assert "amv_goes19" in result.stdout
        assert "amv_goes18" not in result.stdout

    @pytest.mark.parametrize(
        "cadence_minutes, expected_cron",
        [
            ("10", "*/10 * * * *"),
            ("60", f"{BACKSTOP_MINUTE_OF_HOUR} * * * *"),
            ("360", "0 0,6,12,18 * * *"),
        ],
    )
    def test_cadence_reaches_the_backstop(self, cadence_minutes, expected_cron):
        # Every cadence the pipeline supports must produce a loadable
        # code location, offset or no offset.
        result = _run_module_script(
            """
            from dagster import Definitions

            from operational.definitions import defs
            from operational.jobs import BACKSTOP_SCHEDULE_NAME

            Definitions.validate_loadable(defs)
            print("CRON", defs.resolve_schedule_def(
                BACKSTOP_SCHEDULE_NAME).cron_schedule)
            """,
            {"STEREO_WINDS_OP_CADENCE_MINUTES": cadence_minutes},
        )
        assert result.returncode == 0, result.stderr
        assert f"CRON {expected_cron}" in result.stdout

    def test_mismatched_partitions_warn(self, caplog):
        # The asset modules were imported at one cadence; asking for a
        # different one now means the two were configured at different
        # moments.
        other = 30 if BUILT_CADENCE_MINUTES != 30 else 15
        with caplog.at_level("WARNING", logger="operational.definitions"):
            resolved = resolve_partitions_def(
                OperationalConfig(cadence_minutes=other), env={}
            )
        assert "the assets win" in caplog.text
        assert resolved == defs.resolve_all_asset_specs()[0].partitions_def

    def test_matching_partitions_are_quiet(self, caplog):
        with caplog.at_level("WARNING", logger="operational.definitions"):
            resolve_partitions_def(
                OperationalConfig(cadence_minutes=BUILT_CADENCE_MINUTES), env={}
            )
        assert caplog.text == ""

    def test_mismatched_satellites_warn(self, caplog):
        with caplog.at_level("WARNING", logger="operational.definitions"):
            difference = check_satellite_agreement(
                OperationalConfig(satellites=("goes19",)), DEFAULT_SATELLITES
            )
        assert difference
        assert "the assets win" in caplog.text

    def test_matching_satellite_list_is_quiet(self, caplog):
        with caplog.at_level("WARNING", logger="operational.definitions"):
            assert check_mosaic_inputs(
                AssetKey(f"amv_{sat_id}") for sat_id in DEFAULT_SATELLITES
            ) == set()
        assert caplog.text == ""

    def test_check_reports_the_unbuilt_inputs(self):
        dangling = check_mosaic_inputs([AssetKey("amv_goes19")])
        assert AssetKey("amv_goes18") in dangling

    def test_satellite_list_decides_the_rerun_jobs(self):
        result = _run_module_script(
            """
            from operational.definitions import defs
            print(sorted(job.name for job in defs.jobs))
            """,
            {"STEREO_WINDS_OP_SATELLITES": "goes19"},
        )
        assert result.returncode == 0, result.stderr
        assert satellite_job_name("goes19") in result.stdout
        assert satellite_job_name("goes18") not in result.stdout
        assert FULL_JOB_NAME in result.stdout

    def test_env_is_honoured_and_not_quietly_replaced(self):
        # A passed mapping must actually be read; falling back to the
        # real os.environ would make every `env=` argument a lie.
        assert check_satellite_agreement(
            OperationalConfig(satellites=("goes19",)), ["goes19"]
        ) == set()
        rebuilt = build_definitions(env={"STEREO_WINDS_OP_DEVICE": "cuda"})
        assert rebuilt.resources["model"].device == "cuda"

    def test_max_concurrent_reaches_the_jobs(self):
        # The one setting you least want read from an unexpected place.
        rebuilt = build_definitions(env={MAX_CONCURRENT_ENV_VAR: "4"})
        assert rebuilt.resolve_job_def(FULL_JOB_NAME).executor_def.name == (
            "multiprocess"
        )
        assert max_concurrent_from_env({MAX_CONCURRENT_ENV_VAR: "4"}) == 4


class TestImportIsSideEffectFree:
    """Loading the code location must not touch disk, model or network."""

    def test_import_creates_nothing_on_disk(self, tmp_path):
        root = tmp_path / "nowhere"
        overrides = {
            "STEREO_WINDS_OP_OUTPUT_DIR": str(root / "out"),
            "STEREO_WINDS_OP_STORE_URI": str(root / "store.icechunk"),
            CHECKPOINT_ENV_VAR: str(root / "student.ckpt"),
        }
        result = _run_module_script(
            """
            from operational.definitions import defs
            assert defs is not None
            print("imported")
            """,
            overrides,
        )
        assert result.returncode == 0, result.stderr
        assert "imported" in result.stdout
        # No output directory, no store, nothing.
        assert not root.exists()

    def test_import_never_opens_the_checkpoint(self, tmp_path):
        checkpoint = tmp_path / "student.ckpt"
        checkpoint.write_bytes(b"not really a checkpoint")
        result = _run_module_script(
            """
            import os
            import sys

            watched = os.environ["STEREO_WINDS_OP_CHECKPOINT"]
            opened = []

            def _audit(event, args):
                if event == "open" and args and isinstance(args[0], str):
                    if args[0] == watched:
                        opened.append(args[0])

            sys.addaudithook(_audit)
            import operational.definitions  # noqa: F401
            print("OPENED", opened)
            """,
            {CHECKPOINT_ENV_VAR: str(checkpoint)},
        )
        assert result.returncode == 0, result.stderr
        assert "OPENED []" in result.stdout

    def test_loads_the_way_dagster_dev_loads_it(self, tmp_path):
        # ``dagster dev -m operational.definitions`` imports the module
        # and looks for a module-level Definitions.
        result = _run_module_script(
            """
            import importlib

            from dagster import Definitions

            module = importlib.import_module("operational.definitions")
            assert isinstance(module.defs, Definitions)
            Definitions.validate_loadable(module.defs)
            print("loadable")
            """,
            {"STEREO_WINDS_OP_OUTPUT_DIR": str(tmp_path / "missing")},
        )
        assert result.returncode == 0, result.stderr
        assert "loadable" in result.stdout


class TestMaterializeOnePartition:
    """The wired graph actually runs, end to end, for one timestamp."""

    def test_full_graph_materializes(self, tmp_path):
        config = OperationalConfig(
            output_dir=tmp_path / "out",
            store_uri=str(tmp_path / "store.icechunk"),
        )
        result = materialize(
            list(defs.assets),
            partition_key=A_PARTITION_KEY,
            resources=default_resources(config, env={}),
        )
        assert result.success
        materialized = {
            event.asset_key
            for event in result.get_asset_materialization_events()
        }
        assert materialized == EXPECTED_ASSET_KEYS

    def test_one_satellite_materializes_on_its_own(self, tmp_path):
        config = OperationalConfig(output_dir=tmp_path / "out")
        goes18 = [
            asset_def
            for asset_def in defs.assets
            if AssetKey("amv_goes18") in getattr(asset_def, "keys", ())
        ]
        result = materialize(
            goes18,
            partition_key=A_PARTITION_KEY,
            resources=default_resources(config, env={}),
        )
        assert result.success

    def test_retry_policy_is_a_real_policy(self):
        # Cheap guard against the policy degrading to None in a refactor:
        # a step that never retries turns a blip into a lost timestamp.
        assert isinstance(DEFAULT_RETRY_POLICY, RetryPolicy)
        assert DEFAULT_RETRY_POLICY.max_retries >= 1
        assert DEFAULT_RETRY_POLICY.delay and DEFAULT_RETRY_POLICY.delay > 0
