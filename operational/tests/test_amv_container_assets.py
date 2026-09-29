"""Tests for the containerised per-satellite AMV assets."""

import datetime as dt
from unittest.mock import MagicMock

import pytest
from dagster import build_asset_context

from operational.amv_container import AmvContainerResource, satellite_env
from operational.assets.amv_container_assets import (
    AMV_CONTAINER_GROUP,
    amv_container_asset_name,
    build_amv_container_asset,
    build_amv_container_assets,
)
from operational.core.partitions import DEFAULT_START, build_partitions_def
from operational.resources import RunSettingsResource

PARTITIONS = build_partitions_def(DEFAULT_START, 30)
T0 = dt.datetime(2026, 9, 20, 12, 0)


@pytest.fixture
def resource():
    return AmvContainerResource(
        aws_access_key_id="aws-key",
        aws_secret_access_key="aws-secret",
    )


@pytest.fixture
def settings():
    return RunSettingsResource(
        satellites=["goes18"],
        flow_bands=["C08", "C14"],
        rad_bands=["C07", "C13"],
        cadence_minutes=30,
        availability_tolerance_minutes=5,
        resolution_m=5000.0,
    )


class TestEnvironment:
    def test_asks_for_one_satellite_only(self):
        env = satellite_env(
            "mtg-i1",
            T0,
            icechunk_base="s3://b/amv",
            resolution_m=5000,
            flow_bands=["C08"],
            rad_bands=["C13"],
        )
        assert env["AMV_SATELLITES"] == "mtg-i1"

    def test_skips_the_global_mosaic(self):
        """A single-satellite run asked for a mosaic would build one with
        exactly one contributor; the mosaic is a separate step."""
        env = satellite_env(
            "goes18",
            T0,
            icechunk_base="s3://b/amv",
            resolution_m=5000,
            flow_bands=["C08"],
            rad_bands=["C13"],
        )
        assert env["AMV_SKIP_GLOBAL"] == "true"

    def test_writes_no_netcdf(self):
        """The NetCDF is only readable by a mosaic on the same host, which
        is the coupling the container removes."""
        env = satellite_env(
            "goes18",
            T0,
            icechunk_base="s3://b/amv",
            resolution_m=5000,
            flow_bands=["C08"],
            rad_bands=["C13"],
        )
        assert env["AMV_NO_NETCDF"] == "true"
        assert env["AMV_SATELLITE_ICECHUNK_BASE"] == "s3://b/amv"

    def test_carries_no_credentials(self):
        env = satellite_env(
            "goes18",
            T0,
            icechunk_base="s3://b/amv",
            resolution_m=5000,
            flow_bands=["C08"],
            rad_bands=["C13"],
        )
        assert not [k for k in env if "KEY" in k or "SECRET" in k]

    def test_timestamp_is_what_the_script_parses(self):
        env = satellite_env(
            "goes18",
            T0,
            icechunk_base="s3://b/amv",
            resolution_m=5000,
            flow_bands=["C08"],
            rad_bands=["C13"],
        )
        assert env["AMV_TIME"] == "2026-09-20T12:00"


class TestResource:
    def test_store_is_per_satellite(self, resource):
        assert resource.store_uri("mtg-i1").endswith("/amv_mtg_i1.icechunk")
        assert resource.store_uri("goes18").endswith("/amv_goes18.icechunk")

    def test_constructs_without_credentials(self, monkeypatch):
        for v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
            monkeypatch.delenv(v, raising=False)
        # Definition time must not raise, or every asset in the location
        # goes down with it -- including the ones needing no credentials.
        AmvContainerResource()

    def test_missing_credentials_name_the_store(self, monkeypatch):
        for v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
            monkeypatch.delenv(v, raising=False)
        with pytest.raises(RuntimeError, match="AWS_ACCESS_KEY_ID"):
            AmvContainerResource().credential_env()

    def test_environment_credentials_are_used(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "from-env")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
        assert AmvContainerResource().credential_env()["AWS_ACCESS_KEY_ID"] == "from-env"


class TestAssets:
    def test_one_per_satellite_and_independent(self):
        assets = build_amv_container_assets(["goes18", "goes19", "mtg-i1"], PARTITIONS)
        keys = [k for a in assets.values() for k in a.keys]
        assert len(keys) == len(set(keys)) == 3

    def test_name_mangling(self):
        assert amv_container_asset_name("mtg-i1") == "amv_container_mtg_i1"

    def test_grouped_apart_from_the_in_process_assets(self):
        a = build_amv_container_asset("goes18", PARTITIONS)
        assert set(a.group_names_by_key.values()) == {AMV_CONTAINER_GROUP}

    def test_declares_its_resources(self):
        a = build_amv_container_asset("goes18", PARTITIONS)
        assert {"amv_container", "pipes_docker_client", "run_settings"} <= set(
            a.required_resource_keys
        )

    @staticmethod
    def _run(sat_id, resource, settings, partition):
        a = build_amv_container_asset(sat_id, PARTITIONS)
        client = MagicMock()
        client.run.return_value.get_materialize_results.return_value = []
        ctx = build_asset_context(partition_key=partition)
        fn = a.op.compute_fn.decorated_fn
        return fn(ctx, resource, settings, client), client

    def test_runs_the_image_with_the_satellite_env(self, resource, settings):
        result, client = self._run("goes18", resource, settings, "2026-09-20-12:00")
        env = client.run.call_args.kwargs["env"]
        assert env["AMV_SATELLITES"] == "goes18"
        assert env["AMV_RESOLUTION_M"] == "5000"
        # Credentials join only at the container boundary.
        assert env["AWS_SECRET_ACCESS_KEY"] == "aws-secret"
        assert client.run.call_args.kwargs["image"] == resource.image
        assert result.metadata["satellite"].text == "goes18"

    def test_metadata_names_the_store_it_published_to(self, resource, settings):
        result, _ = self._run("mtg-i1", resource, settings, "2026-09-20-12:00")
        assert result.metadata["store"].text.endswith("/amv_mtg_i1.icechunk")

    def test_bands_come_from_run_settings(self, resource, settings):
        _, client = self._run("goes18", resource, settings, "2026-09-20-12:00")
        env = client.run.call_args.kwargs["env"]
        assert env["AMV_FLOW_BANDS"] == "C08,C14"
        assert env["AMV_RAD_BANDS"] == "C07,C13"
