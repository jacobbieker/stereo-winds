"""Tests for the EUMETSAT ingest assets."""

import datetime as dt
from unittest.mock import MagicMock

import pytest
from dagster import build_asset_context

from operational.assets.consumer_assets import (
    CONSUMER_GROUP,
    build_consumer_asset,
    build_consumer_assets,
    consumer_asset_name,
)
from operational.core.partitions import DEFAULT_START, build_partitions_def
from operational.satellite_consumer import SatelliteConsumerResource

PARTITIONS = build_partitions_def(DEFAULT_START, 15)


@pytest.fixture
def resource():
    return SatelliteConsumerResource(
        eumetsat_key="eum-key",
        eumetsat_secret="eum-secret",
        aws_access_key_id="aws-key",
        aws_secret_access_key="aws-secret",
    )


class TestAssetNames:
    def test_dashes_become_underscores(self):
        """Dagster asset names cannot contain the consumer's dashes."""
        assert consumer_asset_name("odegree-12") == "consume_odegree_12"

    def test_one_asset_per_satellite(self):
        assets = build_consumer_assets(PARTITIONS)
        assert set(assets) == {"odegree-12", "odegree", "iodc"}

    def test_assets_are_independent(self):
        """A satellite failing must not hold up the others."""
        assets = build_consumer_assets(PARTITIONS)
        keys = [k for a in assets.values() for k in a.keys]
        assert len(keys) == len(set(keys)) == 3

    def test_subset_can_be_selected(self):
        assets = build_consumer_assets(PARTITIONS, keys=["iodc"])
        assert set(assets) == {"iodc"}


class TestAssetDefinition:
    def test_declares_the_resources_it_uses(self):
        asset_def = build_consumer_asset("iodc", PARTITIONS)
        assert {"satellite_consumer", "pipes_docker_client"} <= set(
            asset_def.required_resource_keys
        )

    def test_grouped_apart_from_the_retrieval(self):
        asset_def = build_consumer_asset("iodc", PARTITIONS)
        assert set(asset_def.group_names_by_key.values()) == {CONSUMER_GROUP}

    def test_metadata_records_the_destination(self):
        asset_def = build_consumer_asset("odegree-12", PARTITIONS)
        meta = next(iter(asset_def.metadata_by_key.values()))
        assert meta["store"] == "geo/mtg_2000m.icechunk"
        assert meta["resolution_m"] == 2000


class TestExecution:
    """The asset runs the container; Pipes is stubbed."""

    @staticmethod
    def _run(key, resource, partition):
        asset_def = build_consumer_asset(key, PARTITIONS)
        client = MagicMock()
        client.run.return_value.get_materialize_results.return_value = []
        context = build_asset_context(partition_key=partition)
        fn = asset_def.op.compute_fn.decorated_fn
        result = fn(context, resource, client)
        return result, client

    def test_passes_the_window_and_credentials_to_the_container(self, resource):
        result, client = self._run("odegree-12", resource, "2026-09-20-00:00")
        env = client.run.call_args.kwargs["env"]
        assert env["SATCONS_SATELLITE"] == "odegree-12"
        assert env["SATCONS_ZARR_PATH"].endswith("geo/mtg_2000m.icechunk")
        # Credentials are merged in only at the container boundary.
        assert env["EUMETSAT_CONSUMER_KEY"] == "eum-key"
        assert env["AWS_SECRET_ACCESS_KEY"] == "aws-secret"
        assert result.metadata["satellite"].text == "odegree-12"

    def test_window_is_the_satellites_own_cycle(self, resource):
        """A 15 minute partition must not ask a 10 minute instrument for
        a cycle that does not exist, or the reverse."""
        result, _ = self._run("odegree-12", resource, "2026-09-20-00:00")
        start = dt.datetime.fromisoformat(result.metadata["window_start"].text)
        end = dt.datetime.fromisoformat(result.metadata["window_end"].text)
        assert end - start == dt.timedelta(minutes=10)  # FCI

        result, _ = self._run("iodc", resource, "2026-09-20-00:00")
        start = dt.datetime.fromisoformat(result.metadata["window_start"].text)
        end = dt.datetime.fromisoformat(result.metadata["window_end"].text)
        assert end - start == dt.timedelta(minutes=15)  # SEVIRI

    def test_runs_the_configured_image(self, resource):
        _, client = self._run("iodc", resource, "2026-09-20-00:00")
        assert client.run.call_args.kwargs["image"] == resource.image

    def test_window_start_follows_the_partition(self, resource):
        result, _ = self._run("iodc", resource, "2026-09-20-06:15")
        assert result.metadata["window_start"].text.startswith("2026-09-20T06:15")
