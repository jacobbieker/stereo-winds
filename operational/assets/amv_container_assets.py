"""Per-satellite AMV assets that run as containers.

One asset per satellite, each materialisation running the retrieval
image for its own satellite and partition and publishing the full disk
to that satellite's icechunk store.

Why containers rather than the in-process assets in ``amv_assets``: the
in-process ones write a NetCDF to local disk that the mosaic reads back
off the same filesystem, so all six satellites have to be retrieved on
one host, one after another.  A container carries its own environment
and publishes to S3, so the satellites can be retrieved wherever there
is a GPU free and the mosaic reads the stores instead of the disk.
"""

import logging

from dagster import (
    AssetExecutionContext,
    AssetsDefinition,
    MaterializeResult,
    MetadataValue,
    PartitionsDefinition,
    RetryPolicy,
    asset,
)
from dagster_docker import PipesDockerClient

from operational.amv_container import AmvContainerResource, satellite_env
from operational.core.partitions import time_for
from operational.resources import RunSettingsResource

logger = logging.getLogger(__name__)

__all__ = [
    "AMV_CONTAINER_GROUP",
    "amv_container_asset_name",
    "build_amv_container_asset",
    "build_amv_container_assets",
]

#: Group for the containerised retrieval, kept apart from the in-process
#: assets so a deployment can run one or the other without the two
#: writing the same partition from different directions.
AMV_CONTAINER_GROUP = "amv_containers"

#: A retrieval reads from object stores and runs a GPU for minutes; its
#: failures are mostly transient (a scene not yet published, a throttled
#: read).  One retry costs less than a hole in the mosaic.
DEFAULT_RETRY_POLICY = RetryPolicy(max_retries=2, delay=180)


def amv_container_asset_name(sat_id: str) -> str:
    """Asset name for a satellite; dashes are not valid in one."""
    return f"amv_container_{sat_id.replace('-', '_')}"


def build_amv_container_asset(
    sat_id: str,
    partitions_def: PartitionsDefinition,
    *,
    group_name: str = AMV_CONTAINER_GROUP,
    retry_policy: RetryPolicy | None = None,
    name: str | None = None,
    key_prefix: str | list[str] | None = None,
) -> AssetsDefinition:
    """Build the containerised AMV asset for one satellite."""

    @asset(
        name=name or amv_container_asset_name(sat_id),
        key_prefix=key_prefix,
        partitions_def=partitions_def,
        group_name=group_name,
        retry_policy=retry_policy or DEFAULT_RETRY_POLICY,
        description=(
            f"Full-disk student AMV retrieval for {sat_id}, run as a "
            f"container and published to that satellite's icechunk store. "
            f"Independent of the other satellites: it shares no filesystem "
            f"with them and can run on any host with a GPU."
        ),
        metadata={"satellite": sat_id},
        kinds={"docker", "pytorch", "icechunk"},
        op_tags={"satellite": sat_id},
    )
    def _amv_container_asset(
        context: AssetExecutionContext,
        amv_container: AmvContainerResource,
        run_settings: RunSettingsResource,
        pipes_docker_client: PipesDockerClient,
    ) -> MaterializeResult:
        t0 = time_for(context.partition_key)
        env = satellite_env(
            sat_id,
            t0,
            icechunk_base=amv_container.icechunk_base,
            resolution_m=run_settings.resolution_m,
            flow_bands=list(run_settings.flow_bands),
            rad_bands=list(run_settings.rad_bands),
            device=amv_container.device,
            row_strip=amv_container.row_strip,
            workdir=amv_container.workdir,
        )
        store = amv_container.store_uri(sat_id)
        context.log.info("%s: retrieving %s -> %s", sat_id, t0.isoformat(), store)

        # The window is what gets logged; the credentials join it only on
        # the way into the container.
        result = pipes_docker_client.run(
            image=amv_container.image,
            env={**env, **amv_container.credential_env()},
            container_kwargs=amv_container.container_kwargs(),
            context=context,
        )

        return MaterializeResult(
            metadata={
                "satellite": MetadataValue.text(sat_id),
                "timestamp": MetadataValue.text(t0.isoformat()),
                "store": MetadataValue.text(store),
                "image": MetadataValue.text(amv_container.image),
                "resolution_m": MetadataValue.float(run_settings.resolution_m),
                **_pipes_metadata(result),
            },
        )

    return _amv_container_asset


def _pipes_metadata(result) -> dict:
    """Whatever the Pipes session reported, if anything.

    The retrieval is an ordinary program that knows nothing about Pipes,
    so this is normally empty; read defensively so a future image that
    does report is not ignored.
    """
    try:
        materialisations = result.get_materialize_results()
    except Exception:  # pragma: no cover - depends on dagster internals
        return {}
    for item in materialisations:
        if item.metadata:
            return dict(item.metadata)
    return {}


def build_amv_container_assets(
    satellites: tuple[str, ...] | list[str],
    partitions_def: PartitionsDefinition,
    **kwargs,
) -> dict[str, AssetsDefinition]:
    """One containerised AMV asset per satellite."""
    return {
        sat_id: build_amv_container_asset(sat_id, partitions_def, **kwargs) for sat_id in satellites
    }
