"""Dagster assets that pull the EUMETSAT satellites into icechunk.

MTG, MSG and IODC are not in the public S3 buckets the rest of the ring
reads, so their imagery has to be fetched from EUMETSAT before any
retrieval can use it.  One asset per satellite, partitioned on the same
time axis as the AMV assets, each materialisation running the
satellite-consumer container for its own partition's window.

The container is the unit of work rather than a library call: it pins
its own conda environment, for reasons ``docker/satellite-consumer.
Dockerfile`` sets out, and nothing in it is importable from here.
"""

# No `from __future__ import annotations` here, as in the sibling asset
# modules: Dagster resolves the context and resource parameters from
# their annotations when the asset is defined, and stringified
# annotations it cannot look up are a definition-time error.
import logging
from datetime import datetime, timedelta

from dagster import (
    AssetExecutionContext,
    AssetsDefinition,
    MaterializeResult,
    MetadataValue,
    PartitionsDefinition,
    RetryPolicy,
    asset,
)

# Imported for real, not under TYPE_CHECKING: Dagster resolves resource
# parameters from their annotations at definition time, and this module
# uses `from __future__ import annotations`, so a string it cannot look
# up is a definition-time error rather than a typing nicety.
from dagster_docker import PipesDockerClient

from operational.core.partitions import time_for
from operational.satellite_consumer import (
    CONSUMER_SATELLITES,
    ConsumerSatellite,
    SatelliteConsumerResource,
    consumer_satellite,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CONSUMER_GROUP",
    "build_consumer_asset",
    "build_consumer_assets",
    "consumer_asset_name",
]

#: Asset group for the ingest layer, kept apart from the retrieval assets
#: so the two can be materialised and scheduled independently.  The full
#: retrieval job excludes this group by name; ``jobs.INGEST_GROUP`` is
#: the same string and a test holds the two together.
CONSUMER_GROUP = "satellite_ingest"

#: A EUMETSAT fetch is network-bound and its failures are mostly
#: transient -- an expired token, a product not yet published.  Retrying
#: is cheap next to leaving a hole in the store.
DEFAULT_RETRY_POLICY = RetryPolicy(max_retries=2, delay=120)


def consumer_asset_name(key: str) -> str:
    """Asset name for a consumer satellite key.

    ``-`` is not valid in a Dagster asset name, and the consumer's keys
    contain it (``odegree-12``).
    """
    return f"consume_{key.replace('-', '_')}"


def _window(sat: ConsumerSatellite, t0: datetime) -> "tuple[datetime, datetime]":
    """The ``[t0, t0 + cadence)`` window a partition stands for.

    The partition is an instant; the consumer takes a range and treats
    the end as exclusive of nothing in particular, so the window is one
    of the satellite's own repeat cycles.  A 10 minute partition on a 15
    minute instrument would otherwise ask for a cycle that does not
    exist and record a materialisation that consumed nothing.
    """
    return t0, t0 + timedelta(minutes=sat.cadence_mins)


def build_consumer_asset(
    key: str,
    partitions_def: PartitionsDefinition,
    *,
    group_name: str = CONSUMER_GROUP,
    retry_policy: RetryPolicy | None = None,
    name: str | None = None,
    key_prefix: str | list[str] | None = None,
) -> AssetsDefinition:
    """Build the partitioned ingest asset for one EUMETSAT satellite."""
    sat = consumer_satellite(key)

    @asset(
        name=name or consumer_asset_name(key),
        key_prefix=key_prefix,
        partitions_def=partitions_def,
        group_name=group_name,
        retry_policy=retry_policy or DEFAULT_RETRY_POLICY,
        description=(
            f"Fetch {sat.description} from EUMETSAT into {sat.store}, one "
            f"partition per {sat.cadence_mins} minute repeat cycle.  Runs "
            f"the satellite-consumer container; fails and resumes "
            f"independently of the other satellites."
        ),
        metadata={
            "satellite": sat.key,
            "store": sat.store,
            "resolution_m": sat.resolution_m,
        },
        kinds={"docker", "icechunk"},
        op_tags={"satellite": sat.key},
    )
    def _consumer_asset(
        context: AssetExecutionContext,
        satellite_consumer: SatelliteConsumerResource,
        pipes_docker_client: PipesDockerClient,
    ) -> MaterializeResult:
        t0 = time_for(context.partition_key)
        start, end = _window(sat, t0)
        env = satellite_consumer.window_env(sat, start, end)
        context.log.info(
            "%s: consuming %s to %s into %s",
            sat.key,
            start.isoformat(),
            end.isoformat(),
            satellite_consumer.store_url(sat),
        )

        # The window is safe to log; the credentials are not, so they are
        # merged in only here, on the way into the container.
        container_env = {**env, **satellite_consumer.credential_env()}

        result = pipes_docker_client.run(
            image=satellite_consumer.image,
            env=container_env,
            context=context,
        )

        # The consumer does not speak the Pipes protocol, so it reports
        # no materialisation of its own; the metadata is what we know
        # from the outside plus whatever Pipes captured.
        return MaterializeResult(
            metadata={
                "satellite": MetadataValue.text(sat.key),
                "window_start": MetadataValue.text(start.isoformat()),
                "window_end": MetadataValue.text(end.isoformat()),
                "store": MetadataValue.text(satellite_consumer.store_url(sat)),
                "resolution_m": MetadataValue.int(sat.resolution_m),
                "image": MetadataValue.text(satellite_consumer.image),
                **_pipes_metadata(result),
            },
        )

    return _consumer_asset


def _pipes_metadata(result) -> dict:
    """Whatever the Pipes session reported, if anything.

    The consumer is an ordinary program that knows nothing about Pipes,
    so this is normally empty; it is read defensively rather than
    assumed so that a future consumer that does report is not ignored.
    """
    try:
        materialisations = result.get_materialize_results()
    except Exception:  # pragma: no cover - depends on dagster internals
        return {}
    for item in materialisations:
        if item.metadata:
            return dict(item.metadata)
    return {}


def build_consumer_assets(
    partitions_def: PartitionsDefinition,
    keys: list[str] | None = None,
    **kwargs,
) -> dict[str, AssetsDefinition]:
    """One ingest asset per EUMETSAT satellite, keyed by consumer key."""
    chosen = list(CONSUMER_SATELLITES) if keys is None else list(keys)
    return {key: build_consumer_asset(key, partitions_def, **kwargs) for key in chosen}
