"""One Dagster asset per satellite, partitioned by scan time.

Why one asset each rather than a single asset that loops: the ring is a
best-effort constellation.  GK-2A's archive stalls, a GOES scan is
republished late, a node runs out of memory partway through Himawari.
With an asset per satellite, such a failure is contained -- the other
satellites still materialize for that partition, and the repair is
re-materializing exactly one asset for exactly one partition key.  The
mosaic downstream is built from whatever files exist.

Each asset is also idempotent: when ``skip_existing`` is on and the
per-satellite NetCDF for the partition is already on disk, the retrieval
is reused rather than recomputed -- and the checkpoints are not even
loaded -- so a resumed backfill costs a directory listing instead of a
forward pass.

The asset set itself is fixed when the code location is imported, from
:class:`~operational.config.OperationalConfig`; run config scopes what a
run does, not which satellites exist.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from dagster import (
    AssetExecutionContext,
    AssetsDefinition,
    Backoff,
    Jitter,
    MaterializeResult,
    MetadataValue,
    PartitionsDefinition,
    RetryPolicy,
    asset,
)

from operational.config import OperationalConfig
from operational.core.amv import AmvResult, run_satellite_amv
from operational.core.partitions import DEFAULT_START, build_partitions_def, time_for
from operational.resources import ModelResource, PathsResource, RunSettingsResource

logger = logging.getLogger(__name__)

#: Dagster group every per-satellite AMV asset belongs to.
AMV_GROUP = "amv"

#: Retry policy attached to every per-satellite asset.  Most operational
#: failures are transient -- a partly written object in a bucket, a
#: throttled listing -- so two spaced retries recover the partition
#: without an operator noticing, while a genuine outage still surfaces.
DEFAULT_RETRY_POLICY = RetryPolicy(
    max_retries=2, delay=30, backoff=Backoff.EXPONENTIAL, jitter=Jitter.PLUS_MINUS,
)


def amv_asset_name(sat_id: str) -> str:
    """Dagster asset name for a satellite.

    Parameters
    ----------
    sat_id : str
        Satellite id, e.g. ``"goes19"`` or ``"mtg-i1"``.

    Returns
    -------
    str
        A valid Dagster asset name, e.g. ``"amv_goes19"``, ``"amv_mtg_i1"``.
    """
    slug = "".join(c if c.isalnum() else "_" for c in sat_id.lower())
    return f"amv_{slug}"


def _existing_output(
    output_dir: Path | str, sat_id: str, t0: datetime,
) -> Path:
    """Path the retrieval for ``(sat_id, t0)`` writes to.

    Parameters
    ----------
    output_dir : Path or str
        Root of the per-day NetCDF layout.
    sat_id : str
        Satellite id.
    t0 : datetime
        Nominal scan time.

    Returns
    -------
    Path
        The deterministic output path, which may or may not exist.
    """
    # Imported here, not at module scope: the ring script pulls in torch and
    # the whole stereo_winds stack, and a Dagster code location is imported
    # by the webserver and daemon far more often than it is run.
    from operational.adapters.ring import sat_nc_path

    return sat_nc_path(Path(output_dir), sat_id, t0)


def _materialization_metadata(
    sat_id: str, t0: datetime, partition_key: str, result: AmvResult,
) -> dict[str, Any]:
    """Metadata describing how thin (or complete) a retrieval turned out.

    Parameters
    ----------
    sat_id : str
        Satellite the retrieval ran for.
    t0 : datetime
        Timestamp decoded from the partition key.
    partition_key : str
        The partition that was materialized.
    result : AmvResult
        Outcome returned by :func:`~operational.core.amv.run_satellite_amv`.

    Returns
    -------
    dict
        Dagster metadata entries, keyed for the asset catalog.
    """
    bands_missing = list(result.bands_missing or ())
    return {
        "satellite": MetadataValue.text(sat_id),
        "partition": MetadataValue.text(partition_key),
        "timestamp": MetadataValue.text(t0.isoformat()),
        # Not "path": the IO manager writes its own "path" entry onto the
        # materialization, which would shadow the retrieval output.
        "output_path": MetadataValue.path(str(result.path)),
        "reused": MetadataValue.bool(bool(result.reused)),
        "n_bands_missing": MetadataValue.int(int(result.n_bands_missing)),
        "bands_missing": MetadataValue.text(", ".join(bands_missing) or "none"),
        "quality_degraded": MetadataValue.bool(bool(result.quality_degraded)),
        "status": MetadataValue.text(
            "reused existing output" if result.reused else "computed"
        ),
    }


def build_amv_asset(
    sat_id: str,
    partitions_def: PartitionsDefinition,
    *,
    group_name: str = AMV_GROUP,
    retry_policy: RetryPolicy | None = None,
    name: str | None = None,
    key_prefix: str | list[str] | None = None,
) -> AssetsDefinition:
    """Build the partitioned AMV asset for one satellite.

    Parameters
    ----------
    sat_id : str
        Satellite id passed straight through to the retrieval.
    partitions_def : dagster.PartitionsDefinition
        Time partitions whose keys decode via
        :func:`operational.core.partitions.time_for`.
    group_name : str, optional
        Dagster group for the asset.  Defaults to ``"amv"``.
    retry_policy : dagster.RetryPolicy, optional
        Overrides :data:`DEFAULT_RETRY_POLICY`.
    name : str, optional
        Overrides the derived :func:`amv_asset_name`.
    key_prefix : str or list of str, optional
        Optional asset key prefix.

    Returns
    -------
    dagster.AssetsDefinition
        An asset that materializes ``sat_id`` for a single partition and
        reports what the retrieval had to work with.
    """

    @asset(
        name=name or amv_asset_name(sat_id),
        key_prefix=key_prefix,
        partitions_def=partitions_def,
        group_name=group_name,
        retry_policy=retry_policy or DEFAULT_RETRY_POLICY,
        description=(
            f"Full-disk student AMV retrieval for {sat_id}, one partition per "
            "scan time.  Fails and resumes independently of the rest of the "
            "geostationary ring."
        ),
        metadata={"satellite": sat_id},
        kinds={"pytorch"},
        op_tags={"satellite": sat_id},
    )
    def _amv_asset(
        context: AssetExecutionContext,
        paths: PathsResource,
        model: ModelResource,
        run_settings: RunSettingsResource,
    ) -> MaterializeResult:
        partition_key = context.partition_key
        t0 = time_for(partition_key)
        context.log.info("%s: retrieving AMVs for %s", sat_id, t0)

        # A partition already on disk needs no checkpoints, and loading them
        # anyway would cost minutes of GPU time per already-complete
        # partition of a resumed backfill.  The retrieval re-checks the file
        # itself; this only decides whether the model is worth resolving.
        reuse_expected = (
            run_settings.skip_existing
            and _existing_output(paths.output_dir, sat_id, t0).exists()
        )
        if reuse_expected:
            context.log.info("%s %s: output already present", sat_id, t0)
        loaded_model = None if reuse_expected else model.model()
        loaded_disp = None if reuse_expected else model.disparity()

        result = run_satellite_amv(
            sat_id,
            t0,
            loaded_model,
            loaded_disp,
            list(run_settings.flow_bands),
            list(run_settings.rad_bands),
            paths.output_dir,
            device=model.device,
            row_strip=model.row_strip,
            skip_existing=run_settings.skip_existing,
        )

        if result.quality_degraded:
            context.log.warning(
                "%s %s: degraded retrieval -- %d band(s) missing (%s)",
                sat_id, t0, result.n_bands_missing,
                ", ".join(result.bands_missing) or "unnamed",
            )

        # The value is the path, not the dataset: a full disk is gigabytes
        # and downstream mosaicing reads it back from the file anyway.
        return MaterializeResult(
            value=str(result.path),
            metadata=_materialization_metadata(sat_id, t0, partition_key, result),
        )

    return _amv_asset


def build_amv_assets(
    satellites: tuple[str, ...] | list[str],
    partitions_def: PartitionsDefinition,
    **kwargs: Any,
) -> dict[str, AssetsDefinition]:
    """Build one AMV asset per satellite.

    Parameters
    ----------
    satellites : sequence of str
        Satellite ids to build assets for.
    partitions_def : dagster.PartitionsDefinition
        Shared time partitions.
    **kwargs
        Forwarded to :func:`build_amv_asset`.

    Returns
    -------
    dict of str to dagster.AssetsDefinition
        Assets keyed by satellite id, in the order given.
    """
    return {
        sat_id: build_amv_asset(sat_id, partitions_def, **kwargs)
        for sat_id in satellites
    }


# Read from the environment, not bare defaults: definitions.py shapes
# the code location with OperationalConfig.from_env(), and the asset
# set has to agree with it.  Built from bare defaults, the assets
# ignored STEREO_WINDS_OP_SATELLITES entirely -- the satellite list
# reached the resources but never the graph, so a deployment
# configured for two satellites still defined assets for four.
DEFAULT_CONFIG = OperationalConfig.from_env()

#: Partitions the default asset set is built on.
AMV_PARTITIONS_DEF = build_partitions_def(
    DEFAULT_START, DEFAULT_CONFIG.cadence_minutes,
)

#: Default per-satellite assets, keyed by satellite id.
AMV_ASSETS_BY_SAT: dict[str, AssetsDefinition] = build_amv_assets(
    DEFAULT_CONFIG.satellites, AMV_PARTITIONS_DEF,
)

#: Default per-satellite assets, for inclusion in ``Definitions``.
AMV_ASSETS: list[AssetsDefinition] = list(AMV_ASSETS_BY_SAT.values())

__all__ = [
    "AMV_ASSETS",
    "AMV_ASSETS_BY_SAT",
    "AMV_GROUP",
    "AMV_PARTITIONS_DEF",
    "DEFAULT_RETRY_POLICY",
    "amv_asset_name",
    "build_amv_asset",
    "build_amv_assets",
]
