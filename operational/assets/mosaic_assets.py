"""Mosaic and publish assets for the operational pipeline.

Two partitioned assets close out each retrieval cycle:

``global_mosaic``
    Merges whichever per-satellite AMV retrievals exist for the partition
    onto one global lat/lon grid and writes the mosaic NetCDF.

``published_mosaic``
    Appends that mosaic to the icechunk store as its own commit.

Both are built by :func:`build_mosaic_assets`, which takes the satellite
list once and uses it for *both* the declared dependencies and the set the
mosaic looks for at run time — so the two cannot drift apart.  The
module-level ``global_mosaic`` / ``published_mosaic`` are that factory
applied to :class:`~operational.config.OperationalConfig` defaults.

Why the mosaic reads NetCDF instead of taking asset inputs
----------------------------------------------------------
The ring is deliberately fault-tolerant: a satellite can be down, late, or
short a few bands, and the cycle still has to produce a mosaic from the
rest.  Declaring the per-satellite retrievals as *loaded* inputs would put
that guarantee in the hands of the I/O manager — the mosaic step would
fail while loading the output of a satellite that produced nothing, before
any of this module's code ran, and the whole cycle would be lost to one
absent satellite.

So the per-satellite retrievals are declared with ``deps`` (a dependency
for scheduling and lineage, with no value loaded) and their products are
read straight off disk from :class:`~operational.resources.PathsResource`
at the partition's canonical path.  Absence is then just a file that is not
there — an ordinary, expected branch — and the mosaic proceeds with what it
has, recording the shortfall in its metadata and in the mosaic's own
attributes.  Only a cycle that yields *no* gridded cells at all fails, and
it fails loudly, because an empty mosaic would otherwise publish as a
perfectly ordinary timestep and hide the outage behind it.
"""

import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from dagster import (
    AssetExecutionContext,
    AssetKey,
    AssetsDefinition,
    Backoff,
    Failure,
    Jitter,
    MetadataValue,
    PartitionsDefinition,
    RetryPolicy,
    asset,
)

from operational.config import OperationalConfig
from operational.core.mosaic import (
    build_mosaic,
    missing_satellites,
    write_mosaic_netcdf,
)
from operational.core.partitions import OPERATIONAL_PARTITIONS, time_for
from operational.core.publish import existing_timestamps, publish_mosaic
from operational.resources import (
    IcechunkStoreResource,
    PathsResource,
    RunSettingsResource,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MOSAIC_GROUP",
    "PUBLISH_RETRY_POLICY",
    "per_satellite_path",
    "load_available_retrievals",
    "amv_asset_name",
    "build_mosaic_assets",
    "global_mosaic",
    "published_mosaic",
]

MOSAIC_GROUP = "mosaic"

# From the environment, matching amv_assets and definitions.py: the
# deps global_mosaic declares must name the same satellites the AMV
# layer actually defines assets for, or dagster treats the difference
# as external assets nothing materialises.
_CONFIG = OperationalConfig.from_env()

#: Publishing is a write to an object store, so it fails for reasons that
#: go away on their own.  Retrying is safe in the deployment's default
#: shape: icechunk commits atomically, so a failed attempt leaves nothing
#: half-written, and with ``skip_existing`` a retry after a commit that
#: landed on the far side of a dropped connection is a no-op.  With
#: ``skip_existing`` turned off a retry would append the timestamp twice,
#: which is one more reason to leave it on outside a controlled backfill.
PUBLISH_RETRY_POLICY = RetryPolicy(
    max_retries=4, delay=10, backoff=Backoff.EXPONENTIAL, jitter=Jitter.PLUS_MINUS,
)

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_]")


# ---------------------------------------------------------------------------
# Locating what the per-satellite assets wrote
# ---------------------------------------------------------------------------

def per_satellite_path(output_dir: Path | str, sat_id: str, t0: datetime) -> Path:
    """Canonical path of one satellite's AMV file for a cycle.

    Parameters
    ----------
    output_dir : Root output directory.
    sat_id : Satellite id, e.g. ``"goes18"``.
    t0 : Cycle timestamp.

    Returns
    -------
    pathlib.Path
        ``<output_dir>/<YYYYMMDD>/student_amv_<sat>_<YYYYMMDDTHHMM>.nc``.
    """
    # Deferred: loading the ring module pulls in torch, and nothing in this
    # module needs it until an asset actually runs.
    from operational.adapters.ring import sat_nc_path

    return Path(sat_nc_path(Path(output_dir), sat_id, t0))


def load_available_retrievals(
    output_dir: Path | str, satellites: list[str], t0: datetime,
) -> dict[str, xr.Dataset]:
    """Load every per-satellite retrieval that made it to disk for ``t0``.

    A satellite whose file is absent — or present but unreadable, which a
    crash mid-write can leave behind — is skipped rather than raised on.

    Parameters
    ----------
    output_dir : Root output directory.
    satellites : Satellites the cycle asked for.
    t0 : Cycle timestamp.

    Returns
    -------
    dict
        Satellite id -> its dataset, loaded into memory (the file handle is
        closed), for the satellites that produced one.
    """
    per_sat: dict[str, xr.Dataset] = {}
    for sat_id in satellites:
        path = per_satellite_path(output_dir, sat_id, t0)
        if not path.exists():
            logger.info("%s: no retrieval at %s", sat_id, path)
            continue
        try:
            with xr.open_dataset(path) as handle:
                per_sat[sat_id] = handle.load()
        except Exception:
            logger.exception("%s: could not read %s — treating as missing",
                             sat_id, path)
    return per_sat


def amv_asset_name(sat_id: str) -> str:
    """Dagster-safe op name for a satellite's AMV asset.

    Dagster names must match ``[A-Za-z0-9_]+``, and two ring satellites
    (``mtg-i1``, ``msg-iodc``) carry hyphens.
    """
    return f"amv_{_UNSAFE_NAME_CHARS.sub('_', sat_id)}"


def _amv_asset_keys(satellites: tuple[str, ...]) -> list[AssetKey]:
    """Asset keys of the per-satellite retrievals, for lineage only.

    Taken from the AMV module when it can be asked, so the keys stay right
    if that module changes how it names them; otherwise from the shared
    ``amv_<slug>`` convention :func:`amv_asset_name` implements.
    """
    by_sat: dict = {}
    try:
        from operational.assets.amv_assets import AMV_ASSETS_BY_SAT

        by_sat = dict(AMV_ASSETS_BY_SAT)
    except Exception:  # pragma: no cover - only before that unit lands
        logger.warning(
            "amv_assets not importable; falling back to the amv_<sat> naming "
            "convention for dependency keys", exc_info=True,
        )

    keys = []
    for sat in satellites:
        definition = by_sat.get(sat)
        keys.append(definition.key if definition is not None
                    else AssetKey(amv_asset_name(sat)))
    return keys


def _note_missing(ds: xr.Dataset, absent: list[str]) -> None:
    """Record absent satellites in the mosaic NetCDF's quality attributes.

    Scoped to the file on disk: these are dataset-level attributes, and the
    icechunk store keeps one attribute set for the whole time series, so a
    later complete cycle overwrites them there.  Per-cycle degradation is
    authoritative in this NetCDF and in the asset's Dagster metadata.
    """
    if not absent:
        return
    ds.attrs["missing_satellites"] = ",".join(absent)
    ds.attrs["quality_degraded"] = 1
    note = str(ds.attrs.get("quality_note", "")).strip()
    gap = (
        f"COVERAGE GAP — no usable retrieval from {', '.join(absent)} for "
        f"this cycle, so the regions those satellites view are unfilled."
    )
    ds.attrs["quality_note"] = f"{note} {gap}".strip() if note else gap


# ---------------------------------------------------------------------------
# Asset factory
# ---------------------------------------------------------------------------

def build_mosaic_assets(
    satellites: tuple[str, ...] | list[str] | None = None,
    partitions_def: PartitionsDefinition | None = None,
) -> list[AssetsDefinition]:
    """Build the mosaic and publish assets for one deployment.

    Parameters
    ----------
    satellites : Satellites the cycle depends on and looks for. Used both
        for the declared ``deps`` and for the run-time search, so the
        dependency graph and the mosaic can never disagree. Defaults to
        :class:`~operational.config.OperationalConfig`.
    partitions_def : Partitions both assets are keyed by. Defaults to
        :data:`~operational.core.partitions.OPERATIONAL_PARTITIONS`.

    Returns
    -------
    list of AssetsDefinition
        ``[global_mosaic, published_mosaic]``.
    """
    declared = tuple(satellites if satellites is not None else _CONFIG.satellites)
    partitions = partitions_def or OPERATIONAL_PARTITIONS

    @asset(
        name="global_mosaic",
        partitions_def=partitions,
        deps=_amv_asset_keys(declared),
        group_name=MOSAIC_GROUP,
        compute_kind="xarray",
        description=(
            "Global min-zenith AMV mosaic for one cycle, merged from "
            "whichever per-satellite retrievals reached disk. Missing "
            "satellites leave holes and are reported in the metadata; they "
            "do not fail the step."
        ),
    )
    def _global_mosaic(
        context: AssetExecutionContext,
        paths: PathsResource,
        run_settings: RunSettingsResource,
    ) -> str:
        """Merge the cycle's per-satellite retrievals into one global mosaic.

        Returns
        -------
        str
            Path of the mosaic NetCDF. A path rather than the dataset
            itself: a global mosaic is far too large to hand through an I/O
            manager, and the file is the product consumers already expect.
        """
        t0 = time_for(context.partition_key)
        output_dir = Path(paths.output_dir)
        expected = list(run_settings.satellites)

        undeclared = sorted(set(expected) - set(declared))
        if undeclared:
            # Not a dependency, so nothing orders the mosaic after them:
            # they would look like an outage whenever they run late.
            logger.warning(
                "run_settings.satellites includes %s, which this asset does not "
                "declare as a dependency — rebuild the assets with "
                "build_mosaic_assets(satellites=...) so scheduling matches",
                ", ".join(undeclared),
            )

        per_sat = load_available_retrievals(output_dir, expected, t0)
        absent = missing_satellites(per_sat, expected)

        if not per_sat:
            raise Failure(
                description=(
                    f"No satellite produced a retrieval for {t0.isoformat()} "
                    f"— nothing to mosaic. Expected: {', '.join(expected)}."
                ),
                metadata={
                    "partition": t0.isoformat(),
                    "missing_satellites": ",".join(absent),
                    "searched_under": MetadataValue.path(str(output_dir)),
                },
            )

        ds_global = build_mosaic(per_sat, t0, resolution_m=run_settings.resolution_m)
        loaded = sorted(per_sat)
        # The per-satellite datasets are full disks; let them go before the
        # mosaic is serialised.
        per_sat.clear()

        # A satellite only enters ``satellites`` once it wins a cell, so a
        # file full of unusable pixels contributes nothing while looking
        # present.  Count it with the outright absent ones, or the mosaic
        # would report a hole in its coverage as healthy.
        contributing = [str(s) for s in (ds_global.attrs.get("satellites") or [])]
        empty = sorted(set(loaded) - set(contributing))
        absent = sorted(set(absent) | set(empty))

        source_index = ds_global["source_satellite_index"].values
        valid_cells = int(np.count_nonzero(source_index >= 0))
        n_cells = int(source_index.size)

        if not valid_cells:
            raise Failure(
                description=(
                    f"Every retrieval loaded for {t0.isoformat()} gridded to "
                    f"zero cells ({', '.join(loaded)}); the mosaic would be "
                    f"empty. Publishing it would hide the outage behind an "
                    f"ordinary-looking timestep."
                ),
                metadata={
                    "partition": t0.isoformat(),
                    "loaded_satellites": ",".join(loaded),
                    "missing_satellites": ",".join(absent),
                },
            )

        if absent:
            logger.warning("Mosaicking %s without %s", t0.isoformat(),
                           ", ".join(absent))
        _note_missing(ds_global, absent)

        out_path = write_mosaic_netcdf(ds_global, output_dir, t0)

        context.add_output_metadata({
            "partition": t0.isoformat(),
            "contributing_satellites": ",".join(contributing) or "(none)",
            "n_contributing": len(contributing),
            "missing_satellites": ",".join(absent) or "(none)",
            "n_missing": len(absent),
            "empty_satellites": ",".join(empty) or "(none)",
            "quality_degraded": MetadataValue.bool(
                bool(int(ds_global.attrs.get("quality_degraded", 0)))
            ),
            "quality_note": MetadataValue.text(
                str(ds_global.attrs.get("quality_note", ""))
            ),
            "output_path": MetadataValue.path(str(out_path)),
            "valid_cells": valid_cells,
            "valid_cell_fraction": round(valid_cells / n_cells, 6) if n_cells else 0.0,
            "grid_shape": f"{source_index.shape[0]} x {source_index.shape[1]}",
            "resolution_m": float(run_settings.resolution_m),
        })
        return str(out_path)

    @asset(
        name="published_mosaic",
        partitions_def=partitions,
        group_name=MOSAIC_GROUP,
        compute_kind="icechunk",
        retry_policy=PUBLISH_RETRY_POLICY,
        description=(
            "Appends the cycle's mosaic to the icechunk store as one commit "
            "per timestep, skipping a timestamp the store already holds."
        ),
    )
    def _published_mosaic(
        context: AssetExecutionContext,
        global_mosaic: str,
        store: IcechunkStoreResource,
        run_settings: RunSettingsResource,
    ) -> str:
        """Append one mosaic to the icechunk store.

        Parameters
        ----------
        global_mosaic : Path of the mosaic NetCDF, from the upstream asset.

        Returns
        -------
        str
            The store URI, so downstream assets can depend on the store
            rather than on a file.
        """
        t0 = time_for(context.partition_key)
        mosaic_path = Path(global_mosaic)
        with xr.open_dataset(mosaic_path) as handle:
            ds = handle.load()

        repo = store.repo()
        result = publish_mosaic(
            repo,
            ds,
            t0,
            branch=store.branch,
            chunk=store.chunk,
            skip_existing=run_settings.skip_existing,
        )

        times = sorted(existing_timestamps(repo, store.branch))
        context.add_output_metadata({
            "partition": t0.isoformat(),
            "store_uri": MetadataValue.text(store.store_uri),
            "branch": store.branch,
            "written": MetadataValue.bool(bool(result.written)),
            "skipped_reason": result.skipped_reason or "(not skipped)",
            "time_size": len(times),
            "time_range": (
                f"{times[0].isoformat()} .. {times[-1].isoformat()}"
                if times else "(empty)"
            ),
            "satellite_vocabulary": ",".join(result.vocabulary) or "(none)",
            "source_mosaic": MetadataValue.path(str(mosaic_path)),
        })
        return store.store_uri

    return [_global_mosaic, _published_mosaic]


global_mosaic, published_mosaic = build_mosaic_assets()
