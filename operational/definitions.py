"""The Dagster code location for the operational AMV pipeline.

Load it with::

    dagster dev -m operational.definitions

Everything the deployment needs is assembled here and nowhere else: one
AMV asset per satellite, the mosaic and icechunk publish assets, the jobs
and backstop schedule from :mod:`operational.jobs`, the availability
sensor from :mod:`operational.sensors`, and the resources from
:mod:`operational.resources`.

Importing this module is side-effect free
-----------------------------------------
Loading a code location happens on every ``dagster dev``, every daemon
restart and every run launch, on machines that may have no GPU, no
credentials and no data.  So nothing here reads a checkpoint, opens an
icechunk store, creates a directory or makes a network call.  The
resources are lazy by contract — they hold *where* things are and open
them on first use inside a run — which is why every default below can
point at a path that does not exist and the module still imports.

The one thing import time does read is :data:`os.environ`, to pick the
deployment's defaults.

Environment variables
---------------------
Read through :meth:`operational.config.OperationalConfig.from_env`:

``STEREO_WINDS_OP_SATELLITES``
    Comma-separated satellite ids.  Default
    ``goes18,goes19,himawari9,gk2a``.
``STEREO_WINDS_OP_FLOW_BANDS``, ``STEREO_WINDS_OP_RAD_BANDS``
    Comma-separated band ids fed to the student model.
``STEREO_WINDS_OP_CADENCE_MINUTES``
    Spacing of the timestamp partitions.  Default 60.
``STEREO_WINDS_OP_AVAILABILITY_TOLERANCE_MINUTES``
    Slack between a scene's own time and its nominal slot.  Default 5.
``STEREO_WINDS_OP_OUTPUT_DIR``
    Root for per-satellite NetCDFs and mosaics.
``STEREO_WINDS_OP_STORE_URI``
    Local path or ``s3://bucket/prefix`` of the icechunk store.
``STEREO_WINDS_OP_RESOLUTION_M``
    Mosaic grid spacing in meters.  Default 10000.
``STEREO_WINDS_OP_DEVICE``
    Torch device for inference.  Default ``cpu``.
``STEREO_WINDS_OP_ROW_STRIP``
    Rows per forward-pass strip.  Default 1024.

Read elsewhere, and listed here because they shape this code location:

``STEREO_WINDS_OP_CHECKPOINT``
    Student model checkpoint path, wired into
    :class:`~operational.resources.ModelResource`.  Never read at import.
``STEREO_WINDS_OP_PARTITION_START``
    ISO date/time of the first partition.  Owned by the AMV asset module,
    which builds the partitions definition the whole graph shares.
``STEREO_WINDS_OP_MAX_CONCURRENT``
    Steps in flight at once, read by :mod:`operational.jobs`.  Defaults
    to 1 because a single full-disk retrieval already peaks at several
    GB.

Which settings are late-bound
-----------------------------
The path-like and device settings are wired into the resources as
:class:`~dagster.EnvVar` when they are set in the process environment, so
their values are re-read when a run launches and an output root or a
checkpoint can be repointed without a redeploy.  Structural settings —
the satellite list, the cadence — are read eagerly, because they change
the asset graph and so need the code location reloaded anyway.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from types import ModuleType
from typing import Any

from dagster import (
    AssetKey,
    AssetsDefinition,
    Definitions,
    EnvVar,
    PartitionsDefinition,
    SensorDefinition,
)

from operational import sensors as sensors_module
from operational.assets import amv_assets as amv_assets_module
from operational.assets.mosaic_assets import global_mosaic, published_mosaic
from operational.config import OperationalConfig
from operational.core.partitions import cron_for_cadence
from operational.jobs import (
    build_backstop_schedule,
    build_full_job,
    build_satellite_jobs,
    max_concurrent_from_env,
)
from operational.resources import (
    IcechunkStoreResource,
    ModelResource,
    PathsResource,
    RunSettingsResource,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CHECKPOINT_ENV_VAR",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_RAFT_CHECKPOINT",
    "RAFT_CHECKPOINT_ENV_VAR",
    "build_definitions",
    "check_mosaic_inputs",
    "check_satellite_agreement",
    "default_resources",
    "defs",
    "discover_sensors",
    "resolve_amv_assets",
    "resolve_partitions_def",
]

#: Environment variable naming the student model checkpoint.
CHECKPOINT_ENV_VAR = "STEREO_WINDS_OP_CHECKPOINT"

#: Checkpoint path when nothing overrides it.  Not read at import time,
#: so it is allowed not to exist on the machine loading this module.
DEFAULT_CHECKPOINT = "checkpoints/student_zeus.ckpt"

#: RAFT optical-flow checkpoint.  ModelResource needs both this and the
#: student checkpoint: the student predicts winds, RAFT supplies the
#: displacement field it is conditioned on.
DEFAULT_RAFT_CHECKPOINT = "checkpoints/windflow.raft.sonde-tuned.ckpt"

#: Environment override for the RAFT checkpoint, matching the name the
#: rest of the repo already uses.
RAFT_CHECKPOINT_ENV_VAR = "STEREO_WINDS_RAFT_CKPT"


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------

def resolve_partitions_def(
    config: OperationalConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> PartitionsDefinition:
    """The one partitions definition the whole asset graph shares.

    The asset modules build their assets at import and carry the
    partitions definition with them, so it is taken from there rather
    than rebuilt: an upstream and a downstream asset on two grids that
    differ by a setting read at two different moments is a confusing
    failure to diagnose, and Dagster only reports it as an unresolvable
    job much later.

    The cadence the configuration *currently* asks for is still compared
    against it, because a difference means the asset modules were
    imported under a different environment from the one loading them now.

    Parameters
    ----------
    config : OperationalConfig, optional
        Supplies the cadence to cross-check.  Defaults to
        :meth:`OperationalConfig.from_env`.
    env : Mapping[str, str], optional
        Mapping to read the configuration from instead of
        :data:`os.environ`.

    Returns
    -------
    dagster.PartitionsDefinition
        The partitions the mosaic asset was built with.

    Raises
    ------
    ValueError
        If the mosaic asset is unpartitioned.  Every asset in this
        pipeline is per-timestamp, so an unpartitioned one means the
        asset module is broken, and failing here names the cause.
    """
    cfg = OperationalConfig.from_env(env) if config is None else config
    existing = global_mosaic.partitions_def
    if existing is None:
        raise ValueError(
            "global_mosaic has no partitions definition, but every asset in "
            "the operational pipeline is partitioned by timestamp. Check "
            "operational.assets.mosaic_assets."
        )
    try:
        expected_cron = cron_for_cadence(cfg.cadence_minutes)
    except ValueError:
        # A cadence with no cron expression is the configuration's own
        # problem to report, not something to cross-check against.
        return existing
    actual_cron = getattr(existing, "cron_schedule", None)
    if actual_cron != expected_cron:
        logger.warning(
            "the asset modules are partitioned as %r but the current "
            "configuration asks for %r; the assets win. Reload the code "
            "location if the configuration is the one you meant.",
            actual_cron,
            expected_cron,
        )
    return existing


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def _env_or(var: str, default: str, env: Mapping[str, str] | None = None) -> str:
    """Resolve a resource field from the environment, or fall back.

    Reading the *process* environment yields an :class:`~dagster.EnvVar`,
    a ``str`` subclass Dagster resolves when a run starts, so the value
    can be repointed without redeploying the code location.  An explicit
    ``env`` mapping yields the literal value instead: Dagster would
    resolve an ``EnvVar`` against :data:`os.environ` at run launch, not
    against that mapping, so a resource built from a mapping would fail
    at the start of the first run rather than here.

    A variable set to blank counts as unset, matching
    :meth:`OperationalConfig.from_env`, so an empty value in a unit file
    or a compose file falls back instead of binding to ``""``.

    Parameters
    ----------
    var : str
        Environment variable name.
    default : str
        Value used when ``var`` is unset or blank.
    env : Mapping[str, str], optional
        Mapping to read instead of :data:`os.environ`.

    Returns
    -------
    str
        An ``EnvVar``, a literal value from ``env``, or ``default``.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    raw = source.get(var, "").strip()
    if not raw:
        return default
    return EnvVar(var) if env is None else raw


def default_resources(
    config: OperationalConfig | None = None,
    env: Mapping[str, str] | None = None,
    satellites: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the resource set the code location ships with.

    Every resource here is constructible with no checkpoint, no network
    and no GPU: they record locations and settings, and open them lazily
    inside a run.

    Parameters
    ----------
    config : OperationalConfig, optional
        Defaults to :meth:`OperationalConfig.from_env`.
    env : Mapping[str, str], optional
        Mapping to read environment overrides from.
    satellites : sequence of str, optional
        Satellite list to advertise to the steps.  Defaults to the
        configuration's, but :func:`build_definitions` passes the
        satellites the asset graph was actually built for, so no step is
        told about a satellite that has no asset.

    Returns
    -------
    dict
        Keyed ``"paths"``, ``"store"``, ``"model"`` and
        ``"run_settings"`` — the keys the assets declare.
    """
    cfg = OperationalConfig.from_env(env) if config is None else config
    ring = list(cfg.satellites if satellites is None else satellites)
    return {
        "paths": PathsResource(
            output_dir=_env_or(
                "STEREO_WINDS_OP_OUTPUT_DIR", str(cfg.output_dir), env
            ),
        ),
        "store": IcechunkStoreResource(
            store_uri=_env_or("STEREO_WINDS_OP_STORE_URI", cfg.store_uri, env),
        ),
        "model": ModelResource(
            student_ckpt=_env_or(CHECKPOINT_ENV_VAR, DEFAULT_CHECKPOINT, env),
            raft_ckpt=_env_or(
                RAFT_CHECKPOINT_ENV_VAR, DEFAULT_RAFT_CHECKPOINT, env),
            device=_env_or("STEREO_WINDS_OP_DEVICE", cfg.device, env),
            row_strip=cfg.row_strip,
        ),
        "run_settings": RunSettingsResource(
            satellites=ring,
            flow_bands=list(cfg.flow_bands),
            rad_bands=list(cfg.rad_bands),
            cadence_minutes=cfg.cadence_minutes,
            availability_tolerance_minutes=cfg.availability_tolerance_minutes,
            resolution_m=cfg.resolution_m,
        ),
    }


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------

def discover_sensors(module: ModuleType = sensors_module) -> list[SensorDefinition]:
    """Collect the sensor objects a module defines.

    Discovering them rather than importing them by name keeps this module
    from having to track what the sensor unit calls its sensors, and
    picks up a second sensor automatically if one is added.

    Parameters
    ----------
    module : module, optional
        Module to scan.  Defaults to :mod:`operational.sensors`.

    Returns
    -------
    list of dagster.SensorDefinition
        Sensors in name order, de-duplicated by name so one exported
        under an alias is not registered twice.
    """
    found: dict[str, SensorDefinition] = {}
    for attr in dir(module):
        if attr.startswith("_"):
            continue
        obj = getattr(module, attr)
        if isinstance(obj, SensorDefinition):
            found.setdefault(obj.name, obj)
    return [found[name] for name in sorted(found)]


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------

def check_satellite_agreement(
    config: OperationalConfig, satellites: Iterable[str]
) -> set[str]:
    """Compare the configured ring with the one the assets were built for.

    The asset modules read the satellite list when *they* are imported;
    this module reads it again when the code location is assembled.  The
    two normally agree, and when they do not the assets win — but every
    step would then be told about a ring that does not match the assets
    that exist, so the difference is worth a line in the log and is
    corrected before it reaches the resources.

    Parameters
    ----------
    config : OperationalConfig
        The configuration this code location was assembled with.
    satellites : iterable of str
        Satellite ids the asset graph was actually built for.

    Returns
    -------
    set of str
        Ids the two disagree about; empty when they match.
    """
    configured = set(config.satellites)
    built = set(satellites)
    difference = configured ^ built
    if difference:
        logger.warning(
            "the asset modules were built for %s but the current "
            "configuration lists %s; the assets win. Reload the code "
            "location if the configuration is the one you meant.",
            sorted(built),
            sorted(configured),
        )
    return difference


def check_mosaic_inputs(amv_keys: Iterable[AssetKey]) -> set[AssetKey]:
    """Compare the mosaic's declared inputs with the assets actually built.

    The satellite list reaches the graph twice — once as the per-satellite
    assets, once as the inputs ``global_mosaic`` declares — and a
    mismatch does not raise.  Dagster treats the missing satellites as
    external assets that nothing in this code location materialises, and
    the mosaic quietly waits on inputs that will never arrive.

    Both layers are built from the same source today, so this normally
    cannot fire; it earns its place once the mosaic asset takes its
    satellite list from somewhere of its own.  It warns rather than
    raises because a deployment deliberately running a subset of the ring
    is legitimate.

    Parameters
    ----------
    amv_keys : iterable of dagster.AssetKey
        Keys of the per-satellite assets this code location built.

    Returns
    -------
    set of dagster.AssetKey
        Inputs the mosaic declares that no built asset supplies.  Empty
        when the two agree.
    """
    built = set(amv_keys)
    declared: set[AssetKey] = set()
    for upstream in global_mosaic.asset_deps.values():
        declared.update(upstream)
    dangling = declared - built
    if dangling:
        logger.warning(
            "global_mosaic depends on %s, which this code location does not "
            "build; check that the satellite list is the same one "
            "operational.assets.mosaic_assets was imported with",
            sorted(key.to_user_string() for key in dangling),
        )
    unused = built - declared
    if unused:
        logger.warning(
            "built AMV assets %s feed nothing; global_mosaic does not list "
            "them among its inputs",
            sorted(key.to_user_string() for key in unused),
        )
    return dangling


# ---------------------------------------------------------------------------
# The code location
# ---------------------------------------------------------------------------

def resolve_amv_assets(
    partitions_def: PartitionsDefinition,
    config: OperationalConfig,
    module: ModuleType = amv_assets_module,
) -> dict[str, AssetsDefinition]:
    """The per-satellite AMV assets, keyed by satellite id.

    The AMV module builds its assets at module scope and publishes them
    as ``AMV_ASSETS_BY_SAT``.  Those objects are used as-is when they are
    there: re-deriving the satellite list here would let the graph's
    upstream layer disagree with the inputs ``global_mosaic`` declared
    when *it* was imported, which Dagster does not complain about — it
    just treats the difference as external assets nothing materialises.

    The fallback builds them from ``config`` through the module's
    ``build_amv_asset`` factory, for an AMV module that offers only that.

    Parameters
    ----------
    partitions_def : dagster.PartitionsDefinition
        Partitions for assets built by the fallback path.
    config : OperationalConfig
        Supplies the satellite list for the fallback path.
    module : module, optional
        Module to take the assets from.  Defaults to
        :mod:`operational.assets.amv_assets`.

    Returns
    -------
    dict
        Satellite id to that satellite's :class:`~dagster.AssetsDefinition`.
    """
    prebuilt = getattr(module, "AMV_ASSETS_BY_SAT", None)
    if isinstance(prebuilt, Mapping) and prebuilt:
        return {str(sat_id): asset_def for sat_id, asset_def in prebuilt.items()}
    logger.debug(
        "%s exposes no AMV_ASSETS_BY_SAT; building from the configuration",
        module.__name__,
    )
    return {
        sat_id: module.build_amv_asset(sat_id, partitions_def)
        for sat_id in config.satellites
    }


def build_definitions(
    config: OperationalConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> Definitions:
    """Assemble the code location.

    The asset graph comes from the asset modules, which build it at
    import from the environment.  ``config`` and ``env`` therefore shape
    the resources, the jobs and the cross-checks — not which assets
    exist.  Changing the satellite list or the cadence means reloading
    the code location, which is what a Dagster deployment does anyway.

    Parameters
    ----------
    config : OperationalConfig, optional
        Defaults to :meth:`OperationalConfig.from_env`.
    env : Mapping[str, str], optional
        Mapping to read environment overrides from, for tests.

    Returns
    -------
    dagster.Definitions
        Assets, jobs, the backstop schedule, the sensors and the
        resources.
    """
    cfg = OperationalConfig.from_env(env) if config is None else config
    partitions = resolve_partitions_def(cfg, env)

    amv_by_sat = resolve_amv_assets(partitions, cfg)
    satellite_keys = {
        sat_id: list(asset_def.keys) for sat_id, asset_def in amv_by_sat.items()
    }
    check_satellite_agreement(cfg, satellite_keys)
    check_mosaic_inputs(key for keys in satellite_keys.values() for key in keys)

    max_concurrent = max_concurrent_from_env(env)
    full_job = build_full_job(max_concurrent=max_concurrent)
    jobs = [
        full_job,
        *build_satellite_jobs(satellite_keys, max_concurrent=max_concurrent),
    ]
    schedule = build_backstop_schedule(full_job, partitions)

    logger.debug(
        "operational code location: %d satellites, %d jobs, partitions %r",
        len(amv_by_sat),
        len(jobs),
        getattr(partitions, "cron_schedule", partitions),
    )
    return Definitions(
        assets=[*amv_by_sat.values(), global_mosaic, published_mosaic],
        jobs=jobs,
        schedules=[schedule],
        sensors=discover_sensors(),
        resources=default_resources(cfg, env, satellites=list(amv_by_sat)),
    )


#: The code location ``dagster dev -m operational.definitions`` loads.
defs = build_definitions()
