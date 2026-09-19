"""Jobs and the backstop schedule for the operational AMV pipeline.

Two kinds of job are built here.

``operational_ring_job``
    The whole graph for one timestamp partition: every per-satellite AMV
    retrieval, the global mosaic that merges them, and the icechunk
    publish step.  This is what the availability sensor launches and what
    the backstop schedule re-launches.

``operational_amv_<sat_id>_job``
    One satellite's retrieval on its own, for the common operational
    chore of re-running a single satellite whose data arrived late or
    whose retrieval failed.  By default it stops at that asset, so
    re-running GOES-19 does not silently rebuild the mosaic from a
    half-refreshed set of inputs; pass ``include_downstream=True`` when
    you do want the mosaic and publish steps to follow.

Why concurrency is throttled
----------------------------
Each per-satellite step runs the student model over a whole full disk and
peaks at several GB of resident memory.  The CLI equivalent of this
pipeline has been OOM-killed running satellites back-to-back in one
process, so the default here is the *multiprocess* executor with
:data:`DEFAULT_MAX_CONCURRENT` = 1: steps run one at a time, each in its
own process, so the operating system reclaims every satellite's memory
before the next one starts.  Four full disks in flight at once is exactly
the configuration that has failed before and is never the default.

Raise the limit only on a machine whose memory you have measured, via the
:data:`MAX_CONCURRENT_ENV_VAR` environment variable, e.g.
``STEREO_WINDS_OP_MAX_CONCURRENT=2``.  Dagster's own per-run and
per-instance concurrency limits still apply on top of this.

Retries
-------
:data:`DEFAULT_RETRY_POLICY` retries a step a couple of times with
exponential backoff and jitter.  The failures these steps actually see
are transient object-store reads and half-written upstream files, which a
delayed retry clears; a genuinely bad partition fails all the same, just
a few minutes later.

It is a *default*, not a guarantee: Dagster prefers a policy declared on
the asset itself, and the AMV assets declare one.  The policy here
therefore covers the steps that do not, so changing it does not change
how the per-satellite retrievals retry.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from dagster import (
    AssetKey,
    AssetSelection,
    Backoff,
    DefaultScheduleStatus,
    ExecutorDefinition,
    Jitter,
    PartitionsDefinition,
    RetryPolicy,
    build_schedule_from_partitioned_job,
    define_asset_job,
    multiprocess_executor,
)

if TYPE_CHECKING:
    # Not exported from the top-level ``dagster`` namespace in 1.13, and
    # only ever needed as an annotation, so it stays behind TYPE_CHECKING.
    from dagster._core.definitions.unresolved_asset_job_definition import (
        UnresolvedAssetJobDefinition,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "BACKSTOP_MINUTE_OF_HOUR",
    "BACKSTOP_SCHEDULE_NAME",
    "DEFAULT_MAX_CONCURRENT",
    "DEFAULT_RETRY_POLICY",
    "FULL_JOB_NAME",
    "MAX_CONCURRENT_ENV_VAR",
    "RUN_TAGS",
    "build_backstop_schedule",
    "build_full_job",
    "build_satellite_job",
    "build_satellite_jobs",
    "supports_minute_of_hour",
    "max_concurrent_from_env",
    "modest_executor",
    "satellite_job_name",
]

#: Name of the job covering the full asset graph for one partition.
#: Other units target the pipeline by this name rather than by importing
#: the job object, so keep it stable.
FULL_JOB_NAME = "operational_ring_job"

#: Name of the belt-and-braces schedule built by
#: :func:`build_backstop_schedule`.
BACKSTOP_SCHEDULE_NAME = "operational_ring_backstop_schedule"

#: Environment variable overriding how many steps may run at once.
MAX_CONCURRENT_ENV_VAR = "STEREO_WINDS_OP_MAX_CONCURRENT"

#: Steps in flight at once when nothing overrides it.  One, because a
#: single full-disk retrieval already peaks at several GB — see the
#: module docstring.
DEFAULT_MAX_CONCURRENT = 1

#: Minute of the hour the backstop fires at.  A partitioned schedule
#: requests the most recently *closed* partition, so the exact minute
#: changes which wall-clock moment the tick happens at, not which
#: partition it asks for.  A few minutes past the hour keeps the backstop
#: off the top of the hour, where the sensor is busy launching the slot
#: that has just opened.
BACKSTOP_MINUTE_OF_HOUR = 15

#: Partition crons a minute-of-hour offset can legally be applied to:
#: a literal minute, and an hour field that is either every hour or a
#: single hour.  ``*/10 * * * *`` and ``0 0,6,12,18 * * *`` are not.
_OFFSETTABLE_CRON = re.compile(r"^\d+ (\*|\d+) \* \* \*$")

#: Characters Dagster will not accept in a job name.
_UNSAFE_IN_NAME = re.compile(r"[^0-9A-Za-z_]+")

#: Retry policy for steps that do not carry one of their own.  An asset
#: that declares ``retry_policy=`` on its decorator wins over this, so
#: changing it here does not change how those assets retry.
DEFAULT_RETRY_POLICY = RetryPolicy(
    max_retries=2,
    delay=120,
    backoff=Backoff.EXPONENTIAL,
    jitter=Jitter.PLUS_MINUS,
)

#: Tags stamped on every run, so operational runs are filterable in the
#: UI and distinguishable from ad-hoc backfills.
RUN_TAGS: Mapping[str, str] = {
    "stereo_winds/pipeline": "operational-ring",
}

#: Additional tags identifying a run the backstop schedule launched.
BACKSTOP_RUN_TAGS: Mapping[str, str] = {
    **RUN_TAGS,
    "stereo_winds/trigger": "backstop-schedule",
}


def satellite_job_name(sat_id: str) -> str:
    """Name of the single-satellite re-run job.

    Half the ring carries punctuation in its id — ``mtg-i1``,
    ``msg-iodc`` — and Dagster rejects a job name that is not an
    identifier, at repository-resolution time rather than at
    construction, so an unslugified name fails the whole code location
    rather than just that job.  Punctuation is therefore replaced with
    underscores here.

    Parameters
    ----------
    sat_id : str
        Satellite id, e.g. ``"goes18"`` or ``"mtg-i1"``.

    Returns
    -------
    str
        ``"operational_amv_<slugified sat_id>_job"``, e.g.
        ``"operational_amv_mtg_i1_job"``.

    Raises
    ------
    ValueError
        If ``sat_id`` has no alphanumeric characters to build a name
        from.
    """
    safe = _UNSAFE_IN_NAME.sub("_", sat_id).strip("_")
    if not safe:
        raise ValueError(f"satellite id {sat_id!r} yields no usable job name")
    return f"operational_amv_{safe}_job"


def max_concurrent_from_env(env: Mapping[str, str] | None = None) -> int:
    """Read the step-concurrency limit from the environment.

    Parameters
    ----------
    env : Mapping[str, str], optional
        Mapping to read instead of :data:`os.environ`.

    Returns
    -------
    int
        The value of :data:`MAX_CONCURRENT_ENV_VAR` when it is set to a
        positive integer, otherwise :data:`DEFAULT_MAX_CONCURRENT`.  A
        malformed or non-positive value is logged and ignored rather than
        raised on, because this is read while a code location loads and a
        typo in a deployment's environment should not take the location
        down.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    raw = source.get(MAX_CONCURRENT_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_MAX_CONCURRENT
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; falling back to %d",
            MAX_CONCURRENT_ENV_VAR,
            raw,
            DEFAULT_MAX_CONCURRENT,
        )
        return DEFAULT_MAX_CONCURRENT
    if value < 1:
        logger.warning(
            "%s=%d is below 1; falling back to %d",
            MAX_CONCURRENT_ENV_VAR,
            value,
            DEFAULT_MAX_CONCURRENT,
        )
        return DEFAULT_MAX_CONCURRENT
    return value


def modest_executor(max_concurrent: int | None = None) -> ExecutorDefinition:
    """Build the memory-conscious executor these jobs run under.

    Parameters
    ----------
    max_concurrent : int, optional
        Steps allowed in flight at once.  Defaults to
        :func:`max_concurrent_from_env`.

    Returns
    -------
    dagster.ExecutorDefinition
        A configured multiprocess executor.  Process-per-step matters as
        much as the limit itself: the memory a full-disk retrieval
        allocates goes back to the OS when its process exits, which it
        would not do inside a long-lived in-process run.

    Raises
    ------
    ValueError
        If ``max_concurrent`` is given and is below 1.
    """
    limit = max_concurrent_from_env() if max_concurrent is None else max_concurrent
    if limit < 1:
        raise ValueError(f"max_concurrent must be >= 1, got {limit}")
    logger.debug("operational executor: multiprocess, max_concurrent=%d", limit)
    return multiprocess_executor.configured({"max_concurrent": limit})


def _job_kwargs(
    max_concurrent: int | None,
    retry_policy: RetryPolicy | None,
    tags: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Assemble the defaults shared by every job built in this module."""
    return {
        "executor_def": modest_executor(max_concurrent),
        "op_retry_policy": DEFAULT_RETRY_POLICY if retry_policy is None else retry_policy,
        "tags": dict(RUN_TAGS if tags is None else tags),
    }


def build_full_job(
    *,
    name: str = FULL_JOB_NAME,
    selection: Any = None,
    max_concurrent: int | None = None,
    retry_policy: RetryPolicy | None = None,
    tags: Mapping[str, str] | None = None,
) -> UnresolvedAssetJobDefinition:
    """Build the job covering the whole pipeline for one partition.

    Parameters
    ----------
    name : str, optional
        Job name.  Defaults to :data:`FULL_JOB_NAME`.
    selection : CoercibleToAssetSelection, optional
        Assets to include.  Defaults to every asset in the code location,
        which is the per-satellite AMV assets plus the mosaic and publish
        assets; pass a narrower selection to carve out a subset without
        depending on the assets' naming.
    max_concurrent : int, optional
        Steps in flight at once.  See :func:`modest_executor`.
    retry_policy : dagster.RetryPolicy, optional
        Per-step retry policy.  Defaults to :data:`DEFAULT_RETRY_POLICY`.
    tags : Mapping[str, str], optional
        Run tags.  Defaults to :data:`RUN_TAGS`.

    Returns
    -------
    dagster.UnresolvedAssetJobDefinition
        The job, resolved against the asset graph by
        :class:`~dagster.Definitions`.  Its partitions are inferred from
        the selected assets, so it is partitioned by timestamp.
    """
    return _define_asset_job(
        name=name,
        selection=AssetSelection.all() if selection is None else selection,
        description=(
            "Full operational AMV pipeline for one timestamp: per-satellite "
            "retrievals, global mosaic, icechunk publish."
        ),
        max_concurrent=max_concurrent,
        retry_policy=retry_policy,
        tags=tags,
    )


def build_satellite_job(
    sat_id: str,
    asset_keys: Iterable[AssetKey],
    *,
    include_downstream: bool = False,
    max_concurrent: int | None = None,
    retry_policy: RetryPolicy | None = None,
    tags: Mapping[str, str] | None = None,
) -> UnresolvedAssetJobDefinition:
    """Build the re-run job for a single satellite.

    Parameters
    ----------
    sat_id : str
        Satellite id, used for the job name and description.
    asset_keys : iterable of dagster.AssetKey
        Keys of that satellite's AMV asset.  Passed in rather than
        derived from ``sat_id`` so this module makes no assumption about
        how the AMV unit names its assets.
    include_downstream : bool, optional
        When True the mosaic and publish steps run after the retrieval.
        Off by default: re-running one satellite normally means fixing
        one input, and rebuilding the mosaic from a partially refreshed
        set of inputs is rarely what is wanted.
    max_concurrent : int, optional
        Steps in flight at once.  See :func:`modest_executor`.
    retry_policy : dagster.RetryPolicy, optional
        Per-step retry policy.
    tags : Mapping[str, str], optional
        Run tags.

    Returns
    -------
    dagster.UnresolvedAssetJobDefinition
        A job named by :func:`satellite_job_name`.

    Raises
    ------
    ValueError
        If ``asset_keys`` is empty.
    """
    keys = list(asset_keys)
    if not keys:
        raise ValueError(f"no asset keys given for satellite {sat_id!r}")
    selection = AssetSelection.assets(*keys)
    if include_downstream:
        selection = selection.downstream()
        scope = "and everything downstream of it"
    else:
        scope = "only"
    return _define_asset_job(
        name=satellite_job_name(sat_id),
        selection=selection,
        description=f"Re-run the AMV retrieval for {sat_id} {scope}.",
        max_concurrent=max_concurrent,
        retry_policy=retry_policy,
        tags=tags,
    )


def build_satellite_jobs(
    satellite_asset_keys: Mapping[str, Iterable[AssetKey]],
    **kwargs: Any,
) -> list[UnresolvedAssetJobDefinition]:
    """Build one re-run job per satellite.

    Parameters
    ----------
    satellite_asset_keys : Mapping[str, iterable of dagster.AssetKey]
        Satellite id to that satellite's asset keys.
    **kwargs
        Forwarded to :func:`build_satellite_job`.

    Returns
    -------
    list of dagster.UnresolvedAssetJobDefinition
        Jobs in the iteration order of ``satellite_asset_keys``.

    Raises
    ------
    ValueError
        If two satellite ids slug to the same job name — ``mtg-i1`` and
        ``mtg_i1``, say.  Dagster rejects a repository with duplicate job
        names while resolving it, which fails the whole code location
        with no mention of the satellites that caused it.
    """
    jobs = [
        build_satellite_job(sat_id, keys, **kwargs)
        for sat_id, keys in satellite_asset_keys.items()
    ]
    seen: dict[str, str] = {}
    for sat_id, job in zip(satellite_asset_keys, jobs):
        clash = seen.get(job.name)
        if clash is not None:
            raise ValueError(
                f"satellite ids {clash!r} and {sat_id!r} both give the job "
                f"name {job.name!r}; rename one of them"
            )
        seen[job.name] = sat_id
    return jobs


def supports_minute_of_hour(partitions_def: PartitionsDefinition) -> bool:
    """Whether a partitions definition accepts a minute-of-hour offset.

    Dagster derives a partitioned schedule's cron from the partitions'
    own cron, and only lets an offset be applied to the plain hourly and
    daily shapes — a minute field of ``*/10`` or an hour field of
    ``0,6,12,18`` has no single "minute past the hour" to move.  Passing
    an offset to anything else does not warn, it raises while the code
    location is loading, which would take the whole deployment down for
    the sake of a cosmetic firing time.

    Parameters
    ----------
    partitions_def : dagster.PartitionsDefinition
        The partitions the schedule's job is built on.

    Returns
    -------
    bool
        True when the partition cron is ``<minute> * * * *`` or
        ``<minute> <hour> * * *``; False for everything else, including
        partitions with no cron at all.
    """
    cron = getattr(partitions_def, "cron_schedule", None)
    if not isinstance(cron, str):
        return False
    return bool(_OFFSETTABLE_CRON.match(cron))


def build_backstop_schedule(
    job: UnresolvedAssetJobDefinition,
    partitions_def: PartitionsDefinition,
    *,
    name: str = BACKSTOP_SCHEDULE_NAME,
    minute_of_hour: int | None = BACKSTOP_MINUTE_OF_HOUR,
    default_status: DefaultScheduleStatus = DefaultScheduleStatus.RUNNING,
    tags: Mapping[str, str] | None = None,
):
    """Build the belt-and-braces schedule behind the availability sensor.

    The availability sensor is what normally starts a run: it notices new
    imagery within a tick or two of it landing, which is both faster and
    more precise than a clock.  But a sensor is a daemon, and daemons get
    restarted, redeployed and occasionally wedged — and a timestamp missed
    while the sensor is down is simply never retrieved, because nothing
    re-examines it.  This schedule closes that hole: it fires once per
    partition regardless of sensor health, so the worst case for an
    outage is a late timestamp rather than a lost one.

    Each tick requests the most recently *closed* partition, which is the
    previous slot — with the default hourly cadence, the 05:00 partition
    is requested by the tick just after 06:00.  Imagery for a slot lands
    ten to fifteen minutes into it, so a healthy sensor has had the best
    part of an hour to launch that run before the backstop repeats it.

    When both do fire for the same partition, two runs exist for it: the
    schedule de-duplicates against its own earlier ticks, never against
    the sensor.  That makes per-timestamp idempotency a requirement of
    the steps rather than a nicety — re-running a partition has to
    replace that timestamp's outputs, in the icechunk store as much as on
    disk, or the backstop turns a missed tick into a duplicated one.
    Dagster's run-level concurrency limits are the place to cap the
    wasted work.

    Parameters
    ----------
    job : dagster.UnresolvedAssetJobDefinition
        The partitioned job to launch — normally the output of
        :func:`build_full_job`.
    partitions_def : dagster.PartitionsDefinition
        The partitions ``job`` resolves to.  It must be that grid and not
        a separately-read setting: Dagster validates the offset against
        the job's own partitions, but not until the code location is
        being resolved, so passing a different grid here turns a wrong
        firing time into a code location that will not load.
    name : str, optional
        Schedule name.  Defaults to :data:`BACKSTOP_SCHEDULE_NAME`.
    minute_of_hour : int, optional
        Minute past the hour to fire at, or None to fire exactly on the
        partition boundary.  Dropped when the partitions do not accept an
        offset — see :func:`supports_minute_of_hour`.
    default_status : dagster.DefaultScheduleStatus, optional
        Whether the schedule is running when first deployed.  Running by
        default — a backstop that has to be switched on is not a backstop.
    tags : Mapping[str, str], optional
        Run tags.  Defaults to :data:`BACKSTOP_RUN_TAGS`, which marks the
        run as schedule-triggered.

    Returns
    -------
    dagster.UnresolvedPartitionedAssetScheduleDefinition
        A schedule targeting ``job``, requesting the most recently closed
        partition on each tick with the partition key as the run key.
    """
    offset = minute_of_hour if supports_minute_of_hour(partitions_def) else None
    if minute_of_hour is not None and offset is None:
        logger.debug(
            "partitions %r take no minute-of-hour offset; the backstop will "
            "fire on the partition boundary instead",
            getattr(partitions_def, "cron_schedule", partitions_def),
        )
    return build_schedule_from_partitioned_job(
        job,
        name=name,
        minute_of_hour=offset,
        default_status=default_status,
        tags=dict(BACKSTOP_RUN_TAGS if tags is None else tags),
        description=(
            "Backstop for the availability sensor: launches "
            f"{job.name} for the most recently closed partition so a "
            "missed sensor tick delays a timestamp instead of losing it."
        ),
    )


def _define_asset_job(
    *,
    name: str,
    selection: Any,
    description: str,
    max_concurrent: int | None,
    retry_policy: RetryPolicy | None,
    tags: Mapping[str, str] | None,
) -> UnresolvedAssetJobDefinition:
    """Create an asset job carrying this module's operational defaults."""
    return define_asset_job(
        name=name,
        selection=selection,
        description=description,
        **_job_kwargs(max_concurrent, retry_policy, tags),
    )
