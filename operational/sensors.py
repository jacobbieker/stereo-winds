"""Dagster sensors that detect newly available satellite data.

The availability sensor is the entry point of the operational pipeline: it polls
the satellite archives on a fixed interval and turns "these timestamps are now
downloadable" into partitioned Dagster runs.

Readiness rule
--------------
``require_all`` decides when a partition timestamp is emitted:

``True`` (default)
    Every configured satellite must be able to deliver the timestamp.  Waiting
    for the full roster keeps the mosaic complete; the downstream mosaic step
    still degrades gracefully to a partial mosaic if a satellite fails *after*
    the run has started, so requiring all here only costs latency, never
    coverage.
``False``
    The timestamp is emitted as soon as any satellite can deliver it, trading
    mosaic completeness for latency.

Bounded look-back
-----------------
Every tick searches at most ``lookback_hours`` back from "now".  A watermark
older than that is clamped to the window start, so a sensor that has been off
for a week resumes at the recent edge of the archive instead of replaying it.

Watermarks vs. the Dagster cursor
---------------------------------
The durable :class:`~operational.core.watermark.WatermarkStore` is the single
source of truth for "what has already been emitted": it is the only input used
to compute each satellite's search window, so it survives restarts, code
reloads, and Dagster instance wipes.  The sensor also writes a JSON *mirror* of
that state into ``SensorEvaluationContext.cursor`` so an operator can read the
sensor's position straight off the tick in the Dagster UI, and so a disagreement
between the two (a wiped or rolled-back watermark file) shows up as a warning in
the tick log.  The cursor is never used to widen or narrow the search window --
clearing the watermark store alone is enough to force a replay.

Watermarks advance only up to the oldest candidate that was *not* emitted, and
that barrier applies to every satellite at once -- including satellites that
never reported the withheld timestamp.  If a tick sees ``T1`` from one satellite
and ``T1, T2`` from another, and only ``T2`` clears the readiness rule, no
watermark may pass ``T1``: letting the satellite that is missing ``T1`` jump to
``T2`` would mean it never looks at ``T1`` again, and the partition could never
complete.  Timestamps above the barrier are simply re-reported next tick and
de-duplicated by their ``run_key``.

Failures are per satellite: an archive outage is logged, that satellite
contributes nothing to this tick, and its watermark is left untouched.

Delivery semantics
------------------
Watermarks are advanced while the tick is being evaluated, before Dagster has
submitted the resulting runs, so a daemon that dies mid-tick can leave a
timestamp marked as emitted without a run behind it.  The trade is deliberate:
the alternative -- advancing only after submission -- is not available to a
sensor (Dagster drains the generator before it launches anything), and the
failure it would replace is duplicate runs on every crash rather than a missed
partition on a rare one.  Recovering a missed partition is a one-liner:
``WatermarkStore.reset(sat_id)``, or a backfill of the partition itself.  In the
common direction the ``run_key`` guarantees the opposite hazard never bites --
a repeated tick for an already-requested partition cannot start a second run.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dagster import (
    DefaultSensorStatus,
    RunRequest,
    SensorDefinition,
    SensorEvaluationContext,
    SkipReason,
    sensor,
)

from operational.config import OperationalConfig
from operational.core import availability as availability_mod
from operational.core.partitions import key_for
from operational.core.watermark import WatermarkStore

logger = logging.getLogger(__name__)

DEFAULT_SENSOR_NAME = "amv_availability_sensor"
#: Job this sensor requests runs of.  Must match the name the job
#: is actually defined under (``jobs.FULL_JOB_NAME``); dagster
#: rejects the whole repository if a sensor targets a job that
#: does not exist.  Kept as a string so this module stays below
#: jobs.py in the dependency order.
DEFAULT_JOB_NAME = "operational_ring_job"
DEFAULT_LOOKBACK_HOURS = 6.0
DEFAULT_MINIMUM_INTERVAL_SECONDS = 300
WATERMARK_FILENAME = "watermarks.json"


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _as_utc(t: datetime) -> datetime:
    """Return ``t`` as a *naive* UTC datetime (naive values are already UTC).

    Naive UTC is this codebase's convention -- the ring script, the
    watermark store and the partition helpers all speak it -- so the
    sensor normalises inward rather than making everything it touches
    timezone-aware.  Dagster hands us aware datetimes, and comparing one
    of those against a watermark read back from disk raises TypeError.
    """
    if t.tzinfo is None:
        return t
    return t.astimezone(timezone.utc).replace(tzinfo=None)


def _encode_cursor(watermarks: Mapping[str, datetime], tick_time: datetime) -> str:
    """Serialise the watermark mirror written to the Dagster cursor."""
    return json.dumps(
        {
            "watermarks": {k: _as_utc(v).isoformat() for k, v in sorted(watermarks.items())},
            "tick_time": _as_utc(tick_time).isoformat(),
        },
        sort_keys=True,
    )


def _decode_cursor(raw: str | None) -> dict[str, datetime]:
    """Parse the watermark mirror from a Dagster cursor, tolerating garbage."""
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        items = payload["watermarks"]
    except (ValueError, TypeError, KeyError):
        logger.warning("Ignoring unparseable sensor cursor %r", raw)
        return {}
    out: dict[str, datetime] = {}
    if not isinstance(items, dict):
        return out
    for sat_id, value in items.items():
        try:
            out[str(sat_id)] = _as_utc(datetime.fromisoformat(str(value)))
        except ValueError:
            logger.warning("Ignoring bad cursor watermark %r for %s", value, sat_id)
    return out


def _resolve_watermark_path(
    watermark_path: Path | str | None, config: OperationalConfig
) -> Path:
    """Pick the watermark file and pin it to an absolute path.

    Precedence is explicit argument, then the ``STEREO_WINDS_OPERATIONAL_WATERMARK``
    environment variable, then ``config.output_dir / "watermarks.json"``.

    Parameters
    ----------
    watermark_path
        Explicit override, or ``None`` to fall back to env/config.
    config
        Operational settings supplying the default output directory.

    Returns
    -------
    Path
        Absolute path, resolved once at definition time.

    Notes
    -----
    Resolving eagerly matters: the watermark store is the sensor's source of
    truth, and a relative default would follow the working directory of whatever
    process hosts the code location.  Two daemons started from different
    directories would then keep two private watermarks and replay each other's
    partitions.
    """
    if watermark_path is None:
        watermark_path = os.environ.get("STEREO_WINDS_OPERATIONAL_WATERMARK")
    if watermark_path is None:
        watermark_path = Path(config.output_dir) / WATERMARK_FILENAME
    return Path(watermark_path).expanduser().resolve()


def _fmt(t: datetime) -> str:
    """Format a timestamp compactly for operator-facing log/skip messages."""
    return _as_utc(t).strftime("%Y-%m-%dT%H:%MZ")


def _collect_per_satellite(
    *,
    config: OperationalConfig,
    satellites: Sequence[str],
    store: WatermarkStore,
    window_start: datetime,
    window_end: datetime,
    log: logging.Logger,
) -> tuple[dict[str, list[datetime]], dict[str, str]]:
    """Ask each satellite what is newly available inside the look-back window.

    Parameters
    ----------
    config
        Operational settings (bands, cadence, tolerance).
    satellites
        Satellite ids to poll.
    store
        Watermark store used to pick each satellite's search start.
    window_start, window_end
        Bounds of the look-back window; a watermark older than ``window_start``
        is clamped to it.
    log
        Logger used for per-satellite diagnostics.

    Returns
    -------
    reported : dict
        Sorted, de-duplicated timestamps per satellite that answered.
    failures : dict
        Error text per satellite whose lookup raised.
    """
    reported: dict[str, list[datetime]] = {}
    failures: dict[str, str] = {}

    for sat_id in satellites:
        watermark = store.get(sat_id)
        since = window_start if watermark is None else max(_as_utc(watermark), window_start)
        if since >= window_end:
            # Clock skew, or a watermark written by a future-dated replay: the
            # window is empty, so do not hand the archive an inverted range.
            log.warning(
                "Watermark for %s (%s) is at or ahead of the tick time (%s); "
                "nothing to search this tick",
                sat_id,
                _fmt(since),
                _fmt(window_end),
            )
            reported[sat_id] = []
            continue
        try:
            found = availability_mod.new_timestamps(
                sat_id,
                since=since,
                until=window_end,
                flow_bands=list(config.flow_bands),
                rad_bands=list(config.rad_bands),
                tolerance_minutes=config.availability_tolerance_minutes,
                cadence_minutes=config.cadence_minutes,
            )
        except Exception as exc:  # noqa: BLE001 - one bad archive must not kill the tick
            failures[sat_id] = f"{type(exc).__name__}: {exc}"
            log.exception(
                "Availability lookup failed for %s over %s..%s; skipping it this tick "
                "and leaving its watermark at %s",
                sat_id,
                _fmt(since),
                _fmt(window_end),
                "unset" if watermark is None else _fmt(watermark),
            )
            continue

        timestamps = sorted({_as_utc(t) for t in (found or [])})
        reported[sat_id] = timestamps
        log.info(
            "%s: %d new timestamp(s) in %s..%s%s",
            sat_id,
            len(timestamps),
            _fmt(since),
            _fmt(window_end),
            f" ({', '.join(_fmt(t) for t in timestamps)})" if timestamps else "",
        )

    return reported, failures


def _select_timestamps(
    reported: Mapping[str, Sequence[datetime]],
    satellites: Sequence[str],
    require_all: bool,
) -> tuple[list[datetime], dict[datetime, list[str]]]:
    """Apply the readiness rule to the per-satellite availability.

    Parameters
    ----------
    reported
        Timestamps each satellite that answered can deliver.
    satellites
        The full configured roster -- a satellite missing from ``reported``
        (because its lookup failed) can never satisfy ``require_all``.
    require_all
        ``True`` to emit only timestamps every configured satellite can deliver.

    Returns
    -------
    emitted : list of datetime
        Timestamps to request runs for, ascending.
    providers : dict
        Satellites backing each candidate timestamp (all candidates, emitted or
        not), for logging.
    """
    providers: dict[datetime, list[str]] = {}
    for sat_id in satellites:
        for t in reported.get(sat_id, ()):  # type: ignore[arg-type]
            providers.setdefault(t, []).append(sat_id)

    if require_all:
        needed = len(satellites)
        emitted = [t for t in sorted(providers) if len(providers[t]) == needed]
    else:
        emitted = sorted(providers)
    return emitted, providers


def _advance_watermarks(
    *,
    store: WatermarkStore,
    reported: Mapping[str, Sequence[datetime]],
    emitted: Iterable[datetime],
    providers: Mapping[datetime, Sequence[str]],
    log: logging.Logger,
) -> None:
    """Advance every watermark up to, but never past, the oldest withheld candidate.

    Parameters
    ----------
    store
        Watermark store to update.
    reported
        Timestamps each satellite reported this tick (ascending).
    emitted
        Timestamps that actually produced a run request.
    providers
        Every candidate timestamp seen this tick and the satellites backing it.
    log
        Logger for the resulting watermark moves.

    Notes
    -----
    The barrier is *global*, not per satellite.  A timestamp withheld because one
    satellite was late has to stay below **every** watermark, including the
    watermarks of satellites that never reported it: if a late satellite were
    allowed to advance past a timestamp it has not delivered yet, its next search
    would start after that timestamp, the data would never be reported once it
    landed, and the partition could never be emitted -- while the satellites that
    did report it stay pinned below the barrier forever.

    Holding every watermark at the barrier costs nothing but a repeat report of
    the already-emitted timestamps above it; those re-emit under the same
    ``run_key`` and Dagster drops them.  The stall is also self-limiting: once
    the withheld candidate falls out of the bounded look-back window nobody
    reports it, the barrier disappears, and the watermarks advance.
    """
    emitted_set = set(emitted)
    withheld = [t for t in providers if t not in emitted_set]
    barrier = min(withheld) if withheld else None

    for sat_id, timestamps in reported.items():
        eligible = [
            t for t in timestamps if t in emitted_set and (barrier is None or t < barrier)
        ]
        if not eligible:
            continue
        advanced_to = max(eligible)
        if store.advance(sat_id, advanced_to):
            log.info("Watermark for %s advanced to %s", sat_id, _fmt(advanced_to))

    if barrier is not None:
        log.info(
            "Holding all watermarks below %s: it is the oldest candidate still "
            "waiting on a satellite",
            _fmt(barrier),
        )


def _skip_message(
    *,
    satellites: Sequence[str],
    reported: Mapping[str, Sequence[datetime]],
    failures: Mapping[str, str],
    providers: Mapping[datetime, Sequence[str]],
    require_all: bool,
    window_start: datetime,
    window_end: datetime,
) -> str:
    """Build the operator-facing explanation for a tick that emitted nothing."""
    parts = [
        f"No new partitions ready in {_fmt(window_start)}..{_fmt(window_end)} "
        f"(rule: {'all' if require_all else 'any'} of {len(satellites)} satellites)."
    ]
    if providers:
        withheld = ", ".join(
            f"{_fmt(t)} [{'+'.join(providers[t])}]" for t in sorted(providers)
        )
        parts.append(f"Waiting on {len(providers)} candidate(s): {withheld}.")
        contributing = {sat for backers in providers.values() for sat in backers}
        # Distinguish "answered, had nothing new" from "could not be asked" --
        # only the latter is an archive problem worth chasing.
        quiet = sorted(s for s in satellites if s not in contributing and s in reported)
        if quiet:
            parts.append(f"Reported no new data: {', '.join(quiet)}.")
    else:
        answered = ", ".join(sat for sat in satellites if sat in reported) or "none"
        parts.append(f"Satellites reporting nothing new: {answered}.")
    if failures:
        parts.append(
            "Availability lookup failed (watermark unchanged) for "
            + "; ".join(f"{sat}: {err}" for sat, err in sorted(failures.items()))
            + "."
        )
    return " ".join(parts)


def build_availability_sensor(
    *,
    config: OperationalConfig | None = None,
    watermark_path: Path | str | None = None,
    lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
    require_all: bool = True,
    minimum_interval_seconds: int = DEFAULT_MINIMUM_INTERVAL_SECONDS,
    name: str = DEFAULT_SENSOR_NAME,
    job_name: str = DEFAULT_JOB_NAME,
    job: object | None = None,
    default_status: DefaultSensorStatus = DefaultSensorStatus.STOPPED,
    now_fn: Callable[[], datetime] | None = None,
) -> SensorDefinition:
    """Build the sensor that requests runs for newly available timestamps.

    Parameters
    ----------
    config
        Operational settings; defaults to :class:`OperationalConfig` defaults.
    watermark_path
        Watermark file; defaults to ``config.output_dir / "watermarks.json"``.
    lookback_hours
        Bound on how far back a tick searches.  Must be positive.
    require_all
        ``True`` (default) emits a timestamp only when every configured
        satellite can deliver it; ``False`` emits as soon as any can.
    minimum_interval_seconds
        Dagster's floor on the gap between ticks.
    name, job_name, job, default_status
        Standard Dagster sensor wiring.  ``job`` takes precedence over
        ``job_name`` when given.
    now_fn
        Clock used for the look-back window; injectable for tests.

    Returns
    -------
    SensorDefinition
        A sensor yielding :class:`RunRequest` per ready partition, or a single
        :class:`SkipReason` when nothing is ready.
    """
    cfg = config or OperationalConfig()
    if lookback_hours <= 0:
        raise ValueError("lookback_hours must be positive")
    if not cfg.satellites:
        raise ValueError("OperationalConfig.satellites must not be empty")
    if cfg.cadence_minutes <= 0:
        # Caught here rather than inside the per-satellite try, where it would
        # masquerade as an archive outage on every tick, forever.
        raise ValueError("OperationalConfig.cadence_minutes must be positive")
    wm_path = _resolve_watermark_path(watermark_path, cfg)
    clock = now_fn or _utcnow
    # The satellites a timestamp is judged on, which is not necessarily
    # the whole ring: the icechunk-only satellites get their own assets
    # but are too sparsely covered to hold up a run, and the mosaic
    # already tolerates one being absent.
    satellites = tuple(cfg.required_satellites or cfg.satellites)
    if set(satellites) != set(cfg.satellites):
        logger.info(
            "Waiting on %s; %s will be mosaicked when present but not "
            "waited for", ", ".join(satellites),
            ", ".join(sorted(set(cfg.satellites) - set(satellites))))
    rule = "all" if require_all else "any"

    description = (
        f"Polls {', '.join(satellites)} for newly available scenes over the last "
        f"{lookback_hours:g} h and requests one AMV pipeline run per partition that "
        f"{rule} of them can deliver. Incremental via the watermark store at {wm_path}."
    )

    @sensor(
        name=name,
        description=description,
        minimum_interval_seconds=minimum_interval_seconds,
        default_status=default_status,
        **({"job": job} if job is not None else {"job_name": job_name}),  # type: ignore[arg-type]
    )
    def _availability_sensor(context: SensorEvaluationContext):
        log = getattr(context, "log", None) or logger
        store = WatermarkStore(wm_path)

        now = _as_utc(clock())
        window_start = now - timedelta(hours=lookback_hours)

        mirrored = _decode_cursor(getattr(context, "cursor", None))
        persisted = store.all()
        for sat_id, cursor_t in mirrored.items():
            stored = persisted.get(sat_id)
            if stored is None or stored < cursor_t:
                log.warning(
                    "Watermark for %s (%s) is behind the sensor cursor (%s); the "
                    "watermark store is authoritative, so %s may be re-emitted.",
                    sat_id,
                    "unset" if stored is None else _fmt(stored),
                    _fmt(cursor_t),
                    sat_id,
                )

        reported, failures = _collect_per_satellite(
            config=cfg,
            satellites=satellites,
            store=store,
            window_start=window_start,
            window_end=now,
            log=log,
        )
        emitted, providers = _select_timestamps(reported, satellites, require_all)

        _advance_watermarks(
            store=store,
            reported=reported,
            emitted=emitted,
            providers=providers,
            log=log,
        )
        context.update_cursor(_encode_cursor(store.all(), now))

        if not emitted:
            yield SkipReason(
                _skip_message(
                    satellites=satellites,
                    reported=reported,
                    failures=failures,
                    providers=providers,
                    require_all=require_all,
                    window_start=window_start,
                    window_end=now,
                )
            )
            return

        log.info(
            "Requesting %d run(s): %s",
            len(emitted),
            ", ".join(key_for(t) for t in emitted),
        )
        for t in emitted:
            partition_key = key_for(t)
            yield RunRequest(
                run_key=partition_key,
                partition_key=partition_key,
                tags={
                    "operational/satellites": ",".join(providers[t]),
                    "operational/readiness_rule": rule,
                },
            )

    return _availability_sensor


#: Module-level sensor built from the default :class:`OperationalConfig`, for
#: ``Definitions(sensors=[availability_sensor])``.
availability_sensor = build_availability_sensor()
