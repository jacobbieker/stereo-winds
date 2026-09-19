"""The single definition of what an operational partition key is.

Every Dagster asset in the operational pipeline is partitioned by
timestamp: the availability sensor emits run requests keyed by
timestamp, the per-satellite AMV assets and the mosaic asset are
materialized per timestamp, and the icechunk writer appends one mosaic
per timestamp.  All of them agree on the format defined here so they
cannot drift apart.

Key format
----------
A partition key looks like ``"2026-08-01-06:00"``, i.e. ``%Y-%m-%d-%H:%M``.
Two reasons for that choice:

* It is the default ``fmt`` of Dagster's
  :class:`~dagster.TimeWindowPartitionsDefinition` for hourly and
  sub-daily schedules, so keys built here are exactly the keys Dagster
  builds itself.
* Every field is zero-padded and ordered most-significant first, so
  **lexicographic order over keys equals chronological order** — the
  cheap sort the sensor and the backfill logic rely on.

Not to be confused with the *filename* tag
------------------------------------------
``scripts/infer_student_global_ring.py`` renders the same instant as
``"20260801T0600"`` via its ``time_tag`` helper, and that string is baked
into the NetCDF filenames already on disk.  The two formats are
deliberately different and both are kept: ``time_tag`` names files,
:func:`key_for` names partitions.  Convert between them through a
:class:`~datetime.datetime`, never by string surgery.

Time zones
----------
The operational pipeline is UTC throughout.  Functions here accept
either naive datetimes (interpreted as UTC) or aware ones (converted to
UTC), and always *return* naive UTC datetimes, which is what the ring
script's own helpers expect.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dagster import TimeWindowPartitionsDefinition

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_START",
    "OPERATIONAL_PARTITIONS",
    "PARTITION_KEY_FORMAT",
    "MINUTES_PER_DAY",
    "align_to_cadence",
    "build_partitions_def",
    "cron_for_cadence",
    "is_on_cadence",
    "key_for",
    "keys_between",
    "time_for",
    "validate_cadence",
    "validate_on_cadence",
    "window_for",
]

#: ``strftime``/``strptime`` format of a partition key.  Matches Dagster's
#: default ``fmt`` for sub-daily :class:`TimeWindowPartitionsDefinition`.
PARTITION_KEY_FORMAT = "%Y-%m-%d-%H:%M"

MINUTES_PER_DAY = 24 * 60


# ---------------------------------------------------------------------------
# Normalisation and validation helpers
# ---------------------------------------------------------------------------

def _as_naive_utc(t: datetime, *, argname: str = "t") -> datetime:
    """Return ``t`` as a timezone-naive UTC datetime.

    Parameters
    ----------
    t : datetime
        Naive (assumed UTC) or aware timestamp.
    argname : str, optional
        Name used in the error message.

    Returns
    -------
    datetime
        Naive UTC equivalent of ``t``.

    Raises
    ------
    TypeError
        If ``t`` is not a :class:`~datetime.datetime`.
    """
    if not isinstance(t, datetime):
        raise TypeError(
            f"{argname} must be a datetime, got {type(t).__name__!r}"
        )
    if t.tzinfo is None:
        return t
    return t.astimezone(timezone.utc).replace(tzinfo=None)


def validate_cadence(cadence_minutes: int) -> int:
    """Check that a cadence tiles the day exactly.

    Parameters
    ----------
    cadence_minutes : int
        Spacing between consecutive partitions, in minutes.

    Returns
    -------
    int
        ``cadence_minutes`` unchanged, as an ``int``.

    Raises
    ------
    ValueError
        If the cadence is not a positive whole number of minutes, or if
        it does not divide the 1440-minute day evenly.  A cadence that
        does not divide the day would make the grid shift from one day
        to the next, so partition keys would stop lining up with the
        satellite scan schedule.
    """
    if isinstance(cadence_minutes, bool) or not isinstance(
        cadence_minutes, int
    ):
        raise ValueError(
            "cadence_minutes must be a positive integer number of minutes, "
            f"got {cadence_minutes!r}"
        )
    if cadence_minutes <= 0:
        raise ValueError(
            "cadence_minutes must be > 0, got " f"{cadence_minutes}"
        )
    if MINUTES_PER_DAY % cadence_minutes != 0:
        raise ValueError(
            f"cadence_minutes={cadence_minutes} does not divide the "
            f"{MINUTES_PER_DAY}-minute day evenly; the partition grid "
            "would drift across midnight. Use a divisor of 1440 "
            "(e.g. 10, 15, 30, 60, 180, 360, 720, 1440)."
        )
    return cadence_minutes


def is_on_cadence(t: datetime, cadence_minutes: int) -> bool:
    """Report whether ``t`` lands exactly on the cadence grid.

    Parameters
    ----------
    t : datetime
        Timestamp to test.
    cadence_minutes : int
        Cadence defining the grid.

    Returns
    -------
    bool
        True when ``t`` has no sub-minute component and its
        minutes-since-midnight is a multiple of ``cadence_minutes``.
    """
    validate_cadence(cadence_minutes)
    naive = _as_naive_utc(t)
    if naive.second or naive.microsecond:
        return False
    minutes = naive.hour * 60 + naive.minute
    return minutes % cadence_minutes == 0


def validate_on_cadence(
    t: datetime, cadence_minutes: int, *, argname: str = "t"
) -> datetime:
    """Return ``t`` as naive UTC, rejecting anything off the grid.

    Parameters
    ----------
    t : datetime
        Timestamp to check.
    cadence_minutes : int
        Cadence defining the grid.
    argname : str, optional
        Name used in the error message.

    Returns
    -------
    datetime
        Naive UTC ``t``.

    Raises
    ------
    ValueError
        If ``t`` is not on the grid; the message names the expected grid
        and the nearest grid point at or before ``t``.
    """
    naive = _as_naive_utc(t, argname=argname)
    if not is_on_cadence(naive, cadence_minutes):
        floored = align_to_cadence(naive, cadence_minutes)
        raise ValueError(
            f"{argname}={naive.isoformat()} is not on the "
            f"{cadence_minutes}-minute grid (every {cadence_minutes} min "
            f"from 00:00 UTC); the nearest grid point at or before it is "
            f"{floored.isoformat()}"
        )
    return naive


# ---------------------------------------------------------------------------
# Key <-> datetime
# ---------------------------------------------------------------------------

def key_for(t: datetime) -> str:
    """Render a timestamp as a partition key.

    Parameters
    ----------
    t : datetime
        Timestamp on a whole minute.  Naive input is treated as UTC;
        aware input is converted to UTC.

    Returns
    -------
    str
        The key, e.g. ``"2026-08-01-06:00"``.

    Raises
    ------
    ValueError
        If ``t`` carries seconds or microseconds.  The key format has
        minute resolution, so such a timestamp could not round-trip.

    Examples
    --------
    >>> key_for(datetime(2026, 8, 1, 6, 0))
    '2026-08-01-06:00'
    """
    naive = _as_naive_utc(t)
    if naive.second or naive.microsecond:
        raise ValueError(
            f"cannot render {naive.isoformat()} as a partition key: the key "
            f"format {PARTITION_KEY_FORMAT!r} has minute resolution, but the "
            "timestamp has a non-zero second/microsecond component"
        )
    return naive.strftime(PARTITION_KEY_FORMAT)


def time_for(key: str) -> datetime:
    """Parse a partition key back into a timestamp.

    Parameters
    ----------
    key : str
        A key in ``%Y-%m-%d-%H:%M`` form.

    Returns
    -------
    datetime
        Naive UTC timestamp.  ``key_for(time_for(k)) == k`` for every
        well-formed key.

    Raises
    ------
    TypeError
        If ``key`` is not a string.
    ValueError
        If ``key`` does not match the expected format; the message shows
        both the format and an example.
    """
    if not isinstance(key, str):
        raise TypeError(
            f"partition key must be a str, got {type(key).__name__!r}"
        )
    try:
        return datetime.strptime(key, PARTITION_KEY_FORMAT)
    except ValueError as exc:
        raise ValueError(
            f"malformed partition key {key!r}: expected format "
            f"{PARTITION_KEY_FORMAT!r}, e.g. '2026-08-01-06:00' ({exc})"
        ) from exc


# ---------------------------------------------------------------------------
# Grid arithmetic
# ---------------------------------------------------------------------------

def align_to_cadence(t: datetime, cadence_minutes: int) -> datetime:
    """Floor a timestamp onto the cadence grid.

    The grid is anchored at 00:00 UTC each day, which is well defined
    precisely because :func:`validate_cadence` requires the cadence to
    divide the day.

    Parameters
    ----------
    t : datetime
        Timestamp to floor.
    cadence_minutes : int
        Cadence defining the grid.

    Returns
    -------
    datetime
        Naive UTC timestamp, the latest grid point at or before ``t``.
        A timestamp already on the grid is returned unchanged.

    Raises
    ------
    ValueError
        If ``cadence_minutes`` is invalid.
    """
    validate_cadence(cadence_minutes)
    naive = _as_naive_utc(t)
    minutes = naive.hour * 60 + naive.minute
    floored = (minutes // cadence_minutes) * cadence_minutes
    midnight = naive.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + timedelta(minutes=floored)


def window_for(key: str, cadence_minutes: int) -> tuple[datetime, datetime]:
    """Return the half-open interval a partition covers.

    Parameters
    ----------
    key : str
        Partition key.
    cadence_minutes : int
        Cadence the key belongs to.

    Returns
    -------
    tuple of datetime
        ``(start, end)`` as naive UTC, where ``start`` is the key's own
        timestamp and ``end`` is one cadence later.  The interval is
        half-open: ``end`` belongs to the next partition.

    Raises
    ------
    ValueError
        If the key is malformed, the cadence invalid, or the key's
        timestamp does not sit on the cadence grid.
    """
    validate_cadence(cadence_minutes)
    start = validate_on_cadence(time_for(key), cadence_minutes, argname="key")
    return start, start + timedelta(minutes=cadence_minutes)


def keys_between(
    start: datetime, end: datetime, cadence_minutes: int
) -> list[str]:
    """List the partition keys covering ``[start, end]``.

    Parameters
    ----------
    start, end : datetime
        Inclusive bounds.  Both must sit on the cadence grid; a bound
        that does not is a sign of an upstream bug, so it is rejected
        rather than silently snapped.
    cadence_minutes : int
        Cadence defining the grid.

    Returns
    -------
    list of str
        Keys in chronological (equivalently, lexicographic) order,
        including both endpoints.  ``start == end`` yields one key.

    Raises
    ------
    ValueError
        If ``end`` precedes ``start``, the cadence is invalid, or either
        bound is off the grid.
    """
    validate_cadence(cadence_minutes)
    start_t = validate_on_cadence(start, cadence_minutes, argname="start")
    end_t = validate_on_cadence(end, cadence_minutes, argname="end")
    if end_t < start_t:
        raise ValueError(
            f"end ({end_t.isoformat()}) is before start "
            f"({start_t.isoformat()})"
        )
    step = timedelta(minutes=cadence_minutes)
    keys: list[str] = []
    current = start_t
    while current <= end_t:
        keys.append(key_for(current))
        current += step
    return keys


# ---------------------------------------------------------------------------
# Dagster partitions definition
# ---------------------------------------------------------------------------

def cron_for_cadence(cadence_minutes: int) -> str:
    """Build the cron schedule expressing a cadence.

    Parameters
    ----------
    cadence_minutes : int
        Cadence defining the grid.

    Returns
    -------
    str
        A five-field cron expression firing exactly on the grid, e.g.
        ``"0 * * * *"`` for 60 minutes and ``"0 0,6,12,18 * * *"`` for
        360 minutes.

    Raises
    ------
    ValueError
        If the cadence is invalid, or if it divides the day but is not
        expressible as a repeating cron schedule.  ``*/N`` restarts every
        hour, so a sub-hourly cadence must divide 60 (10 and 15 are fine,
        16 and 45 are not); an hourly-or-longer cadence must be a whole
        number of hours.
    """
    validate_cadence(cadence_minutes)
    if cadence_minutes < 60:
        if 60 % cadence_minutes != 0:
            raise ValueError(
                f"cadence_minutes={cadence_minutes} divides the day but not "
                "the hour, so it has no cron expression ('*/N' restarts each "
                "hour). Use a divisor of 60 below the hour "
                "(1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30)."
            )
        return f"*/{cadence_minutes} * * * *"
    if cadence_minutes % 60 != 0:
        raise ValueError(
            f"cadence_minutes={cadence_minutes} is above an hour but not a "
            "whole number of hours, so it has no cron expression. Use a "
            "multiple of 60 that divides 1440 (60, 120, 180, 240, 360, 480, "
            "720, 1440)."
        )
    step_hours = cadence_minutes // 60
    if step_hours == 1:
        return "0 * * * *"
    if step_hours == 24:
        return "0 0 * * *"
    hours = ",".join(str(h) for h in range(0, 24, step_hours))
    return f"0 {hours} * * *"


def build_partitions_def(
    start: datetime, cadence_minutes: int
) -> "TimeWindowPartitionsDefinition":
    """Build the Dagster partitions definition for the run cadence.

    Dagster is imported lazily here so that the key, grid and cron
    helpers in this module stay usable — and unit-testable — in a
    process that has no Dagster installed.

    Parameters
    ----------
    start : datetime
        First partition's timestamp.  Must sit on the cadence grid so
        that keys Dagster generates match :func:`key_for` exactly.
    cadence_minutes : int
        Spacing between partitions, in minutes.  60 and 360 are the
        operational cadences; any divisor of 60, or any whole number of
        hours dividing 1440, works.

    Returns
    -------
    dagster.TimeWindowPartitionsDefinition
        Partitions whose keys are formatted with
        :data:`PARTITION_KEY_FORMAT`, so they are interchangeable with
        the output of :func:`key_for`.

    Raises
    ------
    ValueError
        If the cadence is invalid or ``start`` is off the grid.

    Examples
    --------
    >>> pd = build_partitions_def(datetime(2026, 8, 1), 360)
    >>> pd.get_partition_keys(
    ...     current_time=datetime(2026, 8, 1, 18, 0)
    ... )[:2]
    ['2026-08-01-00:00', '2026-08-01-06:00']
    """
    from dagster import TimeWindowPartitionsDefinition

    cron = cron_for_cadence(cadence_minutes)
    start_t = validate_on_cadence(start, cadence_minutes, argname="start")
    logger.debug(
        "building partitions: start=%s cadence=%d cron=%r",
        start_t.isoformat(),
        cadence_minutes,
        cron,
    )
    return TimeWindowPartitionsDefinition(
        start=start_t,
        cron_schedule=cron,
        fmt=PARTITION_KEY_FORMAT,
        timezone="UTC",
    )


#: First partition of the operational time space.  Chosen to predate any
#: satellite archive this pipeline reads, so a backfill can reach as far
#: back as the stores allow without redefining the partition set (which
#: would invalidate every existing partition key).
DEFAULT_START = datetime(2024, 1, 1, 0, 0)


def __getattr__(name: str):
    """Build :data:`OPERATIONAL_PARTITIONS` on first access.

    Evaluating it eagerly would import dagster at module import, which
    this module deliberately avoids so ``operational.core`` stays usable
    -- and testable -- without the orchestrator installed.  PEP 562 lets
    ``from ... import OPERATIONAL_PARTITIONS`` work regardless.
    """
    if name == "OPERATIONAL_PARTITIONS":
        from operational.config import OperationalConfig

        value = build_partitions_def(
            DEFAULT_START, OperationalConfig.from_env().cadence_minutes)
        globals()[name] = value          # cache; __getattr__ won't run again
        return value
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")
