"""Detection of the timestamps each satellite can actually deliver.

This is the "has new data arrived?" half of the operational service.  A
single-satellite retrieval consumes three frames — ``t - dt``, ``t`` and
``t + dt``, where ``dt`` is that satellite's full-disk repeat cycle — so a
timestamp only counts as available once all three frames are on hand.

Instruments do not share a slot convention (ABI full disk stamps at
``HH:00:00``, AMI at ``HH:09:35``), and the readers snap to the nearest
scan, so every frame test is a tolerance test rather than an equality
test.

Scan-time lookup itself is *not* reimplemented here: it comes from the
global-ring script via :mod:`operational.adapters.ring`
(``satellite_available_times`` / ``availability_band`` /
``scan_interval``), which already knows about icechunk stores, public-S3
fallbacks and per-satellite band coverage.

All datetimes are naive UTC; tz-aware inputs are converted on the way in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Default GOES product passed through to the upstream S3 listing.
DEFAULT_PRODUCT = "ABI-L1b-RadF"

#: Anchor for the operational cadence grid, so slots are reproducible
#: across runs and across window boundaries.
_EPOCH = datetime(1970, 1, 1)

_EMPTY_TIMES = np.array([], dtype="datetime64[ns]")


def _naive_utc(t: datetime) -> datetime:
    """Return ``t`` as a naive UTC datetime."""
    if t.tzinfo is None:
        return t
    return t.astimezone(timezone.utc).replace(tzinfo=None)


def _to_datetime64(times: Iterable[datetime]) -> np.ndarray:
    """Convert datetimes to a ``datetime64[ns]`` array."""
    values = [np.datetime64(_naive_utc(t), "ns") for t in times]
    if not values:
        return _EMPTY_TIMES
    return np.array(values, dtype="datetime64[ns]")


def _to_datetimes(times: np.ndarray) -> list[datetime]:
    """Convert a ``datetime64`` array to a list of naive datetimes."""
    if times.size == 0:
        return []
    return list(times.astype("datetime64[us]").astype(datetime))


def _nearest_scan(
    scan_times: np.ndarray,
    targets: np.ndarray,
    tol: np.timedelta64,
) -> np.ndarray:
    """Index of the nearest scan to each target, or ``-1`` if none is close.

    A target equidistant between two different scans that are both in
    tolerance is *ambiguous* and returns ``-1``: this module and the
    readers each pick a nearest scan independently, so a tie is a frame
    whose identity cannot be predicted here — and two frames breaking the
    tie in opposite directions load the same scan twice.
    """
    if scan_times.size == 0 or targets.size == 0:
        return np.full(targets.shape, -1, dtype=np.int64)
    idx = np.searchsorted(scan_times, targets)
    # The nearest scan lies on one side or the other of the insertion point.
    left = np.clip(idx - 1, 0, scan_times.size - 1)
    right = np.clip(idx, 0, scan_times.size - 1)
    d_left = np.abs(scan_times[left] - targets)
    d_right = np.abs(scan_times[right] - targets)
    best = np.where(d_left <= d_right, left, right)
    best_dist = np.minimum(d_left, d_right)
    # A tie between two *distinct* scans, both in tolerance, is ambiguous;
    # when the clips collapse both sides onto one scan it is not.
    ambiguous = (left != right) & (d_left == d_right) & (d_right <= tol)
    return np.where((best_dist <= tol) & ~ambiguous, best, -1).astype(np.int64)


# ``eq=False`` keeps identity semantics: the dataclass holds a numpy
# array, whose element-wise ``==`` would make a generated ``__eq__``
# raise and a generated ``__hash__`` unusable.
@dataclass(frozen=True, eq=False)
class SatelliteAvailability:
    """What one satellite can deliver over a window."""

    sat_id: str
    band: str | None
    scan_times: np.ndarray
    scan_interval_minutes: int
    tolerance_minutes: float

    def __post_init__(self) -> None:
        """Normalise ``scan_times`` to a sorted ``datetime64[ns]`` array."""
        times = np.asarray(self.scan_times, dtype="datetime64[ns]").ravel()
        if times.size > 1 and not np.all(np.diff(times) >= np.timedelta64(0, "ns")):
            times = np.sort(times)
        object.__setattr__(self, "scan_times", times)

    @property
    def has_band(self) -> bool:
        """True when the satellite carries one of the requested bands."""
        return self.band is not None

    @property
    def dt(self) -> timedelta:
        """Triplet spacing as a :class:`~datetime.timedelta`."""
        return timedelta(minutes=self.scan_interval_minutes)

    def deliverable(self, candidates: Sequence[datetime]) -> list[datetime]:
        """Subset of ``candidates`` with a full ``(t-dt, t, t+dt)`` triplet.

        The three frames must match three *distinct*, unambiguously
        nearest scans.  Tolerance is typically half the repeat cycle, so
        without those checks a sparse or half-offset schedule can satisfy
        two frames with the same scan — the readers then snap both to it
        and the student sees a duplicated frame, i.e. near-zero
        displacement, rather than a failure.
        """
        if not candidates or not self.has_band:
            return []
        targets = _to_datetime64(candidates)
        tol = np.timedelta64(int(round(self.tolerance_minutes * 60e9)), "ns")
        step = np.timedelta64(self.scan_interval_minutes, "m").astype("timedelta64[ns]")
        matched = np.stack(
            [
                _nearest_scan(self.scan_times, frame, tol)
                for frame in (targets - step, targets, targets + step)
            ]
        )  # (3, n_candidates)
        ok = np.all(matched >= 0, axis=0)
        # Three frames, three different scans.
        ok &= (matched[0] != matched[1]) & (matched[1] != matched[2]) & (matched[0] != matched[2])
        return [t for t, keep in zip(candidates, ok.tolist()) if keep]

    def can_deliver(self, timestamp: datetime) -> bool:
        """True when a full triplet exists around ``timestamp``."""
        return bool(self.deliverable([timestamp]))


def probe_satellite(
    sat_id: str,
    start: datetime,
    end: datetime,
    flow_bands: Sequence[str],
    rad_bands: Sequence[str],
    tolerance_minutes: float = 5.0,
    product: str = DEFAULT_PRODUCT,
    include_s3_fallback: bool = True,
) -> SatelliteAvailability:
    """Look up one satellite's scan times around a window of candidates.

    The window is widened by ``dt + tolerance`` on both sides so the
    neighbour frames of the first and last candidate are covered.
    """
    from operational.adapters import ring

    start = _naive_utc(start)
    end = _naive_utc(end)
    dt_minutes = int(ring.scan_interval(sat_id))
    band = ring.availability_band(sat_id, list(flow_bands), list(rad_bands))
    if band is None:
        logger.warning(
            "%s carries none of the requested bands (flow=%s, rad=%s); "
            "treating as nothing available",
            sat_id,
            list(flow_bands),
            list(rad_bands),
        )
        return SatelliteAvailability(
            sat_id=sat_id,
            band=None,
            scan_times=_EMPTY_TIMES,
            scan_interval_minutes=dt_minutes,
            tolerance_minutes=tolerance_minutes,
        )

    pad = timedelta(minutes=dt_minutes + tolerance_minutes)
    times = ring.satellite_available_times(
        sat_id,
        band,
        start - pad,
        end + pad,
        product,
        include_s3_fallback,
    )
    logger.debug("%s: scans on band %s over %s .. %s", sat_id, band, start - pad, end + pad)
    return SatelliteAvailability(
        sat_id=sat_id,
        band=band,
        scan_times=times,
        scan_interval_minutes=dt_minutes,
        tolerance_minutes=tolerance_minutes,
    )


def available_timestamps(
    sat_id: str,
    start: datetime,
    end: datetime,
    flow_bands: Sequence[str],
    rad_bands: Sequence[str],
    tolerance_minutes: float = 5.0,
    product: str = DEFAULT_PRODUCT,
    include_s3_fallback: bool = True,
) -> list[datetime]:
    """Timestamps ``sat_id`` can deliver a retrieval for within a window.

    Candidates are the satellite's own scan times inside ``[start, end]``;
    each survives only if the satellite also has the ``t - dt`` and
    ``t + dt`` frames within ``tolerance_minutes``.
    """
    start = _naive_utc(start)
    end = _naive_utc(end)
    if end < start:
        return []
    avail = probe_satellite(
        sat_id,
        start,
        end,
        flow_bands,
        rad_bands,
        tolerance_minutes,
        product,
        include_s3_fallback,
    )
    if not avail.has_band:
        return []
    lo = np.datetime64(start, "ns")
    hi = np.datetime64(end, "ns")
    in_window = avail.scan_times[(avail.scan_times >= lo) & (avail.scan_times <= hi)]
    return avail.deliverable(_to_datetimes(in_window))


def cadence_grid(
    start: datetime,
    end: datetime,
    cadence_minutes: int,
) -> list[datetime]:
    """Grid slots within ``[start, end]``, anchored on the Unix epoch.

    Anchoring means a 60-minute cadence always lands on the hour,
    independent of where the window happens to begin.
    """
    if cadence_minutes <= 0:
        raise ValueError(f"cadence_minutes must be positive, got {cadence_minutes}")
    start = _naive_utc(start)
    end = _naive_utc(end)
    if end < start:
        return []
    step = timedelta(minutes=cadence_minutes)
    n_first = -((_EPOCH - start) // step)  # ceil division towards +inf
    first = _EPOCH + step * n_first
    slots: list[datetime] = []
    t = first
    while t <= end:
        slots.append(t)
        t += step
    return slots


def new_timestamps(
    sat_id: str,
    since: datetime | None,
    until: datetime,
    flow_bands: Sequence[str],
    rad_bands: Sequence[str],
    cadence_minutes: int = 60,
    tolerance_minutes: float = 5.0,
    lookback_hours: float = 24.0,
    product: str = DEFAULT_PRODUCT,
    include_s3_fallback: bool = True,
) -> list[datetime]:
    """Cadence slots strictly after ``since`` that ``sat_id`` can deliver.

    This is the sensor poll: hand it the last timestamp already
    processed and it returns the work that has become possible since.
    """
    until = _naive_utc(until)
    horizon = until - timedelta(hours=lookback_hours)
    if since is None:
        window_start = horizon
    else:
        since = _naive_utc(since)
        if since >= until:
            return []
        window_start = max(since, horizon)
        if window_start > since:
            logger.warning(
                "%s: cursor %s is older than the %.1f h catch-up limit; " "skipping ahead to %s",
                sat_id,
                since,
                lookback_hours,
                window_start,
            )
    candidates = [
        t for t in cadence_grid(window_start, until, cadence_minutes) if since is None or t > since
    ]
    if not candidates:
        return []
    avail = probe_satellite(
        sat_id,
        candidates[0],
        candidates[-1],
        flow_bands,
        rad_bands,
        tolerance_minutes,
        product,
        include_s3_fallback,
    )
    ready = avail.deliverable(candidates)
    logger.info(
        "%s: %d of %d cadence slot(s) ready after %s", sat_id, len(ready), len(candidates), since
    )
    return ready


def ready_satellites(
    timestamp: datetime,
    satellites: Sequence[str],
    flow_bands: Sequence[str],
    rad_bands: Sequence[str],
    tolerance_minutes: float = 5.0,
    product: str = DEFAULT_PRODUCT,
    include_s3_fallback: bool = True,
) -> dict[str, bool]:
    """Which satellites can deliver a retrieval at ``timestamp``."""
    timestamp = _naive_utc(timestamp)
    ready: dict[str, bool] = {}
    for sat_id in satellites:
        avail = probe_satellite(
            sat_id,
            timestamp,
            timestamp,
            flow_bands,
            rad_bands,
            tolerance_minutes,
            product,
            include_s3_fallback,
        )
        ready[sat_id] = avail.can_deliver(timestamp)
    logger.info(
        "Ready at %s: %s",
        timestamp,
        ", ".join(f"{k}={'y' if v else 'n'}" for k, v in ready.items()),
    )
    return ready
