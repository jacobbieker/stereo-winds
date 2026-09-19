"""Local L1b download cache: where it lives, and how it is kept in check.

Every reader that falls back to public object storage caches what it
downloads so consecutive timestamps can reuse a scan — at a 10 minute
step a slot is fetched as ``t+dt``, then reused as ``t0``, then as
``t-dt``.  Without pruning that cache grows without bound: a full-disk
ABI band is ~17 MB and a Himawari band-slot ~40 MB across ten segments,
so a multi-thousand-timestamp run fills the disk.

Entries are therefore removed once the scan they hold is more than
``retention`` older than the scene being read.  Pruning keys off the
*observation* time parsed from the cache layout, not file mtime, so it
is unaffected by when a file happened to be downloaded:

``<cache>/<satellite>/<YYYYmmdd_HHMM>/``
    Slot directories (AHI HSD segments, AMI netCDF)
``<cache>/<satellite>/OR_ABI-...-M6C14_G19_s20261821200205_...nc``
    Flat ABI files stamped ``_s<YYYYJJJHHMMSS>``

Set ``STEREO_WINDS_L1B_RETENTION_MIN`` to change the window, or to a
non-positive value to keep everything.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# How far back a cached scan stays useful.  One hour is generous against
# the +/-15 minute reach of a temporal triplet.
DEFAULT_RETENTION = dt.timedelta(hours=1)

_SLOT_DIR_FMT = "%Y%m%d_%H%M"
_ABI_STAMP_RE = re.compile(r"_s(\d{13})")


def default_cache_dir() -> Path:
    """Where downloaded L1b files are kept between runs."""
    env = os.environ.get("STEREO_WINDS_DATA_DIR")
    base = Path(env) if env else Path.home() / ".cache" / "stereo_winds"
    return base / "l1b"


def default_retention() -> dt.timedelta | None:
    """Retention window, or None when pruning is switched off."""
    raw = os.environ.get("STEREO_WINDS_L1B_RETENTION_MIN")
    if raw is None or not raw.strip():
        return DEFAULT_RETENTION
    try:
        minutes = float(raw)
    except ValueError:
        logger.warning(
            "STEREO_WINDS_L1B_RETENTION_MIN=%r is not a number — keeping the "
            "default %s cache retention", raw, DEFAULT_RETENTION)
        return DEFAULT_RETENTION
    return dt.timedelta(minutes=minutes) if minutes > 0 else None


def entry_time(path: Path) -> dt.datetime | None:
    """Observation time a cache entry holds, or None if it is not one.

    Anything unrecognised returns None and is therefore never pruned —
    that includes the ``satpy-scratch-*`` directories in use by a load
    running right now.
    """
    if path.is_dir():
        try:
            return dt.datetime.strptime(path.name, _SLOT_DIR_FMT)
        except ValueError:
            return None
    match = _ABI_STAMP_RE.search(path.name)
    if match is None:
        return None
    try:
        return dt.datetime.strptime(match.group(1), "%Y%j%H%M%S")
    except ValueError:
        return None


def _size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def prune_cache(root: Path, older_than: dt.datetime) -> tuple[int, int]:
    """Remove cache entries under ``root`` holding scans before ``older_than``.

    Only entries whose name parses as an observation time are considered,
    so partial downloads and in-flight scratch directories are left
    alone.  Returns ``(entries removed, bytes freed)``.
    """
    root = Path(root)
    if not root.is_dir():
        return 0, 0

    removed = 0
    freed = 0
    for entry in root.iterdir():
        stamp = entry_time(entry)
        if stamp is None or stamp >= older_than:
            continue
        try:
            size = _size_of(entry)
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        except OSError as exc:
            # Another process may be reading it; it will be pruned later.
            logger.debug("Could not prune %s: %s", entry, exc)
            continue
        removed += 1
        freed += size
    if removed:
        logger.info("Pruned %d cached entr%s older than %s from %s (%.1f MB)",
                    removed, "y" if removed == 1 else "ies",
                    older_than.isoformat(sep=" "), root, freed / 1e6)
    return removed, freed

# ---------------------------------------------------------------------------
# Concurrency and in-memory sizing
# ---------------------------------------------------------------------------

# Object-store round trips dominate these downloads, so more workers than
# cores is right; the ceiling keeps us from hammering the bucket.
DEFAULT_DOWNLOAD_WORKERS = 8

# Headroom left for everything that is not the scene cache: the model
# inputs for one satellite (~3.7 GB at full disk), the mosaic
# accumulator, the model itself, CUDA's host-side allocations and
# allocator fragmentation.  Measured at ~14 GB for a six-satellite ring.
DEFAULT_MEMORY_RESERVE = 16 * 2**30

# Hard ceiling as a share of *total* RAM.  MemAvailable counts
# reclaimable page cache, so sizing a cache to all of it drives the
# process to fill RAM and leaves nothing for the page cache the reads
# themselves need — which is how this ends in an OOM kill rather than an
# eviction.
MAX_CACHE_FRACTION = 0.35

# If available memory ever falls below this, the cache gives memory back
# rather than waiting to be killed.
MEMORY_PRESSURE_FLOOR = 6 * 2**30


def download_workers() -> int:
    """Concurrent object-store downloads per scene."""
    raw = os.environ.get("STEREO_WINDS_DOWNLOAD_WORKERS")
    if raw and raw.strip():
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("STEREO_WINDS_DOWNLOAD_WORKERS=%r is not an "
                           "integer — using %d", raw, DEFAULT_DOWNLOAD_WORKERS)
    return DEFAULT_DOWNLOAD_WORKERS


def total_memory_bytes() -> int | None:
    """Total physical RAM, or None if unknown."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:  # pragma: no cover - non-Linux fallback
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def available_memory_bytes() -> int | None:
    """Memory that can be allocated without swapping, or None if unknown.

    ``MemAvailable`` accounts for reclaimable page cache, which is what
    actually matters here; ``MemFree`` would badly understate it on a box
    that has been reading imagery.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:  # pragma: no cover - non-Linux fallback
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def default_scene_cache_bytes(reserve: int = DEFAULT_MEMORY_RESERVE) -> int:
    """How much RAM the decoded-scene cache may hold.

    Just under what is free, less ``reserve`` for the working set the
    pipeline needs alongside it.  Override with
    ``STEREO_WINDS_SCENE_CACHE_GB``.
    """
    raw = os.environ.get("STEREO_WINDS_SCENE_CACHE_GB")
    if raw and raw.strip():
        try:
            return max(0, int(float(raw) * 2**30))
        except ValueError:
            logger.warning("STEREO_WINDS_SCENE_CACHE_GB=%r is not a number "
                           "— sizing from free memory instead", raw)
    available = available_memory_bytes()
    if available is None:
        return 4 * 2**30
    limit = available - reserve
    total = total_memory_bytes()
    if total is not None:
        # Never take more than a share of the box, however much happens to
        # look free at the moment we are asked.
        limit = min(limit, int(total * MAX_CACHE_FRACTION))
    # Floor at 1 GB: below that the cache cannot even hold one satellite's
    # triplet and prefetching would thrash.
    return max(2**30, limit)
