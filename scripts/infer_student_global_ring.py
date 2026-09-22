r"""Global geostationary ring student AMV inference.

For a given time (or a range of times), loads imagery from every available
geostationary satellite (GOES-18, GOES-19, Himawari-9, GK-2A, MTG-I1 and
MSG/SEVIRI at 45.5°E) via icechunk — falling back to public-S3 L1b where
a store does not cover the time — runs the single-satellite student model
on each, and produces:

1. Per-satellite NetCDF files with full-disk AMVs
2. A combined global NetCDF on a regular lat/lon grid, where overlapping
   regions use the AMV from whichever satellite has the smallest zenith
   angle (closest to the sub-satellite point)

The mosaic records its winning satellite per cell as
``source_satellite_index`` — ``int8`` codes into the ``satellites``
attribute, with ``-1`` where nothing contributed.  Use
``decode_source_satellite`` to turn a slice back into names; the full
string array is 9 GB at 2 km, which is why the codes are what gets
stored.

Memory scales with the mosaic grid, not the number of satellites:
full disks are gridded as they finish and released immediately.  Peak
RSS is logged after every satellite, so an approaching limit is
visible before the kernel intervenes.  If the accumulator itself is too
large for the machine, raise ``--resolution-m``.

Outputs use a deterministic layout under ``--output-dir``::

    <output-dir>/<YYYYMMDD>/student_amv_<sat_id>_<YYYYMMDDTHHMM>.nc
    <output-dir>/<YYYYMMDD>/student_amv_global_<YYYYMMDDTHHMM>.nc

so a run over a time range can be re-run, resumed (``--skip-existing``)
and globbed without ambiguity.

With ``--icechunk-store`` that layout is not used: the store holds the
mosaics, and the per-satellite disks become intermediates.  They are
written to a per-timestamp scratch folder under ``--temp-dir``
(``output/`` by default)::

    <temp-dir>/ring_scratch_<YYYYMMDDTHHMM>_<random>/<YYYYMMDD>/*.nc

which is removed once that timestamp's mosaic has been committed.  Six
full disks are ~1.4 GB per timestamp, so a month-long run would
otherwise leave hundreds of gigabytes behind for files nothing reads
again.  ``--keep-temp`` retains a folder for inspection, and
``--keep-netcdf`` restores the old behaviour of persisting both sets of
NetCDF files under ``--output-dir`` alongside the store.

Satellites are not all on the same schedule: SEVIRI repeats every 15
minutes where the rest of the ring scans every 10, so temporal pairs use
each satellite's own cadence and the resulting flow is rescaled to the
interval the student was trained on.

``--require-all-satellites`` lists each satellite's scan times up front
(icechunk time coordinates, or an S3 listing for GOES) and processes
only the timestamps where every satellite can supply a full
t-10min/t/t+10min triplet, so the mosaic has the same contributors at
every output time.

With ``--icechunk-store`` the global mosaic is written to an icechunk
store — ``s3://bucket/prefix`` or a local directory — as one commit per
timestamp, and ``--skip-existing`` then resumes from the timestamps that
store already holds.  Pass ``--no-netcdf`` to skip the per-satellite
scratch files too, keeping the whole timestamp in memory.

Navigation always uses the projection metadata carried by the scenes
actually loaded — sub-satellite longitude, perspective height and grid
scale/offset — never a hardcoded nominal value, because geostationary
satellites drift within their station-keeping box and are periodically
relocated.  ``SATELLITE_CONFIGS`` serves only as a cross-check, and a
disagreement beyond ``SUB_LON_TOL_DEG`` is logged.

Usage::

    # single time
    pixi run python scripts/infer_student_global_ring.py \
        --time "2025-03-10T12:00" \
        --student-ckpt checkpoints/student.abi.mb-v3.ep21.ckpt \
        --raft-ckpt checkpoints/windflow.raft.sonde-tuned.ckpt \
        --output-dir output/global_ring \
        --device cuda

    # time range appended to an S3 icechunk store, resuming what it holds;
    # per-satellite disks go to output/ and are deleted after each commit
    pixi run python scripts/infer_student_global_ring.py \
        --time "2026-08-01T00:00" --end-time "2026-08-30T00:00" \
        --step-minutes 10 --skip-existing \
        --icechunk-store s3://my-bucket/student-amv.icechunk \
        --temp-dir output \
        --student-ckpt checkpoints/student.abi.mb-v3.ep21.ckpt \
        --raft-ckpt checkpoints/windflow.raft.sonde-tuned.ckpt \
        --device cuda

    # time range, every 30 minutes, resuming a previous run
    pixi run python scripts/infer_student_global_ring.py --time "2026-08-01T00:00" --end-time "2026-08-02T00:00" --step-minutes 10 --student-ckpt checkpoints/student.abi.mb-v3.ep21.ckpt --raft-ckpt checkpoints/windflow.raft.sonde-tuned.ckpt --output-dir output/global_ring --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import resource
import shutil
import tempfile
import threading
import sys
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
import xarray as xr

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from stereo_winds.config import SATELLITE_CONFIGS, SatelliteConfig
from stereo_winds.readers._satpy_s3 import SceneNotInStore
from stereo_winds.icechunk_output import (
    icechunk_existing_times,
    icechunk_storage,
    open_icechunk_repo,
    write_mosaic_to_icechunk,
)
from stereo_winds.readers._cache import (
    DEFAULT_DOWNLOAD_WORKERS,
    DEFAULT_MEMORY_RESERVE,
    MEMORY_PRESSURE_FLOOR,
    available_memory_bytes,
    default_scene_cache_bytes,
)
from stereo_winds.disparity import StereoDisparity
from stereo_winds.navigation import (
    compute_grid_latlon,
    compute_grid_zenith,
    compute_pixel_scale,
    grid_cache_budget,
)
from stereo_winds.student_dataset import (
    DEFAULT_FLOW_BANDS,
    DEFAULT_RAD_BANDS,
    FLOW_SCALE,
    PIXEL_SCALE_NORM,
    ZENITH_NORM,
)
from stereo_winds.student_zeus_model import StudentWindsModel

logger = logging.getLogger(__name__)

DT_MINUTES = 10
# Sub-satellite longitude agreement tolerance (deg).  Station-keeping
# boxes are typically +/-0.1 deg, so anything larger is worth flagging.
SUB_LON_TOL_DEG = 0.05
# A retrieval is flagged degraded once this share of the requested bands
# had to be zero-filled.  Some absences are normal — SEVIRI has no 1.4 or
# 2.2 µm channel, FCI none at 6.9 or 11.2 µm — so a handful of missing
# bands is expected; losing a quarter of them is not.
DEGRADED_BAND_FRACTION = 0.25
OUTPUT_VARS = [
    "u_wind", "v_wind", "cloud_top_height",
    "quality_flag", "sigma_u", "sigma_v", "sigma_h",
]

# Satellites to process (in longitude order, west to east).
# Each entry: (sat_id, loader_type, flow_bands_key, rad_bands_key)
# loader_type determines which data_loading function to call.
RING_SATELLITES = [
    "goes18",     # 137°W
    "goes19",     # 75°W
    "mtg-i1",     # 0°E
    "msg-iodc",   # 45.5°E
    "gk2a",       # 128.2°E
    "himawari9",  # 140.7°E
]

# Full-disk repeat cycle per satellite.  Everything in the ring scans
# every 10 minutes except MSG/SEVIRI, which takes 15.
SCAN_INTERVAL_MINUTES: dict[str, int] = {"msg-iodc": 15}


def scan_interval(sat_id: str) -> int:
    """Minutes between consecutive full disks for this satellite."""
    return SCAN_INTERVAL_MINUTES.get(sat_id, DT_MINUTES)


# ---------------------------------------------------------------------------
# Memory reporting
# ---------------------------------------------------------------------------

def peak_rss_gb() -> float:
    """Peak resident set size of this process, in GiB."""
    # ru_maxrss is kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def log_peak_rss(tag: str) -> None:
    """Log the high-water mark, so an OOM kill can be anticipated."""
    logger.info("%s: peak RSS %.2f GB", tag, peak_rss_gb())


# ---------------------------------------------------------------------------
# Deterministic output naming
# ---------------------------------------------------------------------------

TIME_TAG_FMT = "%Y%m%dT%H%M"
DAY_DIR_FMT = "%Y%m%d"


def time_tag(t: datetime) -> str:
    """Canonical timestamp tag used in every output filename."""
    return t.strftime(TIME_TAG_FMT)


def day_dir(out_dir: Path, t: datetime) -> Path:
    """Per-day subdirectory holding all files for timestamps on that day."""
    return Path(out_dir) / t.strftime(DAY_DIR_FMT)


def sat_nc_path(out_dir: Path, sat_id: str, t: datetime) -> Path:
    """Path of the per-satellite full-disk AMV file for ``t``."""
    return day_dir(out_dir, t) / f"student_amv_{sat_id}_{time_tag(t)}.nc"


def global_nc_path(out_dir: Path, t: datetime) -> Path:
    """Path of the merged global mosaic file for ``t``."""
    return day_dir(out_dir, t) / f"student_amv_global_{time_tag(t)}.nc"


#: Prefix of the per-timestamp scratch folders, so a folder left behind
#: by a killed run is recognisable (and safe to delete) afterwards.
SCRATCH_PREFIX = "ring_scratch_"


def make_scratch_dir(temp_root: Path, t: datetime) -> Path:
    """Create this timestamp's scratch folder under ``temp_root``.

    Per-satellite mosaics written here are intermediates: they feed the
    global mosaic and are removed once it has been committed.  The name
    carries the timestamp so a folder surviving a crash says which one
    it belongs to, and ``mkdtemp`` keeps concurrent runs apart.
    """
    temp_root = Path(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(
        prefix=f"{SCRATCH_PREFIX}{time_tag(t)}_", dir=temp_root))


def remove_scratch_dir(scratch: Path | None) -> None:
    """Delete a scratch folder, reporting rather than raising on failure.

    A scratch folder that cannot be removed is worth knowing about — it
    is how a long run fills its disk — but it must not lose a mosaic
    that has already been committed.
    """
    if scratch is None or not Path(scratch).exists():
        return
    try:
        freed = sum(p.stat().st_size for p in Path(scratch).rglob("*")
                    if p.is_file())
    except OSError:
        freed = 0
    try:
        shutil.rmtree(scratch)
    except OSError:
        logger.exception("Could not remove the scratch folder %s — it will "
                         "keep using disk until removed by hand", scratch)
        return
    logger.info("Removed scratch folder %s (%.1f GB)", scratch, freed / 2**30)


def time_steps(
    start: datetime, end: datetime | None, step_minutes: int,
) -> list[datetime]:
    """Inclusive list of timestamps from ``start`` to ``end`` every ``step``.

    An ``end`` of None (or equal to ``start``) yields a single timestamp.
    """
    if end is None or end == start:
        return [start]
    if end < start:
        raise ValueError(f"--end-time ({end}) is before --time ({start})")
    if step_minutes <= 0:
        raise ValueError(f"--step-minutes must be positive, got {step_minutes}")
    step = timedelta(minutes=step_minutes)
    out: list[datetime] = []
    t = start
    while t <= end:
        out.append(t)
        t += step
    return out


# ---------------------------------------------------------------------------
# Data loading: 3-frame radiance cubes from icechunk
# ---------------------------------------------------------------------------

def _load_scene(
    sat_id: str, band: str, t: datetime,
) -> tuple[np.ndarray, SatelliteConfig]:
    """Load one 2D radiance scene plus the config the loader read from it.

    The returned ``SatelliteConfig`` carries the projection parameters of
    *this* scene — sub-satellite longitude, perspective height and grid
    scale/offset as recorded in the file or icechunk store — and is what
    navigation must use.  Geostationary satellites drift within their
    station-keeping box and are periodically relocated, so the nominal
    values in ``SATELLITE_CONFIGS`` are only a sanity-check reference.
    """
    if "goes" in sat_id:
        from stereo_winds.data_loading import load_goes_scene
        return load_goes_scene(t, band, sat_id)
    elif "himawari" in sat_id:
        from stereo_winds.data_loading import load_himawari_scene
        return load_himawari_scene(t, band, sat_id)
    elif "gk2a" in sat_id:
        from stereo_winds.data_loading import load_gk2a_scene
        return load_gk2a_scene(t, band, sat_id)
    elif "mtg" in sat_id:
        from stereo_winds.data_loading import load_fci_scene
        return load_fci_scene(t, band, sat_id)
    elif "msg" in sat_id:
        from stereo_winds.data_loading import load_msg_scene
        return load_msg_scene(t, band, sat_id)
    raise ValueError(f"Unknown satellite: {sat_id}")


def _grid_key(cfg: SatelliteConfig) -> tuple:
    """Grid identity used to check that frames share one fixed grid."""
    return (cfg.n_rows, cfg.n_cols,
            round(cfg.scale_x, 12), round(cfg.scale_y, 12),
            round(cfg.x_offset, 9), round(cfg.y_offset, 9))


def _check_same_grid(
    ref: SatelliteConfig, other: SatelliteConfig, what: str,
) -> None:
    """Warn when a frame does not sit on the same fixed grid as the reference.

    Optical flow between frames on different grids is meaningless, and a
    changed sub-satellite longitude mid-triplet means the satellite was
    manoeuvred between scans.
    """
    if abs(other.sub_lon_deg - ref.sub_lon_deg) > SUB_LON_TOL_DEG:
        logger.warning(
            "  %s: sub-satellite longitude changed %.4f° -> %.4f° between "
            "frames — the satellite moved mid-triplet",
            what, ref.sub_lon_deg, other.sub_lon_deg,
        )
    if _grid_key(other) != _grid_key(ref):
        logger.warning("  %s: fixed grid differs from the reference frame", what)


def _load_three_frames(
    sat_id: str, band: str, t0: datetime, dt_min: int | None = None,
    prefetcher: ScenePrefetcher | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SatelliteConfig, int]:
    """Load t-dt, t0, t+dt radiance frames plus the t0 scene's config.

    ``dt_min`` defaults to the satellite's own repeat cycle: asking MSG
    for frames 10 minutes apart would just return the same 15-minute
    scans, mislabelled.  The interval actually used is returned so the
    caller can scale the resulting flow.
    """
    if dt_min is None:
        dt_min = scan_interval(sat_id)
    delta = timedelta(minutes=dt_min)
    fetch = (prefetcher.get if prefetcher is not None
             else lambda sat, b, when: _load_scene(sat, b, when))
    a_m, cfg_m = fetch(sat_id, band, t0 - delta)
    a_0, cfg_0 = fetch(sat_id, band, t0)
    a_p, cfg_p = fetch(sat_id, band, t0 + delta)
    # t0 defines the retrieval grid; the neighbours must match it.
    _check_same_grid(cfg_0, cfg_m, f"{sat_id} {band} t-{dt_min}min")
    _check_same_grid(cfg_0, cfg_p, f"{sat_id} {band} t+{dt_min}min")
    return a_m, a_0, a_p, cfg_0, dt_min


# ---------------------------------------------------------------------------
# Scene prefetch: keep the GPU fed rather than waiting on object storage
# ---------------------------------------------------------------------------

SceneKey = tuple[str, str, datetime]


class SceneCache:
    """Bounded LRU of decoded scenes, keyed by (satellite, band, time).

    Decoding a full disk costs a download plus a bz2/zarr decode for
    ~120 MB of float32.  Consecutive timestamps genuinely reuse scans —
    at a 10 minute step a slot is read as ``t+dt``, then as ``t0``, then
    as ``t-dt`` — so holding the decoded array is worth real time.

    The bound is in bytes and is enforced on insert; entries are dropped
    least-recently-used first.  Eviction only drops the cache's
    reference, so a scene a caller still holds stays alive.
    """

    # How often to re-check free memory, in inserts.  Each check is one
    # small read of /proc/meminfo.
    _PRESSURE_CHECK_EVERY = 16

    # Never shrink below this: a retrieval needs its scenes in flight.
    _SHRINK_FLOOR = 2**30

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.limit = self.max_bytes
        self._entries: OrderedDict[SceneKey, tuple] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self._since_check = 0
        self.hits = 0
        self.misses = 0
        self.shrinks = 0

    def get(self, key: SceneKey):
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: SceneKey, value: tuple) -> None:
        data = value[0]
        size = int(getattr(data, "nbytes", 0))
        if self.max_bytes == 0 or size > self.max_bytes:
            return
        self._check_pressure()
        with self._lock:
            if key in self._entries:
                self._bytes -= int(getattr(self._entries[key][0], "nbytes", 0))
                del self._entries[key]
            self._entries[key] = value
            self._bytes += size
            while self._bytes > self.limit and self._entries:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= int(getattr(evicted[0], "nbytes", 0))

    def _check_pressure(self) -> None:
        """Lower the working limit when the box is running out of memory.

        The limit is chosen once from what was free at startup, which can
        be wrong later: other work starts, or the pipeline's own
        allocations grow.  Holding scenes until the kernel intervenes
        loses the whole run, so the cache gives memory back instead.
        """
        with self._lock:
            self._since_check += 1
            if self._since_check < self._PRESSURE_CHECK_EVERY:
                return
            self._since_check = 0
        available = available_memory_bytes()
        if available is None:
            return
        if available < MEMORY_PRESSURE_FLOOR:
            with self._lock:
                # Halve, but keep enough for a retrieval's scenes in
                # flight — or the whole budget, if it was already small.
                floor = min(self._SHRINK_FLOOR, self.max_bytes)
                new_limit = max(floor, self.limit // 2)
                if new_limit < self.limit:
                    self.limit = new_limit
                    self.shrinks += 1
                    logger.warning(
                        "Only %.1f GB free — shrinking the scene cache to "
                        "%.1f GB", available / 2**30, self.limit / 2**30)
        elif self.limit < self.max_bytes and available > 3 * MEMORY_PRESSURE_FLOOR:
            with self._lock:
                self.limit = min(self.max_bytes, self.limit * 2)

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._bytes

    def summary(self) -> str:
        total = self.hits + self.misses
        rate = 100.0 * self.hits / total if total else 0.0
        shrunk = "" if self.limit == self.max_bytes else (
            f", limit lowered to {self.limit / 2**30:.1f} GB "
            f"({self.shrinks}x under pressure)")
        return (f"scene cache {self.nbytes / 2**30:.1f}/"
                f"{self.max_bytes / 2**30:.1f} GB, "
                f"{self.hits}/{total} hits ({rate:.0f}%){shrunk}")

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0


# Cache size when consecutive timestamps cannot share a scene.  Enough to
# smooth a repeat within one retrieval, small enough to be irrelevant.
NO_REUSE_CACHE_BYTES = 2**30


def scene_cache_is_useful(step_minutes: int, sats: list[str]) -> bool:
    """Can consecutive timestamps share a scene?

    A timestamp reads ``{t-dt, t, t+dt}`` and the next reads that window
    shifted by the step, so they overlap only when the step is at most
    twice the scan interval.  At coarser steps every cached scene is one
    that will never be read again, and holding them fills memory for no
    benefit.
    """
    return any(step_minutes <= 2 * scan_interval(sat) for sat in sats)


class ScenePrefetcher:
    """Loads scenes on background threads so inference is not IO-bound.

    Reading a scene is nearly all waiting — object-store round trips,
    then a decode that releases the GIL — while the forward pass is
    GPU-bound, so the two overlap well.  Requests are deduplicated, and
    asking for a scene that is still in flight simply waits for it.
    """

    # Backstop on queued-but-uncollected work.  One satellite of
    # lookahead is ~20 scenes; beyond a few of those something has gone
    # wrong and the oldest are dropped rather than held forever.
    _MAX_INFLIGHT = 64

    def __init__(self, cache: SceneCache, max_workers: int = 4) -> None:
        self.cache = cache
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, max_workers), thread_name_prefix="scene")
        self._inflight: OrderedDict[SceneKey, Future] = OrderedDict()
        self._lock = threading.Lock()
        self.dropped = 0

    def _load(self, key: SceneKey) -> tuple:
        sat_id, band, t = key
        value = _load_scene(sat_id, band, t)
        self.cache.put(key, value)
        return value

    def submit(self, requests: list[SceneKey]) -> None:
        """Start loading these scenes in the background, skipping duplicates."""
        for key in requests:
            with self._lock:
                if key in self._inflight:
                    continue
                if self.cache.get(key) is not None:
                    continue
                self._inflight[key] = self._pool.submit(self._load, key)
                while len(self._inflight) > self._MAX_INFLIGHT:
                    _, stale = self._inflight.popitem(last=False)
                    stale.cancel()
                    self.dropped += 1

    def reset(self) -> int:
        """Drop everything queued but never collected; returns how many.

        A satellite can be skipped because its output already exists, or
        fail after its scenes were queued.  Those futures hold a decoded
        full disk each — ~120 MB — and nothing will ever read them, so
        holding them for the rest of the run is a leak.  Anything worth
        keeping is in the cache already.
        """
        with self._lock:
            futures = list(self._inflight.values())
            self._inflight.clear()
        for future in futures:
            future.cancel()
        self.dropped += len(futures)
        if futures:
            logger.debug("Dropped %d prefetched scene(s) nothing asked for",
                         len(futures))
        return len(futures)

    @property
    def inflight(self) -> int:
        with self._lock:
            return len(self._inflight)

    def get(self, sat_id: str, band: str, t: datetime) -> tuple:
        """Return a scene, waiting on a prefetch or loading it inline."""
        key = (sat_id, band, t)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        with self._lock:
            future = self._inflight.pop(key, None)
        if future is not None:
            return future.result()
        return self._load(key)

    def shutdown(self) -> None:
        with self._lock:
            futures = list(self._inflight.values())
            self._inflight.clear()
        for future in futures:
            future.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)


def _needs_inference(
    sat_id: str, t0: datetime, out_dir: Path,
    skip_existing: bool, write_netcdf: bool,
) -> bool:
    """False when this satellite's output already exists and will be reused.

    Prefetching for a satellite that is about to be skipped queues work
    nobody collects.
    """
    if not (skip_existing and write_netcdf):
        return True
    return not sat_nc_path(out_dir, sat_id, t0).exists()


def scene_requests(
    sat_id: str,
    t0: datetime,
    flow_bands: list[str],
    rad_bands: list[str],
    rad_time_frames: int = 1,
) -> list[SceneKey]:
    """Every scene ``_build_input_stack`` will ask for, in the order it asks.

    Knowing this up front is what lets the loads run ahead of the GPU.
    """
    delta = timedelta(minutes=scan_interval(sat_id))
    requests: list[SceneKey] = []
    seen: set[SceneKey] = set()

    def add(band: str, when: datetime) -> None:
        key = (sat_id, band, when)
        if key not in seen:
            seen.add(key)
            requests.append(key)

    for band in flow_bands:
        if _band_available(sat_id, band):
            for when in (t0 - delta, t0, t0 + delta):
                add(band, when)
    for band in rad_bands:
        if not _band_available(sat_id, band):
            continue
        if rad_time_frames == 3:
            for when in (t0 - delta, t0, t0 + delta):
                add(band, when)
        else:
            add(band, t0)
    return requests


# ---------------------------------------------------------------------------
# Availability: which timestamps every satellite can actually deliver
# ---------------------------------------------------------------------------

def availability_band(
    sat_id: str, flow_bands: list[str], rad_bands: list[str],
) -> str | None:
    """First requested band this satellite actually carries.

    Availability is checked per satellite on one band: the scan times are
    a property of the instrument's schedule, not of the channel.
    """
    for band in list(flow_bands) + list(rad_bands):
        if _band_available(sat_id, band):
            return band
    return None


def _icechunk_source(sat_id: str, band: str):
    """(reader, store resolution tier) for an icechunk-backed satellite."""
    if "himawari" in sat_id:
        from stereo_winds.readers.himawari import _BAND_RESOLUTION, Himawari
        src = Himawari(satellite=sat_id, bands=[band])
    elif "gk2a" in sat_id:
        from stereo_winds.readers.gk2a import _BAND_RESOLUTION, GK2A
        src = GK2A(satellite=sat_id, bands=[band])
    elif "mtg" in sat_id:
        from stereo_winds.readers.mtg import _BAND_RESOLUTION, MTG
        src = MTG(satellite=sat_id, bands=[band])
    elif "msg" in sat_id:
        from stereo_winds.readers.msg import _BAND_RESOLUTION, MSG
        src = MSG(satellite=sat_id, bands=[band])
    else:
        raise ValueError(f"No icechunk reader for {sat_id!r}")
    return src, _BAND_RESOLUTION[src.bands[0]]


def _icechunk_available_times(
    sat_id: str, band: str, start: datetime, end: datetime,
) -> np.ndarray:
    """Scan times between ``start`` and ``end``, across every store.

    The loader picks whichever store carries the band and covers the
    time, so availability has to consider the same set — otherwise the
    filter rejects timestamps the pipeline could actually retrieve.
    """
    src, _ = _icechunk_source(sat_id, band)
    resolved = src.bands[0]
    lo = np.datetime64(start, "ns")
    hi = np.datetime64(end, "ns")

    found: list[np.ndarray] = []
    for prefix in src._candidate_stores(resolved):
        contents = src._store_contents(prefix)
        if contents is None or resolved not in contents[0]:
            continue
        bands, first, last = contents
        if last < lo or first > hi:
            continue
        ds = src._open_dataset_at(prefix)
        times = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        found.append(times[(times >= lo) & (times <= hi)])
    if not found:
        return np.array([], dtype="datetime64[ns]")
    return np.unique(np.concatenate(found))


_ABI_START_RE = re.compile(r"_s(\d{13})")


def _goes_available_times(
    sat_id: str, band: str, start: datetime, end: datetime,
    product: str = "ABI-L1b-RadF",
) -> np.ndarray:
    """Scan start times on NOAA's public S3 bucket, listed one day at a time."""
    from stereo_winds.readers.goes import _GNUM, GOES

    src = GOES(satellite=sat_id, product=product, bands=[band])
    gnum = _GNUM[sat_id]
    found: list[np.datetime64] = []
    day = start.date()
    last = end.date()
    while day <= last:
        d = datetime(day.year, day.month, day.day)
        pattern = (f"{src.bucket}/{product}/{d:%Y/%j}/*/"
                   f"OR_{product}-M*{band}_G{gnum}_s*")
        for key in src.fs.glob(pattern):
            match = _ABI_START_RE.search(key)
            if match is None:
                continue
            try:
                t = datetime.strptime(match.group(1), "%Y%j%H%M%S")
            except ValueError:
                continue
            if start <= t <= end:
                found.append(np.datetime64(t, "ns"))
        day += timedelta(days=1)
    return np.unique(np.array(found, dtype="datetime64[ns]"))


_AHI_SLOT_RE = re.compile(r"HS_H\d\d_(\d{8})_(\d{4})_")
_AMI_SLOT_RE = re.compile(r"_(\d{12})\.nc$")


def _s3_l1b_times(
    sat_id: str, band: str, start: datetime, end: datetime,
) -> np.ndarray:
    """Slot times on the public NOAA bucket the readers fall back to.

    Listed one day at a time, one file per slot (a single HSD segment
    for AHI, the single netCDF for AMI).
    """
    from stereo_winds.readers._satpy_s3 import s3_filesystem

    if "himawari" in sat_id:
        from stereo_winds.readers.himawari import (
            _S3_BUCKET, _S3_PLATFORM, _S3_PREFIX, Himawari,
        )
        resolved = Himawari(satellite=sat_id, bands=[band]).bands[0]
        def pattern(d: datetime) -> str:
            return (f"{_S3_BUCKET[sat_id]}/{_S3_PREFIX}/{d:%Y/%m/%d}/*/"
                    f"HS_{_S3_PLATFORM[sat_id]}_*_{resolved}_FLDK_R*_S0110.DAT*")
        def parse(key: str) -> datetime | None:
            m = _AHI_SLOT_RE.search(key)
            return (datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
                    if m else None)
    elif "gk2a" in sat_id:
        from stereo_winds.readers.gk2a import _S3_BUCKET, _S3_PREFIX, GK2A
        resolved = GK2A(satellite=sat_id, bands=[band]).bands[0].lower()
        def pattern(d: datetime) -> str:
            return (f"{_S3_BUCKET}/{_S3_PREFIX}/{d:%Y%m}/{d:%d}/*/"
                    f"gk2a_ami_le1b_{resolved}_fd*ge_*.nc")
        def parse(key: str) -> datetime | None:
            m = _AMI_SLOT_RE.search(key)
            return (datetime.strptime(m.group(1), "%Y%m%d%H%M")
                    if m else None)
    else:
        return np.array([], dtype="datetime64[ns]")

    fs = s3_filesystem()
    found: list[np.datetime64] = []
    day = start.date()
    while day <= end.date():
        d = datetime(day.year, day.month, day.day)
        for key in fs.glob(pattern(d)):
            t = parse(key)
            if t is not None and start <= t <= end:
                found.append(np.datetime64(t, "ns"))
        day += timedelta(days=1)
    return np.unique(np.array(found, dtype="datetime64[ns]"))


def satellite_available_times(
    sat_id: str, band: str, start: datetime, end: datetime,
    product: str = "ABI-L1b-RadF",
    include_s3_fallback: bool = True,
) -> np.ndarray:
    """Sorted scan times available for ``sat_id`` within the window.

    For the icechunk-backed satellites this is the union of the store's
    coverage and, unless disabled, the public-S3 L1b the readers fall
    back to — so the filter does not reject times the pipeline could
    actually load.
    """
    if "goes" in sat_id:
        times = _goes_available_times(sat_id, band, start, end, product)
        logger.info("  %s: %d scans available (S3)", sat_id, len(times))
        return times

    store = _icechunk_available_times(sat_id, band, start, end)
    if not include_s3_fallback or "mtg" in sat_id or "msg" in sat_id:
        logger.info("  %s: %d scans available (icechunk)", sat_id, len(store))
        return store

    s3 = _s3_l1b_times(sat_id, band, start, end)
    times = np.unique(np.concatenate([store, s3])) if s3.size else store
    logger.info("  %s: %d scans available (%d icechunk, %d public S3)",
                sat_id, len(times), len(store), len(s3))
    return times


def _has_scan_near(times: np.ndarray, target: datetime, tol: timedelta) -> bool:
    """True if ``times`` (sorted) holds a scan within ``tol`` of ``target``.

    Instruments do not share a slot convention — AMI stamps scans at
    HH:09:35, ABI full disk at HH:00:00 — and the readers snap to the
    nearest scan, so availability is a tolerance test, not equality.
    """
    if times.size == 0:
        return False
    t64 = np.datetime64(target, "ns")
    i = int(np.searchsorted(times, t64))
    tol64 = np.timedelta64(int(tol.total_seconds()), "s").astype("timedelta64[ns]")
    for j in (i - 1, i):
        if 0 <= j < times.size and abs(times[j] - t64) <= tol64:
            return True
    return False


def filter_to_common_times(
    times: list[datetime],
    sats: list[str],
    flow_bands: list[str],
    rad_bands: list[str],
    dt_min: int | None = None,
    tolerance_min: float = 5.0,
    product: str = "ABI-L1b-RadF",
) -> list[datetime]:
    """Keep only timestamps every satellite can deliver a full triplet for.

    Each retrieval needs three frames per satellite (t-dt, t, t+dt), so a
    timestamp survives only when every satellite has a scan within
    ``tolerance_min`` of all three.  ``dt`` is the satellite's own repeat
    cycle unless ``dt_min`` overrides it, since SEVIRI's is 15 minutes
    where the rest of the ring scans every 10.
    """
    if not times or not sats:
        return list(times)

    offsets = {sat_id: timedelta(minutes=dt_min if dt_min is not None
                                 else scan_interval(sat_id))
               for sat_id in sats}
    tol = timedelta(minutes=tolerance_min)
    widest = max(offsets.values())
    window_start = times[0] - widest - tol
    window_end = times[-1] + widest + tol

    logger.info("Scanning availability for %d satellite(s) over %s .. %s",
                len(sats), window_start, window_end)

    available: dict[str, np.ndarray] = {}
    for sat_id in sats:
        band = availability_band(sat_id, flow_bands, rad_bands)
        if band is None:
            raise RuntimeError(
                f"{sat_id} carries none of the requested bands "
                f"(flow={flow_bands}, rad={rad_bands})"
            )
        available[sat_id] = satellite_available_times(
            sat_id, band, window_start, window_end, product,
        )

    keep: list[datetime] = []
    missing: dict[str, int] = {sat_id: 0 for sat_id in sats}
    for t in times:
        absent = [
            sat_id for sat_id in sats
            if not all(_has_scan_near(available[sat_id], w, tol)
                       for w in (t - offsets[sat_id], t, t + offsets[sat_id]))
        ]
        for sat_id in absent:
            missing[sat_id] += 1
        if not absent:
            keep.append(t)

    dropped = len(times) - len(keep)
    logger.info("Common-time filter: keeping %d of %d timestamps (%d dropped)",
                len(keep), len(times), dropped)
    for sat_id, count in sorted(missing.items(), key=lambda kv: -kv[1]):
        if count:
            logger.info("  %s lacked a full triplet at %d timestamp(s)",
                        sat_id, count)
    return keep


# ---------------------------------------------------------------------------
# Build student model input stack
# ---------------------------------------------------------------------------

def _band_available(sat_id: str, band: str) -> bool:
    """Check whether an ABI band can be loaded for the given satellite.

    Bands that have no spectral equivalent on a satellite (e.g. ABI C09
    and C14 on MTG FCI) return False so the caller can substitute zeros.
    """
    if "goes" in sat_id:
        return True  # ABI has all ABI bands
    if "himawari" in sat_id:
        from stereo_winds.readers.himawari import _ABI_TO_AHI, _BAND_RESOLUTION
        return band in _BAND_RESOLUTION or band in _ABI_TO_AHI
    if "gk2a" in sat_id:
        from stereo_winds.readers.gk2a import _ABI_TO_AMI, _BAND_RESOLUTION
        return band in _BAND_RESOLUTION or band in _ABI_TO_AMI
    if "mtg" in sat_id:
        from stereo_winds.config import ABI_TO_FCI_BAND
        from stereo_winds.readers.mtg import _BAND_RESOLUTION
        return band in _BAND_RESOLUTION or band in ABI_TO_FCI_BAND
    if "msg" in sat_id:
        # SEVIRI has eleven narrow channels against ABI's sixteen, so
        # C04 (1.4 µm cirrus) and C06 (2.2 µm) have no counterpart.
        from stereo_winds.readers.msg import ABI_TO_SEVIRI, _BAND_RESOLUTION
        return band in _BAND_RESOLUTION or band in ABI_TO_SEVIRI
    return True


def _build_input_stack(
    sat_id: str,
    t0: datetime,
    disp: StereoDisparity,
    flow_bands: list[str],
    rad_bands: list[str],
    rad_time_frames: int = 1,
    prefetcher: ScenePrefetcher | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SatelliteConfig,
           dict[str, list[str]]]:
    """Build the (C, H, W) flow/rad/geom input stack from icechunk data.

    Bands that have no spectral equivalent on the target satellite are
    filled with NaN (and masked out in the finite_mask), keeping the
    channel count consistent with the trained student checkpoint.

    The geometry channels (pixel scale, satellite zenith) are derived
    from the projection metadata of the scenes actually loaded, so a
    drifted or relocated satellite is navigated where it really is.

    Channels are written straight into their slot in the output array
    and their sources freed as we go: stacking a list of 20 full disks
    and then running ``nan_to_num`` over it would hold three copies of
    the same 2.3 GB at once.

    Returns (flow_arr, rad_arr, geom_arr, finite_mask, scene_config,
    missing), where ``missing`` names the bands that had to be
    zero-filled — the instrument has no equivalent channel, or no store
    covers that band at this time.
    """
    n_flow_ch = 4 * len(flow_bands)
    n_rad_frames = 3 if rad_time_frames == 3 else 1
    n_rad_ch = n_rad_frames * len(rad_bands)

    flow_arr: np.ndarray | None = None
    rad_arr: np.ndarray | None = None
    finite_mask: np.ndarray | None = None
    scene_cfg: SatelliteConfig | None = None
    # Channels of bands this satellite does not carry; zeroed once a
    # loaded scene has established the grid shape.
    missing_flow: list[int] = []
    missing_rad: list[int] = []
    missing: dict[str, list[str]] = {"flow": [], "rad": []}
    # Frames the radiance pass will need again, kept only for those bands.
    rad_needed = {b for b in rad_bands if _band_available(sat_id, b)}
    cached: dict[str, tuple] = {}

    def _mark_finite(channel: np.ndarray) -> None:
        """Fold one channel into the running finite mask."""
        nonlocal finite_mask
        if finite_mask is None:
            finite_mask = np.isfinite(channel)
        else:
            np.logical_and(finite_mask, np.isfinite(channel), out=finite_mask)

    logger.info("  Loading flow bands: %s", flow_bands)
    for i, band in enumerate(flow_bands):
        base = 4 * i
        if not _band_available(sat_id, band):
            logger.warning("  Band %s unavailable on %s — filling with zeros", band, sat_id)
            missing_flow.extend(range(base, base + 4))
            missing["flow"].append(band)
            continue
        try:
            a_m, a_0, a_p, cfg, dt_used = _load_three_frames(
                sat_id, band, t0, prefetcher=prefetcher)
        except SceneNotInStore as exc:
            # No store covers this band at this time and there is no S3
            # fallback: zero-fill it rather than losing the satellite.
            logger.warning("  %s %s: %s — filling with zeros", sat_id, band, exc)
            missing_flow.extend(range(base, base + 4))
            missing["flow"].append(band)
            continue
        if scene_cfg is None:
            scene_cfg = cfg
        else:
            _check_same_grid(scene_cfg, cfg, f"{sat_id} {band}")
        if flow_arr is None:
            H, W = a_0.shape
            flow_arr = np.empty((n_flow_ch, H, W), dtype=np.float32)
            rad_arr = np.empty((n_rad_ch, H, W), dtype=np.float32)

        valid = np.isfinite(a_m) & np.isfinite(a_0) & np.isfinite(a_p)
        fb = disp._run_pair(a_0, a_m)
        ff = disp._run_pair(a_0, a_p)
        # The student reads displacement over DT_MINUTES.  A satellite
        # that scans on a longer cycle (SEVIRI's 15 min) produces
        # proportionally larger displacements for the same wind, so
        # rescale to the nominal interval rather than inflating speeds.
        if dt_used != DT_MINUTES:
            scale = DT_MINUTES / dt_used
            logger.info("  %s: %d min pair scaled by %.3f to the %d min "
                        "interval the student expects",
                        sat_id, dt_used, scale, DT_MINUTES)
            fb *= scale
            ff *= scale

        for k, component in enumerate((fb[0], fb[1], ff[0], ff[1])):
            channel = flow_arr[base + k]
            np.copyto(channel, component)
            channel[~valid] = np.nan
            channel /= FLOW_SCALE
            _mark_finite(channel)
        del fb, ff

        if band in rad_needed:
            cached[band] = ((a_m, a_0, a_p, valid) if rad_time_frames == 3
                            else (None, a_0, None, valid))
        del a_m, a_p

    rad_offset = 0
    logger.info("  Loading rad bands: %s", rad_bands)
    for band in rad_bands:
        base = rad_offset
        rad_offset += n_rad_frames
        if not _band_available(sat_id, band):
            logger.warning("  Band %s unavailable on %s — filling with zeros", band, sat_id)
            missing_rad.extend(range(base, base + n_rad_frames))
            missing["rad"].append(band)
            continue

        if band in cached:
            a_m, a_0, a_p, vb = cached.pop(band)
        else:
            try:
                if rad_time_frames == 3:
                    a_m, a_0, a_p, cfg, _ = _load_three_frames(
                        sat_id, band, t0, prefetcher=prefetcher)
                    vb = np.isfinite(a_m) & np.isfinite(a_0) & np.isfinite(a_p)
                else:
                    a_0, cfg = (prefetcher.get(sat_id, band, t0)
                                if prefetcher is not None
                                else _load_scene(sat_id, band, t0))
                    vb = np.isfinite(a_0)
                    a_m = a_p = None
            except SceneNotInStore as exc:
                logger.warning("  %s %s: %s — filling with zeros",
                               sat_id, band, exc)
                missing_rad.extend(range(base, base + n_rad_frames))
                missing["rad"].append(band)
                continue
            if scene_cfg is None:
                scene_cfg = cfg
            else:
                _check_same_grid(scene_cfg, cfg, f"{sat_id} {band}")
        if rad_arr is None:
            H, W = a_0.shape
            flow_arr = np.empty((n_flow_ch, H, W), dtype=np.float32)
            rad_arr = np.empty((n_rad_ch, H, W), dtype=np.float32)

        frames = (a_m, a_0, a_p) if rad_time_frames == 3 else (a_0,)
        for k, frame in enumerate(frames):
            channel = rad_arr[base + k]
            np.copyto(channel, frame)
            channel[~vb] = np.nan
            _mark_finite(channel)
        del a_m, a_0, a_p, frames

    del cached

    if scene_cfg is None or flow_arr is None or rad_arr is None:
        raise RuntimeError(
            f"{sat_id}: none of the requested bands are available "
            f"(flow={flow_bands}, rad={rad_bands}) — nothing to navigate from"
        )

    if missing_flow:
        flow_arr[missing_flow] = 0.0
    if missing_rad:
        rad_arr[missing_rad] = 0.0

    dx_m, dy_m = compute_pixel_scale(scene_cfg)
    zen = compute_grid_zenith(scene_cfg)
    geom_arr = np.stack(
        [dx_m / PIXEL_SCALE_NORM, dy_m / PIXEL_SCALE_NORM, zen / ZENITH_NORM], 0,
    ).astype(np.float32)
    del dx_m, dy_m, zen

    # In place: the model only needs the NaNs replaced, not a fresh copy.
    np.nan_to_num(flow_arr, copy=False)
    np.nan_to_num(rad_arr, copy=False)
    np.nan_to_num(geom_arr, copy=False)
    return flow_arr, rad_arr, geom_arr, finite_mask, scene_cfg, missing


# ---------------------------------------------------------------------------
# Forward pass (row-strip tiling)
# ---------------------------------------------------------------------------

def _forward_full_disk(
    model: StudentWindsModel,
    flow_arr: np.ndarray,
    rad_arr: np.ndarray,
    geom_arr: np.ndarray,
    row_strip: int = 1024,
    halo: int = 8,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Run the student model in row-strips over the full disk."""
    H, W = flow_arr.shape[1], flow_arr.shape[2]
    keys = ["u_mean", "v_mean", "h_mean", "u_logvar", "v_logvar", "h_logvar"]
    if getattr(model, "predict_chi2", False):
        keys.append("chi2")
    out = {k: np.full((H, W), np.nan, np.float32) for k in keys}
    with torch.no_grad():
        for r in range(0, H, row_strip):
            r1_keep = min(H, r + row_strip)
            r0_in = max(0, r - halo)
            r1_in = min(H, r1_keep + halo)
            ft = torch.from_numpy(flow_arr[:, r0_in:r1_in]).unsqueeze(0).to(device)
            rt = torch.from_numpy(rad_arr[:, r0_in:r1_in]).unsqueeze(0).to(device)
            gt = torch.from_numpy(geom_arr[:, r0_in:r1_in]).unsqueeze(0).to(device)
            o = model.predict(ft, rt, gt)
            keep0 = r - r0_in
            keep1 = keep0 + (r1_keep - r)
            # Multi-band model: select band 0 (or squeeze single-band)
            for k in keys:
                val = o[k]
                if val.ndim == 4:
                    out[k][r:r1_keep] = val[0, 0, keep0:keep1].cpu().numpy()
                else:
                    out[k][r:r1_keep] = val[0, keep0:keep1].cpu().numpy()
    return out


def quality_attrs(
    flow_bands: list[str],
    rad_bands: list[str],
    missing: dict[str, list[str]],
) -> dict:
    """Describe how much of the requested input the retrieval actually got.

    Zero-filled channels still produce winds — the network sees a valid
    channel count — but with less information behind them, so the
    shortfall travels with the data rather than being inferable only from
    the run log.
    """
    # Count distinct bands: the flow set is a subset of the rad set, so a
    # band absent from both would otherwise count twice and can tip the
    # threshold on its own.
    requested = list(dict.fromkeys(list(flow_bands) + list(rad_bands)))
    absent = sorted(set(missing.get("flow", [])) | set(missing.get("rad", [])))
    fraction = len(absent) / len(requested) if requested else 0.0
    degraded = fraction > DEGRADED_BAND_FRACTION

    if not absent:
        note = "all requested bands available"
    else:
        note = (f"{len(absent)} of {len(requested)} requested bands were "
                f"unavailable and zero-filled: {', '.join(absent)}")
        if degraded:
            note = ("DEGRADED QUALITY — " + note +
                    f". Winds and heights here rest on "
                    f"{len(requested) - len(absent)} channels and should be "
                    f"treated as lower confidence.")
    return {
        "bands_requested": ",".join(requested),
        "bands_missing": ",".join(absent),
        "n_bands_requested": len(requested),
        "n_bands_missing": len(absent),
        "flow_bands_missing": ",".join(sorted(set(missing.get("flow", [])))),
        "quality_degraded": int(degraded),
        "quality_note": note,
    }


def _assemble_vars(
    raw: dict[str, np.ndarray], finite_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert raw model output to the standard AMV variable schema.

    ``raw`` is consumed: each field is dropped as it is converted, so
    the two representations of a full disk are never both resident.
    """
    # Mask in place — raw is ours to spend.
    u = raw.pop("u_mean")
    v = raw.pop("v_mean")
    h_km = raw.pop("h_mean")
    for arr in (u, v, h_km):
        arr[~finite_mask] = np.nan

    valid = finite_mask & np.isfinite(u) & np.isfinite(v) & np.isfinite(h_km)
    qf = np.where(valid, 2.0, 0.0).astype(np.float32)
    del valid

    out = {
        "u_wind": u.astype(np.float32, copy=False),
        "v_wind": v.astype(np.float32, copy=False),
        "quality_flag": qf,
    }
    h_km *= 1000.0
    out["cloud_top_height"] = h_km.astype(np.float32, copy=False)

    for name, key, scale in (("sigma_u", "u_logvar", 1.0),
                             ("sigma_v", "v_logvar", 1.0),
                             ("sigma_h", "h_logvar", 1000.0)):
        logvar = raw.pop(key)
        logvar *= 0.5
        np.exp(logvar, out=logvar)
        if scale != 1.0:
            logvar *= scale
        out[name] = logvar.astype(np.float32, copy=False)
    return out


# ---------------------------------------------------------------------------
# Per-satellite inference
# ---------------------------------------------------------------------------

def infer_satellite(
    sat_id: str,
    t0: datetime,
    model: StudentWindsModel,
    disp: StereoDisparity,
    flow_bands: list[str],
    rad_bands: list[str],
    device: str = "cuda",
    row_strip: int = 1024,
    prefetcher: ScenePrefetcher | None = None,
) -> xr.Dataset:
    """Run student inference for one satellite and return an xr.Dataset.

    The dataset has dimensions (y, x) with lat/lon coordinate arrays
    and the standard AMV variables.
    """
    nominal = SATELLITE_CONFIGS[sat_id]
    logger.info("Processing %s (nominal sub_lon=%.1f°)",
                sat_id, nominal.sub_lon_deg)

    rad_tf = int(getattr(model, "rad_time_frames", 1))
    flow_arr, rad_arr, geom_arr, finite_mask, sat, missing = _build_input_stack(
        sat_id, t0, disp, flow_bands, rad_bands,
        rad_time_frames=rad_tf, prefetcher=prefetcher,
    )

    # Navigate with what the data says, and say so when it disagrees with
    # the nominal slot — that is a drift or a relocation, not an error.
    drift = sat.sub_lon_deg - nominal.sub_lon_deg
    if abs(drift) > SUB_LON_TOL_DEG:
        logger.warning(
            "  %s: scene sub_lon=%.4f° differs from the nominal %.4f° "
            "by %.4f° — navigating with the value from the data",
            sat_id, sat.sub_lon_deg, nominal.sub_lon_deg, drift,
        )
    else:
        logger.info("  %s: scene sub_lon=%.4f° (height %.0f m)",
                    sat_id, sat.sub_lon_deg, sat.satellite_height_m)

    logger.info("  Forward pass (%dx%d)...", sat.n_rows, sat.n_cols)
    raw = _forward_full_disk(model, flow_arr, rad_arr, geom_arr,
                             row_strip=row_strip, device=device)
    # The model inputs are ~3.7 GB for a full disk and are finished with
    # the moment the forward pass returns.
    del flow_arr, rad_arr, geom_arr
    amvs = _assemble_vars(raw, finite_mask)
    del raw

    # Compute lat/lon for each pixel
    lat, lon = compute_grid_latlon(sat)
    zen = compute_grid_zenith(sat)

    ds = xr.Dataset(
        {k: (("y", "x"), amvs[k]) for k in OUTPUT_VARS},
        coords={
            "latitude": (("y", "x"), lat.astype(np.float32)),
            "longitude": (("y", "x"), lon.astype(np.float32)),
            "zenith_angle": (("y", "x"), zen.astype(np.float32)),
        },
        attrs={
            "satellite_id": sat_id,
            **quality_attrs(flow_bands, rad_bands, missing),
            "sub_satellite_longitude": sat.sub_lon_deg,
            "sub_satellite_longitude_source": "scene projection metadata",
            "nominal_sub_satellite_longitude": nominal.sub_lon_deg,
            "satellite_height_m": sat.satellite_height_m,
            "time": str(t0),
            "source": "student_amv",
        },
    )
    return ds


# ---------------------------------------------------------------------------
# Global mosaic: merge per-satellite datasets on a regular lat/lon grid
# ---------------------------------------------------------------------------

# Sentinel for "no satellite contributed here" in the source index.
NO_SOURCE = -1


class GlobalMosaic:
    """Incremental min-zenith mosaic on a regular lat/lon grid.

    Satellites are gridded one at a time and dropped immediately after,
    so peak memory holds the accumulator plus a single full disk rather
    than every satellite at once.

    The winning satellite is recorded as an ``int8`` index into
    :attr:`sources` rather than a string: at 2 km the grid is ~200 M
    cells, where a ``U12`` string array costs 9 GB against 0.2 GB for
    the codes.  ``decode_source_satellite`` turns them back into names.

    Parameters
    ----------
    resolution_m : output grid spacing in meters (default 2000 m)
    """

    def __init__(self, resolution_m: float = 2000.0) -> None:
        # Convert metres to degrees: 1° latitude ≈ 111 320 m
        dlat = resolution_m / 111_320.0
        # Use the same angular spacing for longitude; pixels are ~square at
        # the equator and compress toward the poles (equirectangular).
        self.resolution_m = resolution_m
        self.step_deg = dlat
        # Edges as exact multiples of the step rather than an accumulated
        # arange, so a cell index can be computed arithmetically and still
        # agree with the edges.  Cell counts are unchanged; edges move by
        # under 1e-10 degrees.
        n_lat = int(np.ceil(180.0 / dlat))
        n_lon = int(np.ceil(360.0 / dlat))
        self.lat_bins = -90.0 + np.arange(n_lat + 1) * dlat
        self.lon_bins = -180.0 + np.arange(n_lon + 1) * dlat
        self.lat_centers = 0.5 * (self.lat_bins[:-1] + self.lat_bins[1:])
        self.lon_centers = 0.5 * (self.lon_bins[:-1] + self.lon_bins[1:])
        shape = (len(self.lat_centers), len(self.lon_centers))
        logger.info("Global grid: %d x %d (%.0f m ≈ %.4f°), %.1f GB accumulator",
                    shape[0], shape[1], resolution_m, dlat,
                    shape[0] * shape[1] * (4 * (len(OUTPUT_VARS) + 1) + 1) / 2**30)

        self.best_zen = np.full(shape, np.inf, dtype=np.float32)
        self.merged = {v: np.full(shape, np.nan, dtype=np.float32)
                       for v in OUTPUT_VARS}
        self.source_index = np.full(shape, NO_SOURCE, dtype=np.int8)
        self.sources: list[str] = []
        self.time: str | None = None
        # satellite -> its quality note, for satellites that contributed
        self.quality: dict[str, dict] = {}

    def _bin_index(self, values: np.ndarray, origin: float,
                   n_bins: int) -> np.ndarray:
        """Cell index for ``values`` on a uniform grid, clipped in range."""
        idx = ((values - origin) / self.step_deg).astype(np.int64)
        return np.clip(idx, 0, n_bins - 1, out=idx)

    def add(self, sat_id: str, ds: xr.Dataset) -> int:
        """Grid one satellite, keeping cells where it beats the zenith so far.

        Returns the number of grid cells this satellite won.
        """
        logger.info("Gridding %s onto %.0f m global grid...",
                    sat_id, self.resolution_m)
        if self.time is None:
            self.time = ds.attrs.get("time")
        if ds.attrs.get("n_bands_missing"):
            self.quality[sat_id] = {
                "missing": ds.attrs.get("bands_missing", ""),
                "n_missing": int(ds.attrs.get("n_bands_missing", 0)),
                "n_requested": int(ds.attrs.get("n_bands_requested", 0)),
                "degraded": int(ds.attrs.get("quality_degraded", 0)),
            }

        lat_2d = ds["latitude"].values
        lon_2d = ds["longitude"].values
        qf = ds["quality_flag"].values

        # Only grid pixels with valid AMVs
        valid = (qf >= 2) & np.isfinite(lat_2d) & np.isfinite(lon_2d)
        n_valid = int(valid.sum())
        if not n_valid:
            logger.warning("  %s: no valid pixels to grid", sat_id)
            return 0

        flat_zen = ds["zenith_angle"].values[valid]

        # Which cell each pixel falls in.  The bins are uniform, so this
        # is arithmetic rather than a binary search per pixel: ~50x
        # faster than np.digitize over 20 M points, and verified to give
        # identical indices on real full-disk geolocation.
        ri = self._bin_index(lat_2d[valid], self.lat_bins[0],
                             len(self.lat_centers))
        ci = self._bin_index(lon_2d[valid], self.lon_bins[0],
                             len(self.lon_centers))

        # Where this satellite beats the current best zenith
        better = flat_zen < self.best_zen[ri, ci]
        idx_r = ri[better]
        idx_c = ci[better]
        del ri, ci

        n_won = int(idx_r.size)
        if n_won:
            for var_name in OUTPUT_VARS:
                # One variable at a time: the flattened copy is freed
                # before the next is taken.
                flat_var = ds[var_name].values[valid]
                self.merged[var_name][idx_r, idx_c] = flat_var[better]
                del flat_var
            self.best_zen[idx_r, idx_c] = flat_zen[better]
            if sat_id not in self.sources:
                self.sources.append(sat_id)
            self.source_index[idx_r, idx_c] = self.sources.index(sat_id)

        logger.info("  %s: %d grid cells contributed (of %d valid pixels)",
                    sat_id, n_won, n_valid)
        return n_won

    def _quality_attrs(self) -> dict:
        """Which contributors ran on incomplete input, and how badly.

        The mosaic takes each cell from one satellite, so quality varies
        across the grid: ``source_satellite_index`` says which satellite
        a cell came from, and these attributes say what that satellite
        was working with.
        """
        contributing = {s: q for s, q in self.quality.items()
                        if s in self.sources}
        if not contributing:
            return {"quality_degraded": 0,
                    "quality_note": "all contributing satellites had every "
                                    "requested band"}
        detail = "; ".join(
            f"{sat}: {q['n_missing']}/{q['n_requested']} bands missing"
            f"{' (DEGRADED)' if q['degraded'] else ''} [{q['missing']}]"
            for sat, q in sorted(contributing.items()))
        degraded = sorted(s for s, q in contributing.items() if q["degraded"])
        note = ("Some contributing satellites ran on incomplete input. "
                + detail)
        if degraded:
            note = (f"DEGRADED QUALITY over cells sourced from "
                    f"{', '.join(degraded)} — " + detail +
                    ". Use source_satellite_index to find the affected cells.")
        return {
            "quality_degraded": int(bool(degraded)),
            "degraded_satellites": ",".join(degraded),
            "quality_note": note,
        }

    def to_dataset(self) -> xr.Dataset:
        """Assemble the accumulated grids into the output dataset."""
        ds_global = xr.Dataset(
            {v: (("latitude", "longitude"), self.merged[v]) for v in OUTPUT_VARS},
            coords={
                "latitude": self.lat_centers,
                "longitude": self.lon_centers,
            },
            attrs={
                "title": "Global student AMV mosaic",
                "resolution_m": self.resolution_m,
                "merge_rule": "minimum zenith angle (closest to sub-satellite point)",
                "satellites": list(self.sources),
                "time": self.time,
                **self._quality_attrs(),
            },
        )
        ds_global["source_satellite_index"] = (
            ("latitude", "longitude"), self.source_index,
        )
        ds_global["source_satellite_index"].attrs.update({
            "long_name": "index into the satellites attribute of the "
                         "satellite that won each cell",
            "flag_values": list(range(len(self.sources))),
            "flag_meanings": " ".join(self.sources),
            # Deliberately not _FillValue/missing_value: either makes
            # xarray mask on read, promoting the int8 codes to float and
            # quadrupling what this variable costs a consumer.
            "no_source_index": NO_SOURCE,
        })
        return ds_global


def _satellite_names(value) -> list[str]:
    """Normalise a ``satellites`` attribute to a list of names.

    It is a real list in memory, a comma-separated string once written
    to NetCDF or zarr, and that string's characters if handed to
    ``list()`` by mistake.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v) for v in value]
    return [part for part in str(value).split(",") if part]


def decode_source_satellite(ds: xr.Dataset) -> np.ndarray:
    """Satellite ids behind ``source_satellite_index``, as a string array.

    Materialised on demand — at 2 km this is a 9 GB array, which is why
    the mosaic stores codes instead.  Slice the index variable first if
    you only need a region.
    """
    index = ds["source_satellite_index"].values
    names = str(
        ds["source_satellite_index"].attrs.get("flag_meanings", "")
    ).split()
    if not names:
        # ``satellites`` is a comma-separated string, so list() on it
        # yields single characters and every cell decodes to "g".  It
        # arrives as a real list only before a round trip through a file.
        names = _satellite_names(ds.attrs.get("satellites"))
    out = np.full(index.shape, "", dtype=f"U{max((len(n) for n in names), default=1)}")
    for code, name in enumerate(names):
        out[index == code] = name
    return out


def merge_global(
    per_sat: dict[str, xr.Dataset],
    resolution_m: float = 2000.0,
) -> xr.Dataset:
    """Merge per-satellite AMV datasets onto a global 2 km lat/lon grid.

    At each grid cell, if multiple satellites contribute, the one with
    the smallest zenith angle (closest to the sub-satellite point) wins.

    Holding every satellite in a dict costs ~6 GB of full disks; the
    pipeline itself streams them through :class:`GlobalMosaic` instead.

    Parameters
    ----------
    per_sat : dict mapping satellite_id -> xr.Dataset (from infer_satellite)
    resolution_m : output grid spacing in meters (default 2000 m)
    """
    mosaic = GlobalMosaic(resolution_m=resolution_m)
    for sat_id, ds in per_sat.items():
        mosaic.add(sat_id, ds)
    return mosaic.to_dataset()


# ---------------------------------------------------------------------------
# Icechunk output (optional): append the global mosaic along a time axis
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def process_time(
    t0: datetime,
    sats: list[str],
    model: StudentWindsModel,
    disp: StereoDisparity,
    flow_bands: list[str],
    rad_bands: list[str],
    out_dir: Path,
    device: str = "cuda",
    row_strip: int = 1024,
    resolution_m: float = 2000.0,
    skip_global: bool = False,
    skip_existing: bool = False,
    repo=None,
    icechunk_branch: str = "main",
    icechunk_chunk: int = 1024,
    icechunk_times: set[datetime] | None = None,
    write_netcdf: bool = True,
    global_netcdf: bool = True,
    temp_dir: Path | None = None,
    keep_temp: bool = False,
    prefetcher: ScenePrefetcher | None = None,
) -> bool:
    """Run the full ring for a single timestamp and write its output.

    Output goes to per-day NetCDF files under ``out_dir`` and, when
    ``repo`` is given, to an icechunk store as a new commit.

    With ``temp_dir`` the per-satellite mosaics are intermediates rather
    than deliverables: they go to a scratch folder created under it and
    are removed once the global mosaic has been committed, so a long run
    does not accumulate full disks it will never read again.  Set
    ``keep_temp`` to leave the folder in place for inspection.  It is
    ignored under ``skip_global``, where those files are the only output
    and deleting them would leave the timestamp with nothing.

    Returns True if output exists for this timestamp when the call ends
    (either newly written or already present under ``--skip-existing``).
    """
    tag = time_tag(t0)
    icechunk_times = icechunk_times if icechunk_times is not None else set()

    if skip_existing and not skip_global:
        # Done only when every configured sink already holds this timestamp.
        # The scratch copies are not a sink: they are deleted after every
        # commit, so their absence says nothing about what has been written.
        sinks = []
        if global_netcdf:
            sinks.append(("netcdf", global_nc_path(out_dir, t0).exists()))
        if repo is not None:
            sinks.append(("icechunk", t0 in icechunk_times))
        if sinks and all(present for _, present in sinks):
            logger.info("[%s] already in %s — skipping", tag,
                        " and ".join(name for name, _ in sinks))
            return True

    # Per-satellite files land in a scratch folder when one is configured,
    # and beside the mosaic otherwise.  Never under --skip-global: with no
    # mosaic to commit, they are the output rather than an intermediate.
    scratch = (make_scratch_dir(temp_dir, t0)
               if write_netcdf and temp_dir is not None and not skip_global
               else None)
    sat_root = scratch if scratch is not None else out_dir
    if scratch is not None:
        logger.info("[%s] Per-satellite mosaics -> %s (removed after the "
                    "global mosaic is committed)", tag, scratch)

    try:
        return _process_time_inner(
            t0, sats, model, disp, flow_bands, rad_bands, out_dir,
            sat_root=sat_root, device=device, row_strip=row_strip,
            resolution_m=resolution_m, skip_global=skip_global,
            skip_existing=skip_existing, repo=repo,
            icechunk_branch=icechunk_branch, icechunk_chunk=icechunk_chunk,
            icechunk_times=icechunk_times, write_netcdf=write_netcdf,
            global_netcdf=global_netcdf, prefetcher=prefetcher,
        )
    finally:
        # After the commit, whether or not it succeeded: a scratch folder
        # kept "just in case" is what fills the disk on a long run.
        if scratch is not None and not keep_temp:
            remove_scratch_dir(scratch)
        elif scratch is not None:
            logger.info("[%s] Keeping scratch folder %s", tag, scratch)


def _process_time_inner(
    t0: datetime,
    sats: list[str],
    model: StudentWindsModel,
    disp: StereoDisparity,
    flow_bands: list[str],
    rad_bands: list[str],
    out_dir: Path,
    sat_root: Path,
    device: str = "cuda",
    row_strip: int = 1024,
    resolution_m: float = 2000.0,
    skip_global: bool = False,
    skip_existing: bool = False,
    repo=None,
    icechunk_branch: str = "main",
    icechunk_chunk: int = 1024,
    icechunk_times: set[datetime] | None = None,
    write_netcdf: bool = True,
    global_netcdf: bool = True,
    prefetcher: ScenePrefetcher | None = None,
) -> bool:
    """One timestamp, with the per-satellite directory already decided.

    Split out so :func:`process_time` can own the scratch folder's
    lifetime in one place rather than threading cleanup through every
    early return.
    """
    tag = time_tag(t0)
    icechunk_times = icechunk_times if icechunk_times is not None else set()

    if write_netcdf:
        day_dir(sat_root, t0).mkdir(parents=True, exist_ok=True)
    if global_netcdf and not skip_global:
        day_dir(out_dir, t0).mkdir(parents=True, exist_ok=True)

    # Satellites are gridded as they finish and dropped immediately:
    # holding all six full disks costs ~6 GB for no benefit.
    mosaic = None if skip_global else GlobalMosaic(resolution_m=resolution_m)
    done: list[str] = []
    todo = [s for s in sats if s in SATELLITE_CONFIGS]
    for unknown in [s for s in sats if s not in SATELLITE_CONFIGS]:
        logger.warning("Unknown satellite %r, skipping", unknown)

    rad_tf = int(getattr(model, "rad_time_frames", 1)) if model is not None else 1

    def queue(sat_id: str) -> None:
        """Start this satellite's reads, unless it will be skipped."""
        if prefetcher is None:
            return
        if not _needs_inference(sat_id, t0, sat_root, skip_existing,
                                write_netcdf):
            return
        prefetcher.submit(scene_requests(sat_id, t0, flow_bands, rad_bands,
                                         rad_tf))

    if todo:
        # Start the first satellite's reads before anything else happens.
        queue(todo[0])

    for i, sat_id in enumerate(todo):
        nc_path = sat_nc_path(sat_root, sat_id, t0)
        ds = None
        if skip_existing and write_netcdf and nc_path.exists():
            logger.info("[%s] %s already exists — reusing %s",
                        tag, sat_id, nc_path)
            try:
                ds = xr.load_dataset(nc_path)
            except Exception:
                logger.exception("Failed to read %s — recomputing", nc_path)
                ds = None
        if i + 1 < len(todo):
            # Queue the next satellite now: its downloads then overlap this
            # satellite's forward pass instead of following it.
            queue(todo[i + 1])
        if ds is None:
            try:
                ds = infer_satellite(
                    sat_id, t0, model, disp,
                    flow_bands, rad_bands,
                    device=device, row_strip=row_strip,
                    prefetcher=prefetcher,
                )
                if write_netcdf:
                    ds.to_netcdf(nc_path)
                    logger.info("Saved %s", nc_path)
            except Exception:
                logger.exception("[%s] Failed to process %s — skipping",
                                 tag, sat_id)
                continue
        try:
            if mosaic is not None:
                mosaic.add(sat_id, ds)
            done.append(sat_id)
        finally:
            ds.close()
            del ds
        log_peak_rss(f"[{tag}] after {sat_id}")

    if not done:
        if prefetcher is not None:
            prefetcher.reset()
        logger.error("[%s] No satellites produced output.", tag)
        return False

    logger.info("[%s] Completed %d/%d satellites: %s",
                tag, len(done), len(sats), done)
    if prefetcher is not None:
        left = prefetcher.reset()
        logger.info("[%s] %s%s", tag, prefetcher.cache.summary(),
                    f", released {left} unread prefetch(es)" if left else "")

    if mosaic is not None:
        logger.info("[%s] Assembling global mosaic (%.0f m grid)...",
                    tag, resolution_m)
        ds_global = mosaic.to_dataset()

        if global_netcdf:
            global_path = global_nc_path(out_dir, t0)
            ds_global.to_netcdf(global_path)
            logger.info("Saved global mosaic: %s", global_path)

        if repo is not None:
            if t0 in icechunk_times:
                logger.warning(
                    "[%s] already in the icechunk store — not appending a "
                    "duplicate (use --skip-existing to skip it entirely)", tag)
            else:
                write_mosaic_to_icechunk(
                    repo, ds_global, t0,
                    branch=icechunk_branch, chunk=icechunk_chunk,
                )
                icechunk_times.add(t0)

        # Summary stats.  Count in place rather than materialising a
        # full boolean copy of the grid.
        n_valid = int(np.count_nonzero(ds_global["quality_flag"].values >= 2))
        n_total = ds_global["quality_flag"].size
        logger.info("[%s] Global mosaic: %d / %d grid cells with valid AMVs "
                    "(%.1f%%)", tag, n_valid, n_total,
                    100 * n_valid / n_total if n_total else 0)
        del ds_global, mosaic

    log_peak_rss(f"[{tag}] timestamp complete")
    return True


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description="Global geostationary ring student AMV inference",
    )
    ap.add_argument("--time", required=True,
                    help="ISO timestamp, or the start of the range when "
                         "--end-time is given (e.g. 2025-03-10T12:00)")
    ap.add_argument("--end-time", default=None,
                    help="ISO timestamp ending the range (inclusive). "
                         "Omit to process only --time.")
    ap.add_argument("--step-minutes", type=int, default=DT_MINUTES,
                    help=f"Spacing between timestamps in a range "
                         f"(default {DT_MINUTES})")
    ap.add_argument("--student-ckpt", required=True,
                    help="Student Lightning checkpoint")
    ap.add_argument("--raft-ckpt", required=True,
                    help="RAFT optical-flow checkpoint")
    ap.add_argument("--output-dir", default="output/global_ring",
                    help="Output directory; files land in "
                         "<output-dir>/<YYYYMMDD>/ with timestamped names")
    ap.add_argument("--device", default="cuda",
                    choices=["cuda", "cpu"])
    ap.add_argument("--satellites", default=None,
                    help="Comma-separated satellite IDs to process "
                         "(default: all in the ring)")
    ap.add_argument("--flow-bands", default=",".join(DEFAULT_FLOW_BANDS),
                    help="Comma-separated flow bands (ABI names)")
    ap.add_argument("--rad-bands", default=",".join(DEFAULT_RAD_BANDS),
                    help="Comma-separated radiance bands (ABI names)")
    ap.add_argument("--resolution-m", type=float, default=2000.0,
                    help="Global grid resolution in meters (default 2000)")
    ap.add_argument("--row-strip", type=int, default=1024,
                    help="Row strip height for tiled inference")
    ap.add_argument("--skip-global", action="store_true",
                    help="Skip the global mosaic, only produce per-satellite files")
    ap.add_argument("--require-all-satellites", action="store_true",
                    help="Before processing, list each satellite's scan "
                         "times and keep only the timestamps where every "
                         "satellite can supply a full t-dt/t/t+dt triplet")
    ap.add_argument("--availability-tolerance", type=float, default=5.0,
                    help="How close a scan must be to count as covering a "
                         "frame, in minutes (default 5)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip timestamps whose output already exists in "
                         "every configured sink (resume an interrupted range)")

    perf = ap.add_argument_group("throughput")
    perf.add_argument("--scene-cache-gb", type=float, default=None,
                      help="RAM the decoded-scene cache may use (default: "
                           "free memory minus a reserve for the working "
                           "set; 0 disables caching)")
    perf.add_argument("--memory-reserve-gb", type=float,
                      default=DEFAULT_MEMORY_RESERVE / 2**30,
                      help="Memory left for everything that is not the "
                           "scene cache when sizing it automatically "
                           f"(default {DEFAULT_MEMORY_RESERVE / 2**30:.0f})")
    perf.add_argument("--prefetch-workers", type=int, default=4,
                      help="Scenes decoded concurrently ahead of the GPU "
                           "(default 4; 0 loads inline)")
    perf.add_argument("--download-workers", type=int, default=None,
                      help="Concurrent object-store downloads within one "
                           "scene, e.g. AHI's ten segments "
                           f"(default {DEFAULT_DOWNLOAD_WORKERS})")

    ic = ap.add_argument_group("icechunk output")
    ic.add_argument("--icechunk-store", default=None,
                    help="Write the global mosaic to this icechunk store, "
                         "appending along time: s3://bucket/prefix (or a "
                         "local directory path)")
    ic.add_argument("--icechunk-branch", default="main",
                    help="Branch to commit to (default: main)")
    ic.add_argument("--icechunk-chunk", type=int, default=1024,
                    help="Spatial chunk size for icechunk arrays (default 1024)")
    ic.add_argument("--icechunk-endpoint", default=None,
                    help="S3 endpoint URL for non-AWS stores "
                         "(e.g. https://data.source.coop)")
    ic.add_argument("--icechunk-region", default=None,
                    help="S3 region")
    ic.add_argument("--icechunk-anonymous", action="store_true",
                    help="Access the store anonymously (read-only; writes "
                         "need credentials from the environment)")
    ic.add_argument("--icechunk-force-path-style", action="store_true",
                    help="Use path-style S3 addressing (needed by minio and "
                         "source.coop)")
    ic.add_argument("--no-netcdf", action="store_true",
                    help="Write only to the icechunk store, skipping the "
                         "per-satellite files entirely (they are otherwise "
                         "written to the scratch folder and deleted after "
                         "each commit)")
    ic.add_argument("--keep-netcdf", action="store_true",
                    help="Also keep per-day NetCDF files under --output-dir, "
                         "as before --temp-dir existed: per-satellite and "
                         "global mosaics both persist alongside the store")

    tmp = ap.add_argument_group("scratch space")
    tmp.add_argument("--temp-dir", default="output",
                     help="Parent of the per-timestamp scratch folder "
                          "holding per-satellite mosaics, which is removed "
                          "once the global mosaic is committed "
                          "(default: output)")
    tmp.add_argument("--keep-temp", action="store_true",
                     help="Leave each scratch folder in place instead of "
                          "removing it, for inspecting a bad timestamp")
    args = ap.parse_args()

    if args.no_netcdf and not args.icechunk_store:
        ap.error("--no-netcdf requires --icechunk-store; otherwise nothing "
                 "would be written")
    if args.no_netcdf and args.keep_netcdf:
        ap.error("--no-netcdf and --keep-netcdf ask for opposite things")
    if args.keep_temp and not args.icechunk_store:
        ap.error("--keep-temp only applies to the scratch folder, which is "
                 "used when --icechunk-store is given")
    if args.icechunk_store and args.skip_global:
        ap.error("--icechunk-store writes the global mosaic, which "
                 "--skip-global disables")

    t_start = datetime.fromisoformat(args.time)
    t_end = datetime.fromisoformat(args.end_time) if args.end_time else None
    try:
        times = time_steps(t_start, t_end, args.step_minutes)
    except ValueError as exc:
        ap.error(str(exc))

    out_dir = Path(args.output_dir)

    # With a store, the NetCDF files are intermediates unless asked for:
    # the mosaic lives in the store, and the per-satellite disks go to a
    # scratch folder that is emptied after every commit.
    scratch_mode = bool(args.icechunk_store) and not args.keep_netcdf
    temp_dir = Path(args.temp_dir) if scratch_mode else None
    global_netcdf = not args.no_netcdf and not scratch_mode
    if global_netcdf or not args.icechunk_store:
        out_dir.mkdir(parents=True, exist_ok=True)

    sats = (args.satellites.split(",") if args.satellites
            else RING_SATELLITES)
    flow_bands = [b for b in args.flow_bands.split(",") if b]
    rad_bands = [b for b in args.rad_bands.split(",") if b]

    logger.info("Processing %d timestamp(s): %s%s",
                len(times), time_tag(times[0]),
                f" .. {time_tag(times[-1])} every {args.step_minutes} min"
                if len(times) > 1 else "")

    if args.require_all_satellites:
        try:
            times = filter_to_common_times(
                times, sats, flow_bands, rad_bands,
                # triplet offsets come from each satellite's own cadence
                tolerance_min=args.availability_tolerance,
            )
        except Exception:
            logger.exception("Availability scan failed — cannot determine "
                             "which timestamps every satellite covers")
            sys.exit(1)
        if not times:
            logger.error("No timestamp in the range has data from all of: %s",
                         ", ".join(sats))
            sys.exit(1)

    if args.download_workers is not None:
        os.environ["STEREO_WINDS_DOWNLOAD_WORKERS"] = str(args.download_workers)

    reusable = scene_cache_is_useful(args.step_minutes, sats) and len(times) > 1
    if args.scene_cache_gb is not None:
        cache_bytes = int(args.scene_cache_gb * 2**30)
    elif not reusable:
        # Nothing read at this timestamp will be read at the next one, so
        # a large cache would just accumulate scenes until the kernel
        # takes exception to it.
        cache_bytes = NO_REUSE_CACHE_BYTES
        logger.info(
            "--step-minutes %d is more than twice every satellite's scan "
            "interval, so no scene is read twice: holding the scene cache "
            "to %.1f GB rather than filling memory with scenes that will "
            "never be reused", args.step_minutes, cache_bytes / 2**30)
    else:
        # The navigation grids are cached too; that memory is not
        # available to the scene cache.
        cache_bytes = default_scene_cache_bytes(
            reserve=int(args.memory_reserve_gb * 2**30) + grid_cache_budget())
    available = available_memory_bytes()
    logger.info("Scene cache: %.1f GB (%.1f GB available, %.1f GB reserved "
                "for the working set, %.1f GB for navigation grids)",
                cache_bytes / 2**30, (available or 0) / 2**30,
                args.memory_reserve_gb, grid_cache_budget() / 2**30)
    prefetcher = None
    if args.prefetch_workers > 0:
        prefetcher = ScenePrefetcher(SceneCache(cache_bytes),
                                     max_workers=args.prefetch_workers)
        logger.info("Prefetching scenes on %d worker(s)", args.prefetch_workers)

    # Optional icechunk sink, and the timestamps it already holds (resume)
    repo = None
    icechunk_times: set[datetime] = set()
    if args.icechunk_store:
        repo = open_icechunk_repo(
            args.icechunk_store,
            endpoint_url=args.icechunk_endpoint,
            region=args.icechunk_region,
            anonymous=args.icechunk_anonymous,
            force_path_style=args.icechunk_force_path_style,
        )
        icechunk_times = icechunk_existing_times(repo, args.icechunk_branch)
        if icechunk_times and not args.skip_existing:
            logger.warning(
                "Store already holds %d timestamp(s) and --skip-existing was "
                "not given; timestamps already present will not be appended "
                "again", len(icechunk_times),
            )

    # Load models once and reuse across the whole range
    logger.info("Loading student checkpoint: %s", args.student_ckpt)
    model = StudentWindsModel.load_from_checkpoint(
        args.student_ckpt, map_location=args.device,
    ).eval()

    logger.info("Loading RAFT checkpoint: %s", args.raft_ckpt)
    disp = StereoDisparity(
        model_ckpt_path=args.raft_ckpt,
        tile_size=512, overlap=128, batch_size=8,
        device=args.device,
    )

    n_ok = 0
    failed: list[str] = []
    for i, t0 in enumerate(times, 1):
        logger.info("=== [%d/%d] %s ===", i, len(times), t0.isoformat())
        try:
            ok = process_time(
                t0, sats, model, disp, flow_bands, rad_bands, out_dir,
                device=args.device,
                row_strip=args.row_strip,
                resolution_m=args.resolution_m,
                skip_global=args.skip_global,
                skip_existing=args.skip_existing,
                repo=repo,
                icechunk_branch=args.icechunk_branch,
                icechunk_chunk=args.icechunk_chunk,
                icechunk_times=icechunk_times,
                write_netcdf=not args.no_netcdf,
                global_netcdf=global_netcdf,
                temp_dir=temp_dir,
                keep_temp=args.keep_temp,
                prefetcher=prefetcher,
            )
        except Exception:
            logger.exception("Failed to process %s — continuing", t0.isoformat())
            ok = False
        if ok:
            n_ok += 1
        else:
            failed.append(t0.isoformat())

    if prefetcher is not None:
        logger.info("Final %s", prefetcher.cache.summary())
        prefetcher.shutdown()

    if not args.icechunk_store:
        where = str(out_dir)
    elif global_netcdf:
        where = f"{out_dir} and {args.icechunk_store}"
    else:
        where = str(args.icechunk_store)
    logger.info("Done: %d/%d timestamps produced output in %s",
                n_ok, len(times), where)
    if failed:
        logger.warning("Failed timestamps (%d): %s", len(failed),
                       ", ".join(failed))
    if n_ok == 0:
        logger.error("No timestamps produced output. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
