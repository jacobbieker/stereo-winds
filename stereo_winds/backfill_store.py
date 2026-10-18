"""Per-satellite AMV stores filled in parallel: a preallocated time axis.

The operational writer (:func:`stereo_winds.icechunk_output.write_satellite_to_icechunk`)
appends one scan per commit along ``time``.  That is right for a live
store with one writer, and wrong for a backfill: appends must arrive in
time order, and two writers appending to one repository always conflict
on the ``time`` axis.  A backfill of a year and a half of hourly scans
needs many GPU jobs per satellite writing the same store at once.

So a backfill store is laid out up front instead:

1. :func:`preallocate_satellite_store` creates the whole hourly time axis
   once, with every grid field and per-scan variable allocated but no
   chunk written.  Unwritten chunks read as their fill value (NaN for the
   fields), so a scan that never arrives is simply an empty slot.
2. :func:`fill_slot` writes one scan into its slot with a region write.
   Every array is chunked one slot deep along ``time`` (the 1-D per-scan
   variables too), so writers filling different slots never touch the
   same chunk, and their commits rebase cleanly onto each other
   (``icechunk.ConflictDetector``).  A genuine conflict (two writers on
   the same slot, or both writing the static grid) is retried from a
   fresh session.
3. ``slot_filled`` (int8, 1 once written) says which slots hold a scan.
   A filled slot is skipped unless ``replace`` is set, so a retried job
   only does the slots it has not done yet.

The schema is that of the operational stores -- the same grid fields on
the satellite's native fixed grid, 2-D ``latitude``/``longitude`` and a
static ``zenith_angle`` (written once, by the first scan to land), and the
same per-scan variables -- plus ``slot_filled``.  Readers must skip slots
whose ``slot_filled`` is 0: their ``scan_time`` is empty and every field
is NaN.
"""

from __future__ import annotations

import logging
import random
import time as _time
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from stereo_winds.icechunk_output import (
    _TIME_ENCODING,
    _satellite_with_time,
    _store_has_dataset,
    time_tag,
)

logger = logging.getLogger(__name__)

#: The grid fields every scan carries (the ring script's ``OUTPUT_VARS``).
DEFAULT_FIELDS = (
    "u_wind",
    "v_wind",
    "cloud_top_height",
    "quality_flag",
    "sigma_u",
    "sigma_v",
    "sigma_h",
)

#: The static grid coordinates, written once by the first scan to land.
GRID_COORDS = ("latitude", "longitude", "zenith_angle")

#: Per-scan variables and their stored dtypes; the names are those
#: :func:`_satellite_with_time` derives from the scan's attributes.
PER_SLOT_DTYPES: dict[str, type | str] = {
    "scan_time": object,
    "sub_satellite_longitude": "float64",
    "sub_satellite_longitude_source": object,
    "nominal_sub_satellite_longitude": "float64",
    "satellite_height_m": "float64",
    "quality_degraded": "int64",
    "quality_note": object,
    "bands_requested": object,
    "bands_missing": object,
    "n_bands_requested": "int64",
    "n_bands_missing": "int64",
    "flow_bands_missing": object,
}

#: 1 where a slot holds a scan, 0 where it was never filled.
FILLED_VAR = "slot_filled"

#: Group attribute marking the layout, so a reader or a filler can tell a
#: preallocated store from an appended one.
LAYOUT_ATTR = "time_axis_layout"
LAYOUT = "preallocated"


def slot_times(start: datetime, end: datetime, step_minutes: int = 60) -> pd.DatetimeIndex:
    """The slots from ``start`` to ``end`` inclusive, ``step_minutes`` apart."""
    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive")
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    return pd.date_range(start, end, freq=f"{int(step_minutes)}min")


def _grid_shape(sat_id: str) -> tuple[int, int]:
    from stereo_winds.config import SATELLITE_CONFIGS

    cfg = SATELLITE_CONFIGS[sat_id]
    return int(cfg.n_rows), int(cfg.n_cols)


def _template(
    sat_id: str,
    times: pd.DatetimeIndex,
    shape: tuple[int, int],
    chunk: int,
    fields: tuple[str, ...],
) -> tuple[xr.Dataset, dict]:
    """A lazy (dask) dataset of the whole store and its encoding.

    One dask chunk per variable: nothing is computed (the store is written
    with ``compute=False``), and a dask chunk per zarr chunk would build a
    graph of millions of tasks for a year and a half of full disks.  The
    zarr chunking comes from the encoding.
    """
    import dask.array as da

    n = len(times)
    ny, nx = shape
    cy, cx = min(chunk, ny), min(chunk, nx)
    data_vars: dict[str, tuple] = {}
    encoding: dict[str, dict] = {"time": dict(_TIME_ENCODING)}
    for name in fields:
        data_vars[name] = (
            ("time", "y", "x"),
            da.full((n, ny, nx), np.nan, dtype="float32", chunks=-1),
        )
        encoding[name] = {"chunks": (1, cy, cx), "fill_value": np.nan}
    for name, dtype in PER_SLOT_DTYPES.items():
        if dtype is object:
            arr = da.full((n,), "", dtype=object, chunks=-1)
        elif np.dtype(dtype).kind == "f":
            arr = da.full((n,), np.nan, dtype=dtype, chunks=-1)
        else:
            arr = da.zeros((n,), dtype=dtype, chunks=-1)
        data_vars[name] = (("time",), arr)
        # One slot per chunk: concurrent fillers of different slots must
        # never share a chunk, or their commits conflict.
        encoding[name] = {"chunks": (1,)}
    data_vars[FILLED_VAR] = (("time",), da.zeros((n,), dtype="int8", chunks=-1))
    encoding[FILLED_VAR] = {"chunks": (1,), "fill_value": 0}
    coords = {"time": times.values.astype("datetime64[ns]")}
    for name in GRID_COORDS:
        coords[name] = (
            ("y", "x"),
            da.full((ny, nx), np.nan, dtype="float32", chunks=-1),
        )
        encoding[name] = {"chunks": (cy, cx), "fill_value": np.nan}
    ds = xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "satellite_id": sat_id,
            "source": "student_amv",
            LAYOUT_ATTR: LAYOUT,
            "slot_step_minutes": int(round((times[1] - times[0]).total_seconds() / 60))
            if n > 1
            else 0,
            "slot_note": f"slots with {FILLED_VAR} == 0 hold no scan (all fields NaN)",
        },
    )
    ds[FILLED_VAR].attrs["long_name"] = "1 where this time slot holds a retrieved scan"
    return ds, encoding


def preallocate_satellite_store(
    repo,
    sat_id: str,
    start: datetime,
    end: datetime,
    step_minutes: int = 60,
    branch: str = "main",
    chunk: int = 1024,
    shape: tuple[int, int] | None = None,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
) -> str:
    """Lay out ``sat_id``'s store with one empty slot per time step.

    Only metadata and the time axis are written: a field chunk does not
    exist until a scan is filled into it.  Idempotent: on a store that is
    already laid out with the same time axis this returns ``"exists"``;
    any other existing dataset is an error rather than something to
    overwrite.

    ``shape`` defaults to the satellite's native fixed grid.
    """
    times = slot_times(start, end, step_minutes)
    shape = shape or _grid_shape(sat_id)
    session = repo.writable_session(branch)
    if _store_has_dataset(session):
        ds = xr.open_zarr(session.store, consolidated=False)
        same = (
            ds.attrs.get(LAYOUT_ATTR) == LAYOUT
            and ds.sizes["time"] == len(times)
            and np.array_equal(ds["time"].values.astype("datetime64[ns]"), times.values)
            and (ds.sizes["y"], ds.sizes["x"]) == tuple(shape)
        )
        if not same:
            raise ValueError(
                f"{sat_id}: the store already holds a different dataset "
                f"({dict(ds.sizes)}, layout {ds.attrs.get(LAYOUT_ATTR)!r}); "
                "refusing to overwrite it"
            )
        logger.info("%s: store already preallocated (%d slots)", sat_id, len(times))
        return "exists"
    ds, encoding = _template(sat_id, times, shape, chunk, fields)
    ds.to_zarr(
        session.store,
        mode="w",
        compute=False,
        consolidated=False,
        zarr_format=3,
        encoding=encoding,
    )
    session.commit(
        f"preallocate {sat_id} {time_tag(times[0])}..{time_tag(times[-1])} "
        f"every {step_minutes} min ({len(times)} slots)"
    )
    logger.info(
        "%s: preallocated %d slots %s .. %s on a %dx%d grid",
        sat_id,
        len(times),
        time_tag(times[0]),
        time_tag(times[-1]),
        shape[0],
        shape[1],
    )
    return "created"


def is_preallocated(repo, branch: str = "main") -> bool:
    """True if the store at ``repo`` uses the preallocated layout."""
    try:
        ds = xr.open_zarr(repo.readonly_session(branch).store, consolidated=False)
    except Exception:
        return False
    return ds.attrs.get(LAYOUT_ATTR) == LAYOUT


def slot_status(repo, branch: str = "main") -> dict[datetime, bool]:
    """Every slot's nominal time and whether it is filled."""
    ds = xr.open_zarr(repo.readonly_session(branch).store, consolidated=False)
    times = pd.DatetimeIndex(ds["time"].values)
    filled = np.asarray(ds[FILLED_VAR].values) != 0
    return {t.to_pydatetime(): bool(f) for t, f in zip(times, filled)}


def _slot_index(ds: xr.Dataset, t0: datetime) -> int | None:
    times = np.asarray(ds["time"].values, "datetime64[ns]")
    hits = np.flatnonzero(times == np.datetime64(t0, "ns"))
    return int(hits[0]) if hits.size else None


def _grid_written(session, ds: xr.Dataset) -> bool:
    """True once some scan has written the static grid.

    Read at the disk centre, which is on the disk for every geostationary
    imager, so an unwritten grid (all fill value) is unambiguous.
    """
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    centre = ds["latitude"].isel(y=ny // 2, x=nx // 2).values
    return bool(np.isfinite(centre))


def _region_dataset(ds: xr.Dataset, store_ds: xr.Dataset, sat_id: str) -> xr.Dataset:
    """The time-dimensioned part of one scan, matched to the store's schema."""
    region = ds.drop_vars([n for n in ds.variables if "time" not in ds[n].dims])
    region = region.drop_vars("time")
    extra = [n for n in region.data_vars if n not in store_ds.variables]
    if extra:
        logger.warning(
            "%s: store has no %s; not writing them", sat_id, ", ".join(sorted(extra))
        )
        region = region.drop_vars(extra)
    for name in list(region.data_vars):
        want = store_ds[name].dtype
        have = region[name]
        if have.dims[1:] != store_ds[name].dims[1:] or have.shape[1:] != store_ds[name].shape[1:]:
            raise ValueError(
                f"{sat_id}: {name} is {dict(zip(have.dims, have.shape))} but the "
                f"store's slots are {dict(zip(store_ds[name].dims, store_ds[name].shape))}"
            )
        if want.kind == "O" or want.kind in "US":
            region[name] = have.astype(str).astype(object)
        elif have.dtype != want:
            region[name] = have.astype(want)
    region[FILLED_VAR] = ("time", np.array([1], dtype="int8"))
    for var in region.variables.values():
        var.encoding = {}
        var.attrs = {}
    return region


def fill_slot(
    repo,
    ds_sat: xr.Dataset,
    t0: datetime,
    branch: str = "main",
    replace: bool = False,
    max_attempts: int = 30,
    backoff_s: float = 2.0,
) -> str:
    """Write one scan into its preallocated slot as one commit.

    Safe to run from many processes at once on different slots: the
    commit rebases over whatever the others committed meanwhile, and a
    real conflict (the same slot, or the static grid written twice)
    restarts from a fresh session, which then sees the other writer's
    work -- an already-filled slot is skipped, an already-written grid is
    not written again.

    Returns ``"filled"``, ``"replaced"`` or ``"skipped"``.  Raises
    ``KeyError`` when ``t0`` is not one of the store's slots.
    """
    import icechunk

    sat_id = str(ds_sat.attrs.get("satellite_id", "?"))
    ds = _satellite_with_time(ds_sat, t0)
    for attempt in range(1, max_attempts + 1):
        session = repo.writable_session(branch)
        store_ds = xr.open_zarr(session.store, consolidated=False)
        if store_ds.attrs.get(LAYOUT_ATTR) != LAYOUT:
            raise ValueError(f"{sat_id}: the store is not preallocated; run --preallocate first")
        idx = _slot_index(store_ds, t0)
        if idx is None:
            raise KeyError(f"{sat_id}: {time_tag(t0)} is not a slot of this store")
        was_filled = bool(store_ds[FILLED_VAR].isel(time=idx).values)
        if was_filled and not replace:
            logger.info("%s %s: slot already filled; leaving it alone", sat_id, time_tag(t0))
            return "skipped"

        region = _region_dataset(ds, store_ds, sat_id)
        region.to_zarr(session.store, region={"time": slice(idx, idx + 1)}, consolidated=False)
        grid_now = False
        if not _grid_written(session, store_ds):
            grid = xr.Dataset({n: ds_sat[n] for n in GRID_COORDS if n in ds_sat.variables})
            for var in grid.variables.values():
                var.encoding = {}
                var.attrs = {}
            grid = grid.reset_coords().astype("float32")
            grid.to_zarr(
                session.store,
                region={"y": slice(0, store_ds.sizes["y"]), "x": slice(0, store_ds.sizes["x"])},
                consolidated=False,
            )
            grid_now = True
        message = f"student AMV {sat_id} {time_tag(t0)} (slot {idx})"
        if grid_now:
            message += " + grid"
        try:
            commit = session.commit(message, rebase_with=icechunk.ConflictDetector())
        except (icechunk.ConflictError, icechunk.RebaseFailedError) as exc:
            delay = backoff_s * min(attempt, 10) * (0.5 + random.random())
            logger.info(
                "%s %s: commit conflicted (%s); retrying in %.1f s (attempt %d/%d)",
                sat_id,
                time_tag(t0),
                type(exc).__name__,
                delay,
                attempt,
                max_attempts,
            )
            _time.sleep(delay)
            continue
        outcome = "replaced" if was_filled else "filled"
        logger.info("%s %s: %s slot %d (%s)", sat_id, time_tag(t0), outcome, idx, commit)
        return outcome
    raise RuntimeError(
        f"{sat_id} {time_tag(t0)}: still conflicting after {max_attempts} attempts"
    )
