"""Public-S3 + satpy fallback for the AHI and AMI icechunk readers.

The source.coop icechunk stores cover only part of each satellite's
record (the AMI store, for instance, holds three weeks of January 2025).
Outside that, the native Level-1b files are still on NOAA's open buckets:

- ``s3://noaa-himawari8`` / ``s3://noaa-himawari9`` — AHI HSD segments,
  ``AHI-L1b-FLDK/YYYY/MM/DD/HHMM/HS_H09_<date>_<time>_<band>_FLDK_R<res>_S<seg>10.DAT.bz2``
  (ten segments per band), read by satpy's ``ahi_hsd``
- ``s3://noaa-gk2a-pds`` — AMI L1b netCDF,
  ``AMI/L1B/FD/YYYYMM/DD/HH/gk2a_ami_le1b_<band>_fd<res>ge_<stamp>.nc``
  (one file per band), read by satpy's ``ami_l1b``

This module downloads those files and converts a loaded satpy scene into
the same shape the icechunk readers return: ``Rad`` as ``(time, band, y,
x)`` with x/y in metres running west->east and south->north, and
``Rad.attrs["orbital_parameters"]`` carrying the projection actually
recorded for the scene.

Requires ``satpy`` and ``s3fs``.
"""

from __future__ import annotations

import bz2
import contextlib
import datetime as dt
import logging
import os
import shutil
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from stereo_winds.readers._cache import default_cache_dir, download_workers
from stereo_winds.readers._geos_meta import (
    scene_ellipsoid,
    scene_orbital_parameters,
)

# GRS80, when the scene's CRS states no ellipsoid of its own.
_GRS80_SEMI_MAJOR = 6378137.0
_GRS80_SEMI_MINOR = 6356752.31414

logger = logging.getLogger(__name__)


class SceneNotInStore(LookupError):
    """The icechunk store has no scan close enough to the requested time."""


def s3_filesystem():
    """Anonymous handle on the public NOAA buckets."""
    import s3fs

    return s3fs.S3FileSystem(anon=True)


def download_keys(
    fs, keys: list[str], cache_dir: Path, max_workers: int | None = None,
) -> list[Path]:
    """Fetch ``keys`` into ``cache_dir``, skipping anything already there.

    Downloads run concurrently: an AHI band is ten separate segment
    objects, and fetching them one after another leaves the link idle
    between round trips.  Order of the returned paths matches ``keys``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [cache_dir / key.rsplit("/", 1)[-1] for key in keys]
    todo = [(key, local) for key, local in zip(keys, paths)
            if not local.exists() or local.stat().st_size == 0]
    if not todo:
        return paths

    def fetch(item):
        key, local = item
        # Download to a sidecar first so an interrupted run cannot leave a
        # truncated file that later looks like a cache hit.
        tmp = local.with_suffix(local.suffix + f".part{os.getpid()}")
        try:
            fs.get(key, str(tmp))
            tmp.replace(local)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return local

    workers = max_workers or download_workers()
    if workers > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as pool:
            # list() re-raises the first failure once all are done.
            list(pool.map(fetch, todo))
    else:
        for item in todo:
            fetch(item)
    logger.info("  downloaded %d file(s) to %s (%d worker%s)",
                len(todo), cache_dir, min(workers, len(todo)),
                "" if min(workers, len(todo)) == 1 else "s")
    return paths


@contextlib.contextmanager
def _quiet_nan_arithmetic():
    """Silence the expected NaN arithmetic in satpy's IR calibration.

    Off-disk pixels carry no counts, so the radiance -> brightness
    temperature conversion takes the log of NaN (and of the odd negative
    radiance) for every space pixel.  NaN in, NaN out is the intended
    result, but numpy warns per chunk and dask surfaces it from a worker
    thread.  Filters are process-wide, which is what reaches those
    threads; the scope is one satpy load.
    """
    with warnings.catch_warnings():
        for message in ("invalid value encountered in log",
                        "divide by zero encountered in log",
                        "invalid value encountered in divide"):
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                    message=message)
        yield


def _decompress(paths: list[Path], scratch: Path) -> list[Path]:
    """Expand any bz2 inputs into ``scratch``; pass other files through.

    Doing this ourselves rather than letting satpy do it matters under
    concurrency: satpy decompresses into ``satpy.config["tmp_dir"]``,
    which is process-global, so parallel loads would land in each
    other's scratch directories and delete files still being read.
    Segments expand independently, so they expand together.
    """
    resolved: list[Path] = []
    todo: list[tuple[Path, Path]] = []
    for path in paths:
        if path.suffix == ".bz2":
            target = scratch / path.with_suffix("").name
            todo.append((path, target))
            resolved.append(target)
        else:
            resolved.append(path)
    if not todo:
        return resolved

    def expand(item: tuple[Path, Path]) -> None:
        source, target = item
        with bz2.open(source, "rb") as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)

    workers = min(len(todo), download_workers())
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(expand, todo))
    else:
        for item in todo:
            expand(item)
    return resolved


def load_scene_array(
    reader: str, paths: list[Path], band: str,
    scratch_dir: Path | None = None,
):
    """Load one band from local L1b files, returning an in-memory DataArray.

    AHI ships its HSD segments bz2-compressed.  They are expanded into a
    scratch directory of our own and removed as soon as the scene is
    read, so nothing accumulates: satpy's own decompression writes to
    ``/tmp`` and only unlinks in a finalizer that runs at garbage
    collection, which across a long run fills the disk.

    The data is materialised before the scratch directory goes away —
    the returned array must not be lazily backed by files we are about
    to delete.
    """
    from satpy import Scene

    base = Path(scratch_dir) if scratch_dir else default_cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="satpy-scratch-", dir=str(base)))
    try:
        local = _decompress(list(paths), tmp_dir)
        with _quiet_nan_arithmetic():
            scn = Scene(reader=reader, filenames=[str(p) for p in local])
            scn.load([band])
            da = scn[band].compute()
        del scn
        return da
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# CF grid-mapping key -> the proj-style key the metadata reader expects.
_CF_TO_PROJ = {
    "longitude_of_projection_origin": "lon_0",
    "perspective_point_height": "h",
    "semi_major_axis": "a",
    "semi_minor_axis": "b",
    "inverse_flattening": "rf",
    "sweep_angle_axis": "sweep",
}


def _crs_parameters(area: Any) -> dict | None:
    """Projection parameters from a pyresample area, as a proj-style dict.

    Read through ``CRS.to_cf()`` rather than ``CRS.to_dict()``: the latter
    round-trips via a PROJ string and warns "you will likely lose
    important projection information" on every single scene.
    """
    try:
        cf = area.crs.to_cf()
    except Exception:  # pragma: no cover - depends on pyproj internals
        return None
    params = {proj_key: cf[cf_key]
              for cf_key, proj_key in _CF_TO_PROJ.items() if cf_key in cf}
    if cf.get("grid_mapping_name") == "geostationary":
        params["proj"] = "geos"
    return params or None


def _area_info(da: Any, want_crs: bool = True) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """Cell-centre x/y (metres, row 0 = north) and the CRS, from satpy."""
    area = da.attrs.get("area")
    if area is None:
        raise ValueError("satpy scene carries no area definition")
    ny, nx = da.shape[-2], da.shape[-1]
    ll_x, ll_y, ur_x, ur_y = (float(v) for v in area.area_extent)
    dx = (ur_x - ll_x) / nx
    dy = (ur_y - ll_y) / ny
    x = ll_x + (np.arange(nx) + 0.5) * dx
    # pyresample puts row 0 at the upper (north) edge of the extent.
    y_top_down = ur_y - (np.arange(ny) + 0.5) * dy
    crs = _crs_parameters(area) if want_crs else None
    return x.astype(np.float64), y_top_down.astype(np.float64), crs


def scene_to_rad(
    da: Any,
    band: str,
    *,
    sweep: str,
    fallback_sub_lon: float,
    fallback_height: float,
    label: str = "",
) -> xr.Dataset:
    """Convert a satpy DataArray into the reader's standard Rad dataset.

    Output matches the icechunk path exactly: ``Rad`` shaped
    ``(1, 1, y, x)``, x ascending west->east, y ascending south->north,
    coordinates in metres, and satpy's own projection metadata.
    """
    values = np.asarray(da.values, dtype=np.float32)
    if values.ndim > 2:
        values = values.reshape(values.shape[-2:])
    orbital_attr = da.attrs.get("orbital_parameters")
    # The CRS is only a fallback for the projection, but it is always the
    # authority on the ellipsoid.
    x_m, y_m, crs = _area_info(da)

    # Normalise to ascending axes, moving the data with the coordinates.
    if x_m[0] > x_m[-1]:
        x_m = x_m[::-1]
        values = values[:, ::-1]
    if y_m[0] > y_m[-1]:
        y_m = y_m[::-1]
        values = values[::-1, :]

    # Reuse the store metadata reader by handing it satpy's attrs.
    meta = xr.Dataset(attrs={})
    if isinstance(orbital_attr, dict):
        meta.attrs["orbital_parameters"] = orbital_attr
    if crs is not None:
        meta.attrs["area"] = {"projection": crs}
    orbital = scene_orbital_parameters(
        meta,
        fallback_sub_lon=fallback_sub_lon,
        fallback_height=fallback_height,
        label=label,
    )

    Rad = xr.DataArray(
        values[None, None, :, :],
        dims=("time", "band", "y", "x"),
        coords={"x": ("x", x_m), "y": ("y", y_m)},
        name="Rad",
    )
    Rad.attrs["orbital_parameters"] = orbital

    start = da.attrs.get("start_time")
    end = da.attrs.get("end_time", start)
    if isinstance(start, dt.datetime):
        Rad.attrs["time_coverage_start"] = start.isoformat(timespec="seconds")
    if isinstance(end, dt.datetime):
        Rad.attrs["time_coverage_end"] = end.isoformat(timespec="seconds")

    out = xr.Dataset({"Rad": Rad})
    out.attrs["sweep_angle_axis"] = sweep
    out.attrs["source"] = "public S3 L1b via satpy"
    semi_major, semi_minor = scene_ellipsoid(
        meta, fallback_semi_major=_GRS80_SEMI_MAJOR,
        fallback_semi_minor=_GRS80_SEMI_MINOR,
    )
    out.attrs["ellipsoid"] = {
        "semi_major_m": semi_major, "semi_minor_m": semi_minor,
    }
    return out
