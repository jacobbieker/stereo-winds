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

import datetime as dt
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from stereo_winds.readers._geos_meta import scene_orbital_parameters

logger = logging.getLogger(__name__)


class SceneNotInStore(LookupError):
    """The icechunk store has no scan close enough to the requested time."""


def default_cache_dir() -> Path:
    """Where downloaded L1b files are kept between runs."""
    env = os.environ.get("STEREO_WINDS_DATA_DIR")
    base = Path(env) if env else Path.home() / ".cache" / "stereo_winds"
    return base / "l1b"


def s3_filesystem():
    """Anonymous handle on the public NOAA buckets."""
    import s3fs

    return s3fs.S3FileSystem(anon=True)


def download_keys(fs, keys: list[str], cache_dir: Path) -> list[Path]:
    """Fetch ``keys`` into ``cache_dir``, skipping anything already there."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    fetched = 0
    for key in keys:
        local = cache_dir / key.rsplit("/", 1)[-1]
        if not local.exists() or local.stat().st_size == 0:
            tmp = local.with_suffix(local.suffix + ".part")
            fs.get(key, str(tmp))
            tmp.replace(local)
            fetched += 1
        paths.append(local)
    if fetched:
        logger.info("  downloaded %d file(s) to %s", fetched, cache_dir)
    return paths


def load_scene_array(reader: str, paths: list[Path], band: str):
    """Load one band from local L1b files and return the satpy DataArray."""
    from satpy import Scene

    scn = Scene(reader=reader, filenames=[str(p) for p in paths])
    scn.load([band])
    return scn[band]


def _area_info(da: Any) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """Cell-centre x/y (metres, row 0 = north) and the CRS dict, from satpy."""
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
    try:
        crs = dict(area.crs.to_dict())
    except Exception:  # pragma: no cover - depends on pyproj internals
        crs = None
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
    if isinstance(da.attrs.get("orbital_parameters"), dict):
        meta.attrs["orbital_parameters"] = da.attrs["orbital_parameters"]
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
    return out
