"""Minimal public-S3 GOES ABI L1b reader (no satpy, no auth).

Reads GOES-16/17/18/19 ABI Level-1b radiance from NOAA's public S3 buckets
(``noaa-goes16`` ...) and returns a dataset shaped exactly like the internal
satpy-based reader stereo-winds was built against: ``Rad`` as ``(time, band,
y, x)`` oriented south->north, ``x``/``y`` in **meters** (scan angle x
perspective height), and ``Rad.attrs["orbital_parameters"]`` carrying
``projection_altitude`` and ``satellite_nominal_longitude``.

This is the standalone replacement for the zeus GOES data source. Only the
native fixed-grid, single-band path used by the stereo solver is implemented
(no reprojection / HEALPix / multi-band resampling).

Requires ``s3fs`` (and an xarray NetCDF backend: ``h5netcdf`` or ``netCDF4``).
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from stereo_winds.readers._cache import default_retention, prune_cache

logger = logging.getLogger(__name__)

PRODUCT_TIMESTEPS = {"ABI-L1b-RadF": 10, "ABI-L1b-RadC": 5, "ABI-L1b-RadM": 1}
_GNUM = {"goes16": "16", "goes17": "17", "goes18": "18", "goes19": "19"}


def _abi_to_satpy_like(raw: xr.Dataset) -> xr.Dataset:
    """Reshape a raw ABI L1b NetCDF into the satpy-style dataset the solver
    pipeline consumes (see module docstring for the exact contract)."""
    proj = raw["goes_imager_projection"].attrs
    height = float(proj["perspective_point_height"])
    sub_lon = float(proj["longitude_of_projection_origin"])

    x_rad = raw["x"].values.astype(np.float64)  # west->east, increasing
    y_rad = raw["y"].values.astype(np.float64)  # north->south, decreasing
    rad = np.asarray(raw["Rad"].values, dtype=np.float32)  # (y, x), row 0 = north

    # Flip to south->north (y ascending) to match the satpy convention.
    y_rad_asc = y_rad[::-1]
    rad_sn = rad[::-1, :]

    x_m = x_rad * height
    y_m = y_rad_asc * height

    Rad = xr.DataArray(
        rad_sn[None, None, :, :],
        dims=("time", "band", "y", "x"),
        coords={"x": ("x", x_m), "y": ("y", y_m)},
        name="Rad",
    )
    Rad.attrs["orbital_parameters"] = {
        "projection_altitude": height,
        "satellite_nominal_longitude": sub_lon,
        "projection_longitude": sub_lon,
    }
    for k in ("time_coverage_start", "time_coverage_end"):
        if k in raw.attrs:
            Rad.attrs[k] = raw.attrs[k]
    # Planck constants (emissive bands only) so downstream can convert
    # radiance -> brightness temperature without re-reading the file.
    for k in ("planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2"):
        if k in raw:
            Rad.attrs[k] = float(raw[k].values)
    ds = xr.Dataset({"Rad": Rad})
    ds.attrs["sweep_angle_axis"] = proj.get("sweep_angle_axis", "x")
    return ds


class GOES:
    """Public-S3 GOES ABI L1b reader (drop-in for the zeus GOES source).

    Parameters
    ----------
    satellite : "goes16" | "goes17" | "goes18" | "goes19"
    product : "ABI-L1b-RadF" (full disk), "ABI-L1b-RadC" (CONUS), "ABI-L1b-RadM"
    bands : list with a single ABI band, e.g. ``["C14"]``
    cache_dir : local download cache (default ~/.cache/stereo_winds)
    cache_retention : drop cached scans more than this far before the
        scene being read (default one hour; None keeps everything)
    """

    def __init__(
        self,
        satellite="goes16",
        product="ABI-L1b-RadF",
        bands=None,
        cache_dir=None,
        cache_retention=-1,
    ):
        self.satellite = satellite
        self.product = product
        self.bands = list(bands) if bands else ["C13"]
        self.cache_dir = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "stereo_winds")
        self.bucket = f"noaa-{satellite}"
        self.step = PRODUCT_TIMESTEPS.get(product, 10)
        # Downloads older than this (relative to the scene being read) are
        # dropped; None keeps everything.  A full-disk ABI band is ~17 MB,
        # so an unpruned cache fills the disk over a long run.
        self.cache_retention = default_retention() if cache_retention == -1 else cache_retention
        self._fs = None

    @property
    def fs(self):
        if self._fs is None:
            import s3fs

            self._fs = s3fs.S3FileSystem(anon=True)
        return self._fs

    def _snap_time(self, t: dt.datetime) -> dt.datetime:
        return t.replace(minute=(t.minute // self.step) * self.step, second=0, microsecond=0)

    def _find_key(self, t: dt.datetime, band: str) -> str:
        t = self._snap_time(t)
        gnum = _GNUM[self.satellite]
        # RadC/RadM scans start +1 minute past the nominal slot.
        ts = t + dt.timedelta(minutes=1) if self.product != "ABI-L1b-RadF" else t
        prefix = f"{self.bucket}/{self.product}/{t:%Y/%j/%H}"
        pattern = f"{prefix}/OR_{self.product}-M*{band}_G{gnum}_s{ts:%Y%j%H%M}*"
        matches = self.fs.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No ABI file on S3: {pattern}")
        return matches[0]

    def data_at_time(self, t: dt.datetime, download: bool = True, **_) -> xr.Dataset:
        """Return the ABI scene at (snapped) time ``t`` for ``self.bands[0]``.

        Falls back to the virtualized icechunk tier when the public
        bucket has no file for the slot.  That gap is real and recurring
        -- a sweep over a week of the ring lost GOES-18 and GOES-19 for a
        whole day to ``No ABI file on S3`` -- and the virtualized stores
        reference the same objects, so what comes back is the same
        scene, not a substitute.
        """
        band = self.bands[0]
        try:
            key = self._find_key(t, band)
        except FileNotFoundError:
            scene = self._virtual_scene(t, band)
            if scene is None:
                raise
            return scene
        if download:
            local = self.cache_dir / self.satellite / Path(key).name
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                self.fs.get(key, str(local))
            self.prune_download_cache(t)
            raw = xr.open_dataset(local)
        else:
            raw = xr.open_dataset(self.fs.open(key))
        try:
            return _abi_to_satpy_like(raw)
        finally:
            raw.close()

    def _virtual_scene(self, t: dt.datetime, band: str) -> xr.Dataset | None:
        """The scene from the virtualized tier, or None if it has none.

        Returns None rather than raising so the caller re-raises the
        original S3 error: "no file on S3" is the useful message when
        neither source has the scan, and a failure to reach source.coop
        should not mask it.
        """
        from stereo_winds.readers._geos_store import GeoStoreReader
        from stereo_winds.readers import _virtual_store as virtual

        try:
            prefixes = virtual.virtual_store_for(
                GeoStoreReader.bucket, GeoStoreReader.endpoint, self.satellite, band
            )
        except Exception:
            logger.info("virtualized tier unreachable for %s %s", self.satellite, band)
            return None

        target = np.datetime64(self._snap_time(t).replace(tzinfo=None), "ns")
        # Half a slot: the fallback must not quietly hand back a
        # neighbouring scan, because the retrieval takes the interval
        # between frames as its baseline.
        tolerance = np.timedelta64(int(self.step * 30), "s")
        for prefix in prefixes:
            try:
                ds = virtual.open_virtual_dataset(
                    GeoStoreReader.bucket, GeoStoreReader.endpoint, prefix
                )
                # `t` in an ABI store is the mid-scan instant; the slot
                # is named by when the scan started.
                starts = virtual._scan_start(ds)
                i = int(np.argmin(np.abs(starts - target)))
                if abs(starts[i] - target) > tolerance:
                    continue
                snap = ds.isel(t=i)
                # _abi_to_satpy_like reads the Planck coefficients with
                # float(); they carry a length-1 axis here.
                for name in ("planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2"):
                    if name in snap:
                        snap[name] = snap[name].squeeze(drop=True)
                logger.info(
                    "%s %s at %s: no file on S3, read from %s",
                    self.satellite,
                    band,
                    target,
                    prefix,
                )
                return _abi_to_satpy_like(snap)
            except Exception:
                logger.info("virtualized store %s did not serve %s", prefix, band, exc_info=True)
        return None

    def download(self, t: dt.datetime) -> list[Path]:
        """Download the ABI file(s) for ``self.bands`` at ``t``; return paths."""
        paths = []
        for band in self.bands:
            key = self._find_key(t, band)
            local = self.cache_dir / self.satellite / Path(key).name
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                self.fs.get(key, str(local))
            paths.append(local)
        self.prune_download_cache(t)
        return paths

    def prune_download_cache(self, t: dt.datetime) -> None:
        """Drop cached scans more than ``cache_retention`` before ``t``."""
        if self.cache_retention is None:
            return
        prune_cache(self.cache_dir / self.satellite, t - self.cache_retention)

    def __repr__(self):
        return (
            f"GOES(satellite={self.satellite!r}, product={self.product!r}, "
            f"bands={self.bands!r})"
        )
