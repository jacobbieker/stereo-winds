"""Minimal GOES ABI L1b reader (no satpy, no auth).

Reads GOES-16/17/18/19 ABI Level-1b radiance from the virtualized icechunk
stores on source.coop (``geo/virtualized/goes19_radf_C13.icechunk`` ...,
whose chunks reference the L1b objects in NOAA's buckets), falling back to
NOAA's public S3 buckets (``noaa-goes16`` ...) themselves, and returns a dataset shaped exactly like the internal
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
import threading
import time
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


#: Reader product -> the ``<product>`` field of a virtualized store name.
_VIRTUAL_PRODUCT = {"ABI-L1b-RadF": "radf", "ABI-L1b-RadC": "radc", "ABI-L1b-RadM": "radm"}

# Opened virtualized stores, keyed by prefix: (dataset, scan starts,
# when opened).  Opening one reads a time axis of ~10^4 entries (a few
# seconds cold), and a scene reads several bands from several frames, so
# without this every band of every frame would pay it again.  Shared
# across readers and threads, as _geos_store._OPEN_DATASETS is.
_VIRTUAL_OPEN: dict[str, tuple[xr.Dataset, np.ndarray, float]] = {}
_VIRTUAL_LOCK = threading.Lock()
_VIRTUAL_PREFIX_LOCKS: dict[str, threading.Lock] = {}

#: The undated stores are still appended to.  A cached handle older than
#: this is reopened when asked for a time past its last scan, so a
#: long-running process picks up new scans rather than falling back to S3
#: for them forever.
VIRTUAL_REFRESH_SECONDS = 600.0


def _iso_utc(value) -> str:
    """``2026-09-15T12:00:21.800Z``: the form the L1b files carry."""
    return str(np.datetime_as_string(np.datetime64(value, "ms"), unit="ms")) + "Z"


def clear_virtual_cache() -> None:
    """Drop cached virtualized-store handles (tests, or to force a reopen)."""
    with _VIRTUAL_LOCK:
        _VIRTUAL_OPEN.clear()


def _open_virtual(
    prefix: str, target: np.datetime64, tolerance: np.timedelta64
) -> tuple[xr.Dataset, np.ndarray]:
    """The store at ``prefix`` and its scan start times, cached.

    Reopened only when the cached copy is stale *and* ``target`` lies
    past its last scan -- the one case a newer snapshot could change the
    answer.
    """
    from stereo_winds.readers._geos_store import GeoStoreReader
    from stereo_winds.readers import _virtual_store as virtual

    def _usable(cached):
        if cached is None:
            return False
        _, starts, opened = cached
        stale = time.monotonic() - opened > VIRTUAL_REFRESH_SECONDS
        beyond = starts.size == 0 or target > starts.max() + tolerance
        return not (stale and beyond)

    with _VIRTUAL_LOCK:
        cached = _VIRTUAL_OPEN.get(prefix)
        if _usable(cached):
            return cached[0], cached[1]
        # One lock per store: an open is a network round trip of
        # seconds, and holding the shared lock across it would queue
        # every other band and satellite -- and a hung source.coop would
        # hold up the S3 fallback for all of them.
        prefix_lock = _VIRTUAL_PREFIX_LOCKS.setdefault(prefix, threading.Lock())
    with prefix_lock:
        with _VIRTUAL_LOCK:
            cached = _VIRTUAL_OPEN.get(prefix)
        if _usable(cached):
            return cached[0], cached[1]
        logger.info("Opening virtualized store %s", prefix)
        ds = virtual.open_virtual_dataset(GeoStoreReader.bucket, GeoStoreReader.endpoint, prefix)
        starts = np.asarray(virtual._scan_start(ds), dtype="datetime64[ns]")
        with _VIRTUAL_LOCK:
            _VIRTUAL_OPEN[prefix] = (ds, starts, time.monotonic())
        return ds, starts


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

        The virtualized icechunk tier is asked first and the public
        bucket second.  The virtualized stores reference the same L1b
        objects in the NOAA bucket, so what comes back is the same scene
        either way; the tier only saves the bucket listing and the
        NetCDF download.  Once a store's time axis has been read (it is
        cached per process, see :func:`_open_virtual`) a band costs a few
        ranged reads.

        Anything the virtualized tier cannot serve -- no store for the
        band, a time outside its coverage, source.coop unreachable, a
        store that fails to open -- falls through to the S3 path, whose
        "No ABI file on S3" is the error raised when neither has it.
        """
        band = self.bands[0]
        try:
            scene = self._virtual_scene(t, band)
        except Exception:
            logger.info(
                "virtualized tier failed for %s %s at %s; trying S3",
                self.satellite,
                band,
                t,
                exc_info=True,
            )
            scene = None
        if scene is not None:
            return scene

        key = self._find_key(t, band)
        logger.info("%s %s at %s: read from S3 %s", self.satellite, band, t, key)
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

    def _virtual_prefixes(self, band: str, when: dt.datetime | None) -> list[str]:
        """Virtualized stores for this satellite, product and band.

        Newest first, with the undated (still appended) store ahead of
        the dated snapshots; stores cut before ``when`` are left out.
        Only stores of this reader's product: a CONUS request must not
        be answered from a full-disk store whose scan happens to start
        within the tolerance.
        """
        from stereo_winds.readers._geos_store import GeoStoreReader
        from stereo_winds.readers import _virtual_store as virtual

        wanted = _VIRTUAL_PRODUCT.get(self.product)
        if wanted is None:
            return []
        prefixes = virtual.virtual_store_for(
            GeoStoreReader.bucket, GeoStoreReader.endpoint, self.satellite, band, when=when
        )
        out = []
        for prefix in prefixes:
            parsed = virtual.parse_store_name(prefix.rsplit("/", 1)[-1])
            if parsed is not None and parsed.product.lower() == wanted:
                out.append(prefix)
        return out

    def _virtual_scene(self, t: dt.datetime, band: str) -> xr.Dataset | None:
        """The scene from the virtualized tier, or None if it has none.

        Each store that might cover ``t`` is tried in turn; one that
        fails to open or read is logged and skipped rather than raised,
        so that the caller goes on to the public bucket.
        """
        try:
            prefixes = self._virtual_prefixes(band, t.replace(tzinfo=None))
        except Exception:
            logger.info(
                "virtualized tier unreachable for %s %s", self.satellite, band, exc_info=True
            )
            return None

        target = np.datetime64(self._snap_time(t).replace(tzinfo=None), "ns")
        # Half a slot: the tier must not quietly hand back a
        # neighbouring scan, because the retrieval takes the interval
        # between frames as its baseline.
        tolerance = np.timedelta64(int(self.step * 30), "s")
        for prefix in prefixes:
            try:
                ds, starts = _open_virtual(prefix, target, tolerance)
                if starts.size == 0:
                    continue
                # `t` in an ABI store is the mid-scan instant; the slot
                # is named by when the scan started.
                i = int(np.argmin(np.abs(starts - target)))
                if abs(starts[i] - target) > tolerance:
                    continue
                snap = ds.isel(t=i)
                # _abi_to_satpy_like reads the Planck coefficients with
                # float(); they carry a length-1 axis here.
                for name in ("planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2"):
                    if name in snap:
                        snap[name] = snap[name].squeeze(drop=True)
                scene = _abi_to_satpy_like(snap)
                # The store's global attributes describe the store, not
                # this scan; the per-pixel scan-time model reads these,
                # so they must be this scan's own, as on the S3 path.
                if "time_bounds" in snap:
                    bounds = np.asarray(snap["time_bounds"].values).ravel()
                    if bounds.size == 2:
                        for key, value in zip(("time_coverage_start", "time_coverage_end"), bounds):
                            scene["Rad"].attrs[key] = _iso_utc(value)
                else:
                    scene["Rad"].attrs["time_coverage_start"] = _iso_utc(starts[i])
                    scene["Rad"].attrs.pop("time_coverage_end", None)
                logger.info(
                    "%s %s at %s: read from virtualized store %s",
                    self.satellite,
                    band,
                    target,
                    prefix,
                )
                return scene
            except Exception:
                logger.info("virtualized store %s did not serve %s", prefix, band, exc_info=True)
        return None

    def virtual_times(self, band: str, start: dt.datetime, end: dt.datetime) -> np.ndarray:
        """Scan start times the virtualized tier holds within [start, end].

        Empty when the tier is unreachable: availability then rests on
        the public bucket alone, as it did before the tier existed.
        """
        lo = np.datetime64(start.replace(tzinfo=None), "ns")
        hi = np.datetime64(end.replace(tzinfo=None), "ns")
        found = []
        try:
            prefixes = self._virtual_prefixes(band, start.replace(tzinfo=None))
        except Exception:
            logger.info("virtualized tier unreachable for %s %s", self.satellite, band)
            return np.array([], dtype="datetime64[ns]")
        for prefix in prefixes:
            try:
                _, starts = _open_virtual(prefix, hi, np.timedelta64(0, "s"))
            except Exception:
                logger.info("virtualized store %s could not be opened", prefix, exc_info=True)
                continue
            found.append(starts[(starts >= lo) & (starts <= hi)])
        if not found:
            return np.array([], dtype="datetime64[ns]")
        return np.unique(np.concatenate(found))

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
