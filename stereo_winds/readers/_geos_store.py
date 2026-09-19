"""Shared implementation for the icechunk-backed geostationary readers.

AHI, AMI, FCI and SEVIRI are all read the same way: open the store for
the band's resolution tier, pick the scan nearest the requested time,
pull out a 2-D array, and hand back ``Rad`` as ``(time, band, y, x)``
with x/y in metres running west->east and south->north.  What differs
between instruments is data, not logic — band names, store prefixes,
grid sizes, scan cadence — so it lives in class attributes on the
subclasses and the behaviour lives here.

A subclass declares:

``sub_lon`` / ``sat_height``
    Nominal projection origin and perspective height, as a scalar or a
    dict keyed by satellite id.  Fallbacks only: the store's own
    metadata wins whenever it has any.
``band_resolution``
    Native band name -> store resolution tier, which also defines the
    set of valid native names.
``abi_to_native``
    ABI band name -> native equivalent, so callers can ask in ABI terms.
``store_template`` or ``store_prefixes``
    Where the icechunk store lives.
``scan_interval_minutes``
    Full-disk repeat cycle, which sets how far the nearest-scan lookup
    may reach before it reports a miss.

Readers with a public-S3 fallback set ``supports_s3_fallback`` and
implement ``_s3_data_at_time``; the rest surface ``SceneNotInStore``.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from stereo_winds.readers._geos_meta import (
    scene_ellipsoid,
    scene_orbital_parameters,
)
from stereo_winds.readers._cache import (
    default_cache_dir,
    default_retention,
    prune_cache,
)
from stereo_winds.readers._satpy_s3 import SceneNotInStore, s3_filesystem

logger = logging.getLogger(__name__)

# Opened stores, keyed by (bucket, endpoint, prefix).  Shared across
# readers and threads: a store is immutable for the life of a run.
_OPEN_DATASETS: dict[tuple[str, str, str], xr.Dataset] = {}
# prefix -> (bands, first scan, last scan); None when the store is unusable.
_STORE_CONTENTS: dict[str, tuple | None] = {}
_BUCKET_LISTINGS: dict[tuple[str, str, str], list[str]] = {}
_OPEN_LOCK = threading.Lock()


def clear_store_cache() -> None:
    """Drop cached store handles (tests, or to pick up a newer snapshot)."""
    with _OPEN_LOCK:
        _OPEN_DATASETS.clear()
        _STORE_CONTENTS.clear()
        _BUCKET_LISTINGS.clear()

# GRS80, the default when a store states no ellipsoid of its own.
_GRS80_SEMI_MAJOR = 6378137.0
_GRS80_SEMI_MINOR = 6356752.31414


class GeoStoreReader:
    """Base class for the source.coop icechunk geostationary readers."""

    # ── instrument description (overridden by subclasses) ──────────────
    instrument: str = "geostationary imager"
    sweep: str = "y"          # only GOES ABI sweeps x
    bucket: str = "bkr"
    endpoint: str = "https://data.source.coop"

    sub_lon: Any = {}
    sat_height: Any = 0.0
    semi_major: float = _GRS80_SEMI_MAJOR
    semi_minor: float = _GRS80_SEMI_MINOR

    band_resolution: dict[str, str] = {}
    abi_to_native: dict[str, str] = {}
    abi_without_native: tuple[str, ...] = ()
    missing_band_reason: str = ""
    band_name_hint: str = ""
    default_band: str = ""

    store_template: str | None = None
    store_prefixes: dict[str, str] | None = None
    store_root: str = "geo"
    # Stores whose names start with this belong to this instrument, and
    # are considered when the named one lacks the band or the coverage.
    # Empty disables discovery and keeps the named store only.
    store_discovery_prefix: str = ""

    scan_interval_minutes: int = 10

    coord_names: dict[str, list[str]] = {
        "x": ["x", "x_geostationary", "column", "longitude", "lon"],
        "y": ["y", "y_geostationary", "row", "latitude", "lat"],
    }
    synth_scales: dict[int, float] = {}
    default_synth_scale: float = 5.6e-05

    # Layout 3 of the radiance lookup (take the first spatial variable)
    # is a guess; stores with predictable layouts switch it off.
    allow_variable_fallback: bool = True

    supports_s3_fallback: bool = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        satellite: str | None = None,
        bands: list[str] | None = None,
        allow_s3_fallback: bool = True,
        cache_dir: str | None = None,
        cache_retention: dt.timedelta | None = -1,
    ) -> None:
        self.satellite = satellite if satellite is not None else self.satellites()[0]
        if self.satellite not in self.satellites():
            raise ValueError(
                f"Unknown satellite {self.satellite!r}. "
                f"Supported: {sorted(self.satellites())}"
            )
        raw_bands = list(bands) if bands else [self.default_band]
        self.bands = [self.resolve_band(b) for b in raw_bands]
        # Inert on readers without a fallback (supports_s3_fallback False).
        self.allow_s3_fallback = allow_s3_fallback
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        # Downloads older than this (relative to the scene being read) are
        # dropped; None keeps everything.
        self.cache_retention = (default_retention() if cache_retention == -1
                                else cache_retention)
        self._fs = None

    @classmethod
    def satellites(cls) -> tuple[str, ...]:
        """Satellite ids this reader serves."""
        if isinstance(cls.sub_lon, dict):
            return tuple(cls.sub_lon)
        return (cls.instrument,)

    @classmethod
    def resolve_band(cls, band: str) -> str:
        """Accept either a native band name or its ABI equivalent."""
        if band in cls.band_resolution:
            return band
        native = cls.abi_to_native.get(band)
        if native is not None:
            return native
        if band in cls.abi_without_native:
            raise ValueError(
                f"ABI band {band!r} has no {cls.instrument.upper()} "
                f"counterpart — {cls.missing_band_reason}"
            )
        raise ValueError(f"Unknown band {band!r}. {cls.band_name_hint}")

    # ------------------------------------------------------------------
    # Per-satellite nominal values (fallbacks; the store is the authority)
    # ------------------------------------------------------------------

    def _nominal(self, value: Any) -> float:
        return float(value[self.satellite] if isinstance(value, dict) else value)

    @property
    def nominal_sub_lon(self) -> float:
        return self._nominal(self.sub_lon)

    @property
    def nominal_height(self) -> float:
        return self._nominal(self.sat_height)

    @property
    def store_tolerance(self) -> dt.timedelta:
        """How far the nearest scan may be before it counts as a miss."""
        return dt.timedelta(minutes=self.scan_interval_minutes)

    # ------------------------------------------------------------------
    # Store access
    # ------------------------------------------------------------------

    def _open_store(self, prefix: str) -> Any:
        """Open an icechunk store in read-only mode."""
        import icechunk

        storage = icechunk.s3_storage(
            bucket=self.bucket,
            prefix=prefix,
            endpoint_url=self.endpoint,
            anonymous=True,
            force_path_style=True,
        )
        return icechunk.Repository.open(storage).readonly_session("main").store

    def _store_prefix(self, resolution: str) -> str:
        """The store this reader would open from the name alone."""
        if self.store_prefixes is not None:
            return self.store_prefixes[self.satellite]
        if self.store_template is None:
            raise NotImplementedError(
                f"{type(self).__name__} declares no store location")
        return self.store_template.format(resolution=resolution)

    # ------------------------------------------------------------------
    # Choosing a store by what it holds, not by what it is called
    # ------------------------------------------------------------------

    @classmethod
    def _bucket_stores(cls) -> list[str]:
        """Every store in the bucket, listed once per process."""
        key = (cls.bucket, cls.endpoint, cls.store_root)
        with _OPEN_LOCK:
            cached = _BUCKET_LISTINGS.get(key)
        if cached is not None:
            return cached
        try:
            import s3fs

            fs = s3fs.S3FileSystem(anon=True, endpoint_url=cls.endpoint)
            names = sorted(k.split("/")[-1]
                           for k in fs.ls(f"{cls.bucket}/{cls.store_root}"))
        except Exception:
            logger.exception("Could not list %s/%s; falling back to the "
                             "store names built from the band table",
                             cls.bucket, cls.store_root)
            names = []
        with _OPEN_LOCK:
            _BUCKET_LISTINGS[key] = names
        return names

    def _candidate_stores(self, band: str) -> list[str]:
        """Stores that might hold ``band``, most likely first.

        The band's resolution tier is a hint, not the answer: bands move
        between tiers and newer ingests land in separately named stores,
        so the named store is tried first and the instrument's other
        stores after it.
        """
        preferred = self._store_prefix(self.band_resolution[band])
        if not self.store_discovery_prefix:
            return [preferred]

        tier = self.band_resolution[band]
        others = [f"{self.store_root}/{name}"
                  for name in self._bucket_stores()
                  if name.startswith(self.store_discovery_prefix)]
        # Same tier first (a newer ingest of the same grid), then the rest.
        same_tier = [p for p in others if tier in p and p != preferred]
        rest = [p for p in others if tier not in p and p != preferred]
        return [preferred, *same_tier, *rest]

    def _store_contents(self, prefix: str):
        """(bands, first scan, last scan) for a store, or None if unusable."""
        with _OPEN_LOCK:
            cached = _STORE_CONTENTS.get(prefix)
        if cached is not None:
            return cached
        try:
            ds = self._open_dataset_at(prefix)
            times = np.asarray(ds["time"].values, "datetime64[ns]")
            contents = (frozenset(ds.data_vars), times.min(), times.max())
        except Exception:
            logger.info("Store %s is unusable; skipping it", prefix)
            contents = None
        with _OPEN_LOCK:
            _STORE_CONTENTS[prefix] = contents
        return contents

    def _select_store(self, band: str, t: dt.datetime) -> str:
        """The store holding ``band`` with coverage at ``t``.

        Raises ``SceneNotInStore`` when no store has both, which lets the
        caller fall back to public S3 where one exists.
        """
        target = np.datetime64(t.replace(tzinfo=None), "ns")
        tolerance = np.timedelta64(
            int(self.store_tolerance.total_seconds()), "s")
        tried: list[str] = []
        for prefix in self._candidate_stores(band):
            contents = self._store_contents(prefix)
            if contents is None:
                continue
            bands, first, last = contents
            if band not in bands:
                logger.debug("%s does not carry %s", prefix, band)
                continue
            tried.append(prefix)
            if first - tolerance <= target <= last + tolerance:
                return prefix
        raise SceneNotInStore(
            f"no store carries {self.satellite} {band} at {t} "
            f"(checked: {', '.join(tried) or 'none with this band'})"
        )

    def _open_dataset(self, resolution: str) -> xr.Dataset:
        """Open the icechunk store for the given resolution as xr.Dataset.

        Cached per store for the life of the process.  Opening costs a
        round trip to the object store plus a dedupe-and-sort over the
        whole time coordinate — tens of thousands of entries — and a
        single timestamp asks for ~20 scenes from the same store, so
        doing it per band made the reads dominate the retrieval.

        The session is a read-only snapshot, so every scene in a run
        also sees a consistent view of the store.
        """
        return self._open_dataset_at(self._store_prefix(resolution))

    def _open_dataset_at(self, prefix: str) -> xr.Dataset:
        """Open a specific store, cached for the life of the process."""
        key = (self.bucket, self.endpoint, prefix)
        cached = _OPEN_DATASETS.get(key)
        if cached is not None:
            return cached
        with _OPEN_LOCK:
            # Re-check: another thread may have opened it while we waited.
            cached = _OPEN_DATASETS.get(key)
            if cached is not None:
                return cached
            logger.info("Opening icechunk store %s/%s", self.bucket, prefix)
            ds = xr.open_zarr(self._open_store(prefix))
            if "time" in ds.dims:
                # Drop duplicate timestamps, then sort for nearest-neighbour
                # lookup.
                _, unique_idx = np.unique(ds["time"].values, return_index=True)
                ds = ds.isel(time=np.sort(unique_idx)).sortby("time")
            _OPEN_DATASETS[key] = ds
            return ds

    def _select_time(
        self, ds: xr.Dataset, t: dt.datetime,
        tolerance: dt.timedelta | None = -1,  # sentinel: use store_tolerance
    ) -> xr.Dataset:
        """Select the nearest time step, within ``tolerance`` if given.

        Raises ``SceneNotInStore`` when nothing is close enough, so the
        caller can fall back rather than silently using a scan from a
        different day.  Pass ``tolerance=None`` for the old unbounded
        nearest-neighbour behaviour.
        """
        if tolerance == -1:
            tolerance = self.store_tolerance
        target = np.datetime64(t.replace(tzinfo=None), "ns")
        if tolerance is None:
            return ds.sel(time=target, method="nearest")
        tol = np.timedelta64(int(tolerance.total_seconds()), "s")
        try:
            return ds.sel(time=target, method="nearest", tolerance=tol)
        except KeyError as exc:
            raise SceneNotInStore(
                f"no {self.satellite} scan within {tolerance} of {t}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def data_at_time(self, t: dt.datetime, **_: Any) -> xr.Dataset:
        """Return the scene closest to time ``t`` for ``self.bands[0]``.

        Returns
        -------
        xr.Dataset
            ``Rad`` shaped ``(1, 1, y, x)``, y ascending (south -> north)
            and x ascending (west -> east), coordinates in **metres**
            (scan angle * perspective height).
            ``Rad.attrs["orbital_parameters"]`` carries the projection
            recorded for this scene, ``ds.attrs["sweep_angle_axis"]`` the
            sweep convention, and ``ds.attrs["ellipsoid"]`` the reference
            ellipsoid the instrument navigates on.
        """
        band = self.bands[0]
        if not self.supports_s3_fallback:
            return self._icechunk_data_at_time(t, band)

        try:
            return self._icechunk_data_at_time(t, band)
        except SceneNotInStore as exc:
            if not self.allow_s3_fallback:
                raise
            logger.info("%s — falling back to %s on public S3",
                        exc, self.s3_bucket_label())
        except Exception:
            if not self.allow_s3_fallback:
                raise
            logger.exception(
                "icechunk read failed for %s %s at %s — falling back to "
                "public S3", self.satellite, band, t)
        out = self._s3_data_at_time(t, band)
        self.prune_download_cache(t)
        return out

    def prune_download_cache(self, t: dt.datetime) -> None:
        """Drop cached scans more than ``cache_retention`` before ``t``."""
        if self.cache_retention is None:
            return
        prune_cache(self.cache_dir / self.satellite, t - self.cache_retention)

    def _icechunk_data_at_time(self, t: dt.datetime, band: str) -> xr.Dataset:
        """Read the scene from whichever icechunk store covers it."""
        prefix = self._select_store(band, t)
        ds = self._open_dataset_at(prefix)
        snap = self._select_time(ds, t)
        rad_2d = self._extract_radiance(snap, band)

        # Projection metadata for *this* timestamp, not a nominal
        # constant: geostationary satellites drift within their
        # station-keeping box and are periodically relocated.
        orbital = scene_orbital_parameters(
            snap, band,
            fallback_sub_lon=self.nominal_sub_lon,
            fallback_height=self.nominal_height,
            label=f"{self.satellite} {band}",
        )
        x_m, y_m_asc, rad_sn = self._build_coords(
            ds, rad_2d, sat_height=orbital["projection_altitude"],
        )

        Rad = xr.DataArray(
            rad_sn[None, None, :, :],
            dims=("time", "band", "y", "x"),
            coords={"x": ("x", x_m), "y": ("y", y_m_asc)},
            name="Rad",
        )
        Rad.attrs["orbital_parameters"] = orbital

        sel_time = snap["time"].values if "time" in snap.coords else None
        if sel_time is not None:
            actual = str(np.datetime_as_string(
                np.datetime64(sel_time, "ns"), unit="s"))
            Rad.attrs["time_coverage_start"] = actual
            Rad.attrs["time_coverage_end"] = actual

        out = xr.Dataset({"Rad": Rad})
        out.attrs["sweep_angle_axis"] = self.sweep
        out.attrs["source"] = "icechunk"
        semi_major, semi_minor = scene_ellipsoid(
            snap, band,
            fallback_semi_major=self.semi_major,
            fallback_semi_minor=self.semi_minor,
        )
        out.attrs["ellipsoid"] = {
            "semi_major_m": semi_major, "semi_minor_m": semi_minor,
        }
        return out

    # ------------------------------------------------------------------
    # Radiance extraction (flexible to different store layouts)
    # ------------------------------------------------------------------

    def _extract_radiance(self, snap: xr.Dataset, band: str) -> np.ndarray:
        """Extract a 2-D (y, x) radiance array from the time-selected slice."""
        # Layout 1: band name is a data variable
        if band in snap.data_vars:
            return self._as_2d(snap[band].values)

        if not self.allow_variable_fallback:
            raise KeyError(
                f"Cannot find radiance for band {band!r} in dataset. "
                f"Variables: {list(snap.data_vars)}"
            )

        band_num = self._band_number(band)

        # Layout 2: single "Rad" variable with a band dimension
        for vname in ("Rad", "radiance", "rad", "toa_brightness_temperature"):
            if vname in snap.data_vars:
                da = snap[vname]
                if "band" in da.dims:
                    if "band" in da.coords:
                        band_vals = da.coords["band"].values
                        if band in band_vals:
                            return self._as_2d(da.sel(band=band).values)
                        if band_num is not None and band_num in band_vals:
                            return self._as_2d(da.sel(band=band_num).values)
                    return self._as_2d(da.isel(band=0).values)
                return self._as_2d(da.values)

        # Layout 3: discover the first spatial data variable
        for vname in snap.data_vars:
            if snap[vname].ndim >= 2:
                return self._as_2d(snap[vname].values)

        raise KeyError(
            f"Cannot find radiance for band {band!r} in dataset. "
            f"Variables: {list(snap.data_vars)}"
        )

    @staticmethod
    def _as_2d(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        return (arr if arr.ndim == 2 else arr.squeeze()).astype(np.float32)

    @staticmethod
    def _band_number(band: str) -> int | None:
        """Trailing digits of a band name, for stores indexed by number."""
        digits = "".join(c for c in band if c.isdigit())
        return int(digits) if digits else None

    # ------------------------------------------------------------------
    # Coordinate handling
    # ------------------------------------------------------------------

    def _build_coords(
        self, ds: xr.Dataset, rad_2d: np.ndarray,
        sat_height: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build x/y metre coordinates, oriented west->east and south->north.

        Coordinates come from the store when it carries them; the
        synthesised fallback is only for stores (and test fixtures) that
        do not.  Radiance columns/rows are flipped alongside the
        coordinate arrays so the two always agree.
        """
        if sat_height is None:
            sat_height = self.nominal_height
        ny, nx = rad_2d.shape
        x_vals = self._get_coord(ds, "x", nx)
        y_vals = self._get_coord(ds, "y", ny)

        # Radians if the span is tiny; otherwise the store already holds metres.
        if float(np.abs(x_vals[-1] - x_vals[0])) < 1.0:
            x_m = x_vals * sat_height
            y_m = y_vals * sat_height
        else:
            x_m = x_vals
            y_m = y_vals

        # Normalise the column order.  The AHI store keeps its columns
        # east->west (x_geostationary descends); the canonical fixed grid
        # runs west->east, so flip the data with the coordinate.
        if x_m[0] > x_m[-1]:
            x_m = x_m[::-1]
            rad_2d = rad_2d[:, ::-1]

        if y_m[0] > y_m[-1]:
            y_m = y_m[::-1]
            rad_2d = rad_2d[::-1, :]

        return x_m.astype(np.float64), y_m.astype(np.float64), rad_2d

    def _get_coord(
        self, ds: xr.Dataset, axis: str, expected_len: int,
    ) -> np.ndarray:
        """Retrieve the x or y coordinate array, synthesising if absent."""
        for name in self.coord_names.get(axis, []):
            if name in ds.coords:
                vals = ds.coords[name].values.astype(np.float64)
                if len(vals) == expected_len:
                    return vals

        scale = self.synth_scales.get(expected_len, self.default_synth_scale)
        logger.warning(
            "No %s coordinate found in store; synthesising with "
            "scale=%.2e rad/px (grid %d)", axis, scale, expected_len,
        )
        half = expected_len / 2.0
        if axis == "x":
            return (np.arange(expected_len) - half + 0.5) * scale
        return (half - 0.5 - np.arange(expected_len)) * scale

    # ------------------------------------------------------------------
    # Public-S3 fallback hooks
    # ------------------------------------------------------------------

    @property
    def fs(self):
        """Anonymous handle on the public NOAA buckets."""
        if self._fs is None:
            self._fs = s3_filesystem()
        return self._fs

    def s3_bucket_label(self) -> str:
        """Bucket named in fallback log messages."""
        return "public S3"

    def _snap_slot(self, t: dt.datetime) -> dt.datetime:
        """Floor to the full-disk slot containing ``t``."""
        step = self.scan_interval_minutes
        return t.replace(minute=(t.minute // step) * step,
                         second=0, microsecond=0, tzinfo=None)

    def _s3_data_at_time(self, t: dt.datetime, band: str) -> xr.Dataset:
        raise NotImplementedError(
            f"{type(self).__name__} has no public-S3 fallback")

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(satellite={self.satellite!r}, "
                f"bands={self.bands!r})")
