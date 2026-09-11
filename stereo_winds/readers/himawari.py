"""Icechunk-backed Himawari AHI reader for stereo-winds.

Reads Himawari-8/9 AHI Level-1b radiance from icechunk stores hosted at
source.coop and returns a dataset shaped exactly like the GOES reader:
``Rad`` as ``(time, band, y, x)`` oriented south->north, ``x``/``y`` in
**meters** (scan angle x perspective height), and
``Rad.attrs["orbital_parameters"]`` carrying the projection recorded for
the scene.  The shared mechanics live in
:class:`stereo_winds.readers._geos_store.GeoStoreReader`.

Icechunk stores
---------------
- ``geo/himawari_500m.icechunk``  — AHI bands B01-B03  (500 m VIS)
- ``geo/himawari_1000m.icechunk`` — AHI band  B04      (1 km VIS)
- ``geo/himawari_2000m.icechunk`` — AHI bands B05-B16  (2 km IR/WV/NIR)

Outside the stores' coverage the native HSD segments are read from
NOAA's public buckets (``s3://noaa-himawari8`` / ``s3://noaa-himawari9``)
with satpy instead.

Requires ``icechunk`` and ``zarr>=3``.
"""
from __future__ import annotations

import datetime as dt
import logging

import xarray as xr

from stereo_winds.readers._geos_store import GeoStoreReader
from stereo_winds.readers._satpy_s3 import (
    download_keys,
    load_scene_array,
    scene_to_rad,
)

logger = logging.getLogger(__name__)

# Himawari geostationary projection constants
_SAT_HEIGHT = 35786023.0  # m (perspective point height above geoid)
_SUB_LON: dict[str, float] = {
    "himawari8": 140.7,
    "himawari9": 140.7,
}

# AHI band -> resolution tier used to select the correct icechunk store.
_BAND_RESOLUTION: dict[str, str] = {
    "B01": "500m",  "B02": "500m",  "B03": "500m",
    "B04": "1000m",
    "B05": "2000m", "B06": "2000m", "B07": "2000m", "B08": "2000m",
    "B09": "2000m", "B10": "2000m", "B11": "2000m", "B12": "2000m",
    "B13": "2000m", "B14": "2000m", "B15": "2000m", "B16": "2000m",
}

# ABI band name -> AHI equivalent (closest spectral centre).
_ABI_TO_AHI: dict[str, str] = {
    "C01": "B01",  "C02": "B03",  "C03": "B04",  "C04": "B05",
    "C05": "B06",  "C06": "B07",  "C07": "B07",  "C08": "B08",
    "C09": "B10",  "C10": "B11",  "C11": "B12",  "C12": "B13",
    "C13": "B13",  "C14": "B14",  "C15": "B15",  "C16": "B16",
}

_BUCKET = "bkr"
_ENDPOINT = "https://data.source.coop"

# Public NOAA buckets holding the native HSD segments, used when the
# icechunk store does not cover the requested time.
_S3_BUCKET: dict[str, str] = {
    "himawari8": "noaa-himawari8",
    "himawari9": "noaa-himawari9",
}
_S3_PLATFORM: dict[str, str] = {"himawari8": "H08", "himawari9": "H09"}
_S3_PREFIX = "AHI-L1b-FLDK"
_FULL_DISK_MINUTES = 10          # AHI full-disk scan cadence
_SEGMENTS = 10                   # HSD segments per band per slot


def _resolve_band(band: str) -> str:
    """Accept either AHI-native (B14) or ABI-style (C14) band names."""
    return Himawari.resolve_band(band)


class Himawari(GeoStoreReader):
    """Icechunk-backed Himawari AHI reader.

    Parameters
    ----------
    satellite : "himawari8" | "himawari9"
    bands : list with a single AHI band, e.g. ``["B14"]`` or ``["C14"]``
    allow_s3_fallback : when the icechunk store has no scan near the
        requested time, read the native HSD files from NOAA's public
        bucket with satpy instead (default True)
    cache_dir : where downloaded HSD segments are kept
    """

    instrument = "ahi"
    sweep = "y"
    bucket = _BUCKET
    endpoint = _ENDPOINT

    sub_lon = _SUB_LON
    sat_height = _SAT_HEIGHT

    band_resolution = _BAND_RESOLUTION
    abi_to_native = _ABI_TO_AHI
    band_name_hint = "Use AHI names (B01-B16) or ABI names (C01-C16)."
    default_band = "B14"

    store_template = "geo/himawari_{resolution}.icechunk"
    scan_interval_minutes = _FULL_DISK_MINUTES

    coord_names = {
        "x": ["x", "x_geostationary", "column", "longitude", "lon", "phgeo"],
        "y": ["y", "y_geostationary", "row", "latitude", "lat", "thgeo"],
    }
    synth_scales = {5500: 5.6e-05, 11000: 2.8e-05, 22000: 1.4e-05}
    default_synth_scale = 5.6e-05

    supports_s3_fallback = True

    def __init__(
        self,
        satellite: str = "himawari9",
        bands: list[str] | None = None,
        allow_s3_fallback: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(satellite, bands, allow_s3_fallback, cache_dir)

    # ------------------------------------------------------------------
    # Public-S3 fallback (native HSD segments via satpy)
    # ------------------------------------------------------------------

    def s3_bucket_label(self) -> str:
        return _S3_BUCKET[self.satellite]

    def _s3_keys(self, slot: dt.datetime, band: str) -> list[str]:
        """All HSD segment keys for one band at one slot.

        The resolution tag is globbed rather than hardcoded so the band
        table here cannot drift out of step with the bucket.
        """
        bucket = _S3_BUCKET[self.satellite]
        platform = _S3_PLATFORM[self.satellite]
        pattern = (
            f"{bucket}/{_S3_PREFIX}/{slot:%Y/%m/%d/%H%M}/"
            f"HS_{platform}_{slot:%Y%m%d_%H%M}_{band}_FLDK_R*_S*.DAT*"
        )
        return sorted(self.fs.glob(pattern))

    def _s3_data_at_time(self, t: dt.datetime, band: str) -> xr.Dataset:
        """Read the scene from NOAA's public bucket via satpy."""
        slot = self._snap_slot(t)
        logger.info("Loading %s %s at %s from %s", self.satellite, band,
                    slot, _S3_BUCKET[self.satellite])
        keys = self._s3_keys(slot, band)
        if not keys:
            raise FileNotFoundError(
                f"No AHI HSD files for {self.satellite} {band} at {slot} "
                f"in s3://{_S3_BUCKET[self.satellite]}"
            )
        if len(keys) != _SEGMENTS:
            # A partial set would decode to a disk with missing stripes.
            raise FileNotFoundError(
                f"Expected {_SEGMENTS} HSD segments for {self.satellite} "
                f"{band} at {slot}, found {len(keys)}"
            )
        paths = download_keys(
            self.fs, keys,
            self.cache_dir / self.satellite / f"{slot:%Y%m%d_%H%M}",
        )
        da = load_scene_array("ahi_hsd", paths, band)
        return scene_to_rad(
            da, band, sweep=self.sweep,
            fallback_sub_lon=self.nominal_sub_lon,
            fallback_height=self.nominal_height,
            label=f"{self.satellite} {band} (S3)",
        )
