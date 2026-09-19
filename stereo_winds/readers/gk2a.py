"""Icechunk-backed GK-2A AMI reader for stereo-winds.

Reads GK-2A AMI Level-1b radiance from icechunk stores hosted at
source.coop and returns a dataset shaped exactly like the GOES reader:
``Rad`` as ``(time, band, y, x)`` oriented south->north, ``x``/``y`` in
**meters** (scan angle x perspective height), and
``Rad.attrs["orbital_parameters"]`` carrying the projection recorded for
the scene.  The shared mechanics live in
:class:`stereo_winds.readers._geos_store.GeoStoreReader`.

Icechunk stores
---------------
- ``geo/gk2a_500m.icechunk``  — AMI band VI006          (500 m VIS)
- ``geo/gk2a_1000m.icechunk`` — AMI bands VI004-VI008   (1 km VIS)
- ``geo/gk2a_2000m.icechunk`` — AMI bands NR013-IR133   (2 km NIR/IR)

Outside the stores' coverage the native AMI L1b netCDFs are read from
NOAA's public bucket (``s3://noaa-gk2a-pds``) with satpy instead.

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

# GK-2A geostationary projection constants
_SAT_HEIGHT = 35786023.0  # m (perspective point height above geoid)
_SUB_LON = 128.2  # degrees East

# AMI band -> resolution tier used to select the correct icechunk store.
_BAND_RESOLUTION: dict[str, str] = {
    # 500 m visible
    "VI004": "500m",
    "VI005": "500m",
    "VI006": "500m",
    # 1000 m visible / near-IR
    "VI008": "1000m",
    "NR013": "1000m",
    "NR016": "1000m",
    # 2000 m shortwave IR, water vapour, and infrared
    "SW038": "2000m",
    "WV063": "2000m",
    "WV069": "2000m",
    "WV073": "2000m",
    "IR087": "2000m",
    "IR096": "2000m",
    "IR105": "2000m",
    "IR112": "2000m",
    "IR123": "2000m",
    "IR133": "2000m",
}

# ABI band name -> AMI equivalent (closest spectral centre).
_ABI_TO_AMI: dict[str, str] = {
    "C01": "VI004",   # 0.47 / 0.47 um
    "C02": "VI006",   # 0.64 / 0.64 um
    "C03": "VI008",   # 0.865 / 0.86 um
    "C04": "NR013",   # 1.378 / 1.37 um
    "C05": "NR016",   # 1.61 / 1.6 um
    # C06 (2.25 um) has no AMI equivalent -- omitted
    "C07": "SW038",   # 3.90 / 3.8 um
    "C08": "WV063",   # 6.19 / 6.3 um
    "C09": "WV069",   # 6.95 / 6.9 um
    "C10": "WV073",   # 7.34 / 7.3 um
    "C11": "IR087",   # 8.44 / 8.7 um
    "C12": "IR096",   # 9.61 / 9.6 um
    "C13": "IR105",   # 10.35 / 10.5 um
    "C14": "IR112",   # 11.20 / 11.2 um
    "C15": "IR123",   # 12.30 / 12.3 um
    "C16": "IR133",   # 13.30 / 13.3 um
}

_BUCKET = "bkr"
_ENDPOINT = "https://data.source.coop"

# Public NOAA bucket holding the native AMI L1b files, used when the
# icechunk store does not cover the requested time.
_S3_BUCKET = "noaa-gk2a-pds"
_S3_PREFIX = "AMI/L1B/FD"
_FULL_DISK_MINUTES = 10          # AMI full-disk scan cadence


def _resolve_band(band: str) -> str:
    """Accept either AMI-native (IR112) or ABI-style (C14) band names."""
    return GK2A.resolve_band(band)


class GK2A(GeoStoreReader):
    """Icechunk-backed GK-2A AMI reader.

    Parameters
    ----------
    satellite : "gk2a"
    bands : list with a single AMI band, e.g. ``["IR112"]`` or ``["C14"]``
    allow_s3_fallback : when the icechunk store has no scan near the
        requested time, read the native AMI files from NOAA's public
        bucket with satpy instead (default True)
    cache_dir : where downloaded AMI files are kept
    cache_retention : drop cached scans more than this far
        before the scene being read (default one hour;
        None keeps everything)
    """

    instrument = "ami"
    sweep = "y"
    bucket = _BUCKET
    endpoint = _ENDPOINT

    sub_lon = {"gk2a": _SUB_LON}
    sat_height = _SAT_HEIGHT

    band_resolution = _BAND_RESOLUTION
    abi_to_native = _ABI_TO_AMI
    band_name_hint = (
        "Use AMI names (VI004-IR133) or ABI names (C01-C16)."
    )
    default_band = "IR112"

    store_template = "geo/gk2a_{resolution}.icechunk"
    # Newer ingests land in separately named stores; consider
    # them when the named one lacks the band or the coverage.
    store_discovery_prefix = "gk2a_"
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
        satellite: str = "gk2a",
        bands: list[str] | None = None,
        allow_s3_fallback: bool = True,
        cache_dir: str | None = None,
        cache_retention: dt.timedelta | None = -1,
    ) -> None:
        super().__init__(satellite, bands, allow_s3_fallback, cache_dir,
                         cache_retention)

    # ------------------------------------------------------------------
    # Public-S3 fallback (native AMI L1b via satpy)
    # ------------------------------------------------------------------

    def s3_bucket_label(self) -> str:
        return _S3_BUCKET

    def _s3_keys(self, slot: dt.datetime, band: str) -> list[str]:
        """AMI L1b key for one band at one slot.

        The resolution tag (``fd020ge`` and friends) is globbed rather
        than hardcoded, since it varies by channel.
        """
        pattern = (
            f"{_S3_BUCKET}/{_S3_PREFIX}/{slot:%Y%m}/{slot:%d}/{slot:%H}/"
            f"gk2a_ami_le1b_{band.lower()}_fd*ge_{slot:%Y%m%d%H%M}.nc"
        )
        return sorted(self.fs.glob(pattern))

    def _s3_data_at_time(self, t: dt.datetime, band: str) -> xr.Dataset:
        """Read the scene from NOAA's public bucket via satpy."""
        slot = self._snap_slot(t)
        logger.info("Loading %s %s at %s from %s", self.satellite, band,
                    slot, _S3_BUCKET)
        keys = self._s3_keys(slot, band)
        if not keys:
            raise FileNotFoundError(
                f"No AMI L1b file for {self.satellite} {band} at {slot} "
                f"in s3://{_S3_BUCKET}"
            )
        paths = download_keys(
            self.fs, keys[:1],
            self.cache_dir / self.satellite / f"{slot:%Y%m%d_%H%M}",
        )
        # Decompression scratch goes beside the download cache,
        # not /tmp, and is removed as soon as the scene is read.
        da = load_scene_array(
            "ami_l1b", paths, band, scratch_dir=self.cache_dir,
        )
        return scene_to_rad(
            da, band, sweep=self.sweep,
            fallback_sub_lon=self.nominal_sub_lon,
            fallback_height=self.nominal_height,
            label=f"{self.satellite} {band} (S3)",
        )
