"""Icechunk-backed MTG FCI reader for stereo-winds.

Reads MTG-I1 FCI Level-1c radiance from icechunk stores hosted at
source.coop and returns a dataset shaped exactly like the GOES reader:
``Rad`` as ``(time, band, y, x)`` oriented south->north, ``x``/``y`` in
**meters** (scan angle x perspective height), and
``Rad.attrs["orbital_parameters"]`` carrying ``projection_altitude`` and
``satellite_nominal_longitude``.

Icechunk stores
---------------
- ``geo/mtg_500m.icechunk``          -- FCI VIS bands at 500 m
- ``geo/mtg_1000m.icechunk``         -- FCI VIS/NIR bands at 1 km
- ``geo/mtg_2000m.icechunk``         -- FCI IR/WV bands at 2 km
- ``geo/mtg_highres_1000m.icechunk`` -- FCI high-res bands at 1 km

Requires ``icechunk`` and ``zarr>=3``.
"""
from __future__ import annotations

import logging

from stereo_winds.config import ABI_TO_FCI_BAND
from stereo_winds.readers._geos_store import GeoStoreReader

logger = logging.getLogger(__name__)

# MTG-I geostationary projection constants
_SAT_HEIGHT: dict[str, float] = {
    "mtg-i1": 35786400.0,
}
_SUB_LON: dict[str, float] = {
    "mtg-i1": 0.0,
}

# FCI channel -> resolution tier for selecting the correct icechunk store.
_BAND_RESOLUTION: dict[str, str] = {
    # 500 m VIS
    "vis_04": "500m",
    "vis_05": "500m",
    "vis_06": "500m",
    "vis_08": "500m",
    "vis_09": "500m",
    # 1 km VIS/NIR
    "nir_13": "1000m",
    "nir_16": "1000m",
    "nir_22": "1000m",
    # 2 km IR/WV
    "ir_38": "2000m",
    "wv_63": "2000m",
    "wv_73": "2000m",
    "ir_87": "2000m",
    "ir_97": "2000m",
    "ir_105": "2000m",
    "ir_123": "2000m",
    "ir_133": "2000m",
}

_BUCKET = "bkr"
_ENDPOINT = "https://data.source.coop"

_FULL_DISK_MINUTES = 10          # FCI full-disk repeat cycle


def _resolve_band(band: str) -> str:
    """Accept either FCI-native (ir_105) or ABI-style (C13) band names.

    Returns
    -------
    str
        Canonical FCI channel name.
    """
    return MTG.resolve_band(band)


class MTG(GeoStoreReader):
    """Icechunk-backed MTG-I FCI reader.

    Parameters
    ----------
    satellite : "mtg-i1"
    bands : list with a single FCI channel, e.g. ``["ir_105"]`` or an
        ABI name such as ``["C13"]``
    """

    instrument = "fci"
    sweep = "y"
    bucket = _BUCKET
    endpoint = _ENDPOINT

    sub_lon = _SUB_LON
    sat_height = _SAT_HEIGHT

    band_resolution = _BAND_RESOLUTION
    abi_to_native = ABI_TO_FCI_BAND
    band_name_hint = (
        "Use FCI names (e.g. ir_105) or ABI names (e.g. C13). "
        "Not every ABI band has an FCI equivalent."
    )
    default_band = "ir_105"

    store_template = "geo/mtg_{resolution}.icechunk"
    scan_interval_minutes = _FULL_DISK_MINUTES

    coord_names = {
        "x": ["x", "x_geostationary", "column", "longitude", "lon"],
        "y": ["y", "y_geostationary", "row", "latitude", "lat"],
    }
    # FCI FDHSI: 5568 x 5568 at 2 km, scale 5.58871e-05 rad/px
    synth_scales = {
        5568: 5.58871e-05,   # 2 km
        11136: 2.79436e-05,  # 1 km
        22272: 1.39718e-05,  # 500 m
    }
    default_synth_scale = 5.58871e-05

    def __init__(
        self,
        satellite: str = "mtg-i1",
        bands: list[str] | None = None,
    ) -> None:
        super().__init__(satellite, bands)
