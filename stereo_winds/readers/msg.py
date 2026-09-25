"""Icechunk-backed Meteosat Second Generation (MSG/SEVIRI) reader.

Reads MSG SEVIRI Level-1.5 brightness temperatures / reflectances from
icechunk stores hosted at source.coop and returns a dataset shaped
exactly like the other readers: ``Rad`` as ``(time, band, y, x)``
oriented south->north and west->east, ``x``/``y`` in **meters** (scan
angle x perspective height), and ``Rad.attrs["orbital_parameters"]``
carrying the projection recorded for the scene.

Icechunk stores
---------------
- ``geo/iodc_3000m_test.icechunk`` — Indian Ocean Data Coverage service
  (Meteosat at 45.5°E), eleven narrow channels on the 3 km 3712² grid
- ``geo/msg_3000m.icechunk`` — the 0 degree (prime) service, the same
  eleven channels on the same grid, written by the satellite consumer

Differences from the other geostationary imagers
------------------------------------------------
SEVIRI repeats its full disk every **15 minutes**, not 10, so callers
building temporal pairs must use the instrument's own cadence
(``SCAN_INTERVAL_MINUTES``) rather than assuming a 10 minute step.

SEVIRI also carries only eleven narrow channels against ABI's sixteen,
so several ABI bands map onto the same SEVIRI channel and two have no
counterpart at all (see ``ABI_TO_SEVIRI``).

MSG uses its own reference ellipsoid (a = 6378169 m, 1/f = 295.488),
which differs from GRS80 by ~168 m in the semi-minor axis — enough to
shift geolocation by a couple of hundred metres if ignored — so the
ellipsoid is read from the store and passed on with the scene.

Requires ``icechunk`` and ``zarr>=3``.
"""

from __future__ import annotations

import logging

from stereo_winds.readers._geos_store import GeoStoreReader

logger = logging.getLogger(__name__)

# MSG geostationary projection constants (fallbacks; the store is the
# authority and these are only used when it carries no metadata).
_SAT_HEIGHT = 35785831.0  # m (perspective point height above the ellipsoid)
_SUB_LON: dict[str, float] = {
    "msg-iodc": 45.5,  # Indian Ocean Data Coverage service
    "msg-0deg": 0.0,  # the 0 degree (prime) service
}

# MSG reference ellipsoid, used when the store does not state one.
_SEMI_MAJOR = 6378169.0
_INVERSE_FLATTENING = 295.488065897014
_SEMI_MINOR = _SEMI_MAJOR * (1.0 - 1.0 / _INVERSE_FLATTENING)

# SEVIRI full-disk repeat cycle.
SCAN_INTERVAL_MINUTES = 15

# Narrow SEVIRI channels and their store resolution tier.  HRV is not
# carried by these stores.
_BAND_RESOLUTION: dict[str, str] = {
    "VIS006": "3000m",
    "VIS008": "3000m",
    "IR_016": "3000m",
    "IR_039": "3000m",
    "WV_062": "3000m",
    "WV_073": "3000m",
    "IR_087": "3000m",
    "IR_097": "3000m",
    "IR_108": "3000m",
    "IR_120": "3000m",
    "IR_134": "3000m",
}

# ABI band -> SEVIRI equivalent, by nearest band centre:
#
#   ABI                    SEVIRI                 centres (µm)
#   C01 blue      0.47  ->  VIS006      0.635   approximate: SEVIRI has no blue
#   C02 red       0.64  ->  VIS006      0.635
#   C03 veggie    0.86  ->  VIS008      0.81
#   C04 cirrus    1.38  ->  (none)              no 1.4 µm channel on SEVIRI
#   C05 snow/ice  1.61  ->  IR_016      1.64
#   C06 cloud     2.25  ->  (none)              no 2.2 µm channel on SEVIRI
#   C07 shortwave 3.90  ->  IR_039      3.92
#   C08 upper WV  6.19  ->  WV_062      6.25
#   C09 mid WV    6.95  ->  WV_073      7.35    nearer than WV_062
#   C10 lower WV  7.34  ->  WV_073      7.35
#   C11 cloud top 8.50  ->  IR_087      8.70
#   C12 ozone     9.61  ->  IR_097      9.66
#   C13 clean IR 10.35  ->  IR_108     10.80
#   C14 IR       11.20  ->  IR_108     10.80    nearer than IR_120
#   C15 dirty IR 12.30  ->  IR_120     12.00
#   C16 CO2      13.30  ->  IR_134     13.40
#
# C09/C10 and C13/C14 therefore resolve to the same SEVIRI channel: with
# eleven channels against sixteen, some ABI pairs cannot be separated.
ABI_TO_SEVIRI: dict[str, str] = {
    "C01": "VIS006",
    "C02": "VIS006",
    "C03": "VIS008",
    "C05": "IR_016",
    "C07": "IR_039",
    "C08": "WV_062",
    "C09": "WV_073",
    "C10": "WV_073",
    "C11": "IR_087",
    "C12": "IR_097",
    "C13": "IR_108",
    "C14": "IR_108",
    "C15": "IR_120",
    "C16": "IR_134",
}

# ABI bands with no SEVIRI counterpart, listed so callers can explain
# themselves rather than just failing a lookup.
ABI_WITHOUT_SEVIRI = ("C04", "C06")

_BUCKET = "bkr"
_ENDPOINT = "https://data.source.coop"

_STORE_PREFIX: dict[str, str] = {
    "msg-iodc": "geo/iodc_3000m_test.icechunk",
    "msg-0deg": "geo/msg_3000m.icechunk",
}

# Per satellite, because the two services do not share stores: one
# prefix would let a 0 degree request fall through to an IODC store
# whenever the named one was short of a band or a time, and hand back
# imagery from a satellite 45.5 degrees away without saying so.
_STORE_DISCOVERY: dict[str, str] = {
    "msg-iodc": "iodc_",
    "msg-0deg": "msg_",
}


def _resolve_band(band: str) -> str:
    """Accept either SEVIRI-native (IR_108) or ABI-style (C14) band names."""
    return MSG.resolve_band(band)


class MSG(GeoStoreReader):
    """Icechunk-backed MSG SEVIRI reader.

    Parameters
    ----------
    satellite : "msg-iodc" (45.5°E) or "msg-0deg" (0°)
    bands : list with a single band, SEVIRI (``["IR_108"]``) or ABI
        (``["C14"]``) named
    """

    instrument = "seviri"
    sweep = "y"
    bucket = _BUCKET
    endpoint = _ENDPOINT

    sub_lon = _SUB_LON
    sat_height = _SAT_HEIGHT
    semi_major = _SEMI_MAJOR
    semi_minor = _SEMI_MINOR

    band_resolution = _BAND_RESOLUTION
    abi_to_native = ABI_TO_SEVIRI
    abi_without_native = ABI_WITHOUT_SEVIRI
    missing_band_reason = "SEVIRI carries no 1.4 µm cirrus or 2.2 µm channel"
    band_name_hint = "Use SEVIRI names (VIS006-IR_134) or ABI names (C01-C16)."
    default_band = "IR_108"

    store_prefixes = _STORE_PREFIX
    # Also consider the instrument's other stores for this service.
    store_discovery_prefix = _STORE_DISCOVERY
    scan_interval_minutes = SCAN_INTERVAL_MINUTES

    coord_names = {
        "x": ["x", "x_geostationary", "column"],
        "y": ["y", "y_geostationary", "row"],
    }
    # SEVIRI 3 km full disk: 3712² at 3000.403 m/px.
    default_synth_scale = 3000.403 / _SAT_HEIGHT

    # The store keeps one variable per channel, so a "first spatial
    # variable" guess would silently hand back the wrong band.
    allow_variable_fallback = False

    def __init__(
        self,
        satellite: str = "msg-iodc",
        bands: list[str] | None = None,
    ) -> None:
        super().__init__(satellite, bands)
