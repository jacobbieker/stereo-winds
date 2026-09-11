"""Per-scene geostationary projection metadata read from icechunk stores.

Geostationary satellites are not fixed points.  They drift within a
station-keeping box, are occasionally relocated (GOES-19 taking over the
GOES-East slot, Meteosat moving between 0° and 41.5°E), and their
perspective height varies by a few km over an orbit.  Hardcoding a
nominal sub-satellite longitude therefore mis-navigates every pixel:
0.2° of longitude error is roughly 22 km of geolocation error at nadir.

The source.coop ``geo/*.icechunk`` stores carry satpy-derived metadata
alongside the radiance bands, in one of two layouts:

*Packed* (the AHI store) — a per-timestep ``orbital_parameters`` variable
holding a stringified dict, plus a per-timestep ``area`` variable.

*Flattened* (the AMI and FCI stores) — one per-timestep variable per
field: ``projection_longitude``, ``projection_altitude``,
``satellite_actual_longitude`` and friends, with ``area`` carried as a
dataset attribute instead.

Either way the fields are the same: ``projection_*`` describes the fixed
grid the imagery was resampled onto, ``satellite_actual_*`` and
``nadir_*`` describe where the spacecraft actually was, and ``area``
holds the pyresample definition (``proj``, ``lon_0``, ``h``, ``a``,
``rf``, ``shape``, ``area_extent``).

This module reads whichever layout is present for the selected timestamp
and falls back to caller-supplied nominal values only when the store
carries no projection metadata at all.

Navigation uses ``projection_longitude``, not ``satellite_actual_longitude``:
the imagery has already been resampled onto the fixed grid defined by the
projection, so that is the origin the pixel→lat/lon transform must use.
The actual spacecraft position is passed through for consumers that need
the true viewing geometry (parallax, zenith angles).
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Keys that should be coerced to float when present.
_NUMERIC_KEYS = (
    "projection_longitude",
    "projection_latitude",
    "projection_altitude",
    "satellite_actual_longitude",
    "satellite_actual_latitude",
    "satellite_actual_altitude",
    "nadir_longitude",
    "nadir_latitude",
)

# Warn when the spacecraft has wandered this far from the projection origin.
_DRIFT_WARN_DEG = 0.5


def _as_dict(value: Any) -> dict | None:
    """Coerce a stored attribute/variable into a dict, or None.

    Store metadata arrives as a Python-dict ``repr`` (satpy's own
    stringification), occasionally as JSON, sometimes already as a dict,
    and always wrapped in a 0-d numpy array when read from a variable.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", "replace")
    if not isinstance(value, (str, np.str_)):
        return None
    text = str(value).strip()
    if not text:
        return None
    for parse in (ast.literal_eval, json.loads):
        try:
            parsed = parse(text)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _to_float(value: Any) -> float | None:
    """Best-effort float conversion (store values are often strings)."""
    if value is None or isinstance(value, (dict, list, tuple)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _scalar(value: Any) -> Any:
    """Unwrap a 0-d or single-element numpy array to a Python scalar."""
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        return value.reshape(())[()]
    return value


def _variable(ds: Any, name: str) -> Any:
    """Fetch variable ``name`` from a dataset without raising if absent."""
    if ds is None:
        return None
    try:
        if name in ds.variables:
            return ds[name].values
    except (AttributeError, TypeError):
        pass
    try:
        return ds[name].values
    except (KeyError, AttributeError, TypeError):
        return None


def _attr(ds: Any, name: str) -> Any:
    """Fetch dataset attribute ``name`` without raising if absent."""
    try:
        return ds.attrs.get(name)
    except AttributeError:
        return None


def area_projection(area: Any) -> dict | None:
    """Extract the ``projection`` sub-dict from a pyresample area dict.

    The stored area is keyed by area id (e.g. ``{"FLDK": {...}}``), so
    descend one level when necessary.
    """
    area = _as_dict(area)
    if area is None:
        return None
    if "projection" in area:
        proj = area.get("projection")
        return proj if isinstance(proj, dict) else None
    for value in area.values():
        if isinstance(value, dict) and isinstance(value.get("projection"), dict):
            return value["projection"]
    return None


def scene_ellipsoid(
    snap: Any,
    band: str | None = None,
    *,
    fallback_semi_major: float,
    fallback_semi_minor: float,
) -> tuple[float, float]:
    """Reference ellipsoid (semi-major, semi-minor) recorded for the scene.

    Read from the area definition's ``a``/``rf`` (or ``b``/``ellps``).
    It matters: MSG navigates on a = 6378169 m, 1/f = 295.488, whose
    semi-minor axis sits ~168 m from GRS80's.
    """
    proj = area_projection(_variable(snap, "area"))
    if proj is None:
        proj = area_projection(_attr(snap, "area"))
    if proj is None and band is not None:
        try:
            proj = area_projection(snap[band].attrs.get("area"))
        except (KeyError, AttributeError, TypeError):
            proj = None
    if proj is None:
        return float(fallback_semi_major), float(fallback_semi_minor)

    semi_major = _to_float(proj.get("a"))
    semi_minor = _to_float(proj.get("b"))
    inverse_flattening = _to_float(proj.get("rf"))
    if semi_major is None:
        return float(fallback_semi_major), float(fallback_semi_minor)
    if semi_minor is None and inverse_flattening:
        semi_minor = semi_major * (1.0 - 1.0 / inverse_flattening)
    if semi_minor is None:
        return float(semi_major), float(fallback_semi_minor)
    return float(semi_major), float(semi_minor)


def scene_orbital_parameters(
    snap: Any,
    band: str | None = None,
    *,
    fallback_sub_lon: float,
    fallback_height: float,
    label: str = "",
) -> dict[str, float]:
    """Build satpy-style ``orbital_parameters`` for one time-selected scene.

    Parameters
    ----------
    snap : time-selected ``xr.Dataset`` (the output of ``.sel(time=...)``)
    band : band variable name, used as a source of last resort for
        store-level (non-time-varying) attributes
    fallback_sub_lon, fallback_height : nominal values used only when the
        store carries no projection metadata at all
    label : identifier used in log messages

    Returns
    -------
    dict with at least ``projection_longitude``, ``projection_altitude``
    and ``satellite_nominal_longitude``; any spacecraft-position keys the
    store provides (``satellite_actual_*``, ``nadir_*``) are passed
    through unchanged.
    """
    tag = f"{label}: " if label else ""

    out: dict[str, float] = {}
    source = None

    # Flattened layout: one per-timestep variable per field.
    for key in _NUMERIC_KEYS:
        value = _to_float(_scalar(_variable(snap, key)))
        if value is not None:
            out[key] = value
            source = source or "store per-field variables"

    # Packed layout: a single stringified dict, per timestep or as an
    # attribute.  Only fills fields the flattened layout did not supply.
    orbital = _as_dict(_variable(snap, "orbital_parameters"))
    packed_source = "store orbital_parameters variable"
    if orbital is None:
        orbital = _as_dict(_attr(snap, "orbital_parameters"))
        packed_source = "store orbital_parameters attribute"
    if orbital is None and band is not None:
        try:
            orbital = _as_dict(snap[band].attrs.get("orbital_parameters"))
        except (KeyError, AttributeError, TypeError):
            orbital = None
        packed_source = f"{band} orbital_parameters attribute"
    if orbital:
        for key in _NUMERIC_KEYS:
            if key in out:
                continue
            value = _to_float(orbital.get(key))
            if value is not None:
                out[key] = value
                source = source or packed_source

    # The area definition is the authority on the fixed grid, and is the
    # fallback when the orbital fields are missing or incomplete.
    proj = area_projection(_variable(snap, "area"))
    if proj is None:
        proj = area_projection(_attr(snap, "area"))
    if proj is None and band is not None:
        try:
            proj = area_projection(snap[band].attrs.get("area"))
        except (KeyError, AttributeError, TypeError):
            proj = None
    if proj is not None:
        lon_0 = _to_float(proj.get("lon_0"))
        height = _to_float(proj.get("h"))
        if "projection_longitude" not in out and lon_0 is not None:
            out["projection_longitude"] = lon_0
            source = source or "store area definition"
        if "projection_altitude" not in out and height is not None:
            out["projection_altitude"] = height
        if (
            lon_0 is not None
            and "projection_longitude" in out
            and abs(lon_0 - out["projection_longitude"]) > 1e-6
        ):
            logger.warning(
                "%sarea lon_0=%.4f° disagrees with orbital_parameters "
                "projection_longitude=%.4f°; using the latter",
                tag, lon_0, out["projection_longitude"],
            )

    if "projection_longitude" not in out:
        logger.warning(
            "%sno sub-satellite longitude in store metadata — falling back "
            "to the nominal %.3f°. Navigation may be wrong if the satellite "
            "has drifted or been relocated.", tag, fallback_sub_lon,
        )
        out["projection_longitude"] = float(fallback_sub_lon)
        source = "hardcoded nominal fallback"
    source = source or "hardcoded nominal fallback"
    if "projection_altitude" not in out:
        out["projection_altitude"] = float(fallback_height)

    out.setdefault("projection_latitude", 0.0)
    out["satellite_nominal_longitude"] = out["projection_longitude"]

    actual = out.get("satellite_actual_longitude")
    if actual is not None:
        drift = abs(actual - out["projection_longitude"])
        # Longitudes may straddle the dateline (e.g. 179.9 vs -179.9).
        drift = min(drift, 360.0 - drift)
        if drift > _DRIFT_WARN_DEG:
            logger.warning(
                "%sspacecraft is %.3f° from the projection origin "
                "(actual=%.3f°, projection=%.3f°) — imagery may predate a "
                "relocation", tag, drift, actual, out["projection_longitude"],
            )

    logger.info(
        "%ssub_lon=%.4f° height=%.0f m (from %s)",
        tag, out["projection_longitude"], out["projection_altitude"], source,
    )
    return out
