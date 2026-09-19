"""Mosaic step: merge whatever per-satellite retrievals arrived.

One AMV asset per satellite means any single satellite can fail without
taking the cycle with it, and this step is where that tolerance is
cashed in: it mosaics the datasets it was handed and records, in the
output's attributes, which of the expected satellites are not in there.

The gridding itself is not reimplemented — it is upstream's
``GlobalMosaic``, which merges by smallest viewing zenith angle and
accumulates one satellite at a time.  That module is reached lazily,
from inside the functions that need it, so importing this step does not
drag in the inference stack (torch, the student model) on a worker whose
only job is to mosaic.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping, MutableMapping
from datetime import datetime
from pathlib import Path
from typing import Any

import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    "EmptyMosaicError",
    "build_mosaic",
    "missing_satellites",
    "write_mosaic_netcdf",
]

#: Variables/coordinates ``GlobalMosaic.add`` reads off every scene.
REQUIRED_FIELDS = ("latitude", "longitude", "zenith_angle", "quality_flag")


class EmptyMosaicError(ValueError):
    """Raised when nothing ended up in the mosaic.

    A cycle where *some* satellites are missing is normal and produces a
    mosaic; a cycle where nothing was handed over, or where everything
    handed over turned out to have no usable pixels, is a failed cycle.
    It is surfaced as this error rather than as an empty grid that would
    be published and read as a valid all-NaN product.
    """


def missing_satellites(
    per_sat: Mapping[str, Any] | Iterable[str],
    expected: Iterable[str],
) -> list[str]:
    """Which expected satellites are not among the contributors.

    Parameters
    ----------
    per_sat : Mapping of satellite id -> dataset (only the keys are
        used), or any iterable of contributing satellite ids.
    expected : Satellite ids the cycle was supposed to include.

    Returns
    -------
    list of str
        The expected ids that did not contribute, in the order they
        appear in ``expected`` and without duplicates.
    """
    contributed = set(per_sat.keys()) if isinstance(per_sat, Mapping) else set(per_sat)
    seen: set[str] = set()
    out: list[str] = []
    for sat_id in expected:
        if sat_id not in contributed and sat_id not in seen:
            seen.add(sat_id)
            out.append(sat_id)
    return out


def _check_scene(sat_id: str, ds: xr.Dataset) -> None:
    """Fail loudly if a scene lacks what the gridder needs."""
    absent = [f for f in REQUIRED_FIELDS if f not in ds.variables]
    if absent:
        raise ValueError(
            f"{sat_id}: per-satellite dataset is missing "
            f"{', '.join(absent)}; expected the schema infer_satellite "
            f"produces (2-D latitude/longitude/zenith_angle and quality_flag)"
        )


def build_mosaic(
    per_sat: Mapping[str, xr.Dataset],
    t0: datetime,
    *,
    resolution_m: float = 10000.0,
    expected: Iterable[str] | None = None,
    consume: bool = False,
) -> xr.Dataset:
    """Merge the available per-satellite retrievals onto the global grid.

    Satellites are gridded one at a time: nothing here accumulates the
    disks, so the gridded state is a single accumulator however many
    satellites take part.  With ``consume=True`` each scene is also
    dropped from ``per_sat`` once it has been gridded, which is what
    makes peak memory the accumulator plus one full disk rather than the
    accumulator plus every disk the caller is still holding.

    Parameters
    ----------
    per_sat : Satellite id -> AMV dataset, in ``infer_satellite``'s
        schema: dims ``(y, x)``, 2-D ``latitude`` / ``longitude`` /
        ``zenith_angle`` coordinates, and ``quality_flag >= 2`` marking
        the cells worth gridding.  A partial mapping is fine.
    t0 : Nominal timestamp of the cycle.  Recorded as ``nominal_time``,
        and used for the ``time`` attribute if no scene carried one.
    resolution_m : Output grid spacing, in metres.
    expected : Satellite ids the cycle was supposed to include.  Defaults
        to the keys of ``per_sat``, which reports only satellites that
        were handed over but won no cells; pass the full roster to have
        satellites that never produced a dataset reported too.
    consume : Drop each scene from ``per_sat`` after gridding it.  The
        mapping must be mutable.  Off by default, because emptying a
        caller's dict is not something to do unasked.

    Returns
    -------
    xr.Dataset
        Upstream's mosaic — dims ``(latitude, longitude)``, the output
        variables plus ``source_satellite_index`` — with the upstream
        attributes intact and provenance attributes added.

    Raises
    ------
    EmptyMosaicError
        If ``per_sat`` is empty, or if no satellite in it grids a single
        cell: either way there is no mosaic to publish.
    TypeError
        If ``consume`` is set but ``per_sat`` cannot be mutated.
    ValueError
        If a scene is missing a field the gridder needs.
    """
    from operational.adapters.ring import GlobalMosaic

    if not per_sat:
        raise EmptyMosaicError(
            f"no per-satellite retrievals available for {t0.isoformat()}: "
            f"nothing to mosaic"
        )
    if consume and not isinstance(per_sat, MutableMapping):
        raise TypeError(
            f"consume=True needs a mutable mapping, got {type(per_sat).__name__}"
        )

    sat_ids = list(per_sat)
    expected_ids = list(sat_ids) if expected is None else list(dict.fromkeys(expected))

    mosaic = GlobalMosaic(resolution_m=resolution_m)
    contributed: list[str] = []
    empty_handed: list[str] = []
    for sat_id in sat_ids:
        ds_sat = per_sat.pop(sat_id) if consume else per_sat[sat_id]
        _check_scene(sat_id, ds_sat)
        n_won = mosaic.add(sat_id, ds_sat)
        # Drop the local reference before gridding the next satellite; if
        # the caller asked us to consume, this was the last one holding it.
        del ds_sat
        if n_won > 0:
            contributed.append(sat_id)
        else:
            empty_handed.append(sat_id)
            logger.warning("%s produced no grid cells for %s",
                           sat_id, t0.isoformat())

    if not contributed:
        raise EmptyMosaicError(
            f"none of {', '.join(sat_ids)} produced a usable grid cell for "
            f"{t0.isoformat()}: nothing to mosaic"
        )

    ds = mosaic.to_dataset()

    absent = missing_satellites(contributed, expected_ids)
    if absent:
        logger.warning("Mosaic for %s is missing %d of %d satellites: %s",
                       t0.isoformat(), len(absent), len(expected_ids),
                       ", ".join(absent))

    # Upstream takes ``time`` from the first scene's attrs; if none of
    # them carried one it is None, which no NetCDF attribute can hold.
    if ds.attrs.get("time") is None:
        ds.attrs["time"] = t0.isoformat()

    # Provenance, under names upstream does not use, so nothing it set
    # (satellites, time, resolution_m, merge_rule, quality_degraded,
    # degraded_satellites, quality_note) is disturbed.  Comma-joined
    # rather than lists so an empty value survives a NetCDF round trip.
    ds.attrs.update({
        "nominal_time": t0.isoformat(),
        "satellites_expected": ",".join(expected_ids),
        "satellites_contributing": ",".join(contributed),
        "satellites_missing": ",".join(absent),
        "n_satellites_expected": len(expected_ids),
        "n_satellites_contributing": len(contributed),
        "n_satellites_missing": len(absent),
        "satellites_empty": ",".join(empty_handed),
        "mosaic_complete": int(not absent),
    })
    return ds


def write_mosaic_netcdf(
    ds: xr.Dataset,
    output_dir: str | os.PathLike[str],
    t0: datetime,
) -> Path:
    """Write the mosaic to its canonical path, atomically.

    The file is written to a temporary name in the destination directory
    and then :func:`os.replace`\\ d into place, so a reader globbing the
    output tree never sees a half-written mosaic and a crashed run leaves
    no partial file under the canonical name.

    Parameters
    ----------
    ds : The mosaic dataset, as returned by :func:`build_mosaic`.
    output_dir : Root of the per-day output layout.
    t0 : Timestamp the mosaic belongs to.

    Returns
    -------
    Path
        The canonical path now holding the mosaic.
    """
    from operational.adapters.ring import global_nc_path

    path = global_nc_path(Path(output_dir), t0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        ds.to_netcdf(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    logger.info("Wrote mosaic for %s to %s", t0.isoformat(), path)
    return path
