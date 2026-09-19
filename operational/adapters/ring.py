"""Importable view of ``scripts/infer_student_global_ring.py``.

The global-ring pipeline lives in a CLI script rather than in the
:mod:`stereo_winds` package, so it cannot be imported with a normal
``import`` statement.  This module is the operational package's *single*
place that loads it by path, so no Dagster asset has to repeat the
``importlib`` dance and no two callers end up with two copies of the
script's module-level state (grid caches, loggers, torch handles).

The script is executed at most once per process, behind a lock, under the
``sys.modules`` name ``"operational_ring"`` — deliberately distinct from
the names the repository's own tests use (``infer_student_global_ring``,
``ring_prefetch``, ``write_mosaics``, ``ring_msg``) so the two loads
cannot collide.

Nothing here reimplements the upstream logic; every public name below is
the script's own object, re-exported so that static readers and IDEs can
see what the operational code depends on.  If one of them ever disappears
upstream, importing this module fails immediately with an explicit
message instead of raising :class:`AttributeError` deep inside a Dagster
run.

Examples
--------
>>> from operational.adapters.ring import GlobalMosaic, RING_SATELLITES
>>> RING_SATELLITES[0]
'goes18'
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    "load_ring",
    "REPO_ROOT",
    "RING_SCRIPT",
    "RING_MODULE_NAME",
    "infer_satellite",
    "GlobalMosaic",
    "quality_attrs",
    "satellite_available_times",
    "availability_band",
    "scan_interval",
    "time_tag",
    "sat_nc_path",
    "global_nc_path",
    "filter_to_common_times",
    "RING_SATELLITES",
    "OUTPUT_VARS",
    "DT_MINUTES",
    "SCAN_INTERVAL_MINUTES",
]

#: Repository root, resolved from this file (``<repo>/operational/adapters/
#: ring.py``) rather than from the working directory, so the adapter works
#: from any cwd and from a git worktree.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Absolute path of the CLI script this module wraps.
RING_SCRIPT: Path = REPO_ROOT / "scripts" / "infer_student_global_ring.py"

#: ``sys.modules`` key the script is registered under.
RING_MODULE_NAME: str = "operational_ring"

#: Names the operational package depends on; each must exist upstream.
_REQUIRED_NAMES: tuple[str, ...] = (
    "infer_satellite",
    "GlobalMosaic",
    "quality_attrs",
    "satellite_available_times",
    "availability_band",
    "scan_interval",
    "time_tag",
    "sat_nc_path",
    "global_nc_path",
    "filter_to_common_times",
    "RING_SATELLITES",
    "OUTPUT_VARS",
    "DT_MINUTES",
    "SCAN_INTERVAL_MINUTES",
)

_RING_MODULE: ModuleType | None = None
_LOAD_LOCK = threading.Lock()


def _exec_ring_module() -> ModuleType:
    """Execute the ring script and return it as a module.

    Registers the module in :data:`sys.modules` *before* executing it, so
    classes defined by the script have a resolvable ``__module__`` and
    recursive imports see a partially-initialised module rather than
    re-executing the file.

    Returns
    -------
    types.ModuleType
        The freshly executed script module.

    Raises
    ------
    FileNotFoundError
        If the script is missing from the checkout.
    ImportError
        If Python cannot build a loader for the script.
    """
    existing = sys.modules.get(RING_MODULE_NAME)
    if existing is not None and getattr(existing, "__file__", None) == str(
        RING_SCRIPT
    ) and hasattr(existing, _REQUIRED_NAMES[0]):
        # Someone (a reloaded copy of this adapter, say) already executed
        # the script under our name; reuse it rather than running the
        # module-level code a second time.
        logger.debug("Reusing already-loaded %s", RING_MODULE_NAME)
        return existing
    if not RING_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Global ring script not found at {RING_SCRIPT}. The operational "
            f"package must live inside the stereo-winds checkout "
            f"(expected repo root {REPO_ROOT})."
        )
    spec = importlib.util.spec_from_file_location(RING_MODULE_NAME, RING_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build an import spec for {RING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[RING_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Do not leave a half-initialised module behind for the next caller.
        sys.modules.pop(RING_MODULE_NAME, None)
        raise
    logger.debug("Loaded global ring script from %s", RING_SCRIPT)
    return module


def load_ring() -> ModuleType:
    """Return the global-ring script as a module, loading it once.

    The first call executes ``scripts/infer_student_global_ring.py``;
    every later call in the same process returns the cached module, so
    repeated use is cheap.  Concurrent callers are serialised on a lock
    and all receive the same object.

    Returns
    -------
    types.ModuleType
        The ring script module.

    Raises
    ------
    FileNotFoundError
        If the script is missing from the checkout.
    """
    global _RING_MODULE
    module = _RING_MODULE
    if module is not None:
        return module
    with _LOAD_LOCK:
        if _RING_MODULE is None:
            _RING_MODULE = _exec_ring_module()
        return _RING_MODULE


def _require(module: ModuleType, name: str) -> Any:
    """Fetch ``name`` from ``module`` or explain that it vanished upstream.

    Parameters
    ----------
    module : types.ModuleType
        The loaded ring script module.
    name : str
        Attribute the operational package re-exports.

    Returns
    -------
    Any
        The attribute's value.

    Raises
    ------
    AttributeError
        If the script no longer defines ``name``.
    """
    try:
        return getattr(module, name)
    except AttributeError:
        raise AttributeError(
            f"{RING_SCRIPT} no longer defines {name!r}, which "
            f"operational.adapters.ring re-exports. Either the script was "
            f"refactored or this adapter is out of date; update the "
            f"re-export list in operational/adapters/ring.py to match."
        ) from None


_ring = load_ring()

infer_satellite: Callable[..., xr.Dataset] = _require(_ring, "infer_satellite")
"""Run student inference for one satellite and return an ``(y, x)`` dataset.

``infer_satellite(sat_id, t0, model, disp, flow_bands, rad_bands,
device="cuda", row_strip=1024, prefetcher=None)``.
"""

GlobalMosaic: type = _require(_ring, "GlobalMosaic")
"""Incremental min-zenith mosaic on a regular lat/lon grid.

``GlobalMosaic(resolution_m=2000.0)``; ``add(sat_id, ds)`` grids one
satellite and returns the number of cells it won, ``to_dataset()`` returns
the accumulated mosaic.
"""

quality_attrs: Callable[..., dict] = _require(_ring, "quality_attrs")
"""``quality_attrs(flow_bands, rad_bands, missing)`` -> provenance attrs.

Describes how much of the requested input the retrieval actually received,
so zero-filled channels travel with the data.
"""

satellite_available_times: Callable[..., np.ndarray] = _require(
    _ring, "satellite_available_times"
)
"""Sorted scan times a satellite can supply within a window.

``satellite_available_times(sat_id, band, start, end,
product="ABI-L1b-RadF", include_s3_fallback=True)``. Hits icechunk and/or
S3 — never call it from an offline test.
"""

availability_band: Callable[..., str | None] = _require(
    _ring, "availability_band"
)
"""``availability_band(sat_id, flow_bands, rad_bands)`` -> band id or None.

The first requested band the satellite actually carries; scan times are a
property of the instrument schedule, so one band suffices.
"""

scan_interval: Callable[[str], int] = _require(_ring, "scan_interval")
"""``scan_interval(sat_id)`` -> full-disk repeat cycle in minutes.

10 for the whole ring except ``msg-iodc``, which is 15.
"""

time_tag: Callable[[datetime], str] = _require(_ring, "time_tag")
"""``time_tag(t)`` -> compact timestamp tag, e.g. ``"20260801T0000"``."""

sat_nc_path: Callable[..., Path] = _require(_ring, "sat_nc_path")
"""``sat_nc_path(out_dir, sat_id, t)`` -> per-satellite NetCDF path."""

global_nc_path: Callable[..., Path] = _require(_ring, "global_nc_path")
"""``global_nc_path(out_dir, t)`` -> mosaic NetCDF path for a timestamp."""

filter_to_common_times: Callable[..., list] = _require(
    _ring, "filter_to_common_times"
)
"""Keep only timestamps every satellite can deliver a full triplet for.

``filter_to_common_times(times, sats, flow_bands, rad_bands, dt_min=None,
tolerance_min=5.0, product="ABI-L1b-RadF")``. Queries availability, so it
is a network call.
"""

RING_SATELLITES: list[str] = _require(_ring, "RING_SATELLITES")
"""The geostationary ring, in longitude order from west to east."""

OUTPUT_VARS: list[str] = _require(_ring, "OUTPUT_VARS")
"""AMV variables carried by per-satellite datasets and by the mosaic."""

DT_MINUTES: int = _require(_ring, "DT_MINUTES")
"""Default temporal-pair spacing, in minutes."""

SCAN_INTERVAL_MINUTES: dict[str, int] = _require(_ring, "SCAN_INTERVAL_MINUTES")
"""Per-satellite overrides of :data:`DT_MINUTES` for the scan cycle."""

del _ring
