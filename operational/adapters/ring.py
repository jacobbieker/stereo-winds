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

The script is found relative to this file, so the adapter works from any
cwd and from a git worktree; set ``STEREO_WINDS_RING_SCRIPT`` to point it
elsewhere, per the repo's convention that host paths live in env vars.

Nothing here reimplements the upstream logic.  Functions and classes are
re-exported as the script's own objects; the three mutable constants are
re-exported as copies, so a caller appending to :data:`OUTPUT_VARS`
cannot break ``GlobalMosaic`` halfway through a mosaic.  If any
re-exported name ever disappears upstream, importing this module fails
immediately with a message listing every casualty, instead of raising
:class:`AttributeError` deep inside a Dagster run.
"""

from __future__ import annotations

import importlib.util
import logging
import os
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
    "RING_SCRIPT_ENV",
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

#: Environment variable that overrides the script location, for layouts
#: where ``scripts/`` does not sit next to this package.
RING_SCRIPT_ENV: str = "STEREO_WINDS_RING_SCRIPT"

#: Absolute path of the CLI script this module wraps.
RING_SCRIPT: Path = Path(
    os.environ.get(
        RING_SCRIPT_ENV,
        str(REPO_ROOT / "scripts" / "infer_student_global_ring.py"),
    )
).expanduser()

#: ``sys.modules`` key the script is registered under.
RING_MODULE_NAME: str = "operational_ring"

#: Every name this module re-exports.  Checked against the loaded
#: script in one pass, so a rename upstream names all the casualties
#: at once rather than only the first.
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
    """
    existing = sys.modules.get(RING_MODULE_NAME)
    if (
        existing is not None
        and getattr(existing, "__file__", None) == str(RING_SCRIPT)
        and hasattr(existing, _REQUIRED_NAMES[0])
    ):
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
    """Fetch ``name`` from ``module`` or explain that it vanished upstream."""
    try:
        return getattr(module, name)
    except AttributeError:
        raise AttributeError(
            f"{RING_SCRIPT} no longer defines {name!r}, which "
            f"operational.adapters.ring re-exports. Either the script was "
            f"refactored or this adapter is out of date; update the "
            f"re-export list in operational/adapters/ring.py to match."
        ) from None


def _check_required(module: ModuleType) -> None:
    """Fail loudly if the script stopped defining a re-exported name."""
    missing = [name for name in _REQUIRED_NAMES if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"{RING_SCRIPT} no longer defines {', '.join(map(repr, missing))}, "
            f"which operational.adapters.ring re-exports. Either the script "
            f"was refactored or this adapter is out of date; update the "
            f"re-export list in operational/adapters/ring.py to match."
        )


_ring = load_ring()
_check_required(_ring)

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

satellite_available_times: Callable[..., np.ndarray] = _require(_ring, "satellite_available_times")
"""Sorted scan times a satellite can supply within a window.

``satellite_available_times(sat_id, band, start, end,
product="ABI-L1b-RadF", include_s3_fallback=True)``. Hits icechunk and/or
S3 — never call it from an offline test.
"""

availability_band: Callable[..., str | None] = _require(_ring, "availability_band")
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

filter_to_common_times: Callable[..., list] = _require(_ring, "filter_to_common_times")
"""Keep only timestamps every satellite can deliver a full triplet for.

``filter_to_common_times(times, sats, flow_bands, rad_bands, dt_min=None,
tolerance_min=5.0, product="ABI-L1b-RadF")``. Queries availability, so it
is a network call.
"""

RING_SATELLITES: list[str] = list(_require(_ring, "RING_SATELLITES"))
"""The geostationary ring, in longitude order from west to east.

A copy: the script iterates its own list, so mutating this one cannot
corrupt a running mosaic.
"""

OUTPUT_VARS: list[str] = list(_require(_ring, "OUTPUT_VARS"))
"""AMV variables carried by per-satellite datasets and by the mosaic.

A copy, for the same reason as :data:`RING_SATELLITES`: ``GlobalMosaic``
indexes its accumulator by the script's own list, and an appended entry
there would raise ``KeyError`` for every satellite.
"""

DT_MINUTES: int = _require(_ring, "DT_MINUTES")
"""Default temporal-pair spacing, in minutes."""

SCAN_INTERVAL_MINUTES: dict[str, int] = dict(_require(_ring, "SCAN_INTERVAL_MINUTES"))
"""Per-satellite overrides of :data:`DT_MINUTES` for the scan cycle.

A copy; call :func:`scan_interval` rather than reading this directly,
since it applies the default for the satellites absent from it.
"""

del _ring
