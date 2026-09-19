"""Shared fixtures and synthetic data for the operational test suite.

Everything here is synthetic: no network, no checkpoints, no GPU.  The
:func:`synthetic_scene` helper is a module-level function (not a fixture)
so other test modules can import it directly::

    from operational.tests.conftest import synthetic_scene
"""

from __future__ import annotations

import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from operational.config import OperationalConfig

#: Variables every per-satellite AMV dataset carries, in output order.
AMV_VARS = [
    "u_wind",
    "v_wind",
    "cloud_top_height",
    "quality_flag",
    "sigma_u",
    "sigma_v",
    "sigma_h",
]

#: Nominal sub-satellite longitudes used to place synthetic scenes.  Any
#: satellite id not listed here is centred on the prime meridian.
_SUB_LON_DEG = {
    "goes16": -75.2,
    "goes18": -137.0,
    "goes19": -75.0,
    "himawari8": 140.7,
    "himawari9": 140.7,
    "gk2a": 128.2,
    "mtg-i1": 0.0,
    "msg-iodc": 45.5,
}

#: Half-width, in degrees, of a synthetic scene footprint.
_HALF_SPAN_DEG = 30.0

#: Public names for the two constants above.  The integration tests
#: build their coverage expectations from them, so they are part of
#: this module's contract rather than private detail.
SUB_LON_DEG = _SUB_LON_DEG
HALF_WIDTH_DEG = _HALF_SPAN_DEG

#: Stand-in for the band set the student model is fed.
SYNTHETIC_BANDS = [
    "C07", "C08", "C09", "C10", "C11", "C13", "C14", "C15",
]


def synthetic_wind(sat_id: str) -> tuple[float, float]:
    """Wind components this satellite reports, distinct per satellite.

    Deterministic, so a test can name the satellite it expects to have
    won a mosaic cell from the value it finds there.
    """
    rank = sorted(SUB_LON_DEG).index(sat_id) if sat_id in SUB_LON_DEG else 0
    return 10.0 + rank, -3.0 - rank


def synthetic_quality_attrs(bands_missing: tuple[str, ...] = ()) -> dict:
    """Quality attributes shaped like ``quality_attrs``' output."""
    absent = sorted(bands_missing)
    requested = list(SYNTHETIC_BANDS)
    degraded = len(absent) / len(requested) > 0.25
    if not absent:
        note = "all requested bands available"
    else:
        note = (f"{len(absent)} of {len(requested)} requested bands were "
                f"unavailable and zero-filled: {', '.join(absent)}")
        if degraded:
            note = "DEGRADED QUALITY - " + note
    return {
        "bands_requested": ",".join(requested),
        "bands_missing": ",".join(absent),
        "n_bands_requested": len(requested),
        "n_bands_missing": len(absent),
        "quality_degraded": int(degraded),
        "quality_note": note,
    }


def _sub_lon(sat_id: str) -> float:
    """Sub-satellite longitude for ``sat_id``, from the real config if known."""
    try:
        from stereo_winds.config import SATELLITE_CONFIGS

        cfg = SATELLITE_CONFIGS.get(sat_id)
        if cfg is not None:
            return float(cfg.sub_lon_deg)
    except Exception:  # pragma: no cover - stereo_winds always importable here
        pass
    return float(_SUB_LON_DEG.get(sat_id, 0.0))


def synthetic_scene(
    sat_id: str,
    t0: datetime,
    ny: int = 64,
    nx: int = 64,
    zenith: float = 10.0,
    bands_missing: tuple[str, ...] = (),
) -> xr.Dataset:
    """Build a per-satellite AMV dataset shaped like ``infer_satellite``'s output.

    The dataset has dimensions ``(y, x)``, the seven standard AMV
    variables as ``float32``, two-dimensional ``latitude`` / ``longitude``
    / ``zenith_angle`` coordinates, and the attributes the mosaic step
    reads.  Values are smooth, finite and deterministic for a given
    ``sat_id``, so tests can assert on them.

    Parameters
    ----------
    sat_id
        Satellite id; sets the ``satellite_id`` attribute and the
        longitude the footprint is centred on.
    t0
        Scene timestamp, written to the ``time`` attribute as a string.
    ny, nx
        Grid shape (rows, columns); default 64x64.
    zenith
        Constant satellite zenith angle, in degrees, filled across the
        scene.  Mosaicking keeps the lowest-zenith contributor, so tests
        control the winner by varying this.

    Returns
    -------
    xarray.Dataset
        Synthetic AMV scene with ``quality_flag`` set to 2.0 everywhere.
    """
    if ny <= 0 or nx <= 0:
        raise ValueError(f"ny and nx must be positive, got ({ny}, {nx})")

    sub_lon = _sub_lon(sat_id)
    lat_1d = np.linspace(-_HALF_SPAN_DEG, _HALF_SPAN_DEG, ny, dtype=np.float64)
    lon_1d = np.linspace(
        sub_lon - _HALF_SPAN_DEG, sub_lon + _HALF_SPAN_DEG, nx, dtype=np.float64
    )
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    # Keep longitudes in [-180, 180) the way the real navigation does.
    lon_2d = ((lon_2d + 180.0) % 360.0) - 180.0

    # Deterministic per-satellite offset: crc32 is stable across runs,
    # unlike hash() which is salted per interpreter.
    offset = zlib.crc32(sat_id.encode("utf-8")) % 17

    phase = np.deg2rad(lat_2d)
    u = 10.0 + offset + 5.0 * np.sin(phase)
    v = -3.0 + 2.0 * np.cos(phase)
    height = 6000.0 + 100.0 * offset + 500.0 * np.sin(2.0 * phase)

    data = {
        "u_wind": u,
        "v_wind": v,
        "cloud_top_height": height,
        "quality_flag": np.full((ny, nx), 2.0),
        "sigma_u": np.full((ny, nx), 1.5),
        "sigma_v": np.full((ny, nx), 1.5),
        "sigma_h": np.full((ny, nx), 750.0),
    }

    return xr.Dataset(
        {name: (("y", "x"), data[name].astype(np.float32)) for name in AMV_VARS},
        coords={
            "latitude": (("y", "x"), lat_2d.astype(np.float32)),
            "longitude": (("y", "x"), lon_2d.astype(np.float32)),
            "zenith_angle": (
                ("y", "x"),
                np.full((ny, nx), zenith, dtype=np.float32),
            ),
        },
        attrs={
            "satellite_id": sat_id,
            "time": str(t0),
            "source": "student_amv",
            **synthetic_quality_attrs(bands_missing),
        },
    )


@pytest.fixture
def op_config(tmp_path: Path) -> OperationalConfig:
    """An :class:`OperationalConfig` writing everything under ``tmp_path``.

    Uses a coarse mosaic grid and a two-satellite ring so tests stay fast.
    """
    return OperationalConfig(
        satellites=("goes19", "himawari9"),
        output_dir=tmp_path / "operational",
        store_uri=str(tmp_path / "operational.icechunk"),
        resolution_m=200_000.0,
        device="cpu",
        row_strip=32,
    )


@pytest.fixture
def tmp_store_uri(tmp_path: Path) -> str:
    """A local icechunk store URI under ``tmp_path`` (the store is not created)."""
    return str(tmp_path / "store.icechunk")
