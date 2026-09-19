"""Shared offline fixtures for the operational tests.

Nothing here touches the network, a GPU or a model checkpoint: the
per-satellite AMV datasets are synthesised in the same schema
``infer_satellite`` produces, so every step downstream of inference can
be exercised on a laptop.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from operational.config import OperationalConfig

#: The variables every per-satellite AMV dataset carries.
OUTPUT_VARS = [
    "u_wind", "v_wind", "cloud_top_height",
    "quality_flag", "sigma_u", "sigma_v", "sigma_h",
]


def synthetic_scene(
    sat_id: str,
    t0: datetime,
    ny: int = 64,
    nx: int = 64,
    zenith: float = 10.0,
    lat_range: tuple[float, float] = (-10.0, 10.0),
    lon_range: tuple[float, float] = (-10.0, 10.0),
) -> xr.Dataset:
    """A tiny stand-in for one satellite's full-disk AMV retrieval.

    Parameters
    ----------
    sat_id : Satellite id recorded in ``attrs["satellite_id"]``.
    t0 : Nominal retrieval time.
    ny, nx : Scene shape.
    zenith : Constant viewing zenith angle, in degrees — the mosaic's
        merge rule is "smallest zenith wins", so this decides who wins
        an overlap.
    lat_range, lon_range : Geographic extent the scene is spread over.

    Returns
    -------
    xr.Dataset
        Dims ``(y, x)``, the seven output variables, and 2-D
        ``latitude`` / ``longitude`` / ``zenith_angle`` coordinates.
    """
    lat, lon = np.meshgrid(
        np.linspace(lat_range[0], lat_range[1], ny),
        np.linspace(lon_range[0], lon_range[1], nx),
        indexing="ij",
    )
    data = {v: np.full((ny, nx), 1.0, np.float32) for v in OUTPUT_VARS}
    data["quality_flag"] = np.full((ny, nx), 2.0, np.float32)
    return xr.Dataset(
        {v: (("y", "x"), data[v]) for v in OUTPUT_VARS},
        coords={
            "latitude": (("y", "x"), lat.astype(np.float32)),
            "longitude": (("y", "x"), lon.astype(np.float32)),
            "zenith_angle": (("y", "x"),
                             np.full((ny, nx), zenith, np.float32)),
        },
        attrs={
            "satellite_id": sat_id,
            "time": t0.isoformat(),
            "source": "synthetic",
        },
    )


@pytest.fixture
def op_config(tmp_path: Path) -> OperationalConfig:
    """An :class:`OperationalConfig` pointed entirely at ``tmp_path``."""
    return OperationalConfig(
        satellites=("goes18", "goes19"),
        output_dir=tmp_path / "output",
        store_uri=str(tmp_path / "store.icechunk"),
        resolution_m=200_000.0,
        device="cpu",
    )


@pytest.fixture
def tmp_store_uri(tmp_path: Path) -> str:
    """A local icechunk store URI under ``tmp_path``."""
    return str(tmp_path / "store.icechunk")
