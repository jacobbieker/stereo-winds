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

import datetime as dt
import logging
from typing import Any

import numpy as np
import xarray as xr

from stereo_winds.config import ABI_TO_FCI_BAND

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


def _open_store(prefix: str) -> Any:
    """Open an icechunk store in read-only mode."""
    import icechunk

    storage = icechunk.StorageConfig.s3_from_config(
        bucket=_BUCKET,
        prefix=prefix,
        config=icechunk.S3Config(
            endpoint_url=_ENDPOINT,
            region="us-east-1",
            allow_http=False,
            anonymous=True,
        ),
    )
    return icechunk.IcechunkStore.open_existing(storage=storage, mode="r")


def _resolve_band(band: str) -> str:
    """Accept either FCI-native (ir_105) or ABI-style (C13) band names.

    Returns
    -------
    str
        Canonical FCI channel name.
    """
    if band in _BAND_RESOLUTION:
        return band
    fci = ABI_TO_FCI_BAND.get(band)
    if fci is not None:
        return fci
    raise ValueError(
        f"Unknown band {band!r}. Use FCI names (e.g. ir_105) or ABI names "
        f"(e.g. C13). Known ABI->FCI map: {ABI_TO_FCI_BAND}"
    )


class MTG:
    """Icechunk-backed MTG FCI reader.

    Parameters
    ----------
    satellite : str
        Satellite identifier, e.g. ``"mtg-i1"``.
    bands : list[str] | None
        List with a single FCI band, e.g. ``["ir_105"]`` or ``["C13"]``.
        Defaults to ``["ir_105"]``.
    """

    def __init__(
        self,
        satellite: str = "mtg-i1",
        bands: list[str] | None = None,
    ) -> None:
        self.satellite = satellite
        if satellite not in _SUB_LON:
            raise ValueError(
                f"Unknown satellite {satellite!r}. "
                f"Supported: {sorted(_SUB_LON)}"
            )
        raw_bands = list(bands) if bands else ["ir_105"]
        self.bands = [_resolve_band(b) for b in raw_bands]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_prefix(self, resolution: str) -> str:
        return f"geo/mtg_{resolution}.icechunk"

    def _open_dataset(self, resolution: str) -> xr.Dataset:
        """Open the icechunk store for the given resolution as xr.Dataset."""
        prefix = self._store_prefix(resolution)
        logger.info("Opening icechunk store %s/%s", _BUCKET, prefix)
        store = _open_store(prefix)
        return xr.open_zarr(store)

    def _select_time(
        self, ds: xr.Dataset, t: dt.datetime,
    ) -> xr.Dataset:
        """Select the nearest time step to the requested datetime."""
        target = np.datetime64(t.replace(tzinfo=None), "ns")
        return ds.sel(time=target, method="nearest")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def data_at_time(self, t: dt.datetime, **_: Any) -> xr.Dataset:
        """Return the FCI scene closest to time ``t`` for ``self.bands[0]``.

        Returns
        -------
        xr.Dataset
            Dataset with ``Rad`` DataArray shaped ``(1, 1, y, x)``,
            y ascending (south -> north), x/y coordinates in **meters**
            (scan angle * satellite height).  ``Rad.attrs["orbital_parameters"]``
            contains ``projection_altitude`` and ``satellite_nominal_longitude``.
            ``ds.attrs["sweep_angle_axis"]`` is ``"y"`` (Meteosat convention).
        """
        band = self.bands[0]
        resolution = _BAND_RESOLUTION[band]
        ds = self._open_dataset(resolution)

        snap = self._select_time(ds, t)
        rad_2d = self._extract_radiance(snap, band)
        x_m, y_m_asc, rad_sn = self._build_coords(ds, rad_2d)

        sat_height = _SAT_HEIGHT[self.satellite]
        sub_lon = _SUB_LON[self.satellite]
        Rad = xr.DataArray(
            rad_sn[None, None, :, :],
            dims=("time", "band", "y", "x"),
            coords={"x": ("x", x_m), "y": ("y", y_m_asc)},
            name="Rad",
        )
        Rad.attrs["orbital_parameters"] = {
            "projection_altitude": sat_height,
            "satellite_nominal_longitude": sub_lon,
            "projection_longitude": sub_lon,
        }

        actual_time = None
        if "time" in ds.coords:
            sel_time = snap["time"].values if "time" in snap.coords else None
            if sel_time is not None:
                actual_time = str(np.datetime_as_string(
                    np.datetime64(sel_time, "ns"), unit="s",
                ))
        if actual_time:
            Rad.attrs["time_coverage_start"] = actual_time
            Rad.attrs["time_coverage_end"] = actual_time

        out = xr.Dataset({"Rad": Rad})
        out.attrs["sweep_angle_axis"] = "y"
        return out

    # ------------------------------------------------------------------
    # Radiance extraction (flexible to different store layouts)
    # ------------------------------------------------------------------

    def _extract_radiance(
        self, snap: xr.Dataset, band: str,
    ) -> np.ndarray:
        """Extract a 2-D (y, x) radiance array from the time-selected slice."""
        if band in snap.data_vars:
            arr = snap[band].values
            if arr.ndim == 2:
                return arr.astype(np.float32)
            return arr.squeeze().astype(np.float32)

        for vname in ("Rad", "radiance", "rad", "effective_radiance",
                       "toa_brightness_temperature"):
            if vname in snap.data_vars:
                da = snap[vname]
                if "band" in da.dims:
                    if "band" in da.coords:
                        band_vals = da.coords["band"].values
                        if band in band_vals:
                            return da.sel(band=band).values.astype(np.float32)
                    return da.isel(band=0).values.astype(np.float32)
                arr = da.values
                if arr.ndim == 2:
                    return arr.astype(np.float32)
                return arr.squeeze().astype(np.float32)

        for vname in snap.data_vars:
            da = snap[vname]
            if da.ndim >= 2:
                arr = da.values
                if arr.ndim == 2:
                    return arr.astype(np.float32)
                return arr.squeeze().astype(np.float32)

        raise KeyError(
            f"Cannot find radiance for band {band!r} in dataset. "
            f"Variables: {list(snap.data_vars)}"
        )

    # ------------------------------------------------------------------
    # Coordinate handling
    # ------------------------------------------------------------------

    def _build_coords(
        self, ds: xr.Dataset, rad_2d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build x/y metre coordinates and ensure y-ascending radiance."""
        ny, nx = rad_2d.shape
        x_vals = self._get_coord(ds, "x", nx)
        y_vals = self._get_coord(ds, "y", ny)

        x_range = float(np.abs(x_vals[-1] - x_vals[0]))
        sat_height = _SAT_HEIGHT[self.satellite]
        if x_range < 1.0:
            x_m = x_vals * sat_height
            y_m = y_vals * sat_height
        else:
            x_m = x_vals
            y_m = y_vals

        if y_m[0] > y_m[-1]:
            y_m = y_m[::-1]
            rad_2d = rad_2d[::-1, :]

        return x_m.astype(np.float64), y_m.astype(np.float64), rad_2d

    def _get_coord(
        self, ds: xr.Dataset, axis: str, expected_len: int,
    ) -> np.ndarray:
        """Retrieve the x or y coordinate array, synthesising if absent."""
        if axis in ds.coords:
            vals = ds.coords[axis].values.astype(np.float64)
            if len(vals) == expected_len:
                return vals

        alt_names = {
            "x": ["column", "longitude", "lon"],
            "y": ["row", "latitude", "lat"],
        }
        for name in alt_names.get(axis, []):
            if name in ds.coords:
                vals = ds.coords[name].values.astype(np.float64)
                if len(vals) == expected_len:
                    return vals

        # FCI FDHSI: 5568 x 5568 at 2 km, scale 5.58871e-05 rad/px
        grid_params = {
            5568: 5.58871e-05,   # 2 km
            11136: 2.79436e-05,  # 1 km
            22272: 1.39718e-05,  # 500 m
        }
        scale = grid_params.get(expected_len, 5.58871e-05)
        logger.warning(
            "No %s coordinate found in store; synthesising from FCI "
            "grid parameters", axis,
        )
        half = expected_len / 2.0
        if axis == "x":
            return (np.arange(expected_len) - half + 0.5) * scale
        else:
            return (half - 0.5 - np.arange(expected_len)) * scale

    def __repr__(self) -> str:
        return f"MTG(satellite={self.satellite!r}, bands={self.bands!r})"
