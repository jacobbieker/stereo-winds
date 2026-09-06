"""Icechunk-backed GK-2A AMI reader for stereo-winds.

Reads GK-2A (GEO-KOMPSAT-2A) AMI Level-1b radiance from icechunk stores
hosted at source.coop and returns a dataset shaped exactly like the GOES
reader: ``Rad`` as ``(time, band, y, x)`` oriented south->north, ``x``/``y``
in **meters** (scan angle x perspective height), and
``Rad.attrs["orbital_parameters"]`` carrying ``projection_altitude`` and
``satellite_nominal_longitude``.

Icechunk stores
---------------
- ``geo/gk2a_500m.icechunk``  -- AMI high-res VIS bands (500 m)
- ``geo/gk2a_1000m.icechunk`` -- AMI mid-res VIS/NIR bands (1 km)
- ``geo/gk2a_2000m.icechunk`` -- AMI IR/WV bands (2 km)

Requires ``icechunk`` and ``zarr>=3``.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import numpy as np
import xarray as xr

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
    """Accept either AMI-native (IR112) or ABI-style (C14) band names."""
    if band in _BAND_RESOLUTION:
        return band
    ami = _ABI_TO_AMI.get(band)
    if ami is not None:
        return ami
    raise ValueError(
        f"Unknown band {band!r}. Use AMI names (VI004, IR112, ...) or ABI "
        f"names (C01-C16). Known ABI->AMI map: {_ABI_TO_AMI}"
    )


class GK2A:
    """Icechunk-backed GK-2A AMI reader.

    Parameters
    ----------
    satellite : "gk2a"
    bands : list with a single AMI band, e.g. ``["IR112"]`` or ``["C14"]``
    """

    def __init__(
        self,
        satellite: str = "gk2a",
        bands: list[str] | None = None,
    ) -> None:
        self.satellite = satellite
        if satellite != "gk2a":
            raise ValueError(
                f"Unknown satellite {satellite!r}. Only 'gk2a' is supported."
            )
        raw_bands = list(bands) if bands else ["IR112"]
        self.bands = [_resolve_band(b) for b in raw_bands]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_prefix(self, resolution: str) -> str:
        return f"geo/gk2a_{resolution}.icechunk"

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
        """Return the AMI scene closest to time ``t`` for ``self.bands[0]``.

        Returns
        -------
        xr.Dataset
            Dataset with ``Rad`` DataArray shaped ``(1, 1, y, x)``,
            y ascending (south -> north), x/y coordinates in **meters**
            (scan angle * satellite height). ``Rad.attrs["orbital_parameters"]``
            contains ``projection_altitude`` and ``satellite_nominal_longitude``.
            ``ds.attrs["sweep_angle_axis"]`` is ``"y"`` (AMI convention).
        """
        band = self.bands[0]
        resolution = _BAND_RESOLUTION[band]
        ds = self._open_dataset(resolution)

        snap = self._select_time(ds, t)
        rad_2d = self._extract_radiance(snap, band)
        x_m, y_m_asc, rad_sn = self._build_coords(ds, rad_2d)

        Rad = xr.DataArray(
            rad_sn[None, None, :, :],
            dims=("time", "band", "y", "x"),
            coords={"x": ("x", x_m), "y": ("y", y_m_asc)},
            name="Rad",
        )
        Rad.attrs["orbital_parameters"] = {
            "projection_altitude": _SAT_HEIGHT,
            "satellite_nominal_longitude": _SUB_LON,
            "projection_longitude": _SUB_LON,
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

        for vname in ("Rad", "radiance", "rad", "toa_brightness_temperature"):
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
        if x_range < 1.0:
            x_m = x_vals * _SAT_HEIGHT
            y_m = y_vals * _SAT_HEIGHT
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
            "x": ["column", "longitude", "lon", "phgeo"],
            "y": ["row", "latitude", "lat", "thgeo"],
        }
        for name in alt_names.get(axis, []):
            if name in ds.coords:
                vals = ds.coords[name].values.astype(np.float64)
                if len(vals) == expected_len:
                    return vals

        _GRID_SCALE = {5500: 5.6e-05, 11000: 2.8e-05, 22000: 1.4e-05}
        scale = _GRID_SCALE.get(expected_len, 5.6e-05)
        logger.warning(
            "No %s coordinate found in store; synthesising with "
            "scale=%.2e rad/px (grid %d)", axis, scale, expected_len,
        )
        half = expected_len / 2.0
        if axis == "x":
            return (np.arange(expected_len) - half + 0.5) * scale
        else:
            return (half - 0.5 - np.arange(expected_len)) * scale

    def __repr__(self) -> str:
        return f"GK2A(satellite={self.satellite!r}, bands={self.bands!r})"
