"""The virtualizarr tier of the source.coop geostationary archive.

``geo/virtualized/`` holds one icechunk store per satellite, product and
band -- ``gk2a_ami_fd_ir112_2026-09-27.icechunk``,
``goes19_radf_C14_2026-09-27.icechunk`` -- whose chunks are *references*
into the original L1b objects in the public NOAA buckets rather than a
second copy of the pixels.  A read therefore costs one ranged GET per
chunk instead of listing a bucket, downloading a NetCDF and handing it
to satpy: a full GK-2A disk comes back in ~2 s where the satpy path
takes ~60.

Three things stop these stores being a drop-in for
:mod:`stereo_winds.readers._geos_store`, and this module deals with all
three so the rest of the reader machinery does not have to:

*Virtual chunks need authorising.*  Icechunk refuses to follow a
reference into another bucket unless the caller names it when opening
the repository.  The repository itself lists the containers it
references, so they are read off it and authorised anonymously -- every
bucket referenced here is public -- rather than kept in a table that
would go stale.

*The time coordinate is called ``t``.*  Everything downstream selects on
``time``.

*The layouts differ by instrument.*  GOES stores ABI radiance with the
Planck coefficients beside it as variables; Himawari stores brightness
temperature already; GK-2A stores raw counts and the constants needed to
calibrate them.  Each is normalised to what the corresponding
non-virtual store would have given, so a caller cannot tell the tiers
apart -- see :func:`normalise`.

*It is not uniformly faster.*  Measured against this pipeline's access
pattern -- many bands for one timestamp -- the satpy path downloads every
band's file in one go and then reads each locally, while this tier opens
a separate store per band.  Five GK-2A bands from cold: 64 s through
satpy, 112 s through here.  A single band from cold goes the other way
(58 s vs 7 s), which is why the comparison has to be made over the bands
a retrieval actually loads.  GOES and Himawari are better served by
their materialised stores again (0.7 s and 0.8 s a band).  So for those
readers this tier sits *after* those stores and before the satpy
fallback, rather than in front of everything.  GOES is the exception:
:mod:`stereo_winds.readers.goes` has no materialised store to prefer
and reads one band per call either way, so it asks this tier first and
the public bucket second, caching each opened store so the cold open is
paid once per process rather than once per band.

*Coverage is a snapshot, not a feed.*  These stores are cut at a date
(the suffix in the name is a cutoff marker, not the coverage end -- a
store named ``2026-09-27`` may hold data only to April).  So this tier
is offered as a *preference*, checked against its real coverage, and the
live tier still answers for anything past its end.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import threading
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    "VIRTUAL_ROOT",
    "VirtualStoreName",
    "list_virtual_stores",
    "normalise",
    "open_virtual_dataset",
    "parse_store_name",
    "virtual_store_for",
]

#: Prefix under the bucket that holds the virtualized stores.
VIRTUAL_ROOT = "geo/virtualized"

# ``<satellite>_<product>_<band>[_<cutoff>].icechunk``.  The satellite
# may itself carry an underscore (``gk2a_ami``), so the trailing fields
# are matched from the right.  The cutoff is optional: the archive
# carries both dated snapshots (``..._C14_2026-09-27``) and undated
# stores (``himawari9_isatss_C14``), and an undated one is treated as
# the most current -- it is the one still being appended to.
_NAME_RE = re.compile(
    r"^(?P<satellite>.+?)_(?P<product>[A-Za-z0-9]+)_"
    r"(?P<band>[A-Za-z0-9]+)(?:_(?P<cutoff>\d{4}-\d{2}-\d{2}))?\.icechunk$"
)

#: Sort key for a store with no cutoff in its name: later than any date.
_NO_CUTOFF = "9999-99-99"

_LISTING_LOCK = threading.Lock()
_LISTINGS: dict[tuple[str, str], list[str]] = {}


class VirtualStoreName(tuple):
    """One parsed store name: satellite, product, band, cutoff date."""

    __slots__ = ()

    def __new__(cls, satellite: str, product: str, band: str, cutoff: str | None = None):
        return super().__new__(cls, (satellite, product, band, cutoff))

    satellite = property(lambda self: self[0])
    product = property(lambda self: self[1])
    band = property(lambda self: self[2])
    cutoff = property(lambda self: self[3])


def parse_store_name(name: str) -> VirtualStoreName | None:
    """Split a virtualized store's file name, or None if it is not one."""
    match = _NAME_RE.match(name)
    if match is None:
        return None
    return VirtualStoreName(**match.groupdict())


def list_virtual_stores(bucket: str, endpoint: str) -> list[str]:
    """Every store name under :data:`VIRTUAL_ROOT`, listed once per process.

    A failure to list is not fatal: the caller falls back to the live
    tier, which is what it would have used anyway.
    """
    key = (bucket, endpoint)
    with _LISTING_LOCK:
        cached = _LISTINGS.get(key)
    if cached is not None:
        return cached
    try:
        import s3fs

        fs = s3fs.S3FileSystem(anon=True, endpoint_url=endpoint)
        names = sorted(k.split("/")[-1] for k in fs.ls(f"{bucket}/{VIRTUAL_ROOT}"))
    except Exception:
        logger.info(
            "Could not list %s/%s; the virtualized tier will be skipped",
            bucket,
            VIRTUAL_ROOT,
            exc_info=True,
        )
        names = []
    with _LISTING_LOCK:
        _LISTINGS[key] = names
    return names


def virtual_store_for(
    bucket: str,
    endpoint: str,
    satellite: str,
    band: str,
    when: "dt.datetime | None" = None,
) -> list[str]:
    """Prefixes of the stores holding ``band`` for ``satellite``.

    Newest cutoff first, because the newest is the one most likely to
    cover a recent request; the caller still checks real coverage, since
    the cutoff in the name is not the coverage end.

    ``when`` drops the stores that cannot hold that instant: a store
    cut at 2024-12-31 has nothing from 2026, and opening it to find
    that out is not free.  Each open reads a time axis of tens of
    thousands of entries and is cached for the life of the process, so
    a timestamp past *every* store -- which is what a satellite looks
    like once its archive ends -- otherwise opens the whole stack, for
    every band, and keeps them all.  That is what wedged a sweep at the
    end of GK-2A's coverage: resident memory reached 57 GB and three
    restarts made no progress.
    """
    wanted_sat = satellite.lower()
    wanted_band = band.lower()
    cutoff_floor = when.strftime("%Y-%m-%d") if when is not None else None
    hits = []
    for name in list_virtual_stores(bucket, endpoint):
        parsed = parse_store_name(name)
        if parsed is None:
            continue
        if parsed.satellite.lower() != wanted_sat or parsed.band.lower() != wanted_band:
            continue
        # An undated store is the rolling one and is never ruled out.
        if cutoff_floor is not None and parsed.cutoff is not None and parsed.cutoff < cutoff_floor:
            continue
        hits.append((parsed.cutoff or _NO_CUTOFF, f"{VIRTUAL_ROOT}/{name}"))
    return [prefix for _, prefix in sorted(hits, reverse=True)]


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


def _authorisation(repo: Any) -> dict:
    """Anonymous credentials for every container the repo references.

    Read off the repository rather than kept in a table: the archive
    adds satellites, and a table would silently stop authorising one.
    Anonymous is correct for all of them -- they are the public NOAA
    open-data buckets -- and an anonymous credential on a bucket that
    turns out to need auth fails loudly on the first chunk read.
    """
    import icechunk

    containers = list(getattr(repo.config, "virtual_chunk_containers", {}) or {})
    if not containers:
        return {}
    return icechunk.containers_credentials(
        {url: icechunk.s3_credentials(anonymous=True) for url in containers}
    )


def open_virtual_dataset(bucket: str, endpoint: str, prefix: str) -> xr.Dataset:
    """Open one virtualized store, with its virtual chunks authorised.

    The repository is opened twice: once to ask which containers it
    references, once with those authorised.  Both are metadata reads
    against the store's own bucket, and the result is cached by the
    caller, so the second open costs far less than hard-coding the
    bucket list and getting it wrong.
    """
    import icechunk

    storage = icechunk.s3_storage(
        bucket=bucket,
        prefix=prefix,
        endpoint_url=endpoint,
        anonymous=True,
        force_path_style=True,
    )
    probe = icechunk.Repository.open(storage)
    auth = _authorisation(probe)
    # Reopen only when there is something to authorise; a store with
    # no virtual chunks is already usable as opened.
    repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth) if auth else probe
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def _scan_angles(ds: xr.Dataset, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Column/row scan angles in radians, from CGMS navigation constants.

    ``CFAC``/``COFF`` are the CGMS LRIT/HRIT scaling of a fixed grid.
    The scaling is defined in *degrees*: column ``c`` sits at
    ``(c - COFF) * 2**16 / CFAC`` degrees, which for GK-2A's
    ``CFAC = 20425338.9`` gives 3.209e-3 deg = 5.6e-5 rad per pixel --
    the 2 km sample spacing, and the same as ABI's.  Getting either the
    power or the unit wrong is not a small error: ``2**-16`` in place of
    ``2**16`` yields 7.5e-13 rad per pixel, a full disk spanning four
    nanoradians, and geolocation that is wrong everywhere while still
    looking like a plausible monotonic axis.

    ``LFAC`` is negative for an instrument whose rows run north to
    south, which makes ``y`` descend; the caller re-orients it.
    """
    try:
        cfac = float(ds.attrs["cfac"])
        coff = float(ds.attrs["coff"])
        lfac = float(ds.attrs["lfac"])
        loff = float(ds.attrs["loff"])
    except (KeyError, TypeError, ValueError):
        return None
    if not cfac or not lfac:
        return None
    # CGMS numbers columns and rows from 1, so array index i is column
    # i + 1.  Dropping the +1 shifts the whole grid by exactly one pixel
    # -- 2 km on the ground for these instruments -- while leaving the
    # spacing, and therefore every sanity check on the axis, correct.
    cols = np.arange(1, nx + 1, dtype=np.float64)
    rows = np.arange(1, ny + 1, dtype=np.float64)
    x = np.radians((cols - coff) * (2.0**16) / cfac)
    y = np.radians((rows - loff) * (2.0**16) / lfac)
    return x, y


# ---------------------------------------------------------------------------
# Calibration, per instrument family
# ---------------------------------------------------------------------------

#: ``product`` field -> the family that knows how to read it.
_FAMILY_BY_PRODUCT = {
    "radf": "abi",
    "isatss": "ahi",
    "fd": "ami",
}

_AMI_WAVELENGTH_LOCK = threading.Lock()
_AMI_WAVELENGTHS: dict[str, float] | None = None


def _ami_wavelengths() -> dict[str, float]:
    """Band -> central wavelength in um, read from satpy's own reader table.

    Read from satpy rather than copied here because the brightness
    temperature this module produces has to equal satpy's to the last
    bit, and satpy's default calibration mode (PYSPECTRAL) converts
    radiance using *this* number.  A copy would drift the two apart
    silently, which is the one failure a units bug hides best.
    """
    global _AMI_WAVELENGTHS
    with _AMI_WAVELENGTH_LOCK:
        if _AMI_WAVELENGTHS is not None:
            return _AMI_WAVELENGTHS
        import satpy
        import yaml
        from pathlib import Path

        # satpy's reader tables name python objects with `!!python/name:`
        # tags, which the safe loader refuses.  Only the dataset
        # wavelengths are wanted, so the tags are read as None rather
        # than resolved -- resolving them would execute whatever the
        # file names.
        class _IgnoreTags(yaml.SafeLoader):
            pass

        _IgnoreTags.add_multi_constructor(
            "tag:yaml.org,2002:python/name:", lambda loader, suffix, node: None
        )

        path = Path(satpy.__file__).parent / "etc" / "readers" / "ami_l1b.yaml"
        with open(path) as handle:
            table = yaml.load(handle, Loader=_IgnoreTags)
        out: dict[str, float] = {}
        for entry in (table.get("datasets") or {}).values():
            name = entry.get("name")
            wavelength = entry.get("wavelength")
            if name and isinstance(wavelength, list) and len(wavelength) == 3:
                out[str(name).lower()] = float(wavelength[1])
        _AMI_WAVELENGTHS = out
        return out


def _calibrate_ami(ds: xr.Dataset, band: str) -> xr.DataArray:
    """GK-2A counts -> brightness temperature, as satpy's ami_l1b does.

    Mirrors ``AMIL1bNetCDF.get_dataset`` in its default configuration:
    the top two bits are the data quality flag and only "no error"
    pixels survive, the low 13 are the count, radiance is the affine
    calibration from the file (a *negative* gain), and the conversion to
    brightness temperature is pyspectral's inverse Planck at the band's
    central wavelength -- not the in-file ``Teff_to_Tbb`` polynomial,
    which satpy only uses when asked for ``calib_mode="FILE"``.
    """
    from pyspectral.blackbody import blackbody_wn_rad2temp as rad2temp

    counts = ds["image_pixel_values"]
    valid_bits = int(counts.attrs["number_of_valid_bits_per_pixel"])

    # Top two bits: 00 no error, 01 conditional, 10 off-earth, 11 error.
    flags = counts & 0b1100000000000000
    dn = counts & ((1 << valid_bits) - 1)

    gain = float(ds.attrs["DN_to_Radiance_Gain"])
    offset = float(ds.attrs["DN_to_Radiance_Offset"])
    radiance = gain * dn + offset
    radiance = radiance.where(flags == 0)

    micron = _ami_wavelengths()[band.lower()]
    # pyspectral works in SI: wavenumber in m^-1, radiance in W m^-2 sr^-1 m.
    wavenumber = 1.0 / (micron / 1e6)
    brightness = xr.apply_ufunc(
        lambda arr: rad2temp(wavenumber, arr * 1e-5),
        radiance,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    brightness.attrs = dict(counts.attrs)
    return brightness


#: ABI variables carrying the Planck constants beside the radiance.  In
#: the live per-band stores these are attributes of ``Rad``; here they
#: are variables, and ``data_loading._rad_to_bt`` reads attributes.
_ABI_PLANCK = ("planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2")


def _calibrate_abi(ds: xr.Dataset, band: str) -> xr.DataArray:
    """ABI radiance, with the Planck coefficients lifted onto its attrs."""
    radiance = ds["Rad"]
    attrs = dict(radiance.attrs)
    for name in _ABI_PLANCK:
        if name not in ds:
            continue
        values = np.asarray(ds[name].values).ravel()
        if values.size:
            attrs[name] = float(values[0])
    radiance = radiance.copy()
    radiance.attrs = attrs
    return radiance


def _calibrate_ahi(ds: xr.Dataset, band: str) -> xr.DataArray:
    """AHI brightness temperature, which the store already holds."""
    return ds["Sectorized_CMI"]


_CALIBRATORS = {"ami": _calibrate_ami, "abi": _calibrate_abi, "ahi": _calibrate_ahi}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _scan_start(ds: xr.Dataset) -> np.ndarray:
    """When each scan *started*, which is what the live tier indexes on.

    Not simply ``t``: ABI documents ``t`` as the mid-point between the
    start and end of the scan, so a full disk labelled 06:05:05 is the
    scan that began at 06:00:20.  Indexing on it would hand every
    caller a frame up to half a scan away from the time it asked for --
    and a wind retrieval takes the interval between frames as its
    baseline, so a systematic offset there biases every vector rather
    than merely blurring them.

    ``time_bounds`` carries the real interval where the instrument
    records one; otherwise ``t`` is taken at face value, which is
    correct for the instruments that record an observation start.
    """
    if "time_bounds" in ds:
        bounds = np.asarray(ds["time_bounds"].values)
        if bounds.ndim == 2 and bounds.shape[1] == 2:
            return np.asarray(bounds[:, 0], "datetime64[ns]")
    long_name = str(ds["t"].attrs.get("long_name", "")).lower()
    if "mid-point" in long_name or "midpoint" in long_name:
        logger.warning(
            "virtualized store indexes on a scan mid-point and carries no "
            "time_bounds; times may be offset by half a scan"
        )
    return np.asarray(ds["t"].values, "datetime64[ns]")


def family_for(parsed: VirtualStoreName) -> str | None:
    """Which instrument family a parsed store name belongs to."""
    return _FAMILY_BY_PRODUCT.get(parsed.product.lower())


def normalise(ds: xr.Dataset, parsed: VirtualStoreName, band: str) -> xr.Dataset:
    """Present a virtualized store the way the live tier presents itself.

    The result carries exactly one data variable, named ``band``, over a
    ``time`` coordinate, holding the same quantity the non-virtual store
    for this instrument would have held.  That is what lets the rest of
    :mod:`stereo_winds.readers._geos_store` -- store selection, nearest
    time, radiance extraction -- work against either tier without
    knowing which it has.

    The calibration stays lazy: it is arithmetic over the store's dask
    arrays, so selecting one timestamp afterwards computes one
    timestamp, not the whole archive.
    """
    family = family_for(parsed)
    if family is None:
        raise KeyError(
            f"no calibration known for virtualized product "
            f"{parsed.product!r} (store {parsed.satellite}_{parsed.product})"
        )

    values = _CALIBRATORS[family](ds, parsed.band)
    values = values.rename({"t": "time"}) if "t" in values.dims else values

    out = xr.Dataset({band: values})
    out.attrs = dict(ds.attrs)
    # Overwrite rather than fill in: renaming the dimension carried the
    # store's own time values across, and for ABI those are scan
    # mid-points.  See _scan_start.
    if "time" in out.dims:
        out = out.assign_coords(time=("time", _scan_start(ds)))

    ny, nx = values.sizes.get("y"), values.sizes.get("x")
    if ny and nx:
        angles = _scan_angles(ds, nx, ny)
        if angles is not None:
            out = out.assign_coords(x=("x", angles[0]), y=("y", angles[1]))

    return _attach_navigation(out, ds)


def _attach_navigation(out: xr.Dataset, ds: xr.Dataset) -> xr.Dataset:
    """Carry the store's per-scan navigation into the scene metadata.

    Under the names :func:`stereo_winds.readers._geos_meta.
    scene_orbital_parameters` looks for, and in its units.  Two
    conversions are needed and neither is cosmetic: the archive records
    the sub-satellite longitude in *radians*, and records the satellite
    position as a distance from the Earth's centre where the projection
    wants the perspective height above the equator.  Left alone, the
    reader falls back to a nominal longitude and warns that navigation
    may be wrong -- which it would be, by however far the spacecraft has
    drifted since the nominal was written down.
    """

    def _series(name: str) -> xr.DataArray | None:
        if name not in ds:
            return None
        field = ds[name]
        return field.rename({"t": "time"}) if "t" in field.dims else field

    sub_lon = _series("nav_sub_longitude")
    if sub_lon is not None:
        out["projection_longitude"] = np.degrees(sub_lon)

    height = _series("nav_satellite_height")
    equatorial = _series("nav_earth_equatorial_radius")
    if height is not None and equatorial is not None:
        out["projection_altitude"] = height - equatorial

    return out


def open_normalised(bucket: str, endpoint: str, prefix: str, band: str) -> xr.Dataset:
    """Open one virtualized store and normalise it for ``band``."""
    parsed = parse_store_name(prefix.rsplit("/", 1)[-1])
    if parsed is None:
        raise ValueError(f"{prefix} is not a virtualized store name")
    return normalise(open_virtual_dataset(bucket, endpoint, prefix), parsed, band)
