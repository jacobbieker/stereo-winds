"""Writing global AMV mosaics to an icechunk store.

One mosaic per timestep, appended along ``time`` and committed
individually, so a run can be resumed and a partially written store is
still a valid store up to its last commit.

Shared by the retrieval pipeline (``scripts/infer_student_global_ring.py``,
which writes each mosaic as it is produced) and by
``scripts/write_mosaics_to_icechunk.py``, which ingests mosaics already
written as NetCDF.
"""

from __future__ import annotations

import datetime as dt
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

TIME_TAG_FMT = "%Y%m%dT%H%M"


def time_tag(t: datetime) -> str:
    """Canonical timestamp tag used in filenames and commit messages."""
    return t.strftime(TIME_TAG_FMT)


# Pin the time encoding on creation.  Without this xarray picks units from
# the first timestamp ("days since <t0>") and later appends re-encode
# against a different unit, silently corrupting every appended timestamp.
_TIME_ENCODING = {
    "units": "seconds since 1970-01-01T00:00:00",
    "calendar": "proleptic_gregorian",
    "dtype": "int64",
}


def icechunk_storage(
    uri: str,
    endpoint_url: str | None = None,
    region: str | None = None,
    anonymous: bool = False,
    force_path_style: bool = False,
):
    """Build icechunk Storage for ``s3://bucket/prefix`` or a local path."""
    import icechunk

    parsed = urlparse(uri)
    if parsed.scheme in ("s3", "s3a"):
        return icechunk.s3_storage(
            bucket=parsed.netloc,
            prefix=parsed.path.lstrip("/") or None,
            region=region,
            endpoint_url=endpoint_url,
            anonymous=True if anonymous else None,
            from_env=None if anonymous else True,
            force_path_style=force_path_style,
        )
    if parsed.scheme in ("", "file"):
        path = Path(parsed.path if parsed.scheme == "file" else uri)
        path.mkdir(parents=True, exist_ok=True)
        return icechunk.local_filesystem_storage(str(path))
    raise ValueError(
        f"Unsupported icechunk store URI {uri!r} — use s3://bucket/prefix "
        f"or a local directory path"
    )


def open_icechunk_repo(uri: str, **storage_kwargs):
    """Open the icechunk repository at ``uri``, creating it if absent."""
    import icechunk

    storage = icechunk_storage(uri, **storage_kwargs)
    repo = icechunk.Repository.open_or_create(storage)
    logger.info("Icechunk store ready: %s", uri)
    return repo


def icechunk_existing_times(repo, branch: str = "main") -> set[datetime]:
    """Timestamps already committed to the store (empty if it is new)."""
    try:
        ds = xr.open_zarr(repo.readonly_session(branch).store, consolidated=False)
    except Exception:
        logger.info("Icechunk store has no dataset yet — starting fresh")
        return set()
    if "time" not in ds.coords:
        return set()
    times = {pd.Timestamp(v).to_pydatetime() for v in ds["time"].values}
    logger.info("Icechunk store already holds %d timestamp(s)", len(times))
    return times


#: Attributes describing *one* mosaic rather than the whole series.
#:
#: Written as group attributes they are silently overwritten by every
#: later append, so a store would end up describing all of its
#: timesteps by whatever the last one happened to be -- a degraded
#: cycle becoming invisible the moment a clean one lands.  They are
#: promoted to ``time``-dimensioned variables instead, so each timestep
#: keeps its own.
_PER_TIMESTEP_ATTRS = (
    "quality_degraded",
    "quality_note",
    "degraded_satellites",
    "bands_requested",
    "bands_missing",
    "n_bands_requested",
    "n_bands_missing",
    "flow_bands_missing",
    "satellites_expected",
    "satellites_contributing",
    "satellites_missing",
    "satellites_empty",
    "n_satellites_expected",
    "n_satellites_contributing",
    "n_satellites_missing",
    "mosaic_complete",
    "nominal_time",
)


def _timestep_value(value) -> np.ndarray:
    """One attribute value as a length-1 array, ready to append.

    Booleans and integers stay numeric so they can be compared and
    summed; everything else becomes a string, which zarr v3 stores as a
    variable-length UTF-8 array.
    """
    if isinstance(value, (bool, np.bool_)):
        return np.array([int(value)], dtype="int8")
    if isinstance(value, (int, np.integer)):
        return np.array([int(value)], dtype="int64")
    if isinstance(value, (float, np.floating)):
        return np.array([float(value)], dtype="float64")
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.array([",".join(str(v) for v in value)], dtype=object)
    return np.array(["" if value is None else str(value)], dtype=object)


def _mosaic_with_time(ds_global: xr.Dataset, t0: datetime) -> xr.Dataset:
    """Add a length-1 time dimension so the mosaic can be appended."""
    ds = ds_global.expand_dims(time=[np.datetime64(t0, "ns")])
    # Attributes that vary per timestamp belong on a variable, not the
    # store; keep only what is invariant across the whole time series.
    per_timestep = {k: ds_global.attrs[k] for k in _PER_TIMESTEP_ATTRS if k in ds_global.attrs}
    ds.attrs = {
        k: v
        for k, v in ds_global.attrs.items()
        if k not in ("time", "satellites") and k not in per_timestep
    }
    ds.attrs["satellites"] = _satellites_attr(ds_global.attrs.get("satellites"))
    for name, value in per_timestep.items():
        ds[name] = ("time", _timestep_value(value))
    # The contributing satellites are a property of this timestep too;
    # the group attribute above is the store-wide vocabulary.  Only
    # derive it when the mosaic did not already state one.
    if "satellites_contributing" not in ds:
        ds["satellites_contributing"] = ("time", _timestep_value(ds.attrs["satellites"]))

    # Encoding picked up from a NetCDF source (chunking, compression,
    # fill values) does not apply to the zarr we are writing.
    for var in ds.variables.values():
        var.encoding = {}
    return ds


def _satellites_attr(value) -> str:
    """Normalise the satellite list to a comma-separated string.

    It arrives as a real list from the pipeline, but as its ``repr`` after
    a NetCDF round trip — and joining a string yields its characters.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            import ast

            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return text
            if isinstance(parsed, (list, tuple)):
                return ",".join(str(v) for v in parsed)
        return text
    if isinstance(value, (list, tuple, np.ndarray)):
        return ",".join(str(v) for v in value)
    return str(value)


def _store_has_dataset(session) -> bool:
    """True if the store already holds a time-dimensioned dataset."""
    try:
        ds = xr.open_zarr(session.store, consolidated=False)
    except Exception:
        return False
    return "time" in ds.coords


# Sentinel used by the mosaic for "no satellite contributed here".
NO_SOURCE = -1


def mosaic_satellites(ds: xr.Dataset) -> list[str]:
    """Satellite names behind this mosaic's ``source_satellite_index``."""
    var = ds.get("source_satellite_index")
    if var is not None:
        meanings = str(var.attrs.get("flag_meanings", "")).split()
        if meanings:
            return meanings
    return [s for s in _satellites_attr(ds.attrs.get("satellites")).split(",") if s]


def align_source_codes(
    ds: xr.Dataset,
    vocabulary: list[str],
) -> xr.Dataset:
    """Remap ``source_satellite_index`` onto a vocabulary shared by the store.

    The codes a mosaic carries index *its own* satellite list, and that
    list varies with which satellites contributed at that timestamp — so
    code 2 can mean gk2a in one timestep and himawari9 in the next.
    Stacked into one array under a single ``flag_meanings`` they would
    silently misattribute provenance, so every timestep is remapped onto
    one vocabulary, extended in place as new satellites appear.  Names are
    only ever appended, so codes already written stay valid.
    """
    if "source_satellite_index" not in ds:
        return ds

    names = mosaic_satellites(ds)
    for name in names:
        if name not in vocabulary:
            vocabulary.append(name)

    # file code -> store code, with the sentinel passed through.
    lookup = np.full(len(names) + 1, NO_SOURCE, dtype=np.int8)
    for code, name in enumerate(names):
        lookup[code] = vocabulary.index(name)

    codes = ds["source_satellite_index"].values
    remapped = np.where(codes < 0, np.int8(NO_SOURCE), lookup[np.clip(codes, 0, len(names))])

    ds = ds.copy()
    ds["source_satellite_index"] = (
        ds["source_satellite_index"].dims,
        remapped.astype(np.int8),
    )
    ds["source_satellite_index"].attrs.update(
        {
            "long_name": "index into the satellites attribute of the satellite "
            "that won each cell",
            "flag_values": list(range(len(vocabulary))),
            "flag_meanings": " ".join(vocabulary),
            "no_source_index": NO_SOURCE,
        }
    )
    return ds


def _update_store_vocabulary(session, vocabulary: list[str]) -> None:
    """Rewrite the stored flag attributes after the vocabulary grew."""
    try:
        import zarr

        group = zarr.open_group(session.store, mode="a")
        array = group["source_satellite_index"]
        array.attrs["flag_values"] = list(range(len(vocabulary)))
        array.attrs["flag_meanings"] = " ".join(vocabulary)
    except Exception:  # pragma: no cover - depends on zarr internals
        logger.exception(
            "Could not update the stored satellite vocabulary; "
            "source_satellite_index may under-describe itself"
        )


def store_vocabulary(repo, branch: str = "main") -> list[str]:
    """Satellite vocabulary the store already uses (empty if it is new)."""
    try:
        ds = xr.open_zarr(repo.readonly_session(branch).store, consolidated=False)
    except Exception:
        return []
    var = ds.get("source_satellite_index")
    if var is None:
        return []
    return str(var.attrs.get("flag_meanings", "")).split()


def _store_variables(session) -> set[str]:
    """Names already present in the store, empty if it has no dataset."""
    try:
        ds = xr.open_zarr(session.store, consolidated=False)
    except Exception:
        return set()
    return set(ds.variables)


def _time_index(session, t0: datetime) -> int | None:
    """Position of ``t0`` on the store's time axis, or None if absent."""
    try:
        ds = xr.open_zarr(session.store, consolidated=False)
    except Exception:
        return None
    if "time" not in ds.coords:
        return None
    times = np.asarray(ds["time"].values, "datetime64[ns]")
    hits = np.flatnonzero(times == np.datetime64(t0, "ns"))
    return int(hits[0]) if hits.size else None


def _align_for_append(ds: xr.Dataset, existing: set[str]) -> xr.Dataset:
    """Match ``ds`` to the per-timestep variables the store already has.

    A store written before the quality variables existed has none of
    them, and appending a dataset that carries extra variables fails.
    Equally, a store that has them needs every append to supply them,
    or the arrays fall out of step with the time axis.
    """
    per_timestep = set(_PER_TIMESTEP_ATTRS) | {"satellites_contributing"}
    extra = [n for n in ds.data_vars if n in per_timestep and n not in existing]
    if extra:
        logger.info(
            "Store predates the per-timestep quality variables; " "dropping %s from this append",
            ", ".join(sorted(extra)),
        )
        ds = ds.drop_vars(extra)
    absent = [n for n in existing & per_timestep if n not in ds]
    for name in absent:
        logger.debug("Filling absent per-timestep variable %s", name)
        ds[name] = ("time", _timestep_value(""))
    return ds


def write_mosaic_to_icechunk(
    repo,
    ds_global: xr.Dataset,
    t0: datetime,
    branch: str = "main",
    chunk: int = 1024,
    vocabulary: list[str] | None = None,
    replace: bool = False,
) -> str:
    """Write one global mosaic to the icechunk store as a new commit.

    ``vocabulary`` is the store's satellite list; pass the same list on
    every call so per-timestep source codes stay comparable.  It is
    extended in place when a mosaic introduces a new satellite.

    A timestamp already in the store is left alone unless ``replace``
    is set, in which case that timestep is overwritten in place.  That
    is what lets a resumed satellite correct an earlier, degraded
    mosaic: appending would duplicate the timestamp, and skipping would
    leave the store permanently short of a satellite that has since
    been retrieved.
    """
    if vocabulary is None:
        vocabulary = store_vocabulary(repo, branch)
    ds_global = align_source_codes(ds_global, vocabulary)
    ds = _mosaic_with_time(ds_global, t0)
    ds.attrs["satellites"] = ",".join(vocabulary) or ds.attrs.get("satellites", "")
    session = repo.writable_session(branch)

    existing_at = _time_index(session, t0) if _store_has_dataset(session) else None
    if existing_at is not None:
        if not replace:
            logger.info("%s already in the store; leaving it alone", time_tag(t0))
            return "skipped"
        # Only variables on the time axis take part in a region write;
        # the grid coordinates are shared by every timestep and are
        # already in the store.
        region_ds = ds.drop_vars([n for n in ds.variables if "time" not in ds[n].dims])
        region_ds = region_ds.drop_vars("time")
        region_ds = _align_for_append(region_ds, _store_variables(session))
        region_ds.to_zarr(
            session.store, consolidated=False, region={"time": slice(existing_at, existing_at + 1)}
        )
        if vocabulary and "source_satellite_index" in region_ds:
            _update_store_vocabulary(session, vocabulary)
        commit = session.commit(f"student AMV global mosaic {time_tag(t0)} (replaced)")
        logger.info("Replaced %s in icechunk (%s)", time_tag(t0), commit)
        return "replaced"

    if not _store_has_dataset(session):
        encoding: dict[str, dict] = {"time": dict(_TIME_ENCODING)}
        for name, var in ds.data_vars.items():
            # The per-timestep quality variables are 1-D; only the grids
            # get a spatial chunking.
            if var.ndim < 3:
                continue
            encoding[name] = {
                "chunks": (1, min(chunk, var.shape[1]), min(chunk, var.shape[2])),
            }
        ds.to_zarr(session.store, mode="w", consolidated=False, zarr_format=3, encoding=encoding)
        logger.info("Created icechunk dataset (chunks %d x %d)", chunk, chunk)
        outcome = "created"
    else:
        ds = _align_for_append(ds, _store_variables(session))
        ds.to_zarr(session.store, mode="a-", append_dim="time", consolidated=False)
        outcome = "appended"

    if vocabulary and "source_satellite_index" in ds:
        # Always restate it: appending leaves an existing array's
        # attributes untouched, and the caller's vocabulary may already
        # have grown past what the store recorded.
        _update_store_vocabulary(session, vocabulary)

    commit = session.commit(f"student AMV global mosaic {time_tag(t0)}")
    logger.info("Committed %s to icechunk (%s)", time_tag(t0), commit)
    return outcome
