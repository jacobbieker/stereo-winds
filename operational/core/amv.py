"""Per-satellite AMV retrieval, as an independently resumable step.

One satellite is one unit of work: it either produces a complete NetCDF
at a canonical path or it fails, and a failure leaves the other
satellites' output untouched.  A resumed run therefore re-does only the
satellites that did not finish, which is the whole point of splitting
the ring up per satellite.

Everything here is plain Python — the Dagster asset wrapping it lives
elsewhere — so the step can be exercised without an orchestrator.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import xarray as xr

from operational.adapters import ring as ring_adapter

logger = logging.getLogger(__name__)

#: Suffix of the in-progress file that ``os.replace`` later renames.
TMP_SUFFIX = ".tmp"

#: Age past which an abandoned temp file is assumed to be dead, not
#: in-flight.  Comfortably longer than any single full-disk write.
STALE_TMP_AGE_S = 6 * 3600.0


@dataclass(frozen=True, eq=False)
class AmvResult:
    """Outcome of one satellite's retrieval at one timestamp."""

    sat_id: str
    timestamp: datetime
    path: Path
    # A full disk in a repr would drown every log line that carries one.
    dataset: xr.Dataset = field(repr=False)
    reused: bool
    n_bands_missing: int
    bands_missing: tuple[str, ...]
    quality_degraded: bool
    quality_note: str = ""


def _as_int(value: Any, default: int = 0) -> int:
    """Coerce an attr to ``int``, falling back when it is absent or junk.

    Attrs survive a NetCDF round trip as numpy scalars or strings
    depending on the writer, and an old file may not carry them at all.
    A missing or unparsable quality attr means "nothing known", never a
    crash in the middle of a run.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Ignoring unparsable quality attr %r", value)
        return default


def _as_flag(value: Any) -> bool:
    """Read a 0/1 quality flag, failing *closed* when it cannot be read.

    Deliberately the opposite of :func:`_as_int`'s fallback.  A count
    that cannot be parsed is genuinely unknown and zero is the honest
    answer, but a quality flag that cannot be parsed must not report
    "fine" — an unreadable ``quality_degraded`` is a reason to treat the
    retrieval as suspect, not a reason to trust it.  Absent entirely is
    different from present-but-unreadable: upstream simply predates the
    attr, so that stays False.
    """
    if value is None:
        return False
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        logger.warning("Unreadable quality_degraded attr %r — treating as degraded", value)
        return True


def _as_band_tuple(value: Any) -> tuple[str, ...]:
    """Split a comma-joined band-name attr into a tuple of names."""
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(b for b in (p.strip() for p in value.split(",")) if b)
    if isinstance(value, (list, tuple)):
        return tuple(str(b) for b in value)
    return (str(value),)


def quality_from_attrs(attrs: dict) -> dict[str, Any]:
    """Lift the upstream quality attrs off a retrieval's attrs.

    Notes
    -----
    The counts are read, never recomputed: upstream owns the definition
    of what counts as degraded, and a second implementation here would
    be free to drift from it.
    """
    bands_missing = _as_band_tuple(attrs.get("bands_missing"))
    # Prefer the explicit count; fall back to the listed names so a file
    # carrying only one of the two attrs is still described correctly.
    n_missing = _as_int(attrs.get("n_bands_missing"), default=len(bands_missing))
    return {
        "n_bands_missing": n_missing,
        "bands_missing": bands_missing,
        "quality_degraded": _as_flag(attrs.get("quality_degraded")),
        "quality_note": str(attrs.get("quality_note", "")),
    }


def _result(
    sat_id: str,
    t0: datetime,
    path: Path,
    ds: xr.Dataset,
    *,
    reused: bool,
) -> AmvResult:
    """Assemble an :class:`AmvResult` from a dataset and its location."""
    return AmvResult(
        sat_id=sat_id,
        timestamp=t0,
        path=path,
        dataset=ds,
        reused=reused,
        **quality_from_attrs(ds.attrs),
    )


def _log_quality(result: AmvResult) -> AmvResult:
    """Report a retrieval's band shortfall, and hand it straight back.

    Applied to reused results as well as fresh ones: a day resumed from
    disk is exactly when an operator most needs to see that three
    satellites were degraded, and logging only on the compute path would
    make that run look spotless.
    """
    if result.quality_degraded:
        logger.warning(
            "[%s] degraded retrieval%s: %s",
            result.sat_id,
            " (reused)" if result.reused else "",
            result.quality_note or "no note recorded",
        )
    elif result.n_bands_missing:
        logger.info(
            "[%s] %d band(s) zero-filled: %s",
            result.sat_id,
            result.n_bands_missing,
            ", ".join(result.bands_missing),
        )
    return result


def _fsync_path(path: Path) -> None:
    """Flush ``path`` to disk, tolerating filesystems that cannot.

    Directory fsync is unsupported on some platforms and network
    filesystems; durability is best-effort there, and failing the write
    over it would be worse than the weaker guarantee.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        logger.debug("fsync unsupported for %s", path)
    finally:
        os.close(fd)


def _reap_stale_temps(path: Path, max_age_s: float = STALE_TMP_AGE_S) -> None:
    """Delete long-abandoned temp files for ``path``.

    The in-``except`` cleanup covers everything this process can catch,
    but a SIGKILL, an OOM kill or a node eviction between the write and
    the rename leaves the temp file behind — and since every attempt now
    picks a fresh random name, a satellite that keeps dying mid-write
    would otherwise pile up a full disk (hundreds of MB) per attempt
    until the volume fills.

    Only files older than ``max_age_s`` go: a younger one may belong to
    a concurrent attempt that is still writing it.
    """
    cutoff = time.time() - max_age_s
    try:
        stale = list(path.parent.glob(f"{path.name}.*{TMP_SUFFIX}"))
    except OSError:
        return
    for candidate in stale:
        try:
            if candidate.stat().st_mtime < cutoff:
                candidate.unlink()
                logger.warning("Removed abandoned temp file %s", candidate)
        except OSError:
            # Raced with another attempt, or not ours to remove.  Never
            # fail a write over housekeeping.
            continue


def _write_atomic(ds: xr.Dataset, path: Path) -> None:
    """Write ``ds`` to ``path`` so no partial file can ever be observed.

    A full disk takes minutes to write.  Writing in place means an
    interrupted run leaves a truncated NetCDF at the canonical path,
    which a resumed run with ``skip_existing`` would happily mistake for
    finished output.  Writing beside it and renaming makes the canonical
    path appear only once the bytes are all there — ``os.replace`` is
    atomic within a filesystem, and the temp file sits in the same
    directory precisely so it is.

    The temp name carries a random token because an orchestrator may
    have two attempts at the same satellite and timestamp in flight at
    once (a retry overlapping a run that has not yet died).  With a
    shared temp name they would write over each other's bytes, and the
    second ``os.replace`` would fail on a path the first had already
    renamed away.  Distinct temp files make the two attempts independent
    and the last rename to land simply wins.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _reap_stale_temps(path)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}{TMP_SUFFIX}")
    try:
        ds.to_netcdf(tmp)
        # os.replace orders the directory entry, not the data behind it.
        # Without the flush a hard crash can leave the canonical path
        # pointing at unpersisted bytes — and skip_existing would then
        # trust that file forever, since it only checks existence.
        _fsync_path(tmp)
        os.replace(tmp, path)
        # Persist the rename itself, so the entry survives the same crash.
        _fsync_path(path.parent)
    except BaseException:
        # Includes KeyboardInterrupt: a cancelled run must not leave
        # residue either.
        tmp.unlink(missing_ok=True)
        raise


def run_satellite_amv(
    sat_id: str,
    t0: datetime,
    model: Any,
    disp: Any,
    flow_bands: Sequence[str],
    rad_bands: Sequence[str],
    output_dir: str | os.PathLike[str],
    *,
    device: str = "cpu",
    row_strip: int = 1024,
    skip_existing: bool = True,
) -> AmvResult:
    """Retrieve winds for one satellite at one timestamp."""
    out_dir = Path(output_dir)
    path = Path(ring_adapter.sat_nc_path(out_dir, sat_id, t0))

    if skip_existing and path.exists():
        # A file that cannot be opened is not usable output — the atomic
        # write means it should never be truncated, but a corrupted or
        # externally-replaced file should cost a recompute, not the run.
        # ``load_dataset`` reads eagerly and closes the handle, so the
        # returned dataset does not pin the file open for a later step.
        try:
            ds = xr.load_dataset(path)
        except Exception:
            logger.exception("Failed to read %s — recomputing", path)
        else:
            logger.info("[%s] reusing existing %s", sat_id, path)
            return _log_quality(_result(sat_id, t0, path, ds, reused=True))

    logger.info("[%s] running retrieval for %s", sat_id, t0)
    ds = ring_adapter.infer_satellite(
        sat_id,
        t0,
        model,
        disp,
        list(flow_bands),
        list(rad_bands),
        device=device,
        row_strip=row_strip,
    )

    _write_atomic(ds, path)
    logger.info("[%s] wrote %s", sat_id, path)

    return _log_quality(_result(sat_id, t0, path, ds, reused=False))
