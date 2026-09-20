"""Publish a global AMV mosaic to an icechunk store, one commit per timestep.

The final step of the operational pipeline.  It is a thin, re-runnable
layer over :mod:`stereo_winds.icechunk_output`: the storage construction,
the append, and the source-code remapping all live upstream and are
reused here rather than reimplemented.

Three properties matter for an always-on service:

**Idempotence.** A step may be retried after a partial run, a crash, or a
backfill that overlaps what is already there.  With ``skip_existing`` a
timestamp already committed is skipped rather than appended a second
time, so the store never grows duplicate timesteps.

**A monotonic time axis.** Mosaics are *appended*, never inserted, so a
timestamp older than the store's last one would leave ``time``
unsorted — and an unsorted index makes ``ds.sel(time=slice(...))`` return
nothing at all rather than raise.  Out-of-order publication is refused
unless the caller opts in.

**A shared satellite vocabulary.** ``source_satellite_index`` in a mosaic
is an ``int8`` code into *that mosaic's own* satellite list, and that list
varies with which satellites contributed at that timestamp — code 2 can
mean gk2a in one timestep and himawari9 in the next.  Stacked into one
array under a single ``flag_meanings`` they would silently misattribute
provenance, so one vocabulary is carried across calls and extended in
place, seeded from what the store already recorded.  See
:func:`stereo_winds.icechunk_output.align_source_codes`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import xarray as xr

from stereo_winds.icechunk_output import (
    icechunk_existing_times,
    open_icechunk_repo,
    store_vocabulary,
    time_tag,
    write_mosaic_to_icechunk,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PublishResult",
    "as_store_time",
    "existing_timestamps",
    "open_store",
    "publish_mosaic",
    "seed_vocabulary",
]


@dataclass(frozen=True)
class PublishResult:
    """Outcome of a single :func:`publish_mosaic` call."""

    timestamp: datetime
    written: bool
    skipped_reason: str | None = None
    vocabulary: tuple[str, ...] = field(default=())
    branch: str = "main"
    action: str = "appended"


def stored_satellites(repo, t0: datetime, branch: str = "main") -> set[str] | None:
    """Satellites behind the stored mosaic at ``t0``.

    Returns None when the store cannot say -- it has no dataset, no such
    timestamp, or it predates the per-timestep ``satellites_contributing``
    variable.  None means "do not compare", never "nothing contributed".
    """
    try:
        ds = xr.open_zarr(repo.readonly_session(branch).store, consolidated=False)
    except Exception:
        return None
    if "time" not in ds.coords or "satellites_contributing" not in ds:
        return None
    import numpy as np

    times = np.asarray(ds["time"].values, "datetime64[ns]")
    hits = np.flatnonzero(times == np.datetime64(as_store_time(t0), "ns"))
    if not hits.size:
        return None
    value = str(ds["satellites_contributing"].values[int(hits[0])])
    return {part for part in value.split(",") if part}


def mosaic_satellite_set(ds_global: xr.Dataset) -> set[str]:
    """Satellites contributing to a mosaic about to be published."""
    value = ds_global.attrs.get("satellites_contributing")
    if value is None:
        value = ds_global.attrs.get("satellites")
    if value is None:
        return set()
    if isinstance(value, (list, tuple)):
        return {str(v) for v in value if str(v)}
    return {part for part in str(value).split(",") if part}


def as_store_time(t0: datetime) -> datetime:
    """Normalise a timestamp to the naive UTC the store records.

    An orchestrator hands out timezone-aware UTC timestamps, while
    ``numpy.datetime64`` — and therefore everything read back out of the
    store — is naive.  Left unconverted, an aware ``t0`` would never
    compare equal to what the store holds, so every re-run would append a
    duplicate instead of skipping, and a non-UTC ``t0`` would be stored
    at its wall-clock value, off by its own offset.
    """
    if t0.tzinfo is None:
        return t0
    return t0.astimezone(timezone.utc).replace(tzinfo=None)


def open_store(
    store_uri: str,
    *,
    endpoint_url: str | None = None,
    region: str | None = None,
    anonymous: bool = False,
    force_path_style: bool = False,
):
    """Open (or create) the icechunk repository the mosaics are published to.

    Notes
    -----
    Storage construction is delegated to
    :func:`stereo_winds.icechunk_output.open_icechunk_repo` so the
    operational path and the research scripts agree on what a store URI
    means.
    """
    repo = open_icechunk_repo(
        store_uri,
        endpoint_url=endpoint_url,
        region=region,
        anonymous=anonymous,
        force_path_style=force_path_style,
    )
    logger.info("Publishing to icechunk store %s", store_uri)
    return repo


def existing_timestamps(repo, branch: str = "main") -> set[datetime]:
    """Timestamps already committed to ``repo`` (empty for a new store)."""
    return icechunk_existing_times(repo, branch)


def seed_vocabulary(
    repo,
    vocabulary: list[str] | None,
    branch: str = "main",
) -> list[str]:
    """Ensure the working vocabulary starts with the store's own.

    Codes already written to the store index the store's
    ``flag_meanings``, so those names must stay at the positions they
    already occupy; new names are only ever appended.  A resumed run that
    started from an empty list would otherwise renumber the satellites
    and silently misattribute every timestep it appended.
    """
    stored = store_vocabulary(repo, branch)
    if vocabulary is None:
        return stored
    if not stored or vocabulary[: len(stored)] == stored:
        return vocabulary
    extras = [name for name in vocabulary if name not in stored]
    logger.info("Re-seeding satellite vocabulary from the store: %s (+%s)", stored, extras)
    vocabulary[:] = stored + extras
    return vocabulary


def _check_stored_vocabulary(repo, branch: str, vocabulary: list[str]) -> None:
    """Fail loudly if the store did not record the vocabulary just used.

    Upstream's attribute rewrite logs and swallows its exceptions, so a
    failure there would leave the store describing fewer satellites than
    the codes it now holds — and the next run, seeding from the store,
    would bind those codes to different satellites.  Checking here turns
    that silent misattribution into a visible error while the commit that
    caused it is still the last one.
    """
    recorded = store_vocabulary(repo, branch)
    if recorded != vocabulary:
        raise RuntimeError(
            f"icechunk store recorded satellite vocabulary {recorded} but "
            f"the mosaic was written against {vocabulary}; "
            f"source_satellite_index would misattribute provenance"
        )


def publish_mosaic(
    repo,
    ds_global: xr.Dataset,
    t0: datetime,
    *,
    branch: str = "main",
    chunk: int = 1024,
    vocabulary: list[str] | None = None,
    skip_existing: bool = True,
    allow_out_of_order: bool = False,
    repair_improved: bool = True,
    replace_existing: bool = False,
) -> PublishResult:
    """Append one global mosaic to the icechunk store as its own commit."""
    t0 = as_store_time(t0)
    vocabulary = seed_vocabulary(repo, vocabulary, branch)
    stored_times = icechunk_existing_times(repo, branch)

    replace = False
    if t0 in stored_times:
        already = stored_satellites(repo, t0, branch)
        incoming = mosaic_satellite_set(ds_global)
        gained = incoming - already if already is not None else set()

        if replace_existing:
            replace = True
            logger.info("Replacing %s on request", time_tag(t0))
        elif repair_improved and gained:
            replace = True
            logger.info(
                "Repairing %s: this mosaic adds %s to the %s already stored",
                time_tag(t0),
                ", ".join(sorted(gained)),
                ", ".join(sorted(already)) or "(none)",
            )
        elif skip_existing:
            reason = f"{time_tag(t0)} is already in the store"
            if already is None and repair_improved:
                reason += " (it does not record which satellites it used)"
            logger.info("Skipping publish: %s", reason)
            return PublishResult(
                timestamp=t0,
                written=False,
                skipped_reason=reason,
                vocabulary=tuple(vocabulary),
                branch=branch,
                action="skipped",
            )
        else:
            replace = True
            logger.info(
                "Publishing %s again with skip_existing=False; replacing the " "stored timestep",
                time_tag(t0),
            )
    elif stored_times and t0 < max(stored_times) and not allow_out_of_order:  # noqa: E501
        raise ValueError(
            f"Refusing to publish {time_tag(t0)}: the store already holds "
            f"{time_tag(max(stored_times))} and mosaics are appended, not "
            f"inserted, so this would leave the time coordinate unsorted. "
            f"Publish in chronological order, or pass allow_out_of_order=True."
        )

    action = write_mosaic_to_icechunk(
        repo,
        ds_global,
        t0,
        branch=branch,
        chunk=chunk,
        vocabulary=vocabulary,
        replace=replace,
    )
    if vocabulary and "source_satellite_index" in ds_global:
        _check_stored_vocabulary(repo, branch, vocabulary)
    logger.info("Published %s (vocabulary: %s)", time_tag(t0), vocabulary)
    return PublishResult(
        timestamp=t0,
        written=True,
        skipped_reason=None,
        vocabulary=tuple(vocabulary),
        branch=branch,
        action=action or "appended",
    )
