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

Examples
--------
>>> repo = open_store("output/operational.icechunk")  # doctest: +SKIP
>>> vocabulary = []  # doctest: +SKIP
>>> for t0, mosaic in sorted(mosaics):  # doctest: +SKIP
...     publish_mosaic(repo, mosaic, t0, vocabulary=vocabulary)
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
    """Outcome of a single :func:`publish_mosaic` call.

    Attributes
    ----------
    timestamp
        The mosaic timestamp this call was about, normalised to naive UTC
        the way the store records it.
    written
        True if a commit was made, False if the timestamp was skipped.
    skipped_reason
        Why nothing was written, or None when ``written`` is True.
    vocabulary
        Snapshot of the satellite vocabulary after the write.  The
        caller's own list is the one that keeps being extended in place;
        this is a copy for logging and assertions.
    branch
        Branch the commit went to.
    """

    timestamp: datetime
    written: bool
    skipped_reason: str | None = None
    vocabulary: tuple[str, ...] = field(default=())
    branch: str = "main"


def as_store_time(t0: datetime) -> datetime:
    """Normalise a timestamp to the naive UTC the store records.

    An orchestrator hands out timezone-aware UTC timestamps, while
    ``numpy.datetime64`` — and therefore everything read back out of the
    store — is naive.  Left unconverted, an aware ``t0`` would never
    compare equal to what the store holds, so every re-run would append a
    duplicate instead of skipping, and a non-UTC ``t0`` would be stored
    at its wall-clock value, off by its own offset.

    Parameters
    ----------
    t0
        Aware or naive timestamp.  A naive one is taken to be UTC already.

    Returns
    -------
    datetime
        The same instant, naive and in UTC.
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

    Parameters
    ----------
    store_uri
        ``s3://bucket/prefix`` or a local directory path.  A local
        directory is created if it does not exist.
    endpoint_url
        Alternative S3 endpoint (MinIO, Ceph, a local test server).
    region
        S3 region.
    anonymous
        Access the bucket without credentials, for a public store.
    force_path_style
        Use path-style S3 addressing, which some S3-compatible servers
        require.

    Returns
    -------
    icechunk.Repository
        The open repository.

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
    """Timestamps already committed to ``repo`` (empty for a new store).

    Parameters
    ----------
    repo
        An open icechunk repository.
    branch
        Branch to inspect.

    Returns
    -------
    set of datetime
        What the store already holds, as naive UTC, for a caller that
        wants to filter a batch of work before doing it.  Compare against
        :func:`as_store_time` of your own timestamps.
    """
    return icechunk_existing_times(repo, branch)


def seed_vocabulary(
    repo, vocabulary: list[str] | None, branch: str = "main",
) -> list[str]:
    """Ensure the working vocabulary starts with the store's own.

    Codes already written to the store index the store's
    ``flag_meanings``, so those names must stay at the positions they
    already occupy; new names are only ever appended.  A resumed run that
    started from an empty list would otherwise renumber the satellites
    and silently misattribute every timestep it appended.

    Parameters
    ----------
    repo
        An open icechunk repository.
    vocabulary
        The caller's working list, extended **in place** so the caller
        keeps the same object across calls.  None means "start from the
        store".
    branch
        Branch to read the stored vocabulary from.

    Returns
    -------
    list of str
        The working vocabulary (the same object as ``vocabulary`` when
        one was given).
    """
    stored = store_vocabulary(repo, branch)
    if vocabulary is None:
        return stored
    if not stored or vocabulary[: len(stored)] == stored:
        return vocabulary
    extras = [name for name in vocabulary if name not in stored]
    logger.info("Re-seeding satellite vocabulary from the store: %s (+%s)",
                stored, extras)
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
) -> PublishResult:
    """Append one global mosaic to the icechunk store as its own commit.

    Parameters
    ----------
    repo
        An open icechunk repository, from :func:`open_store`.
    ds_global
        The mosaic: dims ``(latitude, longitude)``, the seven AMV
        variables plus ``source_satellite_index``.
    t0
        Timestamp of the mosaic; becomes its position along ``time``.
        Normalised to naive UTC by :func:`as_store_time`.
    branch
        Branch to commit to.
    chunk
        Spatial chunk size used when the dataset is first created.  It is
        ignored on later appends, which inherit the stored chunking.
    vocabulary
        Shared satellite vocabulary, **mutated in place** as new
        satellites appear.  Pass the same list on every call of a run;
        None seeds a fresh one from the store.
    skip_existing
        Skip a timestamp the store already holds instead of writing it
        again.  This is what makes a re-run safe.  Setting it False does
        **not** replace the stored mosaic — icechunk is appended to, so
        the timestamp ends up in ``time`` twice and ``sel(time=t0)``
        returns both.  It is an escape hatch, not an overwrite.
    allow_out_of_order
        Permit a ``t0`` older than the store's latest timestamp.  The
        append leaves ``time`` unsorted, and an unsorted index makes
        range selections return nothing instead of raising, so this is
        refused by default.  Backfills should be published in
        chronological order.

    Returns
    -------
    PublishResult
        Whether a commit happened, and the vocabulary after the write.

    Raises
    ------
    ValueError
        If ``t0`` predates the store's latest timestamp and
        ``allow_out_of_order`` is False.
    RuntimeError
        If the store did not record the satellite vocabulary the mosaic
        was written against.
    """
    t0 = as_store_time(t0)
    vocabulary = seed_vocabulary(repo, vocabulary, branch)
    stored_times = icechunk_existing_times(repo, branch)

    if t0 in stored_times:
        if skip_existing:
            reason = f"{time_tag(t0)} is already in the store"
            logger.info("Skipping publish: %s", reason)
            return PublishResult(
                timestamp=t0, written=False, skipped_reason=reason,
                vocabulary=tuple(vocabulary), branch=branch,
            )
        logger.warning(
            "Publishing %s again with skip_existing=False — it will appear "
            "twice along time; icechunk appends, it does not replace",
            time_tag(t0))
    elif stored_times and t0 < max(stored_times) and not allow_out_of_order:
        raise ValueError(
            f"Refusing to publish {time_tag(t0)}: the store already holds "
            f"{time_tag(max(stored_times))} and mosaics are appended, not "
            f"inserted, so this would leave the time coordinate unsorted. "
            f"Publish in chronological order, or pass allow_out_of_order=True."
        )

    write_mosaic_to_icechunk(
        repo, ds_global, t0, branch=branch, chunk=chunk, vocabulary=vocabulary,
    )
    if vocabulary and "source_satellite_index" in ds_global:
        _check_stored_vocabulary(repo, branch, vocabulary)
    logger.info("Published %s (vocabulary: %s)", time_tag(t0), vocabulary)
    return PublishResult(
        timestamp=t0, written=True, skipped_reason=None,
        vocabulary=tuple(vocabulary), branch=branch,
    )
