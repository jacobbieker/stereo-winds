r"""Create an icechunk repo and write global AMV mosaics into it.

Ingests the mosaics ``infer_student_global_ring.py`` wrote as NetCDF —

    <dir>/<YYYYMMDD>/student_amv_global_<YYYYMMDDTHHMM>.nc

— appending each one along ``time`` as its own commit, so the store holds
one timestep per mosaic and a partially ingested store is still valid up
to its last commit.  Re-running skips what is already there, so an
interrupted ingest resumes where it stopped.

The retrieval pipeline can write straight to a store with its own
``--icechunk-store``; this script is for mosaics already on disk, or for
moving a finished run into a store.

Usage::

    # local store
    pixi run python scripts/write_mosaics_to_icechunk.py \
        --mosaic-dir output/global_ring_hourly_10km \
        --store output/global_ring.icechunk

    # S3, resuming an earlier ingest
    pixi run python scripts/write_mosaics_to_icechunk.py \
        --mosaic-dir output/global_ring_hourly_10km \
        --store s3://my-bucket/student-amv.icechunk \
        --skip-existing

    # see what would be written, without writing
    pixi run python scripts/write_mosaics_to_icechunk.py \
        --mosaic-dir output/global_ring_hourly_10km \
        --store output/global_ring.icechunk --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import xarray as xr

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from stereo_winds.icechunk_output import (  # noqa: E402
    icechunk_existing_times,
    icechunk_storage,
    mosaic_satellites,
    open_icechunk_repo,
    store_vocabulary,
    time_tag,
    write_mosaic_to_icechunk,
)

logger = logging.getLogger(__name__)

MOSAIC_GLOB = "student_amv_global_*.nc"
_STAMP_RE = re.compile(r"student_amv_global_(\d{8}T\d{4})\.nc$")


def mosaic_time(path: Path) -> datetime | None:
    """Timestamp a mosaic file is for, from its name, or None."""
    match = _STAMP_RE.search(path.name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M")
    except ValueError:
        return None


def find_mosaics(roots: list[Path]) -> list[tuple[datetime, Path]]:
    """Every mosaic under ``roots``, in time order.

    Accepts directories (searched recursively, matching the pipeline's
    per-day layout) or individual files.  The timestamp comes from the
    filename, which is the canonical one the pipeline wrote.
    """
    found: dict[datetime, Path] = {}
    for root in roots:
        if root.is_file():
            candidates = [root]
        else:
            candidates = sorted(root.rglob(MOSAIC_GLOB))
        for path in candidates:
            when = mosaic_time(path)
            if when is None:
                logger.warning("Skipping %s: no timestamp in the name", path)
                continue
            previous = found.get(when)
            if previous is not None and previous != path:
                logger.warning("Two mosaics for %s; using %s, ignoring %s",
                               time_tag(when), previous, path)
                continue
            found[when] = path
    return sorted(found.items())


def collect_vocabulary(
    mosaics: list[tuple[datetime, Path]], start: list[str] | None = None,
) -> list[str]:
    """Union of the satellites the mosaics name, in first-seen order.

    Each mosaic's source codes index its own satellite list, and that
    list changes with which satellites contributed.  Building the union
    up front means the store has one stable vocabulary from its first
    commit, rather than one that shifts as it is filled.  Reading it
    costs only the file attributes.
    """
    vocabulary = list(start or [])
    for _, path in mosaics:
        try:
            with xr.open_dataset(path) as ds:
                names = mosaic_satellites(ds)
        except Exception:
            logger.exception("Could not read the satellite list from %s", path)
            continue
        for name in names:
            if name not in vocabulary:
                vocabulary.append(name)
    return vocabulary


def ingest(
    mosaics: list[tuple[datetime, Path]],
    repo,
    branch: str = "main",
    chunk: int = 1024,
    skip: set[datetime] | None = None,
    vocabulary: list[str] | None = None,
) -> tuple[int, list[datetime]]:
    """Write each mosaic to the store as its own commit.

    Returns (written, failed timestamps).  A mosaic that cannot be read
    is reported and skipped rather than abandoning the rest.
    """
    skip = skip or set()
    written = 0
    failed: list[datetime] = []
    for i, (when, path) in enumerate(mosaics, 1):
        if when in skip:
            logger.info("[%d/%d] %s already in the store — skipping",
                        i, len(mosaics), time_tag(when))
            continue
        logger.info("[%d/%d] %s <- %s", i, len(mosaics), time_tag(when), path)
        try:
            with xr.open_dataset(path) as ds:
                # Mosaics are ~200 MB; load before writing so the source
                # file is closed and the write is not lazily backed by it.
                mosaic = ds.load()
            write_mosaic_to_icechunk(repo, mosaic, when, branch=branch,
                                     chunk=chunk, vocabulary=vocabulary)
            skip.add(when)
            written += 1
        except Exception:
            logger.exception("Failed to write %s — continuing", path)
            failed.append(when)
        finally:
            mosaic = None
    return written, failed


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description="Create an icechunk repo and write global AMV mosaics "
                    "into it, one timestep per mosaic",
    )
    ap.add_argument("--mosaic-dir", nargs="+", required=True, type=Path,
                    help="Directory (searched recursively) or mosaic files "
                         "to ingest")
    ap.add_argument("--store", required=True,
                    help="Icechunk store to create or extend: "
                         "s3://bucket/prefix, or a local directory path")
    ap.add_argument("--branch", default="main",
                    help="Branch to commit to (default: main)")
    ap.add_argument("--chunk", type=int, default=1024,
                    help="Spatial chunk size for the store's arrays "
                         "(default 1024)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip timestamps the store already holds (resume "
                         "an interrupted ingest)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be written, and write nothing")
    ap.add_argument("--start", default=None,
                    help="Ignore mosaics before this ISO timestamp")
    ap.add_argument("--end", default=None,
                    help="Ignore mosaics after this ISO timestamp")

    s3 = ap.add_argument_group("S3 options")
    s3.add_argument("--endpoint", default=None,
                    help="S3 endpoint URL for non-AWS stores")
    s3.add_argument("--region", default=None, help="S3 region")
    s3.add_argument("--anonymous", action="store_true",
                    help="Access the store anonymously (reads only)")
    s3.add_argument("--force-path-style", action="store_true",
                    help="Use path-style S3 addressing (minio, source.coop)")
    args = ap.parse_args()

    mosaics = find_mosaics(args.mosaic_dir)
    if args.start:
        start = datetime.fromisoformat(args.start)
        mosaics = [(t, p) for t, p in mosaics if t >= start]
    if args.end:
        end = datetime.fromisoformat(args.end)
        mosaics = [(t, p) for t, p in mosaics if t <= end]

    if not mosaics:
        logger.error("No mosaics found under %s",
                     ", ".join(str(p) for p in args.mosaic_dir))
        sys.exit(1)

    total_gb = sum(p.stat().st_size for _, p in mosaics) / 2**30
    logger.info("Found %d mosaic(s), %s .. %s (%.1f GB on disk)",
                len(mosaics), time_tag(mosaics[0][0]), time_tag(mosaics[-1][0]),
                total_gb)

    if args.dry_run:
        for when, path in mosaics:
            logger.info("  would write %s <- %s", time_tag(when), path)
        logger.info("Dry run: nothing written to %s", args.store)
        return

    repo = open_icechunk_repo(
        args.store,
        endpoint_url=args.endpoint,
        region=args.region,
        anonymous=args.anonymous,
        force_path_style=args.force_path_style,
    )

    existing = icechunk_existing_times(repo, args.branch)
    if existing and not args.skip_existing:
        logger.warning(
            "The store already holds %d timestamp(s); those will not be "
            "written again. Pass --skip-existing to silence this.",
            len(existing))

    # One vocabulary for the whole store, so a source code means the same
    # satellite at every timestep.
    vocabulary = collect_vocabulary(mosaics, store_vocabulary(repo, args.branch))
    logger.info("Satellite vocabulary: %s", ", ".join(vocabulary) or "(none)")

    written, failed = ingest(mosaics, repo, branch=args.branch,
                             chunk=args.chunk, skip=set(existing),
                             vocabulary=vocabulary)

    logger.info("Wrote %d of %d mosaic(s) to %s",
                written, len(mosaics), args.store)
    if failed:
        logger.warning("Failed (%d): %s", len(failed),
                       ", ".join(time_tag(t) for t in failed))
    if written == 0 and not existing:
        logger.error("Nothing was written.")
        sys.exit(1)


if __name__ == "__main__":
    main()
