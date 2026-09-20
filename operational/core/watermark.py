"""Durable per-satellite watermark state for the availability sensor.

The availability sensor polls every few minutes and must only emit timestamps
it has not emitted before.  :class:`WatermarkStore` remembers, per satellite,
the most recent timestamp that was emitted.  The state is held in a single
JSON file that is rewritten atomically, so an operational service that is
killed mid-write restarts against either the old or the new state -- never a
truncated one.  A file that is nonetheless unreadable or malformed is treated
as empty rather than being allowed to crash the service.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["WatermarkStore"]


def _to_naive_utc(t: datetime) -> datetime:
    """Return ``t`` as a naive UTC datetime."""
    if t.tzinfo is not None:
        return t.astimezone(timezone.utc).replace(tzinfo=None)
    return t


class WatermarkStore:
    """JSON-backed, monotonic, per-satellite timestamp watermarks.

    Notes
    -----
    In-process mutation is guarded by a :class:`threading.Lock`.  The store is
    not a cross-process lock.  Each instance snapshots the file once at
    construction and rewrites the whole file on every mutation, so two
    processes sharing a path will clobber each other: a writer started before
    another process first wrote satellite ``X`` drops ``X`` from the file
    entirely, and that satellite then re-emits from scratch.  Use one writer
    process per state file.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._state: dict[str, datetime] = self._load()

    @property
    def path(self) -> Path:
        """Path of the backing JSON file."""
        return self._path

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _load(self) -> dict[str, datetime]:
        """Read the state file, tolerating absence and corruption."""
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            logger.warning(
                "Could not read watermark file %s (%s); treating as empty",
                self._path,
                exc,
            )
            return {}

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Watermark file %s is not valid JSON (%s); treating as empty",
                self._path,
                exc,
            )
            return {}

        if not isinstance(payload, dict):
            logger.warning(
                "Watermark file %s holds %s, expected an object; treating as empty",
                self._path,
                type(payload).__name__,
            )
            return {}

        state: dict[str, datetime] = {}
        for sat_id, value in payload.items():
            if not isinstance(sat_id, str):  # pragma: no cover - JSON keys are str
                continue
            if not isinstance(value, str):
                logger.warning(
                    "Watermark for %s in %s is %s, expected an ISO-8601 string; "
                    "treating as unset",
                    sat_id,
                    self._path,
                    type(value).__name__,
                )
                continue
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                logger.warning(
                    "Watermark for %s in %s is not a parseable timestamp (%r); "
                    "treating as unset",
                    sat_id,
                    self._path,
                    value,
                )
                continue
            state[sat_id] = _to_naive_utc(parsed)
        return state

    def _save(self) -> None:
        """Atomically rewrite the state file from the in-memory mapping.

        The payload is written to a temporary file in the destination
        directory and moved into place with :func:`os.replace`, so readers
        never observe a partial file.  The caller must hold ``self._lock``.
        """
        payload = {sat_id: t.isoformat() for sat_id, t in sorted(self._state.items())}
        directory = self._path.parent
        directory.mkdir(parents=True, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f"{self._path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self._path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:  # pragma: no cover - best-effort cleanup
                pass
            raise

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get(self, sat_id: str) -> datetime | None:
        """Return the last emitted timestamp for ``sat_id``."""
        with self._lock:
            return self._state.get(sat_id)

    def set(self, sat_id: str, t: datetime) -> None:
        """Move the watermark for ``sat_id`` forward to ``t``.

        A value at or before the current watermark is ignored, so the
        watermark never moves backwards.
        """
        self.advance(sat_id, t)

    def advance(self, sat_id: str, t: datetime) -> bool:
        """Move the watermark forward and report whether it actually moved."""
        if not isinstance(sat_id, str):
            raise TypeError(f"sat_id must be a str, got {type(sat_id).__name__}")
        if not isinstance(t, datetime):
            raise TypeError(f"t must be a datetime, got {type(t).__name__}")

        candidate = _to_naive_utc(t)
        with self._lock:
            current = self._state.get(sat_id)
            if current is not None and candidate <= current:
                logger.debug(
                    "Ignoring non-monotonic watermark for %s: %s <= %s",
                    sat_id,
                    candidate.isoformat(),
                    current.isoformat(),
                )
                return False
            self._state[sat_id] = candidate
            try:
                self._save()
            except BaseException:
                # Keep memory and disk in step: a caller that never saw the
                # watermark move must be able to retry the same timestamp.
                if current is None:
                    self._state.pop(sat_id, None)
                else:
                    self._state[sat_id] = current
                raise
            return True

    def all(self) -> dict[str, datetime]:
        """Return a copy of every known watermark."""
        with self._lock:
            return dict(self._state)

    def reset(self, sat_id: str | None = None) -> None:
        """Forget one satellite's watermark, or every watermark.

        Notes
        -----
        The state file is rewritten even when the satellite had no watermark
        in memory, so that a reset also clears an entry that was dropped from
        the in-memory state because it failed to parse.
        """
        with self._lock:
            if sat_id is None:
                self._state.clear()
            else:
                if self._state.pop(sat_id, None) is None:
                    logger.debug("No in-memory watermark to reset for %s", sat_id)
            # Always rewrite: this also repairs a corrupt file on disk.
            self._save()
