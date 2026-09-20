"""Configuration for the operational AMV pipeline.

A single frozen :class:`OperationalConfig` describes everything the
Dagster assets need: which satellites to run, which bands to feed the
student model, how often to look for new imagery, and where the results
land.  Instances are immutable so the same object can be shared between
Dagster assets without one step perturbing another.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Mapping

# Mirrors ``stereo_winds.student_dataset.DEFAULT_FLOW_BANDS`` and
# ``DEFAULT_RAD_BANDS``, restated rather than imported: that module
# does ``import torch`` at module scope, and pulling torch in to read
# two lists of strings would put it in the import path of every
# dagster definition -- sensors, schedules, the whole code location.
# ``test_resources.py`` fails if these drift from upstream.
DEFAULT_FLOW_BANDS: tuple[str, ...] = ("C08", "C09", "C10", "C12", "C14")
DEFAULT_RAD_BANDS: tuple[str, ...] = (
    "C07", "C08", "C09", "C10", "C11",
    "C12", "C13", "C14", "C15", "C16",
)

#: Satellites served only from their icechunk stores, with no public-S3
#: L1b behind them.  Their coverage is sparser and lags the rest, so a
#: run does not wait on them by default -- they still get an asset, and
#: the mosaic records them as missing when they are absent.
SPARSE_COVERAGE: tuple[str, ...] = ("mtg-i1", "msg-iodc")

logger = logging.getLogger(__name__)

#: Prefix for every environment variable read by
#: :meth:`OperationalConfig.from_env`.
ENV_PREFIX = "STEREO_WINDS_OP_"

#: Scheme marking a ``store_uri`` as living in object storage.
S3_SCHEME = "s3://"

#: Leading ``scheme://`` of a URI, if it has one.
_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)://")

#: Anchor for cadence alignment.  A fixed epoch keeps the slot grid
#: continuous across midnight even for cadences that do not divide a day.
_EPOCH = datetime(1970, 1, 1)


def uri_scheme(uri: str) -> str | None:
    """Lower-cased ``scheme`` of ``uri``, or ``None`` for a plain path.

    Parameters
    ----------
    uri
        Store location: a filesystem path or a ``scheme://`` URI.

    Returns
    -------
    str or None
        The scheme without the ``://`` separator, lower-cased; ``None``
        when ``uri`` carries no scheme.
    """
    match = _SCHEME_RE.match(str(uri))
    return match.group(1).lower() if match else None


def _known_satellites() -> frozenset[str]:
    """Satellite ids the installed :mod:`stereo_winds` can navigate."""
    from stereo_winds.config import SATELLITE_CONFIGS

    return frozenset(SATELLITE_CONFIGS)


@dataclass(frozen=True)
class OperationalConfig:
    """Settings for one operational run of the AMV pipeline.

    Parameters
    ----------
    satellites
        Satellite ids to retrieve, in the order they are mosaicked.  Ids
        are keys of :data:`stereo_winds.config.SATELLITE_CONFIGS`.
    flow_bands
        Bands the student model runs optical flow on.
    rad_bands
        Bands supplying brightness temperatures to the student model.
    cadence_minutes
        Spacing of the operational timestamps, in minutes.
    availability_tolerance_minutes
        How far a scene's own timestamp may sit from the nominal slot and
        still count as available for that slot.
    output_dir
        Directory for intermediate per-satellite and mosaic files.
    store_uri
        Destination icechunk store: a local path, or an ``s3://`` URI of
        the form ``s3://bucket/prefix``.
    resolution_m
        Grid spacing of the global mosaic, in meters.
    device
        Torch device string for inference (``"cpu"``, ``"cuda"``, ...).
    row_strip
        Number of full-disk rows per forward-pass strip.

    Notes
    -----
    ``satellites`` is the full ring -- every one of them gets its own
    asset.  ``required_satellites`` is the subset a timestamp must have
    before a run is worth starting.

    They differ because MTG-I1 and MSG-IODC are served only from their
    icechunk stores, with no public-S3 fallback behind them, so their
    coverage is sparser and later than the rest.  Requiring them would
    hold up every run for the satellites that *are* there, while the
    mosaic step already tolerates a missing satellite and records it.
    Put a satellite in ``required_satellites`` once its coverage is
    dependable enough to wait for.
    """

    satellites: tuple[str, ...] = (
        "goes18", "goes19", "himawari9", "gk2a", "mtg-i1", "msg-iodc",
    )
    required_satellites: tuple[str, ...] | None = None
    flow_bands: tuple[str, ...] = tuple(DEFAULT_FLOW_BANDS)
    rad_bands: tuple[str, ...] = tuple(DEFAULT_RAD_BANDS)
    cadence_minutes: int = 60
    availability_tolerance_minutes: float = 5.0
    output_dir: Path = Path("output/operational")
    store_uri: str = "output/operational.icechunk"
    resolution_m: float = 10000.0
    device: str = "cpu"
    row_strip: int = 1024

    # -- normalisation & validation ---------------------------------------

    def __post_init__(self) -> None:
        """Coerce sequence/path fields and reject nonsensical values."""
        object.__setattr__(self, "satellites", tuple(self.satellites))
        if self.required_satellites is None:
            dependable = tuple(s for s in self.satellites
                               if s not in SPARSE_COVERAGE)
            object.__setattr__(self, "required_satellites",
                               dependable or tuple(self.satellites))
        else:
            object.__setattr__(
                self, "required_satellites", tuple(self.required_satellites))
        object.__setattr__(self, "flow_bands", tuple(self.flow_bands))
        object.__setattr__(self, "rad_bands", tuple(self.rad_bands))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "cadence_minutes", int(self.cadence_minutes))
        object.__setattr__(
            self,
            "availability_tolerance_minutes",
            float(self.availability_tolerance_minutes),
        )
        object.__setattr__(self, "resolution_m", float(self.resolution_m))
        object.__setattr__(self, "row_strip", int(self.row_strip))

        if not self.satellites:
            raise ValueError("satellites must not be empty")
        if len(set(self.satellites)) != len(self.satellites):
            raise ValueError(f"duplicate satellite ids: {self.satellites}")
        unknown = set(self.required_satellites) - set(self.satellites)
        if unknown:
            raise ValueError(
                f"required_satellites names satellites that are not in the "
                f"ring: {sorted(unknown)}"
            )
        if self.cadence_minutes <= 0:
            raise ValueError(
                f"cadence_minutes must be positive, got {self.cadence_minutes}"
            )
        if self.availability_tolerance_minutes < 0:
            raise ValueError(
                "availability_tolerance_minutes must be non-negative, got "
                f"{self.availability_tolerance_minutes}"
            )
        if self.resolution_m <= 0:
            raise ValueError(
                f"resolution_m must be positive, got {self.resolution_m}"
            )
        if self.row_strip <= 0:
            raise ValueError(f"row_strip must be positive, got {self.row_strip}")
        if not self.flow_bands:
            raise ValueError("flow_bands must not be empty")
        if not self.rad_bands:
            raise ValueError("rad_bands must not be empty")
        if not str(self.store_uri):
            raise ValueError("store_uri must not be empty")

        # A mistyped scheme must fail here rather than quietly create a
        # local directory literally named "gs:" at publish time.
        scheme = uri_scheme(self.store_uri)
        if scheme is not None and scheme != "s3":
            raise ValueError(
                f"store_uri {self.store_uri!r} uses unsupported scheme "
                f"{scheme!r}; give a local path or an s3:// URI"
            )

        unknown = [s for s in self.satellites if s not in _known_satellites()]
        if unknown:
            # A warning, not an error: tests and future readers may use ids
            # the installed stereo_winds does not know about yet.
            logger.warning(
                "satellite id(s) %s are not in stereo_winds SATELLITE_CONFIGS "
                "— retrieval will fail for them unless a reader is registered",
                ", ".join(repr(s) for s in unknown),
            )

    # -- derived views -----------------------------------------------------

    @property
    def cadence(self) -> timedelta:
        """Spacing between operational timestamps."""
        return timedelta(minutes=self.cadence_minutes)

    @property
    def availability_tolerance(self) -> timedelta:
        """Allowed offset between a scene time and its nominal slot."""
        return timedelta(minutes=self.availability_tolerance_minutes)

    @property
    def store_is_s3(self) -> bool:
        """Whether :attr:`store_uri` points at object storage.

        The scheme is matched case-insensitively, so ``S3://bucket`` is
        recognised as remote rather than taken for a local directory.
        """
        return uri_scheme(self.store_uri) == "s3"

    @property
    def store_path(self) -> Path | None:
        """Local filesystem path of the store, or ``None`` when on S3."""
        if self.store_is_s3:
            return None
        return Path(self.store_uri)

    def store_bucket_prefix(self) -> tuple[str, str]:
        """Split an object-storage store URI into bucket and prefix.

        Returns
        -------
        tuple of str
            ``(bucket, prefix)``; the prefix is ``""`` when the URI names
            only a bucket.

        Raises
        ------
        ValueError
            If :attr:`store_uri` is a local path rather than an object
            storage URI.
        """
        if not self.store_is_s3:
            raise ValueError(
                f"store_uri {self.store_uri!r} is a local path, not an "
                "object-storage URI"
            )
        rest = str(self.store_uri).split("://", 1)[1]
        bucket, _, prefix = rest.partition("/")
        if not bucket:
            raise ValueError(f"store_uri {self.store_uri!r} names no bucket")
        return bucket, prefix.strip("/")

    # -- timestamps --------------------------------------------------------

    def floor_to_cadence(self, when: datetime) -> datetime:
        """Round ``when`` down to the most recent cadence-aligned slot.

        Slots are anchored at the Unix epoch, so a 60 minute cadence
        yields the top of the hour and a 10 minute cadence the nearest
        ten-minute mark — and, unlike a per-day anchor, the grid stays
        continuous across midnight for a cadence that does not divide a
        day.  Timezone-aware inputs are floored on absolute time, so a
        DST transition cannot shift the grid either.

        Parameters
        ----------
        when
            Timestamp to floor; naive values are treated as UTC.

        Returns
        -------
        datetime
            The cadence-aligned slot at or before ``when``, carrying
            ``when``'s tzinfo.
        """
        step = self.cadence.total_seconds()
        if when.tzinfo is None:
            elapsed = (when - _EPOCH) // self.cadence
            return _EPOCH + elapsed * self.cadence
        floored = math.floor(when.timestamp() / step) * step
        return datetime.fromtimestamp(floored, tz=when.tzinfo)

    def timestamps(self, start: datetime, end: datetime) -> Iterator[datetime]:
        """Yield cadence-aligned timestamps in ``[start, end]``.

        The window is snapped **inward**: a ``start`` that falls between
        two slots advances to the next one, so every yielded slot lies
        within ``[start, end]`` and a partially elapsed leading slot is
        not reprocessed.

        Parameters
        ----------
        start, end
            Inclusive bounds of the window.

        Yields
        ------
        datetime
            Each operational slot in the window, ascending.
        """
        current = self.floor_to_cadence(start)
        if current < start:
            current += self.cadence
        while current <= end:
            yield current
            current += self.cadence

    # -- paths -------------------------------------------------------------

    @staticmethod
    def timestamp_key(when: datetime) -> str:
        """Filename-safe key for a timestamp (``YYYYmmddTHHMMSS``)."""
        return when.strftime("%Y%m%dT%H%M%S")

    def satellite_path(self, sat_id: str, when: datetime) -> Path:
        """Path of the per-satellite AMV file for ``sat_id`` at ``when``."""
        key = self.timestamp_key(when)
        return self.output_dir / key / f"amv_{sat_id}_{key}.nc"

    def mosaic_path(self, when: datetime) -> Path:
        """Path of the mosaic file for ``when``."""
        key = self.timestamp_key(when)
        return self.output_dir / key / f"mosaic_{key}.nc"

    def with_satellites(self, *satellites: str) -> "OperationalConfig":
        """Return a copy restricted to ``satellites`` (order preserved)."""
        kept = tuple(s for s in self.required_satellites if s in satellites)
        # Narrowing the ring must not leave required_satellites naming
        # something the ring no longer has; what survives the narrowing
        # is kept, and an empty result is re-derived.
        return replace(self, satellites=tuple(satellites),
                       required_satellites=kept or None)

    # -- environment -------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        **overrides: object,
    ) -> "OperationalConfig":
        """Build a config from ``STEREO_WINDS_OP_*`` environment variables.

        Recognised variables (all optional) are the upper-cased field
        names with the :data:`ENV_PREFIX` prefix, for example
        ``STEREO_WINDS_OP_SATELLITES="goes19,himawari9"`` or
        ``STEREO_WINDS_OP_CADENCE_MINUTES=10``.  Sequence fields are
        comma-separated; blank values are ignored.  Explicit keyword
        ``overrides`` win over the environment.

        Parameters
        ----------
        env
            Mapping to read instead of :data:`os.environ`.
        **overrides
            Field values applied after the environment.

        Returns
        -------
        OperationalConfig
        """
        source: Mapping[str, str] = os.environ if env is None else env
        kwargs: dict[str, object] = {}

        def _get(field: str) -> str | None:
            raw = source.get(ENV_PREFIX + field.upper())
            if raw is None:
                return None
            raw = raw.strip()
            return raw or None

        for field in ("satellites", "flow_bands", "rad_bands"):
            raw = _get(field)
            if raw is not None:
                kwargs[field] = tuple(
                    part.strip() for part in raw.split(",") if part.strip()
                )
        for field, caster in (
            ("cadence_minutes", int),
            ("availability_tolerance_minutes", float),
            ("resolution_m", float),
            ("row_strip", int),
            ("output_dir", Path),
            ("store_uri", str),
            ("device", str),
        ):
            raw = _get(field)
            if raw is not None:
                kwargs[field] = caster(raw)

        kwargs.update(overrides)
        config = cls(**kwargs)  # type: ignore[arg-type]
        logger.debug("Built OperationalConfig from environment: %s", config)
        return config


#: Process-wide default configuration.
#:
#: Asset and job definitions are built at module import, before any
#: Dagster resource exists to supply configuration, so they need a
#: concrete config to shape themselves against.  Runtime overrides
#: still arrive through the resources; this only fixes the *shape* of
#: the definitions (which satellites get an asset, what cadence the
#: partition space uses).
DEFAULT_CONFIG = OperationalConfig()
