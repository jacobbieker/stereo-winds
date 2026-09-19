"""Configuration for the operational AMV pipeline.

A single frozen :class:`OperationalConfig` describes everything the
Dagster assets need: which satellites to run, which bands to feed the
student model, how often to look for new imagery, and where the results
land.  Instances are immutable so the same object can be shared between
Dagster assets without one step perturbing another.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Mapping

from stereo_winds.student_dataset import DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS

logger = logging.getLogger(__name__)

#: Prefix for every environment variable read by
#: :meth:`OperationalConfig.from_env`.
ENV_PREFIX = "STEREO_WINDS_OP_"

#: Scheme marking a ``store_uri`` as living in object storage.
S3_SCHEME = "s3://"


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
    """

    satellites: tuple[str, ...] = ("goes18", "goes19", "himawari9", "gk2a")
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
        """Whether :attr:`store_uri` points at object storage."""
        return str(self.store_uri).startswith(S3_SCHEME)

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
        rest = str(self.store_uri)[len(S3_SCHEME):]
        bucket, _, prefix = rest.partition("/")
        if not bucket:
            raise ValueError(f"store_uri {self.store_uri!r} names no bucket")
        return bucket, prefix.strip("/")

    # -- timestamps --------------------------------------------------------

    def floor_to_cadence(self, when: datetime) -> datetime:
        """Round ``when`` down to the most recent cadence-aligned slot.

        Slots are anchored at midnight of the given date, so a 60 minute
        cadence yields the top of the hour and a 10 minute cadence the
        nearest ten-minute mark.

        Parameters
        ----------
        when
            Timestamp to floor.

        Returns
        -------
        datetime
            The cadence-aligned slot at or before ``when``.
        """
        midnight = when.replace(hour=0, minute=0, second=0, microsecond=0)
        elapsed = (when - midnight) // self.cadence
        return midnight + elapsed * self.cadence

    def timestamps(self, start: datetime, end: datetime) -> Iterator[datetime]:
        """Yield cadence-aligned timestamps in ``[start, end]``.

        ``start`` is floored to the cadence first, so a backfill window
        given in wall-clock terms still lands on the operational slots.

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
        return replace(self, satellites=tuple(satellites))

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
