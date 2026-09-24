"""Running the satellite-consumer container for the EUMETSAT satellites.

MTG, MSG and IODC are not in the public S3 buckets the rest of the ring
reads, so their imagery has to be pulled from EUMETSAT and written to
icechunk before any retrieval can run.  That is what
``openclimatefix/satellite-consumer`` does, and it runs here as a
container rather than as a library: it pins its own conda environment
(see ``docker/satellite-consumer.Dockerfile`` for why that matters) and
nothing in it is importable from this package.

Credentials are read from the environment, never from config: a Dagster
config value is rendered into run history and the UI, and these are live
EUMETSAT and S3 keys.
"""

from __future__ import annotations

import dataclasses
import logging
import datetime as dt
from datetime import datetime, timedelta

from dagster import ConfigurableResource, EnvVar
from pydantic import Field

logger = logging.getLogger(__name__)

__all__ = [
    "CONSUMER_SATELLITES",
    "ConsumerSatellite",
    "SatelliteConsumerResource",
    "consumer_satellite",
]


@dataclasses.dataclass(frozen=True)
class ConsumerSatellite:
    """One satellite the consumer can be asked for.

    ``key`` is the consumer's own name for it, which is what
    ``SATCONS_SATELLITE`` takes and is not the ring's satellite id: the
    ring calls MTG-I1 ``mtg-i1`` where the consumer calls it
    ``odegree-12``.
    """

    key: str
    """``SATCONS_SATELLITE`` value."""

    ring_id: str | None
    """Matching ``SATELLITE_CONFIGS`` id, or None if the ring has none."""

    store: str
    """Destination icechunk store, relative to the data bucket."""

    resolution_m: int
    """``SATCONS_RESOLUTION``, in metres."""

    cadence_mins: int
    """Native repeat cycle, from the consumer's own application.conf."""

    description: str


#: The EUMETSAT satellites, keyed by the consumer's name for each.
#:
#: The stores are the ones that already exist in the bucket and that the
#: readers in ``stereo_winds.readers`` discover -- ``mtg_`` for MTG and
#: ``iodc_`` for IODC -- so a consumer run tops up the same store a
#: retrieval later reads, rather than creating a parallel one.  The
#: resolutions are each instrument's native grid: FCI at 1 km, SEVIRI at
#: 3 km.
CONSUMER_SATELLITES: dict[str, ConsumerSatellite] = {
    "odegree-12": ConsumerSatellite(
        key="odegree-12",
        ring_id="mtg-i1",
        store="geo/mtg_1000m.icechunk",
        resolution_m=1000,
        cadence_mins=10,
        description="MTG-I1 FCI at 0 degrees",
    ),
    "odegree": ConsumerSatellite(
        key="odegree",
        ring_id=None,
        store="geo/msg_3000m.icechunk",
        resolution_m=3000,
        cadence_mins=15,
        description="MSG SEVIRI at 0 degrees",
    ),
    "iodc": ConsumerSatellite(
        key="iodc",
        ring_id="msg-iodc",
        store="geo/iodc_3000m.icechunk",
        resolution_m=3000,
        cadence_mins=15,
        description="MSG SEVIRI Indian Ocean, 45.5 degrees East",
    ),
}


def consumer_satellite(key: str) -> ConsumerSatellite:
    """The :class:`ConsumerSatellite` for ``key``, or a clear error."""
    try:
        return CONSUMER_SATELLITES[key]
    except KeyError:
        raise KeyError(
            f"Unknown consumer satellite {key!r}; known: "
            f"{', '.join(sorted(CONSUMER_SATELLITES))}"
        ) from None


#: Consumer settings that do not vary by satellite or by run.
_STATIC_ENV: dict[str, str] = {
    # The consumer writes icechunk, not plain zarr.
    "SATCONS_ICECHUNK": "True",
    # Raw files are scratch: they are downloaded, read once and dropped.
    # Keeping them would fill the volume within a day at MTG's cadence.
    "SATCONS_KEEP_RAW": "False",
}


class SatelliteConsumerResource(ConfigurableResource):
    """Runs the satellite-consumer container for one satellite and window.

    The image is addressed by digest or tag in ECR; see
    ``docker/satellite-consumer.Dockerfile`` for what is in it and why it
    is not the upstream image.
    """

    image: str = Field(
        default=("033040503982.dkr.ecr.us-west-2.amazonaws.com/" "satellite-consumer:80c1b40-pixi"),
        description="Consumer image to run, as a full registry reference.",
    )

    bucket_url: str = Field(
        default="s3://us-west-2.opendata.source.coop/bkr",
        description=(
            "Base URL the per-satellite store paths hang off.  Joined "
            "with a satellite's store to form SATCONS_ZARR_PATH."
        ),
    )

    workdir: str = Field(
        default="/work",
        description=(
            "Container path for raw downloads.  Scratch: the consumer "
            "reads each file once and KEEP_RAW is off."
        ),
    )

    max_workers: int = Field(
        default=4,
        description="SATCONS_MAX_WORKERS, the consumer's download/process pool.",
    )

    timeout_seconds: int = Field(
        default=3 * 60 * 60,
        description=(
            "How long one satellite's window may take before the run is "
            "abandoned.  A backfill of many cycles is slow; a single "
            "cycle should be minutes."
        ),
    )

    # Read from the environment rather than accepted as config: config
    # values are echoed into run history and the Dagster UI.
    eumetsat_key: str = Field(default=EnvVar("EUMETSAT_CONSUMER_KEY"))
    eumetsat_secret: str = Field(default=EnvVar("EUMETSAT_CONSUMER_SECRET"))
    aws_access_key_id: str = Field(default=EnvVar("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key: str = Field(default=EnvVar("AWS_SECRET_ACCESS_KEY"))
    aws_region: str = Field(default="us-west-2")

    def store_url(self, sat: ConsumerSatellite) -> str:
        """Full ``SATCONS_ZARR_PATH`` for ``sat``."""
        return f"{self.bucket_url.rstrip('/')}/{sat.store.lstrip('/')}"

    def window_env(
        self,
        sat: ConsumerSatellite,
        start: datetime,
        end: datetime,
    ) -> dict[str, str]:
        """Every SATCONS_* variable for one satellite and time window.

        The consumer takes its whole configuration from the environment,
        so this is the entire interface to the container.
        """
        if end <= start:
            raise ValueError(
                f"window end {end.isoformat()} is not after start " f"{start.isoformat()}"
            )
        env = dict(_STATIC_ENV)
        env.update(
            {
                "SATCONS_SATELLITE": sat.key,
                "SATCONS_RESOLUTION": str(sat.resolution_m),
                "SATCONS_START_TIMESTAMP": _stamp(start),
                "SATCONS_END_TIMESTAMP": _stamp(end),
                "SATCONS_ZARR_PATH": self.store_url(sat),
                "SATCONS_RAW_PATH": f"{self.workdir.rstrip('/')}/raw",
                "SATCONS_WORKDIR": self.workdir,
                "SATCONS_MAX_WORKERS": str(self.max_workers),
            }
        )
        return env

    def credential_env(self) -> dict[str, str]:
        """The secrets the container needs, resolved from the environment.

        Kept apart from :meth:`window_env` so the window can be logged
        and asserted on in tests without carrying keys through it.
        """
        return {
            "EUMETSAT_CONSUMER_KEY": self.eumetsat_key,
            "EUMETSAT_CONSUMER_SECRET": self.eumetsat_secret,
            "AWS_ACCESS_KEY_ID": self.aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.aws_secret_access_key,
            "AWS_DEFAULT_REGION": self.aws_region,
            "AWS_REGION": self.aws_region,
        }


def _stamp(t: datetime) -> str:
    """A timestamp in the form the consumer parses.

    It calls ``datetime.fromisoformat``, and every timestamp in this
    package is naive UTC, so an aware value is converted to UTC rather
    than having its offset dropped and its meaning silently changed.
    """
    if t.tzinfo is not None:
        t = t.astimezone(dt.UTC).replace(tzinfo=None)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def window_for(
    sat: ConsumerSatellite,
    start: datetime,
    *,
    cycles: int = 1,
) -> tuple[datetime, datetime]:
    """The ``[start, end)`` window covering ``cycles`` of ``sat``'s cadence."""
    if cycles < 1:
        raise ValueError(f"cycles must be positive, got {cycles}")
    return start, start + timedelta(minutes=sat.cadence_mins * cycles)
