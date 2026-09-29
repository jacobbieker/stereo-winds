"""Running one satellite's AMV retrieval as a container.

The retrieval used to run in-process: the asset loaded the student and
RAFT checkpoints, called into the ring script, and wrote a NetCDF beside
the mosaic.  That ties every satellite to one host, because the mosaic
reads those files off local disk, and it means six satellites take turns
on whichever machine the code location happens to run on.

As a container each satellite is independent -- it carries its own
environment, it publishes to its own icechunk store, and nothing about
it assumes the other five are nearby.  The mosaic then reads the stores
rather than the filesystem, so the satellites can be retrieved wherever
there is a GPU free.

Credentials are read from the environment rather than accepted as
config: a Dagster config value is rendered into run history and the UI.
"""

import dataclasses
import datetime as dt
import logging
import os

from dagster import ConfigurableResource
from pydantic import Field

logger = logging.getLogger(__name__)

__all__ = [
    "AmvContainerResource",
    "satellite_env",
]


@dataclasses.dataclass(frozen=True)
class _Window:
    """The frames one retrieval needs, for logging and metadata."""

    t0: dt.datetime


def satellite_env(
    sat_id: str,
    t0: dt.datetime,
    *,
    icechunk_base: str,
    resolution_m: float,
    flow_bands: list[str],
    rad_bands: list[str],
    device: str = "cuda",
    row_strip: int = 512,
    workdir: str = "/work",
) -> dict[str, str]:
    """AMV_* variables for one satellite at one timestamp.

    ``--skip-global`` because this container retrieves one satellite and
    nothing else: the mosaic is a separate step that reads what every
    satellite published, and asking a single-satellite run to build a
    global mosaic would produce one with a single contributor.
    """
    return {
        "AMV_TIME": t0.strftime("%Y-%m-%dT%H:%M"),
        "AMV_SATELLITES": sat_id,
        "AMV_SATELLITE_ICECHUNK_BASE": icechunk_base,
        "AMV_RESOLUTION_M": str(int(resolution_m)),
        "AMV_FLOW_BANDS": ",".join(flow_bands),
        "AMV_RAD_BANDS": ",".join(rad_bands),
        "AMV_DEVICE": device,
        "AMV_ROW_STRIP": str(row_strip),
        "AMV_SKIP_GLOBAL": "true",
        # The disk is published to the store; the NetCDF beside it would
        # only be read by a mosaic on this same host, which is the
        # coupling this is removing.
        "AMV_NO_NETCDF": "true",
        "AMV_OUTPUT_DIR": workdir,
        "AMV_TEMP_DIR": workdir,
    }


class AmvContainerResource(ConfigurableResource):
    """Image and settings for the per-satellite retrieval containers."""

    image: str = Field(
        default=("033040503982.dkr.ecr.us-west-2.amazonaws.com/stereo-winds:latest"),
        description="Retrieval image to run, as a full registry reference.",
    )

    icechunk_base: str = Field(
        default="s3://tipplyai-us-data/amv",
        description=(
            "Base the per-satellite stores hang off; each satellite "
            "publishes to <base>/amv_<sat_id>.icechunk."
        ),
    )

    device: str = Field(default="cuda", description="AMV_DEVICE for the container.")

    row_strip: int = Field(
        default=512,
        description=(
            "Rows per inference tile.  Lower than the 1024 default "
            "because a container has only its own share of the host."
        ),
    )

    workdir: str = Field(
        default="/work",
        description="Container path for scratch; nothing is kept there.",
    )

    gpus: str = Field(
        default="all",
        description=(
            'GPUs to expose to the container: "all", a count, or "none" '
            "for a CPU run.  The retrieval loads its checkpoints straight "
            "onto the device it was asked for, so a container started "
            "without this fails on the first torch.load rather than "
            "quietly falling back to the CPU."
        ),
    )

    aws_region: str = Field(default="us-west-2")

    # Empty and read from os.environ when the container runs: an EnvVar
    # default has to resolve when the code location is *defined*, so a
    # deployment without credentials would fail to load every asset,
    # including the ones that need none.
    aws_access_key_id: str = Field(default="")
    aws_secret_access_key: str = Field(default="")

    def store_uri(self, sat_id: str) -> str:
        """Where ``sat_id`` publishes."""
        from stereo_winds.icechunk_output import satellite_store_uri

        return satellite_store_uri(self.icechunk_base, sat_id)

    def container_kwargs(self) -> dict:
        """Extra ``docker run`` settings for one retrieval container.

        Only the GPU, and only because nothing else grants it: docker
        hands a container no devices by default, so ``AMV_DEVICE=cuda``
        in an otherwise correct container means
        ``torch.cuda.is_available() is False`` and the run dies loading
        its first checkpoint.

        Spelled as the plain dict docker's API takes rather than
        ``docker.types.DeviceRequest`` so that importing this module
        does not need the docker package on a host that only reads the
        configuration.
        """
        wanted = self.gpus.strip().lower()
        if not wanted or wanted in ("none", "0"):
            return {}
        count = -1 if wanted == "all" else int(wanted)
        return {
            "device_requests": [{"Driver": "nvidia", "Count": count, "Capabilities": [["gpu"]]}]
        }

    def credential_env(self) -> dict[str, str]:
        """Credentials for the store, from config or the environment."""
        resolved = {
            var: (configured or os.environ.get(var, ""))
            for var, configured in (
                ("AWS_ACCESS_KEY_ID", self.aws_access_key_id),
                ("AWS_SECRET_ACCESS_KEY", self.aws_secret_access_key),
            )
        }
        missing = sorted(v for v, value in resolved.items() if not value)
        if missing:
            raise RuntimeError(
                f"the retrieval container needs {', '.join(missing)} to "
                f"publish to {self.icechunk_base}; set them in the "
                f"environment of whatever runs the AMV assets"
            )
        resolved["AWS_DEFAULT_REGION"] = self.aws_region
        resolved["AWS_REGION"] = self.aws_region
        return resolved
