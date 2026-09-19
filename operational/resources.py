"""Dagster resources for the operational AMV pipeline.

.. note::

   **Placeholder.**  This module is owned by the resources unit; the
   implementations here are the minimal, contract-satisfying stand-ins
   the jobs/Definitions unit needs in order to build and validate a
   loadable :class:`~dagster.Definitions`.  Replace wholesale with the
   real implementations — only the class names, resource keys and the
   *laziness contract* below are load-bearing for
   :mod:`operational.definitions`.

Laziness contract
-----------------
Constructing any resource in this module must be free of side effects:
no checkpoint is read, no network call is made, no CUDA context is
created, and no icechunk store is opened or created on disk.  That is
what lets ``dagster dev -m operational.definitions`` load the code
location on a machine with no GPU, no credentials and no data — and what
lets the tests point every path at somewhere that does not exist and
still import the module.  Expensive work happens on first *use* inside a
run, behind :meth:`ModelResource.load` /
:meth:`IcechunkStoreResource.open`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dagster import ConfigurableResource

logger = logging.getLogger(__name__)

__all__ = [
    "PathsResource",
    "IcechunkStoreResource",
    "ModelResource",
    "RunSettingsResource",
]


class PathsResource(ConfigurableResource):
    """Where intermediate and final files land on disk.

    Attributes
    ----------
    output_dir : str
        Root directory for per-satellite AMV NetCDFs and mosaics.  It is
        created on first use, never at construction time.
    """

    output_dir: str = "output/operational"

    @property
    def output_path(self) -> Path:
        """The configured output directory as a :class:`~pathlib.Path`."""
        return Path(self.output_dir)

    def ensure_output_dir(self) -> Path:
        """Create the output directory if needed and return it.

        Returns
        -------
        pathlib.Path
            The existing output directory.
        """
        path = self.output_path
        path.mkdir(parents=True, exist_ok=True)
        return path


class IcechunkStoreResource(ConfigurableResource):
    """Handle on the icechunk store the mosaics are published to.

    Attributes
    ----------
    store_uri : str
        Local path or ``s3://bucket/prefix`` URI of the store.
    create_if_missing : bool
        Whether :meth:`open` may create the store on first write.
    """

    store_uri: str = "output/operational.icechunk"
    create_if_missing: bool = True

    def open(self) -> Any:
        """Open (or create) the store.

        Returns
        -------
        Any
            An icechunk repository handle.

        Raises
        ------
        NotImplementedError
            Always, in this placeholder.  The real resource opens the
            store here — the important part is that nothing touches disk
            until this method is called.
        """
        raise NotImplementedError(
            "PathsResource/IcechunkStoreResource placeholders: replace "
            "operational/resources.py with the real implementations."
        )


class ModelResource(ConfigurableResource):
    """Lazy handle on the student model checkpoint.

    Attributes
    ----------
    checkpoint_path : str
        Path to the student checkpoint.  Not read at construction time,
        so the path is allowed not to exist until a run needs it.
    device : str
        Torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``).
    row_strip : int
        Rows per forward-pass strip over the full disk.
    """

    checkpoint_path: str = "checkpoints/student_zeus.ckpt"
    device: str = "cpu"
    row_strip: int = 1024

    def load(self) -> Any:
        """Load the checkpoint onto :attr:`device`.

        Returns
        -------
        Any
            The loaded student model.

        Raises
        ------
        NotImplementedError
            Always, in this placeholder.
        """
        raise NotImplementedError(
            "ModelResource placeholder: replace operational/resources.py "
            "with the real implementation."
        )


class RunSettingsResource(ConfigurableResource):
    """Knobs shared by every step of one operational run.

    Attributes
    ----------
    satellites : list of str
        Satellite ids in the ring, in mosaicking order.
    flow_bands, rad_bands : list of str
        Bands fed to the student model.
    cadence_minutes : int
        Spacing of operational timestamps.
    availability_tolerance_minutes : float
        Slack between a scene's own time and its nominal slot.
    resolution_m : float
        Mosaic grid spacing, in meters.
    """

    satellites: list[str] = ["goes18", "goes19", "himawari9", "gk2a"]
    flow_bands: list[str] = ["C08", "C09", "C10", "C12", "C14"]
    rad_bands: list[str] = ["C07", "C08", "C09", "C10", "C11", "C13", "C14", "C15"]
    cadence_minutes: int = 60
    availability_tolerance_minutes: float = 5.0
    resolution_m: float = 10000.0
