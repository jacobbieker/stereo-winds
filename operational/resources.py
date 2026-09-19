"""Dagster resources for the operational stereo-winds pipeline.

The assets get their output paths, icechunk store, model weights and run
settings from these resources rather than importing them directly.  That
keeps the heavy dependencies — torch, satpy, icechunk, the ring script —
out of the Dagster definitions module, so the graph can be loaded,
validated and unit-tested on a machine with no GPU, no checkpoints and no
store.

Two rules hold throughout:

* **Import-light.** torch, icechunk and :mod:`operational.adapters.ring`
  are imported inside the methods that need them, never at module scope
  — importing this module costs a dagster import and nothing else.  The
  one thing to know is that :class:`PathsResource`'s path helpers go
  through the ring script for the canonical filenames, so the *first*
  call to one of them pays the ring import (torch, satpy; a few seconds,
  once per process) even though it only formats a string.
* **Side-effect-free construction.** Building a resource never touches
  the filesystem or the network.  Directories are created, checkpoints
  read and stores opened only when a method explicitly asks for them,
  and the result is cached for the resource's lifetime.

Timestamps crossing this boundary are **naive UTC** — see
:func:`as_naive_utc`, and use it on anything coming from a Dagster
partition, which is tz-aware.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dagster import ConfigurableResource
from pydantic import Field, PrivateAttr

from operational.config import (
    DEFAULT_FLOW_BANDS,
    DEFAULT_RAD_BANDS,
    OperationalConfig,
)

logger = logging.getLogger(__name__)

_DEFAULTS = OperationalConfig()


def as_naive_utc(t: datetime) -> datetime:
    """Normalise a timestamp to the naive-UTC convention used throughout.

    Every timestamp that crosses this module's boundary — filenames,
    icechunk ``time`` coordinates, ring-script scan times — is naive and
    understood as UTC.  Dagster partition windows, by contrast, are
    tz-aware UTC, so a raw ``t in store.existing_times()`` between the
    two is always ``False`` and a ``<`` comparison raises ``TypeError``.
    Passing timestamps through here first removes that trap.

    Parameters
    ----------
    t : datetime
        Naive (assumed UTC) or tz-aware.

    Returns
    -------
    datetime
        Naive, shifted to UTC when the input carried an offset.
    """
    if t.tzinfo is None:
        return t
    return t.astimezone(timezone.utc).replace(tzinfo=None)


__all__ = [
    "as_naive_utc",
    "PathsResource",
    "IcechunkStoreResource",
    "ModelResource",
    "RunSettingsResource",
]


class PathsResource(ConfigurableResource):
    """Canonical on-disk locations for per-satellite disks and mosaics.

    The naming scheme is owned by the ring script, so the paths are
    delegated to its ``sat_nc_path`` / ``global_nc_path`` rather than
    re-derived here — an operational run and a manual ring run write to
    exactly the same files.
    """

    output_dir: str = Field(
        default=str(_DEFAULTS.output_dir),
        description=(
            "Root directory for NetCDF output.  Per-satellite disks and "
            "global mosaics are written into per-day subdirectories "
            "beneath it."
        ),
    )

    @property
    def root(self) -> Path:
        """The output root as a :class:`~pathlib.Path` (not created)."""
        return Path(self.output_dir)

    def ensure_root(self) -> Path:
        """Create the output root if absent and return it."""
        root = self.root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def sat_path(self, sat_id: str, t: datetime, *, create: bool = False) -> Path:
        """Path of the per-satellite full-disk AMV file for ``t``.

        Parameters
        ----------
        sat_id : str
            Satellite id, e.g. ``"goes18"``.
        t : datetime
            Nominal timestamp of the retrieval.
        create : bool, default False
            Create the containing per-day directory.  The file itself is
            never created here.

        Returns
        -------
        pathlib.Path
        """
        from operational.adapters.ring import sat_nc_path

        path = Path(sat_nc_path(self.root, sat_id, t))
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def mosaic_path(self, t: datetime, *, create: bool = False) -> Path:
        """Path of the merged global mosaic file for ``t``.

        Parameters
        ----------
        t : datetime
            Nominal timestamp of the mosaic.
        create : bool, default False
            Create the containing per-day directory.

        Returns
        -------
        pathlib.Path
        """
        from operational.adapters.ring import global_nc_path

        path = Path(global_nc_path(self.root, t))
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def day_dir(self, t: datetime, *, create: bool = False) -> Path:
        """Per-day subdirectory holding every file for ``t``'s date."""
        # Derived from the mosaic path so there is a single source of
        # truth for the layout.
        parent = self.mosaic_path(t).parent
        if create:
            parent.mkdir(parents=True, exist_ok=True)
        return parent


class IcechunkStoreResource(ConfigurableResource):
    """The icechunk repository the global mosaics are committed to.

    Accepts either a local directory path or ``s3://bucket/prefix``.  The
    repository is opened on the first :meth:`repo` call and reused for
    the rest of the resource's lifetime; constructing the resource does
    not create the store or contact S3.
    """

    store_uri: str = Field(
        default=_DEFAULTS.store_uri,
        description=(
            "Icechunk store location: a local directory path or "
            "'s3://bucket/prefix'."
        ),
    )
    endpoint_url: str | None = Field(
        default=None,
        description="Custom S3 endpoint (MinIO, Ceph, ...). S3 stores only.",
    )
    region: str | None = Field(
        default=None,
        description="AWS region of the bucket. S3 stores only.",
    )
    anonymous: bool = Field(
        default=False,
        description=(
            "Access the bucket anonymously instead of using ambient "
            "credentials. S3 stores only."
        ),
    )
    force_path_style: bool = Field(
        default=False,
        description=(
            "Use path-style S3 addressing (bucket in the path rather "
            "than the host). Needed by most S3-compatible servers."
        ),
    )
    branch: str = Field(
        default="main",
        description="Icechunk branch that mosaics are committed to.",
    )
    chunk: int = Field(
        default=1024,
        description=(
            "Chunk size, in cells, along latitude and longitude when the "
            "store's dataset is first created."
        ),
    )

    _repo: Any = PrivateAttr(default=None)

    @property
    def is_s3(self) -> bool:
        """Whether :attr:`store_uri` points at object storage.

        ``False`` does **not** imply a usable local path: an unsupported
        scheme such as ``gs://`` also reports ``False`` and is rejected
        by :meth:`repo`.  Branch on this only to decide whether the S3
        settings apply, never to build a local path from the URI.
        """
        return self.store_uri.startswith(("s3://", "s3a://"))

    def repo(self) -> "icechunk.Repository":
        """Open (or create) the repository, caching it.

        Returns
        -------
        icechunk.Repository
            The same object on every call for this resource instance.
        """
        if self._repo is None:
            from stereo_winds.icechunk_output import open_icechunk_repo

            logger.info("Opening icechunk store %s", self.store_uri)
            self._repo = open_icechunk_repo(
                self.store_uri,
                endpoint_url=self.endpoint_url,
                region=self.region,
                anonymous=self.anonymous,
                force_path_style=self.force_path_style,
            )
        return self._repo

    def existing_times(self) -> set[datetime]:
        """Timestamps already committed on :attr:`branch`.

        Returns
        -------
        set of datetime
            **Naive** datetimes understood as UTC — that is what the
            store's ``time`` coordinate decodes to.  Compare against them
            with :meth:`has_time` rather than a bare ``in``, which is
            always ``False`` for the tz-aware datetimes Dagster
            partitions hand out.

        Notes
        -----
        Upstream ``icechunk_existing_times`` swallows every exception and
        returns an empty set, so a transient S3 error is indistinguishable
        from a brand-new store.  A caller that skips already-present
        timestamps will therefore reprocess the whole backlog after such
        an error rather than fail — noisy, not silent corruption, but
        worth knowing when reading a run's logs.
        """
        from stereo_winds.icechunk_output import icechunk_existing_times

        return icechunk_existing_times(self.repo(), self.branch)

    def has_time(self, t: datetime) -> bool:
        """Whether ``t`` is already committed, tz-aware input included.

        Parameters
        ----------
        t : datetime
            Naive (assumed UTC) or tz-aware; normalised either way.
        """
        return as_naive_utc(t) in self.existing_times()

    def set_repo(self, repo: Any) -> None:
        """Inject a repository, bypassing :meth:`repo`'s lazy open.

        For direct unit tests that already hold a repository.  Dagster
        rebuilds a resource from its config fields when a run
        initialises resources, so an injection made before
        :func:`~dagster.materialize` does not survive — pass a stand-in
        object as the resource value there instead.
        """
        self._repo = repo


class ModelResource(ConfigurableResource):
    """The student model and the RAFT disparity engine.

    Both checkpoints are loaded on first use and cached, so an asset that
    never runs inference — or a graph that is merely validated — costs
    nothing.  Construction succeeds even when the checkpoint paths do not
    exist; the error surfaces from :meth:`model` / :meth:`disparity`.

    Injecting fakes
    ---------------
    For a direct unit test, call :meth:`set_model` / :meth:`set_disparity`
    to populate the caches; the lazy load is then never reached.  Inside
    Dagster, pass a stand-in object instead::

        class FakeModel:
            def model(self): return my_model
            def disparity(self): return my_disparity
            def available(self): return True

        materialize([my_asset], resources={"model": FakeModel()})

    The stand-in is required there because Dagster rebuilds a
    ``ConfigurableResource`` from its *config fields* when the run
    initialises resources, which drops anything held in a private
    attribute.  :meth:`available` is the cheap predicate an asset can use
    to decide whether real inference is possible at all.
    """

    student_ckpt: str = Field(
        default="",
        description="Path to the student (single-satellite) Lightning checkpoint.",
    )
    raft_ckpt: str = Field(
        default="",
        description="Path to the RAFT optical-flow checkpoint.",
    )
    device: str = Field(
        default=_DEFAULTS.device,
        description="Torch device for inference, e.g. 'cpu' or 'cuda'.",
    )
    row_strip: int = Field(
        default=_DEFAULTS.row_strip,
        description=(
            "Rows per inference strip.  Smaller strips cap peak memory "
            "at the cost of more passes over the disk."
        ),
    )

    _model: Any = PrivateAttr(default=None)
    _disparity: Any = PrivateAttr(default=None)

    def available(self) -> bool:
        """Whether both checkpoints are configured and present on disk.

        Never loads anything — safe to call in a sensor or a test to
        decide whether real inference is possible.
        """
        return all(
            bool(p) and Path(p).exists()
            for p in (self.student_ckpt, self.raft_ckpt)
        )

    def _require_ckpt(self, path: str, what: str) -> Path:
        """Validate one checkpoint path, raising a pointed error."""
        if not path:
            raise ValueError(
                f"ModelResource.{what} is not configured — set it to the "
                f"path of the {what.replace('_', ' ')} file"
            )
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"ModelResource.{what} checkpoint not found: {p}"
            )
        return p

    def model(self) -> "StudentWindsModel":
        """Load the student model on first call and cache it.

        Returns
        -------
        stereo_winds.student_zeus_model.StudentWindsModel
            In ``eval()`` mode, mapped onto :attr:`device`.
        """
        if self._model is None:
            path = self._require_ckpt(self.student_ckpt, "student_ckpt")
            from stereo_winds.student_zeus_model import StudentWindsModel

            logger.info("Loading student checkpoint: %s", path)
            self._model = StudentWindsModel.load_from_checkpoint(
                str(path), map_location=self.device,
            ).eval()
        return self._model

    def disparity(self) -> "StereoDisparity":
        """Load the RAFT disparity engine on first call and cache it.

        Returns
        -------
        stereo_winds.disparity.StereoDisparity
        """
        if self._disparity is None:
            path = self._require_ckpt(self.raft_ckpt, "raft_ckpt")
            from stereo_winds.disparity import StereoDisparity

            logger.info("Loading RAFT checkpoint: %s", path)
            self._disparity = StereoDisparity(
                model_ckpt_path=str(path),
                tile_size=512,
                overlap=128,
                batch_size=8,
                device=self.device,
            )
        return self._disparity

    def set_model(self, model: Any) -> None:
        """Inject a model, bypassing :meth:`model`'s lazy load (tests)."""
        self._model = model

    def set_disparity(self, disparity: Any) -> None:
        """Inject a disparity engine, bypassing the lazy load (tests)."""
        self._disparity = disparity


class RunSettingsResource(ConfigurableResource):
    """Knobs that decide *what* a run retrieves, rather than *how*.

    Defaults mirror :class:`operational.config.OperationalConfig`.
    """

    satellites: list[str] = Field(
        default_factory=lambda: list(_DEFAULTS.satellites),
        description="Satellite ids to retrieve winds for.",
    )
    flow_bands: list[str] = Field(
        default_factory=lambda: list(DEFAULT_FLOW_BANDS),
        description="Bands fed to the optical-flow half of the student model.",
    )
    rad_bands: list[str] = Field(
        default_factory=lambda: list(DEFAULT_RAD_BANDS),
        description="Bands fed to the radiance half of the student model.",
    )
    cadence_minutes: int = Field(
        default=_DEFAULTS.cadence_minutes,
        description="Spacing, in minutes, of the timestamps the service targets.",
    )
    availability_tolerance_minutes: float = Field(
        default=_DEFAULTS.availability_tolerance_minutes,
        description=(
            "How far a satellite's real scan time may sit from the target "
            "timestamp and still count as available."
        ),
    )
    resolution_m: float = Field(
        default=_DEFAULTS.resolution_m,
        description="Mosaic grid spacing, in metres.",
    )
    skip_existing: bool = Field(
        default=True,
        description=(
            "Skip a satellite or mosaic whose output file is already on "
            "disk instead of recomputing it."
        ),
    )

    def to_config(
        self,
        *,
        model: "ModelResource | None" = None,
        paths: "PathsResource | None" = None,
        store: "IcechunkStoreResource | None" = None,
        **overrides: Any,
    ) -> OperationalConfig:
        """Materialise the equivalent frozen :class:`OperationalConfig`.

        The fields this resource does not itself carry — ``device`` and
        ``row_strip`` (on :class:`ModelResource`), ``output_dir`` (on
        :class:`PathsResource`) and ``store_uri`` (on
        :class:`IcechunkStoreResource`) — are read off the sibling
        resources when they are supplied.  Pass them: otherwise the
        result silently carries ``OperationalConfig``'s own defaults, so
        a ``device="cuda"`` :class:`ModelResource` would yield a config
        saying ``"cpu"`` and inference would quietly run on the CPU.

        Parameters
        ----------
        model, paths, store : resource, optional
            Siblings to take the remaining fields from.
        **overrides
            Explicit values, applied last and winning over the siblings.

        Returns
        -------
        OperationalConfig
        """
        fields: dict[str, Any] = {
            "satellites": tuple(self.satellites),
            "flow_bands": tuple(self.flow_bands),
            "rad_bands": tuple(self.rad_bands),
            "cadence_minutes": self.cadence_minutes,
            "availability_tolerance_minutes": self.availability_tolerance_minutes,
            "resolution_m": self.resolution_m,
        }
        if model is not None:
            fields["device"] = model.device
            fields["row_strip"] = model.row_strip
        if paths is not None:
            fields["output_dir"] = Path(paths.output_dir)
        if store is not None:
            fields["store_uri"] = store.store_uri
        fields.update(overrides)
        return OperationalConfig(**fields)
