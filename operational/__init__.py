"""Operational orchestration for the stereo-winds AMV system.

This package wraps the research code in :mod:`stereo_winds` in a Dagster
job that runs on a fixed cadence:

1. detect which satellites have imagery available for a timestamp,
2. retrieve AMVs **per satellite** (so a single failing satellite can be
   re-run on its own),
3. mosaic the per-satellite fields onto a global lat/lon grid,
4. publish the mosaic to an `icechunk <https://icechunk.io>`_ store,
   either on a local path or in S3.

The package is deliberately import-light: nothing here imports Dagster,
torch or satpy at package-import time, so ``import operational`` stays
cheap for tooling and tests.  The Dagster entry point lives in
``operational.definitions``; configuration lives in
:mod:`operational.config`.

See ``operational/README.md`` for the runbook.
"""

__all__ = ["OperationalConfig"]

_VERSION = "0.1.0"
__version__ = _VERSION


def __getattr__(name: str):
    """Lazily expose :class:`operational.config.OperationalConfig`.

    Keeps ``import operational`` free of the :mod:`stereo_winds` import
    chain (which pulls in torch) until the config is actually needed.
    """
    if name == "OperationalConfig":
        from operational.config import OperationalConfig

        return OperationalConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
