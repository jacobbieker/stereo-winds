"""Operational orchestration for the global satellite-wind ring.

This package wraps the research-oriented :mod:`stereo_winds` code and the
``scripts/infer_student_global_ring.py`` CLI in a Dagster-orchestrated
service: timestamp detection, one AMV retrieval per satellite, mosaicing,
and a final write to an icechunk store.
"""

__all__ = ["OperationalConfig"]

from operational.config import OperationalConfig
