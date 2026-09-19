"""Adapters between the operational service and the research codebase.

Each module here owns exactly one seam to code that lives outside the
``operational`` package, so the Dagster assets never have to reach into a
script or reimplement pipeline logic.
"""

__all__ = ["ring"]
