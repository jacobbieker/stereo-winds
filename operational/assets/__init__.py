"""Dagster assets for the operational AMV pipeline.

The graph is three layers deep for every timestamp partition:

``amv_<sat_id>`` (one per satellite)  ->  ``global_mosaic``  ->
``published_mosaic``

:mod:`operational.assets.amv_assets` builds the per-satellite layer,
:mod:`operational.assets.mosaic_assets` the mosaic and the icechunk
publish step.
"""

__all__ = ["amv_assets", "mosaic_assets"]
