"""Global mosaic and icechunk publish assets.

.. note::

   **Placeholder.**  Owned by the mosaic/publish unit.  What
   :mod:`operational.definitions` relies on is that this module exposes
   two timestamp-partitioned assets named :data:`global_mosaic` and
   :data:`published_mosaic`, that ``global_mosaic`` depends on every
   ``amv_<sat_id>`` asset, and that ``published_mosaic`` depends on
   ``global_mosaic``.  The bodies here do no gridding and no writing;
   replace them wholesale.

Both the partitions definition and the satellite list come from
:mod:`operational.assets.amv_assets`, so the mosaic cannot end up
declaring inputs the AMV layer does not build, or sitting on a different
partition grid from its own upstreams.

Note the deliberate absence of ``from __future__ import annotations``:
on Python 3.14 with dagster 1.13 the ``@asset`` decorator compares the
``context`` parameter's annotation by identity, and a stringised
``AssetExecutionContext`` is rejected.
"""

import logging

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    asset,
)

from operational.assets.amv_assets import AMV_ASSETS_BY_SAT, AMV_PARTITIONS_DEF

logger = logging.getLogger(__name__)

__all__ = ["MOSAIC_GROUP_NAME", "global_mosaic", "published_mosaic"]

#: Group the mosaic and publish assets show up under in the Dagster UI.
MOSAIC_GROUP_NAME = "mosaic"

_AMV_KEYS = [
    key for asset_def in AMV_ASSETS_BY_SAT.values() for key in asset_def.keys
]


@asset(
    partitions_def=AMV_PARTITIONS_DEF,
    deps=_AMV_KEYS,
    group_name=MOSAIC_GROUP_NAME,
    description="Min-zenith merge of every satellite's AMV retrieval.",
    kinds={"python"},
)
def global_mosaic(context: AssetExecutionContext) -> MaterializeResult:
    """Merge the per-satellite retrievals into one global grid."""
    logger.info("placeholder mosaic for partition %s", context.partition_key)
    return MaterializeResult(metadata={"partition": context.partition_key})


@asset(
    partitions_def=AMV_PARTITIONS_DEF,
    deps=[global_mosaic],
    group_name=MOSAIC_GROUP_NAME,
    description="The mosaic, appended to the operational icechunk store.",
    kinds={"icechunk"},
)
def published_mosaic(context: AssetExecutionContext) -> MaterializeResult:
    """Append the mosaic for this partition to the icechunk store."""
    logger.info("placeholder publish for partition %s", context.partition_key)
    return MaterializeResult(metadata={"partition": context.partition_key})
