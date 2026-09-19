"""Per-satellite AMV retrieval assets.

.. note::

   **Placeholder.**  Owned by the AMV-asset unit.  What
   :mod:`operational.definitions` relies on is the module's public
   surface — :func:`build_amv_asset`, :data:`AMV_ASSETS`,
   :data:`AMV_ASSETS_BY_SAT` and :data:`AMV_PARTITIONS_DEF` — and the
   asset key naming rule in :func:`amv_asset_key`.  The bodies here run
   no inference; replace them wholesale.

Note the deliberate absence of ``from __future__ import annotations``:
on Python 3.14 with dagster 1.13 the ``@asset`` decorator compares the
``context`` parameter's annotation by identity, and a stringised
``AssetExecutionContext`` is rejected.
"""

import logging
import os
import re
from datetime import datetime

from dagster import (
    AssetExecutionContext,
    AssetKey,
    AssetsDefinition,
    Backoff,
    Jitter,
    MaterializeResult,
    PartitionsDefinition,
    RetryPolicy,
    asset,
)

from operational.config import OperationalConfig
from operational.core.partitions import build_partitions_def

logger = logging.getLogger(__name__)

__all__ = [
    "AMV_ASSETS",
    "AMV_ASSETS_BY_SAT",
    "AMV_ASSET_PREFIX",
    "AMV_GROUP_NAME",
    "AMV_PARTITIONS_DEF",
    "DEFAULT_RETRY_POLICY",
    "amv_asset_key",
    "build_amv_asset",
    "slugify_sat_id",
]

#: Asset names are ``amv_<sat_id>`` so a satellite's asset is greppable
#: and so a job can name one without importing the assets.
AMV_ASSET_PREFIX = "amv"

#: Group the per-satellite assets show up under in the Dagster UI.
AMV_GROUP_NAME = "amv"

#: Retry policy for a per-satellite retrieval.
DEFAULT_RETRY_POLICY = RetryPolicy(
    max_retries=2,
    delay=120,
    backoff=Backoff.EXPONENTIAL,
    jitter=Jitter.PLUS_MINUS,
)

#: Environment variable naming the first timestamp partition.  Read here
#: rather than in :mod:`operational.definitions` because the partitions
#: definition has to exist before the module-scope assets are declared.
PARTITION_START_ENV_VAR = "STEREO_WINDS_OP_PARTITION_START"

#: First partition when nothing overrides it.
DEFAULT_PARTITION_START = "2026-01-01"

_NON_IDENTIFIER = re.compile(r"[^0-9a-zA-Z]+")


def slugify_sat_id(sat_id: str) -> str:
    """Turn a satellite id into something usable in an asset name.

    Dagster asset names must be valid Python identifiers, but satellite
    ids are not: ``mtg-i1`` and ``msg-iodc`` both carry a hyphen.

    Parameters
    ----------
    sat_id : str
        Satellite id, e.g. ``"goes18"`` or ``"mtg-i1"``.

    Returns
    -------
    str
        The id with every run of non-alphanumeric characters replaced by
        a single underscore, e.g. ``"mtg_i1"``.
    """
    return _NON_IDENTIFIER.sub("_", sat_id).strip("_")


def amv_asset_key(sat_id: str) -> AssetKey:
    """Asset key of the AMV retrieval for one satellite.

    Parameters
    ----------
    sat_id : str
        Satellite id, e.g. ``"goes18"``.

    Returns
    -------
    dagster.AssetKey
        The key ``amv_<slugified sat_id>``.
    """
    return AssetKey(f"{AMV_ASSET_PREFIX}_{slugify_sat_id(sat_id)}")


def partition_start_from_env(env=None) -> datetime:
    """First timestamp partition, from the environment or the default.

    Parameters
    ----------
    env : Mapping[str, str], optional
        Mapping to read instead of :data:`os.environ`.

    Returns
    -------
    datetime
        Naive UTC start of the partition range.

    Raises
    ------
    ValueError
        If the variable holds something
        :meth:`datetime.datetime.fromisoformat` cannot parse.  Raising
        with the variable's name beats the bare "Invalid isoformat
        string" an operator would otherwise see while a code location
        fails to load.
    """
    source = os.environ if env is None else env
    raw = source.get(PARTITION_START_ENV_VAR, "").strip() or DEFAULT_PARTITION_START
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(
            f"{PARTITION_START_ENV_VAR}={raw!r} is not an ISO 8601 "
            f"date/time, e.g. '2026-01-01' or '2026-01-01T00:00' ({exc})"
        ) from exc


def build_amv_asset(
    sat_id: str,
    partitions_def: PartitionsDefinition,
    *,
    group_name: str = AMV_GROUP_NAME,
) -> AssetsDefinition:
    """Build the AMV retrieval asset for one satellite.

    Parameters
    ----------
    sat_id : str
        Satellite id to retrieve winds for.
    partitions_def : dagster.PartitionsDefinition
        Timestamp partitions, shared with every other asset in the graph.
    group_name : str, optional
        Dagster asset group.

    Returns
    -------
    dagster.AssetsDefinition
        An asset keyed :func:`amv_asset_key`, partitioned by timestamp.
    """

    @asset(
        name=f"{AMV_ASSET_PREFIX}_{slugify_sat_id(sat_id)}",
        partitions_def=partitions_def,
        group_name=group_name,
        description=f"Student-model AMV retrieval for {sat_id}.",
        retry_policy=DEFAULT_RETRY_POLICY,
        kinds={"python"},
    )
    def _amv_asset(context: AssetExecutionContext) -> MaterializeResult:
        logger.info(
            "placeholder AMV asset for %s, partition %s",
            sat_id,
            context.partition_key,
        )
        return MaterializeResult(
            metadata={"satellite": sat_id, "partition": context.partition_key}
        )

    return _amv_asset


_CONFIG = OperationalConfig.from_env()

#: The timestamp partitions every asset in the pipeline shares.
AMV_PARTITIONS_DEF: PartitionsDefinition = build_partitions_def(
    partition_start_from_env(), _CONFIG.cadence_minutes
)

#: Satellite id to that satellite's AMV asset, in ring order.
AMV_ASSETS_BY_SAT: dict[str, AssetsDefinition] = {
    sat_id: build_amv_asset(sat_id, AMV_PARTITIONS_DEF)
    for sat_id in _CONFIG.satellites
}

#: Every per-satellite AMV asset.
AMV_ASSETS: list[AssetsDefinition] = list(AMV_ASSETS_BY_SAT.values())
