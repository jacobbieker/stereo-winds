"""Sensors driving the operational AMV pipeline.

.. note::

   **Placeholder.**  Owned by the sensor unit.  What
   :mod:`operational.definitions` relies on is only that this module
   exposes one or more Dagster sensor objects at module level, and that
   any job they target exists in the code location — the real sensor
   should keep targeting :data:`operational.jobs.FULL_JOB_NAME` rather
   than hard-coding a job name of its own.  The evaluation below never
   requests a run; replace it wholesale.

Constructing a sensor must not touch the network: the availability probe
belongs inside the evaluation function, which only ever runs in the
daemon, so that importing the code location stays offline.
"""

from __future__ import annotations

import logging

from dagster import (
    DefaultSensorStatus,
    SensorEvaluationContext,
    SensorResult,
    sensor,
)

from operational.jobs import FULL_JOB_NAME

logger = logging.getLogger(__name__)

__all__ = ["AVAILABILITY_SENSOR_NAME", "availability_sensor"]

#: Name of the sensor that watches for newly available imagery.
AVAILABILITY_SENSOR_NAME = "operational_availability_sensor"


@sensor(
    name=AVAILABILITY_SENSOR_NAME,
    job_name=FULL_JOB_NAME,
    minimum_interval_seconds=60,
    default_status=DefaultSensorStatus.RUNNING,
    description=(
        "Watches the satellite archives for newly complete timestamps and "
        "requests a pipeline run for each one."
    ),
)
def availability_sensor(context: SensorEvaluationContext) -> SensorResult:
    """Request a run for every timestamp whose imagery has landed."""
    logger.debug("placeholder availability sensor tick")
    return SensorResult(
        run_requests=[],
        skip_reason="placeholder availability sensor: never requests runs",
    )
