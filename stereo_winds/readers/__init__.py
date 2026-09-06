"""Standalone satellite data readers for stereo-winds.

``GOES`` reads GOES-R ABI L1b radiance from NOAA's public S3 buckets without
authentication or satpy. ``Himawari``, ``GK2A``, and ``MTG`` read from
icechunk stores at source.coop (requires ``icechunk``).
"""
from stereo_winds.readers.goes import GOES
from stereo_winds.readers.gk2a import GK2A
from stereo_winds.readers.himawari import Himawari
from stereo_winds.readers.mtg import MTG

__all__ = ["GOES", "GK2A", "Himawari", "MTG"]
