"""Pure-Python core of the operational pipeline.

Modules here must stay importable without Dagster running and without
touching the network, so they can be unit-tested in isolation.
"""
