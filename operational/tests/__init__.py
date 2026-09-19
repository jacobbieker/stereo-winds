"""Tests for the operational pipeline.

This is a package (rather than a bare directory) on purpose: pytest
imports test modules by basename unless they live in one, and
``operational/tests/test_config.py`` would otherwise collide with the
repo-root ``tests/test_config.py`` and abort collection for the whole
repository.  Keep this file so new test modules here are free to reuse
upstream test basenames.
"""
