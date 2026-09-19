"""The full-disk grids depend only on the projection, so they are cached."""

import numpy as np
import pytest

from stereo_winds.config import SATELLITE_CONFIGS
from stereo_winds.navigation import (
    _grid_cache,
    clear_grid_cache,
    compute_grid_latlon,
    compute_grid_zenith,
    compute_pixel_scale,
)

SAT = SATELLITE_CONFIGS["msg-iodc"]      # smallest full disk, 3712^2


@pytest.fixture(autouse=True)
def _clean():
    clear_grid_cache()
    yield
    clear_grid_cache()


class TestGridCache:
    def test_cached_result_is_identical(self):
        first = compute_grid_zenith(SAT)
        second = compute_grid_zenith(SAT)
        assert np.array_equal(first, second, equal_nan=True)

    def test_second_call_does_not_recompute(self, monkeypatch):
        compute_grid_latlon(SAT)
        import stereo_winds.navigation as nav
        monkeypatch.setattr(nav, "fixed_grid_to_geodetic",
                            lambda *a, **k: pytest.fail("recomputed"))
        compute_grid_latlon(SAT)

    def test_zenith_reuses_the_cached_latlon(self):
        compute_grid_latlon(SAT)
        before = len(_grid_cache)
        compute_grid_zenith(SAT)
        assert len(_grid_cache) == before + 1     # only the zenith is added

    def test_all_three_grids_are_cached(self):
        compute_pixel_scale(SAT)
        compute_grid_latlon(SAT)
        compute_grid_zenith(SAT)
        kinds = {key[0] for key in _grid_cache}
        assert kinds == {"pixel_scale", "latlon", "zenith"}

    def test_a_different_projection_is_a_different_entry(self):
        """A relocated satellite must not get the previous slot's geometry."""
        from dataclasses import replace
        compute_grid_latlon(SAT)
        moved = replace(SAT, sub_lon_deg=SAT.sub_lon_deg + 5.0)
        lat_a, _ = compute_grid_latlon(SAT)
        lat_b, lon_b = compute_grid_latlon(moved)
        assert len(_grid_cache) == 2
        centre = (SAT.n_rows // 2, SAT.n_cols // 2)
        assert lon_b[centre] == pytest.approx(SAT.sub_lon_deg + 5.0, abs=0.01)

    def test_grid_shape_is_part_of_the_key(self):
        from dataclasses import replace
        compute_grid_latlon(SAT)
        smaller = replace(SAT, n_rows=64, n_cols=64)
        lat, _ = compute_grid_latlon(smaller)
        assert lat.shape == (64, 64)

    def test_clear_releases_the_memory(self):
        compute_grid_latlon(SAT)
        assert _grid_cache
        clear_grid_cache()
        assert not _grid_cache

    def test_budget_is_respected(self, monkeypatch):
        import stereo_winds.navigation as nav
        from dataclasses import replace
        monkeypatch.setattr(nav, "_GRID_CACHE_BYTES", 8 * 1024 * 1024)
        for i in range(6):
            compute_grid_latlon(replace(SAT, n_rows=256, n_cols=256,
                                        sub_lon_deg=float(i)))
        held = sum(sum(a.nbytes for a in v) for v in _grid_cache.values())
        assert held <= 8 * 1024 * 1024

    def test_zero_budget_disables_caching(self, monkeypatch):
        import stereo_winds.navigation as nav
        monkeypatch.setattr(nav, "_GRID_CACHE_BYTES", 0)
        compute_grid_latlon(SAT)
        assert not _grid_cache
