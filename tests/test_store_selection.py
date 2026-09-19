"""Tests for choosing an icechunk store by what it holds, not its name."""

import datetime as dt

import numpy as np
import pytest

from stereo_winds.readers import _geos_store
from stereo_winds.readers._geos_store import clear_store_cache
from stereo_winds.readers._satpy_s3 import SceneNotInStore
from stereo_winds.readers.himawari import Himawari
from stereo_winds.readers.mtg import MTG

T2026 = dt.datetime(2026, 8, 1)
T2024 = dt.datetime(2024, 12, 1)


def _contents(bands, first, last):
    return (frozenset(bands), np.datetime64(first, "ns"), np.datetime64(last, "ns"))


@pytest.fixture
def stub(monkeypatch):
    """Stand in for the bucket: store name -> (bands, coverage)."""
    listing: list[str] = []
    contents: dict[str, tuple | None] = {}

    monkeypatch.setattr(_geos_store, "_BUCKET_LISTINGS", {})
    monkeypatch.setattr(_geos_store, "_STORE_CONTENTS", contents)
    monkeypatch.setattr(_geos_store.GeoStoreReader, "_bucket_stores",
                        classmethod(lambda cls: listing))
    return listing, contents


class TestCandidateOrder:
    def test_named_store_is_tried_first(self, stub):
        listing, _ = stub
        listing[:] = ["himawari_2000m.icechunk", "himawari_2000m_test.icechunk",
                      "himawari_500m.icechunk"]
        order = Himawari(bands=["C14"])._candidate_stores("B14")
        assert order[0] == "geo/himawari_2000m.icechunk"

    def test_same_tier_before_other_tiers(self, stub):
        listing, _ = stub
        listing[:] = ["himawari_500m.icechunk", "himawari_2000m.icechunk",
                      "himawari_2000m_test.icechunk"]
        order = Himawari(bands=["C14"])._candidate_stores("B14")
        assert order.index("geo/himawari_2000m_test.icechunk") < \
            order.index("geo/himawari_500m.icechunk")

    def test_other_instruments_are_not_candidates(self, stub):
        listing, _ = stub
        listing[:] = ["himawari_2000m.icechunk", "gk2a_2000m.icechunk",
                      "mtg_2000m.icechunk"]
        order = Himawari(bands=["C14"])._candidate_stores("B14")
        assert all("himawari" in p for p in order)

    def test_discovery_off_keeps_only_the_named_store(self, stub, monkeypatch):
        listing, _ = stub
        listing[:] = ["himawari_2000m_test.icechunk"]
        monkeypatch.setattr(Himawari, "store_discovery_prefix", "")
        assert Himawari(bands=["C14"])._candidate_stores("B14") == [
            "geo/himawari_2000m.icechunk"]


class TestSelection:
    def test_skips_a_store_whose_coverage_ended(self, stub):
        """Regression: the named store stopped in 2025 and was used anyway."""
        listing, contents = stub
        listing[:] = ["himawari_2000m.icechunk", "himawari_2000m_test.icechunk"]
        contents["geo/himawari_2000m.icechunk"] = _contents(
            ["B14"], "2015-07-07", "2025-02-01")
        contents["geo/himawari_2000m_test.icechunk"] = _contents(
            ["B14"], "2026-01-01", "2026-09-05")
        chosen = Himawari(bands=["C14"])._select_store("B14", T2026)
        assert chosen == "geo/himawari_2000m_test.icechunk"

    def test_prefers_the_named_store_when_it_covers(self, stub):
        listing, contents = stub
        listing[:] = ["himawari_2000m.icechunk", "himawari_2000m_test.icechunk"]
        contents["geo/himawari_2000m.icechunk"] = _contents(
            ["B14"], "2015-07-07", "2025-02-01")
        contents["geo/himawari_2000m_test.icechunk"] = _contents(
            ["B14"], "2026-01-01", "2026-09-05")
        chosen = Himawari(bands=["C14"])._select_store("B14", T2024)
        assert chosen == "geo/himawari_2000m.icechunk"

    def test_skips_a_store_that_lacks_the_band(self, stub):
        """Regression: vis_04 was tiered to a store holding only vis_06/nir_22."""
        listing, contents = stub
        listing[:] = ["mtg_500m.icechunk", "mtg_1000m.icechunk"]
        contents["geo/mtg_500m.icechunk"] = _contents(
            ["vis_06", "nir_22"], "2024-12-01", "2026-08-23")
        contents["geo/mtg_1000m.icechunk"] = _contents(
            ["vis_04", "vis_06"], "2024-09-24", "2026-06-15")
        chosen = MTG(bands=["vis_04"])._select_store("vis_04",
                                                    dt.datetime(2026, 1, 1))
        assert chosen == "geo/mtg_1000m.icechunk"

    def test_finds_a_band_in_a_differently_named_store(self, stub):
        """ir_105 lives in mtg_highres_1000m for recent dates."""
        listing, contents = stub
        listing[:] = ["mtg_2000m.icechunk", "mtg_highres_1000m.icechunk"]
        contents["geo/mtg_2000m.icechunk"] = _contents(
            ["ir_105", "wv_63"], "2024-09-24", "2026-07-29")
        contents["geo/mtg_highres_1000m.icechunk"] = _contents(
            ["ir_105", "ir_38"], "2024-09-24", "2026-08-31")
        chosen = MTG(bands=["C13"])._select_store("ir_105", T2026)
        assert chosen == "geo/mtg_highres_1000m.icechunk"

    def test_raises_when_nothing_covers_it(self, stub):
        listing, contents = stub
        listing[:] = ["mtg_2000m.icechunk"]
        contents["geo/mtg_2000m.icechunk"] = _contents(
            ["wv_63"], "2024-09-24", "2026-07-29")
        with pytest.raises(SceneNotInStore, match="no store carries"):
            MTG(bands=["C08"])._select_store("wv_63", dt.datetime(2027, 1, 1))

    def test_unusable_stores_are_skipped(self, stub):
        listing, contents = stub
        listing[:] = ["himawari_2000m.icechunk", "himawari_2000m_test.icechunk"]
        contents["geo/himawari_2000m.icechunk"] = None      # failed to open
        contents["geo/himawari_2000m_test.icechunk"] = _contents(
            ["B14"], "2026-01-01", "2026-09-05")
        assert Himawari(bands=["C14"])._select_store("B14", T2026) == \
            "geo/himawari_2000m_test.icechunk"

    def test_tolerance_covers_the_edges(self, stub):
        """A scan one interval past the last one still counts as covered."""
        listing, contents = stub
        listing[:] = ["himawari_2000m.icechunk"]
        contents["geo/himawari_2000m.icechunk"] = _contents(
            ["B14"], "2026-01-01", "2026-01-02T00:00")
        reader = Himawari(bands=["C14"])
        assert reader._select_store("B14", dt.datetime(2026, 1, 2, 0, 5))
        with pytest.raises(SceneNotInStore):
            reader._select_store("B14", dt.datetime(2026, 1, 2, 1, 0))


def test_cache_clearing_resets_discovery():
    clear_store_cache()
    assert not _geos_store._STORE_CONTENTS
    assert not _geos_store._BUCKET_LISTINGS
