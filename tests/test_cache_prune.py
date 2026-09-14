"""Tests for age-based pruning of the local L1b download cache.

The cache holds two layouts — slot directories for AHI/AMI, flat
timestamped netCDFs for ABI — and grows without bound across a long run
unless old scans are dropped.
"""

import datetime as dt

import pytest

from stereo_winds.readers._cache import (
    DEFAULT_RETENTION,
    default_retention,
    entry_time,
    prune_cache,
)
from stereo_winds.readers.gk2a import GK2A
from stereo_winds.readers.goes import GOES
from stereo_winds.readers.himawari import Himawari

NOW = dt.datetime(2026, 8, 1, 12, 0)


def _slot_dir(root, stamp, n_files=2, size=16):
    d = root / stamp
    d.mkdir(parents=True)
    for i in range(n_files):
        (d / f"HS_H09_{stamp}_B14_FLDK_R20_S{i:02d}10.DAT.bz2").write_bytes(
            b"x" * size)
    return d


def _abi_file(root, when, size=16):
    root.mkdir(parents=True, exist_ok=True)
    name = (f"OR_ABI-L1b-RadF-M6C14_G19_s{when:%Y%j%H%M%S}0"
            f"_e{when:%Y%j%H%M%S}0_c{when:%Y%j%H%M%S}0.nc")
    path = root / name
    path.write_bytes(b"x" * size)
    return path


class TestEntryTime:
    def test_slot_directory(self, tmp_path):
        d = _slot_dir(tmp_path, "20260801_1150")
        assert entry_time(d) == dt.datetime(2026, 8, 1, 11, 50)

    def test_abi_filename(self, tmp_path):
        f = _abi_file(tmp_path, dt.datetime(2026, 8, 1, 11, 50, 20))
        assert entry_time(f) == dt.datetime(2026, 8, 1, 11, 50, 20)

    def test_unrecognised_entries_are_ignored(self, tmp_path):
        (tmp_path / "satpy-scratch-abc123").mkdir()
        (tmp_path / "notes.txt").write_text("hi")
        (tmp_path / "partial.nc.part").write_text("hi")
        for name in ("satpy-scratch-abc123", "notes.txt", "partial.nc.part"):
            assert entry_time(tmp_path / name) is None


class TestPruneCache:
    def test_removes_only_what_is_too_old(self, tmp_path):
        old = _slot_dir(tmp_path, "20260801_1050")     # 70 min before
        edge = _slot_dir(tmp_path, "20260801_1100")    # exactly 60 min
        keep = _slot_dir(tmp_path, "20260801_1150")    # 10 min before
        ahead = _slot_dir(tmp_path, "20260801_1210")   # after NOW

        removed, freed = prune_cache(tmp_path, NOW - DEFAULT_RETENTION)

        assert removed == 1 and freed == 32
        assert not old.exists()
        assert edge.exists() and keep.exists() and ahead.exists()

    def test_prunes_flat_abi_files(self, tmp_path):
        old = _abi_file(tmp_path, dt.datetime(2026, 8, 1, 10, 50, 20))
        keep = _abi_file(tmp_path, dt.datetime(2026, 8, 1, 11, 50, 20))
        removed, _ = prune_cache(tmp_path, NOW - DEFAULT_RETENTION)
        assert removed == 1
        assert not old.exists() and keep.exists()

    def test_in_flight_scratch_is_never_touched(self, tmp_path):
        """A satpy scratch dir may be in use by the load happening now."""
        scratch = tmp_path / "satpy-scratch-inuse"
        scratch.mkdir()
        (scratch / "segment").write_bytes(b"x")
        _slot_dir(tmp_path, "20260801_1000")
        prune_cache(tmp_path, NOW - DEFAULT_RETENTION)
        assert scratch.exists() and (scratch / "segment").exists()

    def test_partial_downloads_are_left_alone(self, tmp_path):
        part = tmp_path / "OR_ABI-L1b-RadF-M6C14_G19.nc.part"
        part.write_bytes(b"x")
        prune_cache(tmp_path, NOW)
        assert part.exists()

    def test_missing_root_is_not_an_error(self, tmp_path):
        assert prune_cache(tmp_path / "nope", NOW) == (0, 0)

    def test_reports_bytes_freed(self, tmp_path):
        _slot_dir(tmp_path, "20260801_0900", n_files=3, size=100)
        removed, freed = prune_cache(tmp_path, NOW - DEFAULT_RETENTION)
        assert (removed, freed) == (1, 300)


class TestRetentionConfig:
    def test_default_is_one_hour(self, monkeypatch):
        monkeypatch.delenv("STEREO_WINDS_L1B_RETENTION_MIN", raising=False)
        assert default_retention() == dt.timedelta(hours=1)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("STEREO_WINDS_L1B_RETENTION_MIN", "30")
        assert default_retention() == dt.timedelta(minutes=30)

    def test_non_positive_disables_pruning(self, monkeypatch):
        monkeypatch.setenv("STEREO_WINDS_L1B_RETENTION_MIN", "0")
        assert default_retention() is None

    def test_garbage_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("STEREO_WINDS_L1B_RETENTION_MIN", "soon")
        assert default_retention() == DEFAULT_RETENTION


class TestReaderIntegration:
    """Every reader that downloads should prune its own subtree."""

    @pytest.mark.parametrize("make, sat", [
        (lambda p: Himawari(satellite="himawari9", bands=["C14"],
                            cache_dir=str(p)), "himawari9"),
        (lambda p: GK2A(bands=["C14"], cache_dir=str(p)), "gk2a"),
        (lambda p: GOES(satellite="goes19", bands=["C14"],
                        cache_dir=str(p)), "goes19"),
    ])
    def test_prunes_only_its_own_satellite(self, tmp_path, make, sat):
        mine = tmp_path / sat
        theirs = tmp_path / "someone-else"
        _slot_dir(mine, "20260801_1000")
        _slot_dir(theirs, "20260801_1000")

        make(tmp_path).prune_download_cache(NOW)

        assert not (mine / "20260801_1000").exists()
        assert (theirs / "20260801_1000").exists()

    def test_retention_none_disables_it(self, tmp_path):
        mine = tmp_path / "himawari9"
        _slot_dir(mine, "20260801_1000")
        reader = Himawari(satellite="himawari9", bands=["C14"],
                          cache_dir=str(tmp_path), cache_retention=None)
        reader.prune_download_cache(NOW)
        assert (mine / "20260801_1000").exists()

    def test_recent_scans_survive_so_triplets_still_hit(self, tmp_path):
        """t-15 must stay cached: the next timestamp reads it as t0."""
        mine = tmp_path / "himawari9"
        for stamp in ("20260801_1145", "20260801_1150", "20260801_1200"):
            _slot_dir(mine, stamp)
        Himawari(satellite="himawari9", bands=["C14"],
                 cache_dir=str(tmp_path)).prune_download_cache(NOW)
        assert len(list(mine.iterdir())) == 3
