"""Tests for :mod:`operational.core.watermark`."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from operational.core.watermark import WatermarkStore

T0 = datetime(2026, 8, 1, 0, 0)


class TestRoundTrip:
    """Persistence across store instances."""

    def test_empty_store_returns_none(self, tmp_path: Path) -> None:
        store = WatermarkStore(tmp_path / "wm.json")
        assert store.get("goes18") is None
        assert store.all() == {}

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        first = WatermarkStore(path)
        first.set("goes18", T0)
        first.set("himawari9", T0 + timedelta(minutes=10))

        second = WatermarkStore(path)
        assert second.get("goes18") == T0
        assert second.get("himawari9") == T0 + timedelta(minutes=10)
        assert second.all() == {
            "goes18": T0,
            "himawari9": T0 + timedelta(minutes=10),
        }

    def test_file_is_iso8601_strings(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        WatermarkStore(path).set("gk2a", T0)
        payload = json.loads(path.read_text())
        assert payload == {"gk2a": "2026-08-01T00:00:00"}

    def test_all_returns_a_copy(self, tmp_path: Path) -> None:
        store = WatermarkStore(tmp_path / "wm.json")
        store.set("goes19", T0)
        snapshot = store.all()
        snapshot["goes19"] = T0 + timedelta(days=1)
        assert store.get("goes19") == T0

    def test_creates_missing_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "wm.json"
        store = WatermarkStore(path)
        store.set("goes18", T0)
        assert path.is_file()
        assert WatermarkStore(path).get("goes18") == T0


class TestMonotonicity:
    """The watermark must never move backwards."""

    def test_backwards_set_is_ignored(self, tmp_path: Path) -> None:
        store = WatermarkStore(tmp_path / "wm.json")
        store.set("goes18", T0)
        store.set("goes18", T0 - timedelta(hours=1))
        assert store.get("goes18") == T0

    def test_backwards_set_not_persisted(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        store = WatermarkStore(path)
        store.set("goes18", T0)
        store.set("goes18", T0 - timedelta(hours=1))
        assert WatermarkStore(path).get("goes18") == T0

    def test_backwards_set_logs_debug(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = WatermarkStore(tmp_path / "wm.json")
        store.set("goes18", T0)
        with caplog.at_level(logging.DEBUG, logger="operational.core.watermark"):
            store.set("goes18", T0 - timedelta(hours=1))
        assert any(record.levelno == logging.DEBUG for record in caplog.records)

    def test_forward_set_moves(self, tmp_path: Path) -> None:
        store = WatermarkStore(tmp_path / "wm.json")
        store.set("goes18", T0)
        store.set("goes18", T0 + timedelta(minutes=30))
        assert store.get("goes18") == T0 + timedelta(minutes=30)

    def test_satellites_are_independent(self, tmp_path: Path) -> None:
        store = WatermarkStore(tmp_path / "wm.json")
        store.set("goes18", T0 + timedelta(hours=2))
        store.set("gk2a", T0)
        assert store.get("gk2a") == T0
        assert store.get("goes18") == T0 + timedelta(hours=2)


class TestAdvance:
    """``advance`` reports whether the watermark moved."""

    def test_first_advance_is_true(self, tmp_path: Path) -> None:
        store = WatermarkStore(tmp_path / "wm.json")
        assert store.advance("goes18", T0) is True

    def test_forward_advance_is_true(self, tmp_path: Path) -> None:
        store = WatermarkStore(tmp_path / "wm.json")
        store.advance("goes18", T0)
        assert store.advance("goes18", T0 + timedelta(minutes=10)) is True

    def test_equal_advance_is_false(self, tmp_path: Path) -> None:
        store = WatermarkStore(tmp_path / "wm.json")
        store.advance("goes18", T0)
        assert store.advance("goes18", T0) is False

    def test_backwards_advance_is_false(self, tmp_path: Path) -> None:
        store = WatermarkStore(tmp_path / "wm.json")
        store.advance("goes18", T0)
        assert store.advance("goes18", T0 - timedelta(minutes=1)) is False
        assert store.get("goes18") == T0

    def test_rejects_non_datetime(self, tmp_path: Path) -> None:
        store = WatermarkStore(tmp_path / "wm.json")
        with pytest.raises(TypeError):
            store.advance("goes18", "2026-08-01T00:00:00")  # type: ignore[arg-type]

    def test_aware_datetime_stored_as_naive_utc(self, tmp_path: Path) -> None:
        store = WatermarkStore(tmp_path / "wm.json")
        aware = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
        store.advance("goes18", aware)
        stored = store.get("goes18")
        assert stored == T0
        assert stored is not None and stored.tzinfo is None


class TestCorruptionRecovery:
    """A damaged state file must not take the service down."""

    def test_truncated_json_recovers_as_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        path.write_text("{ not json")
        store = WatermarkStore(path)
        assert store.all() == {}

        store.set("goes18", T0)
        assert store.get("goes18") == T0
        assert json.loads(path.read_text()) == {"goes18": "2026-08-01T00:00:00"}
        assert WatermarkStore(path).get("goes18") == T0

    def test_corrupt_file_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "wm.json"
        path.write_text("{ not json")
        with caplog.at_level(logging.WARNING, logger="operational.core.watermark"):
            WatermarkStore(path)
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_non_object_json_recovers_as_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        path.write_text(json.dumps(["goes18", "2026-08-01T00:00:00"]))
        store = WatermarkStore(path)
        assert store.all() == {}
        store.set("goes18", T0)
        assert WatermarkStore(path).get("goes18") == T0

    def test_empty_file_recovers_as_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        path.write_text("")
        store = WatermarkStore(path)
        assert store.all() == {}
        assert store.advance("gk2a", T0) is True

    def test_bad_timestamp_isolated_to_one_satellite(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        path.write_text(
            json.dumps(
                {
                    "goes18": "not-a-timestamp",
                    "goes19": "2026-08-01T00:00:00",
                }
            )
        )
        store = WatermarkStore(path)
        assert store.get("goes18") is None
        assert store.get("goes19") == T0
        assert store.all() == {"goes19": T0}

        # The unset satellite still works, and the repaired file round-trips.
        assert store.advance("goes18", T0 - timedelta(days=5)) is True
        assert WatermarkStore(path).get("goes18") == T0 - timedelta(days=5)

    def test_unexpected_value_types_are_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        path.write_text(
            json.dumps(
                {
                    "goes18": 1234567890,
                    "goes19": None,
                    "himawari9": {"t": "2026-08-01T00:00:00"},
                    "gk2a": "2026-08-01T00:00:00",
                }
            )
        )
        store = WatermarkStore(path)
        assert store.all() == {"gk2a": T0}

    def test_unreadable_file_recovers_as_empty(self, tmp_path: Path) -> None:
        # A directory in place of the state file makes reads fail with OSError.
        path = tmp_path / "wm.json"
        path.mkdir()
        store = WatermarkStore(path)
        assert store.all() == {}


class TestAtomicWrites:
    """Writes go through a temp file and leave nothing behind."""

    def test_no_tmp_residue(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        store = WatermarkStore(path)
        for i in range(5):
            store.set("goes18", T0 + timedelta(minutes=i))
        store.reset("goes18")

        leftovers = [p.name for p in tmp_path.iterdir() if p.name != path.name]
        assert leftovers == []
        assert not list(tmp_path.glob("*.tmp"))

    def test_tmp_removed_when_replace_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import operational.core.watermark as wm

        path = tmp_path / "wm.json"
        store = WatermarkStore(path)

        def boom(src: str, dst: str) -> None:
            raise OSError("replace failed")

        monkeypatch.setattr(wm.os, "replace", boom)
        with pytest.raises(OSError):
            store.set("goes18", T0)

        assert not list(tmp_path.glob("*.tmp"))
        assert not path.exists()

    def test_existing_file_untouched_on_failed_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import operational.core.watermark as wm

        path = tmp_path / "wm.json"
        store = WatermarkStore(path)
        store.set("goes18", T0)
        before = path.read_text()

        monkeypatch.setattr(
            wm.os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("nope"))
        )
        with pytest.raises(OSError):
            store.set("goes18", T0 + timedelta(hours=1))

        assert path.read_text() == before
        assert not list(tmp_path.glob("*.tmp"))

    def test_memory_rolls_back_on_failed_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import operational.core.watermark as wm

        store = WatermarkStore(tmp_path / "wm.json")
        store.set("goes18", T0)

        monkeypatch.setattr(
            wm.os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("nope"))
        )
        with pytest.raises(OSError):
            store.advance("goes18", T0 + timedelta(hours=1))
        monkeypatch.undo()

        # The caller never saw the watermark move, so the same timestamp must
        # still be emittable on the next poll.
        assert store.get("goes18") == T0
        assert store.advance("goes18", T0 + timedelta(hours=1)) is True

    def test_first_write_rolls_back_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import operational.core.watermark as wm

        store = WatermarkStore(tmp_path / "wm.json")
        monkeypatch.setattr(
            wm.os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("nope"))
        )
        with pytest.raises(OSError):
            store.advance("goes18", T0)
        monkeypatch.undo()

        assert store.get("goes18") is None
        assert store.all() == {}
        assert store.advance("goes18", T0) is True


class TestReset:
    """Forgetting one satellite versus all of them."""

    def test_reset_one_satellite(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        store = WatermarkStore(path)
        store.set("goes18", T0)
        store.set("goes19", T0)

        store.reset("goes18")
        assert store.get("goes18") is None
        assert store.get("goes19") == T0
        assert WatermarkStore(path).all() == {"goes19": T0}

    def test_reset_all(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        store = WatermarkStore(path)
        store.set("goes18", T0)
        store.set("goes19", T0)

        store.reset()
        assert store.all() == {}
        assert WatermarkStore(path).all() == {}

    def test_reset_unknown_satellite_leaves_others_intact(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        store = WatermarkStore(path)
        store.set("goes18", T0)
        store.reset("mtg_i1")
        assert store.all() == {"goes18": T0}
        assert WatermarkStore(path).all() == {"goes18": T0}

    def test_reset_clears_an_unparseable_on_disk_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        path.write_text(json.dumps({"goes18": "not-a-timestamp", "goes19": "2026-08-01T00:00:00"}))
        store = WatermarkStore(path)
        store.reset("goes18")
        assert json.loads(path.read_text()) == {"goes19": "2026-08-01T00:00:00"}

    def test_reset_then_set_moves_backwards_freely(self, tmp_path: Path) -> None:
        store = WatermarkStore(tmp_path / "wm.json")
        store.set("goes18", T0)
        store.reset("goes18")
        assert store.advance("goes18", T0 - timedelta(days=1)) is True

    def test_reset_all_repairs_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        path.write_text("{ not json")
        store = WatermarkStore(path)
        store.reset()
        assert json.loads(path.read_text()) == {}


class TestConcurrency:
    """Threaded writers leave a valid file and a consistent maximum."""

    def test_concurrent_advances_same_satellite(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        store = WatermarkStore(path)
        n_threads = 8
        per_thread = 25
        start = threading.Barrier(n_threads)
        moved: list[bool] = []
        moved_lock = threading.Lock()

        def worker(offset: int) -> None:
            start.wait()
            local: list[bool] = []
            for i in range(per_thread):
                local.append(store.advance("goes18", T0 + timedelta(minutes=offset + i)))
            with moved_lock:
                moved.extend(local)

        threads = [
            threading.Thread(target=worker, args=(i * per_thread,)) for i in range(n_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        highest = T0 + timedelta(minutes=n_threads * per_thread - 1)
        assert store.get("goes18") == highest
        assert any(moved)

        reloaded = WatermarkStore(path)
        assert reloaded.get("goes18") == highest
        assert not list(tmp_path.glob("*.tmp"))

    def test_concurrent_advances_distinct_satellites(self, tmp_path: Path) -> None:
        path = tmp_path / "wm.json"
        store = WatermarkStore(path)
        sats = [f"sat{i}" for i in range(6)]
        start = threading.Barrier(len(sats))

        def worker(sat_id: str, index: int) -> None:
            start.wait()
            for i in range(20):
                store.advance(sat_id, T0 + timedelta(minutes=i, hours=index))

        threads = [threading.Thread(target=worker, args=(sat, i)) for i, sat in enumerate(sats)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        expected = {sat: T0 + timedelta(minutes=19, hours=i) for i, sat in enumerate(sats)}
        assert store.all() == expected
        assert WatermarkStore(path).all() == expected
