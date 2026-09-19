"""Offline tests for :mod:`operational.core.availability`.

``satellite_available_times`` is monkeypatched on the ring adapter, so no
S3 listing or icechunk store is touched; ``availability_band`` and
``scan_interval`` are exercised for real (they are pure band-table
lookups).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from operational.adapters import ring
from operational.core.availability import (
    SatelliteAvailability,
    available_timestamps,
    cadence_grid,
    new_timestamps,
    probe_satellite,
    ready_satellites,
)

T0 = datetime(2026, 8, 1, 0, 0)


def _t64(*times: datetime) -> np.ndarray:
    """Sorted ``datetime64[ns]`` array from datetimes."""
    return np.array(sorted(times), dtype="datetime64[ns]")


def _every(minutes: int, n: int, skip: tuple[int, ...] = (),
           offset_seconds: int = 0, base: datetime = T0) -> np.ndarray:
    """``n`` scans spaced ``minutes`` apart, minus the ``skip`` indices."""
    return _t64(*[
        base + timedelta(minutes=minutes * i, seconds=offset_seconds)
        for i in range(n) if i not in skip
    ])


class _ScanTable(dict):
    """Scan-time table that also records every lookup it served."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple] = []


@pytest.fixture
def stub_availability(monkeypatch):
    """Replace the upstream scan-time lookup with an in-memory table.

    Mutate the returned dict as ``table[sat_id] = <datetime64 array>``.
    Every call is recorded in ``table.calls`` as
    ``(sat_id, band, start, end, product, include_s3_fallback)``.
    """
    table = _ScanTable()

    def fake(sat_id, band, start, end, product="ABI-L1b-RadF",
             include_s3_fallback=True):
        table.calls.append(
            (sat_id, band, start, end, product, include_s3_fallback))
        return table.get(sat_id, np.array([], dtype="datetime64[ns]"))

    monkeypatch.setattr(ring, "satellite_available_times", fake)
    return table


class TestProbeSatellite:
    def test_reports_band_and_scan_interval(self, stub_availability):
        stub_availability["goes19"] = _every(10, 6)
        avail = probe_satellite("goes19", T0, T0 + timedelta(hours=1),
                                ["C14"], ["C14"])
        assert isinstance(avail, SatelliteAvailability)
        assert avail.has_band and avail.band == "C14"
        assert avail.scan_interval_minutes == 10
        assert avail.dt == timedelta(minutes=10)
        assert avail.scan_times.size == 6

    def test_window_is_padded_for_neighbour_frames(self, stub_availability):
        stub_availability["goes19"] = _every(10, 6)
        probe_satellite("goes19", T0, T0 + timedelta(hours=1),
                        ["C14"], ["C14"], tolerance_minutes=5.0)
        _, _, start, end, _, _ = stub_availability.calls[0]
        # dt (10 min) + tolerance (5 min) either side.
        assert start == T0 - timedelta(minutes=15)
        assert end == T0 + timedelta(hours=1, minutes=15)

    def test_fifteen_minute_satellite_pads_further(self, stub_availability):
        stub_availability["msg-iodc"] = _every(15, 6)
        avail = probe_satellite("msg-iodc", T0, T0, ["C14"], ["C14"])
        assert avail.scan_interval_minutes == 15
        _, _, start, end, _, _ = stub_availability.calls[0]
        assert start == T0 - timedelta(minutes=20)
        assert end == T0 + timedelta(minutes=20)

    def test_unsorted_scan_times_are_sorted(self, stub_availability):
        shuffled = np.array(
            [np.datetime64(T0 + timedelta(minutes=10 * i), "ns")
             for i in (3, 0, 2, 1)], dtype="datetime64[ns]",
        )
        stub_availability["goes19"] = shuffled
        avail = probe_satellite("goes19", T0, T0 + timedelta(minutes=30),
                                ["C14"], ["C14"])
        assert np.all(np.diff(avail.scan_times) > np.timedelta64(0, "ns"))
        assert avail.can_deliver(T0 + timedelta(minutes=10))

    def test_hand_built_instance_sorts_its_scan_times(self):
        avail = SatelliteAvailability(
            sat_id="goes19", band="C14",
            scan_times=np.array(
                [np.datetime64(T0 + timedelta(minutes=10 * i), "ns")
                 for i in (2, 0, 1)], dtype="datetime64[ns]"),
            scan_interval_minutes=10, tolerance_minutes=5.0,
        )
        assert np.all(np.diff(avail.scan_times) > np.timedelta64(0, "ns"))
        assert avail.can_deliver(T0 + timedelta(minutes=10))


class TestDistinctFrames:
    """The three frames must be three different scans, not the same one twice."""

    def _avail(self, scans, tolerance_minutes):
        return SatelliteAvailability(
            sat_id="goes19", band="C14", scan_times=_t64(*scans),
            scan_interval_minutes=10, tolerance_minutes=tolerance_minutes,
        )

    def test_one_scan_cannot_serve_two_frames(self):
        # No 00:10 scan.  A slack tolerance would let 00:00 answer both
        # the t-dt and the t frame, which is a duplicated input, not a
        # retrievable triplet.
        avail = self._avail(
            [T0, T0 + timedelta(minutes=20)], tolerance_minutes=11.0)
        assert avail.can_deliver(T0 + timedelta(minutes=10)) is False

    def test_three_real_scans_are_accepted(self):
        avail = self._avail(
            [T0, T0 + timedelta(minutes=10), T0 + timedelta(minutes=20)],
            tolerance_minutes=11.0)
        assert avail.can_deliver(T0 + timedelta(minutes=10)) is True


class TestAvailableTimestamps:
    def test_full_triplet_is_included(self, stub_availability):
        stub_availability["goes19"] = _every(10, 5)
        got = available_timestamps(
            "goes19", T0, T0 + timedelta(minutes=40), ["C14"], ["C14"])
        # Only the interior scans have both neighbours.
        assert got == [T0 + timedelta(minutes=10),
                       T0 + timedelta(minutes=20),
                       T0 + timedelta(minutes=30)]

    def test_missing_neighbour_frame_is_excluded(self, stub_availability):
        # No 00:30 scan: 00:20 loses t+dt and 00:40 loses t-dt.
        stub_availability["goes19"] = _every(10, 6, skip=(3,))
        got = available_timestamps(
            "goes19", T0, T0 + timedelta(minutes=50), ["C14"], ["C14"])
        assert got == [T0 + timedelta(minutes=10)]

    def test_candidates_are_limited_to_the_window(self, stub_availability):
        stub_availability["goes19"] = _every(10, 10)
        got = available_timestamps(
            "goes19", T0 + timedelta(minutes=30), T0 + timedelta(minutes=50),
            ["C14"], ["C14"])
        assert got == [T0 + timedelta(minutes=30),
                       T0 + timedelta(minutes=40),
                       T0 + timedelta(minutes=50)]

    def test_no_scans_gives_nothing(self, stub_availability):
        stub_availability["goes19"] = np.array([], dtype="datetime64[ns]")
        assert available_timestamps(
            "goes19", T0, T0 + timedelta(hours=1), ["C14"], ["C14"]) == []

    def test_reversed_window_is_empty(self, stub_availability):
        stub_availability["goes19"] = _every(10, 6)
        assert available_timestamps(
            "goes19", T0 + timedelta(hours=1), T0, ["C14"], ["C14"]) == []
        assert stub_availability.calls == []

    def test_tz_aware_bounds_are_accepted(self, stub_availability):
        stub_availability["goes19"] = _every(10, 5)
        got = available_timestamps(
            "goes19",
            T0.replace(tzinfo=timezone.utc),
            (T0 + timedelta(minutes=40)).replace(tzinfo=timezone.utc),
            ["C14"], ["C14"])
        assert got == [T0 + timedelta(minutes=10),
                       T0 + timedelta(minutes=20),
                       T0 + timedelta(minutes=30)]


class TestTolerance:
    def test_offset_scan_schedule_still_matches_the_grid(self, stub_availability):
        """GK-2A stamps at HH:09:35, which must count for the HH:10 slot."""
        stub_availability["gk2a"] = _every(10, 7, offset_seconds=-25)
        got = new_timestamps(
            "gk2a", None, T0 + timedelta(minutes=50), ["C14"], ["C14"],
            cadence_minutes=10, lookback_hours=1.0)
        assert T0 + timedelta(minutes=20) in got
        assert T0 + timedelta(minutes=30) in got

    def test_offset_beyond_tolerance_is_rejected(self, stub_availability):
        stub_availability["gk2a"] = _every(10, 7, offset_seconds=-25)
        got = new_timestamps(
            "gk2a", None, T0 + timedelta(minutes=50), ["C14"], ["C14"],
            cadence_minutes=10, tolerance_minutes=0.25, lookback_hours=1.0)
        assert got == []

    def test_tolerance_boundary(self, stub_availability):
        # Scans 4 minutes off the grid: inside a 5 min tolerance, outside 2.
        stub_availability["goes19"] = _every(10, 7, offset_seconds=240)
        slot = T0 + timedelta(minutes=20)
        kwargs = dict(cadence_minutes=10, lookback_hours=1.0)
        assert slot in new_timestamps(
            "goes19", None, T0 + timedelta(minutes=50), ["C14"], ["C14"],
            tolerance_minutes=5.0, **kwargs)
        assert new_timestamps(
            "goes19", None, T0 + timedelta(minutes=50), ["C14"], ["C14"],
            tolerance_minutes=2.0, **kwargs) == []

    def test_tolerance_is_inclusive_at_the_edge(self):
        """A scan exactly ``tolerance`` away still counts."""
        slot = T0 + timedelta(minutes=20)
        # Frames 00:10/00:20/00:30 match 00:05/00:20/00:35 at exactly
        # 5/0/5 minutes, and each nearest scan is unambiguous (the runner
        # up is 10 minutes away).
        scans = [T0 + timedelta(minutes=5), slot, T0 + timedelta(minutes=35)]
        assert SatelliteAvailability(
            "goes19", "C14", _t64(*scans), 10, 5.0).can_deliver(slot) is True
        assert SatelliteAvailability(
            "goes19", "C14", _t64(*scans), 10, 4.99).can_deliver(slot) is False

    def test_equidistant_scans_are_ambiguous(self):
        """A frame tied between two scans cannot be resolved here.

        This module and the readers each snap independently, so a tie is
        a frame whose identity is unpredictable — and two frames breaking
        it in opposite directions would load the same scan twice.
        """
        # Every frame of the HH:20 slot sits 5 min from two scans.
        avail = SatelliteAvailability(
            "goes19", "C14",
            _t64(*[T0 + timedelta(minutes=10 * i, seconds=300)
                   for i in range(7)]), 10, 5.0)
        assert avail.can_deliver(T0 + timedelta(minutes=20)) is False

    def test_offset_cadence_over_a_10_minute_satellite_is_rejected(
            self, stub_availability):
        """A 15-min cadence on a 10-min satellite half-offsets every slot.

        HH:15 would otherwise be declared ready off the HH:00/HH:10/HH:20
        scans while the loader asks for HH:05/HH:15/HH:25 and can snap two
        of them onto one scan — a duplicated frame, i.e. a near-zero
        displacement, reported silently as a retrieval.
        """
        stub_availability["goes19"] = _every(10, 13)
        got = new_timestamps(
            "goes19", None, T0 + timedelta(minutes=120), ["C14"], ["C14"],
            cadence_minutes=15, tolerance_minutes=5.0, lookback_hours=2.0)
        assert got  # the on-grid slots survive
        assert all(t.minute % 10 == 0 for t in got)
        assert T0 + timedelta(minutes=15) not in got


class TestNewTimestamps:
    def test_since_filters_already_processed_slots(self, stub_availability):
        stub_availability["goes19"] = _every(10, 13)
        got = new_timestamps(
            "goes19", T0 + timedelta(minutes=40), T0 + timedelta(minutes=110),
            ["C14"], ["C14"], cadence_minutes=10)
        assert got == [T0 + timedelta(minutes=50),
                       T0 + timedelta(minutes=60),
                       T0 + timedelta(minutes=70),
                       T0 + timedelta(minutes=80),
                       T0 + timedelta(minutes=90),
                       T0 + timedelta(minutes=100),
                       T0 + timedelta(minutes=110)]

    def test_since_is_strict(self, stub_availability):
        stub_availability["goes19"] = _every(10, 8)
        got = new_timestamps(
            "goes19", T0 + timedelta(minutes=30), T0 + timedelta(minutes=50),
            ["C14"], ["C14"], cadence_minutes=10)
        assert T0 + timedelta(minutes=30) not in got
        assert got[0] == T0 + timedelta(minutes=40)

    def test_since_none_uses_the_lookback_window(self, stub_availability):
        stub_availability["goes19"] = _every(10, 13)
        got = new_timestamps(
            "goes19", None, T0 + timedelta(minutes=120), ["C14"], ["C14"],
            cadence_minutes=60, lookback_hours=2.0)
        assert got == [T0 + timedelta(minutes=60)]

    def test_since_at_or_after_until_is_empty(self, stub_availability):
        stub_availability["goes19"] = _every(10, 13)
        assert new_timestamps(
            "goes19", T0 + timedelta(hours=2), T0 + timedelta(hours=1),
            ["C14"], ["C14"], cadence_minutes=10) == []
        assert new_timestamps(
            "goes19", T0, T0, ["C14"], ["C14"], cadence_minutes=10) == []
        assert stub_availability.calls == []

    def test_stale_cursor_is_capped_by_the_lookback(self, stub_availability, caplog):
        """A weeks-old cursor must not turn one poll into days of listing."""
        until = T0 + timedelta(days=7)
        stub_availability["goes19"] = _t64(*[
            until - timedelta(minutes=10 * i) for i in range(20)])
        with caplog.at_level("WARNING"):
            got = new_timestamps(
                "goes19", T0, until, ["C14"], ["C14"],
                cadence_minutes=60, lookback_hours=2.0)
        assert got and all(t >= until - timedelta(hours=2) for t in got)
        _, _, start, _, _, _ = stub_availability.calls[0]
        assert start >= until - timedelta(hours=3)
        assert "catch-up limit" in caplog.text

    def test_recent_cursor_is_not_capped(self, stub_availability):
        stub_availability["goes19"] = _every(10, 25)
        got = new_timestamps(
            "goes19", T0, T0 + timedelta(hours=3), ["C14"], ["C14"],
            cadence_minutes=60, lookback_hours=24.0)
        assert got == [T0 + timedelta(hours=1), T0 + timedelta(hours=2),
                       T0 + timedelta(hours=3)]

    def test_hourly_cadence_lands_on_the_hour(self, stub_availability):
        base = datetime(2026, 8, 1, 3, 17)
        stub_availability["goes19"] = _t64(*[
            base + timedelta(minutes=10 * i) for i in range(30)])
        got = new_timestamps(
            "goes19", base, base + timedelta(hours=3), ["C14"], ["C14"],
            cadence_minutes=60)
        assert got == [datetime(2026, 8, 1, 4, 0),
                       datetime(2026, 8, 1, 5, 0),
                       datetime(2026, 8, 1, 6, 0)]

    def test_fifteen_minute_satellite_needs_a_15_minute_triplet(
            self, stub_availability):
        """SEVIRI's dt is 15 min, so a 10-min triplet is not enough."""
        ten = _every(10, 13)
        fifteen = _every(15, 9)
        window_end = T0 + timedelta(minutes=120)
        slot = T0 + timedelta(minutes=60)

        stub_availability["msg-iodc"] = ten
        assert new_timestamps(
            "msg-iodc", None, window_end, ["C14"], ["C14"],
            cadence_minutes=15, tolerance_minutes=2.0, lookback_hours=2.0) == []

        stub_availability["msg-iodc"] = fifteen
        got = new_timestamps(
            "msg-iodc", None, window_end, ["C14"], ["C14"],
            cadence_minutes=15, tolerance_minutes=2.0, lookback_hours=2.0)
        assert slot in got

        # The same 10-minute scan table is fine for a 10-minute satellite.
        stub_availability["goes19"] = ten
        assert slot in new_timestamps(
            "goes19", None, window_end, ["C14"], ["C14"],
            cadence_minutes=15, tolerance_minutes=2.0, lookback_hours=2.0)

    def test_no_scans_gives_nothing(self, stub_availability):
        assert new_timestamps(
            "goes19", None, T0 + timedelta(hours=1), ["C14"], ["C14"],
            cadence_minutes=10, lookback_hours=1.0) == []


class TestReadySatellites:
    def test_reports_one_entry_per_satellite(self, stub_availability):
        stub_availability["goes19"] = _every(10, 7)
        stub_availability["gk2a"] = _every(10, 7, offset_seconds=-25)
        stub_availability["himawari9"] = np.array([], dtype="datetime64[ns]")
        ready = ready_satellites(
            T0 + timedelta(minutes=20), ["goes19", "gk2a", "himawari9"],
            ["C14"], ["C14"])
        assert ready == {"goes19": True, "gk2a": True, "himawari9": False}
        assert list(ready) == ["goes19", "gk2a", "himawari9"]

    def test_missing_neighbour_makes_a_satellite_unready(self, stub_availability):
        # goes18 lacks the 00:30 scan, so the 00:20 slot has no t+dt frame.
        stub_availability["goes18"] = _every(10, 7, skip=(3,))
        stub_availability["goes19"] = _every(10, 7)
        ready = ready_satellites(
            T0 + timedelta(minutes=20), ["goes18", "goes19"], ["C14"], ["C14"])
        assert ready == {"goes18": False, "goes19": True}

    def test_no_satellites_gives_empty_dict(self, stub_availability):
        assert ready_satellites(T0, [], ["C14"], ["C14"]) == {}


class TestSatelliteWithoutRequestedBand:
    """MTG/FCI carries no ABI C14 — that is 'nothing available', not an error."""

    def test_probe_returns_band_none_without_a_lookup(self, stub_availability):
        avail = probe_satellite("mtg-i1", T0, T0 + timedelta(hours=1),
                                ["C14"], ["C14"])
        assert avail.band is None
        assert avail.has_band is False
        assert avail.scan_times.size == 0
        assert avail.deliverable([T0]) == []
        assert avail.can_deliver(T0) is False
        # No point asking the store for times on a band it cannot carry.
        assert stub_availability.calls == []

    def test_available_timestamps_is_empty(self, stub_availability):
        assert available_timestamps(
            "mtg-i1", T0, T0 + timedelta(hours=1), ["C14"], ["C14"]) == []

    def test_new_timestamps_is_empty(self, stub_availability):
        assert new_timestamps(
            "mtg-i1", None, T0 + timedelta(hours=1), ["C14"], ["C14"],
            cadence_minutes=10, lookback_hours=1.0) == []

    def test_ready_satellites_marks_it_false(self, stub_availability):
        stub_availability["goes19"] = _every(10, 7)
        ready = ready_satellites(
            T0 + timedelta(minutes=20), ["goes19", "mtg-i1"], ["C14"], ["C14"])
        assert ready == {"goes19": True, "mtg-i1": False}

    def test_empty_band_lists_are_treated_as_nothing_available(
            self, stub_availability):
        avail = probe_satellite("goes19", T0, T0, [], [])
        assert avail.band is None
        assert available_timestamps(
            "goes19", T0, T0 + timedelta(hours=1), [], []) == []

    def test_a_band_the_satellite_lacks_falls_back(self, stub_availability):
        """MTG has no C14 but does carry a C10 equivalent."""
        stub_availability["mtg-i1"] = _every(10, 7)
        avail = probe_satellite("mtg-i1", T0, T0, ["C14", "C10"], ["C14"])
        assert avail.band == "C10"
        assert stub_availability.calls[0][1] == "C10"


class TestCadenceGrid:
    def test_slots_are_anchored_to_the_epoch(self):
        got = cadence_grid(datetime(2026, 8, 1, 0, 5),
                           datetime(2026, 8, 1, 2, 30), 60)
        assert got == [datetime(2026, 8, 1, 1, 0), datetime(2026, 8, 1, 2, 0)]

    def test_bounds_are_inclusive(self):
        got = cadence_grid(datetime(2026, 8, 1, 1, 0),
                           datetime(2026, 8, 1, 2, 0), 60)
        assert got == [datetime(2026, 8, 1, 1, 0), datetime(2026, 8, 1, 2, 0)]

    def test_reversed_window_is_empty(self):
        assert cadence_grid(datetime(2026, 8, 1, 2, 0),
                            datetime(2026, 8, 1, 1, 0), 60) == []

    def test_ten_minute_grid(self):
        got = cadence_grid(T0, T0 + timedelta(minutes=25), 10)
        assert got == [T0, T0 + timedelta(minutes=10),
                       T0 + timedelta(minutes=20)]

    def test_non_positive_cadence_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            cadence_grid(T0, T0 + timedelta(hours=1), 0)
