"""Tests for the canonical operational partition-key definition."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from dagster import TimeWindowPartitionsDefinition

from operational.core.partitions import (
    MINUTES_PER_DAY,
    PARTITION_KEY_FORMAT,
    align_to_cadence,
    build_partitions_def,
    cron_for_cadence,
    is_on_cadence,
    key_for,
    keys_between,
    time_for,
    validate_cadence,
    validate_on_cadence,
    window_for,
)


class TestKeyRoundTrip:
    """``key_for`` and ``time_for`` must be exact inverses."""

    @pytest.mark.parametrize(
        ("t", "expected"),
        [
            (datetime(2026, 8, 1, 0, 0), "2026-08-01-00:00"),
            (datetime(2026, 8, 1, 6, 0), "2026-08-01-06:00"),
            (datetime(2026, 12, 31, 23, 50), "2026-12-31-23:50"),
            (datetime(2024, 2, 29, 12, 30), "2024-02-29-12:30"),
        ],
    )
    def test_key_for_renders_expected_string(self, t, expected):
        assert key_for(t) == expected

    @pytest.mark.parametrize("cadence", [10, 15, 30, 60, 180, 360, 720, 1440])
    def test_round_trip_over_a_full_day(self, cadence):
        t = datetime(2026, 8, 1)
        step = timedelta(minutes=cadence)
        for _ in range(MINUTES_PER_DAY // cadence):
            assert time_for(key_for(t)) == t
            t += step

    def test_key_round_trip_from_key_side(self):
        key = "2026-08-01-06:00"
        assert key_for(time_for(key)) == key

    def test_returned_time_is_naive_utc(self):
        assert time_for("2026-08-01-06:00").tzinfo is None

    def test_aware_input_is_converted_to_utc(self):
        aware = datetime(2026, 8, 1, 6, 0, tzinfo=timezone(timedelta(hours=2)))
        assert key_for(aware) == "2026-08-01-04:00"

    def test_naive_input_is_treated_as_utc(self):
        naive = datetime(2026, 8, 1, 6, 0)
        aware = naive.replace(tzinfo=timezone.utc)
        assert key_for(naive) == key_for(aware)

    def test_sub_minute_timestamp_is_rejected(self):
        with pytest.raises(ValueError, match="minute resolution"):
            key_for(datetime(2026, 8, 1, 6, 0, 30))
        with pytest.raises(ValueError, match="minute resolution"):
            key_for(datetime(2026, 8, 1, 6, 0, 0, 1))

    def test_non_datetime_is_rejected(self):
        with pytest.raises(TypeError):
            key_for("2026-08-01-06:00")


class TestMalformedKeys:
    """A bad key must fail loudly, naming the expected format."""

    @pytest.mark.parametrize(
        "key",
        [
            "",
            "not-a-key",
            "2026-08-01T06:00",
            "20260801T0600",  # the ring script's filename tag
            "2026-08-01-06:00:00",
            "2026-8-1-6:0 extra",
            "2026-13-01-06:00",
            "2026-08-01-25:00",
        ],
    )
    def test_malformed_key_raises_value_error(self, key):
        with pytest.raises(ValueError, match="malformed partition key"):
            time_for(key)

    def test_error_names_the_expected_format(self):
        with pytest.raises(ValueError) as exc:
            time_for("nope")
        assert PARTITION_KEY_FORMAT in str(exc.value)

    def test_non_string_key_raises_type_error(self):
        with pytest.raises(TypeError):
            time_for(datetime(2026, 8, 1))


class TestLexicographicOrdering:
    """Sorting keys as strings must equal sorting the times."""

    @pytest.mark.parametrize("cadence", [60, 360])
    def test_sorted_keys_match_sorted_times(self, cadence):
        start = datetime(2025, 12, 30, 0, 0)
        times = [
            start + timedelta(minutes=cadence * i)
            for i in range(MINUTES_PER_DAY * 4 // cadence)
        ]
        shuffled = times[::7] + times[::-3] + times
        keys = [key_for(t) for t in shuffled]
        assert sorted(keys) == [key_for(t) for t in sorted(shuffled)]

    def test_ordering_survives_year_and_month_rollover(self):
        pairs = [
            (datetime(2025, 12, 31, 23, 0), datetime(2026, 1, 1, 0, 0)),
            (datetime(2026, 1, 31, 23, 0), datetime(2026, 2, 1, 0, 0)),
            (datetime(2026, 8, 1, 9, 0), datetime(2026, 8, 1, 10, 0)),
        ]
        for earlier, later in pairs:
            assert key_for(earlier) < key_for(later)

    def test_keys_between_is_already_sorted(self):
        keys = keys_between(
            datetime(2025, 12, 31, 12, 0), datetime(2026, 1, 1, 12, 0), 60
        )
        assert keys == sorted(keys)


class TestKeysBetween:
    """Inclusive on both ends when the bounds land on the grid."""

    def test_hourly_over_a_day_is_inclusive(self):
        keys = keys_between(
            datetime(2026, 8, 1, 0, 0), datetime(2026, 8, 2, 0, 0), 60
        )
        assert len(keys) == 25
        assert keys[0] == "2026-08-01-00:00"
        assert keys[-1] == "2026-08-02-00:00"

    def test_six_hourly_over_a_day_is_inclusive(self):
        keys = keys_between(
            datetime(2026, 8, 1, 0, 0), datetime(2026, 8, 2, 0, 0), 360
        )
        assert keys == [
            "2026-08-01-00:00",
            "2026-08-01-06:00",
            "2026-08-01-12:00",
            "2026-08-01-18:00",
            "2026-08-02-00:00",
        ]

    @pytest.mark.parametrize("cadence", [10, 15, 30, 60, 180, 360])
    def test_count_matches_cadence(self, cadence):
        keys = keys_between(
            datetime(2026, 8, 1, 0, 0), datetime(2026, 8, 2, 0, 0), cadence
        )
        assert len(keys) == MINUTES_PER_DAY // cadence + 1

    def test_single_point_range_yields_one_key(self):
        t = datetime(2026, 8, 1, 6, 0)
        assert keys_between(t, t, 360) == ["2026-08-01-06:00"]

    def test_reversed_range_raises(self):
        with pytest.raises(ValueError, match="is before start"):
            keys_between(
                datetime(2026, 8, 1, 12, 0), datetime(2026, 8, 1, 0, 0), 60
            )

    def test_off_grid_start_raises_naming_the_grid(self):
        with pytest.raises(ValueError, match="start=.*not on the 360-minute"):
            keys_between(
                datetime(2026, 8, 1, 1, 0), datetime(2026, 8, 1, 18, 0), 360
            )

    def test_off_grid_end_raises(self):
        with pytest.raises(ValueError, match="end=.*not on the 360-minute"):
            keys_between(
                datetime(2026, 8, 1, 0, 0), datetime(2026, 8, 1, 17, 0), 360
            )

    def test_bounds_may_be_timezone_aware(self):
        tz = timezone(timedelta(hours=-5))
        keys = keys_between(
            datetime(2026, 7, 31, 19, 0, tzinfo=tz),
            datetime(2026, 8, 1, 1, 0, tzinfo=tz),
            360,
        )
        assert keys == ["2026-08-01-00:00", "2026-08-01-06:00"]


class TestAlignToCadence:
    """Flooring onto the grid."""

    @pytest.mark.parametrize(
        ("t", "cadence", "expected"),
        [
            # Already on the grid: returned unchanged.
            (
                datetime(2026, 8, 1, 6, 0),
                360,
                datetime(2026, 8, 1, 6, 0),
            ),
            (
                datetime(2026, 8, 1, 6, 0),
                60,
                datetime(2026, 8, 1, 6, 0),
            ),
            # Just before a boundary: floors to the previous grid point.
            (
                datetime(2026, 8, 1, 5, 59, 59),
                360,
                datetime(2026, 8, 1, 0, 0),
            ),
            (
                datetime(2026, 8, 1, 5, 59, 59, 999_999),
                60,
                datetime(2026, 8, 1, 5, 0),
            ),
            # Just after a boundary.
            (
                datetime(2026, 8, 1, 6, 0, 1),
                360,
                datetime(2026, 8, 1, 6, 0),
            ),
            # Mid-interval.
            (
                datetime(2026, 8, 1, 14, 37, 12),
                360,
                datetime(2026, 8, 1, 12, 0),
            ),
            (
                datetime(2026, 8, 1, 14, 37, 12),
                10,
                datetime(2026, 8, 1, 14, 30),
            ),
            # Last second of the day.
            (
                datetime(2026, 8, 1, 23, 59, 59),
                360,
                datetime(2026, 8, 1, 18, 0),
            ),
            (
                datetime(2026, 8, 1, 23, 59, 59),
                1440,
                datetime(2026, 8, 1, 0, 0),
            ),
        ],
    )
    def test_floors_onto_grid(self, t, cadence, expected):
        assert align_to_cadence(t, cadence) == expected

    def test_result_is_idempotent(self):
        t = datetime(2026, 8, 1, 14, 37, 12, 345)
        once = align_to_cadence(t, 360)
        assert align_to_cadence(once, 360) == once

    def test_result_is_on_the_grid(self):
        t = datetime(2026, 8, 1, 14, 37, 12, 345)
        for cadence in (10, 15, 30, 60, 180, 360, 1440):
            assert is_on_cadence(align_to_cadence(t, cadence), cadence)

    def test_never_moves_forward(self):
        t = datetime(2026, 8, 1, 14, 37, 12)
        assert align_to_cadence(t, 360) <= t

    def test_aware_input_is_floored_in_utc(self):
        tz = timezone(timedelta(hours=5, minutes=30))
        aware = datetime(2026, 8, 1, 11, 45, tzinfo=tz)  # 06:15 UTC
        assert align_to_cadence(aware, 360) == datetime(2026, 8, 1, 6, 0)

    def test_does_not_cross_midnight_backwards(self):
        assert align_to_cadence(datetime(2026, 8, 1, 0, 5), 360) == datetime(
            2026, 8, 1, 0, 0
        )


class TestCadenceValidation:
    """Bad cadences must be rejected with a message that explains."""

    @pytest.mark.parametrize("cadence", [0, -1, -60])
    def test_non_positive_cadence_raises(self, cadence):
        with pytest.raises(ValueError, match="must be > 0|positive"):
            validate_cadence(cadence)

    @pytest.mark.parametrize("cadence", [7, 11, 50, 100, 500, 1441])
    def test_cadence_not_dividing_the_day_raises(self, cadence):
        with pytest.raises(ValueError, match="does not divide the 1440"):
            validate_cadence(cadence)

    @pytest.mark.parametrize("cadence", [1.5, "60", None, True])
    def test_non_integer_cadence_raises_value_error(self, cadence):
        with pytest.raises(ValueError, match="positive integer"):
            validate_cadence(cadence)

    @pytest.mark.parametrize("cadence", [1, 10, 15, 30, 60, 180, 360, 1440])
    def test_valid_cadences_accepted(self, cadence):
        assert validate_cadence(cadence) == cadence

    @pytest.mark.parametrize(
        "func",
        [
            lambda c: align_to_cadence(datetime(2026, 8, 1), c),
            lambda c: keys_between(
                datetime(2026, 8, 1), datetime(2026, 8, 2), c
            ),
            lambda c: window_for("2026-08-01-00:00", c),
            lambda c: build_partitions_def(datetime(2026, 8, 1), c),
            lambda c: is_on_cadence(datetime(2026, 8, 1), c),
        ],
    )
    @pytest.mark.parametrize("cadence", [0, -60, 7])
    def test_every_entry_point_validates_cadence(self, func, cadence):
        with pytest.raises(ValueError):
            func(cadence)


class TestOnCadenceChecks:
    """``is_on_cadence`` / ``validate_on_cadence``."""

    @pytest.mark.parametrize(
        ("t", "cadence", "expected"),
        [
            (datetime(2026, 8, 1, 6, 0), 360, True),
            (datetime(2026, 8, 1, 7, 0), 360, False),
            (datetime(2026, 8, 1, 7, 0), 60, True),
            (datetime(2026, 8, 1, 7, 0, 1), 60, False),
            (datetime(2026, 8, 1, 7, 0, 0, 1), 60, False),
            (datetime(2026, 8, 1, 7, 30), 30, True),
        ],
    )
    def test_is_on_cadence(self, t, cadence, expected):
        assert is_on_cadence(t, cadence) is expected

    def test_validate_returns_naive_utc(self):
        aware = datetime(2026, 8, 1, 6, 0, tzinfo=timezone.utc)
        out = validate_on_cadence(aware, 360)
        assert out == datetime(2026, 8, 1, 6, 0)
        assert out.tzinfo is None

    def test_error_names_grid_and_nearest_point(self):
        with pytest.raises(ValueError) as exc:
            validate_on_cadence(datetime(2026, 8, 1, 7, 13), 360)
        message = str(exc.value)
        assert "360-minute grid" in message
        assert "2026-08-01T06:00:00" in message


class TestWindowFor:
    """The interval a partition covers."""

    @pytest.mark.parametrize(
        ("key", "cadence", "expected"),
        [
            (
                "2026-08-01-06:00",
                360,
                (datetime(2026, 8, 1, 6, 0), datetime(2026, 8, 1, 12, 0)),
            ),
            (
                "2026-08-01-23:00",
                60,
                (datetime(2026, 8, 1, 23, 0), datetime(2026, 8, 2, 0, 0)),
            ),
            (
                "2026-08-01-18:00",
                360,
                (datetime(2026, 8, 1, 18, 0), datetime(2026, 8, 2, 0, 0)),
            ),
        ],
    )
    def test_window_bounds(self, key, cadence, expected):
        assert window_for(key, cadence) == expected

    def test_window_start_round_trips_to_the_key(self):
        key = "2026-08-01-12:00"
        start, _ = window_for(key, 360)
        assert key_for(start) == key

    def test_windows_tile_without_gaps_or_overlap(self):
        keys = keys_between(
            datetime(2026, 8, 1), datetime(2026, 8, 1, 18, 0), 360
        )
        windows = [window_for(k, 360) for k in keys]
        for (_, end), (next_start, _) in zip(windows, windows[1:]):
            assert end == next_start

    def test_off_grid_key_raises(self):
        with pytest.raises(ValueError, match="not on the 360-minute grid"):
            window_for("2026-08-01-07:00", 360)

    def test_malformed_key_raises(self):
        with pytest.raises(ValueError, match="malformed partition key"):
            window_for("20260801T0600", 360)


class TestCronForCadence:
    """The cron expression behind the partitions definition."""

    @pytest.mark.parametrize(
        ("cadence", "expected"),
        [
            (10, "*/10 * * * *"),
            (15, "*/15 * * * *"),
            (30, "*/30 * * * *"),
            (60, "0 * * * *"),
            (180, "0 0,3,6,9,12,15,18,21 * * *"),
            (360, "0 0,6,12,18 * * *"),
            (720, "0 0,12 * * *"),
            (1440, "0 0 * * *"),
        ],
    )
    def test_expected_cron(self, cadence, expected):
        assert cron_for_cadence(cadence) == expected

    @pytest.mark.parametrize("cadence", [16, 32, 36, 45, 48])
    def test_sub_hourly_not_dividing_the_hour_raises(self, cadence):
        with pytest.raises(ValueError, match="divides the day but not"):
            cron_for_cadence(cadence)

    @pytest.mark.parametrize("cadence", [90, 96, 288])
    def test_super_hourly_not_whole_hours_raises(self, cadence):
        with pytest.raises(ValueError, match="not a whole number of hours"):
            cron_for_cadence(cadence)


class TestBuildPartitionsDef:
    """Dagster must accept what we hand it, and agree on the keys."""

    def test_returns_a_dagster_partitions_definition(self):
        pdef = build_partitions_def(datetime(2026, 8, 1), 60)
        assert isinstance(pdef, TimeWindowPartitionsDefinition)

    def test_hourly_partition_keys(self):
        pdef = build_partitions_def(datetime(2026, 8, 1), 60)
        keys = pdef.get_partition_keys(
            current_time=datetime(2026, 8, 1, 5, 0, tzinfo=timezone.utc)
        )
        assert keys == [
            "2026-08-01-00:00",
            "2026-08-01-01:00",
            "2026-08-01-02:00",
            "2026-08-01-03:00",
            "2026-08-01-04:00",
        ]

    def test_six_hourly_partition_keys(self):
        pdef = build_partitions_def(datetime(2026, 8, 1), 360)
        keys = pdef.get_partition_keys(
            current_time=datetime(2026, 8, 2, 0, 0, tzinfo=timezone.utc)
        )
        assert keys == [
            "2026-08-01-00:00",
            "2026-08-01-06:00",
            "2026-08-01-12:00",
            "2026-08-01-18:00",
        ]

    @pytest.mark.parametrize("cadence", [10, 15, 30, 60, 180, 360])
    def test_dagster_keys_match_keys_between(self, cadence):
        start = datetime(2026, 8, 1)
        end = datetime(2026, 8, 2)
        pdef = build_partitions_def(start, cadence)
        dagster_keys = pdef.get_partition_keys(
            current_time=end.replace(tzinfo=timezone.utc)
        )
        # Dagster's listing is half-open on the right; ours is inclusive.
        assert dagster_keys == keys_between(start, end, cadence)[:-1]

    @pytest.mark.parametrize("cadence", [60, 360])
    def test_dagster_time_window_matches_window_for(self, cadence):
        pdef = build_partitions_def(datetime(2026, 8, 1), cadence)
        key = "2026-08-01-06:00" if cadence == 360 else "2026-08-01-03:00"
        window = pdef.time_window_for_partition_key(key)
        start, end = window_for(key, cadence)
        assert window.start.replace(tzinfo=None) == start
        assert window.end.replace(tzinfo=None) == end
        assert window.start.tzinfo == timezone.utc

    def test_keys_round_trip_through_time_for(self):
        pdef = build_partitions_def(datetime(2026, 8, 1), 360)
        keys = pdef.get_partition_keys(
            current_time=datetime(2026, 8, 3, 0, 0, tzinfo=timezone.utc)
        )
        assert [key_for(time_for(k)) for k in keys] == keys

    def test_off_grid_start_raises(self):
        with pytest.raises(ValueError, match="start=.*not on the 360-minute"):
            build_partitions_def(datetime(2026, 8, 1, 1, 0), 360)

    def test_aware_start_accepted(self):
        pdef = build_partitions_def(
            datetime(2026, 8, 1, tzinfo=timezone.utc), 360
        )
        keys = pdef.get_partition_keys(
            current_time=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        )
        assert keys == ["2026-08-01-00:00", "2026-08-01-06:00"]


class TestFilenameTagIsDistinct:
    """Partition keys are not the ring script's filename tags."""

    def test_formats_differ_but_denote_the_same_instant(self):
        t = datetime(2026, 8, 1, 0, 0)
        key = key_for(t)
        tag = t.strftime("%Y%m%dT%H%M")
        assert key == "2026-08-01-00:00"
        assert tag == "20260801T0000"
        assert key != tag
        assert time_for(key) == datetime.strptime(tag, "%Y%m%dT%H%M")
