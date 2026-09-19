"""Unit tests for the availability sensor.

Everything here is offline: :func:`operational.core.availability.new_timestamps`
is stubbed with ``monkeypatch`` while the watermark store is the real one,
backed by ``tmp_path``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from dagster import (
    Definitions,
    HourlyPartitionsDefinition,
    RunRequest,
    SkipReason,
    build_sensor_context,
    job,
    op,
)

from operational.config import OperationalConfig
from operational.core import availability as availability_mod
from operational.core.watermark import WatermarkStore
from operational.sensors import build_availability_sensor

# Naive UTC, matching the watermark store and the ring script.
NOW = datetime(2026, 1, 2, 12, 0)
SATS = ("goes18", "goes19")


def _hours_ago(n: float) -> datetime:
    """Return a whole-hour timestamp ``n`` hours before :data:`NOW`."""
    return NOW - timedelta(hours=n)


class StubAvailability:
    """Recording stand-in for ``new_timestamps``.

    Parameters
    ----------
    per_sat
        Timestamps each satellite reports, before the ``since``/``until``
        filtering the real implementation performs (applied here too, so the
        sensor sees a realistic incremental answer).
    raises
        Satellites whose lookup should raise instead of answering.
    """

    def __init__(
        self,
        per_sat: dict[str, list[datetime]],
        raises: dict[str, Exception] | None = None,
    ) -> None:
        self.per_sat = per_sat
        self.raises = raises or {}
        self.calls: list[dict] = []

    def __call__(self, sat_id: str, since, until, **kwargs):
        self.calls.append({"sat_id": sat_id, "since": since, "until": until, **kwargs})
        if sat_id in self.raises:
            raise self.raises[sat_id]
        out = [t for t in self.per_sat.get(sat_id, []) if since < t <= until]
        return sorted(out)

    def calls_for(self, sat_id: str) -> list[dict]:
        """Return every recorded call for one satellite."""
        return [c for c in self.calls if c["sat_id"] == sat_id]


@pytest.fixture
def sensor_config(tmp_path: Path) -> OperationalConfig:
    """Two-satellite config with hourly cadence writing under ``tmp_path``."""
    return OperationalConfig(
        satellites=SATS,
        cadence_minutes=60,
        output_dir=tmp_path / "operational",
    )


@pytest.fixture
def wm_path(tmp_path: Path) -> Path:
    """Path of the watermark file used by the sensors under test."""
    return tmp_path / "watermarks.json"


def make_sensor(config: OperationalConfig, wm_path: Path, **kwargs):
    """Build a sensor pinned to a fixed clock and the tmp watermark file."""
    kwargs.setdefault("now_fn", lambda: NOW)
    kwargs.setdefault("lookback_hours", 6.0)
    return build_availability_sensor(config=config, watermark_path=wm_path, **kwargs)


def evaluate(sensor_def, cursor: str | None = None):
    """Evaluate a sensor once and return ``(results, context)``."""
    context = build_sensor_context(sensor_name=sensor_def.name, cursor=cursor)
    results = list(sensor_def(context))
    return results, context


def run_requests(results) -> list[RunRequest]:
    """Filter run requests out of a tick's results."""
    return [r for r in results if isinstance(r, RunRequest)]


def skip_reasons(results) -> list[SkipReason]:
    """Filter skip reasons out of a tick's results."""
    return [r for r in results if isinstance(r, SkipReason)]


class TestAvailabilitySensorEmission:
    """Run requests produced for newly available timestamps."""

    def test_emits_run_requests_with_partition_and_run_keys(
        self, monkeypatch, sensor_config, wm_path
    ):
        stub = StubAvailability({s: [_hours_ago(2), _hours_ago(1)] for s in SATS})
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        results, _ = evaluate(make_sensor(sensor_config, wm_path))
        requests = run_requests(results)

        assert skip_reasons(results) == []
        assert [r.partition_key for r in requests] == ["2026-01-02-10:00", "2026-01-02-11:00"]
        assert [r.run_key for r in requests] == ["2026-01-02-10:00", "2026-01-02-11:00"]
        assert requests[0].tags["operational/satellites"] == "goes18,goes19"
        assert requests[0].tags["operational/readiness_rule"] == "all"

    def test_run_key_is_stable_for_the_same_partition(
        self, monkeypatch, sensor_config, wm_path
    ):
        stub = StubAvailability({s: [_hours_ago(1)] for s in SATS})
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        first, _ = evaluate(make_sensor(sensor_config, wm_path))
        WatermarkStore(wm_path).reset()  # simulate a replay of the same tick
        second, _ = evaluate(make_sensor(sensor_config, wm_path))

        assert [r.run_key for r in run_requests(first)] == ["2026-01-02-11:00"]
        assert [r.run_key for r in run_requests(second)] == ["2026-01-02-11:00"]

    def test_second_tick_with_no_new_data_skips(self, monkeypatch, sensor_config, wm_path):
        stub = StubAvailability({s: [_hours_ago(1)] for s in SATS})
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)
        sensor_def = make_sensor(sensor_config, wm_path)

        first, ctx = evaluate(sensor_def)
        assert len(run_requests(first)) == 1

        second, _ = evaluate(sensor_def, cursor=ctx.cursor)

        assert run_requests(second) == []
        skips = skip_reasons(second)
        assert len(skips) == 1
        message = skips[0].skip_message
        assert "No new partitions ready" in message
        assert "2026-01-02T06:00Z..2026-01-02T12:00Z" in message
        assert "goes18" in message and "goes19" in message

    def test_emitted_timestamps_resolve_against_a_partitioned_job(
        self, monkeypatch, sensor_config, wm_path
    ):
        """The keys the sensor emits are valid Dagster hourly partition keys."""
        stub = StubAvailability({s: [_hours_ago(1)] for s in SATS})
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        @op
        def noop():
            return None

        @job(partitions_def=HourlyPartitionsDefinition(start_date="2026-01-01-00:00"))
        def operational_ring_job():
            noop()

        sensor_def = make_sensor(sensor_config, wm_path)
        defs = Definitions(jobs=[operational_ring_job], sensors=[sensor_def])
        context = build_sensor_context(
            sensor_name=sensor_def.name, repository_def=defs.get_repository_def()
        )

        result = sensor_def.evaluate_tick(context)

        assert [r.partition_key for r in result.run_requests] == ["2026-01-02-11:00"]
        assert result.run_requests[0].tags["dagster/partition"] == "2026-01-02-11:00"


class TestWatermarkBehaviour:
    """Incremental behaviour driven by the durable watermark store."""

    def test_watermark_advances_only_for_emitted_timestamps(
        self, monkeypatch, sensor_config, wm_path
    ):
        # goes19 is late for the most recent hour, so it must not be emitted and
        # neither satellite's watermark may move past it.
        stub = StubAvailability(
            {
                "goes18": [_hours_ago(2), _hours_ago(1)],
                "goes19": [_hours_ago(2)],
            }
        )
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        results, _ = evaluate(make_sensor(sensor_config, wm_path))

        assert [r.partition_key for r in run_requests(results)] == ["2026-01-02-10:00"]
        store = WatermarkStore(wm_path)
        assert store.get("goes18") == _hours_ago(2)
        assert store.get("goes19") == _hours_ago(2)

    def test_withheld_timestamp_is_emitted_once_the_late_satellite_arrives(
        self, monkeypatch, sensor_config, wm_path
    ):
        late = {
            "goes18": [_hours_ago(2), _hours_ago(1)],
            "goes19": [_hours_ago(2)],
        }
        stub = StubAvailability(late)
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)
        sensor_def = make_sensor(sensor_config, wm_path)

        first, ctx = evaluate(sensor_def)
        assert [r.partition_key for r in run_requests(first)] == ["2026-01-02-10:00"]

        late["goes19"] = [_hours_ago(2), _hours_ago(1)]
        second, _ = evaluate(sensor_def, cursor=ctx.cursor)

        assert [r.partition_key for r in run_requests(second)] == ["2026-01-02-11:00"]
        assert WatermarkStore(wm_path).get("goes18") == _hours_ago(1)

    def test_late_satellite_is_not_advanced_past_a_timestamp_it_never_reported(
        self, monkeypatch, sensor_config, wm_path
    ):
        """Regression: a global barrier, not a per-satellite one.

        ``goes19``'s 10:00 upload is late, so only 11:00 clears ``require_all``.
        If ``goes19`` were allowed to advance to 11:00 it would never look at
        10:00 again and that partition could never complete.
        """
        per_sat = {
            "goes18": [_hours_ago(2), _hours_ago(1)],
            "goes19": [_hours_ago(1)],
        }
        stub = StubAvailability(per_sat)
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)
        sensor_def = make_sensor(sensor_config, wm_path)

        first, ctx = evaluate(sensor_def)
        assert [r.partition_key for r in run_requests(first)] == ["2026-01-02-11:00"]

        store = WatermarkStore(wm_path)
        assert store.get("goes19") is None, "advanced past an undelivered timestamp"
        assert store.get("goes18") is None

        # goes19's 10:00 data lands; the withheld partition must now be emitted.
        per_sat["goes19"] = [_hours_ago(2), _hours_ago(1)]
        second, _ = evaluate(sensor_def, cursor=ctx.cursor)

        keys = [r.partition_key for r in run_requests(second)]
        assert "2026-01-02-10:00" in keys
        # 11:00 may be re-requested; the stable run_key makes that a no-op.
        assert all(r.run_key == r.partition_key for r in run_requests(second))
        assert WatermarkStore(wm_path).get("goes19") == _hours_ago(1)

    def test_barrier_clears_once_the_withheld_candidate_leaves_the_window(
        self, monkeypatch, sensor_config, wm_path
    ):
        """A permanently missing scene must not stall the sensor forever."""
        stub = StubAvailability(
            {"goes18": [_hours_ago(5), _hours_ago(1)], "goes19": [_hours_ago(1)]}
        )
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        first, _ = evaluate(make_sensor(sensor_config, wm_path, lookback_hours=6.0))
        assert [r.partition_key for r in run_requests(first)] == ["2026-01-02-11:00"]
        assert WatermarkStore(wm_path).all() == {}  # held below the 07:00 barrier

        # A later tick whose window no longer reaches the missing 07:00 scene.
        later = make_sensor(sensor_config, wm_path, lookback_hours=2.0)
        evaluate(later)

        store = WatermarkStore(wm_path)
        assert store.get("goes18") == _hours_ago(1)
        assert store.get("goes19") == _hours_ago(1)

    def test_watermark_persists_across_a_fresh_sensor_and_empty_cursor(
        self, monkeypatch, sensor_config, wm_path
    ):
        stub = StubAvailability({s: [_hours_ago(1)] for s in SATS})
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        first, _ = evaluate(make_sensor(sensor_config, wm_path))
        assert len(run_requests(first)) == 1

        # A restart: brand new sensor definition, no cursor carried over.
        second, _ = evaluate(make_sensor(sensor_config, wm_path), cursor=None)

        assert run_requests(second) == []
        assert len(skip_reasons(second)) == 1
        # The second tick searched forward of the persisted watermark.
        assert stub.calls_for("goes18")[-1]["since"] == _hours_ago(1)

    def test_cursor_mirrors_the_watermark_store(self, monkeypatch, sensor_config, wm_path):
        stub = StubAvailability({s: [_hours_ago(1)] for s in SATS})
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        _, ctx = evaluate(make_sensor(sensor_config, wm_path))

        assert ctx.cursor is not None
        stored = WatermarkStore(wm_path).all()
        for sat_id, t in stored.items():
            assert sat_id in ctx.cursor
            assert t.isoformat() in ctx.cursor

    def test_stale_cursor_does_not_gate_the_search_window(
        self, monkeypatch, sensor_config, wm_path
    ):
        """The watermark store, not the cursor, decides what is re-examined."""
        stub = StubAvailability({s: [_hours_ago(1)] for s in SATS})
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)
        sensor_def = make_sensor(sensor_config, wm_path)

        _, ctx = evaluate(sensor_def)
        WatermarkStore(wm_path).reset()  # operator forces a replay

        replay, _ = evaluate(sensor_def, cursor=ctx.cursor)

        assert [r.partition_key for r in run_requests(replay)] == ["2026-01-02-11:00"]


class TestReadinessRule:
    """``require_all`` chooses between a complete mosaic and low latency."""

    def test_require_all_withholds_partially_available_timestamp(
        self, monkeypatch, sensor_config, wm_path
    ):
        stub = StubAvailability({"goes18": [_hours_ago(1)], "goes19": []})
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        results, _ = evaluate(make_sensor(sensor_config, wm_path, require_all=True))

        assert run_requests(results) == []
        message = skip_reasons(results)[0].skip_message
        assert "Waiting on 1 candidate(s)" in message
        assert "2026-01-02T11:00Z [goes18]" in message
        # goes19 answered successfully with nothing new -- it must not be
        # described as unreachable, or an operator chases a phantom outage.
        assert "Reported no new data: goes19." in message
        assert "failed" not in message
        assert WatermarkStore(wm_path).get("goes18") is None

    def test_require_any_emits_partially_available_timestamp(
        self, monkeypatch, sensor_config, wm_path
    ):
        stub = StubAvailability({"goes18": [_hours_ago(1)], "goes19": []})
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        results, _ = evaluate(make_sensor(sensor_config, wm_path, require_all=False))
        requests = run_requests(results)

        assert [r.partition_key for r in requests] == ["2026-01-02-11:00"]
        assert requests[0].tags["operational/satellites"] == "goes18"
        assert requests[0].tags["operational/readiness_rule"] == "any"
        store = WatermarkStore(wm_path)
        assert store.get("goes18") == _hours_ago(1)
        assert store.get("goes19") is None

    def test_require_all_is_the_default(self, sensor_config, wm_path):
        sensor_def = make_sensor(sensor_config, wm_path)
        assert "all of 2 satellites" not in (sensor_def.description or "")
        assert "all of them can deliver" in (sensor_def.description or "")


class TestResilience:
    """One failing archive must not take the tick down with it."""

    def test_failing_satellite_does_not_abort_the_tick(
        self, monkeypatch, sensor_config, wm_path
    ):
        stub = StubAvailability(
            {"goes18": [_hours_ago(1)], "goes19": [_hours_ago(1)]},
            raises={"goes19": RuntimeError("S3 listing timed out")},
        )
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        results, _ = evaluate(make_sensor(sensor_config, wm_path, require_all=False))
        requests = run_requests(results)

        assert [r.partition_key for r in requests] == ["2026-01-02-11:00"]
        assert requests[0].tags["operational/satellites"] == "goes18"
        # The failing satellite was still asked, and its watermark did not move.
        assert stub.calls_for("goes19")
        assert WatermarkStore(wm_path).get("goes19") is None
        assert WatermarkStore(wm_path).get("goes18") == _hours_ago(1)

    def test_failure_under_require_all_withholds_and_explains(
        self, monkeypatch, sensor_config, wm_path
    ):
        stub = StubAvailability(
            {"goes18": [_hours_ago(1)], "goes19": [_hours_ago(1)]},
            raises={"goes19": RuntimeError("S3 listing timed out")},
        )
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        results, _ = evaluate(make_sensor(sensor_config, wm_path, require_all=True))

        assert run_requests(results) == []
        message = skip_reasons(results)[0].skip_message
        assert "goes19: RuntimeError: S3 listing timed out" in message
        assert "watermark unchanged" in message
        store = WatermarkStore(wm_path)
        assert store.get("goes18") is None
        assert store.get("goes19") is None

    def test_all_satellites_failing_skips_without_raising(
        self, monkeypatch, sensor_config, wm_path
    ):
        stub = StubAvailability(
            {s: [_hours_ago(1)] for s in SATS},
            raises={s: OSError("archive down") for s in SATS},
        )
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        results, _ = evaluate(make_sensor(sensor_config, wm_path))

        assert run_requests(results) == []
        assert "archive down" in skip_reasons(results)[0].skip_message
        assert WatermarkStore(wm_path).all() == {}


class TestLookbackWindow:
    """Every tick searches a bounded, recent window."""

    def test_first_tick_uses_a_bounded_window_not_the_whole_archive(
        self, monkeypatch, sensor_config, wm_path
    ):
        stub = StubAvailability({s: [] for s in SATS})
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        evaluate(make_sensor(sensor_config, wm_path, lookback_hours=6.0))

        assert len(stub.calls) == len(SATS)
        for call in stub.calls:
            assert call["since"] == NOW - timedelta(hours=6)
            assert call["until"] == NOW
            assert call["since"] is not None
            assert (call["until"] - call["since"]) == timedelta(hours=6)

    def test_stale_watermark_is_clamped_to_the_lookback_window(
        self, monkeypatch, sensor_config, wm_path
    ):
        store = WatermarkStore(wm_path)
        store.set("goes18", NOW - timedelta(days=30))
        stub = StubAvailability({s: [] for s in SATS})
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        evaluate(make_sensor(sensor_config, wm_path, lookback_hours=6.0))

        assert stub.calls_for("goes18")[0]["since"] == NOW - timedelta(hours=6)

    def test_recent_watermark_narrows_the_window(self, monkeypatch, sensor_config, wm_path):
        WatermarkStore(wm_path).set("goes18", _hours_ago(2))
        stub = StubAvailability({s: [] for s in SATS})
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        evaluate(make_sensor(sensor_config, wm_path, lookback_hours=6.0))

        assert stub.calls_for("goes18")[0]["since"] == _hours_ago(2)
        assert stub.calls_for("goes19")[0]["since"] == NOW - timedelta(hours=6)

    def test_future_watermark_does_not_produce_an_inverted_window(
        self, monkeypatch, sensor_config, wm_path
    ):
        """Clock skew must not hand the archive ``since > until``."""
        WatermarkStore(wm_path).set("goes18", NOW + timedelta(hours=3))
        stub = StubAvailability({s: [_hours_ago(1)] for s in SATS})
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        results, _ = evaluate(make_sensor(sensor_config, wm_path))

        assert stub.calls_for("goes18") == []  # never asked with a bad window
        assert [c["sat_id"] for c in stub.calls] == ["goes19"]
        assert run_requests(results) == []  # goes18 cannot satisfy require_all
        assert WatermarkStore(wm_path).get("goes18") == NOW + timedelta(hours=3)

    def test_availability_call_carries_the_configured_bands_and_cadence(
        self, monkeypatch, sensor_config, wm_path
    ):
        stub = StubAvailability({s: [] for s in SATS})
        monkeypatch.setattr(availability_mod, "new_timestamps", stub)

        evaluate(make_sensor(sensor_config, wm_path))

        call = stub.calls[0]
        assert call["flow_bands"] == list(sensor_config.flow_bands)
        assert call["rad_bands"] == list(sensor_config.rad_bands)
        assert call["cadence_minutes"] == sensor_config.cadence_minutes
        assert call["tolerance_minutes"] == sensor_config.availability_tolerance_minutes


class TestSensorDefinition:
    """Definition-level wiring an operator sees in the Dagster UI."""

    def test_name_interval_and_description(self, sensor_config, wm_path):
        sensor_def = make_sensor(sensor_config, wm_path)

        assert sensor_def.name == "amv_availability_sensor"
        assert sensor_def.minimum_interval_seconds == 300
        description = sensor_def.description or ""
        assert "goes18" in description and "goes19" in description
        assert "6 h" in description

    def test_rejects_nonsense_configuration(self, sensor_config, wm_path):
        with pytest.raises(ValueError, match="lookback_hours"):
            make_sensor(sensor_config, wm_path, lookback_hours=0)
        with pytest.raises(ValueError, match="satellites"):
            build_availability_sensor(
                config=OperationalConfig(satellites=()), watermark_path=wm_path
            )
        # A bad cadence must fail loudly at definition time, not disguise itself
        # as a per-satellite archive failure on every tick.
        with pytest.raises(ValueError, match="cadence_minutes"):
            build_availability_sensor(
                config=OperationalConfig(satellites=SATS, cadence_minutes=0),
                watermark_path=wm_path,
            )

    def test_default_watermark_path_lives_under_the_output_dir(self, sensor_config):
        expected = (Path(sensor_config.output_dir) / "watermarks.json").resolve()
        sensor_def = build_availability_sensor(config=sensor_config, now_fn=lambda: NOW)
        assert str(expected) in (sensor_def.description or "")

    def test_relative_output_dir_is_resolved_to_an_absolute_path(self, monkeypatch, tmp_path):
        """The source of truth must not move with the process CWD."""
        monkeypatch.chdir(tmp_path)
        config = OperationalConfig(satellites=SATS, output_dir=Path("output/operational"))

        sensor_def = build_availability_sensor(config=config, now_fn=lambda: NOW)

        expected = tmp_path / "output" / "operational" / "watermarks.json"
        assert str(expected.resolve()) in (sensor_def.description or "")

    def test_env_var_overrides_the_default_watermark_path(
        self, monkeypatch, sensor_config, tmp_path
    ):
        override = tmp_path / "shared" / "watermarks.json"
        monkeypatch.setenv("STEREO_WINDS_OPERATIONAL_WATERMARK", str(override))

        sensor_def = build_availability_sensor(config=sensor_config, now_fn=lambda: NOW)

        assert str(override.resolve()) in (sensor_def.description or "")
