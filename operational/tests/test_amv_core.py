"""Tests for the per-satellite AMV retrieval step.

Offline throughout: ``infer_satellite`` is replaced on the adapter
module with a synthetic scene, so the canonical naming, the atomic
write, the resume behaviour and the quality lifting can all be
exercised without touching the network, a GPU or a checkpoint.
"""

from __future__ import annotations

import dataclasses
import os
import time
from datetime import datetime
from pathlib import Path

import pytest
import xarray as xr

from operational.adapters import ring as ring_adapter
from operational.core.amv import (
    AmvResult,
    STALE_TMP_AGE_S,
    TMP_SUFFIX,
    quality_from_attrs,
    run_satellite_amv,
)

from .conftest import synthetic_scene

T0 = datetime(2026, 8, 1, 0, 0)
BANDS = ["C08", "C14"]


class CallLog(list):
    """Recorded ``(sat_id, t0)`` calls, plus the attrs the stub returns.

    A list subclass so tests can assert on it directly, with an ``attrs``
    dict tests mutate to control the quality attrs of the synthetic
    scene the stub hands back.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attrs: dict = {}


@pytest.fixture
def stub_infer(monkeypatch):
    """Stub the retrieval; record every ``(sat_id, t0)`` it is asked for.

    The call log is what the resume tests assert on: a reused file must
    not add an entry.
    """
    calls = CallLog()

    def fake(sat_id, t0, model, disp, flow_bands, rad_bands, **kwargs):
        calls.append((sat_id, t0))
        ds = synthetic_scene(sat_id, t0, ny=8, nx=8)
        ds.attrs.update(calls.attrs)
        return ds

    monkeypatch.setattr(ring_adapter, "infer_satellite", fake)
    return calls


def _run(out_dir, sat_id="goes19", t0=T0, **kwargs):
    return run_satellite_amv(
        sat_id,
        t0,
        None,
        None,
        BANDS,
        BANDS,
        out_dir,
        **kwargs,
    )


# ── Successful run ────────────────────────────────────────────────────


class TestSuccessfulRun:
    def test_writes_file_at_canonical_path(self, tmp_path, stub_infer):
        result = _run(tmp_path)
        expected = tmp_path / "20260801" / "student_amv_goes19_20260801T0000.nc"
        assert result.path == expected
        assert expected.exists()

    def test_path_matches_upstream_naming(self, tmp_path, stub_infer):
        result = _run(tmp_path)
        assert result.path == ring_adapter.sat_nc_path(tmp_path, "goes19", T0)

    def test_returns_populated_result(self, tmp_path, stub_infer):
        result = _run(tmp_path)
        assert isinstance(result, AmvResult)
        assert result.sat_id == "goes19"
        assert result.timestamp == T0
        assert result.reused is False
        assert isinstance(result.dataset, xr.Dataset)
        assert set(ring_adapter.OUTPUT_VARS) <= set(result.dataset.data_vars)

    def test_calls_infer_once(self, tmp_path, stub_infer):
        _run(tmp_path)
        assert stub_infer == [("goes19", T0)]

    def test_written_file_is_readable_and_complete(self, tmp_path, stub_infer):
        result = _run(tmp_path)
        with xr.open_dataset(result.path) as ds:
            assert set(ring_adapter.OUTPUT_VARS) <= set(ds.data_vars)
            assert ds.attrs["satellite_id"] == "goes19"
            assert ds["u_wind"].shape == (8, 8)

    def test_creates_parent_directories(self, tmp_path, stub_infer):
        nested = tmp_path / "deep" / "deeper" / "out"
        assert not nested.exists()
        result = _run(nested)
        assert result.path.exists()
        assert result.path.parent == nested / "20260801"

    def test_no_temp_residue(self, tmp_path, stub_infer):
        _run(tmp_path)
        assert list(tmp_path.rglob(f"*{TMP_SUFFIX}")) == []

    def test_forwards_device_and_row_strip(self, tmp_path, monkeypatch):
        seen: dict = {}

        def fake(sat_id, t0, model, disp, flow_bands, rad_bands, **kwargs):
            seen.update(kwargs)
            seen["flow_bands"] = flow_bands
            seen["rad_bands"] = rad_bands
            return synthetic_scene(sat_id, t0, ny=4, nx=4)

        monkeypatch.setattr(ring_adapter, "infer_satellite", fake)
        _run(tmp_path, device="cuda", row_strip=256)
        assert seen["device"] == "cuda"
        assert seen["row_strip"] == 256
        assert seen["flow_bands"] == BANDS
        assert seen["rad_bands"] == BANDS

    def test_each_satellite_gets_its_own_file(self, tmp_path, stub_infer):
        a = _run(tmp_path, sat_id="goes18")
        b = _run(tmp_path, sat_id="goes19")
        assert a.path != b.path
        assert a.path.exists() and b.path.exists()

    def test_day_partitioned_by_timestamp(self, tmp_path, stub_infer):
        late = datetime(2026, 8, 1, 23, 50)
        result = _run(tmp_path, t0=late)
        assert result.path.parent.name == "20260801"
        nxt = _run(tmp_path, t0=datetime(2026, 8, 2, 0, 0))
        assert nxt.path.parent.name == "20260802"


# ── Resume behaviour ──────────────────────────────────────────────────


class TestSkipExisting:
    def test_second_call_reuses_without_recomputing(self, tmp_path, stub_infer):
        first = _run(tmp_path, skip_existing=True)
        second = _run(tmp_path, skip_existing=True)
        assert len(stub_infer) == 1
        assert second.reused is True
        assert second.path == first.path

    def test_reused_result_carries_the_data(self, tmp_path, stub_infer):
        _run(tmp_path)
        second = _run(tmp_path, skip_existing=True)
        assert set(ring_adapter.OUTPUT_VARS) <= set(second.dataset.data_vars)
        assert second.dataset.attrs["satellite_id"] == "goes19"

    def test_skip_existing_false_recomputes(self, tmp_path, stub_infer):
        _run(tmp_path, skip_existing=False)
        result = _run(tmp_path, skip_existing=False)
        assert len(stub_infer) == 2
        assert result.reused is False

    def test_skip_existing_is_per_satellite(self, tmp_path, stub_infer):
        """One satellite already on disk must not skip the others."""
        _run(tmp_path, sat_id="goes18")
        _run(tmp_path, sat_id="goes19")
        assert stub_infer == [("goes18", T0), ("goes19", T0)]

    def test_no_file_means_compute_even_when_skipping(self, tmp_path, stub_infer):
        result = _run(tmp_path, skip_existing=True)
        assert result.reused is False
        assert len(stub_infer) == 1

    def test_unreadable_existing_file_is_recomputed(self, tmp_path, stub_infer):
        path = ring_adapter.sat_nc_path(tmp_path, "goes19", T0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a netcdf")
        result = _run(tmp_path, skip_existing=True)
        assert result.reused is False
        assert len(stub_infer) == 1
        # The corrupt file was replaced by real output.
        with xr.open_dataset(path) as ds:
            assert ds.attrs["satellite_id"] == "goes19"


# ── Quality attrs ─────────────────────────────────────────────────────


class TestQualityAttrs:
    def test_clean_retrieval(self, tmp_path, stub_infer):
        stub_infer.attrs.update(ring_adapter.quality_attrs(BANDS, BANDS, {"flow": [], "rad": []}))
        result = _run(tmp_path)
        assert result.n_bands_missing == 0
        assert result.bands_missing == ()
        assert result.quality_degraded is False
        assert "all requested bands available" in result.quality_note

    def test_degraded_retrieval(self, tmp_path, stub_infer):
        stub_infer.attrs.update(
            ring_adapter.quality_attrs(BANDS, BANDS, {"flow": ["C08"], "rad": []})
        )
        result = _run(tmp_path)
        assert result.n_bands_missing == 1
        assert result.bands_missing == ("C08",)
        assert result.quality_degraded is True
        assert "DEGRADED" in result.quality_note

    def test_missing_band_below_threshold_is_not_degraded(self, tmp_path, stub_infer):
        many = [f"C{i:02d}" for i in range(1, 17)]
        stub_infer.attrs.update(
            ring_adapter.quality_attrs(many, many, {"flow": [], "rad": ["C01"]})
        )
        result = _run(tmp_path)
        assert result.n_bands_missing == 1
        assert result.quality_degraded is False

    def test_attrs_are_lifted_not_recomputed(self, tmp_path, stub_infer):
        """Whatever upstream wrote is reported, even if it looks odd."""
        stub_infer.attrs.update(
            {
                "bands_missing": "C07,C09",
                "n_bands_missing": 2,
                "quality_degraded": 1,
                "quality_note": "upstream said so",
            }
        )
        result = _run(tmp_path)
        assert result.bands_missing == ("C07", "C09")
        assert result.n_bands_missing == 2
        assert result.quality_degraded is True
        assert result.quality_note == "upstream said so"

    def test_absent_attrs_degrade_gracefully(self, tmp_path, stub_infer):
        result = _run(tmp_path)  # synthetic scene carries no quality attrs
        assert result.n_bands_missing == 0
        assert result.bands_missing == ()
        assert result.quality_degraded is False
        assert result.quality_note == ""

    def test_quality_survives_the_netcdf_round_trip(self, tmp_path, stub_infer):
        stub_infer.attrs.update(
            ring_adapter.quality_attrs(BANDS, BANDS, {"flow": ["C08"], "rad": []})
        )
        fresh = _run(tmp_path)
        reused = _run(tmp_path, skip_existing=True)
        assert reused.reused is True
        assert reused.n_bands_missing == fresh.n_bands_missing
        assert reused.bands_missing == fresh.bands_missing
        assert reused.quality_degraded == fresh.quality_degraded

    def test_degraded_is_logged_on_the_resume_path(self, tmp_path, stub_infer, caplog):
        """A resumed run must still report which satellites were degraded."""
        stub_infer.attrs.update(
            ring_adapter.quality_attrs(BANDS, BANDS, {"flow": ["C08"], "rad": []})
        )
        _run(tmp_path)
        caplog.clear()
        with caplog.at_level("WARNING", logger="operational.core.amv"):
            result = _run(tmp_path, skip_existing=True)
        assert result.reused is True
        assert "degraded" in caplog.text

    def test_count_falls_back_to_the_names(self):
        lifted = quality_from_attrs({"bands_missing": "C07,C09"})
        assert lifted["n_bands_missing"] == 2

    def test_unparsable_count_does_not_raise(self):
        lifted = quality_from_attrs({"n_bands_missing": "lots"})
        assert lifted["n_bands_missing"] == 0

    def test_empty_band_string_is_no_bands(self):
        lifted = quality_from_attrs({"bands_missing": "", "n_bands_missing": 0})
        assert lifted["bands_missing"] == ()

    def test_unreadable_flag_fails_closed(self):
        """An unreadable quality flag must not be reported as clean."""
        assert quality_from_attrs({"quality_degraded": "yes"})["quality_degraded"] is True

    def test_absent_flag_is_not_degraded(self):
        """Absent is different from unreadable: upstream predates the attr."""
        assert quality_from_attrs({})["quality_degraded"] is False

    def test_numpy_scalar_flag_round_trips(self):
        import numpy as np

        assert quality_from_attrs({"quality_degraded": np.int64(1)})["quality_degraded"] is True


# ── Failure handling ──────────────────────────────────────────────────


class TestFailurePropagates:
    @pytest.fixture
    def boom(self, monkeypatch):
        def fake(*args, **kwargs):
            raise RuntimeError("no data for this scan")

        monkeypatch.setattr(ring_adapter, "infer_satellite", fake)

    def test_exception_propagates(self, tmp_path, boom):
        with pytest.raises(RuntimeError, match="no data"):
            _run(tmp_path)

    def test_no_file_left_behind(self, tmp_path, boom):
        with pytest.raises(RuntimeError):
            _run(tmp_path)
        assert not ring_adapter.sat_nc_path(tmp_path, "goes19", T0).exists()
        assert list(tmp_path.rglob("*.nc")) == []

    def test_no_temp_residue_after_failure(self, tmp_path, boom):
        with pytest.raises(RuntimeError):
            _run(tmp_path)
        assert list(tmp_path.rglob(f"*{TMP_SUFFIX}")) == []

    def test_failed_write_leaves_nothing(self, tmp_path, monkeypatch, stub_infer):
        """A crash mid-write must not leave a half-file at the real path."""

        def bad_to_netcdf(self, path, *args, **kwargs):
            Path(path).write_bytes(b"half a netcdf")
            raise OSError("disk full")

        monkeypatch.setattr(xr.Dataset, "to_netcdf", bad_to_netcdf)
        with pytest.raises(OSError, match="disk full"):
            _run(tmp_path)
        assert not ring_adapter.sat_nc_path(tmp_path, "goes19", T0).exists()
        assert list(tmp_path.rglob(f"*{TMP_SUFFIX}")) == []

    def test_interrupt_leaves_nothing(self, tmp_path, monkeypatch, stub_infer):
        def interrupted(self, path, *args, **kwargs):
            Path(path).write_bytes(b"partial")
            raise KeyboardInterrupt

        monkeypatch.setattr(xr.Dataset, "to_netcdf", interrupted)
        with pytest.raises(KeyboardInterrupt):
            _run(tmp_path)
        assert list(tmp_path.rglob("*")) == [tmp_path / "20260801"]

    def test_one_satellite_failing_leaves_the_others(self, tmp_path, monkeypatch):
        def fake(sat_id, t0, *args, **kwargs):
            if sat_id == "goes19":
                raise RuntimeError("goes19 is down")
            return synthetic_scene(sat_id, t0, ny=4, nx=4)

        monkeypatch.setattr(ring_adapter, "infer_satellite", fake)
        good = _run(tmp_path, sat_id="goes18")
        with pytest.raises(RuntimeError):
            _run(tmp_path, sat_id="goes19")
        assert good.path.exists()
        assert not ring_adapter.sat_nc_path(tmp_path, "goes19", T0).exists()


# ── Atomic write ──────────────────────────────────────────────────────


class TestAtomicWrite:
    def _tmp_paths_used(self, tmp_path, monkeypatch, n=2):
        """Run ``n`` retrievals, capturing the temp path each one wrote to."""
        seen: list[Path] = []
        real = xr.Dataset.to_netcdf

        def spy(self, path, *args, **kwargs):
            seen.append(Path(path))
            return real(self, path, *args, **kwargs)

        monkeypatch.setattr(xr.Dataset, "to_netcdf", spy)
        for _ in range(n):
            _run(tmp_path, skip_existing=False)
        return seen

    def test_never_writes_directly_to_the_canonical_path(self, tmp_path, monkeypatch, stub_infer):
        final = ring_adapter.sat_nc_path(tmp_path, "goes19", T0)
        seen = self._tmp_paths_used(tmp_path, monkeypatch, n=1)
        assert seen[0] != final
        assert seen[0].name.endswith(TMP_SUFFIX)

    def test_temp_file_sits_beside_the_final_file(self, tmp_path, monkeypatch, stub_infer):
        """Same directory, so ``os.replace`` stays within one filesystem."""
        final = ring_adapter.sat_nc_path(tmp_path, "goes19", T0)
        seen = self._tmp_paths_used(tmp_path, monkeypatch, n=1)
        assert seen[0].parent == final.parent

    def test_concurrent_attempts_use_distinct_temp_files(self, tmp_path, monkeypatch, stub_infer):
        """Two overlapping attempts must not write over each other."""
        seen = self._tmp_paths_used(tmp_path, monkeypatch, n=2)
        assert len(set(seen)) == 2


class TestStaleTempReaping:
    def _orphan(self, tmp_path, age_s):
        """Plant an abandoned temp file of a given age beside the output."""
        final = ring_adapter.sat_nc_path(tmp_path, "goes19", T0)
        final.parent.mkdir(parents=True, exist_ok=True)
        orphan = final.with_name(f"{final.name}.deadbeef{TMP_SUFFIX}")
        orphan.write_bytes(b"abandoned mid-write")
        stamp = time.time() - age_s
        os.utime(orphan, (stamp, stamp))
        return orphan

    def test_abandoned_temp_is_removed(self, tmp_path, stub_infer):
        """A SIGKILL between write and rename must not leak a full disk."""
        orphan = self._orphan(tmp_path, STALE_TMP_AGE_S + 60)
        _run(tmp_path)
        assert not orphan.exists()

    def test_recent_temp_is_left_alone(self, tmp_path, stub_infer):
        """It may belong to a concurrent attempt still writing it."""
        orphan = self._orphan(tmp_path, 5)
        _run(tmp_path)
        assert orphan.exists()

    def test_reaping_does_not_touch_the_output(self, tmp_path, stub_infer):
        self._orphan(tmp_path, STALE_TMP_AGE_S + 60)
        result = _run(tmp_path)
        assert result.path.exists()
        with xr.open_dataset(result.path) as ds:
            assert ds.attrs["satellite_id"] == "goes19"

    def test_reaping_failure_does_not_fail_the_write(self, tmp_path, monkeypatch, stub_infer):
        orphan = self._orphan(tmp_path, STALE_TMP_AGE_S + 60)
        real_unlink = Path.unlink

        def flaky(self, *args, **kwargs):
            if self == orphan:
                raise OSError("not yours to remove")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", flaky)
        assert _run(tmp_path).path.exists()


# ── Result shape ──────────────────────────────────────────────────────


class TestAmvResult:
    def test_is_frozen(self, tmp_path, stub_infer):
        result = _run(tmp_path)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.sat_id = "other"  # type: ignore[misc]
