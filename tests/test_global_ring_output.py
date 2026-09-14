"""Tests for the global ring script's time range, naming and output sinks.

These are offline: ``infer_satellite`` is replaced with a synthetic
dataset so the range walking, deterministic file layout and icechunk
append/resume behaviour can be exercised without touching the network.
"""

import gc
import importlib.util
import sys
import weakref
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

BASE = Path(__file__).resolve().parent.parent


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "infer_student_global_ring",
        BASE / "scripts" / "infer_student_global_ring.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ring = _load_script()

T0 = datetime(2026, 8, 1, 0, 0)


def _fake_scene(sat_id: str, t0: datetime, ny: int = 8, nx: int = 8):
    """A tiny synthetic per-satellite AMV dataset over a small lat/lon patch."""
    lat, lon = np.meshgrid(
        np.linspace(-20, 20, ny), np.linspace(-30, 30, nx), indexing="ij",
    )
    zenith = {"goes18": 10.0, "goes19": 20.0}[sat_id]
    data = {k: np.full((ny, nx), 1.0, np.float32) for k in ring.OUTPUT_VARS}
    data["quality_flag"] = np.full((ny, nx), 2.0, np.float32)
    return xr.Dataset(
        {k: (("y", "x"), data[k]) for k in ring.OUTPUT_VARS},
        coords={
            "latitude": (("y", "x"), lat.astype(np.float32)),
            "longitude": (("y", "x"), lon.astype(np.float32)),
            "zenith_angle": (("y", "x"), np.full((ny, nx), zenith, np.float32)),
        },
        attrs={"satellite_id": sat_id, "time": str(t0)},
    )


@pytest.fixture
def stub_infer(monkeypatch):
    """Replace infer_satellite; record every (satellite, time) it is asked for."""
    calls: list[tuple[str, datetime]] = []

    def fake(sat_id, t0, model, disp, flow_bands, rad_bands, **kwargs):
        calls.append((sat_id, t0))
        return _fake_scene(sat_id, t0)

    monkeypatch.setattr(ring, "infer_satellite", fake)
    return calls


def _run(t, out_dir, calls_sats=("goes18", "goes19"), **kwargs):
    kwargs.setdefault("resolution_m", 200_000.0)
    return ring.process_time(
        t, list(calls_sats), None, None, ["C14"], ["C14"], out_dir, **kwargs,
    )


# ── Time range ────────────────────────────────────────────────────────

class TestTimeSteps:
    def test_single_time_when_no_end(self):
        assert ring.time_steps(T0, None, 10) == [T0]

    def test_single_time_when_end_equals_start(self):
        assert ring.time_steps(T0, T0, 10) == [T0]

    def test_inclusive_of_both_ends(self):
        times = ring.time_steps(T0, T0 + timedelta(hours=1), 30)
        assert times == [T0, T0 + timedelta(minutes=30), T0 + timedelta(hours=1)]

    def test_partial_trailing_step_dropped(self):
        times = ring.time_steps(T0, T0 + timedelta(minutes=55), 30)
        assert times == [T0, T0 + timedelta(minutes=30)]

    def test_backwards_range_raises(self):
        with pytest.raises(ValueError, match="before"):
            ring.time_steps(T0, T0 - timedelta(hours=1), 10)

    def test_non_positive_step_raises(self):
        with pytest.raises(ValueError, match="positive"):
            ring.time_steps(T0, T0 + timedelta(hours=1), 0)


class TestOutputPaths:
    def test_deterministic_and_day_partitioned(self):
        out = Path("/out")
        assert ring.sat_nc_path(out, "goes19", T0) == (
            out / "20260801" / "student_amv_goes19_20260801T0000.nc")
        assert ring.global_nc_path(out, T0) == (
            out / "20260801" / "student_amv_global_20260801T0000.nc")

    def test_day_boundary_splits_directories(self):
        late = datetime(2026, 8, 1, 23, 50)
        assert ring.day_dir(Path("/out"), late).name == "20260801"
        assert ring.day_dir(Path("/out"), late + timedelta(minutes=10)).name == "20260802"


# ── NetCDF output ─────────────────────────────────────────────────────

class TestNetcdfOutput:
    def test_writes_expected_files(self, tmp_path, stub_infer):
        assert _run(T0, tmp_path) is True
        written = sorted(p.relative_to(tmp_path).as_posix()
                         for p in tmp_path.rglob("*.nc"))
        assert written == [
            "20260801/student_amv_global_20260801T0000.nc",
            "20260801/student_amv_goes18_20260801T0000.nc",
            "20260801/student_amv_goes19_20260801T0000.nc",
        ]

    def test_skip_existing_recomputes_nothing(self, tmp_path, stub_infer):
        _run(T0, tmp_path)
        before = len(stub_infer)
        _run(T0, tmp_path, skip_existing=True)
        assert len(stub_infer) == before

    def test_failed_satellite_does_not_abort_timestamp(self, tmp_path, monkeypatch):
        def flaky(sat_id, t0, *a, **k):
            if sat_id == "goes18":
                raise RuntimeError("simulated loader failure")
            return _fake_scene(sat_id, t0)

        monkeypatch.setattr(ring, "infer_satellite", flaky)
        assert _run(T0, tmp_path) is True
        assert ring.sat_nc_path(tmp_path, "goes19", T0).exists()
        assert not ring.sat_nc_path(tmp_path, "goes18", T0).exists()

    def test_all_satellites_failing_returns_false(self, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("simulated loader failure")

        monkeypatch.setattr(ring, "infer_satellite", boom)
        assert _run(T0, tmp_path) is False
        assert list(tmp_path.rglob("*.nc")) == []


# ── Icechunk output ───────────────────────────────────────────────────

@pytest.fixture
def repo(tmp_path):
    pytest.importorskip("icechunk")
    return ring.open_icechunk_repo(str(tmp_path / "store"))


def _store_ds(repo):
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False)


class TestIcechunkOutput:
    def test_appends_one_timestamp_per_commit(self, tmp_path, repo, stub_infer):
        times = ring.time_steps(T0, T0 + timedelta(minutes=20), 10)
        seen: set[datetime] = set()
        for t in times:
            _run(t, tmp_path, repo=repo, icechunk_times=seen, write_netcdf=False)
        ds = _store_ds(repo)
        assert ds.sizes["time"] == 3
        assert list(ds.data_vars) != []

    def test_appended_times_are_not_corrupted(self, tmp_path, repo, stub_infer):
        """Regression: without pinned time encoding, appends land on wrong dates."""
        times = ring.time_steps(T0, T0 + timedelta(minutes=30), 10)
        seen: set[datetime] = set()
        for t in times:
            _run(t, tmp_path, repo=repo, icechunk_times=seen, write_netcdf=False)
        stored = [pd.Timestamp(v).to_pydatetime() for v in _store_ds(repo).time.values]
        assert stored == times

    def test_resume_skips_committed_timestamps(self, tmp_path, repo, stub_infer):
        first = ring.time_steps(T0, T0 + timedelta(minutes=10), 10)
        seen: set[datetime] = set()
        for t in first:
            _run(t, tmp_path, repo=repo, icechunk_times=seen, write_netcdf=False)
        n_before = len(stub_infer)

        # A fresh handle, as a restarted process would have.
        resumed = ring.icechunk_existing_times(repo)
        assert resumed == set(first)

        extended = ring.time_steps(T0, T0 + timedelta(minutes=30), 10)
        for t in extended:
            _run(t, tmp_path, repo=repo, icechunk_times=resumed,
                 write_netcdf=False, skip_existing=True)

        # Only the two new timestamps were computed, for two satellites each.
        assert len(stub_infer) - n_before == 4
        stored = [pd.Timestamp(v).to_pydatetime() for v in _store_ds(repo).time.values]
        assert stored == extended
        assert len(stored) == len(set(stored))

    def test_duplicate_timestamp_is_not_appended(self, tmp_path, repo, stub_infer):
        seen: set[datetime] = set()
        _run(T0, tmp_path, repo=repo, icechunk_times=seen, write_netcdf=False)
        _run(T0, tmp_path, repo=repo, icechunk_times=seen, write_netcdf=False)
        assert _store_ds(repo).sizes["time"] == 1

    def test_empty_store_reports_no_times(self, repo):
        assert ring.icechunk_existing_times(repo) == set()

    def test_chunks_are_bounded(self, tmp_path, repo, stub_infer):
        _run(T0, tmp_path, repo=repo, icechunk_times=set(),
             write_netcdf=False, icechunk_chunk=64)
        chunks = _store_ds(repo).u_wind.encoding["chunks"]
        assert chunks[0] == 1
        assert max(chunks[1:]) <= 64


class TestIcechunkStorage:
    def test_s3_uri_parsed(self):
        pytest.importorskip("icechunk")
        # Builds Storage without contacting S3.
        assert ring.icechunk_storage("s3://bucket/some/prefix") is not None

    def test_rejects_unknown_scheme(self):
        with pytest.raises(ValueError, match="Unsupported"):
            ring.icechunk_storage("gs://bucket/prefix")


# ── Common-time filtering ─────────────────────────────────────────────

def _t64(*times):
    return np.array([np.datetime64(t, "ns") for t in times])


class TestHasScanNear:
    def test_exact_match(self):
        times = _t64(datetime(2026, 8, 1, 0, 0), datetime(2026, 8, 1, 0, 10))
        assert ring._has_scan_near(times, datetime(2026, 8, 1, 0, 10),
                                   timedelta(minutes=5))

    def test_within_tolerance(self):
        """AMI stamps scans at HH:09:35, so slots never line up exactly."""
        times = _t64(datetime(2026, 8, 1, 0, 9, 35))
        assert ring._has_scan_near(times, datetime(2026, 8, 1, 0, 10),
                                   timedelta(minutes=5))

    def test_outside_tolerance(self):
        times = _t64(datetime(2026, 8, 1, 0, 0), datetime(2026, 8, 1, 0, 30))
        assert not ring._has_scan_near(times, datetime(2026, 8, 1, 0, 15),
                                       timedelta(minutes=5))

    def test_before_first_scan(self):
        times = _t64(datetime(2026, 8, 1, 12, 0))
        assert not ring._has_scan_near(times, datetime(2026, 8, 1, 0, 0),
                                       timedelta(minutes=5))

    def test_empty_is_never_available(self):
        assert not ring._has_scan_near(
            np.array([], dtype="datetime64[ns]"),
            datetime(2026, 8, 1, 0, 0), timedelta(minutes=5))


class TestFilterToCommonTimes:
    """The filter needs a full t-10/t/t+10 triplet from every satellite."""

    @pytest.fixture
    def stub_availability(self, monkeypatch):
        table: dict[str, np.ndarray] = {}

        def fake(sat_id, band, start, end, product="ABI-L1b-RadF"):
            return table[sat_id]

        monkeypatch.setattr(ring, "satellite_available_times", fake)
        return table

    def _every_10min(self, n=10, skip=()):
        base = datetime(2026, 8, 1, 0, 0)
        return _t64(*[base + timedelta(minutes=10 * i)
                      for i in range(n) if i not in skip])

    def test_all_present_keeps_everything(self, stub_availability):
        stub_availability["goes19"] = self._every_10min()
        stub_availability["gk2a"] = self._every_10min()
        candidates = [datetime(2026, 8, 1, 0, 10), datetime(2026, 8, 1, 0, 20)]
        kept = ring.filter_to_common_times(
            candidates, ["goes19", "gk2a"], ["C14"], ["C14"])
        assert kept == candidates

    def test_missing_neighbour_frame_drops_timestamp(self, stub_availability):
        # gk2a is missing the 00:30 scan, so t=00:20 loses its t+10 frame
        # and t=00:30 and t=00:40 lose frames too.
        stub_availability["goes19"] = self._every_10min()
        stub_availability["gk2a"] = self._every_10min(skip=(3,))
        candidates = [datetime(2026, 8, 1, 0, 10), datetime(2026, 8, 1, 0, 20),
                      datetime(2026, 8, 1, 0, 30), datetime(2026, 8, 1, 0, 40),
                      datetime(2026, 8, 1, 0, 50)]
        kept = ring.filter_to_common_times(
            candidates, ["goes19", "gk2a"], ["C14"], ["C14"])
        assert kept == [datetime(2026, 8, 1, 0, 10), datetime(2026, 8, 1, 0, 50)]

    def test_satellite_with_no_data_drops_all(self, stub_availability):
        stub_availability["goes19"] = self._every_10min()
        stub_availability["gk2a"] = np.array([], dtype="datetime64[ns]")
        kept = ring.filter_to_common_times(
            [datetime(2026, 8, 1, 0, 10)], ["goes19", "gk2a"], ["C14"], ["C14"])
        assert kept == []

    def test_offset_scan_schedule_still_matches(self, stub_availability):
        """GK-2A's HH:09:35 stamps must count for the HH:10 slot."""
        base = datetime(2026, 8, 1, 0, 0, 0)
        stub_availability["goes19"] = self._every_10min()
        stub_availability["gk2a"] = _t64(
            *[base + timedelta(minutes=10 * i, seconds=-25) for i in range(1, 6)])
        kept = ring.filter_to_common_times(
            [datetime(2026, 8, 1, 0, 20)], ["goes19", "gk2a"], ["C14"], ["C14"])
        assert kept == [datetime(2026, 8, 1, 0, 20)]

    def test_tolerance_is_respected(self, stub_availability):
        base = datetime(2026, 8, 1, 0, 0)
        stub_availability["goes19"] = self._every_10min()
        # Offset by 4 minutes: inside a 5 min tolerance, outside a 2 min one.
        stub_availability["gk2a"] = _t64(
            *[base + timedelta(minutes=10 * i + 4) for i in range(5)])
        candidates = [datetime(2026, 8, 1, 0, 20)]
        assert ring.filter_to_common_times(
            candidates, ["goes19", "gk2a"], ["C14"], ["C14"],
            tolerance_min=5.0) == candidates
        assert ring.filter_to_common_times(
            candidates, ["goes19", "gk2a"], ["C14"], ["C14"],
            tolerance_min=2.0) == []

    def test_band_the_satellite_lacks_is_not_used(self, monkeypatch):
        """MTG has no C14; availability must fall back to a band it carries."""
        asked: list[str] = []

        def fake(sat_id, band, start, end, product="ABI-L1b-RadF"):
            asked.append(band)
            return _t64(*[datetime(2026, 8, 1, 0, 10 * i) for i in range(5)])

        monkeypatch.setattr(ring, "satellite_available_times", fake)
        ring.filter_to_common_times(
            [datetime(2026, 8, 1, 0, 20)], ["mtg-i1"], ["C14", "C10"], ["C14"])
        assert asked == ["C10"]

    def test_satellite_with_no_usable_band_raises(self):
        with pytest.raises(RuntimeError, match="none of the requested bands"):
            ring.filter_to_common_times(
                [datetime(2026, 8, 1, 0, 20)], ["mtg-i1"], ["C14"], ["C14"])

    def test_empty_candidate_list_is_passed_through(self):
        assert ring.filter_to_common_times([], ["goes19"], ["C14"], ["C14"]) == []


class TestAvailabilityBand:
    def test_prefers_first_available(self):
        assert ring.availability_band("goes19", ["C14"], ["C02"]) == "C14"

    def test_skips_bands_the_satellite_lacks(self):
        assert ring.availability_band("mtg-i1", ["C14"], ["C10"]) == "C10"

    def test_none_when_nothing_matches(self):
        assert ring.availability_band("mtg-i1", ["C14"], ["C14"]) is None


# ── Mosaic memory behaviour ───────────────────────────────────────────

class TestGlobalMosaic:
    """The mosaic accumulates satellites one at a time and stores codes."""

    def _sat(self, sat_id, zenith, lon0=0.0, n=6):
        lat, lon = np.meshgrid(np.linspace(-10, 10, n),
                               np.linspace(lon0 - 10, lon0 + 10, n),
                               indexing="ij")
        data = {v: np.full((n, n), 1.0, np.float32) for v in ring.OUTPUT_VARS}
        data["quality_flag"] = np.full((n, n), 2.0, np.float32)
        return xr.Dataset(
            {v: (("y", "x"), data[v]) for v in ring.OUTPUT_VARS},
            coords={"latitude": (("y", "x"), lat.astype(np.float32)),
                    "longitude": (("y", "x"), lon.astype(np.float32)),
                    "zenith_angle": (("y", "x"),
                                     np.full((n, n), zenith, np.float32))},
            attrs={"satellite_id": sat_id, "time": "2026-08-01T00:00:00"},
        )

    def test_source_is_stored_as_int8_codes(self):
        """A U12 string array costs 9 GB on the 2 km grid; codes cost 0.2."""
        mosaic = ring.GlobalMosaic(resolution_m=200_000.0)
        mosaic.add("goes19", self._sat("goes19", 10.0))
        ds = mosaic.to_dataset()
        assert ds["source_satellite_index"].dtype == np.int8

    def test_unfilled_cells_carry_the_sentinel(self):
        mosaic = ring.GlobalMosaic(resolution_m=200_000.0)
        mosaic.add("goes19", self._sat("goes19", 10.0))
        ds = mosaic.to_dataset()
        assert (ds["source_satellite_index"].values == ring.NO_SOURCE).any()
        assert ds["source_satellite_index"].attrs["no_source_index"] == ring.NO_SOURCE

    def test_decodes_back_to_names(self):
        mosaic = ring.GlobalMosaic(resolution_m=200_000.0)
        mosaic.add("goes19", self._sat("goes19", 10.0))
        mosaic.add("goes18", self._sat("goes18", 20.0, lon0=40.0))
        names = ring.decode_source_satellite(mosaic.to_dataset())
        assert set(names[names != ""]) == {"goes18", "goes19"}

    def test_lowest_zenith_wins_the_overlap(self):
        mosaic = ring.GlobalMosaic(resolution_m=200_000.0)
        mosaic.add("far", self._sat("far", 40.0))
        mosaic.add("near", self._sat("near", 5.0))      # same footprint
        names = ring.decode_source_satellite(mosaic.to_dataset())
        assert set(names[names != ""]) == {"near"}

    def test_order_does_not_matter(self):
        a = ring.GlobalMosaic(resolution_m=200_000.0)
        a.add("far", self._sat("far", 40.0))
        a.add("near", self._sat("near", 5.0))
        b = ring.GlobalMosaic(resolution_m=200_000.0)
        b.add("near", self._sat("near", 5.0))
        b.add("far", self._sat("far", 40.0))
        assert np.array_equal(
            ring.decode_source_satellite(a.to_dataset()),
            ring.decode_source_satellite(b.to_dataset()))

    def test_matches_the_dict_api(self):
        """merge_global is the same accumulator, fed from a dict."""
        per_sat = {"far": self._sat("far", 40.0), "near": self._sat("near", 5.0)}
        merged = ring.merge_global(per_sat, resolution_m=200_000.0)
        mosaic = ring.GlobalMosaic(resolution_m=200_000.0)
        for k, v in per_sat.items():
            mosaic.add(k, v)
        streamed = mosaic.to_dataset()
        for var in ring.OUTPUT_VARS:
            assert np.array_equal(merged[var].values, streamed[var].values,
                                  equal_nan=True)

    def test_empty_satellite_is_skipped(self):
        mosaic = ring.GlobalMosaic(resolution_m=200_000.0)
        blank = self._sat("blank", 10.0)
        blank["quality_flag"][:] = 0.0
        assert mosaic.add("blank", blank) == 0
        assert mosaic.to_dataset().attrs["satellites"] == []

    def test_netcdf_roundtrip_keeps_int8(self, tmp_path):
        """_FillValue would make xarray mask on read and promote to float."""
        mosaic = ring.GlobalMosaic(resolution_m=200_000.0)
        mosaic.add("goes19", self._sat("goes19", 10.0))
        path = tmp_path / "mosaic.nc"
        mosaic.to_dataset().to_netcdf(path)
        back = xr.open_dataset(path)
        assert back["source_satellite_index"].dtype == np.int8
        assert set(ring.decode_source_satellite(back).ravel()) >= {"goes19"}


class TestStreamingProcess:
    """process_time must not retain satellites once they are gridded."""

    def test_datasets_are_released_after_gridding(self, tmp_path, monkeypatch):
        alive: list = []

        def fake(sat_id, t0, *a, **k):
            ds = _fake_scene(sat_id, t0)
            alive.append(weakref.ref(ds))
            return ds

        monkeypatch.setattr(ring, "infer_satellite", fake)
        _run(T0, tmp_path, resolution_m=200_000.0, write_netcdf=False)
        gc.collect()
        leaked = [r for r in alive if r() is not None]
        assert alive, "no satellites were processed"
        assert not leaked, f"{len(leaked)} satellite dataset(s) still resident"
