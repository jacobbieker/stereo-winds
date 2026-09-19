"""Tests for the mosaic step.

All offline and deliberately tiny: the mosaic runs at 200 km so the
global accumulator is a ~100x200 grid rather than the operational
~2000x4000, and the per-satellite scenes are synthesised rather than
retrieved.

The theme throughout is tolerance: one satellite per asset only buys
anything if the mosaic survives losing one, and says which one it lost.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import weakref
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import xarray as xr

from operational.adapters.ring import global_nc_path
from operational.core.mosaic import (
    EmptyMosaicError,
    build_mosaic,
    missing_satellites,
    write_mosaic_netcdf,
)
from operational.tests.conftest import synthetic_scene

T0 = datetime(2026, 8, 1, 12, 0)
RES = 200_000.0


def _scene(sat_id, zenith=10.0, u=1.0, n=16, lat=(-10.0, 10.0),
           lon=(-10.0, 10.0), t0=T0, **attrs):
    """A synthetic scene with a recognisable ``u_wind`` and viewing angle."""
    ds = synthetic_scene(sat_id, t0, ny=n, nx=n, zenith=zenith,
                         lat_range=lat, lon_range=lon)
    ds["u_wind"].values[:] = np.float32(u)
    ds.attrs.update(attrs)
    return ds


def _filled(ds: xr.Dataset) -> np.ndarray:
    """Boolean mask of grid cells some satellite actually filled."""
    return ds["source_satellite_index"].values >= 0


def _winner_names(ds: xr.Dataset) -> np.ndarray:
    """Satellite name behind every filled cell, ``""`` elsewhere."""
    names = ds["source_satellite_index"].attrs["flag_meanings"].split()
    index = ds["source_satellite_index"].values
    out = np.full(index.shape, "", dtype="U16")
    for code, name in enumerate(names):
        out[index == code] = name
    return out


# ── Merging ───────────────────────────────────────────────────────────

class TestBuildMosaic:
    def test_two_satellites_produce_a_populated_mosaic(self):
        per_sat = {"goes18": _scene("goes18", zenith=20.0),
                   "goes19": _scene("goes19", zenith=10.0)}
        ds = build_mosaic(per_sat, T0, resolution_m=RES)
        assert ds.sizes["latitude"] > 0 and ds.sizes["longitude"] > 0
        assert _filled(ds).any()
        assert set(ds.attrs["satellites"]) == {"goes18", "goes19"}

    @pytest.mark.parametrize("order", [("goes18", "goes19"),
                                       ("goes19", "goes18")])
    def test_lowest_zenith_wins_the_overlap(self, order):
        """The merge rule is min-zenith, not last-writer-wins."""
        scenes = {"goes18": _scene("goes18", zenith=20.0, u=18.0),
                  "goes19": _scene("goes19", zenith=10.0, u=19.0)}
        ds = build_mosaic({k: scenes[k] for k in order}, T0, resolution_m=RES)

        filled = _filled(ds)
        assert filled.any()
        # Both scenes cover the same patch, so goes19 must own all of it.
        assert set(np.unique(_winner_names(ds)[filled])) == {"goes19"}
        assert np.all(ds["u_wind"].values[filled] == np.float32(19.0))

    def test_disjoint_satellites_both_appear(self):
        per_sat = {"goes19": _scene("goes19", u=19.0, lon=(-40.0, -20.0)),
                   "gk2a": _scene("gk2a", u=42.0, lon=(120.0, 140.0))}
        ds = build_mosaic(per_sat, T0, resolution_m=RES)
        winners = _winner_names(ds)[_filled(ds)]
        assert set(np.unique(winners)) == {"goes19", "gk2a"}
        assert ds.attrs["n_satellites_contributing"] == 2
        assert ds.attrs["satellites_missing"] == ""
        assert ds.attrs["mosaic_complete"] == 1

    def test_input_mapping_is_not_consumed_by_default(self):
        """Streaming must not mean quietly emptying the caller's dict."""
        per_sat = {"goes18": _scene("goes18"), "goes19": _scene("goes19")}
        build_mosaic(per_sat, T0, resolution_m=RES)
        assert list(per_sat) == ["goes18", "goes19"]
        assert all(isinstance(v, xr.Dataset) for v in per_sat.values())

    def test_consume_releases_each_scene_as_it_is_gridded(self):
        """Peak memory is the accumulator plus one disk only if we let go."""
        per_sat = {"goes18": _scene("goes18", zenith=20.0, u=18.0),
                   "goes19": _scene("goes19", zenith=10.0, u=19.0)}
        held = [weakref.ref(ds) for ds in per_sat.values()]

        ds = build_mosaic(per_sat, T0, resolution_m=RES, consume=True)

        assert per_sat == {}
        gc.collect()
        assert all(ref() is None for ref in held)
        # Consuming must not change the answer.
        filled = _filled(ds)
        assert set(np.unique(_winner_names(ds)[filled])) == {"goes19"}
        assert ds.attrs["satellites_contributing"] == "goes18,goes19"

    def test_consume_requires_a_mutable_mapping(self):
        frozen = MappingProxyType({"goes19": _scene("goes19")})
        with pytest.raises(TypeError, match="mutable mapping"):
            build_mosaic(frozen, T0, resolution_m=RES, consume=True)

    def test_malformed_scene_is_rejected_by_name(self):
        bad = _scene("gk2a").drop_vars("zenith_angle")
        with pytest.raises(ValueError, match="gk2a.*zenith_angle"):
            build_mosaic({"gk2a": bad}, T0, resolution_m=RES)


# ── Tolerance of absent satellites ────────────────────────────────────

class TestMissingSatellites:
    def test_absent_ids_are_reported_in_expected_order(self):
        assert missing_satellites({"goes19": None},
                                  ["goes18", "goes19", "gk2a"]) == [
            "goes18", "gk2a"]

    def test_nothing_missing_is_an_empty_list(self):
        assert missing_satellites({"a": None, "b": None}, ["a", "b"]) == []

    def test_accepts_a_plain_iterable_of_contributors(self):
        assert missing_satellites(["goes19"], ["goes18", "goes19"]) == ["goes18"]

    def test_duplicated_expectations_are_reported_once(self):
        assert missing_satellites([], ["gk2a", "gk2a"]) == ["gk2a"]

    def test_unexpected_contributors_are_ignored(self):
        assert missing_satellites(["mtg-i1"], ["goes19"]) == ["goes19"]


class TestPartialMosaic:
    def test_a_missing_satellite_still_yields_a_mosaic(self):
        """GK-2A failing must not cost us the other three."""
        per_sat = {"goes18": _scene("goes18", lon=(-150.0, -130.0)),
                   "goes19": _scene("goes19", lon=(-85.0, -65.0))}
        ds = build_mosaic(per_sat, T0, resolution_m=RES,
                          expected=["goes18", "goes19", "himawari9", "gk2a"])

        assert _filled(ds).any()
        assert ds.attrs["satellites_missing"] == "himawari9,gk2a"
        assert ds.attrs["satellites_contributing"] == "goes18,goes19"
        assert ds.attrs["n_satellites_expected"] == 4
        assert ds.attrs["n_satellites_missing"] == 2
        assert ds.attrs["mosaic_complete"] == 0

    def test_expected_defaults_to_what_was_handed_over(self):
        ds = build_mosaic({"goes19": _scene("goes19")}, T0, resolution_m=RES)
        assert ds.attrs["satellites_expected"] == "goes19"
        assert ds.attrs["satellites_missing"] == ""

    def test_empty_mapping_raises_rather_than_gridding_nothing(self):
        with pytest.raises(EmptyMosaicError, match="nothing to mosaic"):
            build_mosaic({}, T0, resolution_m=RES)

    def test_all_satellites_empty_raises_rather_than_an_all_nan_grid(self):
        """An all-NaN mosaic would be published and read as real data."""
        per_sat = {}
        for sat_id in ("goes18", "goes19"):
            blank = _scene(sat_id)
            blank["quality_flag"].values[:] = 0.0
            per_sat[sat_id] = blank
        with pytest.raises(EmptyMosaicError,
                           match="goes18, goes19.*usable grid cell"):
            build_mosaic(per_sat, T0, resolution_m=RES)

    def test_a_satellite_with_no_valid_pixels_counts_as_missing(self):
        """A dataset that arrived but is all-bad is not a contributor."""
        blank = _scene("gk2a")
        blank["quality_flag"].values[:] = 0.0
        per_sat = {"goes19": _scene("goes19", u=19.0), "gk2a": blank}
        ds = build_mosaic(per_sat, T0, resolution_m=RES)

        assert ds.attrs["satellites_contributing"] == "goes19"
        assert ds.attrs["satellites_empty"] == "gk2a"
        assert ds.attrs["satellites_missing"] == "gk2a"
        assert "gk2a" not in list(ds.attrs["satellites"])
        assert set(np.unique(_winner_names(ds)[_filled(ds)])) == {"goes19"}


# ── Attributes ────────────────────────────────────────────────────────

class TestMosaicAttributes:
    def test_upstream_attributes_are_preserved(self):
        ds = build_mosaic({"goes19": _scene("goes19")}, T0, resolution_m=RES)
        assert ds.attrs["resolution_m"] == RES
        assert "zenith" in ds.attrs["merge_rule"]
        assert list(ds.attrs["satellites"]) == ["goes19"]
        assert ds.attrs["quality_degraded"] == 0
        assert "all contributing satellites" in ds.attrs["quality_note"]

    def test_upstream_degraded_quality_survives(self):
        """Band-shortfall reporting is upstream's; we must not clobber it."""
        ds = build_mosaic(
            {"mtg-i1": _scene("mtg-i1", bands_missing="C08,C09,C10",
                              n_bands_missing=3, n_bands_requested=15,
                              quality_degraded=1)},
            T0, resolution_m=RES,
        )
        assert ds.attrs["quality_degraded"] == 1
        assert ds.attrs["degraded_satellites"] == "mtg-i1"
        assert "DEGRADED" in ds.attrs["quality_note"]

    def test_scene_time_is_kept_and_nominal_time_added(self):
        ds = build_mosaic({"goes19": _scene("goes19")}, T0, resolution_m=RES)
        assert ds.attrs["time"] == T0.isoformat()
        assert ds.attrs["nominal_time"] == T0.isoformat()

    def test_time_falls_back_to_t0_when_no_scene_carries_one(self):
        """A None ``time`` attribute cannot be written to NetCDF."""
        scene = _scene("goes19")
        del scene.attrs["time"]
        ds = build_mosaic({"goes19": scene}, T0, resolution_m=RES)
        assert ds.attrs["time"] == T0.isoformat()


# ── NetCDF output ─────────────────────────────────────────────────────

class TestWriteMosaicNetcdf:
    def test_lands_on_the_canonical_path(self, tmp_path):
        ds = build_mosaic({"goes19": _scene("goes19")}, T0, resolution_m=RES)
        path = write_mosaic_netcdf(ds, tmp_path, T0)
        assert path == global_nc_path(tmp_path, T0)
        assert path.name == "student_amv_global_20260801T1200.nc"
        assert path.parent.name == "20260801"
        assert path.exists()

    def test_round_trips_through_xarray(self, tmp_path):
        per_sat = {"goes18": _scene("goes18", zenith=20.0, u=18.0),
                   "goes19": _scene("goes19", zenith=10.0, u=19.0)}
        ds = build_mosaic(per_sat, T0, resolution_m=RES,
                          expected=["goes18", "goes19", "gk2a"])
        path = write_mosaic_netcdf(ds, tmp_path, T0)

        with xr.open_dataset(path) as back:
            assert back["source_satellite_index"].dtype == np.int8
            filled = back["source_satellite_index"].values >= 0
            np.testing.assert_array_equal(
                back["u_wind"].values[filled],
                ds["u_wind"].values[filled])
            assert back.attrs["satellites_missing"] == "gk2a"
            assert back.attrs["nominal_time"] == T0.isoformat()
            assert back.attrs["quality_degraded"] == 0

    def test_empty_missing_list_survives_the_round_trip(self, tmp_path):
        """Comma-joined strings, so "nothing missing" stays a string."""
        ds = build_mosaic({"goes19": _scene("goes19")}, T0, resolution_m=RES)
        with xr.open_dataset(write_mosaic_netcdf(ds, tmp_path, T0)) as back:
            assert back.attrs["satellites_missing"] == ""

    def test_creates_the_day_directory(self, tmp_path):
        out = tmp_path / "deep" / "output"
        ds = build_mosaic({"goes19": _scene("goes19")}, T0, resolution_m=RES)
        assert write_mosaic_netcdf(ds, out, T0).exists()

    def test_leaves_no_temporary_residue(self, tmp_path):
        ds = build_mosaic({"goes19": _scene("goes19")}, T0, resolution_m=RES)
        path = write_mosaic_netcdf(ds, tmp_path, T0)
        assert sorted(p.name for p in path.parent.iterdir()) == [path.name]

    def test_a_failed_write_leaves_nothing_behind(self, tmp_path, monkeypatch):
        ds = build_mosaic({"goes19": _scene("goes19")}, T0, resolution_m=RES)

        def boom(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(xr.Dataset, "to_netcdf", boom)
        with pytest.raises(RuntimeError, match="disk full"):
            write_mosaic_netcdf(ds, tmp_path, T0)

        day_dir = global_nc_path(tmp_path, T0).parent
        assert list(day_dir.iterdir()) == []

    def test_overwrites_an_existing_mosaic_atomically(self, tmp_path):
        first = build_mosaic({"goes18": _scene("goes18", u=18.0)}, T0,
                             resolution_m=RES)
        path = write_mosaic_netcdf(first, tmp_path, T0)
        second = build_mosaic({"goes19": _scene("goes19", u=19.0)}, T0,
                              resolution_m=RES)
        assert write_mosaic_netcdf(second, tmp_path, T0) == path

        with xr.open_dataset(path) as back:
            assert back.attrs["satellites_contributing"] == "goes19"
        assert sorted(p.name for p in path.parent.iterdir()) == [path.name]

    def test_accepts_a_string_output_dir(self, tmp_path):
        ds = build_mosaic({"goes19": _scene("goes19")}, T0, resolution_m=RES)
        assert write_mosaic_netcdf(ds, str(tmp_path), T0) == (
            global_nc_path(Path(tmp_path), T0))


# ── Import cost ───────────────────────────────────────────────────────

class TestLazyRingImport:
    """A mosaicing worker should not have to carry the inference stack."""

    def test_importing_the_step_does_not_execute_the_ring_script(self):
        repo_root = Path(__file__).resolve().parents[2]
        env = dict(os.environ, PYTHONPATH=str(repo_root))
        probe = (
            "import sys; import operational.core.mosaic as m; "
            "print('operational_ring' in sys.modules, 'torch' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe], cwd=repo_root, env=env,
            capture_output=True, text=True, timeout=300,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.split() == ["False", "False"]
