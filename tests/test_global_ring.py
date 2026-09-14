"""Integration tests for the global geostationary ring student AMV script.

These tests load real satellite data from icechunk for 2026-06-01, run
the student model, and verify the output datasets.  They require
network access, both checkpoints, and are slow — gated behind the
``integration_live`` marker::

    pixi run python -m pytest tests/test_global_ring.py -v -m integration_live
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))

from stereo_winds.config import SATELLITE_CONFIGS
from stereo_winds.student_dataset import DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS

STUDENT_CKPT = BASE / "checkpoints" / "student.abi.mb-v3.ep21.ckpt"
RAFT_CKPT = BASE / "checkpoints" / "windflow.raft.sonde-tuned.ckpt"

T0 = datetime(2026, 6, 1, 12, 0)

OUTPUT_VARS = [
    "u_wind", "v_wind", "cloud_top_height",
    "quality_flag", "sigma_u", "sigma_v", "sigma_h",
]


def _skip_if_no_checkpoints():
    if not STUDENT_CKPT.exists():
        pytest.skip(f"Student checkpoint not found: {STUDENT_CKPT}")
    if not RAFT_CKPT.exists():
        pytest.skip(f"RAFT checkpoint not found: {RAFT_CKPT}")


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    _skip_if_no_checkpoints()
    from stereo_winds.student_zeus_model import StudentWindsModel
    return StudentWindsModel.load_from_checkpoint(
        str(STUDENT_CKPT), map_location=_device(),
    ).eval()


@pytest.fixture(scope="module")
def disp():
    _skip_if_no_checkpoints()
    from stereo_winds.disparity import StereoDisparity
    return StereoDisparity(
        model_ckpt_path=str(RAFT_CKPT),
        tile_size=512, overlap=128, batch_size=8,
        device=_device(),
    )


# ── Per-satellite inference tests ─────────────────────────────────────

@pytest.mark.integration_live
class TestInferSatellite:
    """Run infer_satellite on each satellite for 2026-06-01T12:00."""

    def _check_dataset(self, ds: xr.Dataset, sat_id: str):
        """Validate the structure and content of a per-satellite dataset."""
        # All output variables present
        for v in OUTPUT_VARS:
            assert v in ds.data_vars, f"Missing variable {v}"

        # Correct dimensions
        assert set(ds.dims) == {"y", "x"}

        # Coordinates present
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords
        assert "zenith_angle" in ds.coords

        # Attrs
        assert ds.attrs["satellite_id"] == sat_id
        assert "time" in ds.attrs

        # Some valid pixels (quality_flag >= 2)
        qf = ds["quality_flag"].values
        n_valid = int((qf >= 2).sum())
        assert n_valid > 0, f"{sat_id}: no valid AMV pixels"

        # Wind values are physically reasonable where valid
        valid = qf >= 2
        u = ds["u_wind"].values[valid]
        v = ds["v_wind"].values[valid]
        speed = np.sqrt(u**2 + v**2)
        assert np.nanmedian(speed) < 100, f"{sat_id}: median speed unreasonable"

        # Heights are non-negative where valid
        h = ds["cloud_top_height"].values[valid]
        assert np.nanmin(h) >= -1000, f"{sat_id}: negative heights"
        assert np.nanmax(h) < 25000, f"{sat_id}: heights above 25 km"

        # Uncertainties are positive where valid
        for s in ("sigma_u", "sigma_v", "sigma_h"):
            vals = ds[s].values[valid]
            assert np.all(vals[np.isfinite(vals)] > 0), f"{sat_id}: non-positive {s}"

    def test_goes18(self, model, disp):
        from infer_student_global_ring import infer_satellite
        ds = infer_satellite(
            "goes18", T0, model, disp,
            DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS,
            device=_device(),
        )
        self._check_dataset(ds, "goes18")

    def test_goes19(self, model, disp):
        from infer_student_global_ring import infer_satellite
        ds = infer_satellite(
            "goes19", T0, model, disp,
            DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS,
            device=_device(),
        )
        self._check_dataset(ds, "goes19")

    def test_himawari9(self, model, disp):
        from infer_student_global_ring import infer_satellite
        ds = infer_satellite(
            "himawari9", T0, model, disp,
            DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS,
            device=_device(),
        )
        self._check_dataset(ds, "himawari9")

    def test_gk2a(self, model, disp):
        from infer_student_global_ring import infer_satellite
        ds = infer_satellite(
            "gk2a", T0, model, disp,
            DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS,
            device=_device(),
        )
        self._check_dataset(ds, "gk2a")

    def test_mtg_i1(self, model, disp):
        from infer_student_global_ring import infer_satellite
        ds = infer_satellite(
            "mtg-i1", T0, model, disp,
            DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS,
            device=_device(),
        )
        self._check_dataset(ds, "mtg-i1")


# ── Global merge tests ────────────────────────────────────────────────

@pytest.mark.integration_live
class TestMergeGlobal:
    """Test the global mosaic merge with synthetic per-satellite datasets."""

    def test_merge_synthetic(self):
        """Merge two fake satellite datasets and verify zenith-priority rule."""
        from infer_student_global_ring import (
            decode_source_satellite, merge_global,
        )

        ny, nx = 50, 50
        rng = np.random.default_rng(42)

        def _make_ds(sat_id: str, sub_lon: float, zen_base: float):
            lat = np.linspace(-10, 10, ny)[:, None] * np.ones((1, nx))
            lon = np.linspace(sub_lon - 20, sub_lon + 20, nx)[None, :] * np.ones((ny, 1))
            zen = np.full((ny, nx), zen_base, dtype=np.float32)
            u = rng.standard_normal((ny, nx)).astype(np.float32) * 5
            v = rng.standard_normal((ny, nx)).astype(np.float32) * 5
            h = rng.uniform(2000, 12000, (ny, nx)).astype(np.float32)
            qf = np.full((ny, nx), 2.0, dtype=np.float32)
            return xr.Dataset(
                {
                    "u_wind": (("y", "x"), u),
                    "v_wind": (("y", "x"), v),
                    "cloud_top_height": (("y", "x"), h),
                    "quality_flag": (("y", "x"), qf),
                    "sigma_u": (("y", "x"), np.ones((ny, nx), np.float32)),
                    "sigma_v": (("y", "x"), np.ones((ny, nx), np.float32)),
                    "sigma_h": (("y", "x"), np.ones((ny, nx), np.float32) * 500),
                },
                coords={
                    "latitude": (("y", "x"), lat.astype(np.float32)),
                    "longitude": (("y", "x"), lon.astype(np.float32)),
                    "zenith_angle": (("y", "x"), zen),
                },
                attrs={"satellite_id": sat_id, "time": "2026-06-01T12:00"},
            )

        # sat_a has zenith=20, sat_b has zenith=40 — sat_a should win in overlap
        ds_a = _make_ds("sat_a", sub_lon=0.0, zen_base=20.0)
        ds_b = _make_ds("sat_b", sub_lon=10.0, zen_base=40.0)

        # Use a coarse grid to keep the test fast
        merged = merge_global({"sat_a": ds_a, "sat_b": ds_b},
                              resolution_m=100_000.0)

        assert "latitude" in merged.coords
        assert "longitude" in merged.coords
        assert "source_satellite_index" in merged.data_vars

        # In the overlap region, sat_a (lower zenith) should dominate
        src = decode_source_satellite(merged)
        n_a = (src == "sat_a").sum()
        n_b = (src == "sat_b").sum()
        assert n_a > 0, "sat_a should contribute some cells"
        assert n_b > 0, "sat_b should contribute cells outside sat_a coverage"

        # In the center of the overlap (lon ~ 0–10°), sat_a should win
        lon_mask = (merged.longitude.values > -5) & (merged.longitude.values < 15)
        lat_mask = (merged.latitude.values > -5) & (merged.latitude.values < 5)
        overlap_src = src[np.ix_(lat_mask, lon_mask)]
        overlap_filled = overlap_src[overlap_src != ""]
        if len(overlap_filled) > 0:
            a_frac = (overlap_filled == "sat_a").sum() / len(overlap_filled)
            assert a_frac > 0.8, (
                f"sat_a should dominate overlap (lower zenith), got {a_frac:.0%}")

    def test_merge_two_real_satellites(self, model, disp):
        """Run inference on two real satellites and merge them."""
        from infer_student_global_ring import (
            decode_source_satellite, infer_satellite, merge_global,
        )

        # Use two satellites with nearby longitudes for overlap
        per_sat = {}
        for sat_id in ("goes18", "goes19"):
            ds = infer_satellite(
                sat_id, T0, model, disp,
                DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS,
                device=_device(),
            )
            per_sat[sat_id] = ds

        # Merge at a coarser grid to keep the test fast
        merged = merge_global(per_sat, resolution_m=50_000.0)

        assert "u_wind" in merged.data_vars
        assert "source_satellite_index" in merged.data_vars

        # Both satellites should contribute
        src = decode_source_satellite(merged)
        sats_present = set(src[src != ""])
        assert len(sats_present) == 2, (
            f"Expected both satellites in merge, got {sats_present}")

        # Valid AMVs exist
        qf = merged["quality_flag"].values
        assert (qf >= 2).sum() > 0
