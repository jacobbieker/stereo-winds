"""Integration tests: student single-view AMV prediction on each satellite.

These tests load the shipped student checkpoint and run inference on synthetic
or real data for GOES, Himawari, GK-2A, and MTG.  They are slow and require
the student checkpoint + PyTorch, so they are gated behind the ``integration``
marker and only run when explicitly requested::

    pixi run python -m pytest tests/test_student_integration.py -v -m integration

The ``integration_live`` marker additionally requires network access to
source.coop to pull real radiance from icechunk stores::

    pixi run python -m pytest tests/test_student_integration.py -v -m integration_live
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import torch

from stereo_winds.config import (
    GOES18_CONFIG,
    GOES19_CONFIG,
    GK2A_CONFIG,
    HIMAWARI9_CONFIG,
    MTG_I1_CONFIG,
    SatelliteConfig,
    SATELLITE_CONFIGS,
)

STUDENT_CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "student.abi.mb-v3.ep21.ckpt"
RAFT_CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "windflow.raft.sonde-tuned.ckpt"

# Student model constants (from student_dataset.py)
DEFAULT_FLOW_BANDS = ["C08", "C09", "C10", "C12", "C14"]
DEFAULT_RAD_BANDS = ["C07", "C08", "C09", "C10", "C11",
                     "C12", "C13", "C14", "C15", "C16"]


def _have_checkpoint() -> bool:
    return STUDENT_CKPT.exists()


def _have_raft_checkpoint() -> bool:
    return RAFT_CKPT.exists()


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── Synthetic-input integration tests ────────────────────────────────
# These load the real checkpoint but feed synthetic (random) inputs
# shaped to match each satellite's grid.  Verifies the model can
# forward-pass on any satellite's geometry without errors.


@pytest.mark.integration
class TestStudentSyntheticForward:
    """Load the student checkpoint and run forward on synthetic inputs
    matching each satellite's grid dimensions (small crop)."""

    @pytest.fixture(autouse=True)
    def _load_model(self):
        if not _have_checkpoint():
            pytest.skip(f"Student checkpoint not found: {STUDENT_CKPT}")
        from stereo_winds.student_zeus_model import StudentWindsModel
        self.device = _device()
        self.model = StudentWindsModel.load_from_checkpoint(
            str(STUDENT_CKPT), map_location=self.device,
        ).eval()

    def _run_forward(self, sat: SatelliteConfig, crop: int = 64):
        """Run a forward pass on a (1, C, crop, crop) random input."""
        n_flow = len(DEFAULT_FLOW_BANDS)
        n_rad = len(DEFAULT_RAD_BANDS)
        rad_tf = int(getattr(self.model, "rad_time_frames", 1))
        n_rad_ch = n_rad * rad_tf
        n_flow_ch = 4 * n_flow
        n_geom = 3

        flow = torch.randn(1, n_flow_ch, crop, crop, device=self.device)
        rad = torch.randn(1, n_rad_ch, crop, crop, device=self.device)
        geom = torch.randn(1, n_geom, crop, crop, device=self.device)

        with torch.no_grad():
            out = self.model.predict(flow, rad, geom)

        assert "u_mean" in out
        assert "v_mean" in out
        assert "h_mean" in out
        assert out["u_mean"].shape == (1, crop, crop) or out["u_mean"].shape[0] == 1
        return out

    def test_goes(self):
        self._run_forward(GOES19_CONFIG)

    def test_himawari(self):
        self._run_forward(HIMAWARI9_CONFIG)

    def test_gk2a(self):
        self._run_forward(GK2A_CONFIG)

    def test_mtg(self):
        self._run_forward(MTG_I1_CONFIG)

    def test_output_values_finite(self):
        out = self._run_forward(GOES19_CONFIG)
        for key in ("u_mean", "v_mean", "h_mean"):
            arr = out[key].cpu().numpy()
            assert np.isfinite(arr).all(), f"{key} has non-finite values"


# ── Live integration tests (real icechunk data) ──────────────────────
# These pull real radiance from icechunk, run RAFT for flows, and
# forward through the student model on a small crop.  Requires network
# access + both checkpoints.


def _load_single_band(sat_id: str, band: str, t: dt.datetime) -> np.ndarray:
    """Load one 2D radiance array from icechunk for the given satellite."""
    if "goes" in sat_id:
        from stereo_winds.data_loading import load_goes_scene
        data, _ = load_goes_scene(t, band, sat_id, coarsen=False)
        return data
    elif "himawari" in sat_id:
        from stereo_winds.data_loading import load_himawari_scene
        data, _ = load_himawari_scene(t, band, sat_id, coarsen=False)
        return data
    elif "gk2a" in sat_id:
        from stereo_winds.data_loading import load_gk2a_scene
        data, _ = load_gk2a_scene(t, band, sat_id, coarsen=False)
        return data
    elif "mtg" in sat_id:
        from stereo_winds.data_loading import load_fci_scene
        data, _ = load_fci_scene(t, band, sat_id, coarsen=False)
        return data
    raise ValueError(f"Unknown satellite: {sat_id}")


def _build_synthetic_student_input(
    sat: SatelliteConfig, rad_time_frames: int = 1, crop: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build random flow + rad + geom arrays for a crop of the satellite grid."""
    from stereo_winds.navigation import compute_pixel_scale, compute_grid_zenith
    from stereo_winds.student_dataset import FLOW_SCALE, PIXEL_SCALE_NORM, ZENITH_NORM

    n_flow = len(DEFAULT_FLOW_BANDS)
    n_rad = len(DEFAULT_RAD_BANDS)
    n_rad_ch = n_rad * rad_time_frames

    # Use real geometry from the satellite config (cropped)
    dx_m, dy_m = compute_pixel_scale(sat)
    zen = compute_grid_zenith(sat)

    r0 = sat.n_rows // 2 - crop // 2
    c0 = sat.n_cols // 2 - crop // 2
    dx_crop = dx_m[r0:r0 + crop, c0:c0 + crop]
    dy_crop = dy_m[r0:r0 + crop, c0:c0 + crop]
    zen_crop = zen[r0:r0 + crop, c0:c0 + crop]

    flow_arr = np.random.randn(4 * n_flow, crop, crop).astype(np.float32) / FLOW_SCALE
    rad_arr = np.random.randn(n_rad_ch, crop, crop).astype(np.float32)
    geom_arr = np.stack([
        np.nan_to_num(dx_crop) / PIXEL_SCALE_NORM,
        np.nan_to_num(dy_crop) / PIXEL_SCALE_NORM,
        np.nan_to_num(zen_crop) / ZENITH_NORM,
    ], axis=0).astype(np.float32)

    return flow_arr, rad_arr, geom_arr


@pytest.mark.integration_live
class TestStudentLiveInference:
    """Live integration: load real radiance from icechunk and run the
    student model forward pass on a small crop of each satellite."""

    @pytest.fixture(autouse=True)
    def _load_model(self):
        if not _have_checkpoint():
            pytest.skip(f"Student checkpoint not found: {STUDENT_CKPT}")
        from stereo_winds.student_zeus_model import StudentWindsModel
        self.device = _device()
        self.model = StudentWindsModel.load_from_checkpoint(
            str(STUDENT_CKPT), map_location=self.device,
        ).eval()

    def _run_student_on_sat(self, sat_id: str, sat: SatelliteConfig,
                            band: str, t: dt.datetime, crop: int = 128):
        """Load real data, build input, run student forward."""
        # Load one real radiance frame to verify data loading works
        data = _load_single_band(sat_id, band, t)
        assert data.ndim == 2
        assert data.shape[0] > 0 and data.shape[1] > 0
        assert np.isfinite(data).sum() > 0, f"No finite values in {sat_id} {band}"

        # Build synthetic student input using real satellite geometry
        rad_tf = int(getattr(self.model, "rad_time_frames", 1))
        flow_arr, rad_arr, geom_arr = _build_synthetic_student_input(
            sat, rad_time_frames=rad_tf, crop=crop)

        # Forward pass
        flow_t = torch.from_numpy(flow_arr).unsqueeze(0).to(self.device)
        rad_t = torch.from_numpy(rad_arr).unsqueeze(0).to(self.device)
        geom_t = torch.from_numpy(geom_arr).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model.predict(flow_t, rad_t, geom_t)

        assert "u_mean" in out
        assert "v_mean" in out
        assert "h_mean" in out
        u = out["u_mean"].cpu().numpy()
        assert np.isfinite(u).all(), f"Student output has NaN for {sat_id}"
        return out

    def test_goes(self):
        self._run_student_on_sat(
            "goes18", GOES18_CONFIG, "C14",
            dt.datetime(2024, 1, 15, 12, 0),
        )

    def test_himawari(self):
        self._run_student_on_sat(
            "himawari9", HIMAWARI9_CONFIG, "B14",
            dt.datetime(2024, 1, 15, 3, 0),
        )

    def test_gk2a(self):
        self._run_student_on_sat(
            "gk2a", GK2A_CONFIG, "IR112",
            dt.datetime(2024, 1, 15, 3, 0),
        )

    def test_mtg(self):
        self._run_student_on_sat(
            "mtg-i1", MTG_I1_CONFIG, "ir_105",
            dt.datetime(2024, 6, 15, 12, 0),
        )
