"""Global geostationary ring student AMV inference.

For a given time, loads imagery from every available geostationary satellite
(GOES-18, GOES-19, Himawari-9, GK-2A, MTG-I1) via icechunk, runs the
single-satellite student model on each, and produces:

1. Per-satellite NetCDF files with full-disk AMVs
2. A combined global NetCDF on a regular lat/lon grid, where overlapping
   regions use the AMV from whichever satellite has the smallest zenith
   angle (closest to the sub-satellite point)

Usage::

    pixi run python scripts/infer_student_global_ring.py \\
        --time "2025-03-10T12:00" \\
        --student-ckpt checkpoints/student.abi.mb-v3.ep21.ckpt \\
        --raft-ckpt checkpoints/windflow.raft.sonde-tuned.ckpt \\
        --output-dir output/global_ring \\
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from stereo_winds.config import SATELLITE_CONFIGS, SatelliteConfig
from stereo_winds.disparity import StereoDisparity
from stereo_winds.navigation import (
    compute_grid_latlon,
    compute_grid_zenith,
    compute_pixel_scale,
)
from stereo_winds.student_dataset import (
    DEFAULT_FLOW_BANDS,
    DEFAULT_RAD_BANDS,
    FLOW_SCALE,
    PIXEL_SCALE_NORM,
    ZENITH_NORM,
)
from stereo_winds.student_zeus_model import StudentWindsModel

logger = logging.getLogger(__name__)

DT_MINUTES = 10
OUTPUT_VARS = [
    "u_wind", "v_wind", "cloud_top_height",
    "quality_flag", "sigma_u", "sigma_v", "sigma_h",
]

# Satellites to process (in longitude order, west to east).
# Each entry: (sat_id, loader_type, flow_bands_key, rad_bands_key)
# loader_type determines which data_loading function to call.
RING_SATELLITES = [
    "goes18",     # 137°W
    "goes19",     # 75°W
    "mtg-i1",     # 0°E
    "gk2a",       # 128.2°E
    "himawari9",  # 140.7°E
]


# ---------------------------------------------------------------------------
# Data loading: 3-frame radiance cubes from icechunk
# ---------------------------------------------------------------------------

def _load_scene(sat_id: str, band: str, t: datetime) -> np.ndarray:
    """Load a single 2D radiance scene for the given satellite and band."""
    if "goes" in sat_id:
        from stereo_winds.data_loading import load_goes_scene
        data, _ = load_goes_scene(t, band, sat_id)
        return data
    elif "himawari" in sat_id:
        from stereo_winds.data_loading import load_himawari_scene
        data, _ = load_himawari_scene(t, band, sat_id)
        return data
    elif "gk2a" in sat_id:
        from stereo_winds.data_loading import load_gk2a_scene
        data, _ = load_gk2a_scene(t, band, sat_id)
        return data
    elif "mtg" in sat_id:
        from stereo_winds.data_loading import load_fci_scene
        data, _ = load_fci_scene(t, band, sat_id)
        return data
    raise ValueError(f"Unknown satellite: {sat_id}")


def _load_three_frames(
    sat_id: str, band: str, t0: datetime, dt_min: int = DT_MINUTES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load t-dt, t0, t+dt radiance frames for a single band."""
    delta = timedelta(minutes=dt_min)
    a_m = _load_scene(sat_id, band, t0 - delta)
    a_0 = _load_scene(sat_id, band, t0)
    a_p = _load_scene(sat_id, band, t0 + delta)
    return a_m, a_0, a_p


# ---------------------------------------------------------------------------
# Build student model input stack
# ---------------------------------------------------------------------------

def _band_available(sat_id: str, band: str) -> bool:
    """Check whether an ABI band can be loaded for the given satellite.

    Bands that have no spectral equivalent on a satellite (e.g. ABI C09
    and C14 on MTG FCI) return False so the caller can substitute zeros.
    """
    if "goes" in sat_id:
        return True  # ABI has all ABI bands
    if "himawari" in sat_id:
        from stereo_winds.readers.himawari import _ABI_TO_AHI, _BAND_RESOLUTION
        return band in _BAND_RESOLUTION or band in _ABI_TO_AHI
    if "gk2a" in sat_id:
        from stereo_winds.readers.gk2a import _ABI_TO_AMI, _BAND_RESOLUTION
        return band in _BAND_RESOLUTION or band in _ABI_TO_AMI
    if "mtg" in sat_id:
        from stereo_winds.config import ABI_TO_FCI_BAND
        from stereo_winds.readers.mtg import _BAND_RESOLUTION
        return band in _BAND_RESOLUTION or band in ABI_TO_FCI_BAND
    return True


def _build_input_stack(
    sat_id: str,
    sat: SatelliteConfig,
    t0: datetime,
    disp: StereoDisparity,
    flow_bands: list[str],
    rad_bands: list[str],
    rad_time_frames: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the (C, H, W) flow/rad/geom input stack from icechunk data.

    Bands that have no spectral equivalent on the target satellite are
    filled with NaN (and masked out in the finite_mask), keeping the
    channel count consistent with the trained student checkpoint.

    Returns (flow_arr, rad_arr, geom_arr, finite_mask).
    """
    H, W = sat.n_rows, sat.n_cols
    flow_chans: list[np.ndarray] = []
    cached: dict[str, tuple] = {}  # band -> (a_m, a_0, a_p, valid)

    logger.info("  Loading flow bands: %s", flow_bands)
    for band in flow_bands:
        if not _band_available(sat_id, band):
            logger.warning("  Band %s unavailable on %s — filling with zeros", band, sat_id)
            zero = np.zeros((H, W), dtype=np.float32)
            flow_chans += [zero, zero.copy(), zero.copy(), zero.copy()]
            continue
        a_m, a_0, a_p = _load_three_frames(sat_id, band, t0)
        valid = np.isfinite(a_m) & np.isfinite(a_0) & np.isfinite(a_p)
        fb = disp._run_pair(a_0, a_m)
        ff = disp._run_pair(a_0, a_p)
        for fl in (fb, ff):
            fl[:, ~valid] = np.nan
        flow_chans += [fb[0], fb[1], ff[0], ff[1]]
        cached[band] = (a_m, a_0, a_p, valid)

    rad_chans: list[np.ndarray] = []
    logger.info("  Loading rad bands: %s", rad_bands)
    for band in rad_bands:
        if not _band_available(sat_id, band):
            logger.warning("  Band %s unavailable on %s — filling with zeros", band, sat_id)
            n_frames = rad_time_frames if rad_time_frames == 3 else 1
            for _ in range(n_frames):
                rad_chans.append(np.zeros((H, W), dtype=np.float32))
            continue

        if band in cached:
            a_m, a_0, a_p, vb = cached[band]
        else:
            if rad_time_frames == 3:
                a_m, a_0, a_p = _load_three_frames(sat_id, band, t0)
                vb = np.isfinite(a_m) & np.isfinite(a_0) & np.isfinite(a_p)
            else:
                a_0 = _load_scene(sat_id, band, t0)
                vb = np.isfinite(a_0)
                a_m = a_p = None

        if rad_time_frames == 3:
            for frame in (a_m, a_0, a_p):
                r = frame.copy()
                r[~vb] = np.nan
                rad_chans.append(r)
        else:
            r = a_0.copy()
            r[~vb] = np.nan
            rad_chans.append(r)

    dx_m, dy_m = compute_pixel_scale(sat)
    zen = compute_grid_zenith(sat)

    flow_arr = np.stack(flow_chans, 0).astype(np.float32) / FLOW_SCALE
    rad_arr = np.stack(rad_chans, 0).astype(np.float32)
    geom_arr = np.stack(
        [dx_m / PIXEL_SCALE_NORM, dy_m / PIXEL_SCALE_NORM, zen / ZENITH_NORM], 0,
    ).astype(np.float32)
    finite_mask = np.isfinite(flow_arr).all(0) & np.isfinite(rad_arr).all(0)
    return np.nan_to_num(flow_arr), np.nan_to_num(rad_arr), np.nan_to_num(geom_arr), finite_mask


# ---------------------------------------------------------------------------
# Forward pass (row-strip tiling)
# ---------------------------------------------------------------------------

def _forward_full_disk(
    model: StudentWindsModel,
    flow_arr: np.ndarray,
    rad_arr: np.ndarray,
    geom_arr: np.ndarray,
    row_strip: int = 1024,
    halo: int = 8,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Run the student model in row-strips over the full disk."""
    H, W = flow_arr.shape[1], flow_arr.shape[2]
    keys = ["u_mean", "v_mean", "h_mean", "u_logvar", "v_logvar", "h_logvar"]
    if getattr(model, "predict_chi2", False):
        keys.append("chi2")
    out = {k: np.full((H, W), np.nan, np.float32) for k in keys}
    with torch.no_grad():
        for r in range(0, H, row_strip):
            r1_keep = min(H, r + row_strip)
            r0_in = max(0, r - halo)
            r1_in = min(H, r1_keep + halo)
            ft = torch.from_numpy(flow_arr[:, r0_in:r1_in]).unsqueeze(0).to(device)
            rt = torch.from_numpy(rad_arr[:, r0_in:r1_in]).unsqueeze(0).to(device)
            gt = torch.from_numpy(geom_arr[:, r0_in:r1_in]).unsqueeze(0).to(device)
            o = model.predict(ft, rt, gt)
            keep0 = r - r0_in
            keep1 = keep0 + (r1_keep - r)
            # Multi-band model: select band 0 (or squeeze single-band)
            for k in keys:
                val = o[k]
                if val.ndim == 4:
                    out[k][r:r1_keep] = val[0, 0, keep0:keep1].cpu().numpy()
                else:
                    out[k][r:r1_keep] = val[0, keep0:keep1].cpu().numpy()
    return out


def _assemble_vars(
    raw: dict[str, np.ndarray], finite_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert raw model output to the standard AMV variable schema."""
    u = np.where(finite_mask, raw["u_mean"], np.nan)
    v = np.where(finite_mask, raw["v_mean"], np.nan)
    h_km = np.where(finite_mask, raw["h_mean"], np.nan)
    sigma_u = np.exp(0.5 * raw["u_logvar"])
    sigma_v = np.exp(0.5 * raw["v_logvar"])
    sigma_h_km = np.exp(0.5 * raw["h_logvar"])
    valid = finite_mask & np.isfinite(u) & np.isfinite(v) & np.isfinite(h_km)
    qf = np.where(valid, 2.0, 0.0).astype(np.float32)
    return {
        "u_wind": u.astype(np.float32),
        "v_wind": v.astype(np.float32),
        "cloud_top_height": (h_km * 1000.0).astype(np.float32),
        "quality_flag": qf,
        "sigma_u": sigma_u.astype(np.float32),
        "sigma_v": sigma_v.astype(np.float32),
        "sigma_h": (sigma_h_km * 1000.0).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Per-satellite inference
# ---------------------------------------------------------------------------

def infer_satellite(
    sat_id: str,
    t0: datetime,
    model: StudentWindsModel,
    disp: StereoDisparity,
    flow_bands: list[str],
    rad_bands: list[str],
    device: str = "cuda",
    row_strip: int = 1024,
) -> xr.Dataset:
    """Run student inference for one satellite and return an xr.Dataset.

    The dataset has dimensions (y, x) with lat/lon coordinate arrays
    and the standard AMV variables.
    """
    sat = SATELLITE_CONFIGS[sat_id]
    logger.info("Processing %s (sub_lon=%.1f°)", sat_id, sat.sub_lon_deg)

    rad_tf = int(getattr(model, "rad_time_frames", 1))
    flow_arr, rad_arr, geom_arr, finite_mask = _build_input_stack(
        sat_id, sat, t0, disp, flow_bands, rad_bands,
        rad_time_frames=rad_tf,
    )

    logger.info("  Forward pass (%dx%d)...", sat.n_rows, sat.n_cols)
    raw = _forward_full_disk(model, flow_arr, rad_arr, geom_arr,
                             row_strip=row_strip, device=device)
    amvs = _assemble_vars(raw, finite_mask)

    # Compute lat/lon for each pixel
    lat, lon = compute_grid_latlon(sat)
    zen = compute_grid_zenith(sat)

    ds = xr.Dataset(
        {k: (("y", "x"), amvs[k]) for k in OUTPUT_VARS},
        coords={
            "latitude": (("y", "x"), lat.astype(np.float32)),
            "longitude": (("y", "x"), lon.astype(np.float32)),
            "zenith_angle": (("y", "x"), zen.astype(np.float32)),
        },
        attrs={
            "satellite_id": sat_id,
            "sub_satellite_longitude": sat.sub_lon_deg,
            "time": str(t0),
            "source": "student_amv",
        },
    )
    return ds


# ---------------------------------------------------------------------------
# Global mosaic: merge per-satellite datasets on a regular lat/lon grid
# ---------------------------------------------------------------------------

def merge_global(
    per_sat: dict[str, xr.Dataset],
    resolution_m: float = 2000.0,
) -> xr.Dataset:
    """Merge per-satellite AMV datasets onto a global 2 km lat/lon grid.

    At each grid cell, if multiple satellites contribute, the one with
    the smallest zenith angle (closest to the sub-satellite point) wins.

    Parameters
    ----------
    per_sat : dict mapping satellite_id -> xr.Dataset (from infer_satellite)
    resolution_m : output grid spacing in meters (default 2000 m)
    """
    # Convert metres to degrees: 1° latitude ≈ 111 320 m
    dlat = resolution_m / 111_320.0
    # Use the same angular spacing for longitude; pixels are ~square at the
    # equator and compress toward the poles (standard equirectangular).
    dlon = dlat

    lat_bins = np.arange(-90, 90 + dlat, dlat)
    lon_bins = np.arange(-180, 180 + dlon, dlon)
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    lon_centers = 0.5 * (lon_bins[:-1] + lon_bins[1:])
    n_lat, n_lon = len(lat_centers), len(lon_centers)
    logger.info("Global grid: %d x %d (%.0f m ≈ %.4f°)",
                n_lat, n_lon, resolution_m, dlat)

    # Accumulate: for each grid cell, track the best (lowest zenith) value
    best_zen = np.full((n_lat, n_lon), np.inf, dtype=np.float32)
    merged = {v: np.full((n_lat, n_lon), np.nan, dtype=np.float32)
              for v in OUTPUT_VARS}
    source_sat = np.full((n_lat, n_lon), "", dtype="U12")

    for sat_id, ds in per_sat.items():
        logger.info("Gridding %s onto %.0f m global grid...", sat_id, resolution_m)
        lat_2d = ds["latitude"].values
        lon_2d = ds["longitude"].values
        zen_2d = ds["zenith_angle"].values
        qf = ds["quality_flag"].values

        # Only grid pixels with valid AMVs
        valid = (qf >= 2) & np.isfinite(lat_2d) & np.isfinite(lon_2d)
        if not valid.any():
            logger.warning("  %s: no valid pixels to grid", sat_id)
            continue

        flat_lat = lat_2d[valid]
        flat_lon = lon_2d[valid]
        flat_zen = zen_2d[valid]

        # Digitize into grid bins
        ri = np.digitize(flat_lat, lat_bins) - 1
        ci = np.digitize(flat_lon, lon_bins) - 1
        np.clip(ri, 0, n_lat - 1, out=ri)
        np.clip(ci, 0, n_lon - 1, out=ci)

        # Find where this satellite beats the current best zenith
        better = flat_zen < best_zen[ri, ci]
        idx_r = ri[better]
        idx_c = ci[better]

        if len(idx_r) > 0:
            for var_name in OUTPUT_VARS:
                flat_var = ds[var_name].values[valid]
                merged[var_name][idx_r, idx_c] = flat_var[better]
            best_zen[idx_r, idx_c] = flat_zen[better]
            source_sat[idx_r, idx_c] = sat_id

        logger.info("  %s: %d grid cells contributed (of %d valid pixels)",
                     sat_id, int(better.sum()), int(valid.sum()))

    ds_global = xr.Dataset(
        {v: (("latitude", "longitude"), merged[v]) for v in OUTPUT_VARS},
        coords={
            "latitude": lat_centers,
            "longitude": lon_centers,
        },
        attrs={
            "title": "Global student AMV mosaic",
            "resolution_m": resolution_m,
            "merge_rule": "minimum zenith angle (closest to sub-satellite point)",
            "satellites": list(per_sat.keys()),
            "time": next(iter(per_sat.values())).attrs["time"],
        },
    )
    ds_global["source_satellite"] = (("latitude", "longitude"), source_sat)
    return ds_global


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description="Global geostationary ring student AMV inference",
    )
    ap.add_argument("--time", required=True,
                    help="ISO timestamp (e.g. 2025-03-10T12:00)")
    ap.add_argument("--student-ckpt", required=True,
                    help="Student Lightning checkpoint")
    ap.add_argument("--raft-ckpt", required=True,
                    help="RAFT optical-flow checkpoint")
    ap.add_argument("--output-dir", default="output/global_ring",
                    help="Output directory for NetCDF files")
    ap.add_argument("--device", default="cuda",
                    choices=["cuda", "cpu"])
    ap.add_argument("--satellites", default=None,
                    help="Comma-separated satellite IDs to process "
                         "(default: all in the ring)")
    ap.add_argument("--flow-bands", default=",".join(DEFAULT_FLOW_BANDS),
                    help="Comma-separated flow bands (ABI names)")
    ap.add_argument("--rad-bands", default=",".join(DEFAULT_RAD_BANDS),
                    help="Comma-separated radiance bands (ABI names)")
    ap.add_argument("--resolution-m", type=float, default=2000.0,
                    help="Global grid resolution in meters (default 2000)")
    ap.add_argument("--row-strip", type=int, default=1024,
                    help="Row strip height for tiled inference")
    ap.add_argument("--skip-global", action="store_true",
                    help="Skip the global mosaic, only produce per-satellite files")
    args = ap.parse_args()

    t0 = datetime.fromisoformat(args.time)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sats = (args.satellites.split(",") if args.satellites
            else RING_SATELLITES)
    flow_bands = [b for b in args.flow_bands.split(",") if b]
    rad_bands = [b for b in args.rad_bands.split(",") if b]

    # Load models
    logger.info("Loading student checkpoint: %s", args.student_ckpt)
    model = StudentWindsModel.load_from_checkpoint(
        args.student_ckpt, map_location=args.device,
    ).eval()

    logger.info("Loading RAFT checkpoint: %s", args.raft_ckpt)
    disp = StereoDisparity(
        model_ckpt_path=args.raft_ckpt,
        tile_size=512, overlap=128, batch_size=8,
        device=args.device,
    )

    # Per-satellite inference
    per_sat: dict[str, xr.Dataset] = {}
    for sat_id in sats:
        if sat_id not in SATELLITE_CONFIGS:
            logger.warning("Unknown satellite %r, skipping", sat_id)
            continue
        try:
            ds = infer_satellite(
                sat_id, t0, model, disp,
                flow_bands, rad_bands,
                device=args.device, row_strip=args.row_strip,
            )
            # Save per-satellite file
            tag = t0.strftime("%Y%m%dT%H%M")
            nc_path = out_dir / f"student_amv_{sat_id}_{tag}.nc"
            ds.to_netcdf(nc_path)
            logger.info("Saved %s", nc_path)
            per_sat[sat_id] = ds
        except Exception:
            logger.exception("Failed to process %s — skipping", sat_id)

    if not per_sat:
        logger.error("No satellites produced output. Exiting.")
        sys.exit(1)

    logger.info("Completed %d/%d satellites: %s",
                len(per_sat), len(sats), list(per_sat.keys()))

    # Global mosaic
    if not args.skip_global and len(per_sat) > 0:
        logger.info("Building global mosaic (%.0f m grid)...", args.resolution_m)
        ds_global = merge_global(per_sat, resolution_m=args.resolution_m)
        tag = t0.strftime("%Y%m%dT%H%M")
        global_path = out_dir / f"student_amv_global_{tag}.nc"
        ds_global.to_netcdf(global_path)
        logger.info("Saved global mosaic: %s", global_path)

        # Summary stats
        valid = ds_global["quality_flag"].values >= 2
        n_valid = int(valid.sum())
        n_total = valid.size
        logger.info("Global mosaic: %d / %d grid cells with valid AMVs (%.1f%%)",
                     n_valid, n_total, 100 * n_valid / n_total if n_total else 0)

    logger.info("Done.")


if __name__ == "__main__":
    main()
