"""
Build a sample in Alpamayo format (like load_physical_aiavdataset) from:
- pose.csv (t, latitude, longitude, altitude) or CSV with x, y, z in meters
- orientation.csv (t, orientation.x, orientation.y, orientation.z, orientation.w)
- images_extracted/ + images_index.csv (t, t_ns, topic, path)

Usage: build_alpamayo_sample(t0_us, pose_df, orientation_df, images_df, ...)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.spatial.transform as spt

import os

import torch

import cv2

try:
    from pyproj import Transformer
    _transformer = Transformer.from_crs(4326, 32616, always_xy=True)
except ImportError:
    _transformer = None

# Constants as in Alpamayo / dataset_builder
NUM_HISTORY_STEPS = 16
NUM_FUTURE_STEPS = 64
NUM_FRAMES = 4
TIME_STEP = 0.1
CAMERA_TOPICS_ORDER = [
    "/lucid_vision/lucid_cam_front_center/image_rect/compressed",
    "/lucid_vision/lucid_cam_front_center/image_rect/compressed",
    "/lucid_vision/lucid_cam_front_left/image_rect/compressed",
    "/lucid_vision/lucid_cam_front_right/image_rect/compressed",
]


def _pose_to_xyz_meters(pose_df: pd.DataFrame, ref_lat: float, ref_lon: float, ref_alt: float):
    """Converts pose (lat, lon, alt) to x, y, z in meters (origin at ref)."""
    if "x_cog" in pose_df.columns or "x" in pose_df.columns:
        x = pose_df["x_cog"].values if "x_cog" in pose_df.columns else pose_df["x"].values
        y = pose_df["y_cog"].values if "y_cog" in pose_df.columns else pose_df["y"].values
        z = pose_df["z_cog"].values if "z_cog" in pose_df.columns else pose_df["z"].values
        return x, y, z
    if _transformer is None:
        raise ImportError("pyproj required to convert lat/lon to meters")
    lon = pose_df["longitude"].values
    lat = pose_df["latitude"].values
    alt = pose_df["altitude"].values
    x, y = _transformer.transform(lon, lat)
    x0, y0 = _transformer.transform(ref_lon, ref_lat)
    return x - x0, y - y0, alt - ref_alt


def _nearest_row(df: pd.DataFrame, t_col: str, t_val: float):
    """Get row with timestamp closest to t_val."""
    idx = (df[t_col] - t_val).abs().idxmin()
    return df.loc[idx]


def _interp_quat(t_out: np.ndarray, t_q: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """Interpolate quaternions (linear interpolation and renormalization)."""
    out = np.zeros((len(t_out), 4))
    for i in range(4):
        out[:, i] = np.interp(t_out, t_q, quats[:, i])
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / (norms + 1e-8)
    return out


def build_alpamayo_sample(
    t0_us: int,
    pose_df: pd.DataFrame,
    orientation_df: pd.DataFrame,
    images_df: pd.DataFrame,
    images_root: str | Path = ".",
    use_torch: bool = False,
) -> dict[str, Any]:
    """
    Build a dictionary in Alpamayo format from CSVs and image index.

    - pose_df: must have column "t" (seconds) and (latitude, longitude, altitude) or (x_cog, y_cog, z_cog) / (x,y,z).
    - orientation_df: t, orientation.x, orientation.y, orientation.z, orientation.w.
    - images_df: t, t_ns, topic, path (path relative to images_root or absolute).

    Returns:
        Dict with image_frames, camera_indices, ego_history_xyz, ego_history_rot,
        ego_future_xyz, ego_future_rot, relative_timestamps, absolute_timestamps, t0_us, clip_id.
    """
    t0_s = t0_us * 1e-6
    images_root = Path(images_root)

    # Reference for converting pose to meters (use pose at t0)
    t_col = "t"
    if "latitude" in pose_df.columns:
        ref = _nearest_row(pose_df, t_col, t0_s)
        ref_lat, ref_lon, ref_alt = ref["latitude"], ref["longitude"], ref["altitude"]
    else:
        ref_lat = ref_lon = ref_alt = 0.0
    pose_df = pose_df.sort_values(t_col)
    t_pose = pose_df[t_col].values
    x, y, z = _pose_to_xyz_meters(pose_df, ref_lat, ref_lon, ref_alt)
    xyz_world = np.stack([x, y, z], axis=1)

    ori_df = orientation_df.sort_values(t_col)
    t_ori = ori_df[t_col].values
    quats = ori_df[["orientation.x", "orientation.y", "orientation.z", "orientation.w"]].values

    # Timestamps as in Alpamayo (linspace so that the last history is exactly t0)
    history_ts = t0_s + np.linspace(
        -(NUM_HISTORY_STEPS - 1) * TIME_STEP, 0, NUM_HISTORY_STEPS, dtype=np.float64
    )
    future_ts = t0_s + np.linspace(
        TIME_STEP, NUM_FUTURE_STEPS * TIME_STEP, NUM_FUTURE_STEPS, dtype=np.float64
    )

    # Interpolate xyz and quat in history and future
    hist_xyz_w = np.zeros((NUM_HISTORY_STEPS, 3))
    fut_xyz_w = np.zeros((NUM_FUTURE_STEPS, 3))
    for j in range(3):
        hist_xyz_w[:, j] = np.interp(history_ts, t_pose, xyz_world[:, j])
        fut_xyz_w[:, j] = np.interp(future_ts, t_pose, xyz_world[:, j])
    hist_quat = _interp_quat(history_ts, t_ori, quats)
    fut_quat = _interp_quat(future_ts, t_ori, quats)

    # Local frame at t0 (last step of history)
    t0_xyz = hist_xyz_w[-1].copy()
    t0_quat = hist_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    # Transform to local frame: xyz_local = R_t0^{-1} @ (xyz - xyz_t0)
    ego_history_xyz = t0_rot_inv.apply(hist_xyz_w - t0_xyz)
    ego_future_xyz = t0_rot_inv.apply(fut_xyz_w - t0_xyz)
    ego_history_rot = (t0_rot_inv * spt.Rotation.from_quat(hist_quat)).as_matrix()
    ego_future_rot = (t0_rot_inv * spt.Rotation.from_quat(fut_quat)).as_matrix()

    # Images: [t0-0.3s, t0-0.2s, t0-0.1s, t0] per camera
    image_ts_s = np.array(
        [t0_s - (NUM_FRAMES - 1 - i) * TIME_STEP for i in range(NUM_FRAMES)],
        dtype=np.float64,
    )
    image_ts_ns = (image_ts_s * 1e9).astype(np.int64)
    images_df = images_df.copy()
    if "t_ns" not in images_df.columns and "t" in images_df.columns:
        images_df["t_ns"] = (images_df["t"].values * 1e9).astype(np.int64)

    frames_list = []
    for topic in CAMERA_TOPICS_ORDER:
        sub = images_df[images_df["topic"] == topic]
        if sub.empty:
            raise ValueError(f"No images for topic {topic}")
        t_ns_arr = sub["t_ns"].values
        paths = sub["path"].values
        row_frames = []
        for t_ns in image_ts_ns:
            idx = np.argmin(np.abs(t_ns_arr - t_ns))
            path = Path(paths[idx])
            if not path.is_absolute():
                path = images_root / path
            else:
                path = Path(path)
            try:
                img = cv2.imread(str(path))
                if img is None:
                    raise FileNotFoundError(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                raise FileNotFoundError(f"Could not load {path}: {e}") from e
            # (H, W, 3) -> (3, H, W)
            img = np.transpose(img, (2, 0, 1))
            row_frames.append(img)
        frames_list.append(np.stack(row_frames, axis=0))
    image_frames = np.stack(frames_list, axis=0).astype(np.uint8)  

    absolute_timestamps = (image_ts_s * 1e6).astype(np.int64)
    absolute_timestamps = np.broadcast_to(absolute_timestamps, (len(CAMERA_TOPICS_ORDER), NUM_FRAMES))
    camera_tmin = absolute_timestamps.min()
    relative_timestamps = (absolute_timestamps - camera_tmin) * 1e-6

    ego_history_xyz = ego_history_xyz.astype(np.float32)[np.newaxis, np.newaxis, ...]
    ego_history_rot = ego_history_rot.astype(np.float32)[np.newaxis, np.newaxis, ...]
    ego_future_xyz = ego_future_xyz.astype(np.float32)[np.newaxis, np.newaxis, ...]
    ego_future_rot = ego_future_rot.astype(np.float32)[np.newaxis, np.newaxis, ...]

    out = {
        "image_frames": image_frames,
        "camera_indices": np.array([0, 1, 2, 3], dtype=np.int64),
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "ego_future_xyz": ego_future_xyz,
        "ego_future_rot": ego_future_rot,
        "relative_timestamps": relative_timestamps.astype(np.float32),
        "absolute_timestamps": absolute_timestamps,
        "t0_us": t0_us,
        "clip_id": "from_csv",
    }
    if use_torch:
        for k in ("image_frames", "ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot", "relative_timestamps", "absolute_timestamps", "camera_indices"):
            out[k] = torch.from_numpy(out[k])
    return out


if __name__ == "__main__":

    pose_df = pd.read_csv("pose-raw.csv")
    orientation_df = pd.read_csv("orientation-raw.csv")
    images_df = pd.read_csv("images_extracted_raw/images-raw-index.csv")

    t_min = max(pose_df["t"].min() + 1.6, images_df["t"].min() + 0.3)
    t_max = min(pose_df["t"].max() - 6.4, images_df["t"].max() - 0.01)

    out_dir = "clips"
    os.makedirs(out_dir, exist_ok=True)

    stride_s = 3  
    ok = fail = 0

  
    for t0 in np.arange(t_min, t_max, stride_s):
        t0_us = int(round(t0 * 1e6)) 
        
        try:
           
            data = build_alpamayo_sample(
                t0_us, 
                pose_df.copy(), 
                orientation_df.copy(), 
                images_df.copy(),
                images_root=".", 
                use_torch=False
            )
            

            arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
            np.savez_compressed(os.path.join(out_dir, f"sample_{t0_us}.npz"), **arrays)
            ok += 1
            print(f"Éxito en clip {t0_us}")
            
        except Exception as e:
            fail += 1
            
            print(f"Error en el clip t0_us={t0_us}: {e}")
            continue

    print(f"Done. saved={ok} failed={fail} dir={out_dir}")



