"""Depth/pointcloud geometry helpers."""

from __future__ import annotations

import numpy as np


def backproject_depth(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Backproject a depth map to a camera-frame (Nx3) point cloud on valid pixels."""
    ys, xs = np.where(valid_mask)
    z = depth[valid_mask].astype(np.float32)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def depth_to_pointmap(depth: np.ndarray, cam_k: np.ndarray) -> np.ndarray:
    """Convert a depth map (H, W) to a dense pointmap (H, W, 3) in camera frame."""
    h, w = depth.shape
    fx, fy = float(cam_k[0, 0]), float(cam_k[1, 1])
    cx, cy = float(cam_k[0, 2]), float(cam_k[1, 2])
    ys, xs = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    z = depth
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def sanitize_depth_for_pointmap(depth: np.ndarray) -> np.ndarray:
    """Mark invalid depth (NaN/Inf/non-positive) as NaN for downstream use."""
    depth = np.asarray(depth, dtype=np.float32)
    invalid = (~np.isfinite(depth)) | (depth <= 0.0)
    if not np.any(invalid):
        return depth
    cleaned = depth.copy()
    cleaned[invalid] = np.nan
    return cleaned


def mad_keep_mask(values: np.ndarray, thresh: float) -> np.ndarray:
    """Boolean keep-mask using a robust z-score (MAD). ``thresh <= 0`` disables filtering."""
    if thresh <= 0:
        return np.ones(values.shape[0], dtype=bool)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-8:
        return np.ones(values.shape[0], dtype=bool)
    z_score = 0.6745 * (values - median) / mad
    return np.abs(z_score) <= thresh


def build_filter_keep_mask(
    points: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    shape: tuple[int, int],
    border_margin: int,
    depth_mad: float,
    radius_mad: float,
) -> np.ndarray:
    """Build a keep-mask combining border clipping + depth/radius MAD filtering."""
    keep = np.ones(points.shape[0], dtype=bool)
    if border_margin > 0 and ys.shape[0] == points.shape[0]:
        height, width = shape
        keep &= (
            (ys >= border_margin)
            & (ys < height - border_margin)
            & (xs >= border_margin)
            & (xs < width - border_margin)
        )

    filtered = points[keep]
    if filtered.size == 0:
        return np.zeros(points.shape[0], dtype=bool)

    if depth_mad > 0 or radius_mad > 0:
        sub_keep = np.ones(filtered.shape[0], dtype=bool)
        if depth_mad > 0:
            sub_keep &= mad_keep_mask(filtered[:, 2], depth_mad)
        if radius_mad > 0:
            center = np.median(filtered, axis=0)
            radius = np.linalg.norm(filtered - center, axis=1)
            sub_keep &= mad_keep_mask(radius, radius_mad)
        indices = np.where(keep)[0]
        keep_final = np.zeros(points.shape[0], dtype=bool)
        keep_final[indices[sub_keep]] = True
        return keep_final

    return keep
