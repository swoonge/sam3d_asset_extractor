"""Depth image loading and scale resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def resolve_depth_scale(depth_raw: np.ndarray, scale: str | float) -> float:
    """Turn a user-provided scale (``auto`` or numeric) into a concrete float.

    Heuristic for ``auto``: if the input looks like millimeter-scale uint16
    depth (large integer magnitudes), use 0.001; otherwise assume meters.
    """
    if isinstance(scale, str):
        scale_raw = scale.strip().lower()
        if scale_raw == "auto":
            if depth_raw.dtype.kind in ("u", "i") and float(np.max(depth_raw)) > 50:
                return 0.001
            if 20.0 < float(np.max(depth_raw)) < 10000.0:
                return 0.001
            return 1.0
        try:
            scale_val = float(scale_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid depth scale '{scale}'. Use 'auto' or a numeric value."
            ) from exc
    else:
        scale_val = float(scale)

    if not np.isfinite(scale_val) or scale_val <= 0.0:
        raise ValueError(f"Invalid depth scale: {scale_val}")
    return float(scale_val)


def load_depth_image(path: Path, scale: str | float) -> tuple[np.ndarray, float]:
    """Load a depth image and return (depth_in_meters, scale_applied)."""
    import cv2

    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Failed to read depth image: {path}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    scale_value = resolve_depth_scale(depth_raw, scale)
    depth = depth_raw.astype(np.float32)
    if scale_value != 1.0:
        depth *= float(scale_value)
    return depth, float(scale_value)
