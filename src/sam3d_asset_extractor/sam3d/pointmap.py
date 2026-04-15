"""Build the pointmap input that SAM3D accepts.

SAM3D's pointmap pipeline expects points in the Pytorch3D camera frame. We
backproject depth into the camera frame, optionally mask out non-object pixels
(``cropped`` mode), then rotate into Pytorch3D convention.

The resulting object is a ``PointmapInputCompat`` dict: newer SAM3D pipelines
read keys ``{"pointmap", "intrinsics"}``; older ones call ``.to(device)`` on
the pointmap, so the wrapper exposes both APIs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sam3d_asset_extractor.common.depth import load_depth_image
from sam3d_asset_extractor.common.geometry import depth_to_pointmap, sanitize_depth_for_pointmap
from sam3d_asset_extractor.logging_setup import get_logger

logger = get_logger("sam3d.pointmap")


class PointmapInputCompat(dict):
    """Dict-like pointmap input compatible with both new and legacy SAM3D pipelines."""

    def __init__(self, pointmap: torch.Tensor, intrinsics: np.ndarray | None):
        super().__init__()
        self["pointmap"] = pointmap
        self["intrinsics"] = intrinsics

    def to(self, *args, **kwargs):
        return self["pointmap"].to(*args, **kwargs)


def _apply_mask_to_pointmap(
    pointmap_hwc: np.ndarray, mask_bool: np.ndarray
) -> np.ndarray:
    """Set pointmap entries outside ``mask_bool`` to NaN (keeps shape intact)."""
    out = pointmap_hwc.copy()
    if mask_bool.shape != pointmap_hwc.shape[:2]:
        raise ValueError(
            f"mask shape {mask_bool.shape} != pointmap {pointmap_hwc.shape[:2]}"
        )
    invert = ~mask_bool
    out[invert, :] = np.nan
    return out


def build_pointmap_input(
    *,
    depth_path: Path,
    cam_k: np.ndarray,
    depth_scale: str | float,
    image_hw: tuple[int, int],
    mask_bool: np.ndarray | None,
    sam3d_input_mode: str,
) -> PointmapInputCompat:
    """Build a SAM3D pointmap input dict in Pytorch3D camera frame.

    ``sam3d_input_mode`` is ``"full"`` (whole pointmap, NaN where depth invalid)
    or ``"cropped"`` (additionally NaN outside the provided mask).
    """
    import cv2

    depth, applied_scale = load_depth_image(depth_path, depth_scale)
    logger.info("depth scale applied: %.6g (input=%s)", applied_scale, depth_scale)

    if depth.shape[:2] != image_hw:
        depth = cv2.resize(
            depth, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_NEAREST,
        )

    depth = sanitize_depth_for_pointmap(depth)
    pointmap = depth_to_pointmap(depth, cam_k)

    if sam3d_input_mode == "cropped":
        if mask_bool is None:
            raise ValueError("sam3d_input_mode='cropped' requires a mask.")
        pointmap = _apply_mask_to_pointmap(pointmap, mask_bool)
        logger.info("pointmap cropped to masked region")
    elif sam3d_input_mode == "full":
        logger.info("pointmap uses full image (NaN where depth invalid)")
    else:
        raise ValueError(f"unknown sam3d_input_mode: {sam3d_input_mode}")

    pointmap_t = torch.from_numpy(pointmap).float()

    # Convert camera-frame points to Pytorch3D camera frame, which is what
    # SAM3D's pointmap pipeline consumes internally.
    try:
        from sam3d_objects.pipeline.inference_pipeline_pointmap import (
            camera_to_pytorch3d_camera,
        )
        from pytorch3d.transforms import Transform3d
    except ImportError as exc:
        raise RuntimeError(
            "sam3d_objects or pytorch3d not importable — are you running in "
            "the sam3d-objects conda env?"
        ) from exc

    points_flat = pointmap_t.reshape(-1, 3)
    cam_to_p3d = (
        Transform3d()
        .rotate(camera_to_pytorch3d_camera(device=points_flat.device).rotation)
        .to(points_flat.device)
    )
    points_p3d = cam_to_p3d.transform_points(points_flat).reshape(pointmap_t.shape)
    return PointmapInputCompat(pointmap=points_p3d, intrinsics=cam_k)


def mask_to_bool(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Normalize a raw mask array to a bool array with shape ``target_hw``."""
    import cv2

    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    if np.issubdtype(mask_np.dtype, np.floating):
        mask_bool = mask_np > 0.5
    else:
        mask_bool = mask_np > 0
    if mask_bool.shape != target_hw:
        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8),
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask_bool
