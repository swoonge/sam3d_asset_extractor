"""Unified PLY read/write.

Merges the duplicated PLY handling that used to live across modules. Supports:
  * Plain (x, y, z [rgb]) vertex PLYs
  * Gaussian Splatting PLYs exposing ``f_dc_{0,1,2}`` color channels
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_points_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write (N, 3) points (optionally with N×3 colors in [0, 1]) to ASCII PLY."""
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if points.shape[0] == 0:
            return
        if colors is None:
            np.savetxt(f, points, fmt="%.6f %.6f %.6f")
        else:
            colors_u8 = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(np.uint8)
            for (px, py, pz), (r, g, b) in zip(points, colors_u8):
                f.write(f"{px:.6f} {py:.6f} {pz:.6f} {int(r)} {int(g)} {int(b)}\n")


def read_points_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a PLY and return ``(points, colors|None)``.

    Colors are normalized to [0, 1] floats regardless of the file's channel
    representation (plain RGB uint8 or Gaussian Splat f_dc_*).
    """
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("plyfile is required to parse PLY files.") from exc

    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise RuntimeError(f"PLY file missing vertex data: {path}")
    vertex = ply["vertex"]
    names = vertex.data.dtype.names or ()

    points = np.stack(
        (vertex["x"], vertex["y"], vertex["z"]),
        axis=1,
    ).astype(np.float32)

    colors: np.ndarray | None = None
    if all(name in names for name in ("red", "green", "blue")):
        colors = np.stack(
            (vertex["red"], vertex["green"], vertex["blue"]),
            axis=1,
        ).astype(np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
    elif all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        colors = np.stack(
            (vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]),
            axis=1,
        ).astype(np.float32)
        colors = np.clip(colors + 0.5, 0.0, 1.0)

    return points, colors


def flatten_pointmap(
    pointmap_hwc: np.ndarray, colors_hwc: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Flatten a (H, W, 3) pointmap to (N, 3) after dropping NaN/Inf points."""
    points = pointmap_hwc.reshape(-1, 3)
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    colors = None
    if colors_hwc is not None:
        colors = colors_hwc.reshape(-1, 3)[valid_mask]
        colors = np.clip(colors, 0.0, 1.0)
    return points, colors
