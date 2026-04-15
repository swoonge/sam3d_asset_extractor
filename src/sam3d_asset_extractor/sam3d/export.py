"""Mesh/PLY export utilities for SAM3D outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Counterclockwise +90deg rotation around +X (right-hand rule). SAM3D outputs
# live in the Pytorch3D camera frame; applying this rotation brings meshes
# into a world +Z-up convention that most simulators expect.
OUTPUT_ROT_X_CCW_90 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def export_mesh(mesh, output_path: Path, world_z_up: bool = True) -> None:
    """Save a trimesh mesh to ``output_path`` (glb/ply/obj by extension).

    If ``world_z_up`` is True, rotate the mesh +90° around X before writing so
    that the Pytorch3D camera frame (Y-down, Z-forward) becomes a world Z-up
    orientation. Disable for the raw model frame.
    """
    try:
        import trimesh
    except ImportError:
        if hasattr(mesh, "export"):
            mesh.export(output_path)
        return

    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            return
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))

    if isinstance(mesh, trimesh.Trimesh):
        out = mesh.copy()
        if world_z_up:
            tr = np.eye(4, dtype=np.float32)
            tr[:3, :3] = OUTPUT_ROT_X_CCW_90
            out.apply_transform(tr)
        out.export(output_path)
        return

    if hasattr(mesh, "export"):
        mesh.export(output_path)


def export_posed_mesh(
    mesh,
    output_path: Path,
    scale: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    world_z_up: bool = True,
) -> None:
    """Apply SAM3D's pose (scale, rotation, translation) to a mesh and export it."""
    try:
        import trimesh
    except ImportError:
        return
    if not isinstance(mesh, trimesh.Trimesh):
        return
    posed = mesh.copy()
    tr = np.eye(4, dtype=np.float32)
    tr[:3, :3] = rotation @ np.diag(scale.astype(np.float32))
    tr[:3, 3] = translation.astype(np.float32)
    posed.apply_transform(tr)
    export_mesh(posed, output_path, world_z_up=world_z_up)


def save_pose_transformed_gaussian(
    source_ply: Path,
    output_ply: Path,
    scale: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> None:
    """Apply pose transform to a Gaussian Splatting PLY and write a new PLY.

    Only xyz are repositioned; the remaining per-point fields are preserved.
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as exc:
        raise RuntimeError("plyfile is required to transform Gaussian PLYs.") from exc

    from sam3d_asset_extractor.sam3d.pose import apply_similarity

    ply = PlyData.read(str(source_ply))
    if "vertex" not in ply:
        raise RuntimeError(f"Gaussian PLY missing vertex data: {source_ply}")
    vertex = ply["vertex"]
    points = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1).astype(np.float32)
    points_t = apply_similarity(points, scale, rotation, translation)

    vertex.data["x"] = points_t[:, 0].astype(vertex.data.dtype["x"])
    vertex.data["y"] = points_t[:, 1].astype(vertex.data.dtype["y"])
    vertex.data["z"] = points_t[:, 2].astype(vertex.data.dtype["z"])

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex.data, "vertex")], text=False).write(str(output_ply))


def mesh_format_exts(mesh_format: str) -> list[str]:
    """Expand a --mesh-format flag into concrete extensions."""
    if mesh_format == "all":
        return ["glb", "ply", "obj"]
    if mesh_format == "both":
        return ["glb", "ply"]
    return [mesh_format]
