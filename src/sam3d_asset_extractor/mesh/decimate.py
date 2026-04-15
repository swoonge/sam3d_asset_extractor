"""Mesh decimation.

Three backends, tried in order when ``method='auto'``:
  1. ``open3d.geometry.TriangleMesh.simplify_quadric_decimation`` — highest quality
  2. trimesh's built-in quadric decimation (API name varies by version)
  3. vertex clustering fallback (pure numpy, always works)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sam3d_asset_extractor.config import DecimateMethod, DecimateOptions
from sam3d_asset_extractor.logging_setup import configure_logging, get_logger

logger = get_logger("mesh.decimate")


@dataclass
class DecimationResult:
    """Summary of a decimation run (used by CLI and tests)."""

    output_path: Path
    method: str
    face_count_before: int
    face_count_after: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decimate a mesh to a target face count.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path (default: <input_stem>_decimated<ext>).",
    )
    parser.add_argument(
        "--method", choices=["auto", "open3d", "trimesh", "cluster"], default="auto",
    )
    parser.add_argument(
        "--target-faces", type=int, default=20000,
        help="Target face count. Ignored when <= 0; then --ratio is used.",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.02,
        help="Target face ratio in (0, 1]. Used when --target-faces <= 0.",
    )
    parser.add_argument("--min-faces", type=int, default=200)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_decimated{input_path.suffix}")


def _cleanup_faces_compat(mesh) -> None:
    """Clean duplicate/degenerate faces across trimesh v3/v4 API differences."""
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    elif hasattr(mesh, "unique_faces") and hasattr(mesh, "update_faces"):
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, "nondegenerate_faces") and hasattr(mesh, "update_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()


def load_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"Scene has no geometry: {path}")
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type: {type(mesh)}")
    _cleanup_faces_compat(mesh)
    return mesh


def _vertex_colors(mesh) -> np.ndarray | None:
    if not hasattr(mesh, "visual") or mesh.visual is None:
        return None
    if getattr(mesh.visual, "kind", None) != "vertex":
        return None
    colors = mesh.visual.vertex_colors
    if colors is None or colors.size == 0:
        return None
    colors = colors[:, :3].astype(np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    return colors


def simplify_open3d(mesh, target_faces: int):
    import open3d as o3d
    import trimesh

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    colors = _vertex_colors(mesh)
    if colors is not None and colors.shape[0] == mesh.vertices.shape[0]:
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(int(target_faces))
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_unreferenced_vertices()

    out = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False,
    )
    if o3d_mesh.has_vertex_colors():
        v = np.asarray(o3d_mesh.vertex_colors)
        out.visual.vertex_colors = np.clip(v * 255.0, 0, 255).astype(np.uint8)
    return out, "open3d"


def simplify_trimesh(mesh, target_faces: int):
    if hasattr(mesh, "simplify_quadratic_decimation"):
        return mesh.simplify_quadratic_decimation(int(target_faces)), "trimesh"
    if hasattr(mesh, "simplify_quadric_decimation"):
        try:
            out = mesh.simplify_quadric_decimation(face_count=int(target_faces))
        except TypeError:
            out = mesh.simplify_quadric_decimation(int(target_faces))
        return out, "trimesh"
    raise RuntimeError(
        "No trimesh decimation API found "
        "(expected simplify_quadratic_decimation or simplify_quadric_decimation)."
    )


def simplify_cluster(mesh, target_faces: int):
    """Vertex-clustering fallback using a coarse voxel grid."""
    import trimesh

    if target_faces <= 0:
        return mesh, "cluster"
    faces = mesh.faces
    verts = mesh.vertices
    if faces.shape[0] <= target_faces:
        return mesh, "cluster"

    ratio = float(target_faces) / max(1, faces.shape[0])
    target_vertices = max(4, int(verts.shape[0] * ratio))

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    bbox = maxs - mins
    volume = float(np.prod(bbox))
    if not np.isfinite(volume) or volume <= 0:
        return mesh, "cluster"
    voxel = (volume / max(1, target_vertices)) ** (1.0 / 3.0)
    if voxel <= 0 or not np.isfinite(voxel):
        return mesh, "cluster"

    grid = np.floor((verts - mins) / voxel).astype(np.int64)
    unique, unique_idx, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
    new_verts = verts[unique_idx]
    new_faces = inverse[faces]
    valid = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[valid]
    out = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=True)

    colors = _vertex_colors(mesh)
    if colors is not None and colors.shape[0] == verts.shape[0]:
        sums = np.zeros((unique.shape[0], 3), dtype=np.float32)
        counts = np.zeros((unique.shape[0], 1), dtype=np.float32)
        np.add.at(sums, inverse, colors)
        np.add.at(counts, inverse, 1.0)
        avg = sums / np.maximum(counts, 1.0)
        out.visual.vertex_colors = np.clip(avg * 255.0, 0, 255).astype(np.uint8)
    return out, "cluster"


def pick_target_faces(face_count: int, ratio: float, target_faces: int, min_faces: int) -> int:
    """Resolve the effective target face count from CLI flags."""
    if target_faces > 0:
        target = target_faces
    else:
        ratio = max(1e-6, min(1.0, float(ratio)))
        target = int(face_count * ratio)
    target = max(min_faces, target)
    target = min(face_count, target)
    return int(target)


def decimate_mesh(mesh, target_faces: int, method: DecimateMethod):
    """Dispatch to the selected backend with sensible fallbacks in ``auto``."""
    if method == "auto":
        candidates: list[DecimateMethod] = ["open3d", "trimesh", "cluster"]
    else:
        candidates = [method]
    tried: list[str] = []
    for name in candidates:
        try:
            if name == "open3d":
                return simplify_open3d(mesh, target_faces)
            if name == "trimesh":
                return simplify_trimesh(mesh, target_faces)
            if name == "cluster":
                return simplify_cluster(mesh, target_faces)
        except Exception as exc:
            tried.append(f"{name}: {exc}")
            logger.warning("decimation backend '%s' failed: %s", name, exc)
    raise RuntimeError("All decimation backends failed:\n  " + "\n  ".join(tried))


def decimate_file(
    input_path: Path, output_path: Path | None, options: DecimateOptions,
) -> DecimationResult:
    """High-level API: read, decimate, write. Returns a summary."""
    if output_path is None:
        output_path = default_output_path(input_path)
    mesh = load_mesh(input_path)
    face_count = int(mesh.faces.shape[0])
    if face_count <= 0:
        logger.warning("mesh has no faces; skipping decimation")
        mesh.export(output_path)
        return DecimationResult(output_path, "noop", 0, 0)

    target = pick_target_faces(face_count, options.ratio, options.target_faces, options.min_faces)
    if target >= face_count:
        logger.info("target faces (%d) >= input (%d); copying", target, face_count)
        mesh.export(output_path)
        return DecimationResult(output_path, "copy", face_count, face_count)

    mesh_out, method = decimate_mesh(mesh, target, options.method)
    mesh_out.export(output_path)
    after = int(mesh_out.faces.shape[0])
    logger.info(
        "decimated via %s: %d -> %d faces (target=%d)",
        method, face_count, after, target,
    )
    return DecimationResult(output_path, method, face_count, after)


def main() -> int:
    args = parse_args()
    configure_logging(level=args.log_level)
    input_path = args.input.resolve()
    if not input_path.exists():
        logger.error("missing input mesh: %s", input_path)
        return 1
    output_path = args.output.resolve() if args.output is not None else None
    options = DecimateOptions(
        enabled=True,
        method=args.method,
        target_faces=args.target_faces,
        ratio=args.ratio,
        min_faces=args.min_faces,
    )
    decimate_file(input_path, output_path, options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
