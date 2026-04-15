"""SAM3D inference entry point.

Runs inside the ``sam3d-objects`` conda env (as a subprocess spawned by the
orchestrator, or directly via ``python -m``). Takes an image + mask and
optionally a depth image + camera intrinsics, runs SAM3D, and writes:

  * ``<stem>.ply``                — Gaussian Splatting PLY (SAM3D output)
  * ``<stem>_mesh.{glb,ply,obj}`` — Mesh output (filtered by --mesh-format)
  * ``<stem>_pose.json``          — Pose metadata (rotation/translation/scale)
  * ``<stem>_pose.ply``           — Gaussian PLY transformed into world frame
  * ``<stem>_pose_mesh.{...}``    — Mesh transformed into world frame
  * ``<stem>_pointmap_full.npz``  — (optional) full pointmap used as input
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from sam3d_asset_extractor.common.camera import load_intrinsics_matrix
from sam3d_asset_extractor.common.ply_io import flatten_pointmap, write_points_ply
from sam3d_asset_extractor.logging_setup import configure_logging, get_logger
from sam3d_asset_extractor.paths import default_sam3d_config, resolve_sam3d_root
from sam3d_asset_extractor.sam3d.export import (
    export_mesh,
    export_posed_mesh,
    mesh_format_exts,
    save_pose_transformed_gaussian,
)
from sam3d_asset_extractor.sam3d.pointmap import build_pointmap_input, mask_to_bool
from sam3d_asset_extractor.sam3d.pose import parse_pose

logger = get_logger("sam3d.inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3D inference: image + mask -> mesh/PLY")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Path of the output Gaussian PLY.")
    parser.add_argument("--sam3d-config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--depth-image", type=Path, required=True,
                        help="Depth image aligned to --image (required).")
    parser.add_argument("--cam-k", type=Path, required=True,
                        help="Camera intrinsics (3x3 or fx fy cx cy) text file (required).")
    parser.add_argument("--depth-scale", type=str, default="auto")
    parser.add_argument(
        "--sam3d-input", choices=["full", "cropped"], default="full",
        help=(
            "Pointmap input mode. 'full' (default): whole depth pointmap with "
            "NaN where depth is invalid. 'cropped': additionally NaN outside "
            "the mask."
        ),
    )
    parser.add_argument(
        "--mesh-format", choices=["glb", "ply", "obj", "both", "all"], default="all",
    )
    parser.add_argument(
        "--save-pointmap",
        dest="save_pointmap", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--save-pose",
        dest="save_pose", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--world-z-up",
        dest="world_z_up", action=argparse.BooleanOptionalAction, default=True,
        help="Rotate exported meshes +90° around X so that +Z is up (default: on).",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _load_image_and_mask(image_path: Path, mask_path: Path):
    """Read image/mask via SAM3D's helper module."""
    from inference import load_image, load_mask  # type: ignore[import-not-found]

    return load_image(str(image_path)), load_mask(str(mask_path))


def _prepare_sam3d_env(sam3d_config: Path) -> None:
    """Put sam-3d-objects and its ``notebook/`` dir on ``sys.path`` + chdir.

    The ``notebook`` directory contains ``inference.py`` (SAM3D's high-level
    wrapper) which must be importable before any SAM3D calls are made.
    """
    sam3d_root = resolve_sam3d_root()
    if not sam3d_root.exists():
        raise FileNotFoundError(f"sam-3d-objects root not found: {sam3d_root}")
    if not sam3d_config.exists():
        raise FileNotFoundError(f"Missing SAM3D config: {sam3d_config}")
    notebook_dir = sam3d_root / "notebook"
    for p in (sam3d_root, notebook_dir):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.chdir(sam3d_root)


def _save_full_pointmap(inference_obj, image, mask, output_path: Path) -> None:
    """Save the pointmap SAM3D internally computed (for debugging/reuse)."""
    try:
        rgba = inference_obj.merge_mask_to_rgba(image, mask)
        pointmap_dict = inference_obj._pipeline.compute_pointmap(rgba)
        pointmap_p3d = pointmap_dict["pointmap"].detach()
        colors = pointmap_dict["pts_color"].detach().cpu().permute(1, 2, 0).numpy()
        intrinsics = pointmap_dict.get("intrinsics", None)
        if intrinsics is not None:
            intrinsics = intrinsics.detach().cpu().numpy()
        pointmap = pointmap_p3d.cpu().permute(1, 2, 0).numpy()

        npz_path = output_path.with_name(f"{output_path.stem}_pointmap_full.npz")
        np.savez_compressed(
            npz_path,
            pointmap=pointmap,
            pointmap_p3d=pointmap_p3d.cpu().permute(1, 2, 0).numpy(),
            colors=colors,
            intrinsics=intrinsics,
            pointmap_frame="pytorch3d_camera",
        )
        flat_pts, flat_colors = flatten_pointmap(pointmap, colors)
        ply_path = output_path.with_name(f"{output_path.stem}_pointmap_full.ply")
        write_points_ply(ply_path, flat_pts, flat_colors)
        logger.info("saved pointmap npz/ply alongside %s", output_path.name)
    except Exception as exc:
        logger.warning("failed to save pointmap output: %s", exc)


def _save_pose_artifacts(
    output_path: Path,
    pose_rot: np.ndarray,
    pose_trans: np.ndarray,
    pose_scale: np.ndarray,
    rot_meta: dict,
    frame: str,
) -> None:
    pose_json = output_path.with_name(f"{output_path.stem}_pose.json")
    payload = {
        "rotation": rot_meta,
        "rotation_matrix": pose_rot.tolist(),
        "translation": pose_trans.tolist(),
        "scale": pose_scale.tolist(),
        "frame": frame,
    }
    with pose_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("saved pose json: %s", pose_json)

    pose_ply = output_path.with_name(f"{output_path.stem}_pose.ply")
    save_pose_transformed_gaussian(output_path, pose_ply, pose_scale, pose_rot, pose_trans)
    logger.info("saved posed gaussian ply: %s", pose_ply)


def main() -> int:
    args = parse_args()
    configure_logging(level=args.log_level)

    sam3d_config = (args.sam3d_config or default_sam3d_config()).resolve()
    image_path = args.image.resolve()
    mask_path = args.mask.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    _prepare_sam3d_env(sam3d_config)
    from inference import Inference  # type: ignore[import-not-found]

    logger.info("loading SAM3D Inference (config=%s, compile=%s)", sam3d_config, args.compile)
    inf = Inference(str(sam3d_config), compile=args.compile)

    image, mask = _load_image_and_mask(image_path, mask_path)

    # Depth + intrinsics are always required — build the pointmap input.
    cam_k = load_intrinsics_matrix(args.cam_k.resolve())
    image_hw = (image.shape[0], image.shape[1])
    mask_bool = mask_to_bool(mask, target_hw=image_hw)
    pointmap_input = build_pointmap_input(
        depth_path=args.depth_image.resolve(),
        cam_k=cam_k,
        depth_scale=args.depth_scale,
        image_hw=image_hw,
        mask_bool=mask_bool if args.sam3d_input == "cropped" else None,
        sam3d_input_mode=args.sam3d_input,
    )

    logger.info("running SAM3D inference (seed=%d, input=%s)", args.seed, args.sam3d_input)
    try:
        output = inf(image, mask, seed=args.seed, pointmap=pointmap_input)
    except AttributeError as exc:
        if "has no attribute 'to'" in str(exc):
            logger.warning("legacy pipeline detected; retrying with tensor-only pointmap")
            retry = pointmap_input.get("pointmap", pointmap_input) \
                if isinstance(pointmap_input, dict) else pointmap_input
            output = inf(image, mask, seed=args.seed, pointmap=retry)
        else:
            raise

    if "gs" not in output:
        logger.error("SAM3D output missing 'gs' key")
        return 1

    output["gs"].save_ply(str(output_path))
    logger.info("saved gaussian PLY: %s", output_path)

    if args.save_pointmap:
        _save_full_pointmap(inf, image, mask, output_path)

    # Pose extraction + posed artifacts (pointmap input puts SAM3D in the
    # Pytorch3D camera frame, so pose metadata is reported in that frame).
    pose_rot, pose_trans, pose_scale, rot_meta = parse_pose(output)
    pose_frame = "pytorch3d_camera"
    has_full_pose = (
        args.save_pose
        and pose_rot is not None
        and pose_trans is not None
        and pose_scale is not None
    )
    if has_full_pose:
        _save_pose_artifacts(
            output_path, pose_rot, pose_trans, pose_scale, rot_meta, pose_frame,
        )

    # Mesh export
    mesh_raw = output.get("mesh")
    if isinstance(mesh_raw, (list, tuple)) and mesh_raw:
        mesh_raw = mesh_raw[0]
    mesh_glb = output.get("glb")
    mesh_any = mesh_glb or mesh_raw
    if mesh_any is None:
        logger.warning("SAM3D output missing mesh; skipping mesh export")
        return 0

    exts = mesh_format_exts(args.mesh_format)
    for ext in exts:
        src = mesh_glb if (ext == "glb" and mesh_glb is not None) else mesh_any
        mesh_path = output_path.with_name(f"{output_path.stem}_mesh.{ext}")
        try:
            export_mesh(src, mesh_path, world_z_up=args.world_z_up)
            logger.info("saved mesh: %s", mesh_path)
        except Exception as exc:
            logger.warning("failed to save %s: %s", mesh_path.name, exc)

    if has_full_pose and mesh_raw is not None:
        for ext in exts:
            pose_mesh_path = output_path.with_name(f"{output_path.stem}_pose_mesh.{ext}")
            try:
                export_posed_mesh(
                    mesh_raw, pose_mesh_path, pose_scale, pose_rot, pose_trans,
                    world_z_up=args.world_z_up,
                )
                logger.info("saved posed mesh: %s", pose_mesh_path)
            except Exception as exc:
                logger.warning("failed to save %s: %s", pose_mesh_path.name, exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
