"""Single-command pipeline: image(+depth) -> SAM2 masks -> SAM3D -> decimate."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from sam3d_asset_extractor import __version__
from sam3d_asset_extractor.config import DecimateOptions, PipelineConfig
from sam3d_asset_extractor.logging_setup import configure_logging, get_logger
from sam3d_asset_extractor.mesh.decimate import decimate_file
from sam3d_asset_extractor.paths import default_sam3d_config
from sam3d_asset_extractor.preflight import PreflightError, preflight_repo_layout, run_preflight
from sam3d_asset_extractor.sam2_mask.runner import run_sam2
from sam3d_asset_extractor.sam3d.export import mesh_format_exts
from sam3d_asset_extractor.sam3d.runner import run_sam3d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sam3d-asset-extractor",
        description="Extract per-object meshes from RGB(-D) images for sim asset creation.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Root for this run's output (masks, sam3d, decimated meshes).")
    parser.add_argument("--depth-image", type=Path, required=True,
                        help="Depth image aligned to --image (required).")
    parser.add_argument("--cam-k", type=Path, required=True,
                        help="Camera intrinsics (3x3 or 'fx fy cx cy') text file (required).")
    parser.add_argument("--depth-scale", type=str, default="auto",
                        help="Depth image scale to meters: 'auto' or a numeric value (default: auto).")

    parser.add_argument("--sam2-mode", choices=["auto", "manual"], default="auto",
                        help="SAM2 masking mode (default: auto).")
    parser.add_argument("--sam2-env", default="sam2")
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")

    parser.add_argument("--sam3d-input", choices=["full", "cropped"], default="full",
                        help="Pointmap input to SAM3D when depth is provided (default: full).")
    parser.add_argument("--sam3d-env", default="sam3d-objects")
    parser.add_argument("--sam3d-config", type=Path, default=None)
    parser.add_argument("--sam3d-seed", type=int, default=42)
    parser.add_argument("--sam3d-compile", action="store_true")
    parser.add_argument("--mesh-format", choices=["glb", "ply", "obj", "all"], default="all")

    parser.add_argument("--decimate", dest="decimate", action=argparse.BooleanOptionalAction,
                        default=True, help="Decimate output meshes (default: on).")
    parser.add_argument("--decimate-method",
                        choices=["auto", "open3d", "trimesh", "cluster"], default="auto")
    parser.add_argument("--decimate-target-faces", type=int, default=20000,
                        help="Target face count. If <= 0, --decimate-ratio is used.")
    parser.add_argument("--decimate-ratio", type=float, default=0.02)
    parser.add_argument("--decimate-min-faces", type=int, default=200)

    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite --output-dir if it already exists.")
    parser.add_argument("--latest-only", action="store_true",
                        help="Process only the most recent mask rather than all masks.")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip dependency preflight checks (use at your own risk).")
    parser.add_argument("--skip-hf-check", action="store_true",
                        help="Skip the HF_TOKEN presence check during preflight.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs + repo layout, then exit.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", type=Path, default=None)
    return parser


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    decimate = DecimateOptions(
        enabled=args.decimate,
        method=args.decimate_method,
        target_faces=args.decimate_target_faces,
        ratio=args.decimate_ratio,
        min_faces=args.decimate_min_faces,
    )
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    return PipelineConfig(
        image=args.image.resolve(),
        output_dir=output_dir,
        depth_image=args.depth_image.resolve(),
        cam_k=args.cam_k.resolve(),
        depth_scale=args.depth_scale,
        sam2_mode=args.sam2_mode,
        sam2_env=args.sam2_env,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_cfg=args.sam2_model_cfg,
        sam3d_input=args.sam3d_input,
        sam3d_env=args.sam3d_env,
        sam3d_config=args.sam3d_config.resolve() if args.sam3d_config is not None else None,
        sam3d_seed=args.sam3d_seed,
        sam3d_compile=args.sam3d_compile,
        mesh_format=args.mesh_format,
        decimate=decimate,
        overwrite=args.overwrite,
        process_all_masks=not args.latest_only,
        dry_run=args.dry_run,
    )


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory not empty: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _select_masks(mask_paths: list[Path], process_all: bool) -> list[Path]:
    if not mask_paths:
        return []
    if process_all:
        return mask_paths
    return [max(mask_paths, key=lambda p: p.stat().st_mtime)]


def _decimate_meshes(sam3d_out_dir: Path, stems: list[str], mesh_format: str,
                     options: DecimateOptions, logger) -> list[Path]:
    exts = mesh_format_exts(mesh_format)
    decimated_dir = sam3d_out_dir.parent / "decimated"
    decimated_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []
    for stem in stems:
        for ext in exts:
            # Prefer posed mesh when available, fall back to the raw mesh export.
            candidates = [
                sam3d_out_dir / f"{stem}_pose_mesh.{ext}",
                sam3d_out_dir / f"{stem}_mesh.{ext}",
            ]
            src = next((c for c in candidates if c.exists()), None)
            if src is None:
                logger.warning("no mesh to decimate for stem=%s ext=%s", stem, ext)
                continue
            dst = decimated_dir / f"{src.stem}_decimated.{ext}"
            try:
                decimate_file(src, dst, options)
                results.append(dst)
            except Exception as exc:
                logger.warning("decimation failed for %s: %s", src.name, exc)
    return results


def _write_manifest(
    output_dir: Path, config: PipelineConfig, status: str,
    start_utc: str, end_utc: str,
    masks: list[Path], sam3d_outputs: list[Path], decimated: list[Path],
) -> None:
    payload = {
        "version": __version__,
        "status": status,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "inputs": {
            "image": str(config.image),
            "depth_image": str(config.depth_image),
            "cam_k": str(config.cam_k),
            "depth_scale": config.depth_scale,
        },
        "options": {
            "sam2_mode": config.sam2_mode,
            "sam3d_input": config.sam3d_input,
            "mesh_format": config.mesh_format,
            "decimate": {
                "enabled": config.decimate.enabled,
                "method": config.decimate.method,
                "target_faces": config.decimate.target_faces,
                "ratio": config.decimate.ratio,
                "min_faces": config.decimate.min_faces,
            },
        },
        "envs": {
            "sam2": config.sam2_env,
            "sam3d": config.sam3d_env,
            "sam3d_config": str(config.sam3d_config or default_sam3d_config()),
        },
        "outputs": {
            "masks": [str(p) for p in masks],
            "sam3d_plys": [str(p) for p in sam3d_outputs],
            "decimated_meshes": [str(p) for p in decimated],
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging(level=args.log_level, log_file=args.log_file)

    config = _config_from_args(args)
    logger.info("image=%s, output=%s", config.image, config.output_dir)

    try:
        config.validate()
    except (ValueError, FileNotFoundError) as exc:
        logger.error("invalid config: %s", exc)
        return 2

    if args.dry_run:
        layout = preflight_repo_layout()
        logger.info("dry-run layout: %s", json.dumps(layout, indent=2))
        logger.info("dry-run OK (skipped conda/HF checks; pass --no-dry-run to execute)")
        return 0

    if not args.skip_preflight:
        try:
            run_preflight(config, skip_hf_check=args.skip_hf_check)
        except PreflightError as exc:
            logger.error("%s", exc)
            return 3

    _prepare_output_dir(config.output_dir, config.overwrite)
    mask_dir = config.output_dir / "sam2_masks"
    sam3d_dir = config.output_dir / "sam3d"
    start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 1) SAM2 masking (inside sam2 env)
    masks_all = run_sam2(config, mask_dir)
    masks = _select_masks(masks_all, config.process_all_masks)
    if not masks:
        logger.error("no masks produced; aborting.")
        _write_manifest(config.output_dir, config, "failed_no_masks",
                        start_utc, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        [], [], [])
        return 4

    # 2) SAM3D inference per mask (inside sam3d-objects env)
    sam3d_outputs: list[Path] = []
    failed: list[str] = []
    for mask_path in masks:
        try:
            ply = run_sam3d(config, mask_path, sam3d_dir)
            sam3d_outputs.append(ply)
        except Exception as exc:
            logger.error("SAM3D failed for %s: %s", mask_path.name, exc)
            failed.append(mask_path.name)

    if not sam3d_outputs:
        logger.error("all SAM3D inferences failed.")
        _write_manifest(config.output_dir, config, "failed_sam3d",
                        start_utc, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        masks, [], [])
        return 5

    # 3) Optional mesh decimation
    decimated: list[Path] = []
    if config.decimate.enabled:
        stems = [p.stem for p in sam3d_outputs]
        decimated = _decimate_meshes(
            sam3d_dir, stems, config.mesh_format, config.decimate, logger,
        )
    else:
        logger.info("decimation disabled (--no-decimate)")

    status = "success" if not failed else "partial"
    end_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_manifest(config.output_dir, config, status, start_utc, end_utc,
                    masks, sam3d_outputs, decimated)
    logger.info("done. status=%s masks=%d sam3d=%d decimated=%d failed=%d",
                status, len(masks), len(sam3d_outputs), len(decimated), len(failed))
    return 0 if status == "success" else 6


if __name__ == "__main__":
    raise SystemExit(main())
