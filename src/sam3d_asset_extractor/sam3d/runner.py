"""Spawn SAM3D inference as a subprocess inside the ``sam3d-objects`` env."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from sam3d_asset_extractor.config import PipelineConfig
from sam3d_asset_extractor.logging_setup import get_logger
from sam3d_asset_extractor.paths import default_sam3d_config, repo_root

logger = get_logger("sam3d.runner")


def _conda_available() -> bool:
    return shutil.which("conda") is not None


def _build_cmd(config: PipelineConfig, mask_path: Path, output_ply: Path) -> list[str]:
    sam3d_config = config.sam3d_config or default_sam3d_config()
    cmd = [
        "conda", "run", "--no-capture-output", "-n", config.sam3d_env,
        "python", "-m", "sam3d_asset_extractor.sam3d.inference",
        "--image", str(config.image),
        "--mask", str(mask_path),
        "--output", str(output_ply),
        "--sam3d-config", str(sam3d_config),
        "--seed", str(config.sam3d_seed),
        "--mesh-format", config.mesh_format,
        "--sam3d-input", config.sam3d_input,
        "--depth-image", str(config.depth_image),
        "--cam-k", str(config.cam_k),
        "--depth-scale", str(config.depth_scale),
    ]
    if config.sam3d_compile:
        cmd.append("--compile")
    return cmd


def _prepare_env() -> dict:
    env = os.environ.copy()
    src_dir = repo_root() / "src"
    extras: list[str] = []
    if src_dir.exists():
        extras.append(str(src_dir))
    existing = env.get("PYTHONPATH", "")
    if existing:
        extras.append(existing)
    if extras:
        env["PYTHONPATH"] = ":".join(extras)
    return env


def run_sam3d(config: PipelineConfig, mask_path: Path, output_dir: Path) -> Path:
    """Run SAM3D inference for a single mask and return the primary PLY path."""
    if not _conda_available():
        raise RuntimeError("conda not found in PATH; cannot run SAM3D inference.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_ply = output_dir / f"{mask_path.stem}.ply"
    cmd = _build_cmd(config, mask_path, output_ply)
    logger.info("SAM3D inference for mask=%s -> %s", mask_path.name, output_ply.name)
    logger.debug("cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=_prepare_env())
    return output_ply
