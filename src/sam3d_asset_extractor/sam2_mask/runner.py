"""Spawn a SAM2 subprocess inside the ``sam2`` conda env."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from sam3d_asset_extractor.config import PipelineConfig
from sam3d_asset_extractor.logging_setup import get_logger
from sam3d_asset_extractor.paths import repo_root

logger = get_logger("sam2_mask.runner")


def _conda_available() -> bool:
    return shutil.which("conda") is not None


def _build_sam2_cmd(config: PipelineConfig, mask_dir: Path) -> list[str]:
    module = (
        "sam3d_asset_extractor.sam2_mask.auto"
        if config.sam2_mode == "auto"
        else "sam3d_asset_extractor.sam2_mask.manual"
    )
    cmd = [
        "conda", "run", "--no-capture-output", "-n", config.sam2_env,
        "python", "-m", module,
        "--image", str(config.image),
        "--output-dir", str(mask_dir),
    ]
    if config.sam2_checkpoint is not None:
        cmd += ["--checkpoint", str(config.sam2_checkpoint)]
    if config.sam2_model_cfg:
        cmd += ["--model-cfg", config.sam2_model_cfg]
    if config.sam2_mode == "auto" and config.depth_image is not None:
        cmd += ["--depth-image", str(config.depth_image)]
    return cmd


def run_sam2(config: PipelineConfig, mask_dir: Path) -> list[Path]:
    """Run SAM2 masking and return the list of produced mask PNG paths."""
    if not _conda_available():
        raise RuntimeError("conda not found in PATH; cannot run SAM2 masking.")

    mask_dir.mkdir(parents=True, exist_ok=True)

    env_path = _prepare_pythonpath_env()
    cmd = _build_sam2_cmd(config, mask_dir)
    logger.info("SAM2 (%s) via conda env '%s'", config.sam2_mode, config.sam2_env)
    logger.debug("cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env_path)

    image_stem = config.image.stem
    masks = sorted(mask_dir.glob(f"{image_stem}_*.png"))
    logger.info("produced %d mask(s) in %s", len(masks), mask_dir)
    return masks


def _prepare_pythonpath_env() -> dict:
    """Propagate the current PYTHONPATH so the conda env can import our package.

    The subprocess needs to ``import sam3d_asset_extractor.*``. Since we're
    running from the package sources (not always installed), pass through the
    parent's PYTHONPATH plus the ``src/`` directory.
    """
    import os

    env = os.environ.copy()
    src_dir = repo_root() / "src"
    extra_paths: list[str] = []
    if src_dir.exists():
        extra_paths.append(str(src_dir))
    existing = env.get("PYTHONPATH", "")
    if existing:
        extra_paths.append(existing)
    if extra_paths:
        env["PYTHONPATH"] = ":".join(extra_paths)
    return env
