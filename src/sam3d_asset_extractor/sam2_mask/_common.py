"""Shared helpers for SAM2 mask entry points."""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

from sam3d_asset_extractor.paths import resolve_sam2_root


def autocast_context(device: str):
    """bfloat16 autocast on CUDA, no-op elsewhere."""
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def resolve_device(device_flag: str) -> str:
    """Resolve 'auto' | 'cuda' | 'cpu' to an actual device string."""
    if device_flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_flag


def default_checkpoint_path() -> Path:
    return resolve_sam2_root() / "checkpoints" / "sam2.1_hiera_large.pt"


def default_model_cfg() -> str:
    return "configs/sam2.1/sam2.1_hiera_l.yaml"


def prepare_sam2_import() -> tuple[Path, Path]:
    """Configure sys.path/cwd so ``import sam2.*`` works and return (sam2_root, sam2_pkg_root).

    SAM2's hydra configs resolve relative to ``cwd``, so we must ``chdir`` into
    the repo root after putting it on ``sys.path``.
    """
    sam2_root = resolve_sam2_root()
    sam2_pkg_root = sam2_root / "sam2"
    if not sam2_pkg_root.exists():
        raise FileNotFoundError(
            f"SAM2 package root not found: {sam2_pkg_root}. "
            "Set SAM2_ROOT or place the sam2 repo as a sibling."
        )
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    os.chdir(sam2_root)
    return sam2_root, sam2_pkg_root


def resolve_model_cfg(model_cfg_arg: str, sam2_pkg_root: Path) -> str:
    """Normalize a user-supplied model config path into the form SAM2/hydra expects."""
    model_cfg_path = Path(model_cfg_arg)
    if model_cfg_path.is_absolute():
        if not model_cfg_path.exists():
            raise FileNotFoundError(f"Missing model config: {model_cfg_path}")
        if sam2_pkg_root in model_cfg_path.parents:
            return model_cfg_path.relative_to(sam2_pkg_root).as_posix()
        raise FileNotFoundError(
            "Model config must live under the sam2 package configs directory; "
            f"got: {model_cfg_path}"
        )
    if not (sam2_pkg_root / model_cfg_path).exists():
        raise FileNotFoundError(
            f"Missing model config: {sam2_pkg_root / model_cfg_path}"
        )
    return model_cfg_path.as_posix()
