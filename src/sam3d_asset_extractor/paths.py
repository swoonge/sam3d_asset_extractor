"""Path resolution for external repositories (SAM2 and SAM3D Objects).

The pipeline relies on two sibling repositories that are NOT vendored into
this package. Users must either:
  1. Set ``SAM2_ROOT`` / ``SAM3D_ROOT`` env vars, OR
  2. Place them as siblings of this repo (``../sam2``, ``../sam-3d-objects``), OR
  3. Symlink ``sam2`` / ``sam-3d-objects`` inside this repo.
"""

from __future__ import annotations

import os
from pathlib import Path

from sam3d_asset_extractor.logging_setup import get_logger

logger = get_logger("paths")


def repo_root() -> Path:
    """Absolute path to the package's repository root (one level above ``src/``).

    File layout: ``<repo>/src/sam3d_asset_extractor/paths.py`` — three parents
    up lands on ``<repo>``.
    """
    return Path(__file__).resolve().parents[2]


def _candidate_dirs(env_var: str, local_name: str) -> list[Path]:
    env_val = os.environ.get(env_var)
    candidates: list[Path] = []
    if env_val:
        candidates.append(Path(env_val))
    root = repo_root()
    candidates.append(root / local_name)
    candidates.append(root.parent / local_name)
    return candidates


def resolve_sam2_root() -> Path:
    """Return an existing SAM2 repo path, or the best-guess candidate if none exist."""
    candidates = _candidate_dirs("SAM2_ROOT", "sam2")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    logger.warning("SAM2 repo not found; tried: %s", [str(c) for c in candidates])
    return candidates[0]


def resolve_sam3d_root() -> Path:
    """Return an existing sam-3d-objects repo path, or the best-guess candidate."""
    candidates = _candidate_dirs("SAM3D_ROOT", "sam-3d-objects")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    logger.warning("sam-3d-objects repo not found; tried: %s", [str(c) for c in candidates])
    return candidates[0]


def default_sam3d_config() -> Path:
    """Default SAM3D pipeline YAML inside the sam-3d-objects checkpoints dir."""
    return resolve_sam3d_root() / "checkpoints" / "hf" / "pipeline.yaml"
