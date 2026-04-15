"""Pre-run dependency and input validation."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from sam3d_asset_extractor.config import PipelineConfig
from sam3d_asset_extractor.logging_setup import get_logger
from sam3d_asset_extractor.paths import repo_root, resolve_sam2_root, resolve_sam3d_root

logger = get_logger("preflight")


class PreflightError(RuntimeError):
    """One or more preflight checks failed."""


def _run_python(env_name: str, code: str) -> tuple[bool, str]:
    cmd = ["conda", "run", "-n", env_name, "python", "-c", code]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return True, ""
    stderr = (proc.stderr or "").strip()
    if len(stderr) > 400:
        stderr = stderr[-400:]
    return False, stderr


def _check_local_repo_import(env_name: str, module_name: str, local_root: Path) -> tuple[bool, str]:
    if not local_root.exists():
        return False, f"Local repo for '{module_name}' not found at: {local_root}"
    if module_name == "sam3d_objects":
        body = "import os; os.environ['LIDRA_SKIP_INIT']='true'; import sam3d_objects"
    else:
        body = f"import {module_name}"
    code = f"import sys; sys.path.insert(0, {repr(str(local_root))}); {body}"
    ok, err = _run_python(env_name, code)
    if ok:
        return True, ""
    return False, f"conda env '{env_name}' cannot import '{module_name}' from {local_root}: {err}"


def _check_import(env_name: str, module_name: str) -> tuple[bool, str]:
    ok, err = _run_python(env_name, f"import {module_name}")
    if ok:
        return True, ""
    return False, f"conda env '{env_name}' missing import '{module_name}': {err}"


def _check_hf_token() -> tuple[bool, str]:
    """SAM3D uses the HF hub; require HF_TOKEN to be set in the environment."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True, ""
    return False, (
        "HF_TOKEN not set. Export the token before running "
        "(e.g. `export HF_TOKEN=hf_...`)."
    )


def run_preflight(config: PipelineConfig, skip_hf_check: bool = False) -> None:
    """Validate conda envs, local repos, inputs, and secrets.

    Raises ``PreflightError`` when any check fails. Running with ``dry_run``
    still invokes preflight so users can confirm setup without heavy work.
    """
    errors: list[str] = []

    if shutil.which("conda") is None:
        errors.append("conda not found in PATH.")

    try:
        config.validate()
    except (ValueError, FileNotFoundError) as exc:
        errors.append(str(exc))

    sam2_root = resolve_sam2_root()
    sam3d_root = resolve_sam3d_root()

    if shutil.which("conda") is not None:
        ok, msg = _check_local_repo_import(config.sam2_env, "sam2", sam2_root)
        if not ok:
            errors.append(msg)
        ok, msg = _check_local_repo_import(config.sam3d_env, "sam3d_objects", sam3d_root)
        if not ok:
            errors.append(msg)
        ok, msg = _check_import(config.sam3d_env, "trimesh")
        if not ok:
            errors.append(msg)

    if not skip_hf_check:
        ok, msg = _check_hf_token()
        if not ok:
            errors.append(msg)

    if errors:
        joined = "\n  - " + "\n  - ".join(errors)
        raise PreflightError(f"Preflight failed:{joined}")
    logger.info("preflight OK (sam2=%s, sam3d=%s)", config.sam2_env, config.sam3d_env)


def preflight_repo_layout() -> dict:
    """Light-weight layout check (no subprocess) used by ``--dry-run``."""
    return {
        "repo_root": str(repo_root()),
        "sam2_root": str(resolve_sam2_root()),
        "sam2_root_exists": resolve_sam2_root().exists(),
        "sam3d_root": str(resolve_sam3d_root()),
        "sam3d_root_exists": resolve_sam3d_root().exists(),
    }
