"""Subprocess-mocked tests for :mod:`sam3d_asset_extractor.sam2_mask.runner`.

We never spawn conda here; instead ``subprocess.run`` is replaced with a fake
that records the command + env, and (for the success path) also writes mask
files into the output directory so the runner's glob picks them up.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sam3d_asset_extractor.config import PipelineConfig
from sam3d_asset_extractor.sam2_mask import runner as sam2_runner


@pytest.fixture
def base_config(tmp_path):
    image = tmp_path / "scene.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9")
    depth = tmp_path / "depth.png"
    depth.write_bytes(b"\x00")
    cam_k = tmp_path / "cam_k.txt"
    cam_k.write_text("500 0 320 0 500 240 0 0 1\n")
    return PipelineConfig(
        image=image, output_dir=tmp_path / "out",
        depth_image=depth, cam_k=cam_k,
    )


class _RecordingRun:
    """Captures the subprocess call and simulates the mask-producing side effect."""

    def __init__(self, image_stem: str, mask_count: int = 3, exit_code: int = 0):
        self.calls: list[tuple[list[str], dict]] = []
        self.image_stem = image_stem
        self.mask_count = mask_count
        self.exit_code = exit_code

    def __call__(self, cmd, check=True, env=None, **kwargs):
        self.calls.append((list(cmd), dict(env or {})))
        if self.exit_code != 0:
            raise subprocess.CalledProcessError(self.exit_code, cmd)
        # Find the --output-dir argument and create dummy mask files there.
        if "--output-dir" in cmd:
            idx = cmd.index("--output-dir")
            mask_dir = Path(cmd[idx + 1])
            mask_dir.mkdir(parents=True, exist_ok=True)
            for i in range(self.mask_count):
                (mask_dir / f"{self.image_stem}_{i:03d}.png").write_bytes(b"PNG")
        return subprocess.CompletedProcess(cmd, 0)


@pytest.fixture
def patch_conda_and_subprocess(monkeypatch):
    def _patch(image_stem: str, *, mask_count: int = 3, exit_code: int = 0):
        monkeypatch.setattr(sam2_runner.shutil, "which",
                            lambda name: "/usr/bin/conda" if name == "conda" else None)
        recorder = _RecordingRun(image_stem, mask_count=mask_count, exit_code=exit_code)
        monkeypatch.setattr(sam2_runner.subprocess, "run", recorder)
        return recorder

    return _patch


def test_builds_auto_command(base_config, tmp_path, patch_conda_and_subprocess):
    recorder = patch_conda_and_subprocess(base_config.image.stem)
    base_config.sam2_mode = "auto"
    masks = sam2_runner.run_sam2(base_config, tmp_path / "masks")
    assert len(recorder.calls) == 1
    cmd, _ = recorder.calls[0]
    assert cmd[:5] == ["conda", "run", "--no-capture-output", "-n", "sam2"]
    assert "sam3d_asset_extractor.sam2_mask.auto" in cmd
    assert "--image" in cmd and str(base_config.image) in cmd
    assert "--output-dir" in cmd
    assert len(masks) == 3


def test_builds_manual_command(base_config, tmp_path, patch_conda_and_subprocess):
    recorder = patch_conda_and_subprocess(base_config.image.stem)
    base_config.sam2_mode = "manual"
    sam2_runner.run_sam2(base_config, tmp_path / "masks")
    cmd, _ = recorder.calls[0]
    assert "sam3d_asset_extractor.sam2_mask.manual" in cmd
    assert "sam3d_asset_extractor.sam2_mask.auto" not in cmd


def test_auto_mode_forwards_depth_image(base_config, tmp_path, patch_conda_and_subprocess):
    depth = tmp_path / "depth.png"
    depth.write_bytes(b"\x00")
    base_config.sam2_mode = "auto"
    base_config.depth_image = depth
    base_config.cam_k = tmp_path / "cam_k.txt"
    recorder = patch_conda_and_subprocess(base_config.image.stem)
    sam2_runner.run_sam2(base_config, tmp_path / "masks")
    cmd, _ = recorder.calls[0]
    assert "--depth-image" in cmd
    assert str(depth) in cmd


def test_manual_mode_skips_depth_even_if_provided(base_config, tmp_path, patch_conda_and_subprocess):
    depth = tmp_path / "depth.png"
    depth.write_bytes(b"\x00")
    base_config.sam2_mode = "manual"
    base_config.depth_image = depth
    base_config.cam_k = tmp_path / "cam_k.txt"
    recorder = patch_conda_and_subprocess(base_config.image.stem)
    sam2_runner.run_sam2(base_config, tmp_path / "masks")
    cmd, _ = recorder.calls[0]
    assert "--depth-image" not in cmd


def test_forwards_custom_checkpoint_and_cfg(base_config, tmp_path, patch_conda_and_subprocess):
    base_config.sam2_checkpoint = tmp_path / "custom.pt"
    base_config.sam2_model_cfg = "configs/custom.yaml"
    recorder = patch_conda_and_subprocess(base_config.image.stem)
    sam2_runner.run_sam2(base_config, tmp_path / "masks")
    cmd, _ = recorder.calls[0]
    assert "--checkpoint" in cmd and str(base_config.sam2_checkpoint) in cmd
    assert "--model-cfg" in cmd and "configs/custom.yaml" in cmd


def test_missing_conda_raises(base_config, tmp_path, monkeypatch):
    monkeypatch.setattr(sam2_runner.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="conda not found"):
        sam2_runner.run_sam2(base_config, tmp_path / "masks")


def test_subprocess_failure_propagates(base_config, tmp_path, patch_conda_and_subprocess):
    patch_conda_and_subprocess(base_config.image.stem, exit_code=2)
    with pytest.raises(subprocess.CalledProcessError):
        sam2_runner.run_sam2(base_config, tmp_path / "masks")


def test_pythonpath_is_propagated(base_config, tmp_path, patch_conda_and_subprocess):
    recorder = patch_conda_and_subprocess(base_config.image.stem)
    sam2_runner.run_sam2(base_config, tmp_path / "masks")
    _, env = recorder.calls[0]
    pypath = env.get("PYTHONPATH", "")
    assert "sam3d_asset_extractor/src" in pypath


def test_returns_only_matching_masks(base_config, tmp_path, patch_conda_and_subprocess):
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    # Pre-existing files that should be ignored by the glob (different stem / visualization).
    (mask_dir / "vis_scene.png").write_bytes(b"PNG")
    (mask_dir / "other_image_000.png").write_bytes(b"PNG")
    patch_conda_and_subprocess(base_config.image.stem, mask_count=2)
    masks = sam2_runner.run_sam2(base_config, mask_dir)
    assert len(masks) == 2
    assert all(p.name.startswith(f"{base_config.image.stem}_") for p in masks)
