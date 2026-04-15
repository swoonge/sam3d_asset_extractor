"""Subprocess-mocked tests for :mod:`sam3d_asset_extractor.sam3d.runner`.

The conda+SAM3D subprocess is replaced with a fake that captures args/env and
never runs real inference.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sam3d_asset_extractor.config import PipelineConfig
from sam3d_asset_extractor.sam3d import runner as sam3d_runner


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


@pytest.fixture
def mask_path(tmp_path):
    p = tmp_path / "mask_000.png"
    p.write_bytes(b"PNG")
    return p


class _RecordingRun:
    def __init__(self, exit_code: int = 0):
        self.calls: list[tuple[list[str], dict]] = []
        self.exit_code = exit_code

    def __call__(self, cmd, check=True, env=None, **kwargs):
        self.calls.append((list(cmd), dict(env or {})))
        if self.exit_code != 0:
            raise subprocess.CalledProcessError(self.exit_code, cmd)
        return subprocess.CompletedProcess(cmd, 0)


@pytest.fixture
def patch_conda_and_subprocess(monkeypatch):
    def _patch(*, exit_code: int = 0):
        monkeypatch.setattr(sam3d_runner.shutil, "which",
                            lambda name: "/usr/bin/conda" if name == "conda" else None)
        recorder = _RecordingRun(exit_code=exit_code)
        monkeypatch.setattr(sam3d_runner.subprocess, "run", recorder)
        return recorder

    return _patch


def test_builds_basic_command(base_config, mask_path, tmp_path, patch_conda_and_subprocess):
    """Default configuration passes depth + intrinsics and sam3d-input='full'."""
    recorder = patch_conda_and_subprocess()
    out = sam3d_runner.run_sam3d(base_config, mask_path, tmp_path / "sam3d")
    cmd, _ = recorder.calls[0]
    assert cmd[:5] == ["conda", "run", "--no-capture-output", "-n", "sam3d-objects"]
    assert "sam3d_asset_extractor.sam3d.inference" in cmd
    assert "--image" in cmd and str(base_config.image) in cmd
    assert "--mask" in cmd and str(mask_path) in cmd
    assert "--output" in cmd
    idx = cmd.index("--sam3d-input")
    assert cmd[idx + 1] == "full"
    assert "--depth-image" in cmd and str(base_config.depth_image) in cmd
    assert "--cam-k" in cmd and str(base_config.cam_k) in cmd
    # Output path mirrors the mask stem.
    assert out.name == f"{mask_path.stem}.ply"


def test_input_mode_cropped_forwards(base_config, mask_path, tmp_path, patch_conda_and_subprocess):
    base_config.sam3d_input = "cropped"
    recorder = patch_conda_and_subprocess()
    sam3d_runner.run_sam3d(base_config, mask_path, tmp_path / "sam3d")
    cmd, _ = recorder.calls[0]
    idx = cmd.index("--sam3d-input")
    assert cmd[idx + 1] == "cropped"


def test_depth_and_cam_k_always_forwarded(
    base_config, mask_path, tmp_path, patch_conda_and_subprocess,
):
    recorder = patch_conda_and_subprocess()
    sam3d_runner.run_sam3d(base_config, mask_path, tmp_path / "sam3d")
    cmd, _ = recorder.calls[0]
    # depth + cam-k are mandatory — they must appear in every command.
    assert cmd.count("--depth-image") == 1
    assert cmd.count("--cam-k") == 1
    assert cmd.count("--sam3d-input") == 1


def test_compile_flag_added(base_config, mask_path, tmp_path, patch_conda_and_subprocess):
    base_config.sam3d_compile = True
    recorder = patch_conda_and_subprocess()
    sam3d_runner.run_sam3d(base_config, mask_path, tmp_path / "sam3d")
    cmd, _ = recorder.calls[0]
    assert "--compile" in cmd


def test_mesh_format_and_seed_forwarded(
    base_config, mask_path, tmp_path, patch_conda_and_subprocess,
):
    base_config.sam3d_seed = 777
    base_config.mesh_format = "glb"
    recorder = patch_conda_and_subprocess()
    sam3d_runner.run_sam3d(base_config, mask_path, tmp_path / "sam3d")
    cmd, _ = recorder.calls[0]
    idx = cmd.index("--seed")
    assert cmd[idx + 1] == "777"
    idx = cmd.index("--mesh-format")
    assert cmd[idx + 1] == "glb"


def test_missing_conda_raises(base_config, mask_path, tmp_path, monkeypatch):
    monkeypatch.setattr(sam3d_runner.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="conda not found"):
        sam3d_runner.run_sam3d(base_config, mask_path, tmp_path / "sam3d")


def test_subprocess_failure_propagates(
    base_config, mask_path, tmp_path, patch_conda_and_subprocess,
):
    patch_conda_and_subprocess(exit_code=5)
    with pytest.raises(subprocess.CalledProcessError):
        sam3d_runner.run_sam3d(base_config, mask_path, tmp_path / "sam3d")


def test_pythonpath_is_propagated(base_config, mask_path, tmp_path, patch_conda_and_subprocess):
    recorder = patch_conda_and_subprocess()
    sam3d_runner.run_sam3d(base_config, mask_path, tmp_path / "sam3d")
    _, env = recorder.calls[0]
    pypath = env.get("PYTHONPATH", "")
    assert "sam3d_asset_extractor/src" in pypath


def test_output_dir_is_created(base_config, mask_path, tmp_path, patch_conda_and_subprocess):
    patch_conda_and_subprocess()
    out_dir = tmp_path / "new_sam3d"
    assert not out_dir.exists()
    sam3d_runner.run_sam3d(base_config, mask_path, out_dir)
    assert out_dir.exists()
