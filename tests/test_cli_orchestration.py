"""End-to-end-ish CLI tests with SAM2/SAM3D/decimation mocked.

We patch out the subprocess boundaries and the decimation call so ``cli.main``
executes the full orchestration path (preflight skipped, masks picked,
artifacts collected, manifest written) without requiring GPUs or conda envs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sam3d_asset_extractor import cli


@pytest.fixture
def sample_inputs(tmp_path):
    """Return (image, depth_image, cam_k) — all three required by the CLI."""
    img = tmp_path / "scene.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    depth = tmp_path / "depth.png"
    depth.write_bytes(b"\x00")
    cam_k = tmp_path / "cam_k.txt"
    cam_k.write_text("500 0 320 0 500 240 0 0 1\n")
    return img, depth, cam_k


def _base_argv(image: Path, depth: Path, cam_k: Path, output_dir: Path) -> list[str]:
    return [
        "--image", str(image),
        "--depth-image", str(depth),
        "--cam-k", str(cam_k),
        "--output-dir", str(output_dir),
    ]


def _fake_run_sam2(mask_count: int):
    def _inner(config, mask_dir):
        mask_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(mask_count):
            p = mask_dir / f"{config.image.stem}_{i:03d}.png"
            p.write_bytes(b"PNG")
            paths.append(p)
        return paths

    return _inner


def _fake_run_sam3d(config, mask_path, output_dir):
    """Pretend SAM3D ran: create PLY + pose_mesh.{glb,ply,obj} placeholders."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = mask_path.stem
    ply = output_dir / f"{stem}.ply"
    ply.write_bytes(b"PLY")
    for ext in ("glb", "ply", "obj"):
        (output_dir / f"{stem}_pose_mesh.{ext}").write_bytes(b"MESH")
    return ply


def _fake_decimate_file(src, dst, options):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"DECIMATED")

    class _Result:
        output_path = dst
        method = "fake"
        face_count_before = 1000
        face_count_after = 200

    return _Result()


@pytest.fixture
def patch_all(monkeypatch):
    """Replace preflight + runners + decimation with in-memory fakes."""
    monkeypatch.setattr(cli, "run_preflight", lambda *a, **kw: None)

    def _runner_factory(mask_count=2):
        monkeypatch.setattr(cli, "run_sam2", _fake_run_sam2(mask_count))
        monkeypatch.setattr(cli, "run_sam3d", _fake_run_sam3d)
        monkeypatch.setattr(cli, "decimate_file", _fake_decimate_file)

    return _runner_factory


def _run_cli(argv: list[str]) -> int:
    """Wrap cli.main with a flag that always points at the real argv parser."""
    return cli.main(argv)


def test_happy_path_writes_manifest(tmp_path, sample_inputs, patch_all):
    image, depth, cam_k = sample_inputs
    patch_all(mask_count=2)
    out = tmp_path / "run"
    code = _run_cli(_base_argv(image, depth, cam_k, out) + [
        "--overwrite", "--log-level", "ERROR",
    ])
    assert code == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert manifest["inputs"]["depth_image"] == str(depth)
    assert manifest["inputs"]["cam_k"] == str(cam_k)
    assert len(manifest["outputs"]["masks"]) == 2
    assert len(manifest["outputs"]["sam3d_plys"]) == 2
    # 2 masks × 3 ext (all) = 6 decimated files
    assert len(manifest["outputs"]["decimated_meshes"]) == 6


def test_latest_only_picks_single_mask(tmp_path, sample_inputs, patch_all):
    image, depth, cam_k = sample_inputs
    patch_all(mask_count=4)
    out = tmp_path / "run"
    code = _run_cli(_base_argv(image, depth, cam_k, out) + [
        "--overwrite", "--latest-only", "--log-level", "ERROR",
    ])
    assert code == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["outputs"]["masks"]) == 1
    assert len(manifest["outputs"]["sam3d_plys"]) == 1


def test_no_decimate_skips_decimation(tmp_path, sample_inputs, patch_all):
    image, depth, cam_k = sample_inputs
    patch_all(mask_count=2)
    out = tmp_path / "run"
    code = _run_cli(_base_argv(image, depth, cam_k, out) + [
        "--overwrite", "--no-decimate", "--log-level", "ERROR",
    ])
    assert code == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["options"]["decimate"]["enabled"] is False
    assert manifest["outputs"]["decimated_meshes"] == []
    assert not (out / "decimated").exists()


def test_mesh_format_glb_only_limits_decimated_count(tmp_path, sample_inputs, patch_all):
    image, depth, cam_k = sample_inputs
    patch_all(mask_count=1)
    out = tmp_path / "run"
    code = _run_cli(_base_argv(image, depth, cam_k, out) + [
        "--overwrite", "--mesh-format", "glb", "--log-level", "ERROR",
    ])
    assert code == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["outputs"]["decimated_meshes"]) == 1
    assert manifest["outputs"]["decimated_meshes"][0].endswith(".glb")


def test_zero_masks_fails_fast(tmp_path, sample_inputs, monkeypatch):
    image, depth, cam_k = sample_inputs
    monkeypatch.setattr(cli, "run_preflight", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "run_sam2", _fake_run_sam2(0))
    monkeypatch.setattr(cli, "run_sam3d", _fake_run_sam3d)
    monkeypatch.setattr(cli, "decimate_file", _fake_decimate_file)
    out = tmp_path / "run"
    code = _run_cli(_base_argv(image, depth, cam_k, out) + [
        "--overwrite", "--log-level", "ERROR",
    ])
    assert code == 4
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "failed_no_masks"


def test_missing_depth_argument_is_rejected_by_argparse(tmp_path, sample_inputs):
    """argparse enforces --depth-image as a required flag."""
    image, _, cam_k = sample_inputs
    with pytest.raises(SystemExit):
        _run_cli([
            "--image", str(image),
            "--cam-k", str(cam_k),
            "--output-dir", str(tmp_path / "run"),
        ])


def test_missing_cam_k_argument_is_rejected_by_argparse(tmp_path, sample_inputs):
    image, depth, _ = sample_inputs
    with pytest.raises(SystemExit):
        _run_cli([
            "--image", str(image),
            "--depth-image", str(depth),
            "--output-dir", str(tmp_path / "run"),
        ])


def test_missing_depth_file_on_disk_fails_validation(tmp_path, sample_inputs, monkeypatch):
    """Depth file that doesn't exist should be caught by PipelineConfig.validate()."""
    image, _, cam_k = sample_inputs
    monkeypatch.setattr(cli, "run_sam2",
                        lambda *a, **kw: pytest.fail("should not run SAM2"))
    monkeypatch.setattr(cli, "run_sam3d",
                        lambda *a, **kw: pytest.fail("should not run SAM3D"))
    monkeypatch.setattr(cli, "run_preflight", lambda *a, **kw: None)
    code = _run_cli([
        "--image", str(image),
        "--depth-image", str(tmp_path / "missing_depth.png"),
        "--cam-k", str(cam_k),
        "--output-dir", str(tmp_path / "run"),
        "--log-level", "ERROR",
    ])
    assert code == 2


def test_partial_failure_status(tmp_path, sample_inputs, monkeypatch):
    """When at least one SAM3D inference fails, exit != 0 and status != 'success'."""
    image, depth, cam_k = sample_inputs
    monkeypatch.setattr(cli, "run_preflight", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "run_sam2", _fake_run_sam2(2))
    monkeypatch.setattr(cli, "decimate_file", _fake_decimate_file)
    calls = {"n": 0}

    def _flaky_sam3d(config, mask_path, output_dir):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated SAM3D failure")
        return _fake_run_sam3d(config, mask_path, output_dir)

    monkeypatch.setattr(cli, "run_sam3d", _flaky_sam3d)

    out = tmp_path / "run"
    code = _run_cli(_base_argv(image, depth, cam_k, out) + [
        "--overwrite", "--log-level", "ERROR",
    ])
    assert code == 6
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "partial"
    assert len(manifest["outputs"]["sam3d_plys"]) == 1
