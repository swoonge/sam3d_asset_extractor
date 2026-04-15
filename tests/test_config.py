from pathlib import Path

import pytest

from sam3d_asset_extractor.config import DecimateOptions, PipelineConfig


def _complete_config(tmp_path: Path) -> PipelineConfig:
    """Return a config with all required paths present on disk."""
    image = tmp_path / "img.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9")
    depth = tmp_path / "depth.png"
    depth.write_bytes(b"\x00\x01\x02")
    cam_k = tmp_path / "cam_k.txt"
    cam_k.write_text("500 0 320 0 500 240 0 0 1\n")
    return PipelineConfig(
        image=image, output_dir=tmp_path / "out",
        depth_image=depth, cam_k=cam_k,
    )


def test_defaults(tmp_path):
    cfg = _complete_config(tmp_path)
    assert cfg.sam2_mode == "auto"
    assert cfg.sam3d_input == "full"
    assert cfg.decimate.enabled is True
    assert cfg.decimate.method == "auto"
    assert cfg.decimate.target_faces == 20000


def test_validate_passes_with_full_inputs(tmp_path):
    _complete_config(tmp_path).validate()  # no error


def test_validate_missing_depth(tmp_path):
    cfg = _complete_config(tmp_path)
    cfg.depth_image = tmp_path / "nope.png"
    with pytest.raises(FileNotFoundError, match="depth image"):
        cfg.validate()


def test_validate_missing_cam_k(tmp_path):
    cfg = _complete_config(tmp_path)
    cfg.cam_k = tmp_path / "nope.txt"
    with pytest.raises(FileNotFoundError, match="camera intrinsics"):
        cfg.validate()


def test_validate_missing_image(tmp_path):
    cfg = _complete_config(tmp_path)
    cfg.image = tmp_path / "missing.jpg"
    with pytest.raises(FileNotFoundError, match="input image"):
        cfg.validate()


def test_decimate_options_defaults():
    d = DecimateOptions()
    assert d.enabled is True and d.method == "auto"
    assert d.target_faces == 20000 and d.ratio == 0.02 and d.min_faces == 200
