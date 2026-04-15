import numpy as np
import pytest

from sam3d_asset_extractor.common.camera import (
    _parse_intrinsics_values,
    load_intrinsics_matrix,
    load_intrinsics_tuple,
)


def test_parse_3x3():
    arr = np.array([500, 0, 320, 0, 500, 240, 0, 0, 1], dtype=np.float32)
    k = _parse_intrinsics_values(arr)
    assert k.shape == (3, 3)
    assert k[0, 0] == 500 and k[1, 1] == 500
    assert k[0, 2] == 320 and k[1, 2] == 240


def test_parse_flat_fxfycxcy():
    arr = np.array([600, 601, 330, 250], dtype=np.float32)
    k = _parse_intrinsics_values(arr)
    assert k.shape == (3, 3)
    assert k[0, 0] == 600 and k[1, 1] == 601
    assert k[0, 2] == 330 and k[1, 2] == 250
    assert k[2, 2] == 1.0


def test_parse_invalid_count():
    with pytest.raises(ValueError, match="Invalid camera intrinsics"):
        _parse_intrinsics_values(np.array([1, 2, 3, 4, 5]))


def test_parse_invalid_focal():
    with pytest.raises(ValueError, match="focal lengths"):
        _parse_intrinsics_values(np.array([-10, 0, 100, 0, 500, 100, 0, 0, 1]))


def test_load_from_file(tmp_path):
    path = tmp_path / "cam_k.txt"
    path.write_text("500 0 320\n0 500 240\n0 0 1\n")
    k = load_intrinsics_matrix(path)
    fx, fy, cx, cy = load_intrinsics_tuple(path)
    assert fx == 500 and fy == 500 and cx == 320 and cy == 240
    assert k[0, 0] == 500
