import numpy as np

from sam3d_asset_extractor.common.ply_io import (
    flatten_pointmap,
    read_points_ply,
    write_points_ply,
)


def test_write_and_read_roundtrip(tmp_path):
    pts = np.array([[0, 0, 0], [1, 2, 3], [-1, 0.5, 4]], dtype=np.float32)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    path = tmp_path / "out.ply"
    write_points_ply(path, pts, colors)

    pts_r, colors_r = read_points_ply(path)
    np.testing.assert_allclose(pts, pts_r, atol=1e-5)
    assert colors_r is not None
    np.testing.assert_allclose(colors, colors_r, atol=2.0 / 255.0)


def test_write_without_colors(tmp_path):
    pts = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
    path = tmp_path / "nocolor.ply"
    write_points_ply(path, pts)
    pts_r, colors_r = read_points_ply(path)
    np.testing.assert_allclose(pts, pts_r, atol=1e-5)
    assert colors_r is None


def test_flatten_pointmap_drops_nan():
    pm = np.zeros((2, 2, 3), dtype=np.float32)
    pm[0, 0] = [1, 2, 3]
    pm[0, 1] = [np.nan, 0, 0]
    pm[1, 0] = [4, 5, 6]
    pm[1, 1] = [0, np.inf, 0]

    colors = np.zeros_like(pm)
    colors[0, 0] = [1, 0, 0]
    colors[1, 0] = [0, 1, 0]

    flat_pts, flat_colors = flatten_pointmap(pm, colors)
    assert flat_pts.shape == (2, 3)
    np.testing.assert_allclose(flat_pts, [[1, 2, 3], [4, 5, 6]])
    assert flat_colors is not None
    assert flat_colors.shape == (2, 3)
