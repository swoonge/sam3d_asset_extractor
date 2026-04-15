import numpy as np

from sam3d_asset_extractor.common.geometry import (
    backproject_depth,
    build_filter_keep_mask,
    depth_to_pointmap,
    mad_keep_mask,
    sanitize_depth_for_pointmap,
)


def test_backproject_principal_point():
    depth = np.ones((10, 10), dtype=np.float32)
    valid = np.zeros_like(depth, dtype=bool)
    valid[5, 5] = True
    pts = backproject_depth(depth, valid, fx=100, fy=100, cx=5, cy=5)
    assert pts.shape == (1, 3)
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 1.0], atol=1e-6)


def test_pointmap_matches_backproject():
    k = np.array([[100, 0, 5], [0, 100, 5], [0, 0, 1]], dtype=np.float32)
    depth = np.full((10, 10), 2.0, dtype=np.float32)
    pm = depth_to_pointmap(depth, k)
    assert pm.shape == (10, 10, 3)
    np.testing.assert_allclose(pm[5, 5], [0.0, 0.0, 2.0], atol=1e-5)


def test_sanitize_depth_replaces_invalid():
    depth = np.array([[1.0, -1.0], [np.inf, 2.0]], dtype=np.float32)
    out = sanitize_depth_for_pointmap(depth)
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 0])
    assert out[0, 0] == 1.0 and out[1, 1] == 2.0


def test_mad_keep_mask_drops_outliers():
    # Need spread so MAD > 0; the extreme value 100 should be dropped.
    values = np.array([0.9, 1.0, 1.1, 0.95, 1.05, 100.0])
    keep = mad_keep_mask(values, thresh=2.5)
    assert keep[:5].all()
    assert not keep[5]


def test_mad_keep_mask_zero_mad_keeps_all():
    # When all inputs are identical MAD is 0 and the filter degrades to no-op.
    values = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
    keep = mad_keep_mask(values, thresh=2.5)
    assert keep.all()


def test_mad_keep_mask_disabled_by_thresh_zero():
    values = np.array([0.0, 1.0, 100.0])
    keep = mad_keep_mask(values, thresh=0.0)
    assert keep.all()


def test_build_filter_keep_mask_border():
    points = np.random.default_rng(0).normal(size=(5, 3)).astype(np.float32)
    ys = np.array([0, 5, 5, 5, 9])
    xs = np.array([5, 0, 5, 5, 5])
    keep = build_filter_keep_mask(
        points, ys, xs, shape=(10, 10),
        border_margin=1, depth_mad=0.0, radius_mad=0.0,
    )
    assert keep.tolist() == [False, False, True, True, False]
