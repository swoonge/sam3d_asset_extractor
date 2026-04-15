import numpy as np

from sam3d_asset_extractor.sam3d.pose import (
    apply_similarity,
    parse_pose,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)


def test_quaternion_identity():
    r = quaternion_to_matrix(np.array([1, 0, 0, 0], dtype=np.float32))
    np.testing.assert_allclose(r, np.eye(3), atol=1e-6)


def test_rotation_6d_orthogonal():
    r = rotation_6d_to_matrix(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32))
    np.testing.assert_allclose(r, np.eye(3), atol=1e-6)


def test_apply_similarity_scales_rotates_translates():
    pts = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    r = np.eye(3, dtype=np.float32)
    t = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    out = apply_similarity(pts, scale, r, t)
    np.testing.assert_allclose(out, [[12, 0, 0], [10, 2, 0]])


def test_parse_pose_quaternion():
    out = {"rotation": [1, 0, 0, 0], "translation": [1, 2, 3], "scale": [1.0]}
    rot, trans, scale, meta = parse_pose(out)
    np.testing.assert_allclose(rot, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(trans, [1, 2, 3])
    np.testing.assert_allclose(scale, [1.0, 1.0, 1.0])
    assert meta["type"] == "quaternion_wxyz"


def test_parse_pose_missing_returns_none():
    rot, trans, scale, meta = parse_pose({})
    assert rot is None and trans is None and scale is None
    assert meta == {"type": None, "value": None}


def test_parse_pose_rotation_matrix():
    mat = np.eye(3, dtype=np.float32).flatten().tolist()
    out = {"rotation": mat, "translation": [0, 0, 0], "scale": [1, 1, 1]}
    rot, _, _, meta = parse_pose(out)
    np.testing.assert_allclose(rot, np.eye(3))
    assert meta["type"] == "rotation_matrix"
