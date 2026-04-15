"""SAM3D pose parsing and application.

SAM3D inference may emit the object pose as one of several rotation
conventions (quaternion, 6D, or 3x3 matrix). These helpers normalize the
output and apply a similarity transform (scale -> rotate -> translate) to a
point cloud or mesh.
"""

from __future__ import annotations

import numpy as np


def to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion into a 3x3 rotation matrix."""
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def rotation_6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """Convert a 6D rotation (Zhou et al.) to a 3x3 rotation matrix."""
    r6d = r6d.astype(np.float32).reshape(-1)
    a1 = r6d[:3]
    a2 = r6d[3:6]
    x = a1 / max(1e-12, np.linalg.norm(a1))
    y = a2 - np.dot(x, a2) * x
    y = y / max(1e-12, np.linalg.norm(y))
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def apply_similarity(
    points: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply ``scale -> rotate -> translate`` to (N, 3) points."""
    scaled = points * scale.reshape(1, 3)
    rotated = (rotation @ scaled.T).T
    return rotated + translation.reshape(1, 3)


def parse_pose(
    output: dict,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict]:
    """Extract (rotation_matrix, translation, scale, rot_meta) from SAM3D output.

    Returns ``(None, None, None, {})`` if pose fields are missing. ``rot_meta``
    records which rotation representation was detected for downstream JSON.
    """
    pose_quat = to_numpy(output.get("rotation"))
    pose_trans = to_numpy(output.get("translation"))
    pose_scale = to_numpy(output.get("scale"))

    rot_mat: np.ndarray | None = None
    rot_meta: dict = {"type": None, "value": None}
    if pose_quat is not None:
        pose_quat = np.asarray(pose_quat, dtype=np.float32).reshape(-1)
        if 4 <= pose_quat.size < 6:
            rot_mat = quaternion_to_matrix(pose_quat[:4])
            rot_meta = {"type": "quaternion_wxyz", "value": pose_quat[:4].tolist()}
        elif pose_quat.size == 6:
            rot_mat = rotation_6d_to_matrix(pose_quat[:6])
            rot_meta = {"type": "rotation_6d", "value": pose_quat[:6].tolist()}
        elif pose_quat.size == 9:
            rot_mat = pose_quat.reshape(3, 3)
            rot_meta = {"type": "rotation_matrix", "value": rot_mat.tolist()}

    trans_vec: np.ndarray | None = None
    if pose_trans is not None:
        pose_trans = np.asarray(pose_trans, dtype=np.float32).reshape(-1)
        if pose_trans.size >= 3:
            trans_vec = pose_trans[:3]

    scale_vec: np.ndarray | None = None
    if pose_scale is not None:
        pose_scale = np.asarray(pose_scale, dtype=np.float32).reshape(-1)
        if pose_scale.size == 1:
            scale_vec = np.repeat(pose_scale[0], 3)
        elif pose_scale.size >= 3:
            scale_vec = pose_scale[:3]

    return rot_mat, trans_vec, scale_vec, rot_meta
