import numpy as np
import pytest

from sam3d_asset_extractor.common.depth import resolve_depth_scale


@pytest.mark.parametrize(
    "raw, user, expected",
    [
        (np.array([[0, 100, 200]], dtype=np.uint16), "auto", 0.001),  # mm hint
        (np.array([[0.5, 1.2, 2.0]], dtype=np.float32), "auto", 1.0),  # meter hint
        (np.zeros((1, 1), dtype=np.float32), "0.001", 0.001),
        (np.zeros((1, 1), dtype=np.float32), 0.002, 0.002),
    ],
)
def test_resolve_depth_scale(raw, user, expected):
    assert resolve_depth_scale(raw, user) == pytest.approx(expected)


def test_resolve_depth_scale_invalid_numeric():
    with pytest.raises(ValueError, match="Invalid depth scale"):
        resolve_depth_scale(np.zeros((1, 1)), "not_a_number")


def test_resolve_depth_scale_rejects_non_positive():
    with pytest.raises(ValueError, match="Invalid depth scale"):
        resolve_depth_scale(np.zeros((1, 1)), 0)
