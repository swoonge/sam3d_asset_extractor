import numpy as np
import pytest

from sam3d_asset_extractor.config import DecimateOptions
from sam3d_asset_extractor.mesh.decimate import (
    decimate_file,
    pick_target_faces,
    simplify_cluster,
)


@pytest.fixture
def sphere_mesh():
    pytest.importorskip("trimesh")
    import trimesh

    return trimesh.creation.icosphere(subdivisions=4)


def test_pick_target_faces_uses_count_when_positive():
    assert pick_target_faces(1000, ratio=0.1, target_faces=150, min_faces=10) == 150


def test_pick_target_faces_falls_back_to_ratio():
    # target <= 0 uses the ratio
    assert pick_target_faces(1000, ratio=0.1, target_faces=0, min_faces=10) == 100


def test_pick_target_faces_respects_min_and_total():
    # target is clamped to [min_faces, face_count]
    assert pick_target_faces(100, ratio=0.5, target_faces=5, min_faces=20) == 20
    assert pick_target_faces(100, ratio=0.5, target_faces=500, min_faces=20) == 100


def test_simplify_cluster_reduces_face_count(sphere_mesh):
    before = sphere_mesh.faces.shape[0]
    out, name = simplify_cluster(sphere_mesh, target_faces=before // 4)
    assert name == "cluster"
    assert out.faces.shape[0] <= before
    assert out.vertices.shape[0] > 0


def test_decimate_file_roundtrip(tmp_path, sphere_mesh):
    src = tmp_path / "sphere.ply"
    sphere_mesh.export(src)
    dst = tmp_path / "sphere_dec.ply"
    opts = DecimateOptions(
        enabled=True, method="cluster", target_faces=200, ratio=0.02, min_faces=50,
    )
    result = decimate_file(src, dst, opts)
    assert result.output_path == dst
    assert dst.exists() and dst.stat().st_size > 0
    assert result.face_count_after <= result.face_count_before
