"""Pipeline-wide configuration assembled from CLI flags."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Sam2Mode = Literal["auto", "manual"]
Sam3dInput = Literal["full", "cropped"]
DecimateMethod = Literal["auto", "open3d", "trimesh", "cluster"]


@dataclass
class DecimateOptions:
    """Mesh decimation settings."""

    enabled: bool = True
    method: DecimateMethod = "auto"
    target_faces: int = 20000
    ratio: float = 0.02
    min_faces: int = 200


@dataclass
class PipelineConfig:
    """Full pipeline configuration.

    Populated by ``cli.py`` from argparse; passed down into step runners. Both
    ``depth_image`` and ``cam_k`` are mandatory — this pipeline always feeds a
    metric pointmap into SAM3D.
    """

    image: Path
    output_dir: Path
    depth_image: Path
    cam_k: Path
    depth_scale: str = "auto"

    sam2_mode: Sam2Mode = "auto"
    sam2_env: str = "sam2"
    sam2_checkpoint: Path | None = None
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam3d_input: Sam3dInput = "full"
    sam3d_env: str = "sam3d-objects"
    sam3d_config: Path | None = None
    sam3d_seed: int = 42
    sam3d_compile: bool = False
    mesh_format: Literal["glb", "ply", "obj", "all"] = "all"

    decimate: DecimateOptions = field(default_factory=DecimateOptions)

    overwrite: bool = False
    process_all_masks: bool = True
    dry_run: bool = False

    def validate(self) -> None:
        if self.depth_image is None:
            raise ValueError("--depth-image is required.")
        if self.cam_k is None:
            raise ValueError("--cam-k is required.")
        if not self.image.exists():
            raise FileNotFoundError(f"Missing input image: {self.image}")
        if not self.depth_image.exists():
            raise FileNotFoundError(f"Missing depth image: {self.depth_image}")
        if not self.cam_k.exists():
            raise FileNotFoundError(f"Missing camera intrinsics: {self.cam_k}")
