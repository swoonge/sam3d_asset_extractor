#!/usr/bin/env bash
# Create and configure the two conda envs (sam2, sam3d-objects) used by the
# pipeline. Run once after setup_externals.sh.
#
# Usage:
#   cd sam3d_asset_extractor
#   bash scripts/setup_envs.sh
#
# Prerequisites:
#   - conda (anaconda3 / miniconda / mamba)
#   - sam2/ and sam-3d-objects/ already cloned (run setup_externals.sh first)
#   - NVIDIA GPU with CUDA driver ≥12.1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA="${CONDA_EXE:-conda}"
PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

echo "========================================="
echo " sam3d_asset_extractor: setup conda envs"
echo "========================================="

# ─── 1. sam2 env ──────────────────────────────────────────────────────────────
echo ""
echo ">>> [1/2] Creating sam2 env..."
$CONDA create -y -n sam2 python=3.10
$CONDA install -y -n sam2 \
  -c pytorch -c nvidia -c conda-forge \
  pytorch torchvision pytorch-cuda=12.4 "mkl<2025"

echo ">>> Installing SAM2 package..."
$CONDA run --no-capture-output -n sam2 \
  pip install -e "$REPO_ROOT/sam2"

echo ">>> Installing sam2 env extra deps..."
$CONDA run --no-capture-output -n sam2 \
  pip install numpy opencv-python Pillow

echo ">>> Verifying sam2 env..."
$CONDA run --no-capture-output -n sam2 python -c "
import torch
from sam2.build_sam import build_sam2
print('sam2 env OK | torch', torch.__version__, '| cuda', torch.cuda.is_available())
"

# ─── 2. sam3d-objects env ─────────────────────────────────────────────────────
echo ""
echo ">>> [2/2] Creating sam3d-objects env..."

# Option A: use sam-3d-objects' official conda env yml (recommended if available)
if [ -f "$REPO_ROOT/sam-3d-objects/environments/default.yml" ]; then
  echo "    Using sam-3d-objects/environments/default.yml..."
  $CONDA env create -y -f "$REPO_ROOT/sam-3d-objects/environments/default.yml"
else
  echo "    Official env yml not found, creating minimal env..."
  $CONDA create -y -n sam3d-objects python=3.10
fi

# Ensure MKL < 2025 (PyTorch 2.5.1 compatibility)
$CONDA install -y -n sam3d-objects -c conda-forge "mkl<2025" || true

echo ">>> Installing PyTorch (cu121, matching sam-3d-objects' tested config)..."
$CONDA install -y -n sam3d-objects \
  -c pytorch -c nvidia -c conda-forge \
  pytorch torchvision pytorch-cuda=12.1 "mkl<2025" || true

echo ">>> Installing sam-3d-objects base dependencies (bpy excluded)..."
$CONDA run --no-capture-output -n sam3d-objects \
  pip install \
    --extra-index-url "$PIP_EXTRA_INDEX_URL" \
    -r "$REPO_ROOT/requirements-sam3d-runtime.txt"

echo ">>> Installing pytorch3d + flash_attn..."
$CONDA run --no-capture-output -n sam3d-objects \
  pip install \
    --extra-index-url "$PIP_EXTRA_INDEX_URL" \
    -r "$REPO_ROOT/sam-3d-objects/requirements.p3d.txt" || {
  echo "WARNING: pytorch3d install failed. May need manual build."
}

echo ">>> Installing kaolin + gsplat + seaborn + gradio..."
$CONDA run --no-capture-output -n sam3d-objects \
  pip install \
    --extra-index-url "$PIP_EXTRA_INDEX_URL" \
    --find-links "$PIP_FIND_LINKS" \
    -r "$REPO_ROOT/sam-3d-objects/requirements.inference.txt" || {
  echo "WARNING: kaolin/gsplat install failed. May need manual build."
}

echo ">>> Installing sam3d_asset_extractor + open3d + mesh tools..."
$CONDA run --no-capture-output -n sam3d-objects \
  pip install -e "$REPO_ROOT[dev]" open3d

echo ">>> Verifying sam3d-objects env..."
$CONDA run --no-capture-output -n sam3d-objects python -c "
import os; os.environ['LIDRA_SKIP_INIT'] = 'true'
import torch, trimesh, plyfile, open3d
print('sam3d-objects env OK | torch', torch.__version__, '| cuda', torch.cuda.is_available())
"

echo ""
echo "========================================="
echo " Setup complete."
echo " Envs: sam2, sam3d-objects"
echo "========================================="
