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

# ─── 1. sam2 env (Python 3.10, PyTorch cu124, C++ extension build) ────────────
echo ""
echo ">>> [1/2] Creating sam2 env..."
$CONDA create -y -n sam2 python=3.10
$CONDA install -y -n sam2 \
  -c pytorch -c nvidia -c conda-forge \
  pytorch torchvision pytorch-cuda=12.4 "mkl<2025"

echo ">>> Installing SAM2 package (builds sam2/_C.so)..."
$CONDA run --no-capture-output -n sam2 \
  pip install -e "$REPO_ROOT/sam2"

echo ">>> Installing sam2 env extra deps..."
$CONDA run --no-capture-output -n sam2 \
  pip install numpy opencv-python Pillow

echo ">>> Verifying sam2 env (CWD=/tmp to dodge sam2/ shadowing)..."
(cd /tmp && $CONDA run --no-capture-output -n sam2 python -c "
import torch
from sam2.build_sam import build_sam2
print('sam2 env OK | torch', torch.__version__, '| cuda:', torch.cuda.is_available())
")

# ─── 2. sam3d-objects env ─────────────────────────────────────────────────────
# Official env YAML handles the system-level conda packages (CUDA 12.1 toolkit,
# Qt, X11, compilers). PyTorch itself is installed via pip from the cu121 index
# to avoid the `pytorch-cuda=12.1` ↔ `libcublas=12.1.3.1` version conflict that
# arises when trying to mix conda-forge pytorch with the strict libcublas pin
# shipped in the env YAML.
echo ""
echo ">>> [2/2] Creating sam3d-objects env from official YAML..."

if [ ! -f "$REPO_ROOT/sam-3d-objects/environments/default.yml" ]; then
  echo "ERROR: $REPO_ROOT/sam-3d-objects/environments/default.yml not found." >&2
  echo "       Did setup_externals.sh complete?" >&2
  exit 1
fi

$CONDA env create -y -f "$REPO_ROOT/sam-3d-objects/environments/default.yml"

echo ">>> Installing sam-3d-objects Python deps (PyTorch cu121 + curated deps)..."
# requirements-sam3d-runtime.txt contains the upstream base requirements minus
# `bpy==4.3.0` (Python 3.10/3.11 wheel unavailable). It transitively pulls
# torch==2.5.1+cu121 + torchvision + CUDA wheels via the pytorch.org index.
export PIP_EXTRA_INDEX_URL
$CONDA run --no-capture-output -n sam3d-objects \
  pip install -r "$REPO_ROOT/requirements-sam3d-runtime.txt"

echo ">>> Installing pytorch3d + flash_attn..."
$CONDA run --no-capture-output -n sam3d-objects \
  pip install -r "$REPO_ROOT/sam-3d-objects/requirements.p3d.txt" \
  || echo "WARNING: p3d install had errors — may need manual fix (nvcc/arch)"

echo ">>> Installing kaolin + gsplat + seaborn + gradio..."
export PIP_FIND_LINKS
$CONDA run --no-capture-output -n sam3d-objects \
  pip install -r "$REPO_ROOT/sam-3d-objects/requirements.inference.txt" \
  || echo "WARNING: kaolin/gsplat install had errors — may need manual fix"

echo ">>> Installing sam3d_asset_extractor (editable, dev extras)..."
$CONDA run --no-capture-output -n sam3d-objects \
  pip install -e "$REPO_ROOT[dev]"

echo ">>> Verifying sam3d-objects env..."
(cd /tmp && $CONDA run --no-capture-output -n sam3d-objects python -c "
import os; os.environ['LIDRA_SKIP_INIT'] = 'true'
import torch, trimesh, plyfile, open3d, omegaconf
print('sam3d-objects env OK | torch', torch.__version__,
      '| cuda:', torch.cuda.is_available(), '| cuda_ver:', torch.version.cuda)
")

echo ""
echo "========================================="
echo " Setup complete."
echo " Envs: sam2, sam3d-objects"
echo "========================================="
echo "Next:"
echo "  export HF_TOKEN=hf_..."
echo "  conda run --no-capture-output -n sam3d-objects sam3d-asset-extractor \\"
echo "    --image datas/move_ham_onto_box/rgb/000000.png \\"
echo "    --depth-image datas/move_ham_onto_box/depth/000000.png \\"
echo "    --cam-k datas/move_ham_onto_box/cam_K.txt \\"
echo "    --output-dir outputs/demo --latest-only --overwrite"
