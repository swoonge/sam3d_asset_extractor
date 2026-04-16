#!/usr/bin/env bash
# Clone external repos (sam2 + sam-3d-objects) INSIDE this repo and download
# their model checkpoints. Run once after git clone.
#
# Usage:
#   cd sam3d_asset_extractor
#   bash scripts/setup_externals.sh
#
# Prerequisites:
#   - git, wget (for SAM2 checkpoints)
#   - HF_TOKEN env var set (for SAM3D checkpoints: `export HF_TOKEN=hf_...`)
#   - `huggingface-hub[cli]<1.0` installed in the active Python env
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "========================================="
echo " sam3d_asset_extractor: setup externals"
echo "========================================="

# ─── 1. Clone SAM2 ───────────────────────────────────────────────────────────
if [ -d sam2/.git ]; then
  echo "[sam2] Already cloned."
else
  echo "[sam2] Cloning from GitHub..."
  git clone https://github.com/facebookresearch/sam2.git
fi

# ─── 2. SAM2 checkpoints ─────────────────────────────────────────────────────
if [ -f sam2/checkpoints/sam2.1_hiera_large.pt ]; then
  echo "[sam2] Checkpoints already present."
else
  echo "[sam2] Downloading checkpoints (~1.5 GB)..."
  (cd sam2/checkpoints && bash download_ckpts.sh)
fi

# ─── 3. Clone sam-3d-objects ──────────────────────────────────────────────────
if [ -d sam-3d-objects/.git ]; then
  echo "[sam-3d-objects] Already cloned."
else
  echo "[sam-3d-objects] Cloning from GitHub..."
  git clone https://github.com/facebookresearch/sam-3d-objects.git
fi

# ─── 4. SAM3D checkpoints (HuggingFace, ~12 GB) ─────────────────────────────
if [ -f sam-3d-objects/checkpoints/hf/pipeline.yaml ]; then
  echo "[sam-3d-objects] Checkpoints already present."
else
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Export it before running this script."
    echo "       export HF_TOKEN=hf_..."
    exit 1
  fi
  echo "[sam-3d-objects] Downloading checkpoints from HuggingFace (~12 GB)..."
  TAG=hf
  cd sam-3d-objects
  hf download \
    --repo-type model \
    --local-dir "checkpoints/${TAG}-download" \
    --max-workers 1 \
    facebook/sam-3d-objects
  mv "checkpoints/${TAG}-download/checkpoints" "checkpoints/${TAG}"
  rm -rf "checkpoints/${TAG}-download"
  cd "$REPO_ROOT"
fi

echo ""
echo "Done. Layout:"
echo "  sam2/checkpoints/sam2.1_hiera_large.pt  $(du -sh sam2/checkpoints 2>/dev/null | awk '{print $1}')"
echo "  sam-3d-objects/checkpoints/hf/           $(du -sh sam-3d-objects/checkpoints/hf 2>/dev/null | awk '{print $1}')"
