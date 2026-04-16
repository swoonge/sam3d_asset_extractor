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
#   - Either `hf` already on PATH, or conda is available so the script can
#     install `huggingface-hub[cli]` into the conda base env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "========================================="
echo " sam3d_asset_extractor: setup externals"
echo "========================================="

# ─── Helper: find or install the `hf` CLI ──────────────────────────────────
#
# The HF CLI is only needed for Step 4 (SAM3D checkpoint download). We prefer:
#   1) A pre-existing `hf` on PATH
#   2) Conda base's pip (no sudo, no Python system pollution)
resolve_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    echo "hf"
    return 0
  fi
  local conda_bin
  conda_bin="$(dirname "$(command -v conda 2>/dev/null)" 2>/dev/null || true)"
  if [ -z "$conda_bin" ]; then
    # CONDA_EXE may be set by activated shell
    if [ -n "${CONDA_EXE:-}" ]; then
      conda_bin="$(dirname "$CONDA_EXE")"
    fi
  fi
  if [ -z "$conda_bin" ]; then
    echo "ERROR: neither 'hf' nor 'conda' is on PATH." >&2
    echo "       Install huggingface_hub[cli] manually or add conda to PATH." >&2
    return 1
  fi
  local conda_pip="$conda_bin/pip"
  local conda_hf="$conda_bin/hf"
  if [ ! -x "$conda_hf" ]; then
    echo "[setup] Installing huggingface-hub[cli] into conda base (via $conda_pip)..." >&2
    "$conda_pip" install -q 'huggingface-hub[cli]<1.0' >&2
  fi
  echo "$conda_hf"
}

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
    echo "ERROR: HF_TOKEN not set. Export it before running this script:" >&2
    echo "       export HF_TOKEN=hf_..." >&2
    exit 1
  fi
  HF_BIN="$(resolve_hf_cli)"
  echo "[sam-3d-objects] Using '$HF_BIN' to download from HuggingFace (~12 GB)..."
  TAG=hf
  cd sam-3d-objects
  "$HF_BIN" download \
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
echo "  sam2/checkpoints/            $(du -sh sam2/checkpoints 2>/dev/null | awk '{print $1}')"
echo "  sam-3d-objects/checkpoints/hf/ $(du -sh sam-3d-objects/checkpoints/hf 2>/dev/null | awk '{print $1}')"
