#!/usr/bin/env bash
# Convenience wrapper around ``python -m sam3d_asset_extractor``.
# All arguments are forwarded as-is.
#
# Example:
#   ./run_pipeline.sh \
#     --image  /path/to/rgb.png \
#     --depth-image /path/to/depth.png \
#     --cam-k  /path/to/cam_K.txt \
#     --output-dir outputs/demo \
#     --overwrite
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${script_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m sam3d_asset_extractor "$@"
