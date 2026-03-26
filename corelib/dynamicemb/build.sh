#!/bin/bash
# Build dynamicemb wheel
# Usage: bash build.sh [conda_env]
#   conda_env: conda environment to use (default: dynamicemb)

set -e

CONDA_ENV=${1:-hkv}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building dynamicemb wheel using conda env: $CONDA_ENV"
echo "Source dir: $SCRIPT_DIR"

cd "$SCRIPT_DIR"

# Ensure ninja is installed for parallel compilation
conda run -n "$CONDA_ENV" pip install ninja -q

# Build wheel with parallel jobs (uses all available CPU cores)
MAX_JOBS=$(nproc) conda run --no-capture-output -n "$CONDA_ENV" bash -c \
    "cd $SCRIPT_DIR && MAX_JOBS=$(nproc) python setup.py bdist_wheel"

echo ""
echo "Build complete. Wheel located at:"
ls -lh "$SCRIPT_DIR/dist/"*.whl
