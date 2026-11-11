#!/bin/bash
# Run the many-collections benchmark: the lookup (forward and backward, nothing else) of
# K separate DynamicEmb collections on a few inputs in turn, to measure what blocking
# host syncs inside DynamicEmb cost.
# Single GPU; no torchrun or process group needed.  Every flag is forwarded to
# benchmark_many_collections.py (see benchmark/README.md, section 3).
#
# Usage:
#   bash ./benchmark/benchmark_many_collections.sh                           # 32 collections, HBM
#   bash ./benchmark/benchmark_many_collections.sh --storage caching --sync-count
#   bash ./benchmark/benchmark_many_collections.sh --collections 64 --batch-size 256
#   bash ./benchmark/benchmark_many_collections.sh --sweep --storage hbm --json hbm.json
#   bash ./benchmark/benchmark_many_collections.sh --compare main.json nosync.json  # no GPU needed
#   bash ./benchmark/benchmark_many_collections.sh --help
#
# Compare two trees: run the same command on each (with --json), then --compare.
# PYTHON=/path/to/python overrides the interpreter; CUDA_VISIBLE_DEVICES picks the GPU.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

exec "${PYTHON:-python}" "$(dirname "${BASH_SOURCE[0]}")/benchmark_many_collections.py" "$@"
