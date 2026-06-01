#!/bin/bash
# ============================================================================
# Quick launcher for the HSTU attention kernel MFU heatmap sweep.
# Thin wrapper around the unified run_all_experiments_local.sh flow.
#
# Usage (from examples/hstu/):
#   bash training/benchmark/scripts/run_hstu_attn_kernel_benchmark.sh [options]
#
# Options: forwarded verbatim to run_all_experiments_local.sh
#   --exp-file=FILE   Override config list (default: training/benchmark/kernel_experiments.txt)
#   --hstu-root=PATH  Specify examples/hstu directory path
#   --dry-run         Print commands only, do not execute
#   --help,-h         Show this help
#
# To customize the sweep, edit training/benchmark/kernel_experiments.txt or
# pass --exp-file=<your_list>.
# ============================================================================
set -e

case " $* " in
    *" --help "*|*" -h "*)
        sed -n '2,20p' "$0" | sed 's/^# \?//'
        exit 0
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

exec bash "${SCRIPT_DIR}/run_all_experiments_local.sh" \
    --benchmark-type=hstu-attn-kernel \
    "$@"
