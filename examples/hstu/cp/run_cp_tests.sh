#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# T3.4 helper: run multi-GPU CP pytests under torchrun for N ∈ {2, 4, 8}.
#
# Usage:
#   bash examples/hstu/cp/run_cp_tests.sh        # forward only
#   bash examples/hstu/cp/run_cp_tests.sh --bwd  # forward + backward (Slice 4)
#   bash examples/hstu/cp/run_cp_tests.sh --sizes 2,4   # custom cp_size set
#
# Required: at least N GPUs visible for cp_size=N to run; otherwise that
# size is reported as SKIPPED.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

INCLUDE_BWD=0
SIZES_CSV="2,4,8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bwd)   INCLUDE_BWD=1; shift ;;
    --sizes) SIZES_CSV="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a SIZES <<< "$SIZES_CSV"

N_GPUS=$(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)
echo "==> visible GPUs: $N_GPUS"

PYTEST_FILES="examples/hstu/test/cp/test_cp_forward.py"
if [[ "$INCLUDE_BWD" == "1" ]]; then
  PYTEST_FILES="$PYTEST_FILES examples/hstu/test/cp/test_cp_backward.py"
fi

OVERALL_RC=0
for SIZE in "${SIZES[@]}"; do
  echo ""
  echo "==> cp_size=$SIZE"
  if [[ "$N_GPUS" -lt "$SIZE" ]]; then
    echo "    SKIPPED: only $N_GPUS GPU(s) visible (need $SIZE)"
    continue
  fi
  if torchrun --standalone --nproc-per-node="$SIZE" -m pytest \
      $PYTEST_FILES -v -k "cp${SIZE}" --strict-markers -ra; then
    echo "    OK cp_size=$SIZE"
  else
    echo "    FAILED cp_size=$SIZE" >&2
    OVERALL_RC=1
  fi
done

if [[ "$OVERALL_RC" -ne 0 ]]; then
  echo ""
  echo "==> overall: FAIL (some cp_size groups failed)" >&2
  exit 1
fi
echo ""
echo "==> overall: OK"
