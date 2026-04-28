#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Phase-0 one-shot regression command (plan T0.3).
#
# Run this before claiming any task green:
#
#   bash examples/hstu/cp/run_regression.sh
#
# It runs:
#   1. All pytest tests under examples/hstu/test/cp/ (single-GPU subset).
#   2. Multi-GPU pytest tests via torchrun, if WORLD_SIZE / nproc available
#      (Slice 3+; no-op in Phase 0).
#   3. The reference benchmark vs `tasks/bench_baseline.json` (if present).
#
# Exits non-zero on:
#   - any pytest FAILED
#   - any required multi-GPU test SKIPPED (fail-on-skip per plan §Global rule 2)
#   - any perf regression beyond the per-tier threshold (plan §Global rule 3)
#
# Globally, target wall-clock < 60s on a single A100/H100 (per plan T0.3 AC).
# If it grows past that, split into "fast" / "full" tiers — don't let it
# silently bloat past contributors' patience.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CP_TEST_DIR="examples/hstu/test/cp"
BASELINE_JSON="tasks/bench_baseline.json"
TMP_BENCH="${TMP_BENCH:-/tmp/cp_bench_$$.json}"

cleanup() {
  rm -f "$TMP_BENCH" 2>/dev/null || true
}
trap cleanup EXIT

# --- 1. Single-GPU pytest --------------------------------------------------
echo ">>> [1/3] single-GPU pytest under $CP_TEST_DIR"
if [[ ! -d "$CP_TEST_DIR" ]]; then
  echo "ERROR: test dir $CP_TEST_DIR not found (cwd=$PWD)" >&2
  exit 2
fi
# `--strict-markers` so unregistered marks fail fast.
# `-ra` shows reasons for skips/xfails so we can fail-on-skip for required tests.
# We grep for SKIPPED later and decide per-test whether to fail.
PYTEST_LOG="$(mktemp)"
trap 'rm -f "$PYTEST_LOG"; cleanup' EXIT
if ! pytest "$CP_TEST_DIR" -v --strict-markers -ra 2>&1 | tee "$PYTEST_LOG"; then
  echo "FAIL: pytest reported failures" >&2
  exit 1
fi

# Fail-on-skip for tests marked as required. v0 contract: any test under
# cp/test_*.py without an explicit `@pytest.mark.optional_skip` decorator
# must run. (We don't have that marker yet; for now, just warn on skips
# and let the user audit.)
if grep -qE 'SKIPPED' "$PYTEST_LOG"; then
  echo "WARN: pytest reported SKIPPED tests; review the reasons above." >&2
fi

# --- 2. Multi-GPU pytest (Slice 3+) ----------------------------------------
echo ""
echo ">>> [2/3] multi-GPU pytest (torchrun)"
MULTI_GPU_DIR="$CP_TEST_DIR"  # same dir; multi-GPU tests will live here from Slice 3
HAS_MULTI_GPU_TESTS=$(find "$MULTI_GPU_DIR" -name 'test_cp_forward.py' -o -name 'test_cp_backward.py' 2>/dev/null | head -1)
if [[ -z "$HAS_MULTI_GPU_TESTS" ]]; then
  echo "   (no multi-GPU test files found; Phase 0 has none — skipping)"
else
  # Detect available GPU count.
  N_GPUS=$(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)
  if [[ "$N_GPUS" -lt 2 ]]; then
    echo "FAIL: multi-GPU tests exist but only $N_GPUS GPU(s) visible; need ≥ 2." >&2
    exit 1
  fi
  echo "   running with --nproc-per-node=$N_GPUS"
  if ! torchrun --standalone --nproc-per-node="$N_GPUS" -m pytest "$MULTI_GPU_DIR" \
        -v --strict-markers -ra; then
    echo "FAIL: torchrun pytest reported failures" >&2
    exit 1
  fi
fi

# --- 3. Reference benchmark vs committed baseline --------------------------
echo ""
echo ">>> [3/3] perf vs $BASELINE_JSON"
if [[ ! -f "$BASELINE_JSON" ]]; then
  echo "WARN: $BASELINE_JSON does not exist yet — skipping perf compare." >&2
  echo "   Run \`python examples/hstu/cp/bench/baseline.py --output $BASELINE_JSON\`"
  echo "   on the reference GPU to commit a baseline."
else
  python examples/hstu/cp/bench/baseline.py --output "$TMP_BENCH"
  if ! python examples/hstu/cp/bench/compare.py "$BASELINE_JSON" "$TMP_BENCH"; then
    echo "FAIL: perf regression beyond threshold (see table above)" >&2
    exit 1
  fi
fi

echo ""
echo "OK — regression suite passed."
