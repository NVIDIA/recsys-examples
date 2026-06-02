#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common_paths.sh
source "${SCRIPT_DIR}/common_paths.sh"
cd "${GR_INFERENCE_REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage: scripts/quickstart_offline.sh

Bootstrap the current container and run a small GR-vs-SGLang offline benchmark.

Common overrides:
  MODEL=Qwen/Qwen3-1.7B       HuggingFace model id
  MODEL_DIR=/workspace/...    Existing model directory in the container
  MODEL_REVISION=main         HuggingFace branch, tag, or commit
  RUN_ACCURACY=1              Also run the quick accuracy benchmark
  SKIP_BOOTSTRAP=1            Skip dependency installation
  CONTEXT_LENS="1000 5000"    Override context lengths
  BATCH_SIZES="1 2 4 8"       Override batch sizes
  REPEAT=3                    Performance repeat count
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  usage >&2
  exit 2
fi

MODEL_REF="${MODEL:-${GR_MODEL:-}}"
MODEL_DIR_REF="${MODEL_DIR:-${GR_MODEL_DIR:-}}"
if [[ -z "${MODEL_VARIANT:-}" ]]; then
  if [[ -n "${MODEL_REF}" ]]; then
    MODEL_REF="${MODEL_REF%/}"
    MODEL_VARIANT="${MODEL_REF##*/}"
  elif [[ -n "${MODEL_DIR_REF}" ]]; then
    MODEL_DIR_REF="${MODEL_DIR_REF%/}"
    MODEL_VARIANT="${MODEL_DIR_REF##*/}"
  else
    MODEL_VARIANT="Qwen3-1.7B"
  fi
fi
SGLANG_REPO="${SGLANG_REPO:-$(gr_default_sglang_repo)}"

QUICK_CONTEXT_LENS="${QUICK_CONTEXT_LENS:-${CONTEXT_LENS:-1000}}"
QUICK_BEAM_WIDTHS="${QUICK_BEAM_WIDTHS:-${BEAM_WIDTHS:-256}}"
QUICK_BATCH_SIZES="${QUICK_BATCH_SIZES:-${BATCH_SIZES:-1}}"
QUICK_REPEAT="${QUICK_REPEAT:-${REPEAT:-1}}"
QUICK_CORRECTNESS_REPEAT="${QUICK_CORRECTNESS_REPEAT:-${CORRECTNESS_REPEAT:-1}}"
RUN_ACCURACY="${RUN_ACCURACY:-0}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"

echo "== SID-GR Inference offline quickstart =="
echo "repo: ${GR_INFERENCE_REPO_ROOT}"
echo "model_variant: ${MODEL_VARIANT}"
echo "sglang_repo: ${SGLANG_REPO}"
echo "contexts: ${QUICK_CONTEXT_LENS}; beams: ${QUICK_BEAM_WIDTHS}; batches: ${QUICK_BATCH_SIZES}"
echo

gr_require_sglang_repo "${SGLANG_REPO}" 1

if [[ "${SKIP_BOOTSTRAP}" != "1" ]]; then
  "${SCRIPT_DIR}/bootstrap_container_env.sh" --project-kernels --smoke
fi

echo
echo "== Running quick offline performance benchmark =="
MODEL_VARIANT="${MODEL_VARIANT}" \
SGLANG_REPO="${SGLANG_REPO}" \
CONTEXT_LENS="${QUICK_CONTEXT_LENS}" \
BEAM_WIDTHS="${QUICK_BEAM_WIDTHS}" \
BATCH_SIZES="${QUICK_BATCH_SIZES}" \
REPEAT="${QUICK_REPEAT}" \
"${SCRIPT_DIR}/run_offline_perf_benchmark.sh"

if [[ "${RUN_ACCURACY}" == "1" ]]; then
  echo
  echo "== Running quick offline accuracy benchmark =="
  MODEL_VARIANT="${MODEL_VARIANT}" \
  SGLANG_REPO="${SGLANG_REPO}" \
  CONTEXT_LENS="${QUICK_CONTEXT_LENS}" \
  BEAM_WIDTHS="${QUICK_BEAM_WIDTHS}" \
  BATCH_SIZES="${QUICK_BATCH_SIZES}" \
  CORRECTNESS_REPEAT="${QUICK_CORRECTNESS_REPEAT}" \
  "${SCRIPT_DIR}/run_offline_accuracy_benchmark.sh"
else
  echo
  echo "Accuracy quick check is optional. Run with RUN_ACCURACY=1 to include scripts/run_offline_accuracy_benchmark.sh."
fi
