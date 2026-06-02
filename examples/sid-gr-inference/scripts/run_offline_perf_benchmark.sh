#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common_paths.sh
source "${SCRIPT_DIR}/common_paths.sh"
cd "${GR_INFERENCE_REPO_ROOT}"

MODEL_DIR="${MODEL_DIR:-${GR_MODEL_DIR:-}}"
MODEL="${MODEL:-${GR_MODEL:-}}"
MODEL_REVISION="${MODEL_REVISION:-${GR_MODEL_REVISION:-}}"
MODEL_DOWNLOAD_DIR="${MODEL_DOWNLOAD_DIR:-${GR_MODEL_DOWNLOAD_DIR:-}}"
if [[ -z "${MODEL_VARIANT:-}" ]]; then
  if [[ -n "${MODEL}" ]]; then
    model_ref_for_variant="${MODEL%/}"
    MODEL_VARIANT="${model_ref_for_variant##*/}"
  else
    MODEL_VARIANT="Qwen3-1.7B"
  fi
fi
DEFAULT_MODEL_DIR="$(gr_default_model_dir "${MODEL_VARIANT}")"
if [[ -z "${MODEL_DIR}" && -z "${MODEL}" ]]; then
  if [[ -d "${DEFAULT_MODEL_DIR}" ]]; then
    MODEL_DIR="${DEFAULT_MODEL_DIR}"
  else
    MODEL="$(gr_default_hf_model_id "${MODEL_VARIANT}")"
  fi
fi
if [[ -z "${MODEL_DIR}" && -z "${MODEL_DOWNLOAD_DIR}" ]]; then
  MODEL_DOWNLOAD_DIR="$(gr_model_download_dir "${MODEL}" "${MODEL_ROOT:-${GR_INFERENCE_WORKSPACE_ROOT}/models}" "${MODEL_VARIANT}")"
fi
SGLANG_REPO="${SGLANG_REPO:-$(gr_default_sglang_repo)}"
SGLANG_PYTHON="${SGLANG_PYTHON:-$(gr_default_sglang_python)}"
GR_DECODE_ATTEN_ROOT="${GR_DECODE_ATTEN_ROOT:-$(gr_default_decode_atten_root)}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-benchmark_artifacts/sglang_compare/offline_perf_${RUN_ID}}"

CONTEXT_LENS="${CONTEXT_LENS:-1000 5000}"
BEAM_WIDTHS="${BEAM_WIDTHS:-256}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
GR_DECODE_STEPS="${GR_DECODE_STEPS:-2}"
SGLANG_DECODE_STEPS="${SGLANG_DECODE_STEPS:-3}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
REPEAT="${REPEAT:-3}"

mkdir -p "${OUT_DIR}"
gr_setup_local_cache_env
gr_require_sglang_repo "${SGLANG_REPO}"
if [[ -z "${MODEL_DIR}" && -n "${MODEL_DOWNLOAD_DIR}" ]]; then
  echo "model_download_dir: ${MODEL_DOWNLOAD_DIR}"
fi
MODEL_DIR="$(gr_resolve_model_dir "${MODEL_DIR}" "${MODEL}" "${MODEL_REVISION}" "$(gr_default_hf_model_id "${MODEL_VARIANT}")" "${MODEL_DOWNLOAD_DIR}")"

echo "== Offline performance benchmark =="
echo "model: ${MODEL_DIR}"
if [[ -n "${MODEL}" ]]; then
  echo "model_ref: ${MODEL}"
fi
if [[ -n "${MODEL_REVISION}" ]]; then
  echo "model_revision: ${MODEL_REVISION}"
fi
echo "sglang_repo: ${SGLANG_REPO}"
echo "out_dir: ${OUT_DIR}"
echo "repo: ${GR_INFERENCE_REPO_ROOT}"
echo "run_id: ${RUN_ID}"
echo "contexts: ${CONTEXT_LENS}; beams: ${BEAM_WIDTHS}; batches: ${BATCH_SIZES}"
echo

MODEL_DIR="${MODEL_DIR}" \
SGLANG_REPO="${SGLANG_REPO}" \
SGLANG_PYTHON="${SGLANG_PYTHON}" \
GR_DECODE_ATTEN_ROOT="${GR_DECODE_ATTEN_ROOT}" \
OUT_DIR="${OUT_DIR}" \
CONTEXT_LENS="${CONTEXT_LENS}" \
BEAM_WIDTHS="${BEAM_WIDTHS}" \
BATCH_SIZES="${BATCH_SIZES}" \
GR_DECODE_STEPS="${GR_DECODE_STEPS}" \
SGLANG_DECODE_STEPS="${SGLANG_DECODE_STEPS}" \
WARMUP_RUNS="${WARMUP_RUNS}" \
REPEAT="${REPEAT}" \
SGLANG_DISABLE_RADIX_CACHE=1 \
GR_ENABLE_PREFILL_CACHE=0 \
GR_RETURN_BEAM_DETAILS=0 \
"${SCRIPT_DIR}/run_gr_sglang_perf_sweep.sh"

echo
echo "Offline performance summary:"
echo "  ${OUT_DIR}/summary.md"
echo "  ${OUT_DIR}/summary.csv"
