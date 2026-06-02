#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common_paths.sh
source "${SCRIPT_DIR}/common_paths.sh"

MODEL_VARIANT="${MODEL_VARIANT:-Qwen3-1.7B}"
MODEL_DIR="${MODEL_DIR:-$(gr_default_model_dir "${MODEL_VARIANT}")}"
GR_DECODE_ATTEN_ROOT="${GR_DECODE_ATTEN_ROOT:-$(gr_default_decode_atten_root)}"
SGLANG_REPO="${SGLANG_REPO:-$(gr_default_sglang_repo)}"
SGLANG_PYTHON="${SGLANG_PYTHON:-$(gr_default_sglang_python)}"
OUT_DIR="${OUT_DIR:-benchmark_artifacts/sglang_compare}"
gr_setup_local_cache_env

CONTEXT_LEN="${CONTEXT_LEN:-5000}"
GR_DECODE_STEPS="${GR_DECODE_STEPS:-2}"
SGLANG_DECODE_STEPS="${SGLANG_DECODE_STEPS:-3}"
BEAM_WIDTH="${BEAM_WIDTH:-256}"
REQUESTS="${REQUESTS:-4}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-${REQUESTS}}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
REPEAT="${REPEAT:-3}"
ARRIVAL_MODE="${ARRIVAL_MODE:-batch}"
ARRIVAL_STAGGER_TICKS="${ARRIVAL_STAGGER_TICKS:-1}"
SGLANG_ARRIVAL_STAGGER_MS="${SGLANG_ARRIVAL_STAGGER_MS:-20}"
ARRIVAL_BURST_SIZE="${ARRIVAL_BURST_SIZE:-1}"
SGLANG_STAGGERED_WORKERS="${SGLANG_STAGGERED_WORKERS:-1}"
SGLANG_DISABLE_RADIX_CACHE="${SGLANG_DISABLE_RADIX_CACHE:-1}"
SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH="${SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH:-0}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-0}"
GR_CONTINUOUS="${GR_CONTINUOUS:-1}"
GR_PROFILE_CONTINUOUS_DECODE="${GR_PROFILE_CONTINUOUS_DECODE:-0}"
GR_RETURN_BEAM_DETAILS="${GR_RETURN_BEAM_DETAILS:-0}"
GR_SUPPRESS_TOKEN_IDS="${GR_SUPPRESS_TOKEN_IDS:-}"
GR_ENABLE_PREFILL_CACHE="${GR_ENABLE_PREFILL_CACHE:-0}"

if [[ "${ARRIVAL_MODE}" != "batch" && "${ARRIVAL_MODE}" != "staggered" ]]; then
  echo "ARRIVAL_MODE must be 'batch' or 'staggered', got: ${ARRIVAL_MODE}" >&2
  exit 2
fi
if [[ "${ARRIVAL_MODE}" == "staggered" && "${GR_CONTINUOUS}" != "1" ]]; then
  echo "ARRIVAL_MODE=staggered requires GR_CONTINUOUS=1" >&2
  exit 2
fi

WORKLOAD_JSONL="${OUT_DIR}/qwen3_ctx${CONTEXT_LEN}_req${REQUESTS}.jsonl"
SUFFIX="ctx${CONTEXT_LEN}_w${BEAM_WIDTH}_req${REQUESTS}_${ARRIVAL_MODE}_gr${GR_DECODE_STEPS}_sg${SGLANG_DECODE_STEPS}"
GR_JSON="${OUT_DIR}/gr_qwen3_${SUFFIX}.json"
SGLANG_JSON="${OUT_DIR}/sglang_qwen3_${SUFFIX}.json"
REPORT_JSON="${OUT_DIR}/gr_vs_sglang_qwen3_${SUFFIX}.json"
REPORT_MD="${OUT_DIR}/gr_vs_sglang_qwen3_${SUFFIX}.md"

gr_require_sglang_repo "${SGLANG_REPO}" 1

mkdir -p "${OUT_DIR}"

GR_ARRIVAL_ARGS=()
SGLANG_ARRIVAL_ARGS=(--arrival-mode "${ARRIVAL_MODE}")
SGLANG_CACHE_ARGS=()
gr_append_sglang_cache_args SGLANG_CACHE_ARGS
SGLANG_GRAPH_ARGS=()
gr_append_sglang_graph_args SGLANG_GRAPH_ARGS
GR_PROFILE_ARGS=()
if [[ "${GR_CONTINUOUS}" == "1" && "${GR_PROFILE_CONTINUOUS_DECODE}" == "1" ]]; then
  GR_PROFILE_ARGS=(--profile-continuous-decode --profile-detail coarse)
fi
GR_OUTPUT_DETAIL_ARGS=()
gr_append_gr_beam_detail_args GR_OUTPUT_DETAIL_ARGS --verbose-metadata
GR_LOGITS_PROCESSOR_ARGS=()
gr_append_option_if_env_set GR_LOGITS_PROCESSOR_ARGS GR_SUPPRESS_TOKEN_IDS --suppress-token-ids
GR_PREFILL_CACHE_ARGS=()
gr_append_gr_prefill_cache_args GR_PREFILL_CACHE_ARGS
GR_SERVING_ARGS=()
if [[ "${GR_CONTINUOUS}" == "1" ]]; then
  GR_SERVING_ARGS=(
    --continuous
    --beam-kv-pool-capacity "${MAX_BATCH_SIZE}"
    --context-kv-pool-capacity "${MAX_BATCH_SIZE}"
  )
fi
if [[ "${ARRIVAL_MODE}" == "staggered" ]]; then
  GR_ARRIVAL_ARGS=(
    --arrival-stagger-ticks "${ARRIVAL_STAGGER_TICKS}"
    --arrival-burst-size "${ARRIVAL_BURST_SIZE}"
  )
  SGLANG_ARRIVAL_ARGS+=(
    --arrival-stagger-ms "${SGLANG_ARRIVAL_STAGGER_MS}"
    --arrival-burst-size "${ARRIVAL_BURST_SIZE}"
    --staggered-workers "${SGLANG_STAGGERED_WORKERS}"
  )
fi

echo "[1/4] Build shared workload: ${WORKLOAD_JSONL}"
PYTHONPATH=src \
python tools/make_qwen3_beam_workload.py \
  --model-dir "${MODEL_DIR}" \
  --context-len "${CONTEXT_LEN}" \
  --requests "${REQUESTS}" \
  --no-tokenizer \
  --output-jsonl "${WORKLOAD_JSONL}"

echo "[2/4] Run SID-GR Inference fixed-beam benchmark: ${GR_JSON}"
PYTHONPATH=src \
GR_DECODE_ATTEN_ROOT="${GR_DECODE_ATTEN_ROOT}" \
python tools/run_qwen3_real_weight_serving.py \
  --model-dir "${MODEL_DIR}" \
  --workload-jsonl "${WORKLOAD_JSONL}" \
  --context-len "${CONTEXT_LEN}" \
  --decode-steps "${GR_DECODE_STEPS}" \
  --beam-width "${BEAM_WIDTH}" \
  --requests "${REQUESTS}" \
  --max-batch-size "${MAX_BATCH_SIZE}" \
  --batched-decode \
  --decode-backend real \
  --device cuda \
  "${GR_SERVING_ARGS[@]}" \
  "${GR_PROFILE_ARGS[@]}" \
  "${GR_OUTPUT_DETAIL_ARGS[@]}" \
  "${GR_LOGITS_PROCESSOR_ARGS[@]}" \
  "${GR_PREFILL_CACHE_ARGS[@]}" \
  --record-outputs \
  "${GR_ARRIVAL_ARGS[@]}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --repeat "${REPEAT}" \
  --output-json "${GR_JSON}"

echo "[3/4] Run SGLang PR fixed-beam benchmark: ${SGLANG_JSON}"
PYTHONPATH="${SGLANG_REPO}/python:src:${PYTHONPATH:-}" \
"${SGLANG_PYTHON}" tools/run_sglang_beam_benchmark.py \
  --model-dir "${MODEL_DIR}" \
  --sglang-repo "${SGLANG_REPO}" \
  --workload-jsonl "${WORKLOAD_JSONL}" \
  --context-len "${CONTEXT_LEN}" \
  --decode-steps "${SGLANG_DECODE_STEPS}" \
  --beam-width "${BEAM_WIDTH}" \
  --requests "${REQUESTS}" \
  "${SGLANG_ARRIVAL_ARGS[@]}" \
  "${SGLANG_CACHE_ARGS[@]}" \
  "${SGLANG_GRAPH_ARGS[@]}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --repeat "${REPEAT}" \
  --use-input-ids \
  --no-tokenizer \
  --output-json "${SGLANG_JSON}"

echo "[4/4] Compare performance and outputs: ${REPORT_MD}"
PYTHONPATH=src \
python tools/compare_gr_sglang_beam.py \
  --gr-json "${GR_JSON}" \
  --sglang-json "${SGLANG_JSON}" \
  --output-json "${REPORT_JSON}" \
  --output-markdown "${REPORT_MD}"

echo "Done."
echo "JSON report: ${REPORT_JSON}"
echo "Markdown report: ${REPORT_MD}"
