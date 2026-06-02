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
OUT_DIR="${OUT_DIR:-benchmark_artifacts/sglang_compare/nsys_ctx1000_beam256_b1}"
gr_setup_local_cache_env

CONTEXT_LEN="${CONTEXT_LEN:-1000}"
GR_DECODE_STEPS="${GR_DECODE_STEPS:-2}"
SGLANG_DECODE_STEPS="${SGLANG_DECODE_STEPS:-3}"
BEAM_WIDTH="${BEAM_WIDTH:-256}"
REQUESTS="${REQUESTS:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
CONTEXT_KV_POOL_CAPACITY="${CONTEXT_KV_POOL_CAPACITY:-${MAX_BATCH_SIZE}}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
REPEAT="${REPEAT:-1}"
GR_PROFILE_CONTINUOUS_DECODE="${GR_PROFILE_CONTINUOUS_DECODE:-0}"
GR_PROFILE_SYNC="${GR_PROFILE_SYNC:-1}"
GR_EXECUTOR_SYNC_TIMING="${GR_EXECUTOR_SYNC_TIMING:-0}"
SGLANG_PROFILE_MODULES="${SGLANG_PROFILE_MODULES:-0}"
SGLANG_GR_STAGE_NVTX="${SGLANG_GR_STAGE_NVTX:-1}"
SGLANG_DISABLE_RADIX_CACHE="${SGLANG_DISABLE_RADIX_CACHE:-1}"
SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH="${SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH:-0}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-0}"
GR_ENABLE_PREFILL_CACHE="${GR_ENABLE_PREFILL_CACHE:-0}"
GR_RETURN_BEAM_DETAILS="${GR_RETURN_BEAM_DETAILS:-0}"
GR_NSYS_CUDA_GRAPH_TRACE="${GR_NSYS_CUDA_GRAPH_TRACE:-}"
SGLANG_NSYS_CUDA_GRAPH_TRACE="${SGLANG_NSYS_CUDA_GRAPH_TRACE:-}"

GR_REPORT="${OUT_DIR}/gr_ctx${CONTEXT_LEN}_beam${BEAM_WIDTH}_b${REQUESTS}"
SGLANG_REPORT="${OUT_DIR}/sglang_ctx${CONTEXT_LEN}_beam${BEAM_WIDTH}_b${REQUESTS}"
WORKLOAD_JSONL="${OUT_DIR}/qwen3_ctx${CONTEXT_LEN}_req${REQUESTS}.jsonl"

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys is required but was not found on PATH" >&2
  exit 2
fi
gr_require_sglang_repo "${SGLANG_REPO}"

mkdir -p "${OUT_DIR}"

GR_PROFILE_ARGS=()
GR_NVTX_ENV=()
GR_OUTPUT_DETAIL_ARGS=()
if [[ "${GR_PROFILE_CONTINUOUS_DECODE}" == "1" ]]; then
  GR_PROFILE_ARGS=(--profile-continuous-decode --profile-detail fine)
  GR_NVTX_ENV=(GR_INFERENCE_NVTX=1)
  if [[ "${GR_PROFILE_SYNC}" == "0" ]]; then
    GR_PROFILE_ARGS+=(--no-profile-sync)
  fi
fi
gr_append_gr_beam_detail_args GR_OUTPUT_DETAIL_ARGS --record-outputs
GR_PREFILL_CACHE_ARGS=()
gr_append_gr_prefill_cache_args GR_PREFILL_CACHE_ARGS
GR_EXECUTOR_SYNC_ARGS=(--no-executor-sync-timing)
if [[ "${GR_EXECUTOR_SYNC_TIMING}" == "1" ]]; then
  GR_EXECUTOR_SYNC_ARGS=(--executor-sync-timing)
fi
GR_NSYS_GRAPH_TRACE_ARGS=()
if [[ -n "${GR_NSYS_CUDA_GRAPH_TRACE}" ]]; then
  GR_NSYS_GRAPH_TRACE_ARGS=(--cuda-graph-trace="${GR_NSYS_CUDA_GRAPH_TRACE}")
fi
SGLANG_NSYS_GRAPH_TRACE_ARGS=()
if [[ -n "${SGLANG_NSYS_CUDA_GRAPH_TRACE}" ]]; then
  SGLANG_NSYS_GRAPH_TRACE_ARGS=(--cuda-graph-trace="${SGLANG_NSYS_CUDA_GRAPH_TRACE}")
fi

echo "[1/5] Build shared workload: ${WORKLOAD_JSONL}"
PYTHONPATH=src \
python tools/make_qwen3_beam_workload.py \
  --model-dir "${MODEL_DIR}" \
  --context-len "${CONTEXT_LEN}" \
  --requests "${REQUESTS}" \
  --no-tokenizer \
  --output-jsonl "${WORKLOAD_JSONL}"

echo "[2/5] Capture GR nsys trace: ${GR_REPORT}.nsys-rep"
env \
  PYTHONPATH="${SGLANG_REPO}/python:src:${PYTHONPATH:-}" \
  GR_DECODE_ATTEN_ROOT="${GR_DECODE_ATTEN_ROOT}" \
  "${GR_NVTX_ENV[@]}" \
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  "${GR_NSYS_GRAPH_TRACE_ARGS[@]}" \
  --output "${GR_REPORT}" \
  python tools/run_qwen3_real_weight_serving.py \
    --model-dir "${MODEL_DIR}" \
    --workload-jsonl "${WORKLOAD_JSONL}" \
    --context-len "${CONTEXT_LEN}" \
    --decode-steps "${GR_DECODE_STEPS}" \
    --beam-width "${BEAM_WIDTH}" \
    --requests "${REQUESTS}" \
    --max-batch-size "${MAX_BATCH_SIZE}" \
    --batched-decode \
    --continuous \
    --beam-kv-pool-capacity "${MAX_BATCH_SIZE}" \
    --context-kv-pool-capacity "${CONTEXT_KV_POOL_CAPACITY}" \
    --decode-backend real \
    --device cuda \
    "${GR_EXECUTOR_SYNC_ARGS[@]}" \
    "${GR_PREFILL_CACHE_ARGS[@]}" \
    "${GR_PROFILE_ARGS[@]}" \
    "${GR_OUTPUT_DETAIL_ARGS[@]}" \
    --warmup-runs "${WARMUP_RUNS}" \
    --repeat "${REPEAT}" \
    --cuda-profiler-range \
    --output-json "${OUT_DIR}/gr_ctx${CONTEXT_LEN}_beam${BEAM_WIDTH}_b${REQUESTS}.json"

echo "[3/5] Capture SGLang nsys trace: ${SGLANG_REPORT}.nsys-rep"
SGLANG_PROFILE_ARGS=()
if [[ "${SGLANG_PROFILE_MODULES}" == "1" ]]; then
  SGLANG_PROFILE_ARGS=(--profile-modules)
fi
gr_append_sglang_cache_args SGLANG_PROFILE_ARGS
gr_append_sglang_graph_args SGLANG_PROFILE_ARGS
PYTHONPATH="${SGLANG_REPO}/python:src:${PYTHONPATH:-}" \
SGLANG_GR_STAGE_NVTX="${SGLANG_GR_STAGE_NVTX}" \
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  "${SGLANG_NSYS_GRAPH_TRACE_ARGS[@]}" \
  --output "${SGLANG_REPORT}" \
  "${SGLANG_PYTHON}" tools/run_sglang_beam_benchmark.py \
    --model-dir "${MODEL_DIR}" \
    --sglang-repo "${SGLANG_REPO}" \
    --workload-jsonl "${WORKLOAD_JSONL}" \
    --context-len "${CONTEXT_LEN}" \
    --decode-steps "${SGLANG_DECODE_STEPS}" \
    --beam-width "${BEAM_WIDTH}" \
    --requests "${REQUESTS}" \
    --arrival-mode batch \
    --warmup-runs "${WARMUP_RUNS}" \
    --repeat "${REPEAT}" \
    --use-input-ids \
    --no-tokenizer \
    "${SGLANG_PROFILE_ARGS[@]}" \
    --cuda-profiler-range \
    --output-json "${OUT_DIR}/sglang_ctx${CONTEXT_LEN}_beam${BEAM_WIDTH}_b${REQUESTS}.json"

echo "[4/5] Export sqlite files"
nsys export --type sqlite --force-overwrite=true \
  --output "${GR_REPORT}.sqlite" \
  "${GR_REPORT}.nsys-rep"
nsys export --type sqlite --force-overwrite=true \
  --output "${SGLANG_REPORT}.sqlite" \
  "${SGLANG_REPORT}.nsys-rep"

echo "[5/5] Analyze side-by-side breakdown"
PYTHONPATH=src \
python tools/analyze_nsys_gr_sglang.py \
  --gr-sqlite "${GR_REPORT}.sqlite" \
  --sglang-sqlite "${SGLANG_REPORT}.sqlite" \
  --gr-json "${OUT_DIR}/gr_ctx${CONTEXT_LEN}_beam${BEAM_WIDTH}_b${REQUESTS}.json" \
  --sglang-json "${OUT_DIR}/sglang_ctx${CONTEXT_LEN}_beam${BEAM_WIDTH}_b${REQUESTS}.json" \
  --output-json "${OUT_DIR}/gr_vs_sglang_nsys_ctx${CONTEXT_LEN}_beam${BEAM_WIDTH}_b${REQUESTS}.json" \
  --output-markdown "${OUT_DIR}/gr_vs_sglang_nsys_ctx${CONTEXT_LEN}_beam${BEAM_WIDTH}_b${REQUESTS}.md"

echo "Done. nsys reports and breakdown are under ${OUT_DIR}"
