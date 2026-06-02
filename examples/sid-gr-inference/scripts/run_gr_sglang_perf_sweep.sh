#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common_paths.sh
source "${SCRIPT_DIR}/common_paths.sh"

MODEL_VARIANT="${MODEL_VARIANT:-Qwen3-1.7B}"
MODEL_DIR="${MODEL_DIR:-$(gr_default_model_dir "${MODEL_VARIANT}")}"
SGLANG_REPO="${SGLANG_REPO:-$(gr_default_sglang_repo)}"
SGLANG_PYTHON="${SGLANG_PYTHON:-$(gr_default_sglang_python)}"
GR_DECODE_ATTEN_ROOT="${GR_DECODE_ATTEN_ROOT:-$(gr_default_decode_atten_root)}"
OUT_DIR="${OUT_DIR:-benchmark_artifacts/sglang_compare/perf_sweep}"
gr_setup_local_cache_env

CONTEXT_LENS="${CONTEXT_LENS:-1000 5000}"
BEAM_WIDTHS="${BEAM_WIDTHS:-64 128 256}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
GR_DECODE_STEPS="${GR_DECODE_STEPS:-2}"
SGLANG_DECODE_STEPS="${SGLANG_DECODE_STEPS:-3}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
REPEAT="${REPEAT:-3}"
SGLANG_DISABLE_RADIX_CACHE="${SGLANG_DISABLE_RADIX_CACHE:-1}"
SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH="${SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH:-0}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-0}"
GR_RETURN_BEAM_DETAILS="${GR_RETURN_BEAM_DETAILS:-0}"
GR_SUPPRESS_TOKEN_IDS="${GR_SUPPRESS_TOKEN_IDS:-}"
GR_ENABLE_PREFILL_CACHE="${GR_ENABLE_PREFILL_CACHE:-0}"

mkdir -p "${OUT_DIR}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

gr_require_sglang_repo "${SGLANG_REPO}"

GR_OUTPUT_DETAIL_ARGS=()
gr_append_gr_beam_detail_args GR_OUTPUT_DETAIL_ARGS --record-outputs
GR_LOGITS_PROCESSOR_ARGS=()
gr_append_option_if_env_set GR_LOGITS_PROCESSOR_ARGS GR_SUPPRESS_TOKEN_IDS --suppress-token-ids
GR_PREFILL_CACHE_ARGS=()
gr_append_gr_prefill_cache_args GR_PREFILL_CACHE_ARGS
SGLANG_CACHE_ARGS=()
gr_append_sglang_cache_args SGLANG_CACHE_ARGS
SGLANG_GRAPH_ARGS=()
gr_append_sglang_graph_args SGLANG_GRAPH_ARGS

for context_len in ${CONTEXT_LENS}; do
  for beam_width in ${BEAM_WIDTHS}; do
    for requests in ${BATCH_SIZES}; do
      suffix="ctx${context_len}_beam${beam_width}_req${requests}"
      workload_jsonl="${OUT_DIR}/qwen3_ctx${context_len}_req${requests}.jsonl"
      gr_json="${OUT_DIR}/gr_${suffix}.json"
      sglang_json="${OUT_DIR}/sglang_${suffix}.json"
      gr_log="${LOG_DIR}/gr_${suffix}.log"
      sglang_log="${LOG_DIR}/sglang_${suffix}.log"

      echo "== ${suffix} =="
      PYTHONPATH=src \
      python tools/make_qwen3_beam_workload.py \
        --model-dir "${MODEL_DIR}" \
        --context-len "${context_len}" \
        --requests "${requests}" \
        --no-tokenizer \
        --output-jsonl "${workload_jsonl}"

      echo "[GR] ${gr_json} (log: ${gr_log})"
      if ! PYTHONPATH="${SGLANG_REPO}/python:src:${PYTHONPATH:-}" \
        GR_DECODE_ATTEN_ROOT="${GR_DECODE_ATTEN_ROOT}" \
        python tools/run_qwen3_real_weight_serving.py \
          --model-dir "${MODEL_DIR}" \
          --workload-jsonl "${workload_jsonl}" \
          --context-len "${context_len}" \
          --decode-steps "${GR_DECODE_STEPS}" \
          --beam-width "${beam_width}" \
          --requests "${requests}" \
          --max-batch-size "${requests}" \
          --batched-decode \
          --continuous \
          --beam-kv-pool-capacity "${requests}" \
          --context-kv-pool-capacity "${requests}" \
          --decode-backend real \
          --device cuda \
          "${GR_OUTPUT_DETAIL_ARGS[@]}" \
          "${GR_LOGITS_PROCESSOR_ARGS[@]}" \
          "${GR_PREFILL_CACHE_ARGS[@]}" \
          --warmup-runs "${WARMUP_RUNS}" \
          --repeat "${REPEAT}" \
          --output-json "${gr_json}" >"${gr_log}" 2>&1; then
        echo "GR benchmark failed for ${suffix}; tail of ${gr_log}:" >&2
        tail -n 80 "${gr_log}" >&2 || true
        exit 1
      fi

      echo "[SGLang] ${sglang_json} (log: ${sglang_log})"
      if ! PYTHONPATH="${SGLANG_REPO}/python:src:${PYTHONPATH:-}" \
        "${SGLANG_PYTHON}" tools/run_sglang_beam_benchmark.py \
          --model-dir "${MODEL_DIR}" \
          --sglang-repo "${SGLANG_REPO}" \
          --workload-jsonl "${workload_jsonl}" \
          --context-len "${context_len}" \
          --decode-steps "${SGLANG_DECODE_STEPS}" \
          --beam-width "${beam_width}" \
          --requests "${requests}" \
          --arrival-mode batch \
          "${SGLANG_CACHE_ARGS[@]}" \
          "${SGLANG_GRAPH_ARGS[@]}" \
          --warmup-runs "${WARMUP_RUNS}" \
          --repeat "${REPEAT}" \
          --use-input-ids \
          --no-tokenizer \
          --output-json "${sglang_json}" >"${sglang_log}" 2>&1; then
        echo "SGLang benchmark failed for ${suffix}; tail of ${sglang_log}:" >&2
        tail -n 80 "${sglang_log}" >&2 || true
        exit 1
      fi
    done
  done
done

PYTHONPATH=src python tools/summarize_gr_sglang_perf_sweep.py "${OUT_DIR}"
