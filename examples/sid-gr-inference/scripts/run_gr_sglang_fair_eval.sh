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
OUT_DIR="${OUT_DIR:-benchmark_artifacts/sglang_compare/fair_eval}"
gr_setup_local_cache_env

CONTEXT_LENS="${CONTEXT_LENS:-1000 5000}"
BEAM_WIDTHS="${BEAM_WIDTHS:-64 128 256}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
PERF_REPEAT="${PERF_REPEAT:-3}"
CORRECTNESS_REPEAT="${CORRECTNESS_REPEAT:-1}"
GR_DECODE_STEPS="${GR_DECODE_STEPS:-2}"
SGLANG_DECODE_STEPS="${SGLANG_DECODE_STEPS:-3}"

mkdir -p "${OUT_DIR}"
gr_require_sglang_repo "${SGLANG_REPO}"

run_perf() {
  local cache_label="$1"
  local disable_radix="$2"
  local perf_dir="${OUT_DIR}/perf_${cache_label}"
  local gr_enable_prefill_cache
  if [[ -n "${GR_ENABLE_PREFILL_CACHE+x}" ]]; then
    gr_enable_prefill_cache="${GR_ENABLE_PREFILL_CACHE}"
  elif [[ "${disable_radix}" == "0" ]]; then
    gr_enable_prefill_cache=1
  else
    gr_enable_prefill_cache=0
  fi
  echo "== performance (${cache_label}) -> ${perf_dir} =="
  CONTEXT_LENS="${CONTEXT_LENS}" \
  BEAM_WIDTHS="${BEAM_WIDTHS}" \
  BATCH_SIZES="${BATCH_SIZES}" \
  REPEAT="${PERF_REPEAT}" \
  GR_DECODE_STEPS="${GR_DECODE_STEPS}" \
  SGLANG_DECODE_STEPS="${SGLANG_DECODE_STEPS}" \
  SGLANG_DISABLE_RADIX_CACHE="${disable_radix}" \
  GR_ENABLE_PREFILL_CACHE="${gr_enable_prefill_cache}" \
  GR_RETURN_BEAM_DETAILS=0 \
  MODEL_DIR="${MODEL_DIR}" \
  SGLANG_REPO="${SGLANG_REPO}" \
  SGLANG_PYTHON="${SGLANG_PYTHON}" \
  GR_DECODE_ATTEN_ROOT="${GR_DECODE_ATTEN_ROOT}" \
  OUT_DIR="${perf_dir}" \
  scripts/run_gr_sglang_perf_sweep.sh
}

run_correctness() {
  local cache_label="$1"
  local disable_radix="$2"
  local correctness_dir="${OUT_DIR}/correctness_${cache_label}"
  local log_dir="${correctness_dir}/logs"
  mkdir -p "${correctness_dir}"
  mkdir -p "${log_dir}"
  echo "== correctness (${cache_label}) -> ${correctness_dir} =="

  for context_len in ${CONTEXT_LENS}; do
    for beam_width in ${BEAM_WIDTHS}; do
      for requests in ${BATCH_SIZES}; do
        local suffix="ctx${context_len}_beam${beam_width}_req${requests}"
        local workload_jsonl="${correctness_dir}/qwen3_ctx${context_len}_req${requests}.jsonl"
        local gr_json="${correctness_dir}/gr_${suffix}.json"
        local sglang_json="${correctness_dir}/sglang_${suffix}.json"
        local report_json="${correctness_dir}/compare_${suffix}.json"
        local report_md="${correctness_dir}/compare_${suffix}.md"
        local gr_log="${log_dir}/gr_${suffix}.log"
        local sglang_log="${log_dir}/sglang_${suffix}.log"
        local compare_log="${log_dir}/compare_${suffix}.log"
        local sglang_cache_args=()
        local gr_prefill_cache_args=()
        if [[ "${disable_radix}" == "1" ]]; then
          gr_append_flag_if_one sglang_cache_args 1 --disable-radix-cache
        elif [[ "${GR_ENABLE_PREFILL_CACHE:-1}" == "1" ]]; then
          gr_append_flag_if_one gr_prefill_cache_args 1 --enable-prefill-cache
        fi

        echo "[correctness ${cache_label}] ${suffix}"
        PYTHONPATH=src \
        python tools/make_qwen3_beam_workload.py \
          --model-dir "${MODEL_DIR}" \
          --context-len "${context_len}" \
          --requests "${requests}" \
          --no-tokenizer \
          --output-jsonl "${workload_jsonl}"

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
            --return-beam-details \
            --record-outputs \
            "${gr_prefill_cache_args[@]}" \
            --warmup-runs 1 \
            --repeat "${CORRECTNESS_REPEAT}" \
            --output-json "${gr_json}" >"${gr_log}" 2>&1; then
          echo "GR correctness run failed for ${suffix}; tail of ${gr_log}:" >&2
          tail -n 80 "${gr_log}" >&2 || true
          exit 1
        fi

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
            "${sglang_cache_args[@]}" \
            --warmup-runs 1 \
            --repeat "${CORRECTNESS_REPEAT}" \
            --use-input-ids \
            --no-tokenizer \
            --output-json "${sglang_json}" >"${sglang_log}" 2>&1; then
          echo "SGLang correctness run failed for ${suffix}; tail of ${sglang_log}:" >&2
          tail -n 80 "${sglang_log}" >&2 || true
          exit 1
        fi

        if ! PYTHONPATH=src \
          python tools/compare_gr_sglang_beam.py \
            --gr-json "${gr_json}" \
            --sglang-json "${sglang_json}" \
            --output-json "${report_json}" \
            --output-markdown "${report_md}" >"${compare_log}" 2>&1; then
          echo "Comparison failed for ${suffix}; tail of ${compare_log}:" >&2
          tail -n 80 "${compare_log}" >&2 || true
          exit 1
        fi
      done
    done
  done
}

write_summary() {
  local summary="${OUT_DIR}/summary.md"
  {
    echo "# GR vs SGLang Fair Eval"
    echo
    echo "## Performance, Radix Off"
    echo
    if [[ -f "${OUT_DIR}/perf_radix_off/summary.md" ]]; then
      sed '1,2d' "${OUT_DIR}/perf_radix_off/summary.md"
    else
      echo "Not run."
    fi
    echo
    echo "## Performance, Radix On"
    echo
    if [[ -f "${OUT_DIR}/perf_radix_on/summary.md" ]]; then
      sed '1,2d' "${OUT_DIR}/perf_radix_on/summary.md"
    else
      echo "Not run."
    fi
    echo
    echo "## Correctness Reports"
    echo
    echo "- Radix off reports: \`${OUT_DIR}/correctness_radix_off/compare_*.md\`"
    echo "- Radix on reports: \`${OUT_DIR}/correctness_radix_on/compare_*.md\`"
  } > "${summary}"
  echo "Final summary: ${summary}"
  sed -n '1,220p' "${summary}"
}

run_perf radix_off 1
run_perf radix_on 0
run_correctness radix_off 1
run_correctness radix_on 0
write_summary
