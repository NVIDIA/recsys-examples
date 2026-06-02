#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common_paths.sh
source "${SCRIPT_DIR}/common_paths.sh"

MODEL_VARIANT="${MODEL_VARIANT:-Qwen3-1.7B}"
MODEL_DIR="${MODEL_DIR:-$(gr_default_model_dir "${MODEL_VARIANT}")}"
SGLANG_REPO="${SGLANG_REPO:-$(gr_default_sglang_repo)}"
SGLANG_PYTHON="${SGLANG_PYTHON:-$(gr_default_sglang_python)}"
OUT_DIR="${OUT_DIR:-benchmark_artifacts/sglang_compare}"
gr_setup_local_cache_env

HOST="${GR_BENCH_HOST:-127.0.0.1}"
PORT="${GR_BENCH_PORT:-8000}"
CONTEXT_LEN="${CONTEXT_LEN:-5000}"
DECODE_STEPS="${DECODE_STEPS:-3}"
BEAM_WIDTH="${BEAM_WIDTH:-256}"
REQUESTS="${REQUESTS:-32}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-0}"

mkdir -p "${OUT_DIR}"

EXTRA_REQUEST_BODY="$(gr_beam_sampling_extra_request_body "${DECODE_STEPS}" "${BEAM_WIDTH}")"

echo "Benchmarking GR /generate via sglang.bench_serving: http://${HOST}:${PORT}/generate"
echo "shape: context=${CONTEXT_LEN} output=${DECODE_STEPS} beam=${BEAM_WIDTH} requests=${REQUESTS} max_concurrency=${MAX_CONCURRENCY}"

PYTHONPATH="${SGLANG_REPO}/python:src:${PYTHONPATH:-}" \
"${SGLANG_PYTHON}" -m sglang.bench_serving \
  --backend sglang \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL_DIR}" \
  --tokenizer "${MODEL_DIR}" \
  --dataset-name random \
  --random-input-len "${CONTEXT_LEN}" \
  --random-output-len "${DECODE_STEPS}" \
  --random-range-ratio 1 \
  --num-prompts "${REQUESTS}" \
  --request-rate "${REQUEST_RATE}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --warmup-requests "${WARMUP_REQUESTS}" \
  --disable-stream \
  --disable-ignore-eos \
  --tokenize-prompt \
  --extra-request-body "${EXTRA_REQUEST_BODY}" \
  --output-details \
  --output-file "${OUT_DIR}/gr_sglang_bench_serving_ctx${CONTEXT_LEN}_w${BEAM_WIDTH}_req${REQUESTS}_rr${REQUEST_RATE}_mc${MAX_CONCURRENCY}.jsonl"
