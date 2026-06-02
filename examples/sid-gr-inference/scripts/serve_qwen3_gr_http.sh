#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common_paths.sh
source "${SCRIPT_DIR}/common_paths.sh"
REPO_ROOT="${GR_INFERENCE_REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_VARIANT="${GR_MODEL_VARIANT:-${MODEL_VARIANT:-Qwen3-1.7B}}"
DEFAULT_MODEL_DIR="$(gr_default_model_dir "${MODEL_VARIANT}")"
MODEL_DIR="${MODEL_DIR:-${GR_MODEL_DIR:-}}"
MODEL="${MODEL:-${GR_MODEL:-}}"
MODEL_REVISION="${MODEL_REVISION:-${GR_MODEL_REVISION:-}}"
if [[ -z "${MODEL_DIR}" && -z "${MODEL}" ]]; then
  if [[ -d "${DEFAULT_MODEL_DIR}" ]]; then
    MODEL_DIR="${DEFAULT_MODEL_DIR}"
  else
    MODEL="$(gr_default_hf_model_id "${MODEL_VARIANT}")"
  fi
fi
export GR_DECODE_ATTEN_ROOT="${GR_DECODE_ATTEN_ROOT:-$(gr_default_decode_atten_root)}"
HOST="${GR_HTTP_HOST:-0.0.0.0}"
PORT="${GR_HTTP_PORT:-8000}"
CONTEXT_LEN="${GR_CONTEXT_LEN:-16}"
DECODE_STEPS="${GR_DECODE_STEPS:-1}"
BEAM_WIDTH="${GR_BEAM_WIDTH:-128}"
MAX_BATCH_SIZE="${GR_MAX_BATCH_SIZE:-2}"
DECODE_BACKEND="${GR_DECODE_BACKEND:-real}"
DEVICE="${GR_DEVICE:-cuda}"
BEAM_KV_POOL_CAPACITY="${GR_BEAM_KV_POOL_CAPACITY:-${MAX_BATCH_SIZE}}"
CONTEXT_KV_POOL_CAPACITY="${GR_CONTEXT_KV_POOL_CAPACITY:-${MAX_BATCH_SIZE}}"
MAX_HTTP_REQUEST_BYTES="${GR_MAX_HTTP_REQUEST_BYTES:-1048576}"
MAX_HTTP_SUBMIT_MANY="${GR_MAX_HTTP_SUBMIT_MANY:-32}"
MAX_HTTP_WAITING_REQUESTS="${GR_MAX_HTTP_WAITING_REQUESTS:-64}"
MAX_HTTP_TIMEOUT_TICKS="${GR_MAX_HTTP_TIMEOUT_TICKS:-32}"
MAX_FINISHED_REQUESTS="${GR_MAX_FINISHED_REQUESTS:-1024}"
WORKER_TICK_INTERVAL_S="${GR_WORKER_TICK_INTERVAL_S:-0.001}"
WORKER_IDLE_SLEEP_S="${GR_WORKER_IDLE_SLEEP_S:-0.005}"
DECODE_LOG_INTERVAL="${GR_DECODE_LOG_INTERVAL:-0}"
DECODE_CUDA_GRAPH_BATCH_BUCKETS="${GR_DECODE_CUDA_GRAPH_BATCH_BUCKETS:-1,2,4,8}"

args=(
  "${REPO_ROOT}/tools/serve_qwen3_gr_http.py"
  --host "${HOST}"
  --port "${PORT}"
  --context-len "${CONTEXT_LEN}"
  --decode-steps "${DECODE_STEPS}"
  --beam-width "${BEAM_WIDTH}"
  --max-batch-size "${MAX_BATCH_SIZE}"
  --decode-backend "${DECODE_BACKEND}"
  --device "${DEVICE}"
  --beam-kv-pool-capacity "${BEAM_KV_POOL_CAPACITY}"
  --context-kv-pool-capacity "${CONTEXT_KV_POOL_CAPACITY}"
  --max-http-request-bytes "${MAX_HTTP_REQUEST_BYTES}"
  --max-http-submit-many "${MAX_HTTP_SUBMIT_MANY}"
  --max-http-waiting-requests "${MAX_HTTP_WAITING_REQUESTS}"
  --max-http-timeout-ticks "${MAX_HTTP_TIMEOUT_TICKS}"
  --max-finished-requests "${MAX_FINISHED_REQUESTS}"
  --worker-tick-interval-s "${WORKER_TICK_INTERVAL_S}"
  --worker-idle-sleep-s "${WORKER_IDLE_SLEEP_S}"
  --decode-log-interval "${DECODE_LOG_INTERVAL}"
  --decode-cuda-graph-batch-buckets "${DECODE_CUDA_GRAPH_BATCH_BUCKETS}"
)

if [[ -n "${MODEL_DIR}" ]]; then
  args+=(--model-dir "${MODEL_DIR}")
fi
if [[ -n "${MODEL}" ]]; then
  args+=(--model "${MODEL}")
fi
if [[ -n "${MODEL_REVISION}" ]]; then
  args+=(--revision "${MODEL_REVISION}")
fi

gr_append_option_if_env_set args GR_CATALOG_JSONL --catalog-jsonl
gr_append_option_if_env_set args GR_CATALOG_VOCAB_SIZE --catalog-vocab-size
gr_append_option_if_env_set args GR_CATALOG_EOS_TOKEN_ID --catalog-eos-token-id
gr_append_gr_beam_detail_args args
gr_append_option_if_env_set args GR_SUPPRESS_TOKEN_IDS --suppress-token-ids
if [[ "${GR_SUPPRESS_SPECIAL_TOKENS_ON_IGNORE_EOS:-1}" == "0" ]]; then
  args+=(--no-suppress-special-tokens-on-ignore-eos)
fi
gr_append_flag_if_env_one args GR_PROFILE_CONTINUOUS_DECODE --profile-continuous-decode
gr_append_gr_prefill_cache_args args
gr_append_option_if_env_set args GR_PREFILL_CACHE_MAX_ENTRIES --prefill-cache-max-entries
gr_append_option_if_env_set args GR_PREFILL_CACHE_MAX_TOKENS --prefill-cache-max-tokens
gr_append_option_if_env_set args GR_PREFILL_CACHE_PAGE_SIZE --prefill-cache-page-size
gr_append_option_if_env_set args GR_PREFILL_CACHE_MIN_PREFIX_TOKENS --prefill-cache-min-prefix-tokens
gr_append_option_if_env_set args GR_PREFILL_CACHE_MAX_DECODE_EXTEND_TOKENS --prefill-cache-max-decode-extend-tokens
gr_append_flag_if_env_one args GR_ALLOW_MANUAL_TICK --allow-manual-tick
gr_append_flag_if_env_one args GR_ALLOW_CATALOG_RELOAD --allow-catalog-reload
gr_append_flag_if_env_one args GR_DISABLE_BACKGROUND_WORKER --disable-background-worker
if [[ "${GR_WARMUP_ONLINE_SHAPES:-1}" == "0" ]]; then
  args+=(--no-warmup-online-shapes)
fi
if [[ "${GR_WARMUP_ONLINE_POOL_WINDOWS:-1}" == "0" ]]; then
  args+=(--no-warmup-online-pool-windows)
fi
gr_append_option_if_env_set args GR_WARMUP_ONLINE_MAX_CASES --warmup-online-max-cases
if [[ "${GR_FREEZE_CUDA_GRAPHS_AFTER_WARMUP:-1}" == "0" ]]; then
  args+=(--no-freeze-cuda-graphs-after-warmup)
fi
gr_append_option_if_env_set args GR_HTTP_API_KEY --api-key
gr_append_flag_if_env_one args GR_ENABLE_LOG_REQUESTS --enable-log-requests

exec "${PYTHON_BIN}" "${args[@]}" "$@"
