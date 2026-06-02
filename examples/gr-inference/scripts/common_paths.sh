#!/usr/bin/env bash
# Shared path defaults for local containers and benchmark scripts.

if [[ -z "${GR_INFERENCE_SCRIPT_DIR:-}" ]]; then
  GR_INFERENCE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
GR_INFERENCE_REPO_ROOT="$(cd "${GR_INFERENCE_SCRIPT_DIR}/.." && pwd)"
GR_INFERENCE_WORKSPACE_ROOT="$(cd "${GR_INFERENCE_REPO_ROOT}/.." && pwd)"

gr_first_existing_dir() {
  local first="$1"
  shift || true
  local path
  for path in "${first}" "$@"; do
    if [[ -d "${path}" ]]; then
      printf '%s\n' "${path}"
      return 0
    fi
  done
  printf '%s\n' "${first}"
}

gr_default_model_dir() {
  local variant="$1"
  gr_first_existing_dir \
    "${GR_INFERENCE_REPO_ROOT}/models/${variant}" \
    "${GR_INFERENCE_WORKSPACE_ROOT}/models/${variant}"
}

gr_default_hf_model_id() {
  local variant="$1"
  case "${variant}" in
    Qwen3-0.6B|qwen3-0.6b|qwen3-0_6b|0.6B|0.6b)
      printf '%s\n' "Qwen/Qwen3-0.6B"
      ;;
    Qwen3-1.7B|qwen3-1.7b|qwen3-1_7b|1.7B|1.7b)
      printf '%s\n' "Qwen/Qwen3-1.7B"
      ;;
    *)
      printf '%s\n' "${variant}"
      ;;
  esac
}

gr_resolve_model_dir() {
  local model_dir="$1"
  local model="$2"
  local model_revision="$3"
  local default_model="$4"
  local download_dir="${5:-}"
  PYTHONPATH="${GR_INFERENCE_REPO_ROOT}/src:${PYTHONPATH:-}" python - \
    "${model_dir}" "${model}" "${model_revision}" "${default_model}" "${download_dir}" <<'PY'
import sys

from gr_inference.gr_models import resolve_model_dir

model_dir, model, revision, default_model, download_dir = (
    arg or None for arg in sys.argv[1:6]
)
print(
    resolve_model_dir(
        model_dir=model_dir,
        model=model,
        revision=revision,
        default_model=default_model,
        download_dir=download_dir,
    )
)
PY
}

gr_model_download_dir() {
  local model_ref="$1"
  local model_root="$2"
  local fallback_name="$3"
  python - "${model_ref}" "${model_root}" "${fallback_name}" <<'PY'
import re
import sys
from pathlib import Path

model_ref, model_root, fallback_name = sys.argv[1:4]
name = (model_ref.rstrip("/").split("/")[-1] if model_ref else fallback_name).strip()
name = re.sub(r"[^A-Za-z0-9._-]+", "_", name) or fallback_name
print(Path(model_root).expanduser() / name)
PY
}

gr_default_sglang_repo() {
  gr_first_existing_dir \
    "${GR_INFERENCE_WORKSPACE_ROOT}/sglang_beam_search"
}

gr_default_sglang_python() {
  printf '%s\n' "python"
}

gr_default_decode_atten_root() {
  gr_first_existing_dir \
    "${GR_INFERENCE_REPO_ROOT}/third_party/gr-decode-attention" \
    "${GR_INFERENCE_WORKSPACE_ROOT}/gr-decode-attention"
}

gr_default_cache_root() {
  printf '%s\n' "${GR_INFERENCE_REPO_ROOT}/.cache"
}

gr_append_flag_if_one() {
  local array_name="$1"
  local value="$2"
  local flag="$3"
  local -n args_ref="${array_name}"
  if [[ "${value}" == "1" ]]; then
    args_ref+=("${flag}")
  fi
}

gr_append_flag_if_env_one() {
  local array_name="$1"
  local env_name="$2"
  local flag="$3"
  gr_append_flag_if_one "${array_name}" "${!env_name:-0}" "${flag}"
}

gr_append_option_if_env_set() {
  local array_name="$1"
  local env_name="$2"
  local option="$3"
  local -n args_ref="${array_name}"
  local value="${!env_name:-}"
  if [[ -n "${value}" ]]; then
    args_ref+=("${option}" "${value}")
  fi
}

gr_append_sglang_cache_args() {
  gr_append_flag_if_env_one "$1" SGLANG_DISABLE_RADIX_CACHE --disable-radix-cache
}

gr_append_sglang_graph_args() {
  gr_append_flag_if_env_one "$1" SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH --disable-piecewise-cuda-graph
  gr_append_flag_if_env_one "$1" SGLANG_DISABLE_CUDA_GRAPH --disable-cuda-graph
}

gr_append_gr_prefill_cache_args() {
  gr_append_flag_if_env_one "$1" GR_ENABLE_PREFILL_CACHE --enable-prefill-cache
}

gr_append_gr_beam_detail_args() {
  local array_name="$1"
  shift
  local -n args_ref="${array_name}"
  if [[ "${GR_RETURN_BEAM_DETAILS:-0}" == "1" ]]; then
    args_ref+=(--return-beam-details "$@")
  fi
}

gr_require_sglang_repo() {
  local sglang_repo="$1"
  local show_clone_hint="${2:-0}"
  if [[ -d "${sglang_repo}/python/sglang" ]]; then
    return 0
  fi
  echo "SGLANG_REPO does not look like a SGLang checkout: ${sglang_repo}" >&2
  if [[ "${show_clone_hint}" == "1" ]]; then
    echo "Clone first, for example:" >&2
    echo "  git clone --depth 1 --branch feature/beam_search https://github.com/cswuyg/sglang.git ${sglang_repo}" >&2
  fi
  return 2
}

gr_beam_sampling_extra_request_body() {
  local decode_steps="$1"
  local beam_width="$2"
  python - "${decode_steps}" "${beam_width}" <<'PY'
import json
import sys

print(json.dumps({
    "sampling_params": {
        "temperature": 0.0,
        "max_new_tokens": int(sys.argv[1]),
        "ignore_eos": True,
        "n": int(sys.argv[2]),
    }
}))
PY
}

gr_setup_local_cache_env() {
  local cache_root="${GR_CACHE_ROOT:-$(gr_default_cache_root)}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}/xdg}"
  export HF_HOME="${HF_HOME:-${cache_root}/huggingface}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-${cache_root}/sglang}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${cache_root}/torchinductor}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${cache_root}/torch_extensions}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${cache_root}/triton}"
  export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-${cache_root}/tvm-ffi}"
  export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${cache_root}/flashinfer_workspace}"
  export TRTLLM_DG_CACHE_DIR="${TRTLLM_DG_CACHE_DIR:-${cache_root}/trtllm_dg}"
  export CUTE_DSL_CACHE_DIR="${CUTE_DSL_CACHE_DIR:-${cache_root}/cute_dsl}"
  mkdir -p \
    "${XDG_CACHE_HOME}" \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${SGLANG_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${TRITON_CACHE_DIR}" \
    "${TVM_FFI_CACHE_DIR}" \
    "${FLASHINFER_WORKSPACE_BASE}" \
    "${TRTLLM_DG_CACHE_DIR}" \
    "${CUTE_DSL_CACHE_DIR}"
}
