#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "${REPO_ROOT}/.." && pwd)}"

IMAGE_NAME="${IMAGE_NAME:-lmsysorg/sglang:dev-cu13}"
CONTAINER_NAME="${CONTAINER_NAME:-sid-gr-inference-bench}"
DOCKER_GPUS="${DOCKER_GPUS:-all}"
DOCKER_RUNTIME="${DOCKER_RUNTIME:-nvidia}"
DOCKER_USER="${DOCKER_USER:-$(id -u):$(id -g)}"
SHM_SIZE="${SHM_SIZE:-32g}"
AUTO_INIT_SUBMODULES="${AUTO_INIT_SUBMODULES:-1}"
AUTO_CLONE_SGLANG="${AUTO_CLONE_SGLANG:-1}"
SGLANG_REPO_URL="${SGLANG_REPO_URL:-https://github.com/cswuyg/sglang.git}"
SGLANG_BRANCH="${SGLANG_BRANCH:-feature/beam_search}"

MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
CACHE_DIR="${CACHE_DIR:-${WORKSPACE_DIR}/.cache}"
SGLANG_REPO_DIR="${SGLANG_REPO_DIR:-${WORKSPACE_DIR}/sglang_beam_search}"
CORELIB_GR_DECODE_ATTEN_DIR="${CORELIB_GR_DECODE_ATTEN_DIR:-$(realpath ${WORKSPACE_DIR}/../corelib/gr_decode_atten)}"

CONTAINER_REPO_DIR="${CONTAINER_REPO_DIR:-/workspace/sid-gr-inference}"
CONTAINER_GR_DECODE_ATTEN_DIR="${CONTAINER_REPO_DIR}/third_party/gr-decode-attention"
CONTAINER_MODEL_ROOT="${CONTAINER_MODEL_ROOT:-/workspace/models}"
CONTAINER_CACHE_DIR="${CONTAINER_CACHE_DIR:-/workspace/.cache}"
CONTAINER_SGLANG_REPO="${CONTAINER_SGLANG_REPO:-/workspace/sglang_beam_search}"
CONTAINER_HOME="${CONTAINER_HOME:-${CONTAINER_CACHE_DIR}/home}"
CONTAINER_PYTHONUSERBASE="${CONTAINER_PYTHONUSERBASE:-${CONTAINER_CACHE_DIR}/python_user}"
CONTAINER_USER_NAME="${CONTAINER_USER_NAME:-gruser}"

usage() {
  cat <<'EOF'
Usage: scripts/run_container.sh [command...]

Start an interactive CUDA container for SID-GR Inference benchmarks.

Common overrides:
  IMAGE_NAME=...              Base image. Default: lmsysorg/sglang:dev-cu13
  SGLANG_REPO_DIR=...         Host SGLang checkout mounted into the container
  AUTO_CLONE_SGLANG=0         Do not clone SGLang automatically when missing
  MODEL_ROOT=...              Host model directory mounted as /workspace/models
  CACHE_DIR=...               Host cache directory mounted as /workspace/.cache
  DOCKER_GPUS=all|device=0    GPU selector passed to docker --gpus
  DOCKER_USER=uid:gid         Container user. Default: current host uid:gid

Examples:
  scripts/run_container.sh
  scripts/run_container.sh scripts/quickstart_offline.sh
  MODEL=Qwen/Qwen3-1.7B scripts/run_container.sh scripts/quickstart_offline.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

mkdir -p "${MODEL_ROOT}" "${CACHE_DIR}/home" "${CACHE_DIR}/python_user" "${CACHE_DIR}/pip"

if [[ ! -d "${REPO_ROOT}/third_party/gr-decode-attention" ]] || \
   [[ -z "$(find "${REPO_ROOT}/third_party/gr-decode-attention" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]]; then
  if [[ "${AUTO_INIT_SUBMODULES}" == "1" ]] && git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "== initializing git submodules =="
    git -C "${REPO_ROOT}" submodule update --init --recursive
  else
    echo "WARNING: third_party/gr-decode-attention is missing or empty." >&2
    echo "Run: git submodule update --init --recursive" >&2
  fi
fi

if [[ ! -d "${SGLANG_REPO_DIR}/python/sglang" ]]; then
  if [[ "${AUTO_CLONE_SGLANG}" == "1" ]]; then
    if [[ -d "${SGLANG_REPO_DIR}" ]] && [[ -n "$(find "${SGLANG_REPO_DIR}" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]]; then
      echo "ERROR: SGLang checkout not found, but ${SGLANG_REPO_DIR} already exists and is not empty." >&2
      echo "Set SGLANG_REPO_DIR to a valid SGLang checkout or remove the invalid directory." >&2
      exit 1
    fi
    echo "== cloning SGLang beam-search branch =="
    mkdir -p "$(dirname "${SGLANG_REPO_DIR}")"
    git clone --depth 1 --branch "${SGLANG_BRANCH}" "${SGLANG_REPO_URL}" "${SGLANG_REPO_DIR}"
  else
    echo "WARNING: SGLang checkout not found at ${SGLANG_REPO_DIR}." >&2
    echo "GR-vs-SGLang benchmark scripts require SGLANG_REPO_DIR to point to the beam-search SGLang checkout." >&2
  fi
fi

append_env_if_set() {
  local name="$1"
  if [[ -n "${!name:-}" ]]; then
    docker_args+=(-e "${name}=${!name}")
  fi
}

docker_args=(
  run
  -it
  --rm
  --gpus "${DOCKER_GPUS}"
  --net=host
  --ipc=host
  --shm-size="${SHM_SIZE}"
  --ulimit memlock=-1:-1
  --ulimit stack=67108864
  --name "${CONTAINER_NAME}"
  --user "${DOCKER_USER}"
  -e "SGLANG_REPO=${CONTAINER_SGLANG_REPO}"
  -e "MODEL_ROOT=${CONTAINER_MODEL_ROOT}"
  -e "GR_DECODE_ATTEN_ROOT=${CONTAINER_GR_DECODE_ATTEN_DIR}"
  -e "HOME=${CONTAINER_HOME}"
  -e "USER=${CONTAINER_USER_NAME}"
  -e "LOGNAME=${CONTAINER_USER_NAME}"
  -e "USERNAME=${CONTAINER_USER_NAME}"
  -e "PYTHONUSERBASE=${CONTAINER_PYTHONUSERBASE}"
  -e "PIP_CACHE_DIR=${CONTAINER_CACHE_DIR}/pip"
  -e "HF_HOME=${CONTAINER_CACHE_DIR}/huggingface"
  -e "GR_CACHE_ROOT=${CONTAINER_CACHE_DIR}/sid-gr-inference"
  -e "PATH=${CONTAINER_PYTHONUSERBASE}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  -v "${REPO_ROOT}:${CONTAINER_REPO_DIR}"
  -v "${MODEL_ROOT}:${CONTAINER_MODEL_ROOT}"
  -v "${CACHE_DIR}:${CONTAINER_CACHE_DIR}"
  -v "${CORELIB_GR_DECODE_ATTEN_DIR}:${CONTAINER_GR_DECODE_ATTEN_DIR}"
  -w "${CONTAINER_REPO_DIR}"
)

for env_name in \
  MODEL \
  MODEL_DIR \
  MODEL_DOWNLOAD_DIR \
  MODEL_REVISION \
  GR_MODEL_DIR \
  GR_MODEL_DOWNLOAD_DIR \
  GR_MODEL_REVISION \
  MODEL_VARIANT \
  RUN_ID \
  OUT_DIR \
  RUN_ACCURACY \
  SKIP_BOOTSTRAP \
  CONTEXT_LENS \
  BEAM_WIDTHS \
  BATCH_SIZES \
  REPEAT \
  CORRECTNESS_REPEAT \
  QUICK_CONTEXT_LENS \
  QUICK_BEAM_WIDTHS \
  QUICK_BATCH_SIZES \
  QUICK_REPEAT \
  QUICK_CORRECTNESS_REPEAT \
  GR_DECODE_STEPS \
  SGLANG_DECODE_STEPS \
  WARMUP_RUNS \
  HF_TOKEN \
  HUGGING_FACE_HUB_TOKEN
do
  append_env_if_set "${env_name}"
done

if [[ -n "${DOCKER_RUNTIME}" ]]; then
  docker_args+=(--runtime "${DOCKER_RUNTIME}")
fi

if [[ -d "${SGLANG_REPO_DIR}" ]]; then
  docker_args+=(-v "${SGLANG_REPO_DIR}:${CONTAINER_SGLANG_REPO}")
fi

docker_args+=("${IMAGE_NAME}")
if [[ $# -gt 0 ]]; then
  docker_args+=("$@")
else
  docker_args+=(bash)
fi

exec docker "${docker_args[@]}"
