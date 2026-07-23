#!/usr/bin/env bash

# Launch and test the Triton Python backend in the built container.
# Run this script with `bash`, not `source`.
# The Triton server and both clients run in this same container. The script
# restores the model repository and Gin config after success or failure.
set -Eeuo pipefail

HSTU_DIR="${HSTU_DIR:-/workspace/recsys-examples/examples/hstu}"
CKPT_DIR="${1:-${HSTU_DIR}/ckpt/kuairand_1k_ckpt}"
SERVER_LOG="${SERVER_LOG:-/tmp/hstu-python-backend-tritonserver.log}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-300}"

GIN_CONFIG="${HSTU_DIR}/inference/configs/kuairand_1k_inference_ranking.gin"
MODEL_REPOSITORY="${HSTU_DIR}/inference/triton"
DENSE_MODEL_DIR="${MODEL_REPOSITORY}/hstu_model"
SPARSE_MODEL_DIR="${MODEL_REPOSITORY}/hstu_sparse"
DENSE_CONFIG="${DENSE_MODEL_DIR}/config.pbtxt"
SPARSE_CONFIG="${SPARSE_MODEL_DIR}/config.pbtxt"
CLIENT="${DENSE_MODEL_DIR}/client.py"
PS_MODULE_DIR="${CKPT_DIR}/ps_module"
DYNAMIC_MODULE_DIR="${CKPT_DIR}/dynamicemb_module"

SERVER_PID=
STATE_DIR=

usage() {
  cat <<EOF
Usage: $(basename "$0") [CHECKPOINT_DIR]

Launch Triton and run the KuaiRand-1K evaluation and training-data clients.

Defaults:
  HSTU root:      /workspace/recsys-examples/examples/hstu
  Checkpoint:     \$HSTU_DIR/ckpt/kuairand_1k_ckpt
  Dataset:        \$HSTU_DIR/tmp_data (unless DatasetArgs.dataset_path is set)

Overrides:
  HSTU_DIR=/path/to/examples/hstu $(basename "$0") /path/to/checkpoint
EOF
}

require_command() {
  command -v "$1" >/dev/null || {
    echo "ERROR: required command '$1' was not found in PATH." >&2
    return 1
  }
}

require_directory() {
  local path="$1"
  local setup_hint="$2"

  [[ -d "${path}" ]] || {
    echo "ERROR: required directory does not exist: ${path}" >&2
    echo "Setup: ${setup_hint}" >&2
    return 1
  }
}

require_file() {
  local path="$1"

  [[ -f "${path}" ]] || {
    echo "ERROR: required file does not exist: ${path}" >&2
    return 1
  }
}

restore_directory() {
  local path="$1"
  local backup="$2"

  rm -rf -- "${path}"
  if [[ -e "${backup}" ]]; then
    cp -a -- "${backup}" "${path}"
  fi
}

cleanup() {
  local status=$?
  trap - EXIT

  # Cleanup: stop only the server launched here, then restore the input tree.
  if [[ -n "${SERVER_PID}" ]]; then
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "Stopping Triton server PID ${SERVER_PID}"
      kill -TERM "${SERVER_PID}"
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi

  if [[ -n "${STATE_DIR}" ]] && [[ -d "${STATE_DIR}" ]]; then
    cp -a -- "${STATE_DIR}/ranking.gin" "${GIN_CONFIG}"
    cp -a -- "${STATE_DIR}/dense-config.pbtxt" "${DENSE_CONFIG}"
    cp -a -- "${STATE_DIR}/sparse-config.pbtxt" "${SPARSE_CONFIG}"
    restore_directory "${DENSE_MODEL_DIR}/1" "${STATE_DIR}/dense-version-1"
    restore_directory "${SPARSE_MODEL_DIR}/1" "${STATE_DIR}/sparse-version-1"
    restore_directory "${PS_MODULE_DIR}" "${STATE_DIR}/ps_module"
    rm -rf -- "${STATE_DIR}"
  fi

  return "${status}"
}

append_checkpoint_parameter_if_missing() {
  local config_file="$1"

  if ! grep -q 'key: "HSTU_CHECKPOINT_DIR"' "${config_file}"; then
    printf '\nparameters [\n {\n  key: "HSTU_CHECKPOINT_DIR"\n  value: {\n' \
      >> "${config_file}"
    printf '   string_value: "%s"\n  }\n }\n]\n' "${CKPT_DIR}" \
      >> "${config_file}"
  fi
}

main() {
  # Step 1: parse the simple interface and validate the container layout.
  if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    usage
    return 0
  fi
  if (( $# > 1 )); then
    echo "ERROR: expected at most one checkpoint-directory argument." >&2
    usage >&2
    return 2
  fi

  require_directory "${HSTU_DIR}" \
    "clone recsys-examples into the default container path or set HSTU_DIR."
  cd "${HSTU_DIR}"

  require_command tritonserver
  require_command curl
  python3 -c \
    'import tritonclient.http as http; print("Triton HTTP client:", http.__file__)'

  require_directory "${CKPT_DIR}" \
    "mount or copy kuairand_1k_ckpt there, or pass its path as argument 1."
  require_directory "${DYNAMIC_MODULE_DIR}" \
    "use a checkpoint containing the DynamicEmb dynamicemb_module directory."
  require_file "${GIN_CONFIG}"
  require_file "${DENSE_CONFIG}"
  require_file "${SPARSE_CONFIG}"
  require_file "${DENSE_MODEL_DIR}/model.py"
  require_file "${SPARSE_MODEL_DIR}/model.py"
  require_file "${CLIENT}"

  if ! grep -Eq \
    '^[[:space:]]*DatasetArgs\.dataset_path[[:space:]]*=' "${GIN_CONFIG}"; then
    require_directory "${HSTU_DIR}/tmp_data" \
      "run the KuaiRand-1K inference preprocessor or mount data at ${HSTU_DIR}/tmp_data."
  fi

  if curl --silent --fail http://localhost:8000/v2/health/live >/dev/null 2>&1; then
    echo "A Triton server is already listening on localhost:8000."
    return 1
  fi

  # Step 2: save every file or directory that this test changes temporarily.
  STATE_DIR="$(mktemp -d /tmp/hstu-python-backend.XXXXXX)"
  cp -a -- "${GIN_CONFIG}" "${STATE_DIR}/ranking.gin"
  cp -a -- "${DENSE_CONFIG}" "${STATE_DIR}/dense-config.pbtxt"
  cp -a -- "${SPARSE_CONFIG}" "${STATE_DIR}/sparse-config.pbtxt"
  [[ ! -e "${DENSE_MODEL_DIR}/1" ]] || \
    cp -a -- "${DENSE_MODEL_DIR}/1" "${STATE_DIR}/dense-version-1"
  [[ ! -e "${SPARSE_MODEL_DIR}/1" ]] || \
    cp -a -- "${SPARSE_MODEL_DIR}/1" "${STATE_DIR}/sparse-version-1"
  [[ ! -e "${PS_MODULE_DIR}" ]] || \
    cp -a -- "${PS_MODULE_DIR}" "${STATE_DIR}/ps_module"

  # Step 3: enable NVEmbedding and expose DynamicEmb files through ps_module.
  sed -i '/^[[:space:]]*NetworkArgs\.embedding_backend[[:space:]]*=/d' \
    "${GIN_CONFIG}"
  printf '\nNetworkArgs.embedding_backend = "NVEmb"\n' >> "${GIN_CONFIG}"

  rm -rf -- "${PS_MODULE_DIR}"
  mkdir -p "${PS_MODULE_DIR}"
  while IFS= read -r -d '' checkpoint_file; do
    ln -s "$(realpath "${checkpoint_file}")" \
      "${PS_MODULE_DIR}/$(basename "${checkpoint_file}").dyn"
  done < <(find "${DYNAMIC_MODULE_DIR}" -type f \
    -regex '.*/.*_emb_.*' -print0)

  test -r "${PS_MODULE_DIR}/user_id_emb_keys.rank_0.world_size_1.dyn"
  test -r "${PS_MODULE_DIR}/user_id_emb_values.rank_0.world_size_1.dyn"
  test -r "${PS_MODULE_DIR}/video_id_emb_keys.rank_0.world_size_1.dyn"
  test -r "${PS_MODULE_DIR}/video_id_emb_values.rank_0.world_size_1.dyn"

  # Step 4: create Triton model version 1 and point both models at the checkpoint.
  rm -rf -- "${DENSE_MODEL_DIR}/1" "${SPARSE_MODEL_DIR}/1"
  mkdir -p "${DENSE_MODEL_DIR}/1" "${SPARSE_MODEL_DIR}/1"
  cp -- "${DENSE_MODEL_DIR}/model.py" "${DENSE_MODEL_DIR}/1/model.py"
  cp -- "${SPARSE_MODEL_DIR}/model.py" "${SPARSE_MODEL_DIR}/1/model.py"

  # Current configs may already contain /checkpoints/hstu. Older configs have
  # no checkpoint parameter, so retain the launcher's append behavior for them.
  append_checkpoint_parameter_if_missing "${DENSE_CONFIG}"
  append_checkpoint_parameter_if_missing "${SPARSE_CONFIG}"
  sed -i \
    "/key: \"HSTU_CHECKPOINT_DIR\"/,/}/ s#string_value: \"[^\"]*\"#string_value: \"${CKPT_DIR}\"#" \
    "${DENSE_CONFIG}" "${SPARSE_CONFIG}"
  grep -Fq "string_value: \"${CKPT_DIR}\"" "${DENSE_CONFIG}"
  grep -Fq "string_value: \"${CKPT_DIR}\"" "${SPARSE_CONFIG}"

  export HSTU_INFERENCE_ONLY=1
  export PYTHONPATH="${PYTHONPATH:-}:$(realpath "${HSTU_DIR}/..")"
  export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:\
/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib:/usr/local/cuda/compat/lib:\
/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

  # Step 5: launch Triton directly so SERVER_PID identifies the real process.
  echo "Starting Triton server; log: ${SERVER_LOG}"
  tritonserver --model-repository "${MODEL_REPOSITORY}" \
    > "${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  echo "Triton server PID: ${SERVER_PID}"

  # Step 6: wait for server readiness, then verify both HSTU models explicitly.
  local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  until curl --silent --fail http://localhost:8000/v2/health/ready \
    >/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "Triton exited before becoming ready."
      tail -n 200 "${SERVER_LOG}"
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Triton was not ready within ${READY_TIMEOUT_SECONDS} seconds."
      tail -n 200 "${SERVER_LOG}"
      return 1
    fi
    sleep 2
  done

  curl --silent --fail \
    http://localhost:8000/v2/models/hstu_sparse/ready >/dev/null
  curl --silent --fail \
    http://localhost:8000/v2/models/hstu_model/ready >/dev/null
  echo "Triton and both HSTU models are ready."

  # Step 7: run evaluation and training-data tests at batch size 2, three times.
  python3 "${CLIENT}" --gin_config_file "${GIN_CONFIG}"
  python3 "${CLIENT}" --gin_config_file "${GIN_CONFIG}" --train_dataset

  echo "Evaluation-data and training-data Python-backend tests passed."
}

trap cleanup EXIT
main "$@"
