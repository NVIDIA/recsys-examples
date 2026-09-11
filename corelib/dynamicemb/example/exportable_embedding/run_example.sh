#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dynamicemb_root="$(cd "${example_dir}/../.." && pwd)"
nve_version="${NVE_VERSION:-26.07}"

if [[ "${nve_version}" == "26.05" ]]; then
  nve_root="/opt/nve/26.05/python"
else
  nve_root="/opt/nve/default/python"
fi

export DYNAMICEMB_OPS_LIB_DIR="${dynamicemb_root}/torch_binding_build"
export PYTHONPATH="${nve_root}:${dynamicemb_root}:${PYTHONPATH:-}"

redis_disabled=0
for argument in "$@"; do
  if [[ "${argument}" == "--disable-redis-incremental" ]]; then
    redis_disabled=1
  fi
done

redis_pid=""
redis_runtime=""
cleanup_redis() {
  if [[ -n "${redis_pid}" ]]; then
    kill "${redis_pid}" 2>/dev/null || :
    wait "${redis_pid}" 2>/dev/null || :
  fi
  if [[ -n "${redis_runtime}" && -d "${redis_runtime}" ]]; then
    rm -r -- "${redis_runtime}"
  fi
}
trap cleanup_redis EXIT

if [[ "${nve_version}" != "26.05" \
  && "${redis_disabled}" == "0" \
  && "${START_LOCAL_REDIS:-1}" == "1" ]]; then
  redis_port="${REDIS_PORT:-6379}"
  redis_runtime="$(mktemp -d "${TMPDIR:-/tmp}/exportable-embedding-redis.XXXXXX")"
  redis-server \
    --bind 127.0.0.1 \
    --protected-mode yes \
    --port "${redis_port}" \
    --save "" \
    --appendonly no \
    --dir "${redis_runtime}" \
    --logfile "${redis_runtime}/redis.log" &
  redis_pid=$!

  redis_ready=0
  for _ in {1..100}; do
    if redis-cli -h 127.0.0.1 -p "${redis_port}" ping 2>/dev/null \
      | grep -q '^PONG$'; then
      redis_ready=1
      break
    fi
    if ! kill -0 "${redis_pid}" 2>/dev/null; then
      break
    fi
    sleep 0.05
  done
  if [[ "${redis_ready}" != "1" ]]; then
    echo "local Redis failed to start" >&2
    exit 1
  fi
  set -- "$@" --redis-address "127.0.0.1:${redis_port}"
fi

python3 "${example_dir}/export_and_verify.py" "$@"
