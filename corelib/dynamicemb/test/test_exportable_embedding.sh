#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run inside the application image; requires one GPU and no external dataset.
set -euo pipefail

dynamicemb_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="${1:?Usage: test_exportable_embedding.sh WORK_DIR}"
mkdir -p "${work_dir}"
work_dir="$(realpath "${work_dir}")"
cd "${dynamicemb_root}"
export PYTHONPATH="/opt/nve/default/python:${dynamicemb_root}:${PYTHONPATH:-}"
export DYNAMICEMB_OPS_LIB_DIR="${dynamicemb_root}/torch_binding_build"
unset NVE_VERSION
python3 -c 'import torch; assert torch.cuda.is_available(), "This suite requires one GPU"'

python3 -m pytest -q tests --junitxml="${work_dir}/exportable.xml"
CUDA_VISIBLE_DEVICES= python3 -m pytest -q \
  tests/test_host_embedding.py tests/test_host_incremental_update.py \
  --junitxml="${work_dir}/host-cpu.xml"

# Keep the original Redis workflow, assertions and scheduling intact.
bash example/exportable_embedding/run_example.sh \
  --work-dir "${work_dir}/incremental" \
  --cpp-replayer "${dynamicemb_root}/example/exportable_embedding/cpp/build/replay_and_verify"

cmake -S example/host_embedding -B "${work_dir}/host-cpp" \
  -DNVE_ROOT=/workspace/deps/nve \
  -DNVE_LIB_DIR=/opt/nve/default/python/pynve \
  -DDYNAMICEMB_LIB_DIR="${DYNAMICEMB_OPS_LIB_DIR}"
cmake --build "${work_dir}/host-cpp" --parallel 2
for indexer in identity fused_identity linear_hash_map; do
  for dtype in float32 float16; do
    package="${work_dir}/host-${indexer}-${dtype}"
    CUDA_VISIBLE_DEVICES= python3 example/host_embedding/incremental_cpu.py \
      --indexer "${indexer}" --dtype "${dtype}" --package-dir "${package}"
    CUDA_VISIBLE_DEVICES= "${work_dir}/host-cpp/replay_host" "${package}" -1
  done
done

# The legacy generation supports export/replay, but not incremental updates.
NVE_VERSION=26.05 bash example/exportable_embedding/run_example.sh \
  --work-dir "${work_dir}/legacy-2605" --disable-redis-incremental
