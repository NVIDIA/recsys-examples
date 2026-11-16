#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Builds the CPU-only inference operators and checks them on CPU. Run in an
# environment with CMake, a C++ compiler and a CPU PyTorch, and no CUDA toolkit.
set -euo pipefail

dynamicemb_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:?Usage: test_cpu_only_build.sh BUILD_DIR}"

torch_prefix="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S "${dynamicemb_root}" -B "${build_dir}" -DDYNAMICEMB_CPU_ONLY=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${torch_prefix}"
cmake --build "${build_dir}" --target inference_emb_ops --parallel
library="${build_dir}/inference_emb_ops.so"

if readelf -d "${library}" | grep -E 'NEEDED.*(cuda|python)'; then
  echo "CPU-only inference_emb_ops.so must not depend on CUDA or Python" >&2
  exit 1
fi

python3 - "${library}" <<'EOF'
import sys

import torch

torch.ops.load_library(sys.argv[1])
ops = torch.ops.INFERENCE_EMB

capacities = torch.tensor([4, 4])
storage, offsets = ops.create_cpu_hash(capacities, 16)
storage, offsets, rows, _ = ops.build_cpu_hash(
    storage, offsets, 16, torch.tensor([10, 20, 30]), torch.tensor([0, 0, 1]),
    capacities, torch.zeros(2, dtype=torch.long), 0)
assert rows.tolist() == [0, 1, 2], rows
found = ops.linear_hash_lookup(
    storage, offsets, 16, torch.tensor([-1, -1]),
    torch.tensor([20, 10, 99, 30]), torch.tensor([0, 0, 0, 1]))
assert found.tolist() == [1, 0, -1, 2], found

table_ids = ops.expand_table_ids(torch.tensor([0, 2, 3, 5]), torch.arange(5), None, 0, 1)
assert table_ids.tolist() == [0, 0, 1, 2, 2], table_ids
for name in ("embedding_collection_index", "register_embedding_collection_indexer"):
    assert hasattr(ops, name), name

class IndexGraph(torch.nn.Module):
    def forward(self, marker, keys, offsets):
        tables = ops.expand_table_ids(offsets, keys, None, 0, 1)
        return ops.embedding_collection_index(marker, 1, keys, tables)

# Both operators need shape inference even when only their CPU kernels are built.
torch.export.export(IndexGraph(), (
    torch.tensor([1]), torch.tensor([10, 20]), torch.tensor([0, 2])))
print("CPU-only inference operators and export OK")
EOF
