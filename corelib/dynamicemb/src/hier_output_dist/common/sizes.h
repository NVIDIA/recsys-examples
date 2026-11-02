/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
******************************************************************************/
// The sizes of an op, from its configuration (host/sizes.h): every buffer of
// its window and every table of its calls is laid out by them, on the host
// and on the device.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hier_output_dist/common/topology.h"

namespace dyn_emb {
namespace hier_output_dist {

// Values that follow from the capacities read 0 on one node.
struct Sizes {
  Topo topo;
  int64_t num_features;
  int64_t num_tables;      // the features' tables, numbered densely (dense_tables)
  int64_t lookup_keys;     // keys an owner looks up (a capacity)
  int64_t remote_keys;     // remote occurrences a proxy deduplicates (a capacity)
  int64_t num_proxy_rows;  // rows of proxy_rows
  int64_t rounds_cap;      // the most rounds of a call, what a round table holds
  int64_t row_bytes;       // an embedding row (lookup_rows, proxy_rows)
  int64_t partial_rows;    // rows of proxy_rows a requester's fp32 partial row takes
  int64_t chunk_rows;      // rows of an IB chunk (2 MiB), a multiple of kTaskRows
  int64_t record_bytes;    // a key record: 8, or 16 with its weight
  bool weights;            // the requesters' keys carry weights

  __host__ __device__ bool multi_node() const { return topo.num_nodes > 1; }
};

// Each buffer of the window starts at a multiple of kBufferAlign: a buffer of
// `bytes` takes buffer_end(bytes).
constexpr uint64_t kBufferAlign = 4096;
__host__ __device__ inline uint64_t buffer_end(uint64_t bytes) {
  return (bytes + kBufferAlign - 1) / kBufferAlign * kBufferAlign;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
