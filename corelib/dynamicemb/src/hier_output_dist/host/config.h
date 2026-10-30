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
// What an op is built from (Config), its capacities among them, read once:
// the op never changes them.
#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <string>
#include <vector>

namespace dyn_emb {
namespace hier_output_dist {

// The planner numbers the entries of a capacity with int32 indices, so each
// stays below this.
constexpr int64_t kMaxEntries = (int64_t{1} << 31) - 1024;

// How much a call may hold, in entries; a call that needs more of one runs
// on TorchRec's dists (the caller's fallback). On one node only lookup_keys
// is used.
struct Capacities {
  int64_t lookup_keys = 0;  // keys an owner looks up (rows of lookup_rows)
  int64_t remote_keys = 0;  // keys of other nodes' owners a proxy deduplicates for its node
  int64_t proxy_bytes = 0;  // the most bytes of the proxy rows region (proxy_rows)
};

struct Config {
  int rank = 0, world_size = 0, local_size = 0;  // ranks are node-contiguous
  std::vector<int> feature_tables;               // the table of each feature: across nodes a key is
                                                 // its owner node, table and value
  at::Tensor block_sizes;  // [num_features] int64 on the GPU: the bucketize's rule
  at::Tensor dist_types;   // [num_features] int32 on the GPU: of each feature
  int64_t dim = 0;         // a row of the lookup output: dim values of dtype
  at::ScalarType dtype = at::kFloat;
  bool weights = false;      // the requesters' keys carry weights: whole counts of at
                             // least 0, which the proxy adds as integers
  bool out_weights = false;  // the owner's KJT carries weights: the keys', or across
                             // nodes each node-unique key's occurrences
  bool node_dedup = true;    // across nodes a node sends a key once (else each
                             // occurrence)
  Capacities capacities;
  double timeout_s = 120;  // the longest a kernel waits for a peer before it traps
  int ctas = 0;            // CTAs of an out kernel (0: its default)
  int reduce_ctas = 8;     // CTAs of the backward across nodes that reduce

  int num_features() const { return (int)feature_tables.size(); }
  int num_nodes() const { return world_size / local_size; }
  int64_t row_bytes() const { return dim * (int64_t)c10::elementSize(dtype); }
  // Throws std::invalid_argument (Python: ValueError) for a configuration the
  // op cannot take. It builds nothing, so the ranks can compare their
  // configurations and answers before any of them builds the op.
  void validate() const;
};

// Why the op does not apply to a sharding of these ranks, features and rows
// (static limits of its kernels), or "" if it does.
std::string unsupported(int world_size, int local_size, int num_features, int64_t row_bytes);

}  // namespace hier_output_dist
}  // namespace dyn_emb
