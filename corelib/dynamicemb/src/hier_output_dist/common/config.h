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
// The host side's basics: error checks and small helpers, an op's
// configuration (Config), and its sizes (sizes_of).
#pragma once

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "hier_output_dist/common/common.h"

namespace dyn_emb {
namespace hier_output_dist {

// What every host file of the hierarchical dist shares: the error checks of
// the CUDA runtime, the CUDA driver and NCCL, and small helpers.

#define HIER_NCCL_CHECK(cmd)                                                \
  do {                                                                      \
    ncclResult_t result_ = (cmd);                                           \
    TORCH_CHECK(result_ == ncclSuccess, "hier_output_dist: NCCL failure '", \
                ncclGetErrorString(result_), "' at " #cmd);                 \
  } while (0)
#define HIER_CUDA_CHECK(cmd)                                                                       \
  do {                                                                                             \
    cudaError_t err_ = (cmd);                                                                      \
    TORCH_CHECK(err_ == cudaSuccess, "hier_output_dist: CUDA failure '", cudaGetErrorString(err_), \
                "' at " #cmd);                                                                     \
  } while (0)
#define HIER_CU_CHECK(cmd)                                                                       \
  do {                                                                                           \
    CUresult result_ = (cmd);                                                                    \
    TORCH_CHECK(result_ == CUDA_SUCCESS, "hier_output_dist: CUDA driver failure ", (int)result_, \
                " at " #cmd);                                                                    \
  } while (0)

inline int sms_of(int device) {
  int sms;
  HIER_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  return sms;
}
// CTAs of kThreads threads for `work` items, a thread each, at most two per SM.
inline int grid_for(int64_t work, int sms) {
  return (int)std::min<int64_t>((std::max<int64_t>(work, 1) + kThreads - 1) / kThreads,
                                2 * (int64_t)sms);
}
// Bits that hold the values [0, value).
inline int bits_for(int64_t value) {
  int bits = 1;
  while ((int64_t{1} << bits) < value) ++bits;
  return bits;
}
// Whether `tensor` is a contiguous CUDA tensor of `count` elements of `dtype`.
inline bool is_cuda_array(const at::Tensor &tensor, at::ScalarType dtype, int64_t count) {
  return tensor.is_cuda() && tensor.is_contiguous() && tensor.scalar_type() == dtype &&
         tensor.numel() == count;
}

// What an op is built from (Config), its capacities among them, read once:
// the op never changes them.

// The planner numbers the entries of a capacity with int32 indices, so each
// stays below this.
constexpr int64_t kMaxEntries = (int64_t{1} << 31) - 1024;

// How much a call may hold, in entries; a call that needs more of one is an
// error, which the caller raises at the call's second wait. On one node only
// lookup_keys is used.
struct Capacities {
  int64_t lookup_keys = 0;  // keys an owner looks up (rows of lookup_rows)
  int64_t remote_keys = 0;  // keys of other nodes' owners a proxy deduplicates for its node
  int64_t proxy_bytes = 0;  // the most bytes of proxy_rows, the proxy's rows
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

// A configuration of these fields (Config's), as Python gives them.
Config config_of(int64_t rank, int64_t world_size, int64_t local_size,
                 const std::vector<int64_t> &feature_tables, const at::Tensor &block_sizes,
                 const at::Tensor &dist_types, int64_t dim, at::ScalarType dtype, bool weights,
                 bool out_weights, bool node_dedup, int64_t lookup_keys, int64_t remote_keys,
                 int64_t proxy_bytes, double timeout_s, int64_t ctas, int64_t reduce_ctas);

// Why the op does not apply to a sharding of these ranks, features and rows
// (ranks that are not whole nodes, int32 cell numbers, rows the kernels
// cannot move), or "" if it does.
std::string unsupported(int world_size, int local_size, int num_features, int64_t row_bytes);
// Why the op refuses `config` (Config::validate), or "" if it takes it.
std::string invalid(const Config &config);

// The sizes of an op (Sizes, common.h) from its configuration alone,
// and the rules of the capacities that follow from the configured ones. The
// capacities are fixed, so the sizes are too.

// The sizes of an op of `config`: its ranks, features and rows, and its
// capacities in place: as configured, and across nodes what follows from
// them (0 on one node).
Sizes sizes_of(const Config &config);

// The sizing rules of the capacities that follow from the configured ones
// (sizes_of's, and Config::validate's), across nodes:
//   num_proxy_rows  min(max(remote_keys, the least rows every round needs),
//                   proxy_bytes / row_bytes)
//   rounds          the rounds remote_keys rows take in that many rows
int64_t num_proxy_rows_for(const Config &config);
// rows of an IB chunk, and of proxy_rows a requester's fp32 partial row takes
int chunk_rows_for(int64_t row_bytes);
int partial_rows_for(int64_t dim, int64_t row_bytes);

// The table of each feature, numbered from 0 in the order the tables first
// appear: equal tables keep equal numbers.
std::vector<int> dense_tables(const std::vector<int> &feature_tables);

}  // namespace hier_output_dist
}  // namespace dyn_emb
