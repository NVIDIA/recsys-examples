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

// The payload of a chunk of a proxy's stream, the rows it sends over IB, one
// per key (Sizes::rows_per_payload): at least about Config::chunk_kb KiB, by
// default kChunkBytes. Python reads chunk_kb from DYNAMICEMB_GIN_CHUNK_KB when
// it builds the sharding, and the ranks compare it with the other arguments,
// so every rank has the same value. Config::validate refuses a value below
// kMinChunkKb or not a multiple of 16. A chunk's rows in the ring, a row per
// occurrence of its keys, stop at kChunkRowCapBytes, or at the payload if it
// is larger (Sizes::chunk_rows). A chunk of keys of one row each sends about
// ring_rows / rows_per_payload rows over IB (sizes_of). When the ring holds two
// chunks at the cap, that is chunk_kb rounded up to 2 MiB / floor(2 MiB /
// chunk_kb), and chunk_kb from 2048 KiB. It is chunk_kb at 64, 128, 256, 512
// and 1024 KiB. Other values give up to about twice chunk_kb; above 512 KiB,
// 528..672 KiB give 682 KiB, 688..1008 KiB give 1 MiB, and 1040..2032 KiB
// give 2 MiB.
constexpr int64_t kChunkBytes = int64_t{512} << 10;
constexpr int64_t kChunkRowCapBytes = int64_t{2} << 20;
constexpr int64_t kMinChunkKb = 64;

struct Config {
  int rank = 0, world_size = 0, local_size = 0;  // ranks are node-contiguous
  std::vector<int> feature_tables;               // the table of each feature: across nodes a key is
                                                 // its table and value
  std::vector<int> dist_types;  // [num_features]: each feature's DistType code (src/utils.h)
  int64_t dim = 0;              // a row of the lookup output: dim values of dtype
  at::ScalarType dtype = at::kFloat;
  bool weights = false;    // the requesters' keys carry weights, and so the owner's KJT:
                           // whole counts of at least 0, which the proxy adds as integers
  bool node_dedup = true;  // across nodes a node sends a key once (else each
                           // occurrence)
  Capacities capacities;
  double timeout_s = 120;  // the longest a kernel waits for a peer before it traps
  int ctas = 0;            // CTAs of an out kernel (0: its default)
  int reduce_ctas = 8;     // CTAs of the backward across nodes that reduce
  // across nodes, the least payload (about) of a chunk of a proxy's stream, in KiB
  int64_t chunk_kb = kChunkBytes >> 10;

  int num_features() const { return (int)feature_tables.size(); }
  int num_nodes() const { return world_size / local_size; }
  int64_t row_bytes() const { return dim * (int64_t)c10::elementSize(dtype); }
  // Throws std::invalid_argument (Python: ValueError) for a configuration the
  // op cannot take. It builds nothing, so the ranks can compare their
  // configurations and answers before any of them builds the op. It also
  // refuses a rule other than roundrobin and hash_roundrobin, and a table
  // whose features have different rules (check_rules).
  void validate() const;
};

// A configuration of these fields (Config's), as Python gives them.
Config config_of(int64_t rank, int64_t world_size, int64_t local_size,
                 const std::vector<int64_t> &feature_tables, const std::vector<int64_t> &dist_types,
                 int64_t dim, at::ScalarType dtype, bool weights, bool node_dedup,
                 int64_t lookup_keys, int64_t remote_keys, int64_t proxy_bytes, double timeout_s,
                 int64_t ctas, int64_t reduce_ctas, int64_t chunk_kb);

// Why the op does not apply to a sharding of these ranks, features and rows
// (ranks that are not whole nodes, int32 bucket feature numbers, rows the
// kernels cannot move), or "" if it does.
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

// The sizing rules of the sizes that follow from the configuration
// (sizes_of's, and Config::validate's), across nodes:
//   num_proxy_rows    min(max(remote_keys, min_proxy_rows_for(max_key_rows)),
//                     proxy_bytes / row_bytes)
//   payload_rows      max(payload / row_bytes, kTaskRows), the payload chunk_kb KiB
//   ring_rows         min(max(max(payload, kChunkRowCapBytes) / row_bytes, kTaskRows),
//                     num_proxy_rows / 2)
//   chunk_rows        max(ring_rows, 2 * max_key_rows)
//   rows_per_payload  rows_per_payload_for(ring_rows, payload_rows)
//   max_chunks        rows_per_payload * remote_keys / chunk_span + num_nodes
// rows_per_payload of chunks of at most `ring_rows` rows in the ring and a
// payload of `payload_rows` rows: max(ring_rows / payload_rows, 1). It is 1
// when the payload is at least kChunkRowCapBytes, and a key's counted rows are
// then its rows.
int64_t rows_per_payload_for(int64_t ring_rows, int64_t payload_rows);
// max_chunks of sizes whose other fields are set: a proxy's keys count at most
// rows_per_payload times their rows, so its stream holds at most
// rows_per_payload * remote_keys / chunk_span chunks, and one more per owner
// node.
int64_t max_chunks_for(const Sizes &sizes);

// The table of each feature, numbered from 0 in the order the tables first
// appear: equal tables keep equal numbers.
std::vector<int> dense_tables(const std::vector<int> &feature_tables);

}  // namespace hier_output_dist
}  // namespace dyn_emb
