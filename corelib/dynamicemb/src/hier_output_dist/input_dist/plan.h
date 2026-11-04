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
// The output dist's plans of a call, which its input dist makes: what
// fwd_kernel reads of the call (FwdPlan) and what bwd_kernel reads
// (BwdPlan), device tensors, and the same as the kernels read them
// (FwdPlanView, BwdPlanView: pointers). A table both kernels read is the
// same tensor in both plans.
//
// The requester's rows are its bucketized rows [0,
// offset_per_bucket[world_size]), owner by owner (its block for an owner:
// [offset_per_bucket[owner], offset_per_bucket[owner + 1])). The forward reads
// a row at src[row], the backward writes its gradient at dst[row]: for an owner
// of its node both are its row of the owner's lookup_rows; for an owner on
// another node, rows of the proxy's proxy_rows (the node-unique key's row, and
// the row's own slot). inv[row] is its row of the requester's output, or under
// index dedup its deduplicated row, whose original rows the fan-out lists.
// Across nodes the requester's push order, heavy groups and its tables of the
// node's proxies, the owner's and the proxy's tables, and the put lists say the
// rest (planner.h makes them); the tables' strides are max_chunks and
// num_nodes. On one node the tables across nodes are undefined, nullptr in the
// views.
#pragma once

#include <ATen/ATen.h>
#include <c10/core/Stream.h>
#include <c10/util/Optional.h>

#include <cstdint>
#include <vector>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/input_dist/planner.h"

namespace dyn_emb {
namespace hier_output_dist {

// What both kernels read of a call.
struct PlanView {
  const int64_t *offset_per_bucket;  // [world_size + 1]
  const int32_t *inv;                // [rows]
  Fanout fanout;                     // fanout.exp_off: index dedup
  // across nodes, the requester's
  const int32_t *push_order;    // [rows] its remote rows by proxy and push order
  const int64_t *proxy_chunks;  // [local_size] the chunks of each same-node proxy's stream
  const int64_t *proxy_ends;    // [local_size][max_chunks + 1] each one's chunk ends, then its
                                // stream's end
  const ReqEntry *push_tables;  // [local_size][max_chunks + 1] its push order at each proxy,
                                // positions absolute in push_order, groups in heavy_groups
  // across nodes, the proxy's
  const int64_t *num_chunks;  // the chunks of its stream
  const Chunk *chunks;        // [max_chunks] its chunks, in stream order
  const int64_t *uniq_off;    // [num_nodes + 1] its first node-unique key of each owner node
  // across nodes, the owner's forward and the proxy's backward put lists
  // (planner.h): the segments of each node from put_starts[node]
  const PutSeg *put_list;
  const int32_t *put_starts;  // [2 * num_nodes + 1]: the backward's owner nodes, then the
                              // forward's proxy nodes
  int64_t max_chunks, ring;
  int num_nodes;

  // This rank's push order at a same-node proxy in a chunk; the next entry
  // holds the ends.
  __device__ __forceinline__ const ReqEntry *push_entry(int proxy, int64_t chunk) const {
    return push_tables + (int64_t)proxy * (max_chunks + 1) + chunk;
  }
  // The end of a same-node proxy's chunk (chunk max_chunks: its stream's end).
  __device__ __forceinline__ int64_t proxy_end(int proxy, int64_t chunk) const {
    return proxy_ends[(int64_t)proxy * (max_chunks + 1) + chunk];
  }
  // Whether a same-node proxy's stream wraps its ring.
  __device__ __forceinline__ bool wraps(int proxy) const {
    return proxy_end(proxy, max_chunks) > ring;
  }
  // The chunk of a same-node proxy's stream that holds row `at` of its ring,
  // when the stream fits the ring: the first that ends past it.
  __device__ __forceinline__ int64_t chunk_at(int proxy, int64_t at) const {
    const int64_t chunks = proxy_chunks[proxy];
    int64_t lo = 0, hi = chunks;
    while (lo < hi) {
      const int64_t mid = (lo + hi) / 2;
      if (proxy_end(proxy, mid) <= at)
        lo = mid + 1;
      else
        hi = mid;
    }
    return min(lo, max(chunks - 1, (int64_t)0));
  }
};

// What fwd_kernel reads of a call.
struct FwdPlanView : PlanView {
  const int32_t *src;  // [rows]
  // across nodes, the requester's
  const int64_t *push_order_off;   // [local_size + 1] where its rows for each proxy start in
                                   // push_order
  const int64_t *fwd_task_starts;  // [max_chunks * local_size + 1] its tasks of rows of other nodes
                                   // before each (chunk, proxy in turn from its own): 32 positions
                                   // of its push order there each
  // across nodes, the owner's
  const ChunkPiece *fwd_chunks;  // [owner_chunks] the chunks of its node in each source node's
                                 // stream, its forward put list's (PutSeg::chunk)
  // across nodes, the proxy's
  const int64_t *node_chunk_start;  // [num_nodes + 1] where each owner node's chunks start in
                                    // node_chunks
  const int32_t *node_chunks;       // [max_chunks] each owner node's chunks in order
  const int64_t *uniq_out;          // [num_nodes] its node-unique keys of each owner node

  // The owner's forward put list's segments for a proxy node.
  __device__ __forceinline__ int64_t fwd_puts(int proxy_node) const {
    return put_starts[num_nodes + proxy_node];
  }
};

// What bwd_kernel reads of a call.
struct BwdPlanView : PlanView {
  const int32_t *dst;  // [rows]
  // across nodes, the requester's
  const HeavyGroup *heavy_groups;  // heavy groups, positions in push_order
  const int32_t *group_first_sub;  // each heavy group's first sub-task
  const int32_t *group_of_sub;     // the heavy group of each sub-task
  const int64_t *bwd_task_starts;  // [2 * max_chunks * local_size + 1] its tasks the way
                                   // fwd_task_starts are, each chunk's of heavy groups of every
                                   // proxy, then of its rows
  // across nodes, the owner's
  const int64_t *uniq_in;  // [num_nodes] node-unique keys from each source node
  // across nodes, the proxy's
  const int64_t *reduce_task_starts;  // [max_chunks + 1] its reducers' tasks before each chunk
                                      // (32 keys each)
  const int32_t *reducer_start;       // the reducer list (planner.h)
  const ReducerEntry *reducer_entries;
  // the plan's counts (BwdPlan)
  int64_t heavy_subs;  // index dedup: sub-tasks of the requester's heavy rows
  int64_t group_subs;  // across nodes: sub-tasks of its heavy groups

  // The proxy's backward put list's segments for an owner node.
  __device__ __forceinline__ int64_t bwd_puts(int owner_node) const {
    return put_starts[owner_node];
  }
};

// The plans cross the op's interface as lists of their tensors (tensors(),
// undefined ones too, in the order of their fields here), and the op makes
// them again from such a list (of()). Python holds the lists until the out
// kernels take them: autograd keeps the backward's as saved tensors, and
// frees them after the last backward.

// The forward's plan.
struct FwdPlan {
  at::Tensor offset_per_bucket;  // int64 [world_size + 1]
  at::Tensor src;                // int32 [rows]
  at::Tensor inv;                // int32 [rows]
  FanoutTables fanout;           // index dedup's, or none
  // across nodes, the tables both kernels read
  at::Tensor push_order, proxy_chunks, proxy_ends, push_tables, num_chunks, chunks, uniq_off,
      put_list, put_starts;
  // across nodes, the forward's own
  at::Tensor push_order_off, fwd_task_starts, fwd_chunks, node_chunk_start, node_chunks, uniq_out;

  // Rows of the requester's output: its input's rows (under index dedup the
  // original ones).
  int64_t num_out() const {
    return fanout.exp_idx.defined() ? fanout.exp_idx.numel() : inv.numel();
  }
  FwdPlanView view(const Sizes &sizes) const;
  std::vector<at::Tensor> tensors() const;
  static FwdPlan of(const std::vector<c10::optional<at::Tensor>> &tensors);
};

// The backward's plan, and its counts, which the call's status gives
// (StatusSlots): of() takes them as [num_recv, heavy_subs, group_subs].
struct BwdPlan {
  int64_t num_recv = 0;          // rows of the owner's lookup output (and of its lookup_grad)
  int64_t heavy_subs = 0;        // index dedup: sub-tasks of the requester's heavy rows
  int64_t group_subs = 0;        // across nodes: sub-tasks of its heavy groups
  at::Tensor offset_per_bucket;  // int64 [world_size + 1]
  at::Tensor dst;                // int32 [rows]
  at::Tensor inv;                // int32 [rows]
  FanoutTables fanout;           // index dedup's, or none
  // across nodes, the tables both kernels read
  at::Tensor push_order, proxy_chunks, proxy_ends, push_tables, num_chunks, chunks, uniq_off,
      put_list, put_starts;
  // across nodes, the backward's own
  at::Tensor heavy_groups, group_first_sub, group_of_sub, bwd_task_starts, uniq_in,
      reduce_task_starts, reducer_start, reducer_entries;

  BwdPlanView view(const Sizes &sizes) const;
  std::vector<at::Tensor> tensors() const;
  static BwdPlan of(const std::vector<c10::optional<at::Tensor>> &tensors,
                    const std::vector<int64_t> &counts);
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
