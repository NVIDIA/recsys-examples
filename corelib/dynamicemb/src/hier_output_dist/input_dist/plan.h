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
// The requester's rows are its bucketized rows [0, block_start[world_size]),
// owner by owner (its block for an owner: [block_start[owner],
// block_start[owner + 1])). The forward reads a row at src[row], the backward
// writes its gradient at dst[row]: for an owner of its node both are its row
// of the owner's lookup_rows; for an owner on another node, rows of the
// proxy's proxy_rows (the node-unique key's row, and the row's own slot).
// inv[row] is its row of the requester's output, or under index dedup its
// deduplicated row, whose original rows the fan-out lists. Across nodes the
// requester's push order, heavy groups and its tables of the node's proxies,
// the owner's and the proxy's tables, and the put lists say the rest
// (planner.h makes them); the tables' strides are rounds_cap and num_nodes.
// On one node the tables across nodes are undefined, nullptr in the views.
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
  const int64_t *block_start;  // [world_size + 1]
  const int32_t *inv;          // [rows]
  Fanout fanout;               // fanout.exp_off: index dedup
  // across nodes
  const int32_t *push_order;    // [rows] the requester's remote rows by proxy and push order
  const int64_t *proxy_rounds;  // [local_size] each same-node proxy's rounds
  const ReqEntry *push_tables;  // [local_size][rounds_cap + 1] its push order at each proxy,
                                // positions absolute in push_order, groups in heavy_groups
  const int64_t *num_rounds;    // the proxy's rounds
  const PutSeg *put_list;       // the owner's forward and the proxy's backward put lists
                                // (planner.h)
  int64_t rounds_cap;
  int num_nodes;

  // This rank's push order at a same-node proxy in a round; the next entry
  // holds the ends.
  __device__ __forceinline__ const ReqEntry *push_entry(int proxy, int rnd) const {
    return push_tables + (int64_t)proxy * (rounds_cap + 1) + rnd;
  }
};

// What fwd_kernel reads of a call.
struct FwdPlanView : PlanView {
  const int32_t *src;  // [rows]
  // across nodes, the requester's
  const RoundPiece *proxy_round_tables;  // [local_size][rounds_cap][num_nodes] the round tables
                                         // of the node's proxies
  const int64_t *push_order_off;         // [local_size + 1] where its rows for each proxy start
                                         // in push_order
  const int64_t *fwd_task_starts;        // [rounds_cap + 1] its tasks of rows of other nodes before
                                         // each round, round after round: per round a slot per task
                                         // of its busiest proxy there (32 positions of its push
                                         // order), proxies in turn
  // across nodes, the owner's
  const int32_t *fwd_put_start;  // [num_nodes][rounds_cap + 1] (fwd_puts)
  // across nodes, the proxy's
  const int64_t *uniq_out;    // [num_nodes] its node-unique keys of each owner node
  const int32_t *push_start;  // [local_size][rounds_cap + 1] (push_starts)

  // Where an owner node's node-unique keys of a round sit in a same-node
  // proxy's proxy_rows.
  __device__ __forceinline__ RoundPiece piece(int proxy, int rnd, int owner_node) const {
    return proxy_round_tables[((int64_t)proxy * rounds_cap + rnd) * num_nodes + owner_node];
  }
  // The owner's forward put list's segments for a proxy node, by round: those
  // of round rnd from [rnd], and [rounds_cap] the node's end.
  __device__ __forceinline__ const int32_t *fwd_puts(int proxy_node) const {
    return fwd_put_start + (int64_t)proxy_node * (rounds_cap + 1);
  }
  // Where a same-node requester's rows of each round start in its push order
  // at this proxy ([num_rounds]: the end).
  __device__ __forceinline__ const int32_t *push_starts(int requester) const {
    return push_start + (int64_t)requester * (rounds_cap + 1);
  }
};

// What bwd_kernel reads of a call.
struct BwdPlanView : PlanView {
  const int32_t *dst;  // [rows]
  // across nodes, the requester's
  const HeavyGroup *heavy_groups;  // heavy groups, positions in push_order
  const int32_t *group_first_sub;  // each heavy group's first sub-task
  const int32_t *group_of_sub;     // the heavy group of each sub-task
  const int64_t *bwd_task_starts;  // [2 rounds_cap + 1] its tasks the way fwd_task_starts are:
                                   // before round rnd's push tasks of heavy groups (2 * rnd),
                                   // and before those of its rows (2 * rnd + 1)
  // across nodes, the owner's
  const int64_t *uniq_in;  // [num_nodes] node-unique keys from each source node
  // across nodes, the proxy's
  const int64_t *num_chunks;          // IB chunks of its backward puts
  const int64_t *uniq_off;            // [num_nodes + 1] its first node-unique key of each owner
                                      // node, over all (the reducer list's index)
  const RoundPiece *round_table;      // [rounds_cap][num_nodes] its round table
  const int32_t *bwd_put_start;       // [num_nodes][rounds_cap + 1] (bwd_puts)
  const int64_t *reduce_task_starts;  // [rounds_cap + 1] its reducers' tasks before each round:
                                      // per round its busiest owner node's IB chunks, times the
                                      // owner nodes, times the tasks of a chunk (32 keys each)
  const int32_t *reducer_start;       // the reducer list (planner.h)
  const int64_t *reducer_entries;

  // Where an owner node's node-unique keys of a round sit in this proxy's
  // proxy_rows, and their first IB chunk.
  __device__ __forceinline__ RoundPiece round_piece(int rnd, int owner_node) const {
    return round_table[(int64_t)rnd * num_nodes + owner_node];
  }
  // The proxy's backward put list's segments for an owner node, by round:
  // those of round rnd from [rnd], and [rounds_cap] the node's end.
  __device__ __forceinline__ const int32_t *bwd_puts(int owner_node) const {
    return bwd_put_start + (int64_t)owner_node * (rounds_cap + 1);
  }
};

// The forward's plan.
struct FwdPlan {
  int64_t num_out = 0;     // rows of the requester's output
  at::Tensor block_start;  // int64 [world_size + 1]
  at::Tensor src;          // int32 [rows]
  at::Tensor inv;          // int32 [rows]
  FanoutTables fanout;     // index dedup's, or none
  // across nodes, the tables both kernels read
  at::Tensor push_order, proxy_rounds, push_tables, num_rounds, put_list;
  // across nodes, the forward's own
  at::Tensor proxy_round_tables, push_order_off, fwd_task_starts, fwd_put_start, uniq_out,
      push_start;

  FwdPlanView view(const Sizes &sizes) const;
  // The tensors' use on `stream` (the caching allocator's).
  void record_stream(c10::Stream stream) const;
};

// The backward's plan.
struct BwdPlan {
  int64_t num_recv = 0;    // rows of the owner's lookup output (and of its lookup_grad)
  int64_t heavy_subs = 0;  // index dedup: sub-tasks of the requester's heavy rows
  int64_t group_subs = 0;  // across nodes: sub-tasks of its heavy groups
  at::Tensor block_start;  // int64 [world_size + 1]
  at::Tensor dst;          // int32 [rows]
  at::Tensor inv;          // int32 [rows]
  FanoutTables fanout;     // index dedup's, or none
  // across nodes, the tables both kernels read
  at::Tensor push_order, proxy_rounds, push_tables, num_rounds, put_list;
  // across nodes, the backward's own
  at::Tensor heavy_groups, group_first_sub, group_of_sub, bwd_task_starts, uniq_in, num_chunks,
      uniq_off, round_table, bwd_put_start, reduce_task_starts, reducer_start, reducer_entries;

  BwdPlanView view(const Sizes &sizes) const;
  void record_stream(c10::Stream stream) const;
  // Autograd keeps the plan from the forward to the backward as saved
  // tensors, which it frees after the last backward: tensors() (undefined
  // ones too) and counts(), from which of() makes the plan again.
  std::vector<at::Tensor> tensors() const;
  std::vector<int64_t> counts() const { return {num_recv, heavy_subs, group_subs}; }
  static BwdPlan of(const std::vector<c10::optional<at::Tensor>> &tensors,
                    const std::vector<int64_t> &counts);
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
