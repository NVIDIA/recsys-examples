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
// The kernels' arguments: what both channels share (ChannelStatic), and
// what the out kernels take (OutArgs), with the plan of a call (OutPlan).
#pragma once

#include <nccl_device.h>

#include <cstdint>
#include <type_traits>

#include "hier_output_dist/common.h"
#include "hier_output_dist/window.h"

namespace dyn_emb {
namespace hier_output_dist {

// The part of the kernels' arguments that both channels share and that is
// fixed for the op's life (ChannelStatic): its sizes, by which a kernel lays
// out its window and its tables, and the window itself. The op builds it
// once, after its window is registered; the out channel's static part
// extends it (OutStatic), the in channel's comm kernels take it with the
// call's sequence number (CommArgs), and the exit round's kernel takes it
// alone.

struct ChannelStatic : Sizes {
  ncclWindow_t window;  // the op's window (window.h)
  uint64_t timeout_ns;  // how long a wait for a peer may take
  ncclDevComm dev;      // GIN across nodes
};
static_assert(std::is_trivially_copyable<ChannelStatic>::value,
              "a kernel parameter is copied as bytes");

// The argument of the out kernels (OutArgs): what the op fixed when it was
// made (OutStatic, which extends the part both channels share,
// ChannelStatic) and what each kernel adds (OutCall), with the plan of its
// call (OutPlan).

// One output dist kernel of a call: pointers to the call's tables, which the
// input dist made before the forward. The requester's rows are its
// bucketized rows in [0, block_start[world_size]) (its block for an owner:
// [block_start[owner], block_start[owner + 1])). The forward reads a row at
// src[row], the backward writes its gradient at dst[row]: for a same-node
// owner both are its row of the owner's lookup_rows; for a remote owner,
// rows of the proxy's proxy_rows (the node-unique key's row, and the row's
// own slot). inv[row] is its row of the requester's output, or under index
// dedup its deduplicated row, whose original rows the fan-out lists. Across
// nodes, the requester's push order, heavy groups and its tables of the
// node's proxies, the owner's and the proxy's tables, and the put lists say
// the rest; the tables' strides are rounds_cap and num_nodes. Both kernels
// of a call get the same pointers, but for the fields marked forward or
// backward, which only that kernel gets (the plan lets go of the forward's
// tables after it).
struct OutPlan {
  const int64_t *block_start;  // [world_size + 1]
  const int32_t *rows;         // [sent] forward: src; backward: dst
  const int32_t *inv;          // [sent]
  Fanout fanout;               // fanout.exp_off: index dedup
  // across nodes, the requester's
  const int32_t *push_order;             // [sent] remote rows by proxy and push order
  const HeavyGroup *heavy_groups;        // heavy groups, positions in push_order
  const int32_t *group_first_sub;        // each heavy group's first sub-task
  const int32_t *group_of_sub;           // the heavy group of each sub-task
  const int64_t *proxy_rounds;           // [local_size] each same-node proxy's rounds
  const RoundPiece *proxy_round_tables;  // forward: [local_size][rounds_cap][num_nodes] their
                                         // round tables
  const ReqEntry *push_tables;     // [local_size][rounds_cap + 1] its push order at each proxy,
                                   // positions absolute in push_order, groups in heavy_groups
  const int64_t *push_order_off;   // forward: [local_size + 1] where its rows for each proxy
                                   // start in push_order
  const int64_t *fwd_task_starts;  // forward: [rounds_cap + 1] its tasks of rows of other nodes
                                   // before each round, round after round: per round a slot per
                                   // task of its busiest proxy there (32 positions of its push
                                   // order), proxies in turn
  const int64_t *bwd_task_starts;  // [2 rounds_cap + 1] its backward's the same way: before
                                   // round rnd's push tasks of heavy groups (2 * rnd), and before
                                   // those of its rows (2 * rnd + 1)
  // across nodes, the owner's
  const int64_t *uniq_in;        // [num_nodes] node-unique keys from each source node
  const int32_t *fwd_put_start;  // forward: [num_nodes][rounds_cap + 1] (fwd_puts)
  // across nodes, the proxy's
  const int64_t *num_rounds;          // its rounds
  const int64_t *num_chunks;          // IB chunks of its backward puts
  const int64_t *uniq_out;            // forward: [num_nodes] its node-unique keys of each owner
                                      // node
  const int64_t *uniq_off;            // [num_nodes + 1] its first node-unique key of each owner
                                      // node, over all (the reducer list's index)
  const RoundPiece *round_table;      // [rounds_cap][num_nodes] its round table
  const int32_t *bwd_put_start;       // [num_nodes][rounds_cap + 1] (bwd_puts)
  const int32_t *push_start;          // forward: [local_size][rounds_cap + 1] (push_starts)
  const int64_t *reduce_task_starts;  // [rounds_cap + 1] its reducers' tasks before each round:
                                      // per round its busiest owner node's IB chunks, times the
                                      // owner nodes, times the tasks of a chunk (32 keys each)
  const int32_t *reducer_start;       // the reducer list (planner.h)
  const int64_t *reducer_entries;
  const PutSeg *put_list;  // the owner's forward and the proxy's backward put lists
                           // (planner.h)
  int64_t *ready;          // backward, proxy: [chunks] keys of each IB chunk
                           // reduced; forward, requester: [local_size] rows it
                           // pulled from each proxy (index dedup's heavy rows
                           // wait for them); zeroed when the kernel starts
  int64_t rounds_cap;
  int num_nodes;
  // rows
  char *out;             // forward: [num_out, row] the requester's output
  const char *grad;      // backward: [num_out, row] its gradient
  float *partial;        // backward: [sub-tasks, dim] fp32 sums of the heavy
                         // rows' sub-tasks (index dedup), then of the heavy
                         // groups' (kStatusHeavySubs, then kStatusGroupSubs)
  uint64_t *heavy_done;  // backward: [heavy rows] sub-tasks of each one done
  uint64_t *group_done;  // backward: [heavy groups] the same (both zeroed when
                         // the kernel starts)
  int64_t heavy_subs;    // backward: the heavy rows' sub-tasks
  int64_t group_subs;    // backward: the heavy groups' sub-tasks
  float scale;           // backward: 1 / world_size under TorchRec's gradient
                         // division, else 1

  // Where an owner node's node-unique keys of a round sit in a same-node
  // proxy's proxy_rows.
  __device__ __forceinline__ RoundPiece piece(int proxy, int rnd, int owner_node) const {
    return proxy_round_tables[((int64_t)proxy * rounds_cap + rnd) * num_nodes + owner_node];
  }
  // Where an owner node's node-unique keys of a round sit in this proxy's
  // proxy_rows, and their first IB chunk.
  __device__ __forceinline__ RoundPiece round_piece(int rnd, int owner_node) const {
    return round_table[(int64_t)rnd * num_nodes + owner_node];
  }
  // This rank's push order at a same-node proxy in a round; the next entry
  // holds the ends.
  __device__ __forceinline__ const ReqEntry *push_entry(int proxy, int rnd) const {
    return push_tables + (int64_t)proxy * (rounds_cap + 1) + rnd;
  }
  // The owner's forward put list's segments for a proxy node, by round: those
  // of round rnd from [rnd], and [rounds_cap] the node's end.
  __device__ __forceinline__ const int32_t *fwd_puts(int proxy_node) const {
    return fwd_put_start + (int64_t)proxy_node * (rounds_cap + 1);
  }
  // The proxy's backward put list's segments for an owner node, the same way.
  __device__ __forceinline__ const int32_t *bwd_puts(int owner_node) const {
    return bwd_put_start + (int64_t)owner_node * (rounds_cap + 1);
  }
  // Where a same-node requester's rows of each round start in its push order
  // at this proxy ([num_rounds]: the end).
  __device__ __forceinline__ const int32_t *push_starts(int requester) const {
    return push_start + (int64_t)requester * (rounds_cap + 1);
  }
};

// What every out kernel of the op takes, fixed when the op is made.
struct OutStatic : ChannelStatic {
  int vec;  // a row of lookup_rows in 16-byte vectors
};

// What one out kernel adds: its sequence number, its CTAs' roles and
// arrivals, its timeline, and its call's plan.
struct OutCall {
  Channel ch;                 // the out channel, at this kernel's out_seq
  int reducers;               // backward across nodes: the last CTAs, which reduce
  int64_t arrive0;            // the arrive word (CtrlLayout) before this kernel
  unsigned long long *trace;  // [kTraceWords] this kernel's timeline, or nullptr
  OutPlan plan;
};

// The out kernels' parameter, by value: OutChannel builds it per kernel from
// its OutStatic and the kernel's OutCall.
struct OutArgs : OutStatic, OutCall {};

// The out kernels' dynamic shared memory: the backward's half_seen
// [local_size] (OutTables), then, when the kernel keeps a timeline, the
// CTA's copy of it (TraceSmSlots, OutTimeline).
__device__ __forceinline__ unsigned long long *out_shared() {
  extern __shared__ unsigned long long out_dyn_sm[];
  return out_dyn_sm;
}
// Its bytes, what each out kernel is launched with.
inline size_t out_shared_bytes(int local_size, bool timeline) {
  return 8 * ((size_t)local_size + (timeline ? kTraceSmWords : 0));
}
static_assert(std::is_trivially_copyable<OutArgs>::value, "a kernel parameter is copied as bytes");
static_assert(sizeof(OutArgs) <= 4096, "the out kernels' parameter fits the parameter space");

}  // namespace hier_output_dist
}  // namespace dyn_emb
