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
// The small tables of a call, in one int64 tensor: where each one starts
// (Meta, meta_layout), and the scalars among them (MiscSlots). The planner
// writes them (host/in_channel.cu), the out kernels read them.
#pragma once

#include <cstdint>

namespace dyn_emb {
namespace hier_output_dist {

// The small tables of one call, carved from one int64 tensor that starts at
// zero: offsets in int64 words (int32 tables take whole words, padded), from
// meta_layout. Calls across nodes fill them.
struct Meta {
  // requester (num_nodes > 1)
  int proxy_round_tables;  // [local_size][rounds_cap][num_nodes] RoundPiece: each same-node
                           // proxy's round table
  int push_tables;         // [local_size][rounds_cap + 1] ReqEntry: its push order at each
                           // same-node proxy, positions absolute in push_order, groups
                           // absolute in heavy_groups
  int push_order_off;      // [local_size + 1] where its rows for each same-node proxy start
                           // in push_order
  int proxy_rounds;        // [local_size] each same-node proxy's rounds
  int fwd_task_starts;     // [rounds_cap + 1] its forward's tasks of rows of other nodes
                           // before each round, round after round: per round a slot per
                           // task of its busiest proxy there (32 positions of its push
                           // order), proxies in turn
  int bwd_task_starts;     // [2 rounds_cap + 1] its backward's the same way: before round
                           // rnd's push tasks of heavy groups (2 * rnd), and before those of
                           // its rows (2 * rnd + 1)
  // owner (num_nodes > 1)
  int slice_in;      // [num_nodes][2] each source node's rail_keys slice: byte offset,
                     // rounds
  int uniq_in;       // [num_nodes] node-unique keys from each source node
  int fwd_put_list;  // [num_nodes][rounds_cap][num_features] PutSeg: its forward put
                     // list, its rows of lookup_rows to each rail proxy, per round and
                     // feature (rows 0: none)
  // proxy (num_nodes > 1)
  int uniq_off;            // [num_nodes + 1] int32: its first node-unique key of each owner
                           // node, over all (the reducer list's index)
  int uniq_out;            // [num_nodes] its node-unique keys of each owner node
  int round_table;         // [rounds_cap][num_nodes] RoundPiece: its round table
  int bwd_put_start;       // [rounds_cap * num_nodes + 1] int32: its backward put list's
                           // segments of each (round, owner node)
  int push_start;          // [local_size][rounds_cap + 1] int32: where each same-node
                           // requester's rows of each round start in its push order here
                           // (ReqEntry::start; round num_rounds: the end)
  int reduce_task_starts;  // [rounds_cap + 1] its reducers' tasks before each round: per
                           // round its busiest owner node's IB chunks, times the owner
                           // nodes, times the tasks of a chunk (32 keys each)
  int misc;                // [kNumMisc] scalars, by MiscSlots
  int words;
};
enum MiscSlots : int {
  kMiscRounds = 0,  // its rounds
  kMiscChunks,      // IB chunks of its backward puts
  kMiscRemoteKeys,  // remote occurrences it deduplicates
  kNumMisc
};
__host__ __device__ inline Meta meta_layout(int num_nodes, int local_size, int num_features,
                                            int rounds_cap) {
  auto i32 = [](int64_t count) { return (int)((count + 1) / 2); };
  Meta layout;
  int next = 0;
  auto take = [&](int words) {
    const int at = next;
    next += words;
    return at;
  };
  layout.proxy_round_tables = take(i32((int64_t)local_size * rounds_cap * num_nodes * 4));
  layout.push_tables = take(i32((int64_t)local_size * (rounds_cap + 1) * 4));
  layout.push_order_off = take(local_size + 1);
  layout.proxy_rounds = take(local_size);
  layout.fwd_task_starts = take(rounds_cap + 1);
  layout.bwd_task_starts = take(2 * rounds_cap + 1);
  layout.slice_in = take(2 * num_nodes);
  layout.uniq_in = take(num_nodes);
  layout.fwd_put_list = take(i32((int64_t)num_nodes * rounds_cap * num_features * 4));
  layout.uniq_off = take(i32(num_nodes + 1));
  layout.uniq_out = take(num_nodes);
  layout.round_table = take(i32((int64_t)rounds_cap * num_nodes * 4));
  layout.bwd_put_start = take(i32((int64_t)rounds_cap * num_nodes + 1));
  layout.push_start = take(i32((int64_t)local_size * (rounds_cap + 1)));
  layout.reduce_task_starts = take(rounds_cap + 1);
  layout.misc = take(kNumMisc);
  layout.words = next;
  return layout;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
