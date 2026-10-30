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
// The proxy's part of a call across nodes (the in channel's side stream,
// after node_done). The remote part of its node_keys holds the keys its node
// sends the owners of its rail on other nodes; the proxy deduplicates them
// into node-unique keys, lays their rows out in proxy_rows by rounds, and
// writes what its node's requesters and its rail's owners read of it into its
// windows: remote_map, key_stage and the proxy header. Three steps, each
// taking the tables of the ones before; ATen ops and cub sorts, nothing
// waits for the device.
//
// The occurrences are the entries of the remote part (occurrence index occ).
// A node-unique key is an identity (owner node, table, key), or each
// occurrence without node dedup; its key index u numbers the node-unique keys
// owner node after owner node, each owner node's by the smallest feature of
// the key, then in key order. Entries past the valid part of the remote part,
// and keys past the valid ones, hold garbage that no one reads.
//
// The proxy header (proxy_hdr, int32 words): word 0 the proxy's rounds; from
// word 4 its round table, RoundPiece [rounds_cap][num_nodes]; then its push
// tables, ReqEntry [local_size][rounds_cap + 1], positions relative to each
// requester's push order there and groups relative to its first group.
#pragma once

#include <ATen/ATen.h>

#include "hier_output_dist/plan/tensors.h"

namespace dyn_emb {
namespace hier_output_dist {

// The proxy's inputs: its remote part (views of its windows) and what the
// node layout says of it.
struct ProxyInput {
  at::Tensor keys;             // int64 [remote_keys]: the remote part of node_keys
  at::Tensor weights;          // fp32 [remote_keys]: their weights, or undefined
  at::Tensor cells;            // int16 [remote_keys]: owner node * num_features + feature of
                               // each (remote_cells)
  at::Tensor requester_start;  // int64 [local_size + 1]: each requester's occurrences
  at::Tensor skip;             // bool 0-dim: the node skips the call (it pushed nothing)
  at::Tensor tables;           // int32 [num_features]: the table of each feature
};

// The windows the proxy writes (views of this rank's regions).
struct ProxyWindows {
  at::Tensor remote_map;  // int32 [3][remote_keys]: per occurrence its key's row (bit 31: a
                          // heavy key), its slot, its position in its requester's push order
  at::Tensor key_stage;   // int64 [(num_nodes - 1) * rounds_cap * 2 + remote_keys *
                          // record_bytes / 8 + 2]: per other owner node a slice of its round
                          // pieces and its keys' records; the last 16 bytes take the writes
                          // of no key
  at::Tensor proxy_hdr;   // int32 [4 + 4 * rounds_cap * num_nodes + 4 * local_size *
                          // (rounds_cap + 1)]
};

// Step 1: the node-unique keys, by sorts of the occurrences.
struct ProxyKeys {
  at::Tensor occurrence_count;  // int64 0-dim: the valid occurrences (0 when the node skips)
  at::Tensor num_keys;          // int64 0-dim: the node-unique keys
  // per sorted position q: the occurrences in identity order, each identity's in occ order
  at::Tensor valid;   // bool [remote_keys]: q < occurrence_count (also per occurrence)
  at::Tensor order;   // int32 [remote_keys]: the occurrence at q
  at::Tensor key_at;  // int32 [remote_keys]: the node-unique key of the occurrence at q
  // per node-unique key u
  at::Tensor valid_key;   // bool [remote_keys]: u < num_keys
  at::Tensor key_count;   // int32 [remote_keys]: its occurrences
  at::Tensor key_heavy;   // bool [remote_keys]: more than kHeavy (never without node dedup)
  at::Tensor key_first;   // int32 [remote_keys]: the sorted position of its first occurrence
  at::Tensor key_node;    // int32 [remote_keys]: its owner node
  at::Tensor key_uniq;    // int32 [remote_keys]: u among its owner node's
  at::Tensor key_value;   // int64 [remote_keys]: the key
  at::Tensor key_weight;  // fp32 [remote_keys]: the weight of its record (its occurrences'
                          // weights, or their count), or undefined (8-byte records)
  at::Tensor cell_uniq;   // int64 [num_nodes * num_features]: the keys of each (owner node,
                          // feature)
  at::Tensor uniq_off;    // int64 [num_nodes + 1]: the first key of each owner node
  at::Tensor uniq_out;    // int64 [num_nodes]: the keys of each owner node
};
// node_dedup: equal keys of an owner node and table are one node-unique key.
// Weights must be whole numbers (frequency counts): they are summed as int64.
ProxyKeys proxy_keys(const ProxyInput &input, const PlanSizes &sizes, bool node_dedup);

// Step 2: the rows of the keys, the rounds and their pieces, each
// occurrence's slot. Writes remote_map words 0 and 1, the records and round
// pieces of key_stage, and the proxy header's rounds and round table; drops
// keys.key_value and keys.key_weight, which only the records need.
struct ProxyRounds {
  at::Tensor num_rounds;    // int64 0-dim: 1, or at least 3
  at::Tensor round_table;   // int32 [rounds_cap][num_nodes][4]: RoundPiece {uniq_lo, uniq_hi,
                            // off, 0} of each round and owner node, zeros where it has none
  at::Tensor stage_slice;   // int64 [num_nodes]: where its slice for each owner node starts in
                            // key_stage (bytes)
  at::Tensor key_round;     // int32 [remote_keys]: the round of each key
  at::Tensor key_lo;        // int32 [remote_keys]: the first key of its piece, among its owner
                            // node's
  at::Tensor key_row;       // int32 [remote_keys]: its row in proxy_rows
  at::Tensor partial_base;  // int32 [remote_keys]: a heavy key's first partial row
  at::Tensor slot_at;       // int32 [remote_keys]: the slot of the occurrence at sorted position
                            // q
  at::Tensor requester_of;  // int32 [remote_keys]: the requester of each occurrence
};
ProxyRounds proxy_rounds_and_slots(ProxyKeys &keys, const at::Tensor &requester_start,
                                   const ProxyWindows &windows, const PlanSizes &sizes);

// What the call keeps of the proxy.
struct ProxyPlan {
  at::Tensor num_rounds;        // int64 0-dim
  at::Tensor occurrence_count;  // int64 0-dim
  at::Tensor cell_uniq;         // int64 [num_nodes * num_features]
  at::Tensor uniq_off;          // int64 [num_nodes + 1]
  at::Tensor uniq_out;          // int64 [num_nodes]
  at::Tensor round_table;       // int32 [rounds_cap][num_nodes][4]
  at::Tensor stage_slice;       // int64 [num_nodes]
  at::Tensor push_start;        // int32 [local_size][rounds_cap + 1]: where each requester's
                                // positions of each round start in its push order here (the
                                // end at num_rounds and past it)
  at::Tensor reducer_start;     // int32 [remote_keys + 1]: key u's entries are
                                // reducer_entries[reducer_start[u] .. reducer_start[u + 1])
  at::Tensor reducer_entries;   // int64 [remote_keys]: a slot and the flag of the push task
                                // that fills it (formats.h)
};
// Step 3: each requester's push order (one stable sort of a
// 60-bit key: requester, round, heavy, then (IB chunk, owner node) or the
// key), its push tables, and the reducer list. Writes remote_map word 2 and
// the proxy header's push tables.
ProxyPlan proxy_push_order(ProxyKeys keys, ProxyRounds rounds, const at::Tensor &requester_start,
                           const ProxyWindows &windows, const PlanSizes &sizes);

}  // namespace hier_output_dist
}  // namespace dyn_emb
