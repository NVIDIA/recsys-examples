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
// The rail's part of a call across nodes (the in channel's side stream). A
// rank's rail is the ranks with its local rank, one on each node. Each rank
// sends its rail a count message; from the messages every rank of the rail
// computes the same layout: as proxy where its key records land at each
// owner, as owner its KJT. Then the needs and cell starts a rank publishes to
// its node, and the owner's tables: its forward put list and its KJT's keys.
// ATen ops; nothing waits for the device.
//
// An owner's KJT has num_sources = local_size + num_nodes - 1 cells per
// feature, feature-major: its node's requesters, then the other nodes in
// order. Its own node's cells are the own part of its node_keys; another
// node's cell holds the node-unique keys of that node's proxy on its rail,
// whose records the proxy puts into its rail_keys: one slice per other node
// in node order, each the proxy's round pieces, then its records.
#pragma once

#include <ATen/ATen.h>

#include "hier_output_dist/plan/tensors.h"

namespace dyn_emb {
namespace hier_output_dist {

// The nodes but this rank's, in order (int64 [num_nodes - 1]): a constant of
// the op.
at::Tensor other_nodes_of(const Topo &topo, const at::TensorOptions &options);

// The count message this rank sends its rail (int64 [2 + (num_nodes + 1) *
// num_features]): its node's largest remote part (need_remote), its rounds
// and its node-unique keys per (owner node, feature) as proxy (num_rounds,
// cell_uniq), and per feature the keys its node sends it as owner (from the
// node layout's own_cells).
at::Tensor count_message(const at::Tensor &need_remote, const at::Tensor &own_cells,
                         const at::Tensor &num_rounds, const at::Tensor &cell_uniq,
                         const PlanSizes &sizes);

// The rail's layout, from the rail's messages (int64 [num_nodes][message]),
// the same on every rank of the rail. Each slice a proxy puts holds its round
// pieces (16 bytes each), then its records (record_bytes each).
struct RailLayout {
  // as proxy, to each owner of the rail (its slice in key_stage is the proxy's stage_slice)
  at::Tensor put_dst;          // int64 [num_nodes]: where its slice lands in the owner's
                               // rail_keys (bytes)
  at::Tensor put_bytes;        // int64 [num_nodes]
  at::Tensor put_skip;         // bool [num_nodes]: no put (the owner overflows lookup_keys,
                               // or it is on this node)
  at::Tensor rail_cell_start;  // int64 [num_nodes][num_features]: where its cell (feature,
                               // this node) starts in each owner's KJT
  // as owner
  at::Tensor kjt_lengths;     // int64 [num_features * num_sources]: cell (feature, source)
  at::Tensor kjt_offsets;     // int64 [num_features * num_sources + 1]
  at::Tensor num_recv;        // int64 0-dim: the keys it looks up
  at::Tensor length_per_key;  // int64 [num_features]
  at::Tensor slice_in;        // int64 [num_nodes][2]: each source node's slice in its
                              // rail_keys: byte offset, rounds
  at::Tensor uniq_in;         // int64 [num_nodes]: the node-unique keys of each source node
  at::Tensor rail_lengths;    // int64 [num_features][num_nodes - 1]: the other nodes' cells
  at::Tensor my_cell_starts;  // int64 [local_size][num_features]: where each requester of
                              // its node's cell (feature, requester) starts in its KJT
  at::Tensor needs;           // int64 [3]: the most the rail needs of lookup_keys and
                              // remote_keys, and its most rounds
};
RailLayout rail_layout(const at::Tensor &messages, const at::Tensor &own_cells,
                       const at::Tensor &other_nodes, const PlanSizes &sizes);

// What this rank publishes to its node (int64 [3 + local_size *
// num_features]): its rail's needs, then my_cell_starts.
at::Tensor status_publish(const RailLayout &rail);
// The call's needs (int64 [3]) from its node's publish vectors (peer_needs,
// int64 [local_size][3]): a rank's rail covers every node, and its node holds
// a rank of every rail, so the most of the node is the most of all ranks.
at::Tensor agreed_needs(const at::Tensor &peer_needs);

// The owner's tables, once the rail's proxies put their records into its
// rail_keys (int64 view) and its node's requesters their keys into the own
// part of its node_keys (node_keys, node_weights: views of [lookup_keys];
// node_weights undefined when the keys carry none).
struct OwnerTables {
  at::Tensor fwd_put_list;  // int32 [num_nodes][rounds_cap][num_features][4]: PutSeg {src
                            // row in its KJT, dst row in the proxy's proxy_rows, rows, 0} of
                            // each source node's round pieces in each cell (rows 0: none)
  at::Tensor kjt_values;    // int64 [lookup_keys]: the first num_recv are its KJT's keys
  at::Tensor kjt_weights;   // fp32 [lookup_keys], or undefined (8-byte records): their
                            // weights, 1 for its own node's keys when they carry none
};
OwnerTables owner_tables(const RailLayout &rail, const at::Tensor &own_starts,
                         const at::Tensor &node_keys, const at::Tensor &node_weights,
                         const at::Tensor &rail_keys, const at::Tensor &other_nodes,
                         const PlanSizes &sizes);

}  // namespace hier_output_dist
}  // namespace dyn_emb
