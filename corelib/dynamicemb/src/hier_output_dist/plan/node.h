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
// The node's part of a call (the in channel's side stream): this rank's
// counts for its node, the layout every rank of the node computes alike from
// the counts the node exchanged, and on one node the owner's KJT and the
// requester's map. ATen ops; nothing waits for the device.
//
// node_keys of a target holds its own part at [0, lookup_keys), the keys of
// its node for the target as owner, in the owner's KJT order
// ([feature][requester][requester's order]); and across nodes its remote
// part at [lookup_keys, lookup_keys + remote_keys), the keys of its node for
// the owners on its rail of the other nodes, which it deduplicates as their
// proxy: requester-major, then owner node, then [feature][position].
#pragma once

#include <ATen/ATen.h>

#include <cstdint>

#include "hier_output_dist/common/topology.h"

namespace dyn_emb {
namespace hier_output_dist {

// This rank's counts for its node, int64 [local_size * (num_features + 1)]:
// first its keys for each target's remote part, then its keys for each
// owner of its node, [owner][feature]. From part 1's cell_start.
at::Tensor node_count_vector(const at::Tensor &cell_start, const Topo &topo, int64_t num_features);

// The layout of the node's keys, from node_counts (int64 [local_size][count
// vector], a row per requester) and this rank's part 1. Under a need above
// a capacity no rank of the node pushes (skip). One node has no remote part:
// there requester_start and need_remote stay undefined.
struct NodeLayout {
  at::Tensor own_cells;         // int64 [local_size][num_features * local_size]: keys of cell
                                // (feature, requester) in each owner's own part
  at::Tensor own_starts;        // int64 [local_size][num_features * local_size]: where each
                                // such cell starts there
  at::Tensor own_total;         // int64 [local_size]: each owner's own part
  at::Tensor requester_start;   // int64 [local_size + 1]: where each requester's keys start in
                                // this rank's remote part
  at::Tensor need_lookup;       // int64 0-dim: the largest own part of the node
  at::Tensor need_remote;       // int64 0-dim: the largest remote part of the node
  at::Tensor skip;              // bool 0-dim: a need above its capacity
  at::Tensor rows_to_push;      // int64 0-dim: this rank's valid rows, 0 under skip
  at::Tensor destination_base;  // int64 [world_size * num_features + 1]: row `row` of part 1
                                // goes to destination_base[row_cell[row]] + row in node_keys
                                // of its target
};
NodeLayout node_layout(const at::Tensor &node_counts, const at::Tensor &cell_start,
                       const at::Tensor &block_start, const Topo &topo, int64_t num_features,
                       int64_t lookup_keys, int64_t remote_keys);

// On one node, the KJT of this rank as owner, a copy of its own part of
// node_keys once every requester of the node pushed (node_keys, and
// node_weights when the keys carry weights: views of the own part), and its
// map as requester: its rows' places in their owners' KJTs, which are where
// it pushed them.
struct OneNodeOwner {
  at::Tensor kjt_values;      // int64 [lookup_keys]: the first num_recv are the KJT's
  at::Tensor kjt_weights;     // fp32 [lookup_keys], or undefined
  at::Tensor kjt_lengths;     // int64 [num_features * local_size]: cell (feature, requester)
  at::Tensor kjt_offsets;     // int64 [num_features * local_size + 1]
  at::Tensor num_recv;        // int64 0-dim: the keys it looks up
  at::Tensor length_per_key;  // int64 [num_features]
  at::Tensor src;             // int32 [row_cell.numel()]: the place of each row in its owner's
                              // KJT (and its dst)
};
OneNodeOwner one_node_owner(const NodeLayout &node, const at::Tensor &row_cell,
                            const at::Tensor &node_keys, const at::Tensor &node_weights,
                            const Topo &topo, int64_t num_features);

}  // namespace hier_output_dist
}  // namespace dyn_emb
