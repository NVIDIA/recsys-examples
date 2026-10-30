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
#include "hier_output_dist/plan/node.h"

#include "hier_output_dist/plan/tensors.h"

namespace dyn_emb {
namespace hier_output_dist {

at::Tensor node_count_vector(const at::Tensor &cell_start, const Topo &topo, int64_t num_features) {
  const at::Tensor cell_size =
      cell_start.diff().view({topo.num_nodes, topo.local_size, num_features});
  const at::Tensor same_node = cell_size.select(0, topo.node);           // [owner][feature]
  const at::Tensor remote = cell_size.sum({0, 2}) - same_node.sum({1});  // [target]
  return at::cat({remote, same_node.reshape({-1})});
}

NodeLayout node_layout(const at::Tensor &node_counts, const at::Tensor &cell_start,
                       const at::Tensor &block_start, const Topo &topo, int64_t num_features,
                       int64_t lookup_keys, int64_t remote_keys) {
  const int64_t local_size = topo.local_size, num_nodes = topo.num_nodes,
                world_size = topo.world_size, local_rank = topo.local_rank;
  const int64_t own_columns = num_features * local_size;
  NodeLayout node;
  const at::Tensor same_counts =
      node_counts.narrow(1, local_size, own_columns).view({local_size, local_size, num_features});
  // each owner's own part, in its KJT order: [owner][feature * local_size + requester]
  // (contiguous: with one feature the reshape would be a view)
  node.own_cells = same_counts.permute({1, 2, 0}).reshape({local_size, own_columns}).contiguous();
  const at::Tensor own_ends = node.own_cells.cumsum(1);
  node.own_starts = own_ends - node.own_cells;
  node.own_total = own_ends.select(1, own_columns - 1);
  // the needs of the node's parts, and whether it skips the call: the same on
  // every rank of the node
  node.need_lookup = node.own_total.max();
  node.skip = node.need_lookup.gt(lookup_keys);
  // A row of an owner of this node goes to the owner's own part: where this
  // rank's cell starts there, and the rows of the cell before it.
  const at::Tensor own_base =
      node.own_starts.view({local_size, num_features, local_size}).select(2, local_rank) -
      cell_start.narrow(0, 0, world_size * num_features)
          .view({num_nodes, local_size, num_features})
          .select(0, topo.node);
  at::Tensor destination_base = own_base;  // one node: every owner is on it
  if (num_nodes > 1) {
    const at::Tensor remote_counts = node_counts.narrow(1, 0, local_size);  // [requester][target]
    node.requester_start = exclusive_ends(remote_counts.select(1, local_rank));
    node.need_remote = remote_counts.sum({0}).max();
    node.skip.logical_or_(node.need_remote.gt(remote_keys));
    // A remote row of owner (owner node, target) goes to its target's remote
    // part: after the requesters before this rank, this rank's blocks for the
    // owner nodes before, and the rows of its block before it.
    at::Tensor remote_blocks = block_start.diff().view({num_nodes, local_size});
    remote_blocks.select(0, topo.node).zero_();
    const at::Tensor remote_block_start = remote_blocks.cumsum(0) - remote_blocks;
    const at::Tensor remote_before_me = remote_counts.narrow(0, 0, local_rank).sum({0});
    destination_base =
        (lookup_keys + remote_before_me.view({1, local_size, 1}) + remote_block_start.unsqueeze(2) -
         block_start.narrow(0, 0, world_size).view({num_nodes, local_size, 1}))
            .expand({num_nodes, local_size, num_features})
            .contiguous();
    destination_base.select(0, topo.node).copy_(own_base);
  }
  node.rows_to_push = at::where(node.skip, 0, block_start.select(0, world_size));
  // the rows past the valid ones (cell world_size * num_features) are never pushed
  node.destination_base =
      at::cat({destination_base.reshape({-1}), at::zeros({1}, destination_base.options())});
  return node;
}

OneNodeOwner one_node_owner(const NodeLayout &node, const at::Tensor &row_cell,
                            const at::Tensor &node_keys, const at::Tensor &node_weights,
                            const Topo &topo, int64_t num_features) {
  OneNodeOwner owner;
  // The own part is the KJT: one copy of the bound (wait() narrows it to
  // num_recv). The node's next call rewrites node_keys after its barrier,
  // which does not wait for this call's lookup.
  owner.kjt_values = node_keys.clone();
  if (node_weights.defined()) owner.kjt_weights = node_weights.clone();
  owner.kjt_lengths = node.own_cells.select(0, topo.local_rank).contiguous();
  owner.kjt_offsets = exclusive_ends(owner.kjt_lengths);
  owner.num_recv = node.own_total.select(0, topo.local_rank);
  owner.length_per_key = owner.kjt_lengths.view({num_features, topo.local_size}).sum({1});
  owner.src = (node.destination_base.index_select(0, row_cell) +
               at::arange(row_cell.numel(), row_cell.options()))
                  .to(at::kInt);
  return owner;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
