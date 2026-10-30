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
#include "hier_output_dist/plan/rail.h"

#include <algorithm>
#include <cstdint>

namespace dyn_emb {
namespace hier_output_dist {

at::Tensor other_nodes_of(const Topo &topo, const at::TensorOptions &options) {
  const at::Tensor nodes = at::arange(topo.num_nodes - 1, options.dtype(at::kLong));
  return nodes + nodes.ge(topo.node);
}

at::Tensor count_message(const at::Tensor &need_remote, const at::Tensor &own_cells,
                         const at::Tensor &num_rounds, const at::Tensor &cell_uniq,
                         const PlanSizes &sizes) {
  const at::Tensor node_totals = own_cells.select(0, sizes.topo.local_rank)
                                     .view({sizes.num_features, sizes.topo.local_size})
                                     .sum({1});
  return at::cat({need_remote.view({1}), num_rounds.view({1}), cell_uniq, node_totals});
}

RailLayout rail_layout(const at::Tensor &messages, const at::Tensor &own_cells,
                       const at::Tensor &other_nodes, const PlanSizes &sizes) {
  const int64_t num_nodes = sizes.topo.num_nodes, node = sizes.topo.node,
                local_size = sizes.topo.local_size, num_features = sizes.num_features,
                num_sources = sizes.topo.num_sources;
  RailLayout rail;
  const at::Tensor source_rounds = messages.select(1, 1);  // [source node]
  // keys of each source node's proxy for each owner node and feature, and
  // the keys each owner's node sends it per feature
  const at::Tensor counts =
      messages.narrow(1, 2, num_nodes * num_features).view({num_nodes, num_nodes, num_features});
  const at::Tensor owner_totals = messages.narrow(1, 2 + num_nodes * num_features, num_features);
  const at::Tensor total = owner_totals + counts.sum({0});  // [owner node][feature]
  const at::Tensor pair_keys = counts.sum({2});             // [source node][owner node]
  const at::Tensor recv = total.sum({1});                   // [owner node]
  // a slice: the source's round pieces, then its records; none to its own node
  const at::Tensor slice_bytes =
      (source_rounds.view({num_nodes, 1}) * 16 + pair_keys * sizes.record_bytes).fill_diagonal_(0);
  // As proxy: where its slice and its cells start at each owner. An owner's
  // KJT holds per feature its node's keys, then the other nodes' in order.
  rail.put_dst = slice_bytes.narrow(0, 0, node).sum({0});
  rail.put_bytes = slice_bytes.select(0, node);
  rail.put_skip = recv.gt(sizes.lookup_keys);
  rail.put_skip.narrow(0, node, 1).fill_(true);
  rail.rail_cell_start =
      total.cumsum(1) - total + owner_totals + counts.narrow(0, 0, node).sum({0});
  // as owner: its KJT, cell (feature, source) feature-major
  rail.rail_lengths = counts.select(1, node).index_select(0, other_nodes).t();
  rail.kjt_lengths =
      at::cat({own_cells.select(0, sizes.topo.local_rank).view({num_features, local_size}),
               rail.rail_lengths},
              1)
          .reshape({-1});
  rail.kjt_offsets = exclusive_ends(rail.kjt_lengths);
  rail.num_recv = recv.select(0, node);
  rail.length_per_key = total.select(0, node);
  const at::Tensor into_me = slice_bytes.select(1, node);
  rail.slice_in = at::stack({into_me.cumsum(0) - into_me, source_rounds}, 1);
  rail.uniq_in = pair_keys.select(1, node);
  rail.my_cell_starts = rail.kjt_offsets.narrow(0, 0, num_features * num_sources)
                            .view({num_features, num_sources})
                            .narrow(1, 0, local_size)
                            .t()
                            .contiguous();
  rail.needs = at::stack({recv.max(), messages.select(1, 0).max(), source_rounds.max()});
  return rail;
}

at::Tensor status_publish(const RailLayout &rail) {
  return at::cat({rail.needs, rail.my_cell_starts.view({-1})});
}

at::Tensor agreed_needs(const at::Tensor &peer_needs) { return peer_needs.amax({0}); }

OwnerTables owner_tables(const RailLayout &rail, const at::Tensor &own_starts,
                         const at::Tensor &node_keys, const at::Tensor &node_weights,
                         const at::Tensor &rail_keys, const at::Tensor &other_nodes,
                         const PlanSizes &sizes) {
  const int64_t num_nodes = sizes.topo.num_nodes, local_size = sizes.topo.local_size,
                num_features = sizes.num_features, num_sources = sizes.topo.num_sources,
                rounds_cap = sizes.rounds_cap, lookup_keys = sizes.lookup_keys,
                record_bytes = sizes.record_bytes;
  const auto long_options = rail_keys.options().dtype(at::kLong);
  const auto int_options = rail_keys.options().dtype(at::kInt);
  OwnerTables tables;
  // each other node's slice in rail_keys: its rounds, where it starts (bytes),
  // and where its keys of each feature start among them
  const at::Tensor source_rounds = rail.slice_in.select(1, 1).index_select(0, other_nodes);
  const at::Tensor slice_start = rail.slice_in.select(1, 0).index_select(0, other_nodes);
  // [num_features][num_nodes - 1]
  const at::Tensor cell_first = rail.rail_lengths.cumsum(0) - rail.rail_lengths;
  // The forward put list: the keys of each other node's round piece in each
  // of its cells, from the owner's KJT to the proxy's rows.
  {
    const at::Tensor rail_ints = rail_keys.view(at::kInt);
    const at::Tensor piece_word = (slice_start.div(4, "floor").view({num_nodes - 1, 1}) +
                                   4 * at::arange(rounds_cap, long_options).view({1, rounds_cap}))
                                      .clamp(0, rail_ints.numel() - 4);
    auto piece_field = [&](int field) {  // [num_nodes - 1][rounds_cap][1]
      return rail_ints.take(piece_word + field).to(at::kLong).unsqueeze(2);
    };
    const at::Tensor source_lo = piece_field(0), source_hi = piece_field(1),
                     source_off = piece_field(2);
    const at::Tensor cell_lo = cell_first.t().reshape({num_nodes - 1, 1, num_features});
    const at::Tensor cell_hi =
        cell_lo + rail.rail_lengths.t().reshape({num_nodes - 1, 1, num_features});
    const at::Tensor lo = at::maximum(cell_lo, source_lo), hi = at::minimum(cell_hi, source_hi);
    const at::Tensor kept =
        lo.lt(hi).logical_and_(at::arange(rounds_cap, long_options)
                                   .view({1, rounds_cap, 1})
                                   .lt(source_rounds.view({num_nodes - 1, 1, 1})));
    const at::Tensor rail_offsets = rail.kjt_offsets.narrow(0, 0, num_features * num_sources)
                                        .view({num_features, num_sources})
                                        .narrow(1, local_size, num_nodes - 1)
                                        .t()
                                        .reshape({num_nodes - 1, 1, num_features});
    const at::Tensor segments = at::stack({rail_offsets + lo - cell_lo, source_off + lo - source_lo,
                                           hi - lo, at::zeros_like(lo)},
                                          3)
                                    .mul_(kept.unsqueeze(3));
    tables.fwd_put_list = at::zeros({num_nodes, rounds_cap, num_features, 4}, int_options)
                              .index_copy_(0, other_nodes, segments.to(at::kInt));
  }
  // The KJT's keys at positions [0, lookup_keys), in slices of 2^22 with int32
  // indices: its own node's cells from the own part of node_keys, the other
  // nodes' from their records in rail_keys (int64 words).
  constexpr int64_t kSlice = int64_t{1} << 22;
  const at::Tensor cell_offsets =
      rail.kjt_offsets.narrow(0, 0, num_features * num_sources).to(at::kInt);
  const at::Tensor own_cell_start = own_starts.select(0, sizes.topo.local_rank).to(at::kInt);
  const at::Tensor first_record = (slice_start + source_rounds * 16).div_(8, "floor");
  const at::Tensor cell_first_key = cell_first.reshape({-1}).to(at::kInt);
  tables.kjt_values = at::empty({lookup_keys}, long_options);
  if (record_bytes == 16)
    tables.kjt_weights = at::empty({lookup_keys}, long_options.dtype(at::kFloat));
  for (int64_t first = 0; first < lookup_keys; first += kSlice) {
    const int64_t count = std::min(kSlice, lookup_keys - first);
    at::Tensor same, own_index, record;  // record: the int64 word of the key's record
    {
      const at::Tensor position = at::arange(first, first + count, int_options);
      const at::Tensor cell =
          (at::searchsorted(cell_offsets, position, /*out_int32=*/true, /*right=*/true) - 1)
              .clamp_(0, num_features * num_sources - 1);
      const at::Tensor in_cell = position - cell_offsets.index_select(0, cell);
      const at::Tensor source = cell.remainder(num_sources),
                       feature = cell.div(num_sources, "floor");
      same = source.lt(local_size);
      own_index =
          own_cell_start.index_select(0, feature * local_size + source.clamp_max(local_size - 1))
              .add_(in_cell)
              .clamp_(0, lookup_keys - 1);
      const at::Tensor column = (source - local_size).clamp_(0, num_nodes - 2);
      record = cell_first_key.index_select(0, feature * (num_nodes - 1) + column)
                   .add_(in_cell)
                   .to(at::kLong)
                   .mul_(record_bytes / 8)
                   .add_(first_record.index_select(0, column));
    }
    at::Tensor values = tables.kjt_values.narrow(0, first, count);
    at::where_out(values, same, node_keys.index_select(0, own_index),
                  rail_keys.index_select(0, record.clamp(0, rail_keys.numel() - 1)));
    if (record_bytes == 16) {  // the weight: the low half of the record's second word
      const at::Tensor weight_index = record.mul(2).add_(2).clamp_(0, 2 * rail_keys.numel() - 1);
      const at::Tensor record_weight = rail_keys.view(at::kFloat).index_select(0, weight_index);
      at::Tensor weights = tables.kjt_weights.narrow(0, first, count);
      at::where_out(weights, same,
                    node_weights.defined() ? node_weights.index_select(0, own_index)
                                           : at::ones_like(record_weight),
                    record_weight);
    }
  }
  return tables;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
