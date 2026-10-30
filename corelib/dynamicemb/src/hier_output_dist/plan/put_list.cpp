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
#include "hier_output_dist/plan/put_list.h"

#include <cstdint>

namespace dyn_emb {
namespace hier_output_dist {

PutList proxy_put_list(const at::Tensor &round_table, const at::Tensor &cell_uniq,
                       const at::Tensor &rail_cell_start, const PlanSizes &sizes) {
  const int64_t num_nodes = sizes.topo.num_nodes, num_features = sizes.num_features,
                pieces = sizes.rounds_cap * num_nodes, chunk_rows = sizes.chunk_rows,
                chunks_cap = chunk_bound(sizes), segments_cap = segment_bound(sizes);
  const auto long_options = cell_uniq.options().dtype(at::kLong);
  PutList list;
  // the pieces, round-major, then owner node (none: no keys)
  auto piece_field = [&](int field) {
    return round_table.select(2, field).reshape({-1}).to(at::kLong);
  };
  const at::Tensor piece_lo = piece_field(0), piece_hi = piece_field(1), piece_off = piece_field(2);
  const at::Tensor chunks = ceil_div(piece_hi - piece_lo, chunk_rows);
  const at::Tensor chunk_ends = chunks.cumsum(0);
  const at::Tensor first_chunk = chunk_ends - chunks;
  list.num_chunks = chunk_ends.select(0, pieces - 1);
  // A [chunk][feature] grid: each chunk's piece and keys, cut at the cells
  // (feature) of its owner node; a cumsum keeps the cuts that hold keys, in
  // grid order.
  const at::Tensor chunk = at::arange(chunks_cap, long_options);
  const at::Tensor piece = at::searchsorted(chunk_ends, chunk, /*out_int32=*/false, /*right=*/true)
                               .clamp_max(pieces - 1);
  const at::Tensor nth = chunk - first_chunk.index_select(0, piece);
  const at::Tensor piece_node = piece.remainder(num_nodes);
  const at::Tensor chunk_lo =
      (piece_lo.index_select(0, piece) + nth * chunk_rows).view({chunks_cap, 1});
  const at::Tensor chunk_hi =
      at::minimum(piece_hi.index_select(0, piece).view({chunks_cap, 1}), chunk_lo + chunk_rows);
  const at::Tensor cell_ends = at::cat({at::zeros({num_nodes, 1}, long_options),
                                        cell_uniq.view({num_nodes, num_features}).cumsum(1)},
                                       1);
  const at::Tensor ends_of = cell_ends.index_select(0, piece_node);  // [chunk][feature + 1]
  const at::Tensor cell_lo = ends_of.narrow(1, 0, num_features);
  const at::Tensor lo = at::maximum(chunk_lo, cell_lo);
  const at::Tensor hi = at::minimum(chunk_hi, ends_of.narrow(1, 1, num_features));
  const at::Tensor valid = lo.lt(hi).logical_and_(chunk.lt(list.num_chunks).view({chunks_cap, 1}));
  const at::Tensor segment = at::stack(
      {piece_off.index_select(0, piece).view({chunks_cap, 1}) + lo -
           piece_lo.index_select(0, piece).view({chunks_cap, 1}),   // its rows in proxy_rows
       rail_cell_start.index_select(0, piece_node) + lo - cell_lo,  // in the owner's KJT
       hi - lo, (first_chunk.index_select(0, piece) + nth).view({chunks_cap, 1}).expand_as(lo)},
      2);
  const at::Tensor place = valid.view({-1}).cumsum(0, at::kLong) - 1;
  list.bwd_put_list = at::empty({segments_cap + 1, 4}, cell_uniq.options().dtype(at::kInt))
                          .index_copy_(0, at::where(valid.view({-1}), place, segments_cap),
                                       segment.view({-1, 4}).to(at::kInt))
                          .narrow(0, 0, segments_cap);
  const at::Tensor segments_before = exclusive_ends(valid.sum({1}));  // [chunk + 1]
  list.bwd_put_start = at::cat({segments_before.index_select(0, first_chunk.clamp_max(chunks_cap)),
                                segments_before.narrow(0, chunks_cap, 1)})
                           .to(at::kInt);
  // a round's reducer tasks: its busiest owner node's chunks, for every owner
  // node, chunk_rows / kTaskRows tasks a chunk
  list.reduce_task_starts = exclusive_ends(chunks.view({sizes.rounds_cap, num_nodes}).amax({1}) *
                                           (num_nodes * (chunk_rows / kTaskRows)));
  list.round_table = round_table.clone();
  list.round_table.select(2, 3).copy_(first_chunk.view({sizes.rounds_cap, num_nodes}));
  return list;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
