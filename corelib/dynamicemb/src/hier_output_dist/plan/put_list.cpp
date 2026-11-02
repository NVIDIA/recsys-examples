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
#include <tuple>

namespace dyn_emb {
namespace hier_output_dist {

namespace {

// A cut's sort key: its node, then its key (below 2^31).
constexpr int kNodeShift = 32;
constexpr int64_t kKeyMask = (int64_t{1} << kNodeShift) - 1;

}  // namespace

PutLists put_lists(const at::Tensor &round_table, const at::Tensor &num_rounds,
                   const at::Tensor &cell_uniq, const at::Tensor &rail_cells,
                   const at::Tensor &fwd_pieces, const at::Tensor &fwd_rounds,
                   const at::Tensor &cells_in, const at::Tensor &cell_stage, const Sizes &sizes) {
  const int64_t num_nodes = sizes.topo.num_nodes, num_features = sizes.num_features,
                rounds_cap = sizes.rounds_cap, chunk_rows = sizes.chunk_rows;
  // The two lists as one of 2 * num_nodes nodes: the backward's owner nodes,
  // then the forward's proxy nodes. (A call that does not fit may read pieces
  // no one wrote: they are clamped.)
  const int64_t nodes = 2 * num_nodes, num_pieces = rounds_cap * nodes;
  const auto long_options = cell_uniq.options().dtype(at::kLong);
  const at::Tensor pieces =  // [rounds_cap][nodes]: lo, hi, the row of lo in proxy_rows
      at::cat({round_table.narrow(2, 0, 3).to(at::kLong), fwd_pieces}, 1).clamp_(0, kKeyMask >> 1);
  const at::Tensor lo = pieces.select(2, 0);
  const at::Tensor rounds = at::cat({num_rounds.expand({num_nodes}), fwd_rounds});
  const at::Tensor count = at::cat({cell_uniq.view({num_nodes, num_features}), cells_in});
  const at::Tensor node_base =
      at::arange(nodes, long_options).bitwise_left_shift_(kNodeShift).view({nodes, 1});
  PutLists lists;
  // Each piece's IB chunks and its first chunk, pieces round-major.
  const at::Tensor piece_chunks = ceil_div((pieces.select(2, 1) - lo).clamp_min_(0), chunk_rows);
  const at::Tensor chunk_ends = piece_chunks.view({-1}).cumsum(0);
  const at::Tensor first_chunk = chunk_ends.view({rounds_cap, nodes}) - piece_chunks;
  lists.num_chunks = chunk_ends.select(0, num_pieces - 1);
  // The cuts, by (node, key): the first key of each IB chunk (past the
  // chunks: after every node) and the end of each cell. Segment i goes from
  // cut i to cut i + 1, and holds rows when both are of one node.
  const at::Tensor chunk = at::arange(
      chunk_bound(sizes.remote_keys, sizes) + chunk_bound(sizes.lookup_keys, sizes), long_options);
  const at::Tensor piece = at::searchsorted(chunk_ends, chunk, /*out_int32=*/false, /*right=*/true)
                               .clamp_max_(num_pieces - 1);
  const at::Tensor piece_base =
      (node_base.view({1, nodes}) + lo - first_chunk * chunk_rows).view({-1});
  const at::Tensor chunk_cut =
      at::where(chunk.lt(lists.num_chunks), piece_base.index_select(0, piece) + chunk * chunk_rows,
                nodes << kNodeShift);
  const at::Tensor cell_ends = count.cumsum(1);
  const at::Tensor cell_lo = cell_ends - count;
  const at::Tensor sorted =
      std::get<0>(at::cat({chunk_cut, (node_base + cell_ends).view({-1})}).sort());
  const int64_t cuts = sorted.numel() - 1;
  const at::Tensor key = sorted.narrow(0, 0, cuts), next = sorted.narrow(0, 1, cuts);
  const at::Tensor node = key.bitwise_right_shift(kNodeShift), first = key.bitwise_and(kKeyMask);
  const at::Tensor rows = at::where(next.bitwise_right_shift(kNodeShift).eq(node),
                                    next.bitwise_and(kKeyMask) - first, 0);
  // Each segment's piece and cell: the last of its node's that starts at or
  // before it (a node's pieces past its rounds start at its end).
  const at::Tensor totals = cell_ends.select(1, num_features - 1);
  const at::Tensor piece_start =
      at::where(at::arange(rounds_cap, long_options).view({rounds_cap, 1}).lt(rounds), lo, totals);
  const at::Tensor at_piece =  // node-major: lo, its row, the first chunk
      at::stack({lo, pieces.select(2, 2), first_chunk}, 2)
          .transpose(0, 1)
          .reshape({-1, 3})
          .index_select(0, (at::searchsorted((node_base + piece_start.t()).reshape({-1}), key,
                                             /*out_int32=*/false, /*right=*/true) -
                            1)
                               .clamp_(0, num_pieces - 1));
  const at::Tensor in_piece = first - at_piece.select(1, 0);
  const at::Tensor piece_row = at_piece.select(1, 1) + in_piece;
  const at::Tensor seg_chunk = at_piece.select(1, 2) + in_piece.div(chunk_rows, "floor");
  const at::Tensor at_cell =  // the cell's first key and its row in the owner's KJT
      at::stack({cell_lo, at::cat({rail_cells, cell_stage})}, 2)
          .view({-1, 2})
          .index_select(0, (at::searchsorted((node_base + cell_lo).view({-1}), key,
                                             /*out_int32=*/false, /*right=*/true) -
                            1)
                               .clamp_(0, nodes * num_features - 1));
  const at::Tensor kjt_row = at_cell.select(1, 1) + first - at_cell.select(1, 0);
  const at::Tensor forward = node.ge(num_nodes);
  // The segments that hold rows, in order, and where each (node, round)'s
  // first one is.
  const at::Tensor kept = rows.gt(0);
  const at::Tensor kept_before = exclusive_ends(kept);
  lists.list = at::empty({cuts + 1, 4}, long_options.dtype(at::kInt))
                   .index_copy_(0, at::where(kept, kept_before.narrow(0, 0, cuts), cuts),
                                at::stack({at::where(forward, kjt_row, piece_row),
                                           at::where(forward, piece_row, kjt_row), rows, seg_chunk},
                                          1)
                                    .to(at::kInt))
                   .narrow(0, 0, cuts);
  const at::Tensor starts =
      kept_before
          .index_select(
              0, at::searchsorted(
                     sorted,
                     (node_base + at::cat({piece_start.t(), totals.unsqueeze(1)}, 1)).view({-1}))
                     .clamp_max_(cuts))
          .view({nodes, rounds_cap + 1})
          .to(at::kInt);
  lists.bwd_start = starts.narrow(0, 0, num_nodes);
  lists.fwd_start = starts.narrow(0, num_nodes, num_nodes);
  // a round's reducer tasks: its busiest owner node's chunks, for every owner
  // node, chunk_rows / kTaskRows tasks a chunk
  lists.reduce_task_starts = exclusive_ends(piece_chunks.narrow(1, 0, num_nodes).amax({1}) *
                                            (num_nodes * (chunk_rows / kTaskRows)));
  lists.round_table = round_table.clone();
  lists.round_table.select(2, 3).copy_(first_chunk.narrow(1, 0, num_nodes));
  return lists;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
