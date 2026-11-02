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
// The put lists of a call across nodes (the in channel's side stream), made
// in one step: the rows that cross the rail between an owner's KJT and a
// proxy's proxy_rows. The proxy's backward list sends the summed gradient
// rows of its node-unique keys to their owners; the owner's forward list
// sends its rows to the proxies that asked for them. Each cuts a node's keys
// at its IB chunks and at its feature cells, so that a segment lies in one
// chunk and one cell. ATen ops; nothing waits for the device.
#pragma once

#include <ATen/ATen.h>

#include <cstdint>

#include "hier_output_dist/plan/tensors.h"

namespace dyn_emb {
namespace hier_output_dist {

// The most IB chunks of a list's pieces, for at most `keys` keys in all: a
// piece of k keys takes ceil(k / chunk_rows), and a list has rounds_cap *
// num_nodes pieces.
inline int64_t chunk_bound(int64_t keys, const Sizes &sizes) {
  return keys / sizes.chunk_rows + sizes.rounds_cap * sizes.topo.num_nodes + 1;
}

// Both lists, in one tensor. A list's segments are in (node, key) order:
// node after node, round by round, and those of an IB chunk follow each
// other. IB chunks are numbered piece by piece, round after round, the
// backward's pieces before the forward's in each round.
struct PutLists {
  at::Tensor list;                // int32 [bound][4]: PutSeg {src, dst, rows, chunk}; the
                                  // backward's src is a row of proxy_rows and its dst a row
                                  // of the owner's KJT, the forward's the other way round
  at::Tensor bwd_start;           // int32 [num_nodes][rounds_cap + 1]: the backward list's first
                                  // segment of each (owner node, round); [node][rounds_cap]:
                                  // the node's end
  at::Tensor fwd_start;           // int32 [num_nodes][rounds_cap + 1]: the forward list's of
                                  // each (proxy node, round)
  at::Tensor round_table;         // int32 [rounds_cap][num_nodes][4]: the proxy's RoundPiece
                                  // with each piece's first chunk
  at::Tensor reduce_task_starts;  // int64 [rounds_cap + 1]: the reducers' tasks before each
                                  // round
  at::Tensor num_chunks;          // int64 0-dim: the IB chunks of both lists
};
// The backward list from the proxy's round table (its pieces, chunk 0) and
// rounds, its node-unique keys per (owner node, feature) and where its cells
// start in each rail owner's KJT (rail_cells, int64 [num_nodes]
// [num_features], as the owners put them). The forward list from each
// source node's round pieces (fwd_pieces, int64 [rounds_cap][num_nodes][3]:
// lo, hi and the row of lo in the proxy's proxy_rows; zeros past its rounds
// fwd_rounds), its keys per feature (cells_in) and where those cells start
// in the owner's KJT (cell_stage).
PutLists put_lists(const at::Tensor &round_table, const at::Tensor &num_rounds,
                   const at::Tensor &cell_uniq, const at::Tensor &rail_cells,
                   const at::Tensor &fwd_pieces, const at::Tensor &fwd_rounds,
                   const at::Tensor &cells_in, const at::Tensor &cell_stage, const Sizes &sizes);

}  // namespace hier_output_dist
}  // namespace dyn_emb
