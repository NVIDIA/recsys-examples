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
// The proxy's backward put list (the in channel's side stream, across
// nodes): the summed gradient rows of its node-unique keys go back to their
// owners in IB chunks of chunk_rows keys, piece after piece (round-major,
// then owner node), each chunk as one segment per feature cell it overlaps.
// ATen ops; nothing waits for the device.
#pragma once

#include <ATen/ATen.h>

#include <cstdint>

#include "hier_output_dist/plan/tensors.h"

namespace dyn_emb {
namespace hier_output_dist {

// The most IB chunks of a call (a piece of k keys takes ceil(k / chunk_rows),
// and a call has at most remote_keys keys in rounds_cap * num_nodes pieces),
// and the most segments (each cell start inside a chunk adds one).
inline int64_t chunk_bound(const PlanSizes &sizes) {
  return sizes.remote_keys / sizes.chunk_rows + sizes.rounds_cap * sizes.topo.num_nodes + 1;
}
inline int64_t segment_bound(const PlanSizes &sizes) {
  return chunk_bound(sizes) + sizes.topo.num_nodes * sizes.num_features;
}

struct PutList {
  at::Tensor bwd_put_list;        // int32 [segment_bound][4]: PutSeg {src row in proxy_rows,
                                  // dst row in the owner's KJT, rows, chunk}
  at::Tensor bwd_put_start;       // int32 [rounds_cap * num_nodes + 1]: the segments of each
                                  // (round, owner node)
  at::Tensor round_table;         // int32 [rounds_cap][num_nodes][4]: RoundPiece with each
                                  // piece's first chunk
  at::Tensor reduce_task_starts;  // int64 [rounds_cap + 1]: the reducers' tasks before each
                                  // round
  at::Tensor num_chunks;          // int64 0-dim: the IB chunks
};
// From the proxy's round table (its pieces, chunk 0), its node-unique keys
// per (owner node, feature), and where its cells start in each rail owner's
// KJT (the rail layout's rail_cell_start).
PutList proxy_put_list(const at::Tensor &round_table, const at::Tensor &cell_uniq,
                       const at::Tensor &rail_cell_start, const PlanSizes &sizes);

}  // namespace hier_output_dist
}  // namespace dyn_emb
