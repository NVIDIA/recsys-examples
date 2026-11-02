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
// The requester's part of a call across nodes (the in channel's side
// stream): its maps to the rows of its keys, from what its node's proxies
// published (two peer_gathers: their headers, and its rows' words of their
// remote_map) and where its owners on its node put its cells. ATen ops;
// nothing waits for the device.
#pragma once

#include <ATen/ATen.h>

#include "hier_output_dist/plan/tensors.h"

namespace dyn_emb {
namespace hier_output_dist {

// What a peer_gather reads (GatherFrom says which words): for each item a
// node peer (negative: none) and its first int32 word to read there.
struct PeerWords {
  at::Tensor peer;        // int32 [items]
  at::Tensor first_word;  // int64 [items]
};
// The remote_map words of this requester's rows (row_cell, block_start: part
// 1's; destination_base: the node layout's), kGatherRemoteMap: a row of
// another node's owner is occurrence destination - lookup_keys of its proxy,
// the owner's rank on this node; it reads that entry of each RemoteMap array.
PeerWords remote_rows(const at::Tensor &row_cell, const at::Tensor &block_start,
                      const at::Tensor &destination_base, const Sizes &sizes);
// The words of the node's proxy headers this requester reads
// (kGatherProxyHeader): ProxyHeader's gathered words of each proxy in turn,
// a constant of the op.
PeerWords header_gather(const Sizes &sizes, const at::TensorOptions &options);

struct RequesterTables {
  at::Tensor src, dst;            // int32 [row_cell.numel()]: where each row's key is in the
                                  // forward (a row of its node: its owner's KJT; else its proxy's
                                  // proxy_rows), and where its gradient goes in the backward
  at::Tensor push_order;          // int32 [row_cell.numel()]: its rows at each proxy in their
                                  // push order
  at::Tensor push_order_off;      // int64 [local_size + 1]: where its rows at each proxy start
  at::Tensor push_tables;         // int32 [local_size][rounds_cap + 1][4]: ReqEntry at each
                                  // proxy, positions in push_order, groups in heavy_groups
  at::Tensor proxy_rounds;        // int64 [local_size]: each proxy's rounds
  at::Tensor proxy_round_tables;  // int32 [local_size][rounds_cap * num_nodes * 4]: RoundPiece
  at::Tensor fwd_task_starts;     // int64 [rounds_cap + 1]: its forward's tasks before each round
  at::Tensor bwd_task_starts;     // int64 [2 * rounds_cap + 1]: its backward's, heavy groups
                                  // then rows each round
  at::Tensor heavy_groups;        // int32 [group bound][2]: HeavyGroup, positions of push_order
  at::Tensor group_first_sub;     // int32 [group bound]: each group's first sub-task
  at::Tensor group_of_sub;        // int32 [sub-task bound]: the group of each sub-task
  at::Tensor group_subs;          // int64 0-dim: the sub-tasks of kHeavy positions
};
// words: int32 [3][row_cell.numel()], the remote_map words of its rows
// (remote_rows' gather; rows of its node's owners and past the valid rows hold
// anything); headers: int32 [local_size][ProxyHeader::gathered_words]
// (header_gather's);
// owner_cell_start: int64 [local_size][num_features], where its cell
// (feature, this requester) starts in each owner's KJT on its node
// (node_status_exchange).
RequesterTables requester_tables(const at::Tensor &words, const at::Tensor &headers,
                                 const at::Tensor &owner_cell_start, const at::Tensor &row_cell,
                                 const at::Tensor &cell_start, const at::Tensor &block_start,
                                 const Sizes &sizes);

}  // namespace hier_output_dist
}  // namespace dyn_emb
