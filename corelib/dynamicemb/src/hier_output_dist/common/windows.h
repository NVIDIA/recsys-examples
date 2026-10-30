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
// The symmetric windows of a sharding, their regions (the same on every
// rank) and the capacities they are sized by.
#pragma once

#include <nccl.h>

#include <cstdint>

namespace dyn_emb {
namespace hier_output_dist {

// The three symmetric windows of an op, allocated once at its capacities
// (host/windows.h), in this order, which is also their order in the
// communicator's symmetric address space:
//   ctrl  the control words of both channels;
//   keys  the in channel's data;
//   rows  the out channel's data, lookup_rows first: a view of it may outlive
//         the op, and its window's memory then stays.
enum WinId : int { kWinCtrl = 0, kWinKeys, kWinRows, kNumWins };

// Regions of the windows, the same size on every rank, so a rank addresses a
// peer's region with its own offsets. A region of 0 bytes is absent; the ones
// marked (num_nodes > 1) exist across nodes only.
//   ctrl          (ctrl) control words (Ctrl)
//   node_keys     (keys) keys this rank's node pushed to it (node_key_push):
//                 its own part at [0, lookup_keys), the keys of its node for
//                 it as owner, in its KJT's order ([feature][requester]
//                 [requester's order]); then (num_nodes > 1) its remote part
//                 at [lookup_keys, lookup_keys + remote_keys), those it
//                 deduplicates as proxy, requester-major, then owner node,
//                 then [feature][position]. A remote occurrence is named by
//                 its index occ in the remote part.
//   node_weights  (keys) their weights, when the requester KJT carries them
//   remote_cells  (keys, num_nodes > 1) per remote occurrence, int16: its
//                 owner node * num_features + feature
//   remote_map    (keys, num_nodes > 1) per remote occurrence occ, three int32
//                 arrays of remote_keys: its key's row in proxy_rows (bit 31:
//                 a heavy key), its slot in proxy_rows (the key's row, a
//                 duplicate's or this requester's partial slot) and its
//                 position in its requester's push order
//   proxy_hdr     (keys, num_nodes > 1) the proxy's header for its node's
//                 requesters: its rounds, its round table and each
//                 requester's push tables (plan/proxy.h)
//   key_stage     (keys, num_nodes > 1) the node-unique key records of each
//                 owner node, each slice after its round pieces: what one key
//                 put sends
//   rail_keys     (keys, num_nodes > 1) the key records the rail proxies put,
//                 one slice per source node in node order, each its round
//                 pieces, then its records in uniq order
//   lookup_rows   (rows) the owner's lookup output, row i for its KJT's key i;
//                 the backward writes the gradient of that output over it
//   proxy_rows    (rows, num_nodes > 1) the proxy's rows: the rows of the
//                 node-unique keys the rail owners put (forward), the gradient
//                 rows of its node's requesters and their sums (backward)
//   push_flags    (rows, num_nodes > 1) the requesters' push-task flags; zeroed
//                 when the window is created
enum Region : int {
  kCtrl = 0,
  kNodeKeys,
  kNodeWeights,
  kRemoteCells,
  kRemoteMap,
  kProxyHdr,
  kKeyStage,
  kRailKeys,
  kLookupRows,
  kProxyRows,
  kPushFlags,
  kNumRegions
};
struct Win {
  ncclWindow_t handle[kNumRegions];  // window each region lives in (nullptr: not created)
  uint64_t off[kNumRegions];         // byte offset of the region inside that window
  uint64_t bytes[kNumRegions];       // its size
  int64_t lookup_keys_cap;           // keys an owner looks up (rows of lookup_rows)
  int64_t num_proxy_rows;            // rows of proxy_rows: a proxy whose rows of a call need
                                     // more takes more rounds
  int rounds_cap;                    // rounds a round table holds
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
