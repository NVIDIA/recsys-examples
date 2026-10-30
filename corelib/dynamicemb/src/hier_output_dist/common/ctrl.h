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
// The ctrl window: the control words of both channels and the device state
// they keep across calls (Ctrl), the count message the rail peers exchange,
// and a channel's sequence number (Channel).
#pragma once

#include <cstdint>

namespace dyn_emb {
namespace hier_output_dist {

// Word offsets (uint64 units) of the ctrl window. It holds every control word
// of both channels and the device state they keep across calls; its size
// depends only on local_size, num_nodes and num_features. Every control
// word is tagged with its channel's sequence number, is a running total, or
// is back at its start value when the kernel that uses it ends. The data
// words (in_meta, cell_start, the needs after in_done, the count messages)
// are written whole in each call, before a tagged word, the node barrier or
// a signal publishes them. So none is ever reset, and a call that overflowed
// leaves all of them valid for the next.
struct Ctrl {
  // in channel
  int in_started;     // in_seq of this rank's current node_count_exchange (its node barrier)
  int in_meta[2];     // by in_seq parity: its count vector, local_size * (num_features + 1)
                      // words (plan/node.h)
  int in_written;     // [local_size] in_seq whose keys each requester pushed into node_keys
  int cell_start;     // (num_nodes > 1) [local_size][num_features] where each same-node
                      // requester's cell (feature, requester) starts in this rank's KJT
  int in_done;        // (num_nodes > 1) in_seq of this rank's node_status_exchange; then
                      // [kNumNeeds] its rail's needs
  int count_msg_out;  // (num_nodes > 1) the count message this rank sends its rail peers
  int count_msgs;     // (num_nodes > 1) [2][num_nodes] the count messages of the rail
                      // peers, by in_seq parity and source node (this rank's own too)
  int msg_words;      // words of a count message's slot (whole 16-byte units)
  // out channel
  int out_started;     // out_seq of this rank's current output kernel (node barrier)
  int pushed;          // [local_size] out_seq whose gradient rows each requester pushed
                       // into lookup_rows
  int pulled;          // out_seq after this rank's last pull of a forward (exit barrier)
  int arrive;          // CTAs of this rank's out kernels that reached their end, a
                       // running total (local)
  int tasks;           // [kNumQueues] task queues of the current out kernel, back
                       // at 0 when it ends (local)
  int landed_rows;     // (num_nodes > 1) [num_nodes] (out_seq << 32) | rows of each owner
                       // node's that landed in this proxy's proxy_rows during that
                       // forward (the relay)
  int fwd_tot;         // (num_nodes > 1) [num_nodes] rows each owner node put into
                       // proxy_rows in earlier forwards: what its signal held when this
                       // forward began (local)
  int bwd_tot;         // (num_nodes > 1) [num_nodes] the same of each proxy node's
                       // gradient rows (local)
  int grant_sent;      // (num_nodes > 1) [num_nodes] the grant this proxy sent each rail
                       // owner (local)
  int bwd_ready_sent;  // (num_nodes > 1) [num_nodes] the bwd_ready this owner sent each
                       // rail proxy (local)
  int posted;          // (num_nodes > 1) [2] backward puts this proxy posted in calls of
                       // more than one round, per half of proxy_rows: what its put
                       // counters count (local)
  int done;            // (num_nodes > 1) [local_size][2] rows each same-node requester
                       // pulled from this proxy's proxy_rows in forwards of more than one
                       // round, of its even and its odd rounds: running totals, which
                       // tell the relay a round's half is free again
  int half_free;       // (num_nodes > 1) grant_of(out_seq, rnd): this proxy's node may push
                       // the gradient rows of rounds up to rnd of that backward into its
                       // proxy_rows (rounds 0 and 1 need none)
  // teardown
  int closing;  // this rank reads no window any more
  int closed;   // every node peer, and every rail peer (closing signals),
                // said so (written locally)
  int words;    // size, in words
};

// The words (int64) of a count message, all a rail peer needs to lay out
// every owner of the rail (plan/rail.h, count_message): its node's largest
// remote part, its rounds and its node-unique keys per (owner node, feature)
// as proxy, and the keys it receives from its node per feature as owner.
__host__ __device__ inline int msg_len(int num_nodes, int num_features) {
  return 2 + (num_nodes + 1) * num_features;
}

// Kernels of one channel run one at a time, in issue order, and share a
// sequence number, the same on every rank. The in channel is the input dist
// (side stream), the out channel the output dist's comm kernels (the
// caller's stream).
struct Channel {
  int64_t seq;  // calls (in) or comm kernels (out) issued so far
  int started;  // the word its node barrier publishes seq in (Ctrl)
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
