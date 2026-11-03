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
// The input dist of a call, in three steps on the caller's stream (the in
// channel, InChannel): part 1 when the input dist is called (part_one,
// local), the rest at its first wait (start), and the call's one host sync
// at its second wait (wait). Each step returns the data the next one takes
// (PartOne, Pending), and the last one what the caller gets (InDistResult):
// the owner's KJT, and the output dist's plans (plan.h).
#pragma once

#include <ATen/ATen.h>
#include <c10/core/Stream.h>

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/config.h"
#include "hier_output_dist/common/device.cuh"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/input_dist/plan.h"
#include "hier_output_dist/input_dist/planner.h"

namespace dyn_emb {
namespace hier_output_dist {

// Part 1 of a call: its keys by owner, and under index dedup its fan-out.
struct PartOne {
  PlanInput input;  // held until the call's host sync: part 1 read it on the caller's stream
  Partition part;
  FanoutTables fanout;  // index dedup's, or none
};

// A call between its waits: what start() queued, which wait() reads.
struct Pending {
  PlanInput input;    // held until the host sync
  at::Tensor status;  // int64 [status_words]: what the host reads of the call (StatusSlots)
  // the owner's KJT; of its values (and weights) lookup_keys, the first num_recv its own
  at::Tensor kjt_values, kjt_weights, kjt_lengths, kjt_offsets;
  at::Tensor occurrence_count;  // across nodes, the proxy's remote occurrences
  FwdPlan fwd_plan;
  BwdPlan bwd_plan;  // but for its counts, which the status gives
};

// What the input dist of a call gives the caller after its one host sync.
// The status is the same on every rank. A call that does not fit the
// capacities has its status alone: the caller moves it some other way.
struct InDistResult {
  bool fits = false;
  std::map<std::string, int64_t> needs;  // what the call needs of each capacity (the most of
                                         // any rank)
  std::vector<std::string> overflows;    // the capacities it overflowed
  int64_t sent = 0;                      // rows of the input KJT this rank sends, its valid keys
  std::vector<int64_t> input_length_per_key;  // the per-feature totals of the KJT
                                              // construct_jagged_tensors reads
  // When it fits: the owner's KJT, [num_features][num_sources] cells, and the
  // output dist's plans.
  int64_t num_recv = 0;  // keys this owner looks up
  int64_t num_sources = 0;
  std::vector<int64_t> length_per_key;
  at::Tensor values, weights, lengths, offsets;  // weights: undefined when the KJT has none
  at::Tensor occurrence_count;                   // across nodes, the proxy's remote occurrences
                                                 // (the benchmark's)
  FwdPlan fwd_plan;
  BwdPlan bwd_plan;

  // The tensors' use on `stream` (the caching allocator's).
  void record_stream(c10::Stream stream) const;
};

// The in channel: the input dist of every call, on the caller's stream,
// under one sequence number (in_seq) the same on every rank. Its comm
// kernels (comm_kernels.cu) exchange with the node and rail peers; the
// planner's steps (planner.h) lay the call out between them.
class InChannel {
 public:
  // channel: the part of the kernels' arguments both channels share, over
  // the registered window; window: the window over this rank's copy.
  InChannel(const Config &config, const Sizes &sizes, const ChannelStatic &channel,
            const WindowLayout &window, int device);

  // Part 1 of a call. Local: no sync, no collective, no sequence number, so
  // the caller may drop it.
  PartOne part_one(PlanInput input) const;
  // The first wait: the rest of the call after part 1 (the steps of the
  // node, across nodes those of the proxy, the rail and the requester, then
  // the call's tables and status), under the next sequence number. It never
  // reads a device value on the host. Every rank starts the same calls in
  // the same order.
  Pending start(const PartOne &part);
  // The second wait, the call's one host sync: a copy of its status.
  InDistResult wait(const Pending &pending) const;

 private:
  ChannelStatic channel_;
  Sizes sizes_;
  OwnerRules rules_;  // part 1's
  bool weights_, node_dedup_;
  int device_;
  int64_t seq_ = 0;  // the calls started
  // This rank's buffers the steps read and write, views of its copy of the
  // window without an allocator (the window outlives them): the own part of
  // node_keys (and node_weights), and across nodes its remote part,
  // remote_cells, the buffers the proxy writes, rail_keys, cell_stage and
  // rail_cells.
  at::Tensor own_keys_, own_weights_;
  at::Tensor remote_keys_, remote_weights_, remote_cells_;
  ProxyBuffers proxy_buffers_;
  at::Tensor rail_keys_, cell_stage_, rail_cells_;
  // Constants of the op across nodes: the table of each feature (numbered
  // densely), the other nodes, and the words of the proxy headers this
  // requester reads.
  at::Tensor tables_, other_nodes_;
  PeerWords header_gather_;
};

// The comm kernels (comm_kernels.cu), on the current stream at call seq;
// each allocates its outputs there.

// The node barrier, and the node's count vectors (int64): their sum over the
// node and over the ranks before this one ([my_counts.numel()] each), and
// each requester's counts for this rank ([local_size][num_features + 1]),
// as planner.h's NodeCounts.
std::tuple<at::Tensor, at::Tensor, at::Tensor> launch_node_count_exchange(
    const ChannelStatic &channel, int64_t seq, const at::Tensor &my_counts);
// This rank's first *rows_to_push rows of part 1 (keys, weights or undefined,
// row_cell) into its targets' node_keys at destination_base[cell] + row.
void launch_node_key_push(const ChannelStatic &channel, const at::Tensor &keys,
                          const at::Tensor &weights, const at::Tensor &row_cell,
                          const at::Tensor &destination_base, const at::Tensor &rows_to_push);
// Tells the node this rank's keys are pushed, and waits until every
// requester of the node told it.
void launch_node_done(const ChannelStatic &channel, int64_t seq);
// Sends each rail peer its count message (message: CountMessage::out_words,
// the common part and the part for each node); returns the messages this
// rank got, int64 [num_nodes][CountMessage::length].
at::Tensor launch_rail_count_exchange(const ChannelStatic &channel, int64_t seq,
                                      const at::Tensor &message);
// Puts this proxy's key_stage slices into its rail owners' rail_keys (each
// [num_nodes], int64 but put_skip) and its cell_stage rows into its rail
// proxies' rail_cells, and waits for the other nodes' key streams.
void launch_rail_key_exchange(const ChannelStatic &channel, int64_t seq, const at::Tensor &put_src,
                              const at::Tensor &put_dst, const at::Tensor &put_bytes,
                              const at::Tensor &put_skip);
// Publishes this rank's needs and cell starts to its node (publish, int64
// [kNumNeeds + local_size * num_features]); returns every node peer's needs
// (int64 [local_size][kNumNeeds]) and where each one's KJT holds this
// requester's cells (int64 [local_size][num_features]).
std::pair<at::Tensor, at::Tensor> launch_node_status_exchange(const ChannelStatic &channel,
                                                              int64_t seq,
                                                              const at::Tensor &publish);
// Words of the node peers' `from` (GatherFrom), int32 [words][items]: item i
// reads node peer peer[i]'s word first_word[i] of its proxy header (1 word),
// or entry first_word[i] of each array of its remote_map (RemoteMap::kArrays
// words), word k into [k][i]; an item of a negative peer keeps what the
// allocation held.
at::Tensor launch_peer_gather(const ChannelStatic &channel, GatherFrom from, const at::Tensor &peer,
                              const at::Tensor &first_word);

}  // namespace hier_output_dist
}  // namespace dyn_emb
