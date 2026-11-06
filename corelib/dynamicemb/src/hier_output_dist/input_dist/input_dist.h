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
// The input dist of a call, in one step on the caller's stream (the in
// channel, InChannel::input_dist): the bucketize, the comm kernels and the
// planner's steps, and the call's status. It reads no device value on the
// host: what it gives (InDist) is the owner's KJT at the op's capacity, the
// status, and the output dist's plans (plan.h). The caller copies the status
// to the host once, at the call's second wait (StatusSlots).
#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <tuple>
#include <utility>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/config.h"
#include "hier_output_dist/common/device.cuh"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/input_dist/plan.h"
#include "hier_output_dist/input_dist/planner.h"

namespace dyn_emb {
namespace hier_output_dist {

// What the input dist of a call gives: the owner's KJT, whose values (and
// weights) hold lookup_keys entries, the first num_recv its own (num_recv:
// in the status); the call's status (StatusSlots); and the output dist's
// plans, the backward's but for its counts, which the status gives.
struct InDist {
  at::Tensor kjt_values, kjt_weights, kjt_lengths, kjt_offsets;  // weights: undefined without
  at::Tensor status;                                             // int64 [status_words]
  FwdPlan fwd_plan;
  BwdPlan bwd_plan;
};

// The in channel: the input dist of every call, on the caller's stream,
// under one sequence number (in_seq) the same on every rank. Its comm
// kernels (comm_kernels.cu) exchange with the node and rail peers; the
// planner's steps (planner.h) lay the call out between them.
class InChannel {
 public:
  // window and dev: the op's registered window and its GIN device
  // communicator, which the comm kernels take; layout: the window over this
  // rank's copy.
  InChannel(const Config &config, const Sizes &sizes, ncclWindow_t window, const ncclDevComm &dev,
            const WindowLayout &layout, int device);

  // The input dist of a call, under the next sequence number. Every rank
  // runs the same calls in the same order. A call that does not fit the
  // capacities, or whose keys were not index-deduplicated (kNeedKeyRows),
  // runs every step on what arrived and stays in bounds; its status says so
  // on every rank alike.
  InDist input_dist(const PlanInput &input);

 private:
  // The comm kernels (comm_kernels.cu), at the current call's sequence
  // number (seq_), on the current stream; each allocates its outputs there.

  // The node barrier, and the node's count vectors (int64): their sum over
  // the node and over the ranks before this one ([my_counts.numel()] each),
  // and each requester's counts for this rank ([local_size][num_features +
  // 1]), as planner.h's NodeCounts.
  std::tuple<at::Tensor, at::Tensor, at::Tensor> node_count_exchange(
      const at::Tensor &my_counts) const;
  // This rank's first *rows_to_push bucketized rows (keys, weights or
  // undefined, bucket_feature_of_row) into its targets' node_keys at
  // destination_base[bucket_feature] + row.
  void node_key_push(const at::Tensor &keys, const at::Tensor &weights,
                     const at::Tensor &bucket_feature_of_row, const at::Tensor &destination_base,
                     const at::Tensor &rows_to_push) const;
  // Tells the node this rank's keys are pushed, and waits until every
  // requester of the node told it.
  void node_done() const;
  // Sends each rail peer its count message (message: CountMessage::out_words,
  // the common part and the part for each node); returns the messages this
  // rank got, int64 [num_nodes][CountMessage::length].
  at::Tensor rail_count_exchange(const at::Tensor &message) const;
  // Puts this proxy's key_stage slices into its rail owners' rail_keys (each
  // [num_nodes], int64 but put_skip) and its source_offsets rows into its rail
  // proxies' owner_offsets, and waits for the other nodes' key streams.
  void rail_key_exchange(const at::Tensor &put_src, const at::Tensor &put_dst,
                         const at::Tensor &put_bytes, const at::Tensor &put_skip) const;
  // Publishes this rank's needs and requester offsets to its node (publish,
  // int64 [kNumNeeds + local_size * num_features]); returns every node peer's
  // needs (int64 [local_size][kNumNeeds]) and where each one's KJT holds this
  // requester's keys of each feature (int64 [local_size][num_features]).
  std::pair<at::Tensor, at::Tensor> node_status_exchange(const at::Tensor &publish) const;
  // Words of the node peers' `from` (GatherFrom), int32 [words][items]: item
  // i reads node peer peer[i]'s word first_word[i] of its proxy header (1
  // word), or entry first_word[i] of each array of its remote_map
  // (RemoteMap::kArrays words), word k into [k][i]; an item of a negative
  // peer keeps what the allocation held.
  at::Tensor peer_gather(GatherFrom from, const at::Tensor &peer,
                         const at::Tensor &first_word) const;

  ncclWindow_t window_;
  ncclDevComm dev_;
  Sizes sizes_;
  OwnerRules rules_;  // the bucketize's
  bool weights_, node_dedup_;
  int device_;
  int64_t seq_ = 0;  // the calls run so far: the current one's sequence number
  // This rank's buffers the steps read and write, views of its copy of the
  // window without an allocator (the window outlives them): the own part of
  // node_keys (and node_weights), and across nodes its remote part,
  // bucket_feature_of_occ, the buffers the proxy writes, rail_keys,
  // source_offsets and owner_offsets.
  at::Tensor own_keys_, own_weights_;
  at::Tensor remote_keys_, remote_weights_, bucket_feature_of_occ_;
  ProxyBuffers proxy_buffers_;
  at::Tensor rail_keys_, source_offsets_, owner_offsets_;
  // Constants of the op across nodes: the table of each feature (numbered
  // densely), the other nodes, and the words of the proxy headers this
  // requester reads.
  at::Tensor tables_, other_nodes_;
  PeerWords header_gather_;
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
