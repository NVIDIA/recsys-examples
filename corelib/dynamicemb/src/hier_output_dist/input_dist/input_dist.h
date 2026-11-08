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

  // This rank's offsets into node_offsets at each node peer, the node barrier, and the layout of
  // the node's keys from the node's offsets. The call takes its sequence number (seq_) here, once
  // the kernel is launched: a call that raises before that takes none. delay_ns: the test hook's
  // wait between the barrier and the layout (delay_node_reads).
  NodeLayout node_exchange(const BucketizedKeys &bucketized, int64_t delay_ns);
  // The bucketize's gather: this rank's valid bucketized rows into their places in their
  // targets' node_keys, over NVLink (none under skip). With src (one node, int32 [rows]) also
  // each row's place in its owner's KJT.
  void node_key_push(const BucketizedKeys &bucketized, const c10::optional<at::Tensor> &weights,
                     const NodeLayout &node, const at::Tensor &src) const;
  // Tells the node this rank's keys are pushed, and waits until every
  // requester of the node told it.
  void node_done() const;
  // Sends every rail peer this rank's count message (CountMessage, from its node layout `node`
  // and its proxy plan), gets theirs, and returns the rail's layout from the rail's messages.
  RailLayout rail_exchange(const NodeLayout &node, const ProxyPlan &proxy) const;
  // Puts this proxy's key_stage slices (put_src: where each starts there, int64 [num_nodes]) into
  // its rail owners' rail_keys, where `rail` says (put_dst, put_bytes, put_skip). It does not wait.
  void rail_key_push(const at::Tensor &put_src, const RailLayout &rail) const;
  // Publishes this rank's needs and requester offsets (the rail layout's) to its node; returns
  // every node peer's needs (int64 [local_size][kNumNeeds]) and where each one's KJT holds this
  // requester's keys of each feature (int64 [local_size][num_features]).
  std::pair<at::Tensor, at::Tensor> node_status_exchange(const RailLayout &rail) const;
  // Words of the node peers' `from` (GatherFrom), int32 [words][items]: item
  // i reads node peer peer[i]'s word first_word[i] of its proxy header (1
  // word), or entry first_word[i] of each array of its remote_map
  // (RemoteMap::kArrays words), word k into [k][i]; an item of a negative
  // peer keeps what the allocation held.
  at::Tensor peer_gather(GatherFrom from, const at::Tensor &peer,
                         const at::Tensor &first_word) const;
  // Waits until every rail proxy's key records of the call are in this rank's rail_keys, and
  // flushes this rank's puts on the rail (the NIC read its key_stage). Just before the owner's
  // tables, the first reader of rail_keys.
  void rail_done() const;

  ncclWindow_t window_;
  ncclDevComm dev_;
  Sizes sizes_;
  at::Tensor dist_types_;  // int32 [num_features] on the GPU: the bucketize's rule of each feature
  // int64 [kFanoutWords] on the GPU: the fan-out kernel's counter and CTA ticket (fanout_tables),
  // zeroed once; each call's kernel leaves them at 0
  at::Tensor fanout_words_;
  bool weights_, node_dedup_;
  int device_;
  int64_t seq_ = 0;  // the calls run so far: the current one's sequence number
  // This rank's buffers the steps read and write, views of its copy of the
  // window without an allocator (the window outlives them): the own part of
  // node_keys (and node_key_weights), and across nodes its remote part,
  // node_bucket_feature_of_row, the buffers the proxy writes and rail_keys.
  at::Tensor own_node_keys_, own_node_key_weights_;
  at::Tensor remote_node_keys_, remote_node_key_weights_, node_bucket_feature_of_row_;
  ProxyBuffers proxy_buffers_;
  at::Tensor rail_keys_;
  // Constants of the op across nodes: the table of each feature (numbered
  // densely), the other nodes, and the words of the proxy headers this
  // requester reads.
  at::Tensor tables_, other_nodes_;
  PeerWords header_gather_;
};

// A test hook (testing.cu): each later input dist of this process waits `ns` on the GPU twice (0:
// no wait): in node_exchange between the node barrier and the read of node_offsets, and between
// node_done and the first read of node_keys. It makes a rank read the node's offsets and keys
// late.
void delay_node_reads(int64_t ns);

}  // namespace hier_output_dist
}  // namespace dyn_emb
