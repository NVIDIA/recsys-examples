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
// The comm kernels of the in channel (the first wait, on the op's side
// stream): the input dist's only custom kernels, each one exchange with the
// node peers or the rail peers. The planner's steps (plan/) run between them
// as ATen ops on the same stream, in the order of InChannel::start:
//   node_count_exchange   the node barrier of the call, and the node's
//                         counts as this rank needs them;
//   node_key_push         this rank's keys into its targets' node_keys over
//                         NVLink;
//   node_done             tells each target its keys are there, and waits
//                         until every requester of the node told it;
//   rail_count_exchange   (across nodes) its count message to each rail
//                         peer, and theirs;
//   rail_key_exchange     (across nodes) its key records to each rail owner
//                         and its cell starts to each rail proxy, and theirs;
//   node_status_exchange  (across nodes) its rail's needs and cell starts to
//                         its node, and every node peer's;
//   peer_gather           (across nodes) words of the node peers' remote_map
//                         or proxy header.
// A kernel that waits runs on one CTA and waits only for the same kernel (the
// same seq) of its node or rail peers; the others never wait. Its threads (or
// lanes) take the peers in turn, so any number of them. Memory order:
// what peers wrote is read only after an acquire of their flag or signal, by
// the comm kernel that acquired it (with ld.global.cg) or by a later kernel
// on the stream (peer_gather with ld.global.cg, the planner's ATen steps with
// plain loads). What this rank writes for its peers (its own stores, or
// ATen's earlier on the stream) is ordered before the flag it releases or the
// put it makes: by fence.sc.sys, or in node_count_exchange by the release
// itself after the CTA's barrier.
#pragma once

#include <nccl_device.h>

#include <cstdint>

#include "hier_output_dist/common/channel_args.h"
#include "hier_output_dist/common/constants.h"
#include "hier_output_dist/common/ctrl.h"
#include "hier_output_dist/common/gin_signals.h"
#include "hier_output_dist/common/keys.h"
#include "hier_output_dist/common/status.h"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/device/memory_order.cuh"
#include "hier_output_dist/device/node_sync.cuh"
#include "hier_output_dist/device/node_window.cuh"
#include "hier_output_dist/device/rail.cuh"
#include "hier_output_dist/device/spin.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// The argument of a comm kernel that waits: the part both channels share,
// and the in channel at the call's sequence number, which NodeBarrier and
// PeerFlags read.
struct CommArgs : ChannelStatic {
  Channel ch;
};

// 1 CTA: this rank's count vector (CountVector, plan/node.h) into its
// count_vector of the call, the node barrier, then the node's count vectors
// as this rank needs them: their sum (totals) and that of the ranks before
// it (before), both [words], and each requester's counts for this rank (mine
// [local_size][num_features + 1]: its keys for its remote part, then for
// each feature of its own part). Passing the barrier also means every peer
// finished its previous call on its side stream, whose steps read what this
// call writes into their copies of the window.
__global__ void node_count_exchange_kernel(CommArgs args, const int64_t *my_counts, int64_t *totals,
                                           int64_t *before, int64_t *mine) {
  const int local_size = args.topo.local_size, num_features = (int)args.num_features,
            me = args.topo.local_rank, tid = threadIdx.x;
  const CountVector format{local_size, num_features};
  const int64_t words = format.words();
  const NodeWindow node(args.window, me);
  const CtrlLayout ctrl = node.layout(args).ctrl;
  int64_t *counts = ctrl.count_vector(args.ch.seq);
  for (int i = tid; i < words; i += blockDim.x) counts[i] = my_counts[i];
  __syncthreads();
  const NodeBarrier<CommArgs> barrier(args, node, ctrl.in_started());
  if (tid == 0) barrier.publish();
  for (int peer = tid; peer < local_size; peer += blockDim.x)
    barrier.wait(peer, "in-channel node barrier");
  __syncthreads();
  auto count_of = [&](int peer, int64_t word) {
    return (int64_t)__ldcg((const unsigned long long *)(node.peer(counts, peer) + word));
  };
  for (int i = tid; i < words; i += blockDim.x) {
    int64_t total = 0, earlier = 0;
    for (int peer = 0; peer < local_size; ++peer) {
      const int64_t count = count_of(peer, i);
      total += count;
      if (peer < me) earlier += count;
    }
    totals[i] = total;
    before[i] = earlier;
  }
  for (int i = tid; i < local_size * (num_features + 1); i += blockDim.x) {
    const int peer = i / (num_features + 1), column = i % (num_features + 1);
    mine[i] = count_of(peer, column == 0 ? format.remote(me) : format.own(me, column - 1));
  }
}

// Every CTA, a thread per row: row `row` of part 1 (row < *rows_to_push) goes
// to destination_base[cell] + row of node_keys (and node_weights) of its
// target, the owner's local rank; a destination from remote_start on is an
// occurrence of the target's remote part, whose cell tag (owner node *
// num_features + feature) goes to remote_cells. Consecutive rows of a cell
// go to consecutive destinations. Each CTA's stores are fenced at system
// scope before it ends.
__global__ void __launch_bounds__(kThreads)
    node_key_push_kernel(ChannelStatic channel, const int64_t *keys, const float *weights,
                         const int32_t *row_cell, const int64_t *destination_base,
                         const int64_t *rows_to_push) {
  const int local_size = channel.topo.local_size, num_features = (int)channel.num_features;
  const NodeWindow node(channel.window, channel.topo.local_rank);
  const WindowLayout window = node.layout(channel);
  const int64_t remote_start = window.keys.remote_start();
  const int64_t rows = *rows_to_push;
  for (int64_t row = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; row < rows;
       row += (int64_t)gridDim.x * blockDim.x) {
    const int cell = row_cell[row];
    const int owner = cell / num_features, target = owner % local_size;
    const int64_t destination = destination_base[cell] + row;
    node.peer(window.keys.node_keys(), target)[destination] = keys[row];
    if (weights != nullptr)
      node.peer(window.keys.node_weights(), target)[destination] = weights[row];
    if (destination >= remote_start)
      node.peer(window.keys.remote_cells(), target)[destination - remote_start] =
          owner / local_size * num_features + cell % num_features;
  }
  __syncthreads();
  if (threadIdx.x == 0) fence_sc_sys();
}

// 1 warp: each thread, after a system fence of its own, tells its node peers
// (peer = thread, thread + 32, ...) that this rank's keys are in their
// node_keys, then waits until each of them told this rank.
__global__ void node_done_kernel(CommArgs args) {
  const NodeWindow node(args.window, args.topo.local_rank);
  const PeerFlags<CommArgs> written(args, node, node.layout(args).ctrl.in_written());
  fence_sc_sys();
  for (int peer = threadIdx.x; peer < args.topo.local_size; peer += blockDim.x) written.tell(peer);
  for (int peer = threadIdx.x; peer < args.topo.local_size; peer += blockDim.x)
    written.wait(peer, "pushed keys");
}

// 1 CTA (8 warps): this rank's count messages (CountMessage::out_words
// words) into its count_msg_out, and its own message (the common part and
// the part for its own node) into the call's count_msg from its node. Then
// warp 1's lanes, each taking rail peers in turn, put each peer its message
// in two puts into the same place in the peer's copy: the common part, then
// the part for the peer's node, which signals (the signal also covers the
// first put). Warp 0's lanes wait for the rail peers' messages, and every
// peer's lands in messages [num_nodes][CountMessage::length]. The count
// messages go by in_seq parity: a peer's message of call seq + 2 needs this
// rank's of seq + 1, which it sends once call seq ended here.
__global__ void rail_count_exchange_kernel(CommArgs args, const int64_t *message,
                                           int64_t *messages) {
  const int num_nodes = args.topo.num_nodes, node = args.topo.node, tid = threadIdx.x,
            warp = tid >> 5, lane = tid & 31;
  const CountMessage format{num_nodes, args.num_features};
  const int64_t common = format.common(), length = format.length();
  const NodeWindow node_window(args.window, args.topo.local_rank);
  const CtrlLayout ctrl = node_window.layout(args).ctrl;
  int64_t *out = ctrl.count_msg_out();
  int64_t *own = ctrl.count_msg(args.ch.seq, node);
  for (int i = tid; i < format.out_words(); i += blockDim.x) out[i] = message[i];
  for (int i = tid; i < length; i += blockDim.x)
    own[i] = message[i < common ? i : format.part(node) + (i - common)];
  __syncthreads();
  ncclGin gin(args.dev, kCtxIn);
  const ncclTeam rail = ncclTeamRail(args.dev);
  if (warp == 1) {
    const RailMessage counts(gin, rail, node_window, sig_count(node));
    fence_sc_sys();
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32) {
      if (peer_node == node) continue;
      rail_put(gin, rail, node_window, peer_node, own, out, (size_t)common * 8);
      counts.put(peer_node, own + common, out + format.part(peer_node),
                 (size_t)args.num_features * 8);
    }
  }
  if (warp == 0) {
    Spin spin(args.timeout_ns);
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32)
      while (peer_node != node && (int64_t)gin.readSignal(sig_count(peer_node)) < args.ch.seq)
        spin.pause("count message", peer_node, args.ch.seq, args.topo.rank);
  }
  __syncthreads();
  for (int i = tid; i < num_nodes * length; i += blockDim.x)
    messages[i] = (int64_t)__ldcg(
        (const unsigned long long *)(ctrl.count_msg(args.ch.seq, i / length) + i % length));
  // count_msg_out may be written again once the NIC has read it.
  if (warp == 1) gin.flush(ncclCoopThread{});
}

// 1 CTA (2 warps): warp 1's lanes, each taking rail peers in turn, send each
// peer o this call's key stream: as proxy, this rank's slice of key_stage
// (bytes [put_src[o], put_src[o] + put_bytes[o])) into the owner's rail_keys
// at put_dst[o], unless put_skip[o] (the owner overflows lookup_keys); then
// as owner, where that proxy's cells start in its KJT (row o of cell_stage)
// into the proxy's rail_cells, which signals keys[node]. Its system fence
// orders the planner's writes of key_stage and cell_stage before the puts.
// Warp 0 waits for the other nodes' key streams. A proxy puts only after it
// has the owner's count message of the call, which the owner sends once its
// previous call ended.
__global__ void rail_key_exchange_kernel(CommArgs args, const int64_t *put_src,
                                         const int64_t *put_dst, const int64_t *put_bytes,
                                         const bool *put_skip) {
  const int num_nodes = args.topo.num_nodes, node = args.topo.node, warp = threadIdx.x >> 5,
            lane = threadIdx.x & 31;
  ncclGin gin(args.dev, kCtxIn);
  const ncclTeam rail = ncclTeamRail(args.dev);
  if (warp == 1) {
    const NodeWindow node_window(args.window, args.topo.local_rank);
    const WindowLayout window = node_window.layout(args);
    const RailMessage key_stream(gin, rail, node_window, sig_keys(num_nodes, node));
    const size_t cells_bytes = 8 * (size_t)args.num_features;
    fence_sc_sys();
    for (int owner_node = lane; owner_node < num_nodes; owner_node += 32) {
      if (owner_node == node) continue;
      if (!put_skip[owner_node])
        rail_put(gin, rail, node_window, owner_node, window.keys.rail_keys() + put_dst[owner_node],
                 window.keys.key_stage() + put_src[owner_node], (size_t)put_bytes[owner_node]);
      key_stream.put(owner_node, window.keys.rail_cells(node), window.keys.cell_stage(owner_node),
                     cells_bytes);
    }
  }
  if (warp == 0) {
    Spin spin(args.timeout_ns);
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32)
      while (peer_node != node &&
             (int64_t)gin.readSignal(sig_keys(num_nodes, peer_node)) < args.ch.seq)
        spin.pause("key stream", peer_node, args.ch.seq, args.topo.rank);
  }
  // key_stage and cell_stage may be written again once the NIC has read them.
  if (warp == 1) gin.flush(ncclCoopThread{});
}

// 1 CTA: this rank's publish vector (its rail's kNumNeeds needs, then its
// cell starts [requester][feature]) into its needs and cell_start words;
// after a system fence, which also orders the planner's writes of
// remote_map and the proxy header before it, in_done gets seq. Once every
// node peer's has it, every peer's needs land in peer_needs [local_size]
// [kNumNeeds] and where each one's KJT holds this requester's cells in
// owner_cell_start [local_size][num_features].
__global__ void node_status_exchange_kernel(CommArgs args, const int64_t *publish,
                                            int64_t *peer_needs, int64_t *owner_cell_start) {
  const int local_size = args.topo.local_size, num_features = (int)args.num_features,
            me = args.topo.local_rank, tid = threadIdx.x;
  const NodeWindow node(args.window, me);
  const CtrlLayout ctrl = node.layout(args).ctrl;
  for (int i = tid; i < kNumNeeds + local_size * num_features; i += blockDim.x) {
    if (i < kNumNeeds) {
      ctrl.needs()[i] = publish[i];
    } else {
      const int cell = i - kNumNeeds;  // requester * num_features + feature
      ctrl.cell_start(cell / num_features)[cell % num_features] = publish[i];
    }
  }
  __syncthreads();
  const NodeBarrier<CommArgs> status(args, node, ctrl.in_done());
  if (tid == 0) {
    fence_sc_sys();
    status.publish();
  }
  for (int peer = tid; peer < local_size; peer += blockDim.x) status.wait(peer, "node's statuses");
  __syncthreads();
  // Node peer `peer`'s copy of the word `mine` names in this rank's.
  auto word_of = [&](int peer, const int64_t *mine) {
    return (int64_t)__ldcg((const unsigned long long *)node.peer(mine, peer));
  };
  for (int i = tid; i < local_size * kNumNeeds; i += blockDim.x)
    peer_needs[i] = word_of(i / kNumNeeds, ctrl.needs() + i % kNumNeeds);
  for (int i = tid; i < local_size * num_features; i += blockDim.x)
    owner_cell_start[i] = word_of(i / num_features, ctrl.cell_start(me) + i % num_features);
}

// Every CTA, a thread per word: word k of item i is node peer peer[i]'s word
// first_word[i] of its proxy header (kGatherProxyHeader, k = 0) or entry
// first_word[i] of array k of its remote_map (kGatherRemoteMap), into
// out[k][i] (none when peer[i] < 0). The peers published them before the
// in_done this rank's node_status_exchange acquired earlier on the stream.
__global__ void __launch_bounds__(kThreads)
    peer_gather_kernel(ChannelStatic channel, GatherFrom from, const int32_t *peer,
                       const int64_t *first_word, int64_t items, int words, int32_t *out) {
  const NodeWindow node(channel.window, channel.topo.local_rank);
  const WindowLayout window = node.layout(channel);
  const ProxyHeader header = window.keys.proxy_header();
  const RemoteMap map = window.keys.remote_map();
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < items * words;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t item = i % items, word = i / items;
    const int32_t source = peer[item];
    // the words that word k of the items is read from, in this rank's copy
    const int32_t *mine = from == kGatherProxyHeader ? header.base : map.array((int)word);
    if (source >= 0) out[i] = __ldcg(node.peer(mine, source) + first_word[item]);
  }
}

// Signals and counters of every GIN context start at 0: `bad` counts those
// that do not.
__global__ void gin_zero_kernel(ncclDevComm dev, int signals, int counters, int *bad) {
  ncclGin gin(dev, blockIdx.x);
  for (int signal = threadIdx.x; signal < signals; signal += blockDim.x)
    if (gin.readSignal(signal) != 0) atomicAdd(bad, 1);
  for (int counter = threadIdx.x; counter < counters; counter += blockDim.x)
    if (gin.readCounter(counter) != 0) atomicAdd(bad, 1);
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
