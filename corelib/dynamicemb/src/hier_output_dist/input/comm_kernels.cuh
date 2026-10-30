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
//   node_count_exchange   the node barrier of the call, and every node
//                         peer's counts;
//   node_key_push         this rank's keys into its targets' node_keys over
//                         NVLink;
//   node_done             tells each target its keys are there, and waits
//                         until every requester of the node told it;
//   rail_count_exchange   (across nodes) its count message to its rail peers,
//                         and theirs;
//   rail_key_exchange     (across nodes) its key records to each rail owner,
//                         and the other nodes' key streams;
//   node_status_exchange  (across nodes) its rail's needs and cell starts to
//                         its node, and every node peer's;
//   peer_gather           (across nodes) words of the node peers' remote_map
//                         or proxy header.
// A kernel that waits runs on one CTA and waits only for the same kernel (the
// same seq) of its node or rail peers; the others never wait. Memory order:
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
#include "hier_output_dist/common/status.h"
#include "hier_output_dist/common/windows.h"
#include "hier_output_dist/device/ctrl_words.cuh"
#include "hier_output_dist/device/memory_order.cuh"
#include "hier_output_dist/device/node_sync.cuh"
#include "hier_output_dist/device/rail.cuh"
#include "hier_output_dist/device/spin.cuh"
#include "hier_output_dist/device/sym_window.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// The argument of a comm kernel that waits: the part both channels share,
// and the in channel at the call's sequence number (its node barrier word
// in_started), which NodeBarrier and PeerFlags read.
struct CommArgs : ChannelStatic {
  Channel ch;
};

// 1 CTA: this rank's count vector (plan/node.h) into its in_meta slot of the
// call, the node barrier, then every node peer's count vector into
// node_counts [local_size][words]. Passing the barrier also means every peer
// finished its previous call on its side stream, whose steps read what this
// call writes into their windows.
__global__ void node_count_exchange_kernel(CommArgs args, const int64_t *my_counts,
                                           int64_t *node_counts) {
  __shared__ const unsigned long long *peer_counts[kMaxLocalRanks];
  const int local_size = args.topo.local_size, tid = threadIdx.x;
  const int words = local_size * (args.num_features + 1);
  const int slot = args.ctrl_at.in_meta[args.ch.seq & 1];
  const CtrlWords mine = CtrlWords::of(args, args.topo.local_rank);
  for (int i = tid; i < words; i += blockDim.x) *mine.word(slot + i) = (uint64_t)my_counts[i];
  if (tid < local_size)
    peer_counts[tid] = (const unsigned long long *)CtrlWords::of(args, tid).word(slot);
  __syncthreads();
  const NodeBarrier<CommArgs> barrier(args, args.ch.started);
  if (tid == 0) barrier.publish(mine);
  if (tid < local_size) barrier.wait(tid, "in-channel node barrier");
  __syncthreads();
  for (int i = tid; i < local_size * words; i += blockDim.x)
    node_counts[i] = (int64_t)__ldcg(peer_counts[i / words] + i % words);
}

// Every CTA, a thread per row: row `row` of part 1 (row < *rows_to_push) goes
// to destination_base[cell] + row of node_keys (and node_weights) of its
// target, the owner's local rank; a destination from remote_base on is an
// occurrence of the target's remote part, whose cell tag (owner node *
// num_features + feature) goes to remote_cells. Consecutive rows of a cell
// go to consecutive destinations. Each CTA's stores are fenced at system
// scope before it ends.
__global__ void __launch_bounds__(kThreads)
    node_key_push_kernel(ChannelStatic channel, const int64_t *keys, const float *weights,
                         const int32_t *row_cell, const int64_t *destination_base,
                         const int64_t *rows_to_push, int64_t remote_base) {
  __shared__ int64_t *node_keys[kMaxLocalRanks];
  __shared__ float *node_weights[kMaxLocalRanks];
  __shared__ int16_t *remote_cells[kMaxLocalRanks];
  const int local_size = channel.topo.local_size, num_features = channel.num_features,
            tid = threadIdx.x;
  if (tid < local_size) {
    node_keys[tid] = SymRegion<kNodeKeys, int64_t>(channel.win).peer(tid);
    node_weights[tid] = SymRegion<kNodeWeights, float>(channel.win).peer(tid);
    remote_cells[tid] = SymRegion<kRemoteCells, int16_t>(channel.win).peer(tid);
  }
  __syncthreads();
  const int64_t rows = *rows_to_push;
  for (int64_t row = blockIdx.x * (int64_t)blockDim.x + tid; row < rows;
       row += (int64_t)gridDim.x * blockDim.x) {
    const int cell = row_cell[row];
    const int owner = cell / num_features, target = owner % local_size;
    const int64_t destination = destination_base[cell] + row;
    node_keys[target][destination] = keys[row];
    if (weights != nullptr) node_weights[target][destination] = weights[row];
    if (destination >= remote_base)
      remote_cells[target][destination - remote_base] =
          (int16_t)(owner / local_size * num_features + cell % num_features);
  }
  __syncthreads();
  if (tid == 0) fence_sc_sys();
}

// 1 warp: thread `peer` tells node peer `peer` that this rank's keys are in
// its node_keys (after a system fence of its own), then waits until that
// peer told this rank.
__global__ void node_done_kernel(CommArgs args) {
  const int peer = threadIdx.x, me = args.topo.local_rank;
  if (peer >= args.topo.local_size) return;
  const PeerFlags<CommArgs> written(args, args.ctrl_at.in_written);
  fence_sc_sys();
  written.tell(peer, me);
  written.wait(me, peer, "pushed keys");
}

// 1 CTA (8 warps): this rank's count message (msg_len words) into its
// count_msg_out and its own row of the call's count_msgs slot, then to each
// rail peer by a GIN put into the same row of its slot (warp 1, a lane per
// peer); warp 0 waits for the rail peers' messages, and every peer's lands in
// messages [num_nodes][msg_len]. The slots go by in_seq parity: a peer's
// message of call seq + 2 needs this rank's of seq + 1, which it sends once
// call seq ended here.
__global__ void rail_count_exchange_kernel(CommArgs args, const int64_t *message,
                                           int64_t *messages) {
  const int num_nodes = args.topo.num_nodes, node = args.topo.node, tid = threadIdx.x,
            warp = tid >> 5, lane = tid & 31;
  const int length = msg_len(num_nodes, args.num_features), stride = args.ctrl_at.msg_words;
  const CtrlWords mine = CtrlWords::of(args, args.topo.local_rank);
  const uint64_t slot = args.ctrl_at.count_msgs + (uint64_t)(args.ch.seq & 1) * num_nodes * stride;
  int64_t *out = mine.count_msg_out();
  int64_t *land = (int64_t *)mine.word(slot);
  for (int i = tid; i < length; i += blockDim.x) {
    out[i] = message[i];
    land[node * stride + i] = message[i];
  }
  __syncthreads();
  ncclGin gin(args.dev, kCtxIn);
  const ncclTeam rail = ncclTeamRail(args.dev);
  const bool issuer = warp == 1 && lane < num_nodes && lane != node;
  const bool waiter = warp == 0 && lane < num_nodes && lane != node;
  if (issuer) {
    const uint64_t ctrl = args.win.off[kCtrl];
    fence_sc_sys();
    RailMessage(gin, rail, sig_count(node))
        .put(lane, args.win.handle[kCtrl], ctrl + 8 * (slot + (uint64_t)node * stride),
             ctrl + 8 * (uint64_t)args.ctrl_at.count_msg_out, (size_t)length * 8);
  }
  if (waiter) {
    Spin spin(args.timeout_ns);
    while ((int64_t)gin.readSignal(sig_count(lane)) < args.ch.seq)
      spin.pause("count message", lane, args.ch.seq, args.topo.rank);
  }
  __syncthreads();
  for (int i = tid; i < num_nodes * length; i += blockDim.x)
    messages[i] =
        (int64_t)__ldcg((const unsigned long long *)land + (i / length) * stride + i % length);
  // count_msg_out may be written again once the NIC has read it.
  if (issuer) gin.flush(ncclCoopThread{});
}

// 1 CTA (2 warps): warp 1, a lane per rail owner o, puts this proxy's slice
// of key_stage (bytes [put_src[o], put_src[o] + put_bytes[o])) into the
// owner's rail_keys at put_dst[o], signalling keys[node]; or the signal alone
// when put_skip[o] (the owner overflows lookup_keys). Its system fence orders
// the planner's writes of key_stage before the put. Warp 0 waits for the
// other nodes' key streams. A proxy puts only after it has the owner's count
// message of the call, which the owner sends once its previous call ended.
__global__ void rail_key_exchange_kernel(CommArgs args, const int64_t *put_src,
                                         const int64_t *put_dst, const int64_t *put_bytes,
                                         const bool *put_skip) {
  const int num_nodes = args.topo.num_nodes, node = args.topo.node, warp = threadIdx.x >> 5,
            lane = threadIdx.x & 31;
  ncclGin gin(args.dev, kCtxIn);
  const ncclTeam rail = ncclTeamRail(args.dev);
  const bool issuer = warp == 1 && lane < num_nodes && lane != node;
  const bool waiter = warp == 0 && lane < num_nodes && lane != node;
  if (issuer) {
    const RailMessage key_stream(gin, rail, sig_keys(num_nodes, node));
    if (put_skip[lane]) {
      key_stream.signal(lane);
    } else {
      fence_sc_sys();
      key_stream.put(lane, args.win.handle[kKeyStage], args.win.off[kRailKeys] + put_dst[lane],
                     args.win.off[kKeyStage] + put_src[lane], (size_t)put_bytes[lane]);
    }
  }
  if (waiter) {
    Spin spin(args.timeout_ns);
    while ((int64_t)gin.readSignal(sig_keys(num_nodes, lane)) < args.ch.seq)
      spin.pause("key stream", lane, args.ch.seq, args.topo.rank);
  }
  // key_stage may be written again once the NIC has read it.
  if (issuer) gin.flush(ncclCoopThread{});
}

// 1 CTA: this rank's publish vector (its rail's kNumNeeds needs, then its
// cell starts [requester][feature]) into its in_done needs and cell_start
// words; after a system fence, which also orders the planner's writes of
// remote_map and the proxy header before it, in_done gets seq. Once every
// node peer's has it, every peer's needs land in peer_needs [local_size]
// [kNumNeeds] and where each one's KJT holds this requester's cells in
// owner_cell_start [local_size][num_features].
__global__ void node_status_exchange_kernel(CommArgs args, const int64_t *publish,
                                            int64_t *peer_needs, int64_t *owner_cell_start) {
  __shared__ const unsigned long long *peer_words[kMaxLocalRanks];
  const int local_size = args.topo.local_size, num_features = args.num_features,
            me = args.topo.local_rank, tid = threadIdx.x;
  const CtrlWords mine = CtrlWords::of(args, me);
  for (int i = tid; i < kNumNeeds + local_size * num_features; i += blockDim.x) {
    if (i < kNumNeeds)
      *mine.in_done_need(i) = (uint64_t)publish[i];
    else
      mine.cell_start()[i - kNumNeeds] = publish[i];
  }
  if (tid < local_size)
    peer_words[tid] = (const unsigned long long *)CtrlWords::of(args, tid).word(0);
  __syncthreads();
  const NodeBarrier<CommArgs> status(args, args.ctrl_at.in_done);
  if (tid == 0) {
    fence_sc_sys();
    status.publish(mine);
  }
  if (tid < local_size) status.wait(tid, "node's statuses");
  __syncthreads();
  for (int i = tid; i < local_size * kNumNeeds; i += blockDim.x)
    peer_needs[i] =
        (int64_t)__ldcg(peer_words[i / kNumNeeds] + args.ctrl_at.in_done + 1 + i % kNumNeeds);
  for (int i = tid; i < local_size * num_features; i += blockDim.x)
    owner_cell_start[i] = (int64_t)__ldcg(peer_words[i / num_features] + args.ctrl_at.cell_start +
                                          me * num_features + i % num_features);
}

// Every CTA, a thread per word: word k of item i is int32 word
// first_word[i] + k * word_stride of node peer peer[i]'s region (none when
// peer[i] < 0), into out[k][i]. The peers published the region before the
// in_done this rank's node_status_exchange acquired earlier on the stream.
__global__ void __launch_bounds__(kThreads)
    peer_gather_kernel(ChannelStatic channel, Region region, const int32_t *peer,
                       const int64_t *first_word, int64_t items, int words, int64_t word_stride,
                       int32_t *out) {
  __shared__ const int32_t *base[kMaxLocalRanks];
  if (threadIdx.x < channel.topo.local_size)
    base[threadIdx.x] = (const int32_t *)ncclGetLsaPointer(channel.win.handle[region],
                                                           channel.win.off[region], threadIdx.x);
  __syncthreads();
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < items * words;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t item = i % items, word = i / items;
    const int32_t from = peer[item];
    if (from >= 0) out[i] = __ldcg(base[from] + first_word[item] + word * word_stride);
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
