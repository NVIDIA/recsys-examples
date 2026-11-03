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
// The comm kernels of the in channel: the input dist's exchanges with the
// node peers and the rail peers, and the in channel's functions that launch
// them (input_dist.h). The planner's steps (planner.h) run between
// them on the same stream, in the order of InChannel::input_dist:
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
// on the stream (peer_gather with ld.global.cg, the planner's steps with
// plain loads). What this rank writes for its peers (its own stores, or
// the planner's earlier on the stream) is ordered before the flag it
// releases or the put it makes: by fence.sc.sys, or in node_count_exchange
// by the release itself after the CTA's barrier.

#include <ATen/cuda/CUDAContext.h>
#include <nccl_device.h>

#include <cstdint>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/config.h"
#include "hier_output_dist/common/device.cuh"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/input_dist/input_dist.h"

namespace dyn_emb {
namespace hier_output_dist {

// Each comm kernel takes the op's window (and, on the rail, its GIN device
// communicator), its sizes, and if it waits the call's sequence number
// (seq, in_seq): then the call's tensors.

// 1 CTA: this rank's count vector (CountVector, planner.h) into its
// count_vector of the call, the node barrier, then the node's count vectors
// as this rank needs them: their sum (totals) and that of the ranks before
// it (before), both [words], and each requester's counts for this rank (mine
// [local_size][num_features + 1]: its keys for its remote part, then for
// each feature of its own part). Passing the barrier also means every peer
// finished its previous call's in channel, whose steps read what this call
// writes into their copies of the window.
__global__ void node_count_exchange_kernel(ncclWindow_t window, Sizes sizes, int64_t seq,
                                           const int64_t *my_counts, int64_t *totals,
                                           int64_t *before, int64_t *mine) {
  const int local_size = sizes.topo.local_size, num_features = (int)sizes.num_features,
            me = sizes.topo.local_rank, tid = threadIdx.x;
  const CountVector format{local_size, num_features};
  const int64_t words = format.words();
  const NodeWindow node(window, me);
  const CtrlLayout ctrl = node.layout(sizes).ctrl;
  int64_t *counts = ctrl.count_vector(seq);
  for (int i = tid; i < words; i += blockDim.x) counts[i] = my_counts[i];
  __syncthreads();
  const NodeBarrier barrier{sizes, node, ctrl.in_started(), seq};
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
    node_key_push_kernel(ncclWindow_t window, Sizes sizes, const int64_t *keys,
                         const float *weights, const int32_t *row_cell,
                         const int64_t *destination_base, const int64_t *rows_to_push) {
  const int local_size = sizes.topo.local_size, num_features = (int)sizes.num_features;
  const NodeWindow node(window, sizes.topo.local_rank);
  const KeysLayout keys_layout = node.layout(sizes).keys;
  const int64_t remote_start = keys_layout.remote_start();
  const int64_t rows = *rows_to_push;
  for (int64_t row = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; row < rows;
       row += (int64_t)gridDim.x * blockDim.x) {
    const int cell = row_cell[row];
    const int owner = cell / num_features, target = owner % local_size;
    const int64_t destination = destination_base[cell] + row;
    node.peer(keys_layout.node_keys(), target)[destination] = keys[row];
    if (weights != nullptr)
      node.peer(keys_layout.node_weights(), target)[destination] = weights[row];
    if (destination >= remote_start)
      node.peer(keys_layout.remote_cells(), target)[destination - remote_start] =
          owner / local_size * num_features + cell % num_features;
  }
  __syncthreads();
  if (threadIdx.x == 0) fence_sc_sys();
}

// 1 warp: each thread, after a system fence of its own, tells its node peers
// (peer = thread, thread + 32, ...) that this rank's keys are in their
// node_keys, then waits until each of them told this rank.
__global__ void node_done_kernel(ncclWindow_t window, Sizes sizes, int64_t seq) {
  const NodeWindow node(window, sizes.topo.local_rank);
  const PeerFlags written{sizes, node, node.layout(sizes).ctrl.in_written(), seq};
  fence_sc_sys();
  for (int peer = threadIdx.x; peer < sizes.topo.local_size; peer += blockDim.x) written.tell(peer);
  for (int peer = threadIdx.x; peer < sizes.topo.local_size; peer += blockDim.x)
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
__global__ void rail_count_exchange_kernel(ncclWindow_t window, ncclDevComm dev, Sizes sizes,
                                           int64_t seq, const int64_t *message, int64_t *messages) {
  const int num_nodes = sizes.topo.num_nodes, node = sizes.topo.node, tid = threadIdx.x,
            warp = tid >> 5, lane = tid & 31;
  const CountMessage format{num_nodes, sizes.num_features};
  const int64_t common = format.common(), length = format.length();
  const NodeWindow node_window(window, sizes.topo.local_rank);
  const CtrlLayout ctrl = node_window.layout(sizes).ctrl;
  int64_t *out = ctrl.count_msg_out();
  int64_t *own = ctrl.count_msg(seq, node);
  for (int i = tid; i < format.out_words(); i += blockDim.x) out[i] = message[i];
  for (int i = tid; i < length; i += blockDim.x)
    own[i] = message[i < common ? i : format.part(node) + (i - common)];
  __syncthreads();
  ncclGin gin(dev, kCtxIn);
  const ncclTeam rail = ncclTeamRail(dev);
  if (warp == 1) {
    const RailMessage counts(gin, rail, node_window, sig_count(node));
    fence_sc_sys();
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32) {
      if (peer_node == node) continue;
      rail_put(gin, rail, node_window, peer_node, own, out, (size_t)common * 8);
      counts.put(peer_node, own + common, out + format.part(peer_node),
                 (size_t)sizes.num_features * 8);
    }
  }
  if (warp == 0) {
    Spin spin(sizes.timeout_ns);
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32)
      while (peer_node != node && (int64_t)gin.readSignal(sig_count(peer_node)) < seq)
        spin.pause("count message", peer_node, seq, sizes.topo.rank);
  }
  __syncthreads();
  for (int i = tid; i < num_nodes * length; i += blockDim.x)
    messages[i] =
        (int64_t)__ldcg((const unsigned long long *)(ctrl.count_msg(seq, i / length) + i % length));
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
__global__ void rail_key_exchange_kernel(ncclWindow_t window, ncclDevComm dev, Sizes sizes,
                                         int64_t seq, const int64_t *put_src,
                                         const int64_t *put_dst, const int64_t *put_bytes,
                                         const bool *put_skip) {
  const int num_nodes = sizes.topo.num_nodes, node = sizes.topo.node, warp = threadIdx.x >> 5,
            lane = threadIdx.x & 31;
  ncclGin gin(dev, kCtxIn);
  const ncclTeam rail = ncclTeamRail(dev);
  if (warp == 1) {
    const NodeWindow node_window(window, sizes.topo.local_rank);
    const KeysLayout keys = node_window.layout(sizes).keys;
    const RailMessage key_stream(gin, rail, node_window, sig_keys(num_nodes, node));
    const size_t cells_bytes = 8 * (size_t)sizes.num_features;
    fence_sc_sys();
    for (int owner_node = lane; owner_node < num_nodes; owner_node += 32) {
      if (owner_node == node) continue;
      if (!put_skip[owner_node])
        rail_put(gin, rail, node_window, owner_node, keys.rail_keys() + put_dst[owner_node],
                 keys.key_stage() + put_src[owner_node], (size_t)put_bytes[owner_node]);
      key_stream.put(owner_node, keys.rail_cells(node), keys.cell_stage(owner_node), cells_bytes);
    }
  }
  if (warp == 0) {
    Spin spin(sizes.timeout_ns);
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32)
      while (peer_node != node && (int64_t)gin.readSignal(sig_keys(num_nodes, peer_node)) < seq)
        spin.pause("key stream", peer_node, seq, sizes.topo.rank);
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
__global__ void node_status_exchange_kernel(ncclWindow_t window, Sizes sizes, int64_t seq,
                                            const int64_t *publish, int64_t *peer_needs,
                                            int64_t *owner_cell_start) {
  const int local_size = sizes.topo.local_size, num_features = (int)sizes.num_features,
            me = sizes.topo.local_rank, tid = threadIdx.x;
  const NodeWindow node(window, me);
  const CtrlLayout ctrl = node.layout(sizes).ctrl;
  for (int i = tid; i < kNumNeeds + local_size * num_features; i += blockDim.x) {
    if (i < kNumNeeds) {
      ctrl.needs()[i] = publish[i];
    } else {
      const int cell = i - kNumNeeds;  // requester * num_features + feature
      ctrl.cell_start(cell / num_features)[cell % num_features] = publish[i];
    }
  }
  __syncthreads();
  const NodeBarrier status{sizes, node, ctrl.in_done(), seq};
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
    peer_gather_kernel(ncclWindow_t window, Sizes sizes, GatherFrom from, const int32_t *peer,
                       const int64_t *first_word, int64_t items, int words, int32_t *out) {
  const NodeWindow node(window, sizes.topo.local_rank);
  const KeysLayout keys = node.layout(sizes).keys;
  const ProxyHeader header = keys.proxy_header();
  const RemoteMap map = keys.remote_map();
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < items * words;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t item = i % items, word = i / items;
    const int32_t source = peer[item];
    // the words that word k of the items is read from, in this rank's copy
    const int32_t *mine = from == kGatherProxyHeader ? header.base : map.array((int)word);
    if (source >= 0) out[i] = __ldcg(node.peer(mine, source) + first_word[item]);
  }
}

// The launchers.

namespace {

cudaStream_t current_stream() { return at::cuda::getCurrentCUDAStream(); }

// CTAs of kThreads threads for `work` items on the current device.
int grid_of(int64_t work) {
  return grid_for(work, at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> InChannel::node_count_exchange(
    const at::Tensor &my_counts) const {
  at::Tensor totals = at::empty_like(my_counts), before = at::empty_like(my_counts);
  at::Tensor mine =
      at::empty({sizes_.topo.local_size, sizes_.num_features + 1}, my_counts.options());
  node_count_exchange_kernel<<<1, kThreads, 0, current_stream()>>>(
      window_, sizes_, seq_, my_counts.data_ptr<int64_t>(), totals.data_ptr<int64_t>(),
      before.data_ptr<int64_t>(), mine.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {totals, before, mine};
}

void InChannel::node_key_push(const at::Tensor &keys, const at::Tensor &weights,
                              const at::Tensor &row_cell, const at::Tensor &destination_base,
                              const at::Tensor &rows_to_push) const {
  node_key_push_kernel<<<grid_of(keys.numel()), kThreads, 0, current_stream()>>>(
      window_, sizes_, keys.data_ptr<int64_t>(),
      weights.defined() ? weights.data_ptr<float>() : nullptr, row_cell.data_ptr<int32_t>(),
      destination_base.data_ptr<int64_t>(), rows_to_push.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void InChannel::node_done() const {
  node_done_kernel<<<1, 32, 0, current_stream()>>>(window_, sizes_, seq_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor InChannel::rail_count_exchange(const at::Tensor &message) const {
  const int num_nodes = sizes_.topo.num_nodes;
  const CountMessage format{num_nodes, sizes_.num_features};
  at::Tensor messages = at::empty({num_nodes, format.length()}, message.options());
  rail_count_exchange_kernel<<<1, 256, 0, current_stream()>>>(
      window_, dev_, sizes_, seq_, message.data_ptr<int64_t>(), messages.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return messages;
}

void InChannel::rail_key_exchange(const at::Tensor &put_src, const at::Tensor &put_dst,
                                  const at::Tensor &put_bytes, const at::Tensor &put_skip) const {
  rail_key_exchange_kernel<<<1, 64, 0, current_stream()>>>(
      window_, dev_, sizes_, seq_, put_src.data_ptr<int64_t>(), put_dst.data_ptr<int64_t>(),
      put_bytes.data_ptr<int64_t>(), put_skip.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::pair<at::Tensor, at::Tensor> InChannel::node_status_exchange(const at::Tensor &publish) const {
  const int local_size = sizes_.topo.local_size;
  at::Tensor peer_needs = at::empty({local_size, kNumNeeds}, publish.options());
  at::Tensor owner_cell_start = at::empty({local_size, sizes_.num_features}, publish.options());
  node_status_exchange_kernel<<<1, 256, 0, current_stream()>>>(
      window_, sizes_, seq_, publish.data_ptr<int64_t>(), peer_needs.data_ptr<int64_t>(),
      owner_cell_start.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {peer_needs, owner_cell_start};
}

at::Tensor InChannel::peer_gather(GatherFrom from, const at::Tensor &peer,
                                  const at::Tensor &first_word) const {
  const int64_t items = peer.numel();
  const int words = gather_words(from);
  at::Tensor out = at::empty({words, items}, peer.options());
  peer_gather_kernel<<<grid_of(items * words), kThreads, 0, current_stream()>>>(
      window_, sizes_, from, peer.data_ptr<int32_t>(), first_word.data_ptr<int64_t>(), items, words,
      out.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
