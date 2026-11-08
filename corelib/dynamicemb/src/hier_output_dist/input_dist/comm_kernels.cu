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
//   node_exchange         this rank's offsets into node_offsets at each node peer, over NVLink;
//                         the node barrier of the call; the node layout from the node's offsets;
//   node_key_push         the bucketize's gather: this rank's keys into their places in their
//                         targets' node_keys, over NVLink;
//   node_done             tells each target its keys are there, and waits until every requester
//                         of the node told it;
//   rail_count_exchange   (across nodes) its count message to each rail
//                         peer, and theirs;
//   rail_key_exchange     (across nodes) its key records to each rail owner
//                         and its source_offsets to each rail proxy, and
//                         theirs;
//   node_status_exchange  (across nodes) its rail's needs and
//                         requester_offsets to its node, and every node
//                         peer's;
//   peer_gather           (across nodes) words of the node peers' remote_map
//                         or proxy header.
// A kernel that waits runs on one CTA (or warp). It waits only for the same kernel (the same seq)
// of its node or rail peers. Its threads (or lanes) take the peers in turn, so any number of them.
// Memory order:
// - What peers wrote is read only after an acquire of their flag or signal. The comm kernel that
//   acquired it reads with ld.global.cg. A later kernel on the stream reads it too: peer_gather
//   with ld.global.cg, the planner's steps with plain loads.
// - What this rank writes for its peers (its own stores, or the planner's earlier on the stream)
//   comes before the flag it releases or the put it makes: fence.sc.sys orders it.
// - What it stores into a peer's buffer comes after an acquire, earlier on the stream, of the
//   peer's flag that says the peer read that buffer: node_exchange's barrier for node_keys (a
//   peer publishes it once every kernel of its call before ended), node_done's of the call before
//   for node_offsets (a peer tells it after its node_exchange read them).

#include <ATen/cuda/CUDAContext.h>
#include <nccl_device.h>

#include <cstdint>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/config.h"
#include "hier_output_dist/common/device.cuh"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/input_dist/input_dist.h"
#include "hier_output_dist/input_dist/node.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// Each comm kernel takes the op's window (and, on the rail, its GIN device
// communicator), its sizes, and if it waits the call's sequence number
// (seq, in_seq): then the call's tensors.

// 1 CTA: this rank's offsets (its bucketize's offset_per_bucket_feature) go into its row of
// node_offsets at each node peer, its own too, over NVLink. Then a system fence, and its
// in_started gets the call (the node barrier). It waits until each peer's in_started holds the
// call: every peer's offsets are then in node_offsets here. From them it makes the node layout
// (node_layout_of; on one node also the owner's KJT offsets). Passing the barrier also means
// every peer finished its previous call's in channel, whose steps read what this call writes
// into their copies of the window (node_keys). delay_ns (the test hook delay_node_reads, 0:
// none): thread 0 waits that long after the barrier, so the layout reads node_offsets late.
__global__ void __launch_bounds__(kThreads)
    node_exchange_kernel(ncclWindow_t window, Sizes sizes, int64_t seq,
                         const int64_t *offset_per_bucket_feature, int64_t *scratch,
                         int64_t delay_ns, NodeLayoutView layout) {
  const int local_size = sizes.topo.local_size, me = sizes.topo.local_rank, tid = threadIdx.x;
  const NodeWindow node(window, me);
  const KeysLayout keys = node.layout(sizes).keys;
  const int64_t words = keys.offset_words();
  int64_t *mine = keys.node_offsets(me);
  for (int64_t i = tid; i < local_size * words; i += blockDim.x)
    node.peer(mine, (int)(i / words))[i % words] = offset_per_bucket_feature[i % words];
  __syncthreads();
  const NodeBarrier barrier{sizes, node, node.layout(sizes).ctrl.in_started(), seq};
  if (tid == 0) {
    fence_sc_sys();
    barrier.publish();
  }
  for (int peer = tid; peer < local_size; peer += blockDim.x) barrier.wait(peer, "node offsets");
  if (tid == 0 && delay_ns > 0) {
    const uint64_t start = now_ns();
    while (now_ns() - start < (uint64_t)delay_ns) __nanosleep(1000);
  }
  __syncthreads();
  node_layout_of(keys.node_offsets(0), sizes.topo, sizes.num_features, sizes.lookup_keys,
                 sizes.remote_keys, scratch, layout);
}

// Every CTA, a thread per bucketized row (push_rows): the bucketize's gather. Each valid row's
// key (and weight; across nodes a remote row's bucket feature too) goes to its place in its
// target's node_keys, over NVLink; on one node each row's place in its owner's KJT goes to src.
// It does not wait: node_exchange, before it on the stream, passed the node barrier. Each CTA
// fences its stores at system scope before it ends.
__global__ void __launch_bounds__(kThreads)
    node_key_push_kernel(ncclWindow_t window, Sizes sizes, GatherRows rows, int64_t num_rows,
                         PushPlaces places, int32_t *src) {
  const NodeWindow node(window, sizes.topo.local_rank);
  const KeysLayout keys = node.layout(sizes).keys;
  const int64_t remote_start = keys.remote_start();
  push_rows(rows, num_rows, places, sizes.topo, sizes.num_features, src,
            [&](int64_t target, int64_t place, int64_t key, float weight, int32_t feature) {
              node.peer(keys.node_keys(), (int)target)[place] = key;
              if (rows.weights != nullptr)
                node.peer(keys.node_key_weights(), (int)target)[place] = weight;
              if (place >= remote_start)
                node.peer(keys.node_bucket_feature_of_row(), (int)target)[place - remote_start] =
                    feature;
            });
  __syncthreads();
  if (threadIdx.x == 0) fence_sc_sys();
}

// 1 warp: each thread, after a system fence of its own, tells its node peers
// (peer = thread, thread + 32, ...) that this rank's keys are in their
// node_keys, then waits until each of them told this rank. The tell also says that this rank's
// node_exchange read node_offsets, so its peers may write them again in the next call. node_done
// runs in every call, also with no rows and under skip.
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
// (bytes [put_src[o], put_src[o] + put_bytes[o])) into the owner's rail_keys at
// put_dst[o], unless put_skip[o] (the owner overflows lookup_keys); then as
// owner, where that proxy's keys of each feature start in its KJT (row o of
// source_offsets) into the proxy's owner_offsets, which signals keys[node]. Its
// system fence orders the planner's writes of key_stage and source_offsets
// before the puts. Warp 0 waits for the other nodes' key streams. A proxy puts
// only after it has the owner's count message of the call, which the owner
// sends once its previous call ended.
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
    const size_t offsets_bytes = 8 * (size_t)sizes.num_features;
    fence_sc_sys();
    for (int owner_node = lane; owner_node < num_nodes; owner_node += 32) {
      if (owner_node == node) continue;
      if (!put_skip[owner_node])
        rail_put(gin, rail, node_window, owner_node, keys.rail_keys() + put_dst[owner_node],
                 keys.key_stage() + put_src[owner_node], (size_t)put_bytes[owner_node]);
      key_stream.put(owner_node, keys.owner_offsets(node), keys.source_offsets(owner_node),
                     offsets_bytes);
    }
  }
  if (warp == 0) {
    Spin spin(sizes.timeout_ns);
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32)
      while (peer_node != node && (int64_t)gin.readSignal(sig_keys(num_nodes, peer_node)) < seq)
        spin.pause("key stream", peer_node, seq, sizes.topo.rank);
  }
  // key_stage and source_offsets may be written again once the NIC has read
  // them.
  if (warp == 1) gin.flush(ncclCoopThread{});
}

// 1 CTA: this rank's publish vector (its rail's kNumNeeds needs, then its
// requester offsets [requester][feature]) into its needs and requester_offsets
// words; after a system fence, which also orders the planner's writes of
// remote_map and the proxy header before it, in_done gets seq. Once every node
// peer's has it, every peer's needs land in peer_needs [local_size] [kNumNeeds]
// and where each one's KJT holds this requester's keys of each feature in
// offset_per_owner_feature [local_size][num_features].
__global__ void node_status_exchange_kernel(ncclWindow_t window, Sizes sizes, int64_t seq,
                                            const int64_t *publish, int64_t *peer_needs,
                                            int64_t *offset_per_owner_feature) {
  const int local_size = sizes.topo.local_size, num_features = (int)sizes.num_features,
            me = sizes.topo.local_rank, tid = threadIdx.x;
  const NodeWindow node(window, me);
  const CtrlLayout ctrl = node.layout(sizes).ctrl;
  for (int i = tid; i < kNumNeeds + local_size * num_features; i += blockDim.x) {
    if (i < kNumNeeds) {
      ctrl.needs()[i] = publish[i];
    } else {
      const int index = i - kNumNeeds;  // requester * num_features + feature
      ctrl.requester_offsets(index / num_features)[index % num_features] = publish[i];
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
    offset_per_owner_feature[i] =
        word_of(i / num_features, ctrl.requester_offsets(me) + i % num_features);
}

// Every CTA, a thread per word: word k of item i is node peer peer[i]'s word
// first_word[i] of its proxy header (kFrom kGatherProxyHeader, k = 0) or
// entry first_word[i] of array k of its remote_map (kGatherRemoteMap), into
// out[k][i] (none when peer[i] < 0). The peers published them before the
// in_done this rank's node_status_exchange acquired earlier on the stream.
template <GatherFrom kFrom>
__global__ void __launch_bounds__(kThreads)
    peer_gather_kernel(ncclWindow_t window, Sizes sizes, const int32_t *peer,
                       const int64_t *first_word, int64_t items, int32_t *out) {
  constexpr int kWords = gather_words(kFrom);
  const NodeWindow node(window, sizes.topo.local_rank);
  const KeysLayout keys = node.layout(sizes).keys;
  const ProxyHeader header = keys.proxy_header();
  const RemoteMap map = keys.remote_map();
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < items * kWords;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t item = i % items, word = i / items;
    const int32_t source = peer[item];
    // the words that word k of the items is read from, in this rank's copy
    const int32_t *mine = kFrom == kGatherProxyHeader ? header.base : map.array((int)word);
    if (source >= 0) out[i] = __ldcg(node.peer(mine, source) + first_word[item]);
  }
}

// The launchers.

NodeLayout InChannel::node_exchange(const BucketizedKeys &bucketized, int64_t delay_ns) {
  const auto options = bucketized.offset_per_bucket.options();
  NodeLayout node = empty_node_layout(sizes_.topo, sizes_.num_features, options);
  const at::Tensor scratch =
      at::empty({node_layout_scratch(sizes_.topo, sizes_.num_features)}, options);
  // The call's sequence number. seq_ takes it only once the kernel is launched.
  const int64_t seq = seq_ + 1;
  node_exchange_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      window_, sizes_, seq, bucketized.offset_per_bucket_feature.data_ptr<int64_t>(),
      scratch.data_ptr<int64_t>(), delay_ns, view_of(node));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  seq_ = seq;
  return node;
}

void InChannel::node_key_push(const BucketizedKeys &bucketized,
                              const c10::optional<at::Tensor> &weights, const NodeLayout &node,
                              const at::Tensor &src) const {
  const int64_t rows = bucketized.inv.numel();
  const GatherRows gather{bucketized.inv.data_ptr<int32_t>(), bucketized.stored.data_ptr<int64_t>(),
                          weights.has_value() ? weights->data_ptr<float>() : nullptr};
  const PushPlaces places{bucketized.bucket_feature_of_row.data_ptr<int32_t>(),
                          bucketized.offset_per_bucket.data_ptr<int64_t>(),
                          node.destination_base.data_ptr<int64_t>(), node.skip.data_ptr<bool>()};
  node_key_push_kernel<<<item_grid(rows), kThreads, 0, current_stream()>>>(
      window_, sizes_, gather, rows, places, data_or_null<int32_t>(src));
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
  at::Tensor offset_per_owner_feature =
      at::empty({local_size, sizes_.num_features}, publish.options());
  node_status_exchange_kernel<<<1, 256, 0, current_stream()>>>(
      window_, sizes_, seq_, publish.data_ptr<int64_t>(), peer_needs.data_ptr<int64_t>(),
      offset_per_owner_feature.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {peer_needs, offset_per_owner_feature};
}

at::Tensor InChannel::peer_gather(GatherFrom from, const at::Tensor &peer,
                                  const at::Tensor &first_word) const {
  const int64_t items = peer.numel();
  const int words = gather_words(from);
  at::Tensor out = at::empty({words, items}, peer.options());
  const auto launch = [&](auto kernel) {
    kernel<<<item_grid(items * words), kThreads, 0, current_stream()>>>(
        window_, sizes_, peer.data_ptr<int32_t>(), first_word.data_ptr<int64_t>(), items,
        out.data_ptr<int32_t>());
  };
  if (from == kGatherProxyHeader)
    launch(peer_gather_kernel<kGatherProxyHeader>);
  else
    launch(peer_gather_kernel<kGatherRemoteMap>);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
