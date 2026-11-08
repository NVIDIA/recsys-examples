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
//   rail_exchange         (across nodes) its count message to every rail peer, and theirs; the
//                         rail's layout from the rail's messages (rail.cuh);
//   rail_key_push         (across nodes) its key records to each rail owner; it does not wait;
//   node_status_exchange  (across nodes) its rail's needs and requester_offsets to its node, and
//                         every node peer's;
//   peer_gather           (across nodes) words of the node peers' remote_map or proxy header;
//   rail_done             (across nodes) waits for every rail proxy's key records, and flushes
//                         this rank's puts; just before owner_tables, the first reader of
//                         rail_keys.
// A kernel that waits runs on one CTA (or warp). It waits only for the same kernel (the same seq)
// of its node or rail peers, but rail_done, which waits for its rail peers' rail_key_push of the
// call, a kernel that never waits. Its threads (or lanes) take the peers in turn, so any number of
// them.
// Memory order:
// - What peers wrote is read only after an acquire of their flag or signal. The comm kernel that
//   acquired it reads with ld.global.cg. A later kernel on the stream reads it too: peer_gather
//   with ld.global.cg, the planner's steps with plain loads.
// - What this rank writes for its peers (its own stores, or the planner's earlier on the stream)
//   comes before the flag it releases or the put it makes: fence.sc.sys orders it.
// - What it stores into a peer's buffer comes after an acquire, earlier on the stream, of the
//   peer's flag that says the peer read that buffer: node_exchange's barrier for node_keys (a
//   peer publishes it once every kernel of its call before ended), node_done's of the call before
//   for node_offsets (a peer tells it after its node_exchange read them), rail_exchange's count
//   message of the call for rail_keys (a rail owner sends it after its owner_tables of the call
//   before read them).
// - A buffer the NIC reads is written again only after a flush, earlier on the stream, of the puts
//   that read it: rail_exchange's own for count_msg_out, rail_done's for key_stage.
// - The key puts are waited for late. rail_key_push only posts the puts (each slice with a strong
//   signal, so the owner that sees the signal sees the slice). rail_done acquires every rail
//   proxy's key signal and flushes this rank's puts. Between the two run node_status_exchange,
//   peer_gather, remote_rows and requester_tables: none of them reads rail_keys or writes
//   key_stage. The first reader of rail_keys, owner_tables, runs after rail_done on the stream, and
//   the planner reads the rail's offsets from the layout's tensors, not from the window. The next
//   call writes key_stage in its proxy steps, after this rail_done (and its flush: the NIC read
//   the slices) on the stream, and its rail proxies put into rail_keys only once they have this
//   rank's count message of that call, which its rail_exchange sends after this call's rail_done
//   and owner_tables. The count messages keep their in_seq parity: a rail peer's message of call
//   seq + 2 needs this rank's of seq + 1, which it sends after its rail_exchange of seq read the
//   slots.

#include <ATen/cuda/CUDAContext.h>
#include <nccl_device.h>

#include <cstdint>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/config.h"
#include "hier_output_dist/common/device.cuh"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/input_dist/input_dist.h"
#include "hier_output_dist/input_dist/node.cuh"
#include "hier_output_dist/input_dist/rail.cuh"

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
  const int64_t remote_offset = keys.remote_offset();
  push_rows(rows, num_rows, places, sizes.topo, sizes.num_features, src,
            [&](int64_t target, int64_t place, int64_t key, float weight, int32_t feature) {
              node.peer(keys.node_keys(), (int)target)[place] = key;
              if (rows.weights != nullptr)
                node.peer(keys.node_key_weights(), (int)target)[place] = weight;
              if (place >= remote_offset)
                node.peer(keys.node_bucket_feature_of_row(), (int)target)[place - remote_offset] =
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

// 1 CTA: this rank's count message (count_message_of, rail.cuh: the same for every rail peer) into
// its count_msg_out and into the call's count_msg slot of its own node. Then warp 1's lanes, each
// taking rail peers in turn, put each peer the message into the same slot in the peer's copy, one
// put that signals count[node]. Warp 0's lanes wait for the rail peers' messages. From the rail's
// num_nodes messages, read in their slots of the call, the CTA lays out the rail (rail_layout_of)
// into the call's tensors: every rank of the rail computes it from the same messages. The count
// messages go by in_seq parity: a peer's message of call seq + 2 needs this rank's of seq + 1,
// which it sends once call seq ended here.
__global__ void __launch_bounds__(kThreads)
    rail_exchange_kernel(ncclWindow_t window, ncclDevComm dev, Sizes sizes, int64_t seq,
                         CountParts parts, int64_t *scratch, RailLayoutView layout) {
  const int num_nodes = sizes.topo.num_nodes, node = sizes.topo.node, tid = threadIdx.x,
            warp = tid >> 5, lane = tid & 31;
  const CountMessage format{num_nodes, sizes.num_features};
  const NodeWindow node_window(window, sizes.topo.local_rank);
  const CtrlLayout ctrl = node_window.layout(sizes).ctrl;
  int64_t *out = ctrl.count_msg_out();
  int64_t *own = ctrl.count_msg(seq, node);
  count_message_of(parts, format, sizes.topo.local_size, out, own);
  __syncthreads();
  ncclGin gin(dev, kCtxIn);
  const ncclTeam rail = ncclTeamRail(dev);
  if (warp == 1) {
    const RailMessage counts(gin, rail, node_window, sig_count(node));
    fence_sc_sys();
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32)
      if (peer_node != node) counts.put(peer_node, own, out, (size_t)format.length() * 8);
  }
  if (warp == 0) {
    Spin spin(sizes.timeout_ns);
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32)
      while (peer_node != node && (int64_t)gin.readSignal(sig_count(peer_node)) < seq)
        spin.pause("count message", peer_node, seq, sizes.topo.rank);
  }
  __syncthreads();
  rail_layout_of(RailCounts{ctrl.count_msg(seq, 0), format.slot_words(), format}, parts.own_lengths,
                 sizes, scratch, layout);
  // count_msg_out may be written again once the NIC has read it.
  if (warp == 1) gin.flush(ncclCoopThread{});
}

// 1 warp: its lanes, each taking rail peers in turn, send each owner o this call's key stream:
// this proxy's slice of key_stage (bytes [put_src[o], put_src[o] + put_bytes[o])) into the owner's
// rail_keys at put_dst[o], one put that signals keys[node]; the signal alone when it puts no key
// (put_skip[o]: the owner overflows lookup_keys; or an empty slice). Its system fence orders the
// planner's writes of key_stage before the puts. It does not wait: rail_done waits for the rail's
// key streams and flushes these puts. A proxy puts only after it has the owner's count message of
// the call (rail_exchange, earlier on the stream), which the owner sends once its previous call
// read rail_keys.
__global__ void rail_key_push_kernel(ncclWindow_t window, ncclDevComm dev, Sizes sizes,
                                     const int64_t *put_src, const int64_t *put_dst,
                                     const int64_t *put_bytes, const bool *put_skip) {
  const int num_nodes = sizes.topo.num_nodes, node = sizes.topo.node;
  ncclGin gin(dev, kCtxIn);
  const ncclTeam rail = ncclTeamRail(dev);
  const NodeWindow node_window(window, sizes.topo.local_rank);
  const KeysLayout keys = node_window.layout(sizes).keys;
  const RailMessage key_stream(gin, rail, node_window, sig_keys(num_nodes, node));
  fence_sc_sys();
  for (int owner_node = threadIdx.x; owner_node < num_nodes; owner_node += blockDim.x) {
    if (owner_node == node) continue;
    const int64_t bytes = put_skip[owner_node] ? 0 : put_bytes[owner_node];
    if (bytes > 0)
      key_stream.put(owner_node, keys.rail_keys() + put_dst[owner_node],
                     keys.key_stage() + put_src[owner_node], (size_t)bytes);
    else
      key_stream.signal(owner_node);
  }
}

// 1 CTA: this rank's rail's kNumNeeds needs and its requester offsets [requester][feature] (the
// rail layout's) into its needs and requester_offsets words; after a system fence, which also
// orders the planner's writes of remote_map and the proxy header before it, in_done gets seq.
// Once every node peer's has it, every peer's needs land in peer_needs [local_size] [kNumNeeds]
// and where each one's KJT holds this requester's keys of each feature in
// offset_per_owner_feature [local_size][num_features].
__global__ void node_status_exchange_kernel(ncclWindow_t window, Sizes sizes, int64_t seq,
                                            const int64_t *needs, const int64_t *requester_offsets,
                                            int64_t *peer_needs,
                                            int64_t *offset_per_owner_feature) {
  const int local_size = sizes.topo.local_size, num_features = (int)sizes.num_features,
            me = sizes.topo.local_rank, tid = threadIdx.x;
  const NodeWindow node(window, me);
  const CtrlLayout ctrl = node.layout(sizes).ctrl;
  for (int i = tid; i < kNumNeeds; i += blockDim.x) ctrl.needs()[i] = needs[i];
  for (int i = tid; i < local_size * num_features; i += blockDim.x)  // requester * F + feature
    ctrl.requester_offsets(i / num_features)[i % num_features] = requester_offsets[i];
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

// 1 warp: its lanes, each taking rail peers in turn, wait until every rail proxy's key stream of
// the call has landed here: its key signal (the slice it put, which the strong signal covers, or
// the signal alone). Then the warp flushes this rank's puts on the in channel's context, so the
// NIC has read its key_stage slices before the next call's proxy steps write key_stage again. It
// runs just before owner_tables, the first reader of rail_keys, so the key puts overlap the
// steps after rail_key_push.
__global__ void rail_done_kernel(ncclDevComm dev, Sizes sizes, int64_t seq) {
  const int num_nodes = sizes.topo.num_nodes, node = sizes.topo.node;
  ncclGin gin(dev, kCtxIn);
  Spin spin(sizes.timeout_ns);
  for (int peer_node = threadIdx.x; peer_node < num_nodes; peer_node += blockDim.x)
    while (peer_node != node && (int64_t)gin.readSignal(sig_keys(num_nodes, peer_node)) < seq)
      spin.pause("key stream", peer_node, seq, sizes.topo.rank);
  gin.flush(ncclCoopWarp{});
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
  const GatherRows gather =
      gather_rows_of(bucketized.inv, bucketized.values, weights.value_or(at::Tensor()));
  const PushPlaces places{bucketized.bucket_feature_of_row.data_ptr<int32_t>(),
                          bucketized.offset_per_bucket.data_ptr<int64_t>(),
                          node.row_shift.data_ptr<int64_t>(), node.skip.data_ptr<bool>()};
  node_key_push_kernel<<<item_grid(rows), kThreads, 0, current_stream()>>>(
      window_, sizes_, gather, rows, places, data_or_null<int32_t>(src));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void InChannel::node_done() const {
  node_done_kernel<<<1, 32, 0, current_stream()>>>(window_, sizes_, seq_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

RailLayout InChannel::rail_exchange(const NodeLayout &node, const ProxyPlan &proxy) const {
  const auto options = node.own_lengths.options();
  RailLayout rail = empty_rail_layout(sizes_, options);
  const at::Tensor scratch =
      at::empty({rail_layout_scratch(sizes_.topo, sizes_.num_features)}, options);
  const CountParts parts{node.need_remote.data_ptr<int64_t>(),
                         node.own_total.data_ptr<int64_t>() + sizes_.topo.local_rank,
                         proxy.key_rows.data_ptr<int64_t>(),
                         proxy.chunk_length_per_bucket.data_ptr<int64_t>(),
                         node.own_lengths.data_ptr<int64_t>(),
                         proxy.node_length_per_bucket_feature.data_ptr<int64_t>()};
  rail_exchange_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      window_, dev_, sizes_, seq_, parts, scratch.data_ptr<int64_t>(), view_of(rail));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return rail;
}

void InChannel::rail_key_push(const at::Tensor &put_src, const RailLayout &rail) const {
  rail_key_push_kernel<<<1, 32, 0, current_stream()>>>(
      window_, dev_, sizes_, put_src.data_ptr<int64_t>(), rail.put_dst.data_ptr<int64_t>(),
      rail.put_bytes.data_ptr<int64_t>(), rail.put_skip.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::pair<at::Tensor, at::Tensor> InChannel::node_status_exchange(const RailLayout &rail) const {
  const int local_size = sizes_.topo.local_size;
  const auto options = rail.needs.options();
  at::Tensor peer_needs = at::empty({local_size, kNumNeeds}, options);
  at::Tensor offset_per_owner_feature = at::empty({local_size, sizes_.num_features}, options);
  node_status_exchange_kernel<<<1, 256, 0, current_stream()>>>(
      window_, sizes_, seq_, rail.needs.data_ptr<int64_t>(),
      rail.requester_offsets.data_ptr<int64_t>(), peer_needs.data_ptr<int64_t>(),
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

void InChannel::rail_done() const {
  rail_done_kernel<<<1, 32, 0, current_stream()>>>(dev_, sizes_, seq_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
