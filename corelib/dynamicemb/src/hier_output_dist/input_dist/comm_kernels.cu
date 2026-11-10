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
//   node_exchange         this rank's row (NodeRow) into node_offsets at each node peer, over
//                         NVLink (a large row: node_rows_push, a grid that does not wait, just
//                         before); the node barrier of the call; the node layout from the node's
//                         rows (its items on a grid just after when they are many:
//                         node_layout_items);
//   node_key_push         the bucketize's gather: this rank's keys into their places in their
//                         targets' node_keys, over NVLink;
//   node_done             tells each target its keys are there, and waits until every requester
//                         of the node told it;
//   rail_exchange         (across nodes) its count message to each rail peer, and theirs; the
//                         rail's layout from the rail's messages (rail.cuh);
//   rail_key_push         (across nodes) its key records to each rail owner; it does not wait;
//   node_status_exchange  (across nodes) its rail's needs and its group shifts to its node, and
//                         every node peer's;
//   requester             (across nodes) a grid that does not wait: the requester's tables from
//                         its node peers' proxy headers and remote_map, read in their copies,
//                         and the owner's records (requester.cuh);
//   rail_done             (across nodes) waits for every rail proxy's key records, and flushes
//                         this rank's puts; just before owner_tables, the first reader of
//                         rail_keys.
// A kernel that waits runs on one CTA (or warp). It waits only for the same kernel (the same seq)
// of its node or rail peers, but rail_done, which waits for its rail peers' rail_key_push of the
// call, a kernel that never waits. Its threads (or lanes) take the peers in turn, so any number of
// them.
// Memory order:
// - What peers wrote is read only after an acquire of their flag or signal. The comm kernel that
//   acquired it reads with ld.global.cg. A later kernel on the stream reads it too: the requester
//   kernel with ld.global.cg, the planner's steps with plain loads. A node peer writes its proxy
//   header and remote_map in its proxy steps, before its node_status_exchange publishes in_status;
//   it writes them again only in its next call's proxy steps, after that call's node barrier,
//   which waits for this rank's in_started of that call: this rank releases it in its next
//   node_exchange, which its stream starts only once its requester kernel of this call has ended.
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
//   proxy's key signal and flushes this rank's puts. Between the two run node_status_exchange and
//   the requester kernel: neither reads rail_keys or writes key_stage. The first reader of
//   rail_keys, owner_tables, runs after rail_done on the stream, and the planner reads the rail's
//   layout from the layout's tensors, not from the window. The next call writes key_stage in its
//   proxy steps, after this rail_done (and its flush: the NIC read the slices) on the stream, and
//   its rail proxies put into rail_keys only once they have this rank's count message of that
//   call, which its rail_exchange sends after this call's rail_done and owner_tables. The count
//   messages keep their in_seq parity: a rail peer's message of call seq + 2 needs this rank's of
//   seq + 1, which it sends after its rail_exchange of seq read the slots.

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
#include "hier_output_dist/input_dist/requester.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// Each comm kernel takes the op's window (and, on the rail, its GIN device
// communicator), its sizes, and if it waits the call's sequence number
// (seq, in_seq): then the call's tensors.

// This rank's row of node_offsets (NodeRow, from its bucketize's offset_per_bucket_feature) into
// its place at every node peer, its own too, over NVLink, in 16-byte units: items [first, end) in
// steps of `stride`, an item a unit at a peer, peer after peer.
__device__ __forceinline__ void push_node_row(const NodeWindow &node, const KeysLayout &keys,
                                              const Sizes &sizes, const int64_t *opbf,
                                              int64_t first, int64_t stride) {
  const NodeRow row = NodeRow::of(sizes.topo, sizes.num_features);
  const int64_t units = row.words / 2;
  int4 *mine = (int4 *)keys.node_offsets(sizes.topo.local_rank);
  for (int64_t item = first; item < sizes.topo.local_size * units; item += stride) {
    const int64_t peer = item / units, unit = item % units;
    const int64_t lo = row.word(opbf, sizes.topo, sizes.num_features, 2 * unit),
                  hi = row.word(opbf, sizes.topo, sizes.num_features, 2 * unit + 1);
    int4 value;
    value.x = (int)(uint32_t)lo;
    value.y = (int)(uint32_t)((uint64_t)lo >> 32);
    value.z = (int)(uint32_t)hi;
    value.w = (int)(uint32_t)((uint64_t)hi >> 32);
    node.peer(mine, (int)peer)[unit] = value;
  }
}

// Every CTA, a thread per 16-byte unit of the row at a peer: the push of this rank's row
// (push_node_row) when it is too large for node_exchange's CTA. It does not wait: this rank's
// node_done of the call before, earlier on the stream, saw every peer tell it read its
// node_offsets. Each CTA fences its stores at system scope before it ends, so node_exchange's
// release, after it on the stream, publishes them.
__global__ void __launch_bounds__(kThreads)
    node_rows_push_kernel(ncclWindow_t window, Sizes sizes,
                          const int64_t *offset_per_bucket_feature) {
  const NodeWindow node(window, sizes.topo.local_rank);
  push_node_row(node, node.layout(sizes).keys, sizes, offset_per_bucket_feature, first_item(),
                item_stride());
  __syncthreads();
  if (threadIdx.x == 0) fence_sc_sys();
}

// 1 CTA: this rank's row (NodeRow: what the node layout reads of its bucketize's
// offset_per_bucket_feature) goes into its place in node_offsets at each node peer, its own too,
// over NVLink, unless node_rows_push did (push false). Then a system fence, and its in_started
// gets the call (the node barrier). It waits until each peer's in_started holds the call: every
// peer's row is then in node_offsets here. From them it makes the node layout (node_layout_small;
// and its items (node_layout_items) unless node_layout_items_kernel does them just after, items
// false; on one node also the owner's KJT offsets). Passing the barrier also means every peer
// finished its previous call's in channel, whose steps read what this call writes into their
// copies of the window (node_keys). delay_ns (the test hook delay_node_reads, 0: none): thread 0
// waits that long after the barrier, so the layout reads node_offsets late.
__global__ void __launch_bounds__(kThreads)
    node_exchange_kernel(ncclWindow_t window, Sizes sizes, int64_t seq,
                         const int64_t *offset_per_bucket_feature, bool push, bool items,
                         int64_t *scratch, int64_t delay_ns, NodeLayoutView layout) {
  const int local_size = sizes.topo.local_size, me = sizes.topo.local_rank, tid = threadIdx.x;
  const NodeWindow node(window, me);
  const KeysLayout keys = node.layout(sizes).keys;
  if (push) push_node_row(node, keys, sizes, offset_per_bucket_feature, tid, blockDim.x);
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
  const NodeRows rows{keys.node_offsets(0), NodeRow::of(sizes.topo, sizes.num_features)};
  node_layout_small(rows, offset_per_bucket_feature, sizes.topo, sizes.num_features,
                    sizes.lookup_keys, sizes.remote_keys, scratch, layout);
  if (!items) return;
  __syncthreads();
  node_layout_items(rows, offset_per_bucket_feature, sizes.topo, sizes.num_features, scratch, tid,
                    blockDim.x, layout);
}

// Every CTA, a thread per item: the node layout's items (node_layout_items) when they are too
// many for node_exchange's CTA, just after it on the stream, from node_layout_small's scratch. It
// reads node_offsets before this rank's node_done of the call, as node_exchange does.
__global__ void __launch_bounds__(kThreads)
    node_layout_items_kernel(ncclWindow_t window, Sizes sizes,
                             const int64_t *offset_per_bucket_feature, const int64_t *scratch,
                             NodeLayoutView layout) {
  const NodeWindow node(window, sizes.topo.local_rank);
  const NodeRows rows{node.layout(sizes).keys.node_offsets(0),
                      NodeRow::of(sizes.topo, sizes.num_features)};
  node_layout_items(rows, offset_per_bucket_feature, sizes.topo, sizes.num_features, scratch,
                    first_item(), item_stride(), layout);
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
  push_rows(
      rows, num_rows, places, sizes.topo, sizes.num_features, src,
      [&](int64_t target, int64_t place, int64_t key, float weight, int32_t node_bucket_feature) {
        node.peer(keys.node_keys(), (int)target)[place] = key;
        if (rows.weights != nullptr)
          node.peer(keys.node_key_weights(), (int)target)[place] = weight;
        if (place >= remote_offset)
          node.peer(keys.node_bucket_feature_of_row(), (int)target)[place - remote_offset] =
              node_bucket_feature;
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

// 1 CTA: this rank's count messages (count_messages_of, rail.cuh: a header the same for every
// rail peer, then the peer's own part) into count_msg_out, a row per peer, and its own into the
// call's count_msg slot of its own node. Then warp 1's lanes, each taking rail peers in turn, put
// each peer its row into the same slot in the peer's copy, one put that signals count[node]. Warp
// 0's lanes wait for the rail peers' messages. From the rail's num_nodes messages, read in their
// slots of the call, and its own part's lengths (own_lengths [feature][requester], the node
// layout's), the CTA lays out the rail (rail_layout_of) into the call's tensors: every rank of the
// rail computes the rail's runs and shifts from the same headers. The count messages go by in_seq
// parity: a peer's message of call seq + 2 needs this rank's of seq + 1, which it sends once call
// seq ended here.
__global__ void __launch_bounds__(kThreads)
    rail_exchange_kernel(ncclWindow_t window, ncclDevComm dev, Sizes sizes, int64_t seq,
                         CountParts parts, const int64_t *own_lengths, int64_t *scratch,
                         RailLayoutView layout) {
  const int num_nodes = sizes.topo.num_nodes, node = sizes.topo.node, tid = threadIdx.x,
            warp = tid >> 5, lane = tid & 31;
  const CountMessage format{num_nodes, sizes.num_features, sizes.num_groups};
  const NodeWindow node_window(window, sizes.topo.local_rank);
  const CtrlLayout ctrl = node_window.layout(sizes).ctrl;
  int64_t *own = ctrl.count_msg(seq, node);
  count_messages_of(parts, format, sizes.topo.local_size, node, ctrl.count_msg_out(0), own);
  __syncthreads();
  ncclGin gin(dev, kCtxIn);
  const ncclTeam rail = ncclTeamRail(dev);
  if (warp == 1) {
    const RailMessage counts(gin, rail, node_window, sig_count(node));
    fence_sc_sys();
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32)
      if (peer_node != node)
        counts.put(peer_node, own, ctrl.count_msg_out(peer_node), (size_t)format.length() * 8);
  }
  if (warp == 0) {
    Spin spin(sizes.timeout_ns);
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32)
      while (peer_node != node && (int64_t)gin.readSignal(sig_count(peer_node)) < seq)
        spin.pause("count message", peer_node, seq, sizes.topo.rank);
  }
  __syncthreads();
  rail_layout_of(RailCounts{ctrl.count_msg(seq, 0), format.slot_words(), format}, own_lengths,
                 sizes, scratch, layout);
  // count_msg_out may be written again once the NIC has read it.
  if (warp == 1) gin.flush(ncclCoopThread{});
}

// 1 warp: its lanes, each taking rail peers in turn, send each owner o this call's key message:
// this proxy's slice of key_stage (bytes [put_src[o], put_src[o] + put_bytes[o])) into the owner's
// rail_keys at put_dst[o], one put that signals keys[node]; the signal alone when it puts no key
// (put_skip[o]: the owner overflows lookup_keys; or an empty slice). Its system fence orders the
// planner's writes of key_stage before the puts. It does not wait: rail_done waits for the rail's
// key messages and flushes these puts. A proxy puts only after it has the owner's count message of
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
  const RailMessage key_message(gin, rail, node_window, sig_keys(num_nodes, node));
  fence_sc_sys();
  for (int owner_node = threadIdx.x; owner_node < num_nodes; owner_node += blockDim.x) {
    if (owner_node == node) continue;
    const int64_t bytes = put_skip[owner_node] ? 0 : put_bytes[owner_node];
    if (bytes > 0)
      key_message.put(owner_node, keys.rail_keys() + put_dst[owner_node],
                      keys.key_stage() + put_src[owner_node], (size_t)bytes);
    else
      key_message.signal(owner_node);
  }
}

// 1 CTA: this rank's rail's kNumNeeds needs and its group shifts [group] (the rail layout's) into
// its needs and group_shift words; after a system fence, which also orders the planner's writes of
// remote_map and the proxy header before it, in_status gets seq. Once every node peer's has it,
// every peer's needs land in peer_needs [local_size][kNumNeeds] and its group shifts in
// owner_group_shift [local_size][num_groups].
__global__ void node_status_exchange_kernel(ncclWindow_t window, Sizes sizes, int64_t seq,
                                            const int64_t *needs, const int64_t *group_shift,
                                            int64_t *peer_needs, int64_t *owner_group_shift) {
  const int local_size = sizes.topo.local_size, num_groups = (int)sizes.num_groups,
            me = sizes.topo.local_rank, tid = threadIdx.x;
  const NodeWindow node(window, me);
  const CtrlLayout ctrl = node.layout(sizes).ctrl;
  for (int i = tid; i < kNumNeeds; i += blockDim.x) ctrl.needs()[i] = needs[i];
  for (int i = tid; i < num_groups; i += blockDim.x) ctrl.group_shift()[i] = group_shift[i];
  __syncthreads();
  const NodeBarrier status{sizes, node, ctrl.in_status(), seq};
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
  for (int i = tid; i < local_size * num_groups; i += blockDim.x)
    owner_group_shift[i] = word_of(i / num_groups, ctrl.group_shift() + i % num_groups);
}

// The node's proxies' buffers as the requester kernel reads them (requester.cuh): node peer p's
// proxy header and remote_map, in its copy of the window, at the places of this rank's in its own.
// Their loads go through L2 only. The peers published them before the in_status this rank's
// node_status_exchange acquired earlier on the stream.
struct WindowPeers {
  ncclWindow_t window;
  Sizes sizes;
  __device__ __forceinline__ const int32_t *header(int peer) const {
    const NodeWindow node(window, sizes.topo.local_rank);
    return node.peer(node.layout(sizes).keys.proxy_header().base, peer);
  }
  __device__ __forceinline__ const int32_t *remote_map(int peer) const {
    const NodeWindow node(window, sizes.topo.local_rank);
    return node.peer(node.layout(sizes).keys.remote_map().base, peer);
  }
};

// 1 warp: its lanes, each taking rail peers in turn, wait until every rail proxy's key message of
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
      spin.pause("key message", peer_node, seq, sizes.topo.rank);
  gin.flush(ncclCoopWarp{});
}

// The launchers.

namespace {
// The test hook's paths of node_exchange (force_node_paths): -1 as the sizes choose, 0 a grid,
// 1 its CTA.
int64_t g_node_push = -1, g_node_items = -1;
}  // namespace

void force_node_paths(int64_t push, int64_t items) {
  g_node_push = push;
  g_node_items = items;
}

NodeLayout InChannel::node_exchange(const BucketizedKeys &bucketized, int64_t delay_ns) {
  const auto options = bucketized.offset_per_bucket.options();
  const Topo &topo = sizes_.topo;
  NodeLayout node = empty_node_layout(topo, sizes_.num_features, options);
  const at::Tensor scratch = at::empty({node_layout_scratch(topo, sizes_.num_features)}, options);
  const int64_t *opbf = bucketized.offset_per_bucket_feature.data_ptr<int64_t>();
  // node_exchange's CTA pushes the row and lays out the items when they are few; else a grid
  // just before it pushes, and one just after lays out (no allocation in between: see
  // input_dist.cu).
  const NodeRow row = NodeRow::of(topo, sizes_.num_features);
  const int64_t items = node_layout_items_of(topo, sizes_.num_features);
  const bool push =
      g_node_push >= 0 ? g_node_push > 0 : topo.local_size * row.words <= kNodeRowWordsInCta;
  const bool lay_out = g_node_items >= 0 ? g_node_items > 0 : items <= kNodeItemsInCta;
  if (!push) {
    node_rows_push_kernel<<<item_grid(topo.local_size * row.words / 2), kThreads, 0,
                            current_stream()>>>(window_, sizes_, opbf);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  // The call's sequence number. seq_ takes it only once the kernel is launched.
  const int64_t seq = seq_ + 1;
  node_exchange_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      window_, sizes_, seq, opbf, push, lay_out, scratch.data_ptr<int64_t>(), delay_ns,
      view_of(node));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  seq_ = seq;
  if (!lay_out) {
    node_layout_items_kernel<<<item_grid(items), kThreads, 0, current_stream()>>>(
        window_, sizes_, opbf, scratch.data_ptr<int64_t>(), view_of(node));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
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
      at::empty({rail_layout_scratch(sizes_.topo, sizes_.num_groups)}, options);
  // the keys of this rank's own part: its entry of the node's own_total
  const int64_t *own_total = node.own_total.data_ptr<int64_t>() + sizes_.topo.local_rank;
  const CountParts parts{node.need_remote.data_ptr<int64_t>(),
                         node.own_offsets.data_ptr<int64_t>(),
                         own_total,
                         proxy.need_key_rows.data_ptr<int64_t>(),
                         proxy.chunk_length_per_bucket.data_ptr<int64_t>(),
                         proxy.node_length_per_bucket_feature.data_ptr<int64_t>(),
                         proxy.node_offset_per_bucket_feature.data_ptr<int64_t>(),
                         group_starts_.data_ptr<int64_t>()};
  rail_exchange_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      window_, dev_, sizes_, seq_, parts, node.own_lengths.data_ptr<int64_t>(),
      scratch.data_ptr<int64_t>(), view_of(rail));
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
  at::Tensor owner_group_shift = at::empty({local_size, sizes_.num_groups}, options);
  node_status_exchange_kernel<<<1, 256, 0, current_stream()>>>(
      window_, sizes_, seq_, rail.needs.data_ptr<int64_t>(), rail.group_shift.data_ptr<int64_t>(),
      peer_needs.data_ptr<int64_t>(), owner_group_shift.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {peer_needs, owner_group_shift};
}

std::pair<RequesterTables, OwnerRecords> InChannel::requester(const BucketizedKeys &bucketized,
                                                              const NodeLayout &node,
                                                              const at::Tensor &owner_group_shift,
                                                              const RailLayout &rail) const {
  return requester_tables_of(WindowPeers{window_, sizes_}, bucketized.bucket_feature_of_row,
                             bucketized.offset_per_bucket, node.row_shift, owner_group_shift,
                             group_of_feature_, &rail, other_nodes_, sizes_);
}

void InChannel::rail_done() const {
  rail_done_kernel<<<1, 32, 0, current_stream()>>>(dev_, sizes_, seq_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
