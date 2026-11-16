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
// The input dist of a call: the in channel's one step (input_dist.h).

#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>

#include "hier_output_dist/input_dist/input_dist.h"
#include "hier_output_dist/input_dist/planner.cuh"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

// The test hook's wait (delay_node_reads), in ns.
int64_t g_read_delay_ns = 0;

// One CTA: the call's status (StatusSlots), which the caller copies to the
// host at the call's second wait. Its needs: across nodes the most of the
// node's (peer_needs, each rank's of its rail, kNumNeeds each): a rank's
// rail covers every node, and its node holds a rank of every rail, so the
// most of the node is the most of all ranks. On one node its lookup need
// alone. occurrences: nullptr on one node.
__global__ void __launch_bounds__(kThreads)
    call_status_kernel(const int64_t *peer_needs, int64_t local_size, const int64_t *need_lookup,
                       const int64_t *num_recv, const int64_t *heavy_subs,
                       const int64_t *occurrences, const int64_t *length_per_key,
                       const int64_t *input_length_per_key, int64_t num_features, int64_t *status) {
  const int tid = threadIdx.x;
  if (tid < kNumNeeds) {
    int64_t need = 0;
    if (peer_needs != nullptr)
      for (int64_t peer = 0; peer < local_size; ++peer)
        need = max(need, peer_needs[peer * kNumNeeds + tid]);
    else if (tid == kNeedLookupKeys)
      need = *need_lookup;
    status[kStatusNeed + tid] = need;
  }
  if (tid == 0) {
    status[kStatusNumRecv] = *num_recv;
    status[kStatusHeavySubs] = *heavy_subs;
    status[kStatusOccurrences] = occurrences != nullptr ? *occurrences : 0;
  }
  for (int64_t feature = tid; feature < num_features; feature += kThreads) {
    status[kStatusLengthPerKey + feature] = length_per_key[feature];
    status[kStatusLengthPerKey + num_features + feature] = input_length_per_key[feature];
  }
}

// One thread: waits `ns` on the GPU's clock (delay_node_reads).
__global__ void delay_kernel(uint64_t ns) {
  const uint64_t start = now_ns();
  while (now_ns() - start < ns) __nanosleep(1000);
}

}  // namespace

void delay_node_reads(int64_t ns) { g_read_delay_ns = ns; }

InChannel::InChannel(const Config &config, const Sizes &sizes, ncclWindow_t window,
                     const ncclDevComm &dev, const WindowLayout &layout, int device)
    : window_(window),
      dev_(dev),
      sizes_(sizes),
      weighted_(config.weighted),
      node_dedup_(config.node_dedup),
      device_(device) {
  const auto options = at::TensorOptions().device(at::kCUDA, device);
  // the bucketize's rules, and the fan-out's words, on one node too (before its return below):
  // copies the host waits for, so a call on any stream finds them
  dist_types_ = at::tensor(config.dist_types, at::kInt).to(options);
  fanout_words_ = at::zeros({kFanoutWords}, at::kLong).to(options);
  auto view = [&](void *buffer, at::IntArrayRef shape, at::ScalarType dtype) {
    return at::from_blob(buffer, shape, options.dtype(dtype));
  };
  const KeysLayout &keys = layout.keys;
  const int64_t lookup_keys = sizes_.lookup_keys, remote_keys = sizes_.remote_keys;
  const at::Tensor node_keys = view(keys.node_keys(), {lookup_keys + remote_keys}, at::kLong);
  own_node_keys_ = node_keys.narrow(0, 0, keys.remote_offset());
  at::Tensor node_key_weights;
  if (config.weighted) {
    node_key_weights = view(keys.node_key_weights(), {lookup_keys + remote_keys}, at::kFloat);
    own_node_key_weights_ = node_key_weights.narrow(0, 0, keys.remote_offset());
  }
  if (!sizes_.multi_node()) return;
  remote_node_keys_ = node_keys.narrow(0, keys.remote_offset(), remote_keys);
  if (config.weighted)
    remote_node_key_weights_ = node_key_weights.narrow(0, keys.remote_offset(), remote_keys);
  node_bucket_feature_of_row_ = view(keys.node_bucket_feature_of_row(), {remote_keys}, at::kInt);
  proxy_buffers_ =
      proxy_buffers(view(keys.remote_map().base, {RemoteMap::kArrays, remote_keys}, at::kInt),
                    view(keys.key_stage(), {(int64_t)keys.key_stage_bytes() / 8}, at::kLong),
                    keys.proxy_header(), options);
  rail_keys_ = view(keys.rail_keys(), {(int64_t)keys.rail_keys_bytes() / 8}, at::kLong);
  tables_ = at::tensor(dense_tables(config.feature_tables), at::kInt).to(options);
  other_nodes_ = other_nodes_of(sizes_.topo, options);
  const std::vector<int64_t> starts = config.group_starts();
  std::vector<int> group_of_feature;
  for (int64_t group = 0; group + 1 < (int64_t)starts.size(); ++group)
    group_of_feature.insert(group_of_feature.end(), starts[group + 1] - starts[group], (int)group);
  group_of_feature_ = at::tensor(group_of_feature, at::kInt).to(options);
  group_starts_ = at::tensor(starts, at::kLong).to(options);
}

InDist InChannel::input_dist(const PlanInput &input) {
  c10::cuda::CUDAGuard device_guard(device_);
  at::NoGradGuard no_grad;
  // The inputs first. A call takes its sequence number when it launches its first waiting comm
  // kernel (node_exchange; a large row's push runs just before it and waits for nothing: no peer
  // reads the row before this rank's in_started holds the call, and a later call writes it
  // again). A call that raises before that, on an input check or a failed allocation,
  // takes none, and its node peers wait at the node barrier for the call that does. A call that
  // raises between its first and its last comm kernel can leave its peers waiting.
  TORCH_CHECK(input.weights.has_value() == weighted_,
              "hier_output_dist: the keys carry weights exactly when the op was made with "
              "weights (",
              weighted_ ? "with" : "without", ")");
  const Topo &topo = sizes_.topo;
  check_bucketize_input(input, dist_types_, topo.world_size);
  check_fanout_input(input.reverse_indices);
  // The bucketize up to its gather (the push gathers), and the fan-out of its rows. Its offsets
  // per bucket feature are those node_exchange reads: each owner's first one, and every one of
  // this node's owners.
  const BucketizedKeys bucketized = bucketize_rows(
      input, dist_types_, topo.world_size, (int64_t)topo.node * topo.local_size, topo.local_size);
  const FanoutTables fanout = fanout_tables(input.reverse_indices, bucketized.inv, fanout_words_);
  const int64_t local_size = topo.local_size, num_nodes = topo.num_nodes,
                num_features = sizes_.num_features;
  const auto long_options = bucketized.offset_per_bucket.options();
  InDist dist;
  FwdPlan &fwd = dist.fwd_plan;
  BwdPlan &bwd = dist.bwd_plan;
  fwd.offset_per_bucket = bwd.offset_per_bucket = bucketized.offset_per_bucket;
  fwd.inv = bwd.inv = bucketized.inv;
  fwd.fanout = bwd.fanout = fanout;
  // The node: every rank's row of offsets (and the node barrier), and the layout of the node's
  // keys. Then this rank's keys go to their places in their targets' node_keys, and the node
  // tells that they are there. src is allocated first: nothing allocates from node_exchange (and
  // its row push) to node_done. node_done runs in every call, also with no rows and under skip: its
  // tell also says that this rank's node_exchange read node_offsets, so its peers may write them
  // again.
  const int64_t rows = bucketized.inv.numel();
  at::Tensor src;  // one node: each row's place in its owner's KJT
  if (num_nodes == 1) src = at::empty({rows}, bucketized.inv.options());
  const NodeLayout node = node_exchange(bucketized, g_read_delay_ns);
  if (rows > 0) node_key_push(bucketized, input.weights, node, src);
  node_done();
  if (g_read_delay_ns > 0) {  // the test hook (delay_node_reads)
    delay_kernel<<<1, 1, 0, current_stream()>>>((uint64_t)g_read_delay_ns);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  at::Tensor peer_needs, num_recv, length_per_key, occurrences;
  if (num_nodes == 1) {
    // The own part of node_keys is the owner's KJT, and a row's place in its
    // owner's KJT is where it was pushed.
    const OneNodeOwner owner =
        one_node_owner(node, own_node_keys_, own_node_key_weights_, src, topo);
    fwd.src = bwd.dst = owner.src;
    dist.kjt_values = owner.kjt_values;
    dist.kjt_weights = owner.kjt_weights;
    dist.kjt_lengths = owner.kjt_lengths;
    dist.kjt_offsets = owner.kjt_offsets;
    num_recv = owner.num_recv;
    length_per_key = owner.length_per_key;
  } else {
    // The proxy: the node-unique keys of its remote part, their rows and
    // the chunks of its stream, and its node's push orders (into
    // remote_map, key_stage and the proxy header).
    ProxyKeys keys =
        proxy_keys(ProxyInput{remote_node_keys_, remote_node_key_weights_,
                              node_bucket_feature_of_row_, node.remote_offsets, node.skip, tables_},
                   sizes_, node_dedup_);
    ProxyChunks chunks = proxy_chunks_and_slots(keys, node.remote_offsets, proxy_buffers_, sizes_);
    const ProxyPlan proxy = proxy_push_order(std::move(keys), std::move(chunks),
                                             node.remote_offsets, proxy_buffers_, sizes_);
    // The rail: every rank's count message to its rail, and the rail's layout
    // from them; then the proxy's key records into its owners' rail_keys,
    // which rail_done waits for just before the owner's tables.
    const RailLayout rail = rail_exchange(node, proxy);
    rail_key_push(proxy.stage_offset_per_bucket, rail);
    // The node again: the call's needs, and the group shifts of the node's
    // owners. Then the requester's tables, read from what the node's proxies
    // published, and the owner's records of each other node, in one kernel.
    at::Tensor owner_group_shift;
    std::tie(peer_needs, owner_group_shift) = node_status_exchange(rail);
    RequesterTables requester;
    OwnerRecords records;
    std::tie(requester, records) = this->requester(bucketized, node, owner_group_shift, rail);
    // Every rail proxy's key records are in rail_keys, and the NIC read this
    // proxy's (comm_kernels.cu says why nothing before reads or writes them).
    rail_done();
    // The owner's tables also write the requester's remote task offsets past
    // its proxies' chunks.
    const OwnerTables owner =
        owner_tables(rail, records, requester, node.own_offsets, own_node_keys_,
                     own_node_key_weights_, rail_keys_, other_nodes_, group_of_feature_, sizes_);
    // The plans: the requester's maps and its tables of the node's proxies,
    // the owner's and the proxy's tables, the proxy's rows before each key
    // and the group runs of the rows the rail puts. Then the owner's KJT and
    // its row map.
    fwd.src = requester.src;
    bwd.dst = requester.dst;
    fwd.push_order = bwd.push_order = requester.push_order;
    fwd.proxy_num_chunks = requester.proxy_num_chunks;
    fwd.proxy_ends = bwd.proxy_ends = requester.proxy_ends;
    fwd.proxy_push_tables = bwd.proxy_push_tables = requester.proxy_push_tables;
    fwd.remote_task_offsets = bwd.remote_task_offsets = requester.remote_task_offsets;
    fwd.num_chunks = bwd.num_chunks = proxy.num_chunks;
    fwd.chunks = bwd.chunks = proxy.chunks;
    fwd.chunk_offset_per_bucket = bwd.chunk_offset_per_bucket = proxy.chunk_offset_per_bucket;
    fwd.chunk_inv = bwd.chunk_inv = proxy.chunk_inv;
    fwd.node_offset_per_bucket = bwd.node_offset_per_bucket = proxy.node_offset_per_bucket;
    fwd.push_order_off = requester.push_order_off;
    fwd.fwd_chunks = owner.fwd_chunks;
    fwd.fwd_chunk_offsets = owner.fwd_chunk_offsets;
    fwd.fwd_groups = rail.fwd_groups;
    fwd.node_length_per_bucket = proxy.node_length_per_bucket;
    bwd.bwd_groups = rail.bwd_groups;
    bwd.node_length_per_source = rail.node_length_per_source;
    bwd.reduce_task_offsets = proxy.reduce_task_offsets;
    bwd.node_exp_off = proxy.node_exp_off;
    dist.kjt_values = owner.kjt_values;
    dist.kjt_weights = owner.kjt_weights;
    dist.row_map = owner.row_map;
    dist.kjt_lengths = rail.kjt_lengths;
    dist.kjt_offsets = rail.kjt_offsets;
    occurrences = proxy.occurrence_count;
    num_recv = rail.num_recv;
    length_per_key = rail.length_per_key;
  }
  // The status, which the caller reads at the call's second wait.
  dist.status = at::empty({status_words(num_features)}, long_options);
  call_status_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      data_or_null<int64_t>(peer_needs), local_size, node.need_lookup.data_ptr<int64_t>(),
      num_recv.data_ptr<int64_t>(), fanout.count.data_ptr<int64_t>() + 1,
      data_or_null<int64_t>(occurrences), length_per_key.data_ptr<int64_t>(),
      bucketized.input_length_per_key.data_ptr<int64_t>(), num_features,
      dist.status.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dist;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
