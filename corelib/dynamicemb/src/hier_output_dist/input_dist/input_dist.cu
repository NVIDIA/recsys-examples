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

#include <algorithm>

#include "hier_output_dist/input_dist/input_dist.h"
#include "hier_output_dist/input_dist/planner.cuh"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

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

}  // namespace

InChannel::InChannel(const Config &config, const Sizes &sizes, ncclWindow_t window,
                     const ncclDevComm &dev, const WindowLayout &layout, int device)
    : window_(window),
      dev_(dev),
      sizes_(sizes),
      rules_{config.dist_types, config.block_sizes},
      weights_(config.weights),
      node_dedup_(config.node_dedup),
      device_(device) {
  const auto options = at::TensorOptions().device(at::kCUDA, device);
  auto view = [&](void *buffer, at::IntArrayRef shape, at::ScalarType dtype) {
    return at::from_blob(buffer, shape, options.dtype(dtype));
  };
  const KeysLayout &keys = layout.keys;
  const int64_t lookup_keys = sizes_.lookup_keys, remote_keys = sizes_.remote_keys;
  const at::Tensor node_keys = view(keys.node_keys(), {lookup_keys + remote_keys}, at::kLong);
  own_node_keys_ = node_keys.narrow(0, 0, keys.remote_start());
  at::Tensor node_key_weights;
  if (config.weights) {
    node_key_weights = view(keys.node_key_weights(), {lookup_keys + remote_keys}, at::kFloat);
    own_node_key_weights_ = node_key_weights.narrow(0, 0, keys.remote_start());
  }
  if (!sizes_.multi_node()) return;
  const int64_t num_nodes = sizes_.topo.num_nodes, num_features = sizes_.num_features;
  remote_node_keys_ = node_keys.narrow(0, keys.remote_start(), remote_keys);
  if (config.weights)
    remote_node_key_weights_ = node_key_weights.narrow(0, keys.remote_start(), remote_keys);
  node_bucket_feature_of_row_ = view(keys.node_bucket_feature_of_row(), {remote_keys}, at::kInt);
  proxy_buffers_ =
      proxy_buffers(view(keys.remote_map().base, {RemoteMap::kArrays, remote_keys}, at::kInt),
                    view(keys.key_stage(), {(int64_t)keys.key_stage_bytes() / 8}, at::kLong),
                    keys.proxy_header(), options);
  rail_keys_ = view(keys.rail_keys(), {(int64_t)keys.rail_keys_bytes() / 8}, at::kLong);
  source_offsets_ = view(keys.source_offsets(0), {num_nodes, num_features}, at::kLong);
  owner_offsets_ = view(keys.owner_offsets(0), {num_nodes, num_features}, at::kLong);
  tables_ = at::tensor(dense_tables(config.feature_tables), at::kInt).to(options);
  other_nodes_ = other_nodes_of(sizes_.topo, options);
  header_gather_ = header_gather(sizes_, options);
}

InDist InChannel::input_dist(const PlanInput &input) {
  c10::cuda::CUDAGuard device_guard(device_);
  at::NoGradGuard no_grad;
  TORCH_CHECK(input.weights.has_value() == weights_,
              "hier_output_dist: the keys carry weights exactly when the op was made with "
              "weights (",
              weights_ ? "with" : "without", ")");
  ++seq_;
  // The bucketize: the call's keys by owner, and the fan-out of its
  // deduplicated rows.
  const BucketizedKeys bucketized = bucketize_keys(input, rules_, sizes_.topo.world_size);
  const FanoutTables fanout = fanout_tables(input.reverse_indices, bucketized.inv);
  const Topo &topo = sizes_.topo;
  const int64_t local_size = topo.local_size, num_nodes = topo.num_nodes,
                num_features = sizes_.num_features, lookup_keys = sizes_.lookup_keys,
                remote_keys = sizes_.remote_keys;
  const auto long_options = bucketized.offset_per_bucket.options();
  InDist dist;
  FwdPlan &fwd = dist.fwd_plan;
  BwdPlan &bwd = dist.bwd_plan;
  fwd.offset_per_bucket = bwd.offset_per_bucket = bucketized.offset_per_bucket;
  fwd.inv = bwd.inv = bucketized.inv;
  fwd.fanout = bwd.fanout = fanout;
  // The node: every rank's counts (and the node barrier), the layout of the
  // node's keys, and this rank's keys pushed to their targets.
  const auto [totals, before, mine] = node_count_exchange(
      node_count_vector(bucketized.offset_per_bucket_feature, topo, num_features));
  const NodeLayout node =
      node_layout(NodeCounts{totals, before, mine}, bucketized.offset_per_bucket_feature,
                  bucketized.offset_per_bucket, topo, num_features, lookup_keys, remote_keys);
  node_key_push(bucketized.keys, bucketized.key_weights, bucketized.bucket_feature_of_row,
                node.destination_base, node.rows_to_push);
  node_done();
  at::Tensor peer_needs, num_recv, length_per_key, occurrences;
  if (num_nodes == 1) {
    // The own part of node_keys is the owner's KJT, and a row's place in its
    // owner's KJT is where it was pushed.
    const OneNodeOwner owner =
        one_node_owner(node, bucketized.bucket_feature_of_row, own_node_keys_,
                       own_node_key_weights_, topo, num_features);
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
    ProxyKeys keys = proxy_keys(
        ProxyInput{remote_node_keys_, remote_node_key_weights_, node_bucket_feature_of_row_,
                   node.requester_start, node.skip, tables_},
        sizes_, node_dedup_);
    ProxyChunks chunks = proxy_chunks_and_slots(keys, node.requester_start, proxy_buffers_, sizes_);
    const ProxyPlan proxy = proxy_push_order(std::move(keys), std::move(chunks),
                                             node.requester_start, proxy_buffers_, sizes_);
    // The rail: the count messages, the layout of the rail, then the
    // proxies' key records into the owners' rail_keys and the owners'
    // source_offsets into the proxies' owner_offsets.
    const at::Tensor message =
        count_message(node.need_remote, node.own_total.select(0, topo.local_rank), proxy.key_rows,
                      proxy.node_length_per_bucket, proxy.chunk_length_per_bucket,
                      proxy.node_length_per_bucket_feature);
    const at::Tensor messages = rail_count_exchange(message);
    const RailLayout rail = rail_layout(messages, node.own_lengths, other_nodes_, sizes_);
    source_offsets_.copy_(rail.source_offsets);
    rail_key_exchange(proxy.stage_offset_per_bucket, rail.put_dst, rail.put_bytes, rail.put_skip);
    // The node again: the call's needs, where the node's owners hold this
    // requester's keys of each feature, and what the node's proxies published
    // for it.
    at::Tensor offset_per_owner_feature;
    std::tie(peer_needs, offset_per_owner_feature) = node_status_exchange(status_publish(rail));
    const at::Tensor headers =
        peer_gather(kGatherProxyHeader, header_gather_.peer, header_gather_.first_word)
            .view({local_size, ProxyHeader::of(sizes_).gathered_words()});
    const PeerWords rows = remote_rows(bucketized.bucket_feature_of_row,
                                       bucketized.offset_per_bucket, node.destination_base, sizes_);
    at::Tensor words = peer_gather(kGatherRemoteMap, rows.peer, rows.first_word);
    const RequesterTables requester = requester_tables(
        words, headers, offset_per_owner_feature, bucketized.bucket_feature_of_row,
        bucketized.offset_per_bucket_feature, bucketized.offset_per_bucket, sizes_);
    words = at::Tensor();  // requester_tables is its one reader: free it before the owner tables
    const OwnerTables owner = owner_tables(rail, node.own_offsets, own_node_keys_,
                                           own_node_key_weights_, rail_keys_, other_nodes_, sizes_);
    const PutLists lists = put_lists(
        proxy.chunks, proxy.chunk_offset_per_bucket, proxy.chunk_inv, proxy.node_offset_per_bucket,
        proxy.node_length_per_bucket_feature, owner_offsets_, owner.fwd_chunks,
        owner.fwd_chunk_start, rail.node_length_per_source_feature, rail.source_offsets, sizes_);
    // The plans: the requester's maps and its tables of the node's proxies,
    // the owner's and the proxy's tables, the proxy's rows before each key
    // and the put lists. Then the owner's KJT.
    fwd.src = requester.src;
    bwd.dst = requester.dst;
    fwd.push_order = bwd.push_order = requester.push_order;
    fwd.proxy_chunks = bwd.proxy_chunks = requester.proxy_chunks;
    fwd.proxy_ends = bwd.proxy_ends = requester.proxy_ends;
    fwd.proxy_push_tables = bwd.proxy_push_tables = requester.proxy_push_tables;
    fwd.task_starts = bwd.task_starts = requester.task_starts;
    fwd.num_chunks = bwd.num_chunks = proxy.num_chunks;
    fwd.chunks = bwd.chunks = proxy.chunks;
    fwd.node_offset_per_bucket = bwd.node_offset_per_bucket = proxy.node_offset_per_bucket;
    fwd.put_list = bwd.put_list = lists.list;
    fwd.put_starts = bwd.put_starts = lists.starts;
    fwd.push_order_off = requester.push_order_off;
    fwd.fwd_chunks = owner.fwd_chunks;
    fwd.chunk_offset_per_bucket = proxy.chunk_offset_per_bucket;
    fwd.chunk_inv = proxy.chunk_inv;
    fwd.node_length_per_bucket = proxy.node_length_per_bucket;
    bwd.node_length_per_source = rail.node_length_per_source;
    bwd.reduce_task_starts = proxy.reduce_task_starts;
    bwd.node_exp_off = proxy.node_exp_off;
    dist.kjt_values = owner.kjt_values;
    dist.kjt_weights = owner.kjt_weights;
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
