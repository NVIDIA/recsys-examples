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
// The input dist of a call: the in channel's three steps (input_dist.h).

#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

#include "hier_output_dist/input_dist/input_dist.h"
#include "hier_output_dist/input_dist/planner.cuh"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

// Part 1 runs the rules present: their codes on the host.
OwnerRules owner_rules(const Config &config) {
  const at::Tensor codes = config.dist_types.to(at::kCPU, at::kInt).contiguous();
  return OwnerRules{config.dist_types, config.block_sizes,
                    std::vector<int>(codes.data_ptr<int>(), codes.data_ptr<int>() + codes.numel())};
}

// One CTA: the call's status (StatusSlots), which wait() copies to the host.
// Its needs: across nodes the most of the node's (peer_needs, each rank's of
// its rail, kNumNeeds each): a rank's rail covers every node, and its node
// holds a rank of every rail, so the most of the node is the most of all
// ranks. On one node its lookup need alone. heavy_subs and group_subs:
// nullptr when there are none.
__global__ void __launch_bounds__(kThreads)
    call_status_kernel(const int64_t *peer_needs, int64_t local_size, const int64_t *need_lookup,
                       const int64_t *num_recv, const int64_t *heavy_subs,
                       const int64_t *group_subs, const int64_t *sent,
                       const int64_t *length_per_key, const int64_t *input_length_per_key,
                       int64_t num_features, int64_t *status) {
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
    status[kStatusHeavySubs] = heavy_subs != nullptr ? *heavy_subs : 0;
    status[kStatusGroupSubs] = group_subs != nullptr ? *group_subs : 0;
    status[kStatusSent] = *sent;
  }
  for (int64_t feature = tid; feature < num_features; feature += kThreads) {
    status[kStatusLengthPerKey + feature] = length_per_key[feature];
    status[kStatusLengthPerKey + num_features + feature] = input_length_per_key[feature];
  }
}

}  // namespace

void InDistResult::record_stream(c10::Stream stream) const {
  for (const at::Tensor *tensor : {&values, &weights, &lengths, &offsets, &occurrence_count})
    if (tensor->defined()) tensor->record_stream(stream);
  fwd_plan.record_stream(stream);
  bwd_plan.record_stream(stream);
}

InChannel::InChannel(const Config &config, const Sizes &sizes, ncclWindow_t window,
                     const ncclDevComm &dev, const WindowLayout &layout, int device)
    : window_(window),
      dev_(dev),
      sizes_(sizes),
      rules_(owner_rules(config)),
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
  own_keys_ = node_keys.narrow(0, 0, keys.remote_start());
  at::Tensor node_weights;
  if (config.weights) {
    node_weights = view(keys.node_weights(), {lookup_keys + remote_keys}, at::kFloat);
    own_weights_ = node_weights.narrow(0, 0, keys.remote_start());
  }
  if (!sizes_.multi_node()) return;
  const int64_t num_nodes = sizes_.topo.num_nodes, num_features = sizes_.num_features;
  remote_keys_ = node_keys.narrow(0, keys.remote_start(), remote_keys);
  if (config.weights) remote_weights_ = node_weights.narrow(0, keys.remote_start(), remote_keys);
  remote_cells_ = view(keys.remote_cells(), {remote_keys}, at::kInt);
  proxy_buffers_ =
      proxy_buffers(view(keys.remote_map().base, {RemoteMap::kArrays, remote_keys}, at::kInt),
                    view(keys.key_stage(), {(int64_t)keys.key_stage_bytes() / 8}, at::kLong),
                    keys.proxy_header(), options);
  rail_keys_ = view(keys.rail_keys(), {(int64_t)keys.rail_keys_bytes() / 8}, at::kLong);
  cell_stage_ = view(keys.cell_stage(0), {num_nodes, num_features}, at::kLong);
  rail_cells_ = view(keys.rail_cells(0), {num_nodes, num_features}, at::kLong);
  tables_ = at::tensor(dense_tables(config.feature_tables), at::kInt).to(options);
  other_nodes_ = other_nodes_of(sizes_.topo, options);
  header_gather_ = header_gather(sizes_, options);
}

PartOne InChannel::part_one(PlanInput input) const {
  c10::cuda::CUDAGuard guard(device_);
  at::NoGradGuard no_grad;
  TORCH_CHECK(input.weights.has_value() == weights_,
              "hier_output_dist: the keys carry weights exactly when the op was made with "
              "weights (",
              weights_ ? "with" : "without", ")");
  PartOne one;
  one.part = partition_keys(input, rules_, sizes_.topo.world_size);
  if (input.reverse_indices.has_value())
    one.fanout = fanout_tables(*input.reverse_indices, one.part.inv);
  one.input = std::move(input);
  return one;
}

Pending InChannel::start(const PartOne &one) {
  c10::cuda::CUDAGuard device_guard(device_);
  at::NoGradGuard no_grad;
  const int64_t seq = ++seq_;
  const Partition &part = one.part;
  const Topo &topo = sizes_.topo;
  const int64_t local_size = topo.local_size, num_nodes = topo.num_nodes,
                num_features = sizes_.num_features, lookup_keys = sizes_.lookup_keys,
                remote_keys = sizes_.remote_keys;
  const auto long_options = part.block_start.options();
  Pending pending;
  pending.input = one.input;
  FwdPlan &fwd = pending.fwd_plan;
  BwdPlan &bwd = pending.bwd_plan;
  const bool index_dedup = one.input.reverse_indices.has_value();
  fwd.num_out = index_dedup ? one.input.reverse_indices->numel() : one.input.values.numel();
  fwd.block_start = bwd.block_start = part.block_start;
  fwd.inv = bwd.inv = part.inv;
  fwd.fanout = bwd.fanout = one.fanout;
  // The node: every rank's counts (and the node barrier), the layout of the
  // node's keys, and this rank's keys pushed to their targets.
  const auto [totals, before, mine] = launch_node_count_exchange(
      window_, sizes_, seq, node_count_vector(part.cell_start, topo, num_features));
  const NodeLayout node =
      node_layout(NodeCounts{totals, before, mine}, part.cell_start, part.block_start, topo,
                  num_features, lookup_keys, remote_keys);
  launch_node_key_push(window_, sizes_, part.keys, part.key_weights, part.row_cell,
                       node.destination_base, node.rows_to_push);
  launch_node_done(window_, sizes_, seq);
  at::Tensor peer_needs, num_recv, length_per_key, group_subs;
  if (num_nodes == 1) {
    // The own part of node_keys is the owner's KJT, and a row's place in its
    // owner's KJT is where it was pushed.
    const OneNodeOwner owner =
        one_node_owner(node, part.row_cell, own_keys_, own_weights_, topo, num_features);
    fwd.src = bwd.dst = owner.src;
    pending.kjt_values = owner.kjt_values;
    pending.kjt_weights = owner.kjt_weights;
    pending.kjt_lengths = owner.kjt_lengths;
    pending.kjt_offsets = owner.kjt_offsets;
    num_recv = owner.num_recv;
    length_per_key = owner.length_per_key;
  } else {
    // The proxy: the node-unique keys of its remote part, their rows and
    // rounds, and its node's push orders (into remote_map, key_stage and
    // the proxy header).
    ProxyKeys keys = proxy_keys(ProxyInput{remote_keys_, remote_weights_, remote_cells_,
                                           node.requester_start, node.skip, tables_},
                                sizes_, node_dedup_);
    ProxyRounds rounds = proxy_rounds_and_slots(keys, node.requester_start, proxy_buffers_, sizes_);
    const ProxyPlan proxy = proxy_push_order(std::move(keys), std::move(rounds),
                                             node.requester_start, proxy_buffers_, sizes_);
    // The rail: the count messages, the layout of the rail, then the
    // proxies' key records into the owners' rail_keys and the owners' cell
    // starts into the proxies' rail_cells.
    const at::Tensor message =
        count_message(node.need_remote, proxy.num_rounds, node.own_total.select(0, topo.local_rank),
                      proxy.uniq_out, proxy.cell_uniq);
    const at::Tensor messages = launch_rail_count_exchange(window_, dev_, sizes_, seq, message);
    const RailLayout rail = rail_layout(messages, node.own_cells, other_nodes_, sizes_);
    cell_stage_.copy_(rail.cell_stage);
    launch_rail_key_exchange(window_, dev_, sizes_, seq, proxy.stage_slice, rail.put_dst,
                             rail.put_bytes, rail.put_skip);
    // The node again: the call's needs, where the node's owners hold this
    // requester's cells, and what the node's proxies published for it.
    at::Tensor owner_cell_start;
    std::tie(peer_needs, owner_cell_start) =
        launch_node_status_exchange(window_, sizes_, seq, status_publish(rail));
    const at::Tensor headers = launch_peer_gather(window_, sizes_, kGatherProxyHeader,
                                                  header_gather_.peer, header_gather_.first_word)
                                   .view({local_size, ProxyHeader::of(sizes_).gathered_words()});
    const PeerWords rows =
        remote_rows(part.row_cell, part.block_start, node.destination_base, sizes_);
    at::Tensor words =
        launch_peer_gather(window_, sizes_, kGatherRemoteMap, rows.peer, rows.first_word);
    const RequesterTables requester = requester_tables(
        words, headers, owner_cell_start, part.row_cell, part.cell_start, part.block_start, sizes_);
    words = at::Tensor();  // requester_tables is its one reader: free it before the owner tables
    const OwnerTables owner = owner_tables(rail, node.own_starts, own_keys_, own_weights_,
                                           rail_keys_, other_nodes_, sizes_);
    const PutLists lists =
        put_lists(proxy.round_table, proxy.num_rounds, proxy.cell_uniq, rail_cells_,
                  owner.fwd_pieces, owner.fwd_rounds, rail.cells_in, rail.cell_stage, sizes_);
    // The plans: the requester's maps and its tables of the node's proxies,
    // the owner's and the proxy's tables, the proxy's reducer list and the
    // put lists. Then the owner's KJT.
    fwd.src = requester.src;
    bwd.dst = requester.dst;
    fwd.push_order = bwd.push_order = requester.push_order;
    fwd.proxy_rounds = bwd.proxy_rounds = requester.proxy_rounds;
    fwd.push_tables = bwd.push_tables = requester.push_tables;
    fwd.num_rounds = bwd.num_rounds = proxy.num_rounds;
    fwd.put_list = bwd.put_list = lists.list;
    fwd.proxy_round_tables = requester.proxy_round_tables;
    fwd.push_order_off = requester.push_order_off;
    fwd.fwd_task_starts = requester.fwd_task_starts;
    fwd.fwd_put_start = lists.fwd_start;
    fwd.uniq_out = proxy.uniq_out;
    fwd.push_start = proxy.push_start;
    bwd.heavy_groups = requester.heavy_groups;
    bwd.group_first_sub = requester.group_first_sub;
    bwd.group_of_sub = requester.group_of_sub;
    bwd.bwd_task_starts = requester.bwd_task_starts;
    bwd.uniq_in = rail.uniq_in;
    bwd.num_chunks = lists.num_chunks;
    bwd.uniq_off = proxy.uniq_off;
    bwd.round_table = lists.round_table;
    bwd.bwd_put_start = lists.bwd_start;
    bwd.reduce_task_starts = lists.reduce_task_starts;
    bwd.reducer_start = proxy.reducer_start;
    bwd.reducer_entries = proxy.reducer_entries;
    pending.kjt_values = owner.kjt_values;
    pending.kjt_weights = owner.kjt_weights;
    pending.kjt_lengths = rail.kjt_lengths;
    pending.kjt_offsets = rail.kjt_offsets;
    pending.occurrence_count = proxy.occurrence_count;
    num_recv = rail.num_recv;
    length_per_key = rail.length_per_key;
    group_subs = requester.group_subs;
  }
  // The status, which wait() reads.
  pending.status = at::empty({status_words(num_features)}, long_options);
  call_status_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      data_or_null<int64_t>(peer_needs), local_size, node.need_lookup.data_ptr<int64_t>(),
      num_recv.data_ptr<int64_t>(),
      index_dedup ? one.fanout.count.data_ptr<int64_t>() + 1 : nullptr,
      data_or_null<int64_t>(group_subs), part.block_start.data_ptr<int64_t>() + topo.world_size,
      length_per_key.data_ptr<int64_t>(), part.input_length_per_key.data_ptr<int64_t>(),
      num_features, pending.status.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return pending;
}

// The needs are the same on every rank, so every rank takes the same branch.
InDistResult InChannel::wait(const Pending &pending) const {
  c10::cuda::CUDAGuard guard(device_);
  const at::Tensor status = pending.status.cpu();  // the call's one host sync
  const int64_t *word = status.data_ptr<int64_t>();
  const int64_t num_features = sizes_.num_features;
  InDistResult result;
  const int64_t need_lookup = word[kStatusNeed + kNeedLookupKeys],
                need_remote = word[kStatusNeed + kNeedRemoteKeys];
  result.needs = {{"lookup_keys", need_lookup},
                  {"remote_keys", need_remote},
                  {"rounds", word[kStatusNeed + kNeedRounds]}};
  if (need_lookup > sizes_.lookup_keys) result.overflows.push_back("lookup_keys");
  if (need_remote > sizes_.remote_keys) result.overflows.push_back("remote_keys");
  result.fits = result.overflows.empty();
  result.sent = word[kStatusSent];
  const int64_t *length_per_key = word + kStatusLengthPerKey;
  result.input_length_per_key.assign(length_per_key + num_features,
                                     length_per_key + 2 * num_features);
  if (!result.fits) return result;
  result.num_recv = word[kStatusNumRecv];
  result.num_sources = sizes_.topo.num_sources;
  result.length_per_key.assign(length_per_key, length_per_key + num_features);
  result.values = pending.kjt_values.narrow(0, 0, result.num_recv);
  if (pending.kjt_weights.defined())
    result.weights = pending.kjt_weights.narrow(0, 0, result.num_recv);
  result.lengths = pending.kjt_lengths;
  result.offsets = pending.kjt_offsets;
  result.occurrence_count = pending.occurrence_count;
  result.fwd_plan = pending.fwd_plan;
  result.bwd_plan = pending.bwd_plan;
  result.bwd_plan.num_recv = result.num_recv;
  result.bwd_plan.heavy_subs = word[kStatusHeavySubs];
  result.bwd_plan.group_subs = word[kStatusGroupSubs];
  return result;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
