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
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

#include "hier_output_dist/common/trace_flag.h"
#include "hier_output_dist/common/trace_slots.h"
#include "hier_output_dist/host/in_channel.h"
#include "hier_output_dist/host/sizes.h"
#include "hier_output_dist/kernels.h"
#include "hier_output_dist/plan/node.h"
#include "hier_output_dist/plan/put_list.h"
#include "hier_output_dist/plan/rail.h"

namespace dyn_emb {
namespace hier_output_dist {

InChannel::InChannel(const Config &config, const Sizes &sizes, const ChannelStatic &channel,
                     const SymmetricWindow &window, int device)
    : channel_(channel),
      sizes_(sizes),
      node_dedup_(config.node_dedup),
      // One stream: the channel's steps rely on their issue order.
      side_(at::cuda::getStreamFromPool(/*isHighPriority=*/true, device)) {
  const auto options = at::TensorOptions().device(at::kCUDA, device);
  auto view = [&](void *buffer, at::IntArrayRef shape, at::ScalarType dtype) {
    return at::from_blob(buffer, shape, options.dtype(dtype));
  };
  const KeysLayout keys = window.layout().keys;
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

void InChannel::start(Plan &plan, const std::shared_ptr<at::cuda::CUDAEvent> &out_done) {
  c10::cuda::CUDAGuard device_guard(side_.device_index());
  c10::cuda::CUDAStreamGuard stream_guard(side_);
  at::NoGradGuard no_grad;
  const int64_t seq = ++seq_;
  plan.planned_.block(side_);
  if (previous_out_done_ != nullptr) previous_out_done_->block(side_);
  previous_out_done_ = out_done;
  plan.record_stream(side_);
  // With kTrace, the call's step times: a timing event on the side stream at
  // its start and at the end of each step, which the plan holds for its wait.
  plan.in_seq_ = seq;
  if constexpr (kTrace) {
    for (int event = 0; event <= kInSteps; ++event)
      plan.step_events_.emplace_back(cudaEventDefault);
    plan.step_events_[0].record(side_);
  }
  auto step_done = [&plan, this](int step) {
    if constexpr (kTrace) plan.step_events_[1 + step].record(side_);
  };
  const Topo &topo = sizes_.topo;
  const int64_t local_size = topo.local_size, num_nodes = topo.num_nodes,
                num_features = sizes_.num_features, lookup_keys = sizes_.lookup_keys,
                remote_keys = sizes_.remote_keys;
  const auto long_options = plan.block_start_.options();
  // The node: every rank's counts (and the node barrier), the layout of the
  // node's keys, and this rank's keys pushed to their targets.
  const auto [totals, before, mine] = launch_node_count_exchange(
      channel_, seq, node_count_vector(plan.cell_start_, topo, num_features));
  step_done(kStepCounts);
  const NodeLayout node =
      node_layout(NodeCounts{totals, before, mine}, plan.cell_start_, plan.block_start_, topo,
                  num_features, lookup_keys, remote_keys);
  step_done(kStepLayout);
  launch_node_key_push(channel_, plan.keys_, plan.key_weights_, plan.row_cell_,
                       node.destination_base, node.rows_to_push);
  step_done(kStepPush);
  launch_node_done(channel_, seq);
  step_done(kStepDone);
  at::Tensor needs, num_recv, length_per_key, group_subs;
  if (num_nodes == 1) {
    // The own part of node_keys is the owner's KJT, and a row's place in its
    // owner's KJT is where it was pushed.
    const OneNodeOwner owner =
        one_node_owner(node, plan.row_cell_, own_keys_, own_weights_, topo, num_features);
    plan.src_ = plan.dst_ = owner.src;
    plan.kjt_values_ = owner.kjt_values;
    plan.kjt_weights_ = owner.kjt_weights;
    plan.kjt_lengths_ = owner.kjt_lengths;
    plan.kjt_offsets_ = owner.kjt_offsets;
    needs = at::cat({node.need_lookup.view({1}), at::zeros({2}, long_options)});
    num_recv = owner.num_recv;
    length_per_key = owner.length_per_key;
    group_subs = at::zeros({}, long_options);
    step_done(kStepKjt);
  } else {
    // The proxy: the node-unique keys of its remote part, their rows and
    // rounds, and its node's push orders (into remote_map, key_stage and
    // the proxy header).
    ProxyKeys keys = proxy_keys(ProxyInput{remote_keys_, remote_weights_, remote_cells_,
                                           node.requester_start, node.skip, tables_},
                                sizes_, node_dedup_);
    ProxyRounds rounds = proxy_rounds_and_slots(keys, node.requester_start, proxy_buffers_, sizes_);
    step_done(kStepDedup);
    const ProxyPlan proxy = proxy_push_order(std::move(keys), std::move(rounds),
                                             node.requester_start, proxy_buffers_, sizes_);
    step_done(kStepPushOrder);
    // The rail: the count messages, the layout of the rail, then the
    // proxies' key records into the owners' rail_keys and the owners' cell
    // starts into the proxies' rail_cells.
    const at::Tensor message =
        count_message(node.need_remote, proxy.num_rounds, node.own_total.select(0, topo.local_rank),
                      proxy.uniq_out, proxy.cell_uniq);
    step_done(kStepMessage);
    const at::Tensor messages = launch_rail_count_exchange(channel_, seq, message);
    step_done(kStepRailCounts);
    const RailLayout rail = rail_layout(messages, node.own_cells, other_nodes_, sizes_);
    cell_stage_.copy_(rail.cell_stage);
    step_done(kStepRailLayout);
    launch_rail_key_exchange(channel_, seq, proxy.stage_slice, rail.put_dst, rail.put_bytes,
                             rail.put_skip);
    step_done(kStepRailKeys);
    // The node again: the call's needs, where the node's owners hold this
    // requester's cells, and what the node's proxies published for it.
    const auto [peer_needs, owner_cell_start] =
        launch_node_status_exchange(channel_, seq, status_publish(rail));
    needs = agreed_needs(peer_needs);
    step_done(kStepStatusExchange);
    const at::Tensor headers = launch_peer_gather(channel_, kGatherProxyHeader, header_gather_.peer,
                                                  header_gather_.first_word)
                                   .view({local_size, ProxyHeader::of(sizes_).gathered_words()});
    const PeerWords rows =
        remote_rows(plan.row_cell_, plan.block_start_, node.destination_base, sizes_);
    at::Tensor words = launch_peer_gather(channel_, kGatherRemoteMap, rows.peer, rows.first_word);
    step_done(kStepGathers);
    const RequesterTables requester =
        requester_tables(words, headers, owner_cell_start, plan.row_cell_, plan.cell_start_,
                         plan.block_start_, sizes_);
    words = at::Tensor();  // requester_tables is its one reader: free it before the owner tables
    step_done(kStepRequester);
    const OwnerTables owner = owner_tables(rail, node.own_starts, own_keys_, own_weights_,
                                           rail_keys_, other_nodes_, sizes_);
    step_done(kStepOwner);
    const PutLists lists =
        put_lists(proxy.round_table, proxy.num_rounds, proxy.cell_uniq, rail_cells_,
                  owner.fwd_pieces, owner.fwd_rounds, rail.cells_in, rail.cell_stage, sizes_);
    step_done(kStepPutList);
    // What the out kernels read: the requester's maps and its tables of the
    // node's proxies, the owner's and the proxy's tables, the proxy's reducer
    // list, the put lists (the ready counters zeroed by the kernels), and the
    // owner's KJT.
    plan.src_ = requester.src;
    plan.dst_ = requester.dst;
    plan.push_order_ = requester.push_order;
    plan.heavy_groups_ = requester.heavy_groups;
    plan.group_first_sub_ = requester.group_first_sub;
    plan.group_of_sub_ = requester.group_of_sub;
    plan.proxy_rounds_ = requester.proxy_rounds;
    plan.proxy_round_tables_ = requester.proxy_round_tables;
    plan.push_tables_ = requester.push_tables;
    plan.push_order_off_ = requester.push_order_off;
    plan.fwd_task_starts_ = requester.fwd_task_starts;
    plan.bwd_task_starts_ = requester.bwd_task_starts;
    plan.uniq_in_ = rail.uniq_in;
    plan.fwd_put_start_ = lists.fwd_start;
    plan.num_rounds_ = proxy.num_rounds;
    plan.num_chunks_ = lists.num_chunks;
    plan.occurrence_count_ = proxy.occurrence_count;
    plan.uniq_out_ = proxy.uniq_out;
    plan.uniq_off_ = proxy.uniq_off;
    plan.round_table_ = lists.round_table;
    plan.bwd_put_start_ = lists.bwd_start;
    plan.push_start_ = proxy.push_start;
    plan.reduce_task_starts_ = lists.reduce_task_starts;
    plan.reducer_start_ = proxy.reducer_start;
    plan.reducer_entries_ = proxy.reducer_entries;
    plan.put_list_ = lists.list;
    plan.ready_ = at::empty(
        {std::max(chunk_bound(remote_keys, sizes_) + chunk_bound(lookup_keys, sizes_), local_size)},
        long_options);
    plan.kjt_values_ = owner.kjt_values;
    plan.kjt_weights_ = owner.kjt_weights;
    plan.kjt_lengths_ = rail.kjt_lengths;
    plan.kjt_offsets_ = rail.kjt_offsets;
    num_recv = rail.num_recv;
    length_per_key = rail.length_per_key;
    group_subs = requester.group_subs;
  }
  // The status, and its copy into the pinned buffer the host reads after
  // done (the copy records the host allocator's event: a plan dropped before
  // its wait keeps the buffer until the copy ran).
  const at::Tensor heavy_subs =
      plan.index_dedup_ ? plan.fanout_.count.narrow(0, 1, 1) : at::zeros({1}, long_options);
  plan.status_ = at::cat({needs, num_recv.view({1}), heavy_subs, group_subs.view({1}),
                          plan.block_start_.narrow(0, topo.world_size, 1), length_per_key,
                          plan.input_length_per_key_});
  plan.status_host_.copy_(plan.status_, /*non_blocking=*/true);
  step_done(kStepStatus);
  plan.after_start();
  plan.done_.record(side_);
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
