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
#include "hier_output_dist/kernels.h"
#include "hier_output_dist/plan/node.h"
#include "hier_output_dist/plan/put_list.h"
#include "hier_output_dist/plan/rail.h"

namespace dyn_emb {
namespace hier_output_dist {

InChannel::InChannel(const Config &config, const Layout &layout, const ChannelStatic &channel,
                     const SymmetricWindows &windows, int device)
    : channel_(channel),
      sizes_(layout.plan_sizes()),
      node_dedup_(config.node_dedup),
      // One stream: the channel's steps rely on their issue order.
      side_(at::cuda::getStreamFromPool(/*isHighPriority=*/true, device)) {
  const auto options = at::TensorOptions().device(at::kCUDA, device);
  auto region = [&](Region region, at::IntArrayRef sizes, at::ScalarType dtype) {
    return at::from_blob(windows.region(region), sizes, options.dtype(dtype));
  };
  const int64_t lookup_keys = sizes_.lookup_keys, remote_keys = sizes_.remote_keys;
  const at::Tensor node_keys = region(kNodeKeys, {lookup_keys + remote_keys}, at::kLong);
  own_keys_ = node_keys.narrow(0, 0, lookup_keys);
  at::Tensor node_weights;
  if (config.weights) {
    node_weights = region(kNodeWeights, {lookup_keys + remote_keys}, at::kFloat);
    own_weights_ = node_weights.narrow(0, 0, lookup_keys);
  }
  if (!layout.multi_node()) return;
  const int64_t num_nodes = sizes_.topo.num_nodes, local_size = sizes_.topo.local_size,
                rounds_cap = sizes_.rounds_cap, record_bytes = sizes_.record_bytes;
  remote_keys_ = node_keys.narrow(0, lookup_keys, remote_keys);
  if (config.weights) remote_weights_ = node_weights.narrow(0, lookup_keys, remote_keys);
  remote_cells_ = region(kRemoteCells, {remote_keys}, at::kShort);
  const int64_t piece_words = (num_nodes - 1) * rounds_cap * 2;  // 16 bytes a round piece
  proxy_windows_ = ProxyWindows{
      region(kRemoteMap, {3, remote_keys}, at::kInt),
      region(kKeyStage, {piece_words + remote_keys * record_bytes / 8 + 2}, at::kLong),
      region(kProxyHdr, {4 + 4 * rounds_cap * num_nodes + 4 * local_size * (rounds_cap + 1)},
             at::kInt)};
  rail_keys_ = region(kRailKeys, {lookup_keys * record_bytes / 8 + piece_words}, at::kLong);
  tables_ = at::tensor(config.feature_tables, at::kInt).to(options);
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
                num_features = sizes_.num_features, rounds_cap = sizes_.rounds_cap,
                lookup_keys = sizes_.lookup_keys, remote_keys = sizes_.remote_keys;
  const auto long_options = plan.block_start_.options();
  plan.meta_ = at::zeros({channel_.meta_at.words}, long_options);
  // The node: every rank's counts (and the node barrier), the layout of the
  // node's keys, and this rank's keys pushed to their targets.
  const at::Tensor node_counts = launch_node_count_exchange(
      channel_, seq, node_count_vector(plan.cell_start_, topo, num_features));
  step_done(kStepCounts);
  const NodeLayout node = node_layout(node_counts, plan.cell_start_, plan.block_start_, topo,
                                      num_features, lookup_keys, remote_keys);
  step_done(kStepLayout);
  launch_node_key_push(channel_, plan.keys_, plan.key_weights_, plan.row_cell_,
                       node.destination_base, node.rows_to_push, lookup_keys);
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
    // proxy_hdr).
    ProxyKeys keys = proxy_keys(ProxyInput{remote_keys_, remote_weights_, remote_cells_,
                                           node.requester_start, node.skip, tables_},
                                sizes_, node_dedup_);
    ProxyRounds rounds = proxy_rounds_and_slots(keys, node.requester_start, proxy_windows_, sizes_);
    step_done(kStepDedup);
    const ProxyPlan proxy = proxy_push_order(std::move(keys), std::move(rounds),
                                             node.requester_start, proxy_windows_, sizes_);
    step_done(kStepPushOrder);
    // The rail: every rank's count message, the layout of the rail, and the
    // proxies' key records into the owners' rail_keys.
    const at::Tensor message =
        count_message(node.need_remote, node.own_cells, proxy.num_rounds, proxy.cell_uniq, sizes_);
    step_done(kStepMessage);
    const at::Tensor messages = launch_rail_count_exchange(channel_, seq, message);
    step_done(kStepRailCounts);
    const RailLayout rail = rail_layout(messages, node.own_cells, other_nodes_, sizes_);
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
    const at::Tensor headers = launch_peer_gather(channel_, kProxyHdr, header_gather_.peer,
                                                  header_gather_.first_word, 1, 0)
                                   .view({local_size, header_words(sizes_)});
    const PeerWords rows =
        remote_rows(plan.row_cell_, plan.block_start_, node.destination_base, sizes_);
    at::Tensor words =
        launch_peer_gather(channel_, kRemoteMap, rows.peer, rows.first_word, 3, remote_keys);
    step_done(kStepGathers);
    const RequesterTables requester =
        requester_tables(words, headers, owner_cell_start, plan.row_cell_, plan.cell_start_,
                         plan.block_start_, sizes_);
    words = at::Tensor();  // requester_tables is its one reader: free it before the owner tables
    step_done(kStepRequester);
    const OwnerTables owner = owner_tables(rail, node.own_starts, own_keys_, own_weights_,
                                           rail_keys_, other_nodes_, sizes_);
    step_done(kStepOwner);
    const PutList list =
        proxy_put_list(proxy.round_table, proxy.cell_uniq, rail.rail_cell_start, sizes_);
    step_done(kStepPutList);
    // What the out kernels read: the requester's maps, the proxy's lists
    // (the ready counters zeroed by the kernels), the owner's KJT, and Meta.
    plan.src_ = requester.src;
    plan.dst_ = requester.dst;
    plan.push_order_ = requester.push_order;
    plan.heavy_groups_ = requester.heavy_groups;
    plan.group_first_sub_ = requester.group_first_sub;
    plan.group_of_sub_ = requester.group_of_sub;
    plan.reducer_start_ = proxy.reducer_start;
    plan.reducer_entries_ = proxy.reducer_entries;
    plan.bwd_put_list_ = list.bwd_put_list;
    plan.ready_ = at::empty({std::max(chunk_bound(sizes_), local_size)}, long_options);
    plan.kjt_values_ = owner.kjt_values;
    plan.kjt_weights_ = owner.kjt_weights;
    plan.kjt_lengths_ = rail.kjt_lengths;
    plan.kjt_offsets_ = rail.kjt_offsets;
    const at::Tensor &meta = plan.meta_;
    const Meta &field = channel_.meta_at;
    meta_int32(meta, field.proxy_round_tables, local_size * rounds_cap * num_nodes * 4)
        .view({local_size, -1})
        .copy_(requester.proxy_round_tables);
    meta_int32(meta, field.push_tables, local_size * (rounds_cap + 1) * 4)
        .copy_(requester.push_tables.view({-1}));
    meta_words(meta, field.push_order_off, local_size + 1).copy_(requester.push_order_off);
    meta_words(meta, field.proxy_rounds, local_size).copy_(requester.proxy_rounds);
    meta_words(meta, field.fwd_task_starts, rounds_cap + 1).copy_(requester.fwd_task_starts);
    meta_words(meta, field.bwd_task_starts, 2 * rounds_cap + 1).copy_(requester.bwd_task_starts);
    meta_words(meta, field.slice_in, 2 * num_nodes).copy_(rail.slice_in.view({-1}));
    meta_words(meta, field.uniq_in, num_nodes).copy_(rail.uniq_in);
    meta_int32(meta, field.fwd_put_list, num_nodes * rounds_cap * num_features * 4)
        .copy_(owner.fwd_put_list.view({-1}));
    meta_int32(meta, field.uniq_off, num_nodes + 1).copy_(proxy.uniq_off);
    meta_words(meta, field.uniq_out, num_nodes).copy_(proxy.uniq_out);
    meta_int32(meta, field.round_table, rounds_cap * num_nodes * 4)
        .copy_(list.round_table.view({-1}));
    meta_int32(meta, field.bwd_put_start, rounds_cap * num_nodes + 1).copy_(list.bwd_put_start);
    meta_int32(meta, field.push_start, local_size * (rounds_cap + 1))
        .copy_(proxy.push_start.view({-1}));
    meta_words(meta, field.reduce_task_starts, rounds_cap + 1).copy_(list.reduce_task_starts);
    meta_words(meta, field.misc, kNumMisc)
        .copy_(at::stack({proxy.num_rounds, list.num_chunks, proxy.occurrence_count}));
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
