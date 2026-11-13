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
// The output dist's plans of a call (plan.h).

#include <c10/util/Exception.h>

#include "hier_output_dist/input_dist/plan.h"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

// A tensor's data as the kernels read it: nullptr when it is undefined.
template <typename T, typename Data = T>
const T *data_of(const at::Tensor &tensor) {
  return tensor.defined() ? (const T *)tensor.data_ptr<Data>() : nullptr;
}

Fanout fanout_view(const FanoutTables &tables) {
  Fanout fanout{};
  fanout.exp_off = data_of<int32_t>(tables.exp_off);
  fanout.exp_idx = data_of<int32_t>(tables.exp_idx);
  fanout.heavy_slot = data_of<int32_t>(tables.heavy_slot);
  fanout.sub = data_of<int64_t>(tables.sub);
  fanout.count = data_of<int64_t>(tables.count);
  fanout.pos = data_of<int32_t>(tables.pos);
  fanout.first = data_of<int32_t>(tables.first);
  return fanout;
}

// What both views take, from either plan.
template <typename Plan>
PlanView plan_view(const Plan &plan, const Sizes &sizes) {
  PlanView view{};
  view.offset_per_bucket = data_of<int64_t>(plan.offset_per_bucket);
  view.inv = data_of<int32_t>(plan.inv);
  view.fanout = fanout_view(plan.fanout);
  view.push_order = data_of<int32_t>(plan.push_order);
  view.proxy_ends = data_of<int64_t>(plan.proxy_ends);
  view.proxy_push_tables = data_of<int32_t>(plan.proxy_push_tables);
  view.remote_task_offsets = data_of<int64_t>(plan.remote_task_offsets);
  view.num_chunks = data_of<int64_t>(plan.num_chunks);
  view.chunks = data_of<Chunk, int32_t>(plan.chunks);
  view.chunk_offset_per_bucket = data_of<int64_t>(plan.chunk_offset_per_bucket);
  view.chunk_inv = data_of<int32_t>(plan.chunk_inv);
  view.node_offset_per_bucket = data_of<int64_t>(plan.node_offset_per_bucket);
  view.max_chunks = sizes.max_chunks;
  view.ring = sizes.num_proxy_rows;
  view.num_nodes = sizes.topo.num_nodes;
  return view;
}

// The forward plan's tensors, in the order of tensors() and of().
template <typename Plan>
auto fwd_tensors(Plan &plan) -> std::vector<decltype(&plan.src)> {
  return {&plan.offset_per_bucket,
          &plan.src,
          &plan.inv,
          &plan.fanout.exp_off,
          &plan.fanout.exp_idx,
          &plan.fanout.heavy_slot,
          &plan.fanout.sub,
          &plan.fanout.count,
          &plan.fanout.pos,
          &plan.fanout.first,
          &plan.push_order,
          &plan.proxy_ends,
          &plan.proxy_push_tables,
          &plan.remote_task_offsets,
          &plan.num_chunks,
          &plan.chunks,
          &plan.chunk_offset_per_bucket,
          &plan.chunk_inv,
          &plan.node_offset_per_bucket,
          &plan.push_order_off,
          &plan.fwd_chunks,
          &plan.fwd_chunk_offsets,
          &plan.fwd_groups,
          &plan.node_length_per_bucket,
          &plan.proxy_num_chunks};
}

// The backward plan's tensors, in the order of tensors() and of().
template <typename Plan>
auto bwd_tensors(Plan &plan) -> std::vector<decltype(&plan.dst)> {
  return {&plan.offset_per_bucket,
          &plan.dst,
          &plan.inv,
          &plan.fanout.exp_off,
          &plan.fanout.exp_idx,
          &plan.fanout.heavy_slot,
          &plan.fanout.sub,
          &plan.fanout.count,
          &plan.fanout.pos,
          &plan.fanout.first,
          &plan.push_order,
          &plan.proxy_ends,
          &plan.proxy_push_tables,
          &plan.remote_task_offsets,
          &plan.num_chunks,
          &plan.chunks,
          &plan.chunk_offset_per_bucket,
          &plan.chunk_inv,
          &plan.node_offset_per_bucket,
          &plan.bwd_groups,
          &plan.node_length_per_source,
          &plan.reduce_task_offsets,
          &plan.node_exp_off};
}

}  // namespace

FwdPlanView FwdPlan::view(const Sizes &sizes) const {
  FwdPlanView view{};
  static_cast<PlanView &>(view) = plan_view(*this, sizes);
  view.src = data_of<int32_t>(src);
  view.push_order_off = data_of<int64_t>(push_order_off);
  view.fwd_chunks = data_of<OwnerChunk, int32_t>(fwd_chunks);
  view.fwd_chunk_offsets = data_of<int64_t>(fwd_chunk_offsets);
  view.fwd_groups = GroupRuns{data_of<int64_t>(fwd_groups), sizes.num_groups};
  view.max_owner_chunks = sizes.multi_node() ? sizes.max_owner_chunks() : 0;
  view.node_length_per_bucket = data_of<int64_t>(node_length_per_bucket);
  view.proxy_num_chunks = data_of<int64_t>(proxy_num_chunks);
  return view;
}

std::vector<at::Tensor> FwdPlan::tensors() const {
  std::vector<at::Tensor> tensors;
  for (const at::Tensor *tensor : fwd_tensors(*this)) tensors.push_back(*tensor);
  return tensors;
}

FwdPlan FwdPlan::of(const std::vector<c10::optional<at::Tensor>> &tensors) {
  FwdPlan plan;
  const std::vector<at::Tensor *> fields = fwd_tensors(plan);
  TORCH_CHECK(tensors.size() == fields.size(), "hier_output_dist: a forward plan is made of ",
              fields.size(), " tensors, got ", tensors.size());
  for (size_t field = 0; field < fields.size(); ++field)
    if (tensors[field].has_value()) *fields[field] = *tensors[field];
  return plan;
}

BwdPlanView BwdPlan::view(const Sizes &sizes) const {
  BwdPlanView view{};
  static_cast<PlanView &>(view) = plan_view(*this, sizes);
  view.dst = data_of<int32_t>(dst);
  view.bwd_groups = GroupRuns{data_of<int64_t>(bwd_groups), sizes.num_groups};
  view.node_length_per_source = data_of<int64_t>(node_length_per_source);
  view.reduce_task_offsets = data_of<int64_t>(reduce_task_offsets);
  view.node_exp_off = data_of<int64_t>(node_exp_off);
  return view;
}

std::vector<at::Tensor> BwdPlan::tensors() const {
  std::vector<at::Tensor> tensors;
  for (const at::Tensor *tensor : bwd_tensors(*this)) tensors.push_back(*tensor);
  return tensors;
}

BwdPlan BwdPlan::of(const std::vector<c10::optional<at::Tensor>> &tensors,
                    const std::vector<int64_t> &counts) {
  BwdPlan plan;
  const std::vector<at::Tensor *> fields = bwd_tensors(plan);
  TORCH_CHECK(tensors.size() == fields.size() && counts.size() == 2,
              "hier_output_dist: a backward plan is made of ", fields.size(),
              " tensors and 2 counts");
  for (size_t field = 0; field < fields.size(); ++field)
    if (tensors[field].has_value()) *fields[field] = *tensors[field];
  plan.num_recv = counts[0];
  plan.heavy_subs = counts[1];
  return plan;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
