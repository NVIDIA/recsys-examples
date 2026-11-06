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
  fanout.slot = data_of<int32_t>(tables.slot);
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
  view.proxy_chunks = data_of<int64_t>(plan.proxy_chunks);
  view.proxy_ends = data_of<int64_t>(plan.proxy_ends);
  view.push_tables = data_of<ReqEntry, int32_t>(plan.push_tables);
  view.task_starts = data_of<int64_t>(plan.task_starts);
  view.num_chunks = data_of<int64_t>(plan.num_chunks);
  view.chunks = data_of<Chunk, int32_t>(plan.chunks);
  view.uniq_off = data_of<int64_t>(plan.uniq_off);
  view.put_list = data_of<PutSeg, int32_t>(plan.put_list);
  view.put_starts = data_of<int32_t>(plan.put_starts);
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
          &plan.fanout.slot,
          &plan.fanout.sub,
          &plan.fanout.count,
          &plan.fanout.pos,
          &plan.fanout.first,
          &plan.push_order,
          &plan.proxy_chunks,
          &plan.proxy_ends,
          &plan.push_tables,
          &plan.task_starts,
          &plan.num_chunks,
          &plan.chunks,
          &plan.uniq_off,
          &plan.put_list,
          &plan.put_starts,
          &plan.push_order_off,
          &plan.fwd_chunks,
          &plan.node_chunk_start,
          &plan.node_chunks,
          &plan.uniq_out};
}

// The backward plan's tensors, in the order of tensors() and of().
template <typename Plan>
auto bwd_tensors(Plan &plan) -> std::vector<decltype(&plan.dst)> {
  return {&plan.offset_per_bucket,
          &plan.dst,
          &plan.inv,
          &plan.fanout.exp_off,
          &plan.fanout.exp_idx,
          &plan.fanout.slot,
          &plan.fanout.sub,
          &plan.fanout.count,
          &plan.fanout.pos,
          &plan.fanout.first,
          &plan.push_order,
          &plan.proxy_chunks,
          &plan.proxy_ends,
          &plan.push_tables,
          &plan.task_starts,
          &plan.num_chunks,
          &plan.chunks,
          &plan.uniq_off,
          &plan.put_list,
          &plan.put_starts,
          &plan.uniq_in,
          &plan.reduce_task_starts,
          &plan.reducer_start,
          &plan.reducer_flags};
}

}  // namespace

FwdPlanView FwdPlan::view(const Sizes &sizes) const {
  FwdPlanView view{};
  static_cast<PlanView &>(view) = plan_view(*this, sizes);
  view.src = data_of<int32_t>(src);
  view.push_order_off = data_of<int64_t>(push_order_off);
  view.fwd_chunks = data_of<ChunkPiece, int32_t>(fwd_chunks);
  view.node_chunk_start = data_of<int64_t>(node_chunk_start);
  view.node_chunks = data_of<int32_t>(node_chunks);
  view.uniq_out = data_of<int64_t>(uniq_out);
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
  view.uniq_in = data_of<int64_t>(uniq_in);
  view.reduce_task_starts = data_of<int64_t>(reduce_task_starts);
  view.reducer_start = data_of<int64_t>(reducer_start);
  view.reducer_flags = data_of<int32_t>(reducer_flags);
  view.heavy_subs = heavy_subs;
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
