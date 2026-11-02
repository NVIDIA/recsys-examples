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
  view.block_start = data_of<int64_t>(plan.block_start);
  view.inv = data_of<int32_t>(plan.inv);
  view.fanout = fanout_view(plan.fanout);
  view.push_order = data_of<int32_t>(plan.push_order);
  view.proxy_rounds = data_of<int64_t>(plan.proxy_rounds);
  view.push_tables = data_of<ReqEntry, int32_t>(plan.push_tables);
  view.num_rounds = data_of<int64_t>(plan.num_rounds);
  view.put_list = data_of<PutSeg, int32_t>(plan.put_list);
  view.rounds_cap = sizes.rounds_cap;
  view.num_nodes = sizes.topo.num_nodes;
  return view;
}

void record(const at::Tensor &tensor, c10::Stream stream) {
  if (tensor.defined()) tensor.record_stream(stream);
}

void record(const FanoutTables &fanout, c10::Stream stream) {
  for (const at::Tensor *tensor : {&fanout.exp_off, &fanout.exp_idx, &fanout.slot, &fanout.sub,
                                   &fanout.count, &fanout.pos, &fanout.first})
    record(*tensor, stream);
}

// The backward plan's tensors, in the order of tensors() and of().
template <typename Plan>
auto bwd_tensors(Plan &plan) -> std::vector<decltype(&plan.dst)> {
  return {&plan.block_start,
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
          &plan.proxy_rounds,
          &plan.push_tables,
          &plan.num_rounds,
          &plan.put_list,
          &plan.heavy_groups,
          &plan.group_first_sub,
          &plan.group_of_sub,
          &plan.bwd_task_starts,
          &plan.uniq_in,
          &plan.num_chunks,
          &plan.uniq_off,
          &plan.round_table,
          &plan.bwd_put_start,
          &plan.reduce_task_starts,
          &plan.reducer_start,
          &plan.reducer_entries};
}

}  // namespace

FwdPlanView FwdPlan::view(const Sizes &sizes) const {
  FwdPlanView view{};
  static_cast<PlanView &>(view) = plan_view(*this, sizes);
  view.src = data_of<int32_t>(src);
  view.proxy_round_tables = data_of<RoundPiece, int32_t>(proxy_round_tables);
  view.push_order_off = data_of<int64_t>(push_order_off);
  view.fwd_task_starts = data_of<int64_t>(fwd_task_starts);
  view.fwd_put_start = data_of<int32_t>(fwd_put_start);
  view.uniq_out = data_of<int64_t>(uniq_out);
  view.push_start = data_of<int32_t>(push_start);
  return view;
}

void FwdPlan::record_stream(c10::Stream stream) const {
  for (const at::Tensor *tensor :
       {&block_start, &src, &inv, &push_order, &proxy_rounds, &push_tables, &num_rounds, &put_list,
        &proxy_round_tables, &push_order_off, &fwd_task_starts, &fwd_put_start, &uniq_out,
        &push_start})
    record(*tensor, stream);
  record(fanout, stream);
}

BwdPlanView BwdPlan::view(const Sizes &sizes) const {
  BwdPlanView view{};
  static_cast<PlanView &>(view) = plan_view(*this, sizes);
  view.dst = data_of<int32_t>(dst);
  view.heavy_groups = data_of<HeavyGroup, int32_t>(heavy_groups);
  view.group_first_sub = data_of<int32_t>(group_first_sub);
  view.group_of_sub = data_of<int32_t>(group_of_sub);
  view.bwd_task_starts = data_of<int64_t>(bwd_task_starts);
  view.uniq_in = data_of<int64_t>(uniq_in);
  view.num_chunks = data_of<int64_t>(num_chunks);
  view.uniq_off = data_of<int64_t>(uniq_off);
  view.round_table = data_of<RoundPiece, int32_t>(round_table);
  view.bwd_put_start = data_of<int32_t>(bwd_put_start);
  view.reduce_task_starts = data_of<int64_t>(reduce_task_starts);
  view.reducer_start = data_of<int32_t>(reducer_start);
  view.reducer_entries = data_of<int64_t>(reducer_entries);
  return view;
}

void BwdPlan::record_stream(c10::Stream stream) const {
  for (const at::Tensor *tensor : bwd_tensors(*this)) record(*tensor, stream);
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
  TORCH_CHECK(tensors.size() == fields.size() && counts.size() == 3,
              "hier_output_dist: a backward plan is made of ", fields.size(),
              " tensors and 3 counts");
  for (size_t field = 0; field < fields.size(); ++field)
    if (tensors[field].has_value()) *fields[field] = *tensors[field];
  plan.num_recv = counts[0];
  plan.heavy_subs = counts[1];
  plan.group_subs = counts[2];
  return plan;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
