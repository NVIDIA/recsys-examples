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
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "hier_output_dist/common/status.h"
#include "hier_output_dist/host/plan.h"

namespace dyn_emb {
namespace hier_output_dist {

// The status's pinned buffer is allocated here, after part 1, which the
// side stream then waits for (planned_).
Plan::Plan(const Layout &layout, PlanInput input, Partition part, FanoutTables fanout)
    : num_features_(layout.num_features),
      num_sources_(layout.topo.num_sources),
      lookup_keys_(layout.lookup_keys),
      remote_keys_(layout.remote_keys),
      multi_node_(layout.multi_node()),
      index_dedup_(input.reverse_indices.has_value()),
      input_(std::move(input)),
      keys_(std::move(part.keys)),
      key_weights_(std::move(part.key_weights)),
      inv_(std::move(part.inv)),
      row_cell_(std::move(part.row_cell)),
      cell_start_(std::move(part.cell_start)),
      block_start_(std::move(part.block_start)),
      input_length_per_key_(std::move(part.input_length_per_key)),
      fanout_(std::move(fanout)) {
  device_ = keys_.get_device();
  num_out_ = index_dedup_ ? input_.reverse_indices->numel() : input_.values.numel();
  status_host_ = at::empty({status_words(num_features_)},
                           at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  planned_.record(at::cuda::getCurrentCUDAStream(device_));
}

// The needs are the same on every rank, so every rank takes the same branch.
bool Plan::wait() {
  c10::cuda::CUDAGuard guard(device_);
  done_.synchronize();  // the call's one host sync
  const int64_t *words = status_host_.data_ptr<int64_t>();
  status_values_.assign(words, words + status_host_.numel());
  const bool fits = overflows().empty();
  num_recv_ = status_values_[kStatusNumRecv];
  heavy_subs_ = status_values_[kStatusHeavySubs];
  group_subs_ = status_values_[kStatusGroupSubs];
  sent_ = status_values_[kStatusSent];
  input_ = PlanInput{};
  status_ = status_host_ = at::Tensor();
  if (!fits) {
    // The caller moves the call another way: nothing of it stays.
    inv_ = block_start_ = input_length_per_key_ = meta_ = src_ = dst_ = at::Tensor();
    push_order_ = heavy_groups_ = group_first_sub_ = group_of_sub_ = at::Tensor();
    reducer_start_ = reducer_entries_ = bwd_put_list_ = ready_ = at::Tensor();
    kjt_values_ = kjt_weights_ = kjt_lengths_ = kjt_offsets_ = at::Tensor();
    fanout_ = FanoutTables{};
    return false;
  }
  auto stream = at::cuda::getCurrentCUDAStream(device_);
  done_.block(stream);
  record_stream(stream);
  if (sent_ < src_.numel()) {
    // Index dedup bounded them by the input's keys: keep the rows sent.
    inv_ = inv_.narrow(0, 0, sent_).clone();
    const bool same = dst_.is_same(src_);
    src_ = src_.narrow(0, 0, sent_).clone();
    dst_ = same ? src_ : dst_.narrow(0, 0, sent_).clone();
    if (push_order_.defined()) push_order_ = push_order_.narrow(0, 0, sent_).clone();
  }
  values_ = kjt_values_.narrow(0, 0, num_recv_);
  if (kjt_weights_.defined()) value_weights_ = kjt_weights_.narrow(0, 0, num_recv_);
  lengths_ = kjt_lengths_;
  offsets_ = kjt_offsets_;
  kjt_values_ = kjt_weights_ = kjt_lengths_ = kjt_offsets_ = at::Tensor();
  return true;
}

c10::optional<at::Tensor> Plan::weights() const {
  if (!value_weights_.defined()) return c10::nullopt;
  return value_weights_;
}

std::vector<int64_t> Plan::length_per_key() const {
  auto first = status_values_.begin() + kStatusLengthPerKey;
  return std::vector<int64_t>(first, first + num_features_);
}

std::vector<int64_t> Plan::input_length_per_key() const {
  auto first = status_values_.begin() + kStatusLengthPerKey + num_features_;
  return std::vector<int64_t>(first, first + num_features_);
}

std::map<std::string, int64_t> Plan::needs() const {
  return {{"lookup_keys", status_values_[kStatusNeed + kNeedLookupKeys]},
          {"remote_keys", status_values_[kStatusNeed + kNeedRemoteKeys]},
          {"rounds", status_values_[kStatusNeed + kNeedRounds]}};
}

std::vector<std::string> Plan::overflows() const {
  std::vector<std::string> names;
  if (status_values_[kStatusNeed + kNeedLookupKeys] > lookup_keys_) names.push_back("lookup_keys");
  if (status_values_[kStatusNeed + kNeedRemoteKeys] > remote_keys_) names.push_back("remote_keys");
  return names;
}

std::map<std::string, at::Tensor> Plan::tensors() const {
  std::map<std::string, at::Tensor> named = {{"keys", keys_},
                                             {"key_weights", key_weights_},
                                             {"inv", inv_},
                                             {"row_cell", row_cell_},
                                             {"cell_start", cell_start_},
                                             {"block_start", block_start_},
                                             {"input_length_per_key", input_length_per_key_},
                                             {"exp_off", fanout_.exp_off},
                                             {"exp_idx", fanout_.exp_idx},
                                             {"fanout_slot", fanout_.slot},
                                             {"fanout_sub", fanout_.sub},
                                             {"fanout_count", fanout_.count},
                                             {"fanout_pos", fanout_.pos},
                                             {"fanout_first", fanout_.first},
                                             {"status", status_},
                                             {"meta", meta_},
                                             {"src", src_},
                                             {"dst", dst_},
                                             {"push_order", push_order_},
                                             {"heavy_groups", heavy_groups_},
                                             {"group_first_sub", group_first_sub_},
                                             {"group_of_sub", group_of_sub_},
                                             {"reducer_start", reducer_start_},
                                             {"reducer_entries", reducer_entries_},
                                             {"bwd_put_list", bwd_put_list_},
                                             {"ready", ready_},
                                             {"kjt_values", kjt_values_},
                                             {"kjt_weights", kjt_weights_},
                                             {"kjt_lengths", kjt_lengths_},
                                             {"kjt_offsets", kjt_offsets_}};
  for (auto it = named.begin(); it != named.end();)
    it = it->second.defined() ? std::next(it) : named.erase(it);
  return named;
}

void Plan::record_stream(c10::Stream stream) const {
  // part 1's
  for (const at::Tensor *tensor :
       {&keys_, &key_weights_, &inv_, &row_cell_, &cell_start_, &block_start_,
        &input_length_per_key_, &fanout_.exp_off, &fanout_.exp_idx, &fanout_.slot, &fanout_.sub,
        &fanout_.count, &fanout_.pos, &fanout_.first})
    if (tensor->defined()) tensor->record_stream(stream);
  // start's
  for (const at::Tensor *tensor :
       {&status_, &meta_, &src_, &dst_, &push_order_, &heavy_groups_, &group_first_sub_,
        &group_of_sub_, &reducer_start_, &reducer_entries_, &bwd_put_list_, &ready_, &kjt_values_,
        &kjt_weights_, &kjt_lengths_, &kjt_offsets_})
    if (tensor->defined()) tensor->record_stream(stream);
}

std::vector<at::Tensor *> Plan::backward_members() {
  std::vector<at::Tensor *> members = {&block_start_, &dst_, &inv_, &meta_};
  if (multi_node_)
    members.insert(members.end(), {&push_order_, &heavy_groups_, &group_first_sub_, &group_of_sub_,
                                   &reducer_start_, &reducer_entries_, &bwd_put_list_, &ready_});
  if (index_dedup_)
    members.insert(members.end(), {&fanout_.exp_off, &fanout_.exp_idx, &fanout_.slot, &fanout_.sub,
                                   &fanout_.count, &fanout_.pos, &fanout_.first});
  return members;
}

// The fan-out tables of index dedup, or none.
static Fanout fanout_of(const FanoutTables &tables) {
  Fanout fanout{};
  if (!tables.exp_off.defined()) return fanout;
  fanout.exp_off = tables.exp_off.data_ptr<int32_t>();
  fanout.exp_idx = tables.exp_idx.data_ptr<int32_t>();
  fanout.slot = tables.slot.data_ptr<int32_t>();
  fanout.sub = tables.sub.data_ptr<int64_t>();
  fanout.count = tables.count.data_ptr<int64_t>();
  fanout.pos = tables.pos.data_ptr<int32_t>();
  fanout.first = tables.first.data_ptr<int32_t>();
  return fanout;
}

OutPlan Plan::out_plan(bool backward) const {
  OutPlan plan{};
  plan.block_start = block_start_.data_ptr<int64_t>();
  plan.rows = (backward ? dst_ : src_).data_ptr<int32_t>();
  plan.inv = inv_.data_ptr<int32_t>();
  plan.meta = meta_.data_ptr<int64_t>();
  if (multi_node_) {
    plan.push_order = push_order_.data_ptr<int32_t>();
    plan.heavy_groups = (const HeavyGroup *)heavy_groups_.data_ptr<int32_t>();
    plan.group_first_sub = group_first_sub_.data_ptr<int32_t>();
    plan.group_of_sub = group_of_sub_.data_ptr<int32_t>();
    plan.reducer_start = reducer_start_.data_ptr<int32_t>();
    plan.reducer_entries = reducer_entries_.data_ptr<int64_t>();
    plan.bwd_put_list = (const PutSeg *)bwd_put_list_.data_ptr<int32_t>();
    plan.ready = ready_.data_ptr<int64_t>();
  }
  plan.fanout = fanout_of(fanout_);
  return plan;
}

// The push is the last reader of the keys and their weights, and the node
// layout of row_cell and cell_start. Their use on the side stream is
// recorded, so their blocks are not reused before those steps run.
void Plan::after_start() { keys_ = key_weights_ = row_cell_ = cell_start_ = at::Tensor(); }

std::vector<at::Tensor> Plan::after_forward() {
  src_ = values_ = value_weights_ = lengths_ = offsets_ = at::Tensor();
  std::vector<at::Tensor> maps;
  for (at::Tensor *member : backward_members()) maps.push_back(*member);
  drop_backward_maps();
  return maps;
}

void Plan::hold_backward_maps(const std::vector<at::Tensor> &maps) {
  std::vector<at::Tensor *> members = backward_members();
  for (size_t map = 0; map < maps.size(); ++map) *members[map] = maps[map];
}

void Plan::drop_backward_maps() {
  for (at::Tensor *member : backward_members()) *member = at::Tensor();
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
