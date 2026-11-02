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
// One call of the op (Plan): what part 1 made of its keys, every tensor of
// the call the steps after it read and write, and its status. The op runs
// every step of the call (HierOutputDist::plan, start, wait, lookup_rows,
// forward, backward); the plan holds the call's state between them.
#pragma once

#include <ATen/ATen.h>
#include <c10/core/Stream.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "hier_output_dist/args.cuh"
#include "hier_output_dist/common.h"
#include "hier_output_dist/planner.h"

namespace dyn_emb {
namespace hier_output_dist {

class Plan {
 public:
  // Part 1's tensors (on the caller's stream): its partition and, under
  // index dedup, its fan-out.
  Plan(const Sizes &sizes, PlanInput input, Partition part, FanoutTables fanout);
  Plan(const Plan &) = delete;
  Plan &operator=(const Plan &) = delete;

  // After the op's wait() returned true, until the output dist's forward:
  // the owner's KJT, its values and weights (the keys', or across nodes the
  // occurrence counts, if any).
  at::Tensor values() const { return values_; }
  c10::optional<at::Tensor> weights() const;
  at::Tensor lengths() const { return lengths_; }  // [num_features * num_sources]
  at::Tensor offsets() const { return offsets_; }  // [num_features * num_sources + 1]
  int64_t num_sources() const { return num_sources_; }
  std::vector<int64_t> length_per_key() const;
  int64_t num_recv() const { return num_recv_; }  // keys this owner looks up
  // After wait(): the rows of the input KJT this rank sends (its valid keys),
  // and the per-feature totals of the KJT construct_jagged_tensors reads.
  int64_t sent() const { return sent_; }
  std::vector<int64_t> input_length_per_key() const;
  int64_t num_out() const { return num_out_; }  // rows of this rank's output
  // After wait(): what the call needs of each capacity (the most of any
  // rank), and the capacities it overflowed.
  std::map<std::string, int64_t> needs() const;
  std::vector<std::string> overflows() const;
  // The device tensors' use on `stream` (the caching allocator's).
  void record_stream(c10::Stream stream) const;
  // A test and diagnostic view (testing.cu): the call's device tensors by name.
  std::map<std::string, at::Tensor> tensors() const;

 private:
  friend class HierOutputDist;
  friend class InChannel;
  friend class OutChannel;

  // The second wait (HierOutputDist::wait), on the stream of the start or
  // one ordered after it: the call's one host sync, a copy of its status.
  // False: the call does not fit the capacities (the same on every rank),
  // and only sent(), input_length_per_key(), needs() and overflows() are
  // valid. True: the plan holds the owner's KJT and the maps of its output
  // dist.
  bool wait();
  // What the out kernels read of it: the forward's src or the backward's
  // dst, and the tables the plan holds (the backward's own, once the forward
  // let go of its own).
  OutPlan out_plan(bool backward) const;
  // Its steps are queued (start): what only they read goes.
  void after_start();
  // The members that hold the backward's maps, in their order.
  std::vector<at::Tensor *> backward_members();
  // Its forward is queued: the owner's KJT is the lookup's, the tables only
  // the forward reads go, and the plan lets go of the backward's maps, which
  // it returns for autograd to save.
  std::vector<at::Tensor> after_forward();
  // A backward holds the maps autograd saved until its kernel is queued.
  void hold_backward_maps(const std::vector<at::Tensor> &maps);
  void drop_backward_maps();
  int64_t heavy_subs() const { return heavy_subs_; }
  int64_t group_subs() const { return group_subs_; }
  int64_t heavy_slots() const { return index_dedup_ ? fanout_.pos.numel() : 0; }
  int64_t heavy_groups() const { return group_first_sub_.defined() ? group_first_sub_.numel() : 0; }

  int num_features_, num_sources_, num_nodes_;
  int64_t lookup_keys_, remote_keys_;  // the op's capacities, which the needs are held to
  int64_t rounds_cap_;                 // with num_nodes_, the tables' strides (OutPlan)
  bool multi_node_, index_dedup_;
  int device_;
  PlanInput input_;  // until wait(): part 1 read it on the caller's stream
  // requester, from part 1 (Partition, FanoutTables)
  at::Tensor keys_, key_weights_, inv_, row_cell_, cell_start_, block_start_, input_length_per_key_;
  FanoutTables fanout_;
  int64_t num_out_;
  // the call's tables, from start (OutPlan says what each holds)
  at::Tensor src_, dst_;
  at::Tensor push_order_, heavy_groups_, group_first_sub_, group_of_sub_;  // across nodes
  at::Tensor reducer_start_, reducer_entries_, put_list_, ready_;          // across nodes
  // across nodes, the requester's tables of the node's proxies
  at::Tensor proxy_rounds_, proxy_round_tables_, push_tables_, push_order_off_;
  at::Tensor fwd_task_starts_, bwd_task_starts_;
  at::Tensor uniq_in_, fwd_put_start_;  // across nodes, the owner's
  // across nodes, the proxy's
  at::Tensor num_rounds_, num_chunks_, uniq_out_, uniq_off_, round_table_, bwd_put_start_;
  at::Tensor push_start_, reduce_task_starts_;
  at::Tensor occurrence_count_;  // the proxy's remote occurrences (the benchmark's)
  at::Tensor kjt_values_, kjt_weights_, kjt_lengths_, kjt_offsets_;  // the owner's KJT
  at::Tensor status_;  // [status_words] the call's status on the device
  // after wait()
  std::vector<int64_t> status_values_;
  int64_t num_recv_ = 0, heavy_subs_ = 0, group_subs_ = 0, sent_ = 0;
  at::Tensor values_, value_weights_, lengths_, offsets_;
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
