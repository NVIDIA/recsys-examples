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
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/Stream.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "hier_output_dist/host/layout.h"
#include "hier_output_dist/output/args.cuh"
#include "hier_output_dist/plan/part_one.h"

namespace dyn_emb {
namespace hier_output_dist {

class Plan {
 public:
  // Part 1's tensors (on the caller's stream): its partition and, under
  // index dedup, its fan-out.
  Plan(const Layout &layout, PlanInput input, Partition part, FanoutTables fanout);
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

  // The second wait (HierOutputDist::wait): the call's one host sync. False:
  // the call does not fit the capacities (the same on every rank), and only
  // sent(), input_length_per_key(), needs() and overflows() are valid. True:
  // the plan holds the owner's KJT and the maps of its output dist, and the
  // current stream is ordered after the call.
  bool wait();
  // What the out kernels read of it: the forward's src or the backward's dst.
  OutPlan out_plan(bool backward) const;
  // Its steps are queued (start): what only they read goes.
  void after_start();
  // The members that hold the backward's maps, in their order.
  std::vector<at::Tensor *> backward_members();
  // Its forward is queued: the owner's KJT is the lookup's, and the plan
  // lets go of the backward's maps, which it returns for autograd to save.
  std::vector<at::Tensor> after_forward();
  // A backward holds the maps autograd saved until its kernel is queued.
  void hold_backward_maps(const std::vector<at::Tensor> &maps);
  void drop_backward_maps();
  int64_t heavy_subs() const { return heavy_subs_; }
  int64_t group_subs() const { return group_subs_; }
  int64_t heavy_slots() const { return index_dedup_ ? fanout_.pos.numel() : 0; }
  int64_t heavy_groups() const { return group_first_sub_.defined() ? group_first_sub_.numel() : 0; }

  int num_features_, num_sources_;
  int64_t lookup_keys_, remote_keys_;  // the op's capacities, which the needs are held to
  bool multi_node_, index_dedup_;
  int device_;
  PlanInput input_;  // until wait(): part 1 read it on the caller's stream
  // requester, from part 1 (Partition, FanoutTables)
  at::Tensor keys_, key_weights_, inv_, row_cell_, cell_start_, block_start_, input_length_per_key_;
  FanoutTables fanout_;
  int64_t num_out_;
  // the call's tables, from start
  at::Tensor meta_, src_, dst_;
  at::Tensor push_order_, heavy_groups_, group_first_sub_, group_of_sub_;  // across nodes
  at::Tensor reducer_start_, reducer_entries_, bwd_put_list_, ready_;      // across nodes
  at::Tensor kjt_values_, kjt_weights_, kjt_lengths_, kjt_offsets_;        // the owner's KJT
  at::Tensor status_;            // [status_words] the call's status on the device
  at::Tensor status_host_;       // pinned: start copies the status into it
  at::cuda::CUDAEvent planned_;  // after part 1, on the caller's stream
  at::cuda::CUDAEvent done_;     // after the status copy, on the side stream
  int64_t in_seq_ = 0;           // the in channel's sequence number of the call
  // With kTrace, the call's step times: a timing event on the side stream at
  // its start, then one at the end of each step (InTraceSteps).
  std::vector<at::cuda::CUDAEvent> step_events_;
  // after wait()
  std::vector<int64_t> status_values_;
  int64_t num_recv_ = 0, heavy_subs_ = 0, group_subs_ = 0, sent_ = 0;
  at::Tensor values_, value_weights_, lengths_, offsets_;
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
