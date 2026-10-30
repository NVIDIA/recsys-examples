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
// The hierarchical dist of a row-wise sequence sharding as one operator
// (HierOutputDist, one per sharding). After one host sync per call, every
// owner has a ready KJT of the keys it looks up, and every rank the per-row
// maps of the output dist. Then the output dist runs one kernel per
// direction, which only moves rows.
//
// Across nodes a requester's keys for an owner on another node go to its
// proxy, the GPU of its own node on the owner's rail. The proxy deduplicates
// them per (owner node, table, key) and puts each node-unique key to the
// owner once, over InfiniBand on that rail (NCCL GIN). The output dist moves
// the rows of those keys the same way: one row per node-unique key over
// InfiniBand, the node's own copies over NVLink.
//
// The op's windows are static: its constructor allocates them once, at the
// capacities of its configuration, and they never grow; a call's own tables
// come from the caching allocator. A call that does not fit the capacities
// (wait() returns false, on every rank alike) still runs its steps to their
// end with nothing in their messages, so the op stays valid for the next
// call; the caller moves that call some other way.
//
// A call is a Plan, which holds its state. The op runs each of its steps:
// plan() is part 1, local, on the caller's stream; start() is the first
// wait, the in channel's comm kernels and the planner's steps on the op's
// side stream; wait() is the second wait, the host sync; lookup_rows() gives
// the rows the lookup writes; then forward() and backward().
#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "hier_output_dist/common/channel_args.h"
#include "hier_output_dist/host/communicator.h"
#include "hier_output_dist/host/config.h"
#include "hier_output_dist/host/cta_budget.h"
#include "hier_output_dist/host/in_channel.h"
#include "hier_output_dist/host/layout.h"
#include "hier_output_dist/host/out_channel.h"
#include "hier_output_dist/host/plan.h"
#include "hier_output_dist/host/trace.h"
#include "hier_output_dist/host/windows.h"

namespace dyn_emb {
namespace hier_output_dist {

class HierOutputDist {
 public:
  // A new unique id for the op's communicator (on one rank; the caller
  // hands it to the others).
  static std::string unique_id();
  // Why the op does not apply to a sharding (its kernels' static limits), or
  // "" if it does.
  static std::string unsupported(int world_size, int local_size, int num_features,
                                 int64_t row_bytes);
  // Why the op refuses `config` (Config::validate), or "" if it takes it. It
  // builds nothing, so the ranks can compare their configurations and
  // answers before any of them builds the op.
  static std::string invalid(const Config &config);

  // Collective: every rank of the sharding with the same unique id and
  // configuration (but its rank), one that invalid() takes. Creates the
  // communicator and allocates every window and the lookup gradient's
  // event. Then the ranks agree that each can run (its LSA team, GIN, its
  // allocations) before they register the windows together. If one cannot,
  // every rank throws the same std::runtime_error, which names the ranks and
  // why, and holds nothing. A rank that fails inside the registration throws
  // alone, and the others wait there, as in NCCL's own initialization.
  HierOutputDist(const std::string &unique_id, Config config);
  // Local: waits for this rank's kernels. Then, unless the op was quiesced
  // after the last kernel of every rank, it waits at most min(timeout, 30 s)
  // for every node and rail peer to reach its own teardown. It frees the
  // windows only if they all did; else it keeps them until the process exits.
  ~HierOutputDist();
  HierOutputDist(const HierOutputDist &) = delete;
  HierOutputDist &operator=(const HierOutputDist &) = delete;

  // Part 1 of a call on the current stream: the keys partitioned by owner,
  // and index dedup's fan-out. Local: no sync, no collective, no sequence
  // number, so the plan may be dropped.
  std::shared_ptr<Plan> plan(PlanInput input);
  // The first wait: the rest of the call on the op's side stream, after part
  // 1 (the comm kernels and the planner's steps, InChannel); advances the in
  // channel's sequence number. Every rank starts the same calls in the same
  // order.
  void start(Plan &plan);
  // The second wait, the call's one host sync. False: the call does not fit
  // the capacities (the same on every rank), and the caller moves it some
  // other way. True: the plan holds the owner's KJT and the maps of its
  // output dist, and the current stream is ordered after the call. With
  // kTrace it also keeps the call's step times (in_trace()).
  bool wait(Plan &plan);
  // The rows the call's lookup writes its output into, and the output dist
  // reads it from: lookup_rows' first num_recv rows, [num_recv, dim] (see
  // OutChannel).
  at::Tensor lookup_rows(Plan &plan);
  // The output dist of a call (see OutChannel): forward() returns the rows
  // and the backward's maps, which backward() takes as autograd saved them.
  std::tuple<at::Tensor, std::vector<at::Tensor>> forward(Plan &plan, const at::Tensor &lookup_out);
  at::Tensor backward(Plan &plan, const at::Tensor &grad, const std::vector<at::Tensor> &maps,
                      bool divide);
  // Collective, after the op's last call, on every rank: syncs this rank's
  // device and meets the other ranks over the op's communicator. No peer
  // then reads this rank's windows or sends it a credit any more, so the
  // teardown frees them at once. The next call clears it.
  void quiesce();

  // Diagnostics.
  const Config &config() const { return config_; }
  // The capacities in place, in entries (proxy_bytes in bytes, as
  // configured): lookup_keys; across nodes also remote_keys and what follows
  // from it (num_proxy_rows, rounds_cap), which read 0 on one node.
  std::map<std::string, int64_t> capacities() const;
  // Bytes of each window (whole granules) and region.
  std::map<std::string, std::map<std::string, int64_t>> memory() const;
  bool gin_railed() const { return communicator_->gin_railed(); }
  // Views of lookup_rows alive, and the gradient views among them.
  int64_t live_views() const { return windows_->live_views(); }
  int64_t lookup_grad_live() const { return lookup_grad_->live.load(); }
  // In a build with DYNAMICEMB_HIER_TRACE=1, the out kernels' timelines and
  // the in channel's step times (Trace; empty without it), and the calls of
  // the in channel so far.
  at::Tensor trace() const { return trace_.out_ring(); }
  std::vector<int64_t> trace_launches() const { return trace_.out_launches(); }
  at::Tensor in_trace() const { return trace_.in_ring(); }
  int64_t in_calls() const { return in_->calls(); }

 private:
  void release() noexcept;
  bool exit_round() noexcept;

  Config config_;
  Layout layout_;
  std::unique_ptr<Communicator> communicator_;
  std::unique_ptr<SymmetricWindows> windows_;
  // The part of the kernels' arguments both channels share, once the
  // windows are registered.
  ChannelStatic channel_static_{};
  OwnerRules rules_;  // part 1's
  std::shared_ptr<LookupGrad> lookup_grad_;
  CtaBudget budget_;
  Trace trace_;
  std::unique_ptr<InChannel> in_;
  std::unique_ptr<OutChannel> out_;
  int device_;
  bool quiesced_ = false;
};

// A test hook (testing.cu): the next construction of an op in this process
// fails to allocate (`fail_alloc`). The constructor takes it and clears it.
void fail_next_construction(bool fail_alloc);

}  // namespace hier_output_dist
}  // namespace dyn_emb
