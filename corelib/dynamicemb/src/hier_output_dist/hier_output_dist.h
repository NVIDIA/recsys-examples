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
// The hierarchical dist's operator (HierOutputDist) and what it owns: its
// communicator, its window, and its two channels, the input dist's
// (input_dist/) and the output dist's (output_dist/).
#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/config.h"
#include "hier_output_dist/common/device.cuh"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/input_dist/input_dist.h"
#include "hier_output_dist/input_dist/plan.h"
#include "hier_output_dist/output_dist/output_dist.h"

namespace dyn_emb {
namespace hier_output_dist {

// The op's own NCCL communicator: its device communicator (LSA inside the
// node, and across nodes GIN on the rails), what the device can do, and a
// collective the constructor agrees with.

class Communicator {
 public:
  // Collective: every rank of `topo` with the same unique id (unique_id()).
  // A rank that fails here raises; the others then wait in NCCL's own
  // initialization, as with any communicator.
  Communicator(const std::string &unique_id, const Topo &topo);
  Communicator(const Communicator &) = delete;
  Communicator &operator=(const Communicator &) = delete;
  // The owner ends the communicator (abort() or finish()), or keeps it until the process
  // exits when a peer may still need it: the destructor does neither.
  ~Communicator() = default;

  // A new unique id (on one rank; the caller hands it to the others).
  static std::string unique_id();

  ncclComm_t comm() const { return comm_; }
  const ncclDevComm &dev() const { return dev_; }
  // This rank's parts of whether the op can run: the LSA team is its node,
  // and across nodes GIN connects the rails with at least 4 contexts, whose
  // signals and counters start at 0.
  bool lsa_ok() const { return dev_.lsaSize == topo_.local_size; }
  bool gin_ok() const { return topo_.num_nodes == 1 || gin_ok_; }
  // Across nodes, whether NCCL connected GIN on the rails only, as asked (it
  // does so only when NCCL_CROSS_NIC is 0; else GIN's memory grows with the
  // ranks).
  bool gin_railed() const { return dev_.ginConnectionsRailed; }
  // What ncclMemAlloc rounds an allocation up to.
  size_t granularity() const { return granularity_; }
  // Every rank's `value`, in rank order (collective, synchronous).
  std::vector<int> all_gather(int value);
  // Returns once every rank has called it (collective, synchronous).
  void barrier() { all_gather(0); }
  // Ends the communicator on this rank alone (no collective).
  void abort();
  // Ends the communicator on every rank together (collective): each rank's
  // last communications reach their peers before any rank frees its
  // resources, which abort() does not wait for.
  void finish() noexcept;

 private:
  Topo topo_;
  ncclComm_t comm_ = nullptr;
  ncclDevComm dev_{};
  bool gin_ok_ = false;
  int device_ = 0;
  size_t granularity_ = 0;
};

// The op's window (window.h): this rank's copy of its symmetric
// memory, allocated once at the op's sizes, then registered collectively;
// its layout over that copy, which the host views and reads its buffers by;
// and views of lookup_rows, which the lookup writes into.

class SymmetricWindow {
 public:
  // Allocates this rank's copy and zeroes what must read 0 before the window
  // is registered: the ctrl section (sequence numbers start at 0) and
  // push_flags (tags start at 0). Every other buffer is written before it is
  // read, so it is never initialized. Throws if it cannot, with nothing left
  // allocated.
  SymmetricWindow(const Sizes &sizes, int device);
  SymmetricWindow(const SymmetricWindow &) = delete;
  SymmetricWindow &operator=(const SymmetricWindow &) = delete;

  // Collective: once every rank allocated its copy.
  void register_window(ncclComm_t comm);
  // The registered window, which the kernels take (nullptr before).
  ncclWindow_t handle() const { return handle_; }
  // The window laid out over this rank's copy.
  WindowLayout layout() const { return WindowLayout(sizes_, memory_.get()); }
  // The first `rows` rows of lookup_rows as a [rows, cols] tensor of `dtype`
  // (a row of lookup_rows). The view holds this rank's copy: its memory goes
  // once the window and every view of it are gone.
  at::Tensor lookup_rows(int64_t rows, int64_t cols, at::ScalarType dtype) const;
  // This rank's copy, in whole granules of `granularity`.
  int64_t bytes(size_t granularity) const;

  // Teardown steps, local: deregister the window, then (once the
  // communicator is gone) let go of this rank's copy, or keep it until the
  // process exits (`keep`: a peer may still read it).
  void deregister(ncclComm_t comm) noexcept;
  void release(bool keep) noexcept;

 private:
  int device_;
  Sizes sizes_;
  uint64_t bytes_;                 // as allocated (WindowLayout::bytes)
  std::shared_ptr<char> memory_;   // this rank's copy, freed by its last holder
  ncclWindow_t handle_ = nullptr;  // nullptr until registered
};

// The hierarchical dist of a row-wise sequence sharding as one operator
// (HierOutputDist, one per sharding). After one host sync per call, every
// owner has a ready KJT of the keys it looks up, and every rank the plans
// of the output dist. Then the output dist runs one kernel per
// direction, which only moves rows.
//
// Across nodes a requester's keys for an owner on another node go to its
// proxy, the GPU of its own node on the owner's rail. The proxy deduplicates
// them per (owner node, table, key) and puts each node-unique key to the
// owner once, over InfiniBand on that rail (NCCL GIN). The output dist moves
// the rows of those keys the same way: one row per node-unique key over
// InfiniBand, the node's own copies over NVLink.
//
// The op's window is static: its constructor allocates it once, at the
// capacities of its configuration, and it never grows; a call's own tables
// come from the caching allocator. A call that does not fit the capacities
// (its result does not fit, on every rank alike) still runs its steps to
// their end with nothing in their messages, so the op stays valid for the
// next call; the caller moves that call some other way.
//
// The op runs each step of a call on the caller's current stream, and each
// step returns the data the next one takes: part_one() is part 1, local;
// start() is the first wait, the in channel's comm kernels and the planner's
// steps; wait() is the second wait, the host sync, which gives the owner's
// KJT and the output dist's plans (InDistResult); lookup_rows() gives the
// rows the lookup writes; then forward() with the forward's plan and the
// lookup's output, and backward() with the backward's plan and the
// gradient. The caller orders them: each step runs on the stream of the step
// before it, or on one the caller ordered after it, and the caller keeps the
// tensors a step reads alive on its stream (record_stream).

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
  // communicator and allocates the window.
  // Then the ranks agree that each can run (its LSA team, GIN, its
  // allocations) before they register the window together. If one cannot,
  // every rank throws the same std::runtime_error, which names the ranks and
  // why, and holds nothing. A rank that fails inside the registration throws
  // alone, and the others wait there, as in NCCL's own initialization.
  HierOutputDist(const std::string &unique_id, Config config);
  // Local: waits for this rank's kernels. Then, unless the op was quiesced
  // after the last kernel of every rank, it waits at most min(timeout, 30 s)
  // for every node and rail peer to reach its own teardown. It lets go of
  // the window only if they all did (a view of lookup_rows still alive keeps
  // its memory); else it keeps it until the process exits.
  ~HierOutputDist();
  HierOutputDist(const HierOutputDist &) = delete;
  HierOutputDist &operator=(const HierOutputDist &) = delete;

  // The input dist of a call (InChannel): part 1, local (no sync, no
  // collective, no sequence number, so the caller may drop it); the first
  // wait, which advances the in channel's sequence number (every rank starts
  // the same calls in the same order); the second wait, the call's one host
  // sync.
  PartOne part_one(PlanInput input);
  Pending start(const PartOne &part);
  InDistResult wait(const Pending &pending);
  // The rows a call's lookup writes its output into, and the output dist
  // reads it from: lookup_rows' first num_recv rows, [num_recv, dim]. A view
  // of the window, valid until the next out kernel.
  at::Tensor lookup_rows(int64_t num_recv);
  // The output dist of a call (OutChannel): forward() returns the rows,
  // backward() the gradient of the lookup output (lookup_grad), a view of
  // lookup_rows valid until the next lookup.
  at::Tensor forward(const FwdPlan &plan, const at::Tensor &lookup_out);
  at::Tensor backward(const BwdPlan &plan, const at::Tensor &grad, bool divide);
  // Collective, after the op's last call, on every rank: syncs this rank's
  // device and meets the other ranks over the op's communicator. No peer
  // then reads this rank's window or sends it a credit any more, so the
  // teardown frees it at once. The next call clears it.
  void quiesce();

  // Diagnostics.
  const Config &config() const { return config_; }
  // The capacities in place, in entries (proxy_bytes in bytes, as
  // configured): lookup_keys; across nodes also remote_keys and what follows
  // from it (num_proxy_rows, rounds_cap), which read 0 on one node.
  std::map<std::string, int64_t> capacities() const;
  // Bytes of the window ("windows": this rank's copy, whole granules) and of
  // each of its buffers ("buffers": what each takes in the window).
  std::map<std::string, std::map<std::string, int64_t>> memory() const;
  bool gin_railed() const { return communicator_->gin_railed(); }
  // In a build with DYNAMICEMB_HIER_TRACE=1, the out kernels' timelines
  // (Trace; empty without it).
  at::Tensor trace() const { return out_->trace().out_ring(); }
  std::vector<int64_t> trace_launches() const { return out_->trace().out_launches(); }

 private:
  void release() noexcept;
  bool exit_round() noexcept;

  Config config_;
  Sizes sizes_;
  std::unique_ptr<Communicator> communicator_;
  std::unique_ptr<SymmetricWindow> window_;
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
