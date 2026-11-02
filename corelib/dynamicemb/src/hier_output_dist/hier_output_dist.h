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
// communicator, its window, its trace, and its two channels.
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

#include "hier_output_dist/args.cuh"
#include "hier_output_dist/common.h"
#include "hier_output_dist/config.h"
#include "hier_output_dist/plan.h"
#include "hier_output_dist/planner.h"
#include "hier_output_dist/window.h"

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

// The trace of a build with DYNAMICEMB_HIER_TRACE=1 (kTrace): the out
// kernels' timelines, a ring of rows per direction (TraceSlots) on the GPU.
// Without kTrace its ring is empty and it keeps nothing.

class Trace {
 public:
  explicit Trace(int device);
  // The next out kernel's row of its direction's ring, started over on
  // `stream` (two memsets, no kernel), into `call`; returns whether the
  // kernel keeps a timeline (false without kTrace).
  bool start_out(bool forward, OutCall &call, cudaStream_t stream);
  // That kernel was launched.
  void launched(bool forward);
  // [2, kTraceRing, kTraceWords]: forward, then backward, each kernel in
  // row (launches before it) % ring.
  at::Tensor out_ring() const { return out_; }
  std::vector<int64_t> out_launches() const { return {launches_[0], launches_[1]}; }

 private:
  at::Tensor out_ = at::empty({0}, at::kLong);
  int64_t launches_[2] = {0, 0};  // out kernels of each direction launched so far
};

// The in channel: the input dist of every call after part 1 (the first
// wait), on the caller's stream, under one sequence number (in_seq) the same
// on every rank. Its comm kernels (comm_kernels.cuh)
// exchange with the node and rail peers; the planner's steps (planner.h) lay the
// call out between them, as ATen ops on the same stream.

class InChannel {
 public:
  // channel: the part of the kernels' arguments both channels share, over
  // the registered window.
  InChannel(const Config &config, const Sizes &sizes, const ChannelStatic &channel,
            const SymmetricWindow &window, int device);

  // Queues the call on the current stream, after part 1: the steps of the
  // node (its counts, layout and keys), across nodes those of the proxy, the
  // rail and the requester, then the call's tables and status, which the
  // plan's wait() reads. It never reads a device value on the host.
  void start(Plan &plan);

 private:
  ChannelStatic channel_;
  Sizes sizes_;
  bool node_dedup_;
  int64_t seq_ = 0;  // the calls started
  // This rank's buffers the steps read and write, views of its copy of the
  // window without an allocator (the window outlives them): the own part of
  // node_keys (and node_weights), and across nodes its remote part,
  // remote_cells, the buffers the proxy writes, rail_keys, cell_stage and
  // rail_cells.
  at::Tensor own_keys_, own_weights_;
  at::Tensor remote_keys_, remote_weights_, remote_cells_;
  ProxyBuffers proxy_buffers_;
  at::Tensor rail_keys_, cell_stage_, rail_cells_;
  // Constants of the op across nodes: the table of each feature (numbered
  // densely), the other nodes, and the words of the proxy headers this
  // requester reads.
  at::Tensor tables_, other_nodes_;
  PeerWords header_gather_;
};

// The out channel: the output dist's kernels (one per direction of a call)
// on the caller's stream, under one sequence number (out_seq) the same on
// every rank, and the gradient views of lookup_rows the backward hands out.

class OutChannel {
 public:
  // `channel_static`: the part of the kernels' arguments both channels share.
  OutChannel(const Config &config, const Sizes &sizes, const ChannelStatic &channel_static,
             SymmetricWindow &window, Trace &trace, int device);

  // The rows the lookup of a call writes, the first num_recv rows of
  // lookup_rows as a [num_recv, dim] view, valid until the next out kernel.
  at::Tensor lookup_rows(const Plan &plan);
  // fwd_kernel on the current stream: this rank's rows, from the
  // lookup_rows of its node's owners and (across nodes) the proxy_rows of
  // its node's proxies, into a new tensor of num_out rows. lookup_out must
  // be the view lookup_rows() returned, which the lookup wrote (zero copy).
  // Returns the rows and the backward's maps, which the plan lets go of:
  // autograd saves them for backward().
  std::tuple<at::Tensor, std::vector<at::Tensor>> forward(Plan &plan, const at::Tensor &lookup_out);
  // bwd_kernel on the current stream: the gradient of this rank's rows back
  // into the lookup_rows of their owners (across nodes through the proxies,
  // which sum each node-unique key's), times 1 / world_size if `divide`.
  // `maps` are those forward() returned, as autograd saved them. Returns
  // this rank's lookup_grad, the gradient of its lookup output, a view of
  // lookup_rows valid until the next lookup.
  at::Tensor backward(Plan &plan, at::Tensor grad, const std::vector<at::Tensor> &maps,
                      bool divide);

 private:
  // CTAs of an out kernel, and of the backward's across nodes those that
  // reduce.
  int out_ctas(bool forward) const;
  int reducers(int ctas) const;
  // The output of a forward of `rows` rows.
  at::Tensor output(int64_t rows);
  // A cooperative launch: every CTA is resident, so they may wait for each
  // other. The kernel takes its sequence number and its arrivals only once
  // it is launched: a launch that fails leaves the channel as it was.
  void launch(const void *kernel, bool forward, OutCall &call);

  OutStatic static_{};  // what every kernel takes
  Channel channel_;
  int64_t arrive_ = 0;    // the arrive word (CtrlLayout) after the out kernels issued so far
  int64_t out_rows_ = 0;  // rows of the storage of the forward's outputs
  int64_t dim_;
  at::ScalarType dtype_;
  int world_size_, device_, sms_;
  int ctas_, reduce_ctas_;  // Config's (0: out_ctas' default)
  bool multi_node_;
  SymmetricWindow &window_;
  Trace &trace_;
};

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
// The op's window is static: its constructor allocates it once, at the
// capacities of its configuration, and it never grows; a call's own tables
// come from the caching allocator. A call that does not fit the capacities
// (wait() returns false, on every rank alike) still runs its steps to their
// end with nothing in their messages, so the op stays valid for the next
// call; the caller moves that call some other way.
//
// A call is a Plan, which holds its state. The op runs each of its steps,
// every one on the caller's current stream: plan() is part 1, local;
// start() is the first wait, the in channel's comm kernels and the planner's
// steps; wait() is the second wait, the host sync; lookup_rows() gives the
// rows the lookup writes; then forward() and backward(). The caller orders
// them: each step runs on the stream of the step before it, or on one the
// caller ordered after it, and the caller keeps the tensors a step reads
// alive on its stream (record_stream).

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

  // Part 1 of a call on the current stream: the keys partitioned by owner,
  // and index dedup's fan-out. Local: no sync, no collective, no sequence
  // number, so the plan may be dropped.
  std::shared_ptr<Plan> plan(PlanInput input);
  // The first wait: the rest of the call after part 1 (the comm kernels and
  // the planner's steps, InChannel); advances the in channel's sequence
  // number. Every rank starts the same calls in the same order.
  void start(Plan &plan);
  // The second wait, the call's one host sync. False: the call does not fit
  // the capacities (the same on every rank), and the caller moves it some
  // other way. True: the plan holds the owner's KJT and the maps of its
  // output dist.
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
  at::Tensor trace() const { return trace_.out_ring(); }
  std::vector<int64_t> trace_launches() const { return trace_.out_launches(); }

 private:
  void release() noexcept;
  bool exit_round() noexcept;

  Config config_;
  Sizes sizes_;
  std::unique_ptr<Communicator> communicator_;
  std::unique_ptr<SymmetricWindow> window_;
  // The part of the kernels' arguments both channels share, once the window
  // is registered.
  ChannelStatic channel_static_{};
  OwnerRules rules_;  // part 1's
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
