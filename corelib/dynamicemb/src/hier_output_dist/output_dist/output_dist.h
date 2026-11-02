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
// The output dist: one kernel per direction of a call (the out channel,
// OutChannel), which only moves rows by the call's plans (plan.h), and the
// kernels' trace (Trace).
#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/config.h"
#include "hier_output_dist/common/device.cuh"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/input_dist/plan.h"
#include "hier_output_dist/output_dist/args.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// The trace of a build with DYNAMICEMB_HIER_TRACE=1 (kTrace): the out
// kernels' timelines, a ring of rows per direction (TraceSlots) on the GPU.
// Without kTrace its ring is empty and it keeps nothing.
class Trace {
 public:
  explicit Trace(int device);
  // The next out kernel's row of its direction's ring, started over on
  // `stream` (two memsets, no kernel), into `args`; returns whether the
  // kernel keeps a timeline (false without kTrace).
  bool start_out(bool forward, OutArgs &args, cudaStream_t stream);
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

// The out channel: the output dist's kernels (one per direction of a call)
// on the caller's stream, under one sequence number (out_seq) the same on
// every rank.
class OutChannel {
 public:
  // channel: the part of the kernels' arguments both channels share, over
  // the registered window; window: the window over this rank's copy.
  OutChannel(const Config &config, const Sizes &sizes, const ChannelStatic &channel,
             const WindowLayout &window, int device);

  // fwd_kernel on the current stream: this rank's rows, from the
  // lookup_rows of its node's owners and (across nodes) the proxy_rows of
  // its node's proxies, into a new tensor of plan.num_out rows. lookup_out
  // must be the rows of lookup_rows the lookup wrote (zero copy).
  at::Tensor forward(const FwdPlan &plan, const at::Tensor &lookup_out);
  // bwd_kernel on the current stream: the gradient of this rank's rows back
  // into the lookup_rows of their owners (across nodes through the proxies,
  // which sum each node-unique key's), times 1 / world_size if `divide`.
  // There each owner's lookup backward reads it.
  void backward(const BwdPlan &plan, at::Tensor grad, bool divide);

  const Trace &trace() const { return trace_; }

 private:
  // CTAs of an out kernel, and of the backward's across nodes those that
  // reduce.
  int out_ctas(bool forward) const;
  int reducers(int ctas) const;
  // The output of a forward of `rows` rows.
  at::Tensor output(int64_t rows);
  // A cooperative launch of `kernel` with its parameter `args`, whose
  // OutArgs part it fills: every CTA is resident, so they may wait for each
  // other. The kernel takes its sequence number and its arrivals only once
  // it is launched: a launch that fails leaves the channel as it was.
  template <typename Args>
  void launch(const void *kernel, bool forward, int ctas, Args &args);

  Sizes sizes_;
  OutArgs static_{};  // what every kernel takes of the op: the channels' part and vec
  Channel channel_;
  int64_t arrive_ = 0;       // the arrive word (CtrlLayout) after the out kernels issued so far
  int64_t out_rows_ = 0;     // rows of the storage of the forward's outputs
  const char *lookup_rows_;  // this rank's lookup_rows
  int64_t dim_;
  at::ScalarType dtype_;
  int world_size_, device_, sms_;
  int ctas_, reduce_ctas_;  // Config's (0: out_ctas' default)
  bool multi_node_;
  Trace trace_;
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
