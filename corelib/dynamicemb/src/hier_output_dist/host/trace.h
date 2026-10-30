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
// The trace of a build with DYNAMICEMB_HIER_TRACE=1 (kTrace): the out
// kernels' timelines, a ring of rows per direction (TraceSlots) on the GPU,
// and the in channel's step times, a ring of rows (InTraceSteps) on the host.
// Without kTrace its rings are empty and it keeps nothing.
#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "hier_output_dist/output/args.cuh"

namespace dyn_emb {
namespace hier_output_dist {

class Trace {
 public:
  explicit Trace(int device);
  // The next out kernel's row of its direction's ring, started over on
  // `stream` (two memsets, no kernel), into `call`; returns the dynamic
  // shared memory its CTAs keep a copy of the row in (0 without kTrace).
  size_t start_out(bool forward, OutCall &call, cudaStream_t stream);
  // That kernel was launched.
  void launched(bool forward);
  // [2, kTraceRing, kTraceWords]: forward, then backward, each kernel in
  // row (launches before it) % ring.
  at::Tensor out_ring() const { return out_; }
  std::vector<int64_t> out_launches() const { return {launches_[0], launches_[1]}; }
  // The step times of in-channel call `seq` into row seq % ring, from its
  // plan's events: a timing event at the call's start, then one at the end
  // of each step (InTraceSteps; one never recorded: a step the call did not
  // run). After the call's sync, so they have completed.
  void keep_step_times(int64_t seq, const std::vector<at::cuda::CUDAEvent> &events);
  // [kTraceRing, kInSteps] on the host.
  at::Tensor in_ring() const { return in_; }

 private:
  at::Tensor out_ = at::empty({0}, at::kLong), in_ = at::empty({0}, at::kLong);
  int64_t launches_[2] = {0, 0};  // out kernels of each direction launched so far
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
