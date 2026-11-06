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
// The call an out kernel moves the rows of: its plan and its tensors
// (FwdArgs, BwdArgs). Beside it each out kernel takes the op's window, its
// GIN device communicator and its sizes, and its place in the out channel
// (its out_seq, and the arrive word before it).
#pragma once

#include <cstdint>
#include <type_traits>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/device.cuh"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/input_dist/plan.h"

namespace dyn_emb {
namespace hier_output_dist {

// fwd_kernel's call.
struct FwdArgs {
  FwdPlanView plan;
  char *out;                  // [num_out, row] the requester's output
  int64_t *ready;             // across nodes [local_size]: rows it pulled from each proxy (heavy
                              // rows wait for them); the kernel zeroes it
  unsigned long long *trace;  // [kTraceWords] this kernel's timeline, or nullptr
};

// bwd_kernel's call.
struct BwdArgs {
  BwdPlanView plan;
  const char *grad;           // [num_out, row] the gradient of the requester's output
  float *partial;             // [plan.heavy_subs, dim] fp32 sums of the heavy rows' sub-tasks
  uint64_t *heavy_done;       // [heavy rows] sub-tasks of each one done (the kernel zeroes it)
  int64_t *ready;             // across nodes, the proxy's [max_chunks]: keys of each chunk of
                              // its stream reduced; the kernel zeroes those of its chunks
  unsigned long long *trace;  // [kTraceWords] this kernel's timeline, or nullptr
  int reducers;               // across nodes: the last CTAs, which reduce
  float scale;                // 1 / world_size under TorchRec's gradient division, else 1
};

static_assert(std::is_trivially_copyable<FwdArgs>::value &&
                  std::is_trivially_copyable<BwdArgs>::value,
              "a kernel parameter is copied as bytes");

// The out kernels' dynamic shared memory: the backward's free_seen
// [local_size] (OutTables), then, when the kernel keeps a timeline, the
// CTA's copy of it (TraceSmSlots, OutTimeline).
__device__ __forceinline__ unsigned long long *out_shared() {
  extern __shared__ unsigned long long out_dyn_sm[];
  return out_dyn_sm;
}
// Its bytes, what each out kernel is launched with.
inline size_t out_shared_bytes(int local_size, bool timeline) {
  return 8 * ((size_t)local_size + (timeline ? kTraceSmWords : 0));
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
