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
// The out kernels' arguments: what both take (OutArgs), and what each one
// adds, its call's plan and the tensors of its call (FwdArgs, BwdArgs).
#pragma once

#include <cstdint>
#include <type_traits>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/device.cuh"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/input_dist/plan.h"

namespace dyn_emb {
namespace hier_output_dist {

// What both out kernels take: the part both channels share, the row width,
// the kernel's place in the out channel, and its timeline.
struct OutArgs : ChannelStatic {
  int vec;                    // a row of lookup_rows in 16-byte vectors
  Channel ch;                 // the out channel, at this kernel's out_seq
  int64_t arrive0;            // the arrive word (CtrlLayout) before this kernel
  unsigned long long *trace;  // [kTraceWords] this kernel's timeline, or nullptr
};

// fwd_kernel's parameter.
struct FwdArgs : OutArgs {
  FwdPlanView plan;
  char *out;       // [num_out, row] the requester's output
  int64_t *ready;  // across nodes [local_size]: rows it pulled from each proxy (index dedup's
                   // heavy rows wait for them); the kernel zeroes it
};

// bwd_kernel's parameter.
struct BwdArgs : OutArgs {
  BwdPlanView plan;
  const char *grad;      // [num_out, row] the gradient of the requester's output
  float *partial;        // [sub-tasks, dim] fp32 sums of the heavy rows' sub-tasks (index
                         // dedup), then of the heavy groups' (heavy_subs, then group_subs)
  uint64_t *heavy_done;  // [heavy rows] sub-tasks of each one done
  uint64_t *group_done;  // [heavy groups] the same (the kernel zeroes both)
  int64_t heavy_subs;    // the heavy rows' sub-tasks
  int64_t group_subs;    // the heavy groups' sub-tasks
  int64_t *ready;        // across nodes, the proxy's [chunks]: keys of each IB chunk reduced;
                         // the kernel zeroes it
  int reducers;          // across nodes: the last CTAs, which reduce
  float scale;           // 1 / world_size under TorchRec's gradient division, else 1
};

static_assert(std::is_trivially_copyable<FwdArgs>::value &&
                  std::is_trivially_copyable<BwdArgs>::value,
              "a kernel parameter is copied as bytes");
static_assert(sizeof(FwdArgs) <= 4096 && sizeof(BwdArgs) <= 4096,
              "the out kernels' parameters fit the parameter space");

// The out kernels' dynamic shared memory: the backward's half_seen
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
