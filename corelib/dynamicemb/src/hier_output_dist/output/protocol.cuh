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
// The out channel's protocol, which both out kernels follow: the entry
// barrier, a CTA's arrival at its end, and the landed_rows words.
#pragma once

#include <cstdint>

#include "hier_output_dist/common/trace_slots.h"
#include "hier_output_dist/device/cta.cuh"
#include "hier_output_dist/device/memory_order.cuh"
#include "hier_output_dist/device/node_sync.cuh"
#include "hier_output_dist/device/node_window.cuh"
#include "hier_output_dist/output/args.cuh"
#include "hier_output_dist/output/timeline.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// The out channel's entry barrier, on its node barrier word (out_started):
// CTA 0 publishes this rank's out_seq, after a system
// fence for what earlier kernels wrote (lookup_rows, which node peers read)
// and what CTA 0 reset for this kernel; every CTA then waits until all node
// peers have published theirs. A peer there has ended its lookup (which wrote
// its lookup_rows before the kernel) and every earlier out kernel. Every
// thread of threads [0, threads) (cta_sync), in every CTA.
__device__ __forceinline__ void pass_entry_barrier(const OutArgs &args, int threads) {
  const int tid = threadIdx.x, local_size = args.topo.local_size;
  const NodeWindow node(args.window, args.topo.local_rank);
  const NodeBarrier<OutArgs> barrier(args, node, node.layout(args).ctrl.out_started());
  if (blockIdx.x == 0 && tid == 0) {
    fence_sc_sys();
    barrier.publish();
  }
  for (int peer = tid; peer < local_size; peer += threads)
    barrier.wait(peer, "out-channel node barrier");
  cta_sync(threads);
  if (tid == 0) {
    const OutTimeline timeline(args);
    timeline.stamp_earliest(kTraceEnterFirst);
    timeline.stamp_latest(kTraceEntered);
  }
}

// The end of a CTA of an out kernel: each CTA arrives once in its arrive word, a
// running total over the rank's out kernels (so nothing is reset), and the
// one whose arrival brings it to arrive0 + gridDim.x is this kernel's last.
// Everything a CTA wrote and every load it made are ordered before its
// arrival, so the last CTA sees everything the others did. Every thread of
// the CTA (tracing: the CTA first folds its shared copy of the timeline into
// the kernel's row). Returns, to every thread, whether this CTA is the last.
__device__ __forceinline__ bool arrive_cta(const OutArgs &args) {
  const int64_t target = args.arrive0 + gridDim.x;
  __shared__ int last;
  __syncthreads();
  const OutTimeline timeline(args);
  if (timeline.on()) timeline.fold_cta();
  if (threadIdx.x == 0) {
    fence_sc_sys();
    uint64_t *word = NodeWindow(args.window, args.topo.local_rank).layout(args).ctrl.arrive();
    last = (int64_t)atomicAdd((unsigned long long *)word, 1ull) + 1 == target;
    if (last) fence_sc_sys();
  }
  __syncthreads();
  return last;
}

// A proxy's landed_rows word (one per owner node): out_seq, and the rows of
// that owner node that have landed during forward out_seq.
using LandedRows = TaggedWord<32>;

}  // namespace hier_output_dist
}  // namespace dyn_emb
