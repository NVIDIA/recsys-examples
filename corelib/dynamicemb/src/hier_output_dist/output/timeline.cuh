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
// The out kernels' timeline (a build with DYNAMICEMB_HIER_TRACE=1, kTrace):
// stamps, counters and bins, which each CTA keeps in shared memory and folds
// into the kernel's row. Without kTrace it compiles away.
#pragma once

#include "hier_output_dist/common/trace_flag.h"
#include "hier_output_dist/common/trace_slots.h"
#include "hier_output_dist/device/spin.cuh"
#include "hier_output_dist/output/args.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// The timeline of an out kernel: its row (args.trace, nullptr without
// kTrace) and the CTA's shared copy (TraceSmSlots, in the kernel's dynamic
// shared memory). Stamps check on() themselves; the rest is for when it is
// on. Each member does its work only with kTrace.
struct OutTimeline {
  const OutArgs &args;
  __device__ __forceinline__ explicit OutTimeline(const OutArgs &out_args) : args(out_args) {}
  __device__ __forceinline__ bool on() const { return kTrace && args.trace != nullptr; }

  // One thread: `slot` keeps the earliest (or the latest) time any CTA
  // stamps it with.
  __device__ __forceinline__ void stamp_earliest(int slot) const { stamp(slot, true); }
  __device__ __forceinline__ void stamp_latest(int slot) const { stamp(slot, false); }

  // The CTA's shared copy.
  __device__ __forceinline__ static unsigned long long *shared() {
    extern __shared__ unsigned long long trace_dyn_sm[];
    return trace_dyn_sm;
  }
  // At the top of the kernel, every thread of the CTA (on() is the same for
  // all): the CTA's start, the kernel's first and last CTA start, and the
  // shared copy set to its start values.
  __device__ __forceinline__ void begin_cta() const {
    if constexpr (kTrace) {
      if (args.trace == nullptr) return;
      unsigned long long *sm = shared();
      for (int i = threadIdx.x; i < kTraceSmWords; i += blockDim.x)
        sm[i] = i == kSmNvFirst || i == kSmFirstLanded ? ~0ull : 0ull;
      __syncthreads();
      if (threadIdx.x == 0) {
        const unsigned long long now = now_ns();
        sm[kSmStart] = now;
        atomicMin(args.trace + kTraceStart, now);
        atomicMax(args.trace + kTraceStartLast, now);
      }
      __syncthreads();
    }
  }
  // A worker's task (lane 0), started at `start`: it moved `nv` bytes over
  // NVLink and `local` bytes of rows within this GPU.
  __device__ __forceinline__ void task(unsigned long long start, unsigned long long nv,
                                       unsigned long long local) const {
    if constexpr (kTrace) {
      unsigned long long *sm = shared();
      if (nv > 0) {
        const unsigned long long now = now_ns();
        atomicMin(sm + kSmNvFirst, start);
        atomicMax(sm + kSmNvLast, now);
        atomicAdd(sm + kSmNvBytes, nv);
        add(sm + kSmBins, 0, now, nv);
      }
      if (local > 0) atomicAdd(sm + kSmLocalBytes, local);
    }
  }
  // Rows from another node landed (`rows` more, the first ones when `first`).
  __device__ __forceinline__ void landed(unsigned long long rows, bool first) const {
    if constexpr (kTrace) {
      unsigned long long *sm = shared();
      const unsigned long long now = now_ns();
      if (first) atomicMin(sm + kSmFirstLanded, now);
      atomicAdd(sm + kSmIbIn, rows);
      add(sm + kSmBins, 1, now, rows);
    }
  }
  // The same once the CTA folded its shared copy (the backward's last CTA):
  // straight into the kernel's row.
  __device__ __forceinline__ void landed_late(unsigned long long rows, bool first) const {
    if constexpr (kTrace) {
      const unsigned long long now = now_ns();
      if (first) atomicMin(args.trace + kTraceFirstLanded, now);
      atomicAdd(args.trace + kTraceIbIn, rows);
      add(args.trace + kTraceBinNv, 1, now, rows);
    }
  }
  // Rows put to another node (`rows` more).
  __device__ __forceinline__ void posted(unsigned long long rows) const {
    if constexpr (kTrace) {
      atomicAdd(shared() + kSmIbOut, rows);
      add(shared() + kSmBins, 2, now_ns(), rows);
    }
  }
  // Every thread of the CTA, once every warp of the CTA is done with the
  // shared copy: thread 0 folds the stamps and counters into the kernel's
  // row, every thread some bins.
  __device__ __forceinline__ void fold_cta() const {
    if constexpr (kTrace) {
      const unsigned long long *sm = shared();
      unsigned long long *row = args.trace;
      if (threadIdx.x == 0) {
        if (sm[kSmNvFirst] != ~0ull) atomicMin(row + kTraceNvFirst, sm[kSmNvFirst]);
        if (sm[kSmFirstLanded] != ~0ull) atomicMin(row + kTraceFirstLanded, sm[kSmFirstLanded]);
        if (sm[kSmNvLast] != 0) atomicMax(row + kTraceNvLast, sm[kSmNvLast]);
        const int sums[4][2] = {{kSmNvBytes, kTraceNvBytes},
                                {kSmLocalBytes, kTraceLocalBytes},
                                {kSmIbIn, kTraceIbIn},
                                {kSmIbOut, kTraceIbOut}};
        for (int i = 0; i < 4; ++i)
          if (sm[sums[i][0]] != 0) atomicAdd(row + sums[i][1], sm[sums[i][0]]);
      }
      for (int i = threadIdx.x; i < 3 * kTraceBins; i += blockDim.x)
        if (sm[kSmBins + i] != 0) atomicAdd(row + kTraceBinNv + i, sm[kSmBins + i]);
    }
  }

 private:
  __device__ __forceinline__ void stamp(int slot, bool earliest) const {
    if constexpr (kTrace) {
      if (args.trace == nullptr) return;
      const unsigned long long now = now_ns();
      if (earliest)
        atomicMin(args.trace + slot, now);
      else
        atomicMax(args.trace + slot, now);
    }
  }
  // Timeline `which` (0: NVLink bytes, 1: rows landed, 2: rows posted) of
  // `bins` (the shared copy's or the kernel row's) gets `amount` at `time`,
  // in the bin of `time` after this CTA's start.
  __device__ __forceinline__ void add(unsigned long long *bins, int which, unsigned long long time,
                                      unsigned long long amount) const {
    const unsigned long long start = shared()[kSmStart];
    const unsigned long long bin = time > start ? (time - start) / kTraceBinNs : 0;
    atomicAdd(bins + which * kTraceBins + (int)min(bin, (unsigned long long)kTraceBins - 1),
              amount);
  }
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
