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

// ---------------------------------------------------------------------------
// Giving each of N threads its own index in a growing array.
//
// The same operation shows up three times in this cache, and would otherwise be
// written three times:
//
//   * compacting misses out of a lookup, so the allocate path walks a dense
//     range instead of a sparse one
//   * reserving a slot on the reclaim queue when a node goes idle
//   * taking a ticket off the reclaim queue when a node is wanted
//
// All three are "the threads that want an index get consecutive ones, and the
// counter ends up advanced by exactly the number that wanted one".
//
// ---------------------------------------------------------------------------
// Two levels, and when each is worth it
// ---------------------------------------------------------------------------
//
// warp_reserve is 32 lanes sharing one atomic. No barrier, no scratch, callable
// from anywhere — including a kernel with an ordinary grid-stride loop and
// divergent control flow.
//
// BlockReserve::reserve puts a whole block behind one atomic, which is another
// 8x at 256 threads. It costs three __syncthreads() per call, and that is not
// only their own latency: they stop a block's warps from running independently,
// so a kernel that was hiding memory latency across warps stops doing so. On a
// pointer-chasing kernel that can cost more than the atomics it saves.
//
// So: BlockReserve where the kernel is already block-collective and the work
// per thread is short; warp_reserve where it is not. The block version is built
// out of the warp version rather than repeating it — it aggregates onto a
// counter in shared memory first, then promotes once to the global one.
//
// ---------------------------------------------------------------------------
// A caveat worth checking before trusting the first level
// ---------------------------------------------------------------------------
//
// nvcc has aggregated warp-uniform atomicAdd of a constant by itself since CUDA
// 9, and `atomicAdd(counter, 1)` under divergence is close to that shape. How
// much warp_reserve adds on top of what the compiler already does has to be
// read out of the SASS rather than assumed. The block level is not something
// the compiler can do.
// ---------------------------------------------------------------------------

#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace embedding_cache {

namespace cg = cooperative_groups;

// One atomic per warp instead of one per lane.
//
// Returns this thread's reserved index — zero, and meaningless, for a thread
// that passed want=false, since nothing was reserved for it.
//
// Not collective. A thread that does not want an index need not call this at
// all, which is what makes it usable inside ordinary divergent code.
[[nodiscard]] __forceinline__ __device__ unsigned long long warp_reserve(
    bool want, unsigned long long* counter) {
  if (!want) return 0ull;
  // Formed inside the branch, so the group is exactly the lanes that took it.
  auto g = cg::coalesced_threads();
  unsigned long long base = 0ull;
  if (g.thread_rank() == 0) {
    base = atomicAdd(counter, static_cast<unsigned long long>(g.size()));
  }
  return g.shfl(base, 0) + g.thread_rank();
}

// One atomic per block.
//
// BLOCK-COLLECTIVE. Every thread of the block has to call this, the same number
// of times, in the same order — it contains __syncthreads(), and a thread that
// skips one hangs the block. A caller's loop therefore strides by whole blocks
// and handles the tail of its range with `want` rather than by exiting early:
//
//   for (int64_t base = blockIdx.x * blockDim.x; base < n;
//        base += gridDim.x * blockDim.x) {
//     const int64_t i = base + threadIdx.x;
//     const unsigned long long slot = BlockReserve::reserve(i < n && p(i), c, s);
//     ...
//   }
//
// A block whose whole range is past the end never enters the loop and so never
// reaches a barrier, which is what lets a caller launch a worst-case grid
// against a count only the device knows.
struct BlockReserve {
  // One per block, in shared memory. Left uninitialised on purpose: reserve()
  // zeroes it, so there is no precondition a caller can quietly fail to meet.
  struct Scratch {
    unsigned long long count;
    unsigned long long base;
  };

  [[nodiscard]] __forceinline__ __device__ static unsigned long long reserve(
      bool want, unsigned long long* counter, Scratch& s) {
    if (threadIdx.x == 0) s.count = 0ull;
    __syncthreads();

    // Level one, onto shared memory rather than global.
    const unsigned long long rank_in_block = warp_reserve(want, &s.count);
    __syncthreads();

    // Level two. One thread turns the block's total into one global atomic.
    if (threadIdx.x == 0) {
      s.base = s.count ? atomicAdd(counter, s.count) : 0ull;
    }
    __syncthreads();

    return s.base + rank_in_block;
  }
};

}  // namespace embedding_cache
