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
// How many blocks to launch.
//
// This lived inside HashMapStorage until ReclaimQueueStorage needed the same
// thing. Two components that each launch a grid-stride initialisation kernel is
// the point at which one copy is better than two.
//
// For a grid-stride kernel the answer is "enough to fill the device once".
// Beyond that, extra blocks only queue behind the resident ones and buy
// nothing; below it, SMs sit idle. So the cap is one wave at full occupancy:
//
//     num_sms * (max_thread_per_sm / block)
//
// and the grid is the smaller of that and what the work actually needs.
//
// The formula is the repository's — dynamic_emb_op.cu:211, and again at :521,
// :623 and :763; unique_op.cu spells it three more times with a hand-picked
// BLOCKS_PER_SM instead of deriving it. The numbers come from dyn_emb::DeviceProp
// (src/utils.h), the same singleton those call sites use, which reads
// cudaGetDeviceProperties once and holds the result.
//
// That include is the one place embedding_cache reaches outside itself. utils.h
// brings check.h, <cuda.h>, <cuda_fp16.h> and <cuda_bf16.h> with it.
// ---------------------------------------------------------------------------

#pragma once

#include <cstdint>

#include "../utils.h"

namespace embedding_cache {

// 256 is what every other kernel in this tree uses.
constexpr int kBlock = 256;
static_assert(kBlock <= 1024,
              "a block cannot exceed maxThreadsPerBlock, which is 1024 on "
              "every supported architecture; grid_for relies on that to know "
              "that max_thread_per_sm / kBlock is at least one");

// Blocks for `n` items, capped at one wave.
//
// n <= 0 still yields one block, so a launch is always well-formed and no
// caller has to guard work it may not have.
[[nodiscard]] inline int grid_for(int64_t n) {
  if (n <= 0) return 1;
  const dyn_emb::DeviceProp& p = dyn_emb::DeviceProp::getDeviceProp();
  const int64_t cap =
      static_cast<int64_t>(p.num_sms) * (p.max_thread_per_sm / kBlock);
  const int64_t want = (n + kBlock - 1) / kBlock;
  return static_cast<int>(want < cap ? want : cap);
}

}  // namespace embedding_cache
