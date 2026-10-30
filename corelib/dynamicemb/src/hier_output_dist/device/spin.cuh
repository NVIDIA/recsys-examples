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
// The GPU's clock (%globaltimer) and bounded spin-waits.
#pragma once

#include <cstdint>
#include <cstdio>

namespace dyn_emb {
namespace hier_output_dist {

__device__ __forceinline__ uint64_t now_ns() {
  uint64_t ns;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
  return ns;
}

// Waits report instead of hanging forever: after `limit` ns the kernel prints
// what it was waiting for and traps.
struct Spin {
  uint64_t start_ns, limit;
  uint32_t polls = 0;
  __device__ explicit Spin(uint64_t limit_ns) : start_ns(now_ns()), limit(limit_ns) {}
  __device__ __forceinline__ void pause(const char *what, long long first, long long second,
                                        int rank) {
    if (((++polls) & 1023) == 0 && now_ns() - start_ns > limit) {
      printf("[hier_output_dist] rank %d timed out waiting for %s (%lld, %lld)\n", rank, what,
             first, second);
      __trap();
    }
    __nanosleep(64);
  }
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
