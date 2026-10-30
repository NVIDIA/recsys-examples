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
// The threads of a CTA: a sync of some of them.
#pragma once

#include "hier_output_dist/common/constants.h"

namespace dyn_emb {
namespace hier_output_dist {

// Threads [0, threads) of a CTA of kThreads threads (whole warps) wait for
// each other: all of them, or fewer on named barrier 1 while the other warps
// run on. Every thread of the group calls it.
__device__ __forceinline__ void cta_sync(int threads) {
  if (threads == kThreads)
    __syncthreads();
  else
    asm volatile("bar.sync 1, %0;" ::"r"(threads) : "memory");
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
