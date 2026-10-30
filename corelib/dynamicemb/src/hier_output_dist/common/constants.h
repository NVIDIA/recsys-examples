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
// Constants of the hierarchical dist that host and device code share: CTA
// sizes, the static limits that size the kernels' shared-memory tables, the
// rows of a task and the task queues; and ceil_div.
#pragma once

namespace dyn_emb {
namespace hier_output_dist {

constexpr int kThreads = 512;      // threads per CTA
constexpr int kMaxLocalRanks = 8;  // GPUs per node (sizes the per-peer smem tables)
constexpr int kMaxNodes = 32;      // nodes
constexpr int kMaxFeatures = 64;   // features per sharding (sizes the per-feature smem tables)
constexpr int kMaxRanks = kMaxNodes * kMaxLocalRanks;
// Index dedup's heavy rows (more original rows than kHeavy) and the heavy
// groups take sub-tasks of kHeavy rows. A node-unique key of more than kHeavy
// occurrences is heavy, so a key takes at most kHeavy rows of proxy_rows
// (min_proxy_rows_for).
constexpr int kHeavy = 64;
constexpr int kTaskRows = 32;  // rows of an output dist task, one per lane
constexpr int kWarps = kThreads / 32;
// The most rounds of proxy_rows a call's tables hold: credits and push flags
// number the rounds in 20 bits (grant_of), and the tables take int32 offsets
// (meta_layout).
constexpr int kMaxRounds = 1 << 16;
// Task queues of an out kernel (Ctrl::tasks): the workers', and across nodes
// the backward's reducers'.
enum Queue : int { kQueueWork = 0, kQueueReduce, kNumQueues };

// count / Per rounded up (count >= 0), in the type of count + Per.
template <int Per, typename Count>
__host__ __device__ __forceinline__ constexpr auto ceil_div(Count count) {
  return (count + Per - 1) / Per;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
