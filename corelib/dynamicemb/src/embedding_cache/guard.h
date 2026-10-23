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
// The question the backing's eviction asks the cache: may this key go?
//
// The backing chooses which keys leave the system, by its own score, and the cache
// holds no score and has no say in that — except that a key a step is using must
// not be taken. So when the backing is about to evict a key it asks, and the cache
// answers no for a key whose node holds references. See kernels.cuh's
// EvictionGuard for the rule and docs/lightweight_cache_design.md, 6.5.
//
// What crosses over is a plain struct of two fields, because the asker is built
// apart from this directory: the backing's insert kernels, and the LruLfu eviction
// cubin that is compiled and linked on its own. Neither knows about torch or about
// the cache's storage; both can include this file, which needs only the map view
// and the node's state word. `tables` is null when nothing is asking to be
// protected, which is every use of the backing but a cache's.
// ---------------------------------------------------------------------------

#pragma once

#include <cstdint>

#include "types.h"

namespace embedding_cache {

struct EvictionGuardView {
  // The cache's per-table maps, an array on the device, or null for no guard.
  const void* tables;
  int32_t num_tables;

  // True if the key must NOT be evicted: it is cached in `table` and its node holds
  // a reference. A table the guard does not know — the backing has more than the
  // cache — has nothing cached, so nothing to protect.
  [[nodiscard]] __forceinline__ __device__ bool refuses(int64_t key,
                                                        int64_t table) const {
    if (tables == nullptr || table < 0 || table >= num_tables) return false;
    using Map = HashMap<Node>;
    const Map& map = static_cast<const Map*>(tables)[table];
    const int32_t node = map.lookup(key);
    return node != Map::kNoNode && map.nodes[node].state.load().has_refs();
  }
};

static_assert(sizeof(EvictionGuardView) == 16,
              "two fields and nothing else: it is passed to a kernel by value and "
              "written into a launch-parameter struct");

}  // namespace embedding_cache
