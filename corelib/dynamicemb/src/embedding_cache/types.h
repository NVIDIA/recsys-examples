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
// The node the cache is built from: the map's three fields from the base, the
// concurrency word from the other component. Everything else in this directory is
// generic over it.
// ---------------------------------------------------------------------------

#pragma once

#include <cstdint>

#include "hashmap_view.h"
#include "node_state.h"

namespace embedding_cache {

struct Node : HashMapNode<int64_t, int32_t> {
  AtomicNodeState state;
};

static_assert(sizeof(Node) == 24, "key 8 + links 8 + state 4, padded to 24");

}  // namespace embedding_cache
