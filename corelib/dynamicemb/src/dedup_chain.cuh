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
// The chained insert of DynamicEmb's dedups: index dedup's
// (segmented_unique_core, unique_op.cu) and the hierarchical dist's node
// dedup at the proxy (insert_kernel, hier_output_dist/input_dist/proxy.cu).
// It finds an input entry's identity or claims a new one for it. Each caller
// hashes its own entries to their buckets. Header-only: the hierarchical dist
// has no relocatable device code.
#pragma once

#include <cstddef>

namespace dyn_emb {

// What chain_insert found or made for an input entry: the chain node of its
// identity, and whether the entry claimed it (then the node is its own index).
struct ChainEntry {
  int node;
  bool claimed;
};

// The chained insert of a dedup: input entry `index` (below 2^31) finds its
// identity in the chain of bucket `bucket`, or claims a new one at its end. A
// lock-free tail append.
//
// The chain nodes are input indices, and the keys stay in the input: a node is
// the index of the entry that claimed an identity, and names it. The chains
// are int32 links, -1 for none:
//   head[bucket]: the first node of the bucket's chain;
//   next[node]:   the node after `node` in its chain.
// The caller takes the bucket from the entry's identity alone, so that equal
// entries share a chain. It clears head, and next at every index that may
// claim, before the inserts (in an earlier kernel). equal(node) tells whether
// node's entry has the entry's identity; it reads only the input, which no
// insert writes.
//
// The entry walks its bucket's chain from the head. At a node equal to it, it
// is a duplicate of that node's identity. At the end of the chain it appends
// its index with an atomicCAS of the -1 there (the head, or the last node's
// next), and claims a new identity. A link changes once, from -1 to a node,
// so a plain load sees -1 or that node: a stale -1 only sends the thread to
// the atomicCAS, which returns the node. The loser of an atomicCAS walks on
// from the node that won it, as the chain grew: that node may be an equal
// entry appended at the same time, and then the loser is its duplicate
// (same-key races collapse). So an entry compares every node before the link
// it takes, each identity is claimed once, and all its entries end on its
// node, whatever the order of the threads. Which entry claims depends on that
// order.
//
// Returns the chain node of the entry's identity (`index` when it claimed),
// and whether it claimed.
template <typename Equal>
__device__ __forceinline__ ChainEntry chain_insert(int *head, int *next, size_t bucket,
                                                   size_t index, Equal equal) {
  int kind = 0;  // 0 none, 1 claim, 2 duplicate
  int rep = -1;  // the chain node: index if it claims, the matched node if a duplicate
  int cur = head[bucket];
  if (cur == -1) {
    const int old = ::atomicCAS(&head[bucket], -1, static_cast<int>(index));
    if (old == -1) {
      kind = 1;
      rep = static_cast<int>(index);
    } else {
      cur = old;
    }
  }
  while (kind == 0) {
    if (equal(cur)) {
      kind = 2;
      rep = cur;
      break;
    }
    const int nxt = next[cur];
    if (nxt == -1) {
      const int old = ::atomicCAS(&next[cur], -1, static_cast<int>(index));
      if (old == -1) {
        kind = 1;
        rep = static_cast<int>(index);
        break;
      }
      cur = old;
    } else {
      cur = nxt;
    }
  }
  return ChainEntry{rep, kind == 1};
}

}  // namespace dyn_emb
