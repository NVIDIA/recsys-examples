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
// The kernels that move embedding rows.
//
// kernels.cuh decides which key lives in which node; this moves the data. The two
// meet at AllocSlot: allocate leaves one per miss, and these read it.
//
// A row is the embedding and its optimizer state together, contiguous, in one
// element type. Everything here addresses rows by number, and the numbers are not
// this file's to find out:
//
//   * a node's row in the cache is the node id
//   * a row in the backing is whatever the backing says it is for a key, and that
//     number is only good within one prefetch call — an expansion moves every
//     row. The caller looks the keys up and hands the numbers in.
//
// ---------------------------------------------------------------------------
// The order, and why there is a kernel for each step
// ---------------------------------------------------------------------------
//
//   writeback   victim rows out to the backing        (cache rows are still intact)
//   fill        rows with a history in from the backing (overwrites the cache row)
//   [initialise] rows with no history — not here, see below
//   publish     LOADING -> RESIDENT
//
// writeback has to finish before fill starts: a node's old row is the victim, and
// fill overwrites it. Kernel order says so. Merging them into one pass that holds
// the old row in registers and issues both directions at once is a later change,
// and a measured one.
//
// Initialisation is left to the caller. A key with no history has nothing to
// load, and what it starts as is the embedding initialiser's and the optimizer's
// business, which already write through an index into a tensor. Doing it here
// would mean a second copy of both. So fill skips those rows, the caller writes
// them in place, and publish — a kernel of its own, after that — is what makes
// them visible.
//
// ---------------------------------------------------------------------------
// Row copies
// ---------------------------------------------------------------------------
//
// One warp per row, element by element. That is correct and it is not tuned: a
// row in host memory crosses PCIe, where 16-byte accesses matter, and whether the
// dimensions allow them is the caller's to say. Vectorising belongs with the
// first measurement, not before it.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "ec_assert.h"
#include "kernels.cuh"

namespace embedding_cache {

// Lanes cooperating on one row. A whole warp, always: the loops below hold no
// barrier, but a partial warp would let lanes of one row and the next take
// different trips through them.
constexpr int kLanesPerRow = 32;
static_assert(kBlock % kLanesPerRow == 0,
              "a block must hold whole rows' worth of lanes");

// A run of equal-length rows, addressed by number. Both the cache's rows and the
// backing's value buffer have this shape, and neither is owned by the view.
template <typename ValueT>
struct Rows {
  ValueT* data;
  int64_t row_elems;

  [[nodiscard]] __forceinline__ __host__ __device__ ValueT* row(
      int64_t i) const {
    return data + i * row_elems;
  }
};

namespace detail {

template <typename ValueT>
__forceinline__ __device__ void copy_row(ValueT* dst, const ValueT* src,
                                         int64_t elems, int lane) {
  for (int64_t i = lane; i < elems; i += kLanesPerRow) dst[i] = src[i];
}

}  // namespace detail

// Rows for the victims out: each slot that evicted a key gets its node's row
// copied to the backing, at the row the backing gave that key.
//
// `victim_row` is indexed by slot. A slot without a victim is skipped, so what is
// in its entry does not matter.
//
// Every victim has a backing row: a cached key is always a backing key, since the
// backing evicts a cached key only together with the cache (see invalidate).
// A missing one would mean the invariant broke, and writing nowhere is the least
// harm, so this asserts and skips.

template <typename MapT, typename ValueT>
__global__ void writeback_kernel(
    Rows<ValueT> cache, Rows<ValueT> backing,
    const AllocSlot<typename MapT::Key, typename MapT::NodeId>* slots,
    const unsigned long long* count, const int64_t* victim_row) {
  const int64_t n = static_cast<int64_t>(*count);
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int lane = static_cast<int>(tid % kLanesPerRow);
  const int64_t groups = int64_t(gridDim.x) * blockDim.x / kLanesPerRow;
  for (int64_t s = tid / kLanesPerRow; s < n; s += groups) {
    const auto slot = slots[s];
    if (slot.node == MapT::kNoNode || !slot.has_victim) continue;
    const int64_t r = victim_row[s];
    EC_ASSERT(r >= 0 && "writeback: an evicted key has no row in the backing");
    if (r < 0) continue;
    detail::copy_row(backing.row(r), cache.row(slot.node), cache.row_elems, lane);
  }
}

// Rows with a history in. `new_row` is indexed by slot: the row the backing holds
// for the incoming key, or negative if it has none. A negative entry is a key new
// to the system, and its row is left for the caller to initialise.

template <typename MapT, typename ValueT>
__global__ void fill_kernel(
    Rows<ValueT> cache, Rows<ValueT> backing,
    const AllocSlot<typename MapT::Key, typename MapT::NodeId>* slots,
    const unsigned long long* count, const int64_t* new_row) {
  const int64_t n = static_cast<int64_t>(*count);
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int lane = static_cast<int>(tid % kLanesPerRow);
  const int64_t groups = int64_t(gridDim.x) * blockDim.x / kLanesPerRow;
  for (int64_t s = tid / kLanesPerRow; s < n; s += groups) {
    const auto slot = slots[s];
    if (slot.node == MapT::kNoNode) continue;
    const int64_t r = new_row[s];
    if (r < 0) continue;
    detail::copy_row(cache.row(slot.node), backing.row(r), cache.row_elems, lane);
  }
}

// LOADING -> RESIDENT for every node a slot bound. After the rows are written,
// all of them, including the ones the caller initialised: it is the kernel
// boundary that orders those writes before the release store in publish().

template <typename MapT, typename NodeT>
__global__ void publish_kernel(
    NodeT* nodes,
    const AllocSlot<typename MapT::Key, typename MapT::NodeId>* slots,
    const unsigned long long* count) {
  const int64_t n = static_cast<int64_t>(*count);
  for (int64_t s = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; s < n;
       s += int64_t(gridDim.x) * blockDim.x) {
    const typename MapT::NodeId node = slots[s].node;
    if (node != MapT::kNoNode) nodes[node].state.publish();
  }
}

// The latest values of cached rows, copied to a dense buffer: for each i, the
// cache row of node_ids[i] in table table_ids[i] lands at out + out_offset[i], and
// entries that are kNoNode are skipped.
//
// Over every table at once, with the tables' own row widths, which is why the
// destination is an offset per entry and not a stride: the caller sums the widths.
//
// For find_nodes' answer, between it and invalidate: the backing has evicted
// these keys, and whoever retains evicted items wants what the cache held, which
// is newer than anything in the backing. The nodes are idle, which the eviction
// guard promised, so nothing is writing them and what is read is a settled value.

template <typename MapT, typename ValueT>
__global__ void fetch_rows_kernel(const Rows<ValueT>* cache,
                                  const typename MapT::NodeId* node_ids,
                                  const int64_t* table_ids, int64_t n,
                                  ValueT* out, const int64_t* out_offset) {
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int lane = static_cast<int>(tid % kLanesPerRow);
  const int64_t groups = int64_t(gridDim.x) * blockDim.x / kLanesPerRow;
  for (int64_t i = tid / kLanesPerRow; i < n; i += groups) {
    const typename MapT::NodeId node = node_ids[i];
    if (node == MapT::kNoNode) continue;
    const Rows<ValueT> rows = cache[table_ids[i]];
    detail::copy_row(out + out_offset[i], rows.row(node), rows.row_elems, lane);
  }
}

}  // namespace embedding_cache
