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
// The device code of the node's part of a call (planner.h) that the comm kernels, the
// planner's kernels and the bucketize share: a requester's row of node_offsets (NodeRow), the node
// layout from the node's rows (node_layout_small, node_layout_items), and the bucketize's gather
// of the KJT's keys (gather_rows, and gather_rows_of on the host). The push (push_rows) is that
// gather too.
//
// It does not include device.cuh, so planner.cu and bucketize.cu do not get nccl_device.h. The
// build has no relocatable device code: everything here is inline or a template.
#pragma once

#include <ATen/ATen.h>

#include <cstdint>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/input_dist/planner.cuh"
#include "hier_output_dist/input_dist/planner.h"

namespace dyn_emb {
namespace hier_output_dist {

// The node layout's tables (NodeLayout) as kernels write and read them. nullptr: a table the
// layout does not have (remote_offsets, need_remote on one node; kjt_offsets, length_per_key
// across nodes).
struct NodeLayoutView {
  int64_t *own_lengths, *own_offsets, *own_total, *remote_offsets;
  int64_t *need_lookup, *need_remote;
  bool *skip;
  int64_t *row_shift, *kjt_offsets, *length_per_key;
};
// The data of each table of `node` (nullptr: a table it does not have).
inline NodeLayoutView view_of(const NodeLayout &node) {
  return NodeLayoutView{node.own_lengths.data_ptr<int64_t>(),
                        node.own_offsets.data_ptr<int64_t>(),
                        node.own_total.data_ptr<int64_t>(),
                        data_or_null<int64_t>(node.remote_offsets),
                        node.need_lookup.data_ptr<int64_t>(),
                        data_or_null<int64_t>(node.need_remote),
                        node.skip.data_ptr<bool>(),
                        node.row_shift.data_ptr<int64_t>(),
                        data_or_null<int64_t>(node.kjt_offsets),
                        data_or_null<int64_t>(node.length_per_key)};
}

// The node's rows (rows: [local_size][NodeRow::words], row r requester r's) as the layout reads
// them. Its peers wrote them over NVLink, so loads go through L2 only.
struct NodeRows {
  const int64_t *rows;
  NodeRow format;
  // requester r's keys for target t's remote part
  __device__ __forceinline__ int64_t remote(int64_t requester, int64_t target) const {
    return (int64_t)__ldcg((const unsigned long long *)(rows + requester * format.words + target));
  }
  // requester r's offset at its own node's bucket feature j (X)
  __device__ __forceinline__ int64_t offset(int64_t requester, int64_t j) const {
    return (int64_t)__ldcg(
        (const unsigned long long *)(rows + requester * format.words + format.x0 + j));
  }
  // the node's keys of its own node's bucket features before j: sum over the requesters
  __device__ __forceinline__ int64_t node_offset(int64_t local_size, int64_t j) const {
    int64_t keys = 0;
    for (int64_t requester = 0; requester < local_size; ++requester) keys += offset(requester, j);
    return keys;
  }
};

// The node layout's items (node_layout_items): a row shift per bucket feature and its end, and an
// entry per (feature, requester) of the own part.
__host__ __device__ inline int64_t node_layout_items_of(const Topo &topo, int64_t num_features) {
  return (int64_t)topo.world_size * num_features + 1 + topo.local_size * num_features;
}
// node_exchange's CTA pushes its row when the node's rows hold at most this many words (64 KiB),
// and lays out the items when there are at most this many; else grids do (comm_kernels.cu).
constexpr int64_t kNodeRowWordsInCta = 8192;
constexpr int64_t kNodeItemsInCta = 8192;

// The int64 words the node layout works in, between its two parts (node_layout_small, then
// node_layout_items): where the node's keys of each target's bucket features start (base,
// local_size + 1: node_offset at t * num_features), each remote owner's row shift
// (remote_shift, world_size), and the offsets of this rank's blocks for the owners of each
// target's rail (world_size + 1).
__host__ __device__ inline int64_t node_layout_scratch(const Topo &topo, int64_t num_features) {
  return topo.local_size + 1 + 2 * (int64_t)topo.world_size + 1;
}

// Every thread of a CTA of kThreads: the node layout's tables of O(local_size^2 + world_size)
// work (NodeLayout) from the node's rows: each owner's own part and the largest, across nodes
// where each requester's keys start in this rank's remote part and the largest remote part, skip;
// and into scratch (node_layout_scratch) base and each remote owner's row shift. opbf: this
// rank's offset_per_bucket_feature. Requester r has keys of feature f for owner t of its node
// (its bucket feature (node * local_size + t) * num_features + f), and R(r, t) keys for the
// owners (n, t) of the other nodes, which go to the remote part of target t.
__device__ inline void node_layout_small(const NodeRows &in, const int64_t *opbf, const Topo &topo,
                                         int64_t num_features, int64_t lookup_keys,
                                         int64_t remote_keys, int64_t *scratch,
                                         const NodeLayoutView &out) {
  const int64_t local_size = topo.local_size, num_nodes = topo.num_nodes, node = topo.node,
                me = topo.local_rank;
  int64_t *base = scratch, *remote_shift = base + local_size + 1,
          *block_offsets = remote_shift + topo.world_size;
  for (int64_t target = threadIdx.x; target <= local_size; target += kThreads)
    base[target] = in.node_offset(local_size, target * num_features);
  __syncthreads();
  for (int64_t target = threadIdx.x; target < local_size; target += kThreads)
    out.own_total[target] = base[target + 1] - base[target];
  const int64_t most_own = cta_max<int64_t>(
      local_size, [&](int64_t target) { return base[target + 1] - base[target]; }, 0);
  // Across nodes, where each requester's keys start in this rank's remote part, and the largest
  // remote part. The needs, and whether the node skips the call, are the same on every rank of
  // the node.
  bool skips = most_own > lookup_keys;
  if (num_nodes > 1) {
    cta_offsets_of<int64_t>(
        local_size, [&](int64_t requester) { return in.remote(requester, me); },
        out.remote_offsets);
    const int64_t most_remote = cta_max<int64_t>(
        local_size,
        [&](int64_t target) {
          int64_t keys = 0;
          for (int64_t requester = 0; requester < local_size; ++requester)
            keys += in.remote(requester, target);
          return keys;
        },
        0);
    skips = skips || most_remote > remote_keys;
    if (threadIdx.x == 0) *out.need_remote = most_remote;
    // A row of an owner (n, t) of another node is in t's remote part: after the keys of the
    // requesters before this rank, this rank's blocks for the owner nodes before n, and the rows
    // of its block before it.
    auto block_keys = [&](int64_t owner) {
      return opbf[(owner + 1) * num_features] - opbf[owner * num_features];
    };
    cta_offsets_of<int64_t>(
        topo.world_size,
        [&](int64_t index) {
          const int64_t target = index / num_nodes, owner_node = index % num_nodes;
          return owner_node == node ? 0 : block_keys(owner_node * local_size + target);
        },
        block_offsets);
    for (int64_t owner = threadIdx.x; owner < topo.world_size; owner += kThreads) {
      const int64_t owner_node = owner / local_size, target = owner % local_size;
      int64_t place = lookup_keys + block_offsets[target * num_nodes + owner_node] -
                      block_offsets[target * num_nodes];
      for (int64_t requester = 0; requester < me; ++requester)
        place += in.remote(requester, target);
      remote_shift[owner] = owner_node == node ? 0 : place - opbf[owner * num_features];
    }
  }
  if (threadIdx.x == 0) {
    *out.need_lookup = most_own;
    *out.skip = skips;
  }
}

// The node layout's tables of O(local_size) work per item (NodeLayout), items [first, end) in
// steps of `stride` (a CTA's threads, or a grid's): item b <= world_size * num_features is
// bucket feature b's row shift (row_shift); then item world_size * num_features + 1 + k, k =
// f * local_size + r, this rank's own part's entry (feature f, requester r): its lengths and
// offsets (own_lengths, own_offsets), and on one node the owner's KJT offsets and keys per
// feature. scratch: node_layout_small's. Bucketized row `row` of bucket feature b goes to place
// row + row_shift[b] in its target's node_keys: row_shift[b] is where this rank's rows of b start
// there, less where they start among its rows (opbf[b]), so one add places a row. A row of an
// owner t of this node is in t's own part: where this rank's keys of the feature start there (the
// node's keys of the features before, then those of the requesters before), and the rows of the
// bucket feature before it. Every feature of a remote owner has the same row_shift, its
// remote_shift. The rows past the valid ones (bucket feature world_size * num_features) are never
// pushed: their place is their own index.
__device__ inline void node_layout_items(const NodeRows &in, const int64_t *opbf, const Topo &topo,
                                         int64_t num_features, const int64_t *scratch,
                                         int64_t first, int64_t stride, const NodeLayoutView &out) {
  const int64_t local_size = topo.local_size, node = topo.node, me = topo.local_rank,
                bucket_features = (int64_t)topo.world_size * num_features,
                node_features = local_size * num_features;
  const int64_t *base = scratch, *remote_shift = base + local_size + 1;
  for (int64_t item = first; item < bucket_features + 1 + node_features; item += stride) {
    if (item <= bucket_features) {
      const int64_t owner = item / num_features;
      int64_t shift = 0;
      if (item == bucket_features) {
        shift = 0;
      } else if (owner / local_size != node) {
        shift = remote_shift[owner];
      } else {
        const int64_t j = item - node * node_features, target = j / num_features;
        int64_t before = 0;
        for (int64_t requester = 0; requester < me; ++requester)
          before += in.offset(requester, j + 1) - in.offset(requester, j);
        shift = in.node_offset(local_size, j) - base[target] + before - opbf[item];
      }
      out.row_shift[item] = shift;
      continue;
    }
    const int64_t index = item - bucket_features - 1, feature = index / local_size,
                  requester = index % local_size, j = me * num_features + feature;
    int64_t before = 0;
    for (int64_t earlier = 0; earlier < requester; ++earlier)
      before += in.offset(earlier, j + 1) - in.offset(earlier, j);
    const int64_t at = in.node_offset(local_size, j) - base[me] + before;
    out.own_lengths[index] = in.offset(requester, j + 1) - in.offset(requester, j);
    out.own_offsets[index] = at;
    if (out.kjt_offsets != nullptr) {
      // One node: the own part is the owner's KJT
      out.kjt_offsets[index] = at;
      if (index == 0) out.kjt_offsets[node_features] = base[me + 1] - base[me];
      if (requester == 0)
        out.length_per_key[feature] =
            in.node_offset(local_size, j + 1) - in.node_offset(local_size, j);
    }
  }
}

// A requester's bucketize up to its gather (bucketize_rows), as the gather reads it.
// Bucketized row `row` is KJT row inv[row]: its key is the KJT's key there, widened to int64
// with its sign (the 64-bit key DynamicEmb's lookup makes of an int32 key), and its weight
// weights[inv[row]] (weights nullptr: none).
struct GatherRows {
  const int32_t *inv;    // [rows]
  const void *values;    // [rows]: the KJT's keys, int32 or int64, in its order
  bool int32_keys;       // values are int32
  const float *weights;  // [rows]: KJT order, or nullptr
  __device__ __forceinline__ int64_t key(int64_t source) const {
    return int32_keys ? (int64_t) static_cast<const int32_t *>(values)[source]
                      : static_cast<const int64_t *>(values)[source];
  }
};
// The gather of the bucketized rows inv (int32) from the KJT's keys (values: int32 or int64)
// and weights (fp32, or undefined).
inline GatherRows gather_rows_of(const at::Tensor &inv, const at::Tensor &values,
                                 const at::Tensor &weights) {
  return GatherRows{inv.data_ptr<int32_t>(), values.data_ptr(), values.scalar_type() == at::kInt,
                    weights.defined() ? weights.data_ptr<float>() : nullptr};
}

// The bucketize's gather. Every thread of a grid over the rows, a thread per row: each of the
// first `rows` rows goes to put(row, key, weight).
template <typename Put>
__device__ __forceinline__ void gather_rows(const GatherRows &in, int64_t rows, Put &&put) {
  for (int64_t row = first_item(); row < rows; row += item_stride()) {
    const int64_t source = in.inv[row];
    put(row, in.key(source), in.weights != nullptr ? in.weights[source] : 0.f);
  }
}

// What the push reads besides the gather's rows: the bucketize's bucket_feature_of_row and
// offset_per_bucket, and the node layout's row_shift and skip.
struct PushPlaces {
  const int32_t *bucket_feature_of_row;  // [rows]
  const int64_t *offset_per_bucket;      // [world_size + 1]
  const int64_t *row_shift;              // [world_size * num_features + 1]
  const bool *skip;
};

// The push (node_key_push): the bucketize's gather into its targets' node_keys. Every thread of
// a grid over the `rows` bucketized rows, a thread per row. A valid row (below
// offset_per_bucket[world_size]) of bucket feature b, owner o, goes to place
// row + row_shift[b] in its target's node_keys: put(target, place, key, weight,
// node_bucket_feature), the target o's local rank and node_bucket_feature o's node *
// num_features + its feature. A place from lookup_keys on is in the target's remote part. Under
// skip no row goes. With src (one node), each row's place in its owner's KJT goes to src[row]
// in every call: a valid row's place, and a row past them its own index.
template <typename Put>
__device__ __forceinline__ void push_rows(const GatherRows &in, int64_t rows,
                                          const PushPlaces &places, const Topo &topo,
                                          int64_t num_features, int32_t *src, Put &&put) {
  const int64_t local_size = topo.local_size;
  const int64_t valid = *places.skip ? 0 : places.offset_per_bucket[topo.world_size];
  for (int64_t row = first_item(); row < (src != nullptr ? rows : valid); row += item_stride()) {
    const int64_t bucket_feature = places.bucket_feature_of_row[row];
    const int64_t place = places.row_shift[bucket_feature] + row;
    if (src != nullptr) src[row] = (int32_t)place;
    if (row >= valid) continue;
    const int64_t source = in.inv[row], owner = bucket_feature / num_features;
    put(owner % local_size, place, in.key(source), in.weights != nullptr ? in.weights[source] : 0.f,
        (int32_t)(owner / local_size * num_features + bucket_feature % num_features));
  }
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
