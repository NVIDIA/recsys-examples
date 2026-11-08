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
// planner's kernels and the bucketize share: the node layout from the node's offsets
// (node_layout_of), and the bucketize's gather of the KJT's keys (gather_rows, and
// gather_rows_of on the host). The push (push_rows) is that gather too.
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
// layout does not have (requester_start, need_remote on one node; kjt_offsets, length_per_key
// across nodes).
struct NodeLayoutView {
  int64_t *own_lengths, *own_offsets, *own_total, *requester_start;
  int64_t *need_lookup, *need_remote;
  bool *skip;
  int64_t *row_shift, *kjt_offsets, *length_per_key;
};
// The data of each table of `node` (nullptr: a table it does not have).
inline NodeLayoutView view_of(const NodeLayout &node) {
  return NodeLayoutView{node.own_lengths.data_ptr<int64_t>(),
                        node.own_offsets.data_ptr<int64_t>(),
                        node.own_total.data_ptr<int64_t>(),
                        data_or_null<int64_t>(node.requester_start),
                        node.need_lookup.data_ptr<int64_t>(),
                        data_or_null<int64_t>(node.need_remote),
                        node.skip.data_ptr<bool>(),
                        node.row_shift.data_ptr<int64_t>(),
                        data_or_null<int64_t>(node.kjt_offsets),
                        data_or_null<int64_t>(node.length_per_key)};
}

// The int64 words node_layout_of works in: per (owner of the node, feature) the node's keys, those
// of the requesters before this rank, and their ends over the node's owners; per (requester,
// target) the requester's keys for the target's remote part; and the ends of this rank's blocks
// for the owners of each target's rail.
__host__ __device__ inline int64_t node_layout_scratch(const Topo &topo, int64_t num_features) {
  const int64_t local_size = topo.local_size;
  return 3 * local_size * num_features + 1 + local_size * local_size + topo.world_size + 1;
}

// Every thread of a CTA of kThreads: the layout of the node's keys (NodeLayout) into `out`, from
// the node's offsets: offsets [local_size][world_size * num_features + 1], row r requester r's
// offset_per_bucket_feature (this rank's own is row local_rank). Its peers wrote them over
// NVLink, so loads go through L2 only. scratch: node_layout_scratch words.
// Requester r has c(r, t, f) keys of feature f for owner t of its node (its bucket feature
// (node * local_size + t) * num_features + f), and R(r, t) keys for the owners (n, t) of the
// other nodes, which go to the remote part of target t.
__device__ inline void node_layout_of(const int64_t *offsets, const Topo &topo,
                                      int64_t num_features, int64_t lookup_keys,
                                      int64_t remote_keys, int64_t *scratch,
                                      const NodeLayoutView &out) {
  const int64_t local_size = topo.local_size, num_nodes = topo.num_nodes, node = topo.node,
                me = topo.local_rank, words = topo.world_size * num_features + 1,
                columns = local_size * num_features;
  auto at = [&](int64_t requester, int64_t word) {
    return (int64_t)__ldcg((const unsigned long long *)(offsets + requester * words + word));
  };
  // keys of requester r for owner (owner_node, t), feature f, and for all features
  auto keys_of = [&](int64_t requester, int64_t owner, int64_t feature) {
    const int64_t bucket_feature = owner * num_features + feature;
    return at(requester, bucket_feature + 1) - at(requester, bucket_feature);
  };
  auto bucket_keys = [&](int64_t requester, int64_t owner) {
    return at(requester, (owner + 1) * num_features) - at(requester, owner * num_features);
  };
  // [owner t][feature f]: the node's keys, those of the requesters before this rank, and the ends
  // of the node's keys, owner-major. [requester][target]: its keys for the target's remote part,
  // a warp each, its lanes taking the owner nodes. [target][owner node]: the ends of this rank's
  // blocks for the owners of the target's rail on the other nodes.
  int64_t *total = scratch, *before = total + columns, *ends = before + columns,
          *remote = ends + columns + 1, *block_ends = remote + local_size * local_size;
  for (int64_t index = threadIdx.x; index < columns; index += kThreads) {
    const int64_t owner = node * local_size + index / num_features, feature = index % num_features;
    int64_t keys = 0, earlier = 0;
    for (int64_t requester = 0; requester < local_size; ++requester) {
      const int64_t count = keys_of(requester, owner, feature);
      keys += count;
      if (requester < me) earlier += count;
    }
    total[index] = keys;
    before[index] = earlier;
  }
  const int lane = threadIdx.x & 31;
  for (int64_t index = threadIdx.x >> 5; index < local_size * local_size; index += kWarps) {
    const int64_t requester = index / local_size, target = index % local_size;
    int64_t keys = 0;
    for (int64_t owner_node = lane; owner_node < num_nodes; owner_node += 32)
      if (owner_node != node) keys += bucket_keys(requester, owner_node * local_size + target);
    for (int offset = 16; offset > 0; offset >>= 1)
      keys += __shfl_xor_sync(0xffffffffu, keys, offset);
    if (lane == 0) remote[index] = keys;
  }
  // This rank's own part, in its KJT order [feature][requester]: each requester's keys of each
  // feature for it, and where each starts: the ends of every one but the last.
  for (int64_t index = threadIdx.x; index < columns; index += kThreads)
    out.own_lengths[index] =
        keys_of(index % local_size, node * local_size + me, index / local_size);
  __syncthreads();
  auto own_length = [&](int64_t index) { return out.own_lengths[index]; };
  if (out.kjt_offsets != nullptr) {
    // One node: the own part is the owner's KJT. Its offsets (own_offsets are the first ones),
    // and its keys of each feature.
    cta_exclusive_ends<int64_t>(columns, own_length, out.kjt_offsets);
    for (int64_t i = threadIdx.x; i < columns; i += kThreads)
      out.own_offsets[i] = out.kjt_offsets[i];
    for (int64_t f = threadIdx.x; f < num_features; f += kThreads)
      out.length_per_key[f] =
          out.kjt_offsets[(f + 1) * local_size] - out.kjt_offsets[f * local_size];
  } else {
    cta_exclusive_ends<int64_t>(columns - 1, own_length, out.own_offsets);
  }
  // The node's keys for each owner of the node, and the largest own part.
  cta_exclusive_ends<int64_t>(
      columns, [&](int64_t index) { return total[index]; }, ends);
  for (int64_t target = threadIdx.x; target < local_size; target += kThreads)
    out.own_total[target] = ends[(target + 1) * num_features] - ends[target * num_features];
  __syncthreads();
  const int64_t most_own = cta_max<int64_t>(
      local_size, [&](int64_t target) { return out.own_total[target]; }, 0);
  // Across nodes, where each requester's keys start in this rank's remote part, and the largest
  // remote part. The needs, and whether the node skips the call, are the same on every rank of
  // the node.
  bool skips = most_own > lookup_keys;
  if (num_nodes > 1) {
    cta_exclusive_ends<int64_t>(
        local_size, [&](int64_t requester) { return remote[requester * local_size + me]; },
        out.requester_start);
    const int64_t most_remote = cta_max<int64_t>(
        local_size,
        [&](int64_t target) {
          int64_t keys = 0;
          for (int64_t requester = 0; requester < local_size; ++requester)
            keys += remote[requester * local_size + target];
          return keys;
        },
        0);
    skips = skips || most_remote > remote_keys;
    if (threadIdx.x == 0) *out.need_remote = most_remote;
  }
  if (threadIdx.x == 0) {
    *out.need_lookup = most_own;
    *out.skip = skips;
  }
  // Bucketized row `row` of bucket feature b goes to place row + row_shift[b] in its target's
  // node_keys: row_shift[b] is where this rank's rows of b start there, less where they start
  // among its rows (at(me, b)), so one add places a row. A row of an owner t of this node is in
  // t's own part: where this rank's keys of the feature start there (the node's keys of the
  // features before, then those of the requesters before), and the rows of the bucket feature
  // before it.
  for (int64_t index = threadIdx.x; index < columns; index += kThreads) {
    const int64_t target = index / num_features, feature = index % num_features;
    const int64_t bucket_feature = (node * local_size + target) * num_features + feature;
    out.row_shift[bucket_feature] =
        ends[index] - ends[target * num_features] + before[index] - at(me, bucket_feature);
  }
  // A row of an owner (n, t) of another node is in t's remote part: after the keys of the
  // requesters before this rank, this rank's blocks for the owner nodes before n, and the rows
  // of its block before it. Every feature of the owner has the same row_shift.
  cta_exclusive_ends<int64_t>(
      topo.world_size,
      [&](int64_t index) {
        const int64_t target = index / num_nodes, owner_node = index % num_nodes;
        return owner_node == node ? 0 : bucket_keys(me, owner_node * local_size + target);
      },
      block_ends);
  for (int64_t index = threadIdx.x; index < topo.world_size * num_features; index += kThreads) {
    const int64_t owner = index / num_features, owner_node = owner / local_size,
                  target = owner % local_size;
    if (owner_node == node) continue;
    int64_t block_place =
        lookup_keys + block_ends[target * num_nodes + owner_node] - block_ends[target * num_nodes];
    for (int64_t requester = 0; requester < me; ++requester)
      block_place += remote[requester * local_size + target];
    out.row_shift[index] = block_place - at(me, owner * num_features);
  }
  // the rows past the valid ones (bucket feature world_size * num_features) are never pushed:
  // their place is their own index
  if (threadIdx.x == 0) out.row_shift[topo.world_size * num_features] = 0;
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
