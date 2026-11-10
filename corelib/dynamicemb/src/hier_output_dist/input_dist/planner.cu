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
// The planner's steps of the node, of the rail and the owner, and of the
// requester (planner.h).

#include <algorithm>
#include <cstdint>

#include "hier_output_dist/common/window.h"
#include "hier_output_dist/input_dist/node.cuh"
#include "hier_output_dist/input_dist/planner.cuh"
#include "hier_output_dist/input_dist/planner.h"
#include "hier_output_dist/input_dist/rail.cuh"
#include "hier_output_dist/input_dist/requester.cuh"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

// ── the node ──

// A thread per word of the node's rows (rows [local_size][NodeRow::words]): word w of requester
// r's row, from its offset_per_bucket_feature (node_offsets [local_size][world_size *
// num_features + 1]), as it pushes it (NodeRow::word). The hooks' (node_layout).
__global__ void __launch_bounds__(kThreads)
    node_rows_of_kernel(const int64_t *node_offsets, Topo topo, int64_t num_features, NodeRow row,
                        int64_t *rows) {
  const int64_t full = (int64_t)topo.world_size * num_features + 1;
  for (int64_t item = first_item(); item < topo.local_size * row.words; item += item_stride())
    rows[item] =
        row.word(node_offsets + item / row.words * full, topo, num_features, item % row.words);
}

// One CTA: the layout of the node's keys from the node's rows (node_layout_small, and
// node_layout_items when `items`; node.cuh), as node_exchange makes it; opbf: this rank's
// offset_per_bucket_feature. The hooks' (node_layout).
__global__ void __launch_bounds__(kThreads)
    node_layout_kernel(const int64_t *rows, NodeRow row, const int64_t *opbf, Topo topo,
                       int64_t num_features, int64_t lookup_keys, int64_t remote_keys, bool items,
                       int64_t *scratch, NodeLayoutView out) {
  const NodeRows in{rows, row};
  node_layout_small(in, opbf, topo, num_features, lookup_keys, remote_keys, scratch, out);
  if (!items) return;
  __syncthreads();
  node_layout_items(in, opbf, topo, num_features, scratch, threadIdx.x, blockDim.x, out);
}

// A thread per item: node_layout_items, as node_layout_items_kernel makes them. The hooks'.
__global__ void __launch_bounds__(kThreads)
    node_items_kernel(const int64_t *rows, NodeRow row, const int64_t *opbf, Topo topo,
                      int64_t num_features, const int64_t *scratch, NodeLayoutView out) {
  node_layout_items(NodeRows{rows, row}, opbf, topo, num_features, scratch, first_item(),
                    item_stride(), out);
}

// The push of node_key_push (push_rows, node.cuh) into a test's stand-ins for its targets'
// node_keys (row t: target t's; node_key_weights nullptr: none). The hooks' (push_to_node_keys).
// A place outside a stand-in's row is not stored.
__global__ void __launch_bounds__(kThreads)
    push_to_node_keys_kernel(GatherRows rows, int64_t num_rows, PushPlaces places, Topo topo,
                             int64_t num_features, int64_t lookup_keys, int64_t remote_keys,
                             int64_t *node_keys, float *node_key_weights,
                             int32_t *node_bucket_feature_of_row, int32_t *src) {
  const int64_t entries = lookup_keys + remote_keys;
  push_rows(
      rows, num_rows, places, topo, num_features, src,
      [&](int64_t target, int64_t place, int64_t key, float weight, int32_t node_bucket_feature) {
        if (place < 0 || place >= entries) return;
        node_keys[target * entries + place] = key;
        if (node_key_weights != nullptr) node_key_weights[target * entries + place] = weight;
        if (place >= lookup_keys)
          node_bucket_feature_of_row[target * remote_keys + place - lookup_keys] =
              node_bucket_feature;
      });
}

// A thread per entry of the own part: the KJT's copy of its first num_recv
// keys (and weights), at most own_part; none under skip (no requester pushed).
__global__ void __launch_bounds__(kThreads)
    own_part_keys_kernel(const int64_t *node_keys, const float *node_key_weights,
                         const int64_t *num_recv, const bool *skip, int64_t own_part,
                         int64_t *kjt_values, float *kjt_weights) {
  const int64_t keys = *skip ? 0 : min(*num_recv, own_part);
  for (int64_t position = first_item(); position < keys; position += item_stride()) {
    kjt_values[position] = node_keys[position];
    if (node_key_weights != nullptr) kjt_weights[position] = node_key_weights[position];
  }
}

}  // namespace

NodeLayout empty_node_layout(const Topo &topo, int64_t num_features,
                             const at::TensorOptions &options) {
  const int64_t local_size = topo.local_size;
  const auto long_options = options.dtype(at::kLong);
  NodeLayout node;
  node.own_lengths = at::empty({num_features * local_size}, long_options);
  node.own_offsets = at::empty({num_features * local_size}, long_options);
  node.own_total = at::empty({local_size}, long_options);
  node.need_lookup = at::empty({}, long_options);
  node.skip = at::empty({}, long_options.dtype(at::kBool));
  node.row_shift = at::empty({topo.world_size * num_features + 1}, long_options);
  if (topo.num_nodes > 1) {  // one node has no remote part
    node.remote_offsets = at::empty({local_size + 1}, long_options);
    node.need_remote = at::empty({}, long_options);
  } else {  // its own part is the owner's KJT
    node.kjt_offsets = at::empty({num_features * local_size + 1}, long_options);
    node.length_per_key = at::empty({num_features}, long_options);
  }
  return node;
}

NodeLayout node_layout(const at::Tensor &node_offsets, const Topo &topo, int64_t num_features,
                       int64_t lookup_keys, int64_t remote_keys, c10::optional<bool> items_in_cta) {
  const int64_t full = (int64_t)topo.world_size * num_features + 1;
  TORCH_CHECK(is_cuda_array(node_offsets, at::kLong, topo.local_size * full),
              "hier_output_dist: the node's offsets, local_size x (world_size * num_features + 1)");
  const auto options = node_offsets.options();
  const cudaStream_t stream = current_stream();
  NodeLayout node = empty_node_layout(topo, num_features, options);
  const at::Tensor scratch = at::empty({node_layout_scratch(topo, num_features)}, options);
  // the rows the node's requesters push (NodeRow)
  const NodeRow row = NodeRow::of(topo, num_features);
  const at::Tensor rows = at::empty({topo.local_size, row.words}, options);
  node_rows_of_kernel<<<item_grid(topo.local_size * row.words), kThreads, 0, stream>>>(
      node_offsets.data_ptr<int64_t>(), topo, num_features, row, rows.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t *opbf = node_offsets.data_ptr<int64_t>() + topo.local_rank * full;
  const int64_t items = node_layout_items_of(topo, num_features);
  const bool lay_out = items_in_cta.value_or(items <= kNodeItemsInCta);
  node_layout_kernel<<<kOneCta, kThreads, 0, stream>>>(
      rows.data_ptr<int64_t>(), row, opbf, topo, num_features, lookup_keys, remote_keys, lay_out,
      scratch.data_ptr<int64_t>(), view_of(node));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (!lay_out) {
    node_items_kernel<<<item_grid(items), kThreads, 0, stream>>>(
        rows.data_ptr<int64_t>(), row, opbf, topo, num_features, scratch.data_ptr<int64_t>(),
        view_of(node));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return node;
}

void push_to_node_keys(const at::Tensor &values, const at::Tensor &inv, const at::Tensor &weights,
                       const at::Tensor &bucket_feature_of_row, const at::Tensor &offset_per_bucket,
                       const NodeLayout &node, const Topo &topo, int64_t num_features,
                       int64_t lookup_keys, int64_t remote_keys, const at::Tensor &node_keys,
                       const at::Tensor &node_key_weights,
                       const at::Tensor &node_bucket_feature_of_row, const at::Tensor &src) {
  const int64_t rows = inv.numel(), local_size = topo.local_size,
                entries = lookup_keys + remote_keys;
  TORCH_CHECK(lookup_keys >= 0 && remote_keys >= 0 && (topo.num_nodes > 1 || remote_keys == 0) &&
                  is_cuda_array(node_keys, at::kLong, local_size * entries) &&
                  weights.defined() == node_key_weights.defined() &&
                  (!weights.defined() ||
                   is_cuda_array(node_key_weights, at::kFloat, local_size * entries)) &&
                  (topo.num_nodes == 1 ||
                   is_cuda_array(node_bucket_feature_of_row, at::kInt, local_size * remote_keys)) &&
                  src.defined() == (topo.num_nodes == 1) &&
                  (!src.defined() || is_cuda_array(src, at::kInt, rows)),
              "hier_output_dist: the stand-ins for the targets' node_keys (local_size x "
              "(lookup_keys + remote_keys)), node_key_weights exactly with weights, across nodes "
              "node_bucket_feature_of_row (local_size x remote_keys), and on one node src");
  if (rows == 0) return;
  const GatherRows gather = gather_rows_of(inv, values, weights);
  const PushPlaces places{bucket_feature_of_row.data_ptr<int32_t>(),
                          offset_per_bucket.data_ptr<int64_t>(), node.row_shift.data_ptr<int64_t>(),
                          node.skip.data_ptr<bool>()};
  push_to_node_keys_kernel<<<item_grid(rows), kThreads, 0, current_stream()>>>(
      gather, rows, places, topo, num_features, lookup_keys, remote_keys,
      node_keys.data_ptr<int64_t>(), data_or_null<float>(node_key_weights),
      data_or_null<int32_t>(node_bucket_feature_of_row), data_or_null<int32_t>(src));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

OneNodeOwner one_node_owner(const NodeLayout &node, const at::Tensor &node_keys,
                            const at::Tensor &node_key_weights, const at::Tensor &src,
                            const Topo &topo) {
  const int64_t own_part = node_keys.numel();
  OneNodeOwner owner;
  // The own part is the KJT: a copy of its first num_recv keys (the call's
  // second wait narrows it to num_recv). The node's next call rewrites
  // node_keys after its barrier, which does not wait for this call's lookup.
  owner.num_recv = node.own_total.select(0, topo.local_rank);
  owner.kjt_values = at::empty({own_part}, node_keys.options());
  if (node_key_weights.defined())
    owner.kjt_weights = at::empty({own_part}, node_key_weights.options());
  own_part_keys_kernel<<<item_grid(own_part), kThreads, 0, current_stream()>>>(
      node_keys.data_ptr<int64_t>(), data_or_null<float>(node_key_weights),
      owner.num_recv.data_ptr<int64_t>(), node.skip.data_ptr<bool>(), own_part,
      owner.kjt_values.data_ptr<int64_t>(), data_or_null<float>(owner.kjt_weights));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  owner.kjt_lengths = node.own_lengths;
  owner.kjt_offsets = node.kjt_offsets;
  owner.length_per_key = node.length_per_key;
  owner.src = src;
  return owner;
}

// ── the rail and the owner ──

namespace {

// One CTA: a rank's count messages (count_messages_of, rail.cuh), as rail_exchange makes them
// (messages [num_nodes][CountMessage::length]; own: a scratch row). The hooks' (count_message).
__global__ void __launch_bounds__(kThreads)
    count_message_kernel(CountParts parts, CountMessage format, int64_t local_size, int64_t node,
                         int64_t *messages, int64_t *own) {
  count_messages_of(parts, format, local_size, node, messages, own);
}

// One CTA: the rail's layout (rail_layout_of, rail.cuh) from the rail's count messages (messages
// [num_nodes][CountMessage::length]), as rail_exchange makes it. The hooks' (rail_layout).
__global__ void __launch_bounds__(kThreads)
    rail_layout_kernel(const int64_t *messages, CountMessage format, const int64_t *own_lengths,
                       Sizes sizes, int64_t *scratch, RailLayoutView out) {
  rail_layout_of(RailCounts{messages, format.length(), format}, own_lengths, sizes, scratch, out);
}

// What owner_kjt_kernel writes of the chunks the owner's forward puts (OwnerTables::fwd_chunks,
// [max_owner_chunks], and where each source node's start there, fwd_chunk_offsets), from the
// rail's slice_in and the chunks first in each source node's slice of rail_keys (its bytes
// rail_bytes).
struct ForwardChunks {
  const int64_t *slice_in;
  const char *rail_keys;
  int64_t rail_bytes;
  int64_t *fwd_chunk_offsets;
  OwnerChunk *fwd_chunks;
};
// The requester's tables whose remote task offsets owner_kjt_kernel completes (RequesterTables'
// proxy_num_chunks and remote_task_offsets); remote_task_offsets nullptr: none.
struct RemoteTaskTail {
  const int64_t *proxy_num_chunks;
  int64_t *remote_task_offsets;
};

// Every CTA, a thread per entry over the grid: the entries of the requester's remote task offsets
// between local_size times its proxies' most chunks (maxC) and local_size * max_chunks get the
// total, which the requester kernel wrote at both. Past maxC no proxy has a chunk, so each holds
// the total, as a scan over all local_size * max_chunks entries leaves it, and a search over them
// (TaskPlace) finds what it finds today.
__device__ __forceinline__ void remote_task_tail(const RemoteTaskTail &tail, int64_t local_size,
                                                 int64_t max_chunks) {
  const int64_t most = cta_max<int64_t>(
      local_size, [&](int64_t proxy) { return tail.proxy_num_chunks[proxy]; }, 0);
  const int64_t entries = local_size * max_chunks, total = tail.remote_task_offsets[entries];
  for (int64_t entry = local_size * most + 1 + first_item(); entry < entries;
       entry += item_stride())
    tail.remote_task_offsets[entry] = total;
}

// A thread per entry: the requester's remote task offsets past its proxies' chunks
// (remote_task_tail), which owner_kjt_kernel writes in the in channel. The hooks'
// (requester_tables).
__global__ void __launch_bounds__(kThreads)
    remote_task_tail_kernel(RemoteTaskTail tail, int64_t local_size, int64_t max_chunks) {
  remote_task_tail(tail, local_size, max_chunks);
}

// What owner_kjt_kernel reads of the rail layout to place each key's row
// (OwnerTables::row_map): the lookup group of each feature, the group runs of
// each source node's records (fwd_groups), the group shifts and where each
// group's slice starts in lookup_rows. row_map nullptr: no row map.
struct RowMapTables {
  const int32_t *group_of_feature;
  GroupRuns fwd_groups;
  const int64_t *group_shift, *group_offsets;
  int32_t *row_map;
};

// Every CTA (shared memory: num_nodes + 1 int64 words): where each source node's chunks start in
// fwd_chunks, from the rail's slice_in (its own node sends none; at most max_owner_chunks in
// all), which CTA 0 writes to fwd_chunk_offsets; then a thread per entry of fwd_chunks
// (max_owner_chunks of them): the chunk at that place, its source node's (the last whose chunks
// start at or before it), read from its slice of rail_keys (a slice starts at a multiple of 8
// bytes); zeros past the chunks.
// Then a thread per position of the owner's KJT, up to the keys it looks up (at most
// lookup_keys): the (feature, column) of the KJT's lengths it is in and its
// place there. A key of its own node is in the own part of node_keys, another
// node's in its Record in rail_keys (rail_records of them). With weighted
// records the KJT carries weights: an own key's from node_key_weights. With a
// row map, also the key's row in its group's slice of lookup_rows: an own key
// at its place in the own part plus its group's shift, another node's record
// (its record among the node's, from records_first) plus its group run's
// shift, less where the slice starts.
// Last, with `tail`, the requester's remote task offsets past its proxies' chunks
// (remote_task_tail).
template <typename Record>
__global__ void __launch_bounds__(kThreads)
    owner_kjt_kernel(ForwardChunks chunks, const int64_t *kjt_offsets, const int64_t *num_recv,
                     const int64_t *own_offsets, const int64_t *records_first,
                     const int64_t *record_offsets, const int64_t *node_keys,
                     const float *node_key_weights, const Record *rail_records, int64_t rail_count,
                     const int64_t *other_nodes, RowMapTables rows, RemoteTaskTail tail,
                     Sizes sizes, int64_t *kjt_values, float *kjt_weights) {
  extern __shared__ int64_t chunk_offsets[];  // [num_nodes + 1]: fwd_chunk_offsets
  const int64_t local_size = sizes.topo.local_size, kjt_stride = sizes.topo.kjt_stride,
                num_nodes = sizes.topo.num_nodes, other_columns = num_nodes - 1,
                lookup_keys = sizes.lookup_keys, num_lengths = sizes.num_features * kjt_stride,
                bound = sizes.max_owner_chunks();
  cta_offsets_of<int64_t>(
      num_nodes, [&](int64_t source) { return chunks.slice_in[2 * source + 1]; }, chunk_offsets);
  for (int64_t source = threadIdx.x; source <= num_nodes; source += kThreads)
    chunk_offsets[source] = min(chunk_offsets[source], bound);
  __syncthreads();
  if (blockIdx.x == 0)
    for (int64_t source = threadIdx.x; source <= num_nodes; source += kThreads)
      chunks.fwd_chunk_offsets[source] = chunk_offsets[source];
  const int64_t count = chunk_offsets[num_nodes];
  constexpr int64_t kOwnerChunk = sizeof(OwnerChunk);
  for (int64_t index = first_item(); index < bound; index += item_stride()) {
    OwnerChunk owner_chunk{0, 0, 0, 0};
    if (index < count) {
      const int64_t source =
          clamp_index(upper_bound(chunk_offsets + 1, num_nodes, index), num_nodes);
      const int64_t byte =
          chunks.slice_in[2 * source] + (index - chunk_offsets[source]) * kOwnerChunk;
      owner_chunk = *(const OwnerChunk *)(chunks.rail_keys + min(max(byte, (int64_t)0),
                                                                 chunks.rail_bytes - kOwnerChunk));
    }
    chunks.fwd_chunks[index] = owner_chunk;
  }
  const int64_t keys = min(*num_recv, lookup_keys);
  for (int64_t position = first_item(); position < keys; position += item_stride()) {
    const int64_t index =
        clamp_index(upper_bound(kjt_offsets, num_lengths + 1, position) - 1, num_lengths);
    const int64_t feature = index / kjt_stride, column = index % kjt_stride,
                  nth = position - kjt_offsets[index];
    int64_t row = 0;  // the key's row in lookup_rows
    const int64_t group = rows.row_map != nullptr ? rows.group_of_feature[feature] : 0;
    if (column < local_size) {
      const int64_t place = own_offsets[feature * local_size + column] + nth;
      const int64_t own = clamp_index(place, lookup_keys);
      kjt_values[position] = node_keys[own];
      if constexpr (Record::kWeighted) kjt_weights[position] = node_key_weights[own];
      if (rows.row_map != nullptr) row = place + rows.group_shift[group];
    } else {
      const int64_t other = column - local_size;
      const int64_t record = record_offsets[feature * other_columns + other] + nth;
      const Record &found = rail_records[clamp_index(record, rail_count)];
      kjt_values[position] = found.key;
      if constexpr (Record::kWeighted) kjt_weights[position] = found.weight;
      if (rows.row_map != nullptr)
        row = record - records_first[other] + rows.fwd_groups.shift(other_nodes[other])[group];
    }
    if (rows.row_map != nullptr)
      rows.row_map[position] = (int32_t)(row - rows.group_offsets[group]);
  }
  if (tail.remote_task_offsets != nullptr) remote_task_tail(tail, local_size, sizes.max_chunks);
}

}  // namespace

at::Tensor other_nodes_of(const Topo &topo, const at::TensorOptions &options) {
  const at::Tensor nodes = at::arange(topo.num_nodes - 1, options.dtype(at::kLong));
  return nodes + nodes.ge(topo.node);
}

at::Tensor count_message(const at::Tensor &need_remote, const at::Tensor &own_offsets,
                         const at::Tensor &own_total, const at::Tensor &need_key_rows,
                         const at::Tensor &chunk_length_per_bucket,
                         const at::Tensor &node_length_per_bucket_feature,
                         const at::Tensor &node_offset_per_bucket_feature,
                         const at::Tensor &group_starts) {
  const int64_t num_nodes = chunk_length_per_bucket.numel(),
                num_features = node_length_per_bucket_feature.numel() / num_nodes;
  const CountMessage format{num_nodes, num_features, group_starts.numel() - 1};
  const at::Tensor messages =
      at::empty({num_nodes, format.slot_words()}, chunk_length_per_bucket.options());
  const at::Tensor own = at::empty({format.length()}, chunk_length_per_bucket.options());
  const CountParts parts{need_remote.data_ptr<int64_t>(),
                         own_offsets.data_ptr<int64_t>(),
                         own_total.data_ptr<int64_t>(),
                         need_key_rows.data_ptr<int64_t>(),
                         chunk_length_per_bucket.data_ptr<int64_t>(),
                         node_length_per_bucket_feature.data_ptr<int64_t>(),
                         node_offset_per_bucket_feature.data_ptr<int64_t>(),
                         group_starts.data_ptr<int64_t>()};
  count_message_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      parts, format, own_offsets.numel() / num_features, 0, messages.data_ptr<int64_t>(),
      own.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return messages.narrow(1, 0, format.length()).contiguous();
}

RailLayout empty_rail_layout(const Sizes &sizes, const at::TensorOptions &options) {
  const int64_t num_nodes = sizes.topo.num_nodes, num_features = sizes.num_features,
                num_groups = sizes.num_groups, num_lengths = num_features * sizes.topo.kjt_stride,
                runs = GroupRuns::words_per_node(num_groups);
  const auto long_options = options.dtype(at::kLong);
  RailLayout rail;
  rail.put_dst = at::empty({num_nodes}, long_options);
  rail.put_bytes = at::empty({num_nodes}, long_options);
  rail.put_skip = at::empty({num_nodes}, long_options.dtype(at::kBool));
  rail.bwd_groups = at::empty({num_nodes, runs}, long_options);
  rail.kjt_lengths = at::empty({num_lengths}, long_options);
  rail.kjt_offsets = at::empty({num_lengths + 1}, long_options);
  rail.num_recv = at::empty({}, long_options);
  rail.length_per_key = at::empty({num_features}, long_options);
  rail.slice_in = at::empty({num_nodes, 2}, long_options);
  rail.node_length_per_source = at::empty({num_nodes}, long_options);
  rail.node_length_per_source_feature = at::empty({num_nodes, num_features}, long_options);
  rail.fwd_groups = at::empty({num_nodes, runs}, long_options);
  rail.group_offsets = at::empty({num_groups + 1}, long_options);
  rail.group_shift = at::empty({num_groups}, long_options);
  rail.needs = at::empty({kNumNeeds}, long_options);
  return rail;
}

RailLayout rail_layout(const at::Tensor &messages, const at::Tensor &own_lengths,
                       const Sizes &sizes) {
  const auto options = messages.options();
  RailLayout rail = empty_rail_layout(sizes, options);
  const at::Tensor scratch =
      at::empty({rail_layout_scratch(sizes.topo, sizes.num_groups)}, options);
  const CountMessage format{sizes.topo.num_nodes, sizes.num_features, sizes.num_groups};
  rail_layout_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      messages.data_ptr<int64_t>(), format, own_lengths.data_ptr<int64_t>(), sizes,
      scratch.data_ptr<int64_t>(), view_of(rail));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return rail;
}

OwnerTables owner_tables(const RailLayout &rail, const OwnerRecords &records,
                         const RequesterTables &requester, const at::Tensor &own_offsets,
                         const at::Tensor &node_keys, const at::Tensor &node_key_weights,
                         const at::Tensor &rail_keys, const at::Tensor &other_nodes,
                         const at::Tensor &group_of_feature, const Sizes &sizes) {
  const int64_t num_nodes = sizes.topo.num_nodes, lookup_keys = sizes.lookup_keys,
                rail_words = rail_keys.numel();
  const auto long_options = rail_keys.options().dtype(at::kLong);
  OwnerTables tables;
  // Each source node's slice in rail_keys starts with the chunks of this
  // node in its stream, then its records.
  const int64_t bound = sizes.max_owner_chunks();
  tables.fwd_chunk_offsets = at::empty({num_nodes + 1}, long_options);
  tables.fwd_chunks = at::empty({bound, 4}, long_options.dtype(at::kInt));
  const ForwardChunks chunks{rail.slice_in.data_ptr<int64_t>(),
                             (const char *)rail_keys.data_ptr<int64_t>(), 8 * rail_words,
                             tables.fwd_chunk_offsets.data_ptr<int64_t>(),
                             records_of<OwnerChunk>(tables.fwd_chunks)};
  // The KJT's keys, its first num_recv positions: its own node's from
  // the own part of node_keys, the other nodes' from their records in
  // rail_keys. A group of one feature takes its keys' rows in KJT order, so
  // the row map is needed only when a group has more features.
  tables.kjt_values = at::empty({lookup_keys}, long_options);
  if (sizes.weighted_records())
    tables.kjt_weights = at::empty({lookup_keys}, long_options.dtype(at::kFloat));
  if (sizes.num_groups < sizes.num_features)
    tables.row_map = at::empty({lookup_keys}, long_options.dtype(at::kInt));
  const RowMapTables rows{group_of_feature.data_ptr<int32_t>(),
                          GroupRuns{rail.fwd_groups.data_ptr<int64_t>(), sizes.num_groups},
                          rail.group_shift.data_ptr<int64_t>(),
                          rail.group_offsets.data_ptr<int64_t>(),
                          data_or_null<int32_t>(tables.row_map)};
  const RemoteTaskTail tail{data_or_null<int64_t>(requester.proxy_num_chunks),
                            data_or_null<int64_t>(requester.remote_task_offsets)};
  with_record(sizes, [&](auto record) {
    using Record = decltype(record);
    owner_kjt_kernel<Record><<<item_grid(std::max(lookup_keys, bound)), kThreads,
                               8 * (num_nodes + 1), current_stream()>>>(
        chunks, rail.kjt_offsets.data_ptr<int64_t>(), rail.num_recv.data_ptr<int64_t>(),
        own_offsets.data_ptr<int64_t>(), records.records_first.data_ptr<int64_t>(),
        records.record_offsets.data_ptr<int64_t>(), node_keys.data_ptr<int64_t>(),
        data_or_null<float>(node_key_weights), (const Record *)rail_keys.data_ptr<int64_t>(),
        8 * rail_words / (int64_t)sizeof(Record), other_nodes.data_ptr<int64_t>(), rows, tail,
        sizes, tables.kjt_values.data_ptr<int64_t>(), data_or_null<float>(tables.kjt_weights));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tables;
}

// ── the requester ──

namespace {

// A test's stand-ins for the buffers of its node's proxies, as the requester kernel reads them
// (requester.cuh): proxy p's header is row p of `headers` (header_words each, ProxyHeader), its
// remote_map row p of `remote_maps` (map_words each, RemoteMap). The hooks' (requester_tables,
// owner_records).
struct StandInPeers {
  const int32_t *headers, *remote_maps;
  int64_t header_words, map_words;
  __device__ __forceinline__ const int32_t *header(int peer) const {
    return headers + peer * header_words;
  }
  __device__ __forceinline__ const int32_t *remote_map(int peer) const {
    return remote_maps + peer * map_words;
  }
};

// The test hook's row CTAs of the requester kernel (force_requester_ctas): 0, as their sizes
// choose.
int64_t g_requester_ctas = 0;

}  // namespace

void force_requester_ctas(int64_t ctas) {
  TORCH_CHECK(ctas >= 0 && ctas <= INT32_MAX / 2,
              "hier_output_dist: the requester kernel's row CTAs: 0 (as their sizes choose) or a "
              "grid of at most 2^30 - 1");
  g_requester_ctas = ctas;
}

int requester_row_ctas(int64_t rows, int64_t other_ctas) {
  if (g_requester_ctas > 0) return (int)g_requester_ctas;
  if (rows == 0) return 0;
  // the CTAs the GPU holds at once, two per SM, less the other roles' (at least one per SM)
  const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return (int)std::min<int64_t>(item_grid(rows), std::max(2 * sms - other_ctas, sms));
}

RequesterTables requester_tables(const at::Tensor &remote_maps, const at::Tensor &proxy_headers,
                                 const at::Tensor &owner_group_shift,
                                 const at::Tensor &bucket_feature_of_row,
                                 const at::Tensor &row_shift, const at::Tensor &group_of_feature,
                                 const at::Tensor &offset_per_bucket, const Sizes &sizes) {
  const int64_t local_size = sizes.topo.local_size, max_chunks = sizes.max_chunks;
  const StandInPeers peers{proxy_headers.data_ptr<int32_t>(), remote_maps.data_ptr<int32_t>(),
                           ProxyHeader::of(sizes).words(), RemoteMap::kArrays * sizes.remote_keys};
  const RequesterTables tables =
      requester_tables_of(peers, bucket_feature_of_row, offset_per_bucket, row_shift,
                          owner_group_shift, group_of_feature, nullptr, at::Tensor(), sizes)
          .first;
  // The entries of the remote task offsets that owner_tables writes in the in channel.
  const RemoteTaskTail tail{tables.proxy_num_chunks.data_ptr<int64_t>(),
                            tables.remote_task_offsets.data_ptr<int64_t>()};
  remote_task_tail_kernel<<<item_grid(local_size * max_chunks), kThreads, 0, current_stream()>>>(
      tail, local_size, max_chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tables;
}

std::pair<RequesterTables, OwnerRecords> owner_records(const RailLayout &rail,
                                                       const at::Tensor &other_nodes,
                                                       const at::Tensor &group_of_feature,
                                                       const Sizes &sizes) {
  const int64_t local_size = sizes.topo.local_size, header_words = ProxyHeader::of(sizes).words();
  const auto int_options = other_nodes.options().dtype(at::kInt);
  const auto long_options = other_nodes.options().dtype(at::kLong);
  // proxies whose headers hold 0 (no chunks); no row reads a remote_map
  const at::Tensor headers = at::zeros({local_size, header_words}, int_options);
  const StandInPeers peers{headers.data_ptr<int32_t>(), nullptr, header_words, 0};
  return requester_tables_of(
      peers, at::empty({0}, int_options), at::zeros({sizes.topo.world_size + 1}, long_options),
      at::zeros({1}, long_options), at::zeros({local_size * sizes.num_groups}, long_options),
      group_of_feature, &rail, other_nodes, sizes);
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
