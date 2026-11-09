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

#include <cstdint>

#include "hier_output_dist/common/window.h"
#include "hier_output_dist/input_dist/node.cuh"
#include "hier_output_dist/input_dist/planner.cuh"
#include "hier_output_dist/input_dist/planner.h"
#include "hier_output_dist/input_dist/rail.cuh"

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

// num_nodes CTAs. CTA c >= 1 takes other node c - 1: where its records start
// in rail_keys (records_first [other node]) and its records of each feature
// (record_offsets [feature][other node]), counted in records of record_bytes
// (a slice and its chunks are whole records): after its chunks, and its
// records of the features before (a scan of its keys per feature). CTA 0:
// where each source node's chunks start in fwd_chunks, from the rail's
// slice_in (its own node sends none; at most max_owner_chunks in all); then
// a thread per entry of fwd_chunks (max_owner_chunks of them): the chunk at
// that place, its source node's (the last whose chunks start at or before
// it), read from its slice of rail_keys (its bytes rail_bytes; a slice starts
// at a multiple of 8 bytes); zeros past the chunks.
__global__ void __launch_bounds__(kThreads)
    owner_slices_kernel(const int64_t *slice_in, const int64_t *node_length_per_source_feature,
                        const int64_t *other_nodes, const char *rail_keys, int64_t rail_bytes,
                        Sizes sizes, int64_t *fwd_chunk_offsets, int64_t *records_first,
                        int64_t *record_offsets, OwnerChunk *fwd_chunks) {
  const int64_t num_nodes = sizes.topo.num_nodes, num_features = sizes.num_features,
                bound = sizes.max_owner_chunks();
  if (blockIdx.x > 0) {
    const int64_t other = blockIdx.x - 1, source = other_nodes[other];
    const int64_t first =
        (slice_in[2 * source] + (int64_t)sizeof(OwnerChunk) * slice_in[2 * source + 1]) /
        sizes.record_bytes;
    if (threadIdx.x == 0) records_first[other] = first;
    cta_scan_blocked<kScanItems, int64_t>(
        num_features,
        [&](int64_t feature) {
          return node_length_per_source_feature[source * num_features + feature];
        },
        [&](int64_t feature, int64_t offset) {
          record_offsets[feature * (num_nodes - 1) + other] = first + offset;
        });
    return;
  }
  cta_offsets_of<int64_t>(
      num_nodes, [&](int64_t source) { return slice_in[2 * source + 1]; }, fwd_chunk_offsets);
  for (int64_t source = threadIdx.x; source <= num_nodes; source += kThreads)
    fwd_chunk_offsets[source] = min(fwd_chunk_offsets[source], bound);
  __syncthreads();
  const int64_t count = fwd_chunk_offsets[num_nodes];
  constexpr int64_t kOwnerChunk = sizeof(OwnerChunk);
  for (int64_t index = threadIdx.x; index < bound; index += kThreads) {
    OwnerChunk owner_chunk{0, 0, 0, 0};
    if (index < count) {
      const int64_t source =
          clamp_index(upper_bound(fwd_chunk_offsets + 1, num_nodes, index), num_nodes);
      const int64_t byte = slice_in[2 * source] + (index - fwd_chunk_offsets[source]) * kOwnerChunk;
      owner_chunk =
          *(const OwnerChunk *)(rail_keys + min(max(byte, (int64_t)0), rail_bytes - kOwnerChunk));
    }
    fwd_chunks[index] = owner_chunk;
  }
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

// A thread per position of the owner's KJT, up to the keys it looks up (at most
// lookup_keys): the (feature, column) of the KJT's lengths it is in and its
// place there. A key of its own node is in the own part of node_keys, another
// node's in its Record in rail_keys (rail_records of them). With weighted
// records the KJT carries weights: an own key's from node_key_weights. With a
// row map, also the key's row in its group's slice of lookup_rows: an own key
// at its place in the own part plus its group's shift, another node's record
// (its record among the node's, from records_first) plus its group run's
// shift, less where the slice starts.
template <typename Record>
__global__ void __launch_bounds__(kThreads)
    owner_kjt_kernel(const int64_t *kjt_offsets, const int64_t *num_recv,
                     const int64_t *own_offsets, const int64_t *records_first,
                     const int64_t *record_offsets, const int64_t *node_keys,
                     const float *node_key_weights, const Record *rail_records, int64_t rail_count,
                     const int64_t *other_nodes, RowMapTables rows, Sizes sizes,
                     int64_t *kjt_values, float *kjt_weights) {
  const int64_t local_size = sizes.topo.local_size, kjt_stride = sizes.topo.kjt_stride,
                other_columns = sizes.topo.num_nodes - 1, lookup_keys = sizes.lookup_keys,
                num_lengths = sizes.num_features * kjt_stride;
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
}

}  // namespace

at::Tensor other_nodes_of(const Topo &topo, const at::TensorOptions &options) {
  const at::Tensor nodes = at::arange(topo.num_nodes - 1, options.dtype(at::kLong));
  return nodes + nodes.ge(topo.node);
}

at::Tensor count_message(const at::Tensor &need_remote, const at::Tensor &own_lengths,
                         const at::Tensor &need_key_rows, const at::Tensor &chunk_length_per_bucket,
                         const at::Tensor &node_length_per_bucket_feature,
                         const at::Tensor &group_starts) {
  const int64_t num_nodes = chunk_length_per_bucket.numel(),
                num_features = node_length_per_bucket_feature.numel() / num_nodes;
  const CountMessage format{num_nodes, num_features, group_starts.numel() - 1};
  const at::Tensor messages =
      at::empty({num_nodes, format.slot_words()}, chunk_length_per_bucket.options());
  const at::Tensor own = at::empty({format.length()}, chunk_length_per_bucket.options());
  const CountParts parts{need_remote.data_ptr<int64_t>(),
                         own_lengths.data_ptr<int64_t>(),
                         need_key_rows.data_ptr<int64_t>(),
                         chunk_length_per_bucket.data_ptr<int64_t>(),
                         node_length_per_bucket_feature.data_ptr<int64_t>(),
                         group_starts.data_ptr<int64_t>()};
  count_message_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      parts, format, own_lengths.numel() / num_features, 0, messages.data_ptr<int64_t>(),
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

OwnerTables owner_tables(const RailLayout &rail, const at::Tensor &own_offsets,
                         const at::Tensor &node_keys, const at::Tensor &node_key_weights,
                         const at::Tensor &rail_keys, const at::Tensor &other_nodes,
                         const at::Tensor &group_of_feature, const Sizes &sizes) {
  const int64_t num_nodes = sizes.topo.num_nodes, lookup_keys = sizes.lookup_keys,
                rail_words = rail_keys.numel();
  const auto long_options = rail_keys.options().dtype(at::kLong);
  const cudaStream_t stream = current_stream();
  OwnerTables tables;
  // Each source node's slice in rail_keys starts with the chunks of this
  // node in its stream, then its records.
  const int64_t bound = sizes.max_owner_chunks();
  tables.fwd_chunk_offsets = at::empty({num_nodes + 1}, long_options);
  tables.fwd_chunks = at::empty({bound, 4}, long_options.dtype(at::kInt));
  const at::Tensor records_first = at::empty({num_nodes - 1}, long_options);
  const at::Tensor record_offsets = at::empty({sizes.num_features * (num_nodes - 1)}, long_options);
  owner_slices_kernel<<<num_nodes, kThreads, 0, stream>>>(
      rail.slice_in.data_ptr<int64_t>(), rail.node_length_per_source_feature.data_ptr<int64_t>(),
      other_nodes.data_ptr<int64_t>(), (const char *)rail_keys.data_ptr<int64_t>(), 8 * rail_words,
      sizes, tables.fwd_chunk_offsets.data_ptr<int64_t>(), records_first.data_ptr<int64_t>(),
      record_offsets.data_ptr<int64_t>(), records_of<OwnerChunk>(tables.fwd_chunks));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  with_record(sizes, [&](auto record) {
    using Record = decltype(record);
    owner_kjt_kernel<Record><<<item_grid(lookup_keys), kThreads, 0, stream>>>(
        rail.kjt_offsets.data_ptr<int64_t>(), rail.num_recv.data_ptr<int64_t>(),
        own_offsets.data_ptr<int64_t>(), records_first.data_ptr<int64_t>(),
        record_offsets.data_ptr<int64_t>(), node_keys.data_ptr<int64_t>(),
        data_or_null<float>(node_key_weights), (const Record *)rail_keys.data_ptr<int64_t>(),
        8 * rail_words / (int64_t)sizeof(Record), other_nodes.data_ptr<int64_t>(), rows, sizes,
        tables.kjt_values.data_ptr<int64_t>(), data_or_null<float>(tables.kjt_weights));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tables;
}

// ── the requester ──

namespace {

// A thread per bucketized row: the node peer whose remote_map has its words,
// for a valid row of another node's owner (its proxy: the owner's local
// rank; else -1), and its occurrence there: its place, row + row_shift,
// less lookup_keys, clamped into the remote part.
__global__ void __launch_bounds__(kThreads)
    remote_map_gather_kernel(const int32_t *bucket_feature_of_row, const int64_t *offset_per_bucket,
                             const int64_t *row_shift, int64_t rows, Sizes sizes, int32_t *peer,
                             int64_t *first_word) {
  const int64_t local_size = sizes.topo.local_size, world_size = sizes.topo.world_size;
  const int64_t valid_rows = offset_per_bucket[world_size];
  for (int64_t row = first_item(); row < rows; row += item_stride()) {
    const int64_t bucket_feature = bucket_feature_of_row[row];
    const int64_t owner = min(bucket_feature / sizes.num_features, world_size - 1);
    const bool remote = owner / local_size != sizes.topo.node && row < valid_rows;
    peer[row] = remote ? (int32_t)(owner % local_size) : -1;
    first_word[row] =
        clamp_index(row_shift[bucket_feature] + row - sizes.lookup_keys, sizes.remote_keys);
  }
}

// One CTA: this requester's tables of its node's proxies, from their
// gathered headers (headers [local_size][ProxyHeader::gathered_words]): where
// its rows at each proxy start in its push order (push_order_off: its rows of
// the owners with the proxy's local rank on the other nodes), each proxy's
// chunks and chunk ends, its push tables there made absolute (positions in
// push_order: its rows at the proxies before come first), and the remote
// task offsets.
__global__ void __launch_bounds__(kThreads)
    requester_proxy_tables_kernel(const int32_t *headers, const int64_t *offset_per_bucket,
                                  Sizes sizes, int64_t *push_order_off, int64_t *proxy_num_chunks,
                                  int64_t *proxy_ends, int32_t *proxy_push_tables,
                                  int64_t *remote_task_offsets) {
  const ProxyHeader header = ProxyHeader::of(sizes);
  const int64_t local_size = sizes.topo.local_size, me = sizes.topo.local_rank,
                max_chunks = sizes.max_chunks, entries = max_chunks + 1;
  // its push table as the proxy published it: positions relative to its push
  // order there
  auto published = [&](int64_t proxy) { return header.gathered(headers, proxy).push_table(); };
  auto chunks_of = [&](int64_t proxy) {
    return min(max(header.gathered(headers, proxy).num_chunks(), (int64_t)0), max_chunks);
  };
  auto rows_at = [&](int64_t proxy) {
    int64_t rows = 0;
    for (int64_t owner_node = 0; owner_node < sizes.topo.num_nodes; ++owner_node) {
      const int64_t owner = owner_node * local_size + proxy;
      if (owner_node != sizes.topo.node)
        rows += offset_per_bucket[owner + 1] - offset_per_bucket[owner];
    }
    return rows;
  };
  cta_offsets_of<int64_t>(local_size, rows_at, push_order_off);
  for (int64_t proxy = threadIdx.x; proxy < local_size; proxy += kThreads)
    proxy_num_chunks[proxy] = chunks_of(proxy);
  // each chunk's end, then the stream's (uint32 words)
  for (int64_t index = threadIdx.x; index < local_size * entries; index += kThreads) {
    const int64_t proxy = index / entries, chunk = index % entries;
    proxy_ends[index] = header.gathered(headers, proxy).end(chunk);
  }
  // a thread per entry of a proxy's table
  for (int64_t index = threadIdx.x; index < local_size * entries; index += kThreads) {
    const int64_t proxy = index / entries;
    proxy_push_tables[index] = (int32_t)(published(proxy)[index % entries] + push_order_off[proxy]);
  }
  // The remote task offsets, chunk after chunk: in a chunk the proxies in
  // turn from this rank's own one, kTaskRows positions of its push order a
  // task; a proxy has none past its chunks. (The push tables' differences are
  // the same relative or absolute.)
  cta_offsets_of<int64_t>(
      max_chunks * local_size,
      [&](int64_t index) -> int64_t {
        const int64_t chunk = index / local_size, proxy = (me + index % local_size) % local_size;
        if (chunk >= chunks_of(proxy)) return 0;
        const int32_t *table = published(proxy);
        return ceil_div<kTaskRows>((int64_t)table[chunk + 1] - table[chunk]);
      },
      remote_task_offsets);
}

// A thread per bucketized row. A row of an owner on this node is in the owner's
// lookup_rows at its place in the owner's own part (where this requester
// pushed it, row + row_shift) plus the shift of its feature's lookup group
// there (owner_group_shift [owner][group]), both forward and backward.
// Another node's row reads its key's row in proxy_rows and pushes its
// gradient to its slot, from what it gathered of its proxy's remote_map (map);
// a valid one is at its push position at the proxy in its push order.
__global__ void __launch_bounds__(kThreads)
    requester_rows_kernel(GatheredMap map, const int64_t *owner_group_shift,
                          const int32_t *bucket_feature_of_row, const int64_t *row_shift,
                          const int32_t *group_of_feature, const int64_t *offset_per_bucket,
                          const int64_t *push_order_off, int64_t rows, Sizes sizes, int32_t *src,
                          int32_t *dst, int32_t *push_order) {
  const int64_t local_size = sizes.topo.local_size, num_features = sizes.num_features,
                world_size = sizes.topo.world_size;
  const int64_t valid_rows = offset_per_bucket[world_size];
  for (int64_t row = first_item(); row < rows; row += item_stride()) {
    const int64_t bucket_feature = bucket_feature_of_row[row];
    const int64_t owner = min(bucket_feature / num_features, world_size - 1);
    const int64_t peer = owner % local_size;  // the owner itself on this node, else its proxy
    if (owner / local_size == sizes.topo.node) {
      const int64_t group = group_of_feature[bucket_feature % num_features];
      const int32_t place = (int32_t)(row + row_shift[bucket_feature] +
                                      owner_group_shift[peer * sizes.num_groups + group]);
      src[row] = place;
      dst[row] = place;
      continue;
    }
    src[row] = map.key_row(row);
    dst[row] = map.slot(row);
    // a call that does not fit may read words no one wrote: a position out
    // of range is dropped
    const int64_t position = push_order_off[peer] + map.position(row);
    if (row < valid_rows && position >= 0 && position < rows) push_order[position] = (int32_t)row;
  }
}

}  // namespace

PeerWords remote_map_gather(const at::Tensor &bucket_feature_of_row,
                            const at::Tensor &offset_per_bucket, const at::Tensor &row_shift,
                            const Sizes &sizes) {
  const int64_t rows = bucket_feature_of_row.numel();
  PeerWords gather;
  gather.peer = at::empty({rows}, bucket_feature_of_row.options());
  gather.first_word = at::empty({rows}, offset_per_bucket.options());
  if (rows > 0) {
    remote_map_gather_kernel<<<item_grid(rows), kThreads, 0, current_stream()>>>(
        bucket_feature_of_row.data_ptr<int32_t>(), offset_per_bucket.data_ptr<int64_t>(),
        row_shift.data_ptr<int64_t>(), rows, sizes, gather.peer.data_ptr<int32_t>(),
        gather.first_word.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return gather;
}

PeerWords header_gather(const Sizes &sizes, const at::TensorOptions &options) {
  const ProxyHeader header = ProxyHeader::of(sizes);
  const int64_t local_size = sizes.topo.local_size, words = header.gathered_words(),
                push_table = header.push_table_word(sizes.topo.local_rank);
  const auto long_options = options.dtype(at::kLong);
  // of each proxy's header, in the order of its gathered words: its chunks,
  // its end, each chunk's end, then its push tables for this requester
  const at::Tensor first_word = at::cat(
      {at::arange(ProxyHeader::kNumChunksWord, ProxyHeader::kEndWord + 1, long_options),
       at::arange(ProxyHeader::kEndsWord, ProxyHeader::kEndsWord + header.max_chunks, long_options),
       at::arange(push_table, push_table + header.push_words(), long_options)});
  PeerWords gather;
  gather.peer = at::arange(local_size, options.dtype(at::kInt))
                    .view({local_size, 1})
                    .expand({local_size, words})
                    .reshape({-1});
  gather.first_word = first_word.view({1, words}).expand({local_size, words}).reshape({-1});
  return gather;
}

RequesterTables requester_tables(const at::Tensor &words, const at::Tensor &headers,
                                 const at::Tensor &owner_group_shift,
                                 const at::Tensor &bucket_feature_of_row,
                                 const at::Tensor &row_shift, const at::Tensor &group_of_feature,
                                 const at::Tensor &offset_per_bucket, const Sizes &sizes) {
  const int64_t rows = bucket_feature_of_row.numel(), local_size = sizes.topo.local_size,
                max_chunks = sizes.max_chunks;
  const auto long_options = offset_per_bucket.options().dtype(at::kLong);
  const auto int_options = offset_per_bucket.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  RequesterTables tables;
  // Per proxy, from its gathered header: its chunks and their ends, and this
  // requester's push tables there, made absolute; and the remote task offsets.
  tables.push_order_off = at::empty({local_size + 1}, long_options);
  tables.proxy_num_chunks = at::empty({local_size}, long_options);
  tables.proxy_ends = at::empty({local_size, max_chunks + 1}, long_options);
  tables.proxy_push_tables = at::empty({local_size, max_chunks + 1}, int_options);
  tables.remote_task_offsets = at::empty({max_chunks * local_size + 1}, long_options);
  requester_proxy_tables_kernel<<<kOneCta, kThreads, 0, stream>>>(
      headers.data_ptr<int32_t>(), offset_per_bucket.data_ptr<int64_t>(), sizes,
      tables.push_order_off.data_ptr<int64_t>(), tables.proxy_num_chunks.data_ptr<int64_t>(),
      tables.proxy_ends.data_ptr<int64_t>(), tables.proxy_push_tables.data_ptr<int32_t>(),
      tables.remote_task_offsets.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tables.src = at::empty({rows}, int_options);
  tables.dst = at::empty({rows}, int_options);
  tables.push_order = at::empty({rows}, int_options);
  if (rows == 0) return tables;
  // Per row: its src and dst, and its place in the push order, its rows at
  // each proxy at their push positions there.
  const GatheredMap map{words.data_ptr<int32_t>(), rows};
  requester_rows_kernel<<<item_grid(rows), kThreads, 0, stream>>>(
      map, owner_group_shift.data_ptr<int64_t>(), bucket_feature_of_row.data_ptr<int32_t>(),
      row_shift.data_ptr<int64_t>(), group_of_feature.data_ptr<int32_t>(),
      offset_per_bucket.data_ptr<int64_t>(), tables.push_order_off.data_ptr<int64_t>(), rows, sizes,
      tables.src.data_ptr<int32_t>(), tables.dst.data_ptr<int32_t>(),
      tables.push_order.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tables;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
