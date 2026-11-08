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
// The planner's steps of the node, of the rail and the owner, of the
// requester, and the put lists (planner.h).

#include <algorithm>
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

// One CTA: the layout of the node's keys (node_layout_of, node.cuh) from the node's offsets, as
// node_exchange makes it. The hooks' (node_layout).
__global__ void __launch_bounds__(kThreads)
    node_layout_kernel(const int64_t *node_offsets, Topo topo, int64_t num_features,
                       int64_t lookup_keys, int64_t remote_keys, int64_t *scratch,
                       NodeLayoutView out) {
  node_layout_of(node_offsets, topo, num_features, lookup_keys, remote_keys, scratch, out);
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
  push_rows(rows, num_rows, places, topo, num_features, src,
            [&](int64_t target, int64_t place, int64_t key, float weight, int32_t feature) {
              if (place < 0 || place >= entries) return;
              node_keys[target * entries + place] = key;
              if (node_key_weights != nullptr) node_key_weights[target * entries + place] = weight;
              if (place >= lookup_keys)
                node_bucket_feature_of_row[target * remote_keys + place - lookup_keys] = feature;
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
  node.destination_base = at::empty({topo.world_size * num_features + 1}, long_options);
  if (topo.num_nodes > 1) {  // one node has no remote part
    node.requester_start = at::empty({local_size + 1}, long_options);
    node.need_remote = at::empty({}, long_options);
  } else {  // its own part is the owner's KJT
    node.kjt_offsets = at::empty({num_features * local_size + 1}, long_options);
    node.length_per_key = at::empty({num_features}, long_options);
  }
  return node;
}

NodeLayout node_layout(const at::Tensor &node_offsets, const Topo &topo, int64_t num_features,
                       int64_t lookup_keys, int64_t remote_keys) {
  TORCH_CHECK(is_cuda_array(node_offsets, at::kLong,
                            topo.local_size * (topo.world_size * num_features + 1)),
              "hier_output_dist: the node's offsets, local_size x (world_size * num_features + 1)");
  const auto options = node_offsets.options();
  NodeLayout node = empty_node_layout(topo, num_features, options);
  const at::Tensor scratch = at::empty({node_layout_scratch(topo, num_features)}, options);
  node_layout_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      node_offsets.data_ptr<int64_t>(), topo, num_features, lookup_keys, remote_keys,
      scratch.data_ptr<int64_t>(), view_of(node));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
                          offset_per_bucket.data_ptr<int64_t>(),
                          node.destination_base.data_ptr<int64_t>(), node.skip.data_ptr<bool>()};
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
  // The own part is the KJT: a copy of its first num_recv keys (wait()
  // narrows it to num_recv). The node's next call rewrites node_keys after
  // its barrier, which does not wait for this call's lookup.
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

// One CTA: a rank's count message (count_message_of, rail.cuh), as rail_exchange makes it. The
// hooks' (count_message).
__global__ void __launch_bounds__(kThreads)
    count_message_kernel(CountParts parts, CountMessage format, int64_t local_size,
                         int64_t *message) {
  count_message_of(parts, format, local_size, message, nullptr);
}

// One CTA: the rail's layout (rail_layout_of, rail.cuh) from the rail's count messages (messages
// [num_nodes][CountMessage::length]), as rail_exchange makes it. The hooks' (rail_layout).
__global__ void __launch_bounds__(kThreads)
    rail_layout_kernel(const int64_t *messages, CountMessage format, const int64_t *own_lengths,
                       Sizes sizes, int64_t *scratch, RailLayoutView out) {
  rail_layout_of(RailCounts{messages, format.length(), format}, own_lengths, sizes, scratch, out);
}

// One CTA: where each source node's chunks start in fwd_chunks, from the
// rail's slice_in (its own node sends none; at most owner_chunks in all),
// and where each other node's records of each feature start in rail_keys
// (record_offsets [feature][other node], counted in records of record_bytes:
// a slice and its chunks are whole records): after its chunks and its
// records of the features before. Then a thread per entry of fwd_chunks
// (owner_chunks of them): the chunk at that place, its source node's (the
// last whose chunks start at or before it), read from its slice of
// rail_keys (its bytes rail_bytes; a slice starts at a multiple of 8 bytes);
// zeros past the chunks.
__global__ void __launch_bounds__(kThreads)
    owner_slices_kernel(const int64_t *slice_in, const int64_t *node_length_per_source_feature,
                        const int64_t *other_nodes, const char *rail_keys, int64_t rail_bytes,
                        Sizes sizes, int64_t *fwd_chunk_start, int64_t *record_offsets,
                        ChunkPiece *fwd_chunks) {
  const int64_t num_nodes = sizes.topo.num_nodes, num_features = sizes.num_features,
                bound = sizes.owner_chunks();
  cta_exclusive_ends<int64_t>(
      num_nodes, [&](int64_t source) { return slice_in[2 * source + 1]; }, fwd_chunk_start);
  for (int64_t source = threadIdx.x; source <= num_nodes; source += kThreads)
    fwd_chunk_start[source] = min(fwd_chunk_start[source], bound);
  for (int64_t column = threadIdx.x; column < num_nodes - 1; column += kThreads) {
    const int64_t source = other_nodes[column];
    int64_t record =
        (slice_in[2 * source] + (int64_t)sizeof(ChunkPiece) * slice_in[2 * source + 1]) /
        sizes.record_bytes;
    for (int64_t feature = 0; feature < num_features; ++feature) {
      record_offsets[feature * (num_nodes - 1) + column] = record;
      record += node_length_per_source_feature[source * num_features + feature];
    }
  }
  __syncthreads();
  const int64_t count = fwd_chunk_start[num_nodes];
  constexpr int64_t kPiece = sizeof(ChunkPiece);
  for (int64_t index = threadIdx.x; index < bound; index += kThreads) {
    ChunkPiece piece{0, 0, 0, 0};
    if (index < count) {
      const int64_t source =
          clamp_index(upper_bound(fwd_chunk_start + 1, num_nodes, index), num_nodes);
      const int64_t byte = slice_in[2 * source] + (index - fwd_chunk_start[source]) * kPiece;
      piece = *(const ChunkPiece *)(rail_keys + min(max(byte, (int64_t)0), rail_bytes - kPiece));
    }
    fwd_chunks[index] = piece;
  }
}

// A thread per position of the owner's KJT, up to the keys it looks up (at most
// lookup_keys): the (feature, source) of the KJT's lengths it is in and its
// place there. A key of its own node is in the own part of node_keys, another
// node's in its Record in rail_keys (rail_records of them). With weighted
// records the KJT carries weights: an own key's from node_key_weights.
template <typename Record>
__global__ void __launch_bounds__(kThreads)
    owner_kjt_kernel(const int64_t *kjt_offsets, const int64_t *num_recv,
                     const int64_t *own_offsets, const int64_t *record_offsets,
                     const int64_t *node_keys, const float *node_key_weights,
                     const Record *rail_records, int64_t rail_count, Sizes sizes,
                     int64_t *kjt_values, float *kjt_weights) {
  const int64_t local_size = sizes.topo.local_size, num_sources = sizes.topo.num_sources,
                other_columns = sizes.topo.num_nodes - 1, lookup_keys = sizes.lookup_keys,
                num_lengths = sizes.num_features * num_sources;
  const int64_t keys = min(*num_recv, lookup_keys);
  for (int64_t position = first_item(); position < keys; position += item_stride()) {
    const int64_t index =
        clamp_index(upper_bound(kjt_offsets, num_lengths + 1, position) - 1, num_lengths);
    const int64_t feature = index / num_sources, source = index % num_sources,
                  nth = position - kjt_offsets[index];
    if (source < local_size) {
      const int64_t own =
          clamp_index(own_offsets[feature * local_size + source] + nth, lookup_keys);
      kjt_values[position] = node_keys[own];
      if constexpr (Record::kWeighted) kjt_weights[position] = node_key_weights[own];
    } else {
      const Record &record = rail_records[clamp_index(
          record_offsets[feature * other_columns + source - local_size] + nth, rail_count)];
      kjt_values[position] = record.key;
      if constexpr (Record::kWeighted) kjt_weights[position] = record.weight;
    }
  }
}

}  // namespace

at::Tensor other_nodes_of(const Topo &topo, const at::TensorOptions &options) {
  const at::Tensor nodes = at::arange(topo.num_nodes - 1, options.dtype(at::kLong));
  return nodes + nodes.ge(topo.node);
}

at::Tensor count_message(const at::Tensor &need_remote, const at::Tensor &own_total,
                         const at::Tensor &key_rows, const at::Tensor &chunk_length_per_bucket,
                         const at::Tensor &own_lengths,
                         const at::Tensor &node_length_per_bucket_feature) {
  const int64_t num_nodes = chunk_length_per_bucket.numel();
  const CountMessage format{num_nodes, node_length_per_bucket_feature.numel() / num_nodes};
  const at::Tensor message = at::empty({format.length()}, chunk_length_per_bucket.options());
  const CountParts parts{
      need_remote.data_ptr<int64_t>(), own_total.data_ptr<int64_t>(),
      key_rows.data_ptr<int64_t>(),    chunk_length_per_bucket.data_ptr<int64_t>(),
      own_lengths.data_ptr<int64_t>(), node_length_per_bucket_feature.data_ptr<int64_t>()};
  count_message_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      parts, format, own_lengths.numel() / format.num_features, message.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return message;
}

RailLayout empty_rail_layout(const Sizes &sizes, const at::TensorOptions &options) {
  const int64_t num_nodes = sizes.topo.num_nodes, local_size = sizes.topo.local_size,
                num_features = sizes.num_features,
                num_lengths = num_features * sizes.topo.num_sources;
  const auto long_options = options.dtype(at::kLong);
  RailLayout rail;
  rail.put_dst = at::empty({num_nodes}, long_options);
  rail.put_bytes = at::empty({num_nodes}, long_options);
  rail.put_skip = at::empty({num_nodes}, long_options.dtype(at::kBool));
  rail.owner_offsets = at::empty({num_nodes, num_features}, long_options);
  rail.kjt_lengths = at::empty({num_lengths}, long_options);
  rail.kjt_offsets = at::empty({num_lengths + 1}, long_options);
  rail.num_recv = at::empty({}, long_options);
  rail.length_per_key = at::empty({num_features}, long_options);
  rail.slice_in = at::empty({num_nodes, 2}, long_options);
  rail.node_length_per_source = at::empty({num_nodes}, long_options);
  rail.node_length_per_source_feature = at::empty({num_nodes, num_features}, long_options);
  rail.source_offsets = at::empty({num_nodes, num_features}, long_options);
  rail.requester_offsets = at::empty({local_size, num_features}, long_options);
  rail.needs = at::empty({kNumNeeds}, long_options);
  return rail;
}

RailLayout rail_layout(const at::Tensor &messages, const at::Tensor &own_lengths,
                       const Sizes &sizes) {
  const auto options = messages.options();
  RailLayout rail = empty_rail_layout(sizes, options);
  const at::Tensor scratch =
      at::empty({rail_layout_scratch(sizes.topo, sizes.num_features)}, options);
  rail_layout_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      messages.data_ptr<int64_t>(), CountMessage{sizes.topo.num_nodes, sizes.num_features},
      own_lengths.data_ptr<int64_t>(), sizes, scratch.data_ptr<int64_t>(), view_of(rail));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return rail;
}

OwnerTables owner_tables(const RailLayout &rail, const at::Tensor &own_offsets,
                         const at::Tensor &node_keys, const at::Tensor &node_key_weights,
                         const at::Tensor &rail_keys, const at::Tensor &other_nodes,
                         const Sizes &sizes) {
  const int64_t num_nodes = sizes.topo.num_nodes, lookup_keys = sizes.lookup_keys,
                rail_words = rail_keys.numel();
  const auto long_options = rail_keys.options().dtype(at::kLong);
  const cudaStream_t stream = current_stream();
  OwnerTables tables;
  // Each source node's slice in rail_keys starts with the chunks of this
  // node in its stream, then its records.
  const int64_t bound = sizes.owner_chunks();
  tables.fwd_chunk_start = at::empty({num_nodes + 1}, long_options);
  tables.fwd_chunks = at::empty({bound, 4}, long_options.dtype(at::kInt));
  const at::Tensor record_offsets = at::empty({sizes.num_features * (num_nodes - 1)}, long_options);
  owner_slices_kernel<<<kOneCta, kThreads, 0, stream>>>(
      rail.slice_in.data_ptr<int64_t>(), rail.node_length_per_source_feature.data_ptr<int64_t>(),
      other_nodes.data_ptr<int64_t>(), (const char *)rail_keys.data_ptr<int64_t>(), 8 * rail_words,
      sizes, tables.fwd_chunk_start.data_ptr<int64_t>(), record_offsets.data_ptr<int64_t>(),
      records_of<ChunkPiece>(tables.fwd_chunks));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The KJT's keys, its first num_recv positions: its own node's from
  // the own part of node_keys, the other nodes' from their records in
  // rail_keys.
  tables.kjt_values = at::empty({lookup_keys}, long_options);
  if (sizes.weighted_records())
    tables.kjt_weights = at::empty({lookup_keys}, long_options.dtype(at::kFloat));
  with_record(sizes, [&](auto record) {
    using Record = decltype(record);
    owner_kjt_kernel<Record><<<item_grid(lookup_keys), kThreads, 0, stream>>>(
        rail.kjt_offsets.data_ptr<int64_t>(), rail.num_recv.data_ptr<int64_t>(),
        own_offsets.data_ptr<int64_t>(), record_offsets.data_ptr<int64_t>(),
        node_keys.data_ptr<int64_t>(), data_or_null<float>(node_key_weights),
        (const Record *)rail_keys.data_ptr<int64_t>(), 8 * rail_words / (int64_t)sizeof(Record),
        sizes, tables.kjt_values.data_ptr<int64_t>(), data_or_null<float>(tables.kjt_weights));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tables;
}

// ── the requester ──

namespace {

// A thread per bucketized row: the node peer whose remote_map has its words,
// for a valid row of another node's owner (its proxy: the owner's local
// rank; else -1), and its occurrence there: its destination less
// lookup_keys, clamped into the remote part.
__global__ void __launch_bounds__(kThreads)
    remote_rows_kernel(const int32_t *bucket_feature_of_row, const int64_t *offset_per_bucket,
                       const int64_t *destination_base, int64_t rows, Sizes sizes, int32_t *peer,
                       int64_t *first_word) {
  const int64_t local_size = sizes.topo.local_size, world_size = sizes.topo.world_size;
  const int64_t valid_rows = offset_per_bucket[world_size];
  for (int64_t row = first_item(); row < rows; row += item_stride()) {
    const int64_t bucket_feature = bucket_feature_of_row[row];
    const int64_t owner = min(bucket_feature / sizes.num_features, world_size - 1);
    const bool remote = owner / local_size != sizes.topo.node && row < valid_rows;
    peer[row] = remote ? (int32_t)(owner % local_size) : -1;
    first_word[row] =
        clamp_index(destination_base[bucket_feature] + row - sizes.lookup_keys, sizes.remote_keys);
  }
}

// One CTA: this requester's tables of its node's proxies, from their
// gathered headers (headers [local_size][ProxyHeader::gathered_words]): where
// its rows at each proxy start in its push order (push_order_off: its rows of
// the owners with the proxy's local rank on the other nodes), each proxy's
// chunks and chunk ends, its push tables there made absolute (positions in
// push_order: its rows at the proxies before come first), and the task
// starts.
__global__ void __launch_bounds__(kThreads)
    requester_proxy_tables_kernel(const int32_t *headers, const int64_t *offset_per_bucket,
                                  Sizes sizes, int64_t *push_order_off, int64_t *proxy_chunks,
                                  int64_t *proxy_ends, int32_t *proxy_push_tables,
                                  int64_t *task_starts) {
  const ProxyHeader header = ProxyHeader::of(sizes);
  const int64_t local_size = sizes.topo.local_size, me = sizes.topo.local_rank,
                max_chunks = sizes.max_chunks, entries = max_chunks + 1;
  // its push table as the proxy published it: positions relative to its push
  // order there
  auto published = [&](int64_t proxy) { return header.gathered(headers, proxy).push_table(); };
  auto chunks_of = [&](int64_t proxy) {
    return min(max(header.gathered(headers, proxy).chunks(), (int64_t)0), max_chunks);
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
  cta_exclusive_ends<int64_t>(local_size, rows_at, push_order_off);
  for (int64_t proxy = threadIdx.x; proxy < local_size; proxy += kThreads)
    proxy_chunks[proxy] = chunks_of(proxy);
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
  // The task starts, chunk after chunk: in a chunk the proxies in turn from
  // this rank's own one, kTaskRows positions of its push order a task; a
  // proxy has none past its chunks. (The push tables' differences are the
  // same relative or absolute.)
  cta_exclusive_ends<int64_t>(
      max_chunks * local_size,
      [&](int64_t index) -> int64_t {
        const int64_t chunk = index / local_size, proxy = (me + index % local_size) % local_size;
        if (chunk >= chunks_of(proxy)) return 0;
        const int32_t *table = published(proxy);
        return ceil_div<kTaskRows>((int64_t)table[chunk + 1] - table[chunk]);
      },
      task_starts);
}

// A thread per bucketized row. A row of an owner on this node is in the owner's
// KJT, where this requester's keys of the feature start plus its place among
// them, both forward and backward. Another node's row reads its key's row in
// proxy_rows and pushes its gradient to its slot, from what it gathered of its
// proxy's remote_map (map); a valid one is at its push position at the proxy in
// its push order.
__global__ void __launch_bounds__(kThreads)
    requester_rows_kernel(GatheredMap map, const int64_t *offset_per_owner_feature,
                          const int32_t *bucket_feature_of_row,
                          const int64_t *offset_per_bucket_feature,
                          const int64_t *offset_per_bucket, const int64_t *push_order_off,
                          int64_t rows, Sizes sizes, int32_t *src, int32_t *dst,
                          int32_t *push_order) {
  const int64_t local_size = sizes.topo.local_size, num_features = sizes.num_features,
                world_size = sizes.topo.world_size;
  const int64_t valid_rows = offset_per_bucket[world_size];
  for (int64_t row = first_item(); row < rows; row += item_stride()) {
    const int64_t bucket_feature = bucket_feature_of_row[row];
    const int64_t owner = min(bucket_feature / num_features, world_size - 1);
    const int64_t peer = owner % local_size;  // the owner itself on this node, else its proxy
    if (owner / local_size == sizes.topo.node) {
      const int32_t place =
          (int32_t)(offset_per_owner_feature[peer * num_features + bucket_feature % num_features] +
                    row - offset_per_bucket_feature[bucket_feature]);
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

PeerWords remote_rows(const at::Tensor &bucket_feature_of_row, const at::Tensor &offset_per_bucket,
                      const at::Tensor &destination_base, const Sizes &sizes) {
  const int64_t rows = bucket_feature_of_row.numel();
  PeerWords gather;
  gather.peer = at::empty({rows}, bucket_feature_of_row.options());
  gather.first_word = at::empty({rows}, offset_per_bucket.options());
  if (rows > 0) {
    remote_rows_kernel<<<item_grid(rows), kThreads, 0, current_stream()>>>(
        bucket_feature_of_row.data_ptr<int32_t>(), offset_per_bucket.data_ptr<int64_t>(),
        destination_base.data_ptr<int64_t>(), rows, sizes, gather.peer.data_ptr<int32_t>(),
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
      {at::arange(ProxyHeader::kChunksWord, ProxyHeader::kEndWord + 1, long_options),
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
                                 const at::Tensor &offset_per_owner_feature,
                                 const at::Tensor &bucket_feature_of_row,
                                 const at::Tensor &offset_per_bucket_feature,
                                 const at::Tensor &offset_per_bucket, const Sizes &sizes) {
  const int64_t rows = bucket_feature_of_row.numel(), local_size = sizes.topo.local_size,
                max_chunks = sizes.max_chunks;
  const auto long_options = offset_per_bucket_feature.options().dtype(at::kLong);
  const auto int_options = offset_per_bucket_feature.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  RequesterTables tables;
  // Per proxy, from its gathered header: its chunks and their ends, and
  // this requester's push tables there, made absolute; and the task starts.
  tables.push_order_off = at::empty({local_size + 1}, long_options);
  tables.proxy_chunks = at::empty({local_size}, long_options);
  tables.proxy_ends = at::empty({local_size, max_chunks + 1}, long_options);
  tables.proxy_push_tables = at::empty({local_size, max_chunks + 1}, int_options);
  tables.task_starts = at::empty({max_chunks * local_size + 1}, long_options);
  requester_proxy_tables_kernel<<<kOneCta, kThreads, 0, stream>>>(
      headers.data_ptr<int32_t>(), offset_per_bucket.data_ptr<int64_t>(), sizes,
      tables.push_order_off.data_ptr<int64_t>(), tables.proxy_chunks.data_ptr<int64_t>(),
      tables.proxy_ends.data_ptr<int64_t>(), tables.proxy_push_tables.data_ptr<int32_t>(),
      tables.task_starts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tables.src = at::empty({rows}, int_options);
  tables.dst = at::empty({rows}, int_options);
  tables.push_order = at::empty({rows}, int_options);
  if (rows == 0) return tables;
  // Per row: its src and dst, and its place in the push order, its rows at
  // each proxy at their push positions there.
  const GatheredMap map{words.data_ptr<int32_t>(), rows};
  requester_rows_kernel<<<item_grid(rows), kThreads, 0, stream>>>(
      map, offset_per_owner_feature.data_ptr<int64_t>(), bucket_feature_of_row.data_ptr<int32_t>(),
      offset_per_bucket_feature.data_ptr<int64_t>(), offset_per_bucket.data_ptr<int64_t>(),
      tables.push_order_off.data_ptr<int64_t>(), rows, sizes, tables.src.data_ptr<int32_t>(),
      tables.dst.data_ptr<int32_t>(), tables.push_order.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tables;
}

// ── the put lists ──

namespace {

// A cut of the lists, as the int64 they are sorted by: its node in the high
// 32 bits, its place among the node's keys (below 2^31) in the low ones.
struct CutKey {
  static constexpr int kNodeShift = 32;
  static constexpr int64_t kMaxKey = (int64_t{1} << 31) - 1;
  __host__ __device__ static int64_t of(int64_t node, int64_t key) {
    return (node << kNodeShift) + key;
  }
  __host__ __device__ static int64_t node(int64_t cut) { return cut >> kNodeShift; }
  __host__ __device__ static int64_t key(int64_t cut) {
    return cut & ((int64_t{1} << kNodeShift) - 1);
  }
};

// The two lists as one of 2 * num_nodes nodes: the backward's owner nodes
// (the proxy's chunks of each, and its node-unique keys per (owner node,
// feature)), then the forward's proxy nodes (the chunks of this owner's node
// in each one's stream, and its keys per feature). Their chunks are numbered
// node after node, each node's in order: the backward's, then the forward's.
// A call that does not fit may read words no one wrote: they are clamped.
struct PutChunks {
  const Chunk *chunks;                            // [max_chunks]: the proxy's, in stream order
  const int64_t *chunk_offset_per_bucket;         // [num_nodes + 1]: each owner node's in chunk_inv
  const int32_t *chunk_inv;                       // [max_chunks]: each owner node's chunks in order
  const int64_t *node_offset_per_bucket;          // [num_nodes + 1]: the proxy's first key of each
                                                  // owner node
  const ChunkPiece *fwd_chunks;                   // [owner_chunks]: the owner's, source node after
                                                  // source node
  const int64_t *fwd_chunk_start;                 // [num_nodes + 1]
  const int64_t *node_length_per_bucket_feature;  // [num_nodes][num_features]
  const int64_t *node_length_per_source_feature;  // [num_nodes][num_features]
  int64_t num_nodes, num_features, max_chunks, owner_chunks, ring;

  // A chunk of the numbering: its node, its keys [lo, hi) among the node's
  // keys, the row of lo in the proxy's proxy_rows, and its number in its own
  // list (the proxy's chunk, or the entry of fwd_chunks).
  struct Place {
    int64_t node, lo, hi, row, number;
  };
  __device__ int64_t backward_chunks() const {
    return min(chunk_offset_per_bucket[num_nodes], max_chunks);
  }
  __device__ int64_t chunks_in_all() const {
    return backward_chunks() + min(fwd_chunk_start[num_nodes], owner_chunks);
  }
  __device__ Place place(int64_t index) const {
    Place out;
    const int64_t backward = backward_chunks();
    if (index < backward) {
      const int64_t owner_node =
          clamp_index(upper_bound(chunk_offset_per_bucket + 1, num_nodes, index), num_nodes);
      const Chunk chunk = chunks[clamp_index(chunk_inv[index], max_chunks)];
      out.node = owner_node;
      out.lo = chunk.key_lo - node_offset_per_bucket[owner_node];
      out.hi = out.lo + chunk.keys;
      out.row = chunk.start % ring;
      out.number = clamp_index(chunk_inv[index], max_chunks);
    } else {
      const int64_t entry = clamp_index(index - backward, owner_chunks);
      const int64_t source =
          clamp_index(upper_bound(fwd_chunk_start + 1, num_nodes, entry), num_nodes);
      const ChunkPiece piece = fwd_chunks[entry];
      out.node = num_nodes + source;
      out.lo = piece.uniq_lo;
      out.hi = piece.uniq_hi;
      out.row = piece.start % ring;
      out.number = entry;
    }
    out.lo = min(max(out.lo, (int64_t)0), CutKey::kMaxKey);
    out.hi = min(max(out.hi, out.lo), CutKey::kMaxKey);
    return out;
  }
  // the node's keys of a feature
  __device__ int64_t keys(int64_t node, int64_t feature) const {
    return node < num_nodes
               ? node_length_per_bucket_feature[node * num_features + feature]
               : node_length_per_source_feature[(node - num_nodes) * num_features + feature];
  }
};

// The rows of the segment from a cut to the next cut: the keys between them
// when both are of one node, else none.
__device__ __forceinline__ int64_t segment_rows(int64_t cut, int64_t next) {
  return CutKey::node(next) == CutKey::node(cut) ? CutKey::key(next) - CutKey::key(cut) : 0;
}

// One CTA: where each node's keys of each feature start, as cut keys
// (feature_cut [nodes][num_features]).
__global__ void __launch_bounds__(kThreads)
    feature_cuts_kernel(PutChunks lists, int64_t *feature_cut) {
  const int64_t nodes = 2 * lists.num_nodes, num_features = lists.num_features;
  for (int64_t node = threadIdx.x; node < nodes; node += kThreads) {
    int64_t end = 0;  // the end of the node's features so far
    for (int64_t feature = 0; feature < num_features; ++feature) {
      feature_cut[node * num_features + feature] = CutKey::of(node, end);
      end += lists.keys(node, feature);
    }
  }
}

// A thread per cut (CutKey): first the first key of each chunk (chunk_cuts
// of them, numbered as PutChunks does: past the chunks after every node),
// which also go to chunk_key, then the end of each node's keys of each
// feature.
__global__ void __launch_bounds__(kThreads)
    cut_keys_kernel(PutChunks lists, const int64_t *feature_cut, int64_t chunk_cuts,
                    int64_t *cut_keys, int64_t *chunk_key) {
  const int64_t nodes = 2 * lists.num_nodes, node_features = nodes * lists.num_features;
  const int64_t chunks = lists.chunks_in_all();
  for (int64_t cut = first_item(); cut < chunk_cuts + node_features; cut += item_stride()) {
    int64_t key;
    if (cut >= chunk_cuts) {
      const int64_t node_feature = cut - chunk_cuts;
      key = feature_cut[node_feature] +
            lists.keys(node_feature / lists.num_features, node_feature % lists.num_features);
    } else {
      key = CutKey::of(nodes, 0);
      if (cut < chunks) {
        const PutChunks::Place place = lists.place(cut);
        key = CutKey::of(place.node, place.lo);
      }
      chunk_key[cut] = key;
    }
    cut_keys[cut] = key;
  }
}

// A thread per segment of the sorted cuts: whether it holds rows.
__global__ void __launch_bounds__(kThreads)
    kept_segments_kernel(const int64_t *sorted, int64_t segments, bool *kept) {
  for (int64_t segment = first_item(); segment < segments; segment += item_stride())
    kept[segment] = segment_rows(sorted[segment], sorted[segment + 1]) > 0;
}

// A thread per segment that holds rows: its PutSeg, at its place among them
// (kept_before). Its chunk and its node's feature are the last that start at
// or before it: the chunk gives its row in proxy_rows and its number, the
// feature its row in the owner's KJT (owner_offsets for the backward's owner
// nodes, source_offsets for the forward's proxy nodes). The backward's src is
// the row of proxy_rows, the forward's the row of the KJT.
__global__ void __launch_bounds__(kThreads)
    put_segments_kernel(PutChunks lists, const int64_t *sorted, const int32_t *kept_before,
                        int64_t segments, const int64_t *chunk_key, int64_t chunk_cuts,
                        const int64_t *feature_cut, const int64_t *owner_offsets,
                        const int64_t *source_offsets, PutSeg *list) {
  const int64_t num_nodes = lists.num_nodes, node_features = 2 * num_nodes * lists.num_features;
  for (int64_t segment = first_item(); segment < segments; segment += item_stride()) {
    const int64_t cut = sorted[segment], rows = segment_rows(cut, sorted[segment + 1]);
    if (rows <= 0) continue;
    const int64_t first = CutKey::key(cut);
    const PutChunks::Place chunk =
        lists.place(clamp_index(upper_bound(chunk_key, chunk_cuts, cut) - 1, chunk_cuts));
    const int64_t chunk_row = chunk.row + first - chunk.lo;
    const int64_t node_feature =
        clamp_index(upper_bound(feature_cut, node_features, cut) - 1, node_features);
    const int64_t kjt_offset = node_feature < num_nodes * lists.num_features
                                   ? owner_offsets[node_feature]
                                   : source_offsets[node_feature - num_nodes * lists.num_features];
    const int64_t kjt_row = kjt_offset + first - CutKey::key(feature_cut[node_feature]);
    const bool forward = CutKey::node(cut) >= num_nodes;
    list[kept_before[segment]] =
        PutSeg{(int32_t)(forward ? kjt_row : chunk_row), (int32_t)(forward ? chunk_row : kjt_row),
               (int32_t)rows, (int32_t)chunk.number};
  }
}

// One CTA: where each node's segments start in the list (starts [nodes + 1]):
// the segments kept before the node's first cut.
__global__ void __launch_bounds__(kThreads)
    put_starts_kernel(const int64_t *sorted, const int32_t *kept_before, int64_t segments,
                      int64_t nodes, int32_t *starts) {
  for (int64_t node = threadIdx.x; node <= nodes; node += kThreads)
    starts[node] =
        kept_before[min(lower_bound(sorted, segments + 1, CutKey::of(node, 0)), segments)];
}

}  // namespace

PutLists put_lists(const at::Tensor &chunks, const at::Tensor &chunk_offset_per_bucket,
                   const at::Tensor &chunk_inv, const at::Tensor &node_offset_per_bucket,
                   const at::Tensor &node_length_per_bucket_feature,
                   const at::Tensor &owner_offsets, const at::Tensor &fwd_chunks,
                   const at::Tensor &fwd_chunk_start,
                   const at::Tensor &node_length_per_source_feature,
                   const at::Tensor &source_offsets, const Sizes &sizes) {
  const int64_t num_nodes = sizes.topo.num_nodes, num_features = sizes.num_features;
  // The two lists as one of 2 * num_nodes nodes: the backward's owner nodes,
  // then the forward's proxy nodes.
  const int64_t nodes = 2 * num_nodes, node_features = nodes * num_features;
  const auto long_options = node_length_per_bucket_feature.options().dtype(at::kLong);
  const auto int_options = node_length_per_bucket_feature.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  const PutChunks lists{records_of<const Chunk>(chunks),
                        chunk_offset_per_bucket.data_ptr<int64_t>(),
                        chunk_inv.data_ptr<int32_t>(),
                        node_offset_per_bucket.data_ptr<int64_t>(),
                        records_of<const ChunkPiece>(fwd_chunks),
                        fwd_chunk_start.data_ptr<int64_t>(),
                        node_length_per_bucket_feature.data_ptr<int64_t>(),
                        node_length_per_source_feature.data_ptr<int64_t>(),
                        num_nodes,
                        num_features,
                        sizes.max_chunks,
                        sizes.owner_chunks(),
                        sizes.num_proxy_rows};
  PutLists out;
  const at::Tensor feature_cut = at::empty({node_features}, long_options);
  feature_cuts_kernel<<<kOneCta, kThreads, 0, stream>>>(lists, feature_cut.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The cuts, by (node, key): the first key of each chunk (past the chunks:
  // after every node) and the end of each node's keys of each feature.
  // Segment i goes from cut i to cut i + 1, and holds rows when both are of
  // one node.
  const int64_t chunk_cuts = sizes.max_chunks + sizes.owner_chunks();
  const at::Tensor cut_keys = at::empty({chunk_cuts + node_features}, long_options);
  const at::Tensor chunk_key = at::empty({chunk_cuts}, long_options);
  cut_keys_kernel<<<item_grid(chunk_cuts + node_features), kThreads, 0, stream>>>(
      lists, feature_cut.data_ptr<int64_t>(), chunk_cuts, cut_keys.data_ptr<int64_t>(),
      chunk_key.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const at::Tensor sorted = sort_keys(cut_keys, CutKey::kNodeShift + bits_for(nodes + 1));
  // The segments that hold rows, in order, and where each node's first one is.
  const int64_t segments = sorted.numel() - 1;
  const at::Tensor kept = at::empty({segments}, long_options.dtype(at::kBool));
  kept_segments_kernel<<<item_grid(segments), kThreads, 0, stream>>>(
      sorted.data_ptr<int64_t>(), segments, kept.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const at::Tensor kept_before = exclusive_ends(kept, at::kInt);
  out.list = at::empty({segments, 4}, int_options);
  put_segments_kernel<<<item_grid(segments), kThreads, 0, stream>>>(
      lists, sorted.data_ptr<int64_t>(), kept_before.data_ptr<int32_t>(), segments,
      chunk_key.data_ptr<int64_t>(), chunk_cuts, feature_cut.data_ptr<int64_t>(),
      owner_offsets.data_ptr<int64_t>(), source_offsets.data_ptr<int64_t>(),
      records_of<PutSeg>(out.list));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  out.starts = at::empty({nodes + 1}, int_options);
  put_starts_kernel<<<kOneCta, kThreads, 0, stream>>>(sorted.data_ptr<int64_t>(),
                                                      kept_before.data_ptr<int32_t>(), segments,
                                                      nodes, out.starts.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
