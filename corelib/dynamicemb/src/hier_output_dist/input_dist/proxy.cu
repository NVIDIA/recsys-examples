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
// The proxy's steps of a call across nodes (planner.h): the node-unique keys
// of its remote part, their rows, the chunks of its stream and their slots,
// and its node's push orders with its reducer list.

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/input_dist/planner.cuh"
#include "hier_output_dist/input_dist/planner.h"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

// ── step 1: the node-unique keys ──

// The owner node of a bucket feature owner node * num_features + feature (the
// bucket feature past the node-unique keys, num_nodes * num_features: the last
// node).
__device__ __forceinline__ int64_t owner_node_of(int64_t bucket_feature, int64_t num_features,
                                                 int64_t num_nodes) {
  return min(bucket_feature / num_features, num_nodes - 1);
}

// A thread per occurrence: whether it is valid, among the occurrences the node
// layout counts for this proxy (none when the node skips the call), and its own
// index, which the sort by key carries (and the sort by bucket feature, as an
// identity's index). The first thread writes the count, and key_rows' start.
__global__ void __launch_bounds__(kThreads)
    valid_occurrences_kernel(const bool *skip, const int64_t *requester_start, int64_t local_size,
                             int64_t entries, int64_t *occurrence_count, int64_t *key_rows,
                             bool *valid, int32_t *index) {
  const int64_t count = *skip ? 0 : requester_start[local_size];
  if (first_item() == 0) {
    *occurrence_count = count;
    *key_rows = 0;
  }
  for (int64_t occurrence = first_item(); occurrence < entries; occurrence += item_stride()) {
    valid[occurrence] = occurrence < count;
    index[occurrence] = (int32_t)occurrence;
  }
}

// A thread per position of the occurrences sorted by key: the identity tag
// of the occurrence there, owner node * num_tables + the table of its
// feature, or num_nodes * num_tables for an invalid one (which sorts it last).
__global__ void __launch_bounds__(kThreads)
    identity_tags_kernel(const int32_t *by_key, const bool *valid,
                         const int32_t *bucket_feature_of_occ, const int32_t *tables,
                         int64_t entries, int64_t num_nodes, int64_t num_features,
                         int64_t num_tables, int32_t *tag) {
  for (int64_t position = first_item(); position < entries; position += item_stride()) {
    const int32_t occurrence = by_key[position];
    int64_t identity_tag = num_nodes * num_tables;
    if (valid[occurrence]) {
      const int64_t bucket_feature = bucket_feature_of_occ[occurrence];
      identity_tag = owner_node_of(bucket_feature, num_features, num_nodes) * num_tables +
                     tables[bucket_feature % num_features];
    }
    tag[position] = (int32_t)identity_tag;
  }
}

// A thread per sorted position (the occurrences in identity order): whether
// an identity starts there, where the key or the tag differs from the
// position before (at every position without node dedup, kNodeDedup false);
// and when the records carry the weights, the weight of its occurrence as a
// whole number (0 when invalid), which the identities sum. It also sets the
// smallest feature of identity `position` above every feature, for the next
// kernel to lower.
template <bool kNodeDedup>
__global__ void __launch_bounds__(kThreads)
    identity_starts_kernel(const int32_t *order, const int32_t *tag_sorted,
                           const int64_t *occurrence_keys, const float *weights, const bool *valid,
                           int64_t entries, int64_t num_features, bool *starts_identity,
                           int64_t *position_weight, int32_t *smallest_feature) {
  for (int64_t position = first_item(); position < entries; position += item_stride()) {
    const int32_t occurrence = order[position];
    starts_identity[position] =
        !kNodeDedup || position == 0 ||
        occurrence_keys[occurrence] != occurrence_keys[order[position - 1]] ||
        tag_sorted[position] != tag_sorted[position - 1];
    if (position_weight != nullptr)
      position_weight[position] = valid[position] ? (int64_t)weights[occurrence] : 0;
    smallest_feature[position] = (int32_t)num_features;
  }
}

// A thread per sorted position, and one past them: the identity at each
// position (identity_ends: the identities that start before it), where each
// identity starts (entries past the last one), and the smallest feature of
// each valid identity, an atomicMin of its features. The smallest is at the
// identity's first position or at one where the feature drops, so only those
// take part: few, as an identity's occurrences come requester after
// requester, each requester's by feature. The first thread writes num_keys:
// the identities that start at the valid positions, which come first.
__global__ void __launch_bounds__(kThreads)
    identity_ranges_kernel(const int32_t *identity_ends, const int32_t *order,
                           const int32_t *bucket_feature_of_occ, const bool *valid,
                           const int64_t *occurrence_count, int64_t entries, int64_t num_features,
                           int32_t *identity_of, int32_t *identity_start, int32_t *smallest_feature,
                           int64_t *num_keys) {
  const int32_t identities = identity_ends[entries];
  if (first_item() == 0) *num_keys = identity_ends[*occurrence_count];
  for (int64_t position = first_item(); position <= entries; position += item_stride()) {
    if (position >= identities) identity_start[position] = (int32_t)entries;
    if (position == entries) continue;
    const int32_t identity = identity_ends[position + 1] - 1;
    const bool starts = identity_ends[position] == identity;  // the first of its identity
    identity_of[position] = identity;
    if (starts) identity_start[identity] = (int32_t)position;
    if (!valid[position]) continue;
    const int32_t feature = (int32_t)(bucket_feature_of_occ[order[position]] % num_features);
    if (starts || feature < bucket_feature_of_occ[order[position - 1]] % num_features)
      atomicMin(&smallest_feature[identity], feature);
  }
}

// A thread per identity: its bucket feature, its owner node * num_features +
// its smallest feature; num_nodes * num_features past the node-unique keys,
// which sorts those last.
__global__ void __launch_bounds__(kThreads)
    identity_bucket_features_kernel(const int32_t *identity_start, const int32_t *order,
                                    const int32_t *bucket_feature_of_occ,
                                    const int32_t *smallest_feature, const int64_t *num_keys,
                                    int64_t entries, int64_t num_nodes, int64_t num_features,
                                    int32_t *identity_bucket_feature) {
  const int64_t valid_keys = *num_keys;
  for (int64_t identity = first_item(); identity < entries; identity += item_stride()) {
    int64_t bucket_feature = num_nodes * num_features;
    if (identity < valid_keys) {
      const int64_t first_bucket_feature = bucket_feature_of_occ[order[identity_start[identity]]];
      bucket_feature = owner_node_of(first_bucket_feature, num_features, num_nodes) * num_features +
                       smallest_feature[identity];
    }
    identity_bucket_feature[identity] = (int32_t)bucket_feature;
  }
}

// A thread per key index u (the identities sorted by bucket feature): which key
// index its identity gets (key_of_identity, the inverse of the sort), its
// entries of the per-key tables of ProxyKeys, and with weighted records its
// record's weight, the sum of its occurrences' weights (from weight_ends at
// its positions). key_rows gets the most occurrences of a valid key, one max
// per warp.
__global__ void __launch_bounds__(kThreads)
    key_tables_kernel(const int32_t *identity_of_key, const int32_t *bucket_feature_of_key,
                      const int32_t *identity_start, const int32_t *order,
                      const int64_t *occurrence_keys, const int64_t *weight_ends,
                      const int64_t *uniq_off, const int64_t *num_keys, int64_t entries,
                      int64_t num_nodes, int64_t num_features, int32_t *key_of_identity,
                      bool *valid_key, int32_t *key_count, int32_t *key_first, int32_t *key_node,
                      int32_t *key_uniq, int64_t *key_value, float *key_weight, int64_t *key_rows) {
  const int64_t valid_keys = *num_keys;
  int32_t most = 0;  // the most occurrences of this thread's valid keys
  for (int64_t key = first_item(); key < entries; key += item_stride()) {
    const int32_t identity = identity_of_key[key];
    const int32_t first = identity_start[identity], end = identity_start[identity + 1];
    const int64_t owner_node = owner_node_of(bucket_feature_of_key[key], num_features, num_nodes);
    key_of_identity[identity] = (int32_t)key;
    valid_key[key] = key < valid_keys;
    key_count[key] = end - first;
    if (key < valid_keys) most = max(most, end - first);
    key_first[key] = first;
    key_node[key] = (int32_t)owner_node;
    key_uniq[key] = (int32_t)(key - uniq_off[owner_node]);
    key_value[key] = occurrence_keys[order[min(first, (int32_t)entries - 1)]];
    if (key_weight != nullptr) key_weight[key] = (float)(weight_ends[end] - weight_ends[first]);
  }
  most = __reduce_max_sync(0xffffffffu, most);
  if ((threadIdx.x & 31) == 0 && most > 0)
    atomicMax((unsigned long long *)key_rows, (unsigned long long)most);
}

// A thread per sorted position: the key index of its occurrence's identity.
__global__ void __launch_bounds__(kThreads)
    position_keys_kernel(const int32_t *identity_of, const int32_t *key_of_identity,
                         int64_t entries, int32_t *key_at) {
  for (int64_t position = first_item(); position < entries; position += item_stride())
    key_at[position] = key_of_identity[identity_of[position]];
}

// One CTA: the node-unique keys of each bucket feature and of each owner node,
// from where they start.
__global__ void __launch_bounds__(kThreads)
    key_counts_kernel(const int64_t *key_offset_per_bucket_feature, const int64_t *uniq_off,
                      int64_t num_nodes, int64_t num_features, int64_t *uniq_per_bucket_feature,
                      int64_t *uniq_out) {
  for (int64_t bucket_feature = threadIdx.x; bucket_feature < num_nodes * num_features;
       bucket_feature += kThreads)
    uniq_per_bucket_feature[bucket_feature] = key_offset_per_bucket_feature[bucket_feature + 1] -
                                              key_offset_per_bucket_feature[bucket_feature];
  for (int64_t owner_node = threadIdx.x; owner_node < num_nodes; owner_node += kThreads)
    uniq_out[owner_node] = uniq_off[owner_node + 1] - uniq_off[owner_node];
}

// ── step 2: the rows, the chunks and the slots ──

// A thread per key index: its rows in proxy_rows, one for each occurrence, at
// most max_key_rows (a call with more raises: kNeedKeyRows); 0 past the
// node-unique keys.
__global__ void __launch_bounds__(kThreads)
    key_row_counts_kernel(const bool *valid_key, const int32_t *key_count, int64_t entries,
                          int32_t max_key_rows, int32_t *rows) {
  for (int64_t key = first_item(); key < entries; key += item_stride())
    rows[key] = valid_key[key] ? min(key_count[key], max_key_rows) : 0;
}

// The levels of the proxy's stream (common.h): chunk j of an owner node takes
// its keys whose rows before them, among the node's, are in [j * room,
// (j + 1) * room) (room: Sizes::chunk_room); a key spans at most room rows,
// so no chunk is empty.
struct Levels {
  const int64_t *row_ends;    // [entries + 1]: the rows before each key
  const int64_t *uniq_off;    // [num_nodes + 1]: the first key of each owner node
  const int64_t *chunks_out;  // [num_nodes]: the chunks of each owner node
  int64_t num_nodes, room;

  // The first key of owner node `owner_node`'s chunk j, among the node's (its
  // keys when j is its chunks).
  __device__ int64_t first_key(int64_t owner_node, int64_t j) const {
    const int64_t first = uniq_off[owner_node], keys = uniq_off[owner_node + 1] - first;
    return lower_bound(row_ends + first, keys, row_ends[first] + j * room);
  }
  // The chunks of the levels below j: each owner node's, up to j.
  __device__ int64_t before(int64_t j) const {
    int64_t chunks = 0;
    for (int64_t owner_node = 0; owner_node < num_nodes; ++owner_node)
      chunks += min(j, chunks_out[owner_node]);
    return chunks;
  }
};

// The chunks of the proxy's stream, made by one CTA (stream_kernel) in three
// steps.
struct StreamArgs {
  Levels levels;            // its chunks_out is the first step's
  const int64_t *uniq_out;  // [num_nodes]
  int64_t node, max_chunks, record_bytes, ring, lap_room;
  // what the steps write: of each owner node, its chunks, where they start in
  // node_chunks, and where its slice starts in key_stage; the stream's chunks
  int64_t *chunks_out, *node_chunk_start, *stage_slice;
  int64_t *num_chunks;
  Chunk *chunks;         // [max_chunks]
  int32_t *node_chunks;  // [max_chunks]
  // [max_chunks]: each chunk's rows and reducer tasks; [max_chunks + 1]: their
  // ends in stream order
  int64_t *rows_of, *reduce_tasks, *stream_ends, *reduce_task_starts;
  // the proxy header's words, and key_stage
  int32_t *header_chunks;
  uint32_t *header_ends, *header_end;
  char *key_stage;
};

// Each owner node's chunks (the level of its last key, plus one; none for
// this node), where they start in node_chunks, the stream's chunks (to the
// proxy header too), and where each owner node's slice starts in key_stage:
// its chunks (ChunkPiece), then its keys' records. Every thread of the CTA.
__device__ void chunk_counts(const StreamArgs &args) {
  const Levels &levels = args.levels;
  const int64_t num_nodes = levels.num_nodes, node = args.node, max_chunks = args.max_chunks,
                record_bytes = args.record_bytes;
  const int64_t *uniq_out = args.uniq_out;
  auto chunks_of = [&](int64_t owner_node) -> int64_t {
    const int64_t first = levels.uniq_off[owner_node], keys = uniq_out[owner_node];
    if (owner_node == node || keys <= 0) return 0;
    return (levels.row_ends[first + keys - 1] - levels.row_ends[first]) / levels.room + 1;
  };
  for (int64_t owner_node = threadIdx.x; owner_node < num_nodes; owner_node += kThreads)
    args.chunks_out[owner_node] = chunks_of(owner_node);
  const int64_t total = cta_exclusive_ends<int64_t>(num_nodes, chunks_of, args.node_chunk_start);
  for (int64_t owner_node = threadIdx.x; owner_node < num_nodes; owner_node += kThreads) {
    int64_t start = 0;
    for (int64_t before = 0; before < owner_node; ++before)
      start += chunks_of(before) * (int64_t)sizeof(ChunkPiece) + uniq_out[before] * record_bytes;
    args.stage_slice[owner_node] = start;
  }
  if (threadIdx.x == 0) {
    *args.num_chunks = min(total, max_chunks);
    *args.header_chunks = (int32_t)min(total, max_chunks);
  }
}

// A thread per place c of the stream: its chunk, of level j and owner node o
// (the levels below j hold before(j) chunks, and in level j the owner nodes
// with more than j chunks come in node order); zeros past the chunks. Its
// keys (key_rows_kernel adds their occurrences, the rows its requesters
// pull), its rows (rows_of, for the stream's scan), its reducer tasks of 32
// keys, and its place among its owner node's (node_chunks).
__device__ void chunk_table(const StreamArgs &args) {
  const Levels &levels = args.levels;
  const int64_t count = *args.num_chunks, max_chunks = args.max_chunks;
  for (int64_t place = threadIdx.x; place < max_chunks; place += kThreads) {
    Chunk chunk{};
    int64_t rows = 0;
    if (place < count) {
      // the level: the last j whose levels below hold at most `place` chunks
      int64_t lo = 0, hi = max_chunks;
      while (lo < hi) {
        const int64_t mid = (lo + hi + 1) / 2;
        if (levels.before(mid) <= place)
          lo = mid;
        else
          hi = mid - 1;
      }
      const int64_t level = lo;
      int64_t rank = place - levels.before(level), owner_node = 0;
      for (; owner_node < levels.num_nodes - 1; ++owner_node)
        if (levels.chunks_out[owner_node] > level && rank-- == 0) break;
      const int64_t first = levels.uniq_off[owner_node];
      const int64_t lo_key = first + levels.first_key(owner_node, level),
                    hi_key = first + levels.first_key(owner_node, level + 1);
      chunk.owner_node = (int32_t)owner_node;
      chunk.key_lo = (int32_t)lo_key;
      chunk.keys = (int32_t)(hi_key - lo_key);
      chunk.level = (int32_t)level;
      rows = levels.row_ends[hi_key] - levels.row_ends[lo_key];
      args.node_chunks[clamp_index(args.node_chunk_start[owner_node] + level, max_chunks)] =
          (int32_t)place;
    }
    args.chunks[place] = chunk;
    args.rows_of[place] = rows;
    args.reduce_tasks[place] = ceil_div<kTaskRows>((int64_t)chunk.keys);
  }
}

// The ends of the stream's chunks' rows, in stream order (stream_ends, from
// rows_of), and the reducer tasks before each chunk (reduce_task_starts,
// from reduce_tasks); then, a thread per place c of the stream, where its
// chunk lies, unwrapped. A stream that fits the ring (ring rows) lies as it
// is; else lap k takes the chunks whose rows before them are in
// [k * lap_room, (k + 1) * lap_room), from position k * ring, one after
// another. Its start and end into its Chunk, its end into the proxy header,
// and its ChunkPiece, at its level, first in its owner node's slice of
// key_stage; the stream's end into the header. Every thread of the CTA.
__device__ void chunk_places(const StreamArgs &args) {
  const int64_t max_chunks = args.max_chunks, ring = args.ring, lap_room = args.lap_room;
  const int64_t *stream_ends = args.stream_ends, *uniq_off = args.levels.uniq_off;
  cta_exclusive_ends<int64_t>(
      max_chunks, [&](int64_t place) { return args.rows_of[place]; }, args.stream_ends);
  cta_exclusive_ends<int64_t>(
      max_chunks, [&](int64_t place) { return args.reduce_tasks[place]; }, args.reduce_task_starts);
  const int64_t count = *args.num_chunks, total = stream_ends[count];
  if (threadIdx.x == 0 && count == 0) *args.header_end = 0;
  for (int64_t place = threadIdx.x; place < max_chunks; place += kThreads) {
    if (place >= count) {
      args.header_ends[place] = 0;
      continue;
    }
    int64_t start = stream_ends[place];
    if (total > ring) {
      const int64_t lap = stream_ends[place] / lap_room;
      const int64_t first = lower_bound(stream_ends, count, lap * lap_room);
      start = lap * ring + stream_ends[place] - stream_ends[first];
    }
    const int64_t end = start + stream_ends[place + 1] - stream_ends[place];
    Chunk &chunk = args.chunks[place];
    chunk.start = (uint32_t)start;
    chunk.end = (uint32_t)end;
    args.header_ends[place] = (uint32_t)end;
    if (place == count - 1) *args.header_end = (uint32_t)end;
    const int32_t uniq_lo = (int32_t)(chunk.key_lo - uniq_off[chunk.owner_node]);
    ((ChunkPiece *)(args.key_stage + args.stage_slice[chunk.owner_node]))[chunk.level] =
        ChunkPiece{uniq_lo, uniq_lo + chunk.keys, (uint32_t)start, (uint32_t)end};
  }
}

// One CTA: the chunks of the proxy's stream, their table and their places.
__global__ void __launch_bounds__(kThreads) stream_kernel(StreamArgs args) {
  chunk_counts(args);
  __syncthreads();
  chunk_table(args);
  __syncthreads();
  chunk_places(args);
}

// The duplicates' slots before key `key`, of the keys from `lo` (a key's
// rows but its key row): rows before it less one per key.
__device__ __forceinline__ int64_t duplicates_between(const int64_t *row_ends, int64_t lo,
                                                      int64_t key) {
  return row_ends[key] - row_ends[lo] - (key - lo);
}

// A thread per node-unique key: its chunk (from its level), and in the
// chunk's place in proxy_rows (its start % ring) its row and, after the
// chunk's keys' rows, the first slot of its duplicates (a chunk's rows: its
// keys' rows, then each key's duplicates' slots in key order); and its Record
// (KeyRecord, or WeightedKeyRecord with its weight) in its owner node's
// slice of key_stage, after the node's chunks. Its occurrences add to its
// chunk's pulls, a sum per chunk of a warp's keys.
template <typename Record>
__global__ void __launch_bounds__(kThreads)
    key_rows_kernel(Levels levels, const int32_t *key_node, const int32_t *key_count,
                    const int64_t *key_value, const float *key_weight, const int32_t *node_chunks,
                    const int64_t *node_chunk_start, const int64_t *stage_slice,
                    const int64_t *num_keys, int64_t max_chunks, int64_t ring, Chunk *chunks,
                    int32_t *key_chunk, int32_t *key_row, int32_t *duplicate_base,
                    char *key_stage) {
  const int64_t valid_keys = *num_keys;
  const int lane = threadIdx.x & 31;
  // a warp's keys at a time, so every lane takes part in the sums
  for (int64_t base = first_item() - lane; base < valid_keys; base += item_stride()) {
    const int64_t key = base + lane;
    int64_t place = -1;  // past the keys: none
    int32_t occurrences = 0;
    if (key < valid_keys) {
      const int64_t owner_node = key_node[key], first = levels.uniq_off[owner_node];
      const int64_t level = min((levels.row_ends[key] - levels.row_ends[first]) / levels.room,
                                max(levels.chunks_out[owner_node] - 1, (int64_t)0));
      place = clamp_index(
          node_chunks[clamp_index(node_chunk_start[owner_node] + level, max_chunks)], max_chunks);
      const Chunk &chunk = chunks[place];
      const int64_t at = (int64_t)chunk.start % ring, lo = chunk.key_lo;
      const int64_t keys_end = at + chunk.keys;  // the end of its keys' rows
      key_chunk[key] = (int32_t)place;
      key_row[key] = (int32_t)(at + key - lo);
      duplicate_base[key] = (int32_t)(keys_end + duplicates_between(levels.row_ends, lo, key));
      Record *records = (Record *)(key_stage + stage_slice[owner_node] +
                                   levels.chunks_out[owner_node] * (int64_t)sizeof(ChunkPiece));
      Record record{};
      record.key = key_value[key];
      if constexpr (Record::kWeighted) record.weight = key_weight[key];
      records[key - first] = record;
      occurrences = key_count[key];
    }
    // the lanes of each chunk: their sum, added once
    const unsigned same = __match_any_sync(0xffffffffu, place);
    const int32_t pulls = __reduce_add_sync(same, occurrences);
    if (place >= 0 && lane == __ffs(same) - 1) atomicAdd(&chunks[place].pulls, pulls);
  }
}

// A thread per valid sorted position: its occurrence's requester (from
// requester_start) and its slot: the key's row for its first occurrence, a
// duplicate slot for the others (an occurrence past the key's rows, in a call
// that raises, takes the key's row). remote_map gets the occurrence's key row
// and its slot.
__global__ void __launch_bounds__(kThreads)
    occurrence_slots_kernel(const int32_t *order, const int32_t *key_at, const int32_t *key_first,
                            const int32_t *key_row, const int32_t *duplicate_base,
                            const int64_t *row_ends, const int64_t *requester_start,
                            const int64_t *occurrence_count, int64_t local_size,
                            int32_t *requester_of, int32_t *map_key_row, int32_t *map_slot) {
  const int64_t count = *occurrence_count;
  for (int64_t position = first_item(); position < count; position += item_stride()) {
    const int32_t occurrence = order[position], key = key_at[position];
    const int64_t requester =
        min(max(upper_bound(requester_start, local_size + 1, occurrence) - 1, (int64_t)0),
            local_size - 1);
    const int64_t place = position - key_first[key];  // among the key's occurrences
    const int64_t rows = row_ends[key + 1] - row_ends[key];
    const int64_t slot =
        place == 0 || place >= rows ? key_row[key] : duplicate_base[key] + place - 1;
    requester_of[occurrence] = (int32_t)requester;
    map_key_row[occurrence] = key_row[key];
    map_slot[occurrence] = (int32_t)slot;
  }
}

// ── step 3: the push order and the reducer list ──

// A thread per sorted position: the push key of its occurrence, its part of
// the push order, requester * (max_chunks + 1) + chunk (past the valid
// occurrences local_size * (max_chunks + 1), which sorts them last); and
// `index` at `position`: the occurrences' own indices, which the sort
// carries.
__global__ void __launch_bounds__(kThreads)
    push_keys_kernel(const int32_t *order, const int32_t *key_at, const bool *valid,
                     const int32_t *requester_of, const int32_t *key_chunk, int64_t entries,
                     int64_t local_size, int64_t max_chunks, int64_t *push_key, int32_t *index) {
  for (int64_t position = first_item(); position < entries; position += item_stride()) {
    const int32_t occurrence = order[position];
    int64_t push = local_size * (max_chunks + 1);
    if (valid[occurrence])
      push = requester_of[occurrence] * (max_chunks + 1) + key_chunk[key_at[position]];
    push_key[occurrence] = push;
    index[position] = (int32_t)position;
  }
}

// A thread per push position (the occurrences sorted by push key): where the
// parts of the push order start (part_start; a part per requester and chunk
// up to max_chunks, in push key order). A valid position whose part is not
// the one before's starts its part and the parts in between, which have
// none; the last valid one ends the parts after its own, and the first
// thread every part when there is no valid one.
__global__ void __launch_bounds__(kThreads)
    part_starts_kernel(const int64_t *push_key_sorted, const int64_t *occurrence_count,
                       int64_t entries, int64_t parts, int32_t *part_start) {
  const int64_t count = *occurrence_count;
  if (first_item() == 0 && count == 0)
    for (int64_t part = 0; part < parts; ++part) part_start[part] = 0;
  for (int64_t position = first_item(); position < count; position += item_stride()) {
    const int64_t part = push_key_sorted[position];
    const int64_t before = position == 0 ? -1 : push_key_sorted[position - 1];
    for (int64_t next = before + 1; next <= part; ++next) part_start[next] = (int32_t)position;
    if (position == count - 1)
      for (int64_t next = part + 1; next < parts; ++next) part_start[next] = (int32_t)count;
  }
}

// One CTA: the push tables, a ReqEntry per requester and chunk up to
// max_chunks: where its positions start (part_start) and its first
// push-task flag. The flags go chunk after chunk (flag_start: the first of
// each), in a chunk requester after requester, a task per 32 rows; chunk
// max_chunks has none. push_chunks gets the entries in the positions of
// this proxy's push order; the proxy header gets them relative to each
// requester's push order (requester_start).
__global__ void __launch_bounds__(kThreads)
    push_tables_kernel(const int32_t *part_start, const int64_t *requester_start,
                       int64_t local_size, int64_t max_chunks, ReqEntry *push_chunks,
                       int64_t *flag_start, ReqEntry *header_tables) {
  const int64_t columns = max_chunks + 1;  // the entries of a requester
  for (int64_t index = threadIdx.x; index < local_size * columns; index += kThreads)
    push_chunks[index] = ReqEntry{part_start[index], 0};
  __syncthreads();
  // a requester's tasks in a chunk, of 32 rows
  auto tasks = [&](int64_t requester, int64_t chunk) -> int64_t {
    if (chunk >= max_chunks) return 0;
    const ReqEntry *entry = push_chunks + requester * columns + chunk;
    return ceil_div<kTaskRows>((int64_t)entry[1].start - entry[0].start);
  };
  cta_exclusive_ends<int64_t>(
      columns,
      [&](int64_t chunk) {
        int64_t flags = 0;
        for (int64_t requester = 0; requester < local_size; ++requester)
          flags += tasks(requester, chunk);
        return flags;
      },
      flag_start);
  // a thread per chunk: the flags, requester after requester; then the copies
  for (int64_t chunk = threadIdx.x; chunk < columns; chunk += kThreads) {
    int64_t flags = flag_start[chunk];
    for (int64_t requester = 0; requester < local_size; ++requester) {
      ReqEntry &entry = push_chunks[requester * columns + chunk];
      entry.flags = (int32_t)flags;
      flags += tasks(requester, chunk);
      header_tables[requester * columns + chunk] =
          ReqEntry{entry.start - (int32_t)requester_start[requester], entry.flags};
    }
  }
}

// A thread per valid push position: its occurrence's place in its
// requester's push order (remote_map's positions), and the flag of the push
// task that pushes it, from the ReqEntry of its part (requester and chunk).
__global__ void __launch_bounds__(kThreads)
    push_positions_kernel(const int64_t *push_key_sorted, const int32_t *push_order,
                          const ReqEntry *push_chunks, const int64_t *requester_start,
                          const int64_t *occurrence_count, int64_t max_chunks,
                          int32_t *map_position, int32_t *occurrence_flag) {
  const int64_t count = *occurrence_count;
  for (int64_t position = first_item(); position < count; position += item_stride()) {
    const int64_t part = push_key_sorted[position], requester = part / (max_chunks + 1);
    const ReqEntry entry = push_chunks[part];
    const int32_t occurrence = push_order[position];
    map_position[occurrence] = (int32_t)(position - requester_start[requester]);
    occurrence_flag[occurrence] = entry.flags + (int32_t)((position - entry.start) / kTaskRows);
  }
}

// A thread per valid sorted position: its occurrence's entry of the reducer
// list, at row `place` of its key's rows (its place among the key's
// occurrences): the flag of the push task that fills that row. An occurrence
// past its key's rows (a call that raises) has none.
__global__ void __launch_bounds__(kThreads)
    reducer_flags_kernel(const int32_t *order, const int32_t *key_at, const int32_t *key_first,
                         const int64_t *row_ends, const int32_t *occurrence_flag,
                         const int64_t *occurrence_count, int32_t *reducer_flags) {
  const int64_t count = *occurrence_count;
  for (int64_t position = first_item(); position < count; position += item_stride()) {
    const int32_t key = key_at[position];
    const int64_t row = row_ends[key] + position - key_first[key];
    if (row < row_ends[key + 1]) reducer_flags[row] = occurrence_flag[order[position]];
  }
}

}  // namespace

ProxyBuffers proxy_buffers(at::Tensor remote_map, at::Tensor key_stage, const ProxyHeader &header,
                           const at::TensorOptions &options) {
  const auto int_options = options.dtype(at::kInt);
  return ProxyBuffers{std::move(remote_map),
                      std::move(key_stage),
                      at::from_blob(header.chunks(), {1}, int_options),
                      at::from_blob(header.end(), {1}, int_options),
                      at::from_blob(header.ends(), {header.max_chunks}, int_options),
                      at::from_blob(header.push_table(0),
                                    {header.local_size, header.max_chunks + 1, 2}, int_options)};
}

ProxyKeys proxy_keys(const ProxyInput &input, const Sizes &sizes, bool node_dedup) {
  const int64_t entries = sizes.remote_keys, num_nodes = sizes.topo.num_nodes,
                num_features = sizes.num_features, num_tables = sizes.num_tables,
                bucket_features = num_nodes * num_features;
  const auto long_options = input.keys.options().dtype(at::kLong);
  const auto int_options = input.keys.options().dtype(at::kInt);
  const auto bool_options = input.keys.options().dtype(at::kBool);
  const cudaStream_t stream = current_stream();
  ProxyKeys keys;
  // the valid occurrences
  keys.occurrence_count = at::empty({}, long_options);
  keys.key_rows = at::empty({}, long_options);
  keys.valid = at::empty({entries}, bool_options);
  const at::Tensor index = at::empty({entries}, int_options);
  valid_occurrences_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      input.skip.data_ptr<bool>(), input.requester_start.data_ptr<int64_t>(), sizes.topo.local_size,
      entries, keys.occurrence_count.data_ptr<int64_t>(), keys.key_rows.data_ptr<int64_t>(),
      keys.valid.data_ptr<bool>(), index.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The identities (owner node, table, key): the occurrences sorted by key,
  // then stably by (owner node, table), the invalid ones last.
  const at::Tensor by_key = sort_pairs(input.keys, index, 64).second;
  const at::Tensor tag = at::empty({entries}, int_options);
  identity_tags_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      by_key.data_ptr<int32_t>(), keys.valid.data_ptr<bool>(),
      input.bucket_feature_of_occ.data_ptr<int32_t>(), input.tables.data_ptr<int32_t>(), entries,
      num_nodes, num_features, num_tables, tag.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  at::Tensor tag_sorted;
  std::tie(tag_sorted, keys.order) = sort_pairs(tag, by_key, bits_for(num_nodes * num_tables + 1));
  // The identity at each sorted position and where each identity starts.
  // Per identity: its smallest feature, and the sum of its weights, exact as
  // integers, when the records carry it (the keys carry weights then).
  const bool sum_weights = sizes.weighted_records();
  const at::Tensor starts_identity = at::empty({entries}, bool_options);
  const at::Tensor position_weight =
      sum_weights ? at::empty({entries}, long_options) : at::Tensor();
  const at::Tensor smallest_feature = at::empty({entries}, int_options);
  const auto launch_starts = [&](auto kernel) {
    kernel<<<item_grid(entries), kThreads, 0, stream>>>(
        keys.order.data_ptr<int32_t>(), tag_sorted.data_ptr<int32_t>(),
        input.keys.data_ptr<int64_t>(), data_or_null<float>(input.weights),
        keys.valid.data_ptr<bool>(), entries, num_features, starts_identity.data_ptr<bool>(),
        data_or_null<int64_t>(position_weight), smallest_feature.data_ptr<int32_t>());
  };
  if (node_dedup)
    launch_starts(identity_starts_kernel<true>);
  else
    launch_starts(identity_starts_kernel<false>);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const at::Tensor identity_ends = exclusive_ends(starts_identity, at::kInt);
  const at::Tensor weight_ends = sum_weights ? exclusive_ends(position_weight) : at::Tensor();
  const at::Tensor identity_of = at::empty({entries}, int_options);
  const at::Tensor identity_start = at::empty({entries + 1}, int_options);
  keys.num_keys = at::empty({}, long_options);
  identity_ranges_kernel<<<item_grid(entries + 1), kThreads, 0, stream>>>(
      identity_ends.data_ptr<int32_t>(), keys.order.data_ptr<int32_t>(),
      input.bucket_feature_of_occ.data_ptr<int32_t>(), keys.valid.data_ptr<bool>(),
      keys.occurrence_count.data_ptr<int64_t>(), entries, num_features,
      identity_of.data_ptr<int32_t>(), identity_start.data_ptr<int32_t>(),
      smallest_feature.data_ptr<int32_t>(), keys.num_keys.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The key index u: by bucket feature (owner node, smallest feature), then in
  // identity order (key order inside a bucket feature).
  const at::Tensor identity_bucket_feature = at::empty({entries}, int_options);
  identity_bucket_features_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      identity_start.data_ptr<int32_t>(), keys.order.data_ptr<int32_t>(),
      input.bucket_feature_of_occ.data_ptr<int32_t>(), smallest_feature.data_ptr<int32_t>(),
      keys.num_keys.data_ptr<int64_t>(), entries, num_nodes, num_features,
      identity_bucket_feature.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  at::Tensor bucket_feature_of_key, identity_of_key;
  std::tie(bucket_feature_of_key, identity_of_key) =
      sort_pairs(identity_bucket_feature, index, bits_for(bucket_features + 1));
  const at::Tensor key_offset_per_bucket_feature = at::empty({bucket_features + 1}, long_options);
  keys.uniq_off = at::empty({num_nodes + 1}, long_options);
  run_starts<int32_t, int64_t>(bucket_feature_of_key, bucket_features,
                               key_offset_per_bucket_feature, num_features, keys.uniq_off);
  // per key, and the key of each sorted position
  const at::Tensor key_of_identity = at::empty({entries}, int_options);
  keys.valid_key = at::empty({entries}, bool_options);
  keys.key_count = at::empty({entries}, int_options);
  keys.key_first = at::empty({entries}, int_options);
  keys.key_node = at::empty({entries}, int_options);
  keys.key_uniq = at::empty({entries}, int_options);
  keys.key_value = at::empty({entries}, long_options);
  if (sizes.weighted_records())
    keys.key_weight = at::empty({entries}, input.keys.options().dtype(at::kFloat));
  key_tables_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      identity_of_key.data_ptr<int32_t>(), bucket_feature_of_key.data_ptr<int32_t>(),
      identity_start.data_ptr<int32_t>(), keys.order.data_ptr<int32_t>(),
      input.keys.data_ptr<int64_t>(), data_or_null<int64_t>(weight_ends),
      keys.uniq_off.data_ptr<int64_t>(), keys.num_keys.data_ptr<int64_t>(), entries, num_nodes,
      num_features, key_of_identity.data_ptr<int32_t>(), keys.valid_key.data_ptr<bool>(),
      keys.key_count.data_ptr<int32_t>(), keys.key_first.data_ptr<int32_t>(),
      keys.key_node.data_ptr<int32_t>(), keys.key_uniq.data_ptr<int32_t>(),
      keys.key_value.data_ptr<int64_t>(), data_or_null<float>(keys.key_weight),
      keys.key_rows.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  keys.key_at = at::empty({entries}, int_options);
  position_keys_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      identity_of.data_ptr<int32_t>(), key_of_identity.data_ptr<int32_t>(), entries,
      keys.key_at.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // per bucket feature and per owner node
  keys.uniq_per_bucket_feature = at::empty({bucket_features}, long_options);
  keys.uniq_out = at::empty({num_nodes}, long_options);
  key_counts_kernel<<<kOneCta, kThreads, 0, stream>>>(
      key_offset_per_bucket_feature.data_ptr<int64_t>(), keys.uniq_off.data_ptr<int64_t>(),
      num_nodes, num_features, keys.uniq_per_bucket_feature.data_ptr<int64_t>(),
      keys.uniq_out.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return keys;
}

ProxyChunks proxy_chunks_and_slots(ProxyKeys &keys, const at::Tensor &requester_start,
                                   const ProxyBuffers &buffers, const Sizes &sizes) {
  const int64_t entries = sizes.remote_keys, num_nodes = sizes.topo.num_nodes,
                local_size = sizes.topo.local_size, max_chunks = sizes.max_chunks;
  const auto long_options = keys.order.options().dtype(at::kLong);
  const auto int_options = keys.order.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  ProxyChunks chunks;
  // The rows of each key, a row per occurrence (max_key_rows at most), and
  // the rows before each key.
  const at::Tensor rows = at::empty({entries}, int_options);
  key_row_counts_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      keys.valid_key.data_ptr<bool>(), keys.key_count.data_ptr<int32_t>(), entries,
      (int32_t)sizes.max_key_rows(), rows.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  chunks.row_ends = exclusive_ends(rows);
  // The chunks of the stream (each owner node's, by levels of its keys'
  // rows, level after level), where each lies, and key_stage's slices; the
  // stream's chunks and their ends to the proxy header too.
  chunks.chunks_out = at::empty({num_nodes}, long_options);
  chunks.node_chunk_start = at::empty({num_nodes + 1}, long_options);
  chunks.num_chunks = at::empty({}, long_options);
  chunks.stage_slice = at::empty({num_nodes}, long_options);
  chunks.chunks = at::empty({max_chunks, 8}, int_options);
  chunks.node_chunks = at::empty({max_chunks}, int_options);
  chunks.reduce_task_starts = at::empty({max_chunks + 1}, long_options);
  // each chunk's rows and reducer tasks, and the ends of its rows
  const at::Tensor stream_tables = at::empty({3, max_chunks + 1}, long_options);
  const Levels levels{chunks.row_ends.data_ptr<int64_t>(), keys.uniq_off.data_ptr<int64_t>(),
                      chunks.chunks_out.data_ptr<int64_t>(), num_nodes, sizes.chunk_room()};
  Chunk *chunk_table = records_of<Chunk>(chunks.chunks);
  StreamArgs stream_args;
  stream_args.levels = levels;
  stream_args.uniq_out = keys.uniq_out.data_ptr<int64_t>();
  stream_args.node = sizes.topo.node;
  stream_args.max_chunks = max_chunks;
  stream_args.record_bytes = sizes.record_bytes;
  stream_args.ring = sizes.num_proxy_rows;
  stream_args.lap_room = sizes.lap_room();
  stream_args.chunks_out = chunks.chunks_out.data_ptr<int64_t>();
  stream_args.node_chunk_start = chunks.node_chunk_start.data_ptr<int64_t>();
  stream_args.stage_slice = chunks.stage_slice.data_ptr<int64_t>();
  stream_args.num_chunks = chunks.num_chunks.data_ptr<int64_t>();
  stream_args.chunks = chunk_table;
  stream_args.node_chunks = chunks.node_chunks.data_ptr<int32_t>();
  stream_args.rows_of = stream_tables[0].data_ptr<int64_t>();
  stream_args.reduce_tasks = stream_tables[1].data_ptr<int64_t>();
  stream_args.stream_ends = stream_tables[2].data_ptr<int64_t>();
  stream_args.reduce_task_starts = chunks.reduce_task_starts.data_ptr<int64_t>();
  stream_args.header_chunks = buffers.chunks.data_ptr<int32_t>();
  stream_args.header_ends = (uint32_t *)buffers.ends.data_ptr<int32_t>();
  stream_args.header_end = (uint32_t *)buffers.end.data_ptr<int32_t>();
  stream_args.key_stage = (char *)buffers.key_stage.data_ptr<int64_t>();
  stream_kernel<<<kOneCta, kThreads, 0, stream>>>(stream_args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Per key: its chunk, its row and its duplicates' first slot there, and
  // its record; each chunk's pulls. The records' keys and weights go once
  // written.
  chunks.key_chunk = at::empty({entries}, int_options);
  const at::Tensor key_row = at::empty({entries}, int_options);
  const at::Tensor duplicate_base = at::empty({entries}, int_options);
  with_record(sizes, [&](auto record) {
    key_rows_kernel<decltype(record)><<<item_grid(entries), kThreads, 0, stream>>>(
        levels, keys.key_node.data_ptr<int32_t>(), keys.key_count.data_ptr<int32_t>(),
        keys.key_value.data_ptr<int64_t>(), data_or_null<float>(keys.key_weight),
        chunks.node_chunks.data_ptr<int32_t>(), chunks.node_chunk_start.data_ptr<int64_t>(),
        chunks.stage_slice.data_ptr<int64_t>(), keys.num_keys.data_ptr<int64_t>(), max_chunks,
        sizes.num_proxy_rows, chunk_table, chunks.key_chunk.data_ptr<int32_t>(),
        key_row.data_ptr<int32_t>(), duplicate_base.data_ptr<int32_t>(),
        (char *)buffers.key_stage.data_ptr<int64_t>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  keys.key_value.reset();
  keys.key_weight.reset();
  // Each occurrence's slot: the key's row for its first occurrence, a
  // duplicate slot for the others.
  chunks.requester_of = at::empty({entries}, int_options);
  const RemoteMap remote_map{buffers.remote_map.data_ptr<int32_t>(), entries};
  occurrence_slots_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      keys.order.data_ptr<int32_t>(), keys.key_at.data_ptr<int32_t>(),
      keys.key_first.data_ptr<int32_t>(), key_row.data_ptr<int32_t>(),
      duplicate_base.data_ptr<int32_t>(), chunks.row_ends.data_ptr<int64_t>(),
      requester_start.data_ptr<int64_t>(), keys.occurrence_count.data_ptr<int64_t>(), local_size,
      chunks.requester_of.data_ptr<int32_t>(), remote_map.key_row(), remote_map.slot());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return chunks;
}

ProxyPlan proxy_push_order(ProxyKeys keys, ProxyChunks chunks, const at::Tensor &requester_start,
                           const ProxyBuffers &buffers, const Sizes &sizes) {
  const int64_t entries = sizes.remote_keys, local_size = sizes.topo.local_size,
                max_chunks = sizes.max_chunks, parts = local_size * (max_chunks + 1);
  const auto long_options = keys.order.options().dtype(at::kLong);
  const auto int_options = keys.order.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  // The push order: one stable sort of the occurrences by their part; the
  // invalid ones after the last part. What only the push key needs goes
  // before the sort.
  at::Tensor push_key_sorted, push_order;
  {
    const at::Tensor push_key = at::empty({entries}, long_options);
    const at::Tensor index = at::empty({entries}, int_options);
    push_keys_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
        keys.order.data_ptr<int32_t>(), keys.key_at.data_ptr<int32_t>(),
        keys.valid.data_ptr<bool>(), chunks.requester_of.data_ptr<int32_t>(),
        chunks.key_chunk.data_ptr<int32_t>(), entries, local_size, max_chunks,
        push_key.data_ptr<int64_t>(), index.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    keys.valid.reset();
    keys.key_uniq.reset();
    keys.key_node.reset();
    chunks.requester_of.reset();
    chunks.key_chunk.reset();
    std::tie(push_key_sorted, push_order) = sort_pairs(push_key, index, bits_for(parts + 1));
  }
  // Where the parts of the push order start, then the push tables, per
  // requester and chunk: its first position and its first push-task flag.
  const at::Tensor part_start = at::empty({parts}, int_options);
  part_starts_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      push_key_sorted.data_ptr<int64_t>(), keys.occurrence_count.data_ptr<int64_t>(), entries,
      parts, part_start.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const at::Tensor push_chunks = at::empty({local_size, max_chunks + 1, 2}, int_options);
  const at::Tensor flag_start = at::empty({max_chunks + 2}, long_options);
  push_tables_kernel<<<kOneCta, kThreads, 0, stream>>>(
      part_start.data_ptr<int32_t>(), requester_start.data_ptr<int64_t>(), local_size, max_chunks,
      records_of<ReqEntry>(push_chunks), flag_start.data_ptr<int64_t>(),
      records_of<ReqEntry>(buffers.push_tables));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Each occurrence's place in its requester's push order (remote_map's
  // positions), and the flag of the push task that pushes it.
  const at::Tensor occurrence_flag = at::empty({entries}, int_options);
  const RemoteMap remote_map{buffers.remote_map.data_ptr<int32_t>(), entries};
  push_positions_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      push_key_sorted.data_ptr<int64_t>(), push_order.data_ptr<int32_t>(),
      records_of<ReqEntry>(push_chunks), requester_start.data_ptr<int64_t>(),
      keys.occurrence_count.data_ptr<int64_t>(), max_chunks, remote_map.position(),
      occurrence_flag.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The reducer list: per row of each key, in key order (its key row, then
  // its duplicates' slots), the flag of the push task that fills it.
  ProxyPlan plan;
  plan.reducer_start = chunks.row_ends;
  plan.reducer_flags = at::empty({entries}, int_options);
  reducer_flags_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      keys.order.data_ptr<int32_t>(), keys.key_at.data_ptr<int32_t>(),
      keys.key_first.data_ptr<int32_t>(), chunks.row_ends.data_ptr<int64_t>(),
      occurrence_flag.data_ptr<int32_t>(), keys.occurrence_count.data_ptr<int64_t>(),
      plan.reducer_flags.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  plan.num_chunks = chunks.num_chunks;
  plan.occurrence_count = keys.occurrence_count;
  plan.key_rows = keys.key_rows;
  plan.uniq_per_bucket_feature = keys.uniq_per_bucket_feature;
  plan.uniq_off = keys.uniq_off;
  plan.uniq_out = keys.uniq_out;
  plan.chunks = chunks.chunks;
  plan.chunks_out = chunks.chunks_out;
  plan.node_chunk_start = chunks.node_chunk_start;
  plan.node_chunks = chunks.node_chunks;
  plan.stage_slice = chunks.stage_slice;
  plan.reduce_task_starts = chunks.reduce_task_starts;
  return plan;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
