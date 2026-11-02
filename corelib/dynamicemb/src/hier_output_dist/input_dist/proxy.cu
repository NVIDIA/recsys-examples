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
// of its remote part, their rows, rounds and slots, and its node's push
// orders with its reducer list.

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

// An int32 table of 4-word records (RoundPiece, ReqEntry), as its records.
template <typename Record>
Record *records_of(const at::Tensor &table) {
  return reinterpret_cast<Record *>(table.data_ptr<int32_t>());
}

// ── step 1: the node-unique keys ──

// The owner node of a cell owner node * num_features + feature (the cell
// past the node-unique keys, num_nodes * num_features: the last node).
__device__ __forceinline__ int64_t owner_node_of(int64_t cell, int64_t num_features,
                                                 int64_t num_nodes) {
  return min(cell / num_features, num_nodes - 1);
}

// A thread per occurrence: whether it is valid, among the occurrences the
// node layout counts for this proxy (none when the node skips the call), and
// its own index, which the sort by key carries (and the sort by cell, as an
// identity's index). The first thread writes the count.
__global__ void __launch_bounds__(kThreads)
    valid_occurrences_kernel(const bool *skip, const int64_t *requester_start, int64_t local_size,
                             int64_t entries, int64_t *occurrence_count, bool *valid,
                             int32_t *index) {
  const int64_t count = *skip ? 0 : requester_start[local_size];
  if (first_item() == 0) *occurrence_count = count;
  for (int64_t occurrence = first_item(); occurrence < entries; occurrence += item_stride()) {
    valid[occurrence] = occurrence < count;
    index[occurrence] = (int32_t)occurrence;
  }
}

// A thread per position of the occurrences sorted by key: the identity tag
// of the occurrence there, owner node * num_tables + the table of its
// feature, or num_nodes * num_tables for an invalid one (which sorts it last).
__global__ void __launch_bounds__(kThreads)
    identity_tags_kernel(const int32_t *by_key, const bool *valid, const int32_t *cells,
                         const int32_t *tables, int64_t entries, int64_t num_nodes,
                         int64_t num_features, int64_t num_tables, int32_t *tag) {
  for (int64_t position = first_item(); position < entries; position += item_stride()) {
    const int32_t occurrence = by_key[position];
    int64_t identity_tag = num_nodes * num_tables;
    if (valid[occurrence]) {
      const int64_t cell = cells[occurrence];
      identity_tag =
          owner_node_of(cell, num_features, num_nodes) * num_tables + tables[cell % num_features];
    }
    tag[position] = (int32_t)identity_tag;
  }
}

// A thread per sorted position (the occurrences in identity order): whether
// an identity starts there, where the key or the tag differs from the
// position before (without node dedup: at every position); and when the
// records carry the weights, the weight of its occurrence as a whole number
// (0 when invalid), which the identities sum. It also sets the smallest
// feature of identity `position` above every feature, for the next kernel to
// lower.
__global__ void __launch_bounds__(kThreads)
    identity_starts_kernel(const int32_t *order, const int32_t *tag_sorted,
                           const int64_t *occurrence_keys, const float *weights, const bool *valid,
                           int64_t entries, int64_t num_features, bool node_dedup,
                           bool *starts_identity, int64_t *position_weight,
                           int32_t *smallest_feature) {
  for (int64_t position = first_item(); position < entries; position += item_stride()) {
    const int32_t occurrence = order[position];
    starts_identity[position] =
        !node_dedup || position == 0 ||
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
    identity_ranges_kernel(const int32_t *identity_ends, const int32_t *order, const int32_t *cells,
                           const bool *valid, const int64_t *occurrence_count, int64_t entries,
                           int64_t num_features, int32_t *identity_of, int32_t *identity_start,
                           int32_t *smallest_feature, int64_t *num_keys) {
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
    const int32_t feature = (int32_t)(cells[order[position]] % num_features);
    if (starts || feature < cells[order[position - 1]] % num_features)
      atomicMin(&smallest_feature[identity], feature);
  }
}

// A thread per identity: its cell, its owner node * num_features + its
// smallest feature; num_nodes * num_features past the node-unique keys,
// which sorts those last.
__global__ void __launch_bounds__(kThreads)
    identity_cells_kernel(const int32_t *identity_start, const int32_t *order, const int32_t *cells,
                          const int32_t *smallest_feature, const int64_t *num_keys, int64_t entries,
                          int64_t num_nodes, int64_t num_features, int32_t *identity_cell) {
  const int64_t valid_keys = *num_keys;
  for (int64_t identity = first_item(); identity < entries; identity += item_stride()) {
    int64_t cell = num_nodes * num_features;
    if (identity < valid_keys) {
      const int64_t first_cell = cells[order[identity_start[identity]]];
      cell = owner_node_of(first_cell, num_features, num_nodes) * num_features +
             smallest_feature[identity];
    }
    identity_cell[identity] = (int32_t)cell;
  }
}

// A thread per key index u (the identities sorted by cell): which key index
// its identity gets (key_of_identity, the inverse of the sort), and its
// entries of the per-key tables of ProxyKeys. Its record's weight: the sum of
// its occurrences' weights (from weight_ends at its positions), or their
// count.
__global__ void __launch_bounds__(kThreads)
    key_tables_kernel(const int32_t *identity_of_key, const int32_t *cell_of_key,
                      const int32_t *identity_start, const int32_t *order,
                      const int64_t *occurrence_keys, const int64_t *weight_ends,
                      const int64_t *uniq_off, const int64_t *num_keys, int64_t entries,
                      int64_t num_nodes, int64_t num_features, int32_t *key_of_identity,
                      bool *valid_key, int32_t *key_count, bool *key_heavy, int32_t *key_first,
                      int32_t *key_node, int32_t *key_uniq, int64_t *key_value, float *key_weight) {
  const int64_t valid_keys = *num_keys;
  for (int64_t key = first_item(); key < entries; key += item_stride()) {
    const int32_t identity = identity_of_key[key];
    const int32_t first = identity_start[identity], end = identity_start[identity + 1];
    const int64_t owner_node = owner_node_of(cell_of_key[key], num_features, num_nodes);
    key_of_identity[identity] = (int32_t)key;
    valid_key[key] = key < valid_keys;
    key_count[key] = end - first;
    key_heavy[key] = key < valid_keys && end - first > kHeavy;
    key_first[key] = first;
    key_node[key] = (int32_t)owner_node;
    key_uniq[key] = (int32_t)(key - uniq_off[owner_node]);
    key_value[key] = occurrence_keys[order[min(first, (int32_t)entries - 1)]];
    if (key_weight != nullptr)
      key_weight[key] = weight_ends != nullptr ? (float)(weight_ends[end] - weight_ends[first])
                                               : (float)(end - first);
  }
}

// A thread per sorted position: the key index of its occurrence's identity.
__global__ void __launch_bounds__(kThreads)
    position_keys_kernel(const int32_t *identity_of, const int32_t *key_of_identity,
                         int64_t entries, int32_t *key_at) {
  for (int64_t position = first_item(); position < entries; position += item_stride())
    key_at[position] = key_of_identity[identity_of[position]];
}

// One CTA: the node-unique keys of each cell and of each owner node, from
// where they start.
__global__ void __launch_bounds__(kThreads)
    key_counts_kernel(const int64_t *cell_key_start, const int64_t *uniq_off, int64_t num_nodes,
                      int64_t num_features, int64_t *cell_uniq, int64_t *uniq_out) {
  for (int64_t cell = threadIdx.x; cell < num_nodes * num_features; cell += kThreads)
    cell_uniq[cell] = cell_key_start[cell + 1] - cell_key_start[cell];
  for (int64_t owner_node = threadIdx.x; owner_node < num_nodes; owner_node += kThreads)
    uniq_out[owner_node] = uniq_off[owner_node + 1] - uniq_off[owner_node];
}

// ── step 2: the rows, the rounds and the slots ──

// A thread per key index: its rows in proxy_rows (one for each occurrence,
// or for a heavy key heavy_rows: its row and a partial of partial_rows for
// each requester) and its duplicates' slots (its occurrences but the first;
// none for a heavy key); 0 past the node-unique keys.
__global__ void __launch_bounds__(kThreads)
    key_row_counts_kernel(const bool *valid_key, const bool *key_heavy, const int32_t *key_count,
                          int64_t entries, int32_t heavy_rows, int32_t *rows, int32_t *duplicates) {
  for (int64_t key = first_item(); key < entries; key += item_stride()) {
    const bool valid = valid_key[key], heavy = key_heavy[key];
    rows[key] = !valid ? 0 : heavy ? heavy_rows : key_count[key];
    duplicates[key] = valid && !heavy ? key_count[key] - 1 : 0;
  }
}

// One CTA: the rounds, where each owner node's slice starts in key_stage,
// and the round table. One round when every row fits proxy_rows, else
// enough of 90% of a half (rounds_for). A slice, for each owner node but
// this one: its round pieces (16 bytes each), then its keys' records. The
// pieces cut each owner node's keys into rounds of equal rows, and never
// split a key: the piece of a round holds the keys [cut(round),
// cut(round + 1)), cut(round) the first key whose rows before it reach
// round / rounds of the owner node's rows. The pieces lie round after round
// in a half, owner node after owner node; those past the rounds, and this
// node's, are zeros. The rounds and the round table go to the proxy header
// too, and each piece to its slice.
__global__ void __launch_bounds__(kThreads)
    round_table_kernel(const int64_t *row_ends, const int64_t *uniq_off, const int64_t *uniq_out,
                       int64_t entries, int64_t num_nodes, int64_t node, int64_t rounds_cap,
                       int64_t proxy_rows, int64_t record_bytes, int64_t *num_rounds,
                       int64_t *stage_slice, RoundPiece *round_table, int32_t *header_rounds,
                       RoundPiece *header_table, char *key_stage) {
  const int64_t rounds = min(rounds_for(row_ends[entries], proxy_rows), rounds_cap);
  if (threadIdx.x == 0) {
    *num_rounds = rounds;
    *header_rounds = (int32_t)rounds;
  }
  for (int64_t owner_node = threadIdx.x; owner_node < num_nodes; owner_node += kThreads) {
    int64_t start = 0;
    for (int64_t before = 0; before < owner_node; ++before)
      if (before != node)
        start += rounds * (int64_t)sizeof(RoundPiece) + uniq_out[before] * record_bytes;
    stage_slice[owner_node] = start;
  }
  // cut(round) of an owner node's keys, among them
  auto cut = [&](int64_t round, int64_t owner_node) {
    const int64_t first_key = uniq_off[owner_node], base_rows = row_ends[first_key];
    const int64_t node_rows = row_ends[uniq_off[owner_node + 1]] - base_rows;
    const int64_t share = (min(round, rounds) * node_rows + rounds - 1) / rounds;
    const int64_t key = lower_bound(row_ends, entries + 1, base_rows + share) - first_key;
    return min(max(key, (int64_t)0), uniq_out[owner_node]);
  };
  // a thread per piece: its keys
  for (int64_t index = threadIdx.x; index < rounds_cap * num_nodes; index += kThreads) {
    const int64_t round = index / num_nodes, owner_node = index % num_nodes;
    RoundPiece piece{0, 0, 0, 0};
    if (round < rounds && owner_node != node) {
      piece.uniq_lo = (int32_t)cut(round, owner_node);
      piece.uniq_hi = (int32_t)cut(round + 1, owner_node);
    }
    round_table[index] = piece;
  }
  __syncthreads();
  // a thread per round: the offsets of its pieces, after the rows of those
  // before them; then the copies
  const int64_t half = rounds == 1 ? 0 : proxy_rows / 2;
  for (int64_t round = threadIdx.x; round < rounds_cap; round += kThreads) {
    int64_t off = (round & 1) * half;
    for (int64_t owner_node = 0; owner_node < num_nodes; ++owner_node) {
      RoundPiece &piece = round_table[round * num_nodes + owner_node];
      if (round < rounds && owner_node != node) {
        const int64_t first_key = uniq_off[owner_node];
        piece.off = (int32_t)off;
        off += row_ends[first_key + piece.uniq_hi] - row_ends[first_key + piece.uniq_lo];
        ((RoundPiece *)(key_stage + stage_slice[owner_node]))[round] = piece;
      }
      header_table[round * num_nodes + owner_node] = piece;
    }
  }
}

// A thread per node-unique key: its round, where its rows before it fall
// among its owner node's (the closed form of the cut), its piece and its row
// in proxy_rows; the first slot of its duplicates and its first partial row
// (a piece's rows: its keys' rows, then the duplicates' slots, then the
// partial rows of its heavy keys, heavy_partials each); and its record in
// its owner node's slice of key_stage: the key and, in 16 bytes, its
// weight's bits and 4 zero bytes.
__global__ void __launch_bounds__(kThreads)
    key_rows_kernel(const int64_t *row_ends, const int32_t *duplicate_ends,
                    const int32_t *heavy_ends, const int64_t *uniq_off, const int32_t *key_node,
                    const int32_t *key_uniq, const int64_t *key_value, const float *key_weight,
                    const RoundPiece *round_table, const int64_t *stage_slice,
                    const int64_t *num_rounds, const int64_t *num_keys, int64_t entries,
                    int64_t num_nodes, int64_t rounds_cap, int64_t heavy_partials,
                    int64_t record_bytes, int32_t *key_round, int32_t *key_lo, int32_t *key_row,
                    int32_t *duplicate_base, int32_t *partial_base, int64_t *key_stage) {
  const int64_t rounds = *num_rounds, valid_keys = *num_keys;
  for (int64_t key = first_item(); key < valid_keys; key += item_stride()) {
    const int64_t owner_node = key_node[key], first_key = uniq_off[owner_node];
    const int64_t base_rows = row_ends[first_key];
    const int64_t node_rows = row_ends[uniq_off[owner_node + 1]] - base_rows;
    const int64_t round =
        min(max((row_ends[key] - base_rows) * rounds / max(node_rows, (int64_t)1), (int64_t)0),
            rounds_cap - 1);
    const RoundPiece piece = round_table[round * num_nodes + owner_node];
    const int64_t uniq = key_uniq[key];
    key_round[key] = (int32_t)round;
    key_lo[key] = piece.uniq_lo;
    key_row[key] = (int32_t)(piece.off + uniq - piece.uniq_lo);
    // the piece's first and last keys, and the end of its keys' rows
    const int64_t piece_first = min(first_key + piece.uniq_lo, entries);
    const int64_t piece_last = min(first_key + piece.uniq_hi, entries);
    const int64_t piece_end = piece.off + piece.uniq_hi - piece.uniq_lo;
    duplicate_base[key] = (int32_t)(piece_end + duplicate_ends[key] - duplicate_ends[piece_first]);
    partial_base[key] =
        (int32_t)(piece_end + duplicate_ends[piece_last] - duplicate_ends[piece_first] +
                  (int64_t)(heavy_ends[key] - heavy_ends[piece_first]) * heavy_partials);
    int64_t *record = key_stage +
                      (stage_slice[owner_node] + rounds * (int64_t)sizeof(RoundPiece)) / 8 +
                      uniq * (record_bytes / 8);
    record[0] = key_value[key];
    if (record_bytes == 16) record[1] = (int64_t)__float_as_uint(key_weight[key]);
  }
}

// A thread per valid sorted position: its occurrence's requester (from
// requester_start) and its slot: for a heavy key its requester's partial;
// else the key's row for its first occurrence, a duplicate slot for the
// others. remote_map gets the occurrence's key row (bit 31: a heavy key) and
// its slot.
__global__ void __launch_bounds__(kThreads)
    occurrence_slots_kernel(const int32_t *order, const int32_t *key_at, const int32_t *key_first,
                            const bool *key_heavy, const int32_t *key_row,
                            const int32_t *duplicate_base, const int32_t *partial_base,
                            const int64_t *requester_start, const int64_t *occurrence_count,
                            int64_t local_size, int64_t partial_rows, int32_t *requester_of,
                            int32_t *slot_at, int32_t *map_key_row, int32_t *map_slot) {
  const int64_t count = *occurrence_count;
  for (int64_t position = first_item(); position < count; position += item_stride()) {
    const int32_t occurrence = order[position], key = key_at[position];
    const int64_t requester =
        min(max(upper_bound(requester_start, local_size + 1, occurrence) - 1, (int64_t)0),
            local_size - 1);
    const int64_t place = position - key_first[key];  // among the key's occurrences
    int64_t slot;
    if (key_heavy[key])
      slot = partial_base[key] + requester * partial_rows;
    else if (place == 0)
      slot = key_row[key];
    else
      slot = duplicate_base[key] + place - 1;
    requester_of[occurrence] = (int32_t)requester;
    slot_at[position] = (int32_t)slot;
    map_key_row[occurrence] = (int32_t)((uint32_t)key_row[key] | (uint32_t)key_heavy[key] << 31);
    map_slot[occurrence] = (int32_t)slot;
  }
}

// ── step 3: the push order and the reducer list ──

// A thread per sorted position: the push key of its occurrence (PushKey;
// requester local_size past the valid occurrences, which sorts them last),
// and `index` at `position`: the occurrences' own indices, which the sort
// carries.
__global__ void __launch_bounds__(kThreads)
    push_keys_kernel(const int32_t *order, const int32_t *key_at, const bool *valid,
                     const int32_t *requester_of, const bool *key_heavy, const int32_t *key_round,
                     const int32_t *key_uniq, const int32_t *key_lo, const int32_t *key_node,
                     int64_t entries, int64_t num_nodes, int64_t local_size, int64_t chunk_rows,
                     PushKey layout, int64_t *push_key, int32_t *index) {
  for (int64_t position = first_item(); position < entries; position += item_stride()) {
    const int32_t occurrence = order[position];
    int64_t push = local_size << layout.requester_shift;
    if (valid[occurrence]) {
      const int32_t key = key_at[position];
      const bool heavy = key_heavy[key];
      // a heavy key's index; else its IB chunk in its piece * num_nodes + its owner node
      const int64_t low =
          heavy ? key : (key_uniq[key] - key_lo[key]) / chunk_rows * num_nodes + key_node[key];
      push = ((int64_t)requester_of[occurrence] << layout.requester_shift) |
             ((int64_t)key_round[key] << layout.round_shift) |
             ((int64_t)heavy << layout.heavy_shift) | low;
    }
    push_key[occurrence] = push;
    index[position] = (int32_t)position;
  }
}

// A thread per push position (the occurrences sorted by push key): whether a
// heavy group starts there, at a valid occurrence of a heavy key whose push
// key the position before does not share.
__global__ void __launch_bounds__(kThreads)
    group_starts_kernel(const int64_t *push_key_sorted, const int64_t *occurrence_count,
                        int64_t entries, int heavy_shift, bool *starts_group) {
  const int64_t count = *occurrence_count;
  for (int64_t position = first_item(); position < entries; position += item_stride()) {
    const int64_t push = push_key_sorted[position];
    starts_group[position] = position < count && ((push >> heavy_shift) & 1) == 1 &&
                             (position == 0 || push != push_key_sorted[position - 1]);
  }
}

// One CTA: the push tables, a ReqEntry per requester and round up to
// rounds_cap: where its positions start (at the first push key of the
// requester and round), where its heavy positions start, its first push-task
// flag and its first heavy group (groups_before there). A round's flags:
// each requester's after the lower requesters', its tasks of 32 rows first,
// then its tasks of 32 heavy groups; round rounds_cap has none. push_rounds
// gets the entries in the positions and groups of this proxy's push order;
// the proxy header gets them relative to each requester's push order
// (requester_start) and to its first group, and push_start their starts.
__global__ void __launch_bounds__(kThreads)
    push_tables_kernel(const int64_t *push_key_sorted, const int32_t *groups_before,
                       const int64_t *requester_start, int64_t entries, int64_t local_size,
                       int64_t rounds_cap, PushKey layout, ReqEntry *push_rounds,
                       ReqEntry *header_tables, int32_t *push_start) {
  const int64_t columns = rounds_cap + 1;  // the entries of a requester
  // a thread per entry: where its push keys start
  for (int64_t index = threadIdx.x; index < local_size * columns; index += kThreads) {
    const int64_t requester = index / columns, round = index % columns;
    const int64_t round_key = (requester << layout.requester_shift) | (round << layout.round_shift);
    const int64_t start = lower_bound(push_key_sorted, entries, round_key);
    const int64_t heavy =
        lower_bound(push_key_sorted, entries, round_key | (int64_t{1} << layout.heavy_shift));
    push_rounds[index] = ReqEntry{(int32_t)start, (int32_t)heavy, 0, groups_before[start]};
  }
  __syncthreads();
  // a thread per round: the flags, requester after requester; then the copies
  for (int64_t round = threadIdx.x; round < columns; round += kThreads) {
    int32_t flags = 0;
    for (int64_t requester = 0; requester < local_size; ++requester) {
      ReqEntry &entry = push_rounds[requester * columns + round];
      if (round < rounds_cap) {
        const int32_t next_groups = push_rounds[requester * columns + round + 1].groups;
        entry.flags = flags;
        flags += ceil_div<kTaskRows>(entry.heavy - entry.start) +
                 ceil_div<kTaskRows>(next_groups - entry.groups);
      }
      const int32_t first = (int32_t)requester_start[requester];
      const int32_t first_group = push_rounds[requester * columns].groups;
      header_tables[requester * columns + round] = ReqEntry{
          entry.start - first, entry.heavy - first, entry.flags, entry.groups - first_group};
      push_start[requester * columns + round] = entry.start - first;
    }
  }
}

// A thread per valid push position: its occurrence's place in its
// requester's push order (remote_map's positions), and the flag of the push
// task that pushes it, from the ReqEntry of its requester and round.
__global__ void __launch_bounds__(kThreads)
    push_positions_kernel(const int64_t *push_key_sorted, const int32_t *push_order,
                          const ReqEntry *push_rounds, const int64_t *requester_start,
                          const int64_t *occurrence_count, int64_t rounds_cap, PushKey layout,
                          int32_t *map_position, int32_t *occurrence_flag) {
  const int64_t count = *occurrence_count;
  const int64_t round_mask = (int64_t{1} << (layout.requester_shift - layout.round_shift)) - 1;
  for (int64_t position = first_item(); position < count; position += item_stride()) {
    const int64_t push = push_key_sorted[position];
    const int64_t requester = push >> layout.requester_shift;
    const int64_t round = (push >> layout.round_shift) & round_mask;
    const ReqEntry entry = push_rounds[requester * (rounds_cap + 1) + round];
    const int32_t occurrence = push_order[position];
    map_position[occurrence] = (int32_t)(position - requester_start[requester]);
    occurrence_flag[occurrence] = entry.flags + (int32_t)((position - entry.start) / kTaskRows);
  }
}

// A thread per key index: its entries in the reducer list (one for each
// occurrence; for a heavy key its own row and a partial for each
// requester); 0 past the node-unique keys.
__global__ void __launch_bounds__(kThreads)
    reducer_counts_kernel(const bool *valid_key, const bool *key_heavy, const int32_t *key_count,
                          int64_t entries, int64_t local_size, int32_t *reducer_count) {
  for (int64_t key = first_item(); key < entries; key += item_stride())
    reducer_count[key] = !valid_key[key]  ? 0
                         : key_heavy[key] ? (int32_t)(1 + local_size)
                                          : key_count[key];
}

// A reducer entry: a slot word of proxy_rows low, the flag of its push task
// high (common.h).
__device__ __forceinline__ int64_t reducer_entry(uint32_t slot, int32_t flag) {
  return (int64_t)(((uint64_t)(uint32_t)flag << 32) | slot);
}

// A thread per node-unique key: its entries of the reducer list. A key with a
// row each: the slot of each of its
// occurrences, in order, with the occurrence's flag. A heavy key: its own
// row (kEntryOwnRow), then each requester's partial (kEntryPartial) with the
// flag of that requester's task of the key's heavy group, which a search for
// the group's push key finds (kEntryNoPush: the requester does not ask the
// key).
__global__ void __launch_bounds__(kThreads) reducer_entries_kernel(
    const int32_t *reducer_start, const bool *key_heavy, const int32_t *key_first,
    const int32_t *key_count, const int32_t *key_row, const int32_t *key_round,
    const int32_t *partial_base, const int32_t *slot_at, const int32_t *order,
    const int32_t *occurrence_flag, const int64_t *push_key_sorted, const int32_t *groups_before,
    const ReqEntry *push_rounds, const int64_t *num_keys, int64_t entries, int64_t local_size,
    int64_t rounds_cap, int64_t partial_rows, PushKey layout, int64_t *reducer_entries) {
  const int64_t valid_keys = *num_keys;
  for (int64_t key = first_item(); key < valid_keys; key += item_stride()) {
    int64_t entry = reducer_start[key];
    if (!key_heavy[key]) {
      const int64_t first = key_first[key];
      for (int64_t position = first; position < first + key_count[key]; ++position, ++entry)
        reducer_entries[entry] =
            reducer_entry((uint32_t)slot_at[position], occurrence_flag[order[position]]);
      continue;
    }
    reducer_entries[entry++] = reducer_entry((uint32_t)key_row[key], kEntryOwnRow);
    const int64_t round = key_round[key];
    for (int64_t requester = 0; requester < local_size; ++requester, ++entry) {
      const int64_t group_key = (requester << layout.requester_shift) |
                                (round << layout.round_shift) | (int64_t{1} << layout.heavy_shift) |
                                key;
      const int64_t group = lower_bound(push_key_sorted, entries, group_key);
      int32_t flag = kEntryNoPush;
      if (group < entries && push_key_sorted[group] == group_key) {
        const ReqEntry table = push_rounds[requester * (rounds_cap + 1) + round];
        flag = table.flags + ceil_div<kTaskRows>(table.heavy - table.start) +
               (groups_before[group] - table.groups) / kTaskRows;
      }
      const int64_t partial = partial_base[key] + requester * partial_rows;
      reducer_entries[entry] = reducer_entry((uint32_t)partial | kEntryPartial, flag);
    }
  }
}

}  // namespace

ProxyBuffers proxy_buffers(at::Tensor remote_map, at::Tensor key_stage, const ProxyHeader &header,
                           const at::TensorOptions &options) {
  const auto int_options = options.dtype(at::kInt);
  return ProxyBuffers{
      std::move(remote_map), std::move(key_stage), at::from_blob(header.rounds(), {1}, int_options),
      at::from_blob(header.round_table(), {header.rounds_cap, header.num_nodes, 4}, int_options),
      at::from_blob(header.push_table(0), {header.local_size, header.rounds_cap + 1, 4},
                    int_options)};
}

ProxyKeys proxy_keys(const ProxyInput &input, const Sizes &sizes, bool node_dedup) {
  const int64_t entries = sizes.remote_keys, num_nodes = sizes.topo.num_nodes,
                num_features = sizes.num_features, num_tables = sizes.num_tables,
                cells = num_nodes * num_features;
  const auto long_options = input.keys.options().dtype(at::kLong);
  const auto int_options = input.keys.options().dtype(at::kInt);
  const auto bool_options = input.keys.options().dtype(at::kBool);
  const cudaStream_t stream = current_stream();
  ProxyKeys keys;
  // the valid occurrences
  keys.occurrence_count = at::empty({}, long_options);
  keys.valid = at::empty({entries}, bool_options);
  const at::Tensor index = at::empty({entries}, int_options);
  valid_occurrences_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      input.skip.data_ptr<bool>(), input.requester_start.data_ptr<int64_t>(), sizes.topo.local_size,
      entries, keys.occurrence_count.data_ptr<int64_t>(), keys.valid.data_ptr<bool>(),
      index.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The identities (owner node, table, key): the occurrences sorted by key,
  // then stably by (owner node, table), the invalid ones last.
  const at::Tensor by_key = sort_pairs(input.keys, index, 64).second;
  const at::Tensor tag = at::empty({entries}, int_options);
  identity_tags_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      by_key.data_ptr<int32_t>(), keys.valid.data_ptr<bool>(), input.cells.data_ptr<int32_t>(),
      input.tables.data_ptr<int32_t>(), entries, num_nodes, num_features, num_tables,
      tag.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  at::Tensor tag_sorted;
  std::tie(tag_sorted, keys.order) = sort_pairs(tag, by_key, bits_for(num_nodes * num_tables + 1));
  // The identity at each sorted position and where each identity starts.
  // Per identity: its smallest feature, and the sum of its weights, exact as
  // integers, when the records carry it.
  const bool sum_weights = input.weights.defined() && sizes.record_bytes == 16;
  const at::Tensor starts_identity = at::empty({entries}, bool_options);
  const at::Tensor position_weight =
      sum_weights ? at::empty({entries}, long_options) : at::Tensor();
  const at::Tensor smallest_feature = at::empty({entries}, int_options);
  identity_starts_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      keys.order.data_ptr<int32_t>(), tag_sorted.data_ptr<int32_t>(),
      input.keys.data_ptr<int64_t>(), data_or_null<float>(input.weights),
      keys.valid.data_ptr<bool>(), entries, num_features, node_dedup,
      starts_identity.data_ptr<bool>(), data_or_null<int64_t>(position_weight),
      smallest_feature.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const at::Tensor identity_ends = exclusive_ends(starts_identity, at::kInt);
  const at::Tensor weight_ends = sum_weights ? exclusive_ends(position_weight) : at::Tensor();
  const at::Tensor identity_of = at::empty({entries}, int_options);
  const at::Tensor identity_start = at::empty({entries + 1}, int_options);
  keys.num_keys = at::empty({}, long_options);
  identity_ranges_kernel<<<item_grid(entries + 1), kThreads, 0, stream>>>(
      identity_ends.data_ptr<int32_t>(), keys.order.data_ptr<int32_t>(),
      input.cells.data_ptr<int32_t>(), keys.valid.data_ptr<bool>(),
      keys.occurrence_count.data_ptr<int64_t>(), entries, num_features,
      identity_of.data_ptr<int32_t>(), identity_start.data_ptr<int32_t>(),
      smallest_feature.data_ptr<int32_t>(), keys.num_keys.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The key index u: by cell (owner node, smallest feature), then in
  // identity order (key order inside a cell).
  const at::Tensor identity_cell = at::empty({entries}, int_options);
  identity_cells_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      identity_start.data_ptr<int32_t>(), keys.order.data_ptr<int32_t>(),
      input.cells.data_ptr<int32_t>(), smallest_feature.data_ptr<int32_t>(),
      keys.num_keys.data_ptr<int64_t>(), entries, num_nodes, num_features,
      identity_cell.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  at::Tensor cell_of_key, identity_of_key;
  std::tie(cell_of_key, identity_of_key) = sort_pairs(identity_cell, index, bits_for(cells + 1));
  const at::Tensor cell_key_start = at::empty({cells + 1}, long_options);
  keys.uniq_off = at::empty({num_nodes + 1}, long_options);
  run_starts<int32_t, int64_t>(cell_of_key, cells, cell_key_start, num_features, keys.uniq_off);
  // per key, and the key of each sorted position
  const at::Tensor key_of_identity = at::empty({entries}, int_options);
  keys.valid_key = at::empty({entries}, bool_options);
  keys.key_count = at::empty({entries}, int_options);
  keys.key_heavy = at::empty({entries}, bool_options);
  keys.key_first = at::empty({entries}, int_options);
  keys.key_node = at::empty({entries}, int_options);
  keys.key_uniq = at::empty({entries}, int_options);
  keys.key_value = at::empty({entries}, long_options);
  if (sizes.record_bytes == 16)
    keys.key_weight = at::empty({entries}, input.keys.options().dtype(at::kFloat));
  key_tables_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      identity_of_key.data_ptr<int32_t>(), cell_of_key.data_ptr<int32_t>(),
      identity_start.data_ptr<int32_t>(), keys.order.data_ptr<int32_t>(),
      input.keys.data_ptr<int64_t>(), data_or_null<int64_t>(weight_ends),
      keys.uniq_off.data_ptr<int64_t>(), keys.num_keys.data_ptr<int64_t>(), entries, num_nodes,
      num_features, key_of_identity.data_ptr<int32_t>(), keys.valid_key.data_ptr<bool>(),
      keys.key_count.data_ptr<int32_t>(), keys.key_heavy.data_ptr<bool>(),
      keys.key_first.data_ptr<int32_t>(), keys.key_node.data_ptr<int32_t>(),
      keys.key_uniq.data_ptr<int32_t>(), keys.key_value.data_ptr<int64_t>(),
      data_or_null<float>(keys.key_weight));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  keys.key_at = at::empty({entries}, int_options);
  position_keys_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      identity_of.data_ptr<int32_t>(), key_of_identity.data_ptr<int32_t>(), entries,
      keys.key_at.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // per cell and per owner node
  keys.cell_uniq = at::empty({cells}, long_options);
  keys.uniq_out = at::empty({num_nodes}, long_options);
  key_counts_kernel<<<kOneCta, kThreads, 0, stream>>>(
      cell_key_start.data_ptr<int64_t>(), keys.uniq_off.data_ptr<int64_t>(), num_nodes,
      num_features, keys.cell_uniq.data_ptr<int64_t>(), keys.uniq_out.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return keys;
}

ProxyRounds proxy_rounds_and_slots(ProxyKeys &keys, const at::Tensor &requester_start,
                                   const ProxyBuffers &buffers, const Sizes &sizes) {
  const int64_t entries = sizes.remote_keys, num_nodes = sizes.topo.num_nodes,
                local_size = sizes.topo.local_size, rounds_cap = sizes.rounds_cap,
                partial_rows = sizes.partial_rows, record_bytes = sizes.record_bytes;
  const auto long_options = keys.order.options().dtype(at::kLong);
  const auto int_options = keys.order.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  ProxyRounds rounds;
  // The rows of each key: one for each occurrence, or a heavy key's row and
  // a partial of partial_rows for each requester; and its duplicates' slots.
  const at::Tensor rows = at::empty({entries}, int_options);
  const at::Tensor duplicates = at::empty({entries}, int_options);
  key_row_counts_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      keys.valid_key.data_ptr<bool>(), keys.key_heavy.data_ptr<bool>(),
      keys.key_count.data_ptr<int32_t>(), entries, (int32_t)(1 + local_size * partial_rows),
      rows.data_ptr<int32_t>(), duplicates.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const at::Tensor row_ends = exclusive_ends(rows);
  const at::Tensor duplicate_ends = exclusive_ends(duplicates, at::kInt);
  const at::Tensor heavy_ends = exclusive_ends(keys.key_heavy, at::kInt);
  // The rounds and their pieces: cuts of equal rows of each owner node's
  // keys, which never split a key; round after round in a half, owner node
  // after owner node. key_stage: for each owner node but this one a slice,
  // its round pieces, then its keys' records. The proxy header: the rounds,
  // then the round table.
  rounds.num_rounds = at::empty({}, long_options);
  rounds.stage_slice = at::empty({num_nodes}, long_options);
  rounds.round_table = at::empty({rounds_cap, num_nodes, 4}, int_options);
  round_table_kernel<<<kOneCta, kThreads, 0, stream>>>(
      row_ends.data_ptr<int64_t>(), keys.uniq_off.data_ptr<int64_t>(),
      keys.uniq_out.data_ptr<int64_t>(), entries, num_nodes, sizes.topo.node, rounds_cap,
      sizes.num_proxy_rows, record_bytes, rounds.num_rounds.data_ptr<int64_t>(),
      rounds.stage_slice.data_ptr<int64_t>(), records_of<RoundPiece>(rounds.round_table),
      buffers.rounds.data_ptr<int32_t>(), records_of<RoundPiece>(buffers.round_table),
      (char *)buffers.key_stage.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Per key: its round, its piece and its row, and its record. A piece's
  // rows: its keys' rows, then the duplicates' slots, then the partial rows
  // of its heavy keys, local_size blocks of partial_rows each. The records'
  // keys and weights go once written.
  rounds.key_round = at::empty({entries}, int_options);
  rounds.key_lo = at::empty({entries}, int_options);
  rounds.key_row = at::empty({entries}, int_options);
  rounds.partial_base = at::empty({entries}, int_options);
  const at::Tensor duplicate_base = at::empty({entries}, int_options);
  key_rows_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      row_ends.data_ptr<int64_t>(), duplicate_ends.data_ptr<int32_t>(),
      heavy_ends.data_ptr<int32_t>(), keys.uniq_off.data_ptr<int64_t>(),
      keys.key_node.data_ptr<int32_t>(), keys.key_uniq.data_ptr<int32_t>(),
      keys.key_value.data_ptr<int64_t>(), data_or_null<float>(keys.key_weight),
      records_of<RoundPiece>(rounds.round_table), rounds.stage_slice.data_ptr<int64_t>(),
      rounds.num_rounds.data_ptr<int64_t>(), keys.num_keys.data_ptr<int64_t>(), entries, num_nodes,
      rounds_cap, local_size * partial_rows, record_bytes, rounds.key_round.data_ptr<int32_t>(),
      rounds.key_lo.data_ptr<int32_t>(), rounds.key_row.data_ptr<int32_t>(),
      duplicate_base.data_ptr<int32_t>(), rounds.partial_base.data_ptr<int32_t>(),
      buffers.key_stage.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  keys.key_value.reset();
  keys.key_weight.reset();
  // Each occurrence's slot: for a heavy key its requester's partial; else
  // the key's row for its first occurrence, a duplicate slot for the others.
  rounds.requester_of = at::empty({entries}, int_options);
  rounds.slot_at = at::empty({entries}, int_options);
  const RemoteMap remote_map{buffers.remote_map.data_ptr<int32_t>(), entries};
  occurrence_slots_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      keys.order.data_ptr<int32_t>(), keys.key_at.data_ptr<int32_t>(),
      keys.key_first.data_ptr<int32_t>(), keys.key_heavy.data_ptr<bool>(),
      rounds.key_row.data_ptr<int32_t>(), duplicate_base.data_ptr<int32_t>(),
      rounds.partial_base.data_ptr<int32_t>(), requester_start.data_ptr<int64_t>(),
      keys.occurrence_count.data_ptr<int64_t>(), local_size, partial_rows,
      rounds.requester_of.data_ptr<int32_t>(), rounds.slot_at.data_ptr<int32_t>(),
      remote_map.key_row(), remote_map.slot());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return rounds;
}

PushKey push_key_of(const Sizes &sizes) {
  // a piece holds at most remote_keys keys: at most this many IB chunks
  const int64_t chunks = sizes.remote_keys / sizes.chunk_rows + 1;
  PushKey key;
  key.heavy_shift = std::max(bits_for(sizes.remote_keys), bits_for(chunks * sizes.topo.num_nodes));
  key.round_shift = key.heavy_shift + 1;
  key.requester_shift = key.round_shift + bits_for(sizes.rounds_cap + 1);
  key.bits = key.requester_shift + bits_for(sizes.topo.local_size + 1);
  return key;
}

ProxyPlan proxy_push_order(ProxyKeys keys, ProxyRounds rounds, const at::Tensor &requester_start,
                           const ProxyBuffers &buffers, const Sizes &sizes) {
  const int64_t entries = sizes.remote_keys, num_nodes = sizes.topo.num_nodes,
                local_size = sizes.topo.local_size, rounds_cap = sizes.rounds_cap;
  const auto long_options = keys.order.options().dtype(at::kLong);
  const auto int_options = keys.order.options().dtype(at::kInt);
  const auto bool_options = keys.order.options().dtype(at::kBool);
  const cudaStream_t stream = current_stream();
  const PushKey layout = push_key_of(sizes);
  // The push order: one stable sort of the occurrences by their push key;
  // the invalid ones after requester local_size - 1. What only the push key
  // needs goes before the sort.
  at::Tensor push_key_sorted, push_order;
  {
    const at::Tensor push_key = at::empty({entries}, long_options);
    const at::Tensor index = at::empty({entries}, int_options);
    push_keys_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
        keys.order.data_ptr<int32_t>(), keys.key_at.data_ptr<int32_t>(),
        keys.valid.data_ptr<bool>(), rounds.requester_of.data_ptr<int32_t>(),
        keys.key_heavy.data_ptr<bool>(), rounds.key_round.data_ptr<int32_t>(),
        keys.key_uniq.data_ptr<int32_t>(), rounds.key_lo.data_ptr<int32_t>(),
        keys.key_node.data_ptr<int32_t>(), entries, num_nodes, local_size, sizes.chunk_rows, layout,
        push_key.data_ptr<int64_t>(), index.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    keys.valid.reset();
    keys.key_at.reset();
    keys.key_uniq.reset();
    keys.key_node.reset();
    rounds.key_lo.reset();
    rounds.requester_of.reset();
    std::tie(push_key_sorted, push_order) = sort_pairs(push_key, index, layout.bits);
  }
  // The heavy groups, numbered in push order (a group: the positions of one
  // requester, round and heavy key).
  const at::Tensor starts_group = at::empty({entries}, bool_options);
  group_starts_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      push_key_sorted.data_ptr<int64_t>(), keys.occurrence_count.data_ptr<int64_t>(), entries,
      layout.heavy_shift, starts_group.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const at::Tensor groups_before = exclusive_ends(starts_group, at::kInt);
  // The push tables, per requester and round: its first position, its first
  // heavy position, its first push-task flag, and its first heavy group.
  ProxyPlan plan;
  const at::Tensor push_rounds = at::empty({local_size, rounds_cap + 1, 4}, int_options);
  plan.push_start = at::empty({local_size, rounds_cap + 1}, int_options);
  push_tables_kernel<<<kOneCta, kThreads, 0, stream>>>(
      push_key_sorted.data_ptr<int64_t>(), groups_before.data_ptr<int32_t>(),
      requester_start.data_ptr<int64_t>(), entries, local_size, rounds_cap, layout,
      records_of<ReqEntry>(push_rounds), records_of<ReqEntry>(buffers.push_tables),
      plan.push_start.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Each occurrence's place in its requester's push order (remote_map's
  // positions), and the flag of the push task that pushes it.
  const at::Tensor occurrence_flag = at::empty({entries}, int_options);
  const RemoteMap remote_map{buffers.remote_map.data_ptr<int32_t>(), entries};
  push_positions_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      push_key_sorted.data_ptr<int64_t>(), push_order.data_ptr<int32_t>(),
      records_of<ReqEntry>(push_rounds), requester_start.data_ptr<int64_t>(),
      keys.occurrence_count.data_ptr<int64_t>(), rounds_cap, layout, remote_map.position(),
      occurrence_flag.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The reducer list: a key's entries, in occurrence order for a key with a
  // row each; for a heavy key its own row, then each requester's partial.
  // Each entry is a slot and the flag of the push task that fills it.
  const at::Tensor reducer_count = at::empty({entries}, int_options);
  reducer_counts_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      keys.valid_key.data_ptr<bool>(), keys.key_heavy.data_ptr<bool>(),
      keys.key_count.data_ptr<int32_t>(), entries, local_size, reducer_count.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  plan.reducer_start = exclusive_ends(reducer_count, at::kInt);
  // A heavy key has more than kHeavy occurrences and 1 + local_size entries:
  // with more GPUs per node than kHeavy, more entries than occurrences.
  const int64_t heavy_keys = entries / (kHeavy + 1);
  plan.reducer_entries =
      at::empty({entries + heavy_keys * std::max<int64_t>(local_size - kHeavy, 0)}, long_options);
  reducer_entries_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      plan.reducer_start.data_ptr<int32_t>(), keys.key_heavy.data_ptr<bool>(),
      keys.key_first.data_ptr<int32_t>(), keys.key_count.data_ptr<int32_t>(),
      rounds.key_row.data_ptr<int32_t>(), rounds.key_round.data_ptr<int32_t>(),
      rounds.partial_base.data_ptr<int32_t>(), rounds.slot_at.data_ptr<int32_t>(),
      keys.order.data_ptr<int32_t>(), occurrence_flag.data_ptr<int32_t>(),
      push_key_sorted.data_ptr<int64_t>(), groups_before.data_ptr<int32_t>(),
      records_of<ReqEntry>(push_rounds), keys.num_keys.data_ptr<int64_t>(), entries, local_size,
      rounds_cap, sizes.partial_rows, layout, plan.reducer_entries.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  plan.num_rounds = rounds.num_rounds;
  plan.occurrence_count = keys.occurrence_count;
  plan.cell_uniq = keys.cell_uniq;
  plan.uniq_off = keys.uniq_off;
  plan.uniq_out = keys.uniq_out;
  plan.round_table = rounds.round_table;
  plan.stage_slice = rounds.stage_slice;
  return plan;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
