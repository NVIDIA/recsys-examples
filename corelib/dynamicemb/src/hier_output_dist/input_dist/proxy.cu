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
// and its node's push orders with their push tables.

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/input_dist/planner.cuh"
#include "hier_output_dist/input_dist/planner.h"
#include "murmur_hash.cuh"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

// ── step 1: the node-unique keys ──
//
// A hash table finds each occurrence's identity. Integer atomics give the
// identity its representative, its occurrences, a list of them and the sum of
// their weights. A scan of the representatives and their counts per
// (requester, bucket feature) block then number the node-unique keys. No
// result depends on the order of the atomics: the representative is a
// minimum, the counts and weights are integer sums, and a list is read as a
// set. Nor on the hash: no step reads which slot an identity took.

// The owner node of a bucket feature owner node * num_features + feature
// (clamped to the last node: an index).
__device__ __forceinline__ int64_t owner_node_of(int64_t bucket_feature, int64_t num_features,
                                                 int64_t num_nodes) {
  return min(bucket_feature / num_features, num_nodes - 1);
}

// The requester of a valid occurrence: the last one whose occurrences start
// at or before it (requester_start: local_size + 1 entries).
__device__ __forceinline__ int64_t requester_of_occurrence(const int64_t *requester_start,
                                                           int64_t local_size, int64_t occurrence) {
  return min(max(upper_bound(requester_start, local_size + 1, occurrence) - 1, (int64_t)0),
             local_size - 1);
}

// An identity's representative is its occurrence of the smallest bucket
// feature, and of those the first. As the word bucket feature << 32 |
// occurrence (both below 2^31), it is the least word of the identity's
// occurrences, which an atomicMin finds in any order.
__device__ __forceinline__ unsigned long long representative_word(int64_t bucket_feature,
                                                                  int64_t occurrence) {
  return (unsigned long long)bucket_feature << 32 | (unsigned long long)occurrence;
}
__device__ __forceinline__ int64_t representative_of(unsigned long long word) {
  return (int64_t)(word & 0xffffffffull);
}
__device__ __forceinline__ int64_t representative_bucket_feature(unsigned long long word) {
  return (int64_t)(word >> 32);
}

// The hash table of a call's identities (table, key) under node dedup: open
// addressing with linear probing over 2^slot_bits slots, at least
// 2 * remote_keys. A slot is empty (-1) or holds the occurrence that took it
// for its identity. That occurrence names the identity, and the identity's
// state is at its index. proxy_keys allocates the table for the call, and
// prepare_kernel clears it.
struct Identities {
  int32_t *slot;  // [2^slot_bits]
  int slot_bits;
  // per identity (an occurrence that took a slot)
  unsigned long long *rep;  // [remote_keys]: its representative (representative_word)
  int32_t *count;           // [remote_keys]: its occurrences
  int32_t *head;            // [remote_keys]: the last of them in its list (-1: none)
  int64_t *weight;          // [remote_keys]: the sum of their weights, or nullptr
  // per occurrence
  int32_t *next;         // [remote_keys]: the one before it in its identity's list (-1: none)
  int32_t *identity_of;  // [remote_keys]: its identity
};

// The seeds of identity_slot's hash.
constexpr uint64_t kKeySeed = 0x9e3779b97f4a7c15ull, kTableSeed = 0xc2b2ae3d27d4eb4full;

// Where the probe for identity (table, key) starts. The owner rule fixes some
// bits for all keys of one owner: under roundrobin the key's low bits (key %
// world_size), under hash_roundrobin the low bits of murmur3_fmix64(key) (its
// hash % world_size). All of a proxy's keys share them up to the largest
// power of two dividing local_size (that value % local_size is the proxy's
// rail), at any world size. So the slot comes from neither. The hash is
// fmix64 of the key xor a seed, a function other than the rule's, xor the
// table times another seed (equal keys of two tables go apart). The slot is
// its high bits, which every bit of the key moves: they are those of fmix64's
// last product (its last step changes only the low 31 bits), and only a
// factor's low bits move a product's low bits.
__device__ __forceinline__ uint64_t identity_slot(int64_t key, int32_t table, int slot_bits) {
  const uint64_t hash = murmur3_fmix64((uint64_t)key ^ kKeySeed) ^ (uint64_t)table * kTableSeed;
  return hash >> (64 - slot_bits);
}

// The int64 words of the blocks' tables (prepare_kernel, blocks_kernel): per
// requester where its blocks start and where its range ends, per (requester,
// bucket feature) a base, and per bucket feature where its keys start, and
// their end.
inline int64_t block_scratch(int64_t local_size, int64_t bucket_features) {
  return local_size * (2 * bucket_features + 1) + bucket_features + 1;
}

// A thread per slot of the hash table (`slots`; none without node dedup) and
// per (requester, bucket feature) block of the remote part. The table of this
// call is cleared: every slot empty, and each identity's state without a
// representative, occurrences, a list or weights. A block is the occurrences
// of a bucket feature in a requester's range. Inside a requester's range the
// occurrences come by owner node, then feature, then position (node.cuh:
// row_shift puts a row of owner (n, t) after the requester's rows for
// the owner nodes before n, at its offset in its bucket, which is
// feature-major). So node_bucket_feature_of_row does not decrease there, and a
// binary search finds where each block starts: block_start [local_size]
// [bucket_features + 1], each requester's last entry the end of its range.
// The first thread writes the count of the valid occurrences, those the node
// layout counts for this proxy (none when the node skips the call), and
// key_rows' start.
__global__ void __launch_bounds__(kThreads)
    prepare_kernel(const bool *skip, const int64_t *requester_start,
                   const int32_t *node_bucket_feature_of_row, int64_t local_size,
                   int64_t bucket_features, int64_t entries, int64_t slots, Identities ids,
                   int64_t *block_start, int64_t *occurrence_count, int64_t *key_rows) {
  const int64_t count = *skip ? 0 : requester_start[local_size], columns = bucket_features + 1;
  if (first_item() == 0) {
    *occurrence_count = count;
    *key_rows = 0;
  }
  for (int64_t index = first_item(); index < slots; index += item_stride()) {
    ids.slot[index] = -1;
    if (index >= entries) continue;
    ids.rep[index] = ~0ull;
    ids.count[index] = 0;
    ids.head[index] = -1;
    if (ids.weight != nullptr) ids.weight[index] = 0;
  }
  for (int64_t index = first_item(); index < local_size * columns; index += item_stride()) {
    const int64_t requester = index / columns, bucket_feature = index % columns;
    const int64_t lo = min(requester_start[requester], count),
                  hi = max(min(requester_start[requester + 1], count), lo);
    block_start[index] =
        bucket_feature == bucket_features
            ? hi
            : lo + lower_bound(node_bucket_feature_of_row + lo, hi - lo, bucket_feature);
  }
}

// A thread per valid occurrence, under node dedup: its identity (table, key),
// the table of its feature (dense_tables). The probe goes from identity_slot
// to the identity's slot, or to the first empty one, which it takes. Of the
// occurrences that race for an empty slot, one takes it (atomicCAS) and the
// others compare their identity with its. A plain load of a slot sees -1 or
// its final occurrence: a stale -1 only sends the thread to the atomicCAS.
// The table has at least twice the slots of the occurrences, so a probe ends.
// Then, on the identity: its representative (atomicMin), the occurrence
// counted (atomicAdd) and put first in the list (atomicExch of its head: the
// occurrence's `next` is the old head), and its weight, a whole number, added
// as an int64.
__global__ void __launch_bounds__(kThreads)
    insert_kernel(const int64_t *occurrence_keys, const float *weights,
                  const int32_t *node_bucket_feature_of_row, const int32_t *tables,
                  const int64_t *occurrence_count, int64_t num_features, Identities ids) {
  const int64_t count = *occurrence_count;
  const uint64_t last_slot = (1ull << ids.slot_bits) - 1;
  for (int64_t occurrence = first_item(); occurrence < count; occurrence += item_stride()) {
    const int64_t key = occurrence_keys[occurrence];
    const int32_t bucket_feature = node_bucket_feature_of_row[occurrence];
    const int32_t table = tables[bucket_feature % num_features];
    int32_t identity = (int32_t)occurrence;  // a new one, unless the probe finds it
    for (uint64_t slot = identity_slot(key, table, ids.slot_bits);; slot = (slot + 1) & last_slot) {
      int32_t taken = ids.slot[slot];
      if (taken < 0) taken = atomicCAS(&ids.slot[slot], -1, (int32_t)occurrence);
      if (taken < 0) break;  // the occurrence took the slot
      if (occurrence_keys[taken] == key &&
          tables[node_bucket_feature_of_row[taken] % num_features] == table) {
        identity = taken;
        break;
      }
    }
    atomicMin(&ids.rep[identity], representative_word(bucket_feature, occurrence));
    atomicAdd(&ids.count[identity], 1);
    ids.next[occurrence] = atomicExch(&ids.head[identity], (int32_t)occurrence);
    if (ids.weight != nullptr)
      atomicAdd((unsigned long long *)&ids.weight[identity],
                (unsigned long long)(int64_t)weights[occurrence]);
    ids.identity_of[occurrence] = identity;
  }
}

// A thread per entry of the remote part: whether its occurrence is its
// identity's representative (none past the valid occurrences), which the scan
// of the representatives sums.
__global__ void __launch_bounds__(kThreads)
    representatives_kernel(Identities ids, const int64_t *occurrence_count, int64_t entries,
                           bool *is_rep) {
  const int64_t count = *occurrence_count;
  for (int64_t occurrence = first_item(); occurrence < entries; occurrence += item_stride())
    is_rep[occurrence] =
        occurrence < count && representative_of(ids.rep[ids.identity_of[occurrence]]) == occurrence;
}

// One CTA: the node-unique keys of each block of the remote part
// (prepare_kernel), its representatives: reps_before[end] - reps_before[start]
// (reps_before: the representatives before each occurrence; nullptr without
// node dedup, where each occurrence is one). From them: the node-unique keys
// of each bucket feature and of each owner node, and base[r][b], the keys of
// the bucket features before b and of b at the requesters before r, less the
// representatives before block (r, b). Key u = base[r][b] + reps_before[rep]
// is the representative rep's of block (r, b): the keys go by bucket feature,
// then in their representatives' order. local_size * num_nodes * num_features
// blocks on one CTA, whose reads do not wait on each other: the order of the
// node layout's row_shift (world_size * num_features entries), also
// one CTA's. scratch: block_scratch words, block_start written.
__global__ void __launch_bounds__(kThreads)
    blocks_kernel(const int32_t *reps_before, int64_t local_size, int64_t num_nodes,
                  int64_t num_features, int64_t *scratch, int64_t *node_length_per_bucket_feature,
                  int64_t *node_offset_per_bucket, int64_t *node_length_per_bucket,
                  int64_t *node_num_uniques) {
  const int64_t bucket_features = num_nodes * num_features, columns = bucket_features + 1;
  // [requester][bucket feature]: where each block starts, then where the
  // requester's range ends; the bases; where the keys of each bucket feature
  // start, then their end
  int64_t *block_start = scratch, *base = block_start + local_size * columns,
          *key_start = base + local_size * bucket_features;
  auto before = [&](int64_t occurrence) -> int64_t {
    return reps_before != nullptr ? reps_before[occurrence] : occurrence;
  };
  // the node-unique keys of block (r, b)
  auto keys_of = [&](int64_t requester, int64_t bucket_feature) {
    const int64_t *start = block_start + requester * columns + bucket_feature;
    return before(start[1]) - before(start[0]);
  };
  for (int64_t bucket_feature = threadIdx.x; bucket_feature < bucket_features;
       bucket_feature += kThreads) {
    int64_t keys = 0;
    for (int64_t requester = 0; requester < local_size; ++requester)
      keys += keys_of(requester, bucket_feature);
    node_length_per_bucket_feature[bucket_feature] = keys;
  }
  __syncthreads();
  const int64_t total = cta_exclusive_ends<int64_t>(
      bucket_features,
      [&](int64_t bucket_feature) { return node_length_per_bucket_feature[bucket_feature]; },
      key_start);
  for (int64_t owner_node = threadIdx.x; owner_node <= num_nodes; owner_node += kThreads) {
    node_offset_per_bucket[owner_node] = key_start[owner_node * num_features];
    if (owner_node < num_nodes)
      node_length_per_bucket[owner_node] =
          key_start[(owner_node + 1) * num_features] - key_start[owner_node * num_features];
  }
  if (threadIdx.x == 0) *node_num_uniques = total;
  for (int64_t bucket_feature = threadIdx.x; bucket_feature < bucket_features;
       bucket_feature += kThreads) {
    int64_t key = key_start[bucket_feature];  // the first key of the next block
    for (int64_t requester = 0; requester < local_size; ++requester) {
      base[requester * bucket_features + bucket_feature] =
          key - before(block_start[requester * columns + bucket_feature]);
      key += keys_of(requester, bucket_feature);
    }
  }
}

// A thread per valid occurrence: its node-unique key u (node_reverse_indices)
// from its identity's representative rep (itself without node dedup,
// kNodeDedup false), u = base[rep's requester][rep's bucket feature] +
// reps_before[rep] (blocks_kernel); and its rank among its key's occurrences
// in occurrence order, the occurrences in its identity's list before it. A
// list holds at most max_key_rows occurrences in a call that fits. A key of
// more makes the call raise (kNeedKeyRows): its occurrences take rank
// max_key_rows without a walk, which gives them the key's row. The
// representative writes its key's entries of the tables per key, and key_rows
// gets the most occurrences of a key, one max per warp. Past node_num_uniques
// the tables per key keep what they held: their readers stop there.
template <bool kNodeDedup>
__global__ void __launch_bounds__(kThreads)
    key_tables_kernel(const int64_t *occurrence_keys, const float *weights,
                      const int32_t *node_bucket_feature_of_row, const int64_t *requester_start,
                      const int64_t *occurrence_count, Identities ids, const int32_t *reps_before,
                      const int64_t *base, int64_t entries, int64_t local_size, int64_t num_nodes,
                      int64_t num_features, int32_t max_key_rows, int32_t *node_reverse_indices,
                      int32_t *occurrence_rank, int32_t *key_count, int32_t *key_node,
                      int64_t *node_unique_keys, float *node_unique_key_weights,
                      int64_t *key_rows) {
  const int64_t count = *occurrence_count, bucket_features = num_nodes * num_features;
  int32_t most = 0;  // the most occurrences of this thread's keys
  for (int64_t occurrence = first_item(); occurrence < count; occurrence += item_stride()) {
    int64_t rep = occurrence, bucket_feature = 0, reps = occurrence;
    int32_t identity = 0, occurrences = 1, rank = 0;
    if constexpr (kNodeDedup) {
      identity = ids.identity_of[occurrence];
      const unsigned long long word = ids.rep[identity];
      rep = representative_of(word);
      bucket_feature = representative_bucket_feature(word);
      reps = reps_before[rep];
      occurrences = ids.count[identity];
      if (occurrences > max_key_rows)
        rank = max_key_rows;
      else
        for (int32_t other = ids.head[identity]; other >= 0; other = ids.next[other])
          rank += other < occurrence;
    } else {
      bucket_feature = node_bucket_feature_of_row[occurrence];
    }
    const int64_t requester = requester_of_occurrence(requester_start, local_size, rep);
    const int64_t key = clamp_index(
        base[requester * bucket_features + clamp_index(bucket_feature, bucket_features)] + reps,
        entries);
    node_reverse_indices[occurrence] = (int32_t)key;
    occurrence_rank[occurrence] = rank;
    if (rep != occurrence) continue;
    key_count[key] = occurrences;
    key_node[key] = (int32_t)owner_node_of(bucket_feature, num_features, num_nodes);
    node_unique_keys[key] = occurrence_keys[occurrence];
    if (node_unique_key_weights != nullptr)
      node_unique_key_weights[key] =
          (float)(kNodeDedup ? ids.weight[identity] : (int64_t)weights[occurrence]);
    most = max(most, occurrences);
  }
  most = __reduce_max_sync(0xffffffffu, most);
  if ((threadIdx.x & 31) == 0 && most > 0)
    atomicMax((unsigned long long *)key_rows, (unsigned long long)most);
}

// ── step 2: the rows, the chunks and the slots ──

// A thread per key index: its rows in proxy_rows, one for each occurrence, at
// most max_key_rows (a call with more raises: kNeedKeyRows), and with
// `weights` its weight in its chunk, its rows but at least chunk_weight
// (Sizes::max_key_weight); 0 past the node-unique keys. Without `weights`
// chunk_weight is 1: the weights are the rows.
__global__ void __launch_bounds__(kThreads)
    key_row_counts_kernel(const int64_t *node_num_uniques, const int32_t *key_count,
                          int64_t entries, int32_t max_key_rows, int32_t chunk_weight,
                          int32_t *rows, int32_t *weights) {
  const int64_t valid_keys = *node_num_uniques;
  for (int64_t key = first_item(); key < entries; key += item_stride()) {
    const bool valid = key < valid_keys;
    const int32_t key_rows = valid ? min(key_count[key], max_key_rows) : 0;
    rows[key] = key_rows;
    if (weights != nullptr) weights[key] = valid ? max(key_rows, chunk_weight) : 0;
  }
}

// The levels of the proxy's stream (common.h): chunk j of an owner node takes
// its keys whose weights before them, among the node's, are in [j * room,
// (j + 1) * room) (room: Sizes::chunk_room); a key weighs at most room, so no
// chunk is empty. A chunk's rows are its keys' rows (node_exp_off).
struct Levels {
  const int64_t *weight_off;               // [entries + 1]: the weights before each key
  const int64_t *node_exp_off;             // [entries + 1]: the rows before each key
  const int64_t *node_offset_per_bucket;   // [num_nodes + 1]: the first key of each owner node
  const int64_t *chunk_length_per_bucket;  // [num_nodes]: the chunks of each owner node
  int64_t num_nodes, room;

  // The first key of owner node `owner_node`'s chunk j, among the node's (its
  // keys when j is its chunks).
  __device__ int64_t first_key(int64_t owner_node, int64_t j) const {
    const int64_t first = node_offset_per_bucket[owner_node],
                  keys = node_offset_per_bucket[owner_node + 1] - first;
    return lower_bound(weight_off + first, keys, weight_off[first] + j * room);
  }
  // The level of key `key` of the owner node whose first key is `first` (the
  // level of the node's last key, plus one, is its chunks).
  __device__ int64_t level_of(int64_t first, int64_t key) const {
    return (weight_off[key] - weight_off[first]) / room;
  }
  // The chunks of the levels below j: each owner node's, up to j.
  __device__ int64_t before(int64_t j) const {
    int64_t chunks = 0;
    for (int64_t owner_node = 0; owner_node < num_nodes; ++owner_node)
      chunks += min(j, chunk_length_per_bucket[owner_node]);
    return chunks;
  }
};

// The chunks of the proxy's stream, made by one CTA (stream_kernel) in three
// steps.
struct StreamArgs {
  Levels levels;                          // its chunk_length_per_bucket is the first step's
  const int64_t *node_length_per_bucket;  // [num_nodes]
  int64_t node, max_chunks, record_bytes, ring, lap_room;
  // what the steps write: of each owner node, its chunks, where they start in
  // chunk_inv, and where its slice starts in key_stage; the stream's chunks
  int64_t *chunk_length_per_bucket, *chunk_offset_per_bucket, *stage_offset_per_bucket;
  int64_t *num_chunks;
  Chunk *chunks;       // [max_chunks]
  int32_t *chunk_inv;  // [max_chunks]
  // [max_chunks]: each chunk's rows and reducer tasks; [max_chunks + 1]: their
  // ends in stream order
  int64_t *rows_of, *reduce_tasks, *stream_ends, *reduce_task_starts;
  // the proxy header's words, and key_stage
  int32_t *header_chunks;
  uint32_t *header_ends, *header_end;
  char *key_stage;
};

// Each owner node's chunks (the level of its last key, plus one; none for
// this node), where they start in chunk_inv, the stream's chunks (to the
// proxy header too), and where each owner node's slice starts in key_stage:
// its chunks (ChunkPiece), then its keys' records. Every thread of the CTA.
__device__ void chunk_counts(const StreamArgs &args) {
  const Levels &levels = args.levels;
  const int64_t num_nodes = levels.num_nodes, node = args.node, max_chunks = args.max_chunks,
                record_bytes = args.record_bytes;
  const int64_t *node_length_per_bucket = args.node_length_per_bucket;
  auto chunks_of = [&](int64_t owner_node) -> int64_t {
    const int64_t first = levels.node_offset_per_bucket[owner_node],
                  keys = node_length_per_bucket[owner_node];
    if (owner_node == node || keys <= 0) return 0;
    return levels.level_of(first, first + keys - 1) + 1;
  };
  for (int64_t owner_node = threadIdx.x; owner_node < num_nodes; owner_node += kThreads)
    args.chunk_length_per_bucket[owner_node] = chunks_of(owner_node);
  const int64_t total =
      cta_exclusive_ends<int64_t>(num_nodes, chunks_of, args.chunk_offset_per_bucket);
  for (int64_t owner_node = threadIdx.x; owner_node < num_nodes; owner_node += kThreads) {
    int64_t start = 0;
    for (int64_t before = 0; before < owner_node; ++before)
      start += chunks_of(before) * (int64_t)sizeof(ChunkPiece) +
               node_length_per_bucket[before] * record_bytes;
    args.stage_offset_per_bucket[owner_node] = start;
  }
  if (threadIdx.x == 0) {
    *args.num_chunks = min(total, max_chunks);
    *args.header_chunks = (int32_t)min(total, max_chunks);
  }
}

// A thread per place c of the stream: its chunk, of level j and owner node o
// (the levels below j hold before(j) chunks, and in level j the owner nodes
// with more than j chunks come in turn from the node after the proxy's own:
// node + 1, node + 2, ... mod num_nodes); zeros past the chunks. Its keys
// (key_rows_kernel adds their occurrences, the rows its requesters pull),
// its rows (rows_of, for the stream's scan), its reducer tasks of 32 keys,
// and its place among its owner node's (chunk_inv).
// The agents take the stream in order on every node at about the same pace,
// so with the turn the proxies of a rail mostly serve different owner nodes
// at a place of a level. In node order they served the same owner node at
// the same time: its NIC took the puts of several proxies (backward), or
// sent to several proxies (forward, grants of a stream that wraps the ring),
// while the other NICs of the rail were idle. DeepEP turns its RDMA peers
// the same way against this incast.
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
      int64_t rank = place - levels.before(level), owner_node = args.node;
      for (int64_t turn = 1; turn < levels.num_nodes; ++turn) {
        owner_node = (args.node + turn) % levels.num_nodes;
        if (levels.chunk_length_per_bucket[owner_node] > level && rank-- == 0) break;
      }
      const int64_t first = levels.node_offset_per_bucket[owner_node];
      const int64_t lo_key = first + levels.first_key(owner_node, level),
                    hi_key = first + levels.first_key(owner_node, level + 1);
      chunk.owner_node = (int32_t)owner_node;
      chunk.key_lo = (int32_t)lo_key;
      chunk.keys = (int32_t)(hi_key - lo_key);
      chunk.level = (int32_t)level;
      rows = levels.node_exp_off[hi_key] - levels.node_exp_off[lo_key];
      args.chunk_inv[clamp_index(args.chunk_offset_per_bucket[owner_node] + level, max_chunks)] =
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
  const int64_t *stream_ends = args.stream_ends,
                *node_offset_per_bucket = args.levels.node_offset_per_bucket;
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
    const int32_t uniq_lo = (int32_t)(chunk.key_lo - node_offset_per_bucket[chunk.owner_node]);
    ((ChunkPiece *)(args.key_stage + args.stage_offset_per_bucket[chunk.owner_node]))[chunk.level] =
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
__device__ __forceinline__ int64_t duplicates_between(const int64_t *node_exp_off, int64_t lo,
                                                      int64_t key) {
  return node_exp_off[key] - node_exp_off[lo] - (key - lo);
}

// A thread per node-unique key: its chunk (from its level), and in the
// chunk's place in proxy_rows (its start % ring) its row and, after the
// chunk's keys' rows, the first slot of its duplicates (a chunk's rows: its
// keys' rows, then each key's duplicates' slots, key after key in u order);
// and its Record (KeyRecord, or WeightedKeyRecord with its weight) in its
// owner node's slice of key_stage, after the node's chunks. Its occurrences
// add to its chunk's pulls, a sum per chunk of a warp's keys.
template <typename Record>
__global__ void __launch_bounds__(kThreads)
    key_rows_kernel(Levels levels, const int32_t *key_node, const int32_t *key_count,
                    const int64_t *node_unique_keys, const float *node_unique_key_weights,
                    const int32_t *chunk_inv, const int64_t *chunk_offset_per_bucket,
                    const int64_t *stage_offset_per_bucket, const int64_t *node_num_uniques,
                    int64_t max_chunks, int64_t ring, Chunk *chunks, int32_t *key_chunk,
                    int32_t *key_row, int32_t *duplicate_base, char *key_stage) {
  const int64_t valid_keys = *node_num_uniques;
  const int lane = threadIdx.x & 31;
  // a warp's keys at a time, so every lane takes part in the sums
  for (int64_t base = first_item() - lane; base < valid_keys; base += item_stride()) {
    const int64_t key = base + lane;
    int64_t place = -1;  // past the keys: none
    int32_t occurrences = 0;
    if (key < valid_keys) {
      const int64_t owner_node = key_node[key], first = levels.node_offset_per_bucket[owner_node];
      const int64_t level = min(levels.level_of(first, key),
                                max(levels.chunk_length_per_bucket[owner_node] - 1, (int64_t)0));
      place = clamp_index(
          chunk_inv[clamp_index(chunk_offset_per_bucket[owner_node] + level, max_chunks)],
          max_chunks);
      const Chunk &chunk = chunks[place];
      const int64_t at = (int64_t)chunk.start % ring, lo = chunk.key_lo;
      const int64_t keys_end = at + chunk.keys;  // the end of its keys' rows
      key_chunk[key] = (int32_t)place;
      key_row[key] = (int32_t)(at + key - lo);
      duplicate_base[key] = (int32_t)(keys_end + duplicates_between(levels.node_exp_off, lo, key));
      Record *records =
          (Record *)(key_stage + stage_offset_per_bucket[owner_node] +
                     levels.chunk_length_per_bucket[owner_node] * (int64_t)sizeof(ChunkPiece));
      Record record{};
      record.key = node_unique_keys[key];
      if constexpr (Record::kWeighted) record.weight = node_unique_key_weights[key];
      records[key - first] = record;
      occurrences = key_count[key];
    }
    // the lanes of each chunk: their sum, added once
    const unsigned same = __match_any_sync(0xffffffffu, place);
    const int32_t pulls = __reduce_add_sync(same, occurrences);
    if (place >= 0 && lane == __ffs(same) - 1) atomicAdd(&chunks[place].pulls, pulls);
  }
}

// A thread per valid occurrence: its requester (from requester_start) and its
// slot: the key's row for its key's first occurrence, a duplicate slot for the
// others (every occurrence of a key of more than max_key_rows, in a call that
// raises, has rank max_key_rows and takes the key's row). remote_map gets the
// occurrence's key row and its slot.
__global__ void __launch_bounds__(kThreads)
    occurrence_slots_kernel(const int32_t *node_reverse_indices, const int32_t *occurrence_rank,
                            const int32_t *key_row, const int32_t *duplicate_base,
                            const int64_t *node_exp_off, const int64_t *requester_start,
                            const int64_t *occurrence_count, int64_t local_size,
                            int32_t *requester_of, int32_t *map_key_row, int32_t *map_slot) {
  const int64_t count = *occurrence_count;
  for (int64_t occurrence = first_item(); occurrence < count; occurrence += item_stride()) {
    const int32_t key = node_reverse_indices[occurrence];
    const int64_t place = occurrence_rank[occurrence];  // among the key's occurrences
    const int64_t rows = node_exp_off[key + 1] - node_exp_off[key];
    const int64_t slot =
        place == 0 || place >= rows ? key_row[key] : duplicate_base[key] + place - 1;
    requester_of[occurrence] =
        (int32_t)requester_of_occurrence(requester_start, local_size, occurrence);
    map_key_row[occurrence] = key_row[key];
    map_slot[occurrence] = (int32_t)slot;
  }
}

// ── step 3: the push order ──

// A thread per occurrence: its push key, its part of the push order,
// requester * (max_chunks + 1) + its key's chunk (past the valid occurrences
// local_size * (max_chunks + 1), which sorts them last); and its own index,
// which the sort carries. So the push order holds a part's occurrences in
// occurrence order.
__global__ void __launch_bounds__(kThreads)
    push_keys_kernel(const int32_t *node_reverse_indices, const int64_t *occurrence_count,
                     const int32_t *requester_of, const int32_t *key_chunk, int64_t entries,
                     int64_t local_size, int64_t max_chunks, int64_t *push_key, int32_t *index) {
  const int64_t count = *occurrence_count;
  for (int64_t occurrence = first_item(); occurrence < entries; occurrence += item_stride()) {
    int64_t push = local_size * (max_chunks + 1);
    if (occurrence < count)
      push =
          requester_of[occurrence] * (max_chunks + 1) + key_chunk[node_reverse_indices[occurrence]];
    push_key[occurrence] = push;
    index[occurrence] = (int32_t)occurrence;
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

// A thread per entry of the push tables, a requester's and a chunk's up to
// max_chunks: where its positions start (part_start), relative to the
// requester's push order (requester_start), into the proxy header.
__global__ void __launch_bounds__(kThreads)
    push_tables_kernel(const int32_t *part_start, const int64_t *requester_start,
                       int64_t local_size, int64_t max_chunks, int32_t *header_tables) {
  const int64_t columns = max_chunks + 1;  // the entries of a requester
  for (int64_t index = first_item(); index < local_size * columns; index += item_stride())
    header_tables[index] = part_start[index] - (int32_t)requester_start[index / columns];
}

// A thread per valid push position: its occurrence's place in its
// requester's push order (remote_map's positions).
__global__ void __launch_bounds__(kThreads)
    push_positions_kernel(const int64_t *push_key_sorted, const int32_t *push_order,
                          const int64_t *requester_start, const int64_t *occurrence_count,
                          int64_t max_chunks, int32_t *map_position) {
  const int64_t count = *occurrence_count;
  for (int64_t position = first_item(); position < count; position += item_stride()) {
    const int64_t requester = push_key_sorted[position] / (max_chunks + 1);
    map_position[push_order[position]] = (int32_t)(position - requester_start[requester]);
  }
}

}  // namespace

ProxyBuffers proxy_buffers(at::Tensor remote_map, at::Tensor key_stage, const ProxyHeader &header,
                           const at::TensorOptions &options) {
  const auto int_options = options.dtype(at::kInt);
  return ProxyBuffers{
      std::move(remote_map),
      std::move(key_stage),
      at::from_blob(header.chunks(), {1}, int_options),
      at::from_blob(header.end(), {1}, int_options),
      at::from_blob(header.ends(), {header.max_chunks}, int_options),
      at::from_blob(header.push_table(0), {header.local_size, header.max_chunks + 1}, int_options)};
}

ProxyKeys proxy_keys(const ProxyInput &input, const Sizes &sizes, bool node_dedup) {
  const int64_t entries = sizes.remote_keys, local_size = sizes.topo.local_size,
                num_nodes = sizes.topo.num_nodes, num_features = sizes.num_features,
                bucket_features = num_nodes * num_features;
  const auto long_options = input.node_keys.options().dtype(at::kLong);
  const auto int_options = input.node_keys.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  const int64_t *requester_start = input.requester_start.data_ptr<int64_t>();
  const float *weights = data_or_null<float>(input.node_key_weights);
  // the sum of an identity's weights, exact as integers, when the records
  // carry it (the keys carry weights then)
  const bool sum_weights = sizes.weighted_records();
  ProxyKeys keys;
  keys.occurrence_count = at::empty({}, long_options);
  keys.key_rows = at::empty({}, long_options);
  // the blocks per (requester, bucket feature), and the bases of the key
  // indices (block_scratch)
  const at::Tensor block_tables =
      at::empty({block_scratch(local_size, bucket_features)}, long_options);
  // Under node dedup the identities: a hash table of the call, at least twice
  // the occurrences' slots, cleared, then every valid occurrence's identity
  // in it, and each identity's state.
  const int slot_bits = node_dedup ? bits_for(2 * entries) : 0;
  const int64_t slots = node_dedup ? int64_t{1} << slot_bits : 0;
  at::Tensor slot, rep, count, head, next, identity_of, weight_sum;
  Identities ids{};
  if (node_dedup) {
    slot = at::empty({slots}, int_options);
    rep = at::empty({entries}, long_options);
    count = at::empty({entries}, int_options);
    head = at::empty({entries}, int_options);
    next = at::empty({entries}, int_options);
    identity_of = at::empty({entries}, int_options);
    if (sum_weights) weight_sum = at::empty({entries}, long_options);
    ids = Identities{slot.data_ptr<int32_t>(),
                     slot_bits,
                     reinterpret_cast<unsigned long long *>(rep.data_ptr<int64_t>()),
                     count.data_ptr<int32_t>(),
                     head.data_ptr<int32_t>(),
                     data_or_null<int64_t>(weight_sum),
                     next.data_ptr<int32_t>(),
                     identity_of.data_ptr<int32_t>()};
  }
  prepare_kernel<<<item_grid(std::max(slots, local_size * (bucket_features + 1))), kThreads, 0,
                   stream>>>(input.skip.data_ptr<bool>(), requester_start,
                             input.node_bucket_feature_of_row.data_ptr<int32_t>(), local_size,
                             bucket_features, entries, slots, ids, block_tables.data_ptr<int64_t>(),
                             keys.occurrence_count.data_ptr<int64_t>(),
                             keys.key_rows.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The representatives before each occurrence (without node dedup each
  // occurrence is one: none needed).
  at::Tensor reps_before;
  if (node_dedup) {
    insert_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
        input.node_keys.data_ptr<int64_t>(), weights,
        input.node_bucket_feature_of_row.data_ptr<int32_t>(), input.tables.data_ptr<int32_t>(),
        keys.occurrence_count.data_ptr<int64_t>(), num_features, ids);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    slot.reset();  // the probes are done
    ids.slot = nullptr;
    const at::Tensor is_rep = at::empty({entries}, input.node_keys.options().dtype(at::kBool));
    representatives_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
        ids, keys.occurrence_count.data_ptr<int64_t>(), entries, is_rep.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    reps_before = exclusive_ends(is_rep, at::kInt);
  }
  // The node-unique keys of the blocks: per bucket feature and per owner
  // node, and the bases of the key indices.
  keys.node_length_per_bucket_feature = at::empty({bucket_features}, long_options);
  keys.node_offset_per_bucket = at::empty({num_nodes + 1}, long_options);
  keys.node_length_per_bucket = at::empty({num_nodes}, long_options);
  keys.node_num_uniques = at::empty({}, long_options);
  blocks_kernel<<<kOneCta, kThreads, 0, stream>>>(
      data_or_null<int32_t>(reps_before), local_size, num_nodes, num_features,
      block_tables.data_ptr<int64_t>(), keys.node_length_per_bucket_feature.data_ptr<int64_t>(),
      keys.node_offset_per_bucket.data_ptr<int64_t>(),
      keys.node_length_per_bucket.data_ptr<int64_t>(), keys.node_num_uniques.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // per occurrence its key and its rank there, and per key its tables
  keys.node_reverse_indices = at::empty({entries}, int_options);
  keys.occurrence_rank = at::empty({entries}, int_options);
  keys.key_count = at::empty({entries}, int_options);
  keys.key_node = at::empty({entries}, int_options);
  keys.node_unique_keys = at::empty({entries}, long_options);
  if (sum_weights)
    keys.node_unique_key_weights =
        at::empty({entries}, input.node_keys.options().dtype(at::kFloat));
  const auto launch_tables = [&](auto kernel) {
    kernel<<<item_grid(entries), kThreads, 0, stream>>>(
        input.node_keys.data_ptr<int64_t>(), weights,
        input.node_bucket_feature_of_row.data_ptr<int32_t>(), requester_start,
        keys.occurrence_count.data_ptr<int64_t>(), ids, data_or_null<int32_t>(reps_before),
        block_tables.data_ptr<int64_t>() + local_size * (bucket_features + 1), entries, local_size,
        num_nodes, num_features, (int32_t)sizes.max_key_rows(),
        keys.node_reverse_indices.data_ptr<int32_t>(), keys.occurrence_rank.data_ptr<int32_t>(),
        keys.key_count.data_ptr<int32_t>(), keys.key_node.data_ptr<int32_t>(),
        keys.node_unique_keys.data_ptr<int64_t>(),
        data_or_null<float>(keys.node_unique_key_weights), keys.key_rows.data_ptr<int64_t>());
  };
  if (node_dedup)
    launch_tables(key_tables_kernel<true>);
  else
    launch_tables(key_tables_kernel<false>);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return keys;
}

ProxyChunks proxy_chunks_and_slots(ProxyKeys &keys, const at::Tensor &requester_start,
                                   const ProxyBuffers &buffers, const Sizes &sizes) {
  const int64_t entries = sizes.remote_keys, num_nodes = sizes.topo.num_nodes,
                local_size = sizes.topo.local_size, max_chunks = sizes.max_chunks;
  const auto long_options = keys.key_count.options().dtype(at::kLong);
  const auto int_options = keys.key_count.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  ProxyChunks chunks;
  // The rows of each key, a row per occurrence (max_key_rows at most), and
  // the rows before each key; the weight of each key, its rows but at least
  // chunk_weight, and the weights before each key (the rows' when
  // chunk_weight is 1).
  const bool weighted = sizes.chunk_weight > 1;
  const at::Tensor rows = at::empty({entries}, int_options);
  const at::Tensor weights = weighted ? at::empty({entries}, int_options) : at::Tensor();
  key_row_counts_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      keys.node_num_uniques.data_ptr<int64_t>(), keys.key_count.data_ptr<int32_t>(), entries,
      (int32_t)sizes.max_key_rows(), (int32_t)sizes.chunk_weight, rows.data_ptr<int32_t>(),
      data_or_null<int32_t>(weights));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  chunks.node_exp_off = exclusive_ends(rows);
  const at::Tensor weight_off = weighted ? exclusive_ends(weights) : chunks.node_exp_off;
  // The chunks of the stream (each owner node's, by levels of its keys'
  // weights, level after level), where each lies, and key_stage's slices; the
  // stream's chunks and their ends to the proxy header too.
  chunks.chunk_length_per_bucket = at::empty({num_nodes}, long_options);
  chunks.chunk_offset_per_bucket = at::empty({num_nodes + 1}, long_options);
  chunks.num_chunks = at::empty({}, long_options);
  chunks.stage_offset_per_bucket = at::empty({num_nodes}, long_options);
  chunks.chunks = at::empty({max_chunks, 8}, int_options);
  chunks.chunk_inv = at::empty({max_chunks}, int_options);
  chunks.reduce_task_starts = at::empty({max_chunks + 1}, long_options);
  // each chunk's rows and reducer tasks, and the ends of its rows
  const at::Tensor stream_tables = at::empty({3, max_chunks + 1}, long_options);
  const Levels levels{weight_off.data_ptr<int64_t>(),
                      chunks.node_exp_off.data_ptr<int64_t>(),
                      keys.node_offset_per_bucket.data_ptr<int64_t>(),
                      chunks.chunk_length_per_bucket.data_ptr<int64_t>(),
                      num_nodes,
                      sizes.chunk_room()};
  Chunk *chunk_table = records_of<Chunk>(chunks.chunks);
  StreamArgs stream_args;
  stream_args.levels = levels;
  stream_args.node_length_per_bucket = keys.node_length_per_bucket.data_ptr<int64_t>();
  stream_args.node = sizes.topo.node;
  stream_args.max_chunks = max_chunks;
  stream_args.record_bytes = sizes.record_bytes;
  stream_args.ring = sizes.num_proxy_rows;
  stream_args.lap_room = sizes.lap_room();
  stream_args.chunk_length_per_bucket = chunks.chunk_length_per_bucket.data_ptr<int64_t>();
  stream_args.chunk_offset_per_bucket = chunks.chunk_offset_per_bucket.data_ptr<int64_t>();
  stream_args.stage_offset_per_bucket = chunks.stage_offset_per_bucket.data_ptr<int64_t>();
  stream_args.num_chunks = chunks.num_chunks.data_ptr<int64_t>();
  stream_args.chunks = chunk_table;
  stream_args.chunk_inv = chunks.chunk_inv.data_ptr<int32_t>();
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
        keys.node_unique_keys.data_ptr<int64_t>(),
        data_or_null<float>(keys.node_unique_key_weights), chunks.chunk_inv.data_ptr<int32_t>(),
        chunks.chunk_offset_per_bucket.data_ptr<int64_t>(),
        chunks.stage_offset_per_bucket.data_ptr<int64_t>(),
        keys.node_num_uniques.data_ptr<int64_t>(), max_chunks, sizes.num_proxy_rows, chunk_table,
        chunks.key_chunk.data_ptr<int32_t>(), key_row.data_ptr<int32_t>(),
        duplicate_base.data_ptr<int32_t>(), (char *)buffers.key_stage.data_ptr<int64_t>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  keys.node_unique_keys.reset();
  keys.node_unique_key_weights.reset();
  // Each occurrence's slot: the key's row for its key's first occurrence, a
  // duplicate slot for the others.
  chunks.requester_of = at::empty({entries}, int_options);
  const RemoteMap remote_map{buffers.remote_map.data_ptr<int32_t>(), entries};
  occurrence_slots_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      keys.node_reverse_indices.data_ptr<int32_t>(), keys.occurrence_rank.data_ptr<int32_t>(),
      key_row.data_ptr<int32_t>(), duplicate_base.data_ptr<int32_t>(),
      chunks.node_exp_off.data_ptr<int64_t>(), requester_start.data_ptr<int64_t>(),
      keys.occurrence_count.data_ptr<int64_t>(), local_size,
      chunks.requester_of.data_ptr<int32_t>(), remote_map.key_row(), remote_map.slot());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  keys.occurrence_rank.reset();
  return chunks;
}

ProxyPlan proxy_push_order(ProxyKeys keys, ProxyChunks chunks, const at::Tensor &requester_start,
                           const ProxyBuffers &buffers, const Sizes &sizes) {
  const int64_t entries = sizes.remote_keys, local_size = sizes.topo.local_size,
                max_chunks = sizes.max_chunks, parts = local_size * (max_chunks + 1);
  const auto long_options = keys.node_reverse_indices.options().dtype(at::kLong);
  const auto int_options = keys.node_reverse_indices.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  // The push order: one stable sort of the occurrences by their part; the
  // invalid ones after the last part. What only the push key needs goes
  // before the sort.
  at::Tensor push_key_sorted, push_order;
  {
    const at::Tensor push_key = at::empty({entries}, long_options);
    const at::Tensor index = at::empty({entries}, int_options);
    push_keys_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
        keys.node_reverse_indices.data_ptr<int32_t>(), keys.occurrence_count.data_ptr<int64_t>(),
        chunks.requester_of.data_ptr<int32_t>(), chunks.key_chunk.data_ptr<int32_t>(), entries,
        local_size, max_chunks, push_key.data_ptr<int64_t>(), index.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    keys.node_reverse_indices.reset();
    keys.key_count.reset();
    keys.key_node.reset();
    chunks.requester_of.reset();
    chunks.key_chunk.reset();
    std::tie(push_key_sorted, push_order) = sort_pairs(push_key, index, bits_for(parts + 1));
  }
  // Where the parts of the push order start, then the push tables, per
  // requester and chunk: its first position.
  const at::Tensor part_start = at::empty({parts}, int_options);
  part_starts_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      push_key_sorted.data_ptr<int64_t>(), keys.occurrence_count.data_ptr<int64_t>(), entries,
      parts, part_start.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  push_tables_kernel<<<item_grid(parts), kThreads, 0, stream>>>(
      part_start.data_ptr<int32_t>(), requester_start.data_ptr<int64_t>(), local_size, max_chunks,
      buffers.push_tables.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Each occurrence's place in its requester's push order (remote_map's
  // positions).
  const RemoteMap remote_map{buffers.remote_map.data_ptr<int32_t>(), entries};
  push_positions_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      push_key_sorted.data_ptr<int64_t>(), push_order.data_ptr<int32_t>(),
      requester_start.data_ptr<int64_t>(), keys.occurrence_count.data_ptr<int64_t>(), max_chunks,
      remote_map.position());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ProxyPlan plan;
  plan.node_exp_off = chunks.node_exp_off;
  plan.num_chunks = chunks.num_chunks;
  plan.occurrence_count = keys.occurrence_count;
  plan.key_rows = keys.key_rows;
  plan.node_length_per_bucket_feature = keys.node_length_per_bucket_feature;
  plan.node_offset_per_bucket = keys.node_offset_per_bucket;
  plan.node_length_per_bucket = keys.node_length_per_bucket;
  plan.chunks = chunks.chunks;
  plan.chunk_length_per_bucket = chunks.chunk_length_per_bucket;
  plan.chunk_offset_per_bucket = chunks.chunk_offset_per_bucket;
  plan.chunk_inv = chunks.chunk_inv;
  plan.stage_offset_per_bucket = chunks.stage_offset_per_bucket;
  plan.reduce_task_starts = chunks.reduce_task_starts;
  return plan;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
