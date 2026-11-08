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

#include "dedup_chain.cuh"
#include "hier_output_dist/common/common.h"
#include "hier_output_dist/input_dist/planner.cuh"
#include "hier_output_dist/input_dist/planner.h"
#include "murmur_hash.cuh"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

// ── step 1: the node-unique keys ──
//
// A hash table finds each occurrence's identity, by index dedup's chained
// insert (dedup_chain.cuh). Integer atomics give the identity its
// representative, its occurrences, a list of them and the sum of their
// weights. A scan of the representatives and their counts per (requester,
// bucket feature) block then number the node-unique keys, inside a bucket
// feature the requesters in turn with node dedup and requester after
// requester without it (key_of_representative). No result depends
// on the order of the atomics: the representative is a minimum, the counts
// and weights are integer sums, a list is read as a set, and the occurrence
// that claims an identity only holds its state. Nor on the hash: no step
// reads which bucket an identity took.

// The owner node of a bucket feature owner node * num_features + feature
// (clamped to the last node: an index).
__device__ __forceinline__ int64_t owner_node_of(int64_t bucket_feature, int64_t num_features,
                                                 int64_t num_nodes) {
  return min(bucket_feature / num_features, num_nodes - 1);
}

// The requester of a valid occurrence: the last one whose occurrences start
// at or before it (remote_offsets: local_size + 1 entries).
__device__ __forceinline__ int64_t requester_of_occurrence(const int64_t *remote_offsets,
                                                           int64_t local_size, int64_t occurrence) {
  return min(max(upper_bound(remote_offsets, local_size + 1, occurrence) - 1, (int64_t)0),
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

// The hash table of a call's identities (table, key) under node dedup: index
// dedup's chained insert (dedup_chain.cuh) over one bucket space of
// 2^bucket_bits buckets, at least 2 * remote_keys. A chain links the
// occurrences that claimed an identity, one per identity: that occurrence
// names the identity, and the identity's state is at its index. A list links
// each identity's occurrences; it shares no array with the chains.
// proxy_keys allocates the table for the call, and prepare_kernel clears it.
struct Identities {
  // the chains: int32 occurrences, -1 for none
  int32_t *bucket_head;  // [2^bucket_bits]: the first identity of each bucket's chain
  int bucket_bits;
  int32_t *chain_next;  // [remote_keys]: per identity, the next one in its bucket's chain
  // per identity (the occurrence that claimed it)
  unsigned long long *rep;  // [remote_keys]: its representative (representative_word)
  int32_t *count;           // [remote_keys]: its occurrences
  int32_t *list_head;       // [remote_keys]: the last of them in its list (-1: none)
  int64_t *weight;          // [remote_keys]: the sum of their weights, or nullptr
  // per occurrence
  int32_t *list_next;    // [remote_keys]: the one before it in its identity's list (-1: none)
  int32_t *identity_of;  // [remote_keys]: its identity
};

// The seeds of identity_bucket's hash.
constexpr uint64_t kKeySeed = 0x9e3779b97f4a7c15ull, kTableSeed = 0xc2b2ae3d27d4eb4full;

// The bucket of identity (table, key). The owner rule fixes some bits for all
// keys of one owner: under roundrobin the key's low bits (key % world_size),
// under hash_roundrobin the low bits of murmur3_fmix64(key) (its hash %
// world_size). All of a proxy's keys share them up to the largest power of
// two dividing local_size (that value % local_size is the proxy's rail), at
// any world size. So the bucket comes from neither. The hash is fmix64 of the
// key xor a seed, a function other than the rule's, xor the table times
// another seed (equal keys of two tables go apart). The bucket is its high
// bits, which every bit of the key moves: they are those of fmix64's last
// product (its last step changes only the low 31 bits), and only a factor's
// low bits move a product's low bits.
__device__ __forceinline__ uint64_t identity_bucket(int64_t key, int32_t table, int bucket_bits) {
  const uint64_t hash = murmur3_fmix64((uint64_t)key ^ kKeySeed) ^ (uint64_t)table * kTableSeed;
  return hash >> (64 - bucket_bits);
}

// The int64 words of the blocks' tables (prepare_kernel, blocks_kernel): per
// requester where its blocks start and where its range ends, and per bucket
// feature where its keys start, and their end.
inline int64_t block_scratch(int64_t local_size, int64_t bucket_features) {
  return (local_size + 1) * (bucket_features + 1);
}

// A thread per bucket of the hash table (`buckets`; none without node dedup)
// and per (requester, bucket feature) block of the remote part. The table of
// this call is cleared: every bucket's chain empty, each valid occurrence
// without a next in a chain (any of them may claim an identity), and each
// identity's state without a representative, occurrences, a list or weights.
// A block is the occurrences of a bucket feature in a requester's range.
// Inside a requester's range the occurrences come by owner node, then
// feature, then position (node.cuh: row_shift puts a row of owner (n, t)
// after the requester's rows for the owner nodes before n, at its offset in
// its bucket, which is feature-major). So node_bucket_feature_of_row does not
// decrease there, and a binary search finds where each block starts:
// block_offsets [local_size][bucket_features + 1], each requester's last entry
// the end of its range.
// The first thread writes the count of the valid occurrences, those the node
// layout counts for this proxy (none when the node skips the call), and
// key_rows' start.
__global__ void __launch_bounds__(kThreads)
    prepare_kernel(const bool *skip, const int64_t *remote_offsets,
                   const int32_t *node_bucket_feature_of_row, int64_t local_size,
                   int64_t bucket_features, int64_t entries, int64_t buckets, Identities ids,
                   int64_t *block_offsets, int64_t *occurrence_count, int64_t *key_rows) {
  const int64_t count = *skip ? 0 : remote_offsets[local_size], columns = bucket_features + 1;
  if (first_item() == 0) {
    *occurrence_count = count;
    *key_rows = 0;
  }
  for (int64_t index = first_item(); index < buckets; index += item_stride()) {
    ids.bucket_head[index] = -1;
    if (index < count) ids.chain_next[index] = -1;
    if (index >= entries) continue;
    ids.rep[index] = ~0ull;
    ids.count[index] = 0;
    ids.list_head[index] = -1;
    if (ids.weight != nullptr) ids.weight[index] = 0;
  }
  for (int64_t index = first_item(); index < local_size * columns; index += item_stride()) {
    const int64_t requester = index / columns, bucket_feature = index % columns;
    const int64_t lo = min(remote_offsets[requester], count),
                  hi = max(min(remote_offsets[requester + 1], count), lo);
    block_offsets[index] =
        bucket_feature == bucket_features
            ? hi
            : lo + lower_bound(node_bucket_feature_of_row + lo, hi - lo, bucket_feature);
  }
}

// A thread per valid occurrence, under node dedup: its identity (table, key),
// the table of its feature (dense_tables). The chained insert (chain_insert,
// dedup_chain.cuh) walks the chain of the identity's bucket (identity_bucket)
// to the occurrence that claimed the identity, or claims it at the chain's
// end: then the occurrence itself names it. Of the occurrences of one
// identity, one claims it, whatever their order, and the others end on it.
// Then, on the identity: its representative (atomicMin), the occurrence
// counted (atomicAdd) and put first in the list (atomicExch of its list head:
// the occurrence's list_next is the old head), and its weight, a whole
// number, added as an int64.
__global__ void __launch_bounds__(kThreads)
    insert_kernel(const int64_t *occurrence_keys, const float *weights,
                  const int32_t *node_bucket_feature_of_row, const int32_t *tables,
                  const int64_t *occurrence_count, int64_t num_features, Identities ids) {
  const int64_t count = *occurrence_count;
  for (int64_t occurrence = first_item(); occurrence < count; occurrence += item_stride()) {
    const int64_t key = occurrence_keys[occurrence];
    const int32_t bucket_feature = node_bucket_feature_of_row[occurrence];
    const int32_t table = tables[bucket_feature % num_features];
    // a chain node of the same identity: the same key, of the same table
    const auto same_identity = [&](int32_t node) {
      return occurrence_keys[node] == key &&
             tables[node_bucket_feature_of_row[node] % num_features] == table;
    };
    const ChainEntry entry =
        chain_insert(ids.bucket_head, ids.chain_next, identity_bucket(key, table, ids.bucket_bits),
                     occurrence, same_identity);
    const int32_t identity = entry.node;  // the occurrence that claimed it
    atomicMin(&ids.rep[identity], representative_word(bucket_feature, occurrence));
    atomicAdd(&ids.count[identity], 1);
    ids.list_next[occurrence] = atomicExch(&ids.list_head[identity], (int32_t)occurrence);
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

// The representatives before occurrence `occurrence`: its entry of
// reps_before, or the occurrence itself without node dedup (reps_before
// nullptr), where each occurrence is one.
__device__ __forceinline__ int64_t reps_before_of(const int32_t *reps_before, int64_t occurrence) {
  return reps_before != nullptr ? reps_before[occurrence] : occurrence;
}

// The representatives of block (r, b), its node-unique keys: those before
// the block's next entry of block_offsets ([local_size][columns]; a
// requester's last entry is the end of its range) less those before its own.
__device__ __forceinline__ int64_t block_reps(const int64_t *block_offsets, int64_t columns,
                                              const int32_t *reps_before, int64_t requester,
                                              int64_t bucket_feature) {
  const int64_t *offsets = block_offsets + requester * columns + bucket_feature;
  return reps_before_of(reps_before, offsets[1]) - reps_before_of(reps_before, offsets[0]);
}

// The key u of representative `rep` of block (r, b). With node dedup
// (kNodeDedup), inside bucket feature b the requesters take turns: the k-th
// representative (k from 0, in occurrence order) of block (r, b), k =
// reps_before[rep] - reps_before[block_offsets[r][b]], is key
//   u = key_offsets[b] + sum over r' of min(k, reps(r', b))
//                      + #{r' < r : reps(r', b) > k},
// reps(r', b) the representatives of block (r', b) (block_reps). Before it
// come every representative of b of index below k, of every requester, and
// those of index k of the requesters before r: the keys of b are b's
// representatives in (k, r) order, a bijection onto [key_offsets[b],
// key_offsets[b + 1]). So a chunk mostly takes keys of all the node's
// requesters, and its pushes mostly come from all of them. (Requester after
// requester, as below, a chunk's pushes come from a few requesters, about
// half of them from one.) O(local_size) loads per occurrence: per requester
// two words of block_offsets and two of reps_before, the same for every
// occurrence of b.
// Without node dedup each occurrence is a key, and the keys of b go
// requester after requester, each requester's in occurrence order: u =
// key_offsets[b] + the occurrences of b at the requesters before r + rep's
// place in its block. So the rows of a (table, key) that several requesters
// ask reach the owner's lookup in requester order. With the requesters in
// turn their order would follow index dedup's order of each requester's
// keys, which can change from run to run, and so could the lookup's fp32
// sum of them.
template <bool kNodeDedup>
__device__ __forceinline__ int64_t key_of_representative(
    const int64_t *block_offsets, const int64_t *key_offsets, const int32_t *reps_before,
    int64_t local_size, int64_t columns, int64_t requester, int64_t bucket_feature, int64_t rep) {
  int64_t key = key_offsets[bucket_feature];
  if constexpr (kNodeDedup) {
    const int64_t k =
        reps_before[rep] - reps_before[block_offsets[requester * columns + bucket_feature]];
    for (int64_t other = 0; other < local_size; ++other) {
      const int64_t *offsets = block_offsets + other * columns + bucket_feature;
      const int64_t reps = reps_before[offsets[1]] - reps_before[offsets[0]];
      key += min(k, reps) + (other < requester && reps > k);
    }
  } else {
    key += rep - block_offsets[requester * columns + bucket_feature];
    for (int64_t other = 0; other < requester; ++other) {
      const int64_t *offsets = block_offsets + other * columns + bucket_feature;
      key += offsets[1] - offsets[0];
    }
  }
  return key;
}

// One CTA: the node-unique keys of each block of the remote part
// (prepare_kernel), its representatives (block_reps; reps_before: the
// representatives before each occurrence, nullptr without node dedup, where
// each occurrence is one). From them: the node-unique keys of each bucket
// feature and where they start (key_offsets: key_of_representative numbers
// each bucket feature's keys from there), and those of each owner node.
// local_size * num_nodes * num_features blocks on one CTA, whose reads do
// not wait on each other: the order of the node layout's row_shift
// (world_size * num_features entries), also one CTA's. scratch:
// block_scratch words, block_offsets written.
__global__ void __launch_bounds__(kThreads)
    blocks_kernel(const int32_t *reps_before, int64_t local_size, int64_t num_nodes,
                  int64_t num_features, int64_t *scratch, int64_t *node_length_per_bucket_feature,
                  int64_t *node_offset_per_bucket, int64_t *node_length_per_bucket,
                  int64_t *node_num_uniques) {
  const int64_t bucket_features = num_nodes * num_features, columns = bucket_features + 1;
  // [requester][bucket feature]: where each block starts, then where the
  // requester's range ends; where the keys of each bucket feature start, then
  // their end
  const int64_t *block_offsets = scratch;
  int64_t *key_offsets = scratch + local_size * columns;
  for (int64_t bucket_feature = threadIdx.x; bucket_feature < bucket_features;
       bucket_feature += kThreads) {
    int64_t keys = 0;
    for (int64_t requester = 0; requester < local_size; ++requester)
      keys += block_reps(block_offsets, columns, reps_before, requester, bucket_feature);
    node_length_per_bucket_feature[bucket_feature] = keys;
  }
  __syncthreads();
  const int64_t total = cta_offsets_of<int64_t>(
      bucket_features,
      [&](int64_t bucket_feature) { return node_length_per_bucket_feature[bucket_feature]; },
      key_offsets);
  for (int64_t owner_node = threadIdx.x; owner_node <= num_nodes; owner_node += kThreads) {
    node_offset_per_bucket[owner_node] = key_offsets[owner_node * num_features];
    if (owner_node < num_nodes)
      node_length_per_bucket[owner_node] =
          key_offsets[(owner_node + 1) * num_features] - key_offsets[owner_node * num_features];
  }
  if (threadIdx.x == 0) *node_num_uniques = total;
}

// A thread per valid occurrence: its node-unique key u (node_reverse_indices)
// from its identity's representative rep (itself without node dedup,
// kNodeDedup false): rep's key in its block (rep's requester, rep's bucket
// feature), key_of_representative (block_offsets and key_offsets:
// blocks_kernel's scratch); and its rank among its key's occurrences in
// occurrence order, the occurrences in its identity's list before it. A
// list holds at most max_key_rows occurrences in a call that fits. A key of
// more makes the call raise (kNeedKeyRows): its occurrences take rank
// max_key_rows without a walk, which gives them the key's row. The
// representative writes its key's entries of the tables per key, and key_rows
// gets the most occurrences of a key, one max per warp. Past node_num_uniques
// the tables per key keep what they held: their readers stop there.
template <bool kNodeDedup>
__global__ void __launch_bounds__(kThreads)
    key_tables_kernel(const int64_t *occurrence_keys, const float *weights,
                      const int32_t *node_bucket_feature_of_row, const int64_t *remote_offsets,
                      const int64_t *occurrence_count, Identities ids, const int32_t *reps_before,
                      const int64_t *block_offsets, const int64_t *key_offsets, int64_t entries,
                      int64_t local_size, int64_t num_nodes, int64_t num_features,
                      int32_t max_key_rows, int32_t *node_reverse_indices, int32_t *occurrence_rank,
                      int32_t *key_count, int32_t *key_node, int64_t *node_unique_keys,
                      float *node_unique_key_weights, int64_t *key_rows) {
  const int64_t count = *occurrence_count, bucket_features = num_nodes * num_features;
  int32_t most = 0;  // the most occurrences of this thread's keys
  for (int64_t occurrence = first_item(); occurrence < count; occurrence += item_stride()) {
    int64_t rep = occurrence, bucket_feature = 0;
    int32_t identity = 0, occurrences = 1, rank = 0;
    if constexpr (kNodeDedup) {
      identity = ids.identity_of[occurrence];
      const unsigned long long word = ids.rep[identity];
      rep = representative_of(word);
      bucket_feature = representative_bucket_feature(word);
      occurrences = ids.count[identity];
      if (occurrences > max_key_rows)
        rank = max_key_rows;
      else
        for (int32_t other = ids.list_head[identity]; other >= 0; other = ids.list_next[other])
          rank += other < occurrence;
    } else {
      bucket_feature = node_bucket_feature_of_row[occurrence];
    }
    const int64_t requester = requester_of_occurrence(remote_offsets, local_size, rep);
    const int64_t key =
        clamp_index(key_of_representative<kNodeDedup>(
                        block_offsets, key_offsets, reps_before, local_size, bucket_features + 1,
                        requester, clamp_index(bucket_feature, bucket_features), rep),
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
// `counted_rows` its counted rows in its chunk, its rows but at least
// rows_per_payload (Sizes::max_counted_rows); 0 past the node-unique keys.
// Without `counted_rows` rows_per_payload is 1: the counted rows are the rows.
__global__ void __launch_bounds__(kThreads)
    key_row_counts_kernel(const int64_t *node_num_uniques, const int32_t *key_count,
                          int64_t entries, int32_t max_key_rows, int32_t rows_per_payload,
                          int32_t *rows, int32_t *counted_rows) {
  const int64_t valid_keys = *node_num_uniques;
  for (int64_t key = first_item(); key < entries; key += item_stride()) {
    const bool valid = key < valid_keys;
    const int32_t key_rows = valid ? min(key_count[key], max_key_rows) : 0;
    rows[key] = key_rows;
    if (counted_rows != nullptr) counted_rows[key] = valid ? max(key_rows, rows_per_payload) : 0;
  }
}

// The rounds of the proxy's stream (common.h): chunk j of an owner node, in the
// stream's round j, takes its keys whose counted rows before them, among the
// node's, are in [j * span, (j + 1) * span) (span: Sizes::chunk_span); a key
// counts at most span rows, so no chunk is empty. A chunk's rows are its keys'
// rows (node_exp_off).
struct Rounds {
  const int64_t *counted_off;              // [entries + 1]: the counted rows before each key
  const int64_t *node_exp_off;             // [entries + 1]: the rows before each key
  const int64_t *node_offset_per_bucket;   // [num_nodes + 1]: the first key of each owner node
  const int64_t *chunk_length_per_bucket;  // [num_nodes]: the chunks of each owner node
  int64_t num_nodes, span;

  // The first key of owner node `owner_node`'s chunk j, among the node's (its
  // keys when j is its chunks).
  __device__ int64_t first_key(int64_t owner_node, int64_t j) const {
    const int64_t first = node_offset_per_bucket[owner_node],
                  keys = node_offset_per_bucket[owner_node + 1] - first;
    return lower_bound(counted_off + first, keys, counted_off[first] + j * span);
  }
  // The round of key `key` of the owner node whose first key is `first` (the
  // round of the node's last key, plus one, is its chunks).
  __device__ int64_t round_of(int64_t first, int64_t key) const {
    return (counted_off[key] - counted_off[first]) / span;
  }
  // The chunks of the rounds below j: each owner node's, up to j.
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
  Rounds rounds;                          // its chunk_length_per_bucket is the first step's
  const int64_t *node_length_per_bucket;  // [num_nodes]
  int64_t node, max_chunks, record_bytes, ring, lap_span;
  // what the steps write: of each owner node, its chunks, where they start in
  // chunk_inv, and where its slice starts in key_stage; the stream's chunks
  int64_t *chunk_length_per_bucket, *chunk_offset_per_bucket, *stage_offset_per_bucket;
  int64_t *num_chunks;
  Chunk *chunks;       // [max_chunks]
  int32_t *chunk_inv;  // [max_chunks]
  // [max_chunks]: each chunk's rows and reducer tasks; [max_chunks + 1]: their
  // offsets in stream order
  int64_t *rows_of, *reduce_tasks, *stream_offsets, *reduce_task_offsets;
  // the proxy header's words, and key_stage
  int32_t *header_chunks;
  uint32_t *header_ends, *header_end;
  char *key_stage;
};

// Each owner node's chunks (the round of its last key, plus one; none for
// this node), where they start in chunk_inv, the stream's chunks (to the
// proxy header too), and where each owner node's slice starts in key_stage:
// its chunks (ChunkPiece), then its keys' records. Every thread of the CTA.
__device__ void chunk_counts(const StreamArgs &args) {
  const Rounds &rounds = args.rounds;
  const int64_t num_nodes = rounds.num_nodes, node = args.node, max_chunks = args.max_chunks,
                record_bytes = args.record_bytes;
  const int64_t *node_length_per_bucket = args.node_length_per_bucket;
  auto chunks_of = [&](int64_t owner_node) -> int64_t {
    const int64_t first = rounds.node_offset_per_bucket[owner_node],
                  keys = node_length_per_bucket[owner_node];
    if (owner_node == node || keys <= 0) return 0;
    return rounds.round_of(first, first + keys - 1) + 1;
  };
  for (int64_t owner_node = threadIdx.x; owner_node < num_nodes; owner_node += kThreads)
    args.chunk_length_per_bucket[owner_node] = chunks_of(owner_node);
  const int64_t total = cta_offsets_of<int64_t>(num_nodes, chunks_of, args.chunk_offset_per_bucket);
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

// A thread per place c of the stream: its chunk, of round j and owner node o
// (the rounds below j hold before(j) chunks, and in round j the owner nodes
// with more than j chunks come in turn from the node after the proxy's own:
// node + 1, node + 2, ... mod num_nodes); zeros past the chunks. Its keys
// (key_rows_kernel adds their occurrences, the rows its requesters pull),
// its rows (rows_of, for the stream's scan), its reducer tasks of 32 keys,
// and its place among its owner node's (chunk_inv).
// The agents take the stream in order on every node at about the same pace,
// so with the turn the proxies of a rail mostly serve different owner nodes
// at a place of a round. In node order they served the same owner node at
// the same time: its NIC took the puts of several proxies (backward), or
// sent to several proxies (forward, grants of a stream that wraps the ring),
// while the other NICs of the rail were idle. DeepEP turns its RDMA peers
// the same way against this incast.
__device__ void chunk_table(const StreamArgs &args) {
  const Rounds &rounds = args.rounds;
  const int64_t count = *args.num_chunks, max_chunks = args.max_chunks;
  for (int64_t place = threadIdx.x; place < max_chunks; place += kThreads) {
    Chunk chunk{};
    int64_t rows = 0;
    if (place < count) {
      // the round: the last j whose rounds below hold at most `place` chunks
      int64_t lo = 0, hi = max_chunks;
      while (lo < hi) {
        const int64_t mid = (lo + hi + 1) / 2;
        if (rounds.before(mid) <= place)
          lo = mid;
        else
          hi = mid - 1;
      }
      const int64_t round = lo;
      int64_t rank = place - rounds.before(round), owner_node = args.node;
      for (int64_t turn = 1; turn < rounds.num_nodes; ++turn) {
        owner_node = (args.node + turn) % rounds.num_nodes;
        if (rounds.chunk_length_per_bucket[owner_node] > round && rank-- == 0) break;
      }
      const int64_t first = rounds.node_offset_per_bucket[owner_node];
      const int64_t lo_key = first + rounds.first_key(owner_node, round),
                    hi_key = first + rounds.first_key(owner_node, round + 1);
      chunk.owner_node = (int32_t)owner_node;
      chunk.key_lo = (int32_t)lo_key;
      chunk.keys = (int32_t)(hi_key - lo_key);
      chunk.round = (int32_t)round;
      rows = rounds.node_exp_off[hi_key] - rounds.node_exp_off[lo_key];
      args.chunk_inv[clamp_index(args.chunk_offset_per_bucket[owner_node] + round, max_chunks)] =
          (int32_t)place;
    }
    args.chunks[place] = chunk;
    args.rows_of[place] = rows;
    args.reduce_tasks[place] = ceil_div<kTaskRows>((int64_t)chunk.keys);
  }
}

// The offsets of the stream's chunks' rows, in stream order (stream_offsets,
// from rows_of), and the reducer tasks before each chunk (reduce_task_offsets,
// from reduce_tasks); then, a thread per place c of the stream, where its
// chunk lies, unwrapped. A stream that fits the ring (ring rows) lies as it
// is; else lap k takes the chunks whose rows before them are in
// [k * lap_span, (k + 1) * lap_span), from position k * ring, one after
// another. Its start and end into its Chunk, its end into the proxy header,
// and its ChunkPiece, at its round, first in its owner node's slice of
// key_stage; the stream's end into the header. Every thread of the CTA.
__device__ void chunk_places(const StreamArgs &args) {
  const int64_t max_chunks = args.max_chunks, ring = args.ring, lap_span = args.lap_span;
  const int64_t *stream_offsets = args.stream_offsets,
                *node_offset_per_bucket = args.rounds.node_offset_per_bucket;
  cta_offsets_of<int64_t>(
      max_chunks, [&](int64_t place) { return args.rows_of[place]; }, args.stream_offsets);
  cta_offsets_of<int64_t>(
      max_chunks, [&](int64_t place) { return args.reduce_tasks[place]; },
      args.reduce_task_offsets);
  const int64_t count = *args.num_chunks, total = stream_offsets[count];
  if (threadIdx.x == 0 && count == 0) *args.header_end = 0;
  for (int64_t place = threadIdx.x; place < max_chunks; place += kThreads) {
    if (place >= count) {
      args.header_ends[place] = 0;
      continue;
    }
    int64_t start = stream_offsets[place];
    if (total > ring) {
      const int64_t lap = stream_offsets[place] / lap_span;
      const int64_t first = lower_bound(stream_offsets, count, lap * lap_span);
      start = lap * ring + stream_offsets[place] - stream_offsets[first];
    }
    const int64_t end = start + stream_offsets[place + 1] - stream_offsets[place];
    Chunk &chunk = args.chunks[place];
    chunk.start = (uint32_t)start;
    chunk.end = (uint32_t)end;
    args.header_ends[place] = (uint32_t)end;
    if (place == count - 1) *args.header_end = (uint32_t)end;
    const int32_t uniq_lo = (int32_t)(chunk.key_lo - node_offset_per_bucket[chunk.owner_node]);
    ((ChunkPiece *)(args.key_stage + args.stage_offset_per_bucket[chunk.owner_node]))[chunk.round] =
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

// A thread per node-unique key: its chunk (from its round), and in the
// chunk's place in proxy_rows (its start % ring) its row and, after the
// chunk's keys' rows, the first slot of its duplicates (a chunk's rows: its
// keys' rows, then each key's duplicates' slots, key after key in u order);
// and its Record (KeyRecord, or WeightedKeyRecord with its weight) in its
// owner node's slice of key_stage, after the node's chunks. Its occurrences
// add to its chunk's pulls, a sum per chunk of a warp's keys.
template <typename Record>
__global__ void __launch_bounds__(kThreads)
    key_rows_kernel(Rounds rounds, const int32_t *key_node, const int32_t *key_count,
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
      const int64_t owner_node = key_node[key], first = rounds.node_offset_per_bucket[owner_node];
      const int64_t round = min(rounds.round_of(first, key),
                                max(rounds.chunk_length_per_bucket[owner_node] - 1, (int64_t)0));
      place = clamp_index(
          chunk_inv[clamp_index(chunk_offset_per_bucket[owner_node] + round, max_chunks)],
          max_chunks);
      const Chunk &chunk = chunks[place];
      const int64_t at = (int64_t)chunk.start % ring, lo = chunk.key_lo;
      const int64_t keys_end = at + chunk.keys;  // the end of its keys' rows
      key_chunk[key] = (int32_t)place;
      key_row[key] = (int32_t)(at + key - lo);
      duplicate_base[key] = (int32_t)(keys_end + duplicates_between(rounds.node_exp_off, lo, key));
      Record *records =
          (Record *)(key_stage + stage_offset_per_bucket[owner_node] +
                     rounds.chunk_length_per_bucket[owner_node] * (int64_t)sizeof(ChunkPiece));
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

// A thread per valid occurrence: its requester (from remote_offsets) and its
// slot: the key's row for its key's first occurrence, a duplicate slot for the
// others (every occurrence of a key of more than max_key_rows, in a call that
// raises, has rank max_key_rows and takes the key's row). remote_map gets the
// occurrence's key row and its slot.
__global__ void __launch_bounds__(kThreads)
    occurrence_slots_kernel(const int32_t *node_reverse_indices, const int32_t *occurrence_rank,
                            const int32_t *key_row, const int32_t *duplicate_base,
                            const int64_t *node_exp_off, const int64_t *remote_offsets,
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
        (int32_t)requester_of_occurrence(remote_offsets, local_size, occurrence);
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
// parts of the push order start (part_offsets; a part per requester and chunk
// up to max_chunks, in push key order). A valid position whose part is not
// the one before's starts its part and the parts in between, which have
// none; the last valid one ends the parts after its own, and the first
// thread every part when there is no valid one.
__global__ void __launch_bounds__(kThreads)
    part_offsets_kernel(const int64_t *push_key_sorted, const int64_t *occurrence_count,
                        int64_t entries, int64_t parts, int32_t *part_offsets) {
  const int64_t count = *occurrence_count;
  if (first_item() == 0 && count == 0)
    for (int64_t part = 0; part < parts; ++part) part_offsets[part] = 0;
  for (int64_t position = first_item(); position < count; position += item_stride()) {
    const int64_t part = push_key_sorted[position];
    const int64_t before = position == 0 ? -1 : push_key_sorted[position - 1];
    for (int64_t next = before + 1; next <= part; ++next) part_offsets[next] = (int32_t)position;
    if (position == count - 1)
      for (int64_t next = part + 1; next < parts; ++next) part_offsets[next] = (int32_t)count;
  }
}

// A thread per entry of the push tables, a requester's and a chunk's up to
// max_chunks: where its positions start (part_offsets), relative to the
// requester's push order (remote_offsets), into the proxy header.
__global__ void __launch_bounds__(kThreads)
    push_tables_kernel(const int32_t *part_offsets, const int64_t *remote_offsets,
                       int64_t local_size, int64_t max_chunks, int32_t *header_tables) {
  const int64_t columns = max_chunks + 1;  // the entries of a requester
  for (int64_t index = first_item(); index < local_size * columns; index += item_stride())
    header_tables[index] = part_offsets[index] - (int32_t)remote_offsets[index / columns];
}

// A thread per valid push position: its occurrence's place in its
// requester's push order (remote_map's positions).
__global__ void __launch_bounds__(kThreads)
    push_positions_kernel(const int64_t *push_key_sorted, const int32_t *push_order,
                          const int64_t *remote_offsets, const int64_t *occurrence_count,
                          int64_t max_chunks, int32_t *map_position) {
  const int64_t count = *occurrence_count;
  for (int64_t position = first_item(); position < count; position += item_stride()) {
    const int64_t requester = push_key_sorted[position] / (max_chunks + 1);
    map_position[push_order[position]] = (int32_t)(position - remote_offsets[requester]);
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
  const int64_t *remote_offsets = input.remote_offsets.data_ptr<int64_t>();
  const float *weights = data_or_null<float>(input.node_key_weights);
  // the sum of an identity's weights, exact as integers, when the records
  // carry it (the keys carry weights then)
  const bool sum_weights = sizes.weighted_records();
  ProxyKeys keys;
  keys.occurrence_count = at::empty({}, long_options);
  keys.key_rows = at::empty({}, long_options);
  // where each (requester, bucket feature) block starts, and where the keys
  // of each bucket feature start (block_scratch)
  const at::Tensor block_tables =
      at::empty({block_scratch(local_size, bucket_features)}, long_options);
  // Under node dedup the identities: a hash table of the call, its buckets at
  // least twice the occurrences and its chain links, cleared, then every
  // valid occurrence's identity in it, and each identity's state and list.
  const int bucket_bits = node_dedup ? bits_for(2 * entries) : 0;
  const int64_t buckets = node_dedup ? int64_t{1} << bucket_bits : 0;
  at::Tensor bucket_head, chain_next, rep, count, list_head, list_next, identity_of, weight_sum;
  Identities ids{};
  if (node_dedup) {
    bucket_head = at::empty({buckets}, int_options);
    chain_next = at::empty({entries}, int_options);
    rep = at::empty({entries}, long_options);
    count = at::empty({entries}, int_options);
    list_head = at::empty({entries}, int_options);
    list_next = at::empty({entries}, int_options);
    identity_of = at::empty({entries}, int_options);
    if (sum_weights) weight_sum = at::empty({entries}, long_options);
    ids = Identities{bucket_head.data_ptr<int32_t>(),
                     bucket_bits,
                     chain_next.data_ptr<int32_t>(),
                     reinterpret_cast<unsigned long long *>(rep.data_ptr<int64_t>()),
                     count.data_ptr<int32_t>(),
                     list_head.data_ptr<int32_t>(),
                     data_or_null<int64_t>(weight_sum),
                     list_next.data_ptr<int32_t>(),
                     identity_of.data_ptr<int32_t>()};
  }
  prepare_kernel<<<item_grid(std::max(buckets, local_size * (bucket_features + 1))), kThreads, 0,
                   stream>>>(
      input.skip.data_ptr<bool>(), remote_offsets,
      input.node_bucket_feature_of_row.data_ptr<int32_t>(), local_size, bucket_features, entries,
      buckets, ids, block_tables.data_ptr<int64_t>(), keys.occurrence_count.data_ptr<int64_t>(),
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
    bucket_head.reset();  // the inserts are done
    chain_next.reset();
    ids.bucket_head = ids.chain_next = nullptr;
    const at::Tensor is_rep = at::empty({entries}, input.node_keys.options().dtype(at::kBool));
    representatives_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
        ids, keys.occurrence_count.data_ptr<int64_t>(), entries, is_rep.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    reps_before = offsets_of(is_rep, at::kInt);
  }
  // The node-unique keys of the blocks: per bucket feature, where they
  // start, and per owner node.
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
  // per occurrence its key (with node dedup the requesters in turn inside
  // each bucket feature) and its rank there, and per key its tables
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
        input.node_bucket_feature_of_row.data_ptr<int32_t>(), remote_offsets,
        keys.occurrence_count.data_ptr<int64_t>(), ids, data_or_null<int32_t>(reps_before),
        block_tables.data_ptr<int64_t>(),
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

ProxyChunks proxy_chunks_and_slots(ProxyKeys &keys, const at::Tensor &remote_offsets,
                                   const ProxyBuffers &buffers, const Sizes &sizes) {
  const int64_t entries = sizes.remote_keys, num_nodes = sizes.topo.num_nodes,
                local_size = sizes.topo.local_size, max_chunks = sizes.max_chunks;
  const auto long_options = keys.key_count.options().dtype(at::kLong);
  const auto int_options = keys.key_count.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  ProxyChunks chunks;
  // The rows of each key, a row per occurrence (max_key_rows at most), and
  // the rows before each key; the counted rows of each key, its rows but at
  // least rows_per_payload, and the counted rows before each key (the rows'
  // when rows_per_payload is 1).
  const at::Tensor rows = at::empty({entries}, int_options);
  const at::Tensor counted_rows =
      sizes.rows_per_payload > 1 ? at::empty({entries}, int_options) : at::Tensor();
  key_row_counts_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      keys.node_num_uniques.data_ptr<int64_t>(), keys.key_count.data_ptr<int32_t>(), entries,
      (int32_t)sizes.max_key_rows(), (int32_t)sizes.rows_per_payload, rows.data_ptr<int32_t>(),
      data_or_null<int32_t>(counted_rows));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  chunks.node_exp_off = offsets_of(rows);
  const at::Tensor counted_off =
      sizes.rows_per_payload > 1 ? offsets_of(counted_rows) : chunks.node_exp_off;
  // The chunks of the stream (each owner node's, cut by its keys' counted
  // rows, round after round), where each lies, and key_stage's slices; the
  // stream's chunks and their ends to the proxy header too.
  chunks.chunk_length_per_bucket = at::empty({num_nodes}, long_options);
  chunks.chunk_offset_per_bucket = at::empty({num_nodes + 1}, long_options);
  chunks.num_chunks = at::empty({}, long_options);
  chunks.stage_offset_per_bucket = at::empty({num_nodes}, long_options);
  chunks.chunks = at::empty({max_chunks, 8}, int_options);
  chunks.chunk_inv = at::empty({max_chunks}, int_options);
  chunks.reduce_task_offsets = at::empty({max_chunks + 1}, long_options);
  // each chunk's rows and reducer tasks, and the offsets of its rows
  const at::Tensor stream_tables = at::empty({3, max_chunks + 1}, long_options);
  const Rounds rounds{counted_off.data_ptr<int64_t>(),
                      chunks.node_exp_off.data_ptr<int64_t>(),
                      keys.node_offset_per_bucket.data_ptr<int64_t>(),
                      chunks.chunk_length_per_bucket.data_ptr<int64_t>(),
                      num_nodes,
                      sizes.chunk_span()};
  Chunk *chunk_table = records_of<Chunk>(chunks.chunks);
  StreamArgs stream_args;
  stream_args.rounds = rounds;
  stream_args.node_length_per_bucket = keys.node_length_per_bucket.data_ptr<int64_t>();
  stream_args.node = sizes.topo.node;
  stream_args.max_chunks = max_chunks;
  stream_args.record_bytes = sizes.record_bytes;
  stream_args.ring = sizes.num_proxy_rows;
  stream_args.lap_span = sizes.lap_span();
  stream_args.chunk_length_per_bucket = chunks.chunk_length_per_bucket.data_ptr<int64_t>();
  stream_args.chunk_offset_per_bucket = chunks.chunk_offset_per_bucket.data_ptr<int64_t>();
  stream_args.stage_offset_per_bucket = chunks.stage_offset_per_bucket.data_ptr<int64_t>();
  stream_args.num_chunks = chunks.num_chunks.data_ptr<int64_t>();
  stream_args.chunks = chunk_table;
  stream_args.chunk_inv = chunks.chunk_inv.data_ptr<int32_t>();
  stream_args.rows_of = stream_tables[0].data_ptr<int64_t>();
  stream_args.reduce_tasks = stream_tables[1].data_ptr<int64_t>();
  stream_args.stream_offsets = stream_tables[2].data_ptr<int64_t>();
  stream_args.reduce_task_offsets = chunks.reduce_task_offsets.data_ptr<int64_t>();
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
        rounds, keys.key_node.data_ptr<int32_t>(), keys.key_count.data_ptr<int32_t>(),
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
      chunks.node_exp_off.data_ptr<int64_t>(), remote_offsets.data_ptr<int64_t>(),
      keys.occurrence_count.data_ptr<int64_t>(), local_size,
      chunks.requester_of.data_ptr<int32_t>(), remote_map.key_row(), remote_map.slot());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  keys.occurrence_rank.reset();
  return chunks;
}

ProxyPlan proxy_push_order(ProxyKeys keys, ProxyChunks chunks, const at::Tensor &remote_offsets,
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
  const at::Tensor part_offsets = at::empty({parts}, int_options);
  part_offsets_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      push_key_sorted.data_ptr<int64_t>(), keys.occurrence_count.data_ptr<int64_t>(), entries,
      parts, part_offsets.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  push_tables_kernel<<<item_grid(parts), kThreads, 0, stream>>>(
      part_offsets.data_ptr<int32_t>(), remote_offsets.data_ptr<int64_t>(), local_size, max_chunks,
      buffers.push_tables.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Each occurrence's place in its requester's push order (remote_map's
  // positions).
  const RemoteMap remote_map{buffers.remote_map.data_ptr<int32_t>(), entries};
  push_positions_kernel<<<item_grid(entries), kThreads, 0, stream>>>(
      push_key_sorted.data_ptr<int64_t>(), push_order.data_ptr<int32_t>(),
      remote_offsets.data_ptr<int64_t>(), keys.occurrence_count.data_ptr<int64_t>(), max_chunks,
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
  plan.reduce_task_offsets = chunks.reduce_task_offsets;
  return plan;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
