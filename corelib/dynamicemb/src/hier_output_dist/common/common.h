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
// What host and device code share, before the window's layouts: the
// constants, the ranks of a sharding (Topo) and an op's sizes (Sizes), the
// formats of the tables the kernels of a call share, the status of a call,
// the GIN contexts, signals and counters, and the trace's build flag and row
// format.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace dyn_emb {
namespace hier_output_dist {

// Constants of the hierarchical dist that host and device code share: CTA
// sizes, the rows of a task, the GIN counters of the backward's puts and the
// task queues; and ceil_div.

constexpr int kThreads = 512;  // threads per CTA
// Index dedup's heavy rows (more original rows than kHeavy) take sub-tasks of
// kHeavy original rows.
constexpr int kHeavy = 64;
constexpr int kTaskRows = 32;  // rows of an output dist task, one per lane
constexpr int kWarps = kThreads / 32;
// The GIN counters of the backward's puts, at most (Sizes::counters): when a
// proxy's stream wraps its ring, chunk c's puts count on counter c % counters,
// and at most that many chunks have puts the NIC may still read.
constexpr int kMaxCounters = 64;
// Task queues of an out kernel (CtrlLayout::task): the workers' tasks of the
// node's owners' rows, across nodes their remote tasks (ReadyFirst), and
// across nodes the backward's reducers'.
enum Queue : int { kQueueLocal = 0, kQueueRemote, kQueueReduce, kNumQueues };

// count / Per rounded up (count >= 0), in the type of count + Per.
template <int Per, typename Count>
__host__ __device__ __forceinline__ constexpr auto ceil_div(Count count) {
  return (count + Per - 1) / Per;
}

// The ranks of a sharding: its nodes and each rank's place in them.

// Who is who. Ranks are node-contiguous: rank = node * local_size +
// local_rank. The owner's KJT has num_sources = local_size + num_nodes - 1
// columns: its node's requesters, then the other nodes.
struct Topo {
  int world_size, local_size, num_nodes, num_sources;
  int rank, local_rank, node;
};

// The sizes of an op, from its configuration (config.h): every buffer of
// its window and every table of its calls is laid out by them, on the host
// and on the device. They also hold how long a kernel may wait for a peer.
// The op keeps them, and every kernel of the op takes them.

// Values that follow from the capacities read 0 on one node. Across nodes a
// proxy's rows of a call are a stream of chunks through its ring, proxy_rows
// (the proxy's stream, below).
struct Sizes {
  Topo topo;
  int64_t num_features;
  int64_t num_tables;      // the features' tables, numbered densely (dense_tables)
  int64_t lookup_keys;     // keys an owner looks up (a capacity)
  int64_t remote_keys;     // remote occurrences a proxy deduplicates (a capacity)
  int64_t num_proxy_rows;  // rows of proxy_rows, the proxy's ring
  int64_t row_bytes;       // an embedding row (lookup_rows, proxy_rows)
  int64_t chunk_rows;      // the most rows of a chunk in the ring: 2 MiB, or its payload if
                           // more, at most half the ring
  int64_t chunk_weight;    // the least weight of a key in a chunk (see max_key_weight): a
                           // chunk's ring rows, before the least of two keys of the most rows,
                           // per row of its payload over IB (512 KiB by default), at least 1
  int64_t max_chunks;      // the most chunks of a proxy's stream in a call
  int64_t record_bytes;    // a key record: 8, or 16 with its weight
  bool weights;            // the requesters' keys carry weights
  uint64_t timeout_ns;     // how long a kernel waits for a peer before it traps

  __host__ __device__ bool multi_node() const { return topo.num_nodes > 1; }
  // The key records carry their weights: WeightedKeyRecord, else KeyRecord
  // (below).
  __host__ __device__ bool weighted_records() const { return record_bytes == 16; }
  // A row in 16-byte vectors.
  __host__ __device__ int row_vecs() const { return (int)(row_bytes / 16); }
  // The most rows of one node-unique key in proxy_rows, a row per
  // occurrence: one per GPU of the node, as index dedup leaves each GPU each
  // (table, key) once. A proxy gives a key of more occurrences this many rows,
  // and the call raises (kNeedKeyRows).
  __host__ __device__ int64_t max_key_rows() const { return topo.local_size; }
  // A key's weight in a chunk is its rows, but at least chunk_weight. A key
  // sends one row over IB, and it has a row in the ring per occurrence. So a
  // chunk of keys of few rows holds about chunk_rows / chunk_weight keys (its
  // payload). A chunk of keys of many rows stops at chunk_rows rows. The most
  // weight of a key:
  __host__ __device__ int64_t max_key_weight() const {
    return max_key_rows() > chunk_weight ? max_key_rows() : chunk_weight;
  }
  // A chunk takes the keys whose weights before them, among their owner
  // node's, fall in one span of this much weight (chunk_room). A key weighs
  // at most max_key_weight, and its rows are at most its weight. So a chunk's
  // weights, and so its rows, stay within chunk_rows.
  __host__ __device__ int64_t chunk_room() const { return chunk_rows - max_key_weight() + 1; }
  // A stream that does not fit the ring takes laps of this many rows of
  // chunk starts (lap_room); a chunk spans at most chunk_rows, so a lap's
  // rows stay within the ring.
  __host__ __device__ int64_t lap_room() const { return num_proxy_rows - chunk_rows + 1; }
  // The GIN counters of the backward's puts (kMaxCounters).
  __host__ __device__ int64_t counters() const {
    return max_chunks < kMaxCounters ? max_chunks : kMaxCounters;
  }
  // The most chunks an owner gets from its rail's proxies. A proxy's stream
  // holds at most w / chunk_room + 1 chunks of this owner's node, w the
  // weights of its keys of the node. A key weighs at most chunk_weight per
  // row, and a proxy's rows are at most remote_keys. A key weighs at most
  // max_key_weight too, and this owner looks up at most lookup_keys keys (or
  // no proxy puts to it).
  __host__ __device__ int64_t owner_chunks() const {
    const int64_t others = topo.num_nodes - 1, room = chunk_room();
    const int64_t by_proxy = others * remote_keys, by_owner = max_key_weight() * lookup_keys;
    // min(chunk_weight * by_proxy, by_owner): the product only when it is at
    // most by_owner, so it does not overflow
    const int64_t weights =
        by_proxy <= by_owner / chunk_weight ? chunk_weight * by_proxy : by_owner;
    return others + weights / room;
  }
};

// Each buffer of the window starts at a multiple of kBufferAlign: a buffer of
// `bytes` takes buffer_end(bytes).
constexpr uint64_t kBufferAlign = 4096;
__host__ __device__ inline uint64_t buffer_end(uint64_t bytes) {
  return (bytes + kBufferAlign - 1) / kBufferAlign * kBufferAlign;
}

// Formats of the tables the kernels of a call share (host and device): index
// dedup's fan-out, the proxy's chunks and push orders, and put lists.

// Index dedup's fan-out of the requester's deduplicated rows to its original
// rows (the bucketize's): a row's are
// exp_idx[exp_off[row] .. exp_off[row + 1]), in increasing order. A row with
// more than kHeavy is heavy, and split into sub-tasks of kHeavy original rows.
struct Fanout {
  const int32_t *exp_off;  // [rows + 1]
  const int32_t *exp_idx;  // [original rows]
  const int32_t *slot;     // [rows] heavy slot of a row, -1 if it is not heavy
  const int64_t *sub;      // [2 * subs] (row, first entry) of each heavy sub-task,
                           // a row's sub-tasks consecutive and in entry order
  const int64_t *count;    // [2] heavy rows (slots), heavy sub-tasks
  const int32_t *pos;      // [slots] the bucketized row of each heavy row
  const int32_t *first;    // [slots] its first sub-task
};

// ── the proxy's stream and push orders (num_nodes > 1) ──────────────────────
// A proxy's rows of a call (proxy_rows) are a stream of chunks through its
// ring of num_proxy_rows rows. A chunk is consecutive node-unique keys of one
// owner node: those whose weights before them (Sizes::max_key_weight), among their owner
// node's, fall in one span of chunk_room (level j: the owner node's chunk j). Its rows are its
// keys' rows (the representatives), then its duplicates' slots, each key's in order: at most
// chunk_rows. The stream takes the chunks level after level, owner node
// after owner node in a level from the node after the proxy's own (so the proxies of a rail serve
// different owner nodes at a time), each right after the one before. Its positions are unwrapped: a
// stream that fits the ring lies as it is; else its chunks take laps of lap_room rows of starts,
// lap k from position k * num_proxy_rows, so no chunk crosses the ring's end. A chunk lies at start
// % num_proxy_rows. Every row of the ring is free when an out kernel starts. A chunk is then done
// with in the forward once every requester of the node pulled it, in the backward once the NIC read
// its sums; chunks are done with in order. The rows of a chunk may be written once the first chunk
// not done with starts at end - num_proxy_rows or later: the relay grants the owners those rows
// (forward), the issuer frees them for the node's requesters (backward).
//
// A chunk as its proxy keeps it (ProxyPlan::chunks): its owner node, its keys
// [key_lo, key_lo + keys) among the proxy's node-unique keys, the rows its
// requesters pull of it in the forward (its occurrences), its unwrapped start
// and end, and its level.
struct Chunk {
  int32_t owner_node, key_lo, keys, pulls;
  uint32_t start, end;
  int32_t level;  // its place among its owner node's chunks
  int32_t unused;
};
// A chunk as its owner reads it (each slice of key_stage starts with the
// chunks of its owner node): its keys [uniq_lo, uniq_hi) among the owner
// node's, and its unwrapped start and end.
struct ChunkPiece {
  int32_t uniq_lo, uniq_hi;
  uint32_t start, end;
};
// A node-unique key as its proxy sends it to the owner (key_stage, then the
// owner's rail_keys), after the chunks of the owner's node: the key, and when
// the owner's KJT carries weights (Sizes::weighted_records, record_bytes 16)
// the weight of its occurrences.
struct KeyRecord {
  static constexpr bool kWeighted = false;
  int64_t key;
};
struct WeightedKeyRecord {
  static constexpr bool kWeighted = true;
  int64_t key;
  float weight;
  uint32_t zero;
};
static_assert(sizeof(KeyRecord) == 8 && sizeof(WeightedKeyRecord) == 16, "record_bytes is 8 or 16");
// The least rows of proxy_rows: two chunks of keys of the most rows each.
__host__ __device__ inline int64_t min_proxy_rows_for(int64_t max_key_rows) {
  return 4 * max_key_rows;
}
// A segment of a put list: `rows` rows from src to dst (rows of the source and
// destination buffers), all in one chunk and in one (feature, source) of the
// owner's KJT; and its chunk, in the list's chunks: the backward's the
// proxy's (whose ready counter it waits for), the forward's the pieces its
// owner got (whose end it waits for in the grant).
struct PutSeg {
  int32_t src, dst, rows, chunk;
};

// A call's status, which the host reads once per call: what it needs of
// each capacity, the same on every rank, and this rank's counts.

// What a call needs of each capacity; the status carries the most any rank
// needs of each, the same on every rank. A call fits when no need is above
// its capacity.
//   lookup_keys  keys an owner looks up (lookup_rows, the KJT)
//   remote_keys  remote occurrences a proxy deduplicates (the remote part of
//                node_keys, remote_map, key_stage; num_nodes > 1)
//   key_rows     occurrences of one node-unique key at its proxy (num_nodes >
//                1), against Sizes::max_key_rows: more means the keys were
//                not index-deduplicated
enum Need : int { kNeedLookupKeys = 0, kNeedRemoteKeys, kNeedKeyRows, kNumNeeds };

// The status of one call (int64 words), which the caller copies to the host
// once, at the call's second wait (shard/hier_output_dist.py reads it by these
// slots).
enum StatusSlots : int {
  kStatusNeed = 0,                           // [kNumNeeds] the most any rank needs, by Need
  kStatusNumRecv = kStatusNeed + kNumNeeds,  // keys this owner looks up
  kStatusHeavySubs,                          // sub-tasks of this requester's heavy rows,
                                             // what its backward sums in
  kStatusOccurrences,                        // (num_nodes > 1) the remote occurrences this
                                             // rank deduplicated as proxy
  kStatusLengthPerKey,                       // [num_features] keys this owner looks up per
                                             // feature, then [num_features] keys per feature
                                             // of the input KJT that construct_jagged_tensors
                                             // reads
};
__host__ __device__ inline int status_words(int num_features) {
  return kStatusLengthPerKey + 2 * num_features;
}

// GIN between nodes: its contexts, and the signals and counters of each.

// GIN between nodes (num_nodes > 1): 4 contexts on the rail connection, each
// with 3 signals per node and Sizes::counters counters, all starting at 0
// (checked once), every one a running total that is never reset. A signal
// belongs to one context, whose QP to a peer executes in order, and is named
// after the sender's node:
//   context 0 (forward rows)   fwd_rows[owner_node]: rows the owner on that node
//                              put into this proxy's proxy_rows;
//   context 1 (backward rows)  bwd_rows[node]: gradient rows the proxy on that
//                              node put into this owner's lookup_rows; counter
//                              k: this proxy's backward puts out of the chunks
//                              c with c % counters == k whose source the NIC
//                              has read (in calls whose stream wraps the ring;
//                              the others' puts count on none);
//   context 2 (in channel)     count[node] counts the count messages from the
//                              rail peer on that node, keys[node] its key
//                              streams (one put, or a payload-free signal, per
//                              call each);
//   context 3 (credits)        grant[node] = tag_of(out_seq, position): the
//                              proxy on that node lets this owner fill the
//                              rows of its stream below that position
//                              (unwrapped), bwd_ready[owner_node] = the out_seq
//                              of the backward whose rows the owner on that
//                              node lets this proxy put into its lookup_rows,
//                              closing[node]: the rail peer on that node tears
//                              down.
// Credits are payload-free SignalAdds of the difference to the value the
// sender sent before (its grant_sent and bwd_ready_sent words, CtrlLayout),
// and name the out kernel they admit.
constexpr int kGinContexts = 4;
constexpr int kGinSignalsPerNode = 3;
constexpr int kCtxFwd = 0;
constexpr int kCtxBwd = 1;
constexpr int kCtxIn = 2;
constexpr int kCtxCredit = 3;
__host__ __device__ inline int sig_count(int node) { return node; }
__host__ __device__ inline int sig_keys(int num_nodes, int node) { return num_nodes + node; }
// fwd_rows (context 0), bwd_rows (context 1)
__host__ __device__ inline int sig_rows(int node) { return node; }
__host__ __device__ inline int sig_grant(int node) { return node; }
__host__ __device__ inline int sig_bwd_ready(int num_nodes, int owner_node) {
  return num_nodes + owner_node;
}
__host__ __device__ inline int sig_closing(int num_nodes, int node) { return 2 * num_nodes + node; }
// A word that names an out kernel and a value of it (a position of a stream,
// a chunk), which grows with both.
__host__ __device__ inline uint64_t tag_of(int64_t out_seq, uint64_t value) {
  return ((uint64_t)out_seq << 32) | value;
}

// Whether the build keeps the hierarchical dist's trace: DYNAMICEMB_HIER_TRACE=1
// at build time makes setup.py define DEMB_HIER_TRACE=1. Without it the trace
// compiles away.

#ifndef DEMB_HIER_TRACE
#define DEMB_HIER_TRACE 0
#endif

constexpr bool kTrace = DEMB_HIER_TRACE != 0;

// The trace of a build with DYNAMICEMB_HIER_TRACE=1 (kTrace): the slots of
// the out kernels' rows and of a CTA's shared copy of one.

// The timeline of the out kernels, a row of kTraceWords uint64 words per
// kernel, in a ring of kTraceRing rows per direction. Times are %globaltimer
// stamps in ns: every GPU counts them on a clock of its own (no common
// origin, and drifting by up to tens of microseconds per second), so only
// stamps of one GPU compare directly. Each stamp is made by one thread per
// CTA (or one lane of a GIN role) and keeps the earliest (atomicMin) or the
// latest (atomicMax) of its CTAs'; the counters are sums. The workers, the
// relay, the issuers and the backward's watcher keep theirs in shared memory
// (TraceSmSlots) and the CTA folds them in when it arrives.
enum TraceSlots : int {
  // the earliest of the CTAs' (the row starts with these at UINT64_MAX)
  kTraceStart = 0,    // the first CTA started
  kTraceFirstPut,     // (num_nodes > 1) this rank posted its first row put over GIN
  kTraceFirstLanded,  // (num_nodes > 1) rows another node sent this rank had landed
  kTraceEnterFirst,   // the first CTA passed the entry barrier
  kTraceNvFirst,      // a worker started its first task that moved rows over NVLink
  kNumEarliest,
  // the latest (the rest of the row starts at 0)
  kTraceEntered = kNumEarliest,  // the last CTA passed the entry barrier
  kTraceLastPut,                 // (num_nodes > 1) this rank posted its last row put
  kTracePutsDone,                // (num_nodes > 1) every row put it posted had completed (the
                                 // issuer's flush; a backward whose stream wraps: the put
                                 // counters)
  kTraceLanded,                  // (num_nodes > 1) every row other nodes send this rank had landed
  kTraceNvLast,                  // a worker ended its last task that moved rows over NVLink
  kTraceArrived,                 // the last CTA arrived: this rank's rows have all moved
  kTraceExited,                  // it passed the exit barrier: every node peer's have too
                                 // (backward: and its lookup_grad is complete)
  kTraceStartLast,               // the last CTA started
  // sums
  kTraceNvBytes,     // bytes the workers moved over NVLink: rows loaded from node
                     // peers' lookup_rows and proxy_rows (forward), stored into them
                     // (backward)
  kTraceLocalBytes,  // bytes of rows the workers moved within this GPU (its own
                     // lookup_rows or proxy_rows to its output, its gradient to its
                     // own lookup_rows or proxy_rows)
  kTraceIbIn,        // (num_nodes > 1) rows other nodes sent this rank, as the relay
                     // (forward) or the watcher (backward) saw them land
  kTraceIbOut,       // (num_nodes > 1) rows this rank put over GIN
  kNumTrace
};
// Then three timelines of kTraceBins bins, bin i holding what happened from
// i to i + 1 bin widths (kTraceBinNs) after the start of the CTA that saw it,
// the last bin also what came later: bytes moved over NVLink, rows landed
// from other nodes, rows posted to other nodes. They span 5.12 ms.
constexpr int kTraceBins = 512;
constexpr unsigned long long kTraceBinNs = 10000;  // width of a bin: 10 us
constexpr int kTraceBinNv = kNumTrace;
constexpr int kTraceBinIbIn = kTraceBinNv + kTraceBins;
constexpr int kTraceBinIbOut = kTraceBinIbIn + kTraceBins;
constexpr int kTraceWords = kTraceBinIbOut + kTraceBins;
constexpr int kTraceRing = 64;  // rows of each direction's ring
// A CTA's shared copy (dynamic shared memory, only when tracing): its start,
// then the slots it keeps (first and last NVLink task, first landing, the
// counters), the rows each lane of the backward's watcher saw land from its
// nodes (not folded into the row) and the three timelines.
enum TraceSmSlots : int {
  kSmStart = 0,
  kSmNvFirst,
  kSmFirstLanded,
  kSmNvLast,
  kSmNvBytes,
  kSmLocalBytes,
  kSmIbIn,
  kSmIbOut,
  kSmLanded,                 // [32], a word per lane
  kSmBins = kSmLanded + 32,  // [3][kTraceBins], as in the row
  kTraceSmWords = kSmBins + 3 * kTraceBins
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
