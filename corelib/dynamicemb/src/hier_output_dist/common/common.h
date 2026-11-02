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
// the GIN contexts and signals, and the trace's build flag and row format.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace dyn_emb {
namespace hier_output_dist {

// Constants of the hierarchical dist that host and device code share: CTA
// sizes, the rows of a task, the rounds a call may take and the task queues;
// and ceil_div.

constexpr int kThreads = 512;  // threads per CTA
// Index dedup's heavy rows (more original rows than kHeavy) and the heavy
// groups take sub-tasks of kHeavy rows. A node-unique key of more than kHeavy
// occurrences is heavy, so a key takes at most kHeavy rows of proxy_rows
// (min_proxy_rows_for).
constexpr int kHeavy = 64;
constexpr int kTaskRows = 32;  // rows of an output dist task, one per lane
constexpr int kWarps = kThreads / 32;
// The most rounds of proxy_rows a call's tables hold: credits and push flags
// number the rounds in 20 bits (grant_of).
constexpr int kMaxRounds = 1 << 16;
// Task queues of an out kernel (CtrlLayout::task): the workers', and across
// nodes the backward's reducers'.
enum Queue : int { kQueueWork = 0, kQueueReduce, kNumQueues };

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
// and on the device.

// Values that follow from the capacities read 0 on one node.
struct Sizes {
  Topo topo;
  int64_t num_features;
  int64_t num_tables;      // the features' tables, numbered densely (dense_tables)
  int64_t lookup_keys;     // keys an owner looks up (a capacity)
  int64_t remote_keys;     // remote occurrences a proxy deduplicates (a capacity)
  int64_t num_proxy_rows;  // rows of proxy_rows
  int64_t rounds_cap;      // the most rounds of a call, what a round table holds
  int64_t row_bytes;       // an embedding row (lookup_rows, proxy_rows)
  int64_t partial_rows;    // rows of proxy_rows a requester's fp32 partial row takes
  int64_t chunk_rows;      // rows of an IB chunk (2 MiB), a multiple of kTaskRows
  int64_t record_bytes;    // a key record: 8, or 16 with its weight
  bool weights;            // the requesters' keys carry weights

  __host__ __device__ bool multi_node() const { return topo.num_nodes > 1; }
};

// Each buffer of the window starts at a multiple of kBufferAlign: a buffer of
// `bytes` takes buffer_end(bytes).
constexpr uint64_t kBufferAlign = 4096;
__host__ __device__ inline uint64_t buffer_end(uint64_t bytes) {
  return (bytes + kBufferAlign - 1) / kBufferAlign * kBufferAlign;
}

// Formats of the tables the kernels of a call share (host and device): index
// dedup's fan-out, the proxy's round table and push orders, its reducer
// entries and heavy groups, and put lists.

// Index dedup's fan-out of the requester's deduplicated rows to its original
// rows (part 1): a row's are exp_idx[exp_off[row] .. exp_off[row + 1]), in
// increasing order. A row with more than kHeavy is heavy, and split into
// sub-tasks of kHeavy original rows.
struct Fanout {
  const int32_t *exp_off;  // [rows + 1], nullptr without index dedup
  const int32_t *exp_idx;  // [original rows]
  const int32_t *slot;     // [rows] heavy slot of a row, -1 if it is not heavy
  const int64_t *sub;      // [2 * subs] (row, first entry) of each heavy sub-task,
                           // a row's sub-tasks consecutive and in entry order
  const int64_t *count;    // [2] heavy rows (slots), heavy sub-tasks
  const int32_t *pos;      // [slots] the bucketized row of each heavy row
  const int32_t *first;    // [slots] its first sub-task
};

// ── the proxy's round table and push orders (num_nodes > 1) ──────────────────
// The proxy lays its rows out in proxy_rows by rounds: round rnd of num_rounds
// uses half rnd % 2 of proxy_rows (all of it when num_rounds = 1), owner node
// after owner node; an owner node's piece of a round holds its node-unique keys
// [uniq_lo, uniq_hi): their rows (the representatives) at
// off + uniq - uniq_lo, then their duplicates' slots, then the fp32 partial
// slots of its heavy keys (local_size each). Rounds 0 and 1 are free when an
// out kernel starts; round rnd + 2 refills round rnd's half once that is done
// with: in the forward once every requester of the node pulled round rnd (the
// in-kernel grants), in the backward once the NIC read round rnd's sums
// (half_free).
struct RoundPiece {
  int32_t uniq_lo, uniq_hi;  // the node-unique keys of the piece
  int32_t off;               // slot of its first row in proxy_rows
  int32_t chunk;             // the proxy's own table (BwdPlan::round_table): the piece's first
                             // IB chunk of the backward, numbered over all pieces; else 0
};
// The push-task flags of one round in push_flags: those of round rnd start at
// (rnd % 2) * half_flags. A round holds at most half of proxy_rows (all of it
// when num_rounds = 1), and its flags each cover 32 of its rows or heavy groups
// (which take more than a row each), per requester rounded up.
__host__ __device__ inline int64_t half_flags(int64_t num_proxy_rows, int local_size) {
  return num_proxy_rows / kTaskRows + 2 * local_size;
}
// The rounds for `total` rows in proxy_rows: 1 if they fit, else enough of
// 90% of a half. A round then holds less than half of proxy_rows, so a proxy
// has one round or at least three, never two.
__host__ __device__ inline int64_t rounds_for(int64_t total, int64_t num_proxy_rows) {
  if (total <= num_proxy_rows) return 1;
  const int64_t room = (int64_t)(0.9 * (double)(num_proxy_rows / 2));
  return (total + room - 1) / room;
}
// The rows proxy_rows needs for every round to fit its half: cut fills a
// round with up to 90% of a half, its piece of each other owner node up to
// kHeavy rows over that owner node's share (a key, of at most kHeavy rows, is
// never split), and the other 10% must hold that much.
__host__ __device__ inline int64_t min_proxy_rows_for(int num_nodes) {
  return 20 * (int64_t)kHeavy * (num_nodes - 1);
}
// A requester's rows for one proxy, sorted by their position in its push
// order (push_pos): round after round, in each round its rows of keys with a
// row each, by (IB chunk, owner node), then its heavy groups: its occurrences
// of each heavy key of the round, in position order. Per round:
struct ReqEntry {
  int32_t start;   // first position of the round (round num_rounds: the end of its
                   // rows)
  int32_t heavy;   // first position of its heavy groups
  int32_t flags;   // its first push-task flag of the round (in the round's half of
                   // push_flags): ceil(rows / 32) tasks of rows, then ceil(groups / 32)
                   // of heavy partials, after those of the lower requesters
  int32_t groups;  // its first heavy group (index into the heavy groups;
                   // round num_rounds: the end of its groups)
};
// An entry of a proxy's reducer list (planner.h, reducer_entries): a slot of
// proxy_rows in the low word, with kEntryPartial for a heavy key's fp32
// partial row; in the high word the flag of the push task that fills it, or
// kEntryNoPush (a requester without a partial) or kEntryOwnRow (a heavy key's
// own row, which gets the sum).
constexpr int32_t kEntryNoPush = -1;
constexpr int32_t kEntryOwnRow = -2;
constexpr uint32_t kEntryPartial = 0x80000000u;
// A heavy group: positions [start, end) of a requester's rows for one proxy.
// Its backward sums them in sub-tasks of at most kHeavy rows each (the
// requester's group sub-tasks, numbered group after group), then those sums
// in sub-task order.
struct HeavyGroup {
  int32_t start, end;
};
// A segment of a put list: `rows` rows from src to dst (rows of the source and
// destination buffers), all in one feature's cell of the owner's KJT; for the
// backward also its IB chunk, whose ready counter it waits for.
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
//   rounds       rounds of a proxy (num_nodes > 1; never an overflow)
enum Need : int { kNeedLookupKeys = 0, kNeedRemoteKeys, kNeedRounds, kNumNeeds };

// The status of one call (int64 words), copied to the host once.
enum StatusSlots : int {
  kStatusNeed = 0,                           // [kNumNeeds] the most any rank needs, by Need
  kStatusNumRecv = kStatusNeed + kNumNeeds,  // keys this owner looks up
  kStatusHeavySubs,                          // sub-tasks of this requester's heavy rows
                                             // (index dedup), what its backward sums in
  kStatusGroupSubs,                          // (num_nodes > 1) sub-tasks of its heavy groups
  kStatusSent,                               // rows this requester sends, the valid ones
                                             // of its bound (index dedup: fewer)
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
// with 3 signals per node and 2 counters, all starting at 0 (checked once),
// every one a running total that is never reset. A signal belongs to one
// context, whose QP to a peer executes in order, and is named after the
// sender's node:
//   context 0 (forward rows)   fwd_rows[owner_node]: rows the owner on that node
//                              put into this proxy's proxy_rows;
//   context 1 (backward rows)  bwd_rows[node]: gradient rows the proxy on that
//                              node put into this owner's lookup_rows; counter
//                              sent[half]: this proxy's backward puts out of
//                              that half of proxy_rows whose source the NIC has
//                              read (in calls of more than one round; one
//                              round's puts count on none);
//   context 2 (in channel)     count[node] counts the count messages from the
//                              rail peer on that node, keys[node] its key
//                              streams (one put, or a payload-free signal, per
//                              call each);
//   context 3 (credits)        grant[node] = (out_seq << 20) | rounds of
//                              proxy_rows the proxy on that node lets this
//                              owner fill, bwd_ready[owner_node] = the out_seq
//                              of the backward whose rows the owner on that
//                              node lets this proxy put into its lookup_rows,
//                              closing[node]: the rail peer on that node tears
//                              down.
// Credits are payload-free SignalAdds of the difference to the value the
// sender sent before (its grant_sent and bwd_ready_sent words, CtrlLayout),
// and name the out kernel they admit.
constexpr int kGinContexts = 4;
constexpr int kGinSignalsPerNode = 3;
constexpr int kGinCounters = 2;
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
__host__ __device__ inline uint64_t grant_of(int64_t out_seq, int rounds) {
  return ((uint64_t)out_seq << 20) | (uint64_t)rounds;
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
                                 // issuer's flush; a backward of more than one round: the
                                 // put counters)
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
