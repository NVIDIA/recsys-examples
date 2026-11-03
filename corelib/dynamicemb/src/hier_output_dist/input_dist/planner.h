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
// The planner: the steps of a call between and around the comm kernels, on
// the current stream: kernels of their own (what they share: planner.cuh)
// and CUB's radix sorts and scans (cub.cu), none of which waits for the
// device. Part 1, the node layout, the proxy, the rail and the owner, the
// requester, and the put lists. Every table of a step is sized by the op's
// sizes (Sizes, from its configuration: config.h), never by a device value.
#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/config.h"
#include "hier_output_dist/common/window.h"

namespace dyn_emb {
namespace hier_output_dist {

// The planner's CUB calls (cub.cu), on the current stream.

// A stable sort of `keys` (int32 or int64, contiguous) by their bits
// [0, end_bit), which carries `values` (int32, one per key): the sorted keys,
// and the values in their order. With end_bit below the keys' width the keys
// must not be negative.
std::pair<at::Tensor, at::Tensor> sort_pairs(const at::Tensor &keys, const at::Tensor &values,
                                             int end_bit);
// A sort of int64 keys (not negative) by their bits [0, end_bit).
at::Tensor sort_keys(const at::Tensor &keys, int end_bit);
// The ends of n counts (bool, int32 or int64), [0, c0, c0 + c1, ...]: n + 1
// entries, int64 (or int32 when their sum stays below 2^31).
at::Tensor exclusive_ends(const at::Tensor &counts, at::ScalarType dtype = at::kLong);

// Part 1 of a call (the requester's, local, on the caller's stream): its keys
// partitioned by owner, and under index dedup the fan-out of its
// deduplicated rows: kernels and a CUB sort each (part_one.cu).

// The keys of a call as part 1 takes them: the KJT the input dist was given
// and what construct_jagged_tensors reads. Under index dedup the KJT holds
// the deduplicated keys: only its first keys are valid, and only the device
// knows how many (its offsets say).
struct PlanInput {
  at::Tensor values;                  // its keys, int32 or int64
  c10::optional<at::Tensor> offsets;  // [num_features * batch_size + 1], or
  c10::optional<at::Tensor> lengths;  // [num_features * batch_size] when it has none
  int64_t batch_size = 0;
  c10::optional<at::Tensor> weights;        // fp32, one per key
  c10::optional<at::Tensor> input_offsets;  // the input KJT's offsets, when it is another KJT
  int64_t input_batch_size = 0;
  c10::optional<at::Tensor> reverse_indices;  // index dedup: original row -> row of the KJT
};

// The rule that places the keys of each feature on their owners, the
// bucketize's: a DistType code (src/utils.h; codes other than kCyclic and
// kHashedCyclic are kBlock) and the block size kBlock divides by.
struct OwnerRules {
  at::Tensor dist_types;             // [num_features] int32 on the GPU
  at::Tensor block_sizes;            // [num_features] int64 on the GPU
  std::vector<int> host_dist_types;  // the codes on the host: only the rules present run
};

// A call's keys by owner, as the bucketize orders them:
// [owner][feature][sample][position]. B is values.numel(); the valid keys
// are block_start[world_size] of them (fewer under index dedup).
struct Partition {
  at::Tensor keys;         // int64 [B]: each valid key as its owner stores it
  at::Tensor key_weights;  // fp32 [B], or undefined: their weights
  at::Tensor inv;          // int32 [B]: the KJT row of each (a permutation; row past the valid)
  at::Tensor row_cell;     // int32 [B]: the cell owner * num_features + feature of each,
                           // world_size * num_features past the valid
  at::Tensor cell_start;   // int64 [world_size * num_features + 1]: where each cell starts
  at::Tensor block_start;  // int64 [world_size + 1]: where each owner's block starts
  at::Tensor input_length_per_key;  // int64 [num_features]: keys per feature of the input KJT
};
Partition partition_keys(const PlanInput &input, const OwnerRules &rules, int64_t world_size);

// Index dedup's fan-out of the deduplicated rows to the original rows, in the
// format of Fanout (common.h): a row's original rows are
// exp_idx[exp_off[row] .. exp_off[row + 1]), in increasing order. A row of
// more than kHeavy is heavy: it takes a slot (by increasing row) and
// consecutive sub-tasks of kHeavy original rows. Entries past `count` are
// clamped garbage nobody reads.
struct FanoutTables {
  at::Tensor exp_off;  // int32 [rows + 1]
  at::Tensor exp_idx;  // int32 [original rows]
  at::Tensor slot;     // int32 [rows]: the slot of a heavy row, else -1
  at::Tensor sub;      // int64 [2 * max_subs]: (row, first entry in exp_idx) of each sub-task
  at::Tensor count;    // int64 [2]: heavy rows, sub-tasks
  at::Tensor pos;      // int32 [max_slots]: the row of part 1's keys each slot's row went to
  at::Tensor first;    // int32 [max_slots]: the first sub-task of each slot
};
// From reverse_indices (original row -> deduplicated row) and the partition's
// inv, whose size is the number of deduplicated rows.
FanoutTables fanout_tables(const at::Tensor &reverse_indices, const at::Tensor &inv);

// The node's part of a call (the in channel): this rank's
// counts for its node, the layout every rank of the node computes alike from
// the counts the node exchanged, and on one node the owner's KJT and the
// requester's map (planner.cu).
//
// node_keys of a target holds its own part at [0, lookup_keys), the keys of
// its node for the target as owner, in the owner's KJT order
// ([feature][requester][requester's order]); and across nodes its remote
// part at [lookup_keys, lookup_keys + remote_keys), the keys of its node for
// the owners on its rail of the other nodes, which it deduplicates as their
// proxy: requester-major, then owner node, then [feature][position].

// This rank's count vector for its node (CountVector, window.h), int64
// [local_size * (num_features + 1)]: first its keys for each target's remote
// part, then its keys for each owner of its node, [owner][feature]. From
// part 1's cell_start.
at::Tensor node_count_vector(const at::Tensor &cell_start, const Topo &topo, int64_t num_features);

// The node's count vectors as node_count_exchange gives them to this rank.
struct NodeCounts {
  at::Tensor totals;  // int64 [local_size * (num_features + 1)]: their sum over the node
  at::Tensor before;  // int64 [local_size * (num_features + 1)]: their sum over the ranks
                      // before this one
  at::Tensor mine;    // int64 [local_size][num_features + 1]: each requester's keys for this
                      // rank: for its remote part, then for each feature of its own part
};

// The layout of the node's keys, from the node's counts and this rank's part
// 1. Under a need above a capacity no rank of the node pushes (skip). One
// node has no remote part: there requester_start and need_remote stay
// undefined.
struct NodeLayout {
  at::Tensor own_cells;         // int64 [num_features * local_size]: keys of cell (feature,
                                // requester) in this rank's own part
  at::Tensor own_starts;        // int64 [num_features * local_size]: where each such cell
                                // starts there
  at::Tensor own_total;         // int64 [local_size]: each owner's own part
  at::Tensor requester_start;   // int64 [local_size + 1]: where each requester's keys start in
                                // this rank's remote part
  at::Tensor need_lookup;       // int64 0-dim: the largest own part of the node
  at::Tensor need_remote;       // int64 0-dim: the largest remote part of the node
  at::Tensor skip;              // bool 0-dim: a need above its capacity
  at::Tensor rows_to_push;      // int64 0-dim: this rank's valid rows, 0 under skip
  at::Tensor destination_base;  // int64 [world_size * num_features + 1]: row `row` of part 1
                                // goes to destination_base[row_cell[row]] + row in node_keys
                                // of its target
};
NodeLayout node_layout(const NodeCounts &counts, const at::Tensor &cell_start,
                       const at::Tensor &block_start, const Topo &topo, int64_t num_features,
                       int64_t lookup_keys, int64_t remote_keys);

// On one node, the KJT of this rank as owner, a copy of its own part of
// node_keys once every requester of the node pushed (node_keys, and
// node_weights when the keys carry weights: views of the own part), and its
// map as requester: its rows' places in their owners' KJTs, which are where
// it pushed them.
struct OneNodeOwner {
  at::Tensor kjt_values;      // int64 [lookup_keys]: the first num_recv are the KJT's
  at::Tensor kjt_weights;     // fp32 [lookup_keys], or undefined
  at::Tensor kjt_lengths;     // int64 [num_features * local_size]: cell (feature, requester)
  at::Tensor kjt_offsets;     // int64 [num_features * local_size + 1]
  at::Tensor num_recv;        // int64 0-dim: the keys it looks up
  at::Tensor length_per_key;  // int64 [num_features]
  at::Tensor src;             // int32 [row_cell.numel()]: the place of each row in its owner's
                              // KJT (and its dst)
};
OneNodeOwner one_node_owner(const NodeLayout &node, const at::Tensor &row_cell,
                            const at::Tensor &node_keys, const at::Tensor &node_weights,
                            const Topo &topo, int64_t num_features);

// The proxy's part of a call across nodes (the in channel, after
// node_done). The remote part of its node_keys holds the keys its node
// sends the owners of its rail on other nodes; the proxy deduplicates them
// into node-unique keys, lays their rows out as a stream of chunks through
// its ring, proxy_rows (common.h), and writes what its node's requesters and
// its rail's owners read of it into its buffers of the window: remote_map,
// key_stage and the proxy header (window.h). Three steps, each taking the
// tables of the ones before (proxy.cu).
//
// The occurrences are the entries of the remote part (occurrence index occ).
// A node-unique key is an identity (owner node, table, key), or each
// occurrence without node dedup; its key index u numbers the node-unique keys
// owner node after owner node, each owner node's by the smallest feature of
// the key, then in key order. Entries past the valid part of the remote part,
// and keys past the valid ones, hold garbage that no one reads.

// The proxy's inputs: its remote part (views of its buffers) and what the
// node layout says of it.
struct ProxyInput {
  at::Tensor keys;             // int64 [remote_keys]: the remote part of node_keys
  at::Tensor weights;          // fp32 [remote_keys]: their weights, or undefined
  at::Tensor cells;            // int32 [remote_keys]: owner node * num_features + feature of
                               // each (remote_cells)
  at::Tensor requester_start;  // int64 [local_size + 1]: each requester's occurrences
  at::Tensor skip;             // bool 0-dim: the node skips the call (it pushed nothing)
  at::Tensor tables;           // int32 [num_features]: the table of each feature, numbered
                               // densely (dense_tables)
};

// The buffers the proxy writes, views of this rank's copy of the window.
struct ProxyBuffers {
  at::Tensor remote_map;  // int32 [3][remote_keys] (RemoteMap): per occurrence its key's row
                          // (bit 31: a heavy key), its slot, its position in its requester's
                          // push order
  at::Tensor key_stage;   // int64 [KeysLayout::key_stage_bytes / 8]: per other owner node a
                          // slice of its chunks (ChunkPiece) and its keys' records; the
                          // last 16 bytes take the writes of no key
  // the proxy header's parts (ProxyHeader)
  at::Tensor chunks;       // int32 [1]: its stream's chunks
  at::Tensor end;          // int32 [1]: its stream's end (uint32)
  at::Tensor ends;         // int32 [max_chunks]: each chunk's end (uint32)
  at::Tensor push_tables;  // int32 [local_size][max_chunks + 1][4]: ReqEntry, positions
                           // relative to each requester's push order here and groups
                           // relative to its first group
};
// The buffers the proxy writes: remote_map and key_stage (views), and the
// parts of the proxy header `header`, viewed on the device of `options`.
ProxyBuffers proxy_buffers(at::Tensor remote_map, at::Tensor key_stage, const ProxyHeader &header,
                           const at::TensorOptions &options);

// Step 1: the node-unique keys, by sorts of the occurrences.
struct ProxyKeys {
  at::Tensor occurrence_count;  // int64 0-dim: the valid occurrences (0 when the node skips)
  at::Tensor num_keys;          // int64 0-dim: the node-unique keys
  // per sorted position q: the occurrences in identity order, each identity's in occ order
  at::Tensor valid;   // bool [remote_keys]: q < occurrence_count (also per occurrence)
  at::Tensor order;   // int32 [remote_keys]: the occurrence at q
  at::Tensor key_at;  // int32 [remote_keys]: the node-unique key of the occurrence at q
  // per node-unique key u
  at::Tensor valid_key;   // bool [remote_keys]: u < num_keys
  at::Tensor key_count;   // int32 [remote_keys]: its occurrences
  at::Tensor key_heavy;   // bool [remote_keys]: more than kHeavy (never without node dedup)
  at::Tensor key_first;   // int32 [remote_keys]: the sorted position of its first occurrence
  at::Tensor key_node;    // int32 [remote_keys]: its owner node
  at::Tensor key_uniq;    // int32 [remote_keys]: u among its owner node's
  at::Tensor key_value;   // int64 [remote_keys]: the key
  at::Tensor key_weight;  // fp32 [remote_keys]: the weight of its record (its occurrences'
                          // weights, or their count), or undefined (8-byte records)
  at::Tensor cell_uniq;   // int64 [num_nodes * num_features]: the keys of each (owner node,
                          // feature)
  at::Tensor uniq_off;    // int64 [num_nodes + 1]: the first key of each owner node
  at::Tensor uniq_out;    // int64 [num_nodes]: the keys of each owner node
};
// node_dedup: equal keys of an owner node and table are one node-unique key.
// Weights must be whole numbers (frequency counts): they are summed as int64.
ProxyKeys proxy_keys(const ProxyInput &input, const Sizes &sizes, bool node_dedup);

// Step 2: the rows of the keys, the chunks of the stream and where they
// lie, each occurrence's slot. Writes remote_map's key rows and slots, the
// chunks and records of key_stage, and the proxy header's chunks, end and
// ends; drops keys.key_value and keys.key_weight, which only the records
// need.
struct ProxyChunks {
  at::Tensor num_chunks;        // int64 0-dim: the chunks of the stream
  at::Tensor chunks;            // int32 [max_chunks][8]: Chunk, in stream order (zeros past them)
  at::Tensor chunks_out;        // int64 [num_nodes]: the chunks of each owner node
  at::Tensor node_chunk_start;  // int64 [num_nodes + 1]: where each owner node's chunks start in
                                // node_chunks
  at::Tensor node_chunks;       // int32 [max_chunks]: each owner node's chunks in order, as their
                                // places in the stream
  at::Tensor stage_slice;       // int64 [num_nodes]: where its slice for each owner node starts in
                                // key_stage (bytes)
  at::Tensor reduce_task_starts;  // int64 [max_chunks + 1]: the reducers' tasks before each chunk
  at::Tensor heavy_ends;          // int32 [remote_keys + 1]: the heavy keys before each key
  at::Tensor key_chunk;           // int32 [remote_keys]: the chunk of each key
  at::Tensor key_row;             // int32 [remote_keys]: its row in proxy_rows
  at::Tensor partial_base;        // int32 [remote_keys]: a heavy key's first partial row
  at::Tensor slot_at;             // int32 [remote_keys]: the slot of the occurrence at sorted
                                  // position q
  at::Tensor requester_of;        // int32 [remote_keys]: the requester of each occurrence
};
ProxyChunks proxy_chunks_and_slots(ProxyKeys &keys, const at::Tensor &requester_start,
                                   const ProxyBuffers &buffers, const Sizes &sizes);

// The push key, by which one stable sort gives each requester's push order:
// from its low bits up, a heavy key's number among the heavy keys (0 for a
// key with a row each);
// whether the key is heavy; its chunk; and the requester (local_size: past
// the valid occurrences). Each field is as wide as its largest value needs.
struct PushKey {
  int heavy_shift, chunk_shift, requester_shift;  // where each field above the first starts
  int bits;                                       // the whole key
};
PushKey push_key_of(const Sizes &sizes);

// What the call keeps of the proxy.
struct ProxyPlan {
  at::Tensor num_chunks;          // int64 0-dim
  at::Tensor occurrence_count;    // int64 0-dim
  at::Tensor cell_uniq;           // int64 [num_nodes * num_features]
  at::Tensor uniq_off;            // int64 [num_nodes + 1]
  at::Tensor uniq_out;            // int64 [num_nodes]
  at::Tensor chunks;              // int32 [max_chunks][8]: Chunk
  at::Tensor chunks_out;          // int64 [num_nodes]
  at::Tensor node_chunk_start;    // int64 [num_nodes + 1]
  at::Tensor node_chunks;         // int32 [max_chunks]
  at::Tensor stage_slice;         // int64 [num_nodes]
  at::Tensor reduce_task_starts;  // int64 [max_chunks + 1]: its reducers' tasks before each chunk
                                  // (32 keys each)
  at::Tensor reducer_start;       // int32 [remote_keys + 1]: key u's entries are
                                  // reducer_entries[reducer_start[u] .. reducer_start[u + 1])
  at::Tensor reducer_entries;     // int64 [remote_keys]: a slot and the flag of the push task
                                  // that fills it (common.h)
};
// Step 3: each requester's push order (one stable sort of the push key),
// its push tables, and the reducer list. Writes remote_map's positions and
// the proxy header's push tables.
ProxyPlan proxy_push_order(ProxyKeys keys, ProxyChunks chunks, const at::Tensor &requester_start,
                           const ProxyBuffers &buffers, const Sizes &sizes);

// The rail's part of a call across nodes (the in channel). A
// rank's rail is the ranks with its local rank, one on each node. Each rank
// sends each rail peer a count message; from the messages it gets, a rank
// computes as proxy where its key records land at each owner, and as owner
// its KJT and where each proxy's cells start in it, which it sends back to
// that proxy. Then the needs and cell starts a rank publishes to its node,
// and the owner's tables: its KJT's keys and what its forward put list sends
// (planner.cu).
//
// An owner's KJT has num_sources = local_size + num_nodes - 1 cells per
// feature, feature-major: its node's requesters, then the other nodes in
// order. Its own node's cells are the own part of its node_keys; another
// node's cell holds the node-unique keys of that node's proxy on its rail,
// whose records the proxy puts into its rail_keys: one slice per other node
// in node order, each the chunks of this owner's node in the proxy's stream
// (ChunkPiece), then its records.

// The nodes but this rank's, in order (int64 [num_nodes - 1]): a constant of
// the op.
at::Tensor other_nodes_of(const Topo &topo, const at::TensorOptions &options);

// The count messages this rank sends its rail (int64
// [CountMessage::out_words], window.h): the common part, its node's
// largest remote part (need_remote), the keys of its own node it looks up as
// owner (own_total, its own part), and as proxy its node-unique keys and its
// chunks per owner node (uniq_out, chunks_out); then for each node its
// node-unique keys of each feature there (cell_uniq, [owner node][feature]).
at::Tensor count_message(const at::Tensor &need_remote, const at::Tensor &own_total,
                         const at::Tensor &uniq_out, const at::Tensor &chunks_out,
                         const at::Tensor &cell_uniq);

// The rail's layout, from the messages this rank got (int64 [num_nodes]
// [CountMessage::length], its own too) and its own part's cells (the node
// layout's own_cells). Each slice a proxy puts holds the owner node's chunks
// (ChunkPiece, 16 bytes each), then its records (record_bytes each).
struct RailLayout {
  // as proxy, to each owner of the rail (its slice in key_stage is the proxy's stage_slice)
  at::Tensor put_dst;    // int64 [num_nodes]: where its slice lands in the owner's rail_keys
                         // (bytes)
  at::Tensor put_bytes;  // int64 [num_nodes]
  at::Tensor put_skip;   // bool [num_nodes]: no put of keys (the owner overflows
                         // lookup_keys, or it is on this node)
  // as owner
  at::Tensor kjt_lengths;     // int64 [num_features * num_sources]: cell (feature, source)
  at::Tensor kjt_offsets;     // int64 [num_features * num_sources + 1]
  at::Tensor num_recv;        // int64 0-dim: the keys it looks up
  at::Tensor length_per_key;  // int64 [num_features]
  at::Tensor slice_in;        // int64 [num_nodes][2]: each source node's slice in its
                              // rail_keys: byte offset, chunks
  at::Tensor uniq_in;         // int64 [num_nodes]: the node-unique keys of each source node
  at::Tensor cells_in;        // int64 [num_nodes][num_features]: of each source node, per
                              // feature (its own node: 0)
  at::Tensor cell_stage;      // int64 [num_nodes][num_features]: where the cells of each
                              // source node start in its KJT, what it sends the node's proxy
                              // (its own node: 0)
  at::Tensor my_cell_starts;  // int64 [local_size][num_features]: where each requester of
                              // its node's cell (feature, requester) starts in its KJT
  at::Tensor needs;           // int64 [kNumNeeds]: the most the rail needs of lookup_keys
                              // and remote_keys
};
RailLayout rail_layout(const at::Tensor &messages, const at::Tensor &own_cells,
                       const at::Tensor &other_nodes, const Sizes &sizes);

// What this rank publishes to its node (int64 [kNumNeeds + local_size *
// num_features]): its rail's needs, then my_cell_starts.
at::Tensor status_publish(const RailLayout &rail);

// The owner's tables, once the rail's proxies put their records into its
// rail_keys (int64 view) and its node's requesters their keys into the own
// part of its node_keys (node_keys, node_weights: views of [lookup_keys];
// node_weights undefined when the keys carry none).
struct OwnerTables {
  at::Tensor kjt_values;       // int64 [lookup_keys]: the first num_recv are its KJT's keys
  at::Tensor kjt_weights;      // fp32 [lookup_keys], or undefined (8-byte records): their
                               // weights, 1 for its own node's keys when they carry none
  at::Tensor fwd_chunks;       // int32 [owner_chunks][4]: what its forward put list sends, the
                               // chunks of its node in each source node's stream (ChunkPiece,
                               // first in its slice), source node after source node
  at::Tensor fwd_chunk_start;  // int64 [num_nodes + 1]: where each source node's start there
};
OwnerTables owner_tables(const RailLayout &rail, const at::Tensor &own_starts,
                         const at::Tensor &node_keys, const at::Tensor &node_weights,
                         const at::Tensor &rail_keys, const at::Tensor &other_nodes,
                         const Sizes &sizes);

// The requester's part of a call across nodes (the in channel): its maps to
// the rows of its keys, from what its node's proxies published (two
// peer_gathers: their headers, and its rows' words of their remote_map) and
// where its owners on its node put its cells (planner.cu).

// What a peer_gather reads (GatherFrom says which words): for each item a
// node peer (negative: none) and its first int32 word to read there.
struct PeerWords {
  at::Tensor peer;        // int32 [items]
  at::Tensor first_word;  // int64 [items]
};
// The remote_map words of this requester's rows (row_cell, block_start: part
// 1's; destination_base: the node layout's), kGatherRemoteMap: a row of
// another node's owner is occurrence destination - lookup_keys of its proxy,
// the owner's rank on this node; it reads that entry of each RemoteMap array.
PeerWords remote_rows(const at::Tensor &row_cell, const at::Tensor &block_start,
                      const at::Tensor &destination_base, const Sizes &sizes);
// The words of the node's proxy headers this requester reads
// (kGatherProxyHeader): ProxyHeader's gathered words of each proxy in turn,
// a constant of the op.
PeerWords header_gather(const Sizes &sizes, const at::TensorOptions &options);

struct RequesterTables {
  at::Tensor src, dst;         // int32 [row_cell.numel()]: where each row's key is in the
                               // forward (a row of its node: its owner's KJT; else its proxy's
                               // proxy_rows), and where its gradient goes in the backward
  at::Tensor push_order;       // int32 [row_cell.numel()]: its rows at each proxy in their
                               // push order
  at::Tensor push_order_off;   // int64 [local_size + 1]: where its rows at each proxy start
  at::Tensor push_tables;      // int32 [local_size][max_chunks + 1][4]: ReqEntry at each proxy,
                               // positions in push_order, groups in heavy_groups
  at::Tensor proxy_chunks;     // int64 [local_size]: the chunks of each proxy's stream
  at::Tensor proxy_ends;       // int64 [local_size][max_chunks + 1]: each proxy's chunk ends,
                               // then its stream's end (past num_proxy_rows when it wraps)
  at::Tensor fwd_task_starts;  // int64 [max_chunks * local_size + 1]: its forward's tasks of
                               // rows of other nodes before each (chunk, proxy in turn)
  at::Tensor bwd_task_starts;  // int64 [2 * max_chunks * local_size + 1]: its backward's before
                               // each (chunk, heavy groups or rows, proxy in turn)
  at::Tensor heavy_groups;     // int32 [group bound][2]: HeavyGroup, positions of push_order
  at::Tensor group_first_sub;  // int32 [group bound]: each group's first sub-task
  at::Tensor group_of_sub;     // int32 [sub-task bound]: the group of each sub-task
  at::Tensor group_subs;       // int64 0-dim: the sub-tasks of kHeavy positions
};
// words: int32 [3][row_cell.numel()], the remote_map words of its rows
// (remote_rows' gather; rows of its node's owners and past the valid rows hold
// anything); headers: int32 [local_size][ProxyHeader::gathered_words]
// (header_gather's);
// owner_cell_start: int64 [local_size][num_features], where its cell
// (feature, this requester) starts in each owner's KJT on its node
// (node_status_exchange).
RequesterTables requester_tables(const at::Tensor &words, const at::Tensor &headers,
                                 const at::Tensor &owner_cell_start, const at::Tensor &row_cell,
                                 const at::Tensor &cell_start, const at::Tensor &block_start,
                                 const Sizes &sizes);

// The put lists of a call across nodes (the in channel), made in one step:
// the rows that cross the rail between an owner's KJT and a proxy's
// proxy_rows. The proxy's backward list sends the summed gradient rows of
// its node-unique keys to their owners; the owner's forward list sends its
// rows to the proxies that asked for them. Each cuts a node's keys at its
// chunks and at its feature cells, so that a segment lies in one chunk and
// one cell (planner.cu).

// Both lists, in one tensor, of 2 * num_nodes nodes: the backward's owner
// nodes, then the forward's proxy nodes. A list's segments are in (node,
// key) order, so a node's chunks follow each other in stream order.
struct PutLists {
  at::Tensor
      list;  // int32 [bound][4]: PutSeg {src, dst, rows, chunk}; the backward's src is a
             // row of proxy_rows, its dst a row of the owner's KJT and its chunk the
             // proxy's; the forward's the other way round, its chunk one of the owner's fwd_chunks
  at::Tensor starts;  // int32 [2 * num_nodes + 1]: where each node's segments start
};
// The backward list from the proxy's chunks (chunks, node_chunk_start,
// node_chunks: ProxyChunks'), its node-unique keys per (owner node,
// feature) and where its cells start in each rail owner's KJT (rail_cells,
// int64 [num_nodes][num_features], as the owners put them). The forward list
// from the chunks of this owner's node in each source node's stream
// (fwd_chunks, fwd_chunk_start: OwnerTables'), its keys per feature
// (cells_in) and where those cells start in the owner's KJT (cell_stage).
PutLists put_lists(const at::Tensor &chunks, const at::Tensor &node_chunk_start,
                   const at::Tensor &node_chunks, const at::Tensor &uniq_off,
                   const at::Tensor &cell_uniq, const at::Tensor &rail_cells,
                   const at::Tensor &fwd_chunks, const at::Tensor &fwd_chunk_start,
                   const at::Tensor &cells_in, const at::Tensor &cell_stage, const Sizes &sizes);

}  // namespace hier_output_dist
}  // namespace dyn_emb
