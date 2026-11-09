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
// device. The bucketize, the owner's KJT on one node, the proxy, the rail and
// the owner, and the requester. node_exchange makes the node
// layout (node_layout_small and node_layout_items, node.cuh); node_layout()
// and push_to_node_keys() here are the hooks'. Every table of a step is sized by the op's sizes
// (Sizes, from its configuration: config.h), never by a device value.
#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include <cstdint>
#include <utility>

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
// The offsets of n counts (bool or int32), [0, c0, c0 + c1, ...]: n + 1
// entries, int64 (or int32 when their sum stays below 2^31).
at::Tensor offsets_of(const at::Tensor &counts, at::ScalarType dtype = at::kLong);

// The bucketize of a call (the requester's, local, on the caller's stream):
// its keys by owner, as TorchRec's bucketize orders them, and the fan-out of
// its deduplicated rows: kernels and a CUB sort each (bucketize.cu).

// The keys of a call as the bucketize takes them: the KJT the input dist was
// given, index-deduplicated (each table's keys once: only its first keys are
// valid, and only the device knows how many, its offsets say), and of the
// input KJT before index dedup, which construct_jagged_tensors reads, its
// offsets and the map of its rows to the KJT's.
struct PlanInput {
  at::Tensor values;                  // its keys, int32 or int64
  c10::optional<at::Tensor> offsets;  // [num_features * batch_size + 1], or
  c10::optional<at::Tensor> lengths;  // [num_features * batch_size] when it has none
  int64_t batch_size = 0;
  c10::optional<at::Tensor> weights;  // fp32, one per key
  at::Tensor input_offsets;           // [num_features * batch_size + 1]: the input KJT's
  at::Tensor reverse_indices;         // int64 [original rows]: original row -> row of the KJT
};

// A call's keys by owner, bucketized: [owner][feature][sample][position],
// the order of TorchRec's bucketize. B is values.numel(); the valid keys are
// offset_per_bucket[world_size] of them (index dedup leaves fewer). A
// requester's "bucketized rows" are these, in this order.
struct BucketizedKeys {
  at::Tensor keys;         // int64 [B]: each key, widened to int64 with its sign (bucketize_keys
                           // only)
  at::Tensor key_weights;  // fp32 [B], or undefined: their weights (bucketize_keys only)
  at::Tensor values;       // int32 or int64 [B]: the KJT's keys (input.values, no copy), which the
                           // push gathers
  at::Tensor inv;          // int32 [B]: the KJT row of each (a permutation; row past the valid)
  at::Tensor bucket_feature_of_row;  // int32 [B]: the bucket feature owner * num_features + feature
                                     // of each, world_size * num_features past the valid
  at::Tensor offset_per_bucket_feature;  // int64 [world_size * num_features + 1]: where each bucket
                                         // feature starts
  at::Tensor offset_per_bucket;          // int64 [world_size + 1]: where each bucket (owner) starts
  at::Tensor input_length_per_key;       // int64 [num_features]: keys per feature of the input KJT
};

// The bucketize places the keys of each feature on their owners by the feature's rule, its
// dist_types entry (int32 [num_features] on the keys' GPU): a DistType code (src/utils.h),
// kCyclic or kHashedCyclic (Config::validate refuses the others), which it reads at run time.

// The checks of the bucketize's inputs (the keys, offsets or lengths, the input KJT's offsets,
// weights, the rules) and of the fan-out's (reverse_indices). Each throws (TORCH_CHECK) with the
// message of its check. InChannel::input_dist calls both before it takes a sequence number.
// bucketize_rows and fanout_tables call them too (the hooks).
void check_bucketize_input(const PlanInput &input, const at::Tensor &dist_types,
                           int64_t world_size);
void check_fanout_input(const at::Tensor &reverse_indices);
// The bucketize up to its gather: the in channel's (its push gathers the keys as it pushes them).
// keys and key_weights stay undefined.
BucketizedKeys bucketize_rows(const PlanInput &input, const at::Tensor &dist_types,
                              int64_t world_size);
// The whole bucketize: bucketize_rows, then the keys (and weights) gathered into bucketized order
// (gather_rows): the tests'.
BucketizedKeys bucketize_keys(const PlanInput &input, const at::Tensor &dist_types,
                              int64_t world_size);

// Index dedup's fan-out of the deduplicated rows to the original rows, in the
// format of Fanout (common.h): a row's original rows are
// exp_idx[exp_off[row] .. exp_off[row + 1]), in increasing order. A row of
// more than kHeavy is heavy: it takes a heavy slot and consecutive sub-tasks
// of kHeavy original rows, heavy slot s the sub-tasks
// [first[s], first[s + 1]) (the last slot's up to count[1]). The heavy slots
// go in the order the kernel reserves them, which can change from run to
// run; no sum depends on it (bucketize.cu, fanout_runs_kernel). Entries past
// `count` are clamped garbage nobody reads.
struct FanoutTables {
  at::Tensor exp_off;     // int32 [rows + 1]
  at::Tensor exp_idx;     // int32 [original rows]
  at::Tensor heavy_slot;  // int32 [rows]: the heavy slot of a heavy row, else -1
  at::Tensor sub;         // int64 [2 * max_subs]: (row, first entry in exp_idx) of each sub-task
  at::Tensor count;       // int64 [2]: heavy rows, sub-tasks
  at::Tensor pos;         // int32 [max_heavy_slots]: the bucketized row each slot's row went to
  at::Tensor first;       // int32 [max_heavy_slots]: the first sub-task of each slot
};
// The words the fan-out's kernel keeps across its CTAs: the counter of the
// heavy rows' slots and sub-tasks, and the CTA ticket (int64 each). They are
// 0 between calls: the kernel's last CTA resets them.
constexpr int64_t kFanoutWords = 2;
// From reverse_indices (original row -> deduplicated row, in [0, rows)) and
// the bucketize's inv, whose size is the number of deduplicated rows (rows).
// words: int64 [kFanoutWords] on the GPU of inv, zeroed once, which the calls
// share (the in channel's: its input dists run on one stream); undefined, a
// zeroed tensor of this call's (the hook's).
FanoutTables fanout_tables(const at::Tensor &reverse_indices, const at::Tensor &inv,
                           const at::Tensor &words = at::Tensor());

// The node's part of a call (the in channel): the layout every rank of the node computes alike
// from the node's offsets, and the push of this rank's keys into their places (planner.cu,
// node.cuh).
//
// Each requester pushes its bucketize's offset_per_bucket_feature into its row of every node
// peer's node_offsets (window.h). From the node's offsets each rank knows where every key of the
// node goes, and pushes its keys straight to their places in their targets' node_keys.
// node_keys of a target holds its own part at [0, lookup_keys), the keys of its node for the
// target as owner, in the owner's KJT order ([feature][requester][requester's order]); and across
// nodes its remote part at [lookup_keys, lookup_keys + remote_keys), the keys of its node for the
// owners on its rail of the other nodes, which it deduplicates as their proxy: requester-major,
// then owner node, then [feature][position].

// The layout of the node's keys, from the node's offsets. Under a need above a capacity no rank
// of the node pushes (skip). One node has no remote part: there remote_offsets and need_remote
// stay undefined. Across nodes kjt_offsets and length_per_key stay undefined.
struct NodeLayout {
  at::Tensor own_lengths;     // int64 [num_features * local_size]: keys of each (feature,
                              // requester) in this rank's own part
  at::Tensor own_offsets;     // int64 [num_features * local_size]: where each one
                              // starts there
  at::Tensor own_total;       // int64 [local_size]: each owner's own part
  at::Tensor remote_offsets;  // int64 [local_size + 1]: where each requester's keys start in
                              // this rank's remote part
  at::Tensor need_lookup;     // int64 0-dim: the largest own part of the node
  at::Tensor need_remote;     // int64 0-dim: the largest remote part of the node
  at::Tensor skip;            // bool 0-dim: a need above its capacity
  at::Tensor row_shift;       // int64 [world_size * num_features + 1]: bucketized row `row`
                              // goes to row + row_shift[bucket_feature_of_row[row]] in
                              // node_keys of its target: where this rank's rows of the
                              // bucket feature start there, less where they start among its
                              // rows (offset_per_bucket_feature)
  at::Tensor kjt_offsets;     // (one node) int64 [num_features * local_size + 1]
  at::Tensor length_per_key;  // (one node) int64 [num_features]
};
// The layout as node_exchange makes it, from the node's offsets (node_offsets: int64
// [local_size][world_size * num_features + 1], row r requester r's offset_per_bucket_feature):
// from the rows the requesters push (NodeRow, window.h). items_in_cta: whether one CTA lays out
// the items (node_layout_items, node.cuh) or a grid does (none: as node_exchange chooses).
NodeLayout node_layout(const at::Tensor &node_offsets, const Topo &topo, int64_t num_features,
                       int64_t lookup_keys, int64_t remote_keys,
                       c10::optional<bool> items_in_cta = c10::nullopt);
// The tensors of a NodeLayout, empty: remote_offsets and need_remote across nodes, kjt_offsets
// and length_per_key on one node.
NodeLayout empty_node_layout(const Topo &topo, int64_t num_features,
                             const at::TensorOptions &options);

// The push of node_key_push (push_rows, node.cuh) into a test's stand-ins for its targets'
// node_keys: row t of node_keys ([local_size][lookup_keys + remote_keys]), of node_key_weights
// (the same shape, or undefined) and across nodes of node_bucket_feature_of_row
// ([local_size][remote_keys]) is target t's. Its inputs are the KJT's keys (values: int32 or
// int64) and weights (or undefined), bucketize_rows' outputs (inv, bucket_feature_of_row,
// offset_per_bucket) and the node layout. With src defined (one node) it also writes src. A place
// outside a stand-in's row is not stored.
void push_to_node_keys(const at::Tensor &values, const at::Tensor &inv, const at::Tensor &weights,
                       const at::Tensor &bucket_feature_of_row, const at::Tensor &offset_per_bucket,
                       const NodeLayout &node, const Topo &topo, int64_t num_features,
                       int64_t lookup_keys, int64_t remote_keys, const at::Tensor &node_keys,
                       const at::Tensor &node_key_weights,
                       const at::Tensor &node_bucket_feature_of_row, const at::Tensor &src);

// On one node, the KJT of this rank as owner, a copy of its own part of node_keys once every
// requester of the node pushed (node_keys, and node_key_weights when the keys carry weights:
// views of the own part), and its map as requester (src: its rows' places in their owners' KJTs,
// which the push wrote).
struct OneNodeOwner {
  at::Tensor kjt_values;      // int64 [lookup_keys]: the first num_recv are the KJT's
  at::Tensor kjt_weights;     // fp32 [lookup_keys], or undefined
  at::Tensor kjt_lengths;     // int64 [num_features * local_size]: per (feature, requester)
  at::Tensor kjt_offsets;     // int64 [num_features * local_size + 1]
  at::Tensor num_recv;        // int64 0-dim: the keys it looks up
  at::Tensor length_per_key;  // int64 [num_features]
  at::Tensor src;             // int32 [rows]: the place of each row in its owner's KJT (and
                              // its dst)
};
OneNodeOwner one_node_owner(const NodeLayout &node, const at::Tensor &node_keys,
                            const at::Tensor &node_key_weights, const at::Tensor &src,
                            const Topo &topo);

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
// A node-unique key is an identity (table, key), or each occurrence without
// node dedup. Its representative is its occurrence of the smallest bucket
// feature (owner node, feature), and of those the first. Its key index u
// numbers the node-unique keys by their representative's bucket feature
// (owner node after owner node, each owner node's by the smallest feature of
// the key), then, under node dedup, with the requesters in turn: the first
// representative of each requester there, requester after requester, then
// the second, and so on, each requester's in its order; without node dedup
// requester after requester (proxy.cu: key_of_representative). Index
// dedup leaves each requester each (table, key) once, so a key has at most
// local_size occurrences (Sizes::max_key_rows) and a row of proxy_rows for
// each. Entries past the valid part of the remote part, and keys past the
// valid ones, hold garbage that no one reads.

// The proxy's inputs: its remote part (views of its buffers) and what the
// node layout says of it.
struct ProxyInput {
  at::Tensor node_keys;                   // int64 [remote_keys]: the remote part of node_keys
  at::Tensor node_key_weights;            // fp32 [remote_keys]: their weights, or undefined
  at::Tensor node_bucket_feature_of_row;  // int32 [remote_keys]: owner node * num_features +
                                          // feature of each
  at::Tensor remote_offsets;              // int64 [local_size + 1]: each requester's occurrences
  at::Tensor skip;                        // bool 0-dim: the node skips the call (it pushed nothing)
  at::Tensor tables;  // int32 [num_features]: the table of each feature, numbered
                      // densely (dense_tables)
};

// The buffers the proxy writes, views of this rank's copy of the window.
struct ProxyBuffers {
  at::Tensor remote_map;  // int32 [3][remote_keys] (RemoteMap): per occurrence its key's row,
                          // its slot, its position in its requester's push order
  at::Tensor key_stage;   // int64 [KeysLayout::key_stage_bytes / 8]: per other owner node a
                          // slice of its chunks (OwnerChunk) and its keys' records
  // the proxy header's parts (ProxyHeader)
  at::Tensor num_chunks;   // int32 [1]: its stream's chunks
  at::Tensor end;          // int32 [1]: its stream's end (uint32)
  at::Tensor ends;         // int32 [max_chunks]: each chunk's end (uint32)
  at::Tensor push_tables;  // int32 [local_size][max_chunks + 1]: where each requester's rows of
                           // each chunk start, relative to its push order here
};
// The buffers the proxy writes: remote_map and key_stage (views), and the
// parts of the proxy header `header`, viewed on the device of `options`.
ProxyBuffers proxy_buffers(at::Tensor remote_map, at::Tensor key_stage, const ProxyHeader &header,
                           const at::TensorOptions &options);

// Step 1: the node-unique keys, by a hash table of the identities and a scan
// of their representatives (proxy.cu). Node dedup is index dedup over the
// node's keys, and its grouping by owner node is the bucketize: their tables
// take index dedup's and the bucketize's names with a node_ prefix
// (docs/hier_output_dist.md, section 10.4). The tables per occurrence and per
// key hold garbage past the valid ones (occurrence_count, node_num_uniques).
struct ProxyKeys {
  at::Tensor occurrence_count;  // int64 0-dim: the valid occurrences (0 when the node skips)
  at::Tensor node_num_uniques;  // int64 0-dim: the node-unique keys
  at::Tensor need_key_rows;     // int64 0-dim: the most occurrences of one (kNeedKeyRows)
  // per occurrence occ
  at::Tensor node_reverse_indices;  // int32 [remote_keys]: its node-unique key u
  at::Tensor occurrence_rank;       // int32 [remote_keys]: its key's occurrences before it, in
                                    // occ order (max_key_rows for a key of more occurrences)
  // per node-unique key u
  at::Tensor key_occurrences;                 // int32 [remote_keys]: its occurrences
  at::Tensor key_node;                        // int32 [remote_keys]: its owner node
  at::Tensor node_unique_keys;                // int64 [remote_keys]: the key
  at::Tensor node_unique_key_weights;         // fp32 [remote_keys]: the weight of its record, its
                                              // occurrences' weights, or undefined (8-byte records)
  at::Tensor node_length_per_bucket_feature;  // int64 [num_nodes * num_features]: the keys of each
                                              // (owner node, feature)
  at::Tensor node_offset_per_bucket;  // int64 [num_nodes + 1]: the first key of each owner node
  at::Tensor node_length_per_bucket;  // int64 [num_nodes]: the keys of each owner node
};
// node_dedup: equal keys of a table are one node-unique key. A table's rule
// places equal keys on one owner when the node's requesters give the op keys
// of one dtype. Both rules work in the key's own width, so an int32 key and
// the same int64 key can have different owners. The op's contract excludes
// that, but the proxy would then serve both from its representative's owner
// node, the smallest of them. Weights must be whole numbers (frequency
// counts): they are summed as int64.
ProxyKeys proxy_keys(const ProxyInput &input, const Sizes &sizes, bool node_dedup);

// Step 2: the rows of the keys, the chunks of the stream and where they
// lie, each occurrence's slot. A key's rows are its key row and its
// duplicates' slots, a row per occurrence (max_key_rows at most: a key of
// more occurrences, which the call raises for, takes max_key_rows, and all
// its occurrences take its key row). Writes remote_map's key rows and
// slots, the chunks and records of key_stage, and the proxy header's chunks,
// end and ends; drops keys.node_unique_keys and keys.node_unique_key_weights, which only the
// records need, and keys.occurrence_rank, which only the slots need.
struct ProxyChunks {
  at::Tensor num_chunks;  // int64 0-dim: the chunks of the stream
  at::Tensor chunks;      // int32 [max_chunks][8]: Chunk, in stream order (zeros past them)
  at::Tensor chunk_length_per_bucket;  // int64 [num_nodes]: the chunks of each owner node
  at::Tensor chunk_offset_per_bucket;  // int64 [num_nodes + 1]: where each owner node's chunks
                                       // start in chunk_inv
  at::Tensor chunk_inv;  // int32 [max_chunks]: each owner node's chunks in order, as their
                         // places in the stream
  at::Tensor stage_offset_per_bucket;  // int64 [num_nodes]: where its slice for each owner node
                                       // starts in key_stage (bytes)
  at::Tensor reduce_task_offsets;  // int64 [max_chunks + 1]: the reducers' tasks before each chunk
  at::Tensor node_exp_off;         // int64 [remote_keys + 1]: the rows before each key
  at::Tensor key_chunk;            // int32 [remote_keys]: the chunk of each key
  at::Tensor requester_of;         // int32 [remote_keys]: the requester of each occurrence
};
ProxyChunks proxy_chunks_and_slots(ProxyKeys &keys, const at::Tensor &remote_offsets,
                                   const ProxyBuffers &buffers, const Sizes &sizes);

// What the call keeps of the proxy.
struct ProxyPlan {
  at::Tensor num_chunks;                      // int64 0-dim
  at::Tensor occurrence_count;                // int64 0-dim
  at::Tensor need_key_rows;                   // int64 0-dim
  at::Tensor node_length_per_bucket_feature;  // int64 [num_nodes * num_features]
  at::Tensor node_offset_per_bucket;          // int64 [num_nodes + 1]
  at::Tensor node_length_per_bucket;          // int64 [num_nodes]
  at::Tensor chunks;                          // int32 [max_chunks][8]: Chunk
  at::Tensor chunk_length_per_bucket;         // int64 [num_nodes]
  at::Tensor chunk_offset_per_bucket;         // int64 [num_nodes + 1]
  at::Tensor chunk_inv;                       // int32 [max_chunks]
  at::Tensor stage_offset_per_bucket;         // int64 [num_nodes]
  at::Tensor reduce_task_offsets;  // int64 [max_chunks + 1]: its reducers' tasks before each chunk
                                   // (32 keys each)
  at::Tensor node_exp_off;         // int64 [remote_keys + 1]: the rows before each key
                                   // (ProxyChunks::node_exp_off): key u's are its key row, then its
                                   // duplicates' slots in order
};
// Step 3: each requester's push order (one stable sort by its part, a
// requester and a chunk) and its push tables. Writes remote_map's positions
// and the proxy header's push tables.
ProxyPlan proxy_push_order(ProxyKeys keys, ProxyChunks chunks, const at::Tensor &remote_offsets,
                           const ProxyBuffers &buffers, const Sizes &sizes);

// The rail's part of a call across nodes (the in channel). A rank's rail is the
// ranks with its local rank, one on each node. Each rank sends every rail peer
// a count message, the same header to all (the rail allgathers it) and the
// peer's own part; from the rail's messages every rank computes the rail's
// layout (rail_layout_of, rail.cuh): as proxy where its key records land at
// each owner and where their rows lie in each owner's lookup_rows (the same
// function as the owners'), as owner its KJT and where each proxy's rows lie
// in its lookup_rows, and the rail's needs, which with its group shifts a rank
// publishes to its node. Then the owner's tables: its KJT's keys, the chunks
// its forward puts and the row map of its lookup (planner.cu).
//
// An owner's KJT has kjt_stride = local_size + num_nodes - 1 lengths per
// feature (its stride), feature-major: its node's requesters, then the other
// nodes in order. Its own node's keys are the own part of its node_keys;
// another node's are the node-unique keys of that node's proxy on its rail,
// whose records the proxy puts into its rail_keys: one slice per other node
// in node order, each the chunks of this owner's node in the proxy's stream
// (OwnerChunk), then its records. Its lookup_rows hold each lookup group's
// rows in a slice of their own, in the order they arrive (common.h,
// GroupRuns), so a chunk's rows cross the rail in a put per group.

// The nodes but this rank's, in order (int64 [num_nodes - 1]): a constant of
// the op.
at::Tensor other_nodes_of(const Topo &topo, const at::TensorOptions &options);

// The count messages this rank sends its rail peers (int64 [num_nodes][CountMessage::length], row
// n to the peer on node n, window.h), as rail_exchange makes them (count_messages_of, rail.cuh): a
// header, the same in every row: its node's largest remote part (need_remote), as proxy the most
// occurrences of one of its keys (need_key_rows), the keys of each lookup group (group_starts:
// int64 [num_groups + 1], each group's first feature, then num_features) of its own part
// (own_lengths [feature][requester], summed over its node's requesters), as proxy its chunks per
// owner node (chunk_length_per_bucket) and its node-unique keys of each (owner node, group); then
// row n's own part: its node-unique keys of each feature of node n
// (node_length_per_bucket_feature). The hooks'.
at::Tensor count_message(const at::Tensor &need_remote, const at::Tensor &own_lengths,
                         const at::Tensor &need_key_rows, const at::Tensor &chunk_length_per_bucket,
                         const at::Tensor &node_length_per_bucket_feature,
                         const at::Tensor &group_starts);

// The rail's layout, from the rail's count messages to this rank (int64 [num_nodes]
// [CountMessage::length], row n the message of the rail peer on node n, this rank's own too) and
// its own part's lengths (the node layout's own_lengths). Each slice a proxy puts holds the owner
// node's chunks (OwnerChunk, 16 bytes each), then its records (record_bytes each).
struct RailLayout {
  // as proxy, to each owner of the rail (its slice in key_stage is the proxy's
  // stage_offset_per_bucket)
  at::Tensor put_dst;     // int64 [num_nodes]: where its slice lands in the owner's rail_keys
                          // (bytes)
  at::Tensor put_bytes;   // int64 [num_nodes]
  at::Tensor put_skip;    // bool [num_nodes]: no put of keys (the owner overflows
                          // lookup_keys, or it is on this node)
  at::Tensor bwd_groups;  // int64 [num_nodes][2 * num_groups + 1]: the group runs (GroupRuns,
                          // common.h) of its node-unique keys of each owner node (its own: 0)
  // as owner
  at::Tensor kjt_lengths;             // int64 [num_features * kjt_stride]: per (feature, column)
  at::Tensor kjt_offsets;             // int64 [num_features * kjt_stride + 1]
  at::Tensor num_recv;                // int64 0-dim: the keys it looks up
  at::Tensor length_per_key;          // int64 [num_features]
  at::Tensor slice_in;                // int64 [num_nodes][2]: each source node's slice in its
                                      // rail_keys: byte offset, chunks
  at::Tensor node_length_per_source;  // int64 [num_nodes]: the node-unique keys of each source node
  at::Tensor node_length_per_source_feature;  // int64 [num_nodes][num_features]: of each source
                                              // node, per feature (its own node: 0)
  at::Tensor fwd_groups;     // int64 [num_nodes][2 * num_groups + 1]: the group runs of the records
                             // of each source node (its own node: 0)
  at::Tensor group_offsets;  // int64 [num_groups + 1]: where each lookup group's slice starts in
                             // its lookup_rows (then the keys it looks up)
  at::Tensor group_shift;    // int64 [num_groups]: per lookup group, the rows before its own part
                             // less the own part's keys before it (CtrlLayout::group_shift)
  at::Tensor needs;          // int64 [kNumNeeds]: the most the rail needs of each
                             // capacity (Need)
};
// The layout as rail_exchange makes it (rail_layout_of, rail.cuh). The hooks'.
RailLayout rail_layout(const at::Tensor &messages, const at::Tensor &own_lengths,
                       const Sizes &sizes);
// The tensors of a RailLayout, empty.
RailLayout empty_rail_layout(const Sizes &sizes, const at::TensorOptions &options);

// The owner's tables, once the rail's proxies put their records into its
// rail_keys (int64 view) and its node's requesters their keys into the own
// part of its node_keys (node_keys, node_key_weights: views of [lookup_keys];
// node_key_weights undefined when the keys carry none).
struct OwnerTables {
  at::Tensor kjt_values;         // int64 [lookup_keys]: the first num_recv are its KJT's keys
  at::Tensor kjt_weights;        // fp32 [lookup_keys]: their weights, or undefined (8-byte
                                 // records)
  at::Tensor fwd_chunks;         // int32 [max_owner_chunks][4]: what its forward puts, the chunks
                                 // of its node in each source node's stream (OwnerChunk, first in
                                 // its slice), source node after source node
  at::Tensor fwd_chunk_offsets;  // int64 [num_nodes + 1]: where each source node's start there
  at::Tensor row_map;            // int32 [lookup_keys]: per key of its KJT (the first num_recv),
                                 // its row in its lookup group's slice of lookup_rows (common.h,
                                 // GroupRuns), where the lookup writes it; undefined when key i of
                                 // a group's keys takes row i (no group of more than one feature)
};
// group_of_feature: int32 [num_features], the lookup group of each feature; group_starts: int64
// [num_groups + 1], each group's first feature, then num_features.
OwnerTables owner_tables(const RailLayout &rail, const at::Tensor &own_offsets,
                         const at::Tensor &node_keys, const at::Tensor &node_key_weights,
                         const at::Tensor &rail_keys, const at::Tensor &other_nodes,
                         const at::Tensor &group_of_feature, const Sizes &sizes);

// The requester's part of a call across nodes (the in channel): its maps to
// the rows of its keys, from what its node's proxies published (two
// peer_gathers: their headers, and its rows' words of their remote_map) and
// where its owners on its node put the rows of its keys (planner.cu).

// What a peer_gather reads (GatherFrom says which words): for each item a
// node peer (negative: none) and its first int32 word to read there.
struct PeerWords {
  at::Tensor peer;        // int32 [items]
  at::Tensor first_word;  // int64 [items]
};
// The remote_map words of this requester's rows (bucket_feature_of_row,
// offset_per_bucket: the bucketize's; row_shift: the node layout's),
// kGatherRemoteMap: a row of another node's owner is occurrence
// row + row_shift - lookup_keys of its proxy, the owner's rank on this node
// (its place in its proxy's remote part, where it pushed it); it reads that
// entry of each RemoteMap array.
PeerWords remote_map_gather(const at::Tensor &bucket_feature_of_row,
                            const at::Tensor &offset_per_bucket, const at::Tensor &row_shift,
                            const Sizes &sizes);
// The words of the node's proxy headers this requester reads
// (kGatherProxyHeader): ProxyHeader's gathered words of each proxy in turn,
// a constant of the op.
PeerWords header_gather(const Sizes &sizes, const at::TensorOptions &options);

struct RequesterTables {
  at::Tensor src, dst;    // int32 [bucket_feature_of_row.numel()]: where each row's key is in the
                          // forward (a row of its node: its owner's KJT; else its proxy's
                          // proxy_rows), and where its gradient goes in the backward
  at::Tensor push_order;  // int32 [bucket_feature_of_row.numel()]: its rows at each proxy in their
                          // push order
  at::Tensor push_order_off;     // int64 [local_size + 1]: where its rows at each proxy start
  at::Tensor proxy_push_tables;  // int32 [local_size][max_chunks + 1]: where its rows of each chunk
                                 // at each proxy start in push_order (chunk max_chunks: their end)
  at::Tensor proxy_num_chunks;   // int64 [local_size]: the chunks of each proxy's stream
  at::Tensor proxy_ends;         // int64 [local_size][max_chunks + 1]: each proxy's chunk ends,
                                 // then its stream's end (past num_proxy_rows when it wraps)
  at::Tensor remote_task_offsets;  // int64 [max_chunks * local_size + 1]: its tasks of rows of
                                   // other nodes before each (chunk, proxy in turn), the forward's
                                   // pulls and the backward's pushes alike
};
// words: int32 [3][bucket_feature_of_row.numel()], the remote_map words of its
// rows (remote_map_gather's; rows of its node's owners and past the valid rows
// hold anything); headers: int32 [local_size][ProxyHeader::gathered_words]
// (header_gather's); owner_group_shift: int64 [local_size][num_groups], each
// owner's group shifts on its node (node_status_exchange: a row of an owner of
// its node lies at its place in the owner's own part, row + row_shift, plus
// its group's shift); row_shift: the node layout's; group_of_feature: int32
// [num_features].
RequesterTables requester_tables(const at::Tensor &words, const at::Tensor &headers,
                                 const at::Tensor &owner_group_shift,
                                 const at::Tensor &bucket_feature_of_row,
                                 const at::Tensor &row_shift, const at::Tensor &group_of_feature,
                                 const at::Tensor &offset_per_bucket, const Sizes &sizes);

}  // namespace hier_output_dist
}  // namespace dyn_emb
