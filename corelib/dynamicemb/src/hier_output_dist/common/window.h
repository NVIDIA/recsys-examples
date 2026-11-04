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
// The op's window and its sections: the control words (CtrlLayout) with
// the formats of the count vector and the count message, the in channel's
// data (KeysLayout) with the proxy header and remote_map, the out channel's
// data (RowsLayout), and the window (WindowLayout).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "hier_output_dist/common/common.h"

namespace dyn_emb {
namespace hier_output_dist {

// The ctrl section of the window (CtrlLayout): the control words of both
// channels and the device state they keep across calls, and the formats of
// two messages that live there, the count vector and the count message.

// A rank's count vector (int64 words, planner.h): the keys it pushes each
// GPU of its node, for their remote parts, then for each feature of their
// own parts.
struct CountVector {
  int64_t local_size, num_features;
  __host__ __device__ int64_t words() const { return local_size * (num_features + 1); }
  // Its keys for the remote part of node peer `target`.
  __host__ __device__ int64_t remote(int64_t target) const { return target; }
  // Its keys of `feature` for the own part of node peer `target`.
  __host__ __device__ int64_t own(int64_t target, int64_t feature) const {
    return local_size + target * num_features + feature;
  }
};

// A count message (int64 words, planner.h), all a rail peer needs to lay
// out the keys its rail puts: a common part, the same for every peer (its
// node's largest remote part, the keys of its own node it looks up as owner,
// and as proxy its node-unique keys and its chunks per owner node), then the
// part for the peer's node alone (its node-unique keys of each feature
// there).
struct CountMessage {
  static constexpr int kNeedRemote = 0, kOwnTotal = 1;
  int64_t num_nodes, num_features;
  __host__ __device__ int64_t uniq_out(int64_t owner_node) const { return 2 + owner_node; }
  __host__ __device__ int64_t chunks_out(int64_t owner_node) const {
    return 2 + num_nodes + owner_node;
  }
  __host__ __device__ int64_t common() const { return 2 + 2 * num_nodes; }
  // A message as a peer gets it: the common part, then the part for its node.
  __host__ __device__ int64_t length() const { return common() + num_features; }
  // What a rank sends from: the common part, then the part for each node.
  __host__ __device__ int64_t out_words() const { return common() + num_nodes * num_features; }
  __host__ __device__ int64_t part(int64_t node) const { return common() + node * num_features; }
  // A slot a message lands in: whole 16-byte units.
  __host__ __device__ int64_t slot_words() const { return (length() + 1) / 2 * 2; }
};

// The kernels of one channel run one at a time, in issue order, and each
// takes its sequence number (seq), the same on every rank. The in channel is
// the input dist, whose sequence number (in_seq) counts calls; the out
// channel the output dist's kernels, whose number (out_seq) counts them.

// The ctrl section (uint64 words): every control word of both channels and the device state they
// keep across calls. Its size depends only on local_size, num_nodes, num_features and the most
// chunks of a stream. Every control word is tagged with its channel's sequence number, is a running
// total, or is back at its start value when the kernel that uses it ends. The data words (the count
// vectors, requester_offsets, the needs, the count messages) are written whole in each call, before
// a tagged word, the node barrier or a signal publishes them. So none is ever reset, and a call
// that overflowed leaves all of them valid for the next. Words marked (local) are only this rank's:
// a GIN warp's place with each peer, which it sets when it starts. Words marked (num_nodes > 1)
// exist across nodes only. Each accessor returns the address of a word, or of the first of an array
// of them, in the copy at `base`.
class CtrlLayout {
 public:
  __host__ __device__ CtrlLayout(const Sizes &sizes, void *base)
      : base_((uint64_t *)base),
        num_nodes_(sizes.topo.num_nodes),
        num_features_(sizes.num_features),
        counters_(sizes.counters()),
        count_{sizes.topo.local_size, sizes.num_features},
        message_{sizes.topo.num_nodes, sizes.num_features} {
    const int64_t local_size = sizes.topo.local_size, num_nodes = num_nodes_;
    const bool multi = num_nodes > 1;
    int64_t next = 0;
    auto take = [&](int64_t words) {
      const int64_t at = next;
      next += words;
      return at;
    };
    in_started_ = take(1);
    count_vectors_ = take(2 * count_.words());
    in_written_ = take(local_size);
    requester_offsets_ = take(multi ? local_size * num_features_ : 0);
    in_done_ = take(multi ? 1 + kNumNeeds : 0);
    count_msg_out_ = take(multi ? message_.out_words() : 0);
    count_msgs_ = take(multi ? 2 * num_nodes * message_.slot_words() : 0);
    out_started_ = take(1);
    pushed_ = take(local_size);
    pulled_ = take(1);
    arrive_ = take(1);
    tasks_ = take(kNumQueues);
    landed_rows_ = take(multi ? num_nodes : 0);
    fwd_tot_ = take(multi ? num_nodes : 0);
    bwd_tot_ = take(multi ? num_nodes : 0);
    grant_sent_ = take(multi ? num_nodes : 0);
    bwd_ready_sent_ = take(multi ? num_nodes : 0);
    put_at_ = take(multi ? num_nodes : 0);
    landed_ = take(multi ? sizes.max_chunks : 0);
    pulled_rows_ = take(multi ? sizes.max_chunks : 0);
    posted_ = take(multi ? counters_ : 0);
    slot_free_ = take(multi ? 1 : 0);
    closing_ = take(1);
    closed_ = take(1);
    words_ = next;
  }
  // The section: its first word and its bytes.
  __host__ __device__ uint64_t *base() const { return base_; }
  __host__ __device__ uint64_t bytes() const { return 8 * (uint64_t)words_; }

  // ── the in channel ──
  // in_seq of this rank's current node_count_exchange (its node barrier)
  __host__ __device__ uint64_t *in_started() const { return word(in_started_); }
  // This rank's count vector (CountVector) in call in_seq, by its parity.
  __host__ __device__ int64_t *count_vector(int64_t in_seq) const {
    return (int64_t *)word(count_vectors_ + (in_seq & 1) * count_.words());
  }
  // [local_size] in_seq whose keys each requester pushed into node_keys
  __host__ __device__ uint64_t *in_written() const { return word(in_written_); }
  // (num_nodes > 1) [num_features] where the keys of each feature of a
  // same-node requester start in this rank's KJT
  __host__ __device__ int64_t *requester_offsets(int64_t requester) const {
    return (int64_t *)word(requester_offsets_ + requester * num_features_);
  }
  // (num_nodes > 1) in_seq of this rank's node_status_exchange
  __host__ __device__ uint64_t *in_done() const { return word(in_done_); }
  // (num_nodes > 1) [kNumNeeds] its rail's needs, which in_done publishes
  __host__ __device__ int64_t *needs() const { return (int64_t *)word(in_done_ + 1); }
  // (num_nodes > 1) the count message this rank sends its rail peers
  // (CountMessage::out_words)
  __host__ __device__ int64_t *count_msg_out() const { return (int64_t *)word(count_msg_out_); }
  // (num_nodes > 1) the count message of call in_seq from the rail peer on
  // `node` (this rank's own too), by in_seq parity: a peer's message of call
  // seq + 2 needs this rank's of seq + 1, which it sends once call seq ended
  // here.
  __host__ __device__ int64_t *count_msg(int64_t in_seq, int64_t node) const {
    return (int64_t *)word(count_msgs_ +
                           ((in_seq & 1) * num_nodes_ + node) * message_.slot_words());
  }

  // ── the out channel ──
  // out_seq of this rank's current output kernel (its node barrier)
  __host__ __device__ uint64_t *out_started() const { return word(out_started_); }
  // [local_size] out_seq whose gradient rows each requester pushed into lookup_rows
  __host__ __device__ uint64_t *pushed() const { return word(pushed_); }
  // out_seq after this rank's last pull of a forward (exit barrier)
  __host__ __device__ uint64_t *pulled() const { return word(pulled_); }
  // (local) CTAs of this rank's out kernels that reached their end, a running total
  __host__ __device__ uint64_t *arrive() const { return word(arrive_); }
  // (local) task queue `queue` of the current out kernel, back at 0 when it ends
  __host__ __device__ uint64_t *task(int queue) const { return word(tasks_ + queue); }
  // (num_nodes > 1) (out_seq << 32) | rows of an owner node's that landed in
  // this proxy's proxy_rows during that forward (the relay)
  __host__ __device__ uint64_t *landed_rows(int64_t owner_node) const {
    return word(landed_rows_ + owner_node);
  }
  // (num_nodes > 1, local) rows an owner node put into proxy_rows in earlier
  // forwards: what its signal held when this forward began
  __host__ __device__ uint64_t *fwd_tot(int64_t owner_node) const {
    return word(fwd_tot_ + owner_node);
  }
  // (num_nodes > 1, local) the same of a proxy node's gradient rows
  __host__ __device__ uint64_t *bwd_tot(int64_t proxy_node) const {
    return word(bwd_tot_ + proxy_node);
  }
  // (num_nodes > 1, local) the grant this proxy sent a rail owner
  __host__ __device__ uint64_t *grant_sent(int64_t owner_node) const {
    return word(grant_sent_ + owner_node);
  }
  // (num_nodes > 1, local) the bwd_ready this owner sent a rail proxy
  __host__ __device__ uint64_t *bwd_ready_sent(int64_t proxy_node) const {
    return word(bwd_ready_sent_ + proxy_node);
  }
  // (num_nodes > 1, local) an issuer's place in its put list for the rail
  // peer on node peer_node: its next segment
  __host__ __device__ int64_t *put_at(int64_t peer_node) const {
    return (int64_t *)word(put_at_ + peer_node);
  }
  // (num_nodes > 1) [max_chunks] the out_seq of the forward in which every
  // row of chunk c of this proxy's stream had landed (the relay)
  __host__ __device__ uint64_t *landed(int64_t chunk) const { return word(landed_ + chunk); }
  // (num_nodes > 1) [max_chunks] rows the node's requesters pulled of chunk c
  // of this proxy's stream in a forward whose stream wraps the ring; the
  // relay zeroes those of its chunks before any of them lands
  __host__ __device__ uint64_t *pulled_rows(int64_t chunk) const {
    return word(pulled_rows_ + chunk);
  }
  // (num_nodes > 1, local) [counters] backward puts this proxy posted on each
  // put counter, in calls whose stream wraps: what the counter counts
  __host__ __device__ uint64_t *posted(int64_t counter) const { return word(posted_ + counter); }
  // (num_nodes > 1) tag_of(out_seq, position): this proxy's node may push
  // the gradient rows of that backward's stream below that position into
  // its proxy_rows (the rows below num_proxy_rows need none)
  __host__ __device__ uint64_t *slot_free() const { return word(slot_free_); }

  // ── teardown ──
  // this rank reads no window any more
  __host__ __device__ uint64_t *closing() const { return word(closing_); }
  // every node peer, and every rail peer (closing signals), said so (written locally)
  __host__ __device__ uint64_t *closed() const { return word(closed_); }

 private:
  __host__ __device__ uint64_t *word(int64_t at) const { return base_ + at; }

  uint64_t *base_;
  int64_t num_nodes_, num_features_, counters_;
  CountVector count_;
  CountMessage message_;
  int64_t in_started_, count_vectors_, in_written_, requester_offsets_, in_done_, count_msg_out_,
      count_msgs_;
  int64_t out_started_, pushed_, pulled_, arrive_, tasks_, landed_rows_, fwd_tot_, bwd_tot_,
      grant_sent_, bwd_ready_sent_, put_at_, landed_, pulled_rows_, posted_, slot_free_;
  int64_t closing_, closed_, words_;
};

// The keys section of the window (KeysLayout): the in channel's data, and the
// formats of the two buffers in it that the node's requesters read, the proxy
// header and remote_map.

struct GatheredHeader;  // below

// The proxy header (int32 words), what a proxy publishes to its node's
// requesters: its stream's chunks at word 0 and its end at word 1 (uint32,
// unwrapped: past num_proxy_rows when the stream wraps the ring); from word
// 4 each chunk's end ([max_chunks], uint32); then its push tables, ReqEntry
// [local_size][max_chunks + 1], positions relative to each requester's push
// order there and groups relative to its first group. A requester gathers
// the words it reads (planner.h) by their word numbers.
struct ProxyHeader {
  static constexpr int64_t kChunksWord = 0, kEndWord = 1, kEndsWord = 4;
  int32_t *base;
  int64_t local_size, max_chunks;
  // The header of a proxy of these sizes at `base` (nullptr: its word numbers
  // alone).
  __host__ __device__ static ProxyHeader of(const Sizes &sizes, int32_t *base = nullptr) {
    return ProxyHeader{base, sizes.topo.local_size, sizes.max_chunks};
  }
  __host__ __device__ int64_t push_words() const { return 4 * (max_chunks + 1); }
  __host__ __device__ int64_t push_table_word(int64_t requester) const {
    return kEndsWord + max_chunks + requester * push_words();
  }
  __host__ __device__ int64_t words() const { return push_table_word(local_size); }
  __host__ __device__ int32_t *chunks() const { return base + kChunksWord; }
  __host__ __device__ uint32_t *end() const { return (uint32_t *)(base + kEndWord); }
  __host__ __device__ uint32_t *ends() const { return (uint32_t *)(base + kEndsWord); }
  __host__ __device__ int32_t *push_table(int64_t requester) const {
    return base + push_table_word(requester);
  }
  // What a requester gathers of a header, in this order: the chunks, the
  // end, each chunk's end, then its own push table.
  static constexpr int64_t kGatheredChunks = 0, kGatheredEnd = 1, kGatheredEnds = 2;
  __host__ __device__ int64_t gathered_push_table() const { return kGatheredEnds + max_chunks; }
  __host__ __device__ int64_t gathered_words() const {
    return gathered_push_table() + push_words();
  }
  // Proxy `proxy`'s words in what a requester gathered of every proxy's
  // header (`headers`, gathered_words each).
  __host__ __device__ GatheredHeader gathered(const int32_t *headers, int64_t proxy) const;
};
// What a requester gathered of a proxy's header: its stream's chunks, each
// chunk's end and the stream's end (unwrapped positions), and this
// requester's push table there (positions relative to its push order at the
// proxy, groups relative to its first group).
struct GatheredHeader {
  const int32_t *words;
  int64_t max_chunks;
  __host__ __device__ int64_t chunks() const { return words[ProxyHeader::kGatheredChunks]; }
  // chunk c's end for c < max_chunks, the stream's end for c = max_chunks
  __host__ __device__ int64_t end(int64_t chunk) const {
    const int64_t word =
        chunk < max_chunks ? ProxyHeader::kGatheredEnds + chunk : ProxyHeader::kGatheredEnd;
    return (int64_t)(uint32_t)words[word];
  }
  __host__ __device__ const ReqEntry *push_table() const {
    return (const ReqEntry *)(words + ProxyHeader::kGatheredEnds + max_chunks);
  }
};
__host__ __device__ inline GatheredHeader ProxyHeader::gathered(const int32_t *headers,
                                                                int64_t proxy) const {
  return GatheredHeader{headers + proxy * gathered_words(), max_chunks};
}

// remote_map (int32), what a proxy publishes of each occurrence of its
// remote part (occ) to the occurrence's requester, in three arrays of
// remote_keys: its key's row in proxy_rows (bit 31: a heavy key), its slot
// in proxy_rows (the key's row, a duplicate's or the requester's partial
// slot) and its position in its requester's push order. A requester gathers
// an occurrence's words from entry occ of each array (GatheredMap).
// An occurrence's key row in remote_map: the row, bit 31 set for a heavy key.
struct MapKeyRow {
  uint32_t bits;
  __host__ __device__ static MapKeyRow of(int32_t row, bool heavy) {
    return {(uint32_t)row | (uint32_t)heavy << 31};
  }
  __host__ __device__ int32_t row() const { return (int32_t)(bits & 0x7FFFFFFFu); }
  __host__ __device__ bool heavy() const { return (bits >> 31) != 0; }
};
static_assert(sizeof(MapKeyRow) == 4, "remote_map's arrays are int32");
struct RemoteMap {
  static constexpr int kArrays = 3;
  int32_t *base;
  int64_t remote_keys;  // the entries of each array
  __host__ __device__ int32_t *array(int which) const { return base + which * remote_keys; }
  __host__ __device__ MapKeyRow *key_row() const { return (MapKeyRow *)array(0); }
  __host__ __device__ int32_t *slot() const { return array(1); }
  __host__ __device__ int32_t *position() const { return array(2); }
};
// The remote_map words a requester gathered for its rows (peer_gather's
// [RemoteMap::kArrays][rows]): each row's key row, slot and push position.
struct GatheredMap {
  const int32_t *words;
  int64_t rows;
  __host__ __device__ MapKeyRow key_row(int64_t row) const {
    return ((const MapKeyRow *)words)[row];
  }
  __host__ __device__ int32_t slot(int64_t row) const { return words[rows + row]; }
  __host__ __device__ int32_t position(int64_t row) const { return words[2 * rows + row]; }
};

// The buffers of the keys section whose words a requester gathers from its
// node peers (peer_gather): of a proxy header, one word per item (its word
// number); of remote_map, the RemoteMap::kArrays words of an occurrence (its
// entry occ of each array).
enum GatherFrom : int { kGatherProxyHeader = 0, kGatherRemoteMap };
// The words peer_gather reads per item.
__host__ __device__ constexpr int gather_words(GatherFrom from) {
  return from == kGatherProxyHeader ? 1 : RemoteMap::kArrays;
}

// The keys section, its buffers in this order (those marked num_nodes > 1
// exist across nodes only):
//   node_keys     the keys this rank's node pushed to it (node_key_push):
//                 its own part at [0, lookup_keys), the keys of its node for
//                 it as owner, in its KJT's order ([feature][requester]
//                 [requester's order]); then (num_nodes > 1) its remote part
//                 at [lookup_keys, lookup_keys + remote_keys), those it
//                 deduplicates as proxy, requester-major, then owner node,
//                 then [feature][position]. A remote occurrence is named by
//                 its index occ in the remote part.
//   node_weights  their weights, when the requesters' keys carry them
//   bucket_feature_of_occ
//                 (num_nodes > 1) per remote occurrence, int32: its bucket
//                 feature, owner node * num_features + feature
//   remote_map    (num_nodes > 1) RemoteMap
//   proxy_header  (num_nodes > 1) ProxyHeader
//   key_stage     (num_nodes > 1) the node-unique key records of each owner
//                 node, each slice after the node's chunks (ChunkPiece): what
//                 one key put sends; then 16 bytes that take what no key
//                 writes
//   rail_keys     (num_nodes > 1) the key records the rail proxies put, one
//                 slice per source node in node order, each the chunks of
//                 this rank's node there, then its records in uniq order
//   source_offsets
//                 (num_nodes > 1) int64 [num_nodes][num_features]: as owner,
//                 where each rail proxy's node's keys of each feature start
//                 in its KJT, what it puts to that proxy
//   owner_offsets (num_nodes > 1) int64 [num_nodes][num_features]: as proxy,
//                 where its node's keys of each feature start in each rail
//                 owner's KJT, as the owners put them
// Each accessor returns its buffer in the copy at `base`.
class KeysLayout {
 public:
  __host__ __device__ KeysLayout(const Sizes &sizes, char *base)
      : base_(base),
        num_nodes_(sizes.topo.num_nodes),
        local_size_(sizes.topo.local_size),
        num_features_(sizes.num_features),
        lookup_keys_(sizes.lookup_keys),
        remote_keys_(sizes.remote_keys),
        max_chunks_(sizes.max_chunks) {
    const bool multi = num_nodes_ > 1;
    const uint64_t entries = (uint64_t)(lookup_keys_ + remote_keys_);
    uint64_t next = 0;
    auto take = [&](uint64_t bytes) {
      const uint64_t at = next;
      next += buffer_end(bytes);
      return at;
    };
    node_keys_ = take(8 * entries);
    node_weights_ = take(sizes.weights ? 4 * entries : 0);
    bucket_feature_of_occ_ = take(multi ? 4 * (uint64_t)remote_keys_ : 0);
    remote_map_ = take(multi ? 4 * RemoteMap::kArrays * (uint64_t)remote_keys_ : 0);
    proxy_header_ = take(multi ? 4 * (uint64_t)ProxyHeader::of(sizes).words() : 0);
    key_stage_bytes_ =
        multi ? max_chunks_ * sizeof(ChunkPiece) + remote_keys_ * sizes.record_bytes + 16 : 0;
    key_stage_ = take(key_stage_bytes_);
    rail_keys_bytes_ =
        multi ? lookup_keys_ * sizes.record_bytes + sizes.owner_chunks() * sizeof(ChunkPiece) : 0;
    rail_keys_ = take(rail_keys_bytes_);
    source_offsets_ = take(multi ? 8 * (uint64_t)(num_nodes_ * num_features_) : 0);
    owner_offsets_ = take(multi ? 8 * (uint64_t)(num_nodes_ * num_features_) : 0);
    bytes_ = next;
  }
  __host__ __device__ uint64_t bytes() const { return bytes_; }
  // Each buffer's name and the bytes it takes (HierOutputDist::memory).
  std::vector<std::pair<const char *, uint64_t>> buffers() const {
    return {{"node_keys", node_weights_ - node_keys_},
            {"node_weights", bucket_feature_of_occ_ - node_weights_},
            {"bucket_feature_of_occ", remote_map_ - bucket_feature_of_occ_},
            {"remote_map", proxy_header_ - remote_map_},
            {"proxy_header", key_stage_ - proxy_header_},
            {"key_stage", rail_keys_ - key_stage_},
            {"rail_keys", source_offsets_ - rail_keys_},
            {"source_offsets", owner_offsets_ - source_offsets_},
            {"owner_offsets", bytes_ - owner_offsets_}};
  }

  // node_keys and node_weights, [lookup_keys + remote_keys]
  __host__ __device__ int64_t *node_keys() const { return (int64_t *)(base_ + node_keys_); }
  __host__ __device__ float *node_weights() const { return (float *)(base_ + node_weights_); }
  // where the remote part starts in them
  __host__ __device__ int64_t remote_start() const { return lookup_keys_; }
  __host__ __device__ int32_t *bucket_feature_of_occ() const {
    return (int32_t *)(base_ + bucket_feature_of_occ_);
  }
  __host__ __device__ RemoteMap remote_map() const {
    return RemoteMap{(int32_t *)(base_ + remote_map_), remote_keys_};
  }
  __host__ __device__ ProxyHeader proxy_header() const {
    return ProxyHeader{(int32_t *)(base_ + proxy_header_), local_size_, max_chunks_};
  }
  // key_stage and rail_keys, and their bytes
  __host__ __device__ char *key_stage() const { return base_ + key_stage_; }
  __host__ __device__ uint64_t key_stage_bytes() const { return key_stage_bytes_; }
  __host__ __device__ char *rail_keys() const { return base_ + rail_keys_; }
  __host__ __device__ uint64_t rail_keys_bytes() const { return rail_keys_bytes_; }
  // [num_features] of a node
  __host__ __device__ int64_t *source_offsets(int64_t node) const {
    return (int64_t *)(base_ + source_offsets_) + node * num_features_;
  }
  __host__ __device__ int64_t *owner_offsets(int64_t node) const {
    return (int64_t *)(base_ + owner_offsets_) + node * num_features_;
  }

 private:
  char *base_;
  int64_t num_nodes_, local_size_, num_features_, lookup_keys_, remote_keys_, max_chunks_;
  uint64_t node_keys_, node_weights_, bucket_feature_of_occ_, remote_map_, proxy_header_,
      key_stage_, rail_keys_, source_offsets_, owner_offsets_, bytes_;
  uint64_t key_stage_bytes_, rail_keys_bytes_;
};

// The rows section of the window (RowsLayout): the out channel's data.

// The rows section, its buffers in this order (those marked num_nodes > 1
// exist across nodes only):
//   lookup_rows  the owner's lookup output, row i for its KJT's key i; the
//                backward writes the gradient of that output over it
//   proxy_rows   (num_nodes > 1) the proxy's ring, its stream of chunks
//                (common.h): the rows of the node-unique keys the rail owners
//                put (forward), the gradient rows of its node's requesters
//                and their sums (backward)
//   push_flags   (num_nodes > 1) the requesters' push-task flags of a call
//                (flag_words), each tagged with its out_seq
// Each accessor returns its buffer in the copy at `base`.
class RowsLayout {
 public:
  __host__ __device__ RowsLayout(const Sizes &sizes, char *base)
      : base_(base), row_bytes_(sizes.row_bytes) {
    const bool multi = sizes.multi_node();
    proxy_rows_ = buffer_end(sizes.lookup_keys * row_bytes_);
    push_flags_ = proxy_rows_ + buffer_end(multi ? sizes.num_proxy_rows * row_bytes_ : 0);
    bytes_ = push_flags_ + buffer_end(multi ? 8 * (uint64_t)sizes.flag_words() : 0);
  }
  __host__ __device__ uint64_t bytes() const { return bytes_; }
  // Each buffer's name and the bytes it takes (HierOutputDist::memory).
  std::vector<std::pair<const char *, uint64_t>> buffers() const {
    return {{"lookup_rows", proxy_rows_},
            {"proxy_rows", push_flags_ - proxy_rows_},
            {"push_flags", bytes_ - push_flags_}};
  }

  __host__ __device__ char *lookup_rows() const { return base_; }
  __host__ __device__ char *lookup_row(int64_t row) const {
    return lookup_rows() + row * row_bytes_;
  }
  __host__ __device__ char *proxy_rows() const { return base_ + proxy_rows_; }
  __host__ __device__ char *proxy_row(int64_t slot) const {
    return proxy_rows() + slot * row_bytes_;
  }
  __host__ __device__ uint64_t *push_flags() const { return (uint64_t *)(base_ + push_flags_); }
  __host__ __device__ uint64_t push_flags_bytes() const { return bytes_ - push_flags_; }

 private:
  char *base_;
  int64_t row_bytes_;
  uint64_t proxy_rows_, push_flags_, bytes_;  // lookup_rows starts at base
};

// The op's window: this rank's copy of its symmetric memory, one NCCL window
// laid out the same on every rank (WindowLayout).

// The window: the rows section, the ctrl section, then the keys section, each
// from a multiple of kBufferAlign. The out kernels' sections come first, so
// the places of their buffers take the least to compute. The host lays the
// window out over its copy (the memory it allocated) to size it and to view
// its buffers as tensors; a kernel over the same copy as the node's GPUs map
// it (NodeWindow). A place in a node peer's copy is the same place in this
// rank's: kernels name it by a pointer into their own copy, and GIN puts by
// its offset in the window.
struct WindowLayout {
  RowsLayout rows;
  CtrlLayout ctrl;
  KeysLayout keys;
  __host__ __device__ WindowLayout(const Sizes &sizes, char *base)
      : rows(sizes, base),
        ctrl(sizes, base + rows.bytes()),
        keys(sizes, base + rows.bytes() + buffer_end(ctrl.bytes())) {}
  // The window's bytes.
  __host__ __device__ static uint64_t bytes(const Sizes &sizes) {
    return RowsLayout(sizes, nullptr).bytes() + buffer_end(CtrlLayout(sizes, nullptr).bytes()) +
           KeysLayout(sizes, nullptr).bytes();
  }
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
