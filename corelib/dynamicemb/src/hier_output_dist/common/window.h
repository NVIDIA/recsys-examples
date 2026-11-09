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
// the format of the count message, the in channel's data (KeysLayout) with
// the proxy header and remote_map, the out channel's data (RowsLayout), and
// the window (WindowLayout).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "hier_output_dist/common/common.h"

namespace dyn_emb {
namespace hier_output_dist {

// The ctrl section of the window (CtrlLayout): the control words of both
// channels and the device state they keep across calls, and the format of a
// message that lives there, the count message.

// A count message (int64 words, planner.h): what a rank sends a rail peer, from which every rank of
// the rail lays out the keys the rail puts and the rows they come from and go to (rail.cuh). First
// a header, the same to every peer (the rail allgathers it): its node's largest remote part, as
// proxy the most occurrences of one of its node-unique keys; the keys of each lookup group of its
// own part (as owner: the keys of its own node it looks up); as proxy its chunks per owner node and
// its node-unique keys of each (owner node, lookup group). Then the part of the receiver alone: as
// proxy its node-unique keys of each feature of the receiver's node. 2 + num_groups + num_nodes *
// (num_groups + 1) + num_features words.
struct CountMessage {
  static constexpr int kNeedRemoteWord = 0, kNeedKeyRowsWord = 1;
  int64_t num_nodes, num_features, num_groups;
  __host__ __device__ int64_t own_total_word(int64_t group) const { return 2 + group; }
  __host__ __device__ int64_t chunk_length_word(int64_t owner_node) const {
    return 2 + num_groups + owner_node;
  }
  __host__ __device__ int64_t key_total_word(int64_t owner_node, int64_t group) const {
    return 2 + num_groups + num_nodes + owner_node * num_groups + group;
  }
  __host__ __device__ int64_t header_length() const {
    return 2 + num_groups + num_nodes * (num_groups + 1);
  }
  __host__ __device__ int64_t key_length_word(int64_t feature) const {
    return header_length() + feature;
  }
  __host__ __device__ int64_t length() const { return header_length() + num_features; }
  // A slot a message lands in: whole 16-byte units.
  __host__ __device__ int64_t slot_words() const { return (length() + 1) / 2 * 2; }
};

// The kernels of one channel run one at a time, in issue order, and each
// takes its sequence number (seq), the same on every rank. The in channel is
// the input dist, whose sequence number (in_seq) counts calls; the out
// channel the output dist's kernels, whose number (out_seq) counts them.

// The ctrl section (uint64 words): every control word of both channels and the device state they
// keep across calls. Its size depends only on local_size, num_nodes, num_features, num_groups and
// the most chunks of a stream. Every control word is tagged with its channel's sequence number, is
// a running total, or is back at its start value when the kernel that uses it ends. The data words
// (group_shift, the needs, the count messages) are written whole in each call, before a flag
// (in_status) or a signal publishes them. So none is ever reset, and a call that overflowed leaves
// all of them valid for the next. Words marked (local) are only this rank's:
// a GIN warp's place with each peer, which it sets when it starts. Words marked (num_nodes > 1)
// exist across nodes only. Each accessor returns the address of a word, or of the first of an array
// of them, in the copy at `base`.
class CtrlLayout {
 public:
  __host__ __device__ CtrlLayout(const Sizes &sizes, void *base)
      : base_((uint64_t *)base),
        num_nodes_(sizes.topo.num_nodes),
        count_format_{sizes.topo.num_nodes, sizes.num_features, sizes.num_groups} {
    const int64_t local_size = sizes.topo.local_size, num_nodes = num_nodes_;
    const bool multi = num_nodes > 1;
    int64_t next = 0;
    auto take = [&](int64_t words) {
      const int64_t at = next;
      next += words;
      return at;
    };
    in_started_ = take(1);
    in_written_ = take(local_size);
    group_shift_ = take(multi ? sizes.num_groups : 0);
    in_status_ = take(multi ? 1 + kNumNeeds : 0);
    count_msg_out_ = take(multi ? num_nodes * count_format_.slot_words() : 0);
    count_msgs_ = take(multi ? 2 * num_nodes * count_format_.slot_words() : 0);
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
    pushed_rows_ = take(multi ? sizes.max_chunks : 0);
    posted_ = take(multi ? sizes.counters() : 0);
    bwd_free_end_ = take(multi ? 1 : 0);
    closing_ = take(1);
    closed_ = take(1);
    words_ = next;
  }
  // The section: its first word and its bytes.
  __host__ __device__ uint64_t *base() const { return base_; }
  __host__ __device__ uint64_t bytes() const { return 8 * (uint64_t)words_; }

  // ── the in channel ──
  // in_seq of this rank's current node_exchange (its node barrier). It holds the call once this
  // rank's offsets of the call are in node_offsets at every node peer.
  __host__ __device__ uint64_t *in_started() const { return word(in_started_); }
  // [local_size] in_seq whose keys each requester pushed into node_keys here. The tell also says
  // that the requester's node_exchange read its node_offsets, so this rank may write its row
  // there again in the next call. node_done tells in every call, also with no rows and under skip.
  __host__ __device__ uint64_t *in_written() const { return word(in_written_); }
  // (num_nodes > 1) [num_groups] per lookup group of this rank's KJT, the rows of its
  // lookup_rows before the group's own part, less the own part's keys before it: a key of its own
  // part lies at its place there plus its group's shift (rail.cuh)
  __host__ __device__ int64_t *group_shift() const { return (int64_t *)word(group_shift_); }
  // (num_nodes > 1) in_seq of this rank's node_status_exchange
  __host__ __device__ uint64_t *in_status() const { return word(in_status_); }
  // (num_nodes > 1) [kNumNeeds] its rail's needs, which in_status publishes
  __host__ __device__ int64_t *needs() const { return (int64_t *)word(in_status_ + 1); }
  // (num_nodes > 1) the count message this rank sends the rail peer on
  // `node` ([num_nodes][CountMessage::slot_words]: the header in every one)
  __host__ __device__ int64_t *count_msg_out(int64_t node) const {
    return (int64_t *)word(count_msg_out_ + node * count_format_.slot_words());
  }
  // (num_nodes > 1) the count message of call in_seq from the rail peer on
  // `node` (this rank's own too), by in_seq parity: a peer's message of call
  // seq + 2 needs this rank's of seq + 1, which it sends once call seq ended
  // here. A parity's slots are consecutive, node after node.
  __host__ __device__ int64_t *count_msg(int64_t in_seq, int64_t node) const {
    return (int64_t *)word(count_msgs_ +
                           ((in_seq & 1) * num_nodes_ + node) * count_format_.slot_words());
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
  // (num_nodes > 1, local) an issuer's place among the chunks it puts for
  // the rail peer on node peer_node: its next chunk
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
  // (num_nodes > 1) [max_chunks] gradient rows the node's requesters pushed
  // into chunk c of this proxy's stream in the current backward, which zeroes
  // those of its chunks before it publishes its out_seq
  __host__ __device__ uint64_t *pushed_rows(int64_t chunk) const {
    return word(pushed_rows_ + chunk);
  }
  // (num_nodes > 1, local) [counters] backward puts this proxy posted on each
  // put counter, in calls whose stream wraps: what the counter counts
  __host__ __device__ uint64_t *posted(int64_t counter) const { return word(posted_ + counter); }
  // (num_nodes > 1) tagged(out_seq, position): this proxy's node may push
  // the gradient rows of that backward's stream below that position into
  // its proxy_rows (the rows below num_proxy_rows need none)
  __host__ __device__ uint64_t *bwd_free_end() const { return word(bwd_free_end_); }

  // ── teardown ──
  // this rank reads no window any more
  __host__ __device__ uint64_t *closing() const { return word(closing_); }
  // every node peer, and every rail peer (closing signals), said so (written locally)
  __host__ __device__ uint64_t *closed() const { return word(closed_); }

 private:
  __host__ __device__ uint64_t *word(int64_t at) const { return base_ + at; }

  uint64_t *base_;
  int64_t num_nodes_;
  CountMessage count_format_;
  int64_t in_started_, in_written_, group_shift_, in_status_, count_msg_out_, count_msgs_;
  int64_t out_started_, pushed_, pulled_, arrive_, tasks_, landed_rows_, fwd_tot_, bwd_tot_,
      grant_sent_, bwd_ready_sent_, put_at_, landed_, pulled_rows_, pushed_rows_, posted_,
      bwd_free_end_;
  int64_t closing_, closed_, words_;
};

// The keys section of the window (KeysLayout): the in channel's data, and the
// formats of the two buffers in it that the node's requesters read, the proxy
// header and remote_map.

struct GatheredHeader;  // below

// The proxy header (int32 words), what a proxy publishes to its node's
// requesters: its stream's chunks at word 0 and its end at word 1 (uint32,
// unwrapped: past num_proxy_rows when the stream wraps the ring); from word
// 2 each chunk's end ([max_chunks], uint32); then its push tables
// [local_size][max_chunks + 1]: where each requester's rows of each chunk
// start in its push order there (chunk max_chunks: their end). A requester
// gathers the words it reads (planner.h) by their word numbers.
struct ProxyHeader {
  static constexpr int64_t kNumChunksWord = 0, kEndWord = 1, kEndsWord = 2;
  int32_t *base;
  int64_t local_size, max_chunks;
  // The header of a proxy of these sizes at `base` (nullptr: its word numbers
  // alone).
  __host__ __device__ static ProxyHeader of(const Sizes &sizes, int32_t *base = nullptr) {
    return ProxyHeader{base, sizes.topo.local_size, sizes.max_chunks};
  }
  __host__ __device__ int64_t push_words() const { return max_chunks + 1; }
  __host__ __device__ int64_t push_table_word(int64_t requester) const {
    return kEndsWord + max_chunks + requester * push_words();
  }
  __host__ __device__ int64_t words() const { return push_table_word(local_size); }
  __host__ __device__ int32_t *num_chunks() const { return base + kNumChunksWord; }
  __host__ __device__ uint32_t *end() const { return (uint32_t *)(base + kEndWord); }
  __host__ __device__ uint32_t *ends() const { return (uint32_t *)(base + kEndsWord); }
  __host__ __device__ int32_t *push_table(int64_t requester) const {
    return base + push_table_word(requester);
  }
  // What a requester gathers of a header, in this order: the chunks, the
  // end, each chunk's end, then its own push table.
  static constexpr int64_t kGatheredNumChunks = 0, kGatheredEnd = 1, kGatheredEnds = 2;
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
// proxy).
struct GatheredHeader {
  const int32_t *words;
  int64_t max_chunks;
  __host__ __device__ int64_t num_chunks() const { return words[ProxyHeader::kGatheredNumChunks]; }
  // chunk c's end for c < max_chunks, the stream's end for c = max_chunks
  __host__ __device__ int64_t end(int64_t chunk) const {
    const int64_t word =
        chunk < max_chunks ? ProxyHeader::kGatheredEnds + chunk : ProxyHeader::kGatheredEnd;
    return (int64_t)(uint32_t)words[word];
  }
  __host__ __device__ const int32_t *push_table() const {
    return words + ProxyHeader::kGatheredEnds + max_chunks;
  }
};
__host__ __device__ inline GatheredHeader ProxyHeader::gathered(const int32_t *headers,
                                                                int64_t proxy) const {
  return GatheredHeader{headers + proxy * gathered_words(), max_chunks};
}

// remote_map (int32), what a proxy publishes of each occurrence of its
// remote part (occ) to the occurrence's requester, in three arrays of
// remote_keys: its key's row in proxy_rows, its slot in proxy_rows (the
// key's row or a duplicate's) and its position in its requester's push
// order. A requester gathers an occurrence's words from entry occ of each
// array (GatheredMap).
struct RemoteMap {
  static constexpr int kArrays = 3;
  int32_t *base;
  int64_t remote_keys;  // the entries of each array
  __host__ __device__ int32_t *array(int which) const { return base + which * remote_keys; }
  __host__ __device__ int32_t *key_row() const { return array(0); }
  __host__ __device__ int32_t *slot() const { return array(1); }
  __host__ __device__ int32_t *position() const { return array(2); }
};
// The remote_map words a requester gathered for its rows (peer_gather's
// [RemoteMap::kArrays][rows]): each row's key row, slot and push position.
struct GatheredMap {
  const int32_t *words;
  int64_t rows;
  __host__ __device__ int32_t key_row(int64_t row) const { return words[row]; }
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
//   node_offsets  int64 [local_size][world_size * num_features + 1]: row r is the
//                 offset_per_bucket_feature of the node's requester r in the call, which it
//                 pushed here (node_exchange)
//   node_keys     the keys this rank's node pushed to it (node_key_push):
//                 its own part at [0, lookup_keys), the keys of its node for
//                 it as owner, in its KJT's order ([feature][requester]
//                 [requester's order]); then (num_nodes > 1) its remote part
//                 at [lookup_keys, lookup_keys + remote_keys), those it
//                 deduplicates as proxy, requester-major, then owner node,
//                 then [feature][position]. A remote occurrence is named by
//                 its index occ in the remote part.
//   node_key_weights  their weights, when the requesters' keys carry them
//   node_bucket_feature_of_row
//                 (num_nodes > 1) per remote occurrence, int32: its bucket
//                 feature, owner node * num_features + feature
//   remote_map    (num_nodes > 1) RemoteMap
//   proxy_header  (num_nodes > 1) ProxyHeader
//   key_stage     (num_nodes > 1) the node-unique key records of each owner
//                 node, each slice after the node's chunks (OwnerChunk): what
//                 one key put sends
//   rail_keys     (num_nodes > 1) the key records the rail proxies put, one
//                 slice per source node in node order, each the chunks of
//                 this rank's node there, then its records in uniq order
// Each accessor returns its buffer in the copy at `base`.
class KeysLayout {
 public:
  __host__ __device__ KeysLayout(const Sizes &sizes, char *base)
      : base_(base),
        local_size_(sizes.topo.local_size),
        offset_words_((int64_t)sizes.topo.world_size * sizes.num_features + 1),
        lookup_keys_(sizes.lookup_keys),
        remote_keys_(sizes.remote_keys),
        max_chunks_(sizes.max_chunks) {
    const bool multi = sizes.multi_node();
    const uint64_t entries = (uint64_t)(lookup_keys_ + remote_keys_);
    uint64_t next = 0;
    auto take = [&](uint64_t bytes) {
      const uint64_t at = next;
      next += buffer_end(bytes);
      return at;
    };
    node_offsets_ = take(8 * (uint64_t)(local_size_ * offset_words_));
    node_keys_ = take(8 * entries);
    node_key_weights_ = take(sizes.weighted ? 4 * entries : 0);
    node_bucket_feature_of_row_ = take(multi ? 4 * (uint64_t)remote_keys_ : 0);
    remote_map_ = take(multi ? 4 * RemoteMap::kArrays * (uint64_t)remote_keys_ : 0);
    proxy_header_ = take(multi ? 4 * (uint64_t)ProxyHeader::of(sizes).words() : 0);
    key_stage_bytes_ =
        multi ? max_chunks_ * sizeof(OwnerChunk) + remote_keys_ * sizes.record_bytes : 0;
    key_stage_ = take(key_stage_bytes_);
    rail_keys_bytes_ =
        multi ? lookup_keys_ * sizes.record_bytes + sizes.max_owner_chunks() * sizeof(OwnerChunk)
              : 0;
    rail_keys_ = take(rail_keys_bytes_);
    bytes_ = next;
  }
  __host__ __device__ uint64_t bytes() const { return bytes_; }
  // Each buffer's name and the bytes it takes (HierOutputDist::memory).
  std::vector<std::pair<const char *, uint64_t>> buffers() const {
    return {{"node_offsets", node_keys_ - node_offsets_},
            {"node_keys", node_key_weights_ - node_keys_},
            {"node_key_weights", node_bucket_feature_of_row_ - node_key_weights_},
            {"node_bucket_feature_of_row", remote_map_ - node_bucket_feature_of_row_},
            {"remote_map", proxy_header_ - remote_map_},
            {"proxy_header", key_stage_ - proxy_header_},
            {"key_stage", rail_keys_ - key_stage_},
            {"rail_keys", bytes_ - rail_keys_}};
  }

  // The words of a row of node_offsets, and requester `requester`'s row.
  __host__ __device__ int64_t offset_words() const { return offset_words_; }
  __host__ __device__ int64_t *node_offsets(int64_t requester) const {
    return (int64_t *)(base_ + node_offsets_) + requester * offset_words_;
  }
  // node_keys and node_key_weights, [lookup_keys + remote_keys]
  __host__ __device__ int64_t *node_keys() const { return (int64_t *)(base_ + node_keys_); }
  __host__ __device__ float *node_key_weights() const {
    return (float *)(base_ + node_key_weights_);
  }
  // where the remote part starts in them
  __host__ __device__ int64_t remote_offset() const { return lookup_keys_; }
  __host__ __device__ int32_t *node_bucket_feature_of_row() const {
    return (int32_t *)(base_ + node_bucket_feature_of_row_);
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

 private:
  char *base_;
  int64_t local_size_, offset_words_, lookup_keys_, remote_keys_, max_chunks_;
  uint64_t node_offsets_, node_keys_, node_key_weights_, node_bucket_feature_of_row_, remote_map_,
      proxy_header_, key_stage_, rail_keys_, bytes_;
  uint64_t key_stage_bytes_, rail_keys_bytes_;
};

// The rows section of the window (RowsLayout): the out channel's data.

// The rows section, its buffers in this order (those marked num_nodes > 1
// exist across nodes only):
//   lookup_rows  the owner's lookup output, a row per key of its KJT, each
//                lookup group's in a slice of their own, in the order they
//                arrive (common.h, GroupRuns); the backward writes the
//                gradient of that output over it
//   proxy_rows   (num_nodes > 1) the proxy's ring, its stream of chunks
//                (common.h): the rows of the node-unique keys the rail owners
//                put (forward), the gradient rows of its node's requesters
//                and their sums (backward)
// Each accessor returns its buffer in the copy at `base`.
class RowsLayout {
 public:
  __host__ __device__ RowsLayout(const Sizes &sizes, char *base)
      : base_(base), row_bytes_(sizes.row_bytes) {
    const bool multi = sizes.multi_node();
    proxy_rows_ = buffer_end(sizes.lookup_keys * row_bytes_);
    bytes_ = proxy_rows_ + buffer_end(multi ? sizes.num_proxy_rows * row_bytes_ : 0);
  }
  __host__ __device__ uint64_t bytes() const { return bytes_; }
  // Each buffer's name and the bytes it takes (HierOutputDist::memory).
  std::vector<std::pair<const char *, uint64_t>> buffers() const {
    return {{"lookup_rows", proxy_rows_}, {"proxy_rows", bytes_ - proxy_rows_}};
  }

  __host__ __device__ char *lookup_rows() const { return base_; }
  __host__ __device__ char *lookup_row(int64_t row) const {
    return lookup_rows() + row * row_bytes_;
  }
  __host__ __device__ char *proxy_rows() const { return base_ + proxy_rows_; }
  __host__ __device__ char *proxy_row(int64_t slot) const {
    return proxy_rows() + slot * row_bytes_;
  }

 private:
  char *base_;
  int64_t row_bytes_;
  uint64_t proxy_rows_, bytes_;  // lookup_rows starts at base
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
