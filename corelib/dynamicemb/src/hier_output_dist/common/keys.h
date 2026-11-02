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
// The keys section of the window (KeysLayout): the in channel's data, and the
// formats of the two buffers in it that the node's requesters read, the proxy
// header and remote_map.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/common/sizes.h"

namespace dyn_emb {
namespace hier_output_dist {

// The proxy header (int32 words), what a proxy publishes to its node's
// requesters: its rounds at word 0; from word 4 its round table, RoundPiece
// [rounds_cap][num_nodes]; then its push tables, ReqEntry [local_size]
// [rounds_cap + 1], positions relative to each requester's push order there
// and groups relative to its first group. A requester gathers the words it
// reads (plan/requester.h) by their word numbers.
struct ProxyHeader {
  static constexpr int64_t kRoundsWord = 0, kTableWord = 4;
  int32_t *base;
  int64_t num_nodes, local_size, rounds_cap;
  // The header of a proxy of these sizes at `base` (nullptr: its word numbers
  // alone).
  __host__ __device__ static ProxyHeader of(const Sizes &sizes, int32_t *base = nullptr) {
    return ProxyHeader{base, sizes.topo.num_nodes, sizes.topo.local_size, sizes.rounds_cap};
  }
  __host__ __device__ int64_t table_words() const { return 4 * rounds_cap * num_nodes; }
  __host__ __device__ int64_t push_words() const { return 4 * (rounds_cap + 1); }
  __host__ __device__ int64_t push_table_word(int64_t requester) const {
    return kTableWord + table_words() + requester * push_words();
  }
  __host__ __device__ int64_t words() const { return push_table_word(local_size); }
  __host__ __device__ int32_t *rounds() const { return base + kRoundsWord; }
  __host__ __device__ int32_t *round_table() const { return base + kTableWord; }
  __host__ __device__ int32_t *push_table(int64_t requester) const {
    return base + push_table_word(requester);
  }
  // What a requester gathers of a header, in this order: the rounds, the
  // round table, then its own push table.
  static constexpr int64_t kGatheredRounds = 0, kGatheredTable = 1;
  __host__ __device__ int64_t gathered_push_table() const { return kGatheredTable + table_words(); }
  __host__ __device__ int64_t gathered_words() const {
    return gathered_push_table() + push_words();
  }
};

// remote_map (int32), what a proxy publishes of each occurrence of its
// remote part (occ) to the occurrence's requester, in three arrays of
// remote_keys: its key's row in proxy_rows (bit 31: a heavy key), its slot
// in proxy_rows (the key's row, a duplicate's or the requester's partial
// slot) and its position in its requester's push order. A requester gathers
// an occurrence's words from entry occ of each array.
struct RemoteMap {
  static constexpr int kArrays = 3;
  int32_t *base;
  int64_t remote_keys;  // the entries of each array
  __host__ __device__ int32_t *array(int which) const { return base + which * remote_keys; }
  __host__ __device__ int32_t *key_row() const { return array(0); }
  __host__ __device__ int32_t *slot() const { return array(1); }
  __host__ __device__ int32_t *position() const { return array(2); }
};

// The buffers of the keys section whose words a requester gathers from its
// node peers (peer_gather): of a proxy header, one word per item (its word
// number); of remote_map, the RemoteMap::kArrays words of an occurrence (its
// entry occ of each array).
enum GatherFrom : int { kGatherProxyHeader = 0, kGatherRemoteMap };
// The words peer_gather reads per item.
__host__ __device__ inline int gather_words(GatherFrom from) {
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
//   remote_cells  (num_nodes > 1) per remote occurrence, int32: its owner
//                 node * num_features + feature
//   remote_map    (num_nodes > 1) RemoteMap
//   proxy_header  (num_nodes > 1) ProxyHeader
//   key_stage     (num_nodes > 1) the node-unique key records of each owner
//                 node, each slice after its round pieces: what one key put
//                 sends; then 16 bytes that take what no key writes
//   rail_keys     (num_nodes > 1) the key records the rail proxies put, one
//                 slice per source node in node order, each its round pieces,
//                 then its records in uniq order
//   cell_stage    (num_nodes > 1) int64 [num_nodes][num_features]: as owner,
//                 where the cells (feature, source node) of each rail proxy's
//                 node start in its KJT, what it puts to that proxy
//   rail_cells    (num_nodes > 1) int64 [num_nodes][num_features]: as proxy,
//                 where its node's cells start in each rail owner's KJT, as
//                 the owners put them
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
        rounds_cap_(sizes.rounds_cap) {
    const bool multi = num_nodes_ > 1;
    const uint64_t entries = (uint64_t)(lookup_keys_ + remote_keys_);
    // a slice's round pieces, of every other owner node
    const uint64_t pieces = (uint64_t)(num_nodes_ - 1) * rounds_cap_ * sizeof(RoundPiece);
    uint64_t next = 0;
    auto take = [&](uint64_t bytes) {
      const uint64_t at = next;
      next += buffer_end(bytes);
      return at;
    };
    node_keys_ = take(8 * entries);
    node_weights_ = take(sizes.weights ? 4 * entries : 0);
    remote_cells_ = take(multi ? 4 * (uint64_t)remote_keys_ : 0);
    remote_map_ = take(multi ? 4 * RemoteMap::kArrays * (uint64_t)remote_keys_ : 0);
    proxy_header_ = take(multi ? 4 * (uint64_t)ProxyHeader::of(sizes).words() : 0);
    key_stage_bytes_ = multi ? pieces + remote_keys_ * sizes.record_bytes + 16 : 0;
    key_stage_ = take(key_stage_bytes_);
    rail_keys_bytes_ = multi ? lookup_keys_ * sizes.record_bytes + pieces : 0;
    rail_keys_ = take(rail_keys_bytes_);
    cell_stage_ = take(multi ? 8 * (uint64_t)(num_nodes_ * num_features_) : 0);
    rail_cells_ = take(multi ? 8 * (uint64_t)(num_nodes_ * num_features_) : 0);
    bytes_ = next;
  }
  __host__ __device__ uint64_t bytes() const { return bytes_; }
  // Each buffer's name and the bytes it takes (HierOutputDist::memory).
  std::vector<std::pair<const char *, uint64_t>> buffers() const {
    return {{"node_keys", node_weights_ - node_keys_},
            {"node_weights", remote_cells_ - node_weights_},
            {"remote_cells", remote_map_ - remote_cells_},
            {"remote_map", proxy_header_ - remote_map_},
            {"proxy_header", key_stage_ - proxy_header_},
            {"key_stage", rail_keys_ - key_stage_},
            {"rail_keys", cell_stage_ - rail_keys_},
            {"cell_stage", rail_cells_ - cell_stage_},
            {"rail_cells", bytes_ - rail_cells_}};
  }

  // node_keys and node_weights, [lookup_keys + remote_keys]
  __host__ __device__ int64_t *node_keys() const { return (int64_t *)(base_ + node_keys_); }
  __host__ __device__ float *node_weights() const { return (float *)(base_ + node_weights_); }
  // where the remote part starts in them
  __host__ __device__ int64_t remote_start() const { return lookup_keys_; }
  __host__ __device__ int32_t *remote_cells() const { return (int32_t *)(base_ + remote_cells_); }
  __host__ __device__ RemoteMap remote_map() const {
    return RemoteMap{(int32_t *)(base_ + remote_map_), remote_keys_};
  }
  __host__ __device__ ProxyHeader proxy_header() const {
    return ProxyHeader{(int32_t *)(base_ + proxy_header_), num_nodes_, local_size_, rounds_cap_};
  }
  // key_stage and rail_keys, and their bytes
  __host__ __device__ char *key_stage() const { return base_ + key_stage_; }
  __host__ __device__ uint64_t key_stage_bytes() const { return key_stage_bytes_; }
  __host__ __device__ char *rail_keys() const { return base_ + rail_keys_; }
  __host__ __device__ uint64_t rail_keys_bytes() const { return rail_keys_bytes_; }
  // [num_features] of a node
  __host__ __device__ int64_t *cell_stage(int64_t node) const {
    return (int64_t *)(base_ + cell_stage_) + node * num_features_;
  }
  __host__ __device__ int64_t *rail_cells(int64_t node) const {
    return (int64_t *)(base_ + rail_cells_) + node * num_features_;
  }

 private:
  char *base_;
  int64_t num_nodes_, local_size_, num_features_, lookup_keys_, remote_keys_, rounds_cap_;
  uint64_t node_keys_, node_weights_, remote_cells_, remote_map_, proxy_header_, key_stage_,
      rail_keys_, cell_stage_, rail_cells_, bytes_;
  uint64_t key_stage_bytes_, rail_keys_bytes_;
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
