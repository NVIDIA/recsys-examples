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
// The device code of the rail's part of a call (planner.h) that the comm kernels and the planner's
// kernels share: a rank's count message (count_message_of), and the rail's layout from the rail's
// count messages (rail_layout_of). rail_exchange makes both; the hooks' count_message and
// rail_layout make them again.
//
// It does not include device.cuh, so planner.cu does not get nccl_device.h. The build has no
// relocatable device code: everything here is inline or a template.
#pragma once

#include <ATen/ATen.h>

#include <cstdint>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/input_dist/planner.cuh"
#include "hier_output_dist/input_dist/planner.h"

namespace dyn_emb {
namespace hier_output_dist {

// The tables a rank's count message (CountMessage) is made of, in its order: of its node layout
// need_remote, own_total (this rank's own part) and own_lengths, and of its proxy plan key_rows,
// chunk_length_per_bucket and node_length_per_bucket_feature.
struct CountParts {
  const int64_t *need_remote;                     // 0-dim
  const int64_t *own_total;                       // 0-dim
  const int64_t *key_rows;                        // 0-dim
  const int64_t *chunk_length_per_bucket;         // [num_nodes]
  const int64_t *own_lengths;                     // [num_features][local_size]
  const int64_t *node_length_per_bucket_feature;  // [num_nodes][num_features]
};

// Every thread of a CTA of kThreads: this rank's count message (`format`) into `message`, and into
// `copy` too unless it is nullptr, word by word. Its own part's keys of a feature are those of its
// node's local_size requesters (own_lengths).
__device__ inline void count_message_of(const CountParts &in, const CountMessage &format,
                                        int64_t local_size, int64_t *message, int64_t *copy) {
  const int64_t own_first = format.own_length_per_feature(0),
                keys_first = format.node_length_per_bucket_feature(0, 0);
  for (int64_t word = threadIdx.x; word < format.length(); word += kThreads) {
    int64_t value;
    if (word == CountMessage::kNeedRemote) {
      value = *in.need_remote;
    } else if (word == CountMessage::kOwnTotal) {
      value = *in.own_total;
    } else if (word == CountMessage::kKeyRows) {
      value = *in.key_rows;
    } else if (word < own_first) {
      value = in.chunk_length_per_bucket[word - format.chunk_length_per_bucket(0)];
    } else if (word < keys_first) {
      const int64_t *lengths = in.own_lengths + (word - own_first) * local_size;
      value = 0;
      for (int64_t requester = 0; requester < local_size; ++requester) value += lengths[requester];
    } else {
      value = in.node_length_per_bucket_feature[word - keys_first];
    }
    message[word] = value;
    if (copy != nullptr) copy[word] = value;
  }
}

// The rail's count messages as a rank reads them: row `source` the message of the rail peer on
// node `source` (this rank's own too), `stride` words apart. Every rank of the rail reads the same
// messages. In rail_exchange they are its count_msg slots of the call, which its rail peers put, so
// loads go through L2 only.
struct RailCounts {
  const int64_t *base;
  int64_t stride;
  CountMessage format;
  // word `index` of the message from node `source`
  __device__ __forceinline__ int64_t at(int64_t source, int64_t index) const {
    return (int64_t)__ldcg((const unsigned long long *)(base + source * stride + index));
  }
  // as proxy, node `source`'s node-unique keys of feature `feature` for the owner on node
  // `owner_node`, and its chunks of that owner node
  __device__ __forceinline__ int64_t keys(int64_t source, int64_t owner_node,
                                          int64_t feature) const {
    return at(source, format.node_length_per_bucket_feature(owner_node, feature));
  }
  __device__ __forceinline__ int64_t chunks(int64_t source, int64_t owner_node) const {
    return at(source, format.chunk_length_per_bucket(owner_node));
  }
  // as owner, the keys of feature `feature` of node `owner_node`'s own part
  __device__ __forceinline__ int64_t own(int64_t owner_node, int64_t feature) const {
    return at(owner_node, format.own_length_per_feature(feature));
  }
};

// The rail layout's tables (RailLayout) as kernels write them.
struct RailLayoutView {
  int64_t *put_dst, *put_bytes;
  bool *put_skip;
  int64_t *kjt_lengths, *kjt_offsets, *num_recv, *length_per_key, *slice_in,
      *node_length_per_source, *node_length_per_source_feature, *source_offsets, *owner_offsets,
      *requester_offsets, *needs;
};
// The data of each table of `rail`.
inline RailLayoutView view_of(const RailLayout &rail) {
  return RailLayoutView{rail.put_dst.data_ptr<int64_t>(),
                        rail.put_bytes.data_ptr<int64_t>(),
                        rail.put_skip.data_ptr<bool>(),
                        rail.kjt_lengths.data_ptr<int64_t>(),
                        rail.kjt_offsets.data_ptr<int64_t>(),
                        rail.num_recv.data_ptr<int64_t>(),
                        rail.length_per_key.data_ptr<int64_t>(),
                        rail.slice_in.data_ptr<int64_t>(),
                        rail.node_length_per_source.data_ptr<int64_t>(),
                        rail.node_length_per_source_feature.data_ptr<int64_t>(),
                        rail.source_offsets.data_ptr<int64_t>(),
                        rail.owner_offsets.data_ptr<int64_t>(),
                        rail.requester_offsets.data_ptr<int64_t>(),
                        rail.needs.data_ptr<int64_t>()};
}

// The int64 words rail_layout_of works in: per (source node, owner node) the source's node-unique
// keys for the owner, and the offsets of every rail owner's KJT features, owner after owner.
__host__ __device__ inline int64_t rail_layout_scratch(const Topo &topo, int64_t num_features) {
  const int64_t num_nodes = topo.num_nodes;
  return num_nodes * num_nodes + num_nodes * num_features + 1;
}

// The keys of feature `feature` in the KJT of the rail owner on node `owner_node`: its own part's,
// then those of the proxy on each other node.
__device__ __forceinline__ int64_t owner_feature_keys(const RailCounts &in, int64_t owner_node,
                                                      int64_t feature) {
  int64_t keys = in.own(owner_node, feature);
  for (int64_t source = 0; source < in.format.num_nodes; ++source)
    if (source != owner_node) keys += in.keys(source, owner_node, feature);
  return keys;
}

// Where the keys of feature `feature` from the proxy on node `source` start in the KJT of the rail
// owner on node `owner_node` (another node): after the KJT's features before (feature_offsets, the
// offsets of owner_feature_keys, owner after owner), the owner's own part of the feature, then the
// keys of the other nodes before `source` (the KJT's columns: its node's requesters, then the
// other nodes in order). rail_layout_of computes it as owner for each proxy of the rail
// (source_offsets), and as proxy for each owner (owner_offsets): one function of the same
// messages, so a proxy's owner_offsets are its owners' source_offsets, bit for bit.
__device__ __forceinline__ int64_t source_offset(const RailCounts &in,
                                                 const int64_t *feature_offsets, int64_t owner_node,
                                                 int64_t source, int64_t feature) {
  const int64_t first = owner_node * in.format.num_features;
  int64_t offset =
      feature_offsets[first + feature] - feature_offsets[first] + in.own(owner_node, feature);
  for (int64_t before = 0; before < source; ++before)
    if (before != owner_node) offset += in.keys(before, owner_node, feature);
  return offset;
}

// Every thread of a CTA of kThreads: the rail's layout (RailLayout) into `out`, from the rail's
// count messages (`in`) and this rank's own part's lengths (own_lengths [feature][requester], the
// node layout's). scratch: rail_layout_scratch words. A slice a proxy puts holds the owner node's
// chunks (ChunkPiece, 16 bytes each), then its records; none to its own node. A proxy's keys of an
// owner node are the sum of its keys of each feature there, as proxy_keys counts them.
__device__ inline void rail_layout_of(const RailCounts &in, const int64_t *own_lengths,
                                      const Sizes &sizes, int64_t *scratch,
                                      const RailLayoutView &out) {
  const int64_t num_nodes = sizes.topo.num_nodes, node = sizes.topo.node,
                local_size = sizes.topo.local_size, num_features = sizes.num_features,
                num_sources = sizes.topo.num_sources;
  // [source][owner node]: the source's node-unique keys for the owner node, a warp each, its lanes
  // taking the features. [owner node][feature] (and the total): the offsets of each owner's KJT
  // features, owner after owner.
  int64_t *node_keys = scratch, *feature_offsets = node_keys + num_nodes * num_nodes;
  const int lane = threadIdx.x & 31;
  for (int64_t index = threadIdx.x >> 5; index < num_nodes * num_nodes; index += kWarps) {
    const int64_t source = index / num_nodes, owner_node = index % num_nodes;
    int64_t keys = 0;
    for (int64_t feature = lane; feature < num_features; feature += 32)
      keys += in.keys(source, owner_node, feature);
    for (int offset = 16; offset > 0; offset >>= 1)
      keys += __shfl_xor_sync(0xffffffffu, keys, offset);
    if (lane == 0) node_keys[index] = keys;
  }
  // (its first __syncthreads makes node_keys whole)
  cta_offsets_of<int64_t>(
      num_nodes * num_features,
      [&](int64_t index) {
        return owner_feature_keys(in, index / num_features, index % num_features);
      },
      feature_offsets);
  // the bytes of the slice the proxy on `source` puts at the owner on `owner_node`
  auto slice_bytes = [&](int64_t source, int64_t owner_node) -> int64_t {
    if (source == owner_node) return 0;
    return in.chunks(source, owner_node) * (int64_t)sizeof(ChunkPiece) +
           node_keys[source * num_nodes + owner_node] * sizes.record_bytes;
  };
  // the keys the owner on `owner_node` looks up: those of its own node, and
  // the node-unique keys every proxy of the rail has for it
  auto recv = [&](int64_t owner_node) {
    int64_t keys = in.at(owner_node, CountMessage::kOwnTotal);
    for (int64_t source = 0; source < num_nodes; ++source)
      keys += node_keys[source * num_nodes + owner_node];
    return keys;
  };
  // As proxy: where its slice starts at each owner (after the slices of the
  // nodes before it), its bytes, and whether it puts none (the owner
  // overflows lookup_keys, or it is on this node).
  for (int64_t owner_node = threadIdx.x; owner_node < num_nodes; owner_node += kThreads) {
    int64_t dst = 0;
    for (int64_t source = 0; source < node; ++source) dst += slice_bytes(source, owner_node);
    out.put_dst[owner_node] = dst;
    out.put_bytes[owner_node] = slice_bytes(node, owner_node);
    out.put_skip[owner_node] = owner_node == node || recv(owner_node) > sizes.lookup_keys;
  }
  // As owner: each source node's slice in its rail_keys (byte offset,
  // chunks of this node), its node-unique keys, and its keys of each feature.
  for (int64_t source = threadIdx.x; source < num_nodes; source += kThreads) {
    int64_t start = 0;
    for (int64_t earlier = 0; earlier < source; ++earlier) start += slice_bytes(earlier, node);
    out.slice_in[2 * source] = start;
    out.slice_in[2 * source + 1] = source == node ? 0 : in.chunks(source, node);
    out.node_length_per_source[source] = node_keys[source * num_nodes + node];
  }
  for (int64_t index = threadIdx.x; index < num_nodes * num_features; index += kThreads)
    out.node_length_per_source_feature[index] =
        in.keys(index / num_features, node, index % num_features);
  // Its KJT's lengths, (feature, source) feature-major: its node's
  // requesters, then the other nodes in order. Then where each requester of
  // its node has its keys of each feature in it.
  auto kjt_length = [&](int64_t index) {
    const int64_t feature = index / num_sources, source = index % num_sources;
    if (source < local_size) return own_lengths[feature * local_size + source];
    const int64_t column = source - local_size;  // the other nodes, in order
    return in.keys(column < node ? column : column + 1, node, feature);
  };
  const int64_t num_lengths = num_features * num_sources;
  for (int64_t index = threadIdx.x; index < num_lengths; index += kThreads)
    out.kjt_lengths[index] = kjt_length(index);
  cta_offsets_of<int64_t>(num_lengths, kjt_length, out.kjt_offsets);
  for (int64_t feature = threadIdx.x; feature < num_features; feature += kThreads)
    out.length_per_key[feature] =
        out.kjt_offsets[(feature + 1) * num_sources] - out.kjt_offsets[feature * num_sources];
  for (int64_t index = threadIdx.x; index < local_size * num_features; index += kThreads) {
    const int64_t requester = index / num_features, feature = index % num_features;
    out.requester_offsets[index] = out.kjt_offsets[feature * num_sources + requester];
  }
  // As owner, where each other node's keys of each feature start in its KJT (what that node's
  // proxy computes as its owner_offsets); as proxy, where its node's start in each other node's
  // owner's KJT (what that owner computes as its source_offsets). Its own node: 0.
  for (int64_t index = threadIdx.x; index < num_nodes * num_features; index += kThreads) {
    const int64_t other = index / num_features, feature = index % num_features;
    out.source_offsets[index] =
        other == node ? 0 : source_offset(in, feature_offsets, node, other, feature);
    out.owner_offsets[index] =
        other == node ? 0 : source_offset(in, feature_offsets, other, node, feature);
  }
  // The keys it looks up, and the most the rail needs of each capacity.
  const int64_t most_recv = cta_max<int64_t>(num_nodes, recv, 0);
  const int64_t most_remote = cta_max<int64_t>(
      num_nodes, [&](int64_t source) { return in.at(source, CountMessage::kNeedRemote); }, 0);
  const int64_t most_key_rows = cta_max<int64_t>(
      num_nodes, [&](int64_t source) { return in.at(source, CountMessage::kKeyRows); }, 0);
  if (threadIdx.x == 0) {
    *out.num_recv = recv(node);
    out.needs[kNeedLookupKeys] = most_recv;
    out.needs[kNeedRemoteKeys] = most_remote;
    out.needs[kNeedKeyRows] = most_key_rows;
  }
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
