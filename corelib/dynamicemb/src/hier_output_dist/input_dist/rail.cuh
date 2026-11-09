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
// kernels share: a rank's count messages (count_messages_of), and the rail's layout from the rail's
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

// The tables a rank's count messages (CountMessage) are made of: of its node layout need_remote and
// own_lengths, of its proxy plan need_key_rows, chunk_length_per_bucket and
// node_length_per_bucket_feature, and the op's lookup groups.
struct CountParts {
  const int64_t *need_remote;                     // 0-dim
  const int64_t *own_lengths;                     // [num_features][local_size]
  const int64_t *need_key_rows;                   // 0-dim
  const int64_t *chunk_length_per_bucket;         // [num_nodes]
  const int64_t *node_length_per_bucket_feature;  // [num_nodes][num_features]
  const int64_t *group_starts;  // [num_groups + 1]: each lookup group's first feature, then
                                // num_features
};

// Every thread of a CTA of kThreads: this rank's count message (`format`) to the rail peer on each
// node into row n of `rows` (slot_words apart), and its message to its own node into `own` too,
// word by word. The header goes into row 0 first, from which the other rows copy it. Its own
// part's keys of a group are those of its node's local_size requesters of the group's features
// (own_lengths), its node-unique keys of an (owner node, group) those of the group's features.
__device__ inline void count_messages_of(const CountParts &in, const CountMessage &format,
                                         int64_t local_size, int64_t node, int64_t *rows,
                                         int64_t *own) {
  const int64_t num_nodes = format.num_nodes, num_features = format.num_features,
                num_groups = format.num_groups, stride = format.slot_words(),
                header = format.header_length(), length = format.length();
  const int lane = threadIdx.x & 31;
  // the scalar words and the chunks
  if (threadIdx.x == 0) {
    rows[CountMessage::kNeedRemoteWord] = *in.need_remote;
    rows[CountMessage::kNeedKeyRowsWord] = *in.need_key_rows;
  }
  for (int64_t owner_node = threadIdx.x; owner_node < num_nodes; owner_node += kThreads)
    rows[format.chunk_length_word(owner_node)] = in.chunk_length_per_bucket[owner_node];
  // A warp per group of its own part, its lanes taking the (feature, requester) entries; then a
  // warp per (owner node, group), its lanes taking the group's features.
  for (int64_t index = threadIdx.x >> 5; index < num_groups * (num_nodes + 1); index += kWarps) {
    const int64_t group = index % num_groups, owner_node = index / num_groups - 1;
    const int64_t lo = in.group_starts[group], hi = in.group_starts[group + 1];
    int64_t keys = 0;
    if (owner_node < 0) {
      for (int64_t entry = lo * local_size + lane; entry < hi * local_size; entry += 32)
        keys += in.own_lengths[entry];
    } else {
      const int64_t *lengths = in.node_length_per_bucket_feature + owner_node * num_features;
      for (int64_t feature = lo + lane; feature < hi; feature += 32) keys += lengths[feature];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
      keys += __shfl_xor_sync(0xffffffffu, keys, offset);
    if (lane == 0)
      rows[owner_node < 0 ? format.own_total_word(group)
                          : format.key_total_word(owner_node, group)] = keys;
  }
  __syncthreads();
  // Each row: the header, then its node's part.
  for (int64_t index = threadIdx.x; index < num_nodes * length; index += kThreads) {
    const int64_t row = index / length, word = index % length;
    const int64_t value =
        word < header ? rows[word]
                      : in.node_length_per_bucket_feature[row * num_features + word - header];
    if (row > 0 || word >= header) rows[row * stride + word] = value;
    if (row == node) own[word] = value;
  }
}

// The rail's count messages as a rank reads them: row `source` the message of the rail peer on
// node `source` to this rank (this rank's own too), `stride` words apart. Every rank of the rail
// reads the same headers. In rail_exchange they are its count_msg slots of the call, which its rail
// peers put, so loads go through L2 only.
struct RailCounts {
  const int64_t *base;
  int64_t stride;
  CountMessage format;
  // word `index` of the message from node `source`
  __device__ __forceinline__ int64_t at(int64_t source, int64_t index) const {
    return (int64_t)__ldcg((const unsigned long long *)(base + source * stride + index));
  }
  // as owner, the keys of lookup group `group` of node `owner_node`'s own part
  __device__ __forceinline__ int64_t own(int64_t owner_node, int64_t group) const {
    return at(owner_node, format.own_total_word(group));
  }
  // as proxy, node `source`'s chunks of owner node `owner_node`, and its node-unique keys of
  // lookup group `group` there (none of its own node)
  __device__ __forceinline__ int64_t chunks(int64_t source, int64_t owner_node) const {
    return at(source, format.chunk_length_word(owner_node));
  }
  __device__ __forceinline__ int64_t keys(int64_t source, int64_t owner_node, int64_t group) const {
    return source == owner_node ? 0 : at(source, format.key_total_word(owner_node, group));
  }
  // its node-unique keys of owner node `owner_node`
  __device__ __forceinline__ int64_t node_keys(int64_t source, int64_t owner_node) const {
    int64_t total = 0;
    for (int64_t group = 0; group < format.num_groups; ++group)
      total += keys(source, owner_node, group);
    return total;
  }
  // as proxy, node `source`'s node-unique keys of feature `feature` of this rank's node
  __device__ __forceinline__ int64_t feature_keys(int64_t source, int64_t feature) const {
    return at(source, format.key_length_word(feature));
  }
};

// The rail layout's tables (RailLayout) as kernels write them.
struct RailLayoutView {
  int64_t *put_dst, *put_bytes;
  bool *put_skip;
  int64_t *bwd_groups, *kjt_lengths, *kjt_offsets, *num_recv, *length_per_key, *slice_in,
      *node_length_per_source, *node_length_per_source_feature, *fwd_groups, *group_offsets,
      *group_shift, *needs;
};
// The data of each table of `rail`.
inline RailLayoutView view_of(const RailLayout &rail) {
  return RailLayoutView{rail.put_dst.data_ptr<int64_t>(),
                        rail.put_bytes.data_ptr<int64_t>(),
                        rail.put_skip.data_ptr<bool>(),
                        rail.bwd_groups.data_ptr<int64_t>(),
                        rail.kjt_lengths.data_ptr<int64_t>(),
                        rail.kjt_offsets.data_ptr<int64_t>(),
                        rail.num_recv.data_ptr<int64_t>(),
                        rail.length_per_key.data_ptr<int64_t>(),
                        rail.slice_in.data_ptr<int64_t>(),
                        rail.node_length_per_source.data_ptr<int64_t>(),
                        rail.node_length_per_source_feature.data_ptr<int64_t>(),
                        rail.fwd_groups.data_ptr<int64_t>(),
                        rail.group_offsets.data_ptr<int64_t>(),
                        rail.group_shift.data_ptr<int64_t>(),
                        rail.needs.data_ptr<int64_t>()};
}

// The int64 words rail_layout_of works in: per (owner node, group) the rows of the owner's
// lookup_rows before the group's slice (group_rows, num_groups + 1 per owner node: the last its
// keys in all).
__host__ __device__ inline int64_t rail_layout_scratch(const Topo &topo, int64_t num_groups) {
  return topo.num_nodes * (num_groups + 1);
}

// A rail peer's run of a lookup group (common.h, GroupRuns), the same function for the owner on
// node `owner_node` and the proxy on node `source` (another node), so a proxy's backward puts land
// where its owners' lookups wrote, bit for bit. Where the source's keys of the group start among
// its keys of the owner node (its keys of the groups before), and the shift of their rows: the
// group's slice of the owner's lookup_rows starts after the groups before (group_rows, rail_layout
// of), then the group's own part, then the keys of the other nodes before the source.
__device__ __forceinline__ int64_t group_first(const RailCounts &in, int64_t owner_node,
                                               int64_t source, int64_t group) {
  int64_t first = 0;
  for (int64_t before = 0; before < group; ++before) first += in.keys(source, owner_node, before);
  return first;
}
__device__ __forceinline__ int64_t group_shift_of(const RailCounts &in, const int64_t *group_rows,
                                                  int64_t owner_node, int64_t source, int64_t group,
                                                  int64_t first) {
  int64_t row =
      group_rows[owner_node * (in.format.num_groups + 1) + group] + in.own(owner_node, group);
  for (int64_t before = 0; before < source; ++before) row += in.keys(before, owner_node, group);
  return row - first;
}

// Every thread of a CTA of kThreads: the rail's layout (RailLayout) into `out`, from the rail's
// count messages (`in`) and this rank's own part's lengths (own_lengths [feature][requester], the
// node layout's). scratch: rail_layout_scratch words. A slice a proxy puts holds the owner node's
// chunks (OwnerChunk, 16 bytes each), then its records; none to its own node. The per-node and
// per-group tables come from the headers alone (O(num_nodes^2 * num_groups) words read); this
// rank's KJT from the parts of the messages to it.
__device__ inline void rail_layout_of(const RailCounts &in, const int64_t *own_lengths,
                                      const Sizes &sizes, int64_t *scratch,
                                      const RailLayoutView &out) {
  const int64_t num_nodes = sizes.topo.num_nodes, node = sizes.topo.node,
                local_size = sizes.topo.local_size, num_features = sizes.num_features,
                num_groups = sizes.num_groups, kjt_stride = sizes.topo.kjt_stride,
                runs = GroupRuns::words_per_node(num_groups);
  // [owner node][group]: the rows of its lookup_rows before each group's slice, a thread per
  // owner node: each group's own part and every other node's keys of it.
  int64_t *group_rows = scratch;
  for (int64_t owner_node = threadIdx.x; owner_node < num_nodes; owner_node += kThreads) {
    int64_t rows = 0;
    for (int64_t group = 0; group <= num_groups; ++group) {
      group_rows[owner_node * (num_groups + 1) + group] = rows;
      if (group == num_groups) break;
      rows += in.own(owner_node, group);
      for (int64_t source = 0; source < num_nodes; ++source)
        rows += in.keys(source, owner_node, group);
    }
  }
  __syncthreads();
  // the bytes of the slice the proxy on `source` puts at the owner on `owner_node`
  auto slice_bytes = [&](int64_t source, int64_t owner_node) -> int64_t {
    if (source == owner_node) return 0;
    return in.chunks(source, owner_node) * (int64_t)sizeof(OwnerChunk) +
           in.node_keys(source, owner_node) * sizes.record_bytes;
  };
  // the keys the owner on `owner_node` looks up: those of its own node, and
  // the node-unique keys every proxy of the rail has for it
  auto recv = [&](int64_t owner_node) {
    return group_rows[owner_node * (num_groups + 1) + num_groups];
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
  // chunks of this node), and its node-unique keys.
  for (int64_t source = threadIdx.x; source < num_nodes; source += kThreads) {
    int64_t start = 0;
    for (int64_t earlier = 0; earlier < source; ++earlier) start += slice_bytes(earlier, node);
    out.slice_in[2 * source] = start;
    out.slice_in[2 * source + 1] = source == node ? 0 : in.chunks(source, node);
    out.node_length_per_source[source] = in.node_keys(source, node);
  }
  // The group runs: as owner of each source node's records (fwd_groups), as proxy of its keys of
  // each owner node (bwd_groups), a thread per (other node, group), its own node's all 0. And as
  // owner the shift of each group's own part, and where each group's slice starts.
  for (int64_t index = threadIdx.x; index < num_nodes * runs; index += kThreads) {
    const int64_t other = index / runs, word = index % runs;
    int64_t fwd = 0, bwd = 0;
    if (other != node) {
      const int64_t group = word <= num_groups ? word : word - num_groups - 1;
      const int64_t fwd_first = group_first(in, node, other, group),
                    bwd_first = group_first(in, other, node, group);
      fwd = word <= num_groups ? fwd_first
                               : group_shift_of(in, group_rows, node, other, group, fwd_first);
      bwd = word <= num_groups ? bwd_first
                               : group_shift_of(in, group_rows, other, node, group, bwd_first);
    }
    out.fwd_groups[index] = fwd;
    out.bwd_groups[index] = bwd;
  }
  for (int64_t group = threadIdx.x; group <= num_groups; group += kThreads) {
    const int64_t rows = group_rows[node * (num_groups + 1) + group];
    out.group_offsets[group] = rows;
    if (group == num_groups) continue;
    int64_t own_before = 0;
    for (int64_t before = 0; before < group; ++before) own_before += in.own(node, before);
    out.group_shift[group] = rows - own_before;
  }
  // As owner, each source node's keys of each feature (its own node: 0), and its KJT's lengths,
  // (feature, column) feature-major: its node's requesters, then the other nodes in order.
  for (int64_t index = threadIdx.x; index < num_nodes * num_features; index += kThreads)
    out.node_length_per_source_feature[index] =
        index / num_features == node ? 0
                                     : in.feature_keys(index / num_features, index % num_features);
  auto kjt_length = [&](int64_t index) {
    const int64_t feature = index / kjt_stride, column = index % kjt_stride;
    if (column < local_size) return own_lengths[feature * local_size + column];
    const int64_t other = column - local_size;  // the other nodes, in order
    return in.feature_keys(other < node ? other : other + 1, feature);
  };
  // the lengths and their offsets in one pass, a tile of loads at a time
  const int64_t num_lengths = num_features * kjt_stride;
  const int64_t kjt_keys = cta_scan_blocked<kScanItems, int64_t>(
      num_lengths,
      [&](int64_t index) {
        const int64_t length = kjt_length(index);
        out.kjt_lengths[index] = length;
        return length;
      },
      [&](int64_t index, int64_t offset) { out.kjt_offsets[index] = offset; });
  if (threadIdx.x == 0) out.kjt_offsets[num_lengths] = kjt_keys;
  __syncthreads();
  for (int64_t feature = threadIdx.x; feature < num_features; feature += kThreads)
    out.length_per_key[feature] =
        out.kjt_offsets[(feature + 1) * kjt_stride] - out.kjt_offsets[feature * kjt_stride];
  // The keys it looks up, and the most the rail needs of each capacity.
  const int64_t most_recv = cta_max<int64_t>(num_nodes, recv, 0);
  const int64_t most_remote = cta_max<int64_t>(
      num_nodes, [&](int64_t source) { return in.at(source, CountMessage::kNeedRemoteWord); }, 0);
  const int64_t most_key_rows = cta_max<int64_t>(
      num_nodes, [&](int64_t source) { return in.at(source, CountMessage::kNeedKeyRowsWord); }, 0);
  if (threadIdx.x == 0) {
    *out.num_recv = recv(node);
    out.needs[kNeedLookupKeys] = most_recv;
    out.needs[kNeedRemoteKeys] = most_remote;
    out.needs[kNeedKeyRows] = most_key_rows;
  }
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
