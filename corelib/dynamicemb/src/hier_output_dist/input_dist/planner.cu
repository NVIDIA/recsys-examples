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
// The planner's steps of the node, of the rail and the owner, of the
// requester, and the put lists (planner.h).

#include <algorithm>
#include <cstdint>

#include "hier_output_dist/common/window.h"
#include "hier_output_dist/input_dist/planner.cuh"
#include "hier_output_dist/input_dist/planner.h"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

// `index` clamped into a table of `count` entries: a call that does not fit
// may read words no one wrote.
__device__ __forceinline__ int64_t clamp_index(int64_t index, int64_t count) {
  return min(max(index, (int64_t)0), count - 1);
}

// ── the node ──

// One CTA: this rank's count vector (CountVector) from part 1's cell_start.
__global__ void __launch_bounds__(kThreads)
    count_vector_kernel(const int64_t *cell_start, Topo topo, int64_t num_features,
                        int64_t *counts) {
  const int64_t local_size = topo.local_size;
  const CountVector format{local_size, num_features};
  // its keys for each target's remote part: its blocks of the owners with the
  // target's local rank on the other nodes
  for (int64_t target = threadIdx.x; target < local_size; target += kThreads) {
    int64_t keys = 0;
    for (int64_t owner_node = 0; owner_node < topo.num_nodes; ++owner_node) {
      const int64_t owner = owner_node * local_size + target;
      if (owner_node != topo.node)
        keys += cell_start[(owner + 1) * num_features] - cell_start[owner * num_features];
    }
    counts[format.remote(target)] = keys;
  }
  // its keys for each owner of its node and feature: the cell's
  for (int64_t index = threadIdx.x; index < local_size * num_features; index += kThreads) {
    const int64_t target = index / num_features, feature = index % num_features;
    const int64_t cell = (topo.node * local_size + target) * num_features + feature;
    counts[format.own(target, feature)] = cell_start[cell + 1] - cell_start[cell];
  }
}

// One CTA: the layout of the node's keys (NodeLayout) from the node's counts
// (totals, before, mine: NodeCounts) and this rank's part 1.
// requester_start and need_remote: nullptr on one node.
__global__ void __launch_bounds__(kThreads)
    node_layout_kernel(const int64_t *totals, const int64_t *before, const int64_t *mine,
                       const int64_t *cell_start, const int64_t *block_start, Topo topo,
                       int64_t num_features, int64_t lookup_keys, int64_t remote_keys,
                       int64_t *own_cells, int64_t *own_starts, int64_t *own_total,
                       int64_t *requester_start, int64_t *need_lookup, int64_t *need_remote,
                       bool *skip, int64_t *rows_to_push, int64_t *destination_base) {
  const int64_t local_size = topo.local_size, num_nodes = topo.num_nodes, node = topo.node;
  const CountVector words{local_size, num_features};
  // This rank's own part, in its KJT order [feature][requester]: each
  // requester's keys of each feature for it (in mine, after those for its
  // remote part), and where each cell starts: the ends of every cell but the
  // last.
  const int64_t own_part_cells = num_features * local_size;
  auto own_cell = [&](int64_t cell) {
    const int64_t feature = cell / local_size, requester = cell % local_size;
    return mine[requester * (num_features + 1) + 1 + feature];
  };
  for (int64_t cell = threadIdx.x; cell < own_part_cells; cell += kThreads)
    own_cells[cell] = own_cell(cell);
  cta_exclusive_ends<int64_t>(own_part_cells - 1, own_cell, own_starts);
  // The node's keys for each owner of the node, and the largest own part.
  for (int64_t target = threadIdx.x; target < local_size; target += kThreads) {
    int64_t keys = 0;
    for (int64_t feature = 0; feature < num_features; ++feature)
      keys += totals[words.own(target, feature)];
    own_total[target] = keys;
  }
  __syncthreads();
  const int64_t most_own = cta_max<int64_t>(
      local_size, [&](int64_t target) { return own_total[target]; }, 0);
  // Across nodes, where each requester's keys start in this rank's remote
  // part, and the largest remote part. The needs, and whether the node skips
  // the call, are the same on every rank of the node.
  bool skips = most_own > lookup_keys;
  if (num_nodes > 1) {
    cta_exclusive_ends<int64_t>(
        local_size, [&](int64_t requester) { return mine[requester * (num_features + 1)]; },
        requester_start);
    const int64_t most_remote = cta_max<int64_t>(
        local_size, [&](int64_t target) { return totals[words.remote(target)]; }, 0);
    skips = skips || most_remote > remote_keys;
    if (threadIdx.x == 0) *need_remote = most_remote;
  }
  if (threadIdx.x == 0) {
    *need_lookup = most_own;
    *skip = skips;
    *rows_to_push = skips ? 0 : block_start[topo.world_size];
  }
  // Row `row` of part 1 in cell `cell` goes to destination_base[cell] + row.
  // A row of an owner of this node goes to the owner's own part: where this
  // rank's cell starts there (the node's keys of the features before, then
  // those of the requesters before), and the rows of the cell before it. A
  // thread per owner walks its features.
  for (int64_t target = threadIdx.x; target < local_size; target += kThreads) {
    int64_t feature_start = 0;  // where the owner's keys of `feature` start in its own part
    for (int64_t feature = 0; feature < num_features; ++feature) {
      const int64_t cell = (node * local_size + target) * num_features + feature;
      destination_base[cell] =
          feature_start + before[words.own(target, feature)] - cell_start[cell];
      feature_start += totals[words.own(target, feature)];
    }
  }
  // A remote row of owner (owner node, target) goes to its target's remote
  // part: after the requesters before this rank, this rank's blocks for the
  // owner nodes before, and the rows of its block before it. A thread per
  // (target, feature) walks the nodes.
  for (int64_t index = threadIdx.x; index < local_size * num_features; index += kThreads) {
    const int64_t target = index / num_features, feature = index % num_features;
    int64_t block_place = lookup_keys + before[words.remote(target)];  // where its next block goes
    for (int64_t owner_node = 0; owner_node < num_nodes; ++owner_node) {
      if (owner_node == node) continue;
      const int64_t owner = owner_node * local_size + target;
      destination_base[owner * num_features + feature] = block_place - block_start[owner];
      block_place += block_start[owner + 1] - block_start[owner];
    }
  }
  // the rows past the valid ones (cell world_size * num_features) are never pushed
  if (threadIdx.x == 0) destination_base[topo.world_size * num_features] = 0;
}

// A thread per position of the own part, up to the keys this rank looks up
// (at most the own part): its key and its weight, the owner's KJT.
__global__ void __launch_bounds__(kThreads)
    own_part_keys_kernel(const int64_t *node_keys, const float *node_weights,
                         const int64_t *num_recv, int64_t own_part, int64_t *kjt_values,
                         float *kjt_weights) {
  const int64_t keys = min(*num_recv, own_part);
  for (int64_t position = first_item(); position < keys; position += item_stride()) {
    kjt_values[position] = node_keys[position];
    if (node_weights != nullptr) kjt_weights[position] = node_weights[position];
  }
}

// One CTA: the owner's KJT offsets, the ends of its cells (feature,
// requester), and its keys of each feature.
__global__ void __launch_bounds__(kThreads)
    one_node_offsets_kernel(const int64_t *own_cells, int64_t local_size, int64_t num_features,
                            int64_t *kjt_offsets, int64_t *length_per_key) {
  cta_exclusive_ends<int64_t>(
      num_features * local_size, [&](int64_t cell) { return own_cells[cell]; }, kjt_offsets);
  for (int64_t feature = threadIdx.x; feature < num_features; feature += kThreads)
    length_per_key[feature] =
        kjt_offsets[(feature + 1) * local_size] - kjt_offsets[feature * local_size];
}

// A thread per row of part 1: its place in its owner's KJT, where it was
// pushed (a row past the valid ones: its own index).
__global__ void __launch_bounds__(kThreads)
    one_node_src_kernel(const int32_t *row_cell, const int64_t *destination_base, int64_t rows,
                        int32_t *src) {
  for (int64_t row = first_item(); row < rows; row += item_stride())
    src[row] = (int32_t)(destination_base[row_cell[row]] + row);
}

}  // namespace

at::Tensor node_count_vector(const at::Tensor &cell_start, const Topo &topo, int64_t num_features) {
  const at::Tensor counts =
      at::empty({CountVector{topo.local_size, num_features}.words()}, cell_start.options());
  count_vector_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      cell_start.data_ptr<int64_t>(), topo, num_features, counts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return counts;
}

NodeLayout node_layout(const NodeCounts &counts, const at::Tensor &cell_start,
                       const at::Tensor &block_start, const Topo &topo, int64_t num_features,
                       int64_t lookup_keys, int64_t remote_keys) {
  const int64_t local_size = topo.local_size;
  const auto long_options = cell_start.options();
  NodeLayout node;
  node.own_cells = at::empty({num_features * local_size}, long_options);
  node.own_starts = at::empty({num_features * local_size}, long_options);
  node.own_total = at::empty({local_size}, long_options);
  node.need_lookup = at::empty({}, long_options);
  node.skip = at::empty({}, long_options.dtype(at::kBool));
  node.rows_to_push = at::empty({}, long_options);
  node.destination_base = at::empty({topo.world_size * num_features + 1}, long_options);
  if (topo.num_nodes > 1) {  // one node has no remote part
    node.requester_start = at::empty({local_size + 1}, long_options);
    node.need_remote = at::empty({}, long_options);
  }
  node_layout_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      counts.totals.data_ptr<int64_t>(), counts.before.data_ptr<int64_t>(),
      counts.mine.data_ptr<int64_t>(), cell_start.data_ptr<int64_t>(),
      block_start.data_ptr<int64_t>(), topo, num_features, lookup_keys, remote_keys,
      node.own_cells.data_ptr<int64_t>(), node.own_starts.data_ptr<int64_t>(),
      node.own_total.data_ptr<int64_t>(), data_or_null<int64_t>(node.requester_start),
      node.need_lookup.data_ptr<int64_t>(), data_or_null<int64_t>(node.need_remote),
      node.skip.data_ptr<bool>(), node.rows_to_push.data_ptr<int64_t>(),
      node.destination_base.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return node;
}

OneNodeOwner one_node_owner(const NodeLayout &node, const at::Tensor &row_cell,
                            const at::Tensor &node_keys, const at::Tensor &node_weights,
                            const Topo &topo, int64_t num_features) {
  const int64_t own_part = node_keys.numel(), rows = row_cell.numel();
  const cudaStream_t stream = current_stream();
  OneNodeOwner owner;
  // The own part is the KJT: a copy of its first num_recv keys (wait()
  // narrows it to num_recv). The node's next call rewrites node_keys after
  // its barrier, which does not wait for this call's lookup.
  owner.num_recv = node.own_total.select(0, topo.local_rank);
  owner.kjt_values = at::empty({own_part}, node_keys.options());
  if (node_weights.defined()) owner.kjt_weights = at::empty({own_part}, node_weights.options());
  own_part_keys_kernel<<<item_grid(own_part), kThreads, 0, stream>>>(
      node_keys.data_ptr<int64_t>(), data_or_null<float>(node_weights),
      owner.num_recv.data_ptr<int64_t>(), own_part, owner.kjt_values.data_ptr<int64_t>(),
      data_or_null<float>(owner.kjt_weights));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  owner.kjt_lengths = node.own_cells;
  owner.kjt_offsets = at::empty({node.own_cells.numel() + 1}, node.own_cells.options());
  owner.length_per_key = at::empty({num_features}, node.own_cells.options());
  one_node_offsets_kernel<<<kOneCta, kThreads, 0, stream>>>(
      node.own_cells.data_ptr<int64_t>(), topo.local_size, num_features,
      owner.kjt_offsets.data_ptr<int64_t>(), owner.length_per_key.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  owner.src = at::empty({rows}, row_cell.options());
  if (rows > 0) {
    one_node_src_kernel<<<item_grid(rows), kThreads, 0, stream>>>(
        row_cell.data_ptr<int32_t>(), node.destination_base.data_ptr<int64_t>(), rows,
        owner.src.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return owner;
}

// ── the rail and the owner ──

namespace {

// One CTA: the count message this rank sends its rail (CountMessage), word
// by word: the common part (its node's largest remote part, its rounds as
// proxy, its own part as owner, its node-unique keys per owner node), then
// for each node its node-unique keys of each feature there.
__global__ void __launch_bounds__(kThreads)
    count_message_kernel(const int64_t *need_remote, const int64_t *num_rounds,
                         const int64_t *own_total, const int64_t *uniq_out,
                         const int64_t *cell_uniq, CountMessage format, int64_t *message) {
  for (int64_t word = threadIdx.x; word < format.out_words(); word += kThreads) {
    int64_t value;
    if (word == CountMessage::kNeedRemote)
      value = *need_remote;
    else if (word == CountMessage::kRounds)
      value = *num_rounds;
    else if (word == CountMessage::kOwnTotal)
      value = *own_total;
    else if (word < format.common())
      value = uniq_out[word - format.uniq_out(0)];
    else
      value = cell_uniq[word - format.common()];
    message[word] = value;
  }
}

// One CTA: the rail's layout (RailLayout) from the count messages this rank
// got (messages [num_nodes][CountMessage::length], its own too) and its own
// part's cells (own_cells, [feature][requester]). A slice a proxy puts holds
// its round pieces (16 bytes each), then its records; none to its own node.
__global__ void __launch_bounds__(kThreads)
    rail_layout_kernel(const int64_t *messages, const int64_t *own_cells,
                       const int64_t *other_nodes, Sizes sizes, int64_t *put_dst,
                       int64_t *put_bytes, bool *put_skip, int64_t *kjt_lengths,
                       int64_t *kjt_offsets, int64_t *num_recv, int64_t *length_per_key,
                       int64_t *slice_in, int64_t *uniq_in, int64_t *cells_in, int64_t *cell_stage,
                       int64_t *my_cell_starts, int64_t *needs) {
  const int64_t num_nodes = sizes.topo.num_nodes, node = sizes.topo.node,
                local_size = sizes.topo.local_size, num_features = sizes.num_features,
                num_sources = sizes.topo.num_sources;
  const CountMessage format{num_nodes, num_features};
  // word `word` of the message from the rail peer on node `source`
  auto message = [&](int64_t source, int64_t word) {
    return messages[source * format.length() + word];
  };
  // the bytes of the slice the proxy on `source` puts at the owner on `owner_node`
  auto slice_bytes = [&](int64_t source, int64_t owner_node) -> int64_t {
    if (source == owner_node) return 0;
    return message(source, CountMessage::kRounds) * 16 +
           message(source, format.uniq_out(owner_node)) * sizes.record_bytes;
  };
  // the keys the owner on `owner_node` looks up: those of its own node, and
  // the node-unique keys every proxy of the rail has for it
  auto recv = [&](int64_t owner_node) {
    int64_t keys = message(owner_node, CountMessage::kOwnTotal);
    for (int64_t source = 0; source < num_nodes; ++source)
      keys += message(source, format.uniq_out(owner_node));
    return keys;
  };
  // As proxy: where its slice starts at each owner (after the slices of the
  // nodes before it), its bytes, and whether it puts none (the owner
  // overflows lookup_keys, or it is on this node).
  for (int64_t owner_node = threadIdx.x; owner_node < num_nodes; owner_node += kThreads) {
    int64_t dst = 0;
    for (int64_t source = 0; source < node; ++source) dst += slice_bytes(source, owner_node);
    put_dst[owner_node] = dst;
    put_bytes[owner_node] = slice_bytes(node, owner_node);
    put_skip[owner_node] = owner_node == node || recv(owner_node) > sizes.lookup_keys;
  }
  // As owner: each source node's slice in its rail_keys (byte offset,
  // rounds), its node-unique keys, and its keys of each feature (the part of
  // its message for this node).
  for (int64_t source = threadIdx.x; source < num_nodes; source += kThreads) {
    int64_t start = 0;
    for (int64_t earlier = 0; earlier < source; ++earlier) start += slice_bytes(earlier, node);
    slice_in[2 * source] = start;
    slice_in[2 * source + 1] = message(source, CountMessage::kRounds);
    uniq_in[source] = message(source, format.uniq_out(node));
  }
  for (int64_t index = threadIdx.x; index < num_nodes * num_features; index += kThreads)
    cells_in[index] = message(index / num_features, format.common() + index % num_features);
  // Its KJT, cell (feature, source) feature-major: its node's keys, then the
  // other nodes' in order. Then where each requester of its node has its
  // cells in it, and each other node (what it sends that node's proxy; its
  // own node: 0).
  auto kjt_length = [&](int64_t cell) {
    const int64_t feature = cell / num_sources, source = cell % num_sources;
    return source < local_size
               ? own_cells[feature * local_size + source]
               : message(other_nodes[source - local_size], format.common() + feature);
  };
  const int64_t kjt_cells = num_features * num_sources;
  for (int64_t cell = threadIdx.x; cell < kjt_cells; cell += kThreads)
    kjt_lengths[cell] = kjt_length(cell);
  cta_exclusive_ends<int64_t>(kjt_cells, kjt_length, kjt_offsets);
  for (int64_t feature = threadIdx.x; feature < num_features; feature += kThreads)
    length_per_key[feature] =
        kjt_offsets[(feature + 1) * num_sources] - kjt_offsets[feature * num_sources];
  for (int64_t index = threadIdx.x; index < local_size * num_features; index += kThreads) {
    const int64_t requester = index / num_features, feature = index % num_features;
    my_cell_starts[index] = kjt_offsets[feature * num_sources + requester];
  }
  for (int64_t feature = threadIdx.x; feature < num_features; feature += kThreads)
    cell_stage[node * num_features + feature] = 0;
  for (int64_t index = threadIdx.x; index < (num_nodes - 1) * num_features; index += kThreads) {
    const int64_t column = index / num_features, feature = index % num_features;
    cell_stage[other_nodes[column] * num_features + feature] =
        kjt_offsets[feature * num_sources + local_size + column];
  }
  // The keys it looks up, and the most the rail needs of lookup_keys and
  // remote_keys, and its most rounds.
  const int64_t most_recv = cta_max<int64_t>(num_nodes, recv, 0);
  const int64_t most_remote = cta_max<int64_t>(
      num_nodes, [&](int64_t source) { return message(source, CountMessage::kNeedRemote); }, 0);
  const int64_t most_rounds = cta_max<int64_t>(
      num_nodes, [&](int64_t source) { return message(source, CountMessage::kRounds); }, 0);
  if (threadIdx.x == 0) {
    *num_recv = recv(node);
    needs[kNeedLookupKeys] = most_recv;
    needs[kNeedRemoteKeys] = most_remote;
    needs[kNeedRounds] = most_rounds;
  }
}

// One CTA: what this rank publishes to its node, its rail's needs, then
// where each requester of its node has its cells in its KJT.
__global__ void __launch_bounds__(kThreads)
    status_publish_kernel(const int64_t *needs, const int64_t *my_cell_starts, int64_t cell_words,
                          int64_t *publish) {
  for (int64_t word = threadIdx.x; word < kNumNeeds + cell_words; word += kThreads)
    publish[word] = word < kNumNeeds ? needs[word] : my_cell_starts[word - kNumNeeds];
}

// One CTA: the owner's tables of its slices in rail_keys (rail_ints: its
// int32 words), from the rail's slice_in. Each source node's rounds (its own
// node sends none) and round pieces (fwd_pieces: the lo, hi and row words of
// each RoundPiece, which come first in its slice; zeros past its rounds).
// And where each other node's records of each feature start in rail_keys
// (cell_record [feature][other node], int64 words): after its round pieces
// and its records of the features before.
__global__ void __launch_bounds__(kThreads)
    owner_pieces_kernel(const int64_t *slice_in, const int64_t *cells_in, const int32_t *rail_ints,
                        int64_t rail_int_count, const int64_t *other_nodes, Sizes sizes,
                        int64_t *fwd_rounds, int64_t *fwd_pieces, int64_t *cell_record) {
  const int64_t num_nodes = sizes.topo.num_nodes, num_features = sizes.num_features;
  auto rounds_of = [&](int64_t source) {
    return source == sizes.topo.node ? (int64_t)0 : slice_in[2 * source + 1];
  };
  for (int64_t source = threadIdx.x; source < num_nodes; source += kThreads)
    fwd_rounds[source] = rounds_of(source);
  for (int64_t index = threadIdx.x; index < sizes.rounds_cap * num_nodes * 3; index += kThreads) {
    const int64_t round = index / (num_nodes * 3), source = index / 3 % num_nodes,
                  field = index % 3;
    const int64_t word = slice_in[2 * source] / 4 + 4 * round + field;
    fwd_pieces[index] =
        round < rounds_of(source) ? rail_ints[clamp_index(word, rail_int_count)] : 0;
  }
  for (int64_t column = threadIdx.x; column < num_nodes - 1; column += kThreads) {
    const int64_t source = other_nodes[column];
    int64_t record = (slice_in[2 * source] + 16 * slice_in[2 * source + 1]) / 8;
    for (int64_t feature = 0; feature < num_features; ++feature) {
      cell_record[feature * (num_nodes - 1) + column] = record;
      record += cells_in[source * num_features + feature] * (sizes.record_bytes / 8);
    }
  }
}

// A thread per position of the owner's KJT, up to the keys it looks up (at
// most lookup_keys): the cell (feature, source) it is in and its place
// there. A key of its own node is in the own part of node_keys (its weight
// 1 when the keys carry none), another node's in its record in rail_keys
// (int64 words: the key, then with 16-byte records the weight's bits).
__global__ void __launch_bounds__(kThreads)
    owner_kjt_kernel(const int64_t *kjt_offsets, const int64_t *num_recv, const int64_t *own_starts,
                     const int64_t *cell_record, const int64_t *node_keys,
                     const float *node_weights, const int64_t *rail_keys, int64_t rail_words,
                     Sizes sizes, int64_t *kjt_values, float *kjt_weights) {
  const int64_t local_size = sizes.topo.local_size, num_sources = sizes.topo.num_sources,
                other_columns = sizes.topo.num_nodes - 1, lookup_keys = sizes.lookup_keys,
                kjt_cells = sizes.num_features * num_sources, record_words = sizes.record_bytes / 8;
  const float *rail_floats = (const float *)rail_keys;
  const int64_t keys = min(*num_recv, lookup_keys);
  for (int64_t position = first_item(); position < keys; position += item_stride()) {
    const int64_t cell =
        clamp_index(upper_bound(kjt_offsets, kjt_cells + 1, position) - 1, kjt_cells);
    const int64_t feature = cell / num_sources, source = cell % num_sources,
                  in_cell = position - kjt_offsets[cell];
    if (source < local_size) {
      const int64_t own =
          clamp_index(own_starts[feature * local_size + source] + in_cell, lookup_keys);
      kjt_values[position] = node_keys[own];
      if (kjt_weights != nullptr)
        kjt_weights[position] = node_weights != nullptr ? node_weights[own] : 1.0f;
    } else {
      const int64_t record =
          cell_record[feature * other_columns + source - local_size] + in_cell * record_words;
      kjt_values[position] = rail_keys[clamp_index(record, rail_words)];
      // the weight: the low half of the record's second word
      if (kjt_weights != nullptr)
        kjt_weights[position] = rail_floats[clamp_index(2 * record + 2, 2 * rail_words)];
    }
  }
}

}  // namespace

at::Tensor other_nodes_of(const Topo &topo, const at::TensorOptions &options) {
  const at::Tensor nodes = at::arange(topo.num_nodes - 1, options.dtype(at::kLong));
  return nodes + nodes.ge(topo.node);
}

at::Tensor count_message(const at::Tensor &need_remote, const at::Tensor &num_rounds,
                         const at::Tensor &own_total, const at::Tensor &uniq_out,
                         const at::Tensor &cell_uniq) {
  const int64_t num_nodes = uniq_out.numel();
  const CountMessage format{num_nodes, cell_uniq.numel() / num_nodes};
  const at::Tensor message = at::empty({format.out_words()}, uniq_out.options());
  count_message_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      need_remote.data_ptr<int64_t>(), num_rounds.data_ptr<int64_t>(),
      own_total.data_ptr<int64_t>(), uniq_out.data_ptr<int64_t>(), cell_uniq.data_ptr<int64_t>(),
      format, message.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return message;
}

RailLayout rail_layout(const at::Tensor &messages, const at::Tensor &own_cells,
                       const at::Tensor &other_nodes, const Sizes &sizes) {
  const int64_t num_nodes = sizes.topo.num_nodes, local_size = sizes.topo.local_size,
                num_features = sizes.num_features,
                kjt_cells = num_features * sizes.topo.num_sources;
  const auto long_options = messages.options();
  RailLayout rail;
  rail.put_dst = at::empty({num_nodes}, long_options);
  rail.put_bytes = at::empty({num_nodes}, long_options);
  rail.put_skip = at::empty({num_nodes}, long_options.dtype(at::kBool));
  rail.kjt_lengths = at::empty({kjt_cells}, long_options);
  rail.kjt_offsets = at::empty({kjt_cells + 1}, long_options);
  rail.num_recv = at::empty({}, long_options);
  rail.length_per_key = at::empty({num_features}, long_options);
  rail.slice_in = at::empty({num_nodes, 2}, long_options);
  rail.uniq_in = at::empty({num_nodes}, long_options);
  rail.cells_in = at::empty({num_nodes, num_features}, long_options);
  rail.cell_stage = at::empty({num_nodes, num_features}, long_options);
  rail.my_cell_starts = at::empty({local_size, num_features}, long_options);
  rail.needs = at::empty({kNumNeeds}, long_options);
  rail_layout_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      messages.data_ptr<int64_t>(), own_cells.data_ptr<int64_t>(), other_nodes.data_ptr<int64_t>(),
      sizes, rail.put_dst.data_ptr<int64_t>(), rail.put_bytes.data_ptr<int64_t>(),
      rail.put_skip.data_ptr<bool>(), rail.kjt_lengths.data_ptr<int64_t>(),
      rail.kjt_offsets.data_ptr<int64_t>(), rail.num_recv.data_ptr<int64_t>(),
      rail.length_per_key.data_ptr<int64_t>(), rail.slice_in.data_ptr<int64_t>(),
      rail.uniq_in.data_ptr<int64_t>(), rail.cells_in.data_ptr<int64_t>(),
      rail.cell_stage.data_ptr<int64_t>(), rail.my_cell_starts.data_ptr<int64_t>(),
      rail.needs.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return rail;
}

at::Tensor status_publish(const RailLayout &rail) {
  const int64_t cell_words = rail.my_cell_starts.numel();
  const at::Tensor publish = at::empty({kNumNeeds + cell_words}, rail.needs.options());
  status_publish_kernel<<<kOneCta, kThreads, 0, current_stream()>>>(
      rail.needs.data_ptr<int64_t>(), rail.my_cell_starts.data_ptr<int64_t>(), cell_words,
      publish.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return publish;
}

OwnerTables owner_tables(const RailLayout &rail, const at::Tensor &own_starts,
                         const at::Tensor &node_keys, const at::Tensor &node_weights,
                         const at::Tensor &rail_keys, const at::Tensor &other_nodes,
                         const Sizes &sizes) {
  const int64_t num_nodes = sizes.topo.num_nodes, lookup_keys = sizes.lookup_keys,
                rail_words = rail_keys.numel();
  const auto long_options = rail_keys.options().dtype(at::kLong);
  const cudaStream_t stream = current_stream();
  OwnerTables tables;
  // Each source node's slice in rail_keys starts with its round pieces, then
  // its records.
  tables.fwd_rounds = at::empty({num_nodes}, long_options);
  tables.fwd_pieces = at::empty({sizes.rounds_cap, num_nodes, 3}, long_options);
  const at::Tensor cell_record = at::empty({sizes.num_features * (num_nodes - 1)}, long_options);
  owner_pieces_kernel<<<kOneCta, kThreads, 0, stream>>>(
      rail.slice_in.data_ptr<int64_t>(), rail.cells_in.data_ptr<int64_t>(),
      (const int32_t *)rail_keys.data_ptr<int64_t>(), 2 * rail_words,
      other_nodes.data_ptr<int64_t>(), sizes, tables.fwd_rounds.data_ptr<int64_t>(),
      tables.fwd_pieces.data_ptr<int64_t>(), cell_record.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The KJT's keys, its first num_recv positions: its own node's cells from
  // the own part of node_keys, the other nodes' from their records in
  // rail_keys.
  tables.kjt_values = at::empty({lookup_keys}, long_options);
  if (sizes.record_bytes == 16)
    tables.kjt_weights = at::empty({lookup_keys}, long_options.dtype(at::kFloat));
  owner_kjt_kernel<<<item_grid(lookup_keys), kThreads, 0, stream>>>(
      rail.kjt_offsets.data_ptr<int64_t>(), rail.num_recv.data_ptr<int64_t>(),
      own_starts.data_ptr<int64_t>(), cell_record.data_ptr<int64_t>(),
      node_keys.data_ptr<int64_t>(), data_or_null<float>(node_weights),
      rail_keys.data_ptr<int64_t>(), rail_words, sizes, tables.kjt_values.data_ptr<int64_t>(),
      data_or_null<float>(tables.kjt_weights));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tables;
}

// ── the requester ──

namespace {

// A thread per row of part 1: the node peer whose remote_map has its words,
// for a valid row of another node's owner (its proxy: the owner's local
// rank; else -1), and its occurrence there: its destination less
// lookup_keys, clamped into the remote part.
__global__ void __launch_bounds__(kThreads)
    remote_rows_kernel(const int32_t *row_cell, const int64_t *block_start,
                       const int64_t *destination_base, int64_t rows, Sizes sizes, int32_t *peer,
                       int64_t *first_word) {
  const int64_t local_size = sizes.topo.local_size, world_size = sizes.topo.world_size;
  const int64_t valid_rows = block_start[world_size];
  for (int64_t row = first_item(); row < rows; row += item_stride()) {
    const int64_t cell = row_cell[row];
    const int64_t owner = min(cell / sizes.num_features, world_size - 1);
    const bool remote = owner / local_size != sizes.topo.node && row < valid_rows;
    peer[row] = remote ? (int32_t)(owner % local_size) : -1;
    first_word[row] =
        clamp_index(destination_base[cell] + row - sizes.lookup_keys, sizes.remote_keys);
  }
}

// One CTA: this requester's tables of its node's proxies, from their
// gathered headers (headers [local_size][ProxyHeader::gathered_words]): where
// its rows at each proxy start in its push order (push_order_off: its rows of
// the owners with the proxy's local rank on the other nodes), each proxy's
// rounds (from 1 to rounds_cap) and round table, and its push tables there
// made absolute: positions in push_order (its rows at the proxies before
// come first), groups after its groups at the proxies before; their heavy
// starts in (proxy, round) order (heavy_start); and the task starts.
__global__ void __launch_bounds__(kThreads)
    requester_proxy_tables_kernel(const int32_t *headers, const int64_t *block_start, Sizes sizes,
                                  int64_t *push_order_off, int64_t *proxy_rounds,
                                  int32_t *proxy_round_tables, ReqEntry *push_tables,
                                  int32_t *heavy_start, int64_t *fwd_task_starts,
                                  int64_t *bwd_task_starts) {
  const ProxyHeader header = ProxyHeader::of(sizes);
  const int64_t local_size = sizes.topo.local_size, rounds_cap = sizes.rounds_cap,
                entries = rounds_cap + 1, table_words = header.table_words();
  auto header_of = [&](int64_t proxy) { return headers + proxy * header.gathered_words(); };
  // its push table as the proxy published it: positions relative to its push
  // order there, groups relative to its first group
  auto published = [&](int64_t proxy) {
    return (const ReqEntry *)(header_of(proxy) + header.gathered_push_table());
  };
  auto rounds_of = [&](int64_t proxy) {
    return min(max((int64_t)header_of(proxy)[ProxyHeader::kGatheredRounds], (int64_t)1),
               rounds_cap);
  };
  auto rows_at = [&](int64_t proxy) {
    int64_t rows = 0;
    for (int64_t owner_node = 0; owner_node < sizes.topo.num_nodes; ++owner_node) {
      const int64_t owner = owner_node * local_size + proxy;
      if (owner_node != sizes.topo.node) rows += block_start[owner + 1] - block_start[owner];
    }
    return rows;
  };
  cta_exclusive_ends<int64_t>(local_size, rows_at, push_order_off);
  for (int64_t proxy = threadIdx.x; proxy < local_size; proxy += kThreads)
    proxy_rounds[proxy] = rounds_of(proxy);
  for (int64_t index = threadIdx.x; index < local_size * table_words; index += kThreads)
    proxy_round_tables[index] =
        header_of(index / table_words)[ProxyHeader::kGatheredTable + index % table_words];
  for (int64_t proxy = threadIdx.x; proxy < local_size; proxy += kThreads) {
    int64_t groups_before = 0;
    for (int64_t earlier = 0; earlier < proxy; ++earlier)
      groups_before += published(earlier)[rounds_of(earlier)].groups;
    for (int64_t entry = 0; entry < entries; ++entry) {
      const ReqEntry relative = published(proxy)[entry];
      ReqEntry &absolute = push_tables[proxy * entries + entry];
      absolute.start = (int32_t)(relative.start + push_order_off[proxy]);
      absolute.heavy = (int32_t)(relative.heavy + push_order_off[proxy]);
      absolute.flags = relative.flags;
      absolute.groups = (int32_t)(relative.groups + groups_before);
      heavy_start[proxy * entries + entry] = absolute.heavy;
    }
  }
  // The task starts over all rounds: per round a slot per task of its
  // busiest proxy there, for every proxy, kTaskRows rows or heavy groups a
  // task; a proxy has none past its rounds. The forward's are its rows' and
  // its heavy rows' each round, the backward's its heavy groups' then its
  // rows'. (The push tables' differences are the same relative or absolute.)
  auto tasks = [&](int64_t proxy, int64_t round, int64_t first, int64_t end) -> int64_t {
    return round < rounds_of(proxy) ? ceil_div<kTaskRows>(end - first) : 0;
  };
  auto row_tasks = [&](int64_t proxy, int64_t round) {
    const ReqEntry *table = published(proxy);
    return tasks(proxy, round, table[round].start, table[round].heavy);
  };
  auto heavy_tasks = [&](int64_t proxy, int64_t round) {
    const ReqEntry *table = published(proxy);
    return tasks(proxy, round, table[round].heavy, table[round + 1].start);
  };
  auto group_tasks = [&](int64_t proxy, int64_t round) {
    const ReqEntry *table = published(proxy);
    return tasks(proxy, round, table[round].groups, table[round + 1].groups);
  };
  cta_exclusive_ends<int64_t>(
      rounds_cap,
      [&](int64_t round) {
        int64_t most = 0;
        for (int64_t proxy = 0; proxy < local_size; ++proxy)
          most = max(most, row_tasks(proxy, round) + heavy_tasks(proxy, round));
        return most * local_size;
      },
      fwd_task_starts);
  cta_exclusive_ends<int64_t>(
      2 * rounds_cap,
      [&](int64_t slot) {
        const int64_t round = slot / 2;
        int64_t most = 0;
        for (int64_t proxy = 0; proxy < local_size; ++proxy)
          most = max(most, slot % 2 == 0 ? group_tasks(proxy, round) : row_tasks(proxy, round));
        return most * local_size;
      },
      bwd_task_starts);
}

// A thread per row of part 1. A row of an owner on this node is in the
// owner's KJT, where this requester's cell starts plus its place in the
// cell, both forward and backward. Another node's row reads its key's row in
// proxy_rows (bit 31, a heavy key, dropped) and pushes its gradient to its
// slot, from its proxy's remote_map (words [3][rows], RemoteMap's arrays); a
// valid one is at its push position at the proxy in its push order.
__global__ void __launch_bounds__(kThreads)
    requester_rows_kernel(const int32_t *words, const int64_t *owner_cell_start,
                          const int32_t *row_cell, const int64_t *cell_start,
                          const int64_t *block_start, const int64_t *push_order_off, int64_t rows,
                          Sizes sizes, int32_t *src, int32_t *dst, int32_t *push_order) {
  const int64_t local_size = sizes.topo.local_size, num_features = sizes.num_features,
                world_size = sizes.topo.world_size;
  const int32_t *key_row = words, *slot = words + rows, *push_position = words + 2 * rows;
  const int64_t valid_rows = block_start[world_size];
  for (int64_t row = first_item(); row < rows; row += item_stride()) {
    const int64_t cell = row_cell[row];
    const int64_t owner = min(cell / num_features, world_size - 1);
    const int64_t peer = owner % local_size;  // the owner itself on this node, else its proxy
    if (owner / local_size == sizes.topo.node) {
      const int32_t place = (int32_t)(owner_cell_start[peer * num_features + cell % num_features] +
                                      row - cell_start[cell]);
      src[row] = place;
      dst[row] = place;
      continue;
    }
    src[row] = key_row[row] & 0x7FFFFFFF;
    dst[row] = slot[row];
    // a call that does not fit may read words no one wrote: a position out
    // of range is dropped
    const int64_t position = push_order_off[peer] + push_position[row];
    if (row < valid_rows && position >= 0 && position < rows) push_order[position] = (int32_t)row;
  }
}

// The push order as the heavy groups read it: the row at a position (a call
// that does not fit may hold rows no one wrote: clamped), whether it is
// heavy (bit 31 of its key's row) and its slot.
struct PushOrder {
  const int32_t *order, *key_row, *dst;
  int64_t rows;
  __device__ int64_t row(int64_t position) const { return clamp_index(order[position], rows); }
  __device__ bool heavy(int64_t position) const { return key_row[row(position)] < 0; }
  __device__ int32_t slot(int64_t position) const { return dst[row(position)]; }
};

// A thread per position of the push order (rows of them; its rows at every
// proxy come first): whether a heavy group starts there. A group is a run of
// positions of heavy rows with one slot (dst) inside one (proxy, round)
// heavy range: a heavy position starts one unless the position before is
// heavy too, with its slot and its heavy range. Different heavy keys of a
// range have different slots; rounds t and t + 2 share a half of
// proxy_rows, so their keys' slots may be equal. The absolute heavy starts
// are non-decreasing over (proxy, round): an entry past a proxy's rounds
// holds its end.
__global__ void __launch_bounds__(kThreads)
    group_starts_kernel(PushOrder order, const int64_t *push_order_off, int64_t local_size,
                        const int32_t *heavy_start, int64_t heavy_entries, bool *group_start) {
  const int64_t positions = push_order_off[local_size];
  auto heavy_range = [&](int64_t position) {
    return upper_bound(heavy_start, heavy_entries, position);
  };
  for (int64_t position = first_item(); position < order.rows; position += item_stride()) {
    bool starts = false;
    if (position < positions && order.heavy(position))
      starts = position == 0 || !order.heavy(position - 1) ||
               order.slot(position) != order.slot(position - 1) ||
               heavy_range(position) != heavy_range(position - 1);
    group_start[position] = starts;
  }
}

// A thread per position of its push order, and one past its end: a group's
// start writes it, numbered in position order (group_before); the position
// after a group's last (the next run's start, or the end) writes its end.
// Groups past group_bound (a call that does not fit) are left out.
__global__ void __launch_bounds__(kThreads)
    heavy_groups_kernel(PushOrder order, const int64_t *push_order_off, int64_t local_size,
                        const bool *group_start, const int32_t *group_before, int64_t group_bound,
                        HeavyGroup *heavy_groups) {
  const int64_t positions = push_order_off[local_size];
  for (int64_t position = first_item(); position <= positions; position += item_stride()) {
    const bool starts = position < positions && group_start[position];
    if (starts && group_before[position] < group_bound)
      heavy_groups[group_before[position]].start = (int32_t)position;
    const bool ends = position > 0 && order.heavy(position - 1) &&
                      (position == positions || !order.heavy(position) || starts);
    if (ends && group_before[position] - 1 < group_bound)
      heavy_groups[group_before[position] - 1].end = (int32_t)position;
  }
}

// A thread per heavy group of the bound: its sub-tasks of kHeavy positions
// (none past the groups).
__global__ void __launch_bounds__(kThreads)
    group_subs_kernel(const HeavyGroup *heavy_groups, const int32_t *group_before, int64_t rows,
                      int64_t group_bound, int64_t *subs) {
  const int64_t groups = group_before[rows];
  for (int64_t group = first_item(); group < group_bound; group += item_stride())
    subs[group] =
        group < groups
            ? ceil_div<kHeavy>((int64_t)heavy_groups[group].end - heavy_groups[group].start)
            : 0;
}

// A thread per sub-task of the bound: its group, the first whose sub-tasks
// end after it (past the sub-tasks: the last group). The threads of the
// first group_bound sub-tasks also write each group's first sub-task.
__global__ void __launch_bounds__(kThreads)
    group_of_sub_kernel(const int64_t *sub_before, int64_t group_bound, int64_t sub_bound,
                        int32_t *group_of_sub, int32_t *group_first_sub) {
  for (int64_t sub = first_item(); sub < sub_bound; sub += item_stride()) {
    group_of_sub[sub] =
        (int32_t)min(upper_bound(sub_before + 1, group_bound, sub), group_bound - 1);
    if (sub < group_bound) group_first_sub[sub] = (int32_t)sub_before[sub];
  }
}

}  // namespace

PeerWords remote_rows(const at::Tensor &row_cell, const at::Tensor &block_start,
                      const at::Tensor &destination_base, const Sizes &sizes) {
  const int64_t rows = row_cell.numel();
  PeerWords gather;
  gather.peer = at::empty({rows}, row_cell.options());
  gather.first_word = at::empty({rows}, block_start.options());
  if (rows > 0) {
    remote_rows_kernel<<<item_grid(rows), kThreads, 0, current_stream()>>>(
        row_cell.data_ptr<int32_t>(), block_start.data_ptr<int64_t>(),
        destination_base.data_ptr<int64_t>(), rows, sizes, gather.peer.data_ptr<int32_t>(),
        gather.first_word.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return gather;
}

PeerWords header_gather(const Sizes &sizes, const at::TensorOptions &options) {
  const ProxyHeader header = ProxyHeader::of(sizes);
  const int64_t local_size = sizes.topo.local_size, words = header.gathered_words(),
                push_table = header.push_table_word(sizes.topo.local_rank);
  const auto long_options = options.dtype(at::kLong);
  // of each proxy's header, in the order of its gathered words: its rounds,
  // its round table, then its push tables for this requester
  const at::Tensor first_word =
      at::cat({at::arange(ProxyHeader::kRoundsWord, ProxyHeader::kRoundsWord + 1, long_options),
               at::arange(ProxyHeader::kTableWord, ProxyHeader::kTableWord + header.table_words(),
                          long_options),
               at::arange(push_table, push_table + header.push_words(), long_options)});
  PeerWords gather;
  gather.peer = at::arange(local_size, options.dtype(at::kInt))
                    .view({local_size, 1})
                    .expand({local_size, words})
                    .reshape({-1});
  gather.first_word = first_word.view({1, words}).expand({local_size, words}).reshape({-1});
  return gather;
}

RequesterTables requester_tables(const at::Tensor &words, const at::Tensor &headers,
                                 const at::Tensor &owner_cell_start, const at::Tensor &row_cell,
                                 const at::Tensor &cell_start, const at::Tensor &block_start,
                                 const Sizes &sizes) {
  const int64_t rows = row_cell.numel(), local_size = sizes.topo.local_size,
                rounds_cap = sizes.rounds_cap, heavy_entries = local_size * (rounds_cap + 1);
  const auto long_options = cell_start.options().dtype(at::kLong);
  const auto int_options = cell_start.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  RequesterTables tables;
  // Per proxy, from its gathered header: its rounds, its round table, and
  // this requester's push tables there, made absolute; and the task starts.
  tables.push_order_off = at::empty({local_size + 1}, long_options);
  tables.proxy_rounds = at::empty({local_size}, long_options);
  tables.proxy_round_tables =
      at::empty({local_size, ProxyHeader::of(sizes).table_words()}, int_options);
  tables.push_tables = at::empty({local_size, rounds_cap + 1, 4}, int_options);
  const at::Tensor heavy_start = at::empty({heavy_entries}, int_options);
  tables.fwd_task_starts = at::empty({rounds_cap + 1}, long_options);
  tables.bwd_task_starts = at::empty({2 * rounds_cap + 1}, long_options);
  requester_proxy_tables_kernel<<<kOneCta, kThreads, 0, stream>>>(
      headers.data_ptr<int32_t>(), block_start.data_ptr<int64_t>(), sizes,
      tables.push_order_off.data_ptr<int64_t>(), tables.proxy_rounds.data_ptr<int64_t>(),
      tables.proxy_round_tables.data_ptr<int32_t>(),
      (ReqEntry *)tables.push_tables.data_ptr<int32_t>(), heavy_start.data_ptr<int32_t>(),
      tables.fwd_task_starts.data_ptr<int64_t>(), tables.bwd_task_starts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t group_bound = std::min(rows, local_size * (sizes.remote_keys / (kHeavy + 1) + 1));
  const int64_t sub_bound = rows / kHeavy + group_bound + 1;
  tables.src = at::empty({rows}, int_options);
  tables.dst = at::empty({rows}, int_options);
  tables.push_order = at::empty({rows}, int_options);
  tables.heavy_groups = at::empty({group_bound, 2}, int_options);
  tables.group_first_sub = at::empty({group_bound}, int_options);
  tables.group_of_sub = at::empty({sub_bound}, int_options);
  if (rows == 0) {
    tables.group_subs = at::empty({}, long_options);
    AT_CUDA_CHECK(cudaMemsetAsync(tables.group_subs.data_ptr(), 0, sizeof(int64_t), stream));
    return tables;
  }
  // Per row: its src and dst, and its place in the push order, its rows at
  // each proxy at their push positions there.
  requester_rows_kernel<<<item_grid(rows), kThreads, 0, stream>>>(
      words.data_ptr<int32_t>(), owner_cell_start.data_ptr<int64_t>(), row_cell.data_ptr<int32_t>(),
      cell_start.data_ptr<int64_t>(), block_start.data_ptr<int64_t>(),
      tables.push_order_off.data_ptr<int64_t>(), rows, sizes, tables.src.data_ptr<int32_t>(),
      tables.dst.data_ptr<int32_t>(), tables.push_order.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The heavy groups, in its push order: where each starts, numbered by a
  // scan, then each group's start and end.
  const PushOrder order{tables.push_order.data_ptr<int32_t>(), words.data_ptr<int32_t>(),
                        tables.dst.data_ptr<int32_t>(), rows};
  const at::Tensor group_start = at::empty({rows}, cell_start.options().dtype(at::kBool));
  group_starts_kernel<<<item_grid(rows), kThreads, 0, stream>>>(
      order, tables.push_order_off.data_ptr<int64_t>(), local_size, heavy_start.data_ptr<int32_t>(),
      heavy_entries, group_start.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const at::Tensor group_before = exclusive_ends(group_start, at::kInt);
  HeavyGroup *heavy_groups = (HeavyGroup *)tables.heavy_groups.data_ptr<int32_t>();
  heavy_groups_kernel<<<item_grid(rows + 1), kThreads, 0, stream>>>(
      order, tables.push_order_off.data_ptr<int64_t>(), local_size, group_start.data_ptr<bool>(),
      group_before.data_ptr<int32_t>(), group_bound, heavy_groups);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Each group's sub-tasks of kHeavy positions, numbered in group order.
  const at::Tensor subs = at::empty({group_bound}, long_options);
  group_subs_kernel<<<item_grid(group_bound), kThreads, 0, stream>>>(
      heavy_groups, group_before.data_ptr<int32_t>(), rows, group_bound, subs.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const at::Tensor sub_before = exclusive_ends(subs);
  tables.group_subs = sub_before.select(0, group_bound);
  group_of_sub_kernel<<<item_grid(sub_bound), kThreads, 0, stream>>>(
      sub_before.data_ptr<int64_t>(), group_bound, sub_bound,
      tables.group_of_sub.data_ptr<int32_t>(), tables.group_first_sub.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tables;
}

// ── the put lists ──

namespace {

// A cut's sort key: its node, then its key (below 2^31).
constexpr int kNodeShift = 32;
constexpr int64_t kKeyMask = (int64_t{1} << kNodeShift) - 1;

// The two lists as one of 2 * num_nodes nodes: the backward's owner nodes
// (the proxy's round table, its rounds and its node-unique keys per (owner
// node, feature)), then the forward's proxy nodes (each source node's round
// pieces, rounds and keys per feature). A call that does not fit may read
// pieces no one wrote: their words are clamped to [0, 2^31).
struct PutPieces {
  const int32_t *round_table;  // [rounds_cap][num_nodes][4]
  const int64_t *fwd_pieces;   // [rounds_cap][num_nodes][3]
  const int64_t *num_rounds;   // 0-dim
  const int64_t *fwd_rounds;   // [num_nodes]
  const int64_t *cell_uniq;    // [num_nodes][num_features]
  const int64_t *cells_in;     // [num_nodes][num_features]
  int64_t num_nodes, num_features, rounds_cap, chunk_rows;

  // Word `field` of the piece of `node` in `round`: 0 its first key, 1 its
  // end, 2 the row of its first key in proxy_rows.
  __device__ int64_t word(int64_t round, int64_t node, int field) const {
    const int64_t value = node < num_nodes
                              ? round_table[(round * num_nodes + node) * 4 + field]
                              : fwd_pieces[(round * num_nodes + node - num_nodes) * 3 + field];
    return min(max(value, (int64_t)0), kKeyMask >> 1);
  }
  __device__ int64_t lo(int64_t round, int64_t node) const { return word(round, node, 0); }
  __device__ int64_t hi(int64_t round, int64_t node) const { return word(round, node, 1); }
  __device__ int64_t row(int64_t round, int64_t node) const { return word(round, node, 2); }
  // its IB chunks
  __device__ int64_t chunks(int64_t round, int64_t node) const {
    const int64_t keys = max(hi(round, node) - lo(round, node), (int64_t)0);
    return (keys + chunk_rows - 1) / chunk_rows;
  }
  __device__ int64_t rounds(int64_t node) const {
    return node < num_nodes ? *num_rounds : fwd_rounds[node - num_nodes];
  }
  // the node's keys of a feature
  __device__ int64_t keys(int64_t node, int64_t feature) const {
    return node < num_nodes ? cell_uniq[node * num_features + feature]
                            : cells_in[(node - num_nodes) * num_features + feature];
  }
};

// The rows of the segment from a cut to the next cut: the keys between them
// when both are of one node, else none.
__device__ __forceinline__ int64_t segment_rows(int64_t cut, int64_t next) {
  return (next >> kNodeShift) == (cut >> kNodeShift) ? (next & kKeyMask) - (cut & kKeyMask) : 0;
}

// One CTA: the tables of the pieces and the cells. chunk_before: where each
// piece's IB chunks start, pieces round-major (round, node), the last entry
// the chunks of both lists. piece_key [nodes][rounds_cap]: where each piece
// starts as a cut key, node-major (a node's pieces past its rounds start at
// its end). cell_key [nodes][num_features]: where each cell starts as a cut
// key. Then the proxy's round table with each piece's first chunk of the
// backward, and the reducers' task starts: a round's busiest owner node's
// chunks, for every owner node, chunk_rows / kTaskRows tasks a chunk.
__global__ void __launch_bounds__(kThreads)
    put_pieces_kernel(PutPieces pieces, int64_t *chunk_before, int64_t *piece_key,
                      int64_t *cell_key, int32_t *round_table, int64_t *reduce_task_starts) {
  const int64_t num_nodes = pieces.num_nodes, nodes = 2 * num_nodes,
                num_features = pieces.num_features, rounds_cap = pieces.rounds_cap;
  cta_exclusive_ends<int64_t>(
      rounds_cap * nodes,
      [&](int64_t piece) { return pieces.chunks(piece / nodes, piece % nodes); }, chunk_before);
  for (int64_t node = threadIdx.x; node < nodes; node += kThreads) {
    const int64_t base = node << kNodeShift;
    int64_t end = 0;  // the end of the node's cells so far
    for (int64_t feature = 0; feature < num_features; ++feature) {
      cell_key[node * num_features + feature] = base + end;
      end += pieces.keys(node, feature);
    }
    for (int64_t round = 0; round < rounds_cap; ++round)
      piece_key[node * rounds_cap + round] =
          base + (round < pieces.rounds(node) ? pieces.lo(round, node) : end);
  }
  for (int64_t index = threadIdx.x; index < rounds_cap * num_nodes * 4; index += kThreads) {
    const int64_t round = index / (num_nodes * 4), owner_node = index / 4 % num_nodes;
    round_table[index] = index % 4 == 3 ? (int32_t)chunk_before[round * nodes + owner_node]
                                        : pieces.round_table[index];
  }
  cta_exclusive_ends<int64_t>(
      rounds_cap,
      [&](int64_t round) {
        int64_t most = 0;
        for (int64_t owner_node = 0; owner_node < num_nodes; ++owner_node)
          most = max(most, pieces.chunks(round, owner_node));
        return most * (num_nodes * (pieces.chunk_rows / kTaskRows));
      },
      reduce_task_starts);
}

// A thread per cut, as a cut key (node << kNodeShift) + key: first the first
// key of each IB chunk (chunk_cuts of them; past the chunks: after every
// node), then the end of each cell.
__global__ void __launch_bounds__(kThreads)
    cut_keys_kernel(PutPieces pieces, const int64_t *chunk_before, const int64_t *cell_key,
                    int64_t chunk_cuts, int64_t *cut_keys) {
  const int64_t nodes = 2 * pieces.num_nodes, num_pieces = pieces.rounds_cap * nodes,
                cells = nodes * pieces.num_features;
  const int64_t num_chunks = chunk_before[num_pieces];
  for (int64_t cut = first_item(); cut < chunk_cuts + cells; cut += item_stride()) {
    int64_t key;
    if (cut >= chunk_cuts) {
      const int64_t cell = cut - chunk_cuts;
      key = cell_key[cell] + pieces.keys(cell / pieces.num_features, cell % pieces.num_features);
    } else if (cut < num_chunks) {
      // chunk `cut`, of the piece whose chunks hold it
      const int64_t piece = upper_bound(chunk_before + 1, num_pieces, cut);
      const int64_t round = piece / nodes, node = piece % nodes;
      key = (node << kNodeShift) + pieces.lo(round, node) +
            (cut - chunk_before[piece]) * pieces.chunk_rows;
    } else {
      key = nodes << kNodeShift;
    }
    cut_keys[cut] = key;
  }
}

// A thread per segment of the sorted cuts: whether it holds rows.
__global__ void __launch_bounds__(kThreads)
    kept_segments_kernel(const int64_t *sorted, int64_t segments, bool *kept) {
  for (int64_t segment = first_item(); segment < segments; segment += item_stride())
    kept[segment] = segment_rows(sorted[segment], sorted[segment + 1]) > 0;
}

// A thread per segment that holds rows: its PutSeg, at its place among them
// (kept_before). Its piece and its cell are the last of its node's that
// start at or before it: the piece gives its row in proxy_rows and its IB
// chunk, the cell its row in the owner's KJT (rail_cells for the backward's
// owner nodes, cell_stage for the forward's proxy nodes). The backward's
// src is the row of proxy_rows, the forward's the row of the KJT.
__global__ void __launch_bounds__(kThreads)
    put_segments_kernel(PutPieces pieces, const int64_t *sorted, const int32_t *kept_before,
                        int64_t segments, const int64_t *chunk_before, const int64_t *piece_key,
                        const int64_t *cell_key, const int64_t *rail_cells,
                        const int64_t *cell_stage, PutSeg *list) {
  const int64_t num_nodes = pieces.num_nodes, nodes = 2 * num_nodes, rounds_cap = pieces.rounds_cap,
                num_pieces = rounds_cap * nodes, cells = nodes * pieces.num_features;
  for (int64_t segment = first_item(); segment < segments; segment += item_stride()) {
    const int64_t cut = sorted[segment], rows = segment_rows(cut, sorted[segment + 1]);
    if (rows <= 0) continue;
    const int64_t first = cut & kKeyMask;
    const int64_t piece = clamp_index(upper_bound(piece_key, num_pieces, cut) - 1, num_pieces);
    const int64_t piece_node = piece / rounds_cap, round = piece % rounds_cap;
    const int64_t in_piece = first - pieces.lo(round, piece_node);
    const int64_t piece_row = pieces.row(round, piece_node) + in_piece;
    const int64_t chunk = chunk_before[round * nodes + piece_node] + in_piece / pieces.chunk_rows;
    const int64_t cell = clamp_index(upper_bound(cell_key, cells, cut) - 1, cells);
    const int64_t cell_row = cell < num_nodes * pieces.num_features
                                 ? rail_cells[cell]
                                 : cell_stage[cell - num_nodes * pieces.num_features];
    const int64_t kjt_row = cell_row + first - (cell_key[cell] & kKeyMask);
    const bool forward = (cut >> kNodeShift) >= num_nodes;
    list[kept_before[segment]] =
        PutSeg{(int32_t)(forward ? kjt_row : piece_row), (int32_t)(forward ? piece_row : kjt_row),
               (int32_t)rows, (int32_t)chunk};
  }
}

// One CTA: where each node's segments of each round start in its list,
// starts [nodes][rounds_cap + 1]: the segments kept before the first cut at
// or after the start of the node's piece (round rounds_cap: the node's end).
__global__ void __launch_bounds__(kThreads)
    put_starts_kernel(PutPieces pieces, const int64_t *sorted, const int32_t *kept_before,
                      int64_t segments, const int64_t *piece_key, const int64_t *cell_key,
                      int32_t *starts) {
  const int64_t nodes = 2 * pieces.num_nodes, rounds_cap = pieces.rounds_cap,
                num_features = pieces.num_features;
  for (int64_t index = threadIdx.x; index < nodes * (rounds_cap + 1); index += kThreads) {
    const int64_t node = index / (rounds_cap + 1), round = index % (rounds_cap + 1);
    const int64_t last_cell = node * num_features + num_features - 1;
    const int64_t key = round < rounds_cap
                            ? piece_key[node * rounds_cap + round]
                            : cell_key[last_cell] + pieces.keys(node, num_features - 1);
    starts[index] = kept_before[min(lower_bound(sorted, segments + 1, key), segments)];
  }
}

}  // namespace

PutLists put_lists(const at::Tensor &round_table, const at::Tensor &num_rounds,
                   const at::Tensor &cell_uniq, const at::Tensor &rail_cells,
                   const at::Tensor &fwd_pieces, const at::Tensor &fwd_rounds,
                   const at::Tensor &cells_in, const at::Tensor &cell_stage, const Sizes &sizes) {
  const int64_t num_nodes = sizes.topo.num_nodes, num_features = sizes.num_features,
                rounds_cap = sizes.rounds_cap;
  // The two lists as one of 2 * num_nodes nodes: the backward's owner nodes,
  // then the forward's proxy nodes.
  const int64_t nodes = 2 * num_nodes, num_pieces = rounds_cap * nodes,
                cells = nodes * num_features;
  const auto long_options = cell_uniq.options().dtype(at::kLong);
  const auto int_options = cell_uniq.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  const PutPieces pieces{round_table.data_ptr<int32_t>(),
                         fwd_pieces.data_ptr<int64_t>(),
                         num_rounds.data_ptr<int64_t>(),
                         fwd_rounds.data_ptr<int64_t>(),
                         cell_uniq.data_ptr<int64_t>(),
                         cells_in.data_ptr<int64_t>(),
                         num_nodes,
                         num_features,
                         rounds_cap,
                         sizes.chunk_rows};
  PutLists lists;
  // Each piece's IB chunks and its first chunk, pieces round-major; where
  // each piece and each cell starts.
  const at::Tensor chunk_before = at::empty({num_pieces + 1}, long_options);
  const at::Tensor piece_key = at::empty({num_pieces}, long_options);
  const at::Tensor cell_key = at::empty({cells}, long_options);
  lists.round_table = at::empty({rounds_cap, num_nodes, 4}, int_options);
  lists.reduce_task_starts = at::empty({rounds_cap + 1}, long_options);
  put_pieces_kernel<<<kOneCta, kThreads, 0, stream>>>(
      pieces, chunk_before.data_ptr<int64_t>(), piece_key.data_ptr<int64_t>(),
      cell_key.data_ptr<int64_t>(), lists.round_table.data_ptr<int32_t>(),
      lists.reduce_task_starts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  lists.num_chunks = chunk_before.select(0, num_pieces);
  // The cuts, by (node, key): the first key of each IB chunk (past the
  // chunks: after every node) and the end of each cell. Segment i goes from
  // cut i to cut i + 1, and holds rows when both are of one node.
  const int64_t chunk_cuts =
      chunk_bound(sizes.remote_keys, sizes) + chunk_bound(sizes.lookup_keys, sizes);
  const at::Tensor cut_keys = at::empty({chunk_cuts + cells}, long_options);
  cut_keys_kernel<<<item_grid(chunk_cuts + cells), kThreads, 0, stream>>>(
      pieces, chunk_before.data_ptr<int64_t>(), cell_key.data_ptr<int64_t>(), chunk_cuts,
      cut_keys.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const at::Tensor sorted = sort_keys(cut_keys, kNodeShift + bits_for(nodes + 1));
  // The segments that hold rows, in order, and where each (node, round)'s
  // first one is.
  const int64_t segments = sorted.numel() - 1;
  const at::Tensor kept = at::empty({segments}, long_options.dtype(at::kBool));
  kept_segments_kernel<<<item_grid(segments), kThreads, 0, stream>>>(
      sorted.data_ptr<int64_t>(), segments, kept.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const at::Tensor kept_before = exclusive_ends(kept, at::kInt);
  lists.list = at::empty({segments, 4}, int_options);
  put_segments_kernel<<<item_grid(segments), kThreads, 0, stream>>>(
      pieces, sorted.data_ptr<int64_t>(), kept_before.data_ptr<int32_t>(), segments,
      chunk_before.data_ptr<int64_t>(), piece_key.data_ptr<int64_t>(), cell_key.data_ptr<int64_t>(),
      rail_cells.data_ptr<int64_t>(), cell_stage.data_ptr<int64_t>(),
      (PutSeg *)lists.list.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const at::Tensor starts = at::empty({nodes, rounds_cap + 1}, int_options);
  put_starts_kernel<<<kOneCta, kThreads, 0, stream>>>(
      pieces, sorted.data_ptr<int64_t>(), kept_before.data_ptr<int32_t>(), segments,
      piece_key.data_ptr<int64_t>(), cell_key.data_ptr<int64_t>(), starts.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  lists.bwd_start = starts.narrow(0, 0, num_nodes);
  lists.fwd_start = starts.narrow(0, num_nodes, num_nodes);
  return lists;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
