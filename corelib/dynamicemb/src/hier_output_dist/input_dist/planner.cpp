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

#include "hier_output_dist/input_dist/planner.h"

#include <algorithm>
#include <cstdint>
#include <tuple>

#include "hier_output_dist/common/window.h"

namespace dyn_emb {
namespace hier_output_dist {

at::Tensor node_count_vector(const at::Tensor &cell_start, const Topo &topo, int64_t num_features) {
  const at::Tensor cell_size =
      cell_start.diff().view({topo.num_nodes, topo.local_size, num_features});
  const at::Tensor same_node = cell_size.select(0, topo.node);  // [owner][feature]
  const at::Tensor remote =
      cell_size.sum(at::IntArrayRef{0, 2}) - same_node.sum(at::IntArrayRef{1});  // [target]
  return at::cat({remote, same_node.reshape({-1})});
}

NodeLayout node_layout(const NodeCounts &counts, const at::Tensor &cell_start,
                       const at::Tensor &block_start, const Topo &topo, int64_t num_features,
                       int64_t lookup_keys, int64_t remote_keys) {
  const int64_t local_size = topo.local_size, num_nodes = topo.num_nodes,
                world_size = topo.world_size;
  const CountVector words{local_size, num_features};
  NodeLayout node;
  // the node's keys for each owner of the node and feature, and those of the
  // requesters before this one
  const at::Tensor own_counts =
      counts.totals.narrow(0, words.own(0, 0), local_size * num_features).view({local_size, -1});
  const at::Tensor own_before =
      counts.before.narrow(0, words.own(0, 0), local_size * num_features).view({local_size, -1});
  // this rank's own part, in its KJT order: [feature][requester]
  node.own_cells = counts.mine.narrow(1, 1, num_features).t().contiguous().view({-1});
  node.own_starts = node.own_cells.cumsum(0) - node.own_cells;
  node.own_total = own_counts.sum({1});
  // the needs of the node's parts, and whether it skips the call: the same on
  // every rank of the node
  node.need_lookup = node.own_total.max();
  node.skip = node.need_lookup.gt(lookup_keys);
  // A row of an owner of this node goes to the owner's own part: where this
  // rank's cell starts there (the features before, then the requesters
  // before), and the rows of the cell before it.
  const at::Tensor own_base = own_counts.cumsum(1) - own_counts + own_before -
                              cell_start.narrow(0, 0, world_size * num_features)
                                  .view({num_nodes, local_size, num_features})
                                  .select(0, topo.node);
  at::Tensor destination_base = own_base;  // one node: every owner is on it
  if (num_nodes > 1) {
    node.requester_start = exclusive_ends(counts.mine.select(1, 0));
    node.need_remote = counts.totals.narrow(0, words.remote(0), local_size).max();
    node.skip.logical_or_(node.need_remote.gt(remote_keys));
    // A remote row of owner (owner node, target) goes to its target's remote
    // part: after the requesters before this rank, this rank's blocks for the
    // owner nodes before, and the rows of its block before it.
    at::Tensor remote_blocks = block_start.diff().view({num_nodes, local_size});
    remote_blocks.select(0, topo.node).zero_();
    const at::Tensor remote_block_start = remote_blocks.cumsum(0) - remote_blocks;
    const at::Tensor remote_before_me = counts.before.narrow(0, words.remote(0), local_size);
    destination_base =
        (lookup_keys + remote_before_me.view({1, local_size, 1}) + remote_block_start.unsqueeze(2) -
         block_start.narrow(0, 0, world_size).view({num_nodes, local_size, 1}))
            .expand({num_nodes, local_size, num_features})
            .contiguous();
    destination_base.select(0, topo.node).copy_(own_base);
  }
  node.rows_to_push = at::where(node.skip, 0, block_start.select(0, world_size));
  // the rows past the valid ones (cell world_size * num_features) are never pushed
  node.destination_base =
      at::cat({destination_base.reshape({-1}), at::zeros({1}, destination_base.options())});
  return node;
}

OneNodeOwner one_node_owner(const NodeLayout &node, const at::Tensor &row_cell,
                            const at::Tensor &node_keys, const at::Tensor &node_weights,
                            const Topo &topo, int64_t num_features) {
  OneNodeOwner owner;
  // The own part is the KJT: one copy of the bound (wait() narrows it to
  // num_recv). The node's next call rewrites node_keys after its barrier,
  // which does not wait for this call's lookup.
  owner.kjt_values = node_keys.clone();
  if (node_weights.defined()) owner.kjt_weights = node_weights.clone();
  owner.kjt_lengths = node.own_cells;
  owner.kjt_offsets = exclusive_ends(owner.kjt_lengths);
  owner.num_recv = node.own_total.select(0, topo.local_rank);
  owner.length_per_key = owner.kjt_lengths.view({num_features, topo.local_size}).sum({1});
  owner.src = (node.destination_base.index_select(0, row_cell) +
               at::arange(row_cell.numel(), row_cell.options()))
                  .to(at::kInt);
  return owner;
}

// ── the rail and the owner ──

at::Tensor other_nodes_of(const Topo &topo, const at::TensorOptions &options) {
  const at::Tensor nodes = at::arange(topo.num_nodes - 1, options.dtype(at::kLong));
  return nodes + nodes.ge(topo.node);
}

at::Tensor count_message(const at::Tensor &need_remote, const at::Tensor &num_rounds,
                         const at::Tensor &own_total, const at::Tensor &uniq_out,
                         const at::Tensor &cell_uniq) {
  return at::cat({need_remote.view({1}), num_rounds.view({1}), own_total.view({1}), uniq_out,
                  cell_uniq.view({-1})});
}

RailLayout rail_layout(const at::Tensor &messages, const at::Tensor &own_cells,
                       const at::Tensor &other_nodes, const Sizes &sizes) {
  const int64_t num_nodes = sizes.topo.num_nodes, node = sizes.topo.node,
                local_size = sizes.topo.local_size, num_features = sizes.num_features,
                num_sources = sizes.topo.num_sources;
  const CountMessage message{num_nodes, num_features};
  RailLayout rail;
  // per source node: its rounds, its node-unique keys for each owner node,
  // and its keys of each feature for this owner (the part for this node)
  const at::Tensor source_rounds = messages.select(1, CountMessage::kRounds);
  // [source node][owner node]
  const at::Tensor uniq = messages.narrow(1, message.uniq_out(0), num_nodes);
  rail.cells_in = messages.narrow(1, message.common(), num_features);
  const at::Tensor recv =
      messages.select(1, CountMessage::kOwnTotal) + uniq.sum({0});  // [owner node]
  // a slice: the source's round pieces, then its records; none to its own node
  const at::Tensor slice_bytes =
      (source_rounds.view({num_nodes, 1}) * 16 + uniq * sizes.record_bytes).fill_diagonal_(0);
  // As proxy: where its slice starts at each owner.
  rail.put_dst = slice_bytes.narrow(0, 0, node).sum({0});
  rail.put_bytes = slice_bytes.select(0, node);
  rail.put_skip = recv.gt(sizes.lookup_keys);
  rail.put_skip.narrow(0, node, 1).fill_(true);
  // As owner: its KJT, cell (feature, source) feature-major, its node's keys
  // then the other nodes' in order; and where each other node's cells start.
  rail.kjt_lengths = at::cat({own_cells.view({num_features, local_size}),
                              rail.cells_in.index_select(0, other_nodes).t()},
                             1)
                         .reshape({-1});
  rail.kjt_offsets = exclusive_ends(rail.kjt_lengths);
  rail.num_recv = recv.select(0, node);
  rail.length_per_key = rail.kjt_lengths.view({num_features, num_sources}).sum({1});
  const at::Tensor into_me = slice_bytes.select(1, node);
  rail.slice_in = at::stack({into_me.cumsum(0) - into_me, source_rounds}, 1);
  rail.uniq_in = uniq.select(1, node).contiguous();
  const at::Tensor cell_offsets =
      rail.kjt_offsets.narrow(0, 0, num_features * num_sources).view({num_features, num_sources});
  rail.my_cell_starts = cell_offsets.narrow(1, 0, local_size).t().contiguous();
  rail.cell_stage =
      at::zeros({num_nodes, num_features}, messages.options())
          .index_copy_(0, other_nodes, cell_offsets.narrow(1, local_size, num_nodes - 1).t());
  rail.needs = at::stack(
      {recv.max(), messages.select(1, CountMessage::kNeedRemote).max(), source_rounds.max()});
  return rail;
}

at::Tensor status_publish(const RailLayout &rail) {
  return at::cat({rail.needs, rail.my_cell_starts.view({-1})});
}

at::Tensor agreed_needs(const at::Tensor &peer_needs) { return peer_needs.amax({0}); }

OwnerTables owner_tables(const RailLayout &rail, const at::Tensor &own_starts,
                         const at::Tensor &node_keys, const at::Tensor &node_weights,
                         const at::Tensor &rail_keys, const at::Tensor &other_nodes,
                         const Sizes &sizes) {
  const int64_t num_nodes = sizes.topo.num_nodes, local_size = sizes.topo.local_size,
                num_features = sizes.num_features, num_sources = sizes.topo.num_sources,
                rounds_cap = sizes.rounds_cap, lookup_keys = sizes.lookup_keys,
                record_bytes = sizes.record_bytes;
  const auto long_options = rail_keys.options().dtype(at::kLong);
  const auto int_options = rail_keys.options().dtype(at::kInt);
  OwnerTables tables;
  // each source node's slice in rail_keys: where it starts (bytes) and its
  // rounds (its own node sends none); its round pieces come first
  const at::Tensor slice_start = rail.slice_in.select(1, 0);
  tables.fwd_rounds = rail.slice_in.select(1, 1).clone();
  tables.fwd_rounds.narrow(0, sizes.topo.node, 1).zero_();
  {
    const at::Tensor rail_ints = rail_keys.view(at::kInt);
    const at::Tensor round = at::arange(rounds_cap, long_options).view({rounds_cap, 1, 1});
    const at::Tensor piece_word = (slice_start.div(4, "floor").view({1, num_nodes, 1}) + 4 * round +
                                   at::arange(3, long_options))
                                      .clamp_(0, rail_ints.numel() - 1);
    tables.fwd_pieces = at::where(round.lt(tables.fwd_rounds.view({1, num_nodes, 1})),
                                  rail_ints.take(piece_word).to(at::kLong), 0);
  }
  // The KJT's keys at positions [0, lookup_keys), in slices of 2^22 with int32
  // indices: its own node's cells from the own part of node_keys, the other
  // nodes' from their records in rail_keys (int64 words).
  constexpr int64_t kSlice = int64_t{1} << 22;
  const at::Tensor cell_offsets =
      rail.kjt_offsets.narrow(0, 0, num_features * num_sources).to(at::kInt);
  const at::Tensor own_cell_start = own_starts.to(at::kInt);
  const at::Tensor other_slice_start = slice_start.index_select(0, other_nodes);
  const at::Tensor first_record =
      (other_slice_start + tables.fwd_rounds.index_select(0, other_nodes) * 16).div_(8, "floor");
  // where each other node's keys of each feature start among its records,
  // [num_features][num_nodes - 1]
  const at::Tensor rail_lengths = rail.cells_in.index_select(0, other_nodes).t();
  const at::Tensor cell_first_key =
      (rail_lengths.cumsum(0) - rail_lengths).reshape({-1}).to(at::kInt);
  tables.kjt_values = at::empty({lookup_keys}, long_options);
  if (record_bytes == 16)
    tables.kjt_weights = at::empty({lookup_keys}, long_options.dtype(at::kFloat));
  for (int64_t first = 0; first < lookup_keys; first += kSlice) {
    const int64_t count = std::min(kSlice, lookup_keys - first);
    at::Tensor same, own_index, record;  // record: the int64 word of the key's record
    {
      const at::Tensor position = at::arange(first, first + count, int_options);
      const at::Tensor cell =
          (at::searchsorted(cell_offsets, position, /*out_int32=*/true, /*right=*/true) - 1)
              .clamp_(0, num_features * num_sources - 1);
      const at::Tensor in_cell = position - cell_offsets.index_select(0, cell);
      const at::Tensor source = cell.remainder(num_sources),
                       feature = cell.div(num_sources, "floor");
      same = source.lt(local_size);
      own_index =
          own_cell_start.index_select(0, feature * local_size + source.clamp_max(local_size - 1))
              .add_(in_cell)
              .clamp_(0, lookup_keys - 1);
      const at::Tensor column = (source - local_size).clamp_(0, num_nodes - 2);
      record = cell_first_key.index_select(0, feature * (num_nodes - 1) + column)
                   .add_(in_cell)
                   .to(at::kLong)
                   .mul_(record_bytes / 8)
                   .add_(first_record.index_select(0, column));
    }
    at::Tensor values = tables.kjt_values.narrow(0, first, count);
    at::where_out(values, same, node_keys.index_select(0, own_index),
                  rail_keys.index_select(0, record.clamp(0, rail_keys.numel() - 1)));
    if (record_bytes == 16) {  // the weight: the low half of the record's second word
      const at::Tensor weight_index = record.mul(2).add_(2).clamp_(0, 2 * rail_keys.numel() - 1);
      const at::Tensor record_weight = rail_keys.view(at::kFloat).index_select(0, weight_index);
      at::Tensor weights = tables.kjt_weights.narrow(0, first, count);
      at::where_out(weights, same,
                    node_weights.defined() ? node_weights.index_select(0, own_index)
                                           : at::ones_like(record_weight),
                    record_weight);
    }
  }
  return tables;
}

// ── the requester ──

PeerWords remote_rows(const at::Tensor &row_cell, const at::Tensor &block_start,
                      const at::Tensor &destination_base, const Sizes &sizes) {
  const int64_t rows = row_cell.numel(), local_size = sizes.topo.local_size,
                world_size = sizes.topo.world_size;
  const at::Tensor row = at::arange(rows, row_cell.options());
  const at::Tensor owner = row_cell.div(sizes.num_features, "floor").clamp_max(world_size - 1);
  const at::Tensor remote = owner.div(local_size, "floor")
                                .ne(sizes.topo.node)
                                .logical_and_(row.lt(block_start.select(0, world_size)));
  PeerWords gather;
  gather.peer = at::where(remote, owner.remainder(local_size), -1);
  gather.first_word = (destination_base.index_select(0, row_cell) + row - sizes.lookup_keys)
                          .clamp(0, sizes.remote_keys - 1);
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
  const int64_t rows = row_cell.numel(), num_nodes = sizes.topo.num_nodes,
                local_size = sizes.topo.local_size, world_size = sizes.topo.world_size,
                num_features = sizes.num_features, rounds_cap = sizes.rounds_cap;
  const auto long_options = cell_start.options().dtype(at::kLong);
  const auto int_options = cell_start.options().dtype(at::kInt);
  RequesterTables tables;
  // Per proxy, from its gathered header: its rounds, its round table, and
  // this requester's push tables there, made absolute: positions in
  // push_order (its rows at the proxies before come first), groups after its
  // groups at the proxies before.
  const ProxyHeader header = ProxyHeader::of(sizes);
  const at::Tensor block_size = block_start.diff().view({num_nodes, local_size});
  tables.push_order_off =
      exclusive_ends(block_size.sum({0}) - block_size.select(0, sizes.topo.node));
  tables.proxy_rounds =
      headers.select(1, ProxyHeader::kGatheredRounds).to(at::kLong).clamp(1, rounds_cap);
  tables.proxy_round_tables =
      headers.narrow(1, ProxyHeader::kGatheredTable, header.table_words()).contiguous();
  const at::Tensor raw = headers.narrow(1, header.gathered_push_table(), header.push_words())
                             .reshape({local_size, rounds_cap + 1, 4})
                             .to(at::kLong);
  const at::Tensor group_at = exclusive_ends(
      raw.select(2, 3).gather(1, tables.proxy_rounds.view({local_size, 1})).squeeze(1));
  const at::Tensor push_tables =
      raw + at::stack({tables.push_order_off.narrow(0, 0, local_size),
                       tables.push_order_off.narrow(0, 0, local_size),
                       at::zeros({local_size}, long_options), group_at.narrow(0, 0, local_size)},
                      1)
                .view({local_size, 1, 4});
  tables.push_tables = push_tables.to(at::kInt);
  // The task starts over all rounds: per round a slot per task of its
  // busiest proxy there, for every proxy. Past a proxy's rounds its entries
  // hold its end, so a round past every proxy's adds nothing.
  {
    const at::Tensor live_round = at::arange(rounds_cap, long_options)
                                      .view({1, rounds_cap})
                                      .lt(tables.proxy_rounds.view({local_size, 1}));
    const at::Tensor entry = push_tables.narrow(1, 0, rounds_cap),
                     next = push_tables.narrow(1, 1, rounds_cap);
    auto tasks = [&](const at::Tensor &first, const at::Tensor &end) {
      return at::where(live_round, ceil_div(end - first, kTaskRows), 0);
    };
    const at::Tensor row_tasks = tasks(entry.select(2, 0), entry.select(2, 1));
    const at::Tensor heavy_tasks = tasks(entry.select(2, 1), next.select(2, 0));
    const at::Tensor group_tasks = tasks(entry.select(2, 3), next.select(2, 3));
    tables.fwd_task_starts = exclusive_ends((row_tasks + heavy_tasks).amax({0}) * local_size);
    tables.bwd_task_starts = exclusive_ends(
        at::stack({group_tasks.amax({0}), row_tasks.amax({0})}, 1).view({-1}) * local_size);
  }
  const int64_t group_bound = std::min(rows, local_size * (sizes.remote_keys / (kHeavy + 1) + 1));
  const int64_t sub_bound = rows / kHeavy + group_bound + 1;
  if (rows == 0) {
    tables.src = tables.dst = tables.push_order = at::empty({0}, int_options);
    tables.heavy_groups = at::empty({0, 2}, int_options);
    tables.group_first_sub = at::empty({0}, int_options);
    tables.group_of_sub = at::empty({sub_bound}, int_options);
    tables.group_subs = at::zeros({}, long_options);
    return tables;
  }
  // Per row: a row of an owner on this node is in the owner's KJT, where
  // this requester's cell starts plus its place in the cell, both forward
  // and backward. Another node's row reads its key's row in proxy_rows, and
  // pushes its gradient to its slot, from its proxy's remote_map.
  const at::Tensor row = at::arange(rows, int_options);
  {
    const at::Tensor owner = row_cell.div(num_features, "floor").clamp_max(world_size - 1);
    const at::Tensor proxy = owner.remainder(local_size);
    const at::Tensor local = owner.div(local_size, "floor").eq(sizes.topo.node);
    const at::Tensor local_src = (owner_cell_start.view({-1}).index_select(
                                      0, proxy * num_features + row_cell.remainder(num_features)) +
                                  (row - cell_start.index_select(0, row_cell)))
                                     .to(at::kInt);
    tables.src = at::where(local, local_src, words.select(0, 0).bitwise_and(0x7FFFFFFF));
    tables.dst = at::where(local, local_src, words.select(0, 1));
    // The push order: its rows at each proxy, at their push positions there.
    tables.push_order =
        at::empty({rows + 1}, int_options)
            .index_copy_(
                0,
                at::where(
                    row.lt(block_start.select(0, world_size)).logical_and_(local.logical_not()),
                    tables.push_order_off.index_select(0, proxy) + words.select(0, 2), rows)
                    .clamp_(0, rows),
                row)
            .narrow(0, 0, rows);
  }
  // The heavy groups, in its push order: a group is a run of positions of
  // heavy rows with one slot (dst) inside one (proxy, round) heavy range.
  // Different heavy keys of a range have different slots; rounds t and t + 2
  // share a half of proxy_rows, so their keys' slots may be equal.
  const at::Tensor position = row;
  const at::Tensor in_order = position.lt(tables.push_order_off.select(0, local_size));
  at::Tensor group_starts, run_end;
  {
    const at::Tensor row_at =
        tables.push_order.index_select(0, at::where(in_order, position, 0)).clamp(0, rows - 1);
    const at::Tensor heavy_at =
        words.select(0, 0).lt(0).index_select(0, row_at).logical_and_(in_order);
    const at::Tensor dst_at = tables.dst.index_select(0, row_at);
    // the absolute heavy starts are non-decreasing over (proxy, round): an
    // entry past a proxy's rounds holds its end
    const at::Tensor heavy_range =
        at::searchsorted(push_tables.select(2, 1).reshape({-1}).to(at::kInt), position,
                         /*out_int32=*/true, /*right=*/true);
    at::Tensor change = at::ones({rows}, in_order.options());
    change.narrow(0, 1, rows - 1)
        .copy_(
            dst_at.narrow(0, 1, rows - 1)
                .ne(dst_at.narrow(0, 0, rows - 1))
                .logical_or_(
                    heavy_range.narrow(0, 1, rows - 1).ne(heavy_range.narrow(0, 0, rows - 1)))
                .logical_or_(heavy_at.narrow(0, 1, rows - 1).ne(heavy_at.narrow(0, 0, rows - 1))));
    group_starts = heavy_at.logical_and(change);
    // where the run of each position ends: the next run's start, or rows;
    // runs_so_far is the number of runs started at or before each position
    const at::Tensor runs_so_far = change.logical_or(in_order.logical_not()).cumsum(0, at::kInt);
    run_end = at::searchsorted(runs_so_far, runs_so_far, /*out_int32=*/true, /*right=*/true);
  }
  tables.heavy_groups =
      at::empty({group_bound + 1, 2}, int_options)
          .index_copy_(0,
                       at::where(group_starts, group_starts.cumsum(0, at::kInt) - 1, group_bound)
                           .clamp_(0, group_bound)
                           .to(at::kLong),
                       at::stack({position, run_end}, 1))
          .narrow(0, 0, group_bound);
  // Each group's sub-tasks of kHeavy positions, numbered in group order.
  const at::Tensor subs = at::where(
      at::arange(group_bound, long_options).lt(group_starts.sum()),
      ceil_div(tables.heavy_groups.select(1, 1).to(at::kLong) - tables.heavy_groups.select(1, 0),
               kHeavy),
      0);
  const at::Tensor sub_ends = subs.cumsum(0);
  tables.group_first_sub = (sub_ends - subs).to(at::kInt);
  tables.group_subs = subs.sum();
  tables.group_of_sub = at::searchsorted(sub_ends, at::arange(sub_bound, long_options),
                                         /*out_int32=*/true, /*right=*/true)
                            .clamp_max(group_bound - 1);
  return tables;
}

// ── the put lists ──

namespace {

// A cut's sort key: its node, then its key (below 2^31).
constexpr int kNodeShift = 32;
constexpr int64_t kKeyMask = (int64_t{1} << kNodeShift) - 1;

}  // namespace

PutLists put_lists(const at::Tensor &round_table, const at::Tensor &num_rounds,
                   const at::Tensor &cell_uniq, const at::Tensor &rail_cells,
                   const at::Tensor &fwd_pieces, const at::Tensor &fwd_rounds,
                   const at::Tensor &cells_in, const at::Tensor &cell_stage, const Sizes &sizes) {
  const int64_t num_nodes = sizes.topo.num_nodes, num_features = sizes.num_features,
                rounds_cap = sizes.rounds_cap, chunk_rows = sizes.chunk_rows;
  // The two lists as one of 2 * num_nodes nodes: the backward's owner nodes,
  // then the forward's proxy nodes. (A call that does not fit may read pieces
  // no one wrote: they are clamped.)
  const int64_t nodes = 2 * num_nodes, num_pieces = rounds_cap * nodes;
  const auto long_options = cell_uniq.options().dtype(at::kLong);
  const at::Tensor pieces =  // [rounds_cap][nodes]: lo, hi, the row of lo in proxy_rows
      at::cat({round_table.narrow(2, 0, 3).to(at::kLong), fwd_pieces}, 1).clamp_(0, kKeyMask >> 1);
  const at::Tensor lo = pieces.select(2, 0);
  const at::Tensor rounds = at::cat({num_rounds.expand({num_nodes}), fwd_rounds});
  const at::Tensor count = at::cat({cell_uniq.view({num_nodes, num_features}), cells_in});
  const at::Tensor node_base =
      at::arange(nodes, long_options).bitwise_left_shift_(kNodeShift).view({nodes, 1});
  PutLists lists;
  // Each piece's IB chunks and its first chunk, pieces round-major.
  const at::Tensor piece_chunks = ceil_div((pieces.select(2, 1) - lo).clamp_min_(0), chunk_rows);
  const at::Tensor chunk_ends = piece_chunks.view({-1}).cumsum(0);
  const at::Tensor first_chunk = chunk_ends.view({rounds_cap, nodes}) - piece_chunks;
  lists.num_chunks = chunk_ends.select(0, num_pieces - 1);
  // The cuts, by (node, key): the first key of each IB chunk (past the
  // chunks: after every node) and the end of each cell. Segment i goes from
  // cut i to cut i + 1, and holds rows when both are of one node.
  const at::Tensor chunk = at::arange(
      chunk_bound(sizes.remote_keys, sizes) + chunk_bound(sizes.lookup_keys, sizes), long_options);
  const at::Tensor piece = at::searchsorted(chunk_ends, chunk, /*out_int32=*/false, /*right=*/true)
                               .clamp_max_(num_pieces - 1);
  const at::Tensor piece_base =
      (node_base.view({1, nodes}) + lo - first_chunk * chunk_rows).view({-1});
  const at::Tensor chunk_cut =
      at::where(chunk.lt(lists.num_chunks), piece_base.index_select(0, piece) + chunk * chunk_rows,
                nodes << kNodeShift);
  const at::Tensor cell_ends = count.cumsum(1);
  const at::Tensor cell_lo = cell_ends - count;
  const at::Tensor sorted =
      std::get<0>(at::cat({chunk_cut, (node_base + cell_ends).view({-1})}).sort());
  const int64_t cuts = sorted.numel() - 1;
  const at::Tensor key = sorted.narrow(0, 0, cuts), next = sorted.narrow(0, 1, cuts);
  const at::Tensor node = key.bitwise_right_shift(kNodeShift), first = key.bitwise_and(kKeyMask);
  const at::Tensor rows = at::where(next.bitwise_right_shift(kNodeShift).eq(node),
                                    next.bitwise_and(kKeyMask) - first, 0);
  // Each segment's piece and cell: the last of its node's that starts at or
  // before it (a node's pieces past its rounds start at its end).
  const at::Tensor totals = cell_ends.select(1, num_features - 1);
  const at::Tensor piece_start =
      at::where(at::arange(rounds_cap, long_options).view({rounds_cap, 1}).lt(rounds), lo, totals);
  const at::Tensor at_piece =  // node-major: lo, its row, the first chunk
      at::stack({lo, pieces.select(2, 2), first_chunk}, 2)
          .transpose(0, 1)
          .reshape({-1, 3})
          .index_select(0, (at::searchsorted((node_base + piece_start.t()).reshape({-1}), key,
                                             /*out_int32=*/false, /*right=*/true) -
                            1)
                               .clamp_(0, num_pieces - 1));
  const at::Tensor in_piece = first - at_piece.select(1, 0);
  const at::Tensor piece_row = at_piece.select(1, 1) + in_piece;
  const at::Tensor seg_chunk = at_piece.select(1, 2) + in_piece.div(chunk_rows, "floor");
  const at::Tensor at_cell =  // the cell's first key and its row in the owner's KJT
      at::stack({cell_lo, at::cat({rail_cells, cell_stage})}, 2)
          .view({-1, 2})
          .index_select(0, (at::searchsorted((node_base + cell_lo).view({-1}), key,
                                             /*out_int32=*/false, /*right=*/true) -
                            1)
                               .clamp_(0, nodes * num_features - 1));
  const at::Tensor kjt_row = at_cell.select(1, 1) + first - at_cell.select(1, 0);
  const at::Tensor forward = node.ge(num_nodes);
  // The segments that hold rows, in order, and where each (node, round)'s
  // first one is.
  const at::Tensor kept = rows.gt(0);
  const at::Tensor kept_before = exclusive_ends(kept);
  lists.list = at::empty({cuts + 1, 4}, long_options.dtype(at::kInt))
                   .index_copy_(0, at::where(kept, kept_before.narrow(0, 0, cuts), cuts),
                                at::stack({at::where(forward, kjt_row, piece_row),
                                           at::where(forward, piece_row, kjt_row), rows, seg_chunk},
                                          1)
                                    .to(at::kInt))
                   .narrow(0, 0, cuts);
  const at::Tensor starts =
      kept_before
          .index_select(
              0, at::searchsorted(
                     sorted,
                     (node_base + at::cat({piece_start.t(), totals.unsqueeze(1)}, 1)).view({-1}))
                     .clamp_max_(cuts))
          .view({nodes, rounds_cap + 1})
          .to(at::kInt);
  lists.bwd_start = starts.narrow(0, 0, num_nodes);
  lists.fwd_start = starts.narrow(0, num_nodes, num_nodes);
  // a round's reducer tasks: its busiest owner node's chunks, for every owner
  // node, chunk_rows / kTaskRows tasks a chunk
  lists.reduce_task_starts = exclusive_ends(piece_chunks.narrow(1, 0, num_nodes).amax({1}) *
                                            (num_nodes * (chunk_rows / kTaskRows)));
  lists.round_table = round_table.clone();
  lists.round_table.select(2, 3).copy_(first_chunk.narrow(1, 0, num_nodes));
  return lists;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
