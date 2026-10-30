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
#include "hier_output_dist/plan/requester.h"

#include <algorithm>
#include <cstdint>

namespace dyn_emb {
namespace hier_output_dist {

PeerWords remote_rows(const at::Tensor &row_cell, const at::Tensor &block_start,
                      const at::Tensor &destination_base, const PlanSizes &sizes) {
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

int64_t header_words(const PlanSizes &sizes) {
  return 1 + 4 * sizes.rounds_cap * sizes.topo.num_nodes + 4 * (sizes.rounds_cap + 1);
}

PeerWords header_gather(const PlanSizes &sizes, const at::TensorOptions &options) {
  const int64_t local_size = sizes.topo.local_size, words = header_words(sizes),
                table_words = 4 * sizes.rounds_cap * sizes.topo.num_nodes,
                push_words = 4 * (sizes.rounds_cap + 1);
  // the proxy header (proxy.h): its rounds at word 0, its round table from
  // word 4, then a row of push tables per requester
  const at::Tensor word = at::arange(words, options.dtype(at::kLong));
  const at::Tensor first_word = at::where(
      word.eq(0), 0,
      at::where(word.le(table_words), word + 3, word + (3 + sizes.topo.local_rank * push_words)));
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
                                 const PlanSizes &sizes) {
  const int64_t rows = row_cell.numel(), num_nodes = sizes.topo.num_nodes,
                local_size = sizes.topo.local_size, world_size = sizes.topo.world_size,
                num_features = sizes.num_features, rounds_cap = sizes.rounds_cap;
  const auto long_options = cell_start.options().dtype(at::kLong);
  const auto int_options = cell_start.options().dtype(at::kInt);
  RequesterTables tables;
  // Per proxy, from its header: its rounds, its round table, and this
  // requester's push tables there, made absolute: positions in push_order
  // (its rows at the proxies before come first), groups after its groups at
  // the proxies before.
  const at::Tensor block_size = block_start.diff().view({num_nodes, local_size});
  tables.push_order_off =
      exclusive_ends(block_size.sum({0}) - block_size.select(0, sizes.topo.node));
  tables.proxy_rounds = headers.select(1, 0).to(at::kLong).clamp(1, rounds_cap);
  tables.proxy_round_tables = headers.narrow(1, 1, rounds_cap * num_nodes * 4);
  const at::Tensor raw = headers.narrow(1, 1 + rounds_cap * num_nodes * 4, (rounds_cap + 1) * 4)
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

}  // namespace hier_output_dist
}  // namespace dyn_emb
