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
#include "hier_output_dist/plan/proxy.h"

#include <cstdint>
#include <utility>

#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/plan/sort.h"

namespace dyn_emb {
namespace hier_output_dist {

ProxyKeys proxy_keys(const ProxyInput &input, const PlanSizes &sizes, bool node_dedup) {
  const int64_t entries = sizes.remote_keys, num_nodes = sizes.topo.num_nodes,
                num_features = sizes.num_features;
  const auto int_options = input.keys.options().dtype(at::kInt);
  ProxyKeys keys;
  const at::Tensor occurrence = at::arange(entries, int_options);
  keys.occurrence_count =
      at::where(input.skip, 0, input.requester_start.select(0, sizes.topo.local_size));
  keys.valid = occurrence.lt(keys.occurrence_count);
  // The identities (owner node, table, key): the occurrences sorted by key,
  // then stably by (owner node, table), the invalid ones last.
  at::Tensor owner_node, feature_at, new_identity;
  {
    const at::Tensor tag = at::where(keys.valid, input.cells.to(at::kInt), 0);
    owner_node = tag.div(num_features, "floor").clamp_max(num_nodes - 1);
    const at::Tensor feature = tag.remainder(num_features);
    auto [key_sorted, by_key] = sort_pairs(input.keys, occurrence, 64);
    const at::Tensor identity_tag =
        at::where(keys.valid, owner_node * 64 + input.tables.index_select(0, feature), 2048)
            .to(at::kShort);
    auto [tag_sorted, by_tag] = sort_pairs(identity_tag.index_select(0, by_key), occurrence, 12);
    keys.order = by_key.index_select(0, by_tag);
    feature_at = feature.index_select(0, keys.order);
    if (node_dedup) {
      const at::Tensor key_in_order = key_sorted.index_select(0, by_tag);
      new_identity = at::cat(
          {at::ones({1}, keys.valid.options()),
           key_in_order.narrow(0, 1, entries - 1)
               .ne(key_in_order.narrow(0, 0, entries - 1))
               .logical_or_(
                   tag_sorted.narrow(0, 1, entries - 1).ne(tag_sorted.narrow(0, 0, entries - 1)))});
    } else {
      new_identity = at::ones({entries}, keys.valid.options());  // each occurrence its own
    }
  }
  // the identity at each sorted position, and where each identity starts
  const at::Tensor identity_of = new_identity.cumsum(0, at::kInt) - 1;
  keys.num_keys = new_identity.logical_and(keys.valid).sum();  // the valid positions come first
  new_identity.reset();
  const at::Tensor identity_start =
      at::searchsorted(identity_of, at::arange(entries + 1, int_options), /*out_int32=*/true);
  // Per identity: its cell (owner node, smallest feature), its key, and the
  // sum of its weights, exact as integers. Sorted by (identity, feature), the
  // positions of an identity start with its smallest feature.
  at::Tensor identity_cell, identity_key, identity_weight;
  {
    const at::Tensor first_position = identity_start.narrow(0, 0, entries).clamp_max(entries - 1);
    const at::Tensor first_occurrence = keys.order.index_select(0, first_position);
    const at::Tensor smallest_feature =
        sort_pairs(identity_of.to(at::kLong).mul_(64).add_(feature_at), occurrence,
                   bits_for(entries * 64))
            .first.index_select(0, first_position)
            .bitwise_and(63);
    feature_at.reset();
    identity_cell =
        at::where(occurrence.lt(keys.num_keys),
                  owner_node.index_select(0, first_occurrence) * num_features + smallest_feature,
                  num_nodes * num_features)
            .to(at::kInt);
    owner_node.reset();
    identity_key = input.keys.index_select(0, first_occurrence);
    if (input.weights.defined() && sizes.record_bytes == 16) {
      const at::Tensor weight_ends = exclusive_ends(
          at::where(keys.valid, input.weights.index_select(0, keys.order).to(at::kLong), 0));
      identity_weight = weight_ends.index_select(0, identity_start.narrow(0, 1, entries)) -
                        weight_ends.index_select(0, identity_start.narrow(0, 0, entries));
    }
  }
  // The key index u: by cell, then in identity order (key order inside a cell).
  auto [cell_of_key, identity_of_key] =
      sort_pairs(identity_cell, occurrence, bits_for(num_nodes * num_features + 1));
  identity_cell.reset();
  keys.key_at = inverse_permutation(identity_of_key).index_select(0, identity_of);
  const at::Tensor cell_key_start =
      at::searchsorted(cell_of_key, at::arange(num_nodes * num_features + 1, int_options));
  keys.cell_uniq = cell_key_start.diff();
  keys.uniq_off =
      cell_key_start.slice(0, 0, num_nodes * num_features + 1, num_features).contiguous();
  keys.uniq_out = keys.uniq_off.diff();
  // per key
  keys.valid_key = occurrence.lt(keys.num_keys);
  keys.key_count = identity_start.diff().index_select(0, identity_of_key);
  keys.key_heavy = keys.key_count.gt(kHeavy).logical_and_(keys.valid_key);
  keys.key_first = identity_start.index_select(0, identity_of_key);
  keys.key_node = cell_of_key.div(num_features, "floor").clamp_max(num_nodes - 1);
  keys.key_uniq = occurrence - keys.uniq_off.to(at::kInt).index_select(0, keys.key_node);
  keys.key_value = identity_key.index_select(0, identity_of_key);
  if (sizes.record_bytes == 16)
    keys.key_weight = (identity_weight.defined() ? identity_weight.index_select(0, identity_of_key)
                                                 : keys.key_count)
                          .to(at::kFloat);
  return keys;
}

ProxyRounds proxy_rounds_and_slots(ProxyKeys &keys, const at::Tensor &requester_start,
                                   const ProxyWindows &windows, const PlanSizes &sizes) {
  const int64_t entries = sizes.remote_keys, num_nodes = sizes.topo.num_nodes,
                local_size = sizes.topo.local_size, rounds_cap = sizes.rounds_cap,
                proxy_rows = sizes.num_proxy_rows, partial_rows = sizes.partial_rows,
                record_bytes = sizes.record_bytes;
  const auto long_options = keys.order.options().dtype(at::kLong);
  const auto int_options = keys.order.options().dtype(at::kInt);
  ProxyRounds rounds;
  at::Tensor live;  // [rounds_cap][num_nodes]: the round and owner node have a piece
  {
    // The rows of each key: one for each occurrence, or a heavy key's row
    // and a partial of partial_rows for each requester. One round when every
    // row fits proxy_rows, else enough of 90% of a half.
    const at::Tensor row_ends = exclusive_ends(
        at::where(keys.valid_key,
                  at::where(keys.key_heavy, 1 + local_size * partial_rows, keys.key_count), 0));
    const at::Tensor total_rows = row_ends.select(0, entries);
    const int64_t room = (int64_t)(0.9 * (double)(proxy_rows / 2));
    rounds.num_rounds =
        at::where(total_rows.le(proxy_rows), 1, ceil_div(total_rows, room)).clamp(1, rounds_cap);
    // The pieces of each round and owner node: cuts of equal rows of the
    // owner node's keys, which never split a key; round after round in a
    // half, owner node after owner node.
    live = at::arange(rounds_cap, long_options)
               .view({rounds_cap, 1})
               .lt(rounds.num_rounds)
               .logical_and(at::arange(num_nodes, long_options).ne(sizes.topo.node));
    const at::Tensor first_key = keys.uniq_off.narrow(0, 0, num_nodes);
    const at::Tensor base_rows = row_ends.index_select(0, first_key);
    const at::Tensor node_rows =
        row_ends.index_select(0, keys.uniq_off.narrow(0, 1, num_nodes)) - base_rows;
    const at::Tensor round_index =
        at::arange(rounds_cap + 1, long_options).view({rounds_cap + 1, 1});
    const at::Tensor want =
        ceil_div(at::minimum(round_index, rounds.num_rounds) * node_rows, rounds.num_rounds);
    const at::Tensor cut = at::minimum((at::searchsorted(row_ends, (base_rows + want).view({-1}))
                                            .view({rounds_cap + 1, num_nodes}) -
                                        first_key)
                                           .clamp_min(0),
                                       keys.uniq_out);
    const at::Tensor piece_lo = at::where(live, cut.narrow(0, 0, rounds_cap), 0);
    const at::Tensor piece_hi = at::where(live, cut.narrow(0, 1, rounds_cap), 0);
    const at::Tensor piece_rows =
        row_ends.index_select(0, (first_key + piece_hi).view({-1})).view({rounds_cap, num_nodes}) -
        row_ends.index_select(0, (first_key + piece_lo).view({-1})).view({rounds_cap, num_nodes});
    const at::Tensor half = at::where(rounds.num_rounds.eq(1), 0, proxy_rows / 2);
    const at::Tensor piece_off =
        at::where(live,
                  at::arange(rounds_cap, long_options).view({rounds_cap, 1}).bitwise_and(1) * half +
                      piece_rows.cumsum(1) - piece_rows,
                  0);
    rounds.round_table =
        at::stack({piece_lo, piece_hi, piece_off, at::zeros_like(piece_lo)}, 2).to(at::kInt);
    // each key's round: the closed form of the cut
    rounds.key_round = (row_ends.narrow(0, 0, entries) - base_rows.index_select(0, keys.key_node))
                           .mul_(rounds.num_rounds)
                           .div_(node_rows.index_select(0, keys.key_node).clamp_min_(1), "floor")
                           .clamp_(0, rounds_cap - 1)
                           .to(at::kInt);
  }
  // key_stage: for each owner node but this one a slice, its round pieces
  // (16 bytes each), then its keys' records; what no key writes goes to the
  // last 16 bytes. The records' keys and weights go once written.
  {
    at::Tensor slice_bytes = rounds.num_rounds * 16 + keys.uniq_out * record_bytes;
    slice_bytes.narrow(0, sizes.topo.node, 1).zero_();
    rounds.stage_slice = slice_bytes.cumsum(0) - slice_bytes;
    const int64_t dump_word = windows.key_stage.numel() - 2;
    const at::Tensor record_word =
        at::where(keys.valid_key,
                  (rounds.stage_slice.index_select(0, keys.key_node) + rounds.num_rounds * 16)
                      .div_(8, "floor")
                      .add_(keys.key_uniq, record_bytes / 8),
                  dump_word);
    windows.key_stage.index_copy_(0, record_word, keys.key_value);
    if (record_bytes == 16)  // the weight's bits, then 4 zero bytes
      windows.key_stage.index_copy_(
          0, record_word + 1,
          keys.key_weight.view(at::kInt).to(at::kLong).bitwise_and_(0xFFFFFFFF));
    keys.key_value.reset();
    keys.key_weight.reset();
    const at::Tensor piece_word =
        at::where(live,
                  (rounds.stage_slice.view({1, num_nodes}) +
                   at::arange(rounds_cap, long_options).view({rounds_cap, 1}) * 16)
                      .div(4, "floor"),
                  2 * dump_word);
    windows.key_stage.view(at::kInt).index_copy_(
        0, (piece_word.unsqueeze(2) + at::arange(4, long_options)).view({-1}),
        rounds.round_table.view({-1}));
  }
  // the proxy header: the rounds, then the round table
  windows.proxy_hdr.narrow(0, 0, 1).copy_(rounds.num_rounds.view({1}));
  windows.proxy_hdr.narrow(0, 4, rounds_cap * num_nodes * 4).copy_(rounds.round_table.view({-1}));
  // Per key: its piece and its row. A piece's rows: its keys' rows, then
  // the duplicates' slots, then the partial rows of its heavy keys,
  // local_size blocks of partial_rows each.
  at::Tensor duplicate_base;
  {
    at::Tensor key_hi, key_off;
    {
      const at::Tensor piece_index = rounds.key_round * num_nodes + keys.key_node;
      auto piece_field = [&](int field) {
        return rounds.round_table.select(2, field).reshape({-1}).index_select(0, piece_index);
      };
      rounds.key_lo = piece_field(0);
      key_hi = piece_field(1);
      key_off = piece_field(2);
    }
    rounds.key_row = key_off + keys.key_uniq - rounds.key_lo;
    const at::Tensor node_first = keys.uniq_off.to(at::kInt).index_select(0, keys.key_node);
    const at::Tensor piece_first = (node_first + rounds.key_lo).clamp_max_(entries);
    const at::Tensor piece_last = (node_first + key_hi).clamp_max_(entries);
    const at::Tensor piece_end = (key_off + key_hi).sub_(rounds.key_lo);  // past its key rows
    key_hi.reset();
    key_off.reset();
    const at::Tensor duplicate_ends = exclusive_ends(
        at::where(keys.valid_key.logical_and(keys.key_heavy.logical_not()), keys.key_count - 1, 0),
        at::kInt);
    duplicate_base = piece_end + duplicate_ends.narrow(0, 0, entries) -
                     duplicate_ends.index_select(0, piece_first);
    const at::Tensor heavy_ends = exclusive_ends(keys.key_heavy, at::kInt);
    rounds.partial_base =
        piece_end + duplicate_ends.index_select(0, piece_last) -
        duplicate_ends.index_select(0, piece_first) +
        (heavy_ends.narrow(0, 0, entries) - heavy_ends.index_select(0, piece_first)) *
            (local_size * partial_rows);
  }
  // Each occurrence's slot: for a heavy key its requester's partial; else the
  // key's row for its first occurrence, a duplicate slot for the others.
  rounds.requester_of =
      (at::searchsorted(requester_start, at::arange(entries, long_options), /*out_int32=*/true,
                        /*right=*/true) -
       1)
          .clamp_(0, local_size - 1);
  const at::Tensor heavy_at = keys.key_heavy.index_select(0, keys.key_at);
  const at::Tensor row_at = rounds.key_row.index_select(0, keys.key_at);
  {
    const at::Tensor place_in_key =
        at::arange(entries, int_options) - keys.key_first.index_select(0, keys.key_at);
    at::Tensor slot = duplicate_base.index_select(0, keys.key_at).add_(place_in_key).sub_(1);
    slot = at::where(place_in_key.eq(0), row_at, slot);
    at::Tensor partial = rounds.requester_of.index_select(0, keys.order).mul_(partial_rows);
    partial.add_(rounds.partial_base.index_select(0, keys.key_at));
    rounds.slot_at = at::where(heavy_at, partial, slot);
  }
  duplicate_base.reset();
  const at::Tensor order = keys.order.to(at::kLong);
  windows.remote_map.select(0, 0).index_copy_(
      0, order, row_at.bitwise_or(heavy_at.to(at::kInt).bitwise_left_shift_(31)));
  windows.remote_map.select(0, 1).index_copy_(0, order, rounds.slot_at);
  return rounds;
}

ProxyPlan proxy_push_order(ProxyKeys keys, ProxyRounds rounds, const at::Tensor &requester_start,
                           const ProxyWindows &windows, const PlanSizes &sizes) {
  const int64_t entries = sizes.remote_keys, num_nodes = sizes.topo.num_nodes,
                local_size = sizes.topo.local_size, rounds_cap = sizes.rounds_cap;
  const auto long_options = keys.order.options().dtype(at::kLong);
  const auto int_options = keys.order.options().dtype(at::kInt);
  // The push order: one stable sort of the occurrences by their requester
  // (bits 56-58), round (36-55) and heavy bit (35), then a heavy key's index
  // or the IB chunk and owner node of a key with a row each (chunk << 5 |
  // owner node); the invalid ones after requester local_size - 1. What only
  // the push key needs goes before the sort.
  at::Tensor push_key_sorted, push_order;
  {
    at::Tensor push_key;
    {
      const at::Tensor key =
          at::empty({entries}, int_options).index_copy_(0, keys.order.to(at::kLong), keys.key_at);
      const at::Tensor heavy = keys.key_heavy.index_select(0, key);
      push_key = rounds.requester_of.to(at::kLong).bitwise_left_shift_(56);
      push_key.bitwise_or_(
          rounds.key_round.index_select(0, key).to(at::kLong).bitwise_left_shift_(36));
      push_key.bitwise_or_(heavy.to(at::kLong).bitwise_left_shift_(35));
      at::Tensor low = keys.key_uniq.index_select(0, key);
      low.sub_(rounds.key_lo.index_select(0, key)).div_(sizes.chunk_rows, "floor");
      low.bitwise_left_shift_(5).bitwise_or_(keys.key_node.index_select(0, key));
      push_key.bitwise_or_(at::where(heavy, key, low));
    }
    keys.key_at.reset();
    keys.key_uniq.reset();
    keys.key_node.reset();
    rounds.key_lo.reset();
    rounds.requester_of.reset();
    push_key = at::where(keys.valid, push_key, local_size << 56);
    std::tie(push_key_sorted, push_order) =
        sort_pairs(push_key, at::arange(entries, int_options), 60);
  }
  // The push tables, per requester and round: its first position, its first
  // heavy position, its first push-task flag, and its first heavy group (a
  // group: the positions of one requester, round and heavy key).
  const at::Tensor round_key = at::arange(local_size, long_options)
                                   .view({local_size, 1})
                                   .bitwise_left_shift(56)
                                   .bitwise_or(at::arange(rounds_cap + 1, long_options)
                                                   .view({1, rounds_cap + 1})
                                                   .bitwise_left_shift(36));
  const at::Tensor start_at =
      at::searchsorted(push_key_sorted, round_key.view({-1})).view({local_size, rounds_cap + 1});
  const at::Tensor requester_first = requester_start.narrow(0, 0, local_size).view({local_size, 1});
  const at::Tensor start = start_at - requester_first;
  const at::Tensor heavy_start =
      at::searchsorted(push_key_sorted, (round_key + (int64_t{1} << 35)).view({-1}))
          .view({local_size, rounds_cap + 1}) -
      requester_first;
  const at::Tensor groups_before = exclusive_ends(
      push_key_sorted.bitwise_right_shift(35)
          .bitwise_and_(1)
          .eq(1)
          .logical_and_(keys.valid)
          .logical_and_(at::cat({at::ones({1}, keys.valid.options()),
                                 push_key_sorted.narrow(0, 1, entries - 1)
                                     .ne(push_key_sorted.narrow(0, 0, entries - 1))})),
      at::kInt);
  keys.valid.reset();
  const at::Tensor groups_at =
      groups_before.index_select(0, start_at.view({-1})).view({local_size, rounds_cap + 1});
  const at::Tensor plain_rows = (heavy_start - start).narrow(1, 0, rounds_cap).to(at::kInt);
  // A round's flags: each requester's after the lower requesters', its tasks
  // of 32 rows first, then its tasks of 32 heavy groups.
  const at::Tensor tasks =
      ceil_div(plain_rows, kTaskRows) +
      ceil_div(groups_at.narrow(1, 1, rounds_cap) - groups_at.narrow(1, 0, rounds_cap), kTaskRows);
  const at::Tensor flags =
      at::cat({tasks.cumsum(0, at::kInt) - tasks, at::zeros({local_size, 1}, int_options)}, 1);
  ProxyPlan plan;
  plan.push_start = start.to(at::kInt);
  windows.proxy_hdr.narrow(0, 4 + rounds_cap * num_nodes * 4, local_size * (rounds_cap + 1) * 4)
      .copy_(at::stack({plan.push_start, heavy_start.to(at::kInt), flags,
                        groups_at - groups_at.narrow(1, 0, 1)},
                       2)
                 .view({-1}));
  // Each occurrence's place in its requester's push order (remote_map word
  // 2), and the flag of the push task that pushes it.
  at::Tensor occurrence_flag;
  {
    const at::Tensor push_order64 = push_order.to(at::kLong);
    push_order.reset();
    const at::Tensor requester =
        push_key_sorted.bitwise_right_shift(56).clamp_max_(local_size).to(at::kInt);
    const at::Tensor position =
        at::arange(entries, int_options) - requester_start.to(at::kInt).index_select(0, requester);
    windows.remote_map.select(0, 2).index_copy_(0, push_order64, position);
    const at::Tensor table =
        requester.clamp_max(local_size - 1)
            .mul_(rounds_cap + 1)
            .add_(push_key_sorted.bitwise_right_shift(36).bitwise_and_(0xFFFFF).clamp_max_(
                rounds_cap));
    occurrence_flag =
        at::empty({entries}, int_options)
            .index_copy_(0, push_order64,
                         (position - plan.push_start.view({-1}).index_select(0, table))
                             .div_(kTaskRows, "floor")
                             .add_(flags.view({-1}).index_select(0, table)));
  }
  // The reducer list: a key's entries, in occurrence order for a key with a
  // row each; for a heavy key its own row, then each requester's partial.
  // Each entry is a slot and the flag of the push task that fills it.
  plan.reducer_start = exclusive_ends(
      at::where(keys.valid_key, at::where(keys.key_heavy, 1 + local_size, keys.key_count), 0),
      at::kInt);
  const at::Tensor entry_key =
      (at::searchsorted(plan.reducer_start, at::arange(entries, int_options), /*out_int32=*/true,
                        /*right=*/true) -
       1)
          .clamp_(0, entries - 1);
  const at::Tensor place =
      at::arange(entries, int_options) - plan.reducer_start.index_select(0, entry_key);
  // a key with a row each: its place-th occurrence (sorted position)
  at::Tensor low, high;
  {
    const at::Tensor sorted_position =
        (keys.key_first.index_select(0, entry_key) + place).clamp_(0, entries - 1);
    low = rounds.slot_at.index_select(0, sorted_position);
    high = occurrence_flag.index_select(0, keys.order.index_select(0, sorted_position));
  }
  occurrence_flag.reset();
  rounds.slot_at.reset();
  keys.key_first.reset();
  keys.order.reset();
  // a heavy key: entry 0 is its own row, entry 1 + r requester r's partial,
  // with the flag of r's task of the key's group (none: r does not ask it)
  {
    const at::Tensor requester = (place - 1).clamp_(0, local_size - 1);
    const at::Tensor round = rounds.key_round.index_select(0, entry_key);
    rounds.key_round.reset();
    const at::Tensor table = requester * (rounds_cap + 1) + round;
    at::Tensor group_flag;
    {
      at::Tensor group_position, present;
      {
        const at::Tensor group_key = requester.to(at::kLong)
                                         .bitwise_left_shift_(56)
                                         .bitwise_or_(round.to(at::kLong).bitwise_left_shift_(36))
                                         .bitwise_or_(int64_t{1} << 35)
                                         .bitwise_or_(entry_key);
        group_position = at::searchsorted(push_key_sorted, group_key, /*out_int32=*/true)
                             .clamp_max_(entries - 1);
        present = push_key_sorted.index_select(0, group_position).eq(group_key);
      }
      push_key_sorted.reset();
      group_flag = at::where(
          present,
          (groups_before.index_select(0, group_position) -
           groups_at.view({-1}).index_select(0, table))
              .div_(kTaskRows, "floor")
              .add_(flags.view({-1}).index_select(0, table))
              .add_(ceil_div(plain_rows.view({-1}).index_select(
                                 0, requester * rounds_cap + round.clamp_max(rounds_cap - 1)),
                             kTaskRows)),
          kEntryNoPush);
    }
    const at::Tensor entry_heavy = keys.key_heavy.index_select(0, entry_key);
    const at::Tensor own_row = place.eq(0);
    high = at::where(entry_heavy, at::where(own_row, kEntryOwnRow, group_flag), high);
    group_flag.reset();
    low = at::where(
        entry_heavy,
        at::where(own_row, rounds.key_row.index_select(0, entry_key),
                  (rounds.partial_base.index_select(0, entry_key) + requester * sizes.partial_rows)
                      .bitwise_or_((int32_t)kEntryPartial)),
        low);
  }
  plan.reducer_entries = low.to(at::kLong)
                             .bitwise_and_(0xFFFFFFFF)
                             .bitwise_or_(high.to(at::kLong).bitwise_left_shift_(32));
  plan.num_rounds = rounds.num_rounds;
  plan.occurrence_count = keys.occurrence_count;
  plan.cell_uniq = keys.cell_uniq;
  plan.uniq_off = keys.uniq_off;
  plan.uniq_out = keys.uniq_out;
  plan.round_table = rounds.round_table;
  plan.stage_slice = rounds.stage_slice;
  return plan;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
