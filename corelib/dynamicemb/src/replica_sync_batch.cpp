// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
#include "replica_sync_rows.h"

#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/library.h>

namespace dyn_emb::replica {
namespace {
using Tensors = std::vector<at::Tensor>;
using Group = c10::intrusive_ptr<c10d::ProcessGroup>;
using Work = c10::intrusive_ptr<c10d::Work>;
using Config =
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t>;
using Result =
    std::tuple<at::Tensor, Tensors, Tensors, Tensors, Tensors, Tensors>;

struct Gather {
  Work work;
  at::Tensor output;
  at::Tensor wait() {
    if (work)
      work->wait();
    return output;
  }
};

Gather gather(const at::Tensor &input, const Group &group) {
  auto out = at::empty({group->getSize(), input.numel()}, input.options());
  if (!input.numel())
    return {nullptr, out};
  auto flat = out.view(-1);
  auto send = input.view(-1);
  return {group->_allgather_base(flat, send), out};
}

// Build all routes using one count vector and a single ragged key exchange.
// Cache validation is one group vote, independent of the number of tables.
std::vector<Route> route_batch(const std::vector<Table> &tables,
                               const Tensors &cache, int64_t threshold,
                               const Group &group) {
  auto n = tables.size();
  std::vector<Route> routes(n);
  auto opt = tables.front().row_map.options();
  auto empty = at::empty({0}, opt);
  size_t valid_count = 0;
  if (!cache.empty()) {
    auto votes = at::zeros({int64_t(n)}, opt.dtype(at::kInt));
    for (size_t i = 0; i < n; ++i) {
      if (!cache[3 * i].numel())
        continue;
      auto vote = votes.select(0, i);
      const auto &t = tables[i];
      vote.copy_(t.sizes.sum().eq(cache[3 * i].numel()));
      validate(t.buckets, t.row_map, cache[3 * i], cache[3 * i + 1],
               cache[3 * i + 2], std::get<3>(t.layout), std::get<4>(t.layout),
               threshold, vote);
    }
    c10d::AllreduceOptions opts;
    opts.reduceOp = c10d::ReduceOp::MIN;
    Tensors args{votes};
    group->allreduce(args, opts)->wait();
    auto agreed = votes.cpu();
    for (size_t i = 0; i < n; ++i) {
      if (agreed.const_data_ptr<int>()[i]) {
        routes[i] = {cache[3 * i], cache[3 * i + 1], cache[3 * i + 2], true,
                     true};
        ++valid_count;
      }
    }
  }
  if (valid_count == n)
    return routes;
  auto counts = at::zeros({int64_t(n)}, opt);
  for (size_t i = 0; i < n; ++i)
    if (!routes[i].cached)
      counts.narrow(0, i, 1).copy_(count_touched(tables[i], threshold));

  auto lengths = gather(counts, group).wait().cpu().contiguous();
  auto lens = lengths.accessor<int64_t, 2>();
  int R = group->getSize(), me = group->getRank();
  std::vector<int64_t> offsets(R * (n + 1), 0);
  auto off = [&](int r, size_t i) -> int64_t & {
    return offsets[r * (n + 1) + i];
  };
  int64_t padded = 0;
  for (int r = 0; r < R; ++r) {
    for (size_t i = 0; i < n; ++i)
      off(r, i + 1) = off(r, i) + lens[r][i];
    padded = std::max(padded, off(r, n));
  }
  auto send = at::empty({padded}, opt);
  send.narrow(0, off(me, n), padded - off(me, n)).zero_();
  for (size_t i = 0; i < n; ++i)
    if (lens[me][i])
      export_keys(tables[i], threshold,
                  send.narrow(0, off(me, i), lens[me][i]));
  auto all_keys = gather(send, group).wait();
  auto local_held = at::zeros_like(counts);
  for (size_t i = 0; i < n; ++i) {
    if (routes[i].cached)
      continue;
    Tensors parts;
    for (int r = 0; r < R; ++r)
      if (lens[r][i])
        parts.push_back(all_keys.select(0, r).narrow(0, off(r, i), lens[r][i]));
    if (parts.empty()) {
      routes[i] = {empty, empty, empty};
      continue;
    }
    auto keys = std::get<0>(at::_unique2(at::cat(parts), true, false, false));
    routes[i] = lookup(tables[i], keys);
    local_held.select(0, i).copy_(routes[i].slots.ge(0).sum());
  }
  auto held = gather(local_held, group).wait().cpu().contiguous();
  auto holders = held.accessor<int64_t, 2>();
  for (size_t i = 0; i < n; ++i) {
    if (routes[i].cached)
      continue;
    bool dense = routes[i].keys.numel() > 0;
    for (int r = 0; r < R; ++r)
      dense &= holders[r][i] == routes[i].keys.numel();
    routes[i].dense = dense;
  }
  return routes;
}

struct Segment {
  size_t table;
  int64_t start, count, wire, padded, offset;
  at::Tensor rows, slots;
  std::optional<at::Tensor> indices;
};
using Slab = std::vector<Segment>;
int64_t align16(int64_t x) { return (x + 15) / 16 * 16; }

// Slabs can straddle table boundaries. Even thousands of small tables cost
// one NCCL work item per slab. Packets for each table keep its native dtype.
std::vector<Slab> partition(const std::vector<Table> &tables,
                            const std::vector<Route> &routes,
                            const std::vector<bool> &done, int64_t budget) {
  std::vector<Slab> slabs;
  int64_t used = 0;
  for (size_t i = 0; i < tables.size(); ++i) {
    if (done[i])
      continue;
    int64_t width = wire_width(tables[i]);
    for (int64_t start = 0; start < routes[i].keys.numel();) {
      if (slabs.empty() || budget - used < width) {
        slabs.emplace_back();
        used = 0;
      }
      int64_t count = std::min(routes[i].keys.numel() - start,
                               std::max<int64_t>(1, (budget - used) / width));
      // Sparse index offsets are int32 and relative to this segment.
      count = std::min<int64_t>(INT_MAX, count);
      slabs.back().push_back(
          {i, start, count, width, count, 0, {}, {}, std::nullopt});
      used += align16(count * width);
      start += count;
    }
  }
  return slabs;
}

struct Inserts {
  Tensors keys, values, scores, destinations;
};

Inserts prepare_inserts(const std::vector<Table> &tables,
                        const std::vector<Route> &routes) {
  Inserts pending;
  auto empty = at::empty({0}, tables.front().row_map.options());
  for (size_t i = 0; i < tables.size(); ++i) {
    const auto &r = routes[i];
    const auto &t = tables[i];
    auto missing_keys = empty;
    auto destinations = r.rows;
    if (!r.dense && r.keys.numel()) {
      auto missing = r.slots.lt(0);
      missing_keys = r.keys.masked_select(missing);
      // Missing row j is encoded as ~j, so one writeback can address either
      // an existing value row or this table's compact insertion buffer.
      destinations =
          at::where(missing, missing.cumsum(0, at::kLong).neg(), r.rows);
    }
    pending.destinations.push_back(destinations);
    pending.keys.push_back(missing_keys);
    pending.values.push_back(at::empty({missing_keys.numel(), t.values.size(1)},
                                       t.values.options()));
    pending.scores.push_back(at::empty(
        {missing_keys.numel(), std::get<4>(t.layout)}, r.keys.options()));
  }
  return pending;
}

// Share a mask collective across all sparse segments in the current slab.
// Prefix indices are discarded after merge; their memory cost is independent
// of the total number of tables and keys in a sync.
void layout_slab(Slab &slab, const std::vector<Route> &routes,
                 const Group &group, const Inserts &pending) {
  Tensors masks;
  for (auto &seg : slab) {
    const auto &r = routes[seg.table];
    seg.rows = pending.destinations[seg.table].narrow(0, seg.start, seg.count);
    seg.slots = r.slots.narrow(0, seg.start, seg.count);
    if (!r.dense)
      masks.push_back(seg.slots.ge(0).to(at::kByte));
  }
  at::Tensor counts;
  if (!masks.empty()) {
    auto across = gather(at::cat(masks), group).wait();
    Tensors maxima;
    int64_t offset = 0;
    for (auto &seg : slab) {
      if (routes[seg.table].dense)
        continue;
      auto mask = across.narrow(1, offset, seg.count);
      auto prefix = mask.cumsum(1, at::kInt);
      maxima.push_back(prefix.select(1, seg.count - 1).max());
      seg.indices = prefix.sub_(1).masked_fill_(mask.eq(0), -1);
      offset += seg.count;
    }
    counts = at::stack(maxima).cpu();
  }
  int64_t used = 0;
  size_t ci = 0;
  for (auto &seg : slab) {
    if (seg.indices)
      seg.padded = counts.const_data_ptr<int32_t>()[ci++];
    seg.offset = used;
    used += align16(seg.padded * seg.wire);
  }
}

void exchange_rows(const std::vector<Table> &tables,
                   const std::vector<Route> &routes, const Group &group,
                   std::vector<Slab> &slabs, const Inserts &insert) {
  Gather previous;
  Slab *prior = nullptr;
  auto finish = [&]() {
    if (!prior)
      return;
    auto output = previous.wait();
    for (auto &seg : *prior) {
      auto i = seg.table;
      auto packet = output.narrow(1, seg.offset, seg.padded * seg.wire);
      merge_into(tables[i], seg.rows, seg.slots, packet, seg.indices,
                 insert.values[i], insert.scores[i]);
      seg.indices.reset();
    }
  };
  for (auto &slab : slabs) {
    layout_slab(slab, routes, group, insert);
    auto &last = slab.back();
    auto send = at::empty({align16(last.offset + last.padded * last.wire)},
                          tables.front().buckets.options());
    for (const auto &seg : slab) {
      auto i = seg.table;
      auto rows = routes[i].rows.narrow(0, seg.start, seg.count);
      auto slots = seg.slots;
      if (seg.indices) {
        auto held = slots.ge(0);
        rows = rows.masked_select(held);
        slots = slots.masked_select(held);
      }
      auto bytes = seg.padded * seg.wire;
      pack_into(tables[i], rows, slots, seg.padded,
                send.narrow(0, seg.offset, bytes));
      if (align16(bytes) != bytes)
        send.narrow(0, seg.offset + bytes, align16(bytes) - bytes).zero_();
    }
    auto current = gather(send, group);
    finish();
    previous = std::move(current);
    prior = &slab;
  }
  finish();
}

Result sync_batch(const Tensors &values, const Tensors &buckets,
                  const Tensors &rows, const Tensors &sizes,
                  const Tensors &fresh, const std::vector<Config> &layouts,
                  const Tensors &cache, const Tensors &peer_state,
                  const std::string &group_name, int64_t threshold,
                  int64_t max_staging_bytes) {
  size_t n = values.size();
  TORCH_CHECK(n && buckets.size() == n && rows.size() == n &&
                  sizes.size() == n && fresh.size() == n &&
                  layouts.size() == n &&
                  (cache.empty() || cache.size() == 3 * n),
              "replica_sync_batch_: inconsistent table descriptors");
  TORCH_CHECK(max_staging_bytes > 0, "max_staging_bytes must be positive");
  c10::cuda::CUDAGuard guard(values.front().device());
  auto group = c10d::resolve_process_group(group_name);
  std::vector<Table> tables;
  for (size_t i = 0; i < n; ++i) {
    auto [emb, opt, bytes, cap, scores, constant, global] = layouts[i];
    TORCH_CHECK(values[i].device() == values.front().device() &&
                values[i].is_contiguous());
    TORCH_CHECK(rows[i].device() == values[i].device() && rows[i].dim() == 1 &&
                rows[i].is_contiguous() && rows[i].scalar_type() == at::kLong);
    TORCH_CHECK(sizes[i].device() == values[i].device() &&
                sizes[i].dim() == 1 && sizes[i].is_contiguous() &&
                sizes[i].scalar_type() == at::kInt && cap > 0 &&
                rows[i].numel() / cap == sizes[i].numel() &&
                rows[i].numel() % cap == 0);
    TORCH_CHECK((scores == 1 || scores == 2) &&
                buckets[i].numel() == rows[i].numel() * (9 + 8 * scores));
    TORCH_CHECK(fresh[i].device() == values[i].device() &&
                fresh[i].is_contiguous() &&
                fresh[i].scalar_type() == values[i].scalar_type() &&
                fresh[i].numel() == values[i].size(1));
    tables.push_back({values[i],
                      buckets[i],
                      rows[i],
                      sizes[i],
                      fresh[i],
                      {emb, opt, bytes, cap, scores},
                      constant,
                      global});
    wire_width(tables.back());
  }
  auto routes = route_batch(tables, cache, threshold, group);
  std::vector<bool> done(n, false);
  auto peers =
      batch_peer(tables, routes, peer_state, group, max_staging_bytes, done);
  auto slabs = partition(tables, routes, done,
                         max_staging_bytes / (2 * (group->getSize() + 1)));
  auto insert = prepare_inserts(tables, routes);
  exchange_rows(tables, routes, group, slabs, insert);
  auto stats = at::empty({int64_t(n), 2},
                         at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  auto hstats = stats.accessor<int64_t, 2>();
  Tensors next;
  for (size_t i = 0; i < n; ++i) {
    const auto &r = routes[i];
    hstats[i][0] = r.keys.numel();
    hstats[i][1] = r.keys.numel() - insert.keys[i].numel();
    if (r.dense)
      next.insert(next.end(), {r.keys, r.rows, r.slots});
    else
      next.insert(next.end(), 3, at::empty({0}, r.keys.options()));
  }
  return {stats, next, insert.keys, insert.values, insert.scores, peers};
}

TORCH_LIBRARY_FRAGMENT(dyn_emb, m) {
  m.def("replica_sync_batch_(Tensor(a!)[] tables, Tensor(b!)[] buckets, "
        "Tensor[] row_maps, Tensor[] sizes, "
        "Tensor[] fresh, (int, int, int, int, int, bool, int)[] layouts, "
        "Tensor[] cache, Tensor(c!)[] peer_state, "
        "str group_name, int threshold, int max_staging_bytes) -> (Tensor, "
        "Tensor[], Tensor[], Tensor[], Tensor[], Tensor[])");
}
TORCH_LIBRARY_IMPL(dyn_emb, CUDA, m) {
  m.impl("replica_sync_batch_", TORCH_FN(sync_batch));
}
} // namespace
} // namespace dyn_emb::replica
