// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
#include "replica_peer.h"
#include "replica_sync_batch.h"
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

namespace dyn_emb::replica {
namespace {
using Group = c10::intrusive_ptr<c10d::ProcessGroup>;
namespace symmetric = c10d::symmetric_memory;
constexpr int64_t kSlab = 16 << 20;
int64_t rows_per_copy(int64_t count, int64_t width) {
  int64_t wave_bytes = std::min<int64_t>(
      192 << 20, std::max<int64_t>(8 << 20, count * width / 32));
  return std::max<int64_t>(1, wave_bytes / width);
}
// The op returns [mode (CPU), inbox, (route.rows, ordered_rows, peer_rows,
// ordered_slots, CPU wave ends and direction) for each table]. The tensors
// keep the allocation and permutations alive until the next call. An empty
// CPU tensor stands in for an inactive table. No process-global resources.
constexpr size_t kFields = 5;

bool agree(const Group &group, const at::Tensor &vote, bool local) {
  vote.fill_(local);
  std::vector<at::Tensor> inputs{vote};
  c10d::AllreduceOptions opts;
  opts.reduceOp = c10d::ReduceOp::MIN;
  group->allreduce(inputs, opts)->wait();
  return vote.item<int>() != 0;
}

int64_t row_bytes(const Table &t) {
  return t.values.size(1) * t.values.element_size();
}

// Destroy the old symmetric allocation before returning to NCCL or
// replacing it. The caller's tensor list also refers to this TensorImpl.
void release(const at::Tensor &old) {
  if (old.defined() && old.numel()) {
    at::cuda::getCurrentCUDAStream(old.get_device()).synchronize();
    old.set_();
  }
}

at::Tensor workspace(const Group &group, const at::Tensor &vote, at::Tensor old,
                     int64_t bytes) {
  if (old.defined() && old.numel() >= bytes)
    return old;
  release(old);
  at::Tensor inbox;
  try {
    inbox = symmetric::empty_strided_p2p({bytes}, {1}, at::kByte, vote.device(),
                                         std::nullopt, std::nullopt);
  } catch (const c10::Error &) {
    // Allocation is local. Decide collectively before either peer rendezvous.
  }
  if (!agree(group, vote, inbox.defined()))
    return {};
  return inbox;
}

// Build permutations in a single row-coordinate all-gather. Each table can
// have its own width and a completely different insertion order on each GPU.
bool plan(const Group &group, const std::vector<Table> &tables,
          const std::vector<Route> &routes, const at::Tensor &vote,
          std::vector<at::Tensor> &state) {
  std::vector<at::Tensor> ids;
  at::Tensor local, gathered;
  bool compact = true;
  try {
    for (const auto &r : routes) {
      if (!r.rows.defined() || !r.rows.numel())
        continue;
      ids.push_back(r.rows);
      auto bounds = at::aminmax(r.rows);
      compact &= (std::get<0>(bounds).eq(0) &
                  std::get<1>(bounds).eq(r.rows.numel() - 1))
                     .item<bool>();
    }
    if (compact) {
      local = ids.size() == 1 ? ids.front() : at::cat(ids);
      gathered = at::empty({2 * local.numel()}, local.options());
    }
  } catch (const c10::OutOfMemoryError &) {
    compact = false;
  }
  if (!agree(group, vote, compact))
    return false;
  group->_allgather_base(gathered, local)->wait();

  int64_t base = local.numel() * (1 - group->getRank());
  bool ready = true;
  try {
    for (size_t t = 0; t < tables.size(); ++t) {
      const Route &route = routes[t];
      if (!route.rows.defined() || !route.rows.numel())
        continue;
      const at::Tensor &rows = route.rows;
      int64_t count = rows.numel();
      auto remote = gathered.narrow(0, base, count);
      base += count;
      int64_t per_copy = rows_per_copy(count, row_bytes(tables[t]));
      auto here = at::floor_divide(rows, per_copy),
           there = at::floor_divide(remote, per_copy);
      int64_t last = (count - 1) / per_copy;
      bool reverse = (here + there - last)
                         .abs()
                         .sum()
                         .lt((here - there).abs().sum())
                         .item<bool>();
      if (reverse) {
        if (group->getRank() == 1)
          here = last - here;
        else
          there = last - there;
      }
      auto wave = at::maximum(here, there);
      auto order = at::argsort(wave * count + rows);
      size_t f = 2 + t * kFields;
      state[f] = rows;
      state[f + 1] = rows.index_select(0, order);
      state[f + 2] = remote.index_select(0, order);
      state[f + 3] = route.slots.index_select(0, order);
      auto ends = at::bincount(wave).cumsum(0).cpu();
      auto metadata = at::empty({1 + ends.numel()}, ends.options());
      metadata.data_ptr<int64_t>()[0] = reverse && group->getRank() == 1;
      metadata.slice(0, 1).copy_(ends);
      state[f + 4] = metadata;
    }
  } catch (const c10::OutOfMemoryError &) {
    ready = false;
  }
  return agree(group, vote, ready);
}

void direct(const Group &group, const std::vector<Table> &tables,
            const std::vector<Route> &routes,
            const std::vector<at::Tensor> &state) {
  auto arena = state[1];
  auto handle = symmetric::rendezvous(arena, group->getGroupName());
  auto peer =
      handle->get_buffer(1 - group->getRank(), {arena.numel()}, at::kByte, 0);
  int64_t base = 0;
  // Align each table's first row and reserve two progress words at its end.
  for (size_t t = 0; t < tables.size(); ++t) {
    if (!routes[t].rows.defined() || !routes[t].rows.numel())
      continue;
    int64_t bytes = routes[t].rows.numel() * row_bytes(tables[t]);
    arena.narrow(0, base + bytes, 8).zero_();
    base += bytes + 16;
  }
  handle->barrier(0, 0);
  auto merge = at::cuda::getStreamFromPool();
  base = 0;
  for (size_t t = 0; t < tables.size(); ++t) {
    if (!routes[t].rows.defined() || !routes[t].rows.numel())
      continue;
    const auto &table = tables[t];
    int64_t width = row_bytes(table),
            size = routes[t].rows.numel() * width + 16;
    size_t f = 2 + t * kFields;
    const auto &meta = state[f + 4];
    TORCH_CHECK(meta.device().is_cpu() && meta.scalar_type() == at::kLong &&
                meta.is_contiguous() && meta.dim() == 1 && meta.numel() > 1);
    auto *info = meta.const_data_ptr<int64_t>();
    peer_copy_merge_async(
        table.values, arena.narrow(0, base, size), peer.narrow(0, base, size),
        state[f + 1], state[f + 2], state[f + 3], table.buckets,
        at::IntArrayRef(info + 1, meta.numel() - 1),
        {std::get<3>(table.layout),
         rows_per_copy(routes[t].rows.numel(), width), info[0] != 0},
        merge);
    base += size;
  }
  // Both streams progress across table boundaries. Join them only once all
  // table writes have been enqueued, before releasing the shared inbox.
  at::cuda::CUDAEvent complete;
  complete.record(merge);
  complete.block(at::cuda::getCurrentCUDAStream());
  handle->barrier(1, 0);
}

struct Piece {
  size_t t;
  int64_t row, count, bytes;
};
void slabs(const Group &group, const std::vector<Table> &tables,
           const std::vector<Route> &routes, const at::Tensor &arena,
           int64_t slab) {
  auto handle = symmetric::rendezvous(arena, group->getGroupName());
  auto peer =
      handle->get_buffer(1 - group->getRank(), {arena.numel()}, at::kByte, 0);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto merge = at::cuda::getStreamFromPool();
  at::cuda::CUDAEvent sent, consumed[2];
  int64_t which = 0, used = 0;
  std::vector<Piece> batch;
  // Fill slabs across table boundaries so many small tables share barriers.
  auto drain = [&] {
    if (consumed[which].isCreated())
      consumed[which].block(stream);
    auto transfer = [&](const at::Tensor &mailbox, bool merging) {
      int64_t offset = which * slab;
      for (auto p : batch) {
        const auto &table = tables[p.t];
        const auto &r = routes[p.t];
        peer_transfer(table.values, mailbox.narrow(0, offset, p.bytes),
                      r.rows.narrow(0, p.row, p.count),
                      r.slots.narrow(0, p.row, p.count), table.buckets,
                      std::nullopt, {std::get<3>(table.layout), merging});
        offset += p.bytes;
      }
    };
    transfer(peer, false);
    handle->barrier(which, 0);
    sent.record(stream);
    sent.block(merge);
    {
      c10::cuda::CUDAStreamGuard guard(merge);
      transfer(arena, true);
      handle->barrier(2 + which, 0);
      consumed[which].record(merge);
    }
    which ^= 1;
    used = 0;
    batch.clear();
  };
  for (size_t t = 0; t < tables.size(); ++t) {
    if (!routes[t].rows.defined())
      continue;
    int64_t width = row_bytes(tables[t]), count = routes[t].rows.numel();
    for (int64_t row = 0; row < count;) {
      if (slab - used < width)
        drain();
      int64_t n = std::min((slab - used) / width, count - row);
      batch.push_back({t, row, n, n * width});
      used += n * width;
      row += n;
    }
  }
  if (!batch.empty())
    drain();
  consumed[which ^ 1].block(stream);
}
} // namespace

std::vector<at::Tensor> batch_peer(const std::vector<Table> &tables,
                                   const std::vector<Route> &routes,
                                   const std::vector<at::Tensor> &previous,
                                   const Group &group, int64_t budget,
                                   std::vector<bool> &done) {
  if (previous.empty())
    return {};
  const auto &mode = previous[0];
  TORCH_CHECK(mode.device().is_cpu() && mode.scalar_type() == at::kLong &&
              mode.numel() == 1);
  int64_t option = mode.item<int64_t>();
  if (!option || group->getSize() != 2 || tables.empty())
    return {mode};
  TORCH_CHECK(previous.size() == 1 || previous.size() == 2 ||
              previous.size() == 2 + tables.size() * kFields);
  auto old = previous.size() > 1 ? previous[1] : at::Tensor();
  auto device = tables[0].values.device();
  if (old.defined())
    TORCH_CHECK(old.device() == device && old.scalar_type() == at::kByte &&
                old.dim() == 1 && old.is_contiguous());
  c10::cuda::CUDAGuard guard(device);
  int64_t total = 0, widest = 0;
  bool eligible = true, cached = previous.size() == 2 + tables.size() * kFields;
  for (size_t t = 0; t < tables.size(); ++t) {
    const auto &table = tables[t];
    const auto &r = routes[t];
    if (!r.keys.defined() || !r.keys.numel())
      continue;
    const int64_t n = r.keys.numel(), width = row_bytes(table);
    auto [emb, opt, opt_bytes, capacity, scores] = table.layout;
    auto dtype = table.values.scalar_type();
    eligible &=
        r.dense && scores == 1 && width % 16 == 0 &&
        (!opt || opt_bytes == table.values.element_size()) &&
        width == emb * table.values.element_size() + opt * opt_bytes &&
        (dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16);
    TORCH_CHECK(width > 0 && n <= (INT64_MAX - total - 16) / width);
    total += n * width + 16;
    widest = std::max(widest, width);
    if (cached)
      cached &= r.cached && previous[2 + t * kFields].is_same(r.rows);
  }
  if (!eligible || total < (8LL << 20)) {
    release(old);
    return {mode};
  }
  auto vote = at::empty({}, tables[0].values.options().dtype(at::kInt));
  if (option == 2 && total <= budget) {
    std::vector<at::Tensor> state;
    // A cold/grown table invalidates the exchange plan on the whole group.
    // Previous route identity is sufficient on warm calls since route
    // validation already checked every slot and key on both peers.
    bool have_plan = agree(group, vote, cached);
    if (have_plan)
      state = previous;
    else {
      state.assign(2 + kFields * tables.size(),
                   at::empty({0}, at::TensorOptions().dtype(at::kLong).device(
                                      at::kCPU)));
      state[0] = mode;
      have_plan = plan(group, tables, routes, vote, state);
    }
    if (have_plan) {
      auto inbox = workspace(group, vote, old, total);
      if (inbox.defined()) {
        state[1] = inbox;
        direct(group, tables, routes, state);
        for (size_t t = 0; t < routes.size(); ++t)
          done[t] = true;
        return state;
      }
    }
  }
  int64_t slab = std::min<int64_t>(kSlab, budget / 2) / 16 * 16;
  if (widest > slab) {
    release(old);
    return {mode};
  }
  auto arena = workspace(group, vote, old, 2 * slab);
  if (!arena.defined())
    return {mode};
  slabs(group, tables, routes, arena, slab);
  for (size_t t = 0; t < routes.size(); ++t)
    done[t] = true;
  return {mode, arena};
}
} // namespace dyn_emb::replica
