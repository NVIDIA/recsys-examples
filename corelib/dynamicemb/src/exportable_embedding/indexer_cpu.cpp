// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dynamicemb/exportable_embedding/indexer_snapshot.h"

#include <ATen/Parallel.h>
#include <torch/library.h>
#include <algorithm>
#include <optional>
#include <tuple>

namespace dynamicemb::exportable_embedding {
namespace {

void check_i64(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.device().is_cpu() && tensor.scalar_type() == at::kLong &&
                  tensor.dim() == 1 && tensor.is_contiguous(),
              "CPU indexer expects contiguous one-dimensional CPU int64 tensors");
}

// Same Murmur3 finalizer, bucket placement and byte layout as LinearBucketTable.
uint64_t hash_key(int64_t key) {
  auto h = static_cast<uint64_t>(key);
  h ^= h >> 33;
  h *= UINT64_C(0xff51afd7ed558ccd);
  h ^= h >> 33;
  h *= UINT64_C(0xc4ceb9fe1a85ec53);
  h ^= h >> 33;
  return h & INT64_MAX;
}

bool reserved_key(int64_t key) {
  return (static_cast<uint64_t>(key) & UINT64_C(0xfffffffffffffffc)) ==
         UINT64_C(0xfffffffffffffffc);
}

struct Buckets {
  at::Tensor storage;
  at::Tensor offsets;
  int64_t width;

  int64_t& key(int64_t slot) const {
    auto* base = storage.data_ptr<uint8_t>() + (slot / width) * 17 * width;
    return reinterpret_cast<int64_t*>(base)[slot % width];
  }
  int64_t& value(int64_t slot) const {
    auto* base = storage.data_ptr<uint8_t>() + (slot / width) * 17 * width;
    return reinterpret_cast<int64_t*>(base + 9 * width)[slot % width];
  }
  // Return an existing key, or an empty slot; -1 means the bucket is full.
  int64_t find(int64_t table, int64_t id) const {
    auto* bounds = offsets.data_ptr<int64_t>();
    const auto h = hash_key(id);
    const auto capacity = (bounds[table + 1] - bounds[table]) * width;
    const auto base = bounds[table] * width + (h % capacity / width) * width;
    // CUDA probes 16 digests at a time, checking each four-slot word for
    // matches before empties. Fill from that aligned start, so an earlier
    // empty word cannot hide a key from the CUDA reader.
    const auto start = (h % width / 16) * 16;
    int64_t empty = -1;
    for (int64_t step = 0; step < width; ++step) {
      auto slot = base + (start + step) % width;
      if (key(slot) == id) return slot;
      if (key(slot) == -1 && empty < 0) empty = slot;
    }
    return empty;
  }
  void put(int64_t slot, int64_t id, int64_t row) {
    key(slot) = id;
    value(slot) = row;
    storage.data_ptr<uint8_t>()[(slot / width) * 17 * width +
                               8 * width + slot % width] =
        static_cast<uint8_t>(hash_key(id) >> 32);
  }
};

Buckets empty_buckets(const at::Tensor& offsets, int64_t width) {
  TORCH_CHECK(width > 0 && width % 16 == 0, "bucket_capacity must be a multiple of 16");
  const auto count = offsets.data_ptr<int64_t>()[offsets.numel() - 1];
  Buckets result{at::zeros({count * width * 17}, offsets.options().dtype(at::kByte)),
                 offsets, width};
  for (int64_t slot = 0; slot < count * width; ++slot) result.put(slot, -1, 0);
  return result;
}

Buckets grow(const Buckets& old) {
  auto offsets = old.offsets * 2;
  for (;;) {
    auto next = empty_buckets(offsets, old.width);
    auto* bounds = old.offsets.data_ptr<int64_t>();
    bool fits = true;
    for (int64_t table = 0; table + 1 < old.offsets.numel() && fits; ++table) {
      for (auto slot = bounds[table] * old.width;
           slot < bounds[table + 1] * old.width; ++slot) {
        auto id = old.key(slot);
        if (id == -1) continue;
        auto target = next.find(table, id);
        if (target < 0) { fits = false; break; }
        next.put(target, id, old.value(slot));
      }
    }
    if (fits) return next;
    offsets = offsets * 2;
  }
}

Buckets checked_buckets(const at::Tensor& storage, const at::Tensor& offsets,
                        int64_t width) {
  check_i64(offsets);
  TORCH_CHECK(storage.device().is_cpu() && storage.scalar_type() == at::kByte &&
                  storage.is_contiguous() && offsets.numel() >= 2 &&
                  width > 0 && width % 16 == 0,
              "Invalid CPU hash snapshot layout");
  auto* bounds = offsets.data_ptr<int64_t>();
  TORCH_CHECK(bounds[0] == 0, "Bucket offsets must start at zero");
  for (int64_t i = 1; i < offsets.numel(); ++i)
    TORCH_CHECK(bounds[i] > bounds[i - 1], "Each table needs at least one bucket");
  TORCH_CHECK(storage.numel() == bounds[offsets.numel() - 1] * 17 * width,
              "Invalid CPU hash storage size");
  return {storage, offsets, width};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> create_cpu_hash(
    const at::Tensor& capacities, int64_t width) {
  check_i64(capacities);
  auto offsets = at::zeros({capacities.numel() + 1}, capacities.options());
  auto* bounds = offsets.data_ptr<int64_t>();
  TORCH_CHECK(width > 0 && width % 16 == 0, "Invalid bucket capacity");
  for (int64_t i = 0; i < capacities.numel(); ++i) {
    auto capacity = capacities.data_ptr<int64_t>()[i];
    TORCH_CHECK(capacity >= 0, "Table capacity must be nonnegative");
    bounds[i + 1] = bounds[i] + std::max<int64_t>(1, (2 * capacity + width - 1) / width);
  }
  auto buckets = empty_buckets(offsets, width);
  return {buckets.storage, offsets};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, int64_t> build_cpu_hash(
    const at::Tensor& storage, const at::Tensor& offsets, int64_t width,
    const at::Tensor& ids, const at::Tensor& tables, const at::Tensor& capacities,
    const at::Tensor& counts, int64_t next_row) {
  check_i64(ids); check_i64(tables); check_i64(capacities); check_i64(counts);
  auto buckets = checked_buckets(storage, offsets, width);
  TORCH_CHECK(ids.numel() == tables.numel() && capacities.numel() == counts.numel() &&
                  offsets.numel() == capacities.numel() + 1, "CPU hash build shape mismatch");
  auto rows = at::empty_like(ids);
  for (int64_t i = 0; i < ids.numel(); ++i) {
    auto table = tables.data_ptr<int64_t>()[i];
    auto id = ids.data_ptr<int64_t>()[i];
    TORCH_CHECK(table >= 0 && table < capacities.numel(), "Invalid source table ID");
    TORCH_CHECK(!reserved_key(id), "Source key is reserved by the hash indexer: ", id);
    auto slot = buckets.find(table, id);
    if (slot >= 0 && buckets.key(slot) == id) {
      rows.data_ptr<int64_t>()[i] = buckets.value(slot);
      continue;
    }
    TORCH_CHECK(counts.data_ptr<int64_t>()[table] < capacities.data_ptr<int64_t>()[table],
                "Source exceeds capacity of table ", table);
    while (slot < 0) {
      buckets = grow(buckets);
      slot = buckets.find(table, id);
    }
    buckets.put(slot, id, next_row);
    rows.data_ptr<int64_t>()[i] = next_row++;
    ++counts.data_ptr<int64_t>()[table];
  }
  return {buckets.storage, buckets.offsets, rows, next_row};
}

// Pack explicit storage rows when rebuilding a paused host indexer. Unlike
// initial loading, surviving keys keep their rows and deleted rows can be reused.
std::tuple<at::Tensor, at::Tensor> repack_cpu_hash(
    const at::Tensor& storage, const at::Tensor& offsets, int64_t width,
    const at::Tensor& ids, const at::Tensor& tables, const at::Tensor& rows) {
  check_i64(ids); check_i64(tables); check_i64(rows);
  auto buckets = checked_buckets(storage, offsets, width);
  TORCH_CHECK(ids.numel() == tables.numel() && ids.numel() == rows.numel(),
              "CPU hash repack shape mismatch");
  for (int64_t i = 0; i < ids.numel(); ++i) {
    const auto table = tables.data_ptr<int64_t>()[i];
    const auto id = ids.data_ptr<int64_t>()[i];
    TORCH_CHECK(table >= 0 && table + 1 < offsets.numel() && !reserved_key(id),
                "Invalid CPU hash repack key/table");
    TORCH_CHECK(rows.data_ptr<int64_t>()[i] >= 0, "Invalid storage row");
    auto slot = buckets.find(table, id);
    while (slot < 0) {
      buckets = grow(buckets);
      slot = buckets.find(table, id);
    }
    buckets.put(slot, id, rows.data_ptr<int64_t>()[i]);
  }
  return {buckets.storage, buckets.offsets};
}

at::Tensor linear_hash_lookup_cpu(
    const at::Tensor& storage, const at::Tensor& offsets, int64_t width,
    const at::Tensor& misses, const at::Tensor& ids, const at::Tensor& tables) {
  check_i64(ids); check_i64(tables); check_i64(misses);
  auto buckets = checked_buckets(storage, offsets, width);
  TORCH_CHECK(ids.numel() == tables.numel() && offsets.numel() == misses.numel() + 1,
              "CPU hash lookup shape mismatch");
  auto rows = at::empty_like(ids);
  at::parallel_for(0, ids.numel(), 1024, [&](int64_t begin, int64_t end) {
    for (auto i = begin; i < end; ++i) {
      auto table = tables.data_ptr<int64_t>()[i];
      auto id = ids.data_ptr<int64_t>()[i];
      TORCH_CHECK(table >= 0 && table < misses.numel(), "Invalid lookup table ID");
      auto slot = reserved_key(id) ? -1 : buckets.find(table, id);
      rows.data_ptr<int64_t>()[i] = slot >= 0 && buckets.key(slot) == id
          ? buckets.value(slot) : misses.data_ptr<int64_t>()[table];
    }
  });
  return rows;
}

at::Tensor embedding_collection_index_cpu(const at::Tensor& marker, int64_t marker_value,
                                          const at::Tensor& ids, const at::Tensor& tables) {
  check_i64(ids); check_i64(tables);
  auto snapshot = find_indexer_snapshot(marker, marker_value);
  if (snapshot->kind == IndexerSnapshotKind::LinearHashMap)
    return linear_hash_lookup_cpu(snapshot->table_storage, snapshot->table_bucket_offsets,
                                  snapshot->bucket_capacity, snapshot->miss_storage_indices,
                                  ids, tables);
  TORCH_CHECK(snapshot->kind == IndexerSnapshotKind::FusedIdentity, "Unsupported indexer");
  auto bases = at::index_select(snapshot->valid_bases, 0, tables);
  auto sizes = at::index_select(snapshot->reserved_sizes, 0, tables);
  return at::where((ids >= 0) & (ids < sizes), bases + ids, bases - 1);
}

at::Tensor expand_table_ids_cpu(at::Tensor offsets, at::Tensor indices,
                                std::optional<at::Tensor> feature_offsets,
                                int64_t num_tables, int64_t batch_size) {
  check_i64(offsets);
  TORCH_CHECK(indices.device().is_cpu() && indices.dim() == 1 && batch_size > 0,
              "Invalid CPU table expansion inputs");
  const int64_t* features = nullptr;
  if (feature_offsets && feature_offsets->numel() > 0) {
    check_i64(*feature_offsets);
    TORCH_CHECK(feature_offsets->numel() == num_tables + 1, "Feature offsets size mismatch");
    features = feature_offsets->data_ptr<int64_t>();
  } else {
    num_tables = (offsets.numel() - 1) / batch_size;
  }
  auto* bounds = offsets.data_ptr<int64_t>();
  auto result = at::empty({indices.numel()}, offsets.options());
  int64_t previous = 0;
  for (int64_t table = 0; table <= num_tables; ++table) {
    auto feature = features ? features[table] : table;
    TORCH_CHECK(feature >= 0 && feature * batch_size < offsets.numel(), "Invalid feature boundary");
    auto end = bounds[feature * batch_size];
    TORCH_CHECK(end >= previous && end <= indices.numel(), "Invalid table boundary");
    if (table == 0) TORCH_CHECK(end == 0, "Offsets must start at zero");
    if (end > previous) std::fill(result.data_ptr<int64_t>() + previous,
                                   result.data_ptr<int64_t>() + end, table - 1);
    previous = end;
  }
  TORCH_CHECK(previous == indices.numel(), "Offsets must cover all keys");
  return result;
}

at::Tensor linear_hash_lookup_meta(const at::Tensor&, const at::Tensor&, int64_t,
                                   const at::Tensor&, const at::Tensor& ids,
                                   const at::Tensor&) {
  return at::empty_like(ids, ids.options().dtype(at::kLong));
}
}  // namespace dynamicemb::exportable_embedding

TORCH_LIBRARY_FRAGMENT(INFERENCE_EMB, m) {
  m.def("create_cpu_hash(Tensor capacities, int bucket_capacity) -> (Tensor, Tensor)");
  m.def("build_cpu_hash(Tensor(a!) storage, Tensor(c) offsets, int bucket_capacity, "
        "Tensor ids, Tensor tables, Tensor capacities, Tensor(b!) counts, int next_row) "
        "-> (Tensor(a!), Tensor(c), Tensor, int)");
  m.def("repack_cpu_hash(Tensor(a!) storage, Tensor(b) offsets, int bucket_capacity, "
        "Tensor ids, Tensor tables, Tensor rows) -> (Tensor(a!), Tensor(b))");
  m.def("linear_hash_lookup(Tensor storage, Tensor offsets, int bucket_capacity, "
        "Tensor misses, Tensor ids, Tensor tables) -> Tensor");
}
TORCH_LIBRARY_IMPL(INFERENCE_EMB, CPU, m) {
  using namespace dynamicemb::exportable_embedding;
  m.impl("create_cpu_hash", &create_cpu_hash);
  m.impl("build_cpu_hash", &build_cpu_hash);
  m.impl("repack_cpu_hash", &repack_cpu_hash);
  m.impl("linear_hash_lookup", &linear_hash_lookup_cpu);
  m.impl("embedding_collection_index", &embedding_collection_index_cpu);
  m.impl("expand_table_ids", &expand_table_ids_cpu);
}
TORCH_LIBRARY_IMPL(INFERENCE_EMB, Meta, m) {
  m.impl("linear_hash_lookup", &dynamicemb::exportable_embedding::linear_hash_lookup_meta);
}
