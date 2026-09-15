// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dynamicemb/exportable_embedding/indexer_snapshot.h"

#include <ATen/cuda/CUDAContext.h>

#include <atomic>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/library.h>

namespace dynamicemb::exportable_embedding {
namespace {

constexpr char kSnapshotMagic[8] = {'E', 'C', 'I', 'D', 'X', '0', '0', '1'};
constexpr uint32_t kSchemaVersion = 1;

#pragma pack(push, 1)
struct SnapshotHeader {
  char magic[8];
  uint32_t version;
  uint32_t kind;
  int64_t bucket_capacity;
  int64_t next_fused_key;
  uint64_t table_storage_size;
  uint64_t table_bucket_offsets_size;
  uint64_t miss_storage_indices_size;
  uint64_t valid_bases_size;
  uint64_t reserved_sizes_size;
};
#pragma pack(pop)

struct PendingSnapshot {
  std::shared_ptr<IndexerSnapshot> next;
  std::shared_ptr<IndexerSnapshot> retired;
  cudaEvent_t retirement_event{};
  std::atomic<bool> published{false};
};

struct RegistryEntry {
  std::shared_ptr<IndexerSnapshot> current;
  int64_t marker_value{0};
  std::atomic<PendingSnapshot*> pending{nullptr};
  std::mutex updates_mutex;
  std::unordered_map<int64_t, std::shared_ptr<PendingSnapshot>> updates;
};

std::mutex registry_mutex;
std::unordered_map<const void*, std::shared_ptr<RegistryEntry>> registry;
std::unordered_map<int64_t, std::shared_ptr<RegistryEntry>> registry_by_value;

struct CachedRegistryEntry {
  int64_t marker_value;
  std::weak_ptr<RegistryEntry> entry;
};

thread_local std::unordered_map<const void*, CachedRegistryEntry>
    registry_cache;

std::shared_ptr<RegistryEntry> find_registry_entry(const at::Tensor& marker,
                                                   int64_t marker_value) {
  const auto marker_pointer = marker.data_ptr();
  const auto cached = registry_cache.find(marker_pointer);
  if (cached != registry_cache.end() &&
      cached->second.marker_value == marker_value) {
    if (auto entry = cached->second.entry.lock()) {
      return entry;
    }
  }

  std::shared_ptr<RegistryEntry> entry;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto iterator = registry.find(marker_pointer);
    if (iterator != registry.end()) {
      entry = iterator->second;
    }
    if (!entry) {
      const auto value_iterator = registry_by_value.find(marker_value);
      if (value_iterator == registry_by_value.end()) {
        throw std::runtime_error("No embedding-collection indexer is bound");
      }
      entry = value_iterator->second;
    }
  }
  registry_cache[marker_pointer] = {marker_value, entry};
  return entry;
}

std::shared_ptr<RegistryEntry> find_registered_entry(
    const at::Tensor& marker) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  return registry.at(marker.data_ptr());
}

template <typename T>
at::Tensor read_tensor(std::ifstream& stream, uint64_t count,
                       at::ScalarType dtype, int device_index) {
  if (count == 0) {
    return torch::empty(
        {0}, torch::TensorOptions().dtype(dtype).device(torch::kCUDA,
                                                        device_index));
  }
  std::vector<T> values(count);
  stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
  auto cpu = torch::from_blob(values.data(), {static_cast<int64_t>(count)},
                              torch::TensorOptions().dtype(dtype))
                 .clone();
  return cpu.to(torch::Device(torch::kCUDA, device_index));
}

}  // namespace

bool register_from_tensors(
    const at::Tensor& marker, int64_t kind, const at::Tensor& table_storage,
    const at::Tensor& table_bucket_offsets, int64_t bucket_capacity,
    const at::Tensor& miss_storage_indices, const at::Tensor& valid_bases,
    const at::Tensor& reserved_sizes, int64_t next_fused_key) {
  auto snapshot = std::make_shared<IndexerSnapshot>();
  snapshot->kind = static_cast<IndexerSnapshotKind>(kind);
  snapshot->table_storage = table_storage;
  snapshot->table_bucket_offsets = table_bucket_offsets;
  snapshot->bucket_capacity = bucket_capacity;
  snapshot->miss_storage_indices = miss_storage_indices;
  snapshot->valid_bases = valid_bases;
  snapshot->reserved_sizes = reserved_sizes;
  snapshot->next_fused_key = next_fused_key;
  register_indexer_snapshot(marker, std::move(snapshot));
  return true;
}

bool unregister_from_tensor(const at::Tensor& marker) {
  unregister_indexer_snapshot(marker);
  return true;
}

std::shared_ptr<IndexerSnapshot> load_indexer_snapshot(
    const std::string& path, int device_index) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Cannot open indexer snapshot: " + path);
  }
  SnapshotHeader header{};
  stream.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (std::memcmp(header.magic, kSnapshotMagic, sizeof(kSnapshotMagic)) != 0 ||
      header.version != kSchemaVersion) {
    throw std::runtime_error("Unsupported indexer snapshot: " + path);
  }
  auto snapshot = std::make_shared<IndexerSnapshot>();
  snapshot->kind = static_cast<IndexerSnapshotKind>(header.kind);
  snapshot->bucket_capacity = header.bucket_capacity;
  snapshot->next_fused_key = header.next_fused_key;
  snapshot->table_storage = read_tensor<uint8_t>(
      stream, header.table_storage_size, at::kByte, device_index);
  snapshot->table_bucket_offsets = read_tensor<int64_t>(
      stream, header.table_bucket_offsets_size, at::kLong, device_index);
  snapshot->miss_storage_indices = read_tensor<int64_t>(
      stream, header.miss_storage_indices_size, at::kLong, device_index);
  snapshot->valid_bases = read_tensor<int64_t>(
      stream, header.valid_bases_size, at::kLong, device_index);
  snapshot->reserved_sizes = read_tensor<int64_t>(
      stream, header.reserved_sizes_size, at::kLong, device_index);
  return snapshot;
}

void register_indexer_snapshot(const at::Tensor& marker,
                               std::shared_ptr<IndexerSnapshot> snapshot) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto& entry = registry[marker.data_ptr()];
  if (!entry) {
    entry = std::make_shared<RegistryEntry>();
    entry->marker_value = marker.item<int64_t>();
  }
  std::atomic_store(&entry->current, std::move(snapshot));
  registry_by_value[entry->marker_value] = entry;
}

void unregister_indexer_snapshot(const at::Tensor& marker) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  const auto found = registry.find(marker.data_ptr());
  if (found == registry.end()) {
    return;
  }
  const auto entry = found->second;
  for (auto iterator = registry.begin(); iterator != registry.end();) {
    if (iterator->second == entry) {
      iterator = registry.erase(iterator);
    } else {
      ++iterator;
    }
  }
  const auto value_entry = registry_by_value.find(entry->marker_value);
  if (value_entry != registry_by_value.end() && value_entry->second == entry) {
    registry_by_value.erase(value_entry);
  }
}

void stage_indexer_snapshot(const at::Tensor& marker,
                            std::shared_ptr<IndexerSnapshot> snapshot,
                            int64_t snapshot_id) {
  auto entry = find_registered_entry(marker);
  auto pending = std::make_shared<PendingSnapshot>();
  pending->next = std::move(snapshot);
  if (cudaEventCreateWithFlags(&pending->retirement_event,
                               cudaEventDisableTiming) != cudaSuccess) {
    throw std::runtime_error("Failed to create indexer retirement event");
  }
  {
    std::lock_guard<std::mutex> lock(entry->updates_mutex);
    entry->updates[snapshot_id] = pending;
  }
  entry->pending.store(pending.get(), std::memory_order_release);
}

void wait_for_indexer_snapshot_retirement(const at::Tensor& marker,
                                          int64_t snapshot_id) {
  auto entry = find_registered_entry(marker);
  std::shared_ptr<PendingSnapshot> update;
  {
    std::lock_guard<std::mutex> lock(entry->updates_mutex);
    update = entry->updates.at(snapshot_id);
  }
  while (!update->published.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  cudaEventSynchronize(update->retirement_event);
  cudaEventDestroy(update->retirement_event);
  {
    std::lock_guard<std::mutex> lock(entry->updates_mutex);
    entry->updates.erase(snapshot_id);
  }
}

std::shared_ptr<IndexerSnapshot> find_indexer_snapshot(
    const at::Tensor& marker, int64_t marker_value) {
  auto entry = find_registry_entry(marker, marker_value);
  auto* pending = entry->pending.load(std::memory_order_acquire);
  if (pending != nullptr) {
    pending = entry->pending.exchange(nullptr, std::memory_order_acq_rel);
  }
  if (pending != nullptr) {
    auto next = pending->next;
    const auto stream =
        at::cuda::getCurrentCUDAStream(marker.get_device()).stream();
    if (cudaEventRecord(pending->retirement_event, stream) != cudaSuccess) {
      cudaEventDestroy(pending->retirement_event);
      throw std::runtime_error("Failed to record indexer retirement event");
    }
    auto retired = std::atomic_exchange(&entry->current, next);
    pending->retired = std::move(retired);
    pending->published.store(true, std::memory_order_release);
    return next;
  }
  return std::atomic_load(&entry->current);
}

}  // namespace dynamicemb::exportable_embedding

TORCH_LIBRARY_FRAGMENT(INFERENCE_EMB, m) {
  m.def("register_embedding_collection_indexer(Tensor marker, int kind, "
        "Tensor table_storage, Tensor table_bucket_offsets, int bucket_capacity, "
        "Tensor miss_storage_indices, Tensor valid_bases, Tensor reserved_sizes, "
        "int next_fused_key) -> bool");
  m.def("unregister_embedding_collection_indexer(Tensor marker) -> bool");
}

TORCH_LIBRARY_IMPL(INFERENCE_EMB, CompositeExplicitAutograd, m) {
  m.impl("register_embedding_collection_indexer",
         &dynamicemb::exportable_embedding::register_from_tensors);
  m.impl("unregister_embedding_collection_indexer",
         &dynamicemb::exportable_embedding::unregister_from_tensor);
}
