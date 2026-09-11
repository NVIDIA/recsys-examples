// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dynamicemb/exportable_embedding/indexer_directory.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstring>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <utility>

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

struct RegistryEntry {
  std::shared_ptr<IndexerSnapshot> current;
  int64_t marker_value{0};
};

std::mutex registry_mutex;
std::unordered_map<const void*, std::shared_ptr<RegistryEntry>> registry;
std::unordered_map<int64_t, std::shared_ptr<RegistryEntry>> registry_by_value;

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

std::string marker_fqn(const std::string& module_path) {
  return module_path + ".marker_tensor";
}

struct RetiredSnapshot {
  std::shared_ptr<IndexerSnapshot> snapshot;
  std::vector<cudaEvent_t> events;
};

std::string retirement_key(const std::string& collection_id,
                           int64_t snapshot_id) {
  return collection_id + "\n" + std::to_string(snapshot_id);
}

}  // namespace

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
  snapshot->kind = static_cast<IndexerKind>(header.kind);
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

std::shared_ptr<IndexerSnapshot> find_indexer_snapshot(
    const at::Tensor& marker, int64_t marker_value) {
  std::shared_ptr<RegistryEntry> entry;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto iterator = registry.find(marker.data_ptr());
    if (iterator != registry.end()) {
      entry = iterator->second;
    }
  }
  if (!entry) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto iterator = registry_by_value.find(marker_value);
    if (iterator == registry_by_value.end()) {
      throw std::runtime_error("No embedding-collection indexer is bound");
    }
    entry = iterator->second;
    registry[marker.data_ptr()] = entry;
  }
  return std::atomic_load(&entry->current);
}

bool register_from_tensors(
    const at::Tensor& marker, int64_t kind, const at::Tensor& table_storage,
    const at::Tensor& table_bucket_offsets, int64_t bucket_capacity,
    const at::Tensor& miss_storage_indices, const at::Tensor& valid_bases,
    const at::Tensor& reserved_sizes, int64_t next_fused_key) {
  auto snapshot = std::make_shared<IndexerSnapshot>();
  snapshot->kind = static_cast<IndexerKind>(kind);
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

struct EmbeddingCollectionIndexerDirectory::Impl {
  int device_index{0};
  std::string package_dir;
  std::unordered_map<std::string, EmbeddingCollectionBinding> bindings;
  std::unordered_map<std::string, at::Tensor> markers;
  std::unordered_map<std::string, std::shared_ptr<IndexerSnapshot>> snapshots;
  std::shared_mutex publication_mutex;
  std::mutex event_mutex;
  std::vector<cudaEvent_t> current_events;
  std::unordered_map<std::string, RetiredSnapshot> retired;
};

struct InferenceCall::Impl {
  EmbeddingCollectionIndexerDirectory::Impl* directory;
  std::shared_lock<std::shared_mutex> lock;
  bool recorded{false};

  explicit Impl(EmbeddingCollectionIndexerDirectory::Impl* value)
      : directory(value), lock(value->publication_mutex) {}
};

InferenceCall::InferenceCall(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
InferenceCall::InferenceCall(InferenceCall&&) noexcept = default;
InferenceCall& InferenceCall::operator=(InferenceCall&&) noexcept = default;
InferenceCall::~InferenceCall() = default;

void InferenceCall::record_complete(cudaStream_t stream) {
  if (!impl_ || impl_->recorded) {
    return;
  }
  cudaEvent_t event;
  if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess ||
      cudaEventRecord(event, stream) != cudaSuccess) {
    throw std::runtime_error("Failed to record indexer retirement event");
  }
  {
    std::lock_guard<std::mutex> lock(impl_->directory->event_mutex);
    impl_->directory->current_events.push_back(event);
  }
  impl_->recorded = true;
  impl_->lock.unlock();
}

EmbeddingCollectionIndexerDirectory::EmbeddingCollectionIndexerDirectory(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
EmbeddingCollectionIndexerDirectory::EmbeddingCollectionIndexerDirectory(
    EmbeddingCollectionIndexerDirectory&&) noexcept = default;
EmbeddingCollectionIndexerDirectory&
EmbeddingCollectionIndexerDirectory::operator=(
    EmbeddingCollectionIndexerDirectory&&) noexcept = default;

EmbeddingCollectionIndexerDirectory::~EmbeddingCollectionIndexerDirectory() {
  if (!impl_) {
    return;
  }
  for (const auto& item : impl_->markers) {
    unregister_indexer_snapshot(item.second);
  }
  for (auto& item : impl_->current_events) {
    cudaEventSynchronize(item);
    cudaEventDestroy(item);
  }
  for (auto& item : impl_->retired) {
    for (auto event : item.second.events) {
      cudaEventSynchronize(event);
      cudaEventDestroy(event);
    }
  }
}

std::unique_ptr<EmbeddingCollectionIndexerDirectory>
EmbeddingCollectionIndexerDirectory::load(const std::string& package_dir,
                                           int device_index) {
  auto impl = std::make_unique<Impl>();
  impl->device_index = device_index;
  impl->package_dir = package_dir;
  const std::string root = package_dir + "/embedding_collection_indexers";
  std::ifstream manifest_stream(root + "/manifest.json");
  if (!manifest_stream) {
    throw std::runtime_error("Cannot open embedding-collection indexer manifest");
  }
  const auto document = nlohmann::json::parse(manifest_stream);
  for (auto iterator = document.at("collections").begin();
       iterator != document.at("collections").end(); ++iterator) {
    const auto& value = iterator.value();
    EmbeddingCollectionBinding binding;
    binding.table_names = value.at("table_names").get<std::vector<std::string>>();
    binding.indexer_type = value.at("indexer_type").get<std::string>();
    binding.indexer_module_path =
        value.at("indexer_module_path").get<std::string>();
    binding.nve_layer_module_path =
        value.at("nve_layer_module_path").get<std::string>();
    if (!value.at("snapshot_path").is_null()) {
      binding.snapshot_path = value.at("snapshot_path").get<std::string>();
    }
    if (value.contains("feature_id_bits") &&
        !value.at("feature_id_bits").is_null()) {
      binding.feature_id_bits = value.at("feature_id_bits").get<int64_t>();
    }
    if (value.contains("marker_value") &&
        !value.at("marker_value").is_null()) {
      binding.marker_value = value.at("marker_value").get<int64_t>();
    }
    const std::string collection_id = iterator.key();
    impl->bindings.emplace(collection_id, binding);
    if (binding.snapshot_path.empty()) {
      continue;
    }
    auto marker = torch::tensor(
        {binding.marker_value},
        torch::TensorOptions().dtype(torch::kInt64).device(
            torch::kCUDA, device_index));
    auto snapshot = load_indexer_snapshot(
        root + "/" + binding.snapshot_path, device_index);
    register_indexer_snapshot(marker, snapshot);
    impl->markers.emplace(collection_id, std::move(marker));
    impl->snapshots.emplace(collection_id, std::move(snapshot));
  }
  return std::unique_ptr<EmbeddingCollectionIndexerDirectory>(
      new EmbeddingCollectionIndexerDirectory(std::move(impl)));
}

void EmbeddingCollectionIndexerDirectory::bind(
    torch::inductor::AOTIModelPackageLoader& loader) {
  const auto names = loader.get_constant_fqns();
  const std::unordered_map<std::string, bool> available = [&] {
    std::unordered_map<std::string, bool> result;
    for (const auto& name : names) result.emplace(name, true);
    return result;
  }();
  for (const auto& item : impl_->markers) {
    const std::string fqn = marker_fqn(
        impl_->bindings.at(item.first).indexer_module_path);
    if (available.find(fqn) == available.end()) {
      throw std::runtime_error("Indexer marker is not an AOTI constant: " + fqn);
    }
    std::unordered_map<std::string, at::Tensor> constants{{fqn, item.second}};
    loader.load_constants(constants, false, false, true);
    loader.load_constants(constants, true, false, true);
  }
}

InferenceCall EmbeddingCollectionIndexerDirectory::begin_inference() {
  return InferenceCall(std::make_unique<InferenceCall::Impl>(impl_.get()));
}

void EmbeddingCollectionIndexerDirectory::publish_snapshot(
    const std::string& collection_id, const std::string& snapshot_path,
    int64_t snapshot_id) {
  auto next = load_indexer_snapshot(snapshot_path, impl_->device_index);
  std::unique_lock<std::shared_mutex> publication(impl_->publication_mutex);
  auto old = impl_->snapshots.at(collection_id);
  register_indexer_snapshot(impl_->markers.at(collection_id), next);
  impl_->snapshots[collection_id] = std::move(next);
  std::vector<cudaEvent_t> events;
  {
    std::lock_guard<std::mutex> lock(impl_->event_mutex);
    events.swap(impl_->current_events);
  }
  impl_->retired[retirement_key(collection_id, snapshot_id)] =
      RetiredSnapshot{std::move(old), std::move(events)};
}

void EmbeddingCollectionIndexerDirectory::mark_update(
    const std::string& collection_id, int64_t snapshot_id) {
  std::unique_lock<std::shared_mutex> publication(impl_->publication_mutex);
  std::vector<cudaEvent_t> events;
  {
    std::lock_guard<std::mutex> lock(impl_->event_mutex);
    events.swap(impl_->current_events);
  }
  impl_->retired[retirement_key(collection_id, snapshot_id)] =
      RetiredSnapshot{nullptr, std::move(events)};
}

void EmbeddingCollectionIndexerDirectory::wait_for_retirement(
    const std::string& collection_id, int64_t snapshot_id) {
  auto iterator = impl_->retired.find(retirement_key(collection_id, snapshot_id));
  if (iterator == impl_->retired.end()) {
    return;
  }
  for (auto event : iterator->second.events) {
    cudaEventSynchronize(event);
    cudaEventDestroy(event);
  }
  impl_->retired.erase(iterator);
}

const EmbeddingCollectionBinding&
EmbeddingCollectionIndexerDirectory::binding(
    const std::string& collection_id) const {
  return impl_->bindings.at(collection_id);
}

const std::unordered_map<std::string, EmbeddingCollectionBinding>&
EmbeddingCollectionIndexerDirectory::bindings() const {
  return impl_->bindings;
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
