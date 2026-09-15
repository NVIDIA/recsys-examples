// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dynamicemb/exportable_embedding/indexer_directory.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace dynamicemb::exportable_embedding {
namespace {

std::string marker_fqn(const std::string& module_path) {
  return module_path + ".marker_tensor";
}

bool has_replaceable_state(const std::string& indexer_type) {
  return indexer_type == "linear_hash_map" ||
         indexer_type == "fused_identity";
}

std::string retirement_key(const std::string& collection_id,
                           int64_t snapshot_id) {
  return collection_id + "\n" + std::to_string(snapshot_id);
}

}  // namespace

struct EmbeddingCollectionIndexerDirectory::Impl {
  int device_index{0};
  std::string package_dir;
  std::unordered_map<std::string, EmbeddingCollectionBinding> bindings;
  std::unordered_map<std::string, at::Tensor> markers;
  std::unordered_map<std::string, std::shared_ptr<IndexerSnapshot>> snapshots;
  std::unordered_set<std::string> pending_retirements;
  bool owns_native_registrations{false};
};

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
  if (impl_->owns_native_registrations) {
    for (const auto& item : impl_->markers) {
      unregister_indexer_snapshot(item.second);
    }
  }
}

std::unique_ptr<EmbeddingCollectionIndexerDirectory>
EmbeddingCollectionIndexerDirectory::create(
    std::unordered_map<std::string, EmbeddingCollectionBinding> bindings,
    std::unordered_map<std::string, at::Tensor> markers,
    std::unordered_map<std::string, std::shared_ptr<IndexerSnapshot>> snapshots,
    int device_index) {
  auto impl = std::make_unique<Impl>();
  impl->device_index = device_index;
  impl->bindings = std::move(bindings);
  impl->markers = std::move(markers);
  impl->snapshots = std::move(snapshots);
  for (const auto& item : impl->snapshots) {
    register_indexer_snapshot(impl->markers.at(item.first), item.second);
  }
  return std::unique_ptr<EmbeddingCollectionIndexerDirectory>(
      new EmbeddingCollectionIndexerDirectory(std::move(impl)));
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
    if (value.contains("snapshot_path") &&
        !value.at("snapshot_path").is_null()) {
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
    if (!binding.snapshot_path.has_value()) {
      continue;
    }
    auto marker = torch::tensor(
        {binding.marker_value.value()},
        torch::TensorOptions().dtype(torch::kInt64).device(
            torch::kCUDA, device_index));
    auto snapshot = load_indexer_snapshot(
        root + "/" + binding.snapshot_path.value(), device_index);
    register_indexer_snapshot(marker, snapshot);
    impl->markers.emplace(collection_id, std::move(marker));
    impl->snapshots.emplace(collection_id, std::move(snapshot));
  }
  impl->owns_native_registrations = true;
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
  for (const auto& item : marker_constants()) {
    const std::string& fqn = item.first;
    if (available.find(fqn) == available.end()) {
      throw std::runtime_error("Indexer marker is not an AOTI constant: " + fqn);
    }
    std::unordered_map<std::string, at::Tensor> constants{{fqn, item.second}};
    loader.load_constants(constants, false, false, true);
    loader.load_constants(constants, true, false, true);
  }
}

void EmbeddingCollectionIndexerDirectory::apply_update(
    const std::string& collection_id, const std::string& snapshot_path,
    int64_t snapshot_id,
    const std::function<void()>& enqueue_cache_update) {
  const bool has_replaceable_snapshot =
      impl_->markers.find(collection_id) != impl_->markers.end() &&
      has_replaceable_state(impl_->bindings.at(collection_id).indexer_type);
  if (!has_replaceable_snapshot) {
    enqueue_cache_update();
    return;
  }

  std::shared_ptr<IndexerSnapshot> next;
  if (!snapshot_path.empty()) {
    next = load_indexer_snapshot(snapshot_path, impl_->device_index);
  }

  enqueue_cache_update();
  if (!next) {
    return;
  }

  stage_indexer_snapshot(impl_->markers.at(collection_id), next, snapshot_id);
  impl_->snapshots[collection_id] = std::move(next);
  impl_->pending_retirements.insert(
      retirement_key(collection_id, snapshot_id));
}

void EmbeddingCollectionIndexerDirectory::wait_for_retirement(
    const std::string& collection_id, int64_t snapshot_id) {
  const auto key = retirement_key(collection_id, snapshot_id);
  if (impl_->pending_retirements.find(key) ==
      impl_->pending_retirements.end()) {
    return;
  }
  wait_for_indexer_snapshot_retirement(impl_->markers.at(collection_id),
                                       snapshot_id);
  impl_->pending_retirements.erase(key);
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

std::unordered_map<std::string, at::Tensor>
EmbeddingCollectionIndexerDirectory::marker_constants() const {
  std::unordered_map<std::string, at::Tensor> result;
  for (const auto& item : impl_->markers) {
    result.emplace(marker_fqn(
                       impl_->bindings.at(item.first).indexer_module_path),
                   item.second);
  }
  return result;
}

std::shared_ptr<IndexerSnapshot>
EmbeddingCollectionIndexerDirectory::snapshot(
    const std::string& collection_id) const {
  const auto iterator = impl_->snapshots.find(collection_id);
  return iterator == impl_->snapshots.end() ? nullptr : iterator->second;
}

int EmbeddingCollectionIndexerDirectory::device_index() const {
  return impl_->device_index;
}

}  // namespace dynamicemb::exportable_embedding
