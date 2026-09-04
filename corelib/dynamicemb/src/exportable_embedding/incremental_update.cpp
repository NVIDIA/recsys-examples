// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dynamicemb/exportable_embedding/incremental_update.h"

#include <ATen/cuda/CUDAContext.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <unordered_map>

#include "python/pynve/torch_bindings/nve_loader.hpp"

namespace dynamicemb::exportable_embedding {

EmbeddingCollectionUpdate EmbeddingCollectionUpdate::from_json(
    const std::string& payload) {
  const auto value = nlohmann::json::parse(payload);
  EmbeddingCollectionUpdate update;
  update.collection_id = value.at("collection_id").get<std::string>();
  update.snapshot_id = value.at("snapshot_id").get<int64_t>();
  update.affected_storage_keys =
      value.at("affected_storage_keys").get<std::vector<int64_t>>();
  if (!value.at("indexer_snapshot_path").is_null()) {
    update.indexer_snapshot_path =
        value.at("indexer_snapshot_path").get<std::string>();
  }
  return update;
}

std::string EmbeddingCollectionUpdate::to_json() const {
  nlohmann::json value{
      {"collection_id", collection_id},
      {"snapshot_id", snapshot_id},
      {"affected_storage_keys", affected_storage_keys},
      {"indexer_snapshot_path",
       indexer_snapshot_path.empty() ? nlohmann::json(nullptr)
                                     : nlohmann::json(indexer_snapshot_path)},
  };
  return value.dump();
}

std::string EmbeddingCollectionUpdateAck::to_json() const {
  return nlohmann::json({{"collection_id", collection_id},
                         {"snapshot_id", snapshot_id}})
      .dump();
}

struct EmbeddingCollectionUpdateSubscriber::Impl {
  EmbeddingCollectionIndexerDirectory& indexers;
  nve::LayerDirectory& nve_layers;
  int device_index;
  std::unordered_map<std::string, int64_t> layer_ids;
  std::unordered_map<std::string, bool> has_host_cache;

  Impl(const std::string& package_dir,
       EmbeddingCollectionIndexerDirectory& indexer_directory,
       nve::LayerDirectory& layer_directory, int device)
      : indexers(indexer_directory),
        nve_layers(layer_directory),
        device_index(device) {
    std::ifstream stream(package_dir + "/metadata.json");
    const auto metadata = nlohmann::json::parse(stream);
    const auto& layers = metadata.is_array() ? metadata : metadata.at("layers");
    for (const auto& layer : layers) {
      const std::string path = layer.at("module_path").get<std::string>();
      layer_ids[path] = layer.at("id").get<int64_t>();
      has_host_cache[path] = layer.value("host_cache_size", uint64_t{0}) > 0;
    }
  }
};

EmbeddingCollectionUpdateSubscriber::EmbeddingCollectionUpdateSubscriber(
    const std::string& package_dir,
    EmbeddingCollectionIndexerDirectory& indexers,
    nve::LayerDirectory& nve_layers, int device_index)
    : impl_(std::make_unique<Impl>(package_dir, indexers, nve_layers,
                                   device_index)) {}

EmbeddingCollectionUpdateSubscriber::~EmbeddingCollectionUpdateSubscriber() =
    default;

void EmbeddingCollectionUpdateSubscriber::apply_incremental_load(
    const EmbeddingCollectionUpdate& update) {
  const auto& binding = impl_->indexers.binding(update.collection_id);
  auto keys = torch::tensor(
      update.affected_storage_keys,
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA,
                                                         impl_->device_index));
  if (keys.numel() != 0) {
    auto& layer = impl_->nve_layers.get_layer(
        impl_->layer_ids.at(binding.nve_layer_module_path));
    const uint64_t stream = reinterpret_cast<uint64_t>(
        at::cuda::getCurrentCUDAStream(impl_->device_index).stream());
    layer.binding->erase(keys.numel(),
                         reinterpret_cast<uintptr_t>(keys.data_ptr()), 0,
                         stream);
    if (impl_->has_host_cache.at(binding.nve_layer_module_path)) {
      layer.binding->erase(keys.numel(),
                           reinterpret_cast<uintptr_t>(keys.data_ptr()), 1,
                           stream);
    }
    cudaSetDevice(impl_->device_index);
    cudaDeviceSynchronize();
  }

  if (update.indexer_snapshot_path.empty()) {
    impl_->indexers.mark_update(update.collection_id, update.snapshot_id);
  } else {
    impl_->indexers.publish_snapshot(update.collection_id,
                                     update.indexer_snapshot_path,
                                     update.snapshot_id);
  }
}

EmbeddingCollectionUpdateAck
EmbeddingCollectionUpdateSubscriber::wait_for_retirement(
    const std::string& collection_id, int64_t snapshot_id) {
  impl_->indexers.wait_for_retirement(collection_id, snapshot_id);
  return {collection_id, snapshot_id};
}

}  // namespace dynamicemb::exportable_embedding
