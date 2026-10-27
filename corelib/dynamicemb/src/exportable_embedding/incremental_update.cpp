// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dynamicemb/exportable_embedding/incremental_update.h"
#include "dynamicemb/exportable_embedding/nve_runtime.h"

#include <c10/cuda/CUDAGuard.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "python/pynve/torch_bindings/nve_loader.hpp"

namespace dynamicemb::exportable_embedding {

EmbeddingCollectionUpdate EmbeddingCollectionUpdate::from_json(
    const std::string& payload) {
  const auto value = nlohmann::json::parse(payload);
  EmbeddingCollectionUpdate update;
  update.collection_id = value.at("collection_id").get<std::string>();
  update.snapshot_id = value.at("snapshot_id").get<int64_t>();
  update.cache_update_keys =
      value.at("cache_update_keys").get<std::vector<int64_t>>();
  update.cache_update_values =
      value.at("cache_update_values").get<std::vector<float>>();
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
      {"cache_update_keys", cache_update_keys},
      {"cache_update_values", cache_update_values},
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
  int device_index;
  std::unordered_map<std::string,
                     std::shared_ptr<nve::NVEmbedBinding<int64_t>>>
      nve_layers;
  std::unordered_map<std::string, bool> has_host_cache;
  std::unordered_map<std::string, int64_t> host_rows;
  std::unordered_map<std::string, int64_t> host_update_ids;

  void load_metadata(const std::string& package_dir,
                     nve::LayerDirectory* layer_directory) {
    std::ifstream stream(package_dir + "/metadata.json");
    const auto metadata = nlohmann::json::parse(stream);
    const auto& layers = metadata.is_array() ? metadata : metadata.at("layers");
    for (const auto& layer : layers) {
      const std::string path = layer.at("module_path").get<std::string>();
      has_host_cache[path] = layer.value("host_cache_size", uint64_t{0}) > 0;
      if (layer.value("layer_type", std::string{}) == "HostLayer") {
        host_rows[path] = layer.at("num_embeddings").get<int64_t>();
      }
      if (layer_directory != nullptr) {
        nve_layers[path] =
            layer_directory->get_layer(layer.at("id").get<int64_t>()).binding;
      }
      TORCH_CHECK(host_rows.count(path) || has_nve_tier_update(nve_layers.at(path)),
                  "Tier updates require DynamicEmb NVE bindings; use the DynamicEmb factory/loader");
    }
  }

  Impl(const std::string& package_dir,
       EmbeddingCollectionIndexerDirectory& indexer_directory,
       nve::LayerDirectory& layer_directory, int device)
      : indexers(indexer_directory),
        device_index(device) {
    load_metadata(package_dir, &layer_directory);
  }

  Impl(const std::string& package_dir,
       EmbeddingCollectionIndexerDirectory& indexer_directory,
       std::unordered_map<
           std::string,
           std::shared_ptr<nve::NVEmbedBinding<int64_t>>> layer_bindings,
       int device)
      : indexers(indexer_directory),
        device_index(device),
        nve_layers(std::move(layer_bindings)) {
    load_metadata(package_dir, nullptr);
  }
};

EmbeddingCollectionUpdateSubscriber::EmbeddingCollectionUpdateSubscriber(
    const std::string& package_dir,
    EmbeddingCollectionIndexerDirectory& indexers,
    nve::LayerDirectory& nve_layers, int device_index)
    : impl_(std::make_unique<Impl>(package_dir, indexers, nve_layers,
                                   device_index)) {}

EmbeddingCollectionUpdateSubscriber::EmbeddingCollectionUpdateSubscriber(
    const std::string& package_dir,
    EmbeddingCollectionIndexerDirectory& indexers,
    std::unordered_map<std::string,
                       std::shared_ptr<nve::NVEmbedBinding<int64_t>>> nve_layers,
    int device_index)
    : impl_(std::make_unique<Impl>(package_dir, indexers,
                                   std::move(nve_layers), device_index)) {}

EmbeddingCollectionUpdateSubscriber::~EmbeddingCollectionUpdateSubscriber() =
    default;

void EmbeddingCollectionUpdateSubscriber::apply_incremental_load(
    const EmbeddingCollectionUpdate& update, cudaStream_t inference_stream,
    bool inference_paused) {
  const auto& binding = impl_->indexers.binding(update.collection_id);
  const auto host = impl_->host_rows.find(binding.nve_layer_module_path);
  if (host != impl_->host_rows.end()) {
    TORCH_CHECK(impl_->device_index < 0 && inference_stream == nullptr,
                "HostLayer incremental updates require CPU execution and no CUDA stream");
    const bool linear = binding.indexer_type == "linear_hash_map";
    TORCH_CHECK(linear || binding.indexer_type == "identity" ||
                    binding.indexer_type == "fused_identity", "Unsupported Host indexer");
    TORCH_CHECK(!linear || inference_paused,
                "Host LinearHashMap updates require paused/drained inference");
    TORCH_CHECK(linear || update.indexer_snapshot_path.empty(),
                "Identity/FusedIdentity updates must preserve the mapping");
    TORCH_CHECK(update.snapshot_id == impl_->host_update_ids[update.collection_id] + 1,
                "Host updates must be applied once in snapshot order");
    const auto& layer = impl_->nve_layers.at(binding.nve_layer_module_path);
    const auto dim = layer->get_embedding_dim();
    TORCH_CHECK(update.cache_update_values.size() == update.cache_update_keys.size() * dim,
                "Host update values must contain one complete embedding per key");
    TORCH_CHECK(binding.table_capacities.size() == binding.table_names.size(),
                "Host updates require exported table capacities");
    std::unordered_set<int64_t> misses;
    int64_t base = 0;
    if (binding.indexer_type != "identity") {
      for (const auto capacity : binding.table_capacities) {
        misses.insert(base);
        base += linear ? 1 : capacity + 1;
      }
    }
    for (const auto row : update.cache_update_keys) {
      TORCH_CHECK(row >= 0 && row < host->second && misses.count(row) == 0,
                  "Host update row is outside storage or is a reserved miss row");
    }
    auto keys = torch::tensor(update.cache_update_keys,
                              torch::TensorOptions().dtype(torch::kInt64));
    const auto dtype = layer->get_data_type() == nve::DataType_t::Float16
        ? torch::kFloat16 : torch::kFloat32;
    auto values = torch::tensor(update.cache_update_values,
                                torch::TensorOptions().dtype(torch::kFloat32)).to(dtype);
    impl_->indexers.apply_update(
        update.collection_id, update.indexer_snapshot_path, update.snapshot_id, [&] {
          if (keys.numel() > 0) {
            // CPU tensors, separate update context (stream key 0). NVE CPU
            // lookups use per-thread sentinel keys. Host update completes here;
            // relaxed row access permits concurrent stale/mixed lookup results.
            layer->update(keys.numel(), reinterpret_cast<uintptr_t>(keys.data_ptr()),
                          reinterpret_cast<uintptr_t>(values.data_ptr()), 0);
          }
        });
    impl_->host_update_ids[update.collection_id] = update.snapshot_id;
    return;
  }
  TORCH_CHECK(impl_->device_index >= 0, "CPU updates require HostLayer storage");
  const auto stream = c10::cuda::getStreamFromExternal(
      inference_stream, impl_->device_index);
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  auto keys = torch::tensor(
      update.cache_update_keys,
      torch::TensorOptions().dtype(torch::kInt64));
  impl_->indexers.apply_update(
      update.collection_id, update.indexer_snapshot_path, update.snapshot_id,
      [&] {
        if (keys.numel() == 0) {
          return;
        }
        const auto& layer =
            impl_->nve_layers.at(binding.nve_layer_module_path);
        const auto value_dtype =
            layer->get_data_type() == nve::DataType_t::Float16
            ? torch::kFloat16
            : torch::kFloat32;
        auto values = torch::tensor(
                          update.cache_update_values,
                          torch::TensorOptions().dtype(torch::kFloat32))
                          .to(value_dtype)
                          .view({keys.numel(), layer->get_embedding_dim()});
        const uint64_t nve_stream =
            reinterpret_cast<uint64_t>(inference_stream);
        const auto* key_data = keys.data_ptr<int64_t>();
        const auto* value_data =
            reinterpret_cast<const uint8_t*>(values.data_ptr());
        const auto row_size =
            values.element_size() * layer->get_embedding_dim();
        for (int64_t index = 0; index < keys.numel(); ++index) {
          const auto key = reinterpret_cast<uintptr_t>(key_data + index);
          const auto value = reinterpret_cast<uintptr_t>(
              value_data + index * row_size);
          if (impl_->has_host_cache.at(binding.nve_layer_module_path)) {
            update_nve_tier(layer, 1, key, value, 1, nve_stream);
          }
          update_nve_tier(layer, 1, key, value, 0, nve_stream);
        }
      });
}

EmbeddingCollectionUpdateAck
EmbeddingCollectionUpdateSubscriber::wait_for_retirement(
    const std::string& collection_id, int64_t snapshot_id) {
  impl_->indexers.wait_for_retirement(collection_id, snapshot_id);
  return {collection_id, snapshot_id};
}

}  // namespace dynamicemb::exportable_embedding
