// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dynamicemb/exportable_embedding/indexer_directory.h"

namespace nve {
class LayerDirectory;
template <typename IndexT>
class NVEmbedBinding;
}

namespace dynamicemb::exportable_embedding {

struct DYNAMICEMB_EXPORT EmbeddingCollectionUpdate {
  std::string collection_id;
  int64_t snapshot_id{0};
  std::vector<int64_t> cache_update_keys;
  std::vector<float> cache_update_values;
  std::string indexer_snapshot_path;

  static EmbeddingCollectionUpdate from_json(const std::string& payload);
  std::string to_json() const;
};

struct DYNAMICEMB_EXPORT EmbeddingCollectionUpdateAck {
  std::string collection_id;
  int64_t snapshot_id{0};

  std::string to_json() const;
};

class DYNAMICEMB_EXPORT EmbeddingCollectionUpdateSubscriber {
 public:
  EmbeddingCollectionUpdateSubscriber(
      const std::string& package_dir,
      EmbeddingCollectionIndexerDirectory& indexers,
      nve::LayerDirectory& nve_layers,
      int device_index);
  EmbeddingCollectionUpdateSubscriber(
      const std::string& package_dir,
      EmbeddingCollectionIndexerDirectory& indexers,
      std::unordered_map<
          std::string,
          std::shared_ptr<nve::NVEmbedBinding<int64_t>>> nve_layers,
      int device_index);
  ~EmbeddingCollectionUpdateSubscriber();

  void apply_incremental_load(const EmbeddingCollectionUpdate& update,
                              cudaStream_t inference_stream);
  EmbeddingCollectionUpdateAck wait_for_retirement(
      const std::string& collection_id, int64_t snapshot_id);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dynamicemb::exportable_embedding
