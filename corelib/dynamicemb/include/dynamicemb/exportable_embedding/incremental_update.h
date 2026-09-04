// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "dynamicemb/exportable_embedding/indexer_directory.h"

namespace nve {
class LayerDirectory;
}

namespace dynamicemb::exportable_embedding {

struct EmbeddingCollectionUpdate {
  std::string collection_id;
  int64_t snapshot_id{0};
  std::vector<int64_t> affected_storage_keys;
  std::string indexer_snapshot_path;

  static EmbeddingCollectionUpdate from_json(const std::string& payload);
  std::string to_json() const;
};

struct EmbeddingCollectionUpdateAck {
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
  ~EmbeddingCollectionUpdateSubscriber();

  void apply_incremental_load(const EmbeddingCollectionUpdate& update);
  EmbeddingCollectionUpdateAck wait_for_retirement(
      const std::string& collection_id, int64_t snapshot_id);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dynamicemb::exportable_embedding
