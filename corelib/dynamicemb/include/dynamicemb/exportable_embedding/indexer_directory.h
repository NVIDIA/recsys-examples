// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include "dynamicemb/exportable_embedding/indexer_snapshot.h"

namespace dynamicemb::exportable_embedding {

struct EmbeddingCollectionBinding {
  std::vector<std::string> table_names;
  std::string indexer_type;
  std::string indexer_module_path;
  std::string nve_layer_module_path;
  std::optional<std::string> snapshot_path;
  std::optional<int64_t> feature_id_bits;
  std::optional<int64_t> marker_value;
};

class EmbeddingCollectionIndexerDirectory;
class EmbeddingCollectionUpdateSubscriber;

class DYNAMICEMB_EXPORT EmbeddingCollectionIndexerDirectory {
 public:
  static std::unique_ptr<EmbeddingCollectionIndexerDirectory> create(
      std::unordered_map<std::string, EmbeddingCollectionBinding> bindings,
      std::unordered_map<std::string, at::Tensor> markers,
      std::unordered_map<std::string, std::shared_ptr<IndexerSnapshot>> snapshots,
      int device_index);
  static std::unique_ptr<EmbeddingCollectionIndexerDirectory> load(
      const std::string& package_dir, int device_index);

  EmbeddingCollectionIndexerDirectory(
      EmbeddingCollectionIndexerDirectory&&) noexcept;
  EmbeddingCollectionIndexerDirectory& operator=(
      EmbeddingCollectionIndexerDirectory&&) noexcept;
  ~EmbeddingCollectionIndexerDirectory();

  void bind(torch::inductor::AOTIModelPackageLoader& loader);
  void wait_for_retirement(const std::string& collection_id,
                           int64_t snapshot_id);
  const EmbeddingCollectionBinding& binding(
      const std::string& collection_id) const;
  const std::unordered_map<std::string, EmbeddingCollectionBinding>& bindings()
      const;
  std::unordered_map<std::string, at::Tensor> marker_constants() const;
  std::shared_ptr<IndexerSnapshot> snapshot(
      const std::string& collection_id) const;
  int device_index() const;

 private:
  friend class EmbeddingCollectionUpdateSubscriber;
  struct Impl;
  explicit EmbeddingCollectionIndexerDirectory(std::unique_ptr<Impl> impl);
  void apply_update(const std::string& collection_id,
                    const std::string& snapshot_path, int64_t snapshot_id,
                    const std::function<void()>& enqueue_cache_update);
  std::unique_ptr<Impl> impl_;
};

}  // namespace dynamicemb::exportable_embedding
