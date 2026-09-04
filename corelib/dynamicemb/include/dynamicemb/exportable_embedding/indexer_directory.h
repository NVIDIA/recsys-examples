// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#if defined(_WIN32)
#define DYNAMICEMB_EXPORT __declspec(dllexport)
#else
#define DYNAMICEMB_EXPORT __attribute__((visibility("default")))
#endif

namespace dynamicemb::exportable_embedding {

enum class IndexerKind : int64_t {
  LinearHashMap = 0,
  FusedIdentity = 1,
};

struct IndexerSnapshot {
  IndexerKind kind;
  at::Tensor table_storage;
  at::Tensor table_bucket_offsets;
  int64_t bucket_capacity{0};
  at::Tensor miss_storage_indices;
  at::Tensor valid_bases;
  at::Tensor reserved_sizes;
  int64_t next_fused_key{0};
};

struct EmbeddingCollectionBinding {
  std::vector<std::string> table_names;
  std::string indexer_type;
  std::string indexer_module_path;
  std::string nve_layer_module_path;
  std::string snapshot_path;
  int64_t feature_id_bits{-1};
  int64_t marker_value{0};
};

class EmbeddingCollectionIndexerDirectory;

class DYNAMICEMB_EXPORT InferenceCall {
 public:
  InferenceCall(InferenceCall&&) noexcept;
  InferenceCall& operator=(InferenceCall&&) noexcept;
  ~InferenceCall();

  void record_complete(cudaStream_t stream);

 private:
  friend class EmbeddingCollectionIndexerDirectory;
  struct Impl;
  explicit InferenceCall(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

class DYNAMICEMB_EXPORT EmbeddingCollectionIndexerDirectory {
 public:
  static std::unique_ptr<EmbeddingCollectionIndexerDirectory> load(
      const std::string& package_dir, int device_index);

  EmbeddingCollectionIndexerDirectory(
      EmbeddingCollectionIndexerDirectory&&) noexcept;
  EmbeddingCollectionIndexerDirectory& operator=(
      EmbeddingCollectionIndexerDirectory&&) noexcept;
  ~EmbeddingCollectionIndexerDirectory();

  void bind(torch::inductor::AOTIModelPackageLoader& loader);
  InferenceCall begin_inference();
  void publish_snapshot(const std::string& collection_id,
                        const std::string& snapshot_path,
                        int64_t snapshot_id);
  void mark_update(const std::string& collection_id, int64_t snapshot_id);
  void wait_for_retirement(const std::string& collection_id,
                           int64_t snapshot_id);
  const EmbeddingCollectionBinding& binding(
      const std::string& collection_id) const;
  const std::unordered_map<std::string, EmbeddingCollectionBinding>& bindings()
      const;

 private:
  friend class InferenceCall;
  struct Impl;
  explicit EmbeddingCollectionIndexerDirectory(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

DYNAMICEMB_EXPORT std::shared_ptr<IndexerSnapshot> load_indexer_snapshot(
    const std::string& path, int device_index);

// Shared by the Torch operator and the explicit Python/C++ directory loaders.
void register_indexer_snapshot(const at::Tensor& marker,
                               std::shared_ptr<IndexerSnapshot> snapshot);
void unregister_indexer_snapshot(const at::Tensor& marker);
std::shared_ptr<IndexerSnapshot> find_indexer_snapshot(
    const at::Tensor& marker, int64_t marker_value);

}  // namespace dynamicemb::exportable_embedding
