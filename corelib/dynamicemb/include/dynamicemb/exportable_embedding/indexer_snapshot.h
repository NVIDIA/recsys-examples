// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <torch/torch.h>

#if defined(_WIN32)
#define DYNAMICEMB_EXPORT __declspec(dllexport)
#else
#define DYNAMICEMB_EXPORT __attribute__((visibility("default")))
#endif

namespace dynamicemb::exportable_embedding {

enum class IndexerSnapshotKind : int64_t {
  // Persisted in the snapshot header; keep aligned with indexer_snapshot.py.
  LinearHashMap = 0,
  FusedIdentity = 1,
};

struct IndexerSnapshot {
  IndexerSnapshotKind kind;
  at::Tensor table_storage;
  at::Tensor table_bucket_offsets;
  int64_t bucket_capacity{0};
  at::Tensor miss_storage_indices;
  at::Tensor valid_bases;
  at::Tensor reserved_sizes;
  int64_t next_fused_key{0};
};

DYNAMICEMB_EXPORT std::shared_ptr<IndexerSnapshot> load_indexer_snapshot(
    const std::string& path, int device_index);

void register_indexer_snapshot(const at::Tensor& marker,
                               std::shared_ptr<IndexerSnapshot> snapshot);
void unregister_indexer_snapshot(const at::Tensor& marker);
void stage_indexer_snapshot(const at::Tensor& marker,
                            std::shared_ptr<IndexerSnapshot> snapshot,
                            int64_t snapshot_id);
void wait_for_indexer_snapshot_retirement(const at::Tensor& marker,
                                          int64_t snapshot_id);
std::shared_ptr<IndexerSnapshot> find_indexer_snapshot(
    const at::Tensor& marker, int64_t marker_value);

}  // namespace dynamicemb::exportable_embedding
