// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dynamicemb/exportable_embedding/indexer_directory.h"

#include <optional>

#include <torch/library.h>

#include "table_operation/table.cuh"

namespace dynamicemb::exportable_embedding {

at::Tensor embedding_collection_index_cuda(const at::Tensor& marker,
                                            int64_t marker_value,
                                            const at::Tensor& feature_ids,
                                            const at::Tensor& table_ids) {
  TORCH_CHECK(marker.is_cuda(), "indexer marker must be on CUDA");
  TORCH_CHECK(feature_ids.is_cuda() && table_ids.is_cuda(),
              "embedding collection IDs must be on CUDA");
  auto snapshot = find_indexer_snapshot(marker, marker_value);
  if (snapshot->kind == IndexerKind::LinearHashMap) {
    auto result = dyn_emb::table_lookup(
        snapshot->table_storage, snapshot->table_bucket_offsets,
        snapshot->bucket_capacity, feature_ids, table_ids, std::nullopt,
        dyn_emb::ScorePolicyType::Const);
    auto& fused_keys = std::get<0>(result);
    auto& found = std::get<1>(result);
    auto misses = at::index_select(snapshot->miss_storage_indices, 0, table_ids);
    return at::where(found, fused_keys, misses);
  }
  if (snapshot->kind == IndexerKind::FusedIdentity) {
    return at::index_select(snapshot->valid_bases, 0, table_ids) + feature_ids;
  }
  TORCH_CHECK(false, "Unsupported embedding-collection indexer kind");
}

at::Tensor embedding_collection_index_meta(const at::Tensor& marker,
                                            int64_t marker_value,
                                            const at::Tensor& feature_ids,
                                            const at::Tensor& table_ids) {
  (void)marker;
  (void)marker_value;
  (void)table_ids;
  return at::empty_like(feature_ids, feature_ids.options().dtype(at::kLong));
}

}  // namespace dynamicemb::exportable_embedding

TORCH_LIBRARY_FRAGMENT(INFERENCE_EMB, m) {
  m.def("embedding_collection_index(Tensor marker, int marker_value, "
        "Tensor feature_ids, Tensor table_ids) -> Tensor");
}

TORCH_LIBRARY_IMPL(INFERENCE_EMB, CUDA, m) {
  m.impl("embedding_collection_index",
         &dynamicemb::exportable_embedding::embedding_collection_index_cuda);
}

TORCH_LIBRARY_IMPL(INFERENCE_EMB, Meta, m) {
  m.impl("embedding_collection_index",
         &dynamicemb::exportable_embedding::embedding_collection_index_meta);
}
