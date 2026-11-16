// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared schemas and export shape inference for CPU and CUDA implementations.

#include <ATen/ATen.h>
#include <optional>
#include <torch/library.h>

namespace {
at::Tensor expand_table_ids_meta(
    at::Tensor offsets, at::Tensor indices, std::optional<at::Tensor> table_offsets_in_feature,
    int64_t num_tables, int64_t local_batch_size) {
  return at::empty_like(indices, at::TensorOptions().dtype(at::kLong));
}

at::Tensor embedding_collection_index_meta(const at::Tensor& marker,
                                           int64_t marker_value,
                                           const at::Tensor& feature_ids,
                                           const at::Tensor& table_ids) {
  return at::empty_like(feature_ids, feature_ids.options().dtype(at::kLong));
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(INFERENCE_EMB, m) {
  m.def("expand_table_ids(Tensor offsets, Tensor indices, Tensor? table_offsets_in_feature=None, int num_tables=0, int local_batch_size=1) -> Tensor");
  m.def("embedding_collection_index(Tensor marker, int marker_value, "
        "Tensor feature_ids, Tensor table_ids) -> Tensor");
}

TORCH_LIBRARY_IMPL(INFERENCE_EMB, Meta, m) {
  m.impl("expand_table_ids", &expand_table_ids_meta);
  m.impl("embedding_collection_index", &embedding_collection_index_meta);
}
