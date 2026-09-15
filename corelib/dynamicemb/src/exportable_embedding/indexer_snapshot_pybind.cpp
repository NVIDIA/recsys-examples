// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <utility>

#include "dynamicemb/exportable_embedding/indexer_snapshot.h"

namespace py = pybind11;

namespace dynamicemb::exportable_embedding {

void bind_indexer_snapshot(py::module_& module) {
  py::enum_<IndexerSnapshotKind>(module, "IndexerSnapshotKind")
      .value("LINEAR_HASH_MAP", IndexerSnapshotKind::LinearHashMap)
      .value("FUSED_IDENTITY", IndexerSnapshotKind::FusedIdentity);

  py::class_<IndexerSnapshot, std::shared_ptr<IndexerSnapshot>>(
      module, "IndexerSnapshot")
      .def(py::init([](int64_t kind, at::Tensor table_storage,
                       at::Tensor table_bucket_offsets, int64_t bucket_capacity,
                       at::Tensor miss_storage_indices, at::Tensor valid_bases,
                       at::Tensor reserved_sizes, int64_t next_fused_key) {
        auto snapshot = std::make_shared<IndexerSnapshot>();
        snapshot->kind = static_cast<IndexerSnapshotKind>(kind);
        snapshot->table_storage = std::move(table_storage);
        snapshot->table_bucket_offsets = std::move(table_bucket_offsets);
        snapshot->bucket_capacity = bucket_capacity;
        snapshot->miss_storage_indices = std::move(miss_storage_indices);
        snapshot->valid_bases = std::move(valid_bases);
        snapshot->reserved_sizes = std::move(reserved_sizes);
        snapshot->next_fused_key = next_fused_key;
        return snapshot;
      }))
      .def_property_readonly("kind", [](const IndexerSnapshot& snapshot) {
        return static_cast<int64_t>(snapshot.kind);
      })
      .def_readonly("table_storage", &IndexerSnapshot::table_storage)
      .def_readonly("table_bucket_offsets",
                    &IndexerSnapshot::table_bucket_offsets)
      .def_readonly("bucket_capacity", &IndexerSnapshot::bucket_capacity)
      .def_readonly("miss_storage_indices",
                    &IndexerSnapshot::miss_storage_indices)
      .def_readonly("valid_bases", &IndexerSnapshot::valid_bases)
      .def_readonly("reserved_sizes", &IndexerSnapshot::reserved_sizes)
      .def_readonly("next_fused_key", &IndexerSnapshot::next_fused_key);

  module.def("load_indexer_snapshot", &load_indexer_snapshot, py::arg("path"),
             py::arg("device_index"));
}

}  // namespace dynamicemb::exportable_embedding
