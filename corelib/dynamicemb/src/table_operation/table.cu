/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
******************************************************************************/

#include "table.cuh"

namespace dyn_emb {

std::vector<at::Tensor> table_partition(at::Tensor storage,
                                        std::vector<torch::Dtype> dtypes,
                                        int64_t bucket_capacity,
                                        int64_t num_buckets) {

  int64_t num_types = dtypes.size();
  std::vector<at::Tensor> result;
  result.reserve(num_types);

  std::vector<int> bytes_offset;
  std::vector<int> bucket_bytes_offset;

  bytes_offset.reserve(num_types + 1);
  bucket_bytes_offset.reserve(num_types + 1);

  bytes_offset.push_back(0);
  bucket_bytes_offset.push_back(0);

  for (int64_t i = 0; i < num_types; i++) {
    auto scalar_type = static_cast<torch::ScalarType>(dtypes[i]);
    auto dtype_bytes = get_size(scalar_type);
    auto offset = bytes_offset.back() + dtype_bytes;
    bytes_offset.push_back(offset);

    auto array_bytes = dtype_bytes * bucket_capacity;
    auto array_offset = bucket_bytes_offset.back() + array_bytes;
    bucket_bytes_offset.push_back(array_offset);
  }

  int64_t bucket_bytes = bucket_bytes_offset.back();
  if (bucket_bytes * num_buckets != storage.numel() * storage.element_size()) {
    throw std::runtime_error(
        "Storage size mismatched with bucket_bytes * num_buckets");
  }

  for (int64_t i = 0; i < num_types; i++) {
    int64_t stride = bucket_bytes / (bytes_offset[i + 1] - bytes_offset[i]);
    void *raw_data = storage.data_ptr() + bucket_capacity * bytes_offset[i];
    result.push_back(at::from_blob(raw_data, {num_buckets, bucket_capacity},
                                   {stride, 1},
                                   storage.options().dtype(dtypes[i])));
  }
  return result;
}

std::vector<at::Tensor> tensor_partition(at::Tensor input,
                                         std::vector<int64_t> byte_range,
                                         std::vector<torch::Dtype> dtypes) {
  int num_partition = byte_range.size() - 1;
  std::vector<at::Tensor> result;
  result.reserve(num_partition);
  for (int i = 0; i < num_partition; i++) {
    auto raw_data = input.data_ptr() + byte_range[i];
    int64_t partition_size = byte_range[i + 1] - byte_range[i];
    auto scalar_type = static_cast<torch::ScalarType>(dtypes[i]);
    partition_size = partition_size / get_size(scalar_type);
    result.push_back(at::from_blob(raw_data, {partition_size},
                                   input.options().dtype(dtypes[i])));
  }
  return result;
}

} // namespace dyn_emb

namespace py = pybind11;

void bind_table_operation(py::module &m) {

  m.def("tensor_partition", &dyn_emb::tensor_partition,
        "split the tensor into several sub-partitions.", py::arg("input"),
        py::arg("byte_range"), py::arg("dtypes"));

  m.def("table_partition", &dyn_emb::table_partition,
        "split the tensor into several sub-partitions.", py::arg("storage"),
        py::arg("dtypes"), py::arg("bucket_capacity"), py::arg("num_buckets"));

  m.def("table_lookup", &dyn_emb::table_lookup, "lookup the table",
        py::arg("table_storage"), py::arg("dtypes"), py::arg("bucket_capacity"),
        py::arg("keys"), py::arg("scores"), py::arg("policy_types"),
        py::arg("is_returns"), py::arg("founds"), py::arg("indices"));

  m.def("table_insert", &dyn_emb::table_insert, "insert into the table",
        py::arg("table_storage"), py::arg("dtypes"), py::arg("bucket_capacity"),
        py::arg("bucket_sizes"), py::arg("keys"), py::arg("scores"),
        py::arg("policy_types"), py::arg("is_returns"), py::arg("indices"),
        py::arg("insert_results"));

  m.def("table_insert_and_evict", &dyn_emb::table_insert_and_evict,
        "insert into the table", py::arg("table_storage"), py::arg("dtypes"),
        py::arg("bucket_capacity"), py::arg("bucket_sizes"), py::arg("keys"),
        py::arg("scores"), py::arg("policy_types"), py::arg("is_returns"),
        py::arg("insert_results"), py::arg("indices"), py::arg("num_evicted"),
        py::arg("evicted_keys"), py::arg("evicted_indices"),
        py::arg("evicted_scores"));

  m.def("table_erase", &dyn_emb::table_erase, "erase keys from the table",
        py::arg("table_storage"), py::arg("dtypes"), py::arg("bucket_capacity"),
        py::arg("bucket_sizes"), py::arg("keys"),
        py::arg("indices") = py::none());

  m.def("table_export_batch", &dyn_emb::table_export_batch,
        "export items[offset, offset + batch) from the table",
        py::arg("table_storage"), py::arg("dtypes"), py::arg("bucket_capacity"),
        py::arg("batch"), py::arg("offset"), py::arg("counter"),
        py::arg("keys"), py::arg("scores"), py::arg("thresholds") = py::none(),
        py::arg("indices") = py::none());

  m.def("table_count_matched", &dyn_emb::table_count_matched,
        "count number of items in the table whose scores >= thresholds",
        py::arg("table_storage"), py::arg("dtypes"), py::arg("bucket_capacity"),
        py::arg("thresholds"), py::arg("num_matched"));

  py::enum_<dyn_emb::ScorePolicyType>(m, "ScorePolicy")
      .value("CONST", dyn_emb::ScorePolicyType::Const)
      .value("ASSIGN", dyn_emb::ScorePolicyType::Assign)
      .value("ACCUMULATE", dyn_emb::ScorePolicyType::Accumulate)
      .value("GLOBAL_TIMER", dyn_emb::ScorePolicyType::GlobalTimer)
      .export_values();

  py::enum_<dyn_emb::InsertResult>(m, "InsertResult")
      .value("INSERT", dyn_emb::InsertResult::Insert)
      .value("RECLAIM", dyn_emb::InsertResult::Reclaim)
      .value("ASSIGN", dyn_emb::InsertResult::Assign)
      .value("EVICT", dyn_emb::InsertResult::Evict)
      .value("DUPLICATED", dyn_emb::InsertResult::Duplicated)
      .value("BUSY", dyn_emb::InsertResult::Busy)
      .value("INIT", dyn_emb::InsertResult::Init)
      .export_values();
}
