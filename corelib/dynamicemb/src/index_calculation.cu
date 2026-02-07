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

#include "check.h"
#include "index_calculation.h"
#include "utils.h"
#include <torch/extension.h>

namespace dyn_emb {

template <typename InT, typename OutT>
__global__ void get_table_range_kernel(int64_t num_table,
                                       int64_t feature_x_batch,
                                       InT const *__restrict__ offsets,
                                       OutT const *__restrict__ feature_offsets,
                                       OutT *__restrict__ table_range) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_table + 1) {
    OutT num_feature = feature_offsets[num_table];
    int64_t batch = feature_x_batch / num_feature;
    OutT feature_offset = feature_offsets[tid];
    int64_t feature_x_batch_offset = feature_offset * batch;
    table_range[tid] = static_cast<OutT>(offsets[feature_x_batch_offset]);
  }
}

at::Tensor get_table_range(at::Tensor offsets, at::Tensor feature_offsets) {
  if (!offsets.is_cuda()) {
    throw std::runtime_error("Tensor <offsets> must be on CUDA device.");
  }
  if (!feature_offsets.is_cuda()) {
    throw std::runtime_error(
        "Tensor <feature_offsets> must be on CUDA device.");
  }
  int64_t feature_x_batch = offsets.size(0) - 1;
  int64_t num_table = feature_offsets.size(0) - 1;

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  at::Tensor table_range = at::empty_like(feature_offsets);

  int block_size = 128;
  if (num_table + 1 < block_size) {
    block_size = num_table + 1;
  }
  int grid_size = (num_table + block_size) / block_size;
  auto offset_type = scalartype_to_datatype(offsets.dtype().toScalarType());
  auto range_type =
      scalartype_to_datatype(feature_offsets.dtype().toScalarType());
  DISPATCH_OFFSET_INT_TYPE(offset_type, offset_t, [&] {
    DISPATCH_OFFSET_INT_TYPE(range_type, range_t, [&] {
      get_table_range_kernel<offset_t, range_t>
          <<<grid_size, block_size, 0, stream>>>(
              num_table, feature_x_batch,
              reinterpret_cast<offset_t *>(offsets.data_ptr()),
              reinterpret_cast<range_t *>(feature_offsets.data_ptr()),
              reinterpret_cast<range_t *>(table_range.data_ptr()));
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  return table_range;
}

void select(at::Tensor flags, at::Tensor inputs, at::Tensor outputs,
            at::Tensor num_selected) {

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t num_total = inputs.size(0);
  auto scalar_type = inputs.dtype().toScalarType();
  auto key_type = scalartype_to_datatype(scalar_type);
  auto num_select_iter_type =
      scalartype_to_datatype(num_selected.dtype().toScalarType());

  if (num_total == 0) {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(
        num_select_iter_type, NumSelectedIteratorT, [&] {
          DEMB_CUDA_CHECK(cudaMemsetAsync(
              reinterpret_cast<NumSelectedIteratorT *>(num_selected.data_ptr()),
              0, sizeof(NumSelectedIteratorT), stream));
        });
    return;
  }
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, KeyType, [&] {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(
        num_select_iter_type, NumSelectedIteratorT, [&] {
          select_async<KeyType, NumSelectedIteratorT>(
              num_total, flags.data_ptr<bool>(),
              reinterpret_cast<KeyType *>(inputs.data_ptr()),
              reinterpret_cast<KeyType *>(outputs.data_ptr()),
              reinterpret_cast<NumSelectedIteratorT *>(num_selected.data_ptr()),
              inputs.device(), stream);
        });
  });
}

void select_index(at::Tensor flags, at::Tensor output_indices,
                  at::Tensor num_selected) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t num_total = output_indices.size(0);
  auto scalar_type = output_indices.dtype().toScalarType();
  auto key_type = scalartype_to_datatype(scalar_type);
  auto num_select_iter_type =
      scalartype_to_datatype(num_selected.dtype().toScalarType());

  if (num_total == 0) {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(
        num_select_iter_type, NumSelectedIteratorT, [&] {
          DEMB_CUDA_CHECK(cudaMemsetAsync(
              reinterpret_cast<NumSelectedIteratorT *>(num_selected.data_ptr()),
              0, sizeof(NumSelectedIteratorT), stream));
        });
    return;
  }
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, KeyType, [&] {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(
        num_select_iter_type, NumSelectedIteratorT, [&] {
          select_index_async<KeyType, NumSelectedIteratorT>(
              num_total, flags.data_ptr<bool>(),
              reinterpret_cast<KeyType *>(output_indices.data_ptr()),
              reinterpret_cast<NumSelectedIteratorT *>(num_selected.data_ptr()),
              output_indices.device(), stream);
        });
  });
}

} // namespace dyn_emb

void bind_index_calculation_op(py::module &m) {
  m.def("get_table_range", &dyn_emb::get_table_range,
        "Make offsets from <feature, batch> scope into <table> scope",
        py::arg("offsets"), py::arg("feature_offsets"));

  m.def("select", &dyn_emb::select,
        "Select items in inputs which flags are true.", py::arg("flags"),
        py::arg("inputs"), py::arg("outputs"), py::arg("num_selected"));
  m.def("select_index", &dyn_emb::select_index,
        "Select items' indices where flags are true.", py::arg("flags"),
        py::arg("output_indices"), py::arg("num_selected"));
}
