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

#pragma once
#include "lookup_forward.h"
#include "torch_utils.h"
#include "utils.h"
#include <cstdint>
#include <cub/cub.cuh>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <utility>

namespace dyn_emb {

template <typename T, typename NumSelectedIteratorT>
void select_async(int64_t num_items, bool const *d_flags, T const *d_input,
                  T *d_output, NumSelectedIteratorT *d_num_select,
                  at::Device const &device, cudaStream_t const &stream) {

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // 1. get the size of temp storage.
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_input,
                             d_flags, d_output, d_num_select, num_items,
                             stream);

  // 2. allocate the temp storage.
  d_temp_storage =
      at::empty({static_cast<int64_t>(temp_storage_bytes)},
                at::TensorOptions().dtype(torch::kChar).device(device))
          .data_ptr();

  // 3. select
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_input,
                             d_flags, d_output, d_num_select, num_items,
                             stream);
}

template <typename T, typename NumSelectedIteratorT>
void select_index_async(int64_t num_items, bool const *d_flags, T *d_output,
                        NumSelectedIteratorT *d_num_select,
                        at::Device const &device, cudaStream_t const &stream) {
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::CountingInputIterator<T> counting_iter(0);

  // 1. get the size of temp storage.
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, counting_iter,
                             d_flags, d_output, d_num_select, num_items,
                             stream);

  // 2. allocate the temp storage.
  d_temp_storage =
      at::empty({static_cast<int64_t>(temp_storage_bytes)},
                at::TensorOptions().dtype(torch::kChar).device(device))
          .data_ptr();

  // 3. select
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, counting_iter,
                             d_flags, d_output, d_num_select, num_items,
                             stream);
}

template <size_t... Is>
void flagged_compact_impl(
    int64_t num_items, bool const *d_flags,
    int64_t const *const *d_inputs,
    int64_t *const *d_outputs,
    int64_t *d_out_indices,
    int64_t *d_num_selected,
    at::Device const &device, cudaStream_t const &stream,
    std::index_sequence<Is...>) {

    auto in_zip = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::counting_iterator<int64_t>(0), d_inputs[Is]...));
    auto out_zip = thrust::make_zip_iterator(thrust::make_tuple(
        d_out_indices, d_outputs[Is]...));

    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp, temp_bytes,
        in_zip, d_flags, out_zip, d_num_selected, num_items, stream);
    d_temp = at::empty({static_cast<int64_t>(temp_bytes)},
        at::TensorOptions().dtype(at::kByte).device(device)).data_ptr();
    cub::DeviceSelect::Flagged(d_temp, temp_bytes,
        in_zip, d_flags, out_zip, d_num_selected, num_items, stream);
}

template <size_t N>
void flagged_compact_cub(
    int64_t num_items, bool const *d_flags,
    int64_t const *const *d_inputs,
    int64_t *const *d_outputs,
    int64_t *d_out_indices,
    int64_t *d_num_selected,
    at::Device const &device, cudaStream_t const &stream) {
    flagged_compact_impl(
        num_items, d_flags, d_inputs, d_outputs,
        d_out_indices, d_num_selected, device, stream,
        std::make_index_sequence<N>{});
}

} // namespace dyn_emb

void bind_index_calculation_op(py::module &m);