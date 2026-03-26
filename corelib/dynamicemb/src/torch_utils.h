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
#include "check.h"
#include "utils.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/torch.h>
// #include <ATen/cuda/DeviceUtils.cuh>
#include "ATen/AccumulateType.h"
#include <pybind11/pybind11.h>
// #include <torch/python.h>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace dyn_emb {

// see
// https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/types.h
// torch provide a ScalarType 2 cpp type
// https://discuss.pytorch.org/t/how-to-convert-at-scalartype-to-c-type/195374
// TODO figure out torch's datatype now have caffe2::TypeMeta and
// c10::ScalarType
DataType scalartype_to_datatype(at::ScalarType scalar_type);
at::ScalarType datatype_to_scalartype(dyn_emb::DataType dtype);
at::ScalarType convertTypeMetaToScalarType(const caffe2::TypeMeta &typeMeta);

uint64_t device_timestamp();

inline DataType get_data_type(at::Tensor tensor) {
  return scalartype_to_datatype(tensor.dtype().toScalarType());
}

template <typename T> T *get_pointer(at::Tensor tensor) {
  if (not tensor.defined()) {
    throw std::invalid_argument("Tensor is undefined.");
  }
  return static_cast<T *>(tensor.data_ptr());
}

template <typename T> T *get_pointer(const std::optional<at::Tensor> &tensor) {
  if (not tensor.has_value()) {
    return nullptr;
  }
  auto value = tensor.value();
  if (not value.defined()) {
    throw std::invalid_argument("Tensor is undefined.");
  }
  return static_cast<T *>(value.data_ptr());
}

} // namespace dyn_emb

// PYTHON WRAP
void bind_utils(py::module &m);
