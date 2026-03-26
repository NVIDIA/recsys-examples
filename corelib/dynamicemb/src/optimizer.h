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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/DeviceUtils.cuh"
#include "check.h"
#include "dynamic_variable_base.h"
#include "utils.h"
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/torch.h>

namespace dyn_emb {

void dynamic_emb_sgd_with_table(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table, const uint64_t n,
    const at::Tensor indices, const at::Tensor grads, const float lr,
    DataType weight_type);

void dynamic_emb_adam_with_table(
    std::shared_ptr<dyn_emb::DynamicVariableBase> ht, const uint64_t n,
    const at::Tensor indices, const at::Tensor grads, const float lr,
    const float beta1, const float beta2, const float eps,
    const float weight_decay, const uint32_t iter_num, DataType weight_type);

void dynamic_emb_adagrad_with_table(
    std::shared_ptr<dyn_emb::DynamicVariableBase> ht, const uint64_t n,
    const at::Tensor indices, const at::Tensor grads, const float lr,
    const float eps, DataType weight_type);

void dynamic_emb_rowwise_adagrad_with_table(
    std::shared_ptr<dyn_emb::DynamicVariableBase> ht, const uint64_t n,
    const at::Tensor indices, const at::Tensor grads, const float lr,
    const float eps, DataType weight_type);

void dynamic_emb_sgd_with_pointer(at::Tensor grads, at::Tensor val_pointers,
                                  DataType val_type, float const lr);

void dynamic_emb_adam_with_pointer(at::Tensor grads, at::Tensor val_pointers,
                                   DataType val_type, int64_t state_dim,
                                   const float lr, const float beta1,
                                   const float beta2, const float eps,
                                   const float weight_decay,
                                   const uint32_t iter_num);

void dynamic_emb_adagrad_with_pointer(at::Tensor grads, at::Tensor val_pointers,
                                      DataType val_type, int64_t state_dim,
                                      const float lr, const float eps);

void dynamic_emb_rowwise_adagrad_with_pointer(at::Tensor grads,
                                              at::Tensor val_pointers,
                                              DataType val_type,
                                              int64_t state_dim, const float lr,
                                              const float eps);

} // namespace dyn_emb
#endif // OPTIMIZER_H
