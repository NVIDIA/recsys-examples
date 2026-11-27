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
#include "utils.h"
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/torch.h>

#include "state/mixed_dim_view.h"

namespace dyn_emb {

void sgd_update_for_flat_table(const FlatTablesView &view, at::Tensor grads,
                               at::Tensor indices, at::Tensor table_ids,
                               float const lr);

void adam_update_for_flat_table(const FlatTablesView &view, at::Tensor grads,
                                at::Tensor indices, at::Tensor table_ids,
                                const float lr, const float beta1,
                                const float beta2, const float eps,
                                const float weight_decay,
                                const uint32_t iter_num);

void adagrad_update_for_flat_table(const FlatTablesView &view,
                                   at::Tensor grads, at::Tensor indices,
                                   at::Tensor table_ids, const float lr,
                                   const float eps);

void rowwise_adagrad_for_flat_table(const FlatTablesView &view,
                                    at::Tensor grads, at::Tensor indices,
                                    at::Tensor table_ids, const float lr,
                                    const float eps, int64_t state_dtype);

void ftrl_update_for_flat_table(const FlatTablesView &view, at::Tensor grads,
                                at::Tensor indices, at::Tensor table_ids,
                                const float lr,
                                const float learning_rate_power,
                                const float beta, const float l1_reg,
                                const float l2_reg,
                                const float group_lasso_reg = 0.0f);

// `values` is a padded buffer of the tables `view` describes (rows
// view.max_value_dim() apart).
void sgd_update_for_padded_buffer(const PaddedBufferView &view,
                                  at::Tensor grads, at::Tensor values,
                                  at::Tensor table_ids, float lr);

void adam_update_for_padded_buffer(const PaddedBufferView &view,
                                   at::Tensor grads, at::Tensor values,
                                   at::Tensor table_ids, float lr, float beta1,
                                   float beta2, float eps, float weight_decay,
                                   uint32_t iter_num);

void adagrad_update_for_padded_buffer(const PaddedBufferView &view,
                                      at::Tensor grads, at::Tensor values,
                                      at::Tensor table_ids, float lr,
                                      float eps);

void rowwise_adagrad_for_padded_buffer(const PaddedBufferView &view,
                                       at::Tensor grads, at::Tensor values,
                                       at::Tensor table_ids, float lr,
                                       float eps, int64_t state_dtype);

void ftrl_update_for_padded_buffer(const PaddedBufferView &view,
                                   at::Tensor grads, at::Tensor values,
                                   at::Tensor table_ids, float lr,
                                   float learning_rate_power, float beta,
                                   float l1_reg, float l2_reg,
                                   float group_lasso_reg);

} // namespace dyn_emb
#endif // OPTIMIZER_H
