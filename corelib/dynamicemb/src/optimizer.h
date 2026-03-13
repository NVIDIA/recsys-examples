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

namespace dyn_emb {

void sgd_update_for_flat_table(at::Tensor grads, at::Tensor indices,
                               at::Tensor table_ptrs, at::Tensor table_ids,
                               at::Tensor table_value_dims,
                               at::Tensor table_emb_dims,
                               int64_t max_emb_dim, bool all_dims_vec4,
                               float const lr,
                               int64_t table_dtype);

void adam_update_for_flat_table(at::Tensor grads, at::Tensor indices,
                                at::Tensor table_ptrs, at::Tensor table_ids,
                                at::Tensor table_value_dims,
                                at::Tensor table_emb_dims, const float lr,
                                const float beta1, const float beta2,
                                const float eps, const float weight_decay,
                                const uint32_t iter_num,
                                int64_t max_emb_dim, bool all_dims_vec4,
                                int64_t table_dtype);

void adagrad_update_for_flat_table(at::Tensor grads, at::Tensor indices,
                                   at::Tensor table_ptrs,
                                   at::Tensor table_ids,
                                   at::Tensor table_value_dims,
                                   at::Tensor table_emb_dims, const float lr,
                                   const float eps,
                                   int64_t max_emb_dim, bool all_dims_vec4,
                                   int64_t table_dtype);

void rowwise_adagrad_for_flat_table(at::Tensor grads, at::Tensor indices,
                                    at::Tensor table_ptrs,
                                    at::Tensor table_ids,
                                    at::Tensor table_value_dims,
                                    at::Tensor table_emb_dims, const float lr,
                                    const float eps,
                                    int64_t max_emb_dim, bool all_dims_vec4,
                                    int64_t table_dtype);

void sgd_update_for_padded_buffer(at::Tensor grads, at::Tensor values,
                                  int64_t emb_dim, int64_t value_dim,
                                  float lr);

void adam_update_for_padded_buffer(at::Tensor grads, at::Tensor values,
                                   int64_t emb_dim, int64_t value_dim,
                                   float lr, float beta1, float beta2,
                                   float eps, float weight_decay,
                                   uint32_t iter_num);

void adagrad_update_for_padded_buffer(at::Tensor grads, at::Tensor values,
                                      int64_t emb_dim, int64_t value_dim,
                                      float lr, float eps);

void rowwise_adagrad_for_padded_buffer(at::Tensor grads, at::Tensor values,
                                       int64_t emb_dim, int64_t value_dim,
                                       float lr, float eps);

} // namespace dyn_emb
#endif // OPTIMIZER_H
