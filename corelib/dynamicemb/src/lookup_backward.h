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

#ifndef LOOKUP_BACKWARD_H
#define LOOKUP_BACKWARD_H
#include "index_calculation.h"
#include "utils.h"

namespace dyn_emb {

class LocalReduce {
private:
  c10::Device device_;
  int64_t num_key_;
  int64_t len_vec_;
  DataType key_type_;
  DataType id_type_;
  DataType accum_type_;

  at::Tensor partial_buffer;
  at::Tensor partial_unique_ids;

  static constexpr int32_t WarpSize = 32;

public:
  LocalReduce(c10::Device &device, int64_t num_key, int64_t len_vec,
              DataType id_type, DataType accum_type);

  void local_reduce(const at::Tensor &in_grad, at::Tensor &out_grad,
                    const at::Tensor &sorted_key_ids,
                    const at::Tensor &unique_key_ids, cudaStream_t &stream);
};

void backward(void *grads, void *unique_buffer, void *unique_indices,
              void *inverse_indices, void *biased_offset, const int dim,
              const int batch_size, const int feature_num, const int num_key,
              int combiner, DataType key_type, DataType value_type,
              cudaStream_t stream);
void one_to_one_atomic(void *grads, void *unique_indices, void *reverse_indices,
                       void *unique_grads, const int ev_size,
                       const int64_t key_num, const int64_t unique_key_num,
                       DataType rev_idx_type, DataType grad_type,
                       DataType key_type, int num_sms, cudaStream_t stream);
} // namespace dyn_emb
#endif // LOOKUP_BACKWARD_H
