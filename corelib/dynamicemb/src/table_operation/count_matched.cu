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

#include "kernels.cuh"
#include "table.cuh"

namespace dyn_emb {

void table_count_matched_single_score(at::Tensor table_storage,
                                      torch::Dtype key_dtype,
                                      int64_t bucket_capacity,
                                      ScoreType threshold,
                                      at::Tensor num_matched,
                                      int64_t range_begin,
                                      int64_t range_end) {

  auto key_type = scalartype_to_datatype(toScalarType(key_dtype));
  auto counter_ = get_pointer<CounterType>(num_matched);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  EvalAndCount func(threshold, counter_);

  constexpr int BLOCK_SIZE = 256;

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    constexpr int64_t total_size =
        sizeof(KeyType) + sizeof(DigestType) + sizeof(ScoreType);
    int64_t bucket_bytes = bucket_capacity * total_size;
    int64_t num_buckets =
        table_storage.numel() * table_storage.element_size() / bucket_bytes;

    using Bucket = LinearBucket<KeyType>;
    using Table = LinearBucketTable<Bucket>;

    auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                       num_buckets, bucket_capacity);

    IndexType begin = (range_begin >= 0) ? range_begin : 0;
    IndexType end = (range_end >= 0) ? range_end : num_buckets * bucket_capacity;

    int64_t num_total = end - begin;

    if (num_total <= 0) return;

    if (num_total % 32 == 0) {
      table_traverse_kernel<Table, EvalAndCount, 32>
          <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
             stream>>>(table, begin, end, func);

    } else {
      table_traverse_kernel<Table, EvalAndCount, 1>
          <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
             stream>>>(table, begin, end, func);
    }
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor table_count_matched(at::Tensor table_storage, torch::Dtype key_dtype,
                               int64_t bucket_capacity, ScoreType threshold,
                               int64_t begin, int64_t end) {

  auto device = table_storage.device();
  auto num_matched = torch::zeros(
      {1}, torch::TensorOptions().dtype(torch::kInt64).device(device));

  table_count_matched_single_score(table_storage, key_dtype, bucket_capacity,
                                   threshold, num_matched, begin, end);

  return num_matched;
}
} // namespace dyn_emb
