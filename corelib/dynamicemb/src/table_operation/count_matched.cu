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
                                      std::vector<torch::Dtype> dtypes,
                                      int64_t bucket_capacity,
                                      std::vector<ScoreType> thresholds,
                                      at::Tensor num_matched) {

  auto key_type = scalartype_to_datatype(toScalarType(dtypes[0]));
  auto counter_ = get_pointer<CounterType>(num_matched);

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  ScoreType threshold = thresholds[0];

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

    IndexType begin = 0;
    IndexType end = num_buckets * bucket_capacity;

    int64_t num_total = end - begin;

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

void table_count_matched(at::Tensor table_storage,
                         std::vector<torch::Dtype> dtypes,
                         int64_t bucket_capacity,
                         std::vector<ScoreType> thresholds,
                         at::Tensor num_matched) {

  if (thresholds.size() == 1) {
    table_count_matched_single_score(table_storage, dtypes, bucket_capacity,
                                     thresholds, num_matched);
  } else {
    throw std::runtime_error("Not support multi-scores.");
  }
}
} // namespace dyn_emb