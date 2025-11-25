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

void table_lookup_single_score(at::Tensor table_storage,
                               std::vector<torch::Dtype> dtypes,
                               int64_t bucket_capacity, at::Tensor keys,
                               std::vector<std::optional<at::Tensor>> scores,
                               std::vector<ScorePolicyType> policy_types,
                               std::vector<bool> is_returns, at::Tensor founds,
                               std::optional<at::Tensor> indices) {

  auto key_type = get_data_type(keys);

  bool is_return = is_returns[0];
  ScorePolicyType policy_type = policy_types[0];
  auto scores_ = get_pointer<ScoreType>(scores[0]);
  auto indices_ = get_pointer<IndexType>(indices);
  auto founds_ = founds.data_ptr<bool>();

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int64_t num_total = keys.size(0);

  constexpr int BLOCK_SIZE = 256;

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    auto keys_ = get_pointer<KeyType>(keys);

    constexpr int64_t total_size =
        sizeof(KeyType) + sizeof(DigestType) + sizeof(ScoreType);
    int64_t bucket_bytes = bucket_capacity * total_size;
    int64_t num_buckets =
        table_storage.numel() * table_storage.element_size() / bucket_bytes;

    using Bucket = LinearBucket<KeyType>;
    using Table = LinearBucketTable<Bucket>;

    auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                       num_buckets, bucket_capacity);

    table_lookup_kernel<Table, 1>
        <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            table, num_total, keys_, founds_, indices_, scores_, policy_type,
            is_return);
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void table_lookup(at::Tensor table_storage, std::vector<torch::Dtype> dtypes,
                  int64_t bucket_capacity, at::Tensor keys,
                  std::vector<std::optional<at::Tensor>> scores,
                  std::vector<ScorePolicyType> policy_types,
                  std::vector<bool> is_returns, at::Tensor founds,
                  std::optional<at::Tensor> indices) {

  int64_t num_total = keys.size(0);
  if (num_total == 0)
    return;

  if (scores.size() == 1) {
    table_lookup_single_score(table_storage, dtypes, bucket_capacity, keys,
                              scores, policy_types, is_returns, founds,
                              indices);
  } else {
    throw std::runtime_error("Not support multi-scores.");
  }
}

} // namespace dyn_emb