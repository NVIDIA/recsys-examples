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
                               int64_t bucket_capacity, at::Tensor keys,
                               std::optional<at::Tensor> score_input,
                               ScorePolicyType policy_type,
                               at::Tensor score_output, at::Tensor founds,
                               at::Tensor indices) {

  auto key_type = get_data_type(keys);

  ScoreType *score_input_ptr = nullptr;
  at::Tensor score_input_tensor;
  if (score_input.has_value() && score_input.value().defined()) {
    at::Tensor in = score_input.value();
    if (in.scalar_type() == torch::kUInt64) {
      score_input_ptr = get_pointer<ScoreType>(score_input);
    } else {
      score_input_tensor = in.view(torch::kUInt64);
      score_input_ptr = score_input_tensor.data_ptr<ScoreType>();
    }
  }

  auto score_output_ptr = score_output.data_ptr<int64_t>();
  auto indices_ptr = indices.data_ptr<IndexType>();
  auto founds_ptr = founds.data_ptr<bool>();

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int64_t num_total = keys.size(0);

  constexpr int BLOCK_SIZE = 256;

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    auto keys_ptr = get_pointer<KeyType>(keys);

    constexpr int64_t total_size =
        sizeof(KeyType) + sizeof(DigestType) + sizeof(ScoreType);
    int64_t bucket_bytes = bucket_capacity * total_size;
    int64_t num_buckets =
        table_storage.numel() * table_storage.element_size() / bucket_bytes;

    using Bucket = LinearBucket<KeyType>;
    using Table = LinearBucketTable<Bucket>;

    auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                       num_buckets, bucket_capacity);

    DISPATCH_SCORE_POLICY(policy_type, PolicyTypeV, [&] {
      table_lookup_kernel<Table, 1, PolicyTypeV>
          <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
             stream>>>(table, num_total, keys_ptr, founds_ptr, indices_ptr,
                       score_input_ptr, score_output_ptr);
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
table_lookup(at::Tensor table_storage, int64_t bucket_capacity, at::Tensor keys,
             std::optional<at::Tensor> score_input,
             ScorePolicyType policy_type) {

  int64_t num_total = keys.size(0);
  if (num_total == 0) {
    at::Tensor score_output =
        torch::empty({0}, keys.options().dtype(torch::kInt64));
    at::Tensor founds = torch::empty({0}, keys.options().dtype(torch::kBool));
    at::Tensor indices = torch::empty({0}, keys.options().dtype(torch::kInt64));
    return std::make_tuple(score_output, founds, indices);
  }

  at::Tensor score_output =
      torch::empty({num_total}, keys.options().dtype(torch::kInt64));
  at::Tensor founds =
      torch::empty({num_total}, keys.options().dtype(torch::kBool));
  at::Tensor indices =
      torch::empty({num_total}, keys.options().dtype(torch::kInt64));

  table_lookup_single_score(table_storage, bucket_capacity, keys, score_input,
                            policy_type, score_output, founds, indices);

  return std::make_tuple(score_output, founds, indices);
}

} // namespace dyn_emb
