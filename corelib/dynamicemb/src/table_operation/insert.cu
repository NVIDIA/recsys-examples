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

template <typename Table, ScorePolicyType PolicyTypeV, bool OutputScoreV>
void launch_table_insert_kernel(
    Table table, int64_t *table_bucket_offsets_ptr,
    int *bucket_sizes_ptr, int64_t num_total,
    typename Table::KeyType *keys_ptr, int64_t *table_ids_ptr,
    InsertResult *insert_results_ptr,
    IndexType *indices_ptr, ScoreType *score_input_ptr,
    int64_t *score_output_ptr, typename Table::KeyType **table_key_slots_ptr,
    int32_t *counter_ptr,
    cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  using KernelTraits =
      InsertKernelTraits<BLOCK_SIZE, 1, 1, 1, 8, PolicyTypeV, OutputScoreV>;

  table_insert_kernel<Table, KernelTraits>
      <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
          table, table_bucket_offsets_ptr,
          bucket_sizes_ptr, num_total, keys_ptr, table_ids_ptr,
          insert_results_ptr,
          indices_ptr, score_input_ptr, score_output_ptr, table_key_slots_ptr,
          counter_ptr);

  table_unlock_kernel<Table>
      <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
          table, num_total, keys_ptr, table_key_slots_ptr);
}

void table_insert_single_score(at::Tensor table_storage,
                               at::Tensor table_bucket_offsets,
                               int64_t bucket_capacity, at::Tensor bucket_sizes,
                               at::Tensor keys, at::Tensor table_ids,
                               std::optional<at::Tensor> score_input,
                               ScorePolicyType policy_type, at::Tensor indices,
                               std::optional<at::Tensor> insert_results,
                               std::optional<at::Tensor> score_output,
                               at::Tensor counter) {

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

  int64_t *score_output_ptr = nullptr;
  if (score_output.has_value() && score_output.value().defined()) {
    score_output_ptr = score_output.value().data_ptr<int64_t>();
  }

  auto indices_ptr = indices.data_ptr<IndexType>();
  InsertResult *insert_results_ptr = get_pointer<InsertResult>(insert_results);
  auto bucket_sizes_ptr = get_pointer<int>(bucket_sizes);
  auto table_ids_ptr = table_ids.data_ptr<int64_t>();
  auto table_bucket_offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  auto counter_ptr = counter.data_ptr<int32_t>();

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int64_t num_total = keys.size(0);

  auto table_key_slots = at::zeros(
      num_total, at::TensorOptions().dtype(at::kLong).device(keys.device()));

  bool output_score = (score_output_ptr != nullptr);

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    auto keys_ptr = get_pointer<KeyType>(keys);
    auto table_key_slots_ptr = get_pointer<KeyType *>(table_key_slots);

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
      if (output_score) {
        launch_table_insert_kernel<Table, PolicyTypeV, true>(
            table, table_bucket_offsets_ptr,
            bucket_sizes_ptr, num_total, keys_ptr, table_ids_ptr,
            insert_results_ptr,
            indices_ptr, score_input_ptr, score_output_ptr, table_key_slots_ptr,
            counter_ptr, stream);
      } else {
        launch_table_insert_kernel<Table, PolicyTypeV, false>(
            table, table_bucket_offsets_ptr,
            bucket_sizes_ptr, num_total, keys_ptr, table_ids_ptr,
            insert_results_ptr,
            indices_ptr, score_input_ptr, nullptr, table_key_slots_ptr,
            counter_ptr, stream);
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor table_insert(at::Tensor table_storage, at::Tensor table_bucket_offsets,
                        int64_t bucket_capacity,
                        at::Tensor bucket_sizes, at::Tensor keys,
                        at::Tensor table_ids,
                        std::optional<at::Tensor> score_input,
                        ScorePolicyType policy_type,
                        at::Tensor counter,
                        std::optional<at::Tensor> insert_results,
                        std::optional<at::Tensor> score_output) {

  int64_t num_total = keys.size(0);
  if (num_total == 0) {
    return torch::empty({0}, keys.options().dtype(torch::kInt64));
  }

  at::Tensor indices =
      torch::empty({num_total}, keys.options().dtype(torch::kInt64));

  table_insert_single_score(table_storage, table_bucket_offsets,
                            bucket_capacity, bucket_sizes, keys, table_ids,
                            score_input, policy_type, indices, insert_results,
                            score_output, counter);

  return indices;
}

} // namespace dyn_emb
