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
    Table table, int64_t *table_bucket_offsets_ptr, int *bucket_sizes_ptr,
    int64_t num_total, typename Table::KeyType *keys_ptr,
    int64_t *table_ids_ptr, InsertResult *insert_results_ptr,
    IndexType *indices_ptr, ScoreType *score_input_ptr,
    int64_t *score_output_ptr, typename Table::KeyType **table_key_slots_ptr,
    int32_t *counter_ptr, cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  using KernelTraits =
      InsertKernelTraits<BLOCK_SIZE, 1, 1, 1, 8, PolicyTypeV, OutputScoreV>;

  table_insert_kernel<Table, KernelTraits>
      <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
          table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
          keys_ptr, table_ids_ptr, insert_results_ptr, indices_ptr,
          score_input_ptr, score_output_ptr, table_key_slots_ptr, counter_ptr);

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
                               at::Tensor counter, int64_t num_scores) {

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

    int64_t total_size =
        sizeof(KeyType) + sizeof(DigestType) + num_scores * sizeof(ScoreType);
    int64_t bucket_bytes = bucket_capacity * total_size;
    int64_t num_buckets =
        table_storage.numel() * table_storage.element_size() / bucket_bytes;

    using Bucket = LinearBucket<KeyType>;
    using Table = LinearBucketTable<Bucket>;

    auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                       num_buckets, bucket_capacity, num_scores);

    DISPATCH_SCORE_POLICY(policy_type, PolicyTypeV, [&] {
      if (output_score) {
        launch_table_insert_kernel<Table, PolicyTypeV, true>(
            table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
            keys_ptr, table_ids_ptr, insert_results_ptr, indices_ptr,
            score_input_ptr, score_output_ptr, table_key_slots_ptr, counter_ptr,
            stream);
      } else {
        launch_table_insert_kernel<Table, PolicyTypeV, false>(
            table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
            keys_ptr, table_ids_ptr, insert_results_ptr, indices_ptr,
            score_input_ptr, nullptr, table_key_slots_ptr, counter_ptr, stream);
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor table_insert(at::Tensor table_storage,
                        at::Tensor table_bucket_offsets,
                        int64_t bucket_capacity, at::Tensor bucket_sizes,
                        at::Tensor keys, at::Tensor table_ids,
                        std::optional<at::Tensor> score_input,
                        ScorePolicyType policy_type, at::Tensor counter,
                        std::optional<at::Tensor> insert_results,
                        std::optional<at::Tensor> score_output,
                        int64_t num_scores) {

  int64_t num_total = keys.size(0);
  if (num_total == 0) {
    return torch::empty({0}, keys.options().dtype(torch::kInt64));
  }

  at::Tensor indices =
      torch::empty({num_total}, keys.options().dtype(torch::kInt64));

  table_insert_single_score(table_storage, table_bucket_offsets,
                            bucket_capacity, bucket_sizes, keys, table_ids,
                            score_input, policy_type, indices, insert_results,
                            score_output, counter, num_scores);

  return indices;
}

// Copy all score words for aligned (src_slot, dst_slot) pairs between two tables.
// Used by rehash to preserve multi-word score layouts (e.g. LruLfu) that a
// single-value re-insert cannot restore. Slots are table-relative flat indices;
// *_bkt_begin is each table's first global bucket for the logical table.
void table_copy_score_blocks(at::Tensor src_storage, int64_t src_bucket_capacity,
                             at::Tensor dst_storage, int64_t dst_bucket_capacity,
                             int64_t num_scores, int64_t src_bkt_begin,
                             int64_t dst_bkt_begin, at::Tensor src_slots,
                             at::Tensor dst_slots, torch::Dtype key_dtype) {
  int64_t n = src_slots.size(0);
  if (n == 0)
    return;
  auto key_type = scalartype_to_datatype(toScalarType(key_dtype));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr int BLOCK_SIZE = 256;

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    using Bucket = LinearBucket<KeyType>;
    using Table = LinearBucketTable<Bucket>;
    int64_t total_size =
        sizeof(KeyType) + sizeof(DigestType) + num_scores * sizeof(ScoreType);
    int64_t src_num_buckets = src_storage.numel() * src_storage.element_size() /
                              (src_bucket_capacity * total_size);
    int64_t dst_num_buckets = dst_storage.numel() * dst_storage.element_size() /
                              (dst_bucket_capacity * total_size);
    auto src_table = Table(reinterpret_cast<uint8_t *>(src_storage.data_ptr()),
                           src_num_buckets, src_bucket_capacity, num_scores);
    auto dst_table = Table(reinterpret_cast<uint8_t *>(dst_storage.data_ptr()),
                           dst_num_buckets, dst_bucket_capacity, num_scores);
    copy_score_blocks_kernel<Table>
        <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            src_table, src_bkt_begin, dst_table, dst_bkt_begin, n,
            src_slots.data_ptr<int64_t>(), dst_slots.data_ptr<int64_t>());
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace dyn_emb
