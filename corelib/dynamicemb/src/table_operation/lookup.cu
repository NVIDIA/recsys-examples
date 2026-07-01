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

void table_lookup_single_score(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor keys, at::Tensor table_ids,
    std::optional<at::Tensor> score_input, ScorePolicyType policy_type,
    at::Tensor score_output, at::Tensor founds, at::Tensor indices,
    std::optional<at::Tensor> active_mask,
    std::optional<at::Tensor> active_count,
    std::optional<at::Tensor> acquire_counter) {

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
  auto table_ids_ptr = table_ids.data_ptr<int64_t>();
  auto table_bucket_offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  auto active_mask_ptr = get_pointer<bool>(active_mask);
  auto active_count_ptr = get_pointer<int64_t>(active_count);
  auto acquire_counter_ptr = get_pointer<int32_t>(acquire_counter);

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
      table_lookup_kernel<Table, 1, PolicyTypeV, false>
          <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
             stream>>>(table, table_bucket_offsets_ptr, num_total, keys_ptr,
                       table_ids_ptr, founds_ptr, indices_ptr, score_input_ptr,
                       score_output_ptr, Table(), nullptr, active_mask_ptr,
                       active_count_ptr, acquire_counter_ptr, nullptr);
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

static void table_lookup_with_overflow_single_score(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor keys, at::Tensor table_ids,
    std::optional<at::Tensor> score_input, ScorePolicyType policy_type,
    at::Tensor score_output, at::Tensor founds, at::Tensor indices,
    at::Tensor ovf_storage, int64_t ovf_bucket_capacity,
    at::Tensor ovf_output_offsets, std::optional<at::Tensor> active_mask,
    std::optional<at::Tensor> active_count,
    std::optional<at::Tensor> acquire_counter,
    std::optional<at::Tensor> acquire_ovf_counter) {

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
  auto table_ids_ptr = table_ids.data_ptr<int64_t>();
  auto table_bucket_offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  auto ovf_output_offsets_ptr = ovf_output_offsets.data_ptr<int64_t>();
  auto active_mask_ptr = get_pointer<bool>(active_mask);
  auto active_count_ptr = get_pointer<int64_t>(active_count);
  auto acquire_counter_ptr = get_pointer<int32_t>(acquire_counter);
  auto acquire_ovf_counter_ptr = get_pointer<int32_t>(acquire_ovf_counter);

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

    int64_t ovf_bucket_bytes = ovf_bucket_capacity * total_size;
    int64_t ovf_num_buckets =
        ovf_storage.numel() * ovf_storage.element_size() / ovf_bucket_bytes;

    auto ovf_table = Table(reinterpret_cast<uint8_t *>(ovf_storage.data_ptr()),
                           ovf_num_buckets, ovf_bucket_capacity);

    DISPATCH_SCORE_POLICY(policy_type, PolicyTypeV, [&] {
      table_lookup_kernel<Table, 1, PolicyTypeV, true>
          <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
             stream>>>(table, table_bucket_offsets_ptr, num_total, keys_ptr,
                       table_ids_ptr, founds_ptr, indices_ptr, score_input_ptr,
                       score_output_ptr, ovf_table, ovf_output_offsets_ptr,
                       active_mask_ptr, active_count_ptr, acquire_counter_ptr,
                       acquire_ovf_counter_ptr);
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
table_lookup(at::Tensor table_storage, at::Tensor table_bucket_offsets,
             int64_t bucket_capacity, at::Tensor keys, at::Tensor table_ids,
             std::optional<at::Tensor> score_input, ScorePolicyType policy_type,
             std::optional<at::Tensor> ovf_storage, int64_t ovf_bucket_capacity,
             std::optional<at::Tensor> ovf_output_offsets,
             std::optional<at::Tensor> active_mask,
             std::optional<at::Tensor> active_count,
             std::optional<at::Tensor> score_output_arg,
             std::optional<at::Tensor> founds_output_arg,
             std::optional<at::Tensor> indices_output_arg,
             std::optional<at::Tensor> acquire_counter,
             std::optional<at::Tensor> acquire_ovf_counter) {

  int64_t num_total = keys.size(0);
  bool sparse = (active_mask.has_value() && active_mask->defined()) ||
                (active_count.has_value() && active_count->defined());
  at::Tensor score_output =
      (score_output_arg.has_value() && score_output_arg->defined())
          ? score_output_arg.value()
          : (sparse ? torch::zeros({num_total}, keys.options().dtype(torch::kInt64))
                    : torch::empty({num_total}, keys.options().dtype(torch::kInt64)));
  at::Tensor founds =
      (founds_output_arg.has_value() && founds_output_arg->defined())
          ? founds_output_arg.value()
          : (sparse ? torch::zeros({num_total}, keys.options().dtype(torch::kBool))
                    : torch::empty({num_total}, keys.options().dtype(torch::kBool)));
  at::Tensor indices =
      (indices_output_arg.has_value() && indices_output_arg->defined())
          ? indices_output_arg.value()
          : (sparse ? torch::full({num_total}, -1,
                                  keys.options().dtype(torch::kInt64))
                    : torch::empty({num_total},
                                   keys.options().dtype(torch::kInt64)));
  TORCH_CHECK(score_output.numel() == num_total && founds.numel() == num_total &&
                  indices.numel() == num_total,
              "lookup output tensors must match keys length");
  if (num_total == 0)
    return std::make_tuple(score_output, founds, indices);

  bool use_overflow = ovf_storage.has_value() && ovf_storage.value().defined();

  if (use_overflow) {
    table_lookup_with_overflow_single_score(
        table_storage, table_bucket_offsets, bucket_capacity, keys, table_ids,
        score_input, policy_type, score_output, founds, indices,
        ovf_storage.value(), ovf_bucket_capacity, ovf_output_offsets.value(),
        active_mask, active_count, acquire_counter, acquire_ovf_counter);
  } else {
    table_lookup_single_score(table_storage, table_bucket_offsets,
                              bucket_capacity, keys, table_ids, score_input,
                              policy_type, score_output, founds, indices,
                              active_mask, active_count, acquire_counter);
  }

  return std::make_tuple(score_output, founds, indices);
}

} // namespace dyn_emb
