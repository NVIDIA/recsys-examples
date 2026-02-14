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

void table_export_single_score(at::Tensor table_storage,
                               int64_t bucket_capacity, int64_t batch,
                               int64_t offset, at::Tensor counter,
                               at::Tensor keys, at::Tensor score,
                               std::optional<ScoreType> threshold,
                               at::Tensor indices) {
  auto key_type = get_data_type(keys);
  auto scores_ = reinterpret_cast<ScoreType *>(score.data_ptr<int64_t>());
  auto indices_ = indices.data_ptr<IndexType>();
  auto counter_ = get_pointer<CounterType>(counter);

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  bool score_pred = threshold.has_value();
  ScoreType threshold_val = score_pred ? threshold.value() : 0;

  int64_t num_total = batch;

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

    if (offset + num_total > num_buckets * bucket_capacity) {
      throw std::invalid_argument("Offset and batch size overflow.");
    }

    if (num_total % 32 == 0) {
      DISPATCH_BOOLEAN(score_pred, PredFlag, [&] {
        using PredFunc = ExportPredFunctor<PredFlag>;
        PredFunc pred(threshold_val);
        table_export_batch_kernel<Table, PredFunc, 32>
            <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
               stream>>>(table, offset, offset + num_total, counter_, keys_,
                         scores_, pred, indices_);
      });

    } else {
      DISPATCH_BOOLEAN(score_pred, PredFlag, [&] {
        using PredFunc = ExportPredFunctor<PredFlag>;
        PredFunc pred(threshold_val);
        table_export_batch_kernel<Table, PredFunc, 1>
            <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
               stream>>>(table, offset, offset + num_total, counter_, keys_,
                         scores_, pred, indices_);
      });
    }
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
table_export_batch(at::Tensor table_storage, int64_t bucket_capacity,
                   int64_t batch, int64_t offset, torch::Dtype key_dtype,
                   std::optional<ScoreType> threshold) {
  auto device = table_storage.device();
  auto key_scalar_type = static_cast<torch::ScalarType>(key_dtype);

  auto counter = torch::zeros(
      {1}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto keys = torch::empty(
      {batch}, torch::TensorOptions().dtype(key_scalar_type).device(device));
  auto score = torch::empty(
      {batch}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto indices = torch::empty(
      {batch}, torch::TensorOptions().dtype(torch::kInt64).device(device));

  if (batch == 0)
    return std::make_tuple(counter, keys, score, indices);

  table_export_single_score(table_storage, bucket_capacity, batch, offset,
                            counter, keys, score, threshold, indices);

  return std::make_tuple(counter, keys, score, indices);
}
} // namespace dyn_emb
