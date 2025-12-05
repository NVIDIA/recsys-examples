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
                               std::vector<torch::Dtype> dtypes,
                               int64_t bucket_capacity, int64_t batch,
                               int64_t offset, at::Tensor counter,
                               at::Tensor keys,
                               std::vector<std::optional<at::Tensor>> scores,
                               std::optional<std::vector<ScoreType>> thresholds,
                               std::optional<at::Tensor> indices) {
  auto key_type = get_data_type(keys);
  auto scores_ = get_pointer<ScoreType>(scores[0]);
  auto indices_ = get_pointer<IndexType>(indices);
  auto counter_ = get_pointer<CounterType>(counter);

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  bool score_pred = thresholds.has_value();
  ScoreType threshold = score_pred ? thresholds.value()[0] : 0;

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
        PredFunc pred(threshold);
        table_export_batch_kernel<Table, PredFunc, 32>
            <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
               stream>>>(table, offset, offset + num_total, counter_, keys_,
                         scores_, pred, indices_);
      });

    } else {
      DISPATCH_BOOLEAN(score_pred, PredFlag, [&] {
        using PredFunc = ExportPredFunctor<PredFlag>;
        PredFunc pred(threshold);
        table_export_batch_kernel<Table, PredFunc, 1>
            <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
               stream>>>(table, offset, offset + num_total, counter_, keys_,
                         scores_, pred, indices_);
      });
    }
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void table_export_batch(at::Tensor table_storage,
                        std::vector<torch::Dtype> dtypes,
                        int64_t bucket_capacity, int64_t batch, int64_t offset,
                        at::Tensor counter, at::Tensor keys,
                        std::vector<std::optional<at::Tensor>> scores,
                        std::optional<std::vector<ScoreType>> thresholds,
                        std::optional<at::Tensor> indices) {
  if (batch == 0)
    return;

  if (scores.size() == 1) {
    table_export_single_score(table_storage, dtypes, bucket_capacity, batch,
                              offset, counter, keys, scores, thresholds,
                              indices);
  } else {
    throw std::runtime_error("Not support multi-scores.");
  }
}
} // namespace dyn_emb