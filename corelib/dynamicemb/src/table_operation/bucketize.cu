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

#include "../index_calculation.h"
#include "kernels.cuh"
#include "table.cuh"

namespace {

template <typename KeyType, typename IndexType> struct SegmentedKey {
  struct Decomposer {
    __host__ __device__ cuda::std::tuple<IndexType &, KeyType &>
    operator()(SegmentedKey<KeyType, IndexType> &segmented_key) const {
      return {segmented_key.segment_id, segmented_key.key};
    }
  };
  IndexType segment_id;
  KeyType key;
} __attribute__((packed));

template <typename KeyType, typename IndexType, typename BucketType>
struct BucketizeFunctor {

  __host__ __device__ __forceinline__ IndexType operator()(KeyType key,
                                                           IndexType tid) {

    uint64_t hashcode = BucketType::hash(key);
    uint64_t global_idx =
        static_cast<uint64_t>(hashcode % (bucket_capacity * num_buckets));
    uint64_t bucket_id = global_idx / bucket_capacity;
    return static_cast<IndexType>(bucket_id);
  }
  int64_t bucket_capacity;
  int64_t num_buckets;
};

template <typename KeyType, typename IndexType, typename ComposeKey>
struct BucketizeOutput {

  __host__ __device__ __forceinline__ void
  operator()(ComposeKey *compose_keys, IndexType tid, IndexType segment_id) {
    if (tid + 1 < batch) {
      if (compose_keys[tid + 1].segment_id != segment_id) {
        buckets_end[segment_id] = tid + 1;
        mask[segment_id] = true;
      }
    } else if (tid + 1 == batch) {
      buckets_end[segment_id] = tid + 1;
      mask[segment_id] = true;
    }
  }
  int64_t batch;
  bool *__restrict__ mask;
  int64_t *__restrict__ buckets_end;
};

template <typename KeyType, typename IndexType, typename ComposeKey,
          typename SegmentFunctor>
__global__ void compose_segmented_key_kernel(
    int64_t const num_total, KeyType const *__restrict__ keys,
    ComposeKey *__restrict__ compose_keys, SegmentFunctor func) {

  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (; tid < num_total; tid += blockDim.x * gridDim.x) {
    KeyType key = keys[tid];
    IndexType segment_id = func(key, tid);

    ComposeKey compose;
    compose.segment_id = segment_id;
    compose.key = key;
    compose_keys[tid] = compose;
  }
}

template <typename KeyType, typename IndexType, typename ComposeKey,
          typename SegmentOutput>
__global__ void decompose_segmented_key_kernel(
    int64_t const num_total, ComposeKey *__restrict__ compose_keys,
    KeyType *__restrict__ keys, SegmentOutput out_func) {

  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (; tid < num_total; tid += blockDim.x * gridDim.x) {
    ComposeKey compose = compose_keys[tid];
    keys[tid] = compose.key;
    out_func(compose_keys, tid, compose.segment_id);
  }
}

} // anonymous namespace

namespace dyn_emb {

std::vector<at::Tensor> bucketize_keys(at::Tensor keys, int64_t bucket_capacity,
                                       int64_t num_buckets) {

  int64_t num_total = keys.size(0);
  std::vector<at::Tensor> result;
  result.reserve(3);
  if (num_total == 0) {
    result.push_back(at::empty(
        0, at::TensorOptions().dtype(at::kLong).device(keys.device())));
    result.push_back(at::empty(
        0, at::TensorOptions().dtype(at::kLong).device(keys.device())));
    result.push_back(at::empty(
        0, at::TensorOptions().dtype(at::kLong).device(keys.device())));
    return result;
  }

  auto buckets_end = at::zeros(
      num_buckets, at::TensorOptions().dtype(at::kLong).device(keys.device()));
  auto offsets =
      at::zeros(num_buckets + 1,
                at::TensorOptions().dtype(at::kLong).device(keys.device()));
  auto mask = at::zeros(
      num_buckets, at::TensorOptions().dtype(at::kBool).device(keys.device()));
  auto natural = at::arange(
      0, num_total, at::TensorOptions().dtype(at::kLong).device(keys.device()));
  auto inverse = at::empty(
      num_total, at::TensorOptions().dtype(at::kLong).device(keys.device()));

  auto key_type = get_data_type(keys);

  auto keys_out = at::empty_like(keys);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  constexpr int BLOCK_SIZE = 256;

  using IndexType = int64_t;

  auto natural_ = get_pointer<IndexType>(natural);
  auto inverse_ = get_pointer<IndexType>(inverse);

  auto buckets_end_ = get_pointer<IndexType>(buckets_end);
  auto offsets_ = get_pointer<IndexType>(offsets);
  auto mask_ = mask.data_ptr<bool>();

  auto num_active_buckets =
      at::zeros(1, at::TensorOptions().dtype(at::kLong).device(keys.device()));

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    auto keys_ = get_pointer<KeyType>(keys);

    using Bucket = LinearBucket<KeyType>;
    using ComposeKey = SegmentedKey<KeyType, IndexType>;

    // 1.  compose keys
    using SegmentFunc = BucketizeFunctor<KeyType, IndexType, Bucket>;
    SegmentFunc func{bucket_capacity, num_buckets};
    auto compose_keys_in = at::empty(
        {num_total * static_cast<int64_t>(sizeof(ComposeKey))},
        at::TensorOptions().dtype(torch::kChar).device(keys.device()));

    auto compose_keys_in_ = get_pointer<ComposeKey>(compose_keys_in);

    compose_segmented_key_kernel<KeyType, IndexType, ComposeKey, SegmentFunc>
        <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            num_total, keys_, compose_keys_in_, func);

    // 2. sort compose keys(bucket_id, key)
    uint64_t cub_sort_temp_bytes_ = 0;
    cub::DeviceRadixSort::SortPairs<ComposeKey, IndexType>(
        nullptr, cub_sort_temp_bytes_, nullptr, nullptr, nullptr, nullptr,
        num_total, ComposeKey::Decomposer{}, 0, sizeof(ComposeKey) * 8);
    auto cub_sort_temp_buffer_ = at::empty(
        {static_cast<int64_t>(cub_sort_temp_bytes_)},
        at::TensorOptions().dtype(torch::kChar).device(keys.device()));
    auto compose_keys_out = at::empty(
        {num_total * static_cast<int64_t>(sizeof(ComposeKey))},
        at::TensorOptions().dtype(torch::kChar).device(keys.device()));

    auto compose_keys_out_ = get_pointer<ComposeKey>(compose_keys_out);
    cub::DeviceRadixSort::SortPairs<ComposeKey, IndexType>(
        cub_sort_temp_buffer_.data_ptr(), cub_sort_temp_bytes_,
        compose_keys_in_, compose_keys_out_, natural_, inverse_, num_total,
        ComposeKey::Decomposer{}, 0, sizeof(ComposeKey) * 8, stream);

    // 3. decompose sorted keys
    using SegmentOutput = BucketizeOutput<KeyType, IndexType, ComposeKey>;
    SegmentOutput out_func{num_total, mask_, buckets_end_};
    auto keys_out_ = get_pointer<KeyType>(keys_out);
    decompose_segmented_key_kernel<KeyType, IndexType, ComposeKey,
                                   SegmentOutput>
        <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            num_total, compose_keys_out_, keys_out_, out_func);

    // 4. get offsets
    select_async<IndexType, IndexType>(
        num_buckets, mask_, buckets_end_, offsets_ + 1,
        num_active_buckets.data_ptr<int64_t>(), keys.device(), stream);
  });

  int64_t num_active_buckets_ = num_active_buckets.item<int64_t>();

  result.push_back(keys_out);
  result.push_back(offsets.slice(/*dim=*/0, /*start=*/0,
                                 /*end=*/num_active_buckets_ + 1, /*step=*/1));
  result.push_back(inverse);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

} // namespace dyn_emb
