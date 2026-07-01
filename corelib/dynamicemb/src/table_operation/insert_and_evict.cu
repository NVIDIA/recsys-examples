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

// Converts per-table slot_indices (and optional table_ids) to flat counter
// index inside the kernel and adds delta. No atomic: caller guarantees no
// two threads write the same counter element (e.g. slot_indices from unique
// keys).
__global__ void update_counter_with_layout_kernel(
    int32_t *__restrict__ counter, int64_t total_capacity,
    int64_t const *__restrict__ slot_indices, int64_t n, int32_t delta,
    int64_t const *__restrict__ table_ids,
    int64_t const *__restrict__ table_bucket_offsets, int64_t bucket_capacity,
    int64_t main_capacity, int64_t const *__restrict__ overflow_output_offsets,
    int64_t overflow_bucket_capacity, int64_t num_tables) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int64_t slot = slot_indices[i];
  if (slot < 0)
    return;
  int64_t tid = (table_ids != nullptr) ? table_ids[i] : 0;
  int64_t flat = -1;
  if (overflow_output_offsets != nullptr) {
    int64_t per_table_cap = overflow_output_offsets[tid];
    if (slot < per_table_cap) {
      flat = table_bucket_offsets[tid] * bucket_capacity + slot;
    } else {
      flat = main_capacity + tid * overflow_bucket_capacity +
             (slot - per_table_cap);
    }
  } else {
    if (table_bucket_offsets != nullptr && num_tables > 1) {
      flat = table_bucket_offsets[tid] * bucket_capacity + slot;
    } else {
      flat = slot;
    }
  }
  if (flat >= 0 && flat < total_capacity) {
    atomicAdd(counter + flat, delta);
  }
}

template <typename Table, ScorePolicyType PolicyTypeV, bool OutputScoreV,
          int CompactTileSize, bool EnableOverflowV = false,
          bool FindOrInsertV = false>
void launch_table_insert_and_evict_kernel(
    Table table, int64_t *table_bucket_offsets_ptr, int *bucket_sizes_ptr,
    int64_t num_total, typename Table::KeyType *keys_ptr,
    int64_t *table_ids_ptr, InsertResult *insert_results_ptr,
    IndexType *indices_ptr, ScoreType *score_input_ptr,
    int64_t *score_output_ptr, typename Table::KeyType **table_key_slots_ptr,
    CounterType *evict_counter_ptr, typename Table::KeyType *evicted_keys_ptr,
    int64_t *evicted_scores_ptr, IndexType *evicted_indices_ptr,
    int64_t *evicted_table_ids_ptr, int32_t *counter_ptr, cudaStream_t stream,
    Table ovf_table = Table(), int *ovf_bucket_sizes_ptr = nullptr,
    int32_t *ovf_counter_ptr = nullptr,
    int64_t *ovf_output_offsets_ptr = nullptr, bool *founds_ptr = nullptr,
    bool *evicted_mask_ptr = nullptr,
    bool const *active_mask_ptr = nullptr,
    int64_t const *active_count_ptr = nullptr,
    int64_t max_cooperative_blocks = 0) {
  constexpr int BLOCK_SIZE = 256;
  using KernelTraits =
      InsertKernelTraits<BLOCK_SIZE, 1, 1, CompactTileSize, 8, PolicyTypeV,
                         OutputScoreV, EnableOverflowV, FindOrInsertV>;

  auto kernel = table_insert_and_evict_kernel<Table, KernelTraits>;
  if constexpr (FindOrInsertV) {
    TORCH_CHECK(max_cooperative_blocks > 0,
                "find_or_insert requires a positive cooperative grid limit");
    int64_t needed_blocks =
        (num_total + static_cast<int64_t>(BLOCK_SIZE) - 1) / BLOCK_SIZE;
    int grid_size = static_cast<int>(
        needed_blocks < max_cooperative_blocks ? needed_blocks
                                                : max_cooperative_blocks);

    void *args[] = {
        &table,
        &table_bucket_offsets_ptr,
        &bucket_sizes_ptr,
        &num_total,
        &keys_ptr,
        &table_ids_ptr,
        &insert_results_ptr,
        &indices_ptr,
        &score_input_ptr,
        &score_output_ptr,
        &table_key_slots_ptr,
        &evict_counter_ptr,
        &evicted_keys_ptr,
        &evicted_scores_ptr,
        &evicted_indices_ptr,
        &evicted_table_ids_ptr,
        &counter_ptr,
        &ovf_table,
        &ovf_bucket_sizes_ptr,
        &ovf_counter_ptr,
        &ovf_output_offsets_ptr,
        &founds_ptr,
        &evicted_mask_ptr,
        &active_mask_ptr,
        &active_count_ptr,
    };
    CUDACHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(kernel), dim3(grid_size), dim3(BLOCK_SIZE),
        args, 0, stream));
  } else {
    kernel<<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
             stream>>>(table, table_bucket_offsets_ptr, bucket_sizes_ptr,
                       num_total, keys_ptr, table_ids_ptr, insert_results_ptr,
                       indices_ptr, score_input_ptr, score_output_ptr,
                       table_key_slots_ptr, evict_counter_ptr, evicted_keys_ptr,
                       evicted_scores_ptr, evicted_indices_ptr,
                       evicted_table_ids_ptr, counter_ptr, ovf_table,
                       ovf_bucket_sizes_ptr, ovf_counter_ptr,
                       ovf_output_offsets_ptr, founds_ptr,
                       evicted_mask_ptr, active_mask_ptr,
                       active_count_ptr);
  }

  if constexpr (!FindOrInsertV) {
    table_unlock_kernel<Table>
        <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            table, num_total, keys_ptr, table_key_slots_ptr);
  }
}

template <typename Table, ScorePolicyType PolicyTypeV, bool OutputScoreV,
          bool EnableOverflowV>
int64_t query_find_or_insert_max_cooperative_blocks(int num_sms) {
  constexpr int BLOCK_SIZE = 256;
  using KernelTraits =
      InsertKernelTraits<BLOCK_SIZE, 1, 1, 1, 8, PolicyTypeV, OutputScoreV,
                         EnableOverflowV, true>;
  auto kernel = table_insert_and_evict_kernel<Table, KernelTraits>;
  int blocks_per_sm = 0;
  CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, BLOCK_SIZE, 0));
  TORCH_CHECK(blocks_per_sm > 0,
              "find_or_insert cooperative kernel has zero occupancy");
  return static_cast<int64_t>(blocks_per_sm) * num_sms;
}

template <typename Table, ScorePolicyType PolicyTypeV, bool OutputScoreV>
int64_t query_find_or_insert_max_cooperative_blocks(bool enable_overflow,
                                                    int num_sms) {
  if (enable_overflow) {
    return query_find_or_insert_max_cooperative_blocks<
        Table, PolicyTypeV, OutputScoreV, true>(num_sms);
  }
  return query_find_or_insert_max_cooperative_blocks<
      Table, PolicyTypeV, OutputScoreV, false>(num_sms);
}

int64_t table_find_or_insert_max_cooperative_blocks(
    at::Tensor key_type_device_ref, ScorePolicyType policy_type,
    bool output_score, bool enable_overflow) {
  TORCH_CHECK(key_type_device_ref.is_cuda(),
              "find_or_insert launch configuration requires a CUDA tensor");
  c10::cuda::CUDAGuard device_guard(key_type_device_ref.device());
  int device_id = key_type_device_ref.get_device();
  int cooperative_launch = 0;
  CUDACHECK(cudaDeviceGetAttribute(&cooperative_launch,
                                   cudaDevAttrCooperativeLaunch, device_id));
  TORCH_CHECK(cooperative_launch != 0,
              "find_or_insert requires CUDA cooperative launch support");
  int num_sms = 0;
  CUDACHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount,
                                   device_id));
  TORCH_CHECK(num_sms > 0, "find_or_insert device has no multiprocessors");

  auto key_type = get_data_type(key_type_device_ref);
  int64_t max_cooperative_blocks = 0;
  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    using Bucket = LinearBucket<KeyType>;
    using Table = LinearBucketTable<Bucket>;
    DISPATCH_SCORE_POLICY(policy_type, PolicyTypeV, [&] {
      if (output_score) {
        max_cooperative_blocks =
            query_find_or_insert_max_cooperative_blocks<Table, PolicyTypeV,
                                                        true>(enable_overflow,
                                                              num_sms);
      } else {
        max_cooperative_blocks =
            query_find_or_insert_max_cooperative_blocks<Table, PolicyTypeV,
                                                        false>(enable_overflow,
                                                               num_sms);
      }
    });
  });
  return max_cooperative_blocks;
}

void table_insert_and_evict_single_score(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, std::optional<at::Tensor> score_input,
    ScorePolicyType policy_type, at::Tensor indices,
    std::optional<at::Tensor> insert_results,
    std::optional<at::Tensor> score_output, at::Tensor num_evicted,
    at::Tensor evicted_keys, at::Tensor evicted_indices,
    at::Tensor evicted_scores, at::Tensor evicted_table_ids,
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
  auto insert_results_ptr = get_pointer<InsertResult>(insert_results);
  auto bucket_sizes_ = get_pointer<int>(bucket_sizes);

  auto evict_counter_ = get_pointer<CounterType>(num_evicted);
  auto evicted_scores_ptr = evicted_scores.data_ptr<int64_t>();
  auto evicted_indices_ = get_pointer<IndexType>(evicted_indices);
  auto evicted_table_ids_ptr = evicted_table_ids.data_ptr<int64_t>();
  auto table_ids_ptr = table_ids.data_ptr<int64_t>();
  auto table_bucket_offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  auto counter_ptr = counter.data_ptr<int32_t>();
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int64_t num_total = keys.size(0);

  auto table_key_slots = at::zeros(
      num_total, at::TensorOptions().dtype(at::kLong).device(keys.device()));

  bool output_score = (score_output_ptr != nullptr);

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    auto keys_ = get_pointer<KeyType>(keys);
    auto evicted_keys_ = get_pointer<KeyType>(evicted_keys);
    auto table_key_slots_ = get_pointer<KeyType *>(table_key_slots);

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
      if (num_total % 32 == 0) {
        if (output_score) {
          launch_table_insert_and_evict_kernel<Table, PolicyTypeV, true, 32>(
              table, table_bucket_offsets_ptr, bucket_sizes_, num_total, keys_,
              table_ids_ptr, insert_results_ptr, indices_ptr, score_input_ptr,
              score_output_ptr, table_key_slots_, evict_counter_, evicted_keys_,
              evicted_scores_ptr, evicted_indices_, evicted_table_ids_ptr,
              counter_ptr, stream);
        } else {
          launch_table_insert_and_evict_kernel<Table, PolicyTypeV, false, 32>(
              table, table_bucket_offsets_ptr, bucket_sizes_, num_total, keys_,
              table_ids_ptr, insert_results_ptr, indices_ptr, score_input_ptr,
              nullptr, table_key_slots_, evict_counter_, evicted_keys_,
              evicted_scores_ptr, evicted_indices_, evicted_table_ids_ptr,
              counter_ptr, stream);
        }
      } else {
        if (output_score) {
          launch_table_insert_and_evict_kernel<Table, PolicyTypeV, true, 1>(
              table, table_bucket_offsets_ptr, bucket_sizes_, num_total, keys_,
              table_ids_ptr, insert_results_ptr, indices_ptr, score_input_ptr,
              score_output_ptr, table_key_slots_, evict_counter_, evicted_keys_,
              evicted_scores_ptr, evicted_indices_, evicted_table_ids_ptr,
              counter_ptr, stream);
        } else {
          launch_table_insert_and_evict_kernel<Table, PolicyTypeV, false, 1>(
              table, table_bucket_offsets_ptr, bucket_sizes_, num_total, keys_,
              table_ids_ptr, insert_results_ptr, indices_ptr, score_input_ptr,
              nullptr, table_key_slots_, evict_counter_, evicted_keys_,
              evicted_scores_ptr, evicted_indices_, evicted_table_ids_ptr,
              counter_ptr, stream);
        }
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

// Internal: counter+overflow insert kernel launch
static void table_insert_and_evict_with_counter_and_overflow_single_score(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, std::optional<at::Tensor> score_input,
    ScorePolicyType policy_type, at::Tensor indices,
    std::optional<at::Tensor> insert_results,
    std::optional<at::Tensor> score_output, at::Tensor num_evicted,
    at::Tensor evicted_keys, at::Tensor evicted_indices,
    at::Tensor evicted_scores, at::Tensor evicted_table_ids, at::Tensor counter,
    at::Tensor ovf_storage, int64_t ovf_bucket_capacity,
    at::Tensor ovf_bucket_sizes, at::Tensor ovf_counter,
    at::Tensor ovf_output_offsets) {

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
  auto insert_results_ptr = get_pointer<InsertResult>(insert_results);
  auto bucket_sizes_ = get_pointer<int>(bucket_sizes);

  auto evict_counter_ = get_pointer<CounterType>(num_evicted);
  auto evicted_scores_ptr = evicted_scores.data_ptr<int64_t>();
  auto evicted_indices_ = get_pointer<IndexType>(evicted_indices);
  auto evicted_table_ids_ptr = evicted_table_ids.data_ptr<int64_t>();
  auto table_ids_ptr = table_ids.data_ptr<int64_t>();
  auto table_bucket_offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  auto counter_ptr = counter.data_ptr<int32_t>();
  auto ovf_bucket_sizes_ = get_pointer<int>(ovf_bucket_sizes);
  auto ovf_counter_ptr = ovf_counter.data_ptr<int32_t>();
  auto ovf_output_offsets_ptr = ovf_output_offsets.data_ptr<int64_t>();
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int64_t num_total = keys.size(0);

  auto table_key_slots = at::zeros(
      num_total, at::TensorOptions().dtype(at::kLong).device(keys.device()));

  bool output_score = (score_output_ptr != nullptr);

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    auto keys_ = get_pointer<KeyType>(keys);
    auto evicted_keys_ = get_pointer<KeyType>(evicted_keys);
    auto table_key_slots_ = get_pointer<KeyType *>(table_key_slots);

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
      if (num_total % 32 == 0) {
        if (output_score) {
          launch_table_insert_and_evict_kernel<Table, PolicyTypeV, true, 32,
                                               true>(
              table, table_bucket_offsets_ptr, bucket_sizes_, num_total, keys_,
              table_ids_ptr, insert_results_ptr, indices_ptr, score_input_ptr,
              score_output_ptr, table_key_slots_, evict_counter_, evicted_keys_,
              evicted_scores_ptr, evicted_indices_, evicted_table_ids_ptr,
              counter_ptr, stream, ovf_table, ovf_bucket_sizes_,
              ovf_counter_ptr, ovf_output_offsets_ptr);
        } else {
          launch_table_insert_and_evict_kernel<Table, PolicyTypeV, false, 32,
                                               true>(
              table, table_bucket_offsets_ptr, bucket_sizes_, num_total, keys_,
              table_ids_ptr, insert_results_ptr, indices_ptr, score_input_ptr,
              nullptr, table_key_slots_, evict_counter_, evicted_keys_,
              evicted_scores_ptr, evicted_indices_, evicted_table_ids_ptr,
              counter_ptr, stream, ovf_table, ovf_bucket_sizes_,
              ovf_counter_ptr, ovf_output_offsets_ptr);
        }
      } else {
        if (output_score) {
          launch_table_insert_and_evict_kernel<Table, PolicyTypeV, true, 1,
                                               true>(
              table, table_bucket_offsets_ptr, bucket_sizes_, num_total, keys_,
              table_ids_ptr, insert_results_ptr, indices_ptr, score_input_ptr,
              score_output_ptr, table_key_slots_, evict_counter_, evicted_keys_,
              evicted_scores_ptr, evicted_indices_, evicted_table_ids_ptr,
              counter_ptr, stream, ovf_table, ovf_bucket_sizes_,
              ovf_counter_ptr, ovf_output_offsets_ptr);
        } else {
          launch_table_insert_and_evict_kernel<Table, PolicyTypeV, false, 1,
                                               true>(
              table, table_bucket_offsets_ptr, bucket_sizes_, num_total, keys_,
              table_ids_ptr, insert_results_ptr, indices_ptr, score_input_ptr,
              nullptr, table_key_slots_, evict_counter_, evicted_keys_,
              evicted_scores_ptr, evicted_indices_, evicted_table_ids_ptr,
              counter_ptr, stream, ovf_table, ovf_bucket_sizes_,
              ovf_counter_ptr, ovf_output_offsets_ptr);
        }
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

// Fused cache probe/insert.  Unlike the legacy insert_and_evict wrapper this
// writes full-length hit/slot outputs, increments the slot reference before
// publication, and writes eviction metadata at the corresponding input
// position under evicted_mask.  Optional output buffers let deterministic
// waves fill disjoint active positions without a host count readback.
static void table_find_or_insert_single_score(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, std::optional<at::Tensor> score_input,
    ScorePolicyType policy_type, at::Tensor indices, at::Tensor founds,
    std::optional<at::Tensor> insert_results,
    std::optional<at::Tensor> score_output, at::Tensor evicted_keys,
    at::Tensor evicted_indices,
    at::Tensor evicted_scores, at::Tensor evicted_table_ids,
    at::Tensor evicted_mask, at::Tensor counter,
    int64_t max_cooperative_blocks,
    std::optional<at::Tensor> active_mask,
    std::optional<at::Tensor> active_count,
    std::optional<at::Tensor> ovf_storage, int64_t ovf_bucket_capacity,
    std::optional<at::Tensor> ovf_bucket_sizes,
    std::optional<at::Tensor> ovf_counter,
    std::optional<at::Tensor> ovf_output_offsets) {
  auto key_type = get_data_type(keys);

  ScoreType *score_input_ptr = nullptr;
  at::Tensor score_input_tensor;
  if (score_input.has_value() && score_input->defined()) {
    at::Tensor in = score_input.value();
    if (in.scalar_type() == torch::kUInt64) {
      score_input_ptr = get_pointer<ScoreType>(score_input);
    } else {
      score_input_tensor = in.view(torch::kUInt64);
      score_input_ptr = score_input_tensor.data_ptr<ScoreType>();
    }
  }

  int64_t *score_output_ptr = get_pointer<int64_t>(score_output);
  auto indices_ptr = indices.data_ptr<IndexType>();
  auto founds_ptr = founds.data_ptr<bool>();
  auto insert_results_ptr = get_pointer<InsertResult>(insert_results);
  auto bucket_sizes_ptr = get_pointer<int>(bucket_sizes);
  auto evicted_scores_ptr = evicted_scores.data_ptr<int64_t>();
  auto evicted_indices_ptr = get_pointer<IndexType>(evicted_indices);
  auto evicted_table_ids_ptr = evicted_table_ids.data_ptr<int64_t>();
  auto evicted_mask_ptr = evicted_mask.data_ptr<bool>();
  auto table_ids_ptr = table_ids.data_ptr<int64_t>();
  auto table_bucket_offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  auto counter_ptr = counter.data_ptr<int32_t>();
  auto active_mask_ptr = get_pointer<bool>(active_mask);
  auto active_count_ptr = get_pointer<int64_t>(active_count);
  // The caller keeps the table device current; the explicit index avoids a
  // cudaGetDevice-style lookup while selecting that device's current stream.
  auto stream =
      at::cuda::getCurrentCUDAStream(table_storage.get_device()).stream();
  int64_t num_total = keys.size(0);

  bool output_score = score_output_ptr != nullptr;
  bool use_overflow = ovf_storage.has_value() && ovf_storage->defined();

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    auto keys_ptr = get_pointer<KeyType>(keys);
    auto evicted_keys_ptr = get_pointer<KeyType>(evicted_keys);
    KeyType **table_key_slots_ptr = nullptr;

    constexpr int64_t total_size =
        sizeof(KeyType) + sizeof(DigestType) + sizeof(ScoreType);
    int64_t bucket_bytes = bucket_capacity * total_size;
    int64_t num_buckets =
        table_storage.numel() * table_storage.element_size() / bucket_bytes;
    using Bucket = LinearBucket<KeyType>;
    using Table = LinearBucketTable<Bucket>;
    auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                       num_buckets, bucket_capacity);

    if (use_overflow) {
      int64_t ovf_bucket_bytes = ovf_bucket_capacity * total_size;
      int64_t ovf_num_buckets = ovf_storage->numel() *
                                ovf_storage->element_size() /
                                ovf_bucket_bytes;
      auto ovf_table =
          Table(reinterpret_cast<uint8_t *>(ovf_storage->data_ptr()),
                ovf_num_buckets, ovf_bucket_capacity);
      auto ovf_bucket_sizes_ptr = get_pointer<int>(ovf_bucket_sizes);
      auto ovf_counter_ptr = get_pointer<int32_t>(ovf_counter);
      auto ovf_output_offsets_ptr = get_pointer<int64_t>(ovf_output_offsets);
      DISPATCH_SCORE_POLICY(policy_type, PolicyTypeV, [&] {
        if (output_score) {
          launch_table_insert_and_evict_kernel<Table, PolicyTypeV, true, 1,
                                               true, true>(
              table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
              keys_ptr, table_ids_ptr, insert_results_ptr, indices_ptr,
              score_input_ptr, score_output_ptr, table_key_slots_ptr,
              nullptr, evicted_keys_ptr, evicted_scores_ptr,
              evicted_indices_ptr, evicted_table_ids_ptr, counter_ptr, stream,
              ovf_table, ovf_bucket_sizes_ptr, ovf_counter_ptr,
              ovf_output_offsets_ptr, founds_ptr, evicted_mask_ptr,
              active_mask_ptr, active_count_ptr, max_cooperative_blocks);
        } else {
          launch_table_insert_and_evict_kernel<Table, PolicyTypeV, false, 1,
                                               true, true>(
              table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
              keys_ptr, table_ids_ptr, insert_results_ptr, indices_ptr,
              score_input_ptr, nullptr, table_key_slots_ptr, nullptr,
              evicted_keys_ptr, evicted_scores_ptr, evicted_indices_ptr,
              evicted_table_ids_ptr, counter_ptr, stream, ovf_table,
              ovf_bucket_sizes_ptr, ovf_counter_ptr, ovf_output_offsets_ptr,
              founds_ptr, evicted_mask_ptr, active_mask_ptr,
              active_count_ptr, max_cooperative_blocks);
        }
      });
    } else {
      DISPATCH_SCORE_POLICY(policy_type, PolicyTypeV, [&] {
        if (output_score) {
          launch_table_insert_and_evict_kernel<Table, PolicyTypeV, true, 1,
                                               false, true>(
              table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
              keys_ptr, table_ids_ptr, insert_results_ptr, indices_ptr,
              score_input_ptr, score_output_ptr, table_key_slots_ptr,
              nullptr, evicted_keys_ptr, evicted_scores_ptr,
              evicted_indices_ptr, evicted_table_ids_ptr, counter_ptr, stream,
              Table(), nullptr, nullptr, nullptr, founds_ptr,
              evicted_mask_ptr, active_mask_ptr, active_count_ptr,
              max_cooperative_blocks);
        } else {
          launch_table_insert_and_evict_kernel<Table, PolicyTypeV, false, 1,
                                               false, true>(
              table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
              keys_ptr, table_ids_ptr, insert_results_ptr, indices_ptr,
              score_input_ptr, nullptr, table_key_slots_ptr, nullptr,
              evicted_keys_ptr, evicted_scores_ptr, evicted_indices_ptr,
              evicted_table_ids_ptr, counter_ptr, stream, Table(), nullptr,
              nullptr, nullptr, founds_ptr, evicted_mask_ptr,
              active_mask_ptr, active_count_ptr, max_cooperative_blocks);
        }
      });
    }
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor>
table_find_or_insert(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, std::optional<at::Tensor> score_input,
    ScorePolicyType policy_type, at::Tensor counter,
    int64_t max_cooperative_blocks,
    std::optional<at::Tensor> insert_results,
    std::optional<at::Tensor> score_output,
    std::optional<at::Tensor> active_mask,
    std::optional<at::Tensor> active_count,
    std::optional<at::Tensor> indices_output,
    std::optional<at::Tensor> founds_output,
    std::optional<at::Tensor> evicted_keys_output,
    std::optional<at::Tensor> evicted_indices_output,
    std::optional<at::Tensor> evicted_scores_output,
    std::optional<at::Tensor> evicted_table_ids_output,
    std::optional<at::Tensor> evicted_mask_output,
    std::optional<at::Tensor> ovf_storage, int64_t ovf_bucket_capacity,
    std::optional<at::Tensor> ovf_bucket_sizes,
    std::optional<at::Tensor> ovf_counter,
    std::optional<at::Tensor> ovf_output_offsets) {
  TORCH_CHECK(max_cooperative_blocks > 0,
              "find_or_insert requires a positive cooperative grid limit");
  int64_t n = keys.size(0);
  auto indices = indices_output.has_value() && indices_output->defined()
                     ? indices_output.value()
                     : torch::full({n}, -1,
                                   keys.options().dtype(torch::kInt64));
  auto founds = founds_output.has_value() && founds_output->defined()
                    ? founds_output.value()
                    : torch::zeros({n}, keys.options().dtype(torch::kBool));
  auto evicted_keys =
      evicted_keys_output.has_value() && evicted_keys_output->defined()
          ? evicted_keys_output.value()
          : torch::empty({n}, keys.options().dtype(keys.scalar_type()));
  auto evicted_indices =
      evicted_indices_output.has_value() && evicted_indices_output->defined()
          ? evicted_indices_output.value()
          : torch::empty({n}, keys.options().dtype(torch::kInt64));
  auto evicted_scores =
      evicted_scores_output.has_value() && evicted_scores_output->defined()
          ? evicted_scores_output.value()
          : torch::empty({n}, keys.options().dtype(torch::kInt64));
  auto evicted_table_ids =
      evicted_table_ids_output.has_value() &&
              evicted_table_ids_output->defined()
          ? evicted_table_ids_output.value()
          : torch::empty({n}, keys.options().dtype(torch::kInt64));
  auto evicted_mask =
      evicted_mask_output.has_value() && evicted_mask_output->defined()
          ? evicted_mask_output.value()
          : torch::zeros({n}, keys.options().dtype(torch::kBool));

  auto check_vector = [&](at::Tensor const &tensor, at::ScalarType dtype,
                          int64_t size, char const *name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == keys.device(), name,
                " must be on the same device as keys");
    TORCH_CHECK(tensor.scalar_type() == dtype, name,
                " has an invalid dtype");
    TORCH_CHECK(tensor.dim() == 1 && tensor.is_contiguous(), name,
                " must be a contiguous one-dimensional tensor");
    TORCH_CHECK(tensor.numel() == size, name, " must contain ", size,
                " elements");
  };
  check_vector(indices, torch::kInt64, n, "indices_output");
  check_vector(founds, torch::kBool, n, "founds_output");
  check_vector(evicted_keys, keys.scalar_type(), n, "evicted_keys_output");
  check_vector(evicted_indices, torch::kInt64, n,
               "evicted_indices_output");
  check_vector(evicted_scores, torch::kInt64, n, "evicted_scores_output");
  check_vector(evicted_table_ids, torch::kInt64, n,
               "evicted_table_ids_output");
  check_vector(evicted_mask, torch::kBool, n, "evicted_mask_output");
  if (active_mask.has_value() && active_mask->defined()) {
    check_vector(*active_mask, torch::kBool, n, "active_mask");
  }
  if (active_count.has_value() && active_count->defined()) {
    check_vector(*active_count, torch::kInt64, 1, "active_count");
  }
  if (n != 0) {
    table_find_or_insert_single_score(
        table_storage, table_bucket_offsets, bucket_capacity, bucket_sizes,
        keys, table_ids, score_input, policy_type, indices, founds,
        insert_results, score_output, evicted_keys, evicted_indices,
        evicted_scores, evicted_table_ids, evicted_mask, counter,
        max_cooperative_blocks, active_mask, active_count, ovf_storage,
        ovf_bucket_capacity, ovf_bucket_sizes, ovf_counter,
        ovf_output_offsets);
  }
  return std::make_tuple(indices, founds, evicted_keys, evicted_indices,
                         evicted_scores, evicted_table_ids, evicted_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor>
table_insert_and_evict(at::Tensor table_storage,
                       at::Tensor table_bucket_offsets, int64_t bucket_capacity,
                       at::Tensor bucket_sizes, at::Tensor keys,
                       at::Tensor table_ids,
                       std::optional<at::Tensor> score_input,
                       ScorePolicyType policy_type, at::Tensor counter,
                       std::optional<at::Tensor> insert_results,
                       std::optional<at::Tensor> score_output,
                       std::optional<at::Tensor> ovf_storage,
                       int64_t ovf_bucket_capacity,
                       std::optional<at::Tensor> ovf_bucket_sizes,
                       std::optional<at::Tensor> ovf_counter,
                       std::optional<at::Tensor> ovf_output_offsets) {

  int64_t num_total = keys.size(0);
  if (num_total == 0) {
    at::Tensor indices = torch::empty({0}, keys.options().dtype(torch::kInt64));
    at::Tensor num_evicted =
        torch::zeros({1}, keys.options().dtype(torch::kInt64));
    at::Tensor evicted_keys =
        torch::empty({0}, keys.options().dtype(keys.scalar_type()));
    at::Tensor evicted_indices =
        torch::empty({0}, keys.options().dtype(torch::kInt64));
    at::Tensor evicted_scores =
        torch::empty({0}, keys.options().dtype(torch::kInt64));
    at::Tensor evicted_table_ids =
        torch::empty({0}, keys.options().dtype(torch::kLong));
    return std::make_tuple(indices, num_evicted, evicted_keys, evicted_indices,
                           evicted_scores, evicted_table_ids);
  }

  at::Tensor indices =
      torch::empty({num_total}, keys.options().dtype(torch::kInt64));
  at::Tensor num_evicted =
      torch::zeros({1}, keys.options().dtype(torch::kInt64));
  at::Tensor evicted_keys =
      torch::empty({num_total}, keys.options().dtype(keys.scalar_type()));
  at::Tensor evicted_indices =
      torch::empty({num_total}, keys.options().dtype(torch::kInt64));
  at::Tensor evicted_scores =
      torch::empty({num_total}, keys.options().dtype(torch::kInt64));
  at::Tensor evicted_table_ids =
      torch::empty({num_total}, keys.options().dtype(torch::kLong));

  bool use_overflow = ovf_storage.has_value() && ovf_storage.value().defined();

  if (use_overflow) {
    table_insert_and_evict_with_counter_and_overflow_single_score(
        table_storage, table_bucket_offsets, bucket_capacity, bucket_sizes,
        keys, table_ids, score_input, policy_type, indices, insert_results,
        score_output, num_evicted, evicted_keys, evicted_indices,
        evicted_scores, evicted_table_ids, counter, ovf_storage.value(),
        ovf_bucket_capacity, ovf_bucket_sizes.value(), ovf_counter.value(),
        ovf_output_offsets.value());
  } else {
    table_insert_and_evict_single_score(
        table_storage, table_bucket_offsets, bucket_capacity, bucket_sizes,
        keys, table_ids, score_input, policy_type, indices, insert_results,
        score_output, num_evicted, evicted_keys, evicted_indices,
        evicted_scores, evicted_table_ids, counter);
  }

  return std::make_tuple(indices, num_evicted, evicted_keys, evicted_indices,
                         evicted_scores, evicted_table_ids);
}

void table_update_counter_with_layout(
    at::Tensor counter, at::Tensor slot_indices, int32_t delta,
    at::Tensor table_bucket_offsets, int64_t bucket_capacity,
    int64_t main_capacity, int64_t num_tables,
    c10::optional<at::Tensor> table_ids,
    c10::optional<at::Tensor> overflow_output_offsets,
    int64_t overflow_bucket_capacity) {

  int64_t n = slot_indices.size(0);
  if (n == 0)
    return;

  int64_t total_capacity = counter.numel();
  auto counter_ptr = counter.data_ptr<int32_t>();
  auto slot_indices_ptr = slot_indices.data_ptr<int64_t>();
  auto table_bucket_offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  int64_t const *table_ids_ptr = (table_ids.has_value() && table_ids->defined())
                                     ? table_ids->data_ptr<int64_t>()
                                     : nullptr;
  int64_t const *overflow_output_offsets_ptr =
      (overflow_output_offsets.has_value() &&
       overflow_output_offsets->defined())
          ? overflow_output_offsets->data_ptr<int64_t>()
          : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  constexpr int BLOCK_SIZE = 256;
  update_counter_with_layout_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                                      BLOCK_SIZE, 0, stream>>>(
      counter_ptr, total_capacity, slot_indices_ptr, n, delta, table_ids_ptr,
      table_bucket_offsets_ptr, bucket_capacity, main_capacity,
      overflow_output_offsets_ptr, overflow_bucket_capacity, num_tables);

  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void table_reclaim_by_slot(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, at::Tensor slot_indices,
    at::Tensor counter, std::optional<at::Tensor> mask,
    std::optional<at::Tensor> ovf_storage, int64_t ovf_bucket_capacity,
    std::optional<at::Tensor> ovf_bucket_sizes,
    std::optional<at::Tensor> ovf_counter,
    std::optional<at::Tensor> ovf_output_offsets) {
  int64_t n = keys.numel();
  if (n == 0)
    return;
  auto key_type = get_data_type(keys);
  auto offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  auto bucket_sizes_ptr = bucket_sizes.data_ptr<int>();
  auto table_ids_ptr = table_ids.data_ptr<int64_t>();
  auto slots_ptr = slot_indices.data_ptr<IndexType>();
  auto mask_ptr = get_pointer<bool>(mask);
  auto counter_ptr = counter.data_ptr<int32_t>();
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr int BLOCK_SIZE = 256;
  bool use_overflow = ovf_storage.has_value() && ovf_storage->defined();

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
    auto keys_ptr = keys.data_ptr<KeyType>();
    if (use_overflow) {
      int64_t ovf_bucket_bytes = ovf_bucket_capacity * total_size;
      int64_t ovf_num_buckets = ovf_storage->numel() *
                                ovf_storage->element_size() /
                                ovf_bucket_bytes;
      auto ovf_table =
          Table(reinterpret_cast<uint8_t *>(ovf_storage->data_ptr()),
                ovf_num_buckets, ovf_bucket_capacity);
      table_reclaim_by_slot_kernel<Table, true>
          <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table, offsets_ptr, bucket_sizes_ptr, n, keys_ptr, table_ids_ptr,
              slots_ptr, mask_ptr, counter_ptr, ovf_table,
              get_pointer<int>(ovf_bucket_sizes),
              get_pointer<int32_t>(ovf_counter),
              get_pointer<int64_t>(ovf_output_offsets));
    } else {
      table_reclaim_by_slot_kernel<Table, false>
          <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table, offsets_ptr, bucket_sizes_ptr, n, keys_ptr, table_ids_ptr,
              slots_ptr, mask_ptr, counter_ptr, Table(), nullptr, nullptr,
              nullptr);
    }
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void table_update_score_by_slot(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor slot_indices, at::Tensor table_ids,
    at::Tensor scores, std::optional<at::Tensor> active_mask,
    std::optional<at::Tensor> active_count,
    std::optional<at::Tensor> ovf_storage, int64_t ovf_bucket_capacity,
    std::optional<at::Tensor> ovf_output_offsets) {
  int64_t n = slot_indices.numel();
  if (n == 0)
    return;
  at::Tensor scores_view = scores.scalar_type() == torch::kUInt64
                               ? scores
                               : scores.view(torch::kUInt64);
  auto offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  auto slots_ptr = slot_indices.data_ptr<IndexType>();
  auto table_ids_ptr = table_ids.data_ptr<int64_t>();
  auto scores_ptr = scores_view.data_ptr<ScoreType>();
  auto mask_ptr = get_pointer<bool>(active_mask);
  auto count_ptr = get_pointer<int64_t>(active_count);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr int BLOCK_SIZE = 256;
  // table_storage is byte-addressed, so derive key type from the score-map
  // caller's table configuration via element-size-independent dispatch below.
  // Both supported key types are 64-bit and therefore share the same layout.
  using KeyType = int64_t;
  constexpr int64_t total_size =
      sizeof(KeyType) + sizeof(DigestType) + sizeof(ScoreType);
  int64_t bucket_bytes = bucket_capacity * total_size;
  int64_t num_buckets =
      table_storage.numel() * table_storage.element_size() / bucket_bytes;
  using Bucket = LinearBucket<KeyType>;
  using Table = LinearBucketTable<Bucket>;
  auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                     num_buckets, bucket_capacity);
  bool use_overflow = ovf_storage.has_value() && ovf_storage->defined();
  if (use_overflow) {
    int64_t ovf_bucket_bytes = ovf_bucket_capacity * total_size;
    int64_t ovf_num_buckets = ovf_storage->numel() *
                              ovf_storage->element_size() / ovf_bucket_bytes;
    auto ovf_table =
        Table(reinterpret_cast<uint8_t *>(ovf_storage->data_ptr()),
              ovf_num_buckets, ovf_bucket_capacity);
    table_update_score_by_slot_kernel<Table, true>
        <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            table, offsets_ptr, n, slots_ptr, table_ids_ptr, scores_ptr,
            mask_ptr, count_ptr, ovf_table,
            get_pointer<int64_t>(ovf_output_offsets));
  } else {
    table_update_score_by_slot_kernel<Table, false>
        <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            table, offsets_ptr, n, slots_ptr, table_ids_ptr, scores_ptr,
            mask_ptr, count_ptr, Table(), nullptr);
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace dyn_emb
