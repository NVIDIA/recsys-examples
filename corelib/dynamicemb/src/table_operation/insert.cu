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

#include "jit/jit_link.h"
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
    int32_t *counter_ptr, int64_t score_fn_key, cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  using KernelTraits =
      InsertKernelTraits<BLOCK_SIZE, 1, 1, 1, 8, PolicyTypeV, OutputScoreV>;

  if constexpr (PolicyTypeV == ScorePolicyType::LruLfu) {
    // Route the plain insert through the LruLfu cubin (default Lex when
    // score_fn_key == 0, else the nvJitLink-linked custom evictor) so a full
    // bucket evicts by the ranked comparator, not the single-score reduce().
    // Reuses EvictParams; evicted_* / ovf_* stay null (dyn_emb_insert_entry
    // collects no evicted output).
    EvictParams p{};
    p.table_storage = table.storage_;
    p.num_buckets = table.num_buckets_;
    p.bucket_capacity = table.bucket_capacity_;
    p.num_scores = table.num_scores_;
    p.table_bucket_offsets = table_bucket_offsets_ptr;
    p.bucket_sizes = bucket_sizes_ptr;
    p.batch = num_total;
    p.input_keys = reinterpret_cast<const int64_t *>(keys_ptr);
    p.table_ids = table_ids_ptr;
    p.insert_results = reinterpret_cast<uint8_t *>(insert_results_ptr);
    p.indices = indices_ptr;
    p.score_input = score_input_ptr;
    p.score_output = OutputScoreV ? score_output_ptr : nullptr;
    p.table_key_slots = reinterpret_cast<int64_t **>(table_key_slots_ptr);
    p.counter = counter_ptr;
    CUfunction fn = demb_get_insert_fn(score_fn_key);
    demb_launch_evict(fn, p, num_total, stream);
  } else {
    table_insert_kernel<Table, KernelTraits>
        <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
            keys_ptr, table_ids_ptr, insert_results_ptr, indices_ptr,
            score_input_ptr, score_output_ptr, table_key_slots_ptr,
            counter_ptr);
  }

  table_unlock_kernel<Table>
      <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
          table, num_total, keys_ptr, table_key_slots_ptr);
}

// Plain insert that ALSO retains each evicted victim's (key, table_id). Mirrors
// launch_table_insert_kernel but routes to the collect entry / kernel and passes
// the evicted_* output buffers. Used by the last-tier retain path.
template <typename Table, ScorePolicyType PolicyTypeV, bool OutputScoreV>
void launch_table_insert_collect_kernel(
    Table table, int64_t *table_bucket_offsets_ptr, int *bucket_sizes_ptr,
    int64_t num_total, typename Table::KeyType *keys_ptr,
    int64_t *table_ids_ptr, InsertResult *insert_results_ptr,
    IndexType *indices_ptr, ScoreType *score_input_ptr,
    int64_t *score_output_ptr, typename Table::KeyType **table_key_slots_ptr,
    int32_t *counter_ptr, CounterType *evicted_counter_ptr,
    typename Table::KeyType *evicted_keys_ptr, int64_t *evicted_table_ids_ptr,
    int64_t score_fn_key, cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  using KernelTraits =
      InsertKernelTraits<BLOCK_SIZE, 1, 1, 1, 8, PolicyTypeV, OutputScoreV>;

  if constexpr (PolicyTypeV == ScorePolicyType::LruLfu) {
    EvictParams p{};
    p.table_storage = table.storage_;
    p.num_buckets = table.num_buckets_;
    p.bucket_capacity = table.bucket_capacity_;
    p.num_scores = table.num_scores_;
    p.table_bucket_offsets = table_bucket_offsets_ptr;
    p.bucket_sizes = bucket_sizes_ptr;
    p.batch = num_total;
    p.input_keys = reinterpret_cast<const int64_t *>(keys_ptr);
    p.table_ids = table_ids_ptr;
    p.insert_results = reinterpret_cast<uint8_t *>(insert_results_ptr);
    p.indices = indices_ptr;
    p.score_input = score_input_ptr;
    p.score_output = OutputScoreV ? score_output_ptr : nullptr;
    p.table_key_slots = reinterpret_cast<int64_t **>(table_key_slots_ptr);
    p.evicted_counter = evicted_counter_ptr;
    p.evicted_keys = reinterpret_cast<int64_t *>(evicted_keys_ptr);
    p.evicted_table_ids = evicted_table_ids_ptr;
    p.counter = counter_ptr;
    CUfunction fn = demb_get_insert_collect_fn(score_fn_key);
    demb_launch_evict(fn, p, num_total, stream);
  } else {
    table_insert_collect_kernel<Table, KernelTraits>
        <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
            keys_ptr, table_ids_ptr, insert_results_ptr, indices_ptr,
            score_input_ptr, score_output_ptr, table_key_slots_ptr, counter_ptr,
            evicted_counter_ptr, evicted_keys_ptr, evicted_table_ids_ptr);
  }

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
                               at::Tensor counter, int64_t num_scores,
                               int64_t score_fn_key) {

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
            score_fn_key, stream);
      } else {
        launch_table_insert_kernel<Table, PolicyTypeV, false>(
            table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
            keys_ptr, table_ids_ptr, insert_results_ptr, indices_ptr,
            score_input_ptr, nullptr, table_key_slots_ptr, counter_ptr,
            score_fn_key, stream);
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
                        int64_t num_scores, int64_t score_fn_key) {

  int64_t num_total = keys.size(0);
  if (num_total == 0) {
    return torch::empty({0}, keys.options().dtype(torch::kInt64));
  }

  at::Tensor indices =
      torch::empty({num_total}, keys.options().dtype(torch::kInt64));

  table_insert_single_score(table_storage, table_bucket_offsets,
                            bucket_capacity, bucket_sizes, keys, table_ids,
                            score_input, policy_type, indices, insert_results,
                            score_output, counter, num_scores, score_fn_key);

  return indices;
}

// Same dispatch as table_insert_single_score but routes to the collect launcher
// and threads through the evicted_* output buffers (retain evicted keys).
void table_insert_collect_single_score(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, std::optional<at::Tensor> score_input,
    ScorePolicyType policy_type, at::Tensor indices,
    std::optional<at::Tensor> insert_results,
    std::optional<at::Tensor> score_output, at::Tensor counter,
    int64_t num_scores, int64_t score_fn_key, at::Tensor evicted_counter,
    at::Tensor evicted_keys, at::Tensor evicted_table_ids) {

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
  auto evicted_counter_ptr =
      reinterpret_cast<CounterType *>(evicted_counter.data_ptr<int64_t>());
  auto evicted_table_ids_ptr = evicted_table_ids.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int64_t num_total = keys.size(0);

  auto table_key_slots = at::zeros(
      num_total, at::TensorOptions().dtype(at::kLong).device(keys.device()));

  bool output_score = (score_output_ptr != nullptr);

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    auto keys_ptr = get_pointer<KeyType>(keys);
    auto table_key_slots_ptr = get_pointer<KeyType *>(table_key_slots);
    auto evicted_keys_ptr = get_pointer<KeyType>(evicted_keys);

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
        launch_table_insert_collect_kernel<Table, PolicyTypeV, true>(
            table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
            keys_ptr, table_ids_ptr, insert_results_ptr, indices_ptr,
            score_input_ptr, score_output_ptr, table_key_slots_ptr, counter_ptr,
            evicted_counter_ptr, evicted_keys_ptr, evicted_table_ids_ptr,
            score_fn_key, stream);
      } else {
        launch_table_insert_collect_kernel<Table, PolicyTypeV, false>(
            table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
            keys_ptr, table_ids_ptr, insert_results_ptr, indices_ptr,
            score_input_ptr, nullptr, table_key_slots_ptr, counter_ptr,
            evicted_counter_ptr, evicted_keys_ptr, evicted_table_ids_ptr,
            score_fn_key, stream);
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

// Plain insert that retains evicted victims. Returns
// (indices, num_evicted[int64 scalar tensor], evicted_keys, evicted_table_ids);
// the evicted_* tensors are allocated at the batch upper bound (one victim per
// input key) -- the caller slices them to num_evicted. Same inputs as
// table_insert otherwise.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
table_insert_collect_evicted(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, std::optional<at::Tensor> score_input,
    ScorePolicyType policy_type, at::Tensor counter,
    std::optional<at::Tensor> insert_results,
    std::optional<at::Tensor> score_output, int64_t num_scores,
    int64_t score_fn_key) {

  auto i64 = keys.options().dtype(torch::kInt64);
  int64_t num_total = keys.size(0);
  if (num_total == 0) {
    at::Tensor empty_keys = torch::empty({0}, keys.options());
    at::Tensor empty_i64 = torch::empty({0}, i64);
    return std::make_tuple(empty_i64, torch::zeros({1}, i64), empty_keys,
                           empty_i64);
  }

  at::Tensor indices = torch::empty({num_total}, i64);
  at::Tensor evicted_keys = torch::empty({num_total}, keys.options());
  at::Tensor evicted_table_ids = torch::empty({num_total}, i64);
  at::Tensor evicted_counter = torch::zeros({1}, i64);

  table_insert_collect_single_score(
      table_storage, table_bucket_offsets, bucket_capacity, bucket_sizes, keys,
      table_ids, score_input, policy_type, indices, insert_results,
      score_output, counter, num_scores, score_fn_key, evicted_counter,
      evicted_keys, evicted_table_ids);

  return std::make_tuple(indices, evicted_counter, evicted_keys,
                         evicted_table_ids);
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

// Gather all score words at `slots` into a [n, num_scores] uint64 tensor.
// Used by dump to persist multi-word score layouts (LruLfu).
at::Tensor table_gather_score_blocks(at::Tensor table_storage,
                                     int64_t bucket_capacity, int64_t num_scores,
                                     int64_t bkt_begin, at::Tensor slots,
                                     torch::Dtype key_dtype) {
  int64_t n = slots.size(0);
  auto out = torch::empty(
      {n, num_scores},
      torch::TensorOptions().dtype(torch::kInt64).device(table_storage.device()));
  if (n == 0)
    return out;
  auto key_type = scalartype_to_datatype(toScalarType(key_dtype));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr int BLOCK_SIZE = 256;
  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    using Bucket = LinearBucket<KeyType>;
    using Table = LinearBucketTable<Bucket>;
    int64_t total_size =
        sizeof(KeyType) + sizeof(DigestType) + num_scores * sizeof(ScoreType);
    int64_t num_buckets = table_storage.numel() * table_storage.element_size() /
                          (bucket_capacity * total_size);
    auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                       num_buckets, bucket_capacity, num_scores);
    gather_score_blocks_kernel<Table>
        <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            table, bkt_begin, n, slots.data_ptr<int64_t>(),
            reinterpret_cast<ScoreType *>(out.data_ptr<int64_t>()));
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Scatter a [n, num_scores] uint64 score block into `slots`. Used by load to
// restore multi-word score layouts (LruLfu) after keys are placed.
void table_scatter_score_blocks(at::Tensor table_storage,
                                int64_t bucket_capacity, int64_t num_scores,
                                int64_t bkt_begin, at::Tensor slots,
                                at::Tensor values, torch::Dtype key_dtype) {
  int64_t n = slots.size(0);
  if (n == 0)
    return;
  auto key_type = scalartype_to_datatype(toScalarType(key_dtype));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr int BLOCK_SIZE = 256;
  at::Tensor vals = values.contiguous();
  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    using Bucket = LinearBucket<KeyType>;
    using Table = LinearBucketTable<Bucket>;
    int64_t total_size =
        sizeof(KeyType) + sizeof(DigestType) + num_scores * sizeof(ScoreType);
    int64_t num_buckets = table_storage.numel() * table_storage.element_size() /
                          (bucket_capacity * total_size);
    auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                       num_buckets, bucket_capacity, num_scores);
    scatter_score_blocks_kernel<Table>
        <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            table, bkt_begin, n, slots.data_ptr<int64_t>(),
            reinterpret_cast<const ScoreType *>(vals.data_ptr<int64_t>()));
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

// Write (key, score words) at exact table-relative slots -- the write-back
// behind replay_increment. Returns (status, same_key): status is the slot
// written, or -1 when the slot is not inside the key's home bucket and the key
// therefore could not be placed (the caller raises: the key would be
// unreachable). same_key tells whether the slot already held this very key (so
// its value row / optimizer state can be kept).
std::tuple<at::Tensor, at::Tensor> table_scatter_keys_at_slots(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, at::Tensor slots, at::Tensor scores,
    int64_t num_scores) {

  int64_t num_total = keys.size(0);
  auto status = torch::empty(
      {num_total},
      torch::TensorOptions().dtype(torch::kInt64).device(keys.device()));
  auto same_key = torch::empty(
      {num_total},
      torch::TensorOptions().dtype(torch::kBool).device(keys.device()));
  if (num_total == 0)
    return {status, same_key};

  auto key_type = get_data_type(keys);
  auto bucket_sizes_ = get_pointer<int>(bucket_sizes);
  auto table_ids_ptr = table_ids.data_ptr<int64_t>();
  auto table_bucket_offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  // Callers hand scores over as int64 (SCORE_TYPE) or uint64; reinterpret rather
  // than convert -- score words are opaque bit patterns.
  at::Tensor score_vals = scores.contiguous();
  if (score_vals.scalar_type() != torch::kUInt64)
    score_vals = score_vals.view(torch::kUInt64);
  const ScoreType *score_ptr = score_vals.data_ptr<ScoreType>();

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr int BLOCK_SIZE = 256;

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    auto keys_ = get_pointer<KeyType>(keys);
    int64_t total_size =
        sizeof(KeyType) + sizeof(DigestType) + num_scores * sizeof(ScoreType);
    int64_t bucket_bytes = bucket_capacity * total_size;
    int64_t num_buckets =
        table_storage.numel() * table_storage.element_size() / bucket_bytes;

    using Bucket = LinearBucket<KeyType>;
    using Table = LinearBucketTable<Bucket>;
    auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                       num_buckets, bucket_capacity, num_scores);

    scatter_keys_at_slots_kernel<Table, 1>
        <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            table, table_bucket_offsets_ptr, bucket_sizes_, num_total, keys_,
            table_ids_ptr, slots.data_ptr<int64_t>(), score_ptr,
            status.data_ptr<int64_t>(), same_key.data_ptr<bool>());
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  return {status, same_key};
}

} // namespace dyn_emb
