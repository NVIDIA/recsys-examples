/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

// The launchers of the deterministic insert (see sorted_insert.cuh): a sort of
// the batch by (home bucket, key), one kernel that inserts the keys of each
// bucket in order, for a cache an ordered overflow pass, and a compaction of
// the evictions whose count stays on the device. Everything is enqueued on the
// current stream; nothing is read back.
//
// The include order matters: ../index_calculation.h reaches lookup_kernel.cuh,
// which calls the CUDA atomicAdd(float *, float) from members of non-template
// specializations, and types.cuh (through kernels.cuh and table.cuh) declares
// dyn_emb::atomicAdd overloads that would hide it. It has to come first, as in
// bucketize.cu.
#include "../index_calculation.h"
#include "sorted_insert.cuh"
#include "table.cuh"

namespace dyn_emb {

template <typename Table, ScorePolicyType PolicyTypeV, bool UseOverflow>
void launch_table_insert_sorted_kernel(
    Table table, int64_t *table_bucket_offsets_ptr, int *bucket_sizes_ptr,
    int64_t num_total, typename Table::KeyType *sorted_keys_ptr,
    int64_t *segments_ptr, int64_t *inverse_ptr, int64_t *table_ids_ptr,
    ScoreType *score_input_ptr, InsertResult *insert_results_ptr,
    int64_t *score_output_ptr, int32_t *counter_ptr,
    typename Table::KeyType **held_slots_ptr,
    EvictRecords<typename Table::KeyType> rec, bool *overflow_flag_ptr,
    cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  if constexpr (PolicyTypeV == ScorePolicyType::LruLfu) {
    // Not reachable: the entry points refuse LRU_LFU (two score words, evicted
    // by the cubin) before they dispatch.
    throw std::runtime_error("the sorted insert takes one score word");
  } else {
    using KernelTraits = InsertKernelTraits<BLOCK_SIZE, 1, 1, 1, 8, PolicyTypeV,
                                            true, UseOverflow>;
    table_insert_sorted_kernel<Table, KernelTraits>
        <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
            sorted_keys_ptr, segments_ptr, inverse_ptr, table_ids_ptr,
            score_input_ptr, insert_results_ptr, score_output_ptr, counter_ptr,
            held_slots_ptr, rec, overflow_flag_ptr);
  }
}

template <typename Table, ScorePolicyType PolicyTypeV>
void launch_table_overflow_sorted_kernel(
    Table ovf_table, int64_t *table_bucket_offsets_ptr, int64_t num_tables,
    int *ovf_bucket_sizes_ptr, int32_t *ovf_counter_ptr,
    int64_t *ovf_output_offsets_ptr, typename Table::KeyType *sorted_keys_ptr,
    int64_t *segments_ptr, int64_t *inverse_ptr, int64_t *busy_ptr,
    int64_t *num_busy_ptr, typename Table::KeyType **spill_slots_ptr,
    InsertResult *insert_results_ptr, int64_t *score_output_ptr,
    EvictRecords<typename Table::KeyType> rec, cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 32;
  if constexpr (PolicyTypeV == ScorePolicyType::LruLfu) {
    throw std::runtime_error("the sorted insert takes one score word");
  } else {
    using KernelTraits =
        InsertKernelTraits<256, 1, 1, 1, 8, PolicyTypeV, true, true>;
    table_overflow_sorted_kernel<Table, KernelTraits>
        <<<(num_tables + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            ovf_table, table_bucket_offsets_ptr, num_tables,
            ovf_bucket_sizes_ptr, ovf_counter_ptr, ovf_output_offsets_ptr,
            sorted_keys_ptr, segments_ptr, inverse_ptr, busy_ptr, num_busy_ptr,
            spill_slots_ptr, insert_results_ptr, score_output_ptr, rec);
  }
}

namespace {

using SortedInsertResult =
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

// What the two entry points share. ``collect`` makes the insert report its
// evictions (and, with an overflow storage, run the ordered overflow pass);
// without it the five tensors returned are empty.
SortedInsertResult table_insert_sorted_impl(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, std::optional<at::Tensor> score_input,
    ScorePolicyType policy_type, std::optional<at::Tensor> counter,
    std::optional<at::Tensor> insert_results,
    std::optional<at::Tensor> score_output, bool collect,
    std::optional<at::Tensor> ovf_storage, int64_t ovf_bucket_capacity,
    std::optional<at::Tensor> ovf_bucket_sizes,
    std::optional<at::Tensor> ovf_counter,
    std::optional<at::Tensor> ovf_output_offsets, int64_t num_scores) {

  TORCH_CHECK(num_scores == 1 && policy_type != ScorePolicyType::LruLfu,
              "the sorted insert takes one score word; LRU_LFU keeps two");
  auto i64 = at::TensorOptions().dtype(at::kLong).device(keys.device());
  int64_t num_total = keys.size(0);
  if (num_total == 0) {
    return std::make_tuple(collect ? at::zeros({1}, i64) : at::empty({0}, i64),
                           at::empty({0}, keys.options()), at::empty({0}, i64),
                           at::empty({0}, i64), at::empty({0}, i64));
  }
  bool use_overflow =
      collect && ovf_storage.has_value() && ovf_storage.value().defined();
  if (use_overflow) {
    TORCH_CHECK(
        counter.has_value() && ovf_bucket_sizes.has_value() &&
            ovf_counter.has_value() && ovf_output_offsets.has_value(),
        "the sorted insert with an overflow bucket needs a ref counter, "
        "the overflow bucket sizes, counters and output offsets");
  }

  auto key_type = get_data_type(keys);
  int64_t bucket_bytes =
      bucket_capacity * (keys.element_size() + sizeof(DigestType) +
                         num_scores * sizeof(ScoreType));
  int64_t num_buckets =
      table_storage.numel() * table_storage.element_size() / bucket_bytes;
  int64_t num_tables = table_bucket_offsets.numel() - 1;

  // 1. sort by (home bucket, key)
  auto sorted = bucketize_keys_segmented(keys, table_ids, table_bucket_offsets,
                                         num_buckets, bucket_capacity);
  at::Tensor sorted_keys = sorted[0];
  at::Tensor inverse = sorted[1];
  at::Tensor segments = sorted[2];

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
  InsertResult *insert_results_ptr = nullptr;
  if (insert_results.has_value() && insert_results.value().defined()) {
    insert_results_ptr = get_pointer<InsertResult>(insert_results);
  }
  auto bucket_sizes_ptr = get_pointer<int>(bucket_sizes);
  auto table_ids_ptr = table_ids.data_ptr<int64_t>();
  auto table_bucket_offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  int32_t *counter_ptr = nullptr;
  if (counter.has_value() && counter.value().defined()) {
    counter_ptr = counter.value().data_ptr<int32_t>();
  }
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // What an eviction reports, at sorted positions: written by the kernel at
  // every position (the main kernel for all but the spills, the overflow pass
  // for those), then compacted.
  at::Tensor rec_block, rec_flag, overflow_flag, busy, num_busy;
  // The slots the keys took, one pointer per sorted position: they stay locked
  // until the run of their bucket is done (the main kernel), then the buffer
  // holds the slots the overflow spills took until the overflow pass is done.
  at::Tensor slot_scratch = at::empty({num_total}, i64);
  if (collect) {
    // rows: keys, scores, indices, table ids
    rec_block = at::empty({4, num_total}, i64);
    rec_flag = at::empty({num_total}, i64.dtype(at::kBool));
  }
  if (use_overflow) {
    overflow_flag = at::empty({num_total}, i64.dtype(at::kBool));
    busy = at::empty({num_total}, i64);
    num_busy = at::zeros({1}, i64);
  }

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    using Bucket = LinearBucket<KeyType>;
    using Table = LinearBucketTable<Bucket>;
    auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                       num_buckets, bucket_capacity, num_scores);
    EvictRecords<KeyType> rec{nullptr, nullptr, nullptr, nullptr, nullptr};
    if (collect) {
      rec.flag = rec_flag.data_ptr<bool>();
      rec.keys = reinterpret_cast<KeyType *>(rec_block[0].data_ptr());
      rec.scores = rec_block[1].data_ptr<int64_t>();
      rec.indices = rec_block[2].data_ptr<int64_t>();
      rec.table_ids = rec_block[3].data_ptr<int64_t>();
    }
    Table ovf_table;
    if (use_overflow) {
      int64_t ovf_bucket_bytes =
          ovf_bucket_capacity * (keys.element_size() + sizeof(DigestType) +
                                 num_scores * sizeof(ScoreType));
      int64_t ovf_num_buckets = ovf_storage.value().numel() *
                                ovf_storage.value().element_size() /
                                ovf_bucket_bytes;
      TORCH_CHECK(ovf_num_buckets >= num_tables,
                  "the overflow storage has a bucket for every table");
      ovf_table =
          Table(reinterpret_cast<uint8_t *>(ovf_storage.value().data_ptr()),
                ovf_num_buckets, ovf_bucket_capacity, num_scores);
    }
    auto sorted_keys_ptr = get_pointer<KeyType>(sorted_keys);
    auto segments_ptr = segments.data_ptr<int64_t>();
    auto inverse_ptr = inverse.data_ptr<int64_t>();
    auto slot_scratch_ptr =
        reinterpret_cast<KeyType **>(slot_scratch.data_ptr());

    DISPATCH_SCORE_POLICY(policy_type, PolicyTypeV, [&] {
      // 2. insert the keys of each bucket, one thread per run
      if (use_overflow) {
        launch_table_insert_sorted_kernel<Table, PolicyTypeV, true>(
            table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
            sorted_keys_ptr, segments_ptr, inverse_ptr, table_ids_ptr,
            score_input_ptr, insert_results_ptr, score_output_ptr, counter_ptr,
            slot_scratch_ptr, rec, overflow_flag.data_ptr<bool>(), stream);
        // 3. the keys whose bucket answered Busy, in sorted order, go to the
        // overflow bucket of their table: one thread per table.
        select_index_async<int64_t, int64_t>(
            num_total, overflow_flag.data_ptr<bool>(), busy.data_ptr<int64_t>(),
            num_busy.data_ptr<int64_t>(), keys.device(), stream);
        launch_table_overflow_sorted_kernel<Table, PolicyTypeV>(
            ovf_table, table_bucket_offsets_ptr, num_tables,
            get_pointer<int>(ovf_bucket_sizes),
            ovf_counter.value().data_ptr<int32_t>(),
            ovf_output_offsets.value().data_ptr<int64_t>(), sorted_keys_ptr,
            segments_ptr, inverse_ptr, busy.data_ptr<int64_t>(),
            num_busy.data_ptr<int64_t>(), slot_scratch_ptr, insert_results_ptr,
            score_output_ptr, rec, stream);
      } else {
        launch_table_insert_sorted_kernel<Table, PolicyTypeV, false>(
            table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
            sorted_keys_ptr, segments_ptr, inverse_ptr, table_ids_ptr,
            score_input_ptr, insert_results_ptr, score_output_ptr, counter_ptr,
            slot_scratch_ptr, rec, nullptr, stream);
      }
    });
  });

  if (!collect) {
    return std::make_tuple(at::empty({0}, i64), at::empty({0}, keys.options()),
                           at::empty({0}, i64), at::empty({0}, i64),
                           at::empty({0}, i64));
  }

  // 4. Compact the evictions in sorted order; the count stays on the device.
  // The arrays are inert past it: key -1 (all ones, the reserved key of both
  // key types), index -1, score 0 and table id -1.
  at::Tensor num_evicted = at::zeros({1}, i64);
  at::Tensor evicted_rest = at::full({3, num_total}, -1, i64);
  at::Tensor evicted_scores = at::zeros({num_total}, i64);
  at::Tensor evicted_keys = evicted_rest[0].view(keys.scalar_type());
  at::Tensor evicted_indices = evicted_rest[1];
  at::Tensor evicted_table_ids = evicted_rest[2];
  at::Tensor position_scratch = at::empty({num_total}, i64);
  std::vector<int64_t const *> in_ptrs = {
      reinterpret_cast<int64_t const *>(rec_block[0].data_ptr()),
      rec_block[1].data_ptr<int64_t>(), rec_block[2].data_ptr<int64_t>(),
      rec_block[3].data_ptr<int64_t>()};
  std::vector<int64_t *> out_ptrs = {
      reinterpret_cast<int64_t *>(evicted_keys.data_ptr()),
      evicted_scores.data_ptr<int64_t>(), evicted_indices.data_ptr<int64_t>(),
      evicted_table_ids.data_ptr<int64_t>()};
  flagged_compact_cub<4>(num_total, rec_flag.data_ptr<bool>(), in_ptrs.data(),
                         out_ptrs.data(), position_scratch.data_ptr<int64_t>(),
                         num_evicted.data_ptr<int64_t>(), keys.device(),
                         stream);

  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(num_evicted, evicted_keys, evicted_indices,
                         evicted_scores, evicted_table_ids);
}

} // anonymous namespace

void table_insert_sorted(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, std::optional<at::Tensor> score_input,
    ScorePolicyType policy_type, std::optional<at::Tensor> counter,
    std::optional<at::Tensor> insert_results,
    std::optional<at::Tensor> score_output, int64_t num_scores) {
  table_insert_sorted_impl(table_storage, table_bucket_offsets, bucket_capacity,
                           bucket_sizes, keys, table_ids, score_input,
                           policy_type, counter, insert_results, score_output,
                           /*collect=*/false, std::nullopt, 0, std::nullopt,
                           std::nullopt, std::nullopt, num_scores);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
table_insert_and_evict_sorted(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, std::optional<at::Tensor> score_input,
    ScorePolicyType policy_type, std::optional<at::Tensor> counter,
    std::optional<at::Tensor> insert_results,
    std::optional<at::Tensor> score_output,
    std::optional<at::Tensor> ovf_storage, int64_t ovf_bucket_capacity,
    std::optional<at::Tensor> ovf_bucket_sizes,
    std::optional<at::Tensor> ovf_counter,
    std::optional<at::Tensor> ovf_output_offsets, int64_t num_scores) {
  return table_insert_sorted_impl(
      table_storage, table_bucket_offsets, bucket_capacity, bucket_sizes, keys,
      table_ids, score_input, policy_type, counter, insert_results,
      score_output, /*collect=*/true, ovf_storage, ovf_bucket_capacity,
      ovf_bucket_sizes, ovf_counter, ovf_output_offsets, num_scores);
}

} // namespace dyn_emb
