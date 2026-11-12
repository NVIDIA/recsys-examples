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

#include <algorithm>

namespace dyn_emb {

// Erase the keys of the slots [begin, end) whose score is below ``threshold``,
// sparing those a prefetched batch has pinned. ``counts`` is an int64 tensor of
// two on the device, ``[erased, spared]``, that this adds to: the caller zeroes it
// once and passes it for every chunk, and nothing here waits for the GPU.
//
// ``out_keys`` / ``out_slots`` / ``out_count``, all or none, record the erase: the
// key and the global slot of each erased slot, ``out_count`` (int64 of one, zeroed by
// the caller) the number written. They must hold ``end - begin`` entries.
//
// The binding releases the GIL (see docs/async_erase_design.md, 4.5): it takes only
// tensors and integers and never touches a Python object.
void table_erase_by_score(at::Tensor table_storage, torch::Dtype key_dtype,
                          int64_t bucket_capacity, at::Tensor bucket_sizes,
                          std::optional<at::Tensor> counter,
                          ScoreType threshold, int64_t begin, int64_t end,
                          int64_t num_scores, int64_t score_index,
                          at::Tensor counts, std::optional<at::Tensor> out_keys,
                          std::optional<at::Tensor> out_slots,
                          std::optional<at::Tensor> out_count) {
  TORCH_CHECK(table_storage.is_cuda(),
              "table_erase_by_score: table_storage must be a CUDA tensor");
  // [erased, spared] are written as two neighbours from one pointer.
  TORCH_CHECK(counts.is_cuda() && counts.device() == table_storage.device() &&
                  counts.scalar_type() == torch::kInt64 && counts.numel() == 2 &&
                  counts.is_contiguous(),
              "table_erase_by_score: counts must be a contiguous int64 tensor "
              "of 2 on the table's device");
  TORCH_CHECK(out_keys.has_value() == out_slots.has_value() &&
                  out_keys.has_value() == out_count.has_value(),
              "table_erase_by_score: out_keys, out_slots and out_count are "
              "given together or not at all");
  TORCH_CHECK(bucket_capacity > 0,
              "table_erase_by_score: bucket_capacity must be positive");
  TORCH_CHECK(score_index >= 0 && score_index < num_scores,
              "table_erase_by_score: score_index out of range");

  auto key_type = scalartype_to_datatype(toScalarType(key_dtype));
  auto bucket_sizes_ = get_pointer<int>(bucket_sizes);
  auto counter_ = get_pointer<int32_t>(counter);
  auto num_erased = get_pointer<CounterType>(counts);
  auto num_spared = num_erased + 1;

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  constexpr int BLOCK_SIZE = 256;

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    int64_t total_size =
        sizeof(KeyType) + sizeof(DigestType) + num_scores * sizeof(ScoreType);
    int64_t bucket_bytes = bucket_capacity * total_size;
    int64_t num_buckets =
        table_storage.numel() * table_storage.element_size() / bucket_bytes;
    int64_t num_slots = num_buckets * bucket_capacity;

    TORCH_CHECK(begin >= 0 && begin <= end && end <= num_slots,
                "table_erase_by_score: the range [", begin, ", ", end,
                ") is not inside the table's ", num_slots, " slots");
    // What the kernel indexes by bucket and by slot: one size per bucket, one
    // counter per slot.
    TORCH_CHECK(bucket_sizes.is_cuda() &&
                    bucket_sizes.device() == table_storage.device() &&
                    bucket_sizes.scalar_type() == torch::kInt32 &&
                    bucket_sizes.is_contiguous() &&
                    bucket_sizes.numel() == num_buckets,
                "table_erase_by_score: bucket_sizes must be a contiguous int32 "
                "tensor with one entry for each of the ",
                num_buckets, " buckets");
    if (counter.has_value()) {
      auto const &c = counter.value();
      TORCH_CHECK(c.is_cuda() && c.device() == table_storage.device() &&
                      c.scalar_type() == torch::kInt32 && c.is_contiguous() &&
                      c.numel() >= num_slots,
                  "table_erase_by_score: counter must be a contiguous int32 "
                  "tensor with one entry for each of the ",
                  num_slots, " slots");
    }

    int64_t num_total = end - begin;
    if (out_keys.has_value()) {
      auto const &keys = out_keys.value();
      auto const &slots = out_slots.value();
      auto const &written = out_count.value();
      TORCH_CHECK(keys.is_cuda() && keys.device() == table_storage.device() &&
                      keys.scalar_type() == key_dtype && keys.is_contiguous() &&
                      keys.numel() >= num_total,
                  "table_erase_by_score: out_keys must be a contiguous tensor "
                  "of the key dtype with room for ",
                  num_total, " keys on the table's device");
      TORCH_CHECK(slots.is_cuda() && slots.device() == table_storage.device() &&
                      slots.scalar_type() == torch::kInt64 &&
                      slots.is_contiguous() && slots.numel() >= num_total,
                  "table_erase_by_score: out_slots must be a contiguous int64 "
                  "tensor with room for ",
                  num_total, " slots on the table's device");
      TORCH_CHECK(written.is_cuda() &&
                      written.device() == table_storage.device() &&
                      written.scalar_type() == torch::kInt64 &&
                      written.numel() == 1 && written.is_contiguous(),
                  "table_erase_by_score: out_count must be a contiguous int64 "
                  "tensor of 1 on the table's device");
    }
    if (num_total == 0)
      return;
    auto out_keys_ = get_pointer<KeyType>(out_keys);
    auto out_slots_ = get_pointer<int64_t>(out_slots);
    auto out_count_ = get_pointer<CounterType>(out_count);

    using Bucket = LinearBucket<KeyType>;
    using Table = LinearBucketTable<Bucket>;

    auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                       num_buckets, bucket_capacity, num_scores);

    // The kernel strides over the range, so a long one needs no more blocks than
    // this; a grid is limited to 2^31 - 1 blocks, and the launch takes it as a
    // 32-bit value.
    const int64_t grid = std::min<int64_t>(
        (num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, int64_t(1) << 20);

    // A tile of 32 votes together, which needs every lane to run the same
    // iterations: a range of whole warps. A chunk of whole buckets always is.
    if (num_total % 32 == 0) {
      table_erase_by_score_kernel<Table, 32><<<grid, BLOCK_SIZE, 0, stream>>>(
          table, begin, end, threshold, score_index, bucket_sizes_, counter_,
          num_erased, num_spared, out_keys_, out_slots_,
          out_count_);
    } else {
      table_erase_by_score_kernel<Table, 1><<<grid, BLOCK_SIZE, 0, stream>>>(
          table, begin, end, threshold, score_index, bucket_sizes_, counter_,
          num_erased, num_spared, out_keys_, out_slots_,
          out_count_);
    }
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace dyn_emb
