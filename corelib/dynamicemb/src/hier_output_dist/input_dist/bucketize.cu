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
// The bucketize of a call: its keys by owner, as TorchRec's bucketize orders
// them, and index dedup's fan-out (planner.h).

#include <c10/util/Exception.h>

#include <cstdint>
#include <type_traits>

#include "hier_output_dist/input_dist/node.cuh"
#include "hier_output_dist/input_dist/planner.cuh"
#include "hier_output_dist/input_dist/planner.h"
#include "murmur_hash.cuh"
#include "utils.h"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

// The KJT's keys, offsets and lengths are each int32 or int64: the kernels
// that read them take the type as a template argument (Key, Length,
// Offset), and with_index calls `launch` with the tensor's data as a pointer
// of its dtype; an absent tensor as a null int64_t pointer, which the kernel
// does not read.
template <typename Launch>
void with_index(const at::Tensor &tensor, Launch &&launch) {
  if (tensor.scalar_type() == at::kInt)
    launch(tensor.data_ptr<int32_t>());
  else
    launch(tensor.data_ptr<int64_t>());
}
template <typename Launch>
void with_index(const c10::optional<at::Tensor> &tensor, Launch &&launch) {
  if (tensor.has_value())
    with_index(*tensor, launch);
  else
    launch(static_cast<const int64_t *>(nullptr));
}

// ── the KJT's features ──

// One CTA per feature of a KJT of lengths: its keys, the sum of its batch
// lengths.
template <typename Length>
__global__ void __launch_bounds__(kThreads)
    feature_totals_kernel(const Length *lengths, int64_t batch, int64_t *totals) {
  using Reduce = cub::BlockReduce<int64_t, kThreads>;
  __shared__ typename Reduce::TempStorage reduce;
  const int64_t feature = blockIdx.x;
  int64_t mine = 0;
  for (int64_t sample = threadIdx.x; sample < batch; sample += kThreads)
    mine += lengths[feature * batch + sample];
  const int64_t total = Reduce(reduce).Sum(mine);
  if (threadIdx.x == 0) totals[feature] = total;
}

// One CTA: where each feature's keys start in the KJT (value_start
// [num_features + 1], from its first key), from its offsets (every
// batch-th) or, for a KJT of lengths (offsets nullptr), from the features'
// totals; and the totals construct_jagged_tensors reads
// (input_length_per_key [num_features]), of the input KJT's offsets.
template <typename Offset, typename InputOffset>
__global__ void __launch_bounds__(kThreads)
    feature_starts_kernel(const Offset *offsets, const int64_t *totals, int64_t batch,
                          int64_t num_features, const InputOffset *input_offsets,
                          int64_t *value_start, int64_t *input_length_per_key) {
  if (offsets != nullptr) {
    for (int64_t feature = threadIdx.x; feature <= num_features; feature += kThreads)
      value_start[feature] = (int64_t)offsets[feature * batch] - (int64_t)offsets[0];
    __syncthreads();
  } else {
    cta_exclusive_ends<int64_t>(
        num_features, [&](int64_t feature) { return totals[feature]; }, value_start);
  }
  for (int64_t feature = threadIdx.x; feature < num_features; feature += kThreads)
    input_length_per_key[feature] =
        (int64_t)input_offsets[(feature + 1) * batch] - (int64_t)input_offsets[feature * batch];
}

// ── the owner rules ──

// The owner rules the op takes (DistType, src/utils.h), as DynamicEmb's
// bucketize places a key (src/sparse_block_bucketize_features.cu): in the
// arithmetic of the key's own type (int32_t or int64_t), as unsigned numbers.
// Both keep the key. Config::validate and the bucketize's hook refuse every
// other code.

// kCyclic ("roundrobin"): key % world_size.
template <typename Key>
__device__ __forceinline__ int64_t cyclic_owner(Key key, int64_t world_size) {
  using Unsigned = std::make_unsigned_t<Key>;
  return (int64_t)((Unsigned)key % (Unsigned)world_size);
}

// kHashedCyclic ("hash_roundrobin"): murmur3_fmix64(key) % world_size, the hash
// of an int32 key (zero-extended) taken to its low 32 bits.
template <typename Key>
__device__ __forceinline__ int64_t hashed_cyclic_owner(Key key, int64_t world_size) {
  using Unsigned = std::make_unsigned_t<Key>;
  const Unsigned hash = (Unsigned)murmur3_fmix64((Unsigned)key);
  return (int64_t)(hash % (Unsigned)world_size);
}

// A key's owner by its feature's rule, read at run time (a sharding's tables
// may mix the two). A warp's keys are consecutive, so they are of one feature
// and take one branch, except at the start of a feature.
template <typename Key>
__device__ __forceinline__ int64_t owner_of(Key key, int32_t dist_type, int64_t world_size) {
  return dist_type == (int)DistType::kCyclic ? cyclic_owner(key, world_size)
                                             : hashed_cyclic_owner(key, world_size);
}

// A thread per key of the KJT: its bucket feature, owner * num_features +
// feature with its owner by its feature's rule, or world_size * num_features
// past the valid keys (index dedup's bound); `index` the key's own index,
// which the sort carries. The rules keep the key, so the keys stay in the
// KJT: the push gathers them from there.
template <typename Key>
__global__ void __launch_bounds__(kThreads)
    place_keys_kernel(const Key *values, int64_t num_keys, const int64_t *value_start,
                      int64_t num_features, const int32_t *dist_types, int64_t world_size,
                      int32_t *bucket_feature, int32_t *index) {
  for (int64_t key = first_item(); key < num_keys; key += item_stride()) {
    const int64_t feature = upper_bound(value_start, num_features + 1, key) - 1;
    bucket_feature[key] =
        feature < num_features
            ? (int32_t)(owner_of(values[key], dist_types[feature], world_size) * num_features +
                        feature)
            : (int32_t)(world_size * num_features);
    index[key] = (int32_t)key;
  }
}

// A thread per position of the keys sorted by bucket feature (gather_rows,
// node.cuh): the key there, widened to int64 with its sign, and its weight
// (key_weights nullptr: none).
__global__ void __launch_bounds__(kThreads)
    gather_keys_kernel(GatherRows rows, int64_t num_keys, int64_t *keys, float *key_weights) {
  gather_rows(rows, num_keys, [&](int64_t row, int64_t key, float weight) {
    keys[row] = key;
    if (key_weights != nullptr) key_weights[row] = weight;
  });
}

// ── index dedup's fan-out ──

// A thread per original row: its deduplicated row as an int32 sort key, and
// its own index, which the sort carries.
__global__ void __launch_bounds__(kThreads)
    original_rows_kernel(const int64_t *reverse_indices, int64_t originals, int32_t *row,
                         int32_t *index) {
  for (int64_t original = first_item(); original < originals; original += item_stride()) {
    row[original] = (int32_t)reverse_indices[original];
    index[original] = (int32_t)original;
  }
}

// A thread per deduplicated row (and per bucketized row, both `rows`):
// whether the row is heavy (more original rows than kHeavy) and its
// sub-tasks of kHeavy original rows; and where the bucketize put each row
// (position_of, the inverse of inv).
__global__ void __launch_bounds__(kThreads)
    heavy_counts_kernel(const int32_t *exp_off, const int32_t *inv, int64_t rows, int32_t *heavy,
                        int64_t *subs, int32_t *position_of) {
  for (int64_t row = first_item(); row < rows; row += item_stride()) {
    const int64_t fan = exp_off[row + 1] - exp_off[row];
    heavy[row] = fan > kHeavy;
    subs[row] = fan > kHeavy ? ceil_div<kHeavy>(fan) : 0;
    position_of[inv[row]] = (int32_t)row;
  }
}

// A thread per deduplicated row: its slot (heavy rows numbered in row order,
// heavy_before) or -1, and for a heavy row its bucketized row and its first
// sub-task. The thread past the rows writes the counts.
__global__ void __launch_bounds__(kThreads)
    heavy_rows_kernel(const int32_t *heavy_before, const int64_t *sub_before,
                      const int32_t *position_of, int64_t rows, int32_t *slot, int32_t *pos,
                      int32_t *first, int64_t *count) {
  for (int64_t row = first_item(); row <= rows; row += item_stride()) {
    if (row == rows) {
      count[0] = heavy_before[rows];
      count[1] = sub_before[rows];
      continue;
    }
    const bool is_heavy = heavy_before[row + 1] > heavy_before[row];
    slot[row] = is_heavy ? heavy_before[row] : -1;
    if (!is_heavy) continue;
    const int32_t heavy_slot = heavy_before[row];
    pos[heavy_slot] = position_of[row];
    first[heavy_slot] = (int32_t)sub_before[row];
  }
}

// A thread per sub-task the table holds (max_subs), up to the sub-tasks
// there are: its row, the last whose sub-tasks start at or before it, and
// its first entry in exp_idx, kHeavy entries after the sub-task before it.
__global__ void __launch_bounds__(kThreads)
    heavy_subs_kernel(const int32_t *exp_off, const int64_t *sub_before, int64_t rows,
                      int64_t max_subs, int64_t *sub) {
  const int64_t subs = sub_before[rows];
  for (int64_t task = first_item(); task < min(subs, max_subs); task += item_stride()) {
    const int64_t row = upper_bound(sub_before, rows + 1, task) - 1;
    sub[2 * task] = row;
    sub[2 * task + 1] = exp_off[row] + (task - sub_before[row]) * kHeavy;
  }
}

// contiguous, on the GPU of the keys
bool on_gpu_of(const at::Tensor &tensor, const at::Tensor &values) {
  return tensor.is_cuda() && tensor.get_device() == values.get_device() && tensor.is_contiguous();
}
bool is_index_dtype(const at::Tensor &tensor) {
  return tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong;
}

}  // namespace

void check_bucketize_input(const PlanInput &input, const at::Tensor &dist_types,
                           int64_t world_size) {
  const at::Tensor &values = input.values;
  const int64_t num_keys = values.numel(), batch = input.batch_size,
                num_features = dist_types.numel();
  TORCH_CHECK(num_features >= 1 && world_size >= 1,
              "hier_output_dist: a rule per feature, and some owners");
  TORCH_CHECK(on_gpu_of(dist_types, values) && dist_types.scalar_type() == at::kInt,
              "hier_output_dist: the rules' dist types (int32) must be contiguous on the GPU of "
              "the keys");
  TORCH_CHECK(
      values.is_cuda() && values.is_contiguous() && is_index_dtype(values) && num_keys < INT32_MAX,
      "hier_output_dist: keys must be contiguous int32 or int64 on the GPU, fewer than 2^31");
  TORCH_CHECK(batch > 0, "hier_output_dist: a KJT of at least one sample");
  TORCH_CHECK(input.offsets.has_value() != input.lengths.has_value(),
              "hier_output_dist: the KJT's offsets or its lengths");
  const at::Tensor &bounds = input.offsets.has_value() ? *input.offsets : *input.lengths;
  TORCH_CHECK(on_gpu_of(bounds, values) && is_index_dtype(bounds) &&
                  bounds.numel() == num_features * batch + input.offsets.has_value(),
              "hier_output_dist: offsets (num_features * batch_size + 1) or lengths "
              "(num_features * batch_size) must be contiguous int32 or int64 on the GPU of the "
              "keys");
  TORCH_CHECK(on_gpu_of(input.input_offsets, values) && is_index_dtype(input.input_offsets) &&
                  input.input_offsets.numel() == num_features * batch + 1,
              "hier_output_dist: the input KJT's offsets must be contiguous int32 or int64 on "
              "the GPU of the keys, num_features * batch_size + 1 of them");
  TORCH_CHECK(!input.weights.has_value() || (on_gpu_of(*input.weights, values) &&
                                             input.weights->scalar_type() == at::kFloat &&
                                             input.weights->numel() == num_keys),
              "hier_output_dist: weights must be contiguous fp32, one per key");
}

void check_fanout_input(const at::Tensor &reverse_indices) {
  // Original rows are int32 (exp_idx).
  TORCH_CHECK(reverse_indices.is_cuda() && reverse_indices.is_contiguous() &&
                  reverse_indices.scalar_type() == at::kLong && reverse_indices.numel() < INT32_MAX,
              "hier_output_dist: reverse indices must be contiguous int64 on the GPU, fewer "
              "than 2^31");
}

BucketizedKeys bucketize_rows(const PlanInput &input, const at::Tensor &dist_types,
                              int64_t world_size) {
  check_bucketize_input(input, dist_types, world_size);
  const at::Tensor &values = input.values;
  const int64_t num_keys = values.numel(), batch = input.batch_size,
                num_features = dist_types.numel();
  const int64_t bucket_features = world_size * num_features;
  const auto long_options = values.options().dtype(at::kLong);
  const auto int_options = values.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();

  // where each feature's keys start in the KJT, and the input's totals
  const at::Tensor value_start = at::empty({num_features + 1}, long_options);
  BucketizedKeys bucketized;
  bucketized.input_length_per_key = at::empty({num_features}, long_options);
  at::Tensor totals;
  if (input.lengths.has_value()) {
    totals = at::empty({num_features}, long_options);
    with_index(*input.lengths, [&](const auto *lengths) {
      feature_totals_kernel<<<num_features, kThreads, 0, stream>>>(lengths, batch,
                                                                   totals.data_ptr<int64_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  with_index(input.offsets, [&](const auto *offsets) {
    with_index(input.input_offsets, [&](const auto *input_offsets) {
      feature_starts_kernel<<<kOneCta, kThreads, 0, stream>>>(
          offsets, data_or_null<int64_t>(totals), batch, num_features, input_offsets,
          value_start.data_ptr<int64_t>(), bucketized.input_length_per_key.data_ptr<int64_t>());
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // the bucket feature of each key, then a stable sort by bucket feature: a
  // bucket feature's keys keep the KJT's order, and the keys past the valid
  // ones stay last, in order
  const at::Tensor bucket_feature = at::empty({num_keys}, int_options);
  const at::Tensor index = at::empty({num_keys}, int_options);
  if (num_keys > 0) {
    with_index(values, [&](const auto *keys) {
      place_keys_kernel<<<item_grid(num_keys), kThreads, 0, stream>>>(
          keys, num_keys, value_start.data_ptr<int64_t>(), num_features,
          dist_types.data_ptr<int32_t>(), world_size, bucket_feature.data_ptr<int32_t>(),
          index.data_ptr<int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  std::tie(bucketized.bucket_feature_of_row, bucketized.inv) =
      sort_pairs(bucket_feature, index, bits_for(bucket_features + 1));
  // where each bucket feature, and each bucket (owner), starts
  bucketized.offset_per_bucket_feature = at::empty({bucket_features + 1}, long_options);
  bucketized.offset_per_bucket = at::empty({world_size + 1}, long_options);
  run_starts<int32_t, int64_t>(bucketized.bucket_feature_of_row, bucket_features,
                               bucketized.offset_per_bucket_feature, num_features,
                               bucketized.offset_per_bucket);
  // the keys stay in the KJT: the push gathers them (bucketize_keys here)
  bucketized.values = values;
  return bucketized;
}

BucketizedKeys bucketize_keys(const PlanInput &input, const at::Tensor &dist_types,
                              int64_t world_size) {
  BucketizedKeys bucketized = bucketize_rows(input, dist_types, world_size);
  const at::Tensor &values = input.values;
  const int64_t num_keys = values.numel();
  // the keys (and weights) gathered into bucketized order
  bucketized.keys = at::empty({num_keys}, values.options().dtype(at::kLong));
  if (input.weights.has_value())
    bucketized.key_weights = at::empty({num_keys}, values.options().dtype(at::kFloat));
  if (num_keys > 0) {
    const GatherRows rows =
        gather_rows_of(bucketized.inv, bucketized.values, input.weights.value_or(at::Tensor()));
    gather_keys_kernel<<<item_grid(num_keys), kThreads, 0, current_stream()>>>(
        rows, num_keys, bucketized.keys.data_ptr<int64_t>(),
        data_or_null<float>(bucketized.key_weights));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return bucketized;
}

FanoutTables fanout_tables(const at::Tensor &reverse_indices, const at::Tensor &inv) {
  const int64_t rows = inv.numel(), originals = reverse_indices.numel();
  // Original and deduplicated rows are int32 (exp_idx, exp_off).
  check_fanout_input(reverse_indices);
  TORCH_CHECK(inv.scalar_type() == at::kInt, "hier_output_dist: inv must be int32");
  const int64_t max_slots = originals / (kHeavy + 1) + 1, max_subs = 2 * originals / kHeavy + 2;
  const auto long_options = inv.options().dtype(at::kLong);
  const auto int_options = inv.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  FanoutTables fanout;
  // each row's original rows: a stable sort of the reverse indices, which
  // keeps them in increasing order
  const at::Tensor original_row = at::empty({originals}, int_options);
  const at::Tensor index = at::empty({originals}, int_options);
  if (originals > 0) {
    original_rows_kernel<<<item_grid(originals), kThreads, 0, stream>>>(
        reverse_indices.data_ptr<int64_t>(), originals, original_row.data_ptr<int32_t>(),
        index.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  at::Tensor sorted_row;
  std::tie(sorted_row, fanout.exp_idx) = sort_pairs(original_row, index, bits_for(rows));
  fanout.exp_off = at::empty({rows + 1}, int_options);
  run_starts<int32_t, int32_t>(sorted_row, rows, fanout.exp_off);
  // heavy rows: slots and sub-tasks numbered by scans, in row order
  const at::Tensor heavy = at::empty({rows}, int_options);
  const at::Tensor subs = at::empty({rows}, long_options);
  const at::Tensor position_of = at::empty({rows}, int_options);
  if (rows > 0) {
    heavy_counts_kernel<<<item_grid(rows), kThreads, 0, stream>>>(
        fanout.exp_off.data_ptr<int32_t>(), inv.data_ptr<int32_t>(), rows,
        heavy.data_ptr<int32_t>(), subs.data_ptr<int64_t>(), position_of.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const at::Tensor heavy_before = exclusive_ends(heavy, at::kInt);
  const at::Tensor sub_before = exclusive_ends(subs, at::kLong);
  fanout.slot = at::empty({rows}, int_options);
  fanout.sub = at::empty({2 * max_subs}, long_options);
  fanout.count = at::empty({2}, long_options);
  fanout.pos = at::empty({max_slots}, int_options);
  fanout.first = at::empty({max_slots}, int_options);
  heavy_rows_kernel<<<item_grid(rows + 1), kThreads, 0, stream>>>(
      heavy_before.data_ptr<int32_t>(), sub_before.data_ptr<int64_t>(),
      position_of.data_ptr<int32_t>(), rows, fanout.slot.data_ptr<int32_t>(),
      fanout.pos.data_ptr<int32_t>(), fanout.first.data_ptr<int32_t>(),
      fanout.count.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  heavy_subs_kernel<<<item_grid(max_subs), kThreads, 0, stream>>>(
      fanout.exp_off.data_ptr<int32_t>(), sub_before.data_ptr<int64_t>(), rows, max_subs,
      fanout.sub.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return fanout;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
