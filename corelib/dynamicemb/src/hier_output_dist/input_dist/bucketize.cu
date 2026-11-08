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

#include <algorithm>
#include <cstdint>
#include <tuple>
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

// A thread per original row and per bucketized row (`rows` of each): an
// original row's deduplicated row as an int32 sort key, and its own index,
// which the sort carries; and where the bucketize put each deduplicated row
// (position_of, the inverse of inv).
__global__ void __launch_bounds__(kThreads)
    original_rows_kernel(const int64_t *reverse_indices, int64_t originals, const int32_t *inv,
                         int64_t rows, int32_t *row, int32_t *index, int32_t *position_of) {
  for (int64_t item = first_item(); item < max(originals, rows); item += item_stride()) {
    if (item < originals) {
      row[item] = (int32_t)reverse_indices[item];
      index[item] = (int32_t)item;
    }
    if (item < rows) position_of[inv[item]] = (int32_t)item;
  }
}

// The end of the run of `row` in sorted[0, n) (ascending) that holds
// position `at`: the first position past it. The whole warp searches: each
// round its lanes probe the 32 positions `step` apart after lo, the last
// position known in the run. While all of them are in the run the steps
// grow 32 times (galloping); then each round narrows the end 32 times. A run
// whose last position is at most 31 past `at` takes one round, a run of a
// million positions seven. Every lane of the warp, with the same arguments;
// every lane gets the end. The probes stay below hi (at most n), whatever the
// table holds.
__device__ __forceinline__ int64_t warp_run_end(const int32_t *sorted, int64_t n, int64_t at,
                                                int32_t row, int lane) {
  int64_t lo = at, hi = n, step = 1;  // the run ends in (lo, hi]
  while (hi - lo > 1) {
    const int64_t probe = lo + step * (lane + 1);
    const unsigned ballot = __ballot_sync(0xffffffffu, probe < hi && sorted[probe] == row);
    const int inside = __clz(__brev(~ballot));  // the lanes from lane 0 on whose probe is in it
    if (inside == 32) {
      lo += 32 * step;
      step *= 32;
    } else {
      hi = min(hi, lo + (inside + 1) * step);
      lo += inside * step;
      step = (hi - lo + 31) / 32;
    }
  }
  return hi;
}

// The fan-out's tables from the sorted rows (sorted: each original row's
// deduplicated row, ascending, n of them), in one kernel: a thread per sorted
// position, a warp's lanes at consecutive positions. Position q starts the
// run of row r = sorted[q] when q is 0 or sorted[q - 1] differs. Its thread
// writes exp_off[r] = q, and for the rows between the previous run's row and
// r, which have no original rows, exp_off q and slot -1. The rows before the
// first run get exp_off 0 and those after the last exp_off n (exp_off[rows]
// too), slot -1, in loops over the whole grid: index dedup leaves many rows
// past the valid ones. With no original rows, every row is such a row.
//
// The run is heavy when it has more than kHeavy positions: sorted[q + kHeavy]
// is r too (r not heavy: slot[r] = -1). Its warp takes it: the warp finds its
// end, and the run-start thread reserves its slot and its ceil(length /
// kHeavy) sub-tasks with one 64-bit atomicAdd of (1 << 32) | subs on the
// counter word: the old word's high half is its slot, its low half its first
// sub-task. So consecutive slots have consecutive sub-task ranges, and
// first[s + 1] - first[s] counts slot s's sub-tasks (the last slot's end at
// count[1]). The thread writes slot[r], pos (where the bucketize put r) and
// first; then the warp writes the run's sub-tasks (r, q + t * kHeavy), 32 at
// a time, so a row of a million original rows (15,625 sub-tasks) takes 489
// rounds of one warp, not 15,625 stores of one thread. Slots past max_slots
// and sub-tasks past max_subs are not written (reverse indices in [0, rows)
// never reach them).
//
// The heavy rows' slots are so numbered in the order their runs reserve them,
// which can differ between two calls on the same rows. No sum depends on the
// numbering: a row's sub-tasks stay consecutive and in original-row order, so
// the backward adds up each heavy row's original rows in the same order
// whatever its slot, and the gradient keeps its bits.
//
// count ([heavy rows, sub-tasks]) is the counter once every run reserved:
// each CTA, once its threads are done, takes a ticket (words[1]), and the
// CTA that takes the last ticket writes count from the counter, then resets
// both words to 0 for the next call. No CTA waits: a CTA takes its ticket and
// exits, and the last one goes on only because every other CTA already
// arrived. It is the last one out, not a spin on other CTAs, so it needs no
// CTA to run at the same time as another, and it keeps the planner's rule
// that no kernel waits for another CTA (planner.cuh). words: [counter,
// ticket], 0 when the kernel starts (kFanoutWords).
__global__ void __launch_bounds__(kThreads)
    fanout_runs_kernel(const int32_t *sorted, int64_t n, int64_t rows, const int32_t *position_of,
                       int64_t max_slots, int64_t max_subs, int32_t *exp_off, int32_t *slot,
                       int32_t *pos, int32_t *first, int64_t *sub, int64_t *count,
                       unsigned long long *words) {
  const int lane = threadIdx.x & 31;
  // the rows before the first run and after the last
  const int64_t head = n > 0 ? min(max((int64_t)sorted[0], (int64_t)0), rows + 1) : rows + 1;
  for (int64_t row = first_item(); row < head; row += item_stride()) {
    exp_off[row] = 0;
    if (row < rows) slot[row] = -1;
  }
  const int64_t tail =
      n > 0 ? min(max((int64_t)sorted[n - 1] + 1, (int64_t)0), rows + 1) : rows + 1;
  for (int64_t row = tail + first_item(); row <= rows; row += item_stride()) {
    exp_off[row] = (int32_t)n;
    if (row < rows) slot[row] = -1;
  }
  // a warp's positions at a time, so every lane takes part in its heavy runs
  for (int64_t base = first_item() - lane; base < n; base += item_stride()) {
    const int64_t q = base + lane;
    int32_t row = 0;
    bool heavy = false;  // q starts a heavy run
    if (q < n) {
      row = sorted[q];
      if (q == 0 || sorted[q - 1] != row) {
        // r, and the rows between the previous run's row and r (q 0: the head's loop took them)
        const int64_t from = q == 0 ? row : (int64_t)sorted[q - 1] + 1;
        for (int64_t gap = max(from, (int64_t)0); gap <= min((int64_t)row, rows); ++gap) {
          exp_off[gap] = (int32_t)q;
          if (gap < row && gap < rows) slot[gap] = -1;
        }
        if (row >= 0 && row < rows) {
          heavy = q + kHeavy < n && sorted[q + kHeavy] == row;
          if (!heavy) slot[row] = -1;
        }
      }
    }
    for (unsigned runs = __ballot_sync(0xffffffffu, heavy); runs != 0; runs &= runs - 1) {
      const int starter = __ffs(runs) - 1;
      const int64_t start = base + starter;
      const int32_t run_row = __shfl_sync(0xffffffffu, row, starter);
      const int64_t subs =
          ceil_div<kHeavy>(warp_run_end(sorted, n, start + kHeavy, run_row, lane) - start);
      unsigned long long reserved = 0;
      if (lane == starter) {
        reserved = atomicAdd(&words[0], (1ull << 32) | (unsigned long long)subs);
        const int64_t heavy_slot = (int64_t)(reserved >> 32);
        slot[run_row] = (int32_t)heavy_slot;
        if (heavy_slot < max_slots) {
          pos[heavy_slot] = position_of[run_row];
          first[heavy_slot] = (int32_t)(reserved & 0xffffffffull);
        }
      }
      const int64_t first_sub =
          (int64_t)(__shfl_sync(0xffffffffu, reserved, starter) & 0xffffffffull);
      for (int64_t task = lane; task < subs && first_sub + task < max_subs; task += 32) {
        sub[2 * (first_sub + task)] = run_row;
        sub[2 * (first_sub + task) + 1] = start + task * kHeavy;
      }
    }
  }
  // the last CTA out: count, and the words back to 0
  __syncthreads();  // every reservation of this CTA is made
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(&words[1], 1ull) == gridDim.x - 1) {
      __threadfence();
      const unsigned long long total = atomicExch(&words[0], 0ull);
      count[0] = (int64_t)(total >> 32);
      count[1] = (int64_t)(total & 0xffffffffull);
      atomicExch(&words[1], 0ull);
    }
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

FanoutTables fanout_tables(const at::Tensor &reverse_indices, const at::Tensor &inv,
                           const at::Tensor &words) {
  const int64_t rows = inv.numel(), originals = reverse_indices.numel();
  // Original and deduplicated rows are int32 (exp_idx, exp_off).
  check_fanout_input(reverse_indices);
  TORCH_CHECK(inv.scalar_type() == at::kInt, "hier_output_dist: inv must be int32");
  TORCH_CHECK(!words.defined() || (on_gpu_of(words, inv) && words.scalar_type() == at::kLong &&
                                   words.numel() == kFanoutWords),
              "hier_output_dist: the fan-out's words must be ", kFanoutWords,
              " contiguous int64 on the GPU of inv");
  const int64_t max_slots = originals / (kHeavy + 1) + 1, max_subs = 2 * originals / kHeavy + 2;
  const auto long_options = inv.options().dtype(at::kLong);
  const auto int_options = inv.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  FanoutTables fanout;
  // where the bucketize put each row, and each row's original rows: a stable
  // sort of the reverse indices, which keeps them in increasing order (the
  // sort's inputs live until it is done)
  const at::Tensor position_of = at::empty({rows}, int_options);
  at::Tensor sorted_row;
  {
    const at::Tensor original_row = at::empty({originals}, int_options);
    const at::Tensor index = at::empty({originals}, int_options);
    const int64_t items = std::max(originals, rows);
    if (items > 0) {
      original_rows_kernel<<<item_grid(items), kThreads, 0, stream>>>(
          reverse_indices.data_ptr<int64_t>(), originals, inv.data_ptr<int32_t>(), rows,
          original_row.data_ptr<int32_t>(), index.data_ptr<int32_t>(),
          position_of.data_ptr<int32_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    std::tie(sorted_row, fanout.exp_idx) = sort_pairs(original_row, index, bits_for(rows));
  }
  // the runs of the sorted rows, in one kernel: exp_off, and the heavy rows'
  // slots and sub-tasks, numbered in the order the kernel reserves them
  // (fanout_runs_kernel); the hook's call gets words of its own
  fanout.exp_off = at::empty({rows + 1}, int_options);
  fanout.slot = at::empty({rows}, int_options);
  fanout.sub = at::empty({2 * max_subs}, long_options);
  fanout.count = at::empty({2}, long_options);
  fanout.pos = at::empty({max_slots}, int_options);
  fanout.first = at::empty({max_slots}, int_options);
  const at::Tensor counter = words.defined() ? words : at::zeros({kFanoutWords}, long_options);
  fanout_runs_kernel<<<item_grid(std::max(originals, rows + 1)), kThreads, 0, stream>>>(
      sorted_row.data_ptr<int32_t>(), originals, rows, position_of.data_ptr<int32_t>(), max_slots,
      max_subs, fanout.exp_off.data_ptr<int32_t>(), fanout.slot.data_ptr<int32_t>(),
      fanout.pos.data_ptr<int32_t>(), fanout.first.data_ptr<int32_t>(),
      fanout.sub.data_ptr<int64_t>(), fanout.count.data_ptr<int64_t>(),
      reinterpret_cast<unsigned long long *>(counter.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return fanout;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
