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
#include "hier_output_dist/plan/part_one.h"

#include <c10/util/Exception.h>

#include <cstdint>
#include <tuple>
#include <utility>

#include "hier_output_dist/plan/sort.h"
#include "hier_output_dist/plan/tensors.h"
#include "utils.h"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

// The owner rules read a key as an unsigned number of 64 bits, held in an
// int64 tensor: the same bits, in ATen's int64 arithmetic, whose multiplies
// wrap (as dynamicemb/key_ownership.py relies on).
constexpr int64_t kSignBit = INT64_MIN;
constexpr int64_t kLow32 = 0xFFFFFFFF;

// value >> bits, unsigned.
at::Tensor logical_shift_right(const at::Tensor &value, int bits) {
  return value.bitwise_right_shift(bits).bitwise_and((int64_t)(~0ull >> bits));
}

// left < right, unsigned.
at::Tensor unsigned_less(const at::Tensor &left, const at::Tensor &right) {
  return left.bitwise_xor(kSignBit).lt(right.bitwise_xor(kSignBit));
}
at::Tensor unsigned_less(const at::Tensor &left, int64_t right) {
  return left.bitwise_xor(kSignBit).lt(right ^ kSignBit);
}

// MurmurHash3's finalizer (murmur3_fmix64, src/murmur_hash.cuh).
at::Tensor fmix64(const at::Tensor &key) {
  at::Tensor hash = key.bitwise_xor(logical_shift_right(key, 33));
  hash.mul_((int64_t)0xff51afd7ed558ccdull);
  hash.bitwise_xor_(logical_shift_right(hash, 33));
  hash.mul_((int64_t)0xc4ceb9fe1a85ec53ull);
  return hash.bitwise_xor_(logical_shift_right(hash, 33));
}

// value mod divisor, unsigned: the low bits for a power of two; else a
// negative value is value + 2^64, whose remainder adds 2^64 mod divisor.
at::Tensor unsigned_mod(const at::Tensor &value, int64_t divisor) {
  if ((divisor & (divisor - 1)) == 0) return value.bitwise_and(divisor - 1);
  const uint64_t wrap = (UINT64_MAX % (uint64_t)divisor + 1) % (uint64_t)divisor;
  at::Tensor remainder = value.remainder(divisor);
  return at::where(value.lt(0), (remainder + (int64_t)wrap).remainder(divisor), remainder);
}

// (value / divisor, value % divisor), unsigned, for a divisor in [1, 2^63):
// a number, or a tensor of one per value. A value of 2^63 or more (negative
// as int64) is halved first; the doubled quotient is then at most one short
// (Java's divideUnsigned).
template <typename Divisor>
std::pair<at::Tensor, at::Tensor> unsigned_divmod(const at::Tensor &value, const Divisor &divisor) {
  const at::Tensor non_negative = value.ge(0);
  at::Tensor quotient =
      at::where(non_negative, value, logical_shift_right(value, 1)).div_(divisor, "trunc");
  quotient = at::where(non_negative, quotient, quotient.bitwise_left_shift(1));
  at::Tensor remainder = value - quotient * divisor;
  const at::Tensor carry =
      unsigned_less(remainder, divisor).logical_or_(non_negative).logical_not_();
  quotient += carry;
  remainder -= carry * divisor;
  return {quotient, remainder};
}

// The continuous rule (kBlock) on int64 keys: below the end of the blocks
// (block * world_size, wrapping) the owner is value / block and the owner
// stores value % block; past it value % world_size and value / world_size.
// A block of 0 has no end; one of 1 places those keys alike.
std::pair<at::Tensor, at::Tensor> place_in_blocks(const at::Tensor &value, at::Tensor block,
                                                  int64_t world_size) {
  block.clamp_min_(1);
  const at::Tensor in_blocks = unsigned_less(value, block * world_size);
  auto [world_quotient, world_remainder] = unsigned_divmod(value, world_size);
  auto [block_quotient, block_remainder] = unsigned_divmod(value, block);
  return {at::where(in_blocks, block_quotient, world_remainder),
          at::where(in_blocks, block_remainder, world_quotient)};
}

// The continuous rule on int32 keys, in 32-bit arithmetic: the low 32 bits
// of the block size, an end that wraps at 2^32, and the stored key the int32
// the bucketize writes, widened with its sign. unsigned_value holds the
// keys' 32 bits, so int64 division is exact.
std::pair<at::Tensor, at::Tensor> place_in_blocks_32(const at::Tensor &unsigned_value,
                                                     at::Tensor block, int64_t world_size) {
  block.bitwise_and_(kLow32);
  const at::Tensor in_blocks = unsigned_value.lt((block * world_size).bitwise_and_(kLow32));
  block.clamp_min_(1);  // the divisor of keys in the blocks; a block of 0 has none
  at::Tensor owner = at::where(in_blocks, unsigned_value.div(block, "trunc"),
                               unsigned_value.remainder(world_size));
  at::Tensor stored = at::where(in_blocks, unsigned_value.remainder(block),
                                unsigned_value.div(world_size, "trunc"));
  stored -= stored.bitwise_right_shift(31).bitwise_left_shift(32);
  return {owner, stored};
}

// The owner of each key (int64) and the key its owner stores (int64), by the
// rule of the key's feature (rule_feature) as DynamicEmb's bucketize places
// it (src/sparse_block_bucketize_features.cu): in the arithmetic of the
// keys' own int32 or int64, as unsigned numbers. TorchRec converts the block
// sizes to the keys' dtype, so int32 keys take their low 32 bits. Only the
// rules of some feature run; with several, each key takes its feature's.
std::pair<at::Tensor, at::Tensor> place_keys(const at::Tensor &values,
                                             const at::Tensor &rule_feature,
                                             const OwnerRules &rules, int64_t world_size) {
  const int cyclic_code = (int)DistType::kCyclic, hashed_code = (int)DistType::kHashedCyclic;
  bool cyclic = false, hashed = false, block = false;
  for (int code : rules.host_dist_types) {
    cyclic |= code == cyclic_code;
    hashed |= code == hashed_code;
    block |= code != cyclic_code && code != hashed_code;
  }
  const bool int32_keys = values.scalar_type() == at::kInt;
  const at::Tensor value = values.to(at::kLong);  // int32 keys widened with their sign
  const at::Tensor unsigned_value = int32_keys ? value.bitwise_and(kLow32) : value;
  at::Tensor owner, stored = value;
  if (block) {
    at::Tensor block_size = rules.block_sizes.index_select(0, rule_feature);
    std::tie(owner, stored) =
        int32_keys ? place_in_blocks_32(unsigned_value, std::move(block_size), world_size)
                   : place_in_blocks(value, std::move(block_size), world_size);
  }
  // A rule after the first: the keys of its features take it. The cyclic
  // rules store the key itself; only the continuous rule stores another.
  const at::Tensor rule =
      cyclic + hashed + block > 1 ? rules.dist_types.index_select(0, rule_feature) : at::Tensor();
  auto use_rule = [&](int code, const at::Tensor &rule_owner) {
    if (!owner.defined()) {
      owner = rule_owner;
      return;
    }
    const at::Tensor mine = rule.eq(code);
    owner = at::where(mine, rule_owner, owner);
    if (block) stored = at::where(mine, value, stored);
  };
  if (hashed)
    use_rule(hashed_code, int32_keys
                              ? fmix64(unsigned_value).bitwise_and_(kLow32).remainder(world_size)
                              : unsigned_mod(fmix64(value), world_size));
  if (cyclic)
    use_rule(cyclic_code,
             int32_keys ? unsigned_value.remainder(world_size) : unsigned_mod(value, world_size));
  return {owner, stored};
}

// contiguous, on the GPU of the keys
bool on_gpu_of(const at::Tensor &tensor, const at::Tensor &values) {
  return tensor.is_cuda() && tensor.get_device() == values.get_device() && tensor.is_contiguous();
}
bool is_index_dtype(const at::Tensor &tensor) {
  return tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong;
}

}  // namespace

Partition partition_keys(const PlanInput &input, const OwnerRules &rules, int64_t world_size) {
  const at::Tensor &values = input.values;
  const int64_t rows = values.numel(), batch = input.batch_size,
                num_features = (int64_t)rules.host_dist_types.size();
  const int64_t cells = world_size * num_features;
  TORCH_CHECK(num_features >= 1 && world_size >= 1 && rules.dist_types.numel() == num_features &&
                  rules.block_sizes.numel() == num_features,
              "hier_output_dist: a rule and a block size per feature, and some owners");
  TORCH_CHECK(
      values.is_cuda() && values.is_contiguous() && is_index_dtype(values) && rows < INT32_MAX,
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
  TORCH_CHECK(!input.input_offsets.has_value() ||
                  (input.input_batch_size > 0 && on_gpu_of(*input.input_offsets, values) &&
                   is_index_dtype(*input.input_offsets) &&
                   input.input_offsets->numel() == num_features * input.input_batch_size + 1),
              "hier_output_dist: the input KJT's offsets must be contiguous int32 or int64 on "
              "the GPU of the keys, num_features * input_batch_size + 1 of them");
  TORCH_CHECK(!input.weights.has_value() ||
                  (on_gpu_of(*input.weights, values) &&
                   input.weights->scalar_type() == at::kFloat && input.weights->numel() == rows),
              "hier_output_dist: weights must be contiguous fp32, one per key");
  const auto long_options = values.options().dtype(at::kLong);
  const auto int_options = values.options().dtype(at::kInt);

  Partition part;
  // where each feature's keys start in the KJT [num_features + 1]
  const at::Tensor value_start =
      input.offsets.has_value()
          ? input.offsets->slice(0, 0, num_features * batch + 1, batch).to(at::kLong) -
                input.offsets->narrow(0, 0, 1).to(at::kLong)
          : at::cat(
                {at::zeros({1}, long_options),
                 input.lengths->view({num_features, batch}).sum({1}, false, at::kLong).cumsum(0)});
  // the per-feature totals construct_jagged_tensors reads
  part.input_length_per_key =
      input.input_offsets.has_value()
          ? input.input_offsets
                ->slice(0, 0, num_features * input.input_batch_size + 1, input.input_batch_size)
                .to(at::kLong)
                .diff()
          : value_start.diff();
  if (rows == 0) {
    part.keys = at::empty({0}, long_options);
    if (input.weights.has_value()) part.key_weights = input.weights->clone();
    part.inv = at::empty({0}, int_options);
    part.row_cell = at::empty({0}, int_options);
    part.cell_start = at::zeros({cells + 1}, long_options);
    part.block_start = at::zeros({world_size + 1}, long_options);
    return part;
  }

  // the cell (owner, feature) of each key, and the key its owner stores
  at::Tensor cell, stored;
  {
    // the feature of each key: num_features past the valid keys (index dedup's bound)
    const at::Tensor feature =
        at::searchsorted(value_start, at::arange(rows, long_options), /*out_int32=*/true,
                         /*right=*/true) -
        1;
    at::Tensor owner;
    std::tie(owner, stored) =
        place_keys(values, feature.clamp_max(num_features - 1), rules, world_size);
    cell = at::where(feature.lt(num_features), owner.mul_(num_features).add_(feature), cells)
               .to(at::kInt);
  }
  // a stable sort by cell: a cell's keys keep the KJT's order, and the keys
  // past the valid ones stay last, in order
  std::tie(part.row_cell, part.inv) =
      sort_pairs(cell, at::arange(rows, int_options), bits_for(cells + 1));
  cell.reset();
  part.keys = stored.index_select(0, part.inv);
  stored.reset();
  if (input.weights.has_value()) part.key_weights = input.weights->index_select(0, part.inv);
  part.cell_start = at::searchsorted(part.row_cell, at::arange(cells + 1, int_options));
  part.block_start = part.cell_start.slice(0, 0, cells + 1, num_features).contiguous();
  return part;
}

FanoutTables fanout_tables(const at::Tensor &reverse_indices, const at::Tensor &inv) {
  const int64_t rows = inv.numel(), originals = reverse_indices.numel();
  // Original and deduplicated rows are int32 (exp_idx, exp_off).
  TORCH_CHECK(reverse_indices.is_cuda() && reverse_indices.is_contiguous() &&
                  reverse_indices.scalar_type() == at::kLong && inv.scalar_type() == at::kInt &&
                  originals < INT32_MAX,
              "hier_output_dist: reverse indices must be contiguous int64 on the GPU, fewer "
              "than 2^31, and inv int32");
  const int64_t max_slots = originals / (kHeavy + 1) + 1, max_subs = 2 * originals / kHeavy + 2;
  const auto long_options = inv.options().dtype(at::kLong);
  const auto int_options = inv.options().dtype(at::kInt);
  FanoutTables fanout;
  if (rows == 0 || originals == 0) {  // no row has an original row
    fanout.exp_off = at::zeros({rows + 1}, int_options);
    fanout.exp_idx = at::empty({originals}, int_options);
    fanout.slot = at::full({rows}, -1, int_options);
    fanout.sub = at::empty({2 * max_subs}, long_options);
    fanout.count = at::zeros({2}, long_options);
    fanout.pos = at::empty({max_slots}, int_options);
    fanout.first = at::empty({max_slots}, int_options);
    return fanout;
  }

  // each row's original rows: a stable sort of the reverse indices, which
  // keeps them in increasing order
  at::Tensor sorted_row;
  std::tie(sorted_row, fanout.exp_idx) =
      sort_pairs(reverse_indices.to(at::kInt), at::arange(originals, int_options), bits_for(rows));
  fanout.exp_off =
      at::searchsorted(sorted_row, at::arange(rows + 1, int_options), /*out_int32=*/true);
  sorted_row.reset();
  // heavy rows: slots and sub-tasks numbered by scans, in row order (the
  // per-row temporaries go as soon as they are dead)
  at::Tensor heavy, subs, heavy_row;
  {
    const at::Tensor fan = fanout.exp_off.diff();
    heavy = fan.gt(kHeavy);
    subs = ceil_div(fan, kHeavy).mul_(heavy);
    const at::Tensor heavy_rank = heavy.cumsum(0, at::kInt);  // inclusive
    fanout.slot = at::where(heavy, heavy_rank - 1, -1);
    // the row of each slot, clamped past the slots
    heavy_row =
        at::searchsorted(heavy_rank, at::arange(1, max_slots + 1, int_options)).clamp_max(rows - 1);
  }
  fanout.pos = inverse_permutation(inv).index_select(0, heavy_row);  // where part 1 put it
  const at::Tensor sub_end = subs.cumsum(0, at::kLong);
  const at::Tensor first_sub = sub_end - subs;
  fanout.count = at::stack({heavy.sum(), subs.sum()});
  heavy.reset();
  subs.reset();
  fanout.first = first_sub.index_select(0, heavy_row).to(at::kInt);
  // the row of each sub-task, clamped past the sub-tasks, and its first entry
  const at::Tensor sub_task = at::arange(max_subs, long_options);
  const at::Tensor sub_row =
      at::searchsorted(sub_end, sub_task, /*out_int32=*/false, /*right=*/true).clamp_max(rows - 1);
  const at::Tensor sub_entry = fanout.exp_off.index_select(0, sub_row).to(at::kLong) +
                               (sub_task - first_sub.index_select(0, sub_row)) * kHeavy;
  fanout.sub = at::stack({sub_row, sub_entry}, 1).reshape({-1});
  return fanout;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
