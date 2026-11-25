// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Float8_e4m3fn.h>
#include <torch/library.h>

#include "table_operation/types.cuh"

namespace dyn_emb {
namespace {

// The dispatcher represents a plain layout as a tuple of integers. It is
// unpacked once on the host; both kernels take the same POD by value.
using LayoutTuple = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;
using Bucket = LinearBucket<int64_t>;
struct RowLayout {
  int embedding, opt_columns, opt_bytes, bucket_capacity, scores;
  int runtime_bytes, packed_bytes;
  int64_t table_rows, slot_count;
};

RowLayout get_layout(const at::Tensor &table, const at::Tensor &buckets,
                     const LayoutTuple &layout) {
  const auto [embedding, opt, opt_bytes, capacity, scores] = layout;
  TORCH_CHECK(table.dim() == 2 && table.is_contiguous() && table.is_cuda());
  TORCH_CHECK(buckets.device() == table.device() && buckets.is_contiguous() &&
              buckets.scalar_type() == at::kByte);
  TORCH_CHECK(embedding > 0 && embedding <= table.size(1) && opt >= 0 &&
              (opt_bytes == table.element_size() || opt_bytes == 4));
  TORCH_CHECK(capacity > 0 && (scores == 1 || scores == 2));
  const int64_t slot_bytes = Bucket::ScoreOffset + scores * sizeof(ScoreType);
  TORCH_CHECK(buckets.numel() % (slot_bytes * capacity) == 0);
  TORCH_CHECK(opt <= table.size(1) && embedding <= INT_MAX && opt <= INT_MAX);
  int64_t payload = embedding * table.element_size() + opt * opt_bytes;
  int64_t row = table.size(1) * table.element_size();
  TORCH_CHECK(payload <= row && row < INT_MAX - 16 && capacity <= INT_MAX);
  // Align successive rows for vector copies and the trailing LFU array.
  return {int(embedding),
          int(opt),
          int(opt_bytes),
          int(capacity),
          int(scores),
          int(row),
          int((payload + 15) / 16 * 16),
          table.size(0),
          buckets.numel() / slot_bytes};
}

void check_indices(const at::Tensor &t, const at::Tensor &like) {
  TORCH_CHECK(t.device() == like.device() && t.scalar_type() == at::kLong &&
              t.is_contiguous() && t.dim() == 1);
}

__device__ uint64_t *score_block(uint8_t *buckets, RowLayout l, int64_t slot) {
  int64_t bucket_bytes = Bucket::memory_usage(l.bucket_capacity, l.scores);
  Bucket bucket(buckets + (slot / l.bucket_capacity) * bucket_bytes,
                l.bucket_capacity, l.scores);
  return bucket.scores(slot % l.bucket_capacity);
}

// Validate a retained dense route against the actual hash slots. The caller
// separately checks the live count, ruling out additional keys outside the map.
// Keys are compared bitwise for both int64 and uint64 hash tables.
__global__ void validate_kernel(uint8_t *buckets, const int64_t *table_rows,
                                const int64_t *keys, const int64_t *rows,
                                const int64_t *slots, int64_t n, int capacity,
                                int scores, int64_t slot_count,
                                uint64_t threshold, int32_t *valid) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int64_t slot = slots[i];
  if (slot < 0 || slot >= slot_count || table_rows[slot] != rows[i]) {
    atomicExch(valid, 0);
    return;
  }
  int64_t bucket_bytes = Bucket::memory_usage(capacity, scores);
  Bucket bucket(buckets + slot / capacity * bucket_bytes, capacity, scores);
  int64_t lane = slot % capacity;
  if (*bucket.keys(lane) != keys[i] || *bucket.scores(lane) < threshold)
    atomicExch(valid, 0);
}

void validate(const at::Tensor &buckets, const at::Tensor &table_rows,
              const at::Tensor &keys, const at::Tensor &rows,
              const at::Tensor &slots, int64_t capacity, int64_t scores,
              int64_t threshold, const at::Tensor &valid) {
  c10::cuda::CUDAGuard guard(buckets.device());
  check_indices(keys, buckets);
  check_indices(rows, buckets);
  check_indices(slots, buckets);
  check_indices(table_rows, buckets);
  TORCH_CHECK(keys.numel() == rows.numel() && keys.numel() == slots.numel() &&
              buckets.scalar_type() == at::kByte && buckets.is_contiguous() &&
              valid.numel() == 1 && valid.scalar_type() == at::kInt &&
              valid.device() == buckets.device() && capacity > 0 &&
              capacity <= INT_MAX && (scores == 1 || scores == 2) &&
              table_rows.numel() % capacity == 0 &&
              buckets.numel() ==
                  table_rows.numel() *
                      (Bucket::ScoreOffset + scores * sizeof(ScoreType)));
  if (keys.numel()) {
    validate_kernel<<<(keys.numel() + 255) / 256, 256, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
        buckets.data_ptr<uint8_t>(), table_rows.data_ptr<int64_t>(),
        keys.data_ptr<int64_t>(), rows.data_ptr<int64_t>(),
        slots.data_ptr<int64_t>(), keys.numel(), capacity, scores,
        table_rows.numel(), threshold, valid.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

// One warp per row. Adjacent lanes move adjacent components even when table
// lookups returned unrelated row IDs.
__global__ void pack_kernel(const uint8_t *table, uint8_t *buckets,
                            const int64_t *rows, const int64_t *slots,
                            uint8_t *packet, int64_t count, int64_t padded,
                            RowLayout l, int payload) {
  int lane = threadIdx.x % 32;
  int64_t i = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  if (i >= padded)
    return;
  uint8_t *dst = packet + i * l.packed_bytes;
  if (i < count) {
    CUDA_KERNEL_ASSERT(rows[i] >= 0 && rows[i] < l.table_rows);
    CUDA_KERNEL_ASSERT(slots[i] >= 0 && slots[i] < l.slot_count);
    const uint8_t *src = table + rows[i] * l.runtime_bytes;
    if ((l.runtime_bytes | payload) % 16 == 0) {
      for (int j = lane; j < payload / 16; j += 32)
        reinterpret_cast<uint4 *>(dst)[j] =
            reinterpret_cast<const uint4 *>(src)[j];
    } else {
      for (int j = lane; j < l.packed_bytes; j += 32)
        dst[j] = j < payload ? src[j] : 0;
    }
  } else {
    for (int j = lane; j < l.packed_bytes / 16; j += 32)
      reinterpret_cast<uint4 *>(dst)[j] = make_uint4(0, 0, 0, 0);
  }
  if (l.scores == 2 && lane == 0) {
    auto *frequencies =
        reinterpret_cast<uint64_t *>(packet + padded * l.packed_bytes);
    frequencies[i] = i < count ? score_block(buckets, l, slots[i])[1] : 0;
  }
}

at::Tensor pack(const at::Tensor &table, const at::Tensor &buckets,
                const at::Tensor &rows, const at::Tensor &slots, int64_t padded,
                LayoutTuple layout) {
  c10::cuda::CUDAGuard guard(table.device());
  RowLayout l = get_layout(table, buckets, layout);
  check_indices(rows, table);
  check_indices(slots, table);
  TORCH_CHECK(rows.numel() == slots.numel() && rows.numel() <= padded);
  TORCH_CHECK(padded <= INT64_MAX / (l.packed_bytes + (l.scores - 1) * 8));
  auto out = at::empty({padded * (l.packed_bytes + (l.scores - 1) * 8)},
                       buckets.options());
  if (padded) {
    int payload =
        l.embedding * table.element_size() + l.opt_columns * l.opt_bytes;
    pack_kernel<<<(padded + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        static_cast<const uint8_t *>(table.const_data_ptr()),
        buckets.data_ptr<uint8_t>(), rows.data_ptr<int64_t>(),
        slots.data_ptr<int64_t>(), out.data_ptr<uint8_t>(), rows.numel(),
        padded, l, payload);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

// Raw words retain the exact fp8 encoding and the packed fp32 Adagrad state.
// Averaging the decoded fp8 codes in the table's scaled units cancels the
// common power-of-two scale, including when the initial row contributes.
template <class Scalar>
__global__ void merge_kernel(uint8_t *table, uint8_t *buckets,
                             const int64_t *rows, const int64_t *slots,
                             const uint8_t *packets, const int32_t *offsets,
                             const uint8_t *fresh, int64_t count,
                             int64_t padded, int replicas, RowLayout l,
                             int64_t timestamp, bool constant) {
  int lane = threadIdx.x % 32;
  int64_t i = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  if (i >= count || rows[i] < 0 || slots[i] < 0)
    return;
  CUDA_KERNEL_ASSERT(rows[i] < l.table_rows && slots[i] < l.slot_count);
  auto *dst = table + rows[i] * l.runtime_bytes;
  int64_t rank_bytes = padded * (l.packed_bytes + (l.scores - 1) * 8);
  int payload = l.embedding * sizeof(Scalar) + l.opt_columns * l.opt_bytes;
  for (int j = lane; j < l.embedding + l.opt_columns; j += 32) {
    bool wide = j >= l.embedding && l.opt_bytes != sizeof(Scalar);
    int col = j < l.embedding ? j * sizeof(Scalar)
                              : l.embedding * sizeof(Scalar) +
                                    (j - l.embedding) * l.opt_bytes;
    float sum = 0;
    int holders = 0;
    for (int rank = 0; rank < replicas; ++rank) {
      int64_t pos = offsets ? offsets[int64_t(rank) * count + i] : i;
      if (pos < 0)
        continue;
      CUDA_KERNEL_ASSERT(pos < padded);
      const uint8_t *p =
          packets + rank * rank_bytes + pos * l.packed_bytes + col;
      float x;
      if (wide)
        memcpy(&x, p, sizeof(x));
      else
        x = float(*reinterpret_cast<const Scalar *>(p));
      sum += x;
      ++holders;
    }
    if (!holders)
      continue;
    if (constant && holders != replicas) {
      float seed;
      if (wide)
        memcpy(&seed, fresh + col, sizeof(seed));
      else
        seed = float(*reinterpret_cast<const Scalar *>(fresh + col));
      sum += float(replicas - holders) * seed;
    }
    float mean = sum / float(constant ? replicas : holders);
    if (wide)
      memcpy(dst + col, &mean, sizeof(mean));
    else
      *reinterpret_cast<Scalar *>(dst + col) = Scalar(mean);
  }
  for (int j = payload + lane; j < l.runtime_bytes; j += 32)
    dst[j] = fresh[j];
  if (lane == 0) {
    uint64_t *scores = score_block(buckets, l, slots[i]);
    uint64_t now = timestamp;
    if (!now)
      asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(now));
    scores[0] = now;
    if (l.scores == 2) {
      // Sum quotients and remainders independently to avoid overflow even
      // when frequency counters approach UINT64_MAX.
      uint64_t quotient = 0, remainder = 0;
      for (int rank = 0; rank < replicas; ++rank) {
        int64_t pos = offsets ? offsets[int64_t(rank) * count + i] : i;
        if (pos < 0)
          continue;
        CUDA_KERNEL_ASSERT(pos < padded);
        const uint8_t *p =
            packets + rank * rank_bytes + padded * l.packed_bytes;
        uint64_t f = reinterpret_cast<const uint64_t *>(p)[pos];
        quotient += f / replicas;
        remainder += f % replicas;
      }
      uint64_t mean = quotient + remainder / replicas;
      scores[1] = mean ? mean : 1;
    }
  }
}

void merge(const at::Tensor &table, const at::Tensor &buckets,
           const at::Tensor &rows, const at::Tensor &slots,
           const at::Tensor &packets, const std::optional<at::Tensor> &offsets,
           const at::Tensor &fresh, bool constant, int64_t timestamp,
           LayoutTuple layout) {
  c10::cuda::CUDAGuard guard(table.device());
  RowLayout l = get_layout(table, buckets, layout);
  check_indices(rows, table);
  check_indices(slots, table);
  TORCH_CHECK(rows.numel() == slots.numel() && packets.is_contiguous() &&
              packets.scalar_type() == at::kByte &&
              packets.device() == table.device() && packets.dim() == 2 &&
              packets.size(0) > 0 && packets.size(0) <= INT_MAX);
  TORCH_CHECK(fresh.device() == table.device() && fresh.is_contiguous() &&
              fresh.scalar_type() == table.scalar_type() &&
              fresh.numel() == table.size(1));
  int width = l.packed_bytes + (l.scores - 1) * 8;
  TORCH_CHECK(packets.size(1) % width == 0);
  int64_t padded = packets.size(1) / width;
  const int32_t *idx = nullptr;
  if (offsets) {
    TORCH_CHECK(
        offsets->device() == table.device() && offsets->is_contiguous() &&
        offsets->dim() == 2 && offsets->size(0) == packets.size(0) &&
        offsets->size(1) == rows.numel() && offsets->scalar_type() == at::kInt);
    idx = offsets->data_ptr<int32_t>();
  } else {
    TORCH_CHECK(padded == rows.numel());
  }
  if (!rows.numel())
    return;
  auto launch = [&](auto sample) {
    using Scalar = decltype(sample);
    merge_kernel<Scalar>
        <<<(rows.numel() + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<uint8_t *>(table.data_ptr()),
            buckets.data_ptr<uint8_t>(), rows.data_ptr<int64_t>(),
            slots.data_ptr<int64_t>(), packets.data_ptr<uint8_t>(), idx,
            static_cast<const uint8_t *>(fresh.const_data_ptr()), rows.numel(),
            padded, packets.size(0), l, timestamp, constant);
  };
  switch (table.scalar_type()) {
  case at::kFloat:
    launch(float{});
    break;
  case at::kHalf:
    launch(at::Half{});
    break;
  case at::kBFloat16:
    launch(at::BFloat16{});
    break;
  case at::kFloat8_e4m3fn:
    launch(c10::Float8_e4m3fn{});
    break;
  default:
    TORCH_CHECK(false, "unsupported replica row dtype ", table.scalar_type());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY_FRAGMENT(dyn_emb, m) {
  m.def("replica_validate_(Tensor buckets, Tensor table_rows, Tensor keys, "
        "Tensor rows, Tensor slots, int bucket_capacity, int scores, int "
        "threshold, Tensor(a!) valid) -> ()");
  m.def("replica_pack(Tensor table, Tensor buckets, Tensor rows, Tensor slots, "
        "int padded, (int, int, int, int, int) layout) -> Tensor");
  m.def("replica_merge_(Tensor(a!) table, Tensor(b!) buckets, Tensor rows, "
        "Tensor slots, Tensor packets, Tensor? offsets, Tensor fresh, bool "
        "constant, int timestamp, (int, int, int, int, int) layout) -> ()");
}
TORCH_LIBRARY_IMPL(dyn_emb, CUDA, m) {
  m.impl("replica_validate_", TORCH_FN(validate));
  m.impl("replica_pack", TORCH_FN(pack));
  m.impl("replica_merge_", TORCH_FN(merge));
}

} // namespace
} // namespace dyn_emb
