/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: Apache-2.0
******************************************************************************/

#include "../check.h"
#include "kernels.cuh"
#include "table.cuh"
#include "types.cuh"

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace dyn_emb {

// Map table-relative hash slots to the value rows they own.
//
// A slot owns its row for as long as the table lives: the first key to occupy
// a slot takes the next row of its table, and every later occupant -- the key
// that evicts it, or the one that reuses it after an erase -- inherits that
// row. Negative entries are not slots and pass through (see below). Anything
// else must be a slot of the table's main region, and, unless `allocate`, one
// that already owns a row; both are asserted on the device.
//
// A new row must also lie below `row_limits[table]`, the rows the table's value
// buffer has: handing out one past it would make every later read and write of
// that key run off the buffer, so it is asserted here, where it is first known.
__global__ void table_resolve_rows_kernel(
    int64_t *__restrict__ rows, int64_t *__restrict__ row_counters,
    int64_t const *__restrict__ row_limits,
    int64_t const *__restrict__ table_bucket_offsets, int64_t bucket_capacity,
    int64_t const *__restrict__ slots, int64_t const *__restrict__ table_ids,
    int64_t *__restrict__ out, int64_t n, bool allocate) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t slot = slots[i];
    if (slot < 0) {
      // Not a slot at all: -1 marks a lookup miss, and -(position + 1) marks an
      // input key the table rejected, reported among the evicted ones. Both
      // encodings are consumed downstream (select_insert_failed_values reads
      // the position back), so they must come out as they went in.
      out[i] = slot;
      continue;
    }
    int64_t tid = table_ids[i];
    int64_t bkt_begin = table_bucket_offsets[tid];
    int64_t main_capacity =
        (table_bucket_offsets[tid + 1] - bkt_begin) * bucket_capacity;
    // Overflow slots sit past the main region and have no row; a table that
    // owns rows has no overflow.
    CUDA_KERNEL_ASSERT(slot < main_capacity);
    int64_t flat = bkt_begin * bucket_capacity + slot;
    int64_t row = rows[flat];
    if (row < 0) {
      // Without `allocate` the slot's key was placed earlier, so it owns a row;
      // finding none means a path put a key here without going through rows.
      CUDA_KERNEL_ASSERT(allocate);
      if (allocate) {
        // Keys in one batch are unique, so no other thread touches this slot.
        // ::atomicAdd: dyn_emb has its own overloads that hide the 64-bit one.
        row = static_cast<int64_t>(::atomicAdd(
            reinterpret_cast<unsigned long long *>(row_counters + tid), 1ull));
        CUDA_KERNEL_ASSERT(row < row_limits[tid]);
        rows[flat] = row;
      }
    }
    out[i] = row;
  }
}

at::Tensor table_resolve_rows(at::Tensor rows, at::Tensor row_counters,
                              at::Tensor row_limits,
                              at::Tensor table_bucket_offsets,
                              int64_t bucket_capacity, at::Tensor slots,
                              at::Tensor table_ids, bool allocate) {
  TORCH_CHECK(rows.is_cuda() && rows.scalar_type() == at::kLong);
  TORCH_CHECK(row_counters.is_cuda() && row_counters.scalar_type() == at::kLong);
  TORCH_CHECK(row_limits.is_cuda() && row_limits.scalar_type() == at::kLong &&
              row_limits.numel() == row_counters.numel());
  TORCH_CHECK(slots.is_cuda() && slots.scalar_type() == at::kLong);
  TORCH_CHECK(table_ids.is_cuda() && table_ids.scalar_type() == at::kLong);
  TORCH_CHECK(slots.numel() == table_ids.numel());
  int64_t n = slots.numel();
  at::Tensor out = torch::empty_like(slots);
  if (n == 0)
    return out;
  constexpr int BLOCK = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  table_resolve_rows_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
      rows.data_ptr<int64_t>(), row_counters.data_ptr<int64_t>(),
      row_limits.data_ptr<int64_t>(),
      table_bucket_offsets.data_ptr<int64_t>(), bucket_capacity,
      slots.data_ptr<int64_t>(), table_ids.data_ptr<int64_t>(),
      out.data_ptr<int64_t>(), n, allocate);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ---------------------------------------------------------------------------
// Rehash support: a key map moved to a bigger one without a count on the host
// ---------------------------------------------------------------------------
//
// The key column of a table is already a padded array: an empty, a reclaimed or
// a locked slot holds a reserved key (-1, -2, -3), which every table kernel
// skips. A rehash copies a chunk of the old key column as it is, inserts it
// into the new table (the reserved lanes insert nothing and report slot -1),
// and the two kernels below move what is keyed by the slot -- the value row and
// the reference counter -- and the rows of the reclaimed slots, which have no
// key to hash by. Nothing is compacted and no count is read: the one number
// that is needed, how many rows the reclaimed slots hold, is computed on the
// device from the table's counters.

constexpr int64_t kEmptyKey = static_cast<int64_t>(0xFFFFFFFFFFFFFFFFull);
constexpr int64_t kReclaimedKey = static_cast<int64_t>(0xFFFFFFFFFFFFFFFEull);
// A key with these bits all set is one of the four reserved keys (-4 .. -1), as
// Bucket::is_valid has it: it is no key, nothing was inserted for it and it
// carries nothing.
constexpr uint64_t kReserveKeyMask = 0xFFFFFFFFFFFFFFFCull;

static void check_keys_view(at::Tensor const &keys) {
  TORCH_CHECK(keys.is_cuda() && keys.scalar_type() == at::kLong);
  TORCH_CHECK(keys.dim() == 2 && keys.stride(1) == 1);
}

// Lane i is an old slot of a rehash chunk: keys[i] is the key it held, slots[i]
// the table-relative slot of the NEW table the insert gave that key (-1 where
// nothing was inserted), src_rows[i] and src_ref[i] the row and the reference
// count the old slot owned. The row and the count follow the key to
// dst[dst_base + slot]. A lane whose key is reserved carries nothing. The new
// table has room for every key it is given, so a real key that was refused is a
// bug: asserted on the device, like a row past the row limit. The row and the
// count are optional (a table without rows or without a reference counter);
// each pair is given whole or left out.
__global__ void carry_slot_state_kernel(int64_t const *__restrict__ keys,
                                        int64_t const *__restrict__ slots,
                                        int64_t const *__restrict__ src_rows,
                                        int32_t const *__restrict__ src_ref,
                                        int64_t *__restrict__ dst_rows,
                                        int32_t *__restrict__ dst_ref,
                                        int64_t dst_base, int64_t n) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    uint64_t key = static_cast<uint64_t>(keys[i]);
    if ((key & kReserveKeyMask) == kReserveKeyMask)
      continue;
    int64_t slot = slots[i];
    CUDA_KERNEL_ASSERT(slot >= 0);
    if (slot < 0)
      continue;
    if (dst_rows != nullptr)
      dst_rows[dst_base + slot] = src_rows[i];
    if (dst_ref != nullptr)
      dst_ref[dst_base + slot] = src_ref[i];
  }
}

void table_carry_slot_state(at::Tensor keys, at::Tensor slots, int64_t dst_base,
                            std::optional<at::Tensor> src_rows,
                            std::optional<at::Tensor> dst_rows,
                            std::optional<at::Tensor> src_ref,
                            std::optional<at::Tensor> dst_ref) {
  TORCH_CHECK(keys.is_cuda() && keys.scalar_type() == at::kLong &&
                  keys.dim() == 1 && keys.is_contiguous(),
              "keys must be a contiguous int64 CUDA vector");
  TORCH_CHECK(slots.is_cuda() && slots.scalar_type() == at::kLong &&
                  slots.dim() == 1 && slots.is_contiguous() &&
                  slots.numel() == keys.numel(),
              "slots must be a contiguous int64 CUDA vector as long as keys");
  TORCH_CHECK(src_rows.has_value() == dst_rows.has_value(),
              "src_rows and dst_rows are given together");
  TORCH_CHECK(src_ref.has_value() == dst_ref.has_value(),
              "src_ref and dst_ref are given together");
  int64_t n = keys.numel();
  if (src_rows.has_value()) {
    TORCH_CHECK(
        src_rows->is_cuda() && src_rows->scalar_type() == at::kLong &&
            src_rows->is_contiguous() && src_rows->numel() == n,
        "src_rows must be a contiguous int64 CUDA vector as long as keys");
    TORCH_CHECK(dst_rows->is_cuda() && dst_rows->scalar_type() == at::kLong &&
                    dst_rows->is_contiguous(),
                "dst_rows must be a contiguous int64 CUDA vector");
  }
  if (src_ref.has_value()) {
    TORCH_CHECK(
        src_ref->is_cuda() && src_ref->scalar_type() == at::kInt &&
            src_ref->is_contiguous() && src_ref->numel() == n,
        "src_ref must be a contiguous int32 CUDA vector as long as keys");
    TORCH_CHECK(dst_ref->is_cuda() && dst_ref->scalar_type() == at::kInt &&
                    dst_ref->is_contiguous(),
                "dst_ref must be a contiguous int32 CUDA vector");
  }
  if (n == 0)
    return;
  constexpr int BLOCK = 256;
  int64_t grid = std::min<int64_t>((n + BLOCK - 1) / BLOCK, 1 << 20);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  carry_slot_state_kernel<<<grid, BLOCK, 0, stream>>>(
      keys.data_ptr<int64_t>(), slots.data_ptr<int64_t>(),
      get_pointer<int64_t>(src_rows), get_pointer<int32_t>(src_ref),
      get_pointer<int64_t>(dst_rows), get_pointer<int32_t>(dst_ref), dst_base,
      n);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

// An erased key leaves its slot reclaimed, still owning its row. A rehash must
// keep those rows, but a reclaimed slot has no key to hash by, so its row goes
// to a free slot of the new table: linear probing over the flattened slots from
// a start spread evenly across the table's range, claiming an empty slot by CAS
// and marking it reclaimed, so that the reclaimed slots do not pile into the
// first buckets (a reclaimed slot is only reused once a bucket has no empty
// slot left) and the threads do not all CAS the same few slots.
//
// One thread per OLD slot of the table's bucket range [old_bkt_begin, ...). A
// reclaimed slot that owns a row takes the next ordinal j of an atomic cursor
// and starts at the head of the j-th of n equal stretches of the new range, n =
// *count: every row below a table's counter sits on a live or a reclaimed slot,
// so n is the counter minus the live keys, computed on the device by the
// caller. A count of zero or below promises no row and nothing is carried, as
// it never was. Linear probing fills the same set of slots whatever order the
// threads take their ordinals in, so the key layout does not depend on
// scheduling.
__global__ void carry_reclaimed_rows_kernel(
    int64_t const *__restrict__ old_keys, int64_t old_key_stride,
    int64_t const *__restrict__ old_rows, int64_t old_bkt_begin,
    int64_t old_num_slots, int64_t bucket_capacity,
    int64_t *__restrict__ new_keys, int64_t new_key_stride,
    uint8_t *__restrict__ new_digests, int64_t new_digest_stride,
    int64_t *__restrict__ new_rows, int64_t new_bkt_begin,
    int64_t new_num_slots, int64_t const *__restrict__ count,
    unsigned long long *__restrict__ cursor) {
  int64_t n = *count;
  if (n <= 0)
    return;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < old_num_slots; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t bucket = old_bkt_begin + i / bucket_capacity;
    int64_t iter = i % bucket_capacity;
    if (old_keys[bucket * old_key_stride + iter] != kReclaimedKey)
      continue;
    int64_t row = old_rows[bucket * bucket_capacity + iter];
    if (row < 0)
      continue;
    // ::atomicAdd: dyn_emb has its own overloads that hide the 64-bit one.
    int64_t j = static_cast<int64_t>(::atomicAdd(cursor, 1ull));
    // More reclaimed rows than the counters promised.
    CUDA_KERNEL_ASSERT(j < n);
    if (j >= n)
      continue;
    // The product stays below 2^64 while n and the range are within 2^32 slots,
    // so it is taken unsigned.
    int64_t start = static_cast<int64_t>(
        (static_cast<uint64_t>(j) * static_cast<uint64_t>(new_num_slots)) /
        static_cast<uint64_t>(n));
    bool placed = false;
    for (int64_t k = 0; k < new_num_slots && !placed; ++k) {
      int64_t s = start + k;
      if (s >= new_num_slots)
        s -= new_num_slots;
      int64_t nb = new_bkt_begin + s / bucket_capacity;
      int64_t it = s % bucket_capacity;
      int64_t *key = new_keys + nb * new_key_stride + it;
      if (*reinterpret_cast<volatile int64_t *>(key) != kEmptyKey)
        continue;
      unsigned long long old =
          atomicCAS(reinterpret_cast<unsigned long long *>(key),
                    static_cast<unsigned long long>(kEmptyKey),
                    static_cast<unsigned long long>(kReclaimedKey));
      if (static_cast<int64_t>(old) == kEmptyKey) {
        // The slot holds ReclaimKey now, so it carries that key's digest, as an
        // erase leaves it -- not the empty one the new table was filled with.
        new_digests[nb * new_digest_stride + it] =
            LinearBucket<int64_t>::reclaimed_digest();
        new_rows[nb * bucket_capacity + it] = row;
        placed = true;
      }
    }
    // A rehash only grows a table, so there is always room.
    CUDA_KERNEL_ASSERT(placed);
  }
}

// One thread, after the kernel above: every row below the counter sits on a
// live or a reclaimed slot, so the reclaimed slots found are exactly what the
// counters promised. The one check of a table's row counter against its bucket
// sizes.
__global__ void
check_carried_kernel(int64_t const *__restrict__ count,
                     unsigned long long const *__restrict__ cursor) {
  int64_t n = *count;
  if (n > 0) {
    CUDA_KERNEL_ASSERT(static_cast<int64_t>(*cursor) == n);
  }
}

void table_carry_reclaimed_rows(at::Tensor old_keys, at::Tensor old_rows,
                                int64_t old_bkt_begin, int64_t old_bkt_end,
                                int64_t bucket_capacity, at::Tensor new_keys,
                                at::Tensor new_digests, at::Tensor new_rows,
                                int64_t new_bkt_begin, int64_t new_bkt_end,
                                at::Tensor count) {
  check_keys_view(old_keys);
  check_keys_view(new_keys);
  TORCH_CHECK(old_rows.is_cuda() && old_rows.scalar_type() == at::kLong);
  TORCH_CHECK(new_rows.is_cuda() && new_rows.scalar_type() == at::kLong);
  TORCH_CHECK(
      new_digests.is_cuda() && new_digests.scalar_type() == at::kByte &&
          new_digests.dim() == 2 && new_digests.stride(1) == 1,
      "digests must be the [num_buckets, bucket_capacity] uint8 column");
  TORCH_CHECK(count.is_cuda() && count.scalar_type() == at::kLong &&
                  count.numel() == 1,
              "count is one int64 on the device");
  int64_t old_slots = (old_bkt_end - old_bkt_begin) * bucket_capacity;
  int64_t new_slots = (new_bkt_end - new_bkt_begin) * bucket_capacity;
  if (old_slots <= 0 || new_slots <= 0)
    return;
  at::Tensor cursor = torch::zeros({1}, old_rows.options());
  constexpr int BLOCK = 256;
  int64_t grid = std::min<int64_t>((old_slots + BLOCK - 1) / BLOCK, 1 << 20);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  carry_reclaimed_rows_kernel<<<grid, BLOCK, 0, stream>>>(
      old_keys.data_ptr<int64_t>(), old_keys.stride(0),
      old_rows.data_ptr<int64_t>(), old_bkt_begin, old_slots, bucket_capacity,
      new_keys.data_ptr<int64_t>(), new_keys.stride(0),
      new_digests.data_ptr<uint8_t>(), new_digests.stride(0),
      new_rows.data_ptr<int64_t>(), new_bkt_begin, new_slots,
      count.data_ptr<int64_t>(),
      reinterpret_cast<unsigned long long *>(cursor.data_ptr<int64_t>()));
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  check_carried_kernel<<<1, 1, 0, stream>>>(
      count.data_ptr<int64_t>(),
      reinterpret_cast<unsigned long long const *>(cursor.data_ptr<int64_t>()));
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

// Leaves the slots named by `slot_index` ([batch, 2] int64: slot, row -- the
// form a dump carries) reclaimed and owning their rows: what an erase leaves behind,
// recreated for a replica that never held the erased key. Per slot, by what it
// holds before:
//  - Empty: the replica never held the erased key (it was added and erased
//    within one window, so no earlier delta carried it), so its own erase had
//    nothing to remove. The slot becomes reclaimed and takes the row.
//  - Reclaimed: the replica held the key, and its erase, which ran just before,
//    left the slot reclaimed with the row that key's replay gave it -- the
//    source's. An erase only changes the key, never the row, so there is
//    nothing to write, and a row that differs means the replica has diverged:
//    asserted, not overwritten, so the divergence is not hidden.
//  - A live key: the source evicted it with another key and erased that one
//    before the next dump, so nothing in the delta overwrites it, though the
//    slot is reclaimed there. The key is erased the way the erase kernel does --
//    score words cleared, digest emptied, bucket_sizes decremented -- and the
//    slot then must own the row, like the reclaimed case.
// A slot that ends reclaimed -- the empty one and the live one -- is left as an
// erase leaves it: ReclaimKey, and that key's digest.
// Reclaimed slots are not counted in bucket_sizes. The slots of one call must be
// distinct -- the erased slots are de-duplicated when they are popped, since a
// slot erased, refilled and erased again is still one reclaimed slot with one
// row. A locked slot, or a lost race for one, means something else is writing
// to the table: asserted.
template <typename Table>
__global__ void restore_reclaimed_slots_kernel(
    Table table, int64_t const *__restrict__ table_bucket_offsets,
    int *__restrict__ bucket_sizes, int64_t *__restrict__ rows,
    int64_t const *__restrict__ slot_index, int64_t batch, int64_t table_id) {

  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  int64_t bc = table.bucket_capacity();
  int64_t bkt_begin = table_bucket_offsets[table_id];
  int64_t table_cap = (table_bucket_offsets[table_id + 1] - bkt_begin) * bc;
  int64_t ns = table.num_scores();

  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < batch; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t slot = slot_index[2 * i];
    int64_t row = slot_index[2 * i + 1];
    CUDA_KERNEL_ASSERT(slot < table_cap);

    int64_t bucket_id = bkt_begin + slot / bc;
    Bucket bucket = table[bucket_id];
    Iter iter = static_cast<Iter>(slot % bc);
    int64_t flat = bucket_id * bc + (slot % bc);

    auto key_slot =
        reinterpret_cast<typename Bucket::AtomicKey *>(bucket.keys(iter));
    KeyType old_key = key_slot->load(cuda::std::memory_order_relaxed);
    CUDA_KERNEL_ASSERT(old_key != static_cast<KeyType>(Bucket::LockedKey));

    if (old_key == Bucket::reclaimed_key()) {
      CUDA_KERNEL_ASSERT(rows[flat] == row);
      continue;
    }
    // Empty or a live key: take the slot under the bucket's lock, one attempt,
    // as the other replay write does -- the replay has the table to itself.
    KeyType expected_key = old_key;
    bool locked = bucket.try_lock(iter, expected_key);
    CUDA_KERNEL_ASSERT(locked);
    if (old_key == static_cast<KeyType>(Bucket::EmptyKey)) {
      rows[flat] = row;
    } else {
      for (int64_t k = 0; k < ns; ++k)
        *bucket.scores(iter, k) = ScoreType();
      atomicSub(bucket_sizes + bucket_id, 1);
      CUDA_KERNEL_ASSERT(rows[flat] == row);
    }
    // Either way the slot ends holding ReclaimKey, so it carries that key's
    // digest, as an erase leaves it: an empty slot that became reclaimed still
    // has the empty one.
    *bucket.digests(iter) = Bucket::reclaimed_digest();
    bucket.unlock(iter, Bucket::reclaimed_key());
  }
}

void table_restore_reclaimed_slots(at::Tensor table_storage,
                                   at::Tensor table_bucket_offsets,
                                   int64_t bucket_capacity,
                                   at::Tensor bucket_sizes, at::Tensor rows,
                                   at::Tensor slot_index, int64_t table_id,
                                   int64_t num_scores, torch::Dtype key_dtype) {
  TORCH_CHECK(rows.is_cuda() && rows.scalar_type() == at::kLong);
  TORCH_CHECK(slot_index.is_cuda() && slot_index.scalar_type() == at::kLong);
  TORCH_CHECK(slot_index.dim() == 2 && slot_index.size(1) == 2 &&
              slot_index.is_contiguous());
  int64_t n = slot_index.size(0);
  if (n == 0)
    return;

  auto key_type = scalartype_to_datatype(toScalarType(key_dtype));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr int BLOCK = 256;
  int64_t grid = std::min<int64_t>((n + BLOCK - 1) / BLOCK, 1 << 20);

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    int64_t total_size =
        sizeof(KeyType) + sizeof(DigestType) + num_scores * sizeof(ScoreType);
    int64_t bucket_bytes = bucket_capacity * total_size;
    int64_t num_buckets =
        table_storage.numel() * table_storage.element_size() / bucket_bytes;

    using Bucket = LinearBucket<KeyType>;
    using Table = LinearBucketTable<Bucket>;
    auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                       num_buckets, bucket_capacity, num_scores);

    restore_reclaimed_slots_kernel<Table><<<grid, BLOCK, 0, stream>>>(
        table, table_bucket_offsets.data_ptr<int64_t>(),
        get_pointer<int>(bucket_sizes), rows.data_ptr<int64_t>(),
        slot_index.data_ptr<int64_t>(), n, table_id);
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace dyn_emb
