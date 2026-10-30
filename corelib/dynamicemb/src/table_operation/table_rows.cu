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
// Rehash support: carrying the rows of reclaimed slots
// ---------------------------------------------------------------------------
//
// An erased key leaves its slot reclaimed, still owning its row. When a table
// is rehashed those rows must survive, but a reclaimed slot has no key to hash
// by, so it is re-placed into any free slot of the new table. Both steps below
// touch only the reclaimed slots (n of them, usually few) beyond one streaming
// pass over the table, so nothing is allocated that scales with the capacity.

constexpr int64_t kEmptyKey = static_cast<int64_t>(0xFFFFFFFFFFFFFFFFull);
constexpr int64_t kReclaimedKey = static_cast<int64_t>(0xFFFFFFFFFFFFFFFEull);

// Collects the rows of the reclaimed slots in the bucket range
// [bkt_begin, bkt_end) that still own a row into `out`, which has room for
// exactly `num_rows` of them: every row below a table's counter sits on a live
// or a reclaimed slot, so the caller knows the count without scanning.
// `keys` is the [num_buckets, bucket_capacity] key column, strided; `rows` is
// flat per slot.
__global__ void collect_reclaimed_rows_kernel(
    int64_t const *__restrict__ keys, int64_t key_stride,
    int64_t const *__restrict__ rows, int64_t bkt_begin, int64_t num_slots,
    int64_t bucket_capacity, int64_t *__restrict__ out, int64_t num_rows,
    int64_t *__restrict__ counter) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < num_slots; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t bucket = bkt_begin + i / bucket_capacity;
    int64_t iter = i % bucket_capacity;
    if (keys[bucket * key_stride + iter] != kReclaimedKey)
      continue;
    int64_t row = rows[bucket * bucket_capacity + iter];
    if (row < 0)
      continue;
    int64_t pos = static_cast<int64_t>(
        ::atomicAdd(reinterpret_cast<unsigned long long *>(counter), 1ull));
    CUDA_KERNEL_ASSERT(pos < num_rows);
    out[pos] = row;
  }
}

// Gives each of the n carried rows a free slot of the table region
// [bkt_begin, bkt_end): linear probing over the flattened slots from a start
// spread evenly across the region, claiming an empty slot by CAS and marking it
// reclaimed. Linear probing fills the same set of slots whatever the order the
// threads claim them in, so the key layout does not depend on scheduling.
__global__ void place_reclaimed_rows_kernel(
    int64_t *__restrict__ keys, int64_t key_stride, int64_t *__restrict__ rows,
    int64_t const *__restrict__ carried, int64_t n, int64_t bkt_begin,
    int64_t num_slots, int64_t bucket_capacity) {
  for (int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < n; j += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    // A hole means fewer reclaimed rows were found than the counters promised.
    CUDA_KERNEL_ASSERT(carried[j] >= 0);
    // Row j starts at the head of the j-th of n equal stretches of the region.
    // Spreading the starts keeps the reclaimed slots from piling into the first
    // buckets -- a reclaimed slot is only reused once a bucket has no empty
    // slot left, while its neighbours still do -- and keeps the threads from
    // all CASing the same few slots. The product stays below 2^64 while n and
    // the region are within 2^32 slots, so it is taken unsigned.
    int64_t start = static_cast<int64_t>(
        (static_cast<uint64_t>(j) * static_cast<uint64_t>(num_slots)) /
        static_cast<uint64_t>(n));
    bool placed = false;
    for (int64_t k = 0; k < num_slots && !placed; ++k) {
      int64_t i = start + k;
      if (i >= num_slots)
        i -= num_slots;
      int64_t bucket = bkt_begin + i / bucket_capacity;
      int64_t iter = i % bucket_capacity;
      int64_t *key = keys + bucket * key_stride + iter;
      if (*reinterpret_cast<volatile int64_t *>(key) != kEmptyKey)
        continue;
      unsigned long long old = atomicCAS(
          reinterpret_cast<unsigned long long *>(key),
          static_cast<unsigned long long>(kEmptyKey),
          static_cast<unsigned long long>(kReclaimedKey));
      if (static_cast<int64_t>(old) == kEmptyKey) {
        rows[bucket * bucket_capacity + iter] = carried[j];
        placed = true;
      }
    }
    // A rehash only grows a table, so there is always room.
    CUDA_KERNEL_ASSERT(placed);
  }
}

static void check_keys_view(at::Tensor const &keys) {
  TORCH_CHECK(keys.is_cuda() && keys.scalar_type() == at::kLong);
  TORCH_CHECK(keys.dim() == 2 && keys.stride(1) == 1);
}

at::Tensor table_collect_reclaimed_rows(at::Tensor keys, at::Tensor rows,
                                        int64_t bkt_begin, int64_t bkt_end,
                                        int64_t bucket_capacity,
                                        int64_t num_rows) {
  check_keys_view(keys);
  TORCH_CHECK(rows.is_cuda() && rows.scalar_type() == at::kLong);
  TORCH_CHECK(num_rows > 0, "nothing to collect: the caller knows the count");
  int64_t num_slots = (bkt_end - bkt_begin) * bucket_capacity;
  // Slots not filled (the caller's count was too high) stay -1.
  at::Tensor out = torch::full({num_rows}, -1, rows.options());
  if (num_slots > 0) {
    at::Tensor counter = torch::zeros({1}, rows.options());
    constexpr int BLOCK = 256;
    int64_t grid = std::min<int64_t>((num_slots + BLOCK - 1) / BLOCK, 1 << 20);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    collect_reclaimed_rows_kernel<<<grid, BLOCK, 0, stream>>>(
        keys.data_ptr<int64_t>(), keys.stride(0), rows.data_ptr<int64_t>(),
        bkt_begin, num_slots, bucket_capacity, out.data_ptr<int64_t>(),
        num_rows, counter.data_ptr<int64_t>());
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

void table_place_reclaimed_rows(at::Tensor keys, at::Tensor rows,
                                at::Tensor carried, int64_t bkt_begin,
                                int64_t bkt_end, int64_t bucket_capacity) {
  check_keys_view(keys);
  TORCH_CHECK(rows.is_cuda() && rows.scalar_type() == at::kLong);
  TORCH_CHECK(carried.is_cuda() && carried.scalar_type() == at::kLong);
  int64_t n = carried.numel();
  int64_t num_slots = (bkt_end - bkt_begin) * bucket_capacity;
  if (n == 0)
    return;
  TORCH_CHECK(n <= num_slots, "more reclaimed rows than slots to hold them");
  constexpr int BLOCK = 256;
  int64_t grid = std::min<int64_t>((n + BLOCK - 1) / BLOCK, 1 << 20);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  place_reclaimed_rows_kernel<<<grid, BLOCK, 0, stream>>>(
      keys.data_ptr<int64_t>(), keys.stride(0), rows.data_ptr<int64_t>(),
      carried.data_ptr<int64_t>(), n, bkt_begin, num_slots, bucket_capacity);
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
      *bucket.digests(iter) = Bucket::empty_digest();
      atomicSub(bucket_sizes + bucket_id, 1);
      CUDA_KERNEL_ASSERT(rows[flat] == row);
    }
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
