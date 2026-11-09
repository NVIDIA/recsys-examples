// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The ORACLE of test_rehash_kernels.cpp: the two kernels and wrappers that
// table_carry_reclaimed_rows (src/table_operation/table_rows.cu) replaced --
// table_collect_reclaimed_rows and table_place_reclaimed_rows -- as table_rows.cu had
// them in the tree before the padded rehash (355a1f61), with the launches rewritten for
// the host-only stand-ins (emulate_launch). Only the test includes it; nothing here is
// compiled into the library.
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
    int64_t *__restrict__ keys, int64_t key_stride,
    uint8_t *__restrict__ digests, int64_t digest_stride,
    int64_t *__restrict__ rows, int64_t const *__restrict__ carried, int64_t n,
    int64_t bkt_begin, int64_t num_slots, int64_t bucket_capacity) {
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
        // The slot holds ReclaimKey now, so it carries that key's digest, as an
        // erase leaves it -- not the empty one the new table was filled with.
        digests[bucket * digest_stride + iter] =
            LinearBucket<int64_t>::reclaimed_digest();
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
    emulate_launch(collect_reclaimed_rows_kernel, grid, BLOCK, 0, stream, 
        keys.data_ptr<int64_t>(), keys.stride(0), rows.data_ptr<int64_t>(),
        bkt_begin, num_slots, bucket_capacity, out.data_ptr<int64_t>(),
        num_rows, counter.data_ptr<int64_t>());
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

void table_place_reclaimed_rows(at::Tensor keys, at::Tensor digests,
                                at::Tensor rows, at::Tensor carried,
                                int64_t bkt_begin, int64_t bkt_end,
                                int64_t bucket_capacity) {
  check_keys_view(keys);
  TORCH_CHECK(digests.is_cuda() && digests.scalar_type() == at::kByte &&
                  digests.dim() == 2 && digests.stride(1) == 1,
              "digests must be the [num_buckets, bucket_capacity] uint8 column");
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
  emulate_launch(place_reclaimed_rows_kernel, grid, BLOCK, 0, stream, 
      keys.data_ptr<int64_t>(), keys.stride(0), digests.data_ptr<uint8_t>(),
      digests.stride(0), rows.data_ptr<int64_t>(), carried.data_ptr<int64_t>(),
      n, bkt_begin, num_slots, bucket_capacity);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}


