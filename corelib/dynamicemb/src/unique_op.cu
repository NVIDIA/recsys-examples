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

#include "check.h"
#include "lookup_forward.h"
#include "torch_utils.h"
#include "unique_op.h"
#include "utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
namespace cg = cooperative_groups;
#ifdef DEMB_USE_PYBIND11
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#endif

#include <cassert>
#include <cstdlib>
#include <limits>

#ifdef DEMB_USE_PYBIND11
namespace py = pybind11;
#endif

namespace dyn_emb {

constexpr int BLOCK_SIZE = 256;
// Block size for segmented_unique_core, opened to the H100 maximum so each
// block hides the scattered-probe latency with a full complement of warps.
constexpr int SHARED_BLOCK_SIZE = 1024;

// MurmurHash3_32 hash function
template <typename Key, uint32_t m_seed = 0> struct MurmurHash3_32 {
  __forceinline__ __host__ __device__ static uint32_t rotl32(uint32_t x,
                                                             int8_t r) {
    return (x << r) | (x >> (32 - r));
  }

  __forceinline__ __host__ __device__ static uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  __forceinline__ __host__ __device__ static uint32_t hash(const Key &key) {
    constexpr int len = sizeof(Key);
    const uint8_t *const data = reinterpret_cast<const uint8_t *>(&key);
    constexpr int nblocks = len / 4;
    uint32_t h1 = m_seed;
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;

    const uint32_t *const blocks =
        reinterpret_cast<const uint32_t *>(data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
      uint32_t k1 = blocks[i];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }

    const uint8_t *tail = data + nblocks * 4;
    uint32_t k1 = 0;
    switch (len & 3) {
    case 3:
      k1 ^= tail[2] << 16;
      [[fallthrough]];
    case 2:
      k1 ^= tail[1] << 8;
      [[fallthrough]];
    case 1:
      k1 ^= tail[0];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
    }

    h1 ^= len;
    return fmix32(h1);
  }

  // Combine two hash values (for compound keys)
  __forceinline__ __host__ __device__ static uint32_t
  hash_combine(uint32_t h1, uint32_t h2) {
    h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
    return h1;
  }
};

// Atomic operation overloads for 64-bit types
__forceinline__ __device__ long atomicAdd(long *address, long val) {
  return static_cast<long>(
      ::atomicAdd(reinterpret_cast<unsigned long long *>(address),
                  static_cast<unsigned long long>(val)));
}

__forceinline__ __device__ long long atomicAdd(long long *address,
                                               long long val) {
  return static_cast<long long>(
      ::atomicAdd(reinterpret_cast<unsigned long long *>(address),
                  static_cast<unsigned long long>(val)));
}

__forceinline__ __device__ unsigned long atomicAdd(unsigned long *address,
                                                   unsigned long val) {
  return static_cast<unsigned long>(
      ::atomicAdd(reinterpret_cast<unsigned long long *>(address),
                  static_cast<unsigned long long>(val)));
}

__forceinline__ __device__ uint64_t atomicCAS(uint64_t *address,
                                              uint64_t compare, uint64_t val) {
  return static_cast<uint64_t>(
      ::atomicCAS(reinterpret_cast<unsigned long long *>(address),
                  static_cast<unsigned long long>(compare),
                  static_cast<unsigned long long>(val)));
}

__forceinline__ __device__ int64_t atomicCAS(int64_t *address, int64_t compare,
                                             int64_t val) {
  return static_cast<int64_t>(
      ::atomicCAS(reinterpret_cast<unsigned long long *>(address),
                  static_cast<unsigned long long>(compare),
                  static_cast<unsigned long long>(val)));
}

// Type dispatch helper
template <typename Func>
void dispatch_key_type(at::ScalarType key_type, Func &&func) {
  if (key_type == at::kLong) {
    func.template operator()<int64_t>();
  } else if (key_type == at::kUInt64) {
    func.template operator()<uint64_t>();
  } else {
    throw std::invalid_argument(
        "Unsupported key dtype: must be int64 or uint64");
  }
}

// ============================================================================
// Segmented Unique Implementation
// ============================================================================

// ============================================================================
// Packed value encoding for segmented unique
// ============================================================================
// Pack table_id (high 32 bits) and local_unique_idx (low 32 bits) into int64_t
// This allows us to use only 2 arrays (hash_keys, hash_vals) instead of 3

__device__ __forceinline__ int64_t pack_table_val(int64_t table_id,
                                                  int32_t local_idx) {
  // Use uint32_t cast to avoid sign extension issues
  return (static_cast<int64_t>(static_cast<int32_t>(table_id)) << 32) |
         static_cast<uint32_t>(local_idx);
}

__device__ __forceinline__ int64_t unpack_table_id(int64_t packed) {
  return static_cast<int64_t>(static_cast<int32_t>(packed >> 32));
}

__device__ __forceinline__ int32_t unpack_local_idx(int64_t packed) {
  return static_cast<int32_t>(packed & 0xFFFFFFFF);
}

// Binary search helper: largest t such that arr[t] <= val.  Used to derive a
// key's table_id from segmented_range on the fly (instead of materializing a
// per-key table_ids array) and to compact from the partitioned layout.
__device__ __forceinline__ int binary_search_upper_bound(const int64_t *arr,
                                                         int n, int64_t val) {
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (arr[mid] <= val) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo - 1;
}

// Initialize the chaining hash structures: bucket heads and chain pointers to
// -1 (empty), output_indices to -1 (so duplicates spin until a claimer
// publishes a valid index), node_freq to 0, and the per-table counters to 0.
__global__ void segmented_unique_prepare(int *head, int *next_arr,
                                         int *rep_local,
                                         int64_t *output_indices,
                                         int64_t *node_freq,
                                         int64_t *table_counters,
                                         bool count_freq, size_t num_buckets,
                                         size_t num_keys, int64_t num_tables) {
  const size_t stride = blockDim.x * gridDim.x;
  const size_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = gtid; i < num_buckets; i += stride)
    head[i] = -1;
  for (size_t i = gtid; i < num_keys; i += stride) {
    next_arr[i] = -1;
    rep_local[i] = -1;
    output_indices[i] = -1;
    if (count_freq)
      node_freq[i] = 0;
  }
  for (size_t i = gtid; i < static_cast<size_t>(num_tables); i += stride)
    table_counters[i] = 0;
}

// Core dedup kernel (chaining hash, per-table bucket space).  One pass over the
// input in WAVES of blockDim; each distinct (key, table) gets exactly one chain
// node (= its first-occurrence input index):
//   - table_id t comes from a binary search on segmented_range cached in shared;
//     the bucket space is PER-TABLE (offset 4*seg[t], size 4*table_size) so the
//     hash is just hash(key) -- no table_id stored, hashed, or verified;
//   - dedup is a lock-free tail-append: walk d_head[bucket] -> d_next -> ...; a
//     key match (d_keys[cur]==key; the input array is immutable, always readable)
//     is a duplicate; at the tail, ::atomicCAS(-1, idx) appends a new node, and
//     the loser of the CAS re-walks the new tail (collapsing same-key races);
//   - the per-table unique counter is PRIVATIZED: claimers warp-aggregate into a
//     shared per-wave counter s_wc, then ONE global atomicAdd per table per wave
//     reserves the index range (s_base);
//   - phase 3: a claimer writes its local index to rep_local (compaction marker)
//     and publishes it via output_indices, seeding node_freq; a duplicate spins
//     on the rep's output_indices and adds into the rep's node_freq.
// d_head is int32 and keys live in the reused input array, so the hot table
// footprint is ~half the open-addressing version -> better L2 residency.
//
// IMPORTANT: the deferred cross-block publish requires every block to be
// co-resident (a duplicate may spin on another block's published index).  The
// host MUST size the grid to exactly the max active blocks (no oversubscription).
template <typename KeyType, typename Hasher,
          KeyType empty_key = std::numeric_limits<KeyType>::max(),
          int64_t empty_val = std::numeric_limits<int64_t>::max()>
__global__ void __launch_bounds__(SHARED_BLOCK_SIZE)
segmented_unique_core(
    const KeyType *d_keys, int64_t *d_output_indices, size_t num_keys,
    int *d_head, int *d_next, int64_t *node_freq, int *rep_local,
    int64_t *table_counters, const int64_t *d_segmented_range,
    const int64_t *input_frequencies, int64_t num_tables) {
  const int B = blockDim.x;
  const int tid = threadIdx.x;
  const bool count_freq = node_freq != nullptr;

  extern __shared__ char smem[];
  int *s_wc = reinterpret_cast<int *>(smem);                // [num_tables]
  int *s_base = reinterpret_cast<int *>(s_wc + num_tables); // [num_tables]
  int64_t *s_segrange = reinterpret_cast<int64_t *>(s_base + num_tables);
  for (int i = tid; i <= num_tables; i += B)
    s_segrange[i] = d_segmented_range[i];
  __syncthreads();

  for (size_t wave = static_cast<size_t>(blockIdx.x) * B; wave < num_keys;
       wave += static_cast<size_t>(gridDim.x) * B) {
    for (int t = tid; t < num_tables; t += B)
      s_wc[t] = 0;
    __syncthreads();

    const size_t idx = wave + tid;
    const bool valid = idx < num_keys;
    int t = -1;
    int64_t input_freq = 1;
    int kind = 0; // 0 none, 1 claim, 2 duplicate
    int rep = -1; // representative node (idx if claim, matched node if dup)
    int rank = 0;
    if (valid) {
      const KeyType key = d_keys[idx];
      t = binary_search_upper_bound(s_segrange, static_cast<int>(num_tables) + 1,
                                    static_cast<int64_t>(idx));
      input_freq = input_frequencies ? input_frequencies[idx] : 1;
      const int64_t seg_t = s_segrange[t];
      const uint64_t nbkt =
          4ull * static_cast<uint64_t>(s_segrange[t + 1] - seg_t);
      const size_t bucket = static_cast<size_t>(4ull * static_cast<uint64_t>(seg_t) +
                                                (Hasher::hash(key) % nbkt));

      // Lock-free tail-append dedup.
      int cur = d_head[bucket];
      if (cur == -1) {
        const int old = ::atomicCAS(&d_head[bucket], -1, static_cast<int>(idx));
        if (old == -1) {
          kind = 1;
          rep = static_cast<int>(idx);
        } else {
          cur = old;
        }
      }
      while (kind == 0) {
        if (d_keys[cur] == key) {
          kind = 2;
          rep = cur;
          break;
        }
        const int nxt = d_next[cur];
        if (nxt == -1) {
          const int old = ::atomicCAS(&d_next[cur], -1, static_cast<int>(idx));
          if (old == -1) {
            kind = 1;
            rep = static_cast<int>(idx);
            break;
          }
          cur = old;
        } else {
          cur = nxt;
        }
      }
    }

    // Warp-aggregate the shared per-table counter bump for claimers: lanes
    // claiming for the SAME table share one shared atomic (label non-claimers
    // -1 so they form a harmless separate group).
    {
      const int label = (kind == 1) ? t : -1;
      auto active = cg::coalesced_threads();
      auto grp = cg::labeled_partition(active, label);
      if (kind == 1) {
        int base = 0;
        if (grp.thread_rank() == 0)
          base = ::atomicAdd(&s_wc[t], static_cast<int>(grp.size()));
        rank = grp.shfl(base, 0) + static_cast<int>(grp.thread_rank());
      }
    }
    __syncthreads();

    // One global atomic per table reserves the base for this wave's claims.
    for (int tt = tid; tt < num_tables; tt += B) {
      if (s_wc[tt] > 0)
        s_base[tt] = static_cast<int>(
            atomicAdd(&table_counters[tt], static_cast<int64_t>(s_wc[tt])));
    }
    __syncthreads();

    if (valid && kind == 1) {
      const int local_idx = s_base[t] + rank;
      rep_local[idx] = local_idx;
      if (count_freq)
        atomicAdd(&node_freq[idx], input_freq);
      // Publish last so a duplicate that observes it reads a valid index.
      *reinterpret_cast<volatile int64_t *>(&d_output_indices[idx]) = local_idx;
    } else if (valid && kind == 2) {
      int64_t pub;
      do {
        pub = *reinterpret_cast<volatile int64_t *>(&d_output_indices[rep]);
      } while (pub < 0);
      d_output_indices[idx] = pub;
      if (count_freq)
        atomicAdd(&node_freq[rep], input_freq);
    }
    __syncthreads();
  }
}

// ============================================================================
// Fused tail kernel: compute the per-table unique offsets, compact keys/freq
// from the partitioned layout, AND adjust output_indices to global indices --
// replacing the separate cub::DeviceScan and adjust_output_indices_kernel.
// num_tables is small, so each block builds the inclusive prefix sum of
// table_counters (s_off) and caches segmented_range (s_seg) in shared; block 0
// publishes s_off to the global unique_table_offsets (return value).  Two
// independent grid-stride loops then run off the shared tables:
//   - compaction over [0, total_unique): gather unique keys/freq;
//   - adjust over [0, num_keys): output_indices[i] += s_off[table_id(i)].
// partitioned_freq/output_freq may be nullptr when frequency counting is off.
// Block-cooperative exclusive prefix sum: s_off[t] = sum(counts[0..t)) for t in
// [0, num_tables], with s_off[num_tables] = grand total.  Scales to
// num_tables >> blockDim via contiguous per-thread chunks + a Hillis-Steele
// scan of the per-thread totals (O(num_tables/blockDim + log blockDim) instead
// of the O(num_tables) serial scan).  s_part[blockDim] is shared scratch.
__device__ __forceinline__ void
block_offsets_scan(const int64_t *counts, int64_t *s_off, int64_t *s_part,
                   int num_tables) {
  const int B = blockDim.x, tid = threadIdx.x;
  const int chunk = (num_tables + B - 1) / B;
  const int begin = min(tid * chunk, num_tables);
  const int end = min(begin + chunk, num_tables);
  // 1. per-thread sum of its contiguous chunk
  int64_t sum = 0;
  for (int i = begin; i < end; ++i)
    sum += counts[i];
  s_part[tid] = sum;
  __syncthreads();
  // 2. inclusive Hillis-Steele scan of the blockDim per-thread totals
  for (int d = 1; d < B; d <<= 1) {
    int64_t add = (tid >= d) ? s_part[tid - d] : 0;
    __syncthreads();
    s_part[tid] += add;
    __syncthreads();
  }
  // 3. write this thread's chunk: exclusive base = inclusive[tid] - own sum
  int64_t run = s_part[tid] - sum;
  for (int i = begin; i < end; ++i) {
    s_off[i] = run;
    run += counts[i];
  }
  if (tid == B - 1)
    s_off[num_tables] = s_part[B - 1]; // grand total
  __syncthreads();
}

// Finalize: prefix-sum the per-table counts (block_offsets_scan -> s_off),
// publish unique_table_offsets, then a SINGLE pass over the inputs that
//   - adjusts output_indices to global (output_indices[i] += s_off[table]);
//   - for each representative node (rep_local[i] >= 0) emits its key (from the
//     reused input array) and accumulated frequency to the compacted output.
template <typename KeyType>
__global__ void segmented_unique_finalize(
    const KeyType *d_keys, const int *rep_local, const int64_t *node_freq,
    const int64_t *d_segmented_range, const int64_t *table_counters,
    int64_t num_tables, KeyType *output_keys, int64_t *output_freq,
    int64_t *unique_table_offsets, int64_t *output_indices, size_t num_keys) {
  extern __shared__ int64_t smem_off[];
  int64_t *s_off = smem_off;                        // [num_tables+1] unique offs
  int64_t *s_seg = smem_off + (num_tables + 1);     // [num_tables+1] seg range
  int64_t *s_part = s_seg + (num_tables + 1);       // [blockDim] scan scratch
  // Parallel prefix sum of table_counters -> s_off (scales to large num_tables).
  block_offsets_scan(table_counters, s_off, s_part, static_cast<int>(num_tables));
  for (int i = threadIdx.x; i <= num_tables; i += blockDim.x)
    s_seg[i] = d_segmented_range[i];
  __syncthreads();

  // Block 0 publishes the offsets globally for the returned table_offsets /
  // num_uniques.
  if (blockIdx.x == 0)
    for (int i = threadIdx.x; i <= num_tables; i += blockDim.x)
      unique_table_offsets[i] = s_off[i];

  const int nt1 = static_cast<int>(num_tables) + 1;
  const int64_t stride = blockDim.x * gridDim.x;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_keys;
       idx += stride) {
    const int table_id =
        binary_search_upper_bound(s_seg, nt1, static_cast<int64_t>(idx));
    const int64_t base = s_off[table_id];
    output_indices[idx] += base;

    const int li = rep_local[idx];
    if (li >= 0) { // this input is the representative of a unique key
      const int64_t gidx = base + li;
      output_keys[gidx] = d_keys[idx];
      if (output_freq != nullptr)
        output_freq[gidx] = node_freq[idx];
    }
  }
}

// ============================================================================
// Helper kernel to expand table IDs from jagged offsets
// ============================================================================

// Expand jagged offsets to per-element table_ids (identity mapping,
// local_batch_size=1). For each idx, find largest t such that offsets[t] <= idx
// via binary_search_upper_bound (defined above).
__global__ void expand_table_ids_kernel(const int64_t *offsets,
                                        int64_t *table_ids, int num_tables,
                                        int64_t num_elements) {
  const int64_t stride = blockDim.x * gridDim.x;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements;
       idx += stride) {
    table_ids[idx] = binary_search_upper_bound(offsets, num_tables + 1, idx);
  }
}

// (adjust_output_indices folded into segmented_unique_finalize above.)

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
segmented_unique_cuda(at::Tensor keys, at::Tensor segmented_range,
                      int64_t num_tables, at::Tensor input_frequencies) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const int64_t num_keys = keys.numel();
  const auto device = keys.device();
  const auto key_dtype = keys.scalar_type();
  const int device_sm_count = DeviceProp::getDeviceProp(device.index()).num_sms;

  TORCH_CHECK(segmented_range.numel() == num_tables + 1,
              "segmented_range must have num_tables+1 elements");
  TORCH_CHECK(segmented_range.scalar_type() == at::kLong,
              "segmented_range must be int64");
  TORCH_CHECK(segmented_range.device() == device,
              "segmented_range must be on the same device as keys");
  TORCH_CHECK(segmented_range.is_contiguous(),
              "segmented_range must be contiguous");
  TORCH_CHECK(num_tables > 0, "num_tables must be positive");
  TORCH_CHECK(num_keys < std::numeric_limits<int32_t>::max(),
              "num_keys must be less than std::numeric_limits<int32_t>::max()");
  TORCH_CHECK(
      num_tables < std::numeric_limits<int32_t>::max(),
      "num_tables must be less than std::numeric_limits<int32_t>::max()");

  // Frequency counting behavior:
  // - input_frequencies not defined (None): disable frequency counting entirely
  // - input_frequencies defined with numel()==0: enable counting, each key
  // counts as 1
  // - input_frequencies defined with numel()>0: use provided frequencies (must
  // match num_keys)
  const bool enable_freq_counting = input_frequencies.defined();
  const bool has_input_freq =
      enable_freq_counting && input_frequencies.numel() > 0;

  if (has_input_freq) {
    TORCH_CHECK(input_frequencies.numel() == num_keys,
                "input_frequencies must have same length as keys");
  }

  // Debug validation of segmented_range (enabled via DYNAMICEMB_DEBUG=1).
  // Checks: starts at 0, ends at num_keys, monotonically non-decreasing.
  if (std::getenv("DYNAMICEMB_DEBUG")) {
    at::Tensor sr_cpu = segmented_range.to(at::kCPU);
    const int64_t *sr = sr_cpu.data_ptr<int64_t>();
    TORCH_CHECK(sr[0] == 0,
                "segmented_range[0] must be 0, got ", sr[0]);
    TORCH_CHECK(sr[num_tables] == num_keys,
                "segmented_range[num_tables] must equal num_keys (", num_keys,
                "), got ", sr[num_tables]);
    for (int64_t t = 0; t < num_tables; ++t) {
      TORCH_CHECK(sr[t + 1] >= sr[t],
                  "segmented_range must be non-decreasing: "
                  "segmented_range[", t + 1, "]=", sr[t + 1],
                  " < segmented_range[", t, "]=", sr[t]);
    }
  }

  // Handle empty input
  if (num_keys == 0) {
    at::Tensor unique_table_offsets = at::zeros(
        {num_tables + 1}, at::TensorOptions().dtype(at::kLong).device(device));
    at::Tensor num_uniques =
        unique_table_offsets.slice(0, num_tables, num_tables + 1);
    return std::make_tuple(
        num_uniques, at::empty({0}, keys.options()),
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device)),
        unique_table_offsets,
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device)));
  }

  // Grid for the BLOCK_SIZE helper kernels (prepare / finalize / expand): fill
  // the SMs at full occupancy (8 blocks/SM x 256 threads = 64 warps/SM on
  // sm_90, regs<=32) and never exceed the work.  (segmented_unique_core sizes
  // its own grid separately, below.)
  constexpr int BLOCKS_PER_SM = 8;
  int64_t grid64 = (num_keys + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int64_t full_grid = static_cast<int64_t>(device_sm_count) * BLOCKS_PER_SM;
  if (grid64 > full_grid)
    grid64 = full_grid;
  if (grid64 < 1)
    grid64 = 1;
  const int grid_size = static_cast<int>(grid64);

  // No per-key table_ids array: the dedup and adjust kernels cache
  // segmented_range in shared and binary-search it to get a key's table_id on
  // the fly.  Keys must be sorted by table:
  // keys[segmented_range[t]:segmented_range[t+1]] all belong to table t.

  const auto i32 = at::TensorOptions().dtype(at::kInt).device(device);
  const auto i64 = at::TensorOptions().dtype(at::kLong).device(device);

  // Chaining hash, per-table bucket space: bucket heads (4*num_keys), chain
  // pointers, and the claimer marker (rep_local).  int32 indices + reusing the
  // input keys array keep the hot footprint ~half the open-addressing table.
  const int64_t num_buckets = num_keys * 4;
  at::Tensor head = at::empty({num_buckets}, i32);
  at::Tensor next_arr = at::empty({num_keys}, i32);
  at::Tensor rep_local = at::empty({num_keys}, i32);

  // Output indices (per-table-local during core, adjusted to global in
  // finalize).  Initialized to -1 in prepare so duplicates spin until their
  // representative publishes a valid index.
  at::Tensor output_indices = at::empty({num_keys}, i64);

  // Per-table unique counters, and their cumulative offsets (computed/published
  // in segmented_unique_finalize).
  at::Tensor table_counters = at::empty({num_tables}, i64);
  at::Tensor unique_table_offsets = at::empty({num_tables + 1}, i64);

  // Per-node accumulated frequency (only when frequency counting is enabled).
  at::Tensor node_freq;
  if (enable_freq_counting)
    node_freq = at::empty({num_keys}, i64);

  // Grid for the flush-on-fill shared kernel: each block owns a contiguous
  // chunk of keys.  Cap at SHARED_BLOCKS_PER_SM blocks/SM (the 96KB persistent
  // table at S=4096 allows 2 resident blocks of 1024 threads = full occupancy
  // on sm_90; a larger table drops this to 1).  Never launch more blocks than
  // there is work for (>= 1 wave each).
  constexpr int SHARED_BLOCKS_PER_SM = 2;
  int64_t shared_grid64 =
      (num_keys + SHARED_BLOCK_SIZE - 1) / SHARED_BLOCK_SIZE;
  const int64_t shared_full_grid =
      static_cast<int64_t>(device_sm_count) * SHARED_BLOCKS_PER_SM;
  if (shared_grid64 > shared_full_grid)
    shared_grid64 = shared_full_grid;
  if (shared_grid64 < 1)
    shared_grid64 = 1;
  const int shared_grid = static_cast<int>(shared_grid64);

  // Initialize the chaining structures + counters (not key-type dependent).
  segmented_unique_prepare<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      get_pointer<int32_t>(head), get_pointer<int32_t>(next_arr),
      get_pointer<int32_t>(rep_local), get_pointer<int64_t>(output_indices),
      enable_freq_counting ? get_pointer<int64_t>(node_freq) : nullptr,
      get_pointer<int64_t>(table_counters), enable_freq_counting,
      static_cast<size_t>(num_buckets), num_keys, num_tables);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

  dispatch_key_type(key_dtype, [&]<typename KeyType>() {
    // Core dedup kernel, block 1024.  Shared = s_wc + s_base (privatized
    // per-table counters) + s_segrange[num_tables+1] (cached for the table_id
    // lookup / per-table bucket offset).
    const size_t smem_bytes =
        static_cast<size_t>(num_tables) * 2 * sizeof(int) +
        static_cast<size_t>(num_tables + 1) * sizeof(int64_t);
    auto kernel = segmented_unique_core<KeyType, MurmurHash3_32<KeyType>>;

    // Deferred cross-block publish requires all launched blocks to be
    // co-resident, so size the grid to exactly the max active blocks/SM (never
    // oversubscribe).  grid-stride then covers all keys.
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel, SHARED_BLOCK_SIZE, smem_bytes);
    if (blocks_per_sm < 1)
      blocks_per_sm = 1;
    int64_t pk_grid64 = static_cast<int64_t>(device_sm_count) * blocks_per_sm;
    const int64_t pk_need =
        (num_keys + SHARED_BLOCK_SIZE - 1) / SHARED_BLOCK_SIZE;
    if (pk_grid64 > pk_need)
      pk_grid64 = pk_need;
    if (pk_grid64 < 1)
      pk_grid64 = 1;
    const int pk_grid = static_cast<int>(pk_grid64);

    kernel<<<pk_grid, SHARED_BLOCK_SIZE, smem_bytes, stream>>>(
        get_pointer<const KeyType>(keys),
        get_pointer<int64_t>(output_indices), num_keys,
        get_pointer<int32_t>(head), get_pointer<int32_t>(next_arr),
        enable_freq_counting ? get_pointer<int64_t>(node_freq) : nullptr,
        get_pointer<int32_t>(rep_local), get_pointer<int64_t>(table_counters),
        get_pointer<const int64_t>(segmented_range),
        has_input_freq ? get_pointer<const int64_t>(input_frequencies)
                       : nullptr,
        num_tables);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  });

  // Allocate compacted output with size num_keys (worst case: all keys unique)
  // Actual count is unique_table_offsets[num_tables], available on device
  at::Tensor unique_keys = at::empty({num_keys}, keys.options());
  at::Tensor output_freq_counters;
  if (enable_freq_counting) {
    output_freq_counters = at::empty(
        {num_keys}, at::TensorOptions().dtype(at::kLong).device(device));
  } else {
    // Return an empty tensor when frequency counting is disabled
    output_freq_counters =
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device));
  }

  // Fused tail: build unique_table_offsets (parallel prefix sum of
  // table_counters), compact keys + frequency counters, AND adjust
  // output_indices to global indices -- all in one kernel (no separate scan or
  // adjust kernel).  Shared = s_off[num_tables+1] + s_seg[num_tables+1] +
  // s_part[BLOCK_SIZE] (scan scratch); opt in since it can exceed 48KB for
  // large num_tables.
  const size_t tail_smem =
      (static_cast<size_t>(num_tables + 1) * 2 + BLOCK_SIZE) * sizeof(int64_t);
  dispatch_key_type(key_dtype, [&]<typename KeyType>() {
    cudaFuncSetAttribute(segmented_unique_finalize<KeyType>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(tail_smem));
    segmented_unique_finalize<<<grid_size, BLOCK_SIZE, tail_smem, stream>>>(
        get_pointer<const KeyType>(keys), get_pointer<const int32_t>(rep_local),
        enable_freq_counting ? get_pointer<const int64_t>(node_freq) : nullptr,
        get_pointer<const int64_t>(segmented_range),
        get_pointer<const int64_t>(table_counters),
        num_tables, get_pointer<KeyType>(unique_keys),
        enable_freq_counting ? get_pointer<int64_t>(output_freq_counters)
                             : nullptr,
        get_pointer<int64_t>(unique_table_offsets),
        get_pointer<int64_t>(output_indices), num_keys);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  });

  // (output_indices adjustment is fused into segmented_unique_finalize.)

  // Extract num_uniques as a separate tensor (view of
  // unique_table_offsets[num_tables])
  at::Tensor num_uniques =
      unique_table_offsets.slice(0, num_tables, num_tables + 1);

  return std::make_tuple(num_uniques, unique_keys, output_indices,
                         unique_table_offsets, output_freq_counters);
}

// Expand table IDs from offsets (identity mapping, local_batch_size=1).
// offsets: size = num_tables + 1; offsets[t] is the start index for table t.
// num_tables is derived from offsets.size(0)-1.
at::Tensor expand_table_ids_cuda(at::Tensor offsets, int64_t num_elements) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const auto device = offsets.device();
  const int device_sm_count = DeviceProp::getDeviceProp(device.index()).num_sms;

  TORCH_CHECK(offsets.is_cuda(), "offsets must be on CUDA device");

  // Handle empty input
  if (num_elements == 0) {
    return at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device));
  }

  // num_tables derived from offsets; local_batch_size is always 1
  const int64_t num_tables = offsets.size(0) - 1;

  // Compute grid size based on SM count
  constexpr int BLOCKS_PER_SM = 4;
  const int grid_size =
      std::min((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
               static_cast<int64_t>(device_sm_count * BLOCKS_PER_SM));

  at::Tensor table_ids = at::empty(
      {num_elements}, at::TensorOptions().dtype(at::kLong).device(device));

  expand_table_ids_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      get_pointer<const int64_t>(offsets),
      get_pointer<int64_t>(table_ids), num_tables, num_elements);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

  return table_ids;
}

// Compute dedup lengths and offsets using GPU kernel
std::tuple<at::Tensor, at::Tensor> compute_dedup_lengths_cuda(
    at::Tensor unique_offsets, at::Tensor table_offsets_in_feature,
    int64_t num_tables, int64_t local_batch_size, int64_t new_lengths_size) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const auto device = unique_offsets.device();

  TORCH_CHECK(unique_offsets.is_cuda(),
              "unique_offsets must be on CUDA device");
  TORCH_CHECK(table_offsets_in_feature.is_cuda(),
              "table_offsets_in_feature must be on CUDA device");

  // Handle empty case
  if (new_lengths_size == 0) {
    return std::make_tuple(
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device)),
        at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(device)));
  }

  // Allocate output tensors
  at::Tensor new_lengths = at::empty(
      {new_lengths_size}, at::TensorOptions().dtype(at::kLong).device(device));
  at::Tensor new_offsets =
      at::empty({new_lengths_size + 1},
                at::TensorOptions().dtype(at::kLong).device(device));

  // Convert unique_offsets to uint64_t for the kernel
  // The kernel expects uint64_t*, but int64_t is bit-compatible
  get_new_length_and_offsets(
      reinterpret_cast<uint64_t *>(get_pointer<int64_t>(unique_offsets)),
      get_pointer<int64_t>(table_offsets_in_feature), num_tables,
      new_lengths_size, local_batch_size, DataType::Int64, DataType::Int64,
      get_pointer<int64_t>(new_offsets), get_pointer<int64_t>(new_lengths),
      stream);

  return std::make_tuple(new_lengths, new_offsets);
}

} // namespace dyn_emb

// Python bindings
#ifdef DEMB_USE_PYBIND11
void bind_unique_op(py::module &m) {
  m.def(
      "segmented_unique_cuda",
      [](at::Tensor keys, at::Tensor segmented_range, int64_t num_tables,
         const c10::optional<at::Tensor> &input_frequencies) {
        // Convert optional to tensor:
        // - None -> undefined tensor (disables frequency counting)
        // - Some(tensor) -> that tensor (enables frequency counting)
        at::Tensor freq_tensor;
        if (input_frequencies.has_value()) {
          freq_tensor = input_frequencies.value();
        }
        // If input_frequencies was None, freq_tensor remains undefined
        // which will disable frequency counting in the C++ implementation
        return dyn_emb::segmented_unique_cuda(keys, segmented_range, num_tables,
                                              freq_tensor);
      },
      R"doc(
Segmented unique: deduplicate keys per table using GPU hash table.

Keys must be pre-sorted by table: keys[segmented_range[t]:segmented_range[t+1]]
all belong to table t. Keys are deduplicated within each table independently.
The same key can appear in different tables.

NOTE: This function is fully asynchronous with no GPU-CPU synchronization.

Args:
    keys: Input keys tensor (int64 or uint64), sorted by table.
    segmented_range: Table boundary offsets (int64, size=num_tables+1).
                     segmented_range[t] is the start index in keys for table t;
                     segmented_range[num_tables] must equal len(keys).
    num_tables: Total number of tables
    input_frequencies: Controls frequency counting behavior:
                       - None: Disable frequency counting (output freq_counters empty)
                       - Empty tensor (numel==0): Enable counting, each key counts as 1
                       - Tensor with numel==num_keys: Use provided frequencies

Returns:
    Tuple of (num_uniques, unique_keys, output_indices, table_offsets, frequency_counters)
    - num_uniques: Tensor of size 1 with total unique count (on device)
    - unique_keys: Compacted unique keys with size=len(keys). Only first
                   num_uniques elements are valid.
    - output_indices: Index mapping (input idx -> global unique idx)
    - table_offsets: Tensor of size (num_tables + 1) with cumulative unique counts
                     table_offsets[i] is the start index for table i in unique_keys
    - frequency_counters: Per-unique-key frequency counts (empty if disabled)
)doc",
      py::arg("keys"), py::arg("segmented_range"), py::arg("num_tables"),
      py::arg("input_frequencies") = py::none());

  m.def(
      "expand_table_ids_cuda",
      [](at::Tensor offsets, int64_t num_elements) {
        return dyn_emb::expand_table_ids_cuda(offsets, num_elements);
      },
      R"doc(
Expand table IDs from offsets (identity mapping, local_batch_size=1).

Generates a table_id for each element via binary search on offsets.
num_tables is derived from offsets.size(0)-1.

Args:
    offsets: Table boundary offsets (int64, size = num_tables + 1)
             offsets[t] is the start index for table t's keys.
    num_elements: Total number of elements (keys)

Returns:
    table_ids tensor (int64) with same length as num_elements
)doc",
      py::arg("offsets"), py::arg("num_elements") = 0);

  m.def(
      "compute_dedup_lengths_cuda",
      [](at::Tensor unique_offsets, at::Tensor table_offsets_in_feature,
         int64_t num_tables, int64_t local_batch_size,
         int64_t new_lengths_size) {
        return dyn_emb::compute_dedup_lengths_cuda(
            unique_offsets, table_offsets_in_feature, num_tables,
            local_batch_size, new_lengths_size);
      },
      R"doc(
Compute new lengths and offsets by evenly distributing unique keys.

This is a GPU kernel that evenly distributes unique keys across (feature, batch)
buckets. For each table, unique keys are distributed so each bucket gets
(unique_count / num_buckets) keys, with the first (unique_count % num_buckets)
buckets getting one extra.

Args:
    unique_offsets: Cumulative unique counts per table (int64, device)
    table_offsets_in_feature: Feature offsets per table (int64, device)
    num_tables: Number of tables
    local_batch_size: Batch size per feature
    new_lengths_size: Total output size (num_features * local_batch_size)

Returns:
    Tuple of (new_lengths, new_offsets)
    - new_lengths: Length for each bucket (int64)
    - new_offsets: Offset for each bucket (int64)
)doc",
      py::arg("unique_offsets"), py::arg("table_offsets_in_feature"),
      py::arg("num_tables"), py::arg("local_batch_size"),
      py::arg("new_lengths_size"));
}
#endif
