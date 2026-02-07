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

#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cassert>
#include <limits>

namespace py = pybind11;

namespace dyn_emb {

constexpr int BLOCK_SIZE = 64;

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

__device__ __forceinline__ int64_t pack_table_val(int32_t table_id,
                                                  int32_t local_idx) {
  // Use uint32_t cast to avoid sign extension issues
  return (static_cast<int64_t>(table_id) << 32) |
         static_cast<uint32_t>(local_idx);
}

__device__ __forceinline__ int32_t unpack_table_id(int64_t packed) {
  return static_cast<int32_t>(packed >> 32);
}

__device__ __forceinline__ int32_t unpack_local_idx(int64_t packed) {
  return static_cast<int32_t>(packed & 0xFFFFFFFF);
}

// Initialize segmented hash table kernel (strided loop version)
// Uses packed (table_id, local_idx) in hash_vals for memory efficiency
template <typename KeyType,
          KeyType empty_key = std::numeric_limits<KeyType>::max(),
          int64_t empty_val = std::numeric_limits<int64_t>::max()>
__global__ void segmented_init_kernel(KeyType *hash_keys, int64_t *hash_vals,
                                      int64_t *table_counters, size_t capacity,
                                      int64_t num_tables) {
  const size_t stride = blockDim.x * gridDim.x;
  // Initialize hash table entries
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < capacity;
       idx += stride) {
    hash_keys[idx] = empty_key;
    hash_vals[idx] = empty_val;
  }
  // Initialize per-table counters
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < static_cast<size_t>(num_tables); idx += stride) {
    table_counters[idx] = 0;
  }
}

// Segmented unique kernel - deduplicates (key, table_id) pairs (strided loop
// version) Uses packed (table_id, local_idx) encoding in hash_vals for
// efficiency Only hash_vals needs volatile reads - hash_keys uses CAS for
// synchronization Supports optional frequency counting for LFU eviction
// strategy
template <typename KeyType, typename Hasher,
          KeyType empty_key = std::numeric_limits<KeyType>::max(),
          int64_t empty_val = std::numeric_limits<int64_t>::max()>
__global__ void
segmented_unique_kernel(const KeyType *d_keys, const int32_t *d_table_ids,
                        KeyType *d_unique_keys, int64_t *d_output_indices,
                        size_t num_keys, KeyType *hash_keys, int64_t *hash_vals,
                        size_t capacity, int64_t *table_counters,
                        size_t max_keys_per_table, int64_t *frequency_counters,
                        const int64_t *input_frequencies) {
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_keys;
       idx += stride) {
    const KeyType key = d_keys[idx];
    const int32_t table_id = d_table_ids[idx];
    const int64_t input_freq = input_frequencies ? input_frequencies[idx] : 1;

    // Hash the (key, table_id) pair
    uint32_t key_hash = Hasher::hash(key);
    uint32_t tid_hash = Hasher::hash(static_cast<uint32_t>(table_id));
    uint32_t combined_hash = Hasher::hash_combine(key_hash, tid_hash);
    size_t hash_index = combined_hash % capacity;

    bool done = false;
    for (size_t probe = 0; probe < capacity && !done; ++probe) {
      const KeyType existing_key = hash_keys[hash_index];

      if (existing_key == empty_key) {
        // Try to claim this slot using CAS on hash_keys
        const KeyType old_key =
            atomicCAS(&hash_keys[hash_index], empty_key, key);

        if (old_key == empty_key) {
          // Successfully claimed the slot
          // Get unique index for this table
          int32_t local_unique_idx =
              static_cast<int32_t>(atomicAdd(&table_counters[table_id], 1));

          // Store unique key in partitioned layout
          size_t output_pos =
              static_cast<size_t>(table_id) * max_keys_per_table +
              local_unique_idx;
          d_unique_keys[output_pos] = key;

          // Pack and store (table_id, local_idx) - this signals completion
          // Use volatile write to ensure visibility
          *reinterpret_cast<volatile int64_t *>(&hash_vals[hash_index]) =
              pack_table_val(table_id, local_unique_idx);

          d_output_indices[idx] = local_unique_idx;

          // Update frequency counter for new unique key
          if (frequency_counters) {
            atomicAdd(&frequency_counters[output_pos], input_freq);
          }
          done = true;
        } else if (old_key == key) {
          // Another thread claimed with same key, wait for packed value
          int64_t packed_val;
          do {
            packed_val =
                *reinterpret_cast<volatile int64_t *>(&hash_vals[hash_index]);
            __nanosleep(1);
          } while (packed_val == empty_val);

          // Check if table_id matches
          if (unpack_table_id(packed_val) == table_id) {
            // Same (key, table_id) pair - use existing index
            int32_t local_idx = unpack_local_idx(packed_val);
            d_output_indices[idx] = local_idx;

            // Update frequency counter for duplicate key
            if (frequency_counters) {
              size_t output_pos =
                  static_cast<size_t>(table_id) * max_keys_per_table +
                  local_idx;
              atomicAdd(&frequency_counters[output_pos], input_freq);
            }
            done = true;
          }
          // Different table_id with same key, continue probing
        }
        // Different key claimed this slot, continue probing
      } else if (existing_key == key) {
        // Slot has same key - read packed value to check table_id
        int64_t packed_val;
        do {
          packed_val =
              *reinterpret_cast<volatile int64_t *>(&hash_vals[hash_index]);
          __nanosleep(1);
        } while (packed_val == empty_val);

        if (unpack_table_id(packed_val) == table_id) {
          // Exact (key, table_id) match found
          int32_t local_idx = unpack_local_idx(packed_val);
          d_output_indices[idx] = local_idx;

          // Update frequency counter for duplicate key
          if (frequency_counters) {
            size_t output_pos =
                static_cast<size_t>(table_id) * max_keys_per_table + local_idx;
            atomicAdd(&frequency_counters[output_pos], input_freq);
          }
          done = true;
        }
        // Different table_id with same key, continue probing
      }

      // Linear probing
      hash_index = (hash_index + 1) % capacity;
    }
    assert(done && "segmented_unique_kernel: hash table full");
  }
}

// Binary search helper for compaction
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

// Fused kernel to compact both keys and frequency counters from partitioned
// layout Shares binary search computation between both operations, reducing
// overhead d_total_unique is a device pointer to avoid GPU-CPU synchronization
// partitioned_freq and output_freq can be nullptr if frequency counting is
// disabled
template <typename KeyType>
__global__ void compact_keys_and_freq_kernel(
    const KeyType *partitioned_keys, const int64_t *partitioned_freq,
    size_t max_keys_per_table, const int64_t *table_offsets, int64_t num_tables,
    KeyType *output_keys, int64_t *output_freq, const int64_t *d_total_unique) {
  const int64_t total_unique = *d_total_unique;
  const int64_t stride = blockDim.x * gridDim.x;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_unique;
       idx += stride) {
    // Find which table this index belongs to (shared computation)
    int table_id =
        binary_search_upper_bound(table_offsets, num_tables + 1, idx);

    // Calculate offset within table
    int64_t local_idx = idx - table_offsets[table_id];

    // Compute source position in partitioned layout
    size_t src_pos =
        static_cast<size_t>(table_id) * max_keys_per_table + local_idx;

    // Compact keys
    output_keys[idx] = partitioned_keys[src_pos];

    // Compact frequency counters if enabled
    if (partitioned_freq != nullptr) {
      output_freq[idx] = partitioned_freq[src_pos];
    }
  }
}

// ============================================================================
// Helper kernel to expand table IDs from jagged offsets
// ============================================================================

// Binary search to find which table an index belongs to (for expand_table_ids)
// When table_offsets_in_feature is nullptr, use identity mapping (feature i =
// table i)
__device__ __forceinline__ int32_t find_table_for_index(
    const int64_t *table_offsets_in_feature, const int64_t *offsets,
    int num_tables, int local_batch_size, int64_t global_idx) {
  // Binary search through tables to find which one contains this index
  int lo = 0, hi = num_tables;
  while (lo < hi) {
    int mid = (lo + hi + 1) / 2;
    // If table_offsets_in_feature is nullptr, use identity: feature mid = table
    // mid
    int64_t table_start_feature =
        table_offsets_in_feature ? table_offsets_in_feature[mid] : mid;
    int64_t table_start_offset = table_start_feature * local_batch_size;
    int64_t table_start_idx = offsets[table_start_offset];
    if (table_start_idx <= global_idx) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return static_cast<int32_t>(lo);
}

// Expand jagged offsets to per-element table_ids (strided loop version)
// Given: offsets tensor and table_offsets_in_feature, generate table_id for
// each element
__global__ void expand_table_ids_kernel(const int64_t *offsets,
                                        const int64_t *table_offsets_in_feature,
                                        int32_t *table_ids, int num_tables,
                                        int local_batch_size,
                                        int64_t num_elements) {
  const int64_t stride = blockDim.x * gridDim.x;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements;
       idx += stride) {
    table_ids[idx] = find_table_for_index(table_offsets_in_feature, offsets,
                                          num_tables, local_batch_size, idx);
  }
}

// Adjust output indices to global indices using table offsets (strided loop
// version)
__global__ void adjust_output_indices_kernel(const int32_t *d_table_ids,
                                             const int64_t *table_offsets,
                                             int64_t *d_output_indices,
                                             size_t num_keys) {
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_keys;
       idx += stride) {
    int32_t table_id = d_table_ids[idx];
    d_output_indices[idx] += table_offsets[table_id];
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
segmented_unique_cuda(at::Tensor keys, at::Tensor table_ids, int64_t num_tables,
                      int64_t device_sm_count, at::Tensor input_frequencies) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const int64_t num_keys = keys.numel();
  const auto device = keys.device();
  const auto key_dtype = keys.scalar_type();

  TORCH_CHECK(keys.numel() == table_ids.numel(),
              "keys and table_ids must have the same length");
  TORCH_CHECK(table_ids.scalar_type() == at::kInt, "table_ids must be int32");
  TORCH_CHECK(num_tables > 0, "num_tables must be positive");
  TORCH_CHECK(device_sm_count > 0, "device_sm_count must be positive");
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

  // Handle empty input
  if (num_keys == 0) {
    at::Tensor table_offsets = at::zeros(
        {num_tables + 1}, at::TensorOptions().dtype(at::kLong).device(device));
    at::Tensor num_uniques = table_offsets.slice(0, num_tables, num_tables + 1);
    return std::make_tuple(
        num_uniques, at::empty({0}, keys.options()),
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device)),
        table_offsets,
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device)));
  }

  // Compute grid size based on SM count (4 blocks per SM is a good heuristic)
  constexpr int BLOCKS_PER_SM = 4;
  const int grid_size = device_sm_count * BLOCKS_PER_SM;

  // Max keys per table (worst case: all keys go to one table)
  const int64_t max_keys_per_table = num_keys;

  // Allocate partitioned output buffer (num_tables * max_keys_per_table)
  at::Tensor partitioned_unique_keys =
      at::empty({num_tables * max_keys_per_table}, keys.options());

  // Allocate output indices (local indices within each table, adjusted later)
  at::Tensor output_indices = at::empty(
      {num_keys}, at::TensorOptions().dtype(at::kLong).device(device));

  // Per-table unique counters
  at::Tensor table_counters = at::zeros(
      {num_tables}, at::TensorOptions().dtype(at::kLong).device(device));

  // Allocate partitioned frequency counters if needed
  at::Tensor partitioned_freq_counters;
  if (enable_freq_counting) {
    partitioned_freq_counters =
        at::zeros({num_tables * max_keys_per_table},
                  at::TensorOptions().dtype(at::kLong).device(device));
  }

  // Allocate shared hash table for (key, table_id) pairs
  // capacity = 2 * num_keys for good load factor
  // hash_vals stores packed (table_id << 32 | local_idx)
  const int64_t capacity = num_keys * 2;
  at::Tensor hash_keys = at::empty({capacity}, keys.options());
  at::Tensor hash_vals = at::empty(
      {capacity}, at::TensorOptions().dtype(at::kLong).device(device));

  dispatch_key_type(key_dtype, [&]<typename KeyType>() {
    // Initialize hash table and counters
    segmented_init_kernel<KeyType><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        get_pointer<KeyType>(hash_keys), get_pointer<int64_t>(hash_vals),
        get_pointer<int64_t>(table_counters), capacity, num_tables);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();

    // Run segmented unique kernel with optional frequency counting
    segmented_unique_kernel<KeyType, MurmurHash3_32<KeyType>>
        <<<grid_size, BLOCK_SIZE, 0, stream>>>(
            get_pointer<const KeyType>(keys),
            get_pointer<const int32_t>(table_ids),
            get_pointer<KeyType>(partitioned_unique_keys),
            get_pointer<int64_t>(output_indices), num_keys,
            get_pointer<KeyType>(hash_keys), get_pointer<int64_t>(hash_vals),
            capacity, get_pointer<int64_t>(table_counters), max_keys_per_table,
            enable_freq_counting
                ? get_pointer<int64_t>(partitioned_freq_counters)
                : nullptr,
            has_input_freq ? get_pointer<const int64_t>(input_frequencies)
                           : nullptr);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  });

  // Compute table offsets using inclusive scan
  at::Tensor table_offsets = at::zeros(
      {num_tables + 1}, at::TensorOptions().dtype(at::kLong).device(device));

  // Use CUB for inclusive scan
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(
      nullptr, temp_storage_bytes, get_pointer<int64_t>(table_counters),
      get_pointer<int64_t>(table_offsets) + 1, num_tables, stream);
  at::Tensor temp_storage =
      at::empty({static_cast<int64_t>(temp_storage_bytes)},
                at::TensorOptions().dtype(at::kByte).device(device));
  cub::DeviceScan::InclusiveSum(temp_storage.data_ptr(), temp_storage_bytes,
                                get_pointer<int64_t>(table_counters),
                                get_pointer<int64_t>(table_offsets) + 1,
                                num_tables, stream);

  // Allocate compacted output with size num_keys (worst case: all keys unique)
  // Actual count is table_offsets[num_tables], available on device
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

  // Compact keys and frequency counters in a single fused kernel
  dispatch_key_type(key_dtype, [&]<typename KeyType>() {
    compact_keys_and_freq_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        get_pointer<const KeyType>(partitioned_unique_keys),
        enable_freq_counting
            ? get_pointer<const int64_t>(partitioned_freq_counters)
            : nullptr,
        max_keys_per_table, get_pointer<const int64_t>(table_offsets),
        num_tables, get_pointer<KeyType>(unique_keys),
        enable_freq_counting ? get_pointer<int64_t>(output_freq_counters)
                             : nullptr,
        get_pointer<const int64_t>(table_offsets) + num_tables);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  });

  // Adjust output indices to global indices
  adjust_output_indices_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      get_pointer<const int32_t>(table_ids),
      get_pointer<const int64_t>(table_offsets),
      get_pointer<int64_t>(output_indices), num_keys);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

  // Extract num_uniques as a separate tensor (view of
  // table_offsets[num_tables])
  at::Tensor num_uniques = table_offsets.slice(0, num_tables, num_tables + 1);

  return std::make_tuple(num_uniques, unique_keys, output_indices,
                         table_offsets, output_freq_counters);
}

// Helper function to expand table IDs from offsets
//
// offsets: size = num_features * local_batch_size + 1
//   - Indexed by (feature_id * local_batch_size + batch_id)
//   - Each feature contains local_batch_size buckets
//
// table_offsets_in_feature: size = num_tables + 1
//   - Maps features to tables (adjacent features may belong to same table)
//   - table_offsets_in_feature[t] is the first feature index for table t
//
// When table_offsets_in_feature is None:
//   - Each feature is treated as a separate table
//   - num_tables = num_features = (offsets.size(0) - 1) / local_batch_size
//
at::Tensor
expand_table_ids_cuda(at::Tensor offsets,
                      c10::optional<at::Tensor> table_offsets_in_feature,
                      int64_t num_tables, int64_t local_batch_size,
                      int64_t num_elements, int64_t device_sm_count) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const auto device = offsets.device();

  TORCH_CHECK(offsets.is_cuda(), "offsets must be on CUDA device");
  TORCH_CHECK(device_sm_count > 0, "device_sm_count must be positive");
  TORCH_CHECK(local_batch_size > 0, "local_batch_size must be positive");

  // Handle empty input
  if (num_elements == 0) {
    return at::empty({0}, at::TensorOptions().dtype(at::kInt).device(device));
  }

  // Compute num_features from offsets size
  int64_t num_features = (offsets.size(0) - 1) / local_batch_size;

  // Determine if we have explicit table_offsets_in_feature or use identity
  // mapping
  const int64_t *table_offsets_ptr = nullptr;
  if (table_offsets_in_feature.has_value() &&
      table_offsets_in_feature.value().numel() > 0) {
    const auto &table_offsets = table_offsets_in_feature.value();
    TORCH_CHECK(table_offsets.is_cuda(),
                "table_offsets_in_feature must be on CUDA device");
    table_offsets_ptr = get_pointer<const int64_t>(table_offsets);
  } else {
    // Each feature = one table, so num_tables = num_features
    // Kernel will use identity mapping when table_offsets_ptr is nullptr
    num_tables = num_features;
  }

  // Compute grid size based on SM count
  constexpr int BLOCKS_PER_SM = 4;
  const int grid_size =
      std::min((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
               static_cast<int64_t>(device_sm_count * BLOCKS_PER_SM));

  // Allocate output table_ids
  at::Tensor table_ids = at::empty(
      {num_elements}, at::TensorOptions().dtype(at::kInt).device(device));

  expand_table_ids_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      get_pointer<const int64_t>(offsets), table_offsets_ptr,
      get_pointer<int32_t>(table_ids), num_tables, local_batch_size,
      num_elements);
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
        at::empty({0}, at::TensorOptions().dtype(at::kInt).device(device)),
        at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(device)));
  }

  // Allocate output tensors
  at::Tensor new_lengths = at::empty(
      {new_lengths_size}, at::TensorOptions().dtype(at::kInt).device(device));
  at::Tensor new_offsets =
      at::empty({new_lengths_size + 1},
                at::TensorOptions().dtype(at::kLong).device(device));

  // Convert unique_offsets to uint64_t for the kernel
  // The kernel expects uint64_t*, but int64_t is bit-compatible
  get_new_length_and_offsets(
      reinterpret_cast<uint64_t *>(get_pointer<int64_t>(unique_offsets)),
      get_pointer<int64_t>(table_offsets_in_feature), num_tables,
      new_lengths_size, local_batch_size, DataType::Int32, DataType::Int64,
      get_pointer<int64_t>(new_offsets), get_pointer<int32_t>(new_lengths),
      stream);

  return std::make_tuple(new_lengths, new_offsets);
}

} // namespace dyn_emb

// Python bindings
void bind_unique_op(py::module &m) {
  m.def(
      "segmented_unique_cuda",
      [](at::Tensor keys, at::Tensor table_ids, int64_t num_tables,
         int64_t device_sm_count,
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
        return dyn_emb::segmented_unique_cuda(keys, table_ids, num_tables,
                                              device_sm_count, freq_tensor);
      },
      R"doc(
Segmented unique: deduplicate keys per table using GPU hash table.

Keys are deduplicated within each table independently. The same key can
appear in different tables. Uses compound hashing on (key, table_id) pairs
with a single shared hash table for memory efficiency.

NOTE: This function is fully asynchronous with no GPU-CPU synchronization.

Args:
    keys: Input keys tensor (int64 or uint64)
    table_ids: Table ID for each key (int32, same length as keys,
               must be in ascending order)
    num_tables: Total number of tables
    device_sm_count: Number of SMs on the device
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
    - table_offsets: Tensor of size (num_tables + 1) with cumulative counts
                     table_offsets[i] is the start index for table i
    - frequency_counters: Per-unique-key frequency counts (empty if disabled)
)doc",
      py::arg("keys"), py::arg("table_ids"), py::arg("num_tables"),
      py::arg("device_sm_count"), py::arg("input_frequencies") = py::none());

  m.def(
      "expand_table_ids_cuda",
      [](at::Tensor offsets, c10::optional<at::Tensor> table_offsets_in_feature,
         int64_t num_tables, int64_t local_batch_size, int64_t num_elements,
         int64_t device_sm_count) {
        return dyn_emb::expand_table_ids_cuda(offsets, table_offsets_in_feature,
                                              num_tables, local_batch_size,
                                              num_elements, device_sm_count);
      },
      R"doc(
Expand table IDs from offsets.

Generates a table_id for each element based on the offsets structure.
This is a helper function to prepare input for segmented_unique_cuda.

Args:
    offsets: Jagged tensor offsets (int64)
             Size = num_features * local_batch_size + 1
             Indexed by (feature_id * local_batch_size + batch_id)

    table_offsets_in_feature: Feature offsets per table (int64), or None
             Size = num_tables + 1
             Maps features to tables (adjacent features may share a table)
             table_offsets_in_feature[t] is the first feature index for table t
             If None: each feature is treated as a separate table

    num_tables: Number of tables (ignored if table_offsets_in_feature is None)
    local_batch_size: Batch size per feature
    num_elements: Total number of elements (keys)
    device_sm_count: Number of SMs on the device

Returns:
    table_ids tensor (int32) with same length as num_elements
)doc",
      py::arg("offsets"), py::arg("table_offsets_in_feature") = py::none(),
      py::arg("num_tables") = 0, py::arg("local_batch_size") = 1,
      py::arg("num_elements") = 0, py::arg("device_sm_count") = 0);

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
    - new_lengths: Length for each bucket (int32)
    - new_offsets: Offset for each bucket (int64)
)doc",
      py::arg("unique_offsets"), py::arg("table_offsets_in_feature"),
      py::arg("num_tables"), py::arg("local_batch_size"),
      py::arg("new_lengths_size"));
}
