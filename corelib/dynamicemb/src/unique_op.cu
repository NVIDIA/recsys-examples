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
  __forceinline__ __host__ __device__ static uint32_t hash_combine(uint32_t h1,
                                                                   uint32_t h2) {
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

// Initialize hash table kernel
template <typename KeyType, typename CounterType,
          KeyType empty_key = std::numeric_limits<KeyType>::max(),
          CounterType empty_val = std::numeric_limits<CounterType>::max()>
__global__ void init_kernel(KeyType *keys, CounterType *vals,
                            CounterType *counter, size_t capacity) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < capacity) {
    keys[idx] = empty_key;
    vals[idx] = empty_val;
  }
  if (idx == 0) {
    counter[0] = 0;
  }
}

// Unique kernel with linear probing hash table
template <typename KeyType, typename CounterType, typename Hasher,
          KeyType empty_key = std::numeric_limits<KeyType>::max(),
          CounterType empty_val = std::numeric_limits<CounterType>::max()>
__global__ void unique_kernel(const KeyType *d_key, KeyType *d_unique_key,
                              CounterType *d_output_index, size_t len,
                              KeyType *keys, CounterType *vals, size_t capacity,
                              CounterType *counter,
                              CounterType *d_frequency_counters,
                              const CounterType *d_input_frequencies) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len)
    return;

  const CounterType input_freq =
      d_input_frequencies ? d_input_frequencies[idx] : 1;

  const KeyType target_key = d_key[idx];
  size_t hash_index = Hasher::hash(target_key) % capacity;

  for (size_t probe = 0; probe < capacity; ++probe) {
    const KeyType existing_key = keys[hash_index];
    volatile CounterType &slot_val = vals[hash_index];

    if (existing_key == empty_key) {
      // Try to claim this slot
      const KeyType old_key =
          atomicCAS(&keys[hash_index], empty_key, target_key);

      if (old_key == empty_key) {
        // Successfully claimed - this is a new unique key
        CounterType unique_idx = atomicAdd(counter, 1);
        d_unique_key[unique_idx] = target_key;
        d_output_index[idx] = unique_idx;
        slot_val = unique_idx;

        if (d_frequency_counters) {
          atomicAdd(&d_frequency_counters[unique_idx], input_freq);
        }
        return;
      } else if (old_key == target_key) {
        // Another thread claimed it with same key
        while (slot_val == empty_val) {
          __nanosleep(1);
        }
        d_output_index[idx] = slot_val;
        if (d_frequency_counters) {
          atomicAdd(&d_frequency_counters[slot_val], input_freq);
        }
        return;
      }
    } else if (existing_key == target_key) {
      // Key already exists
      while (slot_val == empty_val) {
        __nanosleep(1);
      }
      d_output_index[idx] = slot_val;
      if (d_frequency_counters) {
        atomicAdd(&d_frequency_counters[slot_val], input_freq);
      }
      return;
    }

    hash_index = (hash_index + 1) % capacity;
  }
  assert(false && "unique_kernel: hash table full");
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

std::tuple<at::Tensor, at::Tensor, at::Tensor>
unique_cuda(at::Tensor keys, at::Tensor frequency_counters,
            at::Tensor input_frequencies) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const int64_t num_keys = keys.numel();
  const auto device = keys.device();
  const auto key_dtype = keys.scalar_type();

  // Handle empty input
  if (num_keys == 0) {
    return std::make_tuple(
        at::empty({0}, keys.options()),
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device)),
        at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(device)));
  }

  // Allocate output tensors
  at::Tensor unique_keys = at::empty({num_keys}, keys.options());
  at::Tensor output_indices = at::empty(
      {num_keys}, at::TensorOptions().dtype(at::kLong).device(device));
  at::Tensor num_unique =
      at::empty({1}, at::TensorOptions().dtype(at::kLong).device(device));

  // Allocate internal hash table buffers (capacity = 2x input size for good
  // load factor)
  const int64_t capacity = num_keys * 2;
  at::Tensor hash_keys = at::empty({capacity}, keys.options());
  at::Tensor hash_vals = at::empty(
      {capacity}, at::TensorOptions().dtype(at::kLong).device(device));
  at::Tensor hash_counter =
      at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(device));

  dispatch_key_type(key_dtype, [&]<typename KeyType>() {
    using CounterType = int64_t;

    // Initialize hash table
    int grid = (capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_kernel<KeyType, CounterType><<<grid, BLOCK_SIZE, 0, stream>>>(
        get_pointer<KeyType>(hash_keys), get_pointer<CounterType>(hash_vals),
        get_pointer<CounterType>(hash_counter), capacity);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();

    // Get optional pointers
    CounterType *freq_ptr =
        (frequency_counters.defined() && frequency_counters.numel() > 0)
            ? get_pointer<CounterType>(frequency_counters)
            : nullptr;
    const CounterType *input_freq_ptr =
        (input_frequencies.defined() && input_frequencies.numel() > 0)
            ? get_pointer<CounterType>(input_frequencies)
            : nullptr;

    // Run unique kernel
    grid = (num_keys + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unique_kernel<KeyType, CounterType, MurmurHash3_32<KeyType>>
        <<<grid, BLOCK_SIZE, 0, stream>>>(
            get_pointer<const KeyType>(keys), get_pointer<KeyType>(unique_keys),
            get_pointer<CounterType>(output_indices), num_keys,
            get_pointer<KeyType>(hash_keys),
            get_pointer<CounterType>(hash_vals), capacity,
            get_pointer<CounterType>(hash_counter),
            freq_ptr, input_freq_ptr);

    // Copy count to output
    cudaMemcpyAsync(num_unique.data_ptr(), hash_counter.data_ptr(),
                    sizeof(CounterType), cudaMemcpyDeviceToDevice, stream);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return std::make_tuple(unique_keys, output_indices, num_unique);
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

// Segmented unique kernel - deduplicates (key, table_id) pairs (strided loop version)
// Uses packed (table_id, local_idx) encoding in hash_vals for efficiency
// Only hash_vals needs volatile reads - hash_keys uses CAS for synchronization
template <typename KeyType, typename Hasher,
          KeyType empty_key = std::numeric_limits<KeyType>::max(),
          int64_t empty_val = std::numeric_limits<int64_t>::max()>
__global__ void segmented_unique_kernel(
    const KeyType *d_keys, const int32_t *d_table_ids, KeyType *d_unique_keys,
    int64_t *d_output_indices, size_t num_keys, KeyType *hash_keys,
    int64_t *hash_vals, size_t capacity, int64_t *table_counters,
    size_t max_keys_per_table) {
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_keys;
       idx += stride) {
    const KeyType key = d_keys[idx];
    const int32_t table_id = d_table_ids[idx];

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
          size_t output_pos = table_id * max_keys_per_table + local_unique_idx;
          d_unique_keys[output_pos] = key;

          // Pack and store (table_id, local_idx) - this signals completion
          // Use volatile write to ensure visibility
          *reinterpret_cast<volatile int64_t *>(&hash_vals[hash_index]) =
              pack_table_val(table_id, local_unique_idx);

          d_output_indices[idx] = local_unique_idx;
          done = true;
        } else if (old_key == key) {
          // Another thread claimed with same key, wait for packed value
          int64_t packed_val;
          do {
            packed_val = *reinterpret_cast<volatile int64_t *>(
                &hash_vals[hash_index]);
            __nanosleep(1);
          } while (packed_val == empty_val);

          // Check if table_id matches
          if (unpack_table_id(packed_val) == table_id) {
            // Same (key, table_id) pair - use existing index
            d_output_indices[idx] = unpack_local_idx(packed_val);
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
          d_output_indices[idx] = unpack_local_idx(packed_val);
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
__device__ __forceinline__ int
binary_search_upper_bound(const int64_t *arr, int n, int64_t val) {
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

// Compact partitioned keys into contiguous output (strided loop version)
// d_total_unique is a device pointer to avoid GPU-CPU synchronization
template <typename KeyType>
__global__ void compact_keys_kernel(const KeyType *partitioned_keys,
                                    size_t max_keys_per_table,
                                    const int64_t *table_offsets,
                                    int64_t num_tables, KeyType *output_keys,
                                    const int64_t *d_total_unique) {
  const int64_t total_unique = *d_total_unique;
  const int64_t stride = blockDim.x * gridDim.x;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_unique;
       idx += stride) {
    // Find which table this index belongs to
    int table_id = binary_search_upper_bound(table_offsets, num_tables + 1, idx);

    // Calculate offset within table
    int64_t local_idx = idx - table_offsets[table_id];

    // Read from partitioned layout
    output_keys[idx] =
        partitioned_keys[table_id * max_keys_per_table + local_idx];
  }
}

// Adjust output indices to global indices using table offsets (strided loop version)
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

std::tuple<at::Tensor, at::Tensor, at::Tensor>
segmented_unique_cuda(at::Tensor keys, at::Tensor table_ids,
                      int64_t num_tables, int64_t device_sm_count) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const int64_t num_keys = keys.numel();
  const auto device = keys.device();
  const auto key_dtype = keys.scalar_type();

  TORCH_CHECK(keys.numel() == table_ids.numel(),
              "keys and table_ids must have the same length");
  TORCH_CHECK(table_ids.scalar_type() == at::kInt,
              "table_ids must be int32");
  TORCH_CHECK(num_tables > 0, "num_tables must be positive");
  TORCH_CHECK(device_sm_count > 0, "device_sm_count must be positive");

  // Handle empty input
  if (num_keys == 0) {
    return std::make_tuple(
        at::empty({0}, keys.options()),
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device)),
        at::zeros({num_tables + 1},
                  at::TensorOptions().dtype(at::kLong).device(device)));
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
  at::Tensor table_counters =
      at::zeros({num_tables}, at::TensorOptions().dtype(at::kLong).device(device));

  // Allocate shared hash table for (key, table_id) pairs
  // capacity = 2 * num_keys for good load factor
  // hash_vals stores packed (table_id << 32 | local_idx), eliminating hash_table_ids
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

    // Run segmented unique kernel
    segmented_unique_kernel<KeyType, MurmurHash3_32<KeyType>>
        <<<grid_size, BLOCK_SIZE, 0, stream>>>(
            get_pointer<const KeyType>(keys),
            get_pointer<const int32_t>(table_ids),
            get_pointer<KeyType>(partitioned_unique_keys),
            get_pointer<int64_t>(output_indices), num_keys,
            get_pointer<KeyType>(hash_keys), get_pointer<int64_t>(hash_vals),
            capacity, get_pointer<int64_t>(table_counters), max_keys_per_table);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  });

  // Compute table offsets using inclusive scan
  at::Tensor table_offsets =
      at::zeros({num_tables + 1}, at::TensorOptions().dtype(at::kLong).device(device));

  // Use CUB for inclusive scan
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes,
                                get_pointer<int64_t>(table_counters),
                                get_pointer<int64_t>(table_offsets) + 1,
                                num_tables, stream);
  at::Tensor temp_storage = at::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      at::TensorOptions().dtype(at::kByte).device(device));
  cub::DeviceScan::InclusiveSum(temp_storage.data_ptr(), temp_storage_bytes,
                                get_pointer<int64_t>(table_counters),
                                get_pointer<int64_t>(table_offsets) + 1,
                                num_tables, stream);

  // Allocate compacted output with size num_keys (worst case: all keys unique)
  // Actual count is table_offsets[num_tables], available on device
  at::Tensor unique_keys = at::empty({num_keys}, keys.options());

  // Compact keys from partitioned layout to contiguous output
  dispatch_key_type(key_dtype, [&]<typename KeyType>() {
    compact_keys_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        get_pointer<const KeyType>(partitioned_unique_keys), max_keys_per_table,
        get_pointer<const int64_t>(table_offsets), num_tables,
        get_pointer<KeyType>(unique_keys),
        get_pointer<const int64_t>(table_offsets) + num_tables);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  });

  // Adjust output indices to global indices
  adjust_output_indices_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      get_pointer<const int32_t>(table_ids),
      get_pointer<const int64_t>(table_offsets),
      get_pointer<int64_t>(output_indices), num_keys);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(unique_keys, output_indices, table_offsets);
}

} // namespace dyn_emb

// Python bindings
void bind_unique_op(py::module &m) {
  m.def(
      "unique_cuda",
      [](at::Tensor keys, const c10::optional<at::Tensor> &frequency_counters,
         const c10::optional<at::Tensor> &input_frequencies) {
        return dyn_emb::unique_cuda(keys,
                                    frequency_counters.value_or(at::Tensor()),
                                    input_frequencies.value_or(at::Tensor()));
      },
      R"doc(
Deduplicate keys using GPU hash table. Uses the current CUDA stream.

Args:
    keys: Input keys tensor (int64 or uint64)
    frequency_counters: Optional output frequency counter tensor
    input_frequencies: Optional input frequency tensor

Returns:
    Tuple of (unique_keys, output_indices, num_unique)
)doc",
      py::arg("keys"), py::arg("frequency_counters") = py::none(),
      py::arg("input_frequencies") = py::none());

  m.def(
      "segmented_unique_cuda",
      [](at::Tensor keys, at::Tensor table_ids, int64_t num_tables,
         int64_t device_sm_count) {
        return dyn_emb::segmented_unique_cuda(keys, table_ids, num_tables,
                                              device_sm_count);
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
    device_sm_count: Number of SMs on the device (used to determine
                     optimal grid size for kernel launches)

Returns:
    Tuple of (unique_keys, output_indices, table_offsets)
    - unique_keys: Compacted unique keys with size=len(keys). Only first
                   table_offsets[num_tables] elements are valid.
    - output_indices: Index mapping (input idx -> global unique idx)
    - table_offsets: Tensor of size (num_tables + 1) with cumulative counts
                     table_offsets[i] is the start index for table i
                     table_offsets[num_tables] is the total unique count
)doc",
      py::arg("keys"), py::arg("table_ids"), py::arg("num_tables"),
      py::arg("device_sm_count"));
}
