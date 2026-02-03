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
template <typename KeyType, typename CounterType>
__global__ void init_kernel(KeyType *keys, CounterType *vals,
                            CounterType *counter, size_t capacity,
                            KeyType empty_key, CounterType empty_val) {
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
template <typename KeyType, typename CounterType, typename Hasher>
__global__ void unique_kernel(const KeyType *d_key, KeyType *d_unique_key,
                              CounterType *d_output_index, size_t len,
                              KeyType *keys, CounterType *vals, size_t capacity,
                              CounterType *counter, KeyType empty_key,
                              CounterType empty_val,
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
    constexpr auto empty_key = std::numeric_limits<KeyType>::max();
    constexpr auto empty_val = std::numeric_limits<CounterType>::max();

    // Initialize hash table
    int grid = (capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        get_pointer<KeyType>(hash_keys), get_pointer<CounterType>(hash_vals),
        get_pointer<CounterType>(hash_counter), capacity, empty_key, empty_val);
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
            get_pointer<CounterType>(hash_counter), empty_key, empty_val,
            freq_ptr, input_freq_ptr);

    // Copy count to output
    cudaMemcpyAsync(num_unique.data_ptr(), hash_counter.data_ptr(),
                    sizeof(CounterType), cudaMemcpyDeviceToDevice, stream);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return std::make_tuple(unique_keys, output_indices, num_unique);
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
}
