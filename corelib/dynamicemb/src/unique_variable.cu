/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
#include "unique_variable.h"

namespace dyn_emb {

__forceinline__ __device__ long atomicAdd(long *address, long val) {
  return (long)::atomicAdd((unsigned long long *)address,
                           (unsigned long long)val);
}

__forceinline__ __device__ long long atomicAdd(long long *address,
                                               long long val) {
  return (long long)::atomicAdd((unsigned long long *)address,
                                (unsigned long long)val);
}

__forceinline__ __device__ unsigned long atomicAdd(unsigned long *address,
                                                   unsigned long val) {
  return (unsigned long)::atomicAdd((unsigned long long *)address,
                                    (unsigned long long)val);
}

__forceinline__ __device__ uint32_t atomicCAS(uint32_t *address,
                                              uint32_t compare, uint32_t val) {
  return (uint32_t)::atomicCAS((unsigned int *)address, (unsigned int)compare,
                               (unsigned int)val);
}

__forceinline__ __device__ int32_t atomicCAS(int32_t *address, int32_t compare,
                                             int32_t val) {
  return (int32_t)::atomicCAS((int *)address, (int)compare, (int)val);
}

__forceinline__ __device__ uint64_t atomicCAS(uint64_t *address,
                                              uint64_t compare, uint64_t val) {
  return (uint64_t)::atomicCAS((unsigned long long *)address,
                               (unsigned long long)compare,
                               (unsigned long long)val);
}

__forceinline__ __device__ int64_t atomicCAS(int64_t *address, int64_t compare,
                                             int64_t val) {
  return (int64_t)::atomicCAS((unsigned long long *)address,
                              (unsigned long long)compare,
                              (unsigned long long)val);
}

template <typename KeyType, typename CounterType>
__global__ void
init_kernel(KeyType *keys, CounterType *vals, CounterType *counter,
            const size_t capacity, const KeyType empty_key,
            const CounterType empty_val, const CounterType init_counter_val) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < capacity) {
    // Simply store every element a unused <K, V> pair
    keys[idx] = empty_key;
    vals[idx] = empty_val;
  }
  if (idx == 0) {
    counter[idx] = init_counter_val;
  }
}

template <typename KeyType, typename CounterType>
__global__ void dump_kernel(KeyType *d_key, const KeyType *keys,
                            const CounterType *vals, const size_t offset,
                            const size_t search_length,
                            uint64_t *d_dump_counter, const KeyType empty_key) {
  /* Per block accumulator */
  __shared__ size_t block_acc;

  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  /* Initialize */
  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  KeyType read_key;
  CounterType read_val;
  bool valid_slot = false;
  // Each thread gather the key and value from slot assigned to them.
  if (idx < search_length) {
    read_key = keys[offset + idx];
    if (read_key != empty_key) {
      valid_slot = true;
      atomicAdd(&block_acc, 1);
      read_val = vals[offset + idx];
    }
  }
  __syncthreads();

  // Each block accumulate the dump count to global counter
  if (threadIdx.x == 0) {
    atomicAdd(d_dump_counter, (uint64_t)block_acc);
  }

  // Each thread store one slot's data back to global memory, d_dump_counter is
  // how many slots in total dumped.
  if (valid_slot) {
    d_key[read_val] = read_key;
  }
}

template <typename KeyType, typename CounterType, typename hasher>
__global__ void get_insert_kernel(
    const KeyType *d_key, KeyType *d_unique_key, CounterType *d_val,
    const size_t len, KeyType *keys, CounterType *vals, const size_t capacity,
    CounterType *d_global_counter, const KeyType empty_key,
    const CounterType empty_val, 
    CounterType *d_frequency_counters, 
    const CounterType *d_input_frequencies,  
    CounterType *offset_ptr = nullptr) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    CounterType offset = 0;
    if (offset_ptr != nullptr) {
      offset = offset_ptr[0];
    }
    
    // if d_input_frequencies is nullptr, set input_freq to 1
    CounterType input_freq = (d_input_frequencies != nullptr) ? 
                            d_input_frequencies[idx] : static_cast<CounterType>(1);
    
    KeyType target_key = d_key[idx];
    size_t hash_index = hasher::hash(target_key) % capacity;
    size_t counter = 0;
    while (true) {
      // Have searched all the slot in the hashtable, but all slots in the
      // hashtable are occupied by other keys
      if (counter >= capacity) {
        assert(false && "error: unique op fails: hashtable is full");
      }
      // Try to set the key for the current slot to target key
      // const KeyType old_key = atomicCAS(keys + hash_index, empty_key,
      // target_key);
      const KeyType existing_key = keys[hash_index];
      volatile CounterType &target_val_pos = vals[hash_index];
      if (empty_key == existing_key) {
        const KeyType old_key =
            atomicCAS(keys + hash_index, empty_key, target_key);
        if (empty_key == old_key) {
          CounterType result_val;
          result_val = atomicAdd(d_global_counter, 1);
          d_unique_key[result_val] = target_key;
          d_val[idx] = result_val + offset;
          target_val_pos = result_val;
          
          if (d_frequency_counters != nullptr) {
            atomicCAS(&d_frequency_counters[result_val], 0, input_freq);
          }
          break;
        } else if (target_key == old_key) {
          while (target_val_pos == empty_val) {
          };
          d_val[idx] = target_val_pos + offset;
          
          // accumulate frequency
          if (d_frequency_counters != nullptr) {
            atomicAdd(&d_frequency_counters[target_val_pos], input_freq);
          }
          break;
        }
      } else if (target_key == existing_key) {
        while (target_val_pos == empty_val) {
        };
        d_val[idx] = target_val_pos + offset;
        
        // accumulate frequency
        if (d_frequency_counters != nullptr) {
          atomicAdd(&d_frequency_counters[target_val_pos], input_freq);
        }
        break;
      }
      counter++;
      hash_index = (hash_index + 1) % capacity;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KeyType, typename CounterType, KeyType empty_key,
          CounterType empty_val, typename hasher>
unique_op<KeyType, CounterType, empty_key, empty_val, hasher>::unique_op(
    KeyType *keys, CounterType *vals, CounterType *counter,
    const size_t capacity, const CounterType init_counter_val)
    : keys_(keys), vals_(vals), counter_(counter), capacity_(capacity),
      init_counter_val_(init_counter_val) {
  // Check parameter
  if (capacity_ == 0) {
    throw std::invalid_argument("Invalid value for unique_op capacity");
    return;
  }

  // Initialization kernel, set all entry to unused <K,V>, set counter to init
  // value
  init_kernel<KeyType, CounterType>
      <<<((capacity_ - 1) / BLOCK_SIZE_) + 1, BLOCK_SIZE_>>>(
          keys_, vals_, counter_, capacity_, empty_key, empty_val,
          init_counter_val_);

  // Wait for initialization to finish
  DEMB_CUDA_CHECK(cudaStreamSynchronize(0));
}

template <typename KeyType, typename CounterType, KeyType empty_key,
          CounterType empty_val, typename hasher>
size_t
unique_op<KeyType, CounterType, empty_key, empty_val, hasher>::get_capacity()
    const {
  return capacity_;
}

template <typename KeyType, typename CounterType, KeyType empty_key,
          CounterType empty_val, typename hasher>
void unique_op<KeyType, CounterType, empty_key, empty_val, hasher>::unique(
    const KeyType *d_key, const uint64_t len, CounterType *d_output_index,
    KeyType *d_unique_key, CounterType *d_output_counter, cudaStream_t stream,
    CounterType *offset_ptr, CounterType *d_frequency_counters,
    const CounterType *d_input_frequencies) {

  if (len == 0) {
    // Set the d_output_counter to 0
    CUDACHECK(cudaMemsetAsync(d_output_counter, 0, sizeof(size_t), stream));
    return;
  }

  // Launch get_insert kernel to do unique
  get_insert_kernel<KeyType, CounterType, hasher>
      <<<(len - 1) / BLOCK_SIZE_ + 1, BLOCK_SIZE_, 0, stream>>>(
          d_key, d_unique_key, d_output_index, len, keys_, vals_, capacity_,
          counter_, empty_key, empty_val, d_frequency_counters, d_input_frequencies, offset_ptr);
  // replace counter_ with input d_output_counter
  cudaMemcpyAsync(d_output_counter, counter_, sizeof(CounterType),
                  cudaMemcpyDeviceToDevice, stream);

  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KeyType, typename CounterType, KeyType empty_key,
          CounterType empty_val, typename hasher>
void unique_op<KeyType, CounterType, empty_key, empty_val, hasher>::clear(
    cudaStream_t stream) {
  // Initialization kernel, set all entry to unused <K,V>, set counter to init
  // value
  init_kernel<KeyType, CounterType>
      <<<((capacity_ - 1) / BLOCK_SIZE_) + 1, BLOCK_SIZE_, 0, stream>>>(
          keys_, vals_, counter_, capacity_, empty_key, empty_val,
          init_counter_val_);

  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KeyType, typename CounterType, KeyType empty_key,
          CounterType empty_val, typename hasher>
void unique_op<KeyType, CounterType, empty_key, empty_val,
               hasher>::reset_capacity(KeyType *keys, CounterType *vals,
                                       const size_t capacity,
                                       cudaStream_t stream) {
  keys_ = keys;
  vals_ = vals;
  capacity_ = capacity;
  // Initialization kernel, set all entry to unused <K,V>, set counter to init
  // value
  init_kernel<KeyType, CounterType>
      <<<((capacity_ - 1) / BLOCK_SIZE_) + 1, BLOCK_SIZE_, 0, stream>>>(
          keys_, vals_, counter_, capacity_, empty_key, empty_val,
          init_counter_val_);

  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template class unique_op<int64_t, uint64_t, std::numeric_limits<int64_t>::max(),
                         std::numeric_limits<uint64_t>::max()>;
template class unique_op<uint64_t, uint64_t,
                         std::numeric_limits<uint64_t>::max(),
                         std::numeric_limits<uint64_t>::max()>;
template class unique_op<int64_t, int64_t, std::numeric_limits<int64_t>::max(),
                         std::numeric_limits<int64_t>::max()>;
template class unique_op<uint64_t, int64_t,
                         std::numeric_limits<uint64_t>::max(),
                         std::numeric_limits<int64_t>::max()>;
} // namespace dyn_emb
