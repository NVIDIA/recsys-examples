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
#include "hkv_variable.h"
#include "utils.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <cooperative_groups.h>
#include "lookup_kernel.cuh"

namespace {

using namespace cooperative_groups;
namespace cg = cooperative_groups;

// Increment the counter when matched
template <class K, class V, class S>
struct EvalAndInc {
  S threshold;
  uint64_t* d_counter;
  EvalAndInc(S threshold, uint64_t* d_counter)
    : threshold(threshold), d_counter(d_counter) {}
  template <int GroupSize>
  __forceinline__ __device__ void operator()(
      const K& key, V* value, S* score, cg::thread_block_tile<GroupSize>& g) {
    S score_val = *score;
    bool match = (not nv::merlin::IS_RESERVED_KEY<K>(key)) && 
                 score_val >= threshold;
    uint32_t vote = g.ballot(match);
    int group_cnt = __popc(vote);
    if (g.thread_rank() == 0) {
      atomicAdd(reinterpret_cast<unsigned long long int*>(d_counter), 
                static_cast<unsigned long long int>(group_cnt));
    }
  }
};

template <class K, class V, class S>
struct ExportIfPredFunctor {
  S threshold;
  ExportIfPredFunctor(S threshold): threshold(threshold) {}
  template <int GroupSize>
  __forceinline__ __device__ bool operator()(
      const K& key, const V* value, const S& score,
      cg::thread_block_tile<GroupSize>& g) {
    return (not nv::merlin::IS_RESERVED_KEY<K>(key)) && 
           score >= threshold;
  }
};

} // end namespace

namespace dyn_emb {

template <typename T>
__global__ void check_safe_pointers_kernel(const uint64_t n, const T **ptrs,
                                           uint64_t *counter) {
  uint64_t id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= n)
    return;
  const T *ptr = ptrs[id];
  if (ptr == nullptr) {
    atomicAdd(counter, 1);
  }
}

template <typename T>
void check_safe_pointers_sync(const uint64_t n, const T **ptrs,
                              const SafeCheckMode safe_check_mode,
                              const cudaStream_t &stream) {
  if (n == 0)
    return;
  static DeviceCounter counter;
  check_safe_pointers_kernel<<< (n + 1023) / 1024, 1024, 0, stream>>>(
      n, ptrs, counter.reset(stream).get());
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  auto result = counter.sync(stream).result();
  if (result == 0) {
    std::cout << "#DynamicEmb LOG: All indices in current batch size " << n
              << " have legal pointers.\n";
    return;
  }
  std::stringstream ss;
  ss << "#DynamicEmb ERROR: Failed to insert " << result
     << " indices in current batch size " << n << ". "
     << "Consider expanding the capacity/num_embedding.\n";
  if (safe_check_mode == SafeCheckMode::WARNING) {
    std::cerr << ss.str();
  } else if (safe_check_mode == SafeCheckMode::ERROR) {
    throw std::runtime_error(ss.str());
  }
}

__global__ static void setup_kernel(unsigned long long seed,
                                    curandState *states) {
  auto grid = cooperative_groups::this_grid();
  curand_init(seed, grid.thread_rank(), 0, &states[grid.thread_rank()]);
}

template <typename ConcretEmbeddingGenerator, typename T>
struct BaseEmbeddingGenerator {
  enum class Action : uint8_t {
    INIT = 0,
    FROM_HKV_TO_TENSOR, // Load embedding from HKV to output Tensor.
    ZERO_TO_TENSOR, // Not found in or inserted into HKV, fill 0 by default.
    TO_HKV_TO_TENSOR, // Generate embedding and store to both HKV and output Tensor.
  };

  DEVICE_INLINE
  BaseEmbeddingGenerator(curandState* state, bool* founds, T** hkv_ptrs) :
    state_(state), founds_(founds), hkv_ptrs_(hkv_ptrs), load_(false), action_(Action::INIT) {}
  
  DEVICE_INLINE
  void destroy() {
    if (load_) {
      state_[GlobalThreadId()] = localState_;
    }
  }

  DEVICE_INLINE
  void set_state(uint64_t emb_id) {
    hkv_ptr_ = hkv_ptrs_[emb_id];
    bool found_ = founds_[emb_id];
    if (hkv_ptr_ == nullptr) {
      action_ = Action::ZERO_TO_TENSOR;
    } else if (found_) {
      action_ = Action::FROM_HKV_TO_TENSOR;
    } else {
      action_ = Action::TO_HKV_TO_TENSOR;
      if (!load_) {
        localState_ = state_[GlobalThreadId()];
        load_ = true;
      }
    }
  }

  DEVICE_INLINE
  T generate(uint32_t i) {
    if (action_ == Action::FROM_HKV_TO_TENSOR) {
      return hkv_ptr_[i];
    } else if (action_ == Action::TO_HKV_TO_TENSOR) {
      auto tmp = static_cast<ConcretEmbeddingGenerator*>(this)->generate_impl(i);
      hkv_ptr_[i] = tmp;
      return tmp;
    }
    return TypeConvertFunc<T, float>::convert(0.0f);
  }

  bool load_;
  Action action_;
  curandState localState_;
  curandState* state_;
  bool* founds_;
  T** hkv_ptrs_;
  T* hkv_ptr_;
};

template <typename T>
struct UniformEmbeddingGenerator : public BaseEmbeddingGenerator<UniformEmbeddingGenerator<T>, T> {
  using Base = BaseEmbeddingGenerator<UniformEmbeddingGenerator<T>, T>;
  struct Args {
    curandState* state;
    T** hkv_ptrs;
    bool* founds;
    float lower;
    float upper;
  };

  DEVICE_INLINE
  UniformEmbeddingGenerator(Args args): Base(args.state, args.founds, args.hkv_ptrs), 
    lower(args.lower), upper(args.upper) {}

  DEVICE_INLINE
  T generate_impl(uint32_t i) {
    auto tmp = curand_uniform_double(&this->localState_);
    return TypeConvertFunc<T, float>::convert((upper - lower) * tmp + lower);
  }

  float lower;
  float upper;
};

template <typename T>
struct NormalEmbeddingGenerator : public BaseEmbeddingGenerator<NormalEmbeddingGenerator<T>, T> {
  using Base = BaseEmbeddingGenerator<NormalEmbeddingGenerator<T>, T>;
  struct Args {
    curandState* state;
    T** hkv_ptrs;
    bool* founds;
    float mean;
    float std_dev;
  };

  DEVICE_INLINE
  NormalEmbeddingGenerator(Args args): Base(args.state, args.founds, args.hkv_ptrs), 
    mean(args.mean), std_dev(args.std_dev) {}

  DEVICE_INLINE
  T generate_impl(uint32_t i) {
    auto tmp = curand_normal_double(&this->localState_);
    return TypeConvertFunc<T, float>::convert(std_dev * tmp + mean);
  }

  float mean;
  float std_dev;
};

template <typename K, typename V>
struct MappingEmbeddingGenerator : public BaseEmbeddingGenerator<MappingEmbeddingGenerator<K,V>, V> {
  using Base = BaseEmbeddingGenerator<MappingEmbeddingGenerator<K,V>, V>;
  struct Args {
    curandState* state;
    V** hkv_ptrs;
    bool* founds;
    const K* keys;
    uint64_t mod;
  };

  DEVICE_INLINE
  MappingEmbeddingGenerator(Args args): Base(args.state, args.founds, args.hkv_ptrs), 
    mod(args.mod), keys(args.keys) {}

  DEVICE_INLINE
  void set_state(uint64_t emb_id) {
    Base::set_state(emb_id);
    key = keys[emb_id];
  }

  DEVICE_INLINE
  V generate_impl(uint32_t i) {
    auto k = static_cast<float>(key % mod);
    return TypeConvertFunc<V, float>::convert(k);
  }

  uint64_t mod;
  const K* keys;
  K key;
};

template <typename T>
struct ConstEmbeddingGenerator : public BaseEmbeddingGenerator<ConstEmbeddingGenerator<T>, T> {
  using Base = BaseEmbeddingGenerator<ConstEmbeddingGenerator<T>, T>;
  struct Args {
    curandState* state;
    T** hkv_ptrs;
    bool* founds;
    float val;
  };

  DEVICE_INLINE
  ConstEmbeddingGenerator(Args args): Base(args.state, args.founds, args.hkv_ptrs), 
    val(args.val) {}

  DEVICE_INLINE
  T generate_impl(uint32_t i) {
    return TypeConvertFunc<T, float>::convert(val);
  }

  float val;
};

template <typename T, typename EmbeddingGenerator, typename Args>
__global__ void fill_embedding_from_generator(
    uint64_t n, uint32_t dim, T* embs, Args args) {
  EmbeddingGenerator generator(args);
  for (uint64_t emb_id = blockIdx.x; emb_id < n; emb_id += gridDim.x) {
    generator.set_state(emb_id);
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
      embs[emb_id * dim + i] = generator.generate(i);
    }
  }
  generator.destroy();
}

static void set_curand_states(curandState **states,
                              const cudaStream_t &stream = 0) {
  auto &deviceProp = DeviceProp::getDeviceProp();
  CUDACHECK(cudaMallocAsync(
      states, sizeof(curandState) * deviceProp.total_threads, stream));
  std::random_device rd;
  auto seed = rd();
  int block_size = deviceProp.max_thread_per_block;
  int grid_size = deviceProp.total_threads / block_size;
  setup_kernel<<<grid_size, block_size, 0, stream>>>(seed, *states);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
HKVVariable<KeyType, ValueType, Strategy>::HKVVariable(
    DataType key_type, DataType value_type, int64_t dim, int64_t init_capacity,
    size_t max_capacity, size_t max_hbm_for_vectors, size_t max_bucket_size,
    float max_load_factor, int block_size, int io_block_size, int device_id,
    bool io_by_cpu, bool use_constant_memory, int reserved_key_start_bit,
    size_t num_of_buckets_per_alloc, const InitializerArgs &initializer_args_,
    const SafeCheckMode safe_check_mode)
    : dim_(dim), max_capacity_(max_capacity),
      initializer_args(initializer_args_), curand_states_(nullptr),
      key_type_(key_type), value_type_(value_type),
      safe_check_mode_(safe_check_mode) {
  if (dim <= 0) {
    throw std::invalid_argument("dimension must > 0 but got " +
                                std::to_string(dim));
  }

  // Init cuda context if necessary.
  if (device_id == -1) { // default value.
    CUDACHECK(cudaGetDevice(&device_id));
  }
  int deviceCount;
  CUDACHECK(cudaGetDeviceCount(&deviceCount));
  if (device_id >= 0 && device_id < deviceCount) {
    const int current_device_id = c10::cuda::current_device();
    if (current_device_id == -1) {
      c10::cuda::set_device(device_id);
    } else if (current_device_id != device_id) {
      throw std::runtime_error(
          "DynamicEmbTable's device id mismatches with torch's.");
    }
    // Init global device property.
    DeviceProp::getDeviceProp(device_id);
  } else {
    throw std::invalid_argument("Invalid device id.");
  }

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  set_curand_states(&curand_states_, stream);
  hkv_table_option_.init_capacity = init_capacity;
  hkv_table_option_.max_capacity = max_capacity;
  hkv_table_option_.max_hbm_for_vectors =
      max_hbm_for_vectors; // nv::merlin::GB(max_hbm_for_vectors);
  hkv_table_option_.max_bucket_size = max_bucket_size;
  hkv_table_option_.dim = dim;
  hkv_table_option_.max_load_factor = max_load_factor;
  hkv_table_option_.block_size = block_size;
  hkv_table_option_.io_block_size = io_block_size;
  hkv_table_option_.device_id = device_id;
  hkv_table_option_.io_by_cpu = io_by_cpu;
  hkv_table_option_.use_constant_memory = use_constant_memory;
  hkv_table_option_.reserved_key_start_bit = reserved_key_start_bit;
  hkv_table_option_.num_of_buckets_per_alloc = num_of_buckets_per_alloc;

  /// TODO: make HKV's init async.
  hkv_table_->init(hkv_table_option_);
  // HKV itself has cuda check, however, it invokes exit() rather than throw
  // error, so we need to disable the HKV check
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
HKVVariable<KeyType, ValueType, Strategy>::~HKVVariable() {
  if (curand_states_) {
    CUDACHECK(cudaFree(curand_states_));
  }
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
int64_t HKVVariable<KeyType, ValueType, Strategy>::rows(cudaStream_t stream) {
  // TODO:do this need a stream?
  return hkv_table_->size(stream);
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
int64_t HKVVariable<KeyType, ValueType, Strategy>::cols() {
  return dim_;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
int64_t HKVVariable<KeyType, ValueType, Strategy>::capacity() {
  return max_capacity_;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
DataType HKVVariable<KeyType, ValueType, Strategy>::key_type() {
  // TODO:do this need a stream?
  return key_type_;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
DataType HKVVariable<KeyType, ValueType, Strategy>::value_type() {
  return value_type_;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
EvictStrategy HKVVariable<KeyType, ValueType, Strategy>::evict_strategy() const {
  return Strategy;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::insert_or_assign(
    const size_t n, const void *keys, const void *values, const void *scores,
    cudaStream_t stream, bool unique_key, bool ignore_evict_strategy) {
  hkv_table_->insert_or_assign(n, (KeyType *)keys, (ValueType *)values,
                               (uint64_t *)scores, stream, unique_key,
                               ignore_evict_strategy);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::insert_and_evict(
    const size_t n, const void *keys, const void *values, const void *scores,
    void *evicted_keys, void *evicted_values, void *evicted_scores,
    uint64_t* d_evicted_counter, cudaStream_t stream, bool unique_key,
    bool ignore_evict_strategy) {
  hkv_table_->insert_and_evict(
      n, (KeyType*)keys, (ValueType*)values, (uint64_t*)scores,
      (KeyType*)evicted_keys, (ValueType*)evicted_values,
      (uint64_t*)evicted_scores, d_evicted_counter, stream,
      unique_key, ignore_evict_strategy);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::accum_or_assign(
    const size_t n, const void *keys, const void *value_or_deltas,
    const bool *accum_or_assigns, const void *scores, cudaStream_t stream,
    bool ignore_evict_strategy) {

  hkv_table_->accum_or_assign(n, (KeyType *)keys, (ValueType *)value_or_deltas,
                              accum_or_assigns, (uint64_t *)scores, stream,
                              ignore_evict_strategy);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}




template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::find_or_insert(
    const size_t n, const void *keys, void **value_ptrs, void *values,
    bool *d_found, void *scores, cudaStream_t stream, bool unique_key,
    bool ignore_evict_strategy) {
  if (n == 0)
    return;
  int64_t dim = cols();
  hkv_table_->find_or_insert(n, (KeyType *)keys, (ValueType **)value_ptrs,
                             d_found, (uint64_t *)scores, stream, unique_key,
                             ignore_evict_strategy);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  if (this->safe_check_mode_ != SafeCheckMode::IGNORE) {
    auto hkv_ptrs = reinterpret_cast<const ValueType **>(
        const_cast<const void **>(value_ptrs));
    check_safe_pointers_sync<ValueType>(n, hkv_ptrs, this->safe_check_mode_,
                                        stream);
  }

  auto &device_prop = DeviceProp::getDeviceProp();
  int block_size = dim < device_prop.max_thread_per_block
                       ? dim
                       : device_prop.max_thread_per_block;
  int grid_size = device_prop.num_sms * (device_prop.max_thread_per_sm / block_size);
  auto &initializer_ = initializer_args.mode;
  if (initializer_ == "normal") {
    using Generator = NormalEmbeddingGenerator<ValueType>;
    using Args = typename Generator::Args;
    auto args = Args {curand_states_, reinterpret_cast<ValueType **>(value_ptrs),
      d_found, initializer_args.mean, initializer_args.std_dev};
    fill_embedding_from_generator<ValueType, Generator, Args><<<grid_size, block_size, 0, stream>>>(
      n, dim, reinterpret_cast<ValueType *>(values), args);
  } else if (initializer_ == "uniform") {
    using Generator = UniformEmbeddingGenerator<ValueType>;
    using Args = typename Generator::Args;
    auto args = Args {curand_states_, reinterpret_cast<ValueType **>(value_ptrs),
      d_found, initializer_args.lower, initializer_args.upper};
    fill_embedding_from_generator<ValueType, Generator, Args><<<grid_size, block_size, 0, stream>>>(
      n, dim, reinterpret_cast<ValueType *>(values), args);
  } else if (initializer_ == "debug") {
    using Generator = MappingEmbeddingGenerator<KeyType, ValueType>;
    using Args = typename Generator::Args;
    auto args = Args {curand_states_, reinterpret_cast<ValueType **>(value_ptrs),
      d_found, reinterpret_cast<const KeyType *>(keys), 100000};
    fill_embedding_from_generator<ValueType, Generator, Args><<<grid_size, block_size, 0, stream>>>(
      n, dim, reinterpret_cast<ValueType *>(values), args);
  } else if (initializer_ == "constant") {
    using Generator = ConstEmbeddingGenerator<ValueType>;
    using Args = typename Generator::Args;
    auto args = Args {curand_states_, reinterpret_cast<ValueType **>(value_ptrs),
      d_found, initializer_args.value};
    fill_embedding_from_generator<ValueType, Generator, Args><<<grid_size, block_size, 0, stream>>>(
      n, dim, reinterpret_cast<ValueType *>(values), args);
  } else {
    throw std::runtime_error("Unrecognized initializer {" + initializer_ + "}");
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::find_or_insert_pointers(
    const size_t n, const void *keys, void **value_ptrs, bool *d_found,
    void *scores, cudaStream_t stream, bool unique_key, bool ignore_evict_strategy) {
  if (n == 0)
    return;
  int64_t dim = cols();
  hkv_table_->find_or_insert(n, (KeyType *)keys, (ValueType **)value_ptrs,
                             d_found, (uint64_t *)scores, stream, unique_key,
                             ignore_evict_strategy);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  if (this->safe_check_mode_ != SafeCheckMode::IGNORE) {
    auto hkv_ptrs = reinterpret_cast<const ValueType **>(
        const_cast<const void **>(value_ptrs));
    check_safe_pointers_sync<ValueType>(n, hkv_ptrs, this->safe_check_mode_,
                                        stream);
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::assign(
    const size_t n, const void *keys, const void *values, const void *scores,
    cudaStream_t stream, bool unique_key) {

  hkv_table_->assign(n, (KeyType *)keys, (ValueType *)values,
                     (uint64_t *)scores, stream, unique_key);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::find(
    const size_t n, const void *keys, void *values, bool *founds, void *scores,
    cudaStream_t stream) const {

  hkv_table_->find(n, (KeyType *)keys, (ValueType *)values, founds,
                   (uint64_t *)scores, stream);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::erase(const size_t n,
                                                      const void *keys,
                                                      cudaStream_t stream) {

  hkv_table_->erase(n, (KeyType *)keys, stream);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType,Strategy >::clear(cudaStream_t stream){
  hkv_table_->clear(stream);
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::reserve(
    const size_t new_capacity, cudaStream_t stream) {

  hkv_table_->reserve(new_capacity, stream);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::export_batch(
    const size_t n, const size_t offset, size_t *d_counter, void *keys,
    void *values, void *scores, cudaStream_t stream) const {

  hkv_table_->export_batch(n, offset, d_counter, (KeyType *)keys,
                           (ValueType *)values, (uint64_t *)scores, stream);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType,Strategy>::count_matched(
    uint64_t threshold, 
    uint64_t* d_counter, 
    cudaStream_t stream) const {
  using ExecutionFunc = EvalAndInc<KeyType, ValueType, uint64_t>;
  ExecutionFunc func(threshold, d_counter);
  hkv_table_->for_each(0, hkv_table_->capacity(), func, stream);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType,Strategy >::export_batch_matched(
    uint64_t threshold,
    const uint64_t n,
    const uint64_t offset,
    uint64_t* d_counter,
    void* keys,              // (n)
    void* values,            // (n, DIM)
    void* scores,            // (n)
    cudaStream_t stream) const {

  using PredFunc = ExportIfPredFunctor<KeyType, ValueType, uint64_t>;
  PredFunc func(threshold);
  hkv_table_->export_batch_if_v2(
    func, n, offset, d_counter, 
    reinterpret_cast<KeyType*>(keys),
    reinterpret_cast<ValueType*>(values),
    reinterpret_cast<uint64_t*>(scores), stream);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
curandState* HKVVariable<KeyType, ValueType, Strategy>::get_curand_states() const {
  return curand_states_;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
const InitializerArgs&  HKVVariable<KeyType, ValueType, Strategy>::get_initializer_args() const {
  return initializer_args;
}

} // namespace dyn_emb
