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

struct UniformEmbeddingGenerator {
  struct Args {
    curandState* state;
    float lower;
    float upper;
  };

  DEVICE_INLINE UniformEmbeddingGenerator(Args args): load_(false), state_(args.state), 
    lower(args.lower), upper(args.upper) {}

  DEVICE_INLINE float generate(int64_t vec_id) {
    if (!load_) {
      localState_ = state_[GlobalThreadId()];
      load_ = true;
    }
    auto tmp = curand_uniform_double(&this->localState_);
    return static_cast<float>((upper - lower) * tmp + lower);
  }

  DEVICE_INLINE void destroy() {
    if (load_) {
      state_[GlobalThreadId()] = localState_;
    }
  }

  bool load_;
  curandState localState_;
  curandState* state_;
  float lower;
  float upper;
};

struct NormalEmbeddingGenerator {
  struct Args {
    curandState* state;
    float mean;
    float std_dev;
  };

  DEVICE_INLINE
  NormalEmbeddingGenerator(Args args): load_(false), state_(args.state),
    mean(args.mean), std_dev(args.std_dev) {}

  DEVICE_INLINE
  float generate(int64_t vec_id) {
    if (!load_) {
      localState_ = state_[GlobalThreadId()];
      load_ = true;
    }
    auto tmp = curand_normal_double(&this->localState_);
    return static_cast<float>(std_dev * tmp + mean);
  }

  DEVICE_INLINE void destroy() {
    if (load_) {
      state_[GlobalThreadId()] = localState_;
    }
  }

  bool load_;
  curandState localState_;
  curandState* state_;
  float mean;
  float std_dev;
};

struct TruncatedNormalEmbeddingGenerator {
  struct Args {
    curandState* state;
    float mean;
    float std_dev;
    float lower;
    float upper;
  };

  DEVICE_INLINE
  TruncatedNormalEmbeddingGenerator(Args args): load_(false), state_(args.state),
    mean(args.mean), std_dev(args.std_dev), lower(args.lower), upper(args.upper) {}

  DEVICE_INLINE
  float generate(int64_t vec_id) {
    if (!load_) {
      localState_ = state_[GlobalThreadId()];
      load_ = true;
    }
    auto l = normcdf((lower - mean) / std_dev);
    auto u = normcdf((upper - mean) / std_dev);
    u = 2 * u - 1;
    l = 2 * l - 1;
    float tmp = curand_uniform_double(&this->localState_);
    tmp = tmp * (u - l) + l;
    tmp = erfinv(tmp);
    tmp *= scale * std_dev;
    tmp += mean;
    tmp = max(tmp, lower);
    tmp = min(tmp, upper);
    return tmp;
  }

  DEVICE_INLINE void destroy() {
    if (load_) {
      state_[GlobalThreadId()] = localState_;
    }
  }

  bool load_;
  curandState localState_;
  curandState* state_;
  float mean;
  float std_dev;
  float lower;
  float upper;
  double scale = sqrt(2.0f);
};

template <typename K>
struct MappingEmbeddingGenerator {
  struct Args {
    const K* keys;
    uint64_t mod;
  };

  DEVICE_INLINE
  MappingEmbeddingGenerator(Args args): mod(args.mod), keys(args.keys) {}

  DEVICE_INLINE
  float generate(int64_t vec_id) {
    K key = keys[vec_id];
    return static_cast<float>(key % mod);
  }

  DEVICE_INLINE void destroy() {}

  uint64_t mod;
  const K* keys;
};

struct ConstEmbeddingGenerator {
  struct Args {
    float val;
  };

  DEVICE_INLINE
  ConstEmbeddingGenerator(Args args): val(args.val) {}
  
  DEVICE_INLINE
  float generate(int64_t vec_id) {
    return val;
  }

  DEVICE_INLINE void destroy() {}

  float val;
};

template <typename ElementType, typename SizeType>
struct OptStateInitializer {
  SizeType dim;
  ElementType initial_optstate;
  DEVICE_INLINE void init(ElementType* vec_ptr) {
    if (vec_ptr == nullptr) return;
    for (SizeType i = threadIdx.x; i < dim; i ++) {
      vec_ptr[i] = initial_optstate;
    }
  }
  DEVICE_INLINE void init4(ElementType* vec_ptr) {
    if (vec_ptr == nullptr) return;
    Vec4T<ElementType> state;
    state.reset(initial_optstate);

    constexpr int VecSize = 4;
    constexpr int kWarpSize = 32;
    const int lane_id = threadIdx.x % kWarpSize;
    for (int i = 0; VecSize * (kWarpSize * i + lane_id) < dim; ++i) {
      int idx4 = VecSize * (kWarpSize * i + lane_id);
      state.store(vec_ptr + idx4);
    }
  }
};

template <typename ElementType>
struct TableVector {

  struct Args {
    ElementType** vec_ptrs {nullptr};
    bool* founds {nullptr};
  };

  DEVICE_INLINE TableVector(Args args) : vec_ptrs_(args.vec_ptrs), 
    founds_(args.founds), vec_id_(-1), vec_ptr_(nullptr),  found_(false) {}

  DEVICE_INLINE bool isInitialized(int64_t vec_id) {
    if (vec_id != vec_id_) {
      load(vec_id);
    }
    return found_;
  }

  DEVICE_INLINE bool isValid(int64_t vec_id) {
    if (vec_id != vec_id_) {
      load(vec_id);
    }
    return vec_ptr_ != nullptr;
  }

  DEVICE_INLINE ElementType* data_ptr(int64_t vec_id, int i = 0) {
    if (vec_id != vec_id_) {
      load(vec_id);
    }
    if (vec_ptr_ != nullptr) {
      return vec_ptr_ + i;
    } else {
      return nullptr;
    }
  }

private:
  DEVICE_INLINE void load(int64_t vec_id) {
    vec_id_ = vec_id;
    found_ = founds_[vec_id];
    vec_ptr_ = vec_ptrs_[vec_id];
  }

  ElementType** vec_ptrs_;
  bool* founds_;
  int64_t vec_id_;
  ElementType* vec_ptr_;
  bool found_;
};

template <
  typename T, 
  typename EmbeddingGenerator,
  typename TableVector>
__global__ void fill_output_with_table_vectors_kernel(
    uint64_t n,
    int emb_dim,
    T* outputs, 
    typename TableVector::Args vector_args,
    typename EmbeddingGenerator::Args generator_args) {
  
  TableVector vectors(vector_args);
  EmbeddingGenerator emb_gen(generator_args);

  for (int64_t emb_id = blockIdx.x; emb_id < n; emb_id += gridDim.x) {
    if (vectors.isInitialized(emb_id)) { // copy embedding from table to outputs.
      for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
        outputs[emb_id * emb_dim + i] = *vectors.data_ptr(emb_id, i);
      }
    } else if (vectors.isValid(emb_id)) { // initialize the embedding as well as outputs.
      for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
        auto tmp = emb_gen.generate(emb_id);
        outputs[emb_id * emb_dim + i] = TypeConvertFunc<T, float>::convert(tmp);
        *vectors.data_ptr(emb_id, i) = TypeConvertFunc<T, float>::convert(tmp);
      }
    } else { // vector not exists in table, set the output to 0.
      for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
        outputs[emb_id * emb_dim + i] = TypeConvertFunc<T, float>::convert(0.0f);
      }
    }
  }

  emb_gen.destroy();
}

template <
  typename T, 
  typename EmbeddingGenerator>
__global__ void load_or_initialize_embeddings_kernel(
    uint64_t n,
    int emb_dim,
    T* outputs, 
    T** inputs_ptr,
    bool* masks,
    typename EmbeddingGenerator::Args generator_args) {

  EmbeddingGenerator emb_gen(generator_args);

  for (int64_t emb_id = blockIdx.x; emb_id < n; emb_id += gridDim.x) {
    T* input_ptr = inputs_ptr[emb_id];
    bool mask = masks[emb_id];
    if (mask) { // copy embedding from inputs to outputs.
      for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
        outputs[emb_id * emb_dim + i] = input_ptr[i];
      }
    } else { // initialize the embeddings directly.
      for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
        auto tmp = emb_gen.generate(emb_id);
        outputs[emb_id * emb_dim + i] = TypeConvertFunc<T, float>::convert(tmp);
      }
    }
  }

  emb_gen.destroy();
}

template <
  typename T,
  typename OptStateInitializer,
  typename TableVector>
__global__ void initialize_optimizer_state_kernel_vec4(
    uint64_t n,
    int emb_dim,
    typename TableVector::Args vector_args,
    OptStateInitializer optstate_initailizer) {
  
  TableVector vectors(vector_args);

  constexpr int kWarpSize = 32;
  const int warp_num_per_block = blockDim.x / kWarpSize;
  const int warp_id_in_block = threadIdx.x / kWarpSize;

  for (int64_t emb_id = warp_num_per_block * blockIdx.x + warp_id_in_block;
      emb_id < n; emb_id += gridDim.x * warp_num_per_block) {
    if ((!vectors.isInitialized(emb_id)) and vectors.isValid(emb_id)) {
      optstate_initailizer.init4(vectors.data_ptr(emb_id, emb_dim));
    }
  }
}

template <
  typename T,
  typename OptStateInitializer,
  typename TableVector>
__global__ void initialize_optimizer_state_kernel(
    uint64_t n,
    int emb_dim,
    typename TableVector::Args vector_args,
    OptStateInitializer optstate_initailizer) {
  
  TableVector vectors(vector_args);

  for (int64_t emb_id = blockIdx.x; emb_id < n; emb_id += gridDim.x) {
    if ((!vectors.isInitialized(emb_id)) and vectors.isValid(emb_id)) {
      optstate_initailizer.init(vectors.data_ptr(emb_id, emb_dim));
    }
  }
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
    const SafeCheckMode safe_check_mode, const OptimizerType optimizer_type)
    : dim_(dim), max_capacity_(max_capacity),
      initializer_args(initializer_args_), curand_states_(nullptr),
      key_type_(key_type), value_type_(value_type),
      safe_check_mode_(safe_check_mode), optimizer_type_(optimizer_type) {
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
  hkv_table_option_.dim = dim + get_optimizer_state_dim<ValueType>(optimizer_type, dim);
  int64_t max_hbm_needed = hkv_table_option_.max_capacity * hkv_table_option_.dim * sizeof (ValueType);
  hkv_table_option_.max_hbm_for_vectors = max_hbm_needed < max_hbm_for_vectors ? max_hbm_needed : max_hbm_for_vectors;
  hkv_table_option_.max_bucket_size = max_bucket_size;
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
void HKVVariable<KeyType, ValueType, Strategy>::find_and_initialize(
    const size_t n, const void *keys, void **value_ptrs, void *values,
    bool *d_found, const cudaStream_t& stream) {
  if (n == 0)
    return;
  int dim = dim_;
  this->find_pointers(n, keys, value_ptrs, d_found, nullptr, stream);
  auto &device_prop = DeviceProp::getDeviceProp();
  int block_size = dim < device_prop.max_thread_per_block
                       ? dim
                       : device_prop.max_thread_per_block;
  int grid_size = device_prop.num_sms * (device_prop.max_thread_per_sm / block_size);

  auto &initializer_ = initializer_args.mode;
  if (initializer_ == "normal") {
    using Generator = NormalEmbeddingGenerator;
    auto generator_args = typename Generator::Args {curand_states_, initializer_args.mean, initializer_args.std_dev};
    load_or_initialize_embeddings_kernel<ValueType, Generator>
      <<<grid_size, block_size, 0, stream>>>(
      n, dim, reinterpret_cast<ValueType *>(values), reinterpret_cast<ValueType **>(value_ptrs), d_found, generator_args);
  } else if (initializer_ == "truncated_normal") {
    using Generator = TruncatedNormalEmbeddingGenerator;
    auto generator_args = typename Generator::Args {curand_states_, initializer_args.mean, initializer_args.std_dev, initializer_args.lower, initializer_args.upper};
    load_or_initialize_embeddings_kernel<ValueType, Generator>
      <<<grid_size, block_size, 0, stream>>>(
      n, dim, reinterpret_cast<ValueType *>(values), reinterpret_cast<ValueType **>(value_ptrs), d_found, generator_args);
  } else if (initializer_ == "uniform") {
    using Generator = UniformEmbeddingGenerator;
    auto generator_args = typename Generator::Args {curand_states_, initializer_args.lower, initializer_args.upper};
    load_or_initialize_embeddings_kernel<ValueType, Generator>
      <<<grid_size, block_size, 0, stream>>>(
      n, dim, reinterpret_cast<ValueType *>(values), reinterpret_cast<ValueType **>(value_ptrs), d_found, generator_args);
  } else if (initializer_ == "debug") {
    using Generator = MappingEmbeddingGenerator<KeyType>;
    auto generator_args = typename Generator::Args {reinterpret_cast<const KeyType *>(keys), 100000};
    load_or_initialize_embeddings_kernel<ValueType, Generator>
      <<<grid_size, block_size, 0, stream>>>(
      n, dim, reinterpret_cast<ValueType *>(values), reinterpret_cast<ValueType **>(value_ptrs), d_found, generator_args);
  } else if (initializer_ == "constant") {
    using Generator = ConstEmbeddingGenerator;
    auto generator_args = typename Generator::Args {initializer_args.value};
    load_or_initialize_embeddings_kernel<ValueType, Generator>
      <<<grid_size, block_size, 0, stream>>>(
      n, dim, reinterpret_cast<ValueType *>(values), reinterpret_cast<ValueType **>(value_ptrs), d_found, generator_args);
  } else {
    throw std::runtime_error("Unrecognized initializer {" + initializer_ + "}");
  }
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
  using TableVector = TableVector<ValueType>;
  auto table_vec_args = typename TableVector::Args {reinterpret_cast<ValueType **>(value_ptrs), d_found};

  auto &initializer_ = initializer_args.mode;
  if (initializer_ == "normal") {
    using Generator = NormalEmbeddingGenerator;
    auto generator_args = typename Generator::Args {curand_states_, initializer_args.mean, initializer_args.std_dev};
    fill_output_with_table_vectors_kernel<ValueType, Generator, TableVector>
      <<<grid_size, block_size, 0, stream>>>(
      n, dim, reinterpret_cast<ValueType *>(values), table_vec_args, generator_args);
  } else if (initializer_ == "truncated_normal") {
    using Generator = TruncatedNormalEmbeddingGenerator;
    auto generator_args = typename Generator::Args {curand_states_, initializer_args.mean, initializer_args.std_dev, initializer_args.lower, initializer_args.upper};
    fill_output_with_table_vectors_kernel<ValueType, Generator, TableVector>
      <<<grid_size, block_size, 0, stream>>>(
      n, dim, reinterpret_cast<ValueType *>(values), table_vec_args, generator_args);
  } else if (initializer_ == "uniform") {
    using Generator = UniformEmbeddingGenerator;
    auto generator_args = typename Generator::Args {curand_states_, initializer_args.lower, initializer_args.upper};
    fill_output_with_table_vectors_kernel<ValueType, Generator, TableVector>
      <<<grid_size, block_size, 0, stream>>>(
      n, dim, reinterpret_cast<ValueType *>(values), table_vec_args, generator_args);
  } else if (initializer_ == "debug") {
    using Generator = MappingEmbeddingGenerator<KeyType>;
    auto generator_args = typename Generator::Args {reinterpret_cast<const KeyType *>(keys), 100000};
    fill_output_with_table_vectors_kernel<ValueType, Generator, TableVector>
      <<<grid_size, block_size, 0, stream>>>(
      n, dim, reinterpret_cast<ValueType *>(values), table_vec_args, generator_args);
  } else if (initializer_ == "constant") {
    using Generator = ConstEmbeddingGenerator;
    auto generator_args = typename Generator::Args {initializer_args.value};
    fill_output_with_table_vectors_kernel<ValueType, Generator, TableVector>
      <<<grid_size, block_size, 0, stream>>>(
      n, dim, reinterpret_cast<ValueType *>(values), table_vec_args, generator_args);
  } else {
    throw std::runtime_error("Unrecognized initializer {" + initializer_ + "}");
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

  int optstate_dim = get_optimizer_state_dim<ValueType>(optimizer_type_, dim);
  if (optstate_dim == 0) return;
  using OptStateInitializer = OptStateInitializer<ValueType, int>;
  OptStateInitializer optstate_initializer {optstate_dim, initial_optstate_};

  constexpr int kWarpSize = 32;
  constexpr int MULTIPLIER = 4;
  constexpr int BLOCK_SIZE_VEC = 64;
  constexpr int WARP_PER_BLOCK = BLOCK_SIZE_VEC / kWarpSize;
  const int max_grid_size =
      device_prop.num_sms *
      (device_prop.max_thread_per_sm / BLOCK_SIZE_VEC);
  
  int grid_size_opt = 0;
  if (n / WARP_PER_BLOCK < max_grid_size) {
    grid_size_opt = (n - 1) / WARP_PER_BLOCK + 1;
  } else if (n / WARP_PER_BLOCK > max_grid_size * MULTIPLIER) {
    grid_size_opt = max_grid_size * MULTIPLIER;
  } else {
    grid_size_opt = max_grid_size;
  }

  if (dim % 4 == 0 and optstate_dim % 4 == 0) {
    initialize_optimizer_state_kernel_vec4<ValueType, OptStateInitializer, TableVector>
      <<<grid_size_opt, BLOCK_SIZE_VEC, 0, stream>>>(
      n, dim, table_vec_args, optstate_initializer);
  } else {
    int block_size = optstate_dim < device_prop.max_thread_per_block
                        ? optstate_dim
                        : device_prop.max_thread_per_block;
    int grid_size = n;
    initialize_optimizer_state_kernel<ValueType, OptStateInitializer, TableVector>
      <<<grid_size, block_size, 0, stream>>>(
      n, dim, table_vec_args, optstate_initializer);
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
void HKVVariable<KeyType, ValueType, Strategy>::find_pointers(
    const size_t n, const void *keys, void **value_ptrs, bool *founds,
    void *scores, cudaStream_t stream) const {
  if (n == 0)
    return;
  hkv_table_->find(n, (KeyType *)keys, (ValueType **)value_ptrs,
                   founds, (uint64_t *)scores, stream);
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

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
const int  HKVVariable<KeyType, ValueType, Strategy>::optstate_dim() const {
  return hkv_table_option_.dim - dim_;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::set_initial_optstate(const float value) {
  this->initial_optstate_ = value;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
const float HKVVariable<KeyType, ValueType, Strategy>::get_initial_optstate() const {
  return this->initial_optstate_;
}

} // namespace dyn_emb
