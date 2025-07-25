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

#include <pybind11/pybind11.h>
#include <torch/extension.h>
//#include <torch/python.h>

#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/DeviceUtils.cuh"
#include "check.h"
#include "dynamic_variable_base.h"
#include "index_calculation.h"
#include "lookup_backward.h"
#include "lookup_forward.h"
#include "torch_utils.h"
#include "unique_op.h"
#include "utils.h"
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <torch/torch.h>
#include <cooperative_groups.h>
#include <optional>
#include "hkv_variable.cuh"
// #include <source_location>
#include <cstring>

namespace py = pybind11;
using namespace dyn_emb;

template <typename T, class = std::enable_if_t<std::is_integral_v<T>>>
inline bool power2(T v) {

  return v && (v & -v) == v;
}

at::Tensor create_sub_tensor(const at::Tensor &original_tensor,
                             int64_t offset) {
  if (offset < 0 || offset >= original_tensor.numel()) {
    throw std::runtime_error("Invalid offset");
  }

  void *data_ptr =
      original_tensor.data_ptr() + offset * original_tensor.element_size();

  int64_t new_size = original_tensor.numel() - offset;

  at::Tensor new_tensor =
      at::from_blob(data_ptr, {new_size}, original_tensor.options());

  return new_tensor;
}

// REMOVE LATER:check result create_sub_tensor correct
void check_sub_tensor(const at::Tensor &original_tensor,
                      const at::Tensor &new_tensor, int64_t offset) {
  void *original_data_ptr = original_tensor.data_ptr();

  void *new_data_ptr = new_tensor.data_ptr();

  std::cout << "Original tensor data pointer: " << original_data_ptr
            << std::endl;
  std::cout << "New tensor data pointer: " << new_data_ptr << std::endl;

  void *expected_new_data_ptr = static_cast<char *>(original_data_ptr) +
                                offset * original_tensor.element_size();

  if (new_data_ptr == expected_new_data_ptr) {
    std::cout << "The new tensor data pointer is correctly referencing the "
                 "original tensor's memory."
              << std::endl;
  } else {
    std::cout << "The new tensor data pointer is NOT correctly referencing the "
                 "original tensor's memory."
              << std::endl;
  }
}

// Dyn_emb API
// TODO all the API need check datatype and dimension continuous
int64_t dyn_emb_rows(std::shared_ptr<dyn_emb::DynamicVariableBase> table) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  return table->rows(stream);
}

int64_t dyn_emb_cols(std::shared_ptr<dyn_emb::DynamicVariableBase> table) {
  return table->cols();
}

int64_t dyn_emb_capacity(std::shared_ptr<dyn_emb::DynamicVariableBase> table) {
  return table->capacity();
}

void insert_or_assign(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
                      const size_t n, const at::Tensor keys,
                      const at::Tensor values,
                      const c10::optional<at::Tensor> &score = c10::nullopt,
                      bool unique_key = true,
                      bool ignore_evict_strategy = false) {

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu) {
    if (score.has_value()) {
      at::Tensor score_ = score.value();
      table->insert_or_assign(n, keys.data_ptr(), values.data_ptr(),
                              score_.data_ptr(), stream, unique_key,
                              ignore_evict_strategy);
    } else {
      throw std::runtime_error("Not provide score in Customized or LFU mode.");
    }
  } else {
    table->insert_or_assign(n, keys.data_ptr(), values.data_ptr(), nullptr,
                            stream, unique_key, ignore_evict_strategy);
  }
}

// If don't need input scores, `scores` can be set to std::nullopt.
void insert_and_evict(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    const size_t n,
    const at::Tensor keys,
    const at::Tensor values,
    const std::optional<uint64_t> score,
    at::Tensor evicted_keys,
    at::Tensor evicted_values,
    at::Tensor evicted_score,
    at::Tensor d_evicted_counter,
    bool unique_key = true,
    bool ignore_evict_strategy = false) {

  if (not score and (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu)) {
    throw std::invalid_argument("Must specify the score when evict strategy is customized or LFU.");
  }
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu) {
    auto&& option = at::TensorOptions().dtype(at::kUInt64).device(keys.device());
    // broadcast scores
    at::Tensor bc_scores = at::empty({static_cast<int64_t>(n)}, option);
    bc_scores.fill_(score.value());
    table->insert_and_evict(
      n, keys.data_ptr(), values.data_ptr(), bc_scores.data_ptr(),
      evicted_keys.data_ptr(), evicted_values.data_ptr(), evicted_score.data_ptr(),
      reinterpret_cast<uint64_t*>(d_evicted_counter.data_ptr()), stream, unique_key, ignore_evict_strategy);
  } else {
    table->insert_and_evict(
      n, keys.data_ptr(), values.data_ptr(), nullptr, 
      evicted_keys.data_ptr(), evicted_values.data_ptr(), evicted_score.data_ptr(),
      reinterpret_cast<uint64_t*>(d_evicted_counter.data_ptr()), stream, unique_key, ignore_evict_strategy);
  }
}

// If don't need input scores, `scores` can be set to std::nullopt.
void insert_and_evict_(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    const size_t n,
    const at::Tensor keys,
    const at::Tensor values,
    c10::optional<at::Tensor> const &scores,
    at::Tensor evicted_keys,
    at::Tensor evicted_values,
    at::Tensor evicted_score,
    at::Tensor d_evicted_counter,
    bool unique_key = true,
    bool ignore_evict_strategy = false) {

  if (not scores and (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu)) {
    throw std::invalid_argument("Must specify the score when evict strategy is customized or LFU.");
  }
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu) {
    table->insert_and_evict(
      n, keys.data_ptr(), values.data_ptr(), scores.value().data_ptr(),
      evicted_keys.data_ptr(), evicted_values.data_ptr(), evicted_score.data_ptr(),
      reinterpret_cast<uint64_t*>(d_evicted_counter.data_ptr()), stream, unique_key, ignore_evict_strategy);
  } else {
    table->insert_and_evict(
      n, keys.data_ptr(), values.data_ptr(), nullptr, 
      evicted_keys.data_ptr(), evicted_values.data_ptr(), evicted_score.data_ptr(),
      reinterpret_cast<uint64_t*>(d_evicted_counter.data_ptr()), stream, unique_key, ignore_evict_strategy);
  }
}

void accum_or_assign(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
                     const size_t n, const at::Tensor keys,
                     const at::Tensor value_or_deltas,
                     const at::Tensor accum_or_assigns,
                     const c10::optional<at::Tensor> &score = c10::nullopt,
                     bool ignore_evict_strategy = false) {

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (score.has_value()) {
    at::Tensor score_ = score.value();
    table->accum_or_assign(n, keys.data_ptr(), value_or_deltas.data_ptr(),
                           accum_or_assigns.data_ptr<bool>(), score_.data_ptr(),
                           stream, ignore_evict_strategy);
  } else {
    table->accum_or_assign(n, keys.data_ptr(), value_or_deltas.data_ptr(),
                           accum_or_assigns.data_ptr<bool>(), nullptr, stream,
                           ignore_evict_strategy);
  }
}


void find_and_initialize(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    const size_t n,
    const at::Tensor keys,
    const at::Tensor values) {

  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  at::Tensor vals_ptr_tensor = at::empty({static_cast<int64_t>(n)}, 
    at::TensorOptions().dtype(at::kLong).device(values.device()));
  auto vals_ptr = reinterpret_cast<void**>(vals_ptr_tensor.data_ptr<int64_t>());
  at::Tensor founds_tensor = at::empty({static_cast<int64_t>(n)},
     at::TensorOptions().dtype(at::kBool).device(keys.device()));
  auto founds = founds_tensor.data_ptr<bool>();

  table->find_and_initialize(n, keys.data_ptr(), vals_ptr, values.data_ptr(), founds, stream);
}

namespace dyn_emb {

template<
  typename key_t,
  typename idx_t>
__global__ void get_missed_keys_kernel(
    int64_t n, const bool* founds, const key_t* keys, key_t* missed_keys, idx_t* missed_ids, int64_t* missed_counter, idx_t* reverse_ids=nullptr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  cg::thread_block_tile<32> g = cg::tiled_partition<32>(cg::this_thread_block());
  int64_t group_begin = (tid >> 5) * 32;
  int lane = tid % 32;
  for (int64_t i = group_begin; i < n; i += gridDim.x * blockDim.x) {
    bool missed;
    key_t key;
    if (i + lane < n) {
      missed = not founds[i + lane];
      key = keys[i + lane];
    } else {
      missed = false;
    }
    uint32_t vote = g.ballot(missed);
    int group_cnt = __popc(vote);
    int64_t group_offset = 0;
    if (g.thread_rank() == 0) {
      group_offset = atomicAdd(missed_counter, static_cast<int64_t>(group_cnt));
    }
    group_offset = g.shfl(group_offset, 0);
    // Each thread gets the count of previous missed ranks.
    int previous_cnt = group_cnt - __popc(vote >> g.thread_rank());
    if (missed) {
      missed_keys[group_offset + previous_cnt] = key;
      missed_ids[i + lane] = static_cast<idx_t>(group_offset + previous_cnt);
      if (reverse_ids) {
        reverse_ids[group_offset + previous_cnt] = i + lane;
      }
    }
  }
}

template<
  typename key_t,
  typename idx_t,
  typename double_counter>
__global__ void get_classified_keys_kernel(
    int n, const bool* founds, const key_t* keys, key_t* classified_keys,
    uint64_t* classified_ids, idx_t* original_ids, double_counter counter) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  cg::thread_block_tile<32> g = cg::tiled_partition<32>(cg::this_thread_block());
  int64_t group_begin = (tid >> 5) * 32;
  int lane = tid % 32;
  for (int64_t i = group_begin; i < n; i += gridDim.x * blockDim.x) {
    bool found = false;
    key_t key;
    if (i + lane < n) {
      key = keys[i + lane];
      found = founds[i + lane];
    } else {
      found = false;
    }
    uint32_t found_vote = g.ballot(found);
    int found_cnt = __popc(found_vote);
    uint32_t missed_vote = ~found_vote;
    int missed_cnt;
    if (i + 31 < n) {
      missed_cnt = 32 - found_cnt;
    } else {
      missed_cnt = (n - i) - found_cnt;
    }
    idx_t found_offset = 0;
    idx_t missed_offset = 0;
    if (g.thread_rank() == 0) {
      ///TODO: if cnt = 0 then not to get offset.
      found_offset = counter.get_offset1(found_cnt);
      missed_offset = counter.get_offset2(missed_cnt);
    }
    found_offset = g.shfl(found_offset, 0);
    missed_offset = g.shfl(missed_offset, 0);
    // Each thread gets the count of before lanes.
    uint32_t before_mask = (1u << lane) - 1;
    int prefix_found_count = __popc(found_vote & before_mask);
    int prefix_missed_cnt = __popc(missed_vote & before_mask);
    
    if (found) {
      idx_t dst_id = found_offset + prefix_found_count;
      classified_keys[dst_id] = key;
      original_ids[dst_id] = static_cast<idx_t>(i + lane);
      classified_ids[i + lane] = static_cast<uint64_t>(dst_id);
    } else if (i + lane < n)  {
      idx_t dst_id = (n - 1) - (missed_offset + prefix_missed_cnt);
      classified_keys[dst_id] = key;
      classified_ids[i + lane] = static_cast<uint64_t>(dst_id);
    }
  }
}

__global__ void remapping_reverse_ids_kernel(
  uint64_t n,
  uint64_t* reverse_ids,
  uint64_t const * remapping_ids,
  uint64_t const * offset_ptr
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint64_t i = tid; i < n; i += blockDim.x * gridDim.x) {
    auto offset = offset_ptr[0];
    uint64_t reverse_id = reverse_ids[i] - offset;
    uint64_t remapping_id = remapping_ids[reverse_id];
    reverse_ids[i] = remapping_id + offset;
  }
}

template <
  typename T,
  typename Idx,
  typename EmbeddingGenerator>
__global__ void load_twice_or_initialize_embeddings_kernel(
    uint64_t n,
    int emb_dim,
    T* outputs, 
    T** inputs_ptr,
    bool* masks,
    Idx* missed_ids,
    T** inputs_ptr2,
    bool* masks2,
    typename EmbeddingGenerator::Args generator_args) {

  EmbeddingGenerator emb_gen(generator_args);

  for (int64_t emb_id = blockIdx.x; emb_id < n; emb_id += gridDim.x) {
    bool mask = masks[emb_id];
    T* input_ptr = inputs_ptr[emb_id];
    Idx missed_id = missed_ids[emb_id];
    if (mask) { // copy embedding from inputs to outputs.
      for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
        outputs[emb_id * emb_dim + i] = input_ptr[i];
      }
    } else {
      bool mask2 = masks2[missed_id];
      T* input_ptr2 = inputs_ptr2[missed_id];
      if (mask2) {
        for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
          outputs[emb_id * emb_dim + i] = input_ptr2[i];
        }
      } else { // initialize the embeddings directly.
        for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
          auto tmp = emb_gen.generate(emb_id);
          outputs[emb_id * emb_dim + i] = TypeConvertFunc<T, float>::convert(tmp);
        }
      }
    }
  }

  emb_gen.destroy();
}

template <
  typename T,
  typename Idx,
  typename EmbeddingGenerator>
__global__ void load_or_initialize_classified_embeddings_kernel(
    int n,
    int emb_dim,
    T* outputs, 
    T** inputs_ptr,
    int found_counter,
    Idx* original_ids,
    T** inputs_ptr2,
    bool* masks2,
    typename EmbeddingGenerator::Args generator_args) {

  EmbeddingGenerator emb_gen(generator_args);

  for (int emb_id = blockIdx.x; emb_id < n; emb_id += gridDim.x) {
    bool mask = false;
    T* input_ptr = nullptr;
    if (emb_id < found_counter) {
      mask = true;
      Idx original_id = original_ids[emb_id];
      input_ptr = inputs_ptr[original_id];
    } else {
      mask = masks2[emb_id - found_counter];
      input_ptr = inputs_ptr2[emb_id - found_counter];
    }
  
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

template <typename CounterType>
struct DoubleCounter {
CounterType * counter;

DEVICE_INLINE
CounterType get_offset1(CounterType increment) {
  return atomicAdd(counter, increment);
}

DEVICE_INLINE
CounterType get_offset2(CounterType increment) {
  return atomicAdd(counter + 1, increment);
}
};

void find_and_initialize_from_hierarchical_table(
    std::shared_ptr<dyn_emb::DynamicVariableBase> device_table,
    std::shared_ptr<dyn_emb::DynamicVariableBase> host_table,
    const size_t n,
    const at::Tensor keys,
    at::Tensor classified_keys,
    at::Tensor classified_ids,
    const at::Tensor values,
    const c10::optional<at::Tensor>& cache_metrics=c10::nullopt
  ) {

  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  at::Tensor vals_dev_ptr_tensor = at::empty({static_cast<int64_t>(n)}, 
    at::TensorOptions().dtype(at::kLong).device(values.device()));
  auto vals_dev_ptr = reinterpret_cast<void**>(vals_dev_ptr_tensor.data_ptr<int64_t>());
  at::Tensor founds_dev_tensor = at::empty({static_cast<int64_t>(n)},
     at::TensorOptions().dtype(at::kBool).device(keys.device()));
  auto founds_dev = founds_dev_tensor.data_ptr<bool>();

  device_table->find_pointers(n, keys.data_ptr(), vals_dev_ptr, founds_dev, nullptr, stream);

  auto original_ids = at::empty({static_cast<int64_t>(n)},
     at::TensorOptions().dtype(at::kInt).device(keys.device()));
  auto counter = at::zeros({static_cast<int64_t>(2)},
     at::TensorOptions().dtype(at::kInt).device(keys.device()));
  DISPATCH_INTEGER_DATATYPE_FUNCTION(host_table->key_type(), key_t, [&] {
    get_classified_keys_kernel<key_t, int, DoubleCounter<int>><<<(n + 127) / 128, 128, 0, stream>>>(
      n, founds_dev, reinterpret_cast<key_t*>(keys.data_ptr()), 
      reinterpret_cast<key_t*>(classified_keys.data_ptr()), reinterpret_cast<uint64_t*>(classified_ids.data_ptr()), 
      original_ids.data_ptr<int>(), DoubleCounter<int> {counter.data_ptr<int>()}
    );
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  int found_counter = 0;
  AT_CUDA_CHECK(cudaMemcpyAsync(&found_counter, counter.data_ptr(),
      sizeof(int), cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  int missed_counter = n - found_counter;
  if (cache_metrics.has_value()) {
    cache_metrics.value()[0] = static_cast<int>(n);
    cache_metrics.value()[1] = static_cast<int>(found_counter);
  }

  auto vals_host_ptr_tensor = at::empty({static_cast<int64_t>(missed_counter)}, vals_dev_ptr_tensor.options());
  auto vals_host_ptr = reinterpret_cast<void**>(vals_host_ptr_tensor.data_ptr<int64_t>());
  auto founds_host = at::empty({static_cast<int64_t>(missed_counter)}, founds_dev_tensor.options()).data_ptr<bool>();

  auto missed_keys_ptr = reinterpret_cast<char*>(classified_keys.data_ptr()) + classified_keys.element_size() * found_counter;

  host_table->find_pointers(missed_counter, (void*)missed_keys_ptr, vals_host_ptr, founds_host, nullptr, stream);

  int dim = host_table->cols();
  auto &device_prop = DeviceProp::getDeviceProp();
  int block_size = dim < device_prop.max_thread_per_block
                       ? dim
                       : device_prop.max_thread_per_block;
  int grid_size = device_prop.num_sms * (device_prop.max_thread_per_sm / block_size);

  auto &initializer_args = device_table->get_initializer_args();
  auto* curand_states_ = device_table->get_curand_states();
  DISPATCH_FLOAT_DATATYPE_FUNCTION(device_table->value_type(), ValueType, [&] {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(device_table->key_type(), KeyType, [&] {
      if (initializer_args.mode == "normal") {
        using Generator = NormalEmbeddingGenerator;
        auto generator_args = typename Generator::Args {curand_states_, initializer_args.mean, initializer_args.std_dev};
        load_or_initialize_classified_embeddings_kernel<ValueType, int, Generator>
          <<<grid_size, block_size, 0, stream>>>(
          n, dim, reinterpret_cast<ValueType *>(values.data_ptr()), (ValueType **)(vals_dev_ptr), found_counter,
          original_ids.data_ptr<int>(), (ValueType **)(vals_host_ptr), founds_host, generator_args);
      } else if (initializer_args.mode == "truncated_normal") {
        using Generator = TruncatedNormalEmbeddingGenerator;
        auto generator_args = typename Generator::Args {curand_states_, initializer_args.mean, initializer_args.std_dev, initializer_args.lower, initializer_args.upper};
        load_or_initialize_classified_embeddings_kernel<ValueType, int, Generator>
          <<<grid_size, block_size, 0, stream>>>(
          n, dim, reinterpret_cast<ValueType *>(values.data_ptr()), (ValueType **)(vals_dev_ptr), found_counter,
          original_ids.data_ptr<int>(), (ValueType **)(vals_host_ptr), founds_host, generator_args);
      } else if (initializer_args.mode == "uniform") {
        using Generator = UniformEmbeddingGenerator;
        auto generator_args = typename Generator::Args {curand_states_, initializer_args.lower, initializer_args.upper};
        load_or_initialize_classified_embeddings_kernel<ValueType, int, Generator>
          <<<grid_size, block_size, 0, stream>>>(
          n, dim, (ValueType *)(values.data_ptr()), (ValueType **)(vals_dev_ptr), found_counter,
          original_ids.data_ptr<int>(), (ValueType **)(vals_host_ptr), founds_host,generator_args);
      } else if (initializer_args.mode == "debug") {
        using Generator = MappingEmbeddingGenerator<KeyType>;
        auto generator_args = typename Generator::Args {reinterpret_cast<const KeyType *>(classified_keys.data_ptr()), 100000};
        load_or_initialize_classified_embeddings_kernel<ValueType, int, Generator>
          <<<grid_size, block_size, 0, stream>>>(
          n, dim, reinterpret_cast<ValueType *>(values.data_ptr()), (ValueType **)(vals_dev_ptr), found_counter,
          original_ids.data_ptr<int>(), (ValueType **)(vals_host_ptr), founds_host, generator_args);
      } else if (initializer_args.mode == "constant") {
        using Generator = ConstEmbeddingGenerator;
        auto generator_args = typename Generator::Args {initializer_args.value};
        load_or_initialize_classified_embeddings_kernel<ValueType, int, Generator>
          <<<grid_size, block_size, 0, stream>>>(
          n, dim, reinterpret_cast<ValueType *>(values.data_ptr()), (ValueType **)(vals_dev_ptr), found_counter,
          original_ids.data_ptr<int>(), (ValueType **)(vals_host_ptr), founds_host, generator_args);
      } else {
        throw std::runtime_error("Unrecognized initializer {" + initializer_args.mode + "}");
      }
      DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    });
  });
}

} // namespace dyn_emb

void find_or_insert(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
                  const size_t n,
                  const at::Tensor keys,
                  const at::Tensor values,
                  const std::optional<uint64_t> score = std::nullopt,
                  bool unique_key = true,
                  bool ignore_evict_strategy = false
                  )
{
  if (not score and (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu)) {
    throw std::invalid_argument("Must specify the score when evict strategy is customized or LFU.");
  }
  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  at::Tensor new_tensor = at::empty({static_cast<int64_t>(n)},
                                    at::TensorOptions().dtype(at::kLong).device(values.device()));

  auto new_tensor_data_ptr = reinterpret_cast<void**>(new_tensor.data_ptr<int64_t>());

  at::Tensor found_tensor = at::empty({static_cast<int64_t>(n)},
                                      at::TensorOptions().dtype(at::kBool).device(keys.device()));

  auto found_tensor_data_ptr = found_tensor.data_ptr<bool>();

  if (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu) {
    auto&& option = at::TensorOptions().dtype(at::kUInt64).device(keys.device());
    // broadcast scores
    at::Tensor bc_scores = at::empty({static_cast<int64_t>(n)}, option);
    bc_scores.fill_(score.value());
    table->find_or_insert(n, keys.data_ptr(), new_tensor_data_ptr, values.data_ptr(), found_tensor_data_ptr,
                              bc_scores.data_ptr(), stream, unique_key, ignore_evict_strategy);

  } else {
    table->find_or_insert(n, keys.data_ptr(), new_tensor_data_ptr, values.data_ptr(), found_tensor_data_ptr, nullptr,
                              stream, unique_key, ignore_evict_strategy);
  }
}

void find_or_insert_pointers(
  std::shared_ptr<dyn_emb::DynamicVariableBase> table,
  const size_t n,
  const at::Tensor keys,
  at::Tensor values,
  at::Tensor founds,
  const std::optional<uint64_t> score = std::nullopt,
  bool unique_key = true,
  bool ignore_evict_strategy = false) {
  if (not score and (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu)) {
    throw std::invalid_argument("Must specify the score when evict strategy is customized or LFU.");
  }
  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto values_data_ptr = reinterpret_cast<void**>(values.data_ptr<int64_t>());
  auto found_tensor_data_ptr = founds.data_ptr<bool>();

  if (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu) {
    auto&& option = at::TensorOptions().dtype(at::kUInt64).device(keys.device());
    // broadcast scores
    at::Tensor bc_scores = at::empty({static_cast<int64_t>(n)}, option);
    bc_scores.fill_(score.value());
    table->find_or_insert_pointers(n, keys.data_ptr(), values_data_ptr, found_tensor_data_ptr, 
      bc_scores.data_ptr(), stream, unique_key, ignore_evict_strategy);
  } else {
    table->find_or_insert_pointers(n, keys.data_ptr(), values_data_ptr, found_tensor_data_ptr, 
      nullptr, stream, unique_key, ignore_evict_strategy);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

void find_pointers(
  std::shared_ptr<dyn_emb::DynamicVariableBase> table,
  const size_t n,
  const at::Tensor keys,
  at::Tensor values,
  at::Tensor founds) {

  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto values_data_ptr = reinterpret_cast<void**>(values.data_ptr<int64_t>());
  auto found_tensor_data_ptr = founds.data_ptr<bool>();

  table->find_pointers(n, keys.data_ptr(), values_data_ptr, found_tensor_data_ptr, 
      nullptr, stream);
}

int64_t find_and_get_missed(
  std::shared_ptr<dyn_emb::DynamicVariableBase> ht,
  uint64_t n,
  at::Tensor keys,
  at::Tensor founds,
  at::Tensor vals_ptr,
  at::Tensor missed_keys,
  at::Tensor missed_ids,
  at::Tensor reverse_ids
) {

  find_pointers(ht, n, keys, vals_ptr, founds);
  auto missed_counter = at::zeros({static_cast<int64_t>(1)},
    at::TensorOptions().dtype(at::kLong).device(keys.device()));
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  DISPATCH_INTEGER_DATATYPE_FUNCTION(ht->key_type(), key_t, [&] {
    get_missed_keys_kernel<key_t, int><<<(n + 127) / 128, 128, 0, stream>>>(
      n, founds.data_ptr<bool>(), reinterpret_cast<key_t*>(keys.data_ptr()), 
      reinterpret_cast<key_t*>(missed_keys.data_ptr()), missed_ids.data_ptr<int>(),
      missed_counter.data_ptr<int64_t>(), reverse_ids.data_ptr<int>()
    );
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  int64_t missed_host_counter = 0;
  AT_CUDA_CHECK(cudaMemcpyAsync(&missed_host_counter, missed_counter.data_ptr(),
      sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  return missed_host_counter;
}

void assign(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n,
            const at::Tensor keys, const at::Tensor values,
            const c10::optional<at::Tensor> &score = c10::nullopt,
            bool unique_key = true) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (score.has_value()) {
    at::Tensor score_ = score.value();
    table->assign(n, keys.data_ptr(), values.data_ptr(), score_.data_ptr(),
                  stream, unique_key);
  } else {
    table->assign(n, keys.data_ptr(), values.data_ptr(), nullptr, stream,
                  unique_key);
  }
}

void find(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n,
          const at::Tensor keys, const at::Tensor values,
          const at::Tensor founds,
          const c10::optional<at::Tensor> &score = c10::nullopt) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (score.has_value()) {
    at::Tensor score_ = score.value();
    table->find(n, keys.data_ptr(), values.data_ptr(), founds.data_ptr<bool>(),
                score_.data_ptr(), stream);
  } else {
    table->find(n, keys.data_ptr(), values.data_ptr(), founds.data_ptr<bool>(),
                nullptr, stream);
  }
}

void erase(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n,
           const at::Tensor keys) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  table->erase(n, keys.data_ptr(), stream);
}

void clear(std::shared_ptr<dyn_emb::DynamicVariableBase> table) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  table->clear(stream);
}

void reserve(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
             const size_t new_capacity) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  table->reserve(new_capacity, stream);
}

void export_batch(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
                  const size_t n, const size_t offset,
                  const at::Tensor d_counter, const at::Tensor keys,
                  const at::Tensor values,
                  const c10::optional<at::Tensor> &score = c10::nullopt) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (score.has_value()) {
    at::Tensor score_ = score.value();
    table->export_batch(n, offset, d_counter.data_ptr<size_t>(),
                        keys.data_ptr(), values.data_ptr(), score_.data_ptr(),
                        stream);
  } else {
    table->export_batch(n, offset, d_counter.data_ptr<size_t>(),
                        keys.data_ptr(), values.data_ptr(), nullptr, stream);
  }
}

void count_matched(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    const uint64_t threshold,
    at::Tensor num_matched) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  table->count_matched(threshold, reinterpret_cast<uint64_t*>(num_matched.data_ptr()), stream);
}

void export_batch_matched(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    const uint64_t threshold,
    const uint64_t n,
    const uint64_t offset,
    at::Tensor num_matched,
    at::Tensor keys,
    at::Tensor values) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  table->export_batch_matched(
    threshold, n, offset, reinterpret_cast<uint64_t*>(num_matched.data_ptr()), 
    keys.data_ptr(), values.data_ptr(), nullptr, stream);
}

void set_table_offset_async(
    at::Tensor h_table_offsets,
    at::Tensor offsets, 
    int table_num, 
    int batch_size,
    const std::vector<int> &table_offsets_in_feature
  ) {

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  for (int i = 0; i < table_num + 1; i++) {
    // the last one is boundary
    int table_offset_begin = table_offsets_in_feature[i];
    table_offset_begin = table_offset_begin * batch_size;
    auto offset_type = scalartype_to_datatype(offsets.dtype().toScalarType());
    DISPATCH_OFFSET_INT_TYPE(offset_type, offset_t, [&] {
      AT_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<offset_t*>(h_table_offsets.data_ptr()) + i,
                                    reinterpret_cast<offset_t*>(offsets.data_ptr()) + table_offset_begin,
                                    offsets.element_size(), cudaMemcpyDeviceToHost, stream));
    });
  }
}

template <typename T>
__global__ void load_embedding_kernel_vec4(
    int batch,
    int emb_dim,
    T* __restrict__ outputs,
    T* const * __restrict__ src_ptrs) {
  
  constexpr int kWarpSize = 32;
  constexpr int VecSize = 4;
  const int warp_num_per_block = blockDim.x / kWarpSize;
  const int warp_id_in_block = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;

  Vec4T<T> emb;
  for (int emb_id = warp_num_per_block * blockIdx.x + warp_id_in_block;
      emb_id < batch; emb_id += gridDim.x * warp_num_per_block) {
    T* const src_ptr = src_ptrs[emb_id];
    T* dst_ptr = outputs + emb_id * emb_dim;
    if (src_ptr != nullptr) {
      for (int i = 0; VecSize * (kWarpSize * i + lane_id) < emb_dim; ++i) {
        int idx4 = VecSize * (kWarpSize * i + lane_id);
        emb.load(src_ptr + idx4);
        emb.store(dst_ptr + idx4);
      }
    }
  }
}

template <typename T>
__global__ void load_embedding_kernel(
    int batch,
    int emb_dim,
    T* __restrict__ outputs,
    T* const * __restrict__ src_ptrs) {

  for (int emb_id = blockIdx.x; emb_id < batch; emb_id += gridDim.x) {
    T* const src_ptr = src_ptrs[emb_id];
    T* dst_ptr = outputs + emb_id * emb_dim;
    if (src_ptr != nullptr) {
      for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
        dst_ptr[i] = src_ptr[i];
      }
    }
  }
}

// name embedding_lookup, because don't return optimizer states
void cache_embedding_lookup(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    int num_total,
    at::Tensor const& keys,
    at::Tensor const & embs,
    at::Tensor& found_keys,
    at::Tensor& num_found,
    at::Tensor& missing_keys,
    at::Tensor& missing_keys_idx
) {
  if (num_total == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  at::Tensor vals_dev_ptr_tensor = at::empty({static_cast<int64_t>(num_total)}, 
    at::TensorOptions().dtype(at::kLong).device(keys.device()));
  auto vals_dev_ptr = reinterpret_cast<void**>(vals_dev_ptr_tensor.data_ptr<int64_t>());
  at::Tensor founds_dev_tensor = at::empty({static_cast<int64_t>(num_total)},
     at::TensorOptions().dtype(at::kBool).device(keys.device()));
  auto founds_dev = founds_dev_tensor.data_ptr<bool>();

  table->find_pointers(num_total, keys.data_ptr(), vals_dev_ptr, founds_dev, nullptr, stream);

  constexpr int kWarpSize = 32;
  constexpr int MULTIPLIER = 4;
  constexpr int BLOCK_SIZE_VEC = 64;
  constexpr int WARP_PER_BLOCK = BLOCK_SIZE_VEC / kWarpSize;
  auto &device_prop = DeviceProp::getDeviceProp();
  const int max_grid_size =
      device_prop.num_sms *
      (device_prop.max_thread_per_sm / BLOCK_SIZE_VEC);
  
  int grid_size = 0;
  if (num_total / WARP_PER_BLOCK < max_grid_size) {
    grid_size = (num_total - 1) / WARP_PER_BLOCK + 1;
  } else if (num_total / WARP_PER_BLOCK > max_grid_size * MULTIPLIER) {
    grid_size = max_grid_size * MULTIPLIER;
  } else {
    grid_size = max_grid_size;
  }

  auto value_type = table->value_type();
  DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, ValueType, [&] {
    int dim = table->cols();
    if (dim % 4 == 0) {
      load_embedding_kernel_vec4<ValueType>
        <<<grid_size, BLOCK_SIZE_VEC, 0, stream>>>(
        num_total, dim, reinterpret_cast<ValueType*>(embs.data_ptr()), 
        reinterpret_cast<ValueType**>(vals_dev_ptr_tensor.data_ptr()));
    } else {
      int block_size = dim < device_prop.max_thread_per_block
                          ? dim
                          : device_prop.max_thread_per_block;
      int grid_size = num_total;
      load_embedding_kernel<ValueType>
        <<<grid_size, block_size, 0, stream>>>(
        num_total, dim, reinterpret_cast<ValueType*>(embs.data_ptr()), 
        reinterpret_cast<ValueType**>(vals_dev_ptr_tensor.data_ptr()));
    }
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

  /// select results using cub.
  auto key_type = table->key_type();
  at::Tensor inv_founds = at::logical_not(founds_dev_tensor);
  at::Tensor select_res = at::empty({2}, keys.options().dtype(at::kInt));
  int* d_num_select = select_res.data_ptr<int>();
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, KeyType, [&] {
    select_async<KeyType>(num_total, founds_dev, reinterpret_cast<KeyType*>(keys.data_ptr()),
      reinterpret_cast<KeyType*>(found_keys.data_ptr()), num_found.data_ptr<int>(), keys.device(), stream);
    select_async<KeyType>(num_total, inv_founds.data_ptr<bool>(), reinterpret_cast<KeyType*>(keys.data_ptr()),
      reinterpret_cast<KeyType*>(missing_keys.data_ptr()), d_num_select, keys.device(), stream);
  });
  
  select_index_async(num_total, inv_founds.data_ptr<bool>(), missing_keys_idx.data_ptr<int>(), 
                     d_num_select+1, keys.device(), stream);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <
  typename T,
  typename EmbeddingGenerator>
__global__ void load_or_initialize_embeddings_twice_kernel(
    int n,
    int value_dim,
    int emb_dim,
    T* __restrict__ out_values,
    T* __restrict__ out_embs,
    int const * __restrict__ out_embs_idx,
    T* const * __restrict__ in_values_ptr,
    typename EmbeddingGenerator::Args generator_args,
    int const * __restrict__ reverse_indices
  ) {

  EmbeddingGenerator emb_gen(generator_args);

  for (int emb_id = blockIdx.x; emb_id < n; emb_id += gridDim.x) {
    T* const in_value_ptr = in_values_ptr[emb_id];
    int reverse_idx = reverse_indices[emb_id];
    int out_emb_idx = out_embs_idx[reverse_idx];
    T* out_value_ptr = out_values + reverse_idx * value_dim;  
    T* out_emb_ptr = out_embs + out_emb_idx * emb_dim;

    if (in_value_ptr) { // copy embedding from inputs.
      for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
        T tmp = in_value_ptr[i];
        out_value_ptr[i] = tmp;
        out_emb_ptr[i] = tmp;
      }
    } else { // initialize the embedding directly.
      for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
        auto tmp = TypeConvertFunc<T, float>::convert(emb_gen.generate(emb_id));
        out_value_ptr[i] = tmp;
        out_emb_ptr[i] = tmp;
      }
    }
  }

  emb_gen.destroy();
}

template <typename T>
__global__ void load_storage_emb_kernel(
    int n,
    int value_dim,
    int emb_dim,
    T* const * __restrict__ in_values_ptr,
    T* __restrict__ out_values,
    T* __restrict__ out_embs,
    int const * __restrict__ out_vals_idx,
    int const * __restrict__ out_embs_idx) {

  for (int emb_id = blockIdx.x; emb_id < n; emb_id += gridDim.x) {
    T* const in_value_ptr = in_values_ptr[emb_id];
    int out_val_idx = out_vals_idx[emb_id];
    int out_emb_idx = out_embs_idx[out_val_idx];
    T* out_value_ptr = out_values + out_val_idx * value_dim;  
    T* out_emb_ptr = out_embs + out_emb_idx * emb_dim;

    // copy embedding from storage
    for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
      T tmp = in_value_ptr[i];
      out_value_ptr[i] = tmp;
      out_emb_ptr[i] = tmp;
    }
  }
}

template <
  typename T,
  typename OptStateInitializer>
__global__ void load_or_initialize_optimizer_state_kernel_vec4(
    int n,
    int emb_dim,
    T* __restrict__ out_values,
    T* const * __restrict__ in_values_ptr,
    OptStateInitializer optstate_initailizer,
    int const * __restrict__ reverse_indices
  ) {
  
  constexpr int kWarpSize = 32;
  constexpr int VecSize = 4;
  const int warp_num_per_block = blockDim.x / kWarpSize;
  const int warp_id_in_block = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;

  Vec4T<T> optim_state;
  for (int emb_id = warp_num_per_block * blockIdx.x + warp_id_in_block;
      emb_id < n; emb_id += gridDim.x * warp_num_per_block) {
    T* const in_value_ptr = in_values_ptr[emb_id];
    int reverse_idx = reverse_indices[emb_id];
    T* out_value_ptr = out_values + (emb_dim + optstate_initailizer.dim) * reverse_idx;
    if (in_value_ptr) {
      for (int i = 0; VecSize * (kWarpSize * i + lane_id) < optstate_initailizer.dim; ++i) {
        int idx4 = VecSize * (kWarpSize * i + lane_id);
        optim_state.load(in_value_ptr + emb_dim + idx4);
        optim_state.store(out_value_ptr + emb_dim + idx4);
      }
    } else {
      optstate_initailizer.init4(out_value_ptr + emb_dim);
    }
  }
}

template <
  typename T,
  typename OptStateInitializer>
__global__ void load_or_initialize_optimizer_state_kernel(
    int n,
    int emb_dim,
    T* __restrict__ out_values,
    T* const * __restrict__ in_values_ptr,
    OptStateInitializer optstate_initailizer,
    int const * __restrict__ reverse_indices
) {
  
  for (int emb_id = blockIdx.x; emb_id < n; emb_id += gridDim.x) {
    T* const in_value_ptr = in_values_ptr[emb_id];
    int reverse_idx = reverse_indices[emb_id];
    T* out_value_ptr = out_values + (emb_dim + optstate_initailizer.dim) * reverse_idx;
    if (in_value_ptr) {
      for (int i = threadIdx.x; i < optstate_initailizer.dim; i += blockDim.x) {
        out_value_ptr[emb_dim + i] = in_value_ptr[emb_dim + i];
      }
    } else {
      optstate_initailizer.init(out_value_ptr + emb_dim);
    }
  }
}

__global__ void update_cache_missed_score_for_lfu(
  int n, uint64_t* __restrict__ scores, bool const * __restrict__ founds
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    bool found = founds[tid];
    uint64_t old_score = scores[tid];
    if (found) {
      scores[tid] = old_score + 1;
    } else {
      scores[tid] = 1;
    }
  }
}


void storage_find_and_initialize(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    int const num_total,
    at::Tensor const& keys,
    at::Tensor& values,
    at::Tensor& scores,
    at::Tensor& embs,
    at::Tensor const& emb_idx,
    const c10::optional<at::Tensor>& cache_metrics
) {

  if (num_total == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto vals_ptr_tensor = at::empty({static_cast<int64_t>(num_total)}, keys.options().dtype(at::kLong));
  auto vals_ptr = reinterpret_cast<void**>(vals_ptr_tensor.data_ptr<int64_t>());
  auto founds_tensor = at::empty({static_cast<int64_t>(num_total)}, keys.options().dtype(at::kBool));
  auto founds_ptr = founds_tensor.data_ptr<bool>();
  ///////////////////////
  void* score_ptr = nullptr;
  // bool is_lfu = table->evict_strategy() == EvictStrategy::kLfu;
  // if (is_lfu) {
  //   score_ptr = scores.data_ptr();
  // }
  // for customized strategy, just use the new score.
  table->find_pointers(num_total, keys.data_ptr(), vals_ptr, founds_ptr, score_ptr, stream);
  // if (is_lfu) {
  //   update_cache_missed_score_for_lfu<<<(num_total + 63)/64, 64, 0, stream>>>(num_total, reinterpret_cast<uint64_t*>(score_ptr), founds_ptr);
  // }
  ///////////////////////

  /// select results using cub.
  // auto value_type = table->value_type();
  // at::Tensor inv_founds = at::logical_not(founds_tensor);
  // auto found_val_ptrs = at::empty_like(vals_ptr_tensor);
  // auto missing_val_ptrs = at::empty_like(vals_ptr_tensor);
  // at::Tensor select_res = at::empty({3}, keys.options().dtype(at::kInt));
  // int* d_num_select = select_res.data_ptr<int>();
  // auto num_found = at::zeros({static_cast<int64_t>(1)}, keys.options().dtype(at::kInt));
  // DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, ValueType, [&] {
  //   select_async<ValueType*>(num_total, founds_ptr, reinterpret_cast<ValueType**>(vals_ptr_tensor.data_ptr()),
  //     reinterpret_cast<ValueType**>(found_val_ptrs.data_ptr()), num_found.data_ptr<int>(), keys.device(), stream);
  //   select_async<ValueType*>(num_total, inv_founds.data_ptr<bool>(), reinterpret_cast<ValueType**>(vals_ptr_tensor.data_ptr()),
  //     reinterpret_cast<ValueType**>(missing_val_ptrs.data_ptr()), d_num_select, keys.device(), stream);
  // });
  // at::Tensor found_keys_idx = at::empty({num_total}, keys.options().dtype(at::kInt));
  // at::Tensor missing_keys_idx = at::empty({num_total}, keys.options().dtype(at::kInt));
  // select_index_async(num_total, founds_ptr, found_keys_idx.data_ptr<int>(), 
  //                   d_num_select + 1, keys.device(), stream);
  // select_index_async(num_total, inv_founds.data_ptr<bool>(), missing_keys_idx.data_ptr<int>(), 
  //                   d_num_select + 2, keys.device(), stream);
  // int h_num_found = 0;
  // AT_CUDA_CHECK(cudaMemcpyAsync(&h_num_found, num_found.data_ptr(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  // AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  auto value_type = table->value_type();
  auto sorted_val_ptrs = at::empty_like(vals_ptr_tensor);
  at::Tensor reverse_keys_idx = at::empty({num_total}, keys.options().dtype(at::kInt));
  DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, ValueType, [&] {
    sort_async<uint64_t>(num_total, (uint64_t*)vals_ptr_tensor.data_ptr(), //reinterpret_cast<ValueType**>(vals_ptr_tensor.data_ptr()),
                           (uint64_t*)sorted_val_ptrs.data_ptr(),/*reinterpret_cast<ValueType**>(sorted_val_ptrs.data_ptr())*/ reverse_keys_idx.data_ptr<int>(),
                           keys.device(), stream);
  });
  if (cache_metrics.has_value()) {
    auto num_found = at::zeros({static_cast<int64_t>(1)}, keys.options().dtype(at::kInt));
    at::Tensor found_keys_idx = at::empty({num_total}, keys.options().dtype(at::kInt));
    select_index_async(num_total, founds_ptr, found_keys_idx.data_ptr<int>(), 
                      num_found.data_ptr<int>(), keys.device(), stream);
    int h_num_found = 0;
    AT_CUDA_CHECK(cudaMemcpyAsync(&h_num_found, num_found.data_ptr(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    cache_metrics.value()[2] = h_num_found;
  }

  int emb_dim = table->cols();
  int value_dim = emb_dim + table->optstate_dim();
  auto &device_prop = DeviceProp::getDeviceProp();
  int block_size = emb_dim < device_prop.max_thread_per_block
                       ? emb_dim
                       : device_prop.max_thread_per_block;
  int grid_size = device_prop.num_sms * (device_prop.max_thread_per_sm / block_size);

  // initialize value'emb, unique emb
  // DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, ValueType, [&] {
  //   load_storage_emb_kernel<ValueType><<<grid_size, block_size, 0, stream>>>(
  //     h_num_found, value_dim, emb_dim, reinterpret_cast<ValueType **>(found_val_ptrs.data_ptr()),
  //     reinterpret_cast<ValueType *>(values.data_ptr()), reinterpret_cast<ValueType *>(embs.data_ptr()), 
  //     found_keys_idx.data_ptr<int>(), emb_idx.data_ptr<int>()
  //   );
  // });
  auto &initializer_args = table->get_initializer_args();
  auto* curand_states_ = table->get_curand_states();
  auto key_type = table->key_type();
  DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, ValueType, [&] {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, KeyType, [&] {
      if (initializer_args.mode == "normal") {
        using Generator = NormalEmbeddingGenerator;
        auto generator_args = typename Generator::Args {curand_states_, initializer_args.mean, initializer_args.std_dev};
        load_or_initialize_embeddings_twice_kernel<ValueType, Generator>
          <<<grid_size, block_size, 0, stream>>>(
          num_total, value_dim, emb_dim, reinterpret_cast<ValueType *>(values.data_ptr()), 
          reinterpret_cast<ValueType *>(embs.data_ptr()), emb_idx.data_ptr<int>(),
          reinterpret_cast<ValueType **>(sorted_val_ptrs.data_ptr()), generator_args, reverse_keys_idx.data_ptr<int>());
      } else if (initializer_args.mode == "truncated_normal") {
        using Generator = TruncatedNormalEmbeddingGenerator;
        auto generator_args = typename Generator::Args {curand_states_, initializer_args.mean, initializer_args.std_dev, initializer_args.lower, initializer_args.upper};
        load_or_initialize_embeddings_twice_kernel<ValueType, Generator>
          <<<grid_size, block_size, 0, stream>>>(
          num_total, value_dim, emb_dim, reinterpret_cast<ValueType *>(values.data_ptr()), 
          reinterpret_cast<ValueType *>(embs.data_ptr()), emb_idx.data_ptr<int>(),
          reinterpret_cast<ValueType **>(sorted_val_ptrs.data_ptr()), generator_args, reverse_keys_idx.data_ptr<int>());
      } else if (initializer_args.mode == "uniform") {
        using Generator = UniformEmbeddingGenerator;
        auto generator_args = typename Generator::Args {curand_states_, initializer_args.lower, initializer_args.upper};
        load_or_initialize_embeddings_twice_kernel<ValueType, Generator>
          <<<grid_size, block_size, 0, stream>>>(
          num_total, value_dim, emb_dim, reinterpret_cast<ValueType *>(values.data_ptr()), 
          reinterpret_cast<ValueType *>(embs.data_ptr()), emb_idx.data_ptr<int>(),
          reinterpret_cast<ValueType **>(sorted_val_ptrs.data_ptr()), generator_args, reverse_keys_idx.data_ptr<int>());
      } else if (initializer_args.mode == "debug") {
        using Generator = MappingEmbeddingGenerator<KeyType>;
        auto generator_args = typename Generator::Args {reinterpret_cast<const KeyType *>(keys.data_ptr()), 100000};
        load_or_initialize_embeddings_twice_kernel<ValueType, Generator>
          <<<grid_size, block_size, 0, stream>>>(
          num_total, value_dim, emb_dim, reinterpret_cast<ValueType *>(values.data_ptr()), 
          reinterpret_cast<ValueType *>(embs.data_ptr()), emb_idx.data_ptr<int>(),
          reinterpret_cast<ValueType **>(sorted_val_ptrs.data_ptr()), generator_args, reverse_keys_idx.data_ptr<int>());
      } else if (initializer_args.mode == "constant") {
        using Generator = ConstEmbeddingGenerator;
        auto generator_args = typename Generator::Args {initializer_args.value};
        load_or_initialize_embeddings_twice_kernel<ValueType, Generator>
          <<<grid_size, block_size, 0, stream>>>(
          num_total, value_dim, emb_dim, reinterpret_cast<ValueType *>(values.data_ptr()), 
          reinterpret_cast<ValueType *>(embs.data_ptr()), emb_idx.data_ptr<int>(),
          reinterpret_cast<ValueType **>(sorted_val_ptrs.data_ptr()), generator_args, reverse_keys_idx.data_ptr<int>());
      } else {
        throw std::runtime_error("Unrecognized initializer {" + initializer_args.mode + "}");
      }
      DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    });
  });

  // initialize value's optim states
  int optim_state_dim = value_dim - emb_dim;
  if (optim_state_dim == 0) return;

  constexpr int kWarpSize = 32;
  constexpr int MULTIPLIER = 4;
  constexpr int BLOCK_SIZE_VEC = 64;
  constexpr int WARP_PER_BLOCK = BLOCK_SIZE_VEC / kWarpSize;
  const int max_grid_size =
      device_prop.num_sms *
      (device_prop.max_thread_per_sm / BLOCK_SIZE_VEC);
  
  int grid_size_opt = 0;
  if (num_total / WARP_PER_BLOCK < max_grid_size) {
    grid_size_opt = (num_total - 1) / WARP_PER_BLOCK + 1;
  } else if (num_total / WARP_PER_BLOCK > max_grid_size * MULTIPLIER) {
    grid_size_opt = max_grid_size * MULTIPLIER;
  } else {
    grid_size_opt = max_grid_size;
  }

  float initial_optim_state = table->get_initial_optstate();
  DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, ValueType, [&] {
    using OptStateInitializer = OptStateInitializer<ValueType, int>;
    OptStateInitializer optim_state_initializer {optim_state_dim, initial_optim_state};
    if (emb_dim % 4 == 0 and optim_state_dim % 4 == 0) {
      load_or_initialize_optimizer_state_kernel_vec4<ValueType, OptStateInitializer>
        <<<grid_size_opt, BLOCK_SIZE_VEC, 0, stream>>>(
        num_total, emb_dim, reinterpret_cast<ValueType *>(values.data_ptr()),
        reinterpret_cast<ValueType **>(sorted_val_ptrs.data_ptr()), optim_state_initializer, reverse_keys_idx.data_ptr<int>());
    } else {
      int block_size = optim_state_dim < device_prop.max_thread_per_block
                          ? optim_state_dim
                          : device_prop.max_thread_per_block;
      int grid_size = num_total;
      load_or_initialize_optimizer_state_kernel<ValueType, OptStateInitializer>
        <<<grid_size, block_size, 0, stream>>>(
        num_total, emb_dim, reinterpret_cast<ValueType *>(values.data_ptr()),
        reinterpret_cast<ValueType **>(sorted_val_ptrs.data_ptr()), optim_state_initializer, reverse_keys_idx.data_ptr<int>());
    }
  });

  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

}

void cache_lock(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    int64_t num_total,
    at::Tensor const& keys,
    at::Tensor& locked_ptr,
    std::optional<uint64_t> const score
) {
  at::Tensor success = at::empty({num_total}, keys.options().dtype(at::kBool));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if ((table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu)) {
    if (not score.has_value()) {
      throw std::invalid_argument("Must specify the score when evict strategy is customized or LFU.");
    }
    auto&& option = at::TensorOptions().dtype(at::kUInt64).device(keys.device());
    at::Tensor bc_scores = at::empty({static_cast<int64_t>(num_total)}, option);
    bc_scores.fill_(score.value());
    table->lock(num_total, keys.data_ptr(), reinterpret_cast<void**>(locked_ptr.data_ptr()), 
                success.data_ptr<bool>(), bc_scores.data_ptr(), stream);
  } else {
    table->lock(num_total, keys.data_ptr(), reinterpret_cast<void**>(locked_ptr.data_ptr()), 
                success.data_ptr<bool>(), nullptr, stream);
  }
}

void cache_unlock(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    int64_t num_total,
    at::Tensor const& keys,
    at::Tensor& locked_ptr
) {
  at::Tensor success = at::empty({num_total}, keys.options().dtype(at::kBool));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  table->unlock(num_total, reinterpret_cast<void**>(locked_ptr.data_ptr()), keys.data_ptr(), success.data_ptr<bool>(), stream);
}

void cache_insert_and_evict(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    int64_t num_total,
    at::Tensor const& keys,
    at::Tensor const& values,
    std::optional<uint64_t> const score,
    c10::optional<at::Tensor> const &scores,
    at::Tensor& evicted_keys,
    at::Tensor& evicted_values,
    at::Tensor& evicted_scores,
    at::Tensor& num_evicted
) {
  if (num_total == 0) return;
  if (not (score.has_value() ^ scores.has_value())) {
    throw std::runtime_error("To provide only one score in uint64_t or at::Tensor");
  }
  if (score.has_value()) {
    insert_and_evict(table, num_total, keys, values, score, evicted_keys, evicted_values, evicted_scores, num_evicted);
  } else {
    insert_and_evict_(table, num_total, keys, values, scores, evicted_keys, evicted_values, evicted_scores, num_evicted);
  }
}

void storage_insert(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    int64_t num_total, 
    at::Tensor const& keys, 
    at::Tensor const& values,
    at::Tensor const& scores,
    const c10::optional<at::Tensor>& cache_metrics,
    std::optional<uint64_t> const score
) {
  if (cache_metrics.has_value()) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    at::Tensor evicted_keys = at::empty({num_total}, keys.options());
    at::Tensor evicted_values = at::empty_like(values);
    at::Tensor evicted_scores = at::empty({num_total}, keys.options().dtype(at::kUInt64));
    at::Tensor num_evicted = at::zeros({static_cast<int64_t>(1)}, keys.options().dtype(at::kUInt64));
    // CUDA_CHECK(cudaDeviceSynchronize());
    // std::cout << "Jiashu " << __FILE__ << " " << __LINE__ << "\n";
    ///make score to 0 to avoid score unexpected grow.
    ///:it works for LRU and LFU, but not for customized.
    insert_and_evict_(table, num_total, keys, values, c10::make_optional<at::Tensor>(scores), evicted_keys, evicted_values, evicted_scores, num_evicted);
    // CUDA_CHECK(cudaDeviceSynchronize());
    // std::cout << "Jiashu " << __FILE__ << " " << __LINE__ << "\n";
    uint64_t h_num_evicted = 0;
    AT_CUDA_CHECK(cudaMemcpyAsync(&h_num_evicted, num_evicted.data_ptr(), sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    cache_metrics.value()[3] = h_num_evicted;
  }  else {
    //update score only.
    insert_or_assign(table, num_total, keys, values, scores);
  }
}

void storage_find_or_insert_with_initialize(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    int64_t num_total, 
    at::Tensor const& keys,
    at::Tensor& embs,
    std::optional<uint64_t> const score
) {
  find_or_insert(table, num_total, keys, embs, score);
}

void cache_storage_find_or_insert_with_initialize(
    std::shared_ptr<dyn_emb::DynamicVariableBase> cache,
    std::shared_ptr<dyn_emb::DynamicVariableBase> storage,
    int64_t total_num,
    at::Tensor const & keys,
    at::Tensor & embs,
    std::optional<uint64_t> const score,
    const c10::optional<at::Tensor>& cache_metrics
) {
  if (total_num == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  at::Tensor found_keys = at::empty({total_num}, keys.options());
  at::Tensor num_found = at::empty({1}, keys.options().dtype(at::kInt));
  at::Tensor missing_keys = at::empty({total_num}, keys.options());
  at::Tensor missing_keys_idx = at::empty({total_num}, keys.options().dtype(at::kInt));
  cache_embedding_lookup(cache, total_num, keys, embs, 
                          found_keys, num_found, missing_keys, missing_keys_idx);
  
  int h_num_found = 0;
  AT_CUDA_CHECK(cudaMemcpyAsync(&h_num_found, num_found.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  if (cache_metrics.has_value()) {
    cache_metrics.value()[0] = static_cast<int>(total_num);
    cache_metrics.value()[1] = static_cast<int>(h_num_found);
  }
  if (h_num_found == total_num) return;
  int64_t h_num_missing = total_num - h_num_found;
  int64_t value_dim = storage->cols() + storage->optstate_dim();
  at::Tensor missing_values = at::empty({h_num_missing, value_dim}, embs.options());
  at::Tensor missing_scores = at::empty({h_num_missing}, keys.options().dtype(at::kUInt64));
  storage_find_and_initialize(storage, h_num_missing, missing_keys, missing_values, missing_scores, embs, missing_keys_idx, cache_metrics);
  
  if (std::getenv("DISABLE_UPDATE_CACHE") != nullptr) {
    return;
  }
  
  at::Tensor locked_ptr = at::empty({h_num_found}, found_keys.options().dtype(at::kLong));
  cache_lock(cache, h_num_found, found_keys, locked_ptr, score);
  at::Tensor evicted_keys = at::empty({h_num_missing}, keys.options());
  at::Tensor evicted_values = at::empty({h_num_missing, value_dim}, keys.options());
  at::Tensor evicted_scores = at::empty({h_num_missing}, keys.options().dtype(at::kUInt64));
  at::Tensor num_evicted = at::zeros({static_cast<int64_t>(1)}, keys.options().dtype(at::kUInt64));
  ///////////////////////
  // if (cache->evict_strategy() == EvictStrategy::kLfu) {
  //   cache_insert_and_evict(cache, h_num_missing, missing_keys, missing_values, std::nullopt, c10::make_optional<at::Tensor>(missing_scores), 
  //                           evicted_keys, evicted_values, evicted_scores, num_evicted);
  // } else {
  cache_insert_and_evict(cache, h_num_missing, missing_keys, missing_values, score, c10::nullopt,
                          evicted_keys, evicted_values, evicted_scores, num_evicted);
  // }
  ///////////////////////
  cache_unlock(cache, h_num_found, found_keys, locked_ptr);
  uint64_t h_num_evicted = 0;
  AT_CUDA_CHECK(cudaMemcpyAsync(&h_num_evicted, num_evicted.data_ptr(), sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  storage_insert(storage, h_num_evicted, evicted_keys, evicted_values, evicted_scores, cache_metrics, score);
}

void lookup_forward_dense(
    std::vector<std::shared_ptr<dyn_emb::DynamicVariableBase>> tables,
    const at::Tensor indices, const at::Tensor offsets, const py::list scores,
    const std::vector<int> &table_offsets_in_feature, at::Tensor table_offsets,
    int table_num, int batch_size, int dim, bool use_index_dedup,
    const at::Tensor unique_idx, const at::Tensor reverse_idx,
    const at::Tensor h_unique_nums, const at::Tensor d_unique_nums,
    const at::Tensor h_unique_offsets, const at::Tensor d_unique_offsets,
    const at::Tensor unique_embs, const at::Tensor output_embs,
    int device_num_sms, std::shared_ptr<dyn_emb::UniqueOpBase> unique_op,
    const c10::optional<at::Tensor>& cache_metrics=c10::nullopt,
    std::optional<std::vector<std::shared_ptr<dyn_emb::DynamicVariableBase>>> host_tables = std::nullopt) {

  if (!offsets.is_cuda() || !indices.is_cuda()) {
    throw std::runtime_error(
        "offsets or indices tensor must be on CUDA device");
  }

  // Check dtype of h_unique_nums and d_unique_nums
  if (h_unique_nums.scalar_type() != at::kUInt64 ||
      d_unique_nums.scalar_type() != at::kUInt64) {
    throw std::runtime_error(
        "h_unique_nums and d_unique_nums must have dtype uint64_t");
  }

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t indices_shape = indices.size(0);
  auto unique_num_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(d_unique_nums.dtype()));
  auto unique_offset_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(d_unique_offsets.dtype()));

  auto options = at::TensorOptions().dtype(offsets.dtype()).device(at::kCPU);
  at::Tensor h_table_offsets = at::empty({static_cast<int64_t>(table_num + 1)}, options);
  set_table_offset_async(h_table_offsets, offsets, table_num, batch_size, table_offsets_in_feature);

  //1.unique
  size_t unique_op_capacity = unique_op->get_capacity();
  if (indices_shape * 2 > unique_op_capacity) {
    at::Tensor new_keys = at::empty({indices_shape * 2}, indices.options());
    at::Tensor new_vals = at::empty(
        {indices_shape * 2},
        at::TensorOptions().dtype(at::kUInt64).device(indices.device()));
    unique_op->reset_capacity(new_keys, new_vals, indices_shape * 2, stream);
  }

  std::vector<at::Tensor> tmp_unique_indices(table_num);
  for (int i = 0; i < table_num; ++i) {
    tmp_unique_indices[i] = at::empty_like(indices);
  }

  // sync for the table offsets.
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  for (int i = 0; i < table_num; ++i) {
    ///TODO: maybe not int64_t
    int64_t indices_begin = h_table_offsets[i].item<int64_t>();
    int64_t indices_end = h_table_offsets[i+1].item<int64_t>();
    int64_t indices_length = indices_end - indices_begin;

    if (indices_length == 0) {
      DEMB_CUDA_CHECK(cudaMemsetAsync(
          reinterpret_cast<uint64_t *>(d_unique_nums.data_ptr()) + i, 0,
          sizeof(uint64_t), stream));
      dyn_emb::add_offset(d_unique_nums.data_ptr(), d_unique_offsets.data_ptr(),
                          i, unique_num_type, unique_offset_type, stream);
    } else {
      at::Tensor tmp_indices = create_sub_tensor(indices, indices_begin);
      at::Tensor tmp_reverse_idx =
          create_sub_tensor(reverse_idx, indices_begin);
      at::Tensor tmp_d_unique_num = create_sub_tensor(d_unique_nums, i);

      at::Tensor previous_d_unique_num = create_sub_tensor(d_unique_offsets, i);
      unique_op->unique(tmp_indices, indices_length, tmp_reverse_idx,
                        tmp_unique_indices[i], tmp_d_unique_num, stream,
                        previous_d_unique_num);
      dyn_emb::add_offset(d_unique_nums.data_ptr(), d_unique_offsets.data_ptr(),
                          i, unique_num_type, unique_offset_type, stream);
    }
  }

  AT_CUDA_CHECK(
      cudaMemcpyAsync(h_unique_nums.data_ptr(), d_unique_nums.data_ptr(),
                      d_unique_nums.numel() * d_unique_nums.element_size(),
                      cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(
      h_unique_offsets.data_ptr(), d_unique_offsets.data_ptr(),
      (d_unique_nums.numel() + 1) * d_unique_nums.element_size(),
      cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  if (not use_index_dedup) {
    AT_CUDA_CHECK(cudaMemcpyAsync(table_offsets.data_ptr(), h_table_offsets.data_ptr(),
      table_offsets.numel() * table_offsets.element_size(),cudaMemcpyHostToDevice, stream));
  }


  // 2. lookup and initialize per table.
  int64_t unique_embs_offset = 0;
  for (int i = 0; i < table_num; ++i) {

    int64_t indices_begin = h_table_offsets[i].item<int64_t>();
    int64_t indices_end = h_table_offsets[i+1].item<int64_t>();
    int64_t indices_length = indices_end - indices_begin;
    int64_t tmp_unique_num = h_unique_nums[i].item<int64_t>();
    if (tmp_unique_num == 0) {
      continue;
    }
    at::Tensor tmp_unique_embs = create_sub_tensor(unique_embs, unique_embs_offset * dim);
    auto score = std::make_optional<uint64_t>(py::cast<uint64_t>(scores[i]));
    std::shared_ptr<dyn_emb::DynamicVariableBase> host_table;
    if (host_tables.has_value()) {
      host_table = host_tables.value()[i];
    }
    // CUDA_CHECK(cudaDeviceSynchronize());
    // std::cout << "Jiashu " << __FILE__ << " " << __LINE__ << "\n";
    if (host_table != nullptr) { // cache + storage.
      // CUDA_CHECK(cudaDeviceSynchronize());
      // std::cout << "Jiashu " << __FILE__ << " " << __LINE__ << "\n";
      cache_storage_find_or_insert_with_initialize(tables[i], host_table, tmp_unique_num, 
        tmp_unique_indices[i], tmp_unique_embs, score, cache_metrics);
      // CUDA_CHECK(cudaDeviceSynchronize());
      // std::cout << "Jiashu " << __FILE__ << " " << __LINE__ << "\n";
    } else { // single storage.
      // CUDA_CHECK(cudaDeviceSynchronize());
      // std::cout << "Jiashu " << __FILE__ << " " << __LINE__ << "\n";
      storage_find_or_insert_with_initialize(tables[i], tmp_unique_num, 
        tmp_unique_indices[i], tmp_unique_embs, score);
      // CUDA_CHECK(cudaDeviceSynchronize());
      // std::cout << "Jiashu " << __FILE__ << " " << __LINE__ << "\n";
    }
    if (use_index_dedup) {
      void *dst_ptr = reinterpret_cast<uint8_t*>(unique_idx.data_ptr()) +
                        unique_embs_offset * unique_idx.element_size();
      void *src_ptr = tmp_unique_indices[i].data_ptr();
      size_t copy_size = tmp_unique_num * unique_idx.element_size();
      AT_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, copy_size, cudaMemcpyDeviceToDevice, stream));
    }
      // CUDA_CHECK(cudaDeviceSynchronize());
      // std::cout << "Jiashu " << __FILE__ << " " << __LINE__ << "\n";
    unique_embs_offset += tmp_unique_num;
  }
  auto src_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(unique_embs.dtype()));
  auto dst_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(output_embs.dtype()));
  auto offset_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(reverse_idx.dtype()));

  dyn_emb::scatter_fused(unique_embs.data_ptr(), output_embs.data_ptr(),
                         reverse_idx.data_ptr(), indices_shape, dim, src_type,
                         dst_type, offset_type, device_num_sms, stream);
}

void lookup_forward_dense(
    std::vector<std::shared_ptr<dyn_emb::DynamicVariableBase>> tables,
    const at::Tensor indices, const at::Tensor offsets,
    const std::vector<int> &table_offsets_in_feature, int table_num,
    int batch_size, int dim, const at::Tensor h_unique_offsets,
    const at::Tensor unique_embs, const at::Tensor output_embs) {

  if (!offsets.is_cuda() || !indices.is_cuda()) {
    throw std::runtime_error(
        "offsets or indices tensor must be on CUDA device");
  }

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t indices_shape = indices.size(0);
  auto scalar_type = unique_embs.dtype().toScalarType();
  auto emb_dtype = scalartype_to_datatype(scalar_type);
  scalar_type = output_embs.dtype().toScalarType();
  auto output_dtype = scalartype_to_datatype(scalar_type);
  auto &device_prop = DeviceProp::getDeviceProp(indices.device().index());

  at::Tensor h_offset =
      at::empty_like(offsets, offsets.options().device(at::kCPU));
  AT_CUDA_CHECK(cudaMemcpyAsync(h_offset.data_ptr(), offsets.data_ptr(),
                                offsets.numel() * offsets.element_size(),
                                cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  h_unique_offsets[0] = 0;
  for (int i = 0; i < table_num; ++i) {
    int table_offset_begin = table_offsets_in_feature[i];
    int table_offset_end = table_offsets_in_feature[i + 1];
    int offset_begin = table_offset_begin * batch_size;
    int offset_end = table_offset_end * batch_size;

    int64_t indices_begin = h_offset[offset_begin].item<int64_t>();
    int64_t indices_end = h_offset[offset_end].item<int64_t>();
    int64_t indices_length = indices_end - indices_begin;
    h_unique_offsets[i + 1] = indices_end;
    at::Tensor tmp_indices = create_sub_tensor(indices, indices_begin);
    at::Tensor tmp_unique_embs =
        create_sub_tensor(unique_embs, indices_begin * dim);
    find_or_insert(tables[i], indices_length, tmp_indices, tmp_unique_embs);
    at::Tensor tmp_output_embs =
        create_sub_tensor(output_embs, indices_begin * dim);
    dyn_emb::batched_vector_copy_device(
        tmp_unique_embs.data_ptr(), output_embs.data_ptr(), indices_length, dim,
        emb_dtype, output_dtype, device_prop.num_sms, stream);
  }
}

void lookup_backward_dense(const at::Tensor indices, const at::Tensor grads,
                           int32_t dim, const at::Tensor table_offsets,
                           at::Tensor unique_indices, at::Tensor unique_grads) {
  // Doc for dynamic embedding's backward:
  //   Step 1: using SegmentedSortDevice to sort the indices per table.
  //   Step 2: using SegmentedUniqueDevice to dedup the indices per table.
  //   Step 3: using 2-stage reduction to reduce the gradients.

  // Initialization
  if (!indices.is_cuda() || !grads.is_cuda() || !table_offsets.is_cuda() ||
      !table_offsets.is_cuda() || !unique_indices.is_cuda() ||
      !unique_grads.is_cuda()) {
    throw std::runtime_error("All argument tensors should on device");
  }
  auto device_ = indices.device();
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  // Number of tables should <= 2^31-1
  int32_t table_num = static_cast<int32_t>(table_offsets.size(0) - 1);
  auto scalar_type = indices.dtype().toScalarType();
  auto key_type = scalartype_to_datatype(scalar_type);
  auto id_stype = table_offsets.dtype().toScalarType(); // scalar type
  auto id_dtype = scalartype_to_datatype(id_stype);     // data type
  auto key_num = indices.size(0);

  // Step 1: using SegmentedSortDevice to sort the indices by table.
  SegmentedSortDevice sort_op =
      SegmentedSortDevice(device_, key_num, table_num, key_type, id_dtype);
  auto original_ids =
      at::empty_like(indices, indices.options().dtype(id_stype));
  auto sorted_keys = at::empty_like(indices, indices.options());
  auto sorted_key_ids =
      at::empty_like(indices, indices.options().dtype(id_stype));
  auto sorted_table_ids =
      at::empty_like(indices, indices.options().dtype(at::kInt));
  sort_op(indices, original_ids, table_offsets, sorted_keys, sorted_key_ids,
          sorted_table_ids, stream, true, true);

  // Step 2: using SegmentedUniqueDevice to dedup the indices by table.
  SegmentedUniqueDevice unique_op =
      SegmentedUniqueDevice(device_, key_num, key_type, id_dtype);
  auto unique_key_ids =
      at::empty_like(indices, indices.options().dtype(id_stype));
  unique_op(sorted_keys, sorted_table_ids, unique_indices, unique_key_ids,
            stream);

  // Step 3: using 2-stage reduction to reduce the gradients.
  LocalReduce localReduceOp(device_, key_num, dim, id_dtype, DataType::Float32);
  localReduceOp.local_reduce(grads, unique_grads, sorted_key_ids,
                             unique_key_ids, stream);
}

void lookup_backward_dense_dedup(const at::Tensor grads,
                                 at::Tensor unique_indices,
                                 at::Tensor reverse_idx, int32_t dim,
                                 at::Tensor unique_grads,
                                 int32_t device_num_sms) {
  // Initialization
  if (!grads.is_cuda() || !unique_indices.is_cuda() || !reverse_idx.is_cuda() ||
      !unique_grads.is_cuda()) {
    throw std::runtime_error("All argument tensors should on device");
  }
  auto device_ = unique_indices.device();
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto rev_idx_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(reverse_idx.dtype()));
  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
  auto idx_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(unique_indices.dtype()));
  auto key_num = reverse_idx.size(0);
  auto unique_key_num = unique_indices.size(0);

  dyn_emb::one_to_one_atomic(grads.data_ptr(), unique_indices.data_ptr(),
                             reverse_idx.data_ptr(), unique_grads.data_ptr(),
                             dim, key_num, unique_key_num, rev_idx_type,
                             grad_type, idx_type, device_num_sms, stream);
}

void dedup_input_indices(
    const at::Tensor indices, const at::Tensor offsets,
    const at::Tensor h_table_offsets_in_feature,
    const at::Tensor d_table_offsets_in_feature, int table_num,
    int local_batch_size, const at::Tensor reverse_idx,
    const at::Tensor h_unique_nums, const at::Tensor d_unique_nums,
    const at::Tensor h_unique_offsets, const at::Tensor d_unique_offsets,
    std::vector<at::Tensor> unique_idx, const at::Tensor new_offsets,
    const at::Tensor new_lengths, int device_num_sms,
    std::shared_ptr<dyn_emb::UniqueOpBase> unique_op) {

  if (!offsets.is_cuda() || !indices.is_cuda()) {
    throw std::runtime_error(
        "offsets or indices tensor must be on CUDA device");
  }

  // Check dtype of h_unique_nums and d_unique_nums
  if (h_unique_nums.scalar_type() != at::kUInt64 ||
      d_unique_nums.scalar_type() != at::kUInt64) {
    throw std::runtime_error(
        "h_unique_nums and d_unique_nums must have dtype uint64_t");
  }

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t indices_shape = indices.size(0);
  auto unique_num_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(d_unique_nums.dtype()));
  auto unique_offset_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(d_unique_offsets.dtype()));
  int64_t new_lengths_size = new_lengths.size(0);

  at::Tensor h_offset =
      at::empty_like(offsets, offsets.options().device(at::kCPU));
  AT_CUDA_CHECK(cudaMemcpyAsync(h_offset.data_ptr(), offsets.data_ptr(),
                                offsets.numel() * offsets.element_size(),
                                cudaMemcpyDeviceToHost, stream));

  size_t unique_op_capacity = unique_op->get_capacity();
  if (indices_shape * 2 > unique_op_capacity) {
    at::Tensor new_keys = at::empty({indices_shape * 2}, indices.options());
    at::Tensor new_vals = at::empty(
        {indices_shape * 2},
        at::TensorOptions().dtype(at::kUInt64).device(indices.device()));
    unique_op->reset_capacity(new_keys, new_vals, indices_shape * 2, stream);
  }

  std::vector<at::Tensor> tmp_unique_indices(table_num);
  for (int i = 0; i < table_num; ++i) {
    tmp_unique_indices[i] = at::empty_like(indices);
  }

  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  for (int i = 0; i < table_num; ++i) {
    int table_offset_begin = h_table_offsets_in_feature[i].item<int64_t>();
    int table_offset_end = h_table_offsets_in_feature[i + 1].item<int64_t>();
    int offset_begin = table_offset_begin * local_batch_size;
    int offset_end = table_offset_end * local_batch_size;

    int64_t indices_begin = h_offset[offset_begin].item<int64_t>();
    int64_t indices_end = h_offset[offset_end].item<int64_t>();
    int64_t indices_length = indices_end - indices_begin;

    if (indices_length == 0) {
      DEMB_CUDA_CHECK(cudaMemsetAsync(
          reinterpret_cast<uint64_t *>(d_unique_nums.data_ptr()) + i, 0,
          sizeof(uint64_t), stream));
      dyn_emb::add_offset(d_unique_nums.data_ptr(), d_unique_offsets.data_ptr(),
                          i, unique_num_type, unique_offset_type, stream);
    } else {
      at::Tensor tmp_indices = create_sub_tensor(indices, indices_begin);
      at::Tensor tmp_reverse_idx =
          create_sub_tensor(reverse_idx, indices_begin);
      at::Tensor tmp_d_unique_num = create_sub_tensor(d_unique_nums, i);
      at::Tensor previous_d_unique_num = create_sub_tensor(d_unique_offsets, i);

      unique_op->unique(tmp_indices, indices_length, tmp_reverse_idx,
                        unique_idx[i], tmp_d_unique_num, stream,
                        previous_d_unique_num);
      dyn_emb::add_offset(d_unique_nums.data_ptr(), d_unique_offsets.data_ptr(),
                          i, unique_num_type, unique_offset_type, stream);
    }
  }

  AT_CUDA_CHECK(
      cudaMemcpyAsync(h_unique_nums.data_ptr(), d_unique_nums.data_ptr(),
                      d_unique_nums.numel() * d_unique_nums.element_size(),
                      cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(
      h_unique_offsets.data_ptr(), d_unique_offsets.data_ptr(),
      d_unique_offsets.numel() * d_unique_offsets.element_size(),
      cudaMemcpyDeviceToHost, stream));

  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  auto offset_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(new_offsets.dtype()));
  auto lengths_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(new_lengths.dtype()));

  get_new_length_and_offsets(
      reinterpret_cast<uint64_t *>(d_unique_offsets.data_ptr()),
      d_table_offsets_in_feature.data_ptr<int64_t>(), table_num,
      new_lengths_size, local_batch_size, lengths_type, offset_type,
      new_offsets.data_ptr(), new_lengths.data_ptr(), stream);
}

void lookup_forward(const at::Tensor src, const at::Tensor dst,
                    const at::Tensor offset, const at::Tensor inverse_idx,
                    int combiner, int total_D, int accum_D, int ev_size,
                    int num_vec, int batch_size, int device_num_sms) {

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto src_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(src.dtype()));
  auto dst_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(dst.dtype()));
  auto offset_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(offset.dtype()));
  if (combiner == -1) { // sequence
    auto &&num_emb = inverse_idx.size(0);
    dyn_emb::scatter(src.data_ptr(), dst.data_ptr(), offset.data_ptr(),
                     inverse_idx.data_ptr(), num_emb, ev_size, src_type,
                     dst_type, offset_type, device_num_sms, stream);
  } else {
    dyn_emb::scatter_combine(src.data_ptr(), dst.data_ptr(), offset.data_ptr(),
                             inverse_idx.data_ptr(), combiner, total_D, accum_D,
                             ev_size, num_vec, batch_size, src_type, dst_type,
                             offset_type, stream);
  }
}

void lookup_backward(const at::Tensor grad, const at::Tensor unique_buffer,
                     const at::Tensor unique_indices,
                     const at::Tensor inverse_indices,
                     const at::Tensor biased_offsets, const int dim,
                     const int table_num, int batch_size, int feature_num,
                     int num_key) {

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto value_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(unique_buffer.dtype()));
  auto key_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(unique_indices.dtype()));
  dyn_emb::backward(grad.data_ptr(), unique_buffer.data_ptr(),
                    unique_indices.data_ptr(), inverse_indices.data_ptr(),
                    biased_offsets.data_ptr(), dim, batch_size, feature_num,
                    num_key, key_type, value_type, stream);
}

// PYTHON WARP
void bind_dyn_emb_op(py::module &m) {
  py::class_<dyn_emb::InitializerArgs>(m, "InitializerArgs")
    .def(py::init([] (const std::string& mode, float mean, float std_dev, float lower, float upper, float value) {
      return dyn_emb::InitializerArgs(mode, mean, std_dev, lower, upper, value);
    }
    ))
    .def(py::pickle(
      [](const InitializerArgs &p) { // __getstate__
        return py::make_tuple(p.mode, p.mean, p.std_dev, p.lower, p.upper, p.value);
      },
      [](py::tuple t) { // __setstate__
        if (t.size() != 6)
          throw std::runtime_error("Invalid number args of InitializerArgs!");
        InitializerArgs p(
          t[0].cast<std::string>(),
          t[1].cast<float>(),
          t[2].cast<float>(),
          t[3].cast<float>(),
          t[4].cast<float>(),
          t[5].cast<float>());
        return p;
      }
     ));
    py::class_<dyn_emb::DynamicVariableBase, std::shared_ptr<dyn_emb::DynamicVariableBase>>(m, "DynamicEmbTable")
        .def(py::init([](dyn_emb::DataType key_type,
					dyn_emb::DataType value_type, 
					dyn_emb::EvictStrategy evict_type,
					int64_t dim = 128,
					int64_t init_capaity = 1024,
					int64_t max_capaity = 2048,
					size_t max_hbm_for_vectors = 0, 
					size_t max_bucket_size  = 128,
					float max_load_factor = 0.5,
					int block_size = 128,
					int io_block_size = 1024, 
					int device_id = -1, 
					bool io_by_cpu = false,
					bool use_constant_memory = false,
					int reserved_key_start_bit = 0,
					size_t num_of_buckets_per_alloc = 1,
					const dyn_emb::InitializerArgs & initializer_args = dyn_emb::InitializerArgs(),
          const int safe_check_mode = static_cast<int>(SafeCheckMode::IGNORE),
          const int optimizer_type = static_cast<int>(OptimizerType::Null)) {

            int64_t pow2_max_capaity = power2(max_capaity);
            int64_t pow2_init_capaity = power2(init_capaity);
            auto table = dyn_emb::VariableFactory::create(key_type,value_type,evict_type,dim,init_capaity,max_capaity,max_hbm_for_vectors,max_bucket_size,max_load_factor,
                                 block_size,io_block_size,device_id,io_by_cpu,use_constant_memory,reserved_key_start_bit,num_of_buckets_per_alloc,initializer_args, 
                                 static_cast<SafeCheckMode>(safe_check_mode), static_cast<OptimizerType>(optimizer_type));
            return table; }))
         .def("key_type", &dyn_emb::DynamicVariableBase::key_type,
             "Get Dynamic Emb Table key type")
         .def("value_type", &dyn_emb::DynamicVariableBase::value_type,
             "Get Dynamic Emb Table value type")
          .def("evict_strategy", &dyn_emb::DynamicVariableBase::evict_strategy,
            "Get evict strategy of Dynamic Emb Table.")
          .def("capacity", &dyn_emb::DynamicVariableBase::capacity,
            "Get capacity of Dynamic Emb Table.")
          .def("optstate_dim", &dyn_emb::DynamicVariableBase::optstate_dim,
            "Get dim of all optimizer states.")
          .def("set_initial_optstate", &dyn_emb::DynamicVariableBase::set_initial_optstate,
            "Set initial value of optimizer state.")
          .def("get_initial_optstate", &dyn_emb::DynamicVariableBase::get_initial_optstate,
            "Get initial value of optimizer state.")
          .def("load_factor", &dyn_emb::DynamicVariableBase::load_factor, "Get the load factor of the table");

  m.def("dyn_emb_rows", &dyn_emb_rows, "Get the number of rows in the table",
        py::arg("table"));

  m.def("dyn_emb_cols", &dyn_emb_cols, "Get the number of columns in the table",
        py::arg("table"));

  m.def("dyn_emb_capacity", &dyn_emb_capacity,
        "Get the capacity in the dynamic table", py::arg("table"));

  m.def("insert_or_assign", &insert_or_assign,
        "Insert or assign a key-value pair in the table", py::arg("table"),
        py::arg("n"), py::arg("keys"), py::arg("values"),
        py::arg("score") = c10::nullopt, py::arg("unique_key") = true,
        py::arg("ignore_evict_strategy") = false);

  m.def("insert_and_evict", &insert_and_evict,
        "Insert keys and values, evicting if necessary", py::arg("table"),
        py::arg("n"), py::arg("keys"), py::arg("values"), py::arg("score"),
        py::arg("evicted_keys"), py::arg("evicted_values"),
        py::arg("evicted_score"), py::arg("d_evicted_counter"),
        py::arg("unique_key") = true, py::arg("ignore_evict_strategy") = false);

  m.def("accum_or_assign", &accum_or_assign,
        "Accumulate or assign values to the table", py::arg("table"),
        py::arg("n"), py::arg("keys"), py::arg("value_or_deltas"),
        py::arg("accum_or_assigns"), py::arg("score") = c10::nullopt,
        py::arg("ignore_evict_strategy") = false);

  m.def("find_or_insert", &find_or_insert,
        "Find or insert a key-value pair in the table", py::arg("table"),
        py::arg("n"), py::arg("keys"), py::arg("values"),
        py::arg("score") = py::none(), py::arg("unique_key") = true, 
        py::arg("ignore_evict_strategy") = false);

  m.def("find_or_insert_pointers", &find_or_insert_pointers,
        "Find or insert a key-value pair in the table , and return every "
        "value's ptr",
        py::arg("table"), py::arg("n"), py::arg("keys"), py::arg("values"), py::arg("founds"),
        py::arg("score") = py::none(), py::arg("unique_key") = true, 
        py::arg("ignore_evict_strategy") = false);

  m.def("assign", &assign, "Assign values to the table based on keys",
        py::arg("table"), py::arg("n"), py::arg("keys"), py::arg("values"),
        py::arg("score") = c10::nullopt, py::arg("unique_key") = true);

  m.def("find", &find, "Find values in the table based on keys",
        py::arg("table"), py::arg("n"), py::arg("keys"), py::arg("values"),
        py::arg("founds"), py::arg("score") = c10::nullopt);

  m.def("erase", &erase, "Erase values from the table based on keys",
        py::arg("table"), py::arg("n"), py::arg("keys"));

  m.def("reserve", &reserve, "reserve hash table capacity", py::arg("table"),
        py::arg("new_capacity"));


  py::enum_<dyn_emb::DataType>(m, "DynamicEmbDataType")
      .value("Float32", dyn_emb::DataType::Float32)
      .value("BFloat16", dyn_emb::DataType::BFloat16)
      .value("Float16", dyn_emb::DataType::Float16)
      .value("Int64", dyn_emb::DataType::Int64)
      .value("UInt64", dyn_emb::DataType::UInt64)
      .value("Int32", dyn_emb::DataType::Int32)
      .value("UInt32", dyn_emb::DataType::UInt32)
      .value("Size_t", dyn_emb::DataType::Size_t)
      .export_values();
    m.def("clear", &clear,
          "Clear all keys in the table",
          py::arg("table"));

    m.def("reserve", &reserve,
          "reserve hash table capacity",
          py::arg("table"),
          py::arg("new_capacity"));

    m.def("export_batch", &export_batch,
          "export key value from table",
          py::arg("table"),
          py::arg("n"),
          py::arg("offset"),
          py::arg("d_counter"),
          py::arg("keys"),
          py::arg("values"),
          py::arg("score") = c10::nullopt);
    
    m.def("count_matched", &count_matched, 
      "Count the KV-pairs whose score > threshold in the whole table.",
      py::arg("table"),
      py::arg("threshold"),
      py::arg("num_matched"));

    m.def("export_batch_matched", &export_batch_matched,
      "Export KV-pairs within [offset, offset + n) whose score > threshold",
      py::arg("table"),
      py::arg("threshold"),
      py::arg("n"),
      py::arg("offset"),
      py::arg("num_matched"),
      py::arg("keys"),
      py::arg("values"));

  py::enum_<dyn_emb::EvictStrategy>(m, "EvictStrategy")
      .value("KLru", dyn_emb::EvictStrategy::kLru)
      .value("KLfu", dyn_emb::EvictStrategy::kLfu)
      .value("KEpochLru", dyn_emb::EvictStrategy::kEpochLru)
      .value("KEpochLfu", dyn_emb::EvictStrategy::kEpochLfu)
      .value("KCustomized", dyn_emb::EvictStrategy::kCustomized)
      .export_values();

  py::enum_<dyn_emb::OptimizerType>(m, "OptimizerType")
    .value("Null", dyn_emb::OptimizerType::Null)
    .value("SGD", dyn_emb::OptimizerType::SGD)
    .value("Adam", dyn_emb::OptimizerType::Adam)
    .value("AdaGrad", dyn_emb::OptimizerType::AdaGrad)
    .value("RowWiseAdaGrad", dyn_emb::OptimizerType::RowWiseAdaGrad)
    .export_values();

  m.def("lookup_forward", &lookup_forward, "scatter and combine",
        py::arg("src"), py::arg("dst"), py::arg("offset"),
        py::arg("inverse_idx"), py::arg("combiner"), py::arg("total_D"),
        py::arg("accum_D"), py::arg("ev_size"), py::arg("num_vec"),
        py::arg("batch_size"), py::arg("device_num_sms"));

  m.def("lookup_backward", &lookup_backward, "backward", py::arg("grad"),
        py::arg("unique_buffer"), py::arg("unique_indices"),
        py::arg("inverse_indices"), py::arg("biased_offsets"), py::arg("dim"),
        py::arg("tables_num"), py::arg("batch_size"), py::arg("num_feature"),
        py::arg("num_key"));

  m.def("lookup_forward_dense",
        (void (*)(std::vector<std::shared_ptr<dyn_emb::DynamicVariableBase>>,
                  const at::Tensor, const at::Tensor, const py::list, 
                  const std::vector<int> &,
                  at::Tensor, int, int, int, bool, const at::Tensor,
                  const at::Tensor, const at::Tensor, const at::Tensor,
                  const at::Tensor, const at::Tensor, const at::Tensor,
                  const at::Tensor, int,
                  std::shared_ptr<dyn_emb::UniqueOpBase>,
                  const c10::optional<at::Tensor>& cache_metrics,
                  std::optional<std::vector<std::shared_ptr<dyn_emb::DynamicVariableBase>>> host_tables)) &
            lookup_forward_dense,
        "lookup forward dense for duplicated keys", py::arg("tables"),
        py::arg("indices"), py::arg("offsets"), py::arg("scores"),
        py::arg("table_offsets_in_feature"), py::arg("table_offsets"),
        py::arg("table_num"), py::arg("batch_size"), py::arg("dim"),
        py::arg("use_index_dedup"), py::arg("unique_idx"),
        py::arg("reverse_idx"), py::arg("h_unique_nums"),
        py::arg("d_unique_nums"), py::arg("h_unique_offsets"),
        py::arg("d_unique_offsets"), py::arg("unique_embs"),
        py::arg("output_embs"), py::arg("device_num_sms"),
        py::arg("unique_op"), py::arg("cache_metrics") = c10::nullopt,  py::arg("host_tables") = py::none());

  m.def("lookup_forward_dense",
        (void (*)(std::vector<std::shared_ptr<dyn_emb::DynamicVariableBase>>,
                  const at::Tensor, const at::Tensor, const std::vector<int> &,
                  int, int, int, const at::Tensor, const at::Tensor,
                  const at::Tensor)) &
            lookup_forward_dense,
        "lookup forward dense for globally deduplicated keys",
        py::arg("tables"), py::arg("indices"), py::arg("offsets"),
        py::arg("table_offsets_in_feature"), py::arg("table_num"),
        py::arg("batch_size"), py::arg("dim"), py::arg("h_unique_offsets"),
        py::arg("unique_embs"), py::arg("output_embs"));

  m.def("lookup_backward_dense", &lookup_backward_dense,
        "lookup backward for dense/sequence", py::arg("indices"),
        py::arg("grads"), py::arg("dim"), py::arg("table_offsets"),
        py::arg("unique_indices"), py::arg("unique_grads"));


  m.def("lookup_backward_dense_dedup", &lookup_backward_dense_dedup,
        "lookup backward for dedup dense/sequence", py::arg("grads"),
        py::arg("unique_indices"), py::arg("reverse_idx"), py::arg("dim"),
        py::arg("unique_grads"), py::arg("device_num_sms"));

  m.def("dedup_input_indices", &dedup_input_indices,
        "duplicate indices from a given list or array of indices",
        py::arg("indices"), py::arg("offset"),
        py::arg("h_table_offsets_in_feature"),
        py::arg("d_table_offsets_in_feature"), py::arg("table_num"),
        py::arg("local_batch_size"), py::arg("reverse_idx"),
        py::arg("h_unique_nums"), py::arg("d_unique_nums"),
        py::arg("h_unique_offsets"), py::arg("d_unique_offsets"),
        py::arg("unique_idx"), py::arg("new_offsets"), py::arg("new_lengths"),
        py::arg("device_num_sms"), py::arg("unique_op"));
}
