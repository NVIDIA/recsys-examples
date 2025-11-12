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

#ifndef DYNAMIC_VARIABLE_BASE_H
#define DYNAMIC_VARIABLE_BASE_H
#include "utils.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <memory>
#include <optional>
#include <string>

namespace dyn_emb {

struct InitializerArgs {
  const std::string mode;
  float mean;
  float std_dev;
  float lower;
  float upper;
  float value;
  InitializerArgs(const std::string &mode_, float mean_, float std_dev_,
                  float lower_, float upper_, float value_)
      : mode(mode_), mean(mean_), std_dev(std_dev_), lower(lower_),
        upper(upper_), value(value_) {}
  InitializerArgs()
      : InitializerArgs("uniform", 0.0f, 1.0f, 0.0f, 1.0f, 0.0f) {}
};

enum class OptimizerType : int {
  Null = 0, // used in inference mode.
  SGD,
  Adam,
  AdaGrad,
  RowWiseAdaGrad,
};

template <typename T>
uint32_t get_optimizer_state_dim(OptimizerType opt_type, uint32_t emb_dim) {
  uint32_t optstate_dim = 0;
  switch (opt_type) {
  case OptimizerType::Null:
  case OptimizerType::SGD: {
    break;
  }
  case OptimizerType::Adam: {
    optstate_dim = emb_dim * 2;
    break;
  }
  case OptimizerType::AdaGrad: {
    optstate_dim = emb_dim;
    break;
  }
  case OptimizerType::RowWiseAdaGrad: {
    optstate_dim = 16 / sizeof(T);
    break;
  }
  default:
    throw std::invalid_argument("Unsupported optimizer type.");
  }
  return optstate_dim;
}

enum class SafeCheckMode : int { ERROR = 0, WARNING = 1, IGNORE = 2 };

class DynamicVariableBase {
public:
  virtual ~DynamicVariableBase() = default;

  virtual int64_t rows(cudaStream_t stream = 0) = 0;
  virtual int64_t cols() = 0;
  virtual int64_t capacity() = 0;

  virtual DataType key_type() = 0;
  virtual DataType value_type() = 0;

  virtual void insert_or_assign(const size_t n,
                                const void *keys,             // (n)
                                const void *values,           // (n, DIM)
                                const void *scores = nullptr, // (n)
                                cudaStream_t stream = 0, bool unique_key = true,
                                bool ignore_evict_strategy = false) = 0;

  virtual void insert_and_evict(const size_t n,
                                const void *keys,            // (n)
                                const void *values,          // (n, DIM)
                                const void *scores,          // (n)
                                void *evicted_keys,          // (n)
                                void *evicted_values,        // (n, DIM)
                                void *evicted_scores,        // (n)
                                uint64_t *d_evicted_counter, // (1)
                                cudaStream_t stream = 0, bool unique_key = true,
                                bool ignore_evict_strategy = false) = 0;

  virtual void accum_or_assign(const size_t n,
                               const void *keys,             // (n)
                               const void *value_or_deltas,  // (n, DIM)
                               const bool *accum_or_assigns, // (n)
                               const void *scores = nullptr, // (n)
                               cudaStream_t stream = 0,
                               bool ignore_evict_strategy = false) = 0;

  virtual void find_or_insert(const size_t n, const void *keys, // (n)
                              void **value_ptrs,                // (n * ptrs)
                              void *values,                     // (n * DIM)
                              bool *d_found,                    // (n * 1)
                              void *scores = nullptr,           // (n)
                              cudaStream_t stream = 0, bool unique_key = true,
                              bool ignore_evict_strategy = false) = 0;

  virtual void find_or_insert_pointers(const size_t n, const void *keys, // (n)
                                       void **value_ptrs,      // (n * ptrs)
                                       bool *d_found,          // (n * 1)
                                       void *scores = nullptr, // (n)
                                       cudaStream_t stream = 0,
                                       bool unique_key = true,
                                       bool ignore_evict_strategy = false) = 0;

  virtual void
  find_and_initialize(const size_t n, const void *keys, void **value_ptrs,
                      void *values, bool *founds,
                      std::optional<InitializerArgs> initializer_args,
                      const cudaStream_t &stream) = 0;

  virtual void assign(const size_t n,
                      const void *keys,             // (n)
                      const void *values,           // (n, DIM)
                      const void *scores = nullptr, // (n)
                      cudaStream_t stream = 0, bool unique_key = true) = 0;

  virtual void find(const size_t n, const void *keys, // (n)
                    void *values,                     // (n, DIM)
                    bool *founds,                     // (n)
                    void *scores = nullptr,           // (n)
                    cudaStream_t stream = 0) const = 0;

  virtual void find_pointers(const size_t n, const void *keys, // (n)
                             void **values,                    // (n)
                             bool *founds,                     // (n)
                             void *scores = nullptr,           // (n)
                             cudaStream_t stream = 0) const = 0;

  virtual void find_pointers(const size_t n, const void *keys, // (n)
                             void **values,                    // (n)
                             bool *founds,                     // (n)
                             void *scores = nullptr,           // (n)
                             cudaStream_t stream = 0) = 0;

  virtual void erase(const size_t n, const void *keys,
                     cudaStream_t stream = 0) = 0;
  virtual void clear(cudaStream_t stream = 0) = 0;

  virtual void reserve(const size_t new_capacity, cudaStream_t stream = 0) = 0;

  virtual void export_batch(const size_t n, const size_t offset,
                            size_t *d_counter,      //(1)
                            void *keys,             // (n)
                            void *values,           // (n, DIM)
                            void *scores = nullptr, // (n)
                            cudaStream_t stream = 0) const = 0;

  virtual EvictStrategy evict_strategy() const = 0;

  virtual void export_batch_matched(uint64_t threshold, const uint64_t n,
                                    const uint64_t offset, uint64_t *d_counter,
                                    void *keys,             // (n)
                                    void *values,           // (n, DIM)
                                    void *scores = nullptr, // (n)
                                    cudaStream_t stream = 0) const = 0;

  virtual void count_matched(uint64_t threshold, uint64_t *d_counter,
                             cudaStream_t stream = 0) const = 0;

  virtual void lock(const size_t n,
                    const void *keys,       // (n)
                    void **locked_keys_ptr, // (n)
                    bool *flags = nullptr,  // (n)
                    void *scores = nullptr, // (n)
                    cudaStream_t stream = 0) = 0;

  virtual void unlock(const size_t n,
                      void **locked_keys_ptr, // (n)
                      const void *keys,       // (n)
                      bool *flags = nullptr,  // (n)
                      cudaStream_t stream = 0) = 0;

  virtual curandState *get_curand_states() const = 0;
  virtual const InitializerArgs &get_initializer_args() const = 0;
  virtual const int optstate_dim() const = 0;
  virtual void set_initial_optstate(const float value) = 0;
  virtual const float get_initial_optstate() const = 0;
};

class VariableFactory {
public:
  static std::shared_ptr<DynamicVariableBase> create(
      DataType keytype, DataType valuetype, EvictStrategy evict_type,
      int64_t dim, size_t init_capacity, size_t max_capacity,
      size_t max_hbm_for_vectors, size_t max_bucket_size, float max_load_factor,
      int block_size, int io_block_size, int device_id, bool io_by_cpu,
      bool use_constant_memory, int reserved_key_start_bit,
      size_t num_of_buckets_per_alloc, const InitializerArgs &initializer_args,
      const SafeCheckMode safe_check_mode, const OptimizerType optimizer_type);
};

} // namespace dyn_emb
#endif // DYNAMIC_VARIABLE_BASE_H
