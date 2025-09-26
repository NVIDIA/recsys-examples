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

#ifndef HKV_VARIABLE_H
#define HKV_VARIABLE_H

#include "dynamic_variable_base.h"
#include "merlin_hashtable.cuh"
#include <curand_kernel.h>
#include <string>

namespace dyn_emb {

template <typename KeyType, typename ValueType,
          EvictStrategy Strategy = EvictStrategy::kLru>
class HKVVariable : public DynamicVariableBase {
public:
  HKVVariable(DataType key_type_input, DataType value_type_input, int64_t dim,
              int64_t init_capacity, size_t max_capacity,
              size_t max_hbm_for_vectors = 0, size_t max_bucket_size = 128,
              float max_load_factor = 0.5f, int block_size = 128,
              int io_block_size = 1024, int device_id = -1,
              bool io_by_cpu = false, bool use_constant_memory = false,
              int reserved_key_start_bit = 0,
              size_t num_of_buckets_per_alloc = 1,
              const InitializerArgs &initializer = InitializerArgs(),
              const SafeCheckMode safe_check_mode = SafeCheckMode::IGNORE,
              const OptimizerType optimizer_type = OptimizerType::Null);

  ~HKVVariable() override;

  int64_t rows(cudaStream_t stream = 0) override;
  int64_t cols() override;
  int64_t capacity() override;

  DataType key_type() override;
  DataType value_type() override;

  void insert_or_assign(const size_t n,
                        const void *keys,             // (n)
                        const void *values,           // (n, DIM)
                        const void *scores = nullptr, // (n)
                        cudaStream_t stream = 0, bool unique_key = true,
                        bool ignore_evict_strategy = false) override;

  void insert_and_evict(const size_t n,
                        const void *keys,          // (n)
                        const void *values,        // (n, DIM)
                        const void *scores,        // (n)
                        void *evicted_keys,        // (n)
                        void *evicted_values,      // (n, DIM)
                        void *evicted_scores,      // (n)
                        size_t *d_evicted_counter, // (1)
                        cudaStream_t stream = 0, bool unique_key = true,
                        bool ignore_evict_strategy = false) override;

  void accum_or_assign(const size_t n,
                       const void *keys,             // (n)
                       const void *value_or_deltas,  // (n, DIM)
                       const bool *accum_or_assigns, // (n)
                       const void *scores = nullptr, // (n)
                       cudaStream_t stream = 0,
                       bool ignore_evict_strategy = false) override;

  void find_or_insert(const size_t n, const void *keys, // (n)
                         void **value_ptrs,                // (n * ptrs)
                         void *values,                     // (n * DIM)
                         bool *d_found,                    // (n * 1)
                         void *scores = nullptr,           // (n)
                         cudaStream_t stream = 0, bool unique_key = true,
                         bool ignore_evict_strategy = false) override;

  void find_and_initialize(
    const size_t n, const void *keys, void **value_ptrs, void *values,
    bool *founds, std::optional<InitializerArgs> initializer_args, const cudaStream_t& stream) override;

  void find_or_insert_pointers(const size_t n, const void *keys, // (n)
                               void **value_ptrs,                // (n * ptrs)
                               bool *d_found,                    // (n * 1)
                               void *scores = nullptr,           // (n)
                               cudaStream_t stream = 0, bool unique_key = true,
                               bool ignore_evict_strategy = false) override;

  void assign(const size_t n,
              const void *keys,             // (n)
              const void *values,           // (n, DIM)
              const void *scores = nullptr, // (n)
              cudaStream_t stream = 0, bool unique_key = true) override;

  void find(const size_t n, const void *keys, // (n)
            void *values,                     // (n, DIM)
            bool *founds,                     // (n)
            void *scores = nullptr,           // (n)
            cudaStream_t stream = 0) const override;

  void find_pointers(
    const size_t n, const void* keys,         // (n)
    void** values,                            // (n)
    bool* founds,                             // (n)
    void* scores = nullptr,                   // (n)
    cudaStream_t stream = 0) const override;

  void find_pointers(
    const size_t n, const void* keys,         // (n)
    void** values,                            // (n)
    bool* founds,                             // (n)
    void* scores = nullptr,                   // (n)
    cudaStream_t stream = 0) override;

  void erase(const size_t n, const void* keys,
           cudaStream_t stream = 0) override;
  
  void clear(cudaStream_t stream = 0) override;

  void reserve(const size_t new_capacity, cudaStream_t stream = 0) override;

  void export_batch(const size_t n, const size_t offset,
                    size_t *d_counter,      //(1)
                    void *keys,             // (n)
                    void *values,           // (n, DIM)
                    void *scores = nullptr, // (n)
                    cudaStream_t stream = 0) const override;

  EvictStrategy evict_strategy() const override;

  void export_batch_matched(
    uint64_t threshold,
    const uint64_t n,
    const uint64_t offset,
    uint64_t* d_counter,
    void* keys,              // (n)
    void* values,            // (n, DIM)
    void* scores = nullptr,  // (n)
    cudaStream_t stream = 0) const override;

  void count_matched(
    uint64_t threshold,
    uint64_t* d_counter, 
    cudaStream_t stream = 0) const override;

  void lock(
    const size_t n,
    const void* keys,            // (n)
    void** locked_keys_ptr,      // (n)
    bool* flags = nullptr,       // (n)
    void* scores = nullptr,     // (n)
    cudaStream_t stream = 0) override;

  void unlock(
    const size_t n,
    void** locked_keys_ptr,      // (n)
    const void* keys,            // (n)
    bool* flags = nullptr,       // (n)
    cudaStream_t stream = 0) override;
  
  curandState* get_curand_states() const override;
  const InitializerArgs& get_initializer_args() const override;
  const int optstate_dim() const override;
  void set_initial_optstate(const float value) override;
  const float get_initial_optstate() const override;


private:
  using HKVTable =
      nv::merlin::HashTable<KeyType, ValueType, uint64_t, (int)Strategy>;
  std::unique_ptr<HKVTable> hkv_table_ = std::make_unique<HKVTable>();
  nv::merlin::HashTableOptions hkv_table_option_;

  size_t dim_;
  size_t max_capacity_;
  const InitializerArgs initializer_args;
  curandState *curand_states_;
  DataType key_type_;
  DataType value_type_;
  SafeCheckMode safe_check_mode_;
  OptimizerType optimizer_type_;
  float initial_optstate_ {0.};

};

} // namespace dyn_emb

#endif // HKV_VARIABLE_H
