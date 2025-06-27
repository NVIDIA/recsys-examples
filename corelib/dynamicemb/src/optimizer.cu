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
#include "optimizer.h"
#include "optimizer_kernel.cuh"
#include "torch_utils.h"
#include "utils.h"
#include <iostream>

void find_pointers(
  std::shared_ptr<dyn_emb::DynamicVariableBase> table,
  const size_t n,
  const at::Tensor keys,
  at::Tensor values,
  at::Tensor founds);

void find_or_insert_pointers(
  std::shared_ptr<dyn_emb::DynamicVariableBase> table,
  const size_t n,
  const at::Tensor keys,
  at::Tensor values,
  at::Tensor founds,
  const std::optional<uint64_t> score = std::nullopt,
  bool unique_key = true,
  bool ignore_evict_strategy = false);

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
    bool ignore_evict_strategy = false);

int64_t find_and_get_missed(
  std::shared_ptr<dyn_emb::DynamicVariableBase> ht,
  uint64_t n,
  at::Tensor keys,
  at::Tensor foundss,
  at::Tensor vals_ptr,
  at::Tensor missed_keys,
  at::Tensor missed_ids,
  at::Tensor reverse_ids
);

at::Tensor create_sub_tensor(const at::Tensor &original_tensor,
                             int64_t offset);

void insert_or_assign(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
                      const size_t n, const at::Tensor keys,
                      const at::Tensor values,
                      const c10::optional<at::Tensor> &score = c10::nullopt,
                      bool unique_key = true,
                      bool ignore_evict_strategy = false);

void assign(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n,
            const at::Tensor keys, const at::Tensor values,
            const c10::optional<at::Tensor> &score = c10::nullopt,
            bool unique_key = true);

namespace dyn_emb {

constexpr int MULTIPLIER = 4;
constexpr int WARPSIZE = 32;
constexpr int OPTIMIZER_BLOCKSIZE_VEC = 64;
constexpr int OPTIMIZER_BLOCKSIZE = 1024;

template<typename IdxType, typename V, int GROUP_SIZE=32>
__global__ void get_missing_values(
    int n, int dim,
    IdxType const * __restrict__ original_ids,
    V const* __restrict__ src, 
    V* __restrict__ dst) {
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int group_id = tid / GROUP_SIZE;
  int lane_id = tid % GROUP_SIZE;
  if (group_id < n) {
    auto src_id = original_ids[group_id];
    for (int i = lane_id; i < dim; i += GROUP_SIZE) {
      dst[group_id * dim + i] = src[src_id * dim + i];
    }
  }
}

void update_heirarchical_tables(
  std::shared_ptr<dyn_emb::DynamicVariableBase> t1,
  std::shared_ptr<dyn_emb::DynamicVariableBase> t2,
  int64_t total_key_num,
  int64_t found_key_num,
  at::Tensor keys,
  at::Tensor values,
  const std::optional<uint64_t> score
) {

  //1.update
  if (t1->need_score()) {
    if (not score) {
      throw std::invalid_argument("Must specify the score.");
    }
    auto&& option = at::TensorOptions().dtype(at::kUInt64).device(keys.device());
    // broadcast scores
    at::Tensor bc_scores = at::empty({static_cast<int64_t>(found_key_num)}, option);
    bc_scores.fill_(score.value());
    c10::optional<at::Tensor> opt_scores(bc_scores);
    assign(t1, found_key_num, keys, values, opt_scores);
  } else {
    assign(t1, found_key_num, keys, values);
  }

  if (found_key_num == total_key_num) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // 2.insert_and_evict
  int64_t missed_key_num = total_key_num - found_key_num;
  auto missed_keys = create_sub_tensor(keys, found_key_num);
  auto missed_values = create_sub_tensor(values, found_key_num * values.size(1));
  at::Tensor evicted_keys = at::empty({static_cast<int64_t>(missed_key_num)}, keys.options());
  at::Tensor evicted_values = at::empty({static_cast<int64_t>(missed_key_num), values.size(1)}, values.options());
  at::Tensor evicted_score = at::empty({static_cast<int64_t>(missed_key_num)}, keys.options().dtype(at::kUInt64));
  at::Tensor d_evicted_counter =  at::zeros({static_cast<int64_t>(1)}, at::TensorOptions().dtype(at::kUInt64).device(keys.device()));
  insert_and_evict(t1, missed_key_num, missed_keys, missed_values, score, evicted_keys, evicted_values, evicted_score, d_evicted_counter);
  uint64_t evict_counter = 0;
  AT_CUDA_CHECK(cudaMemcpyAsync(&evict_counter, d_evicted_counter.data_ptr(),
      sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  if (evict_counter > missed_key_num) {
    throw std::runtime_error("Evict too much keys than new inserted.");
  }
  
  auto evict_score_opt = c10::make_optional(evicted_score);
  insert_or_assign(t2, evict_counter, evicted_keys, evicted_values, evict_score_opt);
}

__global__ void get_found_counter(bool const * __restrict__ found, int64_t* counter, int64_t batch_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < batch_size) {
    bool cur_found = found[tid];
    bool pre_found = tid != 0 ? found[tid - 1] : true;
    if (cur_found != pre_found) {
      *counter = static_cast<int64_t>(tid);
    }
  }
}

__global__ void verify_found_counter(bool const * __restrict__ found, int64_t counter, int64_t batch_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < batch_size) {
    bool found_ = found[tid];
    if (tid < counter) {
      if (not found_) {
        asm("trap;");
      }
    } else {
      if (found_) {
        asm("trap;");
      }  
    }
  }
}

int64_t find_ptr_from_hierarchical_table_for_classified_keys(
  std::shared_ptr<dyn_emb::DynamicVariableBase> ht1,
  std::shared_ptr<dyn_emb::DynamicVariableBase> ht2,
  uint64_t n,
  at::Tensor keys,
  at::Tensor founds,
  at::Tensor vals_ptr
) {

  find_pointers(ht1, n, keys, vals_ptr, founds);
  auto found_counter = at::zeros({static_cast<int64_t>(1)},
    at::TensorOptions().dtype(at::kLong).device(keys.device()));
  
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  
  get_found_counter<<<(n + 127) / 128, 128, 0, stream>>>(
    founds.data_ptr<bool>(), found_counter.data_ptr<int64_t>(), n
  );
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  int64_t found_counter_host = 0;
  AT_CUDA_CHECK(cudaMemcpyAsync(&found_counter_host, found_counter.data_ptr(),
      sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  ///TODO: open it under debug mode
  // verify_found_counter<<<(n + 127) / 128, 128, 0, stream>>>(
  //   founds.data_ptr<bool>(), found_counter_host, n
  // );
  // DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  // AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  auto missed_keys = create_sub_tensor(keys, found_counter_host);
  auto vals_host_ptr = create_sub_tensor(vals_ptr, found_counter_host);
  auto founds_host = create_sub_tensor(founds, found_counter_host);
  int64_t missed_counter = n - found_counter_host;
  find_pointers(ht2, missed_counter, missed_keys, vals_host_ptr, founds_host);
  return found_counter_host;
}
void dynamic_emb_sgd_with_table(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table, const uint64_t n, 
    const at::Tensor indices, const at::Tensor grads, const float lr, DataType weight_type, 
    const std::optional<uint64_t> score, const c10::optional<at::Tensor>& embs,
    const std::optional<std::shared_ptr<dyn_emb::DynamicVariableBase>> host_table) {

  if (n == 0) return;
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  at::Tensor founds = at::empty({static_cast<int64_t>(n)}, 
                                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor weight_ptrs = at::empty({static_cast<int64_t>(n)}, 
                                     at::TensorOptions().dtype(at::kLong).device(indices.device()));

  auto &device_prop = DeviceProp::getDeviceProp(grads.device().index());

  int64_t dim = grads.size(1);
  int64_t ev_nums = weight_ptrs.size(0);

  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (host_table.has_value()) {
    if (not embs.has_value()) {
      throw std::runtime_error("Not provide unique embeddings on caching mode.");
    }

    int64_t found_counter = find_ptr_from_hierarchical_table_for_classified_keys(
      table, host_table.value(), n, indices, founds, weight_ptrs
    );

    int optstate_dim = table->optstate_dim();
    auto emb_optstate_buffer = at::empty({static_cast<int64_t>(n), dim + optstate_dim}, embs.value().options());

    DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
      DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {
    
        SgdVecOptimizerV3<g_t, w_t> opt{lr};
        if (dim % 4 == 0) {
          const int max_grid_size =
              device_prop.num_sms *
              (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
          const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC / WARPSIZE;

          int grid_size = 0;
          if (ev_nums / warp_per_block < max_grid_size) {
            grid_size = (ev_nums - 1) / warp_per_block + 1;
          } else if (ev_nums / warp_per_block > max_grid_size * MULTIPLIER) {
            grid_size = max_grid_size * MULTIPLIER;
          } else {
            grid_size = max_grid_size;
          }

          auto kernel = update4_kernel_v4<g_t, w_t, decltype(opt)>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, optstate_dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t *>(embs.value().data_ptr()), 
            reinterpret_cast<w_t *>(emb_optstate_buffer.data_ptr()), opt,
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), founds.data_ptr<bool>()
          );
        } else {
          int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
          int grid_size = ev_nums;

          auto kernel = update_kernel_v4<g_t, w_t, decltype(opt)>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, optstate_dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t *>(embs.value().data_ptr()), 
            reinterpret_cast<w_t *>(emb_optstate_buffer.data_ptr()), opt,
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), founds.data_ptr<bool>()
          );
        }
      });
    });
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    update_heirarchical_tables(table, host_table.value(), n, found_counter, indices, emb_optstate_buffer, score);
    return;
  }
  if (embs.has_value()) {
    find_or_insert_pointers(table, n, indices, weight_ptrs, founds, score);
  } else {
    find_pointers(table, n, indices, weight_ptrs, founds);
  }

  if (embs.has_value()) {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
      DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {
        
        SgdVecOptimizerV2<g_t, w_t> opt{lr};
        if (dim % 4 == 0) {
          const int max_grid_size =
              device_prop.num_sms *
              (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
          const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC / WARPSIZE;

          int grid_size = 0;
          if (ev_nums / warp_per_block < max_grid_size) {
            grid_size = (ev_nums - 1) / warp_per_block + 1;
          } else if (ev_nums / warp_per_block > max_grid_size * MULTIPLIER) {
            grid_size = max_grid_size * MULTIPLIER;
          } else {
            grid_size = max_grid_size;
          }

          auto kernel = update4_kernel_v2<g_t, w_t, decltype(opt)>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), reinterpret_cast<w_t *>(embs.value().data_ptr()), founds.data_ptr<bool>(), opt);
          DEMB_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
          int grid_size = ev_nums;

          auto kernel = update_kernel_v2<g_t, w_t, decltype(opt)>;
          kernel<<<grid_size, block_size, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), reinterpret_cast<w_t *>(embs.value().data_ptr()), founds.data_ptr<bool>(), opt);
          DEMB_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });
    });
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {
      
      SgdVecOptimizer<g_t, w_t> opt{lr};
      if (dim % 4 == 0) {
        const int max_grid_size =
            device_prop.num_sms *
            (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
        const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC / WARPSIZE;

        int grid_size = 0;
        if (ev_nums / warp_per_block < max_grid_size) {
          grid_size = (ev_nums - 1) / warp_per_block + 1;
        } else if (ev_nums / warp_per_block > max_grid_size * MULTIPLIER) {
          grid_size = max_grid_size * MULTIPLIER;
        } else {
          grid_size = max_grid_size;
        }

        auto kernel = update4_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
          ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
          reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;

        auto kernel = update_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, block_size, 0, stream>>>(
          ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
          reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void dynamic_emb_adam_with_table(
  std::shared_ptr<dyn_emb::DynamicVariableBase> ht,
  const uint64_t n, const at::Tensor indices, const at::Tensor grads, 
  const float lr, const float beta1, const float beta2, const float eps,
  const float weight_decay,
  const uint32_t iter_num, DataType weight_type, 
  const std::optional<uint64_t> score,
  const c10::optional<at::Tensor>& embs,
  const std::optional<std::shared_ptr<dyn_emb::DynamicVariableBase>> host_table) {

  if (n == 0) return;
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  at::Tensor founds = at::empty({static_cast<int64_t>(n)}, 
                                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor vector_ptrs = at::empty({static_cast<int64_t>(n)}, 
                                     at::TensorOptions().dtype(at::kLong).device(indices.device()));

  auto &device_prop = DeviceProp::getDeviceProp(grads.device().index());

  int64_t dim = grads.size(1);
  int64_t ev_nums = n;

  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (host_table.has_value()) {
    if (not embs.has_value()) {
      throw std::runtime_error("Not provide unique embeddings on caching mode.");
    }

    int64_t found_counter = find_ptr_from_hierarchical_table_for_classified_keys(
      ht, host_table.value(), n, indices, founds, vector_ptrs
    );
    int optstate_dim = ht->optstate_dim();
    auto emb_optstate_buffer = at::empty({static_cast<int64_t>(n), dim + optstate_dim}, embs.value().options());

    DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
      DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {
 
        AdamVecOptimizerV3<g_t, w_t> opt{lr,
                                      beta1,
                                      beta2,
                                      eps,
                                      weight_decay,
                                      iter_num};
        if (dim % 4 == 0) {
          const int max_grid_size =
              device_prop.num_sms *
              (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
          const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC / WARPSIZE;

          int grid_size = 0;
          if (ev_nums / warp_per_block < max_grid_size) {
            grid_size = (ev_nums - 1) / warp_per_block + 1;
          } else if (ev_nums / warp_per_block > max_grid_size * MULTIPLIER) {
            grid_size = max_grid_size * MULTIPLIER;
          } else {
            grid_size = max_grid_size;
          }

          auto kernel = update4_kernel_v4<g_t, w_t, decltype(opt)>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, optstate_dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t *>(embs.value().data_ptr()), 
            reinterpret_cast<w_t *>(emb_optstate_buffer.data_ptr()), opt,
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), founds.data_ptr<bool>()
          );
        } else {
          int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
          int grid_size = ev_nums;

          auto kernel = update_kernel_v4<g_t, w_t, decltype(opt)>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, optstate_dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t *>(embs.value().data_ptr()), 
            reinterpret_cast<w_t *>(emb_optstate_buffer.data_ptr()), opt,
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), founds.data_ptr<bool>()
          );
        }
      });
    });
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    update_heirarchical_tables(ht, host_table.value(), n, found_counter, indices, emb_optstate_buffer, score);
    return;
  }

  if (embs.has_value()) {
    find_or_insert_pointers(ht, n, indices, vector_ptrs, founds, score);
  } else {
    find_pointers(ht, n, indices, vector_ptrs, founds);
  }
  
  if (embs.has_value()) {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
      DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {
        AdamVecOptimizerV2<g_t, w_t> opt{lr,
                                      beta1,
                                      beta2,
                                      eps,
                                      weight_decay,
                                      iter_num};
        if (dim % 4 == 0) {
          const int max_grid_size =
              device_prop.num_sms *
              (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
          const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC / WARPSIZE;

          int grid_size = 0;
          if (ev_nums / warp_per_block < max_grid_size) {
            grid_size = (ev_nums - 1) / warp_per_block + 1;
          } else if (ev_nums / warp_per_block > max_grid_size * MULTIPLIER) {
            grid_size = max_grid_size * MULTIPLIER;
          } else {
            grid_size = max_grid_size;
          }

          auto kernel = update4_kernel_v2<g_t, w_t, decltype(opt)>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), reinterpret_cast<w_t *>(embs.value().data_ptr()), founds.data_ptr<bool>(), opt);
          DEMB_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
          int grid_size = ev_nums;

          auto kernel = update_kernel_v2<g_t, w_t, decltype(opt)>;
          kernel<<<grid_size, block_size, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), reinterpret_cast<w_t *>(embs.value().data_ptr()), founds.data_ptr<bool>(), opt);
          DEMB_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });
    });
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {
      AdamVecOptimizer<g_t, w_t> opt{lr,
                                     beta1,
                                     beta2,
                                     eps,
                                     weight_decay,
                                     iter_num};
      if (dim % 4 == 0) {
        const int max_grid_size =
            device_prop.num_sms *
            (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
        const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC / WARPSIZE;

        int grid_size = 0;
        if (ev_nums / warp_per_block < max_grid_size) {
          grid_size = (ev_nums - 1) / warp_per_block + 1;
        } else if (ev_nums / warp_per_block > max_grid_size * MULTIPLIER) {
          grid_size = max_grid_size * MULTIPLIER;
        } else {
          grid_size = max_grid_size;
        }

        auto kernel = update4_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
          ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
          reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;

        auto kernel = update_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, block_size, 0, stream>>>(
          ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
          reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void dynamic_emb_adagrad_with_table(
  std::shared_ptr<dyn_emb::DynamicVariableBase> ht,
  const uint64_t n, const at::Tensor indices,
  const at::Tensor grads,
  const float lr,
  const float eps,
  DataType weight_type,const std::optional<uint64_t> score,
  const c10::optional<at::Tensor>& embs,
  const std::optional<std::shared_ptr<dyn_emb::DynamicVariableBase>> host_table){
  if (n == 0) return;

  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  at::Tensor founds = at::empty({static_cast<int64_t>(n)}, 
                                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor vector_ptrs = at::empty({static_cast<int64_t>(n)}, 
                                     at::TensorOptions().dtype(at::kLong).device(indices.device()));

  auto& device_prop = DeviceProp::getDeviceProp(grads.device().index());

  int64_t dim = grads.size(1);
  int64_t ev_nums = n;

  auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (host_table.has_value()) {
    if (not embs.has_value()) {
      throw std::runtime_error("Not provide unique embeddings on caching mode.");
    }

    int64_t found_counter = find_ptr_from_hierarchical_table_for_classified_keys(
      ht, host_table.value(), n, indices, founds, vector_ptrs
    );

    int optstate_dim = ht->optstate_dim();
    auto emb_optstate_buffer = at::empty({static_cast<int64_t>(n), dim + optstate_dim}, embs.value().options());

    DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
      DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {
        
        AdaGradVecOptimizerV3<g_t,w_t> opt{lr, eps, ht->get_initial_optstate()};

        if (dim % 4 == 0) {
          const int max_grid_size =
              device_prop.num_sms *
              (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
          const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC / WARPSIZE;

          int grid_size = 0;
          if (ev_nums / warp_per_block < max_grid_size) {
            grid_size = (ev_nums - 1) / warp_per_block + 1;
          } else if (ev_nums / warp_per_block > max_grid_size * MULTIPLIER) {
            grid_size = max_grid_size * MULTIPLIER;
          } else {
            grid_size = max_grid_size;
          }

          auto kernel = update4_kernel_v4<g_t, w_t, decltype(opt)>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, optstate_dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t *>(embs.value().data_ptr()), 
            reinterpret_cast<w_t *>(emb_optstate_buffer.data_ptr()), opt,
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), founds.data_ptr<bool>()
          );
        } else {
          int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
          int grid_size = ev_nums;

          auto kernel = update_kernel_v4<g_t, w_t, decltype(opt)>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, optstate_dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t *>(embs.value().data_ptr()), 
            reinterpret_cast<w_t *>(emb_optstate_buffer.data_ptr()), opt,
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), founds.data_ptr<bool>()
          );
        }
      });
    });
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    update_heirarchical_tables(ht, host_table.value(), n, found_counter, indices, emb_optstate_buffer, score);
    return;
  }

  if (embs.has_value()) {
    find_or_insert_pointers(ht, n, indices, vector_ptrs, founds, score);
  } else {
    find_pointers(ht, n, indices, vector_ptrs, founds);
  }

  if (embs.has_value()) {

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {

      AdaGradVecOptimizerV2<g_t,w_t> opt{lr, eps, ht->get_initial_optstate()};

      if (dim % 4 == 0) {
        const int max_grid_size = device_prop.num_sms * (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
        const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC/WARPSIZE;

        int grid_size = 0;
        if (ev_nums/warp_per_block < max_grid_size){
            grid_size = (ev_nums-1)/warp_per_block+1;
        }
        else if (ev_nums/warp_per_block > max_grid_size*MULTIPLIER){
            grid_size = max_grid_size*MULTIPLIER;
        }
        else{
            grid_size = max_grid_size;
        }

        auto kernel = update4_kernel_v2<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
          ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
          reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), reinterpret_cast<w_t *>(embs.value().data_ptr()), founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      } else {

        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;

        auto kernel = update_kernel_v2<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, block_size, 0, stream>>>(
          ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
          reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), reinterpret_cast<w_t *>(embs.value().data_ptr()), founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

    return;
  }

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {

      AdaGradVecOptimizer<g_t,w_t> opt{lr, eps};

      if (dim % 4 == 0) {
        const int max_grid_size = device_prop.num_sms * (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
        const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC/WARPSIZE;

        int grid_size = 0;
        if (ev_nums/warp_per_block < max_grid_size){
            grid_size = (ev_nums-1)/warp_per_block+1;
        }
        else if (ev_nums/warp_per_block > max_grid_size*MULTIPLIER){
            grid_size = max_grid_size*MULTIPLIER;
        }
        else{
            grid_size = max_grid_size;
        }

        auto kernel = update4_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
          ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
          reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      } else {

        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;

        auto kernel = update_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, block_size, 0, stream>>>(
          ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
          reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void dynamic_emb_rowwise_adagrad_with_table(
  std::shared_ptr<dyn_emb::DynamicVariableBase> ht,
  const uint64_t n, const at::Tensor indices,
  const at::Tensor grads,
  const float lr,
  const float eps,
  DataType weight_type,const std::optional<uint64_t> score,
  const c10::optional<at::Tensor>& embs,
  const std::optional<std::shared_ptr<dyn_emb::DynamicVariableBase>> host_table) {
  if (n == 0) return;
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  at::Tensor founds = at::empty({static_cast<int64_t>(n)}, 
                                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor vector_ptrs = at::empty({static_cast<int64_t>(n)}, 
                                     at::TensorOptions().dtype(at::kLong).device(indices.device()));

  auto& device_prop = DeviceProp::getDeviceProp(grads.device().index());

  int64_t dim = grads.size(1);
  int64_t ev_nums = n;

  auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (host_table.has_value()) {
    if (not embs.has_value()) {
      throw std::runtime_error("Not provide unique embeddings on caching mode.");
    }

    int64_t found_counter = find_ptr_from_hierarchical_table_for_classified_keys(
      ht, host_table.value(), n, indices, founds, vector_ptrs
    );

    int optstate_dim = ht->optstate_dim();
    auto emb_optstate_buffer = at::empty({static_cast<int64_t>(n), dim + optstate_dim}, embs.value().options());

    DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
      DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {
        
        RowWiseAdaGradVecOptimizerV3<g_t, w_t> opt {lr, eps, ht->get_initial_optstate()};

        if (dim % 4 == 0) {
          const int max_grid_size =
              device_prop.num_sms *
              (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
          const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC / WARPSIZE;

          int grid_size = 0;
          if (ev_nums / warp_per_block < max_grid_size) {
            grid_size = (ev_nums - 1) / warp_per_block + 1;
          } else if (ev_nums / warp_per_block > max_grid_size * MULTIPLIER) {
            grid_size = max_grid_size * MULTIPLIER;
          } else {
            grid_size = max_grid_size;
          }

          auto kernel = update4_kernel_v4<g_t, w_t, decltype(opt)>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, optstate_dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t *>(embs.value().data_ptr()), 
            reinterpret_cast<w_t *>(emb_optstate_buffer.data_ptr()), opt,
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), founds.data_ptr<bool>()
          );
        } else {
          int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
          int grid_size = ev_nums;

          auto kernel = update_kernel_v4<g_t, w_t, decltype(opt)>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, optstate_dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t *>(embs.value().data_ptr()), 
            reinterpret_cast<w_t *>(emb_optstate_buffer.data_ptr()), opt,
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), founds.data_ptr<bool>()
          );
        }
      });
    });
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    update_heirarchical_tables(ht, host_table.value(), n, found_counter, indices, emb_optstate_buffer, score);
    return;
  }

  if (embs.has_value()) {
    find_or_insert_pointers(ht, n, indices, vector_ptrs, founds, score);
  } else {
    find_pointers(ht, n, indices, vector_ptrs, founds);
  }

  if (embs.has_value()) {

    DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
      DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {

        RowWiseAdaGradVecOptimizerV2<g_t, w_t> opt {lr, eps, ht->get_initial_optstate()};
        if (dim % 4 == 0) {
          const int max_grid_size = device_prop.num_sms * (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
          const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC / WARPSIZE;

          int grid_size = 0;
          if (ev_nums / warp_per_block < max_grid_size) {
            grid_size = (ev_nums-1) / warp_per_block + 1;
          }
          else if (ev_nums / warp_per_block > max_grid_size * MULTIPLIER) {
            grid_size = max_grid_size * MULTIPLIER;
          } else {
            grid_size = max_grid_size;
          }

          auto kernel = update4_kernel_v2<g_t, w_t, decltype(opt)>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), reinterpret_cast<w_t *>(embs.value().data_ptr()), founds.data_ptr<bool>(), opt);
          DEMB_CUDA_KERNEL_LAUNCH_CHECK();

        } else {

          int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
          int grid_size = ev_nums;
          int shared_memory_bytes = block_size * sizeof(float);

          auto kernel = update_kernel_v2<g_t, w_t, decltype(opt)>;
          kernel<<<grid_size, block_size, shared_memory_bytes, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), reinterpret_cast<w_t *>(embs.value().data_ptr()), founds.data_ptr<bool>(), opt);
          DEMB_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });
    });
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();

    return;
  }

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {

      RowWiseAdaGradVecOptimizer<g_t, w_t> opt {lr, eps};
      if (dim % 4 == 0) {
        const int max_grid_size = device_prop.num_sms * (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
        const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC / WARPSIZE;

        int grid_size = 0;
        if (ev_nums / warp_per_block < max_grid_size) {
          grid_size = (ev_nums-1) / warp_per_block + 1;
        }
        else if (ev_nums / warp_per_block > max_grid_size * MULTIPLIER) {
          grid_size = max_grid_size * MULTIPLIER;
        } else {
          grid_size = max_grid_size;
        }

        auto kernel = update4_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
          ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
          reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();

      } else {

        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;
        int shared_memory_bytes = block_size * sizeof(float);

        auto kernel = update_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, block_size, shared_memory_bytes, stream>>>(
          ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
          reinterpret_cast<w_t **>(vector_ptrs.data_ptr()), founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace dyn_emb

// PYTHON WRAP
void bind_optimizer_kernel_op(py::module &m) {
  m.def("dynamic_emb_sgd_with_table", &dyn_emb::dynamic_emb_sgd_with_table,
        "SGD optimizer for Dynamic Emb", py::arg("table"),
        py::arg("n"), py::arg("indices"), py::arg("grads"),
        py::arg("lr"), py::arg("weight_type"), py::arg("score") = py::none(), py::arg("emb") = c10::nullopt,
        py::arg("host_table") = py::none());

  m.def("dynamic_emb_adam_with_table", &dyn_emb::dynamic_emb_adam_with_table,
        "Adam optimizer for Dynamic Emb", py::arg("ht"),
        py::arg("n"), py::arg("indices"), py::arg("grads"),
        py::arg("lr"), py::arg("beta1"),
        py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"), py::arg("iter_num"),
        py::arg("weight_type"), py::arg("score") = py::none(), py::arg("emb") = c10::nullopt,
        py::arg("host_table") = py::none());

  m.def("dynamic_emb_adagrad_with_table", &dyn_emb::dynamic_emb_adagrad_with_table,
        "Adagrad optimizer for Dynamic Emb", py::arg("ht"),
        py::arg("n"), py::arg("indices"), py::arg("grads"),py::arg("lr"),
        py::arg("eps"),
        py::arg("weight_type"), py::arg("score") = py::none(), py::arg("emb") = c10::nullopt,
        py::arg("host_table") = py::none());

  m.def("dynamic_emb_rowwise_adagrad_with_table", &dyn_emb::dynamic_emb_rowwise_adagrad_with_table,
        "Row Wise Adagrad optimizer for Dynamic Emb", py::arg("ht"),
        py::arg("n"), py::arg("indices"), py::arg("grads"),py::arg("lr"),
        py::arg("eps"),
        py::arg("weight_type"), py::arg("score") = py::none(), py::arg("emb") = c10::nullopt,
        py::arg("host_table") = py::none());
}
