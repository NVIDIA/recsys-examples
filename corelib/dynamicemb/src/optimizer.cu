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
#include "optimizer.h"
#include "optimizer_kernel.cuh"
#include "torch_utils.h"
#include "utils.h"
#include <functional>

void find_pointers(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
                   const size_t n, const at::Tensor keys, at::Tensor values,
                   at::Tensor founds,
                   const std::optional<uint64_t> score = std::nullopt);

namespace dyn_emb {

constexpr int MULTIPLIER = 4;
constexpr int WARPSIZE = 32;
constexpr int OPTIMIZER_BLOCKSIZE_VEC = 64;
constexpr int OPTIMIZER_BLOCKSIZE = 1024;

void dynamic_emb_sgd_with_table(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table, const uint64_t n,
    const at::Tensor indices, const at::Tensor grads, const float lr,
    DataType weight_type) {

  if (n == 0)
    return;
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  at::Tensor founds =
      at::empty({static_cast<int64_t>(n)},
                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor weight_ptrs =
      at::empty({static_cast<int64_t>(n)},
                at::TensorOptions().dtype(at::kLong).device(indices.device()));

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  find_pointers(table, n, indices, weight_ptrs, founds);

  auto &device_prop = DeviceProp::getDeviceProp(grads.device().index());

  int64_t dim = grads.size(1);
  int64_t ev_nums = weight_ptrs.size(0);

  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));

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
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()),
            founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;

        auto kernel = update_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, block_size, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()),
            founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void dynamic_emb_adam_with_table(
    std::shared_ptr<dyn_emb::DynamicVariableBase> ht, const uint64_t n,
    const at::Tensor indices, const at::Tensor grads, const float lr,
    const float beta1, const float beta2, const float eps,
    const float weight_decay, const uint32_t iter_num, DataType weight_type) {

  if (n == 0)
    return;
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  at::Tensor founds =
      at::empty({static_cast<int64_t>(n)},
                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor vector_ptrs =
      at::empty({static_cast<int64_t>(n)},
                at::TensorOptions().dtype(at::kLong).device(indices.device()));

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  find_pointers(ht, n, indices, vector_ptrs, founds);

  auto &device_prop = DeviceProp::getDeviceProp(grads.device().index());

  int64_t dim = grads.size(1);
  int64_t ev_nums = n;

  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {
      AdamVecOptimizer<g_t, w_t> opt{lr,  beta1,        beta2,
                                     eps, weight_decay, iter_num};
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
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()),
            founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;

        auto kernel = update_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, block_size, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()),
            founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void dynamic_emb_adagrad_with_table(
    std::shared_ptr<dyn_emb::DynamicVariableBase> ht, const uint64_t n,
    const at::Tensor indices, const at::Tensor grads, const float lr,
    const float eps, DataType weight_type) {
  if (n == 0)
    return;

  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  at::Tensor founds =
      at::empty({static_cast<int64_t>(n)},
                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor vector_ptrs =
      at::empty({static_cast<int64_t>(n)},
                at::TensorOptions().dtype(at::kLong).device(indices.device()));

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  find_pointers(ht, n, indices, vector_ptrs, founds);

  auto &device_prop = DeviceProp::getDeviceProp(grads.device().index());

  int64_t dim = grads.size(1);
  int64_t ev_nums = n;

  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {
      AdaGradVecOptimizer<g_t, w_t> opt{lr, eps};

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
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()),
            founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      } else {

        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;

        auto kernel = update_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, block_size, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()),
            founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void dynamic_emb_rowwise_adagrad_with_table(
    std::shared_ptr<dyn_emb::DynamicVariableBase> ht, const uint64_t n,
    const at::Tensor indices, const at::Tensor grads, const float lr,
    const float eps, DataType weight_type) {
  if (n == 0)
    return;
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  at::Tensor founds =
      at::empty({static_cast<int64_t>(n)},
                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor vector_ptrs =
      at::empty({static_cast<int64_t>(n)},
                at::TensorOptions().dtype(at::kLong).device(indices.device()));

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  find_pointers(ht, n, indices, vector_ptrs, founds);

  auto &device_prop = DeviceProp::getDeviceProp(grads.device().index());

  int64_t dim = grads.size(1);
  int64_t ev_nums = n;

  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {
      RowWiseAdaGradVecOptimizer<g_t, w_t> opt{lr, eps};
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
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()),
            founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();

      } else {

        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;
        int shared_memory_bytes = block_size * sizeof(float);

        auto kernel = update_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, block_size, shared_memory_bytes, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(vector_ptrs.data_ptr()),
            founds.data_ptr<bool>(), opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void dynamic_emb_sgd_with_pointer(at::Tensor grads, at::Tensor val_pointers,
                                  DataType val_type, float const lr) {
  int64_t ev_nums = grads.size(0);
  int64_t dim = grads.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(val_pointers.is_cuda(), "val_pointers must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto &device_prop = DeviceProp::getDeviceProp(grads.device().index());

  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
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
            reinterpret_cast<w_t **>(val_pointers.data_ptr()), nullptr, opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;

        auto kernel = update_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, block_size, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(val_pointers.data_ptr()), nullptr, opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void dynamic_emb_adam_with_pointer(at::Tensor grads, at::Tensor val_pointers,
                                   DataType val_type, int64_t state_dim,
                                   const float lr, const float beta1,
                                   const float beta2, const float eps,
                                   const float weight_decay,
                                   const uint32_t iter_num) {
  int64_t ev_nums = grads.size(0);
  int64_t dim = grads.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(val_pointers.is_cuda(), "val_pointers must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto &device_prop = DeviceProp::getDeviceProp(grads.device().index());

  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      AdamVecOptimizer<g_t, w_t> opt{lr,  beta1,        beta2,
                                     eps, weight_decay, iter_num};
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
            reinterpret_cast<w_t **>(val_pointers.data_ptr()), nullptr, opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;

        auto kernel = update_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, block_size, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(val_pointers.data_ptr()), nullptr, opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void dynamic_emb_adagrad_with_pointer(at::Tensor grads, at::Tensor val_pointers,
                                      DataType val_type, int64_t state_dim,
                                      const float lr, const float eps) {

  int64_t ev_nums = grads.size(0);
  int64_t dim = grads.size(1);
  if (ev_nums == 0)
    return;

  TORCH_CHECK(val_pointers.is_cuda(), "val_pointers must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto &device_prop = DeviceProp::getDeviceProp(grads.device().index());

  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      AdaGradVecOptimizer<g_t, w_t> opt{lr, eps};

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
            reinterpret_cast<w_t **>(val_pointers.data_ptr()), nullptr, opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      } else {

        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;

        auto kernel = update_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, block_size, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(val_pointers.data_ptr()), nullptr, opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void dynamic_emb_rowwise_adagrad_with_pointer(at::Tensor grads,
                                              at::Tensor val_pointers,
                                              DataType val_type,
                                              int64_t state_dim, const float lr,
                                              const float eps) {

  int64_t ev_nums = grads.size(0);
  int64_t dim = grads.size(1);
  if (ev_nums == 0)
    return;

  TORCH_CHECK(val_pointers.is_cuda(), "val_pointers must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto &device_prop = DeviceProp::getDeviceProp(grads.device().index());

  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      RowWiseAdaGradVecOptimizer<g_t, w_t> opt{lr, eps};
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
            reinterpret_cast<w_t **>(val_pointers.data_ptr()), nullptr, opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();

      } else {

        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;
        int shared_memory_bytes = block_size * sizeof(float);

        auto kernel = update_kernel<g_t, w_t, decltype(opt)>;
        kernel<<<grid_size, block_size, shared_memory_bytes, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(val_pointers.data_ptr()), nullptr, opt);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename GradType, typename WeightType, typename IndexType,
          typename OptimizerType>
void launch_update_kernel_for_combined_table(
    GradType *grads, WeightType *dev_table, WeightType *uvm_table,
    IndexType *indices, OptimizerType opt, int64_t const ev_nums,
    uint32_t const dim, int64_t const stride, int64_t const split_index,
    int device_id,
    std::function<float(int)> smem_size_f = [](int block_size) { return 0; }) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto &device_prop = DeviceProp::getDeviceProp(device_id);
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

    auto kernel = update4_with_index_kernel<GradType, WeightType, IndexType,
                                            OptimizerType>;
    kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
        ev_nums, dim, stride, split_index, grads, dev_table, uvm_table, indices,
        nullptr, opt);
  } else {
    int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
    int grid_size = ev_nums;

    auto kernel = update_with_index_kernel<GradType, WeightType, IndexType,
                                           OptimizerType>;
    kernel<<<grid_size, block_size, smem_size_f(block_size), stream>>>(
        ev_nums, dim, stride, split_index, grads, dev_table, uvm_table, indices,
        nullptr, opt);
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void sgd_update_for_combined_table(at::Tensor grads, at::Tensor indices,
                                   std::optional<at::Tensor> dev_table,
                                   std::optional<at::Tensor> uvm_table,
                                   float const lr) {
  int64_t ev_nums = grads.size(0);
  int64_t dim = grads.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  DataType val_type;
  int64_t stride = -1;
  if ((not dev_table.has_value()) and (not uvm_table.has_value())) {
    throw std::runtime_error("Two tables cannot both be None.");
  } else {
    if (dev_table.has_value()) {
      val_type = get_data_type(dev_table.value());
      stride = dev_table.value().size(1);
    } else {
      val_type = get_data_type(uvm_table.value());
      stride = uvm_table.value().size(1);
    }
  }

  auto grad_type = get_data_type(grads);
  auto index_type = get_data_type(indices);

  int64_t split_index = 0;
  if (dev_table.has_value()) {
    split_index = dev_table.value().size(0);
  }

  int device_id = grads.device().index();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      DISPATCH_OFFSET_INT_TYPE(index_type, i_t, [&] {
        auto grad_ptr = get_pointer<g_t>(grads);
        auto dev_ptr = get_pointer<w_t>(dev_table);
        auto uvm_ptr = get_pointer<w_t>(uvm_table);

        auto index_ptr = get_pointer<i_t>(indices);

        SgdVecOptimizer<g_t, w_t> opt{lr};

        launch_update_kernel_for_combined_table<g_t, w_t, i_t, decltype(opt)>(
            grad_ptr, dev_ptr, uvm_ptr, index_ptr, opt, ev_nums, dim, stride,
            split_index, device_id);
      });
    });
  });
}

void adam_update_for_combined_table(at::Tensor grads, at::Tensor indices,
                                    std::optional<at::Tensor> dev_table,
                                    std::optional<at::Tensor> uvm_table,
                                    int64_t state_dim, const float lr,
                                    const float beta1, const float beta2,
                                    const float eps, const float weight_decay,
                                    const uint32_t iter_num) {
  int64_t ev_nums = grads.size(0);
  int64_t dim = grads.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  DataType val_type;
  int64_t stride = -1;
  if ((not dev_table.has_value()) and (not uvm_table.has_value())) {
    throw std::runtime_error("Two tables cannot both be None.");
  } else {
    if (dev_table.has_value()) {
      val_type = get_data_type(dev_table.value());
      stride = dev_table.value().size(1);
    } else {
      val_type = get_data_type(uvm_table.value());
      stride = uvm_table.value().size(1);
    }
  }

  auto grad_type = get_data_type(grads);
  auto index_type = get_data_type(indices);

  int64_t split_index = 0;
  if (dev_table.has_value()) {
    split_index = dev_table.value().size(0);
  }
  int device_id = grads.device().index();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      DISPATCH_OFFSET_INT_TYPE(index_type, i_t, [&] {
        auto grad_ptr = get_pointer<g_t>(grads);
        auto dev_ptr = get_pointer<w_t>(dev_table);
        auto uvm_ptr = get_pointer<w_t>(uvm_table);

        auto index_ptr = get_pointer<i_t>(indices);

        AdamVecOptimizer<g_t, w_t> opt{lr,  beta1,        beta2,
                                       eps, weight_decay, iter_num};

        launch_update_kernel_for_combined_table<g_t, w_t, i_t, decltype(opt)>(
            grad_ptr, dev_ptr, uvm_ptr, index_ptr, opt, ev_nums, dim, stride,
            split_index, device_id);
      });
    });
  });
}

void adagrad_update_for_combined_table(at::Tensor grads, at::Tensor indices,
                                       std::optional<at::Tensor> dev_table,
                                       std::optional<at::Tensor> uvm_table,
                                       int64_t state_dim, const float lr,
                                       const float eps) {

  int64_t ev_nums = grads.size(0);
  int64_t dim = grads.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  DataType val_type;
  int64_t stride = -1;
  if ((not dev_table.has_value()) and (not uvm_table.has_value())) {
    throw std::runtime_error("Two tables cannot both be None.");
  } else {
    if (dev_table.has_value()) {
      val_type = get_data_type(dev_table.value());
      stride = dev_table.value().size(1);
    } else {
      val_type = get_data_type(uvm_table.value());
      stride = uvm_table.value().size(1);
    }
  }

  auto grad_type = get_data_type(grads);
  auto index_type = get_data_type(indices);

  int64_t split_index = 0;
  if (dev_table.has_value()) {
    split_index = dev_table.value().size(0);
  }

  int device_id = grads.device().index();
  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      DISPATCH_OFFSET_INT_TYPE(index_type, i_t, [&] {
        auto grad_ptr = get_pointer<g_t>(grads);
        auto dev_ptr = get_pointer<w_t>(dev_table);
        auto uvm_ptr = get_pointer<w_t>(uvm_table);

        auto index_ptr = get_pointer<i_t>(indices);

        AdaGradVecOptimizer<g_t, w_t> opt{lr, eps};

        launch_update_kernel_for_combined_table<g_t, w_t, i_t, decltype(opt)>(
            grad_ptr, dev_ptr, uvm_ptr, index_ptr, opt, ev_nums, dim, stride,
            split_index, device_id);
      });
    });
  });
}

void rowwise_adagrad_for_combined_table(at::Tensor grads, at::Tensor indices,
                                        std::optional<at::Tensor> dev_table,
                                        std::optional<at::Tensor> uvm_table,
                                        int64_t state_dim, const float lr,
                                        const float eps) {

  int64_t ev_nums = grads.size(0);
  int64_t dim = grads.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  DataType val_type;
  int64_t stride = -1;
  if ((not dev_table.has_value()) and (not uvm_table.has_value())) {
    throw std::runtime_error("Two tables cannot both be None.");
  } else {
    if (dev_table.has_value()) {
      val_type = get_data_type(dev_table.value());
      stride = dev_table.value().size(1);
    } else {
      val_type = get_data_type(uvm_table.value());
      stride = uvm_table.value().size(1);
    }
  }

  auto grad_type = get_data_type(grads);
  auto index_type = get_data_type(indices);

  int64_t split_index = 0;
  if (dev_table.has_value()) {
    split_index = dev_table.value().size(0);
  }

  int device_id = grads.device().index();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      DISPATCH_OFFSET_INT_TYPE(index_type, i_t, [&] {
        auto grad_ptr = get_pointer<g_t>(grads);
        auto dev_ptr = get_pointer<w_t>(dev_table);
        auto uvm_ptr = get_pointer<w_t>(uvm_table);

        auto index_ptr = get_pointer<i_t>(indices);

        RowWiseAdaGradVecOptimizer<g_t, w_t> opt{lr, eps};

        launch_update_kernel_for_combined_table<g_t, w_t, i_t, decltype(opt)>(
            grad_ptr, dev_ptr, uvm_ptr, index_ptr, opt, ev_nums, dim, stride,
            split_index, device_id,
            [](int block_size) { return block_size * sizeof(float); });
      });
    });
  });
}

template <typename opt_t, typename g_t, typename w_t>
void fused_update(const opt_t &opt, at::Tensor grads, at::Tensor values) {

  int64_t ev_nums = grads.size(0);
  int64_t dim = grads.size(1);
  int64_t val_dim = values.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto &device_prop = DeviceProp::getDeviceProp(grads.device().index());

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

    auto kernel = update4_kernel_fused<g_t, w_t, opt_t>;
    kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
        ev_nums, dim, val_dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
        reinterpret_cast<w_t *>(values.data_ptr()), nullptr, opt);
  } else {
    int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
    int grid_size = ev_nums;

    auto kernel = update_kernel_fused<g_t, w_t, opt_t>;
    kernel<<<grid_size, block_size, 0, stream>>>(
        ev_nums, dim, val_dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
        reinterpret_cast<w_t *>(values.data_ptr()), nullptr, opt);
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void dynamic_emb_sgd_fused(at::Tensor grads, at::Tensor values,
                           float const lr) {

  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
  auto val_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      SgdVecOptimizer<g_t, w_t> opt{lr};
      fused_update<decltype(opt), g_t, w_t>(opt, grads, values);
    });
  });
}

void dynamic_emb_adam_fused(at::Tensor grads, at::Tensor values, const float lr,
                            const float beta1, const float beta2,
                            const float eps, const float weight_decay,
                            const uint32_t iter_num) {
  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
  auto val_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      AdamVecOptimizer<g_t, w_t> opt{lr,  beta1,        beta2,
                                     eps, weight_decay, iter_num};
      fused_update<decltype(opt), g_t, w_t>(opt, grads, values);
    });
  });
}

void dynamic_emb_adagrad_fused(at::Tensor grads, at::Tensor values,
                               const float lr, const float eps) {

  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
  auto val_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      AdaGradVecOptimizer<g_t, w_t> opt{lr, eps};
      fused_update<decltype(opt), g_t, w_t>(opt, grads, values);
    });
  });
}

void dynamic_emb_rowwise_adagrad_fused(at::Tensor grads, at::Tensor values,
                                       const float lr, const float eps) {

  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
  auto val_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      RowWiseAdaGradVecOptimizer<g_t, w_t> opt{lr, eps};
      fused_update<decltype(opt), g_t, w_t>(opt, grads, values);
    });
  });
}

} // namespace dyn_emb

// PYTHON WRAP
void bind_optimizer_kernel_op(py::module &m) {
  m.def("dynamic_emb_sgd_with_table", &dyn_emb::dynamic_emb_sgd_with_table,
        "SGD optimizer for Dynamic Emb", py::arg("table"), py::arg("n"),
        py::arg("indices"), py::arg("grads"), py::arg("lr"),
        py::arg("weight_type"));

  m.def("dynamic_emb_adam_with_table", &dyn_emb::dynamic_emb_adam_with_table,
        "Adam optimizer for Dynamic Emb", py::arg("ht"), py::arg("n"),
        py::arg("indices"), py::arg("grads"), py::arg("lr"), py::arg("beta1"),
        py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"),
        py::arg("iter_num"), py::arg("weight_type"));

  m.def("dynamic_emb_adagrad_with_table",
        &dyn_emb::dynamic_emb_adagrad_with_table,
        "Adagrad optimizer for Dynamic Emb", py::arg("ht"), py::arg("n"),
        py::arg("indices"), py::arg("grads"), py::arg("lr"), py::arg("eps"),
        py::arg("weight_type"));

  m.def("dynamic_emb_rowwise_adagrad_with_table",
        &dyn_emb::dynamic_emb_rowwise_adagrad_with_table,
        "Row Wise Adagrad optimizer for Dynamic Emb", py::arg("ht"),
        py::arg("n"), py::arg("indices"), py::arg("grads"), py::arg("lr"),
        py::arg("eps"), py::arg("weight_type"));

  m.def("dynamic_emb_sgd_with_pointer", &dyn_emb::dynamic_emb_sgd_with_pointer,
        "SGD optimizer for Dynamic Emb", py::arg("grads"),
        py::arg("val_pointers"), py::arg("val_dtype"), py::arg("lr"));

  m.def("dynamic_emb_adam_with_pointer",
        &dyn_emb::dynamic_emb_adam_with_pointer,
        "Adam optimizer for Dynamic Emb", py::arg("grads"),
        py::arg("val_pointers"), py::arg("val_dtype"), py::arg("state_dim"),
        py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
        py::arg("weight_decay"), py::arg("iter_num"));

  m.def("dynamic_emb_adagrad_with_pointer",
        &dyn_emb::dynamic_emb_adagrad_with_pointer,
        "Adagrad optimizer for Dynamic Emb", py::arg("grads"),
        py::arg("val_pointers"), py::arg("val_dtype"), py::arg("state_dim"),
        py::arg("lr"), py::arg("eps"));

  m.def("dynamic_emb_rowwise_adagrad_with_pointer",
        &dyn_emb::dynamic_emb_rowwise_adagrad_with_pointer,
        "Row Wise Adagrad optimizer for Dynamic Emb", py::arg("grads"),
        py::arg("val_pointers"), py::arg("val_dtype"), py::arg("state_dim"),
        py::arg("lr"), py::arg("eps"));

  m.def("dynamic_emb_sgd_fused", &dyn_emb::dynamic_emb_sgd_fused,
        "SGD optimizer for Dynamic Emb", py::arg("grads"), py::arg("values"),
        py::arg("lr"));

  m.def("dynamic_emb_adam_fused", &dyn_emb::dynamic_emb_adam_fused,
        "Adam optimizer for Dynamic Emb", py::arg("grads"), py::arg("values"),
        py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
        py::arg("weight_decay"), py::arg("iter_num"));

  m.def("dynamic_emb_adagrad_fused", &dyn_emb::dynamic_emb_adagrad_fused,
        "Adagrad optimizer for Dynamic Emb", py::arg("grads"),
        py::arg("values"), py::arg("lr"), py::arg("eps"));

  m.def("dynamic_emb_rowwise_adagrad_fused",
        &dyn_emb::dynamic_emb_rowwise_adagrad_fused,
        "Row Wise Adagrad optimizer for Dynamic Emb", py::arg("grads"),
        py::arg("values"), py::arg("lr"), py::arg("eps"));

  m.def("sgd_update_for_combined_table",
        &dyn_emb::sgd_update_for_combined_table,
        "SGD optimizer for Dynamic Emb", py::arg("grads"), py::arg("indices"),
        py::arg("dev_table"), py::arg("uvm_table"), py::arg("lr"));

  m.def("adam_update_for_combined_table",
        &dyn_emb::adam_update_for_combined_table,
        "Adam optimizer for Dynamic Emb", py::arg("grads"), py::arg("indices"),
        py::arg("dev_table"), py::arg("uvm_table"), py::arg("state_dim"),
        py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
        py::arg("weight_decay"), py::arg("iter_num"));

  m.def("adagrad_update_for_combined_table",
        &dyn_emb::adagrad_update_for_combined_table,
        "Adagrad optimizer for Dynamic Emb", py::arg("grads"),
        py::arg("indices"), py::arg("dev_table"), py::arg("uvm_table"),
        py::arg("state_dim"), py::arg("lr"), py::arg("eps"));

  m.def("rowwise_adagrad_for_combined_table",
        &dyn_emb::rowwise_adagrad_for_combined_table,
        "Row Wise Adagrad optimizer for Dynamic Emb", py::arg("grads"),
        py::arg("indices"), py::arg("dev_table"), py::arg("uvm_table"),
        py::arg("state_dim"), py::arg("lr"), py::arg("eps"));
}
