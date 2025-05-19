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

void find_or_insert_pointers(
  std::shared_ptr<dyn_emb::DynamicVariableBase> table,
  const size_t n,
  const at::Tensor keys,
  at::Tensor values,
  at::Tensor founds,
  const std::optional<uint64_t> score = std::nullopt,
  bool unique_key = true,
  bool ignore_evict_strategy = false);

namespace dyn_emb {

constexpr int MULTIPLIER = 4;
constexpr int WARPSIZE = 32;
constexpr int OPTIMIZER_BLOCKSIZE_VEC = 64;
constexpr int OPTIMIZER_BLOCKSIZE = 1024;

void dynamic_emb_sgd_with_table(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table, const uint64_t n, 
    const at::Tensor indices, const at::Tensor grads, const float lr, DataType weight_type, 
    const std::optional<uint64_t> score) {

  if (n == 0) return;
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  at::Tensor founds = at::empty({static_cast<int64_t>(n)}, 
                                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor weight_ptrs = at::empty({static_cast<int64_t>(n)}, 
                                     at::TensorOptions().dtype(at::kLong).device(indices.device()));

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  find_or_insert_pointers(table, n, indices, weight_ptrs, founds, score);

  auto &device_prop = DeviceProp::getDeviceProp(grads.device().index());

  int64_t dim = grads.size(1);
  int64_t ev_nums = weight_ptrs.size(0);

  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));

  auto initializer_args = table->get_initializer_args();
  auto* curand_states_ = table->get_curand_states();
  auto &initializer_ = initializer_args.mode;

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {
      
      SgdVecOptimizer<g_t, w_t> opt{lr, founds.data_ptr<bool>()};
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

        if (initializer_ == "normal") {
          using Generator = NormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "truncated_normal") {
          using Generator = TruncatedNormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev, initializer_args.lower, initializer_args.upper};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "uniform") {
          using Generator = UniformEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.lower, initializer_args.upper};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "debug") {
          using Generator = MappingEmbeddingGenerator<int64_t>;
          using Args = typename Generator::Args;
          auto args = Args {reinterpret_cast<const int64_t *>(indices.data_ptr()), 100000};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "constant") {
          using Generator = ConstEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {initializer_args.value};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else {
          throw std::runtime_error("Unrecognized initializer {" + initializer_ + "}");
        }
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;

        if (initializer_ == "normal") {
          using Generator = NormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "truncated_normal") {
          using Generator = TruncatedNormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev, initializer_args.lower, initializer_args.upper};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "uniform") {
          using Generator = UniformEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.lower, initializer_args.upper};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "debug") {
          using Generator = MappingEmbeddingGenerator<int64_t>;
          using Args = typename Generator::Args;
          auto args = Args {reinterpret_cast<const int64_t *>(indices.data_ptr()), 100000};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "constant") {
          using Generator = ConstEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {initializer_args.value};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else {
          throw std::runtime_error("Unrecognized initializer {" + initializer_ + "}");
        }
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void dynamic_emb_adam_with_table(
  std::shared_ptr<dyn_emb::DynamicVariableBase> ht,
  std::shared_ptr<dyn_emb::DynamicVariableBase> m_ht,
  std::shared_ptr<dyn_emb::DynamicVariableBase> v_ht,
  const uint64_t n, const at::Tensor indices, const at::Tensor grads, 
  const float lr, const float beta1, const float beta2, const float eps,
  const float weight_decay,
  const uint32_t iter_num, DataType weight_type, 
  const std::optional<uint64_t> score) {

  if (n == 0) return;
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  at::Tensor founds = at::empty({static_cast<int64_t>(n)}, 
                                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor founds_m = at::empty({static_cast<int64_t>(n)}, 
                                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor founds_v = at::empty({static_cast<int64_t>(n)}, 
                                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor weight_ptrs = at::empty({static_cast<int64_t>(n)}, 
                                     at::TensorOptions().dtype(at::kLong).device(indices.device()));
  at::Tensor m = at::empty({static_cast<int64_t>(n)}, 
                                     at::TensorOptions().dtype(at::kLong).device(indices.device()));
  at::Tensor v = at::empty({static_cast<int64_t>(n)}, 
                                     at::TensorOptions().dtype(at::kLong).device(indices.device()));

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  find_or_insert_pointers(ht, n, indices, weight_ptrs, founds, score);
  find_or_insert_pointers(m_ht, n, indices, m, founds_m, score);
  find_or_insert_pointers(v_ht, n, indices, v, founds_v, score);

  auto &device_prop = DeviceProp::getDeviceProp(grads.device().index());

  int64_t dim = grads.size(1);
  int64_t ev_nums = n;

  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
  auto initializer_args = ht->get_initializer_args();
  auto* curand_states_ = ht->get_curand_states();
  auto &initializer_ = initializer_args.mode;

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {
      AdamVecOptimizer<g_t, w_t> opt{reinterpret_cast<w_t **>(m.data_ptr()),
                                     reinterpret_cast<w_t **>(v.data_ptr()),
                                     lr,
                                     beta1,
                                     beta2,
                                     eps,
                                     weight_decay,
                                     iter_num,
                                     founds.data_ptr<bool>(),
                                     founds_m.data_ptr<bool>(),
                                     founds_v.data_ptr<bool>()};
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

        if (initializer_ == "normal") {
          using Generator = NormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "truncated_normal") {
          using Generator = TruncatedNormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev, initializer_args.lower, initializer_args.upper};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "uniform") {
          using Generator = UniformEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.lower, initializer_args.upper};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "debug") {
          using Generator = MappingEmbeddingGenerator<int64_t>;
          using Args = typename Generator::Args;
          auto args = Args {reinterpret_cast<const int64_t *>(indices.data_ptr()), 100000};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "constant") {
          using Generator = ConstEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {initializer_args.value};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else {
          throw std::runtime_error("Unrecognized initializer {" + initializer_ + "}");
        }
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;

        if (initializer_ == "normal") {
          using Generator = NormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "truncated_normal") {
          using Generator = TruncatedNormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev, initializer_args.lower, initializer_args.upper};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "uniform") {
          using Generator = UniformEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.lower, initializer_args.upper};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "debug") {
          using Generator = MappingEmbeddingGenerator<int64_t>;
          using Args = typename Generator::Args;
          auto args = Args {reinterpret_cast<const int64_t *>(indices.data_ptr()), 100000};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "constant") {
          using Generator = ConstEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {initializer_args.value};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else {
          throw std::runtime_error("Unrecognized initializer {" + initializer_ + "}");
        }
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void dynamic_emb_adagrad_with_table(
  std::shared_ptr<dyn_emb::DynamicVariableBase> ht,
  std::shared_ptr<dyn_emb::DynamicVariableBase> gt_ht,
  const uint64_t n, const at::Tensor indices,
  const at::Tensor grads,
  const float lr,
  const float eps,
  DataType weight_type,const std::optional<uint64_t> score){
  if (n == 0) return;

  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  at::Tensor founds = at::empty({static_cast<int64_t>(n)}, 
                                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor founds_gt = at::empty({static_cast<int64_t>(n)}, 
                                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor weight_ptrs = at::empty({static_cast<int64_t>(n)}, 
                                     at::TensorOptions().dtype(at::kLong).device(indices.device()));
  at::Tensor gt = at::empty({static_cast<int64_t>(n)}, 
                                     at::TensorOptions().dtype(at::kLong).device(indices.device()));

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  find_or_insert_pointers(ht, n, indices, weight_ptrs, founds, score);
  find_or_insert_pointers(gt_ht, n, indices, gt, founds_gt, score);

  auto& device_prop = DeviceProp::getDeviceProp(grads.device().index());

  int64_t dim = grads.size(1);
  int64_t ev_nums = n;

  auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
  auto initializer_args = ht->get_initializer_args();
  auto* curand_states_ = ht->get_curand_states();
  auto &initializer_ = initializer_args.mode;

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {

      AdaGradVecOptimizer<g_t,w_t> opt{reinterpret_cast<w_t**>(gt.data_ptr()),lr, eps,
                                     founds.data_ptr<bool>(),
                                     founds_gt.data_ptr<bool>()};

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

        if (initializer_ == "normal") {
          using Generator = NormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "truncated_normal") {
          using Generator = TruncatedNormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev, initializer_args.lower, initializer_args.upper};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "uniform") {
          using Generator = UniformEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.lower, initializer_args.upper};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "debug") {
          using Generator = MappingEmbeddingGenerator<int64_t>;
          using Args = typename Generator::Args;
          auto args = Args {reinterpret_cast<const int64_t *>(indices.data_ptr()), 100000};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "constant") {
          using Generator = ConstEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {initializer_args.value};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else {
          throw std::runtime_error("Unrecognized initializer {" + initializer_ + "}");
        }
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();

      } else {

        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;

        if (initializer_ == "normal") {
          using Generator = NormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "truncated_normal") {
          using Generator = TruncatedNormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev, initializer_args.lower, initializer_args.upper};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "uniform") {
          using Generator = UniformEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.lower, initializer_args.upper};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "debug") {
          using Generator = MappingEmbeddingGenerator<int64_t>;
          using Args = typename Generator::Args;
          auto args = Args {reinterpret_cast<const int64_t *>(indices.data_ptr()), 100000};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "constant") {
          using Generator = ConstEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {initializer_args.value};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else {
          throw std::runtime_error("Unrecognized initializer {" + initializer_ + "}");
        }
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();

      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void dynamic_emb_rowwise_adagrad_with_table(
  std::shared_ptr<dyn_emb::DynamicVariableBase> ht,
  std::shared_ptr<dyn_emb::DynamicVariableBase> gt_ht,
  const uint64_t n, const at::Tensor indices,
  const at::Tensor grads,
  const float lr,
  const float eps,
  DataType weight_type,const std::optional<uint64_t> score) {
  if (n == 0) return;
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");

  at::Tensor founds = at::empty({static_cast<int64_t>(n)}, 
                                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor founds_gt = at::empty({static_cast<int64_t>(n)}, 
                                at::TensorOptions().dtype(at::kBool).device(indices.device()));
  at::Tensor weight_ptrs = at::empty({static_cast<int64_t>(n)}, 
                                     at::TensorOptions().dtype(at::kLong).device(indices.device()));
  at::Tensor gt = at::empty({static_cast<int64_t>(n)}, 
                                     at::TensorOptions().dtype(at::kLong).device(indices.device()));

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  find_or_insert_pointers(ht, n, indices, weight_ptrs, founds, score);
  find_or_insert_pointers(gt_ht, n, indices, gt, founds_gt, score);

  auto& device_prop = DeviceProp::getDeviceProp(grads.device().index());

  int64_t dim = grads.size(1);
  int64_t ev_nums = n;

  auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
  auto initializer_args = ht->get_initializer_args();
  auto* curand_states_ = ht->get_curand_states();
  auto &initializer_ = initializer_args.mode;

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(weight_type, w_t, [&] {

      RowWiseAdaGradVecOptimizer<g_t,w_t> opt{reinterpret_cast<w_t**>(gt.data_ptr()),lr, eps,
                                     founds.data_ptr<bool>(),
                                     founds_gt.data_ptr<bool>()};

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

        if (initializer_ == "normal") {
          using Generator = NormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "truncated_normal") {
          using Generator = TruncatedNormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev, initializer_args.lower, initializer_args.upper};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "uniform") {
          using Generator = UniformEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.lower, initializer_args.upper};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "debug") {
          using Generator = MappingEmbeddingGenerator<int64_t>;
          using Args = typename Generator::Args;
          auto args = Args {reinterpret_cast<const int64_t *>(indices.data_ptr()), 100000};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "constant") {
          using Generator = ConstEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {initializer_args.value};
          auto kernel = update4_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
            reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else {
          throw std::runtime_error("Unrecognized initializer {" + initializer_ + "}");
        }
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();

      } else {

        int block_size = dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : dim;
        int grid_size = ev_nums;
        int shared_memory_bytes = block_size*sizeof(float);

        if (initializer_ == "normal") {
          using Generator = NormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, shared_memory_bytes, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "truncated_normal") {
          using Generator = TruncatedNormalEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.mean, initializer_args.std_dev, initializer_args.lower, initializer_args.upper};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, shared_memory_bytes, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "uniform") {
          using Generator = UniformEmbeddingGenerator;
          using Args = typename Generator::Args;
          auto args = Args {curand_states_, initializer_args.lower, initializer_args.upper};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, shared_memory_bytes, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "debug") {
          using Generator = MappingEmbeddingGenerator<int64_t>;
          using Args = typename Generator::Args;
          auto args = Args {reinterpret_cast<const int64_t *>(indices.data_ptr()), 100000};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, shared_memory_bytes, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else if (initializer_ == "constant") {
          using Generator = ConstEmbeddingGenerator;
         using Args = typename Generator::Args;
          auto args = Args {initializer_args.value};
          auto kernel = update_kernel_with_generator<g_t, w_t, decltype(opt), Generator, Args>;
          kernel<<<grid_size, block_size, shared_memory_bytes, stream>>>(
              ev_nums, dim, reinterpret_cast<const g_t *>(grads.data_ptr()),
              reinterpret_cast<w_t **>(weight_ptrs.data_ptr()), opt, args);
        } else {
          throw std::runtime_error("Unrecognized initializer {" + initializer_ + "}");
        }
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
        py::arg("lr"), py::arg("weight_type"), py::arg("score") = py::none());

  m.def("dynamic_emb_adam_with_table", &dyn_emb::dynamic_emb_adam_with_table,
        "Adam optimizer for Dynamic Emb", py::arg("ht"), py::arg("m_ht"), py::arg("v_ht"),
        py::arg("n"), py::arg("indices"), py::arg("grads"),
        py::arg("lr"), py::arg("beta1"),
        py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"), py::arg("iter_num"),
        py::arg("weight_type"), py::arg("score") = py::none());

  m.def("dynamic_emb_adagrad_with_table", &dyn_emb::dynamic_emb_adagrad_with_table,
        "Adagrad optimizer for Dynamic Emb", py::arg("ht"), py::arg("gt_ht"), 
        py::arg("n"), py::arg("indices"), py::arg("grads"),py::arg("lr"),
        py::arg("eps"),
        py::arg("weight_type"), py::arg("score") = py::none());

  m.def("dynamic_emb_rowwise_adagrad_with_table", &dyn_emb::dynamic_emb_rowwise_adagrad_with_table,
        "Row Wise Adagrad optimizer for Dynamic Emb", py::arg("ht"), py::arg("gt_ht"),
        py::arg("n"), py::arg("indices"), py::arg("grads"),py::arg("lr"),
        py::arg("eps"),
        py::arg("weight_type"), py::arg("score") = py::none());
}
