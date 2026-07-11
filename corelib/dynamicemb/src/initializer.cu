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

#include "initializer.cuh"

#include <c10/cuda/CUDAGuard.h>

namespace py = pybind11;

namespace dyn_emb {

__global__ void init_curand_state_kernel(unsigned long long seed,
                                         curandState *states) {
  auto grid = cooperative_groups::this_grid();
  curand_init(seed, grid.thread_rank(), 0, &states[grid.thread_rank()]);
}

class CurandStateContext {

public:
  CurandStateContext() {
    CUDACHECK(cudaGetDevice(&device_id_));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto &deviceProp = DeviceProp::getDeviceProp();
    num_worker_ = deviceProp.total_threads;
    CUDACHECK(
        cudaMallocAsync(&states_, sizeof(curandState) * num_worker_, stream));
    std::random_device rd;
    auto seed = rd();
    int block_size = deviceProp.max_thread_per_block;
    int grid_size = num_worker_ / block_size;
    init_curand_state_kernel<<<grid_size, block_size, 0, stream>>>(seed,
                                                                   states_);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  }

  ~CurandStateContext() {
    // not async to avoid stream destroy case.
    c10::cuda::CUDAGuard device_guard(
        c10::Device(c10::DeviceType::CUDA, device_id_));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaFree(states_));
  }

  int64_t num_worker() { return num_worker_; }

  int device_id() const { return device_id_; }

  curandState *ptr() { return states_; }

private:
  curandState *states_;
  int64_t num_worker_;
  int device_id_;
};

template <typename ValueT, typename IndexT, typename GeneratorT>
__global__ void initialize_with_index_addressor_kernel(
    int64_t num, int64_t dim, int64_t stride, ValueT *__restrict__ buffer,
    IndexT const *__restrict__ indices,
    typename GeneratorT::Args generator_args) {

  GeneratorT gen(generator_args);
  int64_t num_task = num * dim;
  int64_t task_id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; task_id < num_task; task_id += gridDim.x * blockDim.x) {
    int64_t emb_id = task_id / dim;
    int64_t index = indices[emb_id];
    ValueT *dst = buffer + index * stride;
    auto tmp = gen.generate(index);
    dst[task_id % dim] = TypeConvertFunc<ValueT, float>::convert(tmp);
  }
  gen.destroy();
}

template <typename ValueT, typename GeneratorT>
__global__ void initialize_with_mask_addressor_kernel(
    int64_t num, int64_t dim, int64_t stride, ValueT *__restrict__ buffer,
    bool const *__restrict__ mask, typename GeneratorT::Args generator_args) {

  GeneratorT gen(generator_args);
  int64_t num_task = num * dim;
  int64_t task_id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; task_id < num_task; task_id += gridDim.x * blockDim.x) {
    int64_t emb_id = task_id / dim;
    if (!mask[emb_id]) {
      continue;
    }
    ValueT *dst = buffer + emb_id * stride;
    auto tmp = gen.generate(emb_id);
    dst[task_id % dim] = TypeConvertFunc<ValueT, float>::convert(tmp);
  }
  gen.destroy();
}

template <typename GeneratorT>
void initialize_with_generator(at::Tensor buffer, at::Tensor indices,
                               typename GeneratorT::Args generator_args,
                               int64_t num_worker = -1) {
  int64_t num_dims = buffer.dim();
  if (num_dims != 2) {
    throw std::runtime_error("Initializer'input buffer's dim have to be 2.");
  }
  if (buffer.stride(1) != 1) {
    throw std::runtime_error(
        "Initializer'input buffer has to be contiguous at dim1.");
  }
  if (indices.dim() != 1 || !indices.is_contiguous()) {
    throw std::runtime_error(
        "Initializer selector must be a contiguous one-dimensional tensor.");
  }
  int64_t num_total = indices.size(0);
  int64_t dim = buffer.size(1);
  int64_t stride = buffer.stride(0);

  bool use_mask = indices.scalar_type() == at::ScalarType::Bool;
  if (use_mask && num_total != buffer.size(0)) {
    throw std::runtime_error(
        "Boolean initializer selector must match buffer.size(0).");
  }

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto &deviceProp = DeviceProp::getDeviceProp();

  int64_t block_size = deviceProp.max_thread_per_block;
  int64_t num_need = num_total * dim;
  if (num_need == 0) {
    return;
  }
  if (num_worker == -1) {
    num_worker = deviceProp.total_threads;
  }
  int64_t max_grid_size = num_worker / block_size;
  if (num_worker > num_need) {
    num_worker = num_need;
  }
  int64_t grid_size = (num_worker - 1) / block_size + 1;
  if (grid_size > max_grid_size) {
    grid_size = max_grid_size;
  }

  auto value_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(buffer.dtype()));
  DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, ValueType, [&] {
    if (use_mask) {
      initialize_with_mask_addressor_kernel<ValueType, GeneratorT>
          <<<grid_size, block_size, 0, stream>>>(
              num_total, dim, stride,
              reinterpret_cast<ValueType *>(buffer.data_ptr()),
              reinterpret_cast<bool const *>(indices.data_ptr()),
              generator_args);
    } else {
      auto index_type =
          scalartype_to_datatype(convertTypeMetaToScalarType(indices.dtype()));
      DISPATCH_INTEGER_DATATYPE_FUNCTION(index_type, IndexType, [&] {
        initialize_with_index_addressor_kernel<ValueType, IndexType, GeneratorT>
            <<<grid_size, block_size, 0, stream>>>(
                num_total, dim, stride,
                reinterpret_cast<ValueType *>(buffer.data_ptr()),
                reinterpret_cast<IndexType *>(indices.data_ptr()),
                generator_args);
      });
    }
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename ValueT, typename GeneratorT>
__global__ void initialize_flat_with_mask_kernel(
    int64_t num, int64_t const *__restrict__ table_ptrs,
    int64_t const *__restrict__ slot_indices,
    int64_t const *__restrict__ table_ids,
    int64_t const *__restrict__ table_value_dims,
    int64_t const *__restrict__ table_emb_dims, bool const *__restrict__ mask,
    float initial_optim_state, typename GeneratorT::Args generator_args) {
  GeneratorT gen(generator_args);

  for (int64_t row = blockIdx.x; row < num; row += gridDim.x) {
    if (!mask[row]) {
      continue;
    }

    int64_t slot = slot_indices[row];
    if (slot < 0) {
      continue;
    }
    int64_t table_id = table_ids[row];
    int64_t value_dim = table_value_dims[table_id];
    int64_t emb_dim = table_emb_dims[table_id];
    ValueT *dst = reinterpret_cast<ValueT *>(table_ptrs[table_id]) +
                  slot * value_dim;

    for (int64_t col = threadIdx.x; col < value_dim; col += blockDim.x) {
      float value =
          col < emb_dim ? gen.generate(row) : initial_optim_state;
      dst[col] = TypeConvertFunc<ValueT, float>::convert(value);
    }
  }
  gen.destroy();
}

template <typename GeneratorT>
void initialize_flat_with_generator(
    at::Tensor table_ptrs, at::Tensor slot_indices, at::Tensor table_ids,
    at::Tensor table_value_dims, at::Tensor table_emb_dims, at::Tensor mask,
    torch::Dtype value_dtype, float initial_optim_state,
    typename GeneratorT::Args generator_args, int64_t num_worker = -1) {
  TORCH_CHECK(table_ptrs.is_cuda(), "table_ptrs must be a CUDA tensor");
  TORCH_CHECK(slot_indices.is_cuda() && table_ids.is_cuda() && mask.is_cuda(),
              "slot_indices, table_ids, and mask must be CUDA tensors");
  TORCH_CHECK(table_value_dims.is_cuda() && table_emb_dims.is_cuda(),
              "table dimension tensors must be CUDA tensors");
  TORCH_CHECK(table_ptrs.dim() == 1 && slot_indices.dim() == 1 &&
                  table_ids.dim() == 1 && table_value_dims.dim() == 1 &&
                  table_emb_dims.dim() == 1 && mask.dim() == 1,
              "flat initializer inputs must be one-dimensional");
  TORCH_CHECK(table_ptrs.is_contiguous() && slot_indices.is_contiguous() &&
                  table_ids.is_contiguous() &&
                  table_value_dims.is_contiguous() &&
                  table_emb_dims.is_contiguous() && mask.is_contiguous(),
              "flat initializer inputs must be contiguous");
  TORCH_CHECK(table_ptrs.scalar_type() == at::ScalarType::Long &&
                  slot_indices.scalar_type() == at::ScalarType::Long &&
                  table_ids.scalar_type() == at::ScalarType::Long &&
                  table_value_dims.scalar_type() == at::ScalarType::Long &&
                  table_emb_dims.scalar_type() == at::ScalarType::Long,
              "flat initializer pointer, slot, table id, and dimension tensors "
              "must be int64");
  TORCH_CHECK(mask.scalar_type() == at::ScalarType::Bool,
              "flat initializer mask must be bool");
  TORCH_CHECK(slot_indices.numel() == table_ids.numel() &&
                  slot_indices.numel() == mask.numel(),
              "slot_indices, table_ids, and mask must have the same length");
  TORCH_CHECK(table_ptrs.numel() == table_value_dims.numel() &&
                  table_ptrs.numel() == table_emb_dims.numel(),
              "table_ptrs and table dimension tensors must have the same "
              "length");
  TORCH_CHECK(table_ptrs.get_device() == slot_indices.get_device() &&
                  table_ptrs.get_device() == table_ids.get_device() &&
                  table_ptrs.get_device() == table_value_dims.get_device() &&
                  table_ptrs.get_device() == table_emb_dims.get_device() &&
                  table_ptrs.get_device() == mask.get_device(),
              "flat initializer inputs must be on the same CUDA device");

  int64_t num_total = slot_indices.numel();
  if (num_total == 0) {
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto &device_prop = DeviceProp::getDeviceProp();
  int64_t block_size = device_prop.max_thread_per_block < 256
                           ? device_prop.max_thread_per_block
                           : 256;
  if (num_worker == -1) {
    num_worker = device_prop.total_threads;
  }
  int64_t max_grid_size = num_worker / block_size;
  int64_t grid_size = num_total < max_grid_size ? num_total : max_grid_size;

  auto value_type = scalartype_to_datatype(toScalarType(value_dtype));
  DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, ValueType, [&] {
    initialize_flat_with_mask_kernel<ValueType, GeneratorT>
        <<<grid_size, block_size, 0, stream>>>(
            num_total, get_pointer<int64_t>(table_ptrs),
            get_pointer<int64_t>(slot_indices), get_pointer<int64_t>(table_ids),
            get_pointer<int64_t>(table_value_dims),
            get_pointer<int64_t>(table_emb_dims), get_pointer<bool>(mask),
            initial_optim_state, generator_args);
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename ValueT, typename GeneratorT, bool Vec4>
__global__ void load_or_initialize_mixed_kernel(
    int64_t num, int64_t output_stride, ValueT *__restrict__ output,
    int64_t const *__restrict__ cache_table_ptrs,
    int64_t const *__restrict__ cache_rows,
    int64_t const *__restrict__ storage_table_ptrs,
    int64_t const *__restrict__ storage_rows,
    int64_t const *__restrict__ table_ids,
    int64_t const *__restrict__ table_value_dims,
    int64_t const *__restrict__ table_emb_dims,
    bool const *__restrict__ preinitialized_mask,
    typename GeneratorT::Args generator_args) {
  constexpr int kWarpSize = 32;
  constexpr int kVecSize = 4;
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  GeneratorT gen(generator_args);

  for (int64_t row =
           static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
       row < num; row += static_cast<int64_t>(gridDim.x) * warps_per_block) {
    int64_t source_row = cache_rows[row];
    int64_t const *source_table_ptrs = cache_table_ptrs;
    if (source_row < 0) {
      source_row = storage_rows[row];
      source_table_ptrs = storage_table_ptrs;
    }

    const int64_t table_id = table_ids[row];
    const int64_t value_dim = table_value_dims[table_id];
    const int64_t emb_dim = table_emb_dims[table_id];
    ValueT *dst = output + row * output_stride;

    if (source_row >= 0) {
      ValueT const *src =
          reinterpret_cast<ValueT const *>(source_table_ptrs[table_id]) +
          source_row * value_dim;
      if constexpr (Vec4) {
        for (int64_t col = static_cast<int64_t>(lane) * kVecSize;
             col < emb_dim;
             col += static_cast<int64_t>(kWarpSize) * kVecSize) {
          Vec4T<ValueT> value;
          value.load(src + col);
          value.store(dst + col);
        }
      } else {
        for (int64_t col = lane; col < emb_dim; col += kWarpSize)
          dst[col] = src[col];
      }
    } else if (preinitialized_mask == nullptr ||
               !preinitialized_mask[row]) {
      // A globally-new row that could not be provisioned in cache is
      // transient for this iteration. Generate directly into the final
      // unique-embedding buffer; a later batch retries cache provisioning.
      for (int64_t col = lane; col < emb_dim; col += kWarpSize) {
        dst[col] = TypeConvertFunc<ValueT, float>::convert(gen.generate(row));
      }
    }
  }
  gen.destroy();
}

template <typename GeneratorT>
void load_or_initialize_mixed_with_generator(
    at::Tensor output, at::Tensor cache_table_ptrs, at::Tensor cache_rows,
    at::Tensor storage_table_ptrs, at::Tensor storage_rows,
    at::Tensor table_ids, at::Tensor table_value_dims,
    at::Tensor table_emb_dims, bool all_dims_vec4,
    std::optional<at::Tensor> preinitialized_mask,
    typename GeneratorT::Args generator_args, int64_t num_worker = -1) {
  TORCH_CHECK(output.is_cuda() && cache_table_ptrs.is_cuda() &&
                  cache_rows.is_cuda() && storage_table_ptrs.is_cuda() &&
                  storage_rows.is_cuda() && table_ids.is_cuda() &&
                  table_value_dims.is_cuda() && table_emb_dims.is_cuda(),
              "mixed load/initialize inputs must be CUDA tensors");
  const c10::Device device = output.device();
  TORCH_CHECK(cache_table_ptrs.device() == device &&
                  cache_rows.device() == device &&
                  storage_table_ptrs.device() == device &&
                  storage_rows.device() == device &&
                  table_ids.device() == device &&
                  table_value_dims.device() == device &&
                  table_emb_dims.device() == device,
              "mixed load/initialize inputs must be on the output device");
  TORCH_CHECK(output.dim() == 2 && output.is_contiguous(),
              "mixed output must be a contiguous matrix");
  TORCH_CHECK(cache_rows.dim() == 1 && storage_rows.dim() == 1 &&
                  table_ids.dim() == 1,
              "mixed row metadata must be one-dimensional");
  TORCH_CHECK(cache_rows.numel() == output.size(0) &&
                  storage_rows.numel() == output.size(0) &&
                  table_ids.numel() == output.size(0),
              "mixed row metadata must match output rows");
  TORCH_CHECK(cache_table_ptrs.scalar_type() == at::ScalarType::Long &&
                  storage_table_ptrs.scalar_type() == at::ScalarType::Long &&
                  cache_rows.scalar_type() == at::ScalarType::Long &&
                  storage_rows.scalar_type() == at::ScalarType::Long &&
                  table_ids.scalar_type() == at::ScalarType::Long &&
                  table_value_dims.scalar_type() == at::ScalarType::Long &&
                  table_emb_dims.scalar_type() == at::ScalarType::Long,
              "mixed pointer, row, table, and dimension tensors must be int64");
  TORCH_CHECK(cache_table_ptrs.numel() == storage_table_ptrs.numel() &&
                  cache_table_ptrs.numel() == table_value_dims.numel() &&
                  cache_table_ptrs.numel() == table_emb_dims.numel(),
              "mixed table metadata must have matching lengths");

  bool const *preinitialized_mask_ptr = nullptr;
  if (preinitialized_mask.has_value() && preinitialized_mask->defined()) {
    TORCH_CHECK(preinitialized_mask->is_cuda() &&
                    preinitialized_mask->device() == device &&
                    preinitialized_mask->scalar_type() == at::kBool &&
                    preinitialized_mask->dim() == 1 &&
                    preinitialized_mask->is_contiguous() &&
                    preinitialized_mask->numel() == output.size(0),
                "preinitialized_mask must be a contiguous boolean vector "
                "on the output device with one entry per output row");
    preinitialized_mask_ptr = preinitialized_mask->data_ptr<bool>();
  }

  const int64_t num = output.size(0);
  if (num == 0)
    return;

  c10::cuda::CUDAGuard device_guard(device);
  constexpr int kBlockSize = 128;
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  auto &device_prop = DeviceProp::getDeviceProp();
  if (num_worker < 0)
    num_worker = device_prop.total_threads;
  int64_t max_grid_size = num_worker / kBlockSize;
  if (max_grid_size < 1)
    max_grid_size = 1;
  int64_t grid_size = (num + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (grid_size > max_grid_size)
    grid_size = max_grid_size;

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto value_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(output.dtype()));
  DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, ValueType, [&] {
    if (all_dims_vec4) {
      load_or_initialize_mixed_kernel<ValueType, GeneratorT, true>
          <<<grid_size, kBlockSize, 0, stream>>>(
              num, output.stride(0), get_pointer<ValueType>(output),
              get_pointer<int64_t>(cache_table_ptrs),
              get_pointer<int64_t>(cache_rows),
              get_pointer<int64_t>(storage_table_ptrs),
              get_pointer<int64_t>(storage_rows),
              get_pointer<int64_t>(table_ids),
              get_pointer<int64_t>(table_value_dims),
              get_pointer<int64_t>(table_emb_dims), preinitialized_mask_ptr,
              generator_args);
    } else {
      load_or_initialize_mixed_kernel<ValueType, GeneratorT, false>
          <<<grid_size, kBlockSize, 0, stream>>>(
              num, output.stride(0), get_pointer<ValueType>(output),
              get_pointer<int64_t>(cache_table_ptrs),
              get_pointer<int64_t>(cache_rows),
              get_pointer<int64_t>(storage_table_ptrs),
              get_pointer<int64_t>(storage_rows),
              get_pointer<int64_t>(table_ids),
              get_pointer<int64_t>(table_value_dims),
              get_pointer<int64_t>(table_emb_dims), preinitialized_mask_ptr,
              generator_args);
    }
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void normal_init_flat(
    at::Tensor table_ptrs, at::Tensor slot_indices, at::Tensor table_ids,
    at::Tensor table_value_dims, at::Tensor table_emb_dims, at::Tensor mask,
    torch::Dtype value_dtype, float initial_optim_state,
    CurandStateContext &curand_state_context, float mean, float std_dev) {
  using GeneratorT = NormalEmbeddingGenerator;
  auto generator_args =
      typename GeneratorT::Args{curand_state_context.ptr(), mean, std_dev};
  initialize_flat_with_generator<GeneratorT>(
      table_ptrs, slot_indices, table_ids, table_value_dims, table_emb_dims,
      mask, value_dtype, initial_optim_state, generator_args,
      curand_state_context.num_worker());
}

void truncated_normal_init_flat(
    at::Tensor table_ptrs, at::Tensor slot_indices, at::Tensor table_ids,
    at::Tensor table_value_dims, at::Tensor table_emb_dims, at::Tensor mask,
    torch::Dtype value_dtype, float initial_optim_state,
    CurandStateContext &curand_state_context, float mean, float std_dev,
    float lower, float upper) {
  using GeneratorT = TruncatedNormalEmbeddingGenerator;
  auto generator_args = typename GeneratorT::Args{
      curand_state_context.ptr(), mean, std_dev, lower, upper};
  initialize_flat_with_generator<GeneratorT>(
      table_ptrs, slot_indices, table_ids, table_value_dims, table_emb_dims,
      mask, value_dtype, initial_optim_state, generator_args,
      curand_state_context.num_worker());
}

void uniform_init_flat(
    at::Tensor table_ptrs, at::Tensor slot_indices, at::Tensor table_ids,
    at::Tensor table_value_dims, at::Tensor table_emb_dims, at::Tensor mask,
    torch::Dtype value_dtype, float initial_optim_state,
    CurandStateContext &curand_state_context, float lower, float upper) {
  using GeneratorT = UniformEmbeddingGenerator;
  auto generator_args = typename GeneratorT::Args{curand_state_context.ptr(),
                                                   lower, upper};
  initialize_flat_with_generator<GeneratorT>(
      table_ptrs, slot_indices, table_ids, table_value_dims, table_emb_dims,
      mask, value_dtype, initial_optim_state, generator_args,
      curand_state_context.num_worker());
}

void const_init_flat(at::Tensor table_ptrs, at::Tensor slot_indices,
                     at::Tensor table_ids, at::Tensor table_value_dims,
                     at::Tensor table_emb_dims, at::Tensor mask,
                     torch::Dtype value_dtype, float initial_optim_state,
                     float value) {
  using GeneratorT = ConstEmbeddingGenerator;
  auto generator_args = typename GeneratorT::Args{value};
  initialize_flat_with_generator<GeneratorT>(
      table_ptrs, slot_indices, table_ids, table_value_dims, table_emb_dims,
      mask, value_dtype, initial_optim_state, generator_args);
}

void debug_init_flat(at::Tensor table_ptrs, at::Tensor slot_indices,
                     at::Tensor table_ids, at::Tensor table_value_dims,
                     at::Tensor table_emb_dims, at::Tensor mask,
                     torch::Dtype value_dtype, float initial_optim_state,
                     at::Tensor keys) {
  TORCH_CHECK(keys.is_cuda() && keys.dim() == 1 && keys.is_contiguous(),
              "debug initializer keys must be a contiguous CUDA vector");
  TORCH_CHECK(keys.numel() == slot_indices.numel(),
              "debug initializer keys must match slot_indices length");
  TORCH_CHECK(keys.get_device() == slot_indices.get_device(),
              "debug initializer keys and slots must be on the same device");
  auto key_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(keys.dtype()));
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, KeyType, [&] {
    using GeneratorT = MappingEmbeddingGenerator<KeyType>;
    auto generator_args = typename GeneratorT::Args{
        reinterpret_cast<const KeyType *>(keys.data_ptr()), 100000};
    initialize_flat_with_generator<GeneratorT>(
        table_ptrs, slot_indices, table_ids, table_value_dims, table_emb_dims,
        mask, value_dtype, initial_optim_state, generator_args);
  });
}

void normal_load_or_initialize_mixed(
    at::Tensor output, at::Tensor cache_table_ptrs, at::Tensor cache_rows,
    at::Tensor storage_table_ptrs, at::Tensor storage_rows,
    at::Tensor table_ids, at::Tensor table_value_dims,
    at::Tensor table_emb_dims, bool all_dims_vec4,
    CurandStateContext &curand_state_context, float mean, float std_dev,
    std::optional<at::Tensor> preinitialized_mask) {
  TORCH_CHECK(output.is_cuda() &&
                  output.get_device() == curand_state_context.device_id(),
              "normal initializer state must be on the output device");
  using GeneratorT = NormalEmbeddingGenerator;
  auto generator_args =
      typename GeneratorT::Args{curand_state_context.ptr(), mean, std_dev};
  load_or_initialize_mixed_with_generator<GeneratorT>(
      output, cache_table_ptrs, cache_rows, storage_table_ptrs, storage_rows,
      table_ids, table_value_dims, table_emb_dims, all_dims_vec4,
      preinitialized_mask, generator_args, curand_state_context.num_worker());
}

void truncated_normal_load_or_initialize_mixed(
    at::Tensor output, at::Tensor cache_table_ptrs, at::Tensor cache_rows,
    at::Tensor storage_table_ptrs, at::Tensor storage_rows,
    at::Tensor table_ids, at::Tensor table_value_dims,
    at::Tensor table_emb_dims, bool all_dims_vec4,
    CurandStateContext &curand_state_context, float mean, float std_dev,
    float lower, float upper,
    std::optional<at::Tensor> preinitialized_mask) {
  TORCH_CHECK(output.is_cuda() &&
                  output.get_device() == curand_state_context.device_id(),
              "truncated-normal initializer state must be on the output "
              "device");
  using GeneratorT = TruncatedNormalEmbeddingGenerator;
  auto generator_args = typename GeneratorT::Args{
      curand_state_context.ptr(), mean, std_dev, lower, upper};
  load_or_initialize_mixed_with_generator<GeneratorT>(
      output, cache_table_ptrs, cache_rows, storage_table_ptrs, storage_rows,
      table_ids, table_value_dims, table_emb_dims, all_dims_vec4,
      preinitialized_mask, generator_args, curand_state_context.num_worker());
}

void uniform_load_or_initialize_mixed(
    at::Tensor output, at::Tensor cache_table_ptrs, at::Tensor cache_rows,
    at::Tensor storage_table_ptrs, at::Tensor storage_rows,
    at::Tensor table_ids, at::Tensor table_value_dims,
    at::Tensor table_emb_dims, bool all_dims_vec4,
    CurandStateContext &curand_state_context, float lower, float upper,
    std::optional<at::Tensor> preinitialized_mask) {
  TORCH_CHECK(output.is_cuda() &&
                  output.get_device() == curand_state_context.device_id(),
              "uniform initializer state must be on the output device");
  using GeneratorT = UniformEmbeddingGenerator;
  auto generator_args = typename GeneratorT::Args{curand_state_context.ptr(),
                                                   lower, upper};
  load_or_initialize_mixed_with_generator<GeneratorT>(
      output, cache_table_ptrs, cache_rows, storage_table_ptrs, storage_rows,
      table_ids, table_value_dims, table_emb_dims, all_dims_vec4,
      preinitialized_mask, generator_args, curand_state_context.num_worker());
}

void const_load_or_initialize_mixed(
    at::Tensor output, at::Tensor cache_table_ptrs, at::Tensor cache_rows,
    at::Tensor storage_table_ptrs, at::Tensor storage_rows,
    at::Tensor table_ids, at::Tensor table_value_dims,
    at::Tensor table_emb_dims, bool all_dims_vec4, float value,
    std::optional<at::Tensor> preinitialized_mask) {
  using GeneratorT = ConstEmbeddingGenerator;
  auto generator_args = typename GeneratorT::Args{value};
  load_or_initialize_mixed_with_generator<GeneratorT>(
      output, cache_table_ptrs, cache_rows, storage_table_ptrs, storage_rows,
      table_ids, table_value_dims, table_emb_dims, all_dims_vec4,
      preinitialized_mask, generator_args);
}

void debug_load_or_initialize_mixed(
    at::Tensor output, at::Tensor cache_table_ptrs, at::Tensor cache_rows,
    at::Tensor storage_table_ptrs, at::Tensor storage_rows,
    at::Tensor table_ids, at::Tensor table_value_dims,
    at::Tensor table_emb_dims, bool all_dims_vec4, at::Tensor keys,
    std::optional<at::Tensor> preinitialized_mask) {
  TORCH_CHECK(keys.is_cuda() && keys.dim() == 1 && keys.is_contiguous(),
              "debug initializer keys must be a contiguous CUDA vector");
  TORCH_CHECK(keys.numel() == cache_rows.numel(),
              "debug initializer keys must match mixed rows");
  TORCH_CHECK(keys.device() == output.device(),
              "debug initializer keys must be on the output device");
  auto key_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(keys.dtype()));
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, KeyType, [&] {
    using GeneratorT = MappingEmbeddingGenerator<KeyType>;
    auto generator_args = typename GeneratorT::Args{
        reinterpret_cast<const KeyType *>(keys.data_ptr()), 100000};
    load_or_initialize_mixed_with_generator<GeneratorT>(
        output, cache_table_ptrs, cache_rows, storage_table_ptrs, storage_rows,
        table_ids, table_value_dims, table_emb_dims, all_dims_vec4,
        preinitialized_mask, generator_args);
  });
}

void normal_init(at::Tensor buffer, at::Tensor indices,
                 CurandStateContext &curand_state_context, float mean,
                 float std_dev) {

  using GeneratorT = NormalEmbeddingGenerator;
  auto generator_args =
      typename GeneratorT::Args{curand_state_context.ptr(), mean, std_dev};
  int num_worker = curand_state_context.num_worker();
  initialize_with_generator<GeneratorT>(buffer, indices, generator_args,
                                        num_worker);
}

void truncated_normal_init(at::Tensor buffer, at::Tensor indices,
                           CurandStateContext &curand_state_context, float mean,
                           float std_dev, float lower, float upper) {
  using GeneratorT = TruncatedNormalEmbeddingGenerator;
  auto generator_args = typename GeneratorT::Args{curand_state_context.ptr(),
                                                  mean, std_dev, lower, upper};
  int num_worker = curand_state_context.num_worker();
  initialize_with_generator<GeneratorT>(buffer, indices, generator_args,
                                        num_worker);
}

void uniform_init(at::Tensor buffer, at::Tensor indices,
                  CurandStateContext &curand_state_context, float lower,
                  float upper) {
  using GeneratorT = UniformEmbeddingGenerator;
  auto generator_args =
      typename GeneratorT::Args{curand_state_context.ptr(), lower, upper};
  int num_worker = curand_state_context.num_worker();
  initialize_with_generator<GeneratorT>(buffer, indices, generator_args,
                                        num_worker);
}

void const_init(at::Tensor buffer, at::Tensor indices, float value) {
  using GeneratorT = ConstEmbeddingGenerator;
  auto generator_args = typename GeneratorT::Args{value};
  initialize_with_generator<GeneratorT>(buffer, indices, generator_args);
}

void debug_init(at::Tensor buffer, at::Tensor indices, at::Tensor keys) {
  auto key_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(keys.dtype()));
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, KeyType, [&] {
    using GeneratorT = MappingEmbeddingGenerator<KeyType>;
    auto generator_args = typename GeneratorT::Args{
        reinterpret_cast<const KeyType *>(keys.data_ptr()), 100000};
    initialize_with_generator<GeneratorT>(buffer, indices, generator_args);
  });
}

} // namespace dyn_emb

void bind_initializer_op(py::module &m) {

  py::class_<dyn_emb::CurandStateContext>(m, "CurandStateContext")
      .def(py::init<>())
      .def("ptr", &dyn_emb::CurandStateContext::ptr,
           py::return_value_policy::reference);

  m.def("normal_init", &dyn_emb::normal_init, "Normal initializer",
        py::arg("buffer"), py::arg("indices"), py::arg("curand_state_context"),
        py::arg("mean"), py::arg("std_dev"));

  m.def("truncated_normal_init", &dyn_emb::truncated_normal_init,
        "Truncated normal initializer", py::arg("buffer"), py::arg("indices"),
        py::arg("curand_state_context"), py::arg("mean"), py::arg("std_dev"),
        py::arg("lower"), py::arg("upper"));

  m.def("uniform_init", &dyn_emb::uniform_init, "Uniform initializer",
        py::arg("buffer"), py::arg("indices"), py::arg("curand_state_context"),
        py::arg("lower"), py::arg("upper"));

  m.def("const_init", &dyn_emb::const_init, "Const initializer",
        py::arg("buffer"), py::arg("indices"), py::arg("value"));

  m.def("debug_init", &dyn_emb::debug_init, "Debug initializer",
        py::arg("buffer"), py::arg("indices"), py::arg("keys"));

  m.def("normal_init_flat", &dyn_emb::normal_init_flat,
        "Masked normal initializer for multi-table flat buffers",
        py::arg("table_ptrs"), py::arg("slot_indices"), py::arg("table_ids"),
        py::arg("table_value_dims"), py::arg("table_emb_dims"),
        py::arg("mask"), py::arg("value_dtype"),
        py::arg("initial_optim_state"), py::arg("curand_state_context"),
        py::arg("mean"), py::arg("std_dev"));

  m.def("truncated_normal_init_flat", &dyn_emb::truncated_normal_init_flat,
        "Masked truncated-normal initializer for multi-table flat buffers",
        py::arg("table_ptrs"), py::arg("slot_indices"), py::arg("table_ids"),
        py::arg("table_value_dims"), py::arg("table_emb_dims"),
        py::arg("mask"), py::arg("value_dtype"),
        py::arg("initial_optim_state"), py::arg("curand_state_context"),
        py::arg("mean"), py::arg("std_dev"), py::arg("lower"),
        py::arg("upper"));

  m.def("uniform_init_flat", &dyn_emb::uniform_init_flat,
        "Masked uniform initializer for multi-table flat buffers",
        py::arg("table_ptrs"), py::arg("slot_indices"), py::arg("table_ids"),
        py::arg("table_value_dims"), py::arg("table_emb_dims"),
        py::arg("mask"), py::arg("value_dtype"),
        py::arg("initial_optim_state"), py::arg("curand_state_context"),
        py::arg("lower"), py::arg("upper"));

  m.def("const_init_flat", &dyn_emb::const_init_flat,
        "Masked constant initializer for multi-table flat buffers",
        py::arg("table_ptrs"), py::arg("slot_indices"), py::arg("table_ids"),
        py::arg("table_value_dims"), py::arg("table_emb_dims"),
        py::arg("mask"), py::arg("value_dtype"),
        py::arg("initial_optim_state"), py::arg("value"));

  m.def("debug_init_flat", &dyn_emb::debug_init_flat,
        "Masked debug initializer for multi-table flat buffers",
        py::arg("table_ptrs"), py::arg("slot_indices"), py::arg("table_ids"),
        py::arg("table_value_dims"), py::arg("table_emb_dims"),
        py::arg("mask"), py::arg("value_dtype"),
        py::arg("initial_optim_state"), py::arg("keys"));

  m.def("normal_load_or_initialize_mixed",
        &dyn_emb::normal_load_or_initialize_mixed,
        "Load cache/storage rows or initialize transient rows with normal values",
        py::arg("output"), py::arg("cache_table_ptrs"),
        py::arg("cache_rows"), py::arg("storage_table_ptrs"),
        py::arg("storage_rows"), py::arg("table_ids"),
        py::arg("table_value_dims"), py::arg("table_emb_dims"),
        py::arg("all_dims_vec4"), py::arg("curand_state_context"),
        py::arg("mean"), py::arg("std_dev"),
        py::arg("preinitialized_mask") = py::none());

  m.def("truncated_normal_load_or_initialize_mixed",
        &dyn_emb::truncated_normal_load_or_initialize_mixed,
        "Load cache/storage rows or initialize transient rows with truncated-normal values",
        py::arg("output"), py::arg("cache_table_ptrs"),
        py::arg("cache_rows"), py::arg("storage_table_ptrs"),
        py::arg("storage_rows"), py::arg("table_ids"),
        py::arg("table_value_dims"), py::arg("table_emb_dims"),
        py::arg("all_dims_vec4"), py::arg("curand_state_context"),
        py::arg("mean"), py::arg("std_dev"), py::arg("lower"),
        py::arg("upper"), py::arg("preinitialized_mask") = py::none());

  m.def("uniform_load_or_initialize_mixed",
        &dyn_emb::uniform_load_or_initialize_mixed,
        "Load cache/storage rows or initialize transient rows with uniform values",
        py::arg("output"), py::arg("cache_table_ptrs"),
        py::arg("cache_rows"), py::arg("storage_table_ptrs"),
        py::arg("storage_rows"), py::arg("table_ids"),
        py::arg("table_value_dims"), py::arg("table_emb_dims"),
        py::arg("all_dims_vec4"), py::arg("curand_state_context"),
        py::arg("lower"), py::arg("upper"),
        py::arg("preinitialized_mask") = py::none());

  m.def("const_load_or_initialize_mixed",
        &dyn_emb::const_load_or_initialize_mixed,
        "Load cache/storage rows or initialize transient rows with a constant",
        py::arg("output"), py::arg("cache_table_ptrs"),
        py::arg("cache_rows"), py::arg("storage_table_ptrs"),
        py::arg("storage_rows"), py::arg("table_ids"),
        py::arg("table_value_dims"), py::arg("table_emb_dims"),
        py::arg("all_dims_vec4"), py::arg("value"),
        py::arg("preinitialized_mask") = py::none());

  m.def("debug_load_or_initialize_mixed",
        &dyn_emb::debug_load_or_initialize_mixed,
        "Load cache/storage rows or initialize transient rows from keys",
        py::arg("output"), py::arg("cache_table_ptrs"),
        py::arg("cache_rows"), py::arg("storage_table_ptrs"),
        py::arg("storage_rows"), py::arg("table_ids"),
        py::arg("table_value_dims"), py::arg("table_emb_dims"),
        py::arg("all_dims_vec4"), py::arg("keys"),
        py::arg("preinitialized_mask") = py::none());
}
