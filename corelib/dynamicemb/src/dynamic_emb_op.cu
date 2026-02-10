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

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "check.h"
#include "lookup_backward.h"
#include "lookup_forward.h"
#include "lookup_kernel.cuh"
#include "torch_utils.h"
#include "utils.h"
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <torch/torch.h>

#include "table_operation/types.cuh"

namespace py = pybind11;
using namespace dyn_emb;

template <typename scalar_t>
__global__ void
compact_offsets(const scalar_t *offsets, scalar_t *features_offsets,
                const int64_t num_features, const int64_t batch_size) {
  for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < num_features;
       tid += blockDim.x * gridDim.x) {
    features_offsets[tid] = offsets[tid * batch_size];
  }
  if (threadIdx.x == 0) {
    features_offsets[num_features] = offsets[num_features * batch_size];
  }
}

std::vector<int64_t> offsets_to_table_features_offsets(
    const at::Tensor &offsets, const std::vector<int> &table_offsets_in_feature,
    const int64_t batch_size, cudaStream_t stream) {
  int64_t table_num = table_offsets_in_feature.size() - 1;
  int64_t num_features = (offsets.numel() - 1) / batch_size;
  at::Tensor h_features_offsets =
      at::empty({num_features + 1},
                offsets.options().device(at::kCPU).pinned_memory(true));
  if (num_features == 0) {
    return {0, 0};
  }
  AT_DISPATCH_INTEGRAL_TYPES(offsets.scalar_type(), "compact_offsets", [&] {
    compact_offsets<<<num_features / 1024 + 1, 1024, 0, stream>>>(
        offsets.data_ptr<scalar_t>(), h_features_offsets.data_ptr<scalar_t>(),
        num_features, batch_size);
  });
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<int64_t> table_features_offsets(table_offsets_in_feature.size(),
                                              0);
  for (int i = 0; i < table_offsets_in_feature.size(); ++i) {
    table_features_offsets[i] =
        h_features_offsets[table_offsets_in_feature[i]].item<int64_t>();
  }
  return table_features_offsets;
}

void gather_embedding(at::Tensor input, at::Tensor output, at::Tensor index) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto &device_prop = DeviceProp::getDeviceProp(index.device().index());
  int num_sms = device_prop.num_sms;
  auto src_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(input.dtype()));
  auto dst_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(output.dtype()));
  auto index_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(index.dtype()));

  int64_t num_total = output.size(0);
  int64_t dim = output.size(1);
  if (num_total != index.numel()) {
    throw std::runtime_error(
        "Number rows of `output` must match with `index`.");
  }
  if (dim != input.size(1)) {
    throw std::runtime_error(
        "Number cols of `output` must match with `input`.");
  }
  dyn_emb::scatter_fused(input.data_ptr(), output.data_ptr(), index.data_ptr(),
                         num_total, dim, src_type, dst_type, index_type,
                         num_sms, stream);
}

void gather_embedding_pooled(
    at::Tensor input, at::Tensor output, at::Tensor index, at::Tensor offsets,
    int combiner, int total_D, int batch_size,
    const std::optional<at::Tensor> &D_offsets = std::nullopt, int max_D = 0) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int num_slots = offsets.size(0) - 1;

  auto src_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(input.dtype()));
  auto dst_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(output.dtype()));
  auto offset_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(offsets.dtype()));

  int dim = D_offsets.has_value() ? max_D : static_cast<int>(input.size(1));
  const int *d_D_offsets = nullptr;
  if (D_offsets.has_value()) {
    TORCH_CHECK(D_offsets.value().scalar_type() == at::kInt,
                "D_offsets must be int32, got ", D_offsets.value().scalar_type());
    d_D_offsets =
        reinterpret_cast<const int *>(D_offsets.value().data_ptr());
  }
  dyn_emb::scatter_combine(
      input.data_ptr(), output.data_ptr(), offsets.data_ptr(), index.data_ptr(),
      combiner, total_D, /*accum_D=*/0, dim, num_slots, batch_size, src_type,
      dst_type, offset_type, stream, d_D_offsets);
}

// Generate permutation-aware gather_ids from CSR offsets.
// grads is [B*F, D] batch-first (row r → b=r/F, f=r%F).
// Each thread processes one slot (bucket) s; slot s owns indices
// [offsets[s], offsets[s+1]).  slot s has f=s/B, b=s%B.
// gather_ids[j] = b*F + f  — the row in [B*F, D] that LocalReduce reads.
template <typename offset_t, typename id_t>
__global__ void
generate_gather_ids_pooled_kernel(const offset_t *__restrict__ offsets,
                                  id_t *__restrict__ gather_ids, int num_slots,
                                  int B, int F) {
  for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < num_slots;
       s += gridDim.x * blockDim.x) {
    int f = s / B;
    int b = s % B;
    id_t val = static_cast<id_t>(b) * static_cast<id_t>(F) + static_cast<id_t>(f);
    offset_t start = offsets[s];
    offset_t end = offsets[s + 1];
    for (offset_t j = start; j < end; ++j) {
      gather_ids[j] = val;
    }
  }
}

at::Tensor
reduce_grads(at::Tensor reverse_indices, at::Tensor grads, int64_t num_unique,
             int batch_size, int64_t out_dim,
             const std::optional<at::Tensor> &offsets = std::nullopt,
             const std::optional<at::Tensor> &D_offsets = std::nullopt,
             int combiner = -1,
             int total_D = 0) {
  // When D_offsets is provided (multi-dim pooling):
  //   grads is [B, total_D].  Permutation-aware gather_ids are generated,
  //   sorted with reverse_indices, then a multi-dim variant of LocalReduce
  //   reads directly from grads using D_offsets to compute per-feature source
  //   offsets and widths.  MEAN scaling is fused in the stage-1 kernel.
  //   No padded intermediate buffer is needed.
  //
  // When offsets is provided without D_offsets (uniform-dim pooling):
  //   grads is [B*F, D] batch-first (free reshape from [B, total_D]).
  //   1. For MEAN, an in-place kernel scales each row by 1/pool_size.
  //   2. Permutation-aware gather_ids are generated via binary search so that
  //      LocalReduce reads from the correct batch-first rows directly — no
  //      intermediate permuted tensor is allocated.
  //
  // When offsets is absent (sequence mode), gather_ids = arange(num_keys)
  //   and LocalReduce gathers directly from grads.

  int64_t num_keys = reverse_indices.size(0);

  if (!reverse_indices.is_cuda() || !grads.is_cuda()) {
    throw std::runtime_error("All argument tensors should be on device");
  }

  auto device_ = reverse_indices.device();
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto id_stype = reverse_indices.dtype().toScalarType();
  auto id_dtype = scalartype_to_datatype(id_stype);

  bool multi_dim = D_offsets.has_value() && offsets.has_value();

  at::Tensor unique_grads = at::empty({num_unique, out_dim}, grads.options());

  if (num_keys == 0 || batch_size == 0)
    return unique_grads;
  // --- Generate gather_ids ---
  at::Tensor gather_ids;
  if (offsets.has_value()) {
    auto &offs = offsets.value();
    int num_slots = static_cast<int>(offs.numel() - 1);
    TORCH_CHECK(batch_size > 0, "batch_size must be greater than 0");
    TORCH_CHECK(num_slots % batch_size == 0, "num_slots (", num_slots, ") must be divisible by batch_size (", batch_size, ")");
    int num_features = num_slots / batch_size;
    auto offset_type =
        scalartype_to_datatype(convertTypeMetaToScalarType(offs.dtype()));

    constexpr int kBlockSize = 256;
    auto &device_prop = DeviceProp::getDeviceProp();
    const int max_grid_size =
        device_prop.num_sms * (device_prop.max_thread_per_sm / kBlockSize);

    // Generate permutation-aware gather_ids — one thread per slot (bucket).
    gather_ids = at::empty({num_keys}, reverse_indices.options());
    int slot_grid = static_cast<int>(
        std::min(((int64_t)num_slots + kBlockSize - 1) / kBlockSize,
                 (int64_t)max_grid_size));

    DISPATCH_INTEGER_DATATYPE_FUNCTION(offset_type, offset_t, [&] {
      DISPATCH_INTEGER_DATATYPE_FUNCTION(id_dtype, id_t, [&] {
        generate_gather_ids_pooled_kernel<offset_t, id_t>
            <<<slot_grid, kBlockSize, 0, stream>>>(
                reinterpret_cast<const offset_t *>(offs.data_ptr()),
                reinterpret_cast<id_t *>(gather_ids.data_ptr()), num_slots,
                batch_size, num_features);
      });
    });
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    gather_ids = at::arange(num_keys, reverse_indices.options());
  }

  // --- Sort (reverse_indices, gather_ids) by reverse_indices ---
  auto sorted_reverse_indices = at::empty_like(reverse_indices);
  auto sorted_gather_ids = at::empty_like(gather_ids);

  int end_bit =
      (num_unique > 1)
          ? (64 - __builtin_clzll(static_cast<uint64_t>(num_unique - 1)))
          : 1;
  DISPATCH_INTEGER_DATATYPE_FUNCTION(id_dtype, id_t, [&] {
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        reinterpret_cast<id_t *>(reverse_indices.data_ptr()),
        reinterpret_cast<id_t *>(sorted_reverse_indices.data_ptr()),
        reinterpret_cast<id_t *>(gather_ids.data_ptr()),
        reinterpret_cast<id_t *>(sorted_gather_ids.data_ptr()), num_keys, 0,
        end_bit, stream);
    auto temp_storage =
        at::empty({static_cast<int64_t>(temp_storage_bytes)},
                  at::TensorOptions().dtype(at::kByte).device(device_));
    cub::DeviceRadixSort::SortPairs(
        temp_storage.data_ptr(), temp_storage_bytes,
        reinterpret_cast<id_t *>(reverse_indices.data_ptr()),
        reinterpret_cast<id_t *>(sorted_reverse_indices.data_ptr()),
        reinterpret_cast<id_t *>(gather_ids.data_ptr()),
        reinterpret_cast<id_t *>(sorted_gather_ids.data_ptr()), num_keys, 0,
        end_bit, stream);
  });

  // --- LocalReduce ---
  // MEAN scaling is fused inside the reduce kernel for both uniform and
  // multi-dim modes, so no separate scaling pass is needed.
  LocalReduce localReduceOp(device_, num_keys, out_dim, id_dtype,
                            DataType::Float32);

  if (offsets.has_value()) {
    auto &offs = offsets.value();
    int num_slots = static_cast<int>(offs.size(0) - 1);
    int num_features = num_slots / batch_size;

    localReduceOp.local_reduce(grads, unique_grads, sorted_gather_ids,
                               sorted_reverse_indices, stream, D_offsets, offs,
                               batch_size, num_features, total_D, combiner);
  } else {
    localReduceOp.local_reduce(grads, unique_grads, sorted_gather_ids,
                               sorted_reverse_indices, stream);
  }

  return unique_grads;
}

template <typename IndexT, typename ValueT>
__global__ void load_from_combined_table_kernel_vec4(
    int64_t batch, int emb_dim, int stride, int64_t output_stride,
    int split_index, ValueT const *__restrict__ dev_table,
    ValueT const *__restrict__ uvm_table, ValueT *__restrict__ output_buffer,
    IndexT const *__restrict__ indices) {

  constexpr int kWarpSize = 32;
  constexpr int VecSize = 4;
  const int warp_num_per_block = blockDim.x / kWarpSize;
  const int warp_id_in_block = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;

  Vec4T<ValueT> emb;
  for (int64_t emb_id = warp_num_per_block * blockIdx.x + warp_id_in_block;
       emb_id < batch; emb_id += gridDim.x * warp_num_per_block) {
    IndexT const index = indices[emb_id];
    ValueT const *src = nullptr;
    if (index < split_index) {
      src = dev_table + index * stride;
    } else {
      src = uvm_table + (index - split_index) * stride;
    }
    ValueT *dst = output_buffer + emb_id * output_stride;
    if (index >= 0) {
      for (int i = 0; VecSize * (kWarpSize * i + lane_id) < emb_dim; ++i) {
        int idx4 = VecSize * (kWarpSize * i + lane_id);
        emb.load(src + idx4);
        emb.store(dst + idx4);
      }
    }
  }
}

template <typename IndexT, typename ValueT>
__global__ void load_from_combined_table_kernel(
    int64_t batch, int emb_dim, int stride, int64_t output_stride,
    int split_index, ValueT const *__restrict__ dev_table,
    ValueT const *__restrict__ uvm_table, ValueT *__restrict__ output_buffer,
    IndexT const *__restrict__ indices) {

  for (int64_t emb_id = blockIdx.x; emb_id < batch; emb_id += gridDim.x) {
    IndexT const index = indices[emb_id];
    ValueT const *src = nullptr;
    if (index < split_index) {
      src = dev_table + index * stride;
    } else {
      src = uvm_table + (index - split_index) * stride;
    }
    ValueT *dst = output_buffer + emb_id * output_stride;
    if (index >= 0) {
      for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
        dst[i] = src[i];
      }
    }
  }
}

void load_from_combined_table(std::optional<at::Tensor> dev_table,
                              std::optional<at::Tensor> uvm_table,
                              at::Tensor indices, at::Tensor output) {

  int64_t num_total = indices.size(0);
  if (num_total == 0) {
    return;
  }
  int64_t stride = -1;
  int64_t dim = output.size(1);
  int64_t output_stride = output.stride(0);
  if ((not dev_table.has_value()) and (not uvm_table.has_value())) {
    throw std::runtime_error("Two tables cannot both be None.");
  } else {
    if (dev_table.has_value()) {
      stride = dev_table.value().size(1);
      if (stride < dim) {
        throw std::runtime_error(
            "Output tensor's dim1 should not be greater than the table's.");
      }
    } else {
      stride = uvm_table.value().size(1);
      if (stride < dim) {
        throw std::runtime_error(
            "Output tensor's dim1 should not be greater than the table's.");
      }
    }
  }

  if (output.dim() != 2) {
    throw std::runtime_error("Output tensor should be 2-dim.");
  }

  if (output.size(0) != indices.size(0)) {
    throw std::runtime_error("Output tensor mismatches with indices at dim-0.");
  }

  int64_t split_index = 0;
  if (dev_table.has_value()) {
    split_index = dev_table.value().size(0);
  }

  auto val_type = get_data_type(output);
  auto index_type = get_data_type(indices);

  constexpr int kWarpSize = 32;
  constexpr int MULTIPLIER = 4;
  constexpr int BLOCK_SIZE_VEC = 64;
  constexpr int WARP_PER_BLOCK = BLOCK_SIZE_VEC / kWarpSize;
  auto &device_prop = DeviceProp::getDeviceProp();
  const int max_grid_size =
      device_prop.num_sms * (device_prop.max_thread_per_sm / BLOCK_SIZE_VEC);

  int grid_size = 0;
  if (num_total / WARP_PER_BLOCK < max_grid_size) {
    grid_size = (num_total - 1) / WARP_PER_BLOCK + 1;
  } else if (num_total / WARP_PER_BLOCK > max_grid_size * MULTIPLIER) {
    grid_size = max_grid_size * MULTIPLIER;
  } else {
    grid_size = max_grid_size;
  }

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, ValueType, [&] {
    DISPATCH_OFFSET_INT_TYPE(index_type, IndexType, [&] {
      auto dev_ptr = get_pointer<ValueType>(dev_table);
      auto uvm_ptr = get_pointer<ValueType>(uvm_table);
      auto out_ptr = get_pointer<ValueType>(output);
      auto index_ptr = get_pointer<IndexType>(indices);

      if (dim % 4 == 0) {
        load_from_combined_table_kernel_vec4<IndexType, ValueType>
            <<<grid_size, BLOCK_SIZE_VEC, 0, stream>>>(
                num_total, dim, stride, output_stride, split_index, dev_ptr,
                uvm_ptr, out_ptr, index_ptr);
      } else {
        int block_size = dim < device_prop.max_thread_per_block
                             ? dim
                             : device_prop.max_thread_per_block;
        int grid_size = num_total;
        load_from_combined_table_kernel<IndexType, ValueType>
            <<<grid_size, block_size, 0, stream>>>(
                num_total, dim, stride, output_stride, split_index, dev_ptr,
                uvm_ptr, out_ptr, index_ptr);
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename IndexT, typename ValueT>
__global__ void store_to_combined_table_kernel_vec4(
    int64_t batch, int stride, int split_index, ValueT *__restrict__ dev_table,
    ValueT *__restrict__ uvm_table, ValueT const *__restrict__ input_buffer,
    IndexT const *__restrict__ indices) {

  constexpr int kWarpSize = 32;
  constexpr int VecSize = 4;
  const int warp_num_per_block = blockDim.x / kWarpSize;
  const int warp_id_in_block = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;

  Vec4T<ValueT> emb;
  for (int64_t emb_id = warp_num_per_block * blockIdx.x + warp_id_in_block;
       emb_id < batch; emb_id += gridDim.x * warp_num_per_block) {
    IndexT const index = indices[emb_id];
    ValueT *dst = nullptr;
    if (index < split_index) {
      dst = dev_table + index * stride;
    } else {
      dst = uvm_table + (index - split_index) * stride;
    }
    ValueT const *src = input_buffer + emb_id * stride;
    if (index >= 0) {
      for (int i = 0; VecSize * (kWarpSize * i + lane_id) < stride; ++i) {
        int idx4 = VecSize * (kWarpSize * i + lane_id);
        emb.load(src + idx4);
        emb.store(dst + idx4);
      }
    }
  }
}

template <typename IndexT, typename ValueT>
__global__ void store_to_combined_table_kernel(
    int64_t batch, int stride, int split_index, ValueT *__restrict__ dev_table,
    ValueT *__restrict__ uvm_table, ValueT const *__restrict__ input_buffer,
    IndexT const *__restrict__ indices) {

  for (int64_t emb_id = blockIdx.x; emb_id < batch; emb_id += gridDim.x) {
    IndexT const index = indices[emb_id];
    ValueT *dst = nullptr;
    if (index < split_index) {
      dst = dev_table + index * stride;
    } else {
      dst = uvm_table + (index - split_index) * stride;
    }
    ValueT const *src = input_buffer + emb_id * stride;
    if (index >= 0) {
      for (int i = threadIdx.x; i < stride; i += blockDim.x) {
        dst[i] = src[i];
      }
    }
  }
}

void store_to_combined_table(std::optional<at::Tensor> dev_table,
                             std::optional<at::Tensor> uvm_table,
                             at::Tensor indices, at::Tensor input) {

  int64_t stride = -1;
  int64_t dim = input.size(1);
  if ((not dev_table.has_value()) and (not uvm_table.has_value())) {
    throw std::runtime_error("Two tables cannot both be None.");
  } else {
    if (dev_table.has_value()) {
      stride = dev_table.value().size(1);
      if (stride != dim) {
        throw std::runtime_error(
            "Input tensor's dim1 should equal to the table's.");
      }
    } else {
      stride = uvm_table.value().size(1);
      if (stride != dim) {
        throw std::runtime_error(
            "Input tensor's dim1 should equal to the table's.");
      }
    }
  }

  if (input.dim() != 2) {
    throw std::runtime_error("Input tensor should be 2-dim.");
  }

  if (input.size(0) != indices.size(0)) {
    throw std::runtime_error("Input tensor mismatches with indices at dim-0.");
  }

  int64_t split_index = 0;
  if (dev_table.has_value()) {
    split_index = dev_table.value().size(0);
  }

  auto val_type = get_data_type(input);
  auto index_type = get_data_type(indices);

  int64_t num_total = indices.size(0);

  constexpr int kWarpSize = 32;
  constexpr int MULTIPLIER = 4;
  constexpr int BLOCK_SIZE_VEC = 64;
  constexpr int WARP_PER_BLOCK = BLOCK_SIZE_VEC / kWarpSize;
  auto &device_prop = DeviceProp::getDeviceProp();
  const int max_grid_size =
      device_prop.num_sms * (device_prop.max_thread_per_sm / BLOCK_SIZE_VEC);

  int grid_size = 0;
  if (num_total / WARP_PER_BLOCK < max_grid_size) {
    grid_size = (num_total - 1) / WARP_PER_BLOCK + 1;
  } else if (num_total / WARP_PER_BLOCK > max_grid_size * MULTIPLIER) {
    grid_size = max_grid_size * MULTIPLIER;
  } else {
    grid_size = max_grid_size;
  }

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, ValueType, [&] {
    DISPATCH_OFFSET_INT_TYPE(index_type, IndexType, [&] {
      auto dev_ptr = get_pointer<ValueType>(dev_table);
      auto uvm_ptr = get_pointer<ValueType>(uvm_table);
      auto input_ptr = get_pointer<ValueType>(input);
      auto index_ptr = get_pointer<IndexType>(indices);

      if (dim % 4 == 0) {
        store_to_combined_table_kernel_vec4<IndexType, ValueType>
            <<<grid_size, BLOCK_SIZE_VEC, 0, stream>>>(
                num_total, stride, split_index, dev_ptr, uvm_ptr, input_ptr,
                index_ptr);
      } else {
        int block_size = dim < device_prop.max_thread_per_block
                             ? dim
                             : device_prop.max_thread_per_block;
        int grid_size = num_total;
        store_to_combined_table_kernel<IndexType, ValueType>
            <<<grid_size, block_size, 0, stream>>>(
                num_total, stride, split_index, dev_ptr, uvm_ptr, input_ptr,
                index_ptr);
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename IndexT, typename ValueT>
__global__ void select_insert_failed_values_kernel_vec4(
    int64_t batch, int64_t stride, ValueT const *__restrict__ in_v_ptr,
    ValueT *__restrict__ out_v_ptr, IndexT *__restrict__ indices) {

  constexpr int kWarpSize = 32;
  constexpr int VecSize = 4;
  const int warp_num_per_block = blockDim.x / kWarpSize;
  const int warp_id_in_block = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;

  Vec4T<ValueT> emb;
  for (int64_t dst_idx = warp_num_per_block * blockIdx.x + warp_id_in_block;
       dst_idx < batch; dst_idx += gridDim.x * warp_num_per_block) {
    IndexT in_idx = indices[dst_idx];
    if (in_idx >= 0) {
      continue;
    }
    IndexT in_idx_pos = -in_idx - 1;
    ValueT *dst = out_v_ptr + dst_idx * stride;
    ValueT const *src = in_v_ptr + in_idx_pos * stride;

    for (int i = 0; VecSize * (kWarpSize * i + lane_id) < stride; ++i) {
      int idx4 = VecSize * (kWarpSize * i + lane_id);
      emb.load(src + idx4);
      emb.store(dst + idx4);
    }

    if (lane_id == 0) {
      indices[dst_idx] = -1;
    }
  }
}

template <typename IndexT, typename ValueT>
__global__ void select_insert_failed_values_kernel(
    int64_t batch, int64_t stride, ValueT const *__restrict__ in_v_ptr,
    ValueT *__restrict__ out_v_ptr, IndexT *__restrict__ indices) {

  for (int64_t dst_idx = blockIdx.x; dst_idx < batch; dst_idx += gridDim.x) {

    IndexT in_idx = indices[dst_idx];
    if (in_idx >= 0) {
      continue;
    }
    IndexT in_idx_pos = -in_idx - 1;
    ValueT *dst = out_v_ptr + dst_idx * stride;
    ValueT const *src = in_v_ptr + in_idx_pos * stride;

    for (int i = threadIdx.x; i < stride; i += blockDim.x) {
      dst[i] = src[i];
    }

    if (threadIdx.x == 0) {
      indices[dst_idx] = -1;
    }
  }
}

void select_insert_failed_values(at::Tensor indices, at::Tensor input_values,
                                 at::Tensor evictd_values) {
  int64_t num_total = indices.numel();
  if (num_total == 0) {
    return;
  }

  int64_t dim = input_values.size(1);

  auto val_type = get_data_type(input_values);
  auto index_type = get_data_type(indices);

  constexpr int kWarpSize = 32;
  constexpr int MULTIPLIER = 4;
  constexpr int BLOCK_SIZE_VEC = 64;
  constexpr int WARP_PER_BLOCK = BLOCK_SIZE_VEC / kWarpSize;
  auto &device_prop = DeviceProp::getDeviceProp();
  const int max_grid_size =
      device_prop.num_sms * (device_prop.max_thread_per_sm / BLOCK_SIZE_VEC);

  int grid_size = 0;
  if (num_total / WARP_PER_BLOCK < max_grid_size) {
    grid_size = (num_total - 1) / WARP_PER_BLOCK + 1;
  } else if (num_total / WARP_PER_BLOCK > max_grid_size * MULTIPLIER) {
    grid_size = max_grid_size * MULTIPLIER;
  } else {
    grid_size = max_grid_size;
  }

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, ValueType, [&] {
    DISPATCH_OFFSET_INT_TYPE(index_type, IndexType, [&] {
      auto in_v_ptr = get_pointer<ValueType>(input_values);
      auto out_v_ptr = get_pointer<ValueType>(evictd_values);
      auto index_ptr = get_pointer<IndexType>(indices);

      if (dim % 4 == 0) {
        select_insert_failed_values_kernel_vec4<IndexType, ValueType>
            <<<grid_size, BLOCK_SIZE_VEC, 0, stream>>>(num_total, dim, in_v_ptr,
                                                       out_v_ptr, index_ptr);
      } else {
        int block_size = dim < device_prop.max_thread_per_block
                             ? dim
                             : device_prop.max_thread_per_block;
        int grid_size = num_total;
        select_insert_failed_values_kernel<IndexType, ValueType>
            <<<grid_size, block_size, 0, stream>>>(num_total, dim, in_v_ptr,
                                                   out_v_ptr, index_ptr);
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

// PYTHON WARP
void bind_dyn_emb_op(py::module &m) {

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

  m.def("reduce_grads", &reduce_grads, "reduce grads",
        py::arg("reverse_indices"), py::arg("grads"), py::arg("num_unique"),
        py::arg("batch_size"), py::arg("out_dim"),
        py::arg("offsets") = py::none(),
        py::arg("D_offsets") = py::none(), py::arg("combiner") = -1,
        py::arg("total_D") = 0);

  m.def("gather_embedding", &gather_embedding,
        "Gather embedding based on index.", py::arg("input"), py::arg("output"),
        py::arg("index"));

  m.def("gather_embedding_pooled", &gather_embedding_pooled,
        "Gather embedding with pooling (SUM/MEAN) based on index and offsets.",
        py::arg("input"), py::arg("output"), py::arg("index"),
        py::arg("offsets"), py::arg("combiner"), py::arg("total_D"),
        py::arg("batch_size"), py::arg("D_offsets") = py::none(),
        py::arg("max_D") = 0);

  m.def("load_from_combined_table", &load_from_combined_table,
        "load_from_combined_table", py::arg("dev_table"), py::arg("uvm_table"),
        py::arg("indices"), py::arg("output"));

  m.def("store_to_combined_table", &store_to_combined_table,
        "store_to_combined_table", py::arg("dev_table"), py::arg("uvm_table"),
        py::arg("indices"), py::arg("input"));

  m.def("select_insert_failed_values", &select_insert_failed_values,
        "select_insert_failed_values", py::arg("indices"),
        py::arg("input_values"), py::arg("evicted_values"));
}
