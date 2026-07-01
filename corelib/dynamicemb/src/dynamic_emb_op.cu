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
  int64_t src_stride = input.stride(0);
  dyn_emb::scatter_fused(input.data_ptr(), output.data_ptr(), index.data_ptr(),
                         num_total, dim, src_stride, src_type, dst_type,
                         index_type, num_sms, stream);
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
  int src_stride = static_cast<int>(input.stride(0));
  const int *d_D_offsets = nullptr;
  if (D_offsets.has_value()) {
    TORCH_CHECK(D_offsets.value().scalar_type() == at::kInt,
                "D_offsets must be int32, got ",
                D_offsets.value().scalar_type());
    d_D_offsets = reinterpret_cast<const int *>(D_offsets.value().data_ptr());
  }
  dyn_emb::scatter_combine(
      input.data_ptr(), output.data_ptr(), offsets.data_ptr(), index.data_ptr(),
      combiner, total_D, /*accum_D=*/0, dim, src_stride, num_slots, batch_size,
      src_type, dst_type, offset_type, stream, d_D_offsets);
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
    id_t val =
        static_cast<id_t>(b) * static_cast<id_t>(F) + static_cast<id_t>(f);
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
             int combiner = -1, int total_D = 0) {
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
    TORCH_CHECK(num_slots % batch_size == 0, "num_slots (", num_slots,
                ") must be divisible by batch_size (", batch_size, ")");
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

// ---------------------------------------------------------------------------
// Flat multi-table load / store
// ---------------------------------------------------------------------------

// NumRegions: 0 = contiguous, 1 = emb-only, 2 = two-region (emb + opt)
// When NumRegions==0, scalar_table_id is used directly (table_ids may be
// nullptr).
template <int NumRegions, typename IndexT, typename ValueT>
__global__ void load_from_flat_table_kernel_vec4(
    int64_t batch, int64_t output_dim, int64_t output_stride,
    ValueT *__restrict__ output, IndexT const *__restrict__ indices,
    int64_t const *__restrict__ table_ids,
    int64_t const *__restrict__ table_ptrs,
    int64_t const *__restrict__ table_value_dims,
    int64_t const *__restrict__ table_emb_dims, int64_t max_emb_dim,
    int64_t scalar_table_id) {

  constexpr int kWarpSize = 32;
  constexpr int VecSize = 4;
  const int warp_num_per_block = blockDim.x / kWarpSize;
  const int warp_id_in_block = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;

  auto copy_region_vec4 = [&](ValueT const *src, ValueT *dst, int64_t len) {
    Vec4T<ValueT> v;
    int64_t aligned = (len / VecSize) * VecSize;
    for (int i = 0; VecSize * (kWarpSize * i + lane_id) < aligned; ++i) {
      int idx4 = VecSize * (kWarpSize * i + lane_id);
      v.load(src + idx4);
      v.store(dst + idx4);
    }
    for (int64_t i = aligned + lane_id; i < len; i += kWarpSize) {
      dst[i] = src[i];
    }
  };

  for (int64_t emb_id = warp_num_per_block * blockIdx.x + warp_id_in_block;
       emb_id < batch; emb_id += gridDim.x * warp_num_per_block) {
    IndexT index = indices[emb_id];
    if (index < 0)
      continue;

    int64_t table_id = NumRegions == 0 ? scalar_table_id : table_ids[emb_id];
    int64_t vdim = table_value_dims[table_id];
    ValueT const *src_base =
        reinterpret_cast<ValueT const *>(table_ptrs[table_id]) +
        static_cast<int64_t>(index) * vdim;
    ValueT *dst_base = output + emb_id * output_stride;

    if constexpr (NumRegions == 0) {
      int64_t copy_len = vdim < output_dim ? vdim : output_dim;
      copy_region_vec4(src_base, dst_base, copy_len);
    } else if constexpr (NumRegions == 1) {
      int64_t edim = table_emb_dims[table_id];
      int64_t emb_copy = edim < output_dim ? edim : output_dim;
      copy_region_vec4(src_base, dst_base, emb_copy);
    } else {
      int64_t edim = table_emb_dims[table_id];
      copy_region_vec4(src_base, dst_base, edim);
      int64_t opt_dim = vdim - edim;
      if (opt_dim > 0) {
        copy_region_vec4(src_base + edim, dst_base + max_emb_dim, opt_dim);
      }
    }
  }
}

template <int NumRegions, typename IndexT, typename ValueT>
__global__ void
load_from_flat_table_kernel(int64_t batch, int64_t output_dim,
                            int64_t output_stride, ValueT *__restrict__ output,
                            IndexT const *__restrict__ indices,
                            int64_t const *__restrict__ table_ids,
                            int64_t const *__restrict__ table_ptrs,
                            int64_t const *__restrict__ table_value_dims,
                            int64_t const *__restrict__ table_emb_dims,
                            int64_t max_emb_dim, int64_t scalar_table_id) {

  for (int64_t emb_id = blockIdx.x; emb_id < batch; emb_id += gridDim.x) {
    IndexT index = indices[emb_id];
    if (index < 0)
      continue;

    int64_t table_id = NumRegions == 0 ? scalar_table_id : table_ids[emb_id];
    int64_t vdim = table_value_dims[table_id];
    ValueT const *src_base =
        reinterpret_cast<ValueT const *>(table_ptrs[table_id]) +
        static_cast<int64_t>(index) * vdim;
    ValueT *dst_base = output + emb_id * output_stride;

    if constexpr (NumRegions == 0) {
      int64_t copy_len = vdim < output_dim ? vdim : output_dim;
      for (int64_t i = threadIdx.x; i < copy_len; i += blockDim.x)
        dst_base[i] = src_base[i];
    } else if constexpr (NumRegions == 1) {
      int64_t edim = table_emb_dims[table_id];
      int64_t emb_copy = edim < output_dim ? edim : output_dim;
      for (int64_t i = threadIdx.x; i < emb_copy; i += blockDim.x)
        dst_base[i] = src_base[i];
    } else {
      int64_t edim = table_emb_dims[table_id];
      for (int64_t i = threadIdx.x; i < edim; i += blockDim.x)
        dst_base[i] = src_base[i];
      int64_t opt_dim = vdim - edim;
      if (opt_dim > 0) {
        for (int64_t i = threadIdx.x; i < opt_dim; i += blockDim.x)
          dst_base[max_emb_dim + i] = src_base[edim + i];
      }
    }
  }
}

template <int NumRegions, typename IndexT, typename ValueT>
__global__ void store_to_flat_table_kernel_vec4(
    int64_t batch, int64_t input_dim, int64_t input_stride,
    ValueT const *__restrict__ input, IndexT const *__restrict__ indices,
    int64_t const *__restrict__ table_ids,
    int64_t const *__restrict__ table_ptrs,
    int64_t const *__restrict__ table_value_dims,
    int64_t const *__restrict__ table_emb_dims, int64_t max_emb_dim,
    int64_t scalar_table_id) {

  constexpr int kWarpSize = 32;
  constexpr int VecSize = 4;
  const int warp_num_per_block = blockDim.x / kWarpSize;
  const int warp_id_in_block = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;

  auto copy_region_vec4 = [&](ValueT const *src, ValueT *dst, int64_t len) {
    Vec4T<ValueT> v;
    int64_t aligned = (len / VecSize) * VecSize;
    for (int i = 0; VecSize * (kWarpSize * i + lane_id) < aligned; ++i) {
      int idx4 = VecSize * (kWarpSize * i + lane_id);
      v.load(src + idx4);
      v.store(dst + idx4);
    }
    for (int64_t i = aligned + lane_id; i < len; i += kWarpSize) {
      dst[i] = src[i];
    }
  };

  for (int64_t emb_id = warp_num_per_block * blockIdx.x + warp_id_in_block;
       emb_id < batch; emb_id += gridDim.x * warp_num_per_block) {
    IndexT index = indices[emb_id];
    if (index < 0)
      continue;

    int64_t table_id = NumRegions == 0 ? scalar_table_id : table_ids[emb_id];
    int64_t vdim = table_value_dims[table_id];
    ValueT const *src_base = input + emb_id * input_stride;
    ValueT *dst_base = reinterpret_cast<ValueT *>(table_ptrs[table_id]) +
                       static_cast<int64_t>(index) * vdim;

    if constexpr (NumRegions == 0) {
      int64_t copy_len = vdim < input_dim ? vdim : input_dim;
      copy_region_vec4(src_base, dst_base, copy_len);
    } else {
      int64_t edim = table_emb_dims[table_id];
      copy_region_vec4(src_base, dst_base, edim);
      int64_t opt_dim = vdim - edim;
      if (opt_dim > 0) {
        copy_region_vec4(src_base + max_emb_dim, dst_base + edim, opt_dim);
      }
    }
  }
}

template <int NumRegions, typename IndexT, typename ValueT>
__global__ void store_to_flat_table_kernel(
    int64_t batch, int64_t input_dim, int64_t input_stride,
    ValueT const *__restrict__ input, IndexT const *__restrict__ indices,
    int64_t const *__restrict__ table_ids,
    int64_t const *__restrict__ table_ptrs,
    int64_t const *__restrict__ table_value_dims,
    int64_t const *__restrict__ table_emb_dims, int64_t max_emb_dim,
    int64_t scalar_table_id) {

  for (int64_t emb_id = blockIdx.x; emb_id < batch; emb_id += gridDim.x) {
    IndexT index = indices[emb_id];
    if (index < 0)
      continue;

    int64_t table_id = NumRegions == 0 ? scalar_table_id : table_ids[emb_id];
    int64_t vdim = table_value_dims[table_id];
    ValueT const *src_base = input + emb_id * input_stride;
    ValueT *dst_base = reinterpret_cast<ValueT *>(table_ptrs[table_id]) +
                       static_cast<int64_t>(index) * vdim;

    if constexpr (NumRegions == 0) {
      int64_t copy_len = vdim < input_dim ? vdim : input_dim;
      for (int64_t i = threadIdx.x; i < copy_len; i += blockDim.x)
        dst_base[i] = src_base[i];
    } else {
      int64_t edim = table_emb_dims[table_id];
      for (int64_t i = threadIdx.x; i < edim; i += blockDim.x)
        dst_base[i] = src_base[i];
      int64_t opt_dim = vdim - edim;
      if (opt_dim > 0) {
        for (int64_t i = threadIdx.x; i < opt_dim; i += blockDim.x)
          dst_base[edim + i] = src_base[max_emb_dim + i];
      }
    }
  }
}

template <int NumRegions>
void load_from_flat_table_impl(at::Tensor table_ptrs, at::Tensor indices,
                               int64_t const *table_ids_ptr,
                               int64_t scalar_table_id, at::Tensor output,
                               at::Tensor table_value_dims,
                               at::Tensor table_emb_dims, int64_t max_emb_dim,
                               bool all_dims_vec4) {

  int64_t num_total = indices.size(0);
  if (num_total == 0)
    return;

  TORCH_CHECK(output.dim() == 2, "output must be 2-D");
  TORCH_CHECK(output.size(0) == num_total,
              "output.size(0) must match indices.size(0)");

  int64_t output_dim = output.size(1);
  int64_t output_stride = output.stride(0);

  auto val_type = get_data_type(output);
  auto index_type = get_data_type(indices);

  auto &device_prop = DeviceProp::getDeviceProp();

  constexpr int kWarpSize = 32;
  constexpr int BLOCK_SIZE_VEC = 64;
  constexpr int WARP_PER_BLOCK = BLOCK_SIZE_VEC / kWarpSize;
  constexpr int MULTIPLIER = 4;
  const int max_grid_size =
      device_prop.num_sms * (device_prop.max_thread_per_sm / BLOCK_SIZE_VEC);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, ValueType, [&] {
    DISPATCH_OFFSET_INT_TYPE(index_type, IndexType, [&] {
      if (all_dims_vec4 && output_dim >= 4) {
        int grid_size;
        if (num_total / WARP_PER_BLOCK < max_grid_size) {
          grid_size = (num_total - 1) / WARP_PER_BLOCK + 1;
        } else if (num_total / WARP_PER_BLOCK > max_grid_size * MULTIPLIER) {
          grid_size = max_grid_size * MULTIPLIER;
        } else {
          grid_size = max_grid_size;
        }
        load_from_flat_table_kernel_vec4<NumRegions, IndexType, ValueType>
            <<<grid_size, BLOCK_SIZE_VEC, 0, stream>>>(
                num_total, output_dim, output_stride,
                get_pointer<ValueType>(output), get_pointer<IndexType>(indices),
                table_ids_ptr, get_pointer<int64_t>(table_ptrs),
                get_pointer<int64_t>(table_value_dims),
                get_pointer<int64_t>(table_emb_dims), max_emb_dim,
                scalar_table_id);
      } else {
        int block_size = output_dim < device_prop.max_thread_per_block
                             ? static_cast<int>(output_dim)
                             : device_prop.max_thread_per_block;
        if (block_size < 1)
          block_size = 1;
        load_from_flat_table_kernel<NumRegions, IndexType, ValueType>
            <<<static_cast<int>(num_total), block_size, 0, stream>>>(
                num_total, output_dim, output_stride,
                get_pointer<ValueType>(output), get_pointer<IndexType>(indices),
                table_ids_ptr, get_pointer<int64_t>(table_ptrs),
                get_pointer<int64_t>(table_value_dims),
                get_pointer<int64_t>(table_emb_dims), max_emb_dim,
                scalar_table_id);
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void load_from_flat_table_contiguous(at::Tensor table_ptrs, at::Tensor indices,
                                     int64_t table_id, at::Tensor output,
                                     at::Tensor table_value_dims,
                                     at::Tensor table_emb_dims,
                                     int64_t max_emb_dim, bool all_dims_vec4) {
  load_from_flat_table_impl<0>(table_ptrs, indices, nullptr, table_id, output,
                               table_value_dims, table_emb_dims, max_emb_dim,
                               all_dims_vec4);
}

void load_from_flat_table_emb(at::Tensor table_ptrs, at::Tensor indices,
                              at::Tensor table_ids, at::Tensor output,
                              at::Tensor table_value_dims,
                              at::Tensor table_emb_dims, int64_t max_emb_dim,
                              bool all_dims_vec4) {
  load_from_flat_table_impl<1>(
      table_ptrs, indices, get_pointer<int64_t>(table_ids), 0, output,
      table_value_dims, table_emb_dims, max_emb_dim, all_dims_vec4);
}

void load_from_flat_table_value(at::Tensor table_ptrs, at::Tensor indices,
                                at::Tensor table_ids, at::Tensor output,
                                at::Tensor table_value_dims,
                                at::Tensor table_emb_dims, int64_t max_emb_dim,
                                bool all_dims_vec4) {
  load_from_flat_table_impl<2>(
      table_ptrs, indices, get_pointer<int64_t>(table_ids), 0, output,
      table_value_dims, table_emb_dims, max_emb_dim, all_dims_vec4);
}

template <int NumRegions>
void store_to_flat_table_impl(at::Tensor table_ptrs, at::Tensor indices,
                              int64_t const *table_ids_ptr,
                              int64_t scalar_table_id, at::Tensor input,
                              at::Tensor table_value_dims,
                              at::Tensor table_emb_dims, int64_t max_emb_dim,
                              bool all_dims_vec4) {

  int64_t num_total = indices.size(0);
  if (num_total == 0)
    return;

  TORCH_CHECK(input.dim() == 2, "input must be 2-D");
  TORCH_CHECK(input.size(0) == num_total,
              "input.size(0) must match indices.size(0)");

  int64_t input_dim = input.size(1);
  int64_t input_stride = input.stride(0);

  auto val_type = get_data_type(input);
  auto index_type = get_data_type(indices);

  auto &device_prop = DeviceProp::getDeviceProp();

  constexpr int kWarpSize = 32;
  constexpr int BLOCK_SIZE_VEC = 64;
  constexpr int WARP_PER_BLOCK = BLOCK_SIZE_VEC / kWarpSize;
  constexpr int MULTIPLIER = 4;
  const int max_grid_size =
      device_prop.num_sms * (device_prop.max_thread_per_sm / BLOCK_SIZE_VEC);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, ValueType, [&] {
    DISPATCH_OFFSET_INT_TYPE(index_type, IndexType, [&] {
      if (all_dims_vec4 && input_dim >= 4) {
        int grid_size;
        if (num_total / WARP_PER_BLOCK < max_grid_size) {
          grid_size = (num_total - 1) / WARP_PER_BLOCK + 1;
        } else if (num_total / WARP_PER_BLOCK > max_grid_size * MULTIPLIER) {
          grid_size = max_grid_size * MULTIPLIER;
        } else {
          grid_size = max_grid_size;
        }
        store_to_flat_table_kernel_vec4<NumRegions, IndexType, ValueType>
            <<<grid_size, BLOCK_SIZE_VEC, 0, stream>>>(
                num_total, input_dim, input_stride,
                get_pointer<ValueType>(input), get_pointer<IndexType>(indices),
                table_ids_ptr, get_pointer<int64_t>(table_ptrs),
                get_pointer<int64_t>(table_value_dims),
                get_pointer<int64_t>(table_emb_dims), max_emb_dim,
                scalar_table_id);
      } else {
        int block_size = input_dim < device_prop.max_thread_per_block
                             ? static_cast<int>(input_dim)
                             : device_prop.max_thread_per_block;
        if (block_size < 1)
          block_size = 1;
        store_to_flat_table_kernel<NumRegions, IndexType, ValueType>
            <<<static_cast<int>(num_total), block_size, 0, stream>>>(
                num_total, input_dim, input_stride,
                get_pointer<ValueType>(input), get_pointer<IndexType>(indices),
                table_ids_ptr, get_pointer<int64_t>(table_ptrs),
                get_pointer<int64_t>(table_value_dims),
                get_pointer<int64_t>(table_emb_dims), max_emb_dim,
                scalar_table_id);
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void store_to_flat_table_contiguous(at::Tensor table_ptrs, at::Tensor indices,
                                    int64_t table_id, at::Tensor input,
                                    at::Tensor table_value_dims,
                                    at::Tensor table_emb_dims,
                                    int64_t max_emb_dim, bool all_dims_vec4) {
  store_to_flat_table_impl<0>(table_ptrs, indices, nullptr, table_id, input,
                              table_value_dims, table_emb_dims, max_emb_dim,
                              all_dims_vec4);
}

void store_to_flat_table_value(at::Tensor table_ptrs, at::Tensor indices,
                               at::Tensor table_ids, at::Tensor input,
                               at::Tensor table_value_dims,
                               at::Tensor table_emb_dims, int64_t max_emb_dim,
                               bool all_dims_vec4) {
  store_to_flat_table_impl<2>(
      table_ptrs, indices, get_pointer<int64_t>(table_ids), 0, input,
      table_value_dims, table_emb_dims, max_emb_dim, all_dims_vec4);
}

// ---------------------------------------------------------------------------
// Sparse cache <-> storage value exchange
// ---------------------------------------------------------------------------

constexpr int kExchangeWarpSize = 32;
constexpr int kExchangeBytesPerThread = 16;
constexpr int kExchangeTileBytes =
    kExchangeWarpSize * kExchangeBytesPerThread;
constexpr int kExchangeDirections = 2;
constexpr int kExchangeWarpsPerBlock = 8;
constexpr int kExchangePipelineDepth = 4;
constexpr int kExchangeRowsPerWarp = 8;
constexpr int kExchangeInputsPerBlock =
    kExchangeWarpsPerBlock * kExchangeRowsPerWarp;
constexpr int kExchangeCacheToStorage = 0;
constexpr int kExchangeStorageToCache = 1;

constexpr uint32_t kExchangeMetadataValid = 1U << 0;
constexpr uint32_t kExchangeMetadataStorageToCache = 1U << 1;
constexpr uint32_t kExchangeMetadataCacheToStorage = 1U << 2;

// The eight warps preload one sparse metadata tensor each. Every warp accesses
// 32 consecutive input positions and writes a shared-memory SoA, so all global
// reads are coalesced even when the found/evicted masks are sparse.
struct ExchangeRawMetadata {
  int64_t table_ids[kExchangeInputsPerBlock];
  int64_t cache_rows[kExchangeInputsPerBlock];
  int64_t storage_src_rows[kExchangeInputsPerBlock];
  int64_t storage_lookup_slots[kExchangeInputsPerBlock];
  int64_t storage_dst_slots[kExchangeInputsPerBlock];
  int64_t storage_dst_rows[kExchangeInputsPerBlock];
  bool storage_founds[kExchangeInputsPerBlock];
  bool evicted_mask[kExchangeInputsPerBlock];
  bool valid[kExchangeInputsPerBlock];
};

// Resolved row pointers and reference-counter locations are reused by every
// lane in the warp's four-stage value pipeline.
struct ExchangeMetadata {
  int64_t storage_to_cache_src;
  int64_t storage_to_cache_dst;
  int64_t cache_to_storage_src;
  int64_t cache_to_storage_dst;
  int64_t value_dim;
  int64_t storage_to_cache_ref_counter;
  int64_t cache_to_storage_ref_counter;
  uint32_t flags;
};

struct alignas(kExchangeBytesPerThread) ExchangeBytes16 {
  uint32_t value[4];
};

struct alignas(8) ExchangeBytes8 {
  uint32_t value[2];
};

struct alignas(4) ExchangeBytes4 {
  uint32_t value;
};

__forceinline__ __device__ bool exchange_address_aligned(const void *ptr,
                                                         uintptr_t alignment) {
  return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

template <typename ValueT>
__forceinline__ __device__ void exchange_stage_tile_async(
    const ValueT *src, int64_t dim, int64_t tile_offset, ValueT *smem_dst,
    bool active) {
  constexpr int kValuesPerThread = kExchangeBytesPerThread / sizeof(ValueT);
  const int lane = threadIdx.x & (kExchangeWarpSize - 1);
  const int64_t offset =
      tile_offset + static_cast<int64_t>(lane) * kValuesPerThread;
  const int valid =
      active && offset < dim
          ? static_cast<int>(dim - offset < kValuesPerThread
                                 ? dim - offset
                                 : kValuesPerThread)
          : 0;
  ValueT *thread_smem = smem_dst + lane * kValuesPerThread;
  const ValueT *thread_src = valid > 0 ? src + offset : nullptr;
  if (valid == kValuesPerThread &&
      exchange_address_aligned(thread_src, kExchangeBytesPerThread)) {
    // CUDA's classic pipeline intrinsic lowers to per-thread cp.async on SM80+
    // and retains build compatibility with the existing SM75 target.
    __pipeline_memcpy_async(thread_smem, thread_src, 16);
  } else if (valid == kValuesPerThread &&
             exchange_address_aligned(thread_src, 8)) {
    constexpr int kValuesPer8B = 8 / sizeof(ValueT);
    __pipeline_memcpy_async(thread_smem, thread_src, 8);
    __pipeline_memcpy_async(thread_smem + kValuesPer8B,
                            thread_src + kValuesPer8B, 8);
  } else if (valid == kValuesPerThread &&
             exchange_address_aligned(thread_src, 4)) {
    constexpr int kValuesPer4B = 4 / sizeof(ValueT);
#pragma unroll
    for (int copy = 0; copy < 4; ++copy) {
      __pipeline_memcpy_async(thread_smem + copy * kValuesPer4B,
                              thread_src + copy * kValuesPer4B, 4);
    }
  } else {
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
      if (i < valid)
        thread_smem[i] = src[offset + i];
    }
  }
}

template <typename ValueT>
__forceinline__ __device__ void exchange_store_tile(
    ValueT *dst, int64_t dim, int64_t tile_offset, const ValueT *smem_src,
    bool active) {
  constexpr int kValuesPerThread = kExchangeBytesPerThread / sizeof(ValueT);
  const int lane = threadIdx.x & (kExchangeWarpSize - 1);
  const int64_t offset =
      tile_offset + static_cast<int64_t>(lane) * kValuesPerThread;
  const int valid =
      active && offset < dim
          ? static_cast<int>(dim - offset < kValuesPerThread
                                 ? dim - offset
                                 : kValuesPerThread)
          : 0;
  const ValueT *thread_smem = smem_src + lane * kValuesPerThread;
  ValueT *thread_dst = valid > 0 ? dst + offset : nullptr;
  if (valid == kValuesPerThread &&
      exchange_address_aligned(thread_dst, 16)) {
    *reinterpret_cast<ExchangeBytes16 *>(dst + offset) =
        *reinterpret_cast<const ExchangeBytes16 *>(thread_smem);
  } else if (valid == kValuesPerThread &&
             exchange_address_aligned(thread_dst, 8)) {
    constexpr int kValuesPer8B = 8 / sizeof(ValueT);
    *reinterpret_cast<ExchangeBytes8 *>(thread_dst) =
        *reinterpret_cast<const ExchangeBytes8 *>(thread_smem);
    *reinterpret_cast<ExchangeBytes8 *>(thread_dst + kValuesPer8B) =
        *reinterpret_cast<const ExchangeBytes8 *>(thread_smem + kValuesPer8B);
  } else if (valid == kValuesPerThread &&
             exchange_address_aligned(thread_dst, 4)) {
    constexpr int kValuesPer4B = 4 / sizeof(ValueT);
#pragma unroll
    for (int copy = 0; copy < 4; ++copy) {
      *reinterpret_cast<ExchangeBytes4 *>(thread_dst + copy * kValuesPer4B) =
          *reinterpret_cast<const ExchangeBytes4 *>(thread_smem +
                                                     copy * kValuesPer4B);
    }
  } else {
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
      if (i < valid)
        dst[offset + i] = thread_smem[i];
    }
  }
}

template <typename ValueT, int Direction>
__forceinline__ __device__ void exchange_issue_stage(
    const ExchangeMetadata &metadata, int64_t tile_offset, ValueT *smem) {
  constexpr uint32_t kActiveFlag =
      Direction == kExchangeCacheToStorage
          ? kExchangeMetadataCacheToStorage
          : kExchangeMetadataStorageToCache;
  const bool active = (metadata.flags & kActiveFlag) != 0;
  const int64_t src_address =
      Direction == kExchangeCacheToStorage
          ? metadata.cache_to_storage_src
          : metadata.storage_to_cache_src;
  exchange_stage_tile_async(reinterpret_cast<const ValueT *>(src_address),
                            metadata.value_dim, tile_offset, smem, active);
  // Keep the following commit converged even when lanes took different
  // alignment/tail paths while issuing their classic cp.async operations.
  __syncwarp();
}

template <typename ValueT, int Direction>
__forceinline__ __device__ void exchange_store_stage(
    const ExchangeMetadata &metadata, int64_t tile_offset,
    const ValueT *smem) {
  constexpr uint32_t kActiveFlag =
      Direction == kExchangeCacheToStorage
          ? kExchangeMetadataCacheToStorage
          : kExchangeMetadataStorageToCache;
  const bool active = (metadata.flags & kActiveFlag) != 0;
  const int64_t dst_address =
      Direction == kExchangeCacheToStorage
          ? metadata.cache_to_storage_dst
          : metadata.storage_to_cache_dst;
  exchange_store_tile(reinterpret_cast<ValueT *>(dst_address),
                      metadata.value_dim, tile_offset, smem, active);
}

template <typename ValueT>
__forceinline__ __device__ void exchange_resolve_metadata(
    const ExchangeRawMetadata &raw, int local_input,
    ExchangeMetadata *metadata, int64_t num_tables,
    const int64_t *__restrict__ cache_table_ptrs,
    const int64_t *__restrict__ cache_table_value_dims,
    const int64_t *__restrict__ storage_table_ptrs,
    const int64_t *__restrict__ storage_table_value_dims,
    const int64_t *__restrict__ storage_table_bucket_offsets,
    int64_t storage_bucket_capacity) {
  *metadata = ExchangeMetadata{};
  if (!raw.valid[local_input])
    return;

  const int64_t table_id = raw.table_ids[local_input];
  const int64_t cache_row = raw.cache_rows[local_input];
  // Every input is provisioned in cache before exchange.  Continuing with a
  // negative row would let forward consume uninitialized memory.
  if (table_id < 0 || table_id >= num_tables || cache_row < 0) {
    __trap();
  }

  const bool storage_to_cache = raw.storage_founds[local_input];
  const bool cache_to_storage = raw.evicted_mask[local_input];
  metadata->flags = kExchangeMetadataValid |
                    (storage_to_cache
                         ? kExchangeMetadataStorageToCache
                         : 0U) |
                    (cache_to_storage
                         ? kExchangeMetadataCacheToStorage
                         : 0U);
  if (!storage_to_cache && !cache_to_storage)
    return;

  const int64_t cache_dim = cache_table_value_dims[table_id];
  const int64_t storage_dim = storage_table_value_dims[table_id];
  if (cache_dim <= 0 || cache_dim != storage_dim) {
    __trap();
  }
  metadata->value_dim = cache_dim;

  const int64_t bucket_begin = storage_table_bucket_offsets[table_id];
  const int64_t table_capacity =
      (storage_table_bucket_offsets[table_id + 1] - bucket_begin) *
      storage_bucket_capacity;
  const int64_t counter_begin = bucket_begin * storage_bucket_capacity;

  ValueT *cache_row_ptr =
      reinterpret_cast<ValueT *>(cache_table_ptrs[table_id]) +
      cache_row * cache_dim;
  if (storage_to_cache) {
    const int64_t storage_row = raw.storage_src_rows[local_input];
    const int64_t lookup_slot = raw.storage_lookup_slots[local_input];
    if (storage_row < 0 || lookup_slot < 0 ||
        lookup_slot >= table_capacity) {
      __trap();
    }
    metadata->storage_to_cache_src =
        reinterpret_cast<int64_t>(
            reinterpret_cast<const ValueT *>(storage_table_ptrs[table_id]) +
            storage_row * storage_dim);
    metadata->storage_to_cache_dst = reinterpret_cast<int64_t>(cache_row_ptr);
    metadata->storage_to_cache_ref_counter = counter_begin + lookup_slot;
  }

  if (cache_to_storage) {
    const int64_t storage_row = raw.storage_dst_rows[local_input];
    const int64_t insert_slot = raw.storage_dst_slots[local_input];
    if (storage_row < 0 || insert_slot < 0 || insert_slot >= table_capacity) {
      // A failed storage insertion would discard the only current copy of the
      // evicted cache row, so fail instead of silently dropping it.
      __trap();
    }
    metadata->cache_to_storage_src = reinterpret_cast<int64_t>(cache_row_ptr);
    metadata->cache_to_storage_dst =
        reinterpret_cast<int64_t>(
            reinterpret_cast<ValueT *>(storage_table_ptrs[table_id]) +
            storage_row * storage_dim);
    metadata->cache_to_storage_ref_counter = counter_begin + insert_slot;
  }
}

template <typename ValueT>
__global__ void exchange_cache_storage_values_kernel(
    int64_t batch, int64_t num_tables,
    const int64_t *__restrict__ cache_table_ptrs,
    const int64_t *__restrict__ cache_table_value_dims,
    const int64_t *__restrict__ storage_table_ptrs,
    const int64_t *__restrict__ storage_table_value_dims,
    const int64_t *__restrict__ storage_table_bucket_offsets,
    int64_t storage_bucket_capacity,
    int32_t *__restrict__ storage_ref_counter,
    const int64_t *__restrict__ input_table_ids,
    const int64_t *__restrict__ cache_slots,
    const int64_t *__restrict__ storage_src_rows,
    const int64_t *__restrict__ storage_lookup_slots,
    const bool *__restrict__ storage_founds,
    const int64_t *__restrict__ evicted_storage_slots,
    const int64_t *__restrict__ evicted_storage_dst_rows,
    const bool *__restrict__ evicted_mask) {

  // The fused callers guarantee that input keys/cache rows are unique and
  // that acquired storage lookup slots cannot be selected by the subsequent
  // publish-and-acquire insertion. Therefore cache rows alias only within the
  // same input pair, while storage source and destination rows never alias.
  // This contract is what permits the two directions to run independently.

  constexpr int kExchangeTileValues = kExchangeTileBytes / sizeof(ValueT);
  __shared__ ExchangeRawMetadata raw_metadata;
  __shared__ ExchangeMetadata metadata[kExchangeInputsPerBlock];
  __shared__ __align__(16) ValueT
      smem[kExchangeWarpsPerBlock][kExchangePipelineDepth]
          [kExchangeDirections][kExchangeTileValues];

  const int warp = threadIdx.x / kExchangeWarpSize;
  const int lane = threadIdx.x & (kExchangeWarpSize - 1);
  const int64_t block_stride =
      static_cast<int64_t>(gridDim.x) * kExchangeInputsPerBlock;

  for (int64_t block_input =
           static_cast<int64_t>(blockIdx.x) * kExchangeInputsPerBlock;
       block_input < batch; block_input += block_stride) {
    // One warp per tensor lets all eight metadata transactions be issued in
    // parallel. Each loop iteration is a contiguous 32-element lane access.
#pragma unroll
    for (int local_input = lane; local_input < kExchangeInputsPerBlock;
         local_input += kExchangeWarpSize) {
      const int64_t input_id = block_input + local_input;
      const bool valid_input = input_id < batch;
      switch (warp) {
      case 0:
        raw_metadata.table_ids[local_input] =
            valid_input ? input_table_ids[input_id] : -1;
        raw_metadata.valid[local_input] = valid_input;
        break;
      case 1:
        raw_metadata.cache_rows[local_input] =
            valid_input ? cache_slots[input_id] : -1;
        break;
      case 2:
        raw_metadata.storage_src_rows[local_input] =
            valid_input ? storage_src_rows[input_id] : -1;
        break;
      case 3:
        raw_metadata.storage_lookup_slots[local_input] =
            valid_input ? storage_lookup_slots[input_id] : -1;
        break;
      case 4:
        raw_metadata.storage_founds[local_input] =
            valid_input && storage_founds[input_id];
        break;
      case 5:
        raw_metadata.storage_dst_slots[local_input] =
            valid_input ? evicted_storage_slots[input_id] : -1;
        break;
      case 6:
        raw_metadata.storage_dst_rows[local_input] =
            valid_input ? evicted_storage_dst_rows[input_id] : -1;
        break;
      default:
        raw_metadata.evicted_mask[local_input] =
            valid_input && evicted_mask[input_id];
        break;
      }
    }
    __syncthreads();

    // Each warp resolves the rows it will process. The raw slot metadata
    // remains in shared memory through this step, matching the sparse API
    // directly without a compact/gathered layout.
    if (lane < kExchangeRowsPerWarp) {
      const int local_input = warp + lane * kExchangeWarpsPerBlock;
      exchange_resolve_metadata<ValueT>(
          raw_metadata, local_input, &metadata[local_input], num_tables,
          cache_table_ptrs, cache_table_value_dims, storage_table_ptrs,
          storage_table_value_dims, storage_table_bucket_offsets,
          storage_bucket_capacity);
    }
    __syncwarp();

    int64_t max_dim = 0;
    bool any_cache_to_storage = false;
#pragma unroll
    for (int row = 0; row < kExchangeRowsPerWarp; ++row) {
      const ExchangeMetadata &item =
          metadata[warp + row * kExchangeWarpsPerBlock];
      if ((item.flags &
           (kExchangeMetadataStorageToCache |
            kExchangeMetadataCacheToStorage)) != 0 &&
          item.value_dim > max_dim) {
        max_dim = item.value_dim;
      }
      any_cache_to_storage |=
          (item.flags & kExchangeMetadataCacheToStorage) != 0;
    }

    const int64_t num_tiles =
        (max_dim + kExchangeTileValues - 1) / kExchangeTileValues;
    for (int64_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      const int64_t tile_offset = tile_idx * kExchangeTileValues;

      // Prime depth - 1 ring slots. Each group contains both directions for
      // one row, and sparse/inactive rows still commit an empty group so every
      // lane observes the same pipeline sequence.
#pragma unroll
      for (int row = 0; row < kExchangePipelineDepth - 1; ++row) {
        exchange_issue_stage<ValueT, kExchangeCacheToStorage>(
            metadata[warp + row * kExchangeWarpsPerBlock], tile_offset,
            &smem[warp][row][kExchangeCacheToStorage][0]);
        exchange_issue_stage<ValueT, kExchangeStorageToCache>(
            metadata[warp + row * kExchangeWarpsPerBlock], tile_offset,
            &smem[warp][row][kExchangeStorageToCache][0]);
        __pipeline_commit();
      }

      // Rolling ring: wait for the oldest cache+storage snapshot, issue the
      // next row into the free tail slot, then exchange the completed head.
      // Tail storage reads can overlap the head's mapped-storage writes.
#pragma unroll
      for (int head = 0; head < kExchangeRowsPerWarp; ++head) {
        const int rows_left = kExchangeRowsPerWarp - head;
        const int pending =
            rows_left < kExchangePipelineDepth - 1
                ? rows_left
                : kExchangePipelineDepth - 1;
        // The loop is fully unrolled, making pending - 1 the immediate operand
        // required by cp.async.wait_group.
        __pipeline_wait_prior(pending - 1);
        __syncwarp();

        const int tail = head + kExchangePipelineDepth - 1;
        if (tail < kExchangeRowsPerWarp) {
          const int tail_slot = tail % kExchangePipelineDepth;
          const ExchangeMetadata &tail_metadata =
              metadata[warp + tail * kExchangeWarpsPerBlock];
          exchange_issue_stage<ValueT, kExchangeCacheToStorage>(
              tail_metadata, tile_offset,
              &smem[warp][tail_slot][kExchangeCacheToStorage][0]);
          exchange_issue_stage<ValueT, kExchangeStorageToCache>(
              tail_metadata, tile_offset,
              &smem[warp][tail_slot][kExchangeStorageToCache][0]);
          __pipeline_commit();
        }

        const int head_slot = head % kExchangePipelineDepth;
        const ExchangeMetadata &head_metadata =
            metadata[warp + head * kExchangeWarpsPerBlock];
        exchange_store_stage<ValueT, kExchangeCacheToStorage>(
            head_metadata, tile_offset,
            &smem[warp][head_slot][kExchangeCacheToStorage][0]);
        exchange_store_stage<ValueT, kExchangeStorageToCache>(
            head_metadata, tile_offset,
            &smem[warp][head_slot][kExchangeStorageToCache][0]);
        // Complete every lane's shared-memory reads before this ring slot is
        // eligible to become a tail on the next iteration.
        __syncwarp();
      }
    }

    // Every lane fences its own mapped-host stores before lane zero releases
    // the insertion references.  Lookup references stay pinned until all
    // storage reads have drained as well.
    if (any_cache_to_storage)
      __threadfence_system();
    __syncwarp();
    if (lane == 0) {
#pragma unroll
      for (int row = 0; row < kExchangeRowsPerWarp; ++row) {
        const ExchangeMetadata &item =
            metadata[warp + row * kExchangeWarpsPerBlock];
        if ((item.flags & kExchangeMetadataStorageToCache) != 0)
          atomicSub(storage_ref_counter +
                        item.storage_to_cache_ref_counter,
                    1);
        if ((item.flags & kExchangeMetadataCacheToStorage) != 0)
          atomicSub(storage_ref_counter +
                        item.cache_to_storage_ref_counter,
                    1);
      }
    }
    __syncwarp();
    __syncthreads();
  }
}

void exchange_cache_storage_values(
    at::Tensor cache_table_ptrs, at::Tensor cache_table_value_dims,
    at::Tensor storage_table_ptrs, at::Tensor storage_table_value_dims,
    at::Tensor storage_table_bucket_offsets, int64_t storage_bucket_capacity,
    at::Tensor storage_ref_counter, at::Tensor input_table_ids,
    at::Tensor cache_slots, at::Tensor storage_src_rows,
    at::Tensor storage_lookup_slots, at::Tensor storage_founds,
    at::Tensor evicted_storage_slots, at::Tensor evicted_storage_dst_rows,
    at::Tensor evicted_mask, int64_t target_grid_size,
    dyn_emb::DataType value_type) {

  TORCH_CHECK(cache_slots.defined(), "cache_slots must be defined");
  const c10::Device device = cache_slots.device();
  auto check_metadata = [&](const at::Tensor &tensor, const char *name,
                            at::ScalarType scalar_type) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == device, name,
                " must be on the same device as cache_slots");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 1, name, " must be 1-D");
    TORCH_CHECK(tensor.scalar_type() == scalar_type, name, " must have dtype ",
                scalar_type, ", got ", tensor.scalar_type());
  };

  check_metadata(cache_table_ptrs, "cache_table_ptrs", at::kLong);
  check_metadata(cache_table_value_dims, "cache_table_value_dims", at::kLong);
  check_metadata(storage_table_ptrs, "storage_table_ptrs", at::kLong);
  check_metadata(storage_table_value_dims, "storage_table_value_dims",
                 at::kLong);
  check_metadata(storage_table_bucket_offsets,
                 "storage_table_bucket_offsets", at::kLong);
  check_metadata(storage_ref_counter, "storage_ref_counter", at::kInt);
  check_metadata(input_table_ids, "input_table_ids", at::kLong);
  check_metadata(cache_slots, "cache_slots", at::kLong);
  check_metadata(storage_src_rows, "storage_src_rows", at::kLong);
  check_metadata(storage_lookup_slots, "storage_lookup_slots", at::kLong);
  check_metadata(storage_founds, "storage_founds", at::kBool);
  check_metadata(evicted_storage_slots, "evicted_storage_slots", at::kLong);
  check_metadata(evicted_storage_dst_rows, "evicted_storage_dst_rows",
                 at::kLong);
  check_metadata(evicted_mask, "evicted_mask", at::kBool);

  const int64_t num_tables = cache_table_ptrs.numel();
  TORCH_CHECK(num_tables > 0, "table pointer arrays must be non-empty");
  TORCH_CHECK(cache_table_value_dims.numel() == num_tables,
              "cache_table_value_dims must match cache_table_ptrs");
  TORCH_CHECK(storage_table_ptrs.numel() == num_tables,
              "cache and storage must have the same number of tables");
  TORCH_CHECK(storage_table_value_dims.numel() == num_tables,
              "storage_table_value_dims must match storage_table_ptrs");
  TORCH_CHECK(storage_table_bucket_offsets.numel() == num_tables + 1,
              "storage_table_bucket_offsets must have num_tables + 1 entries");
  TORCH_CHECK(storage_bucket_capacity > 0,
              "storage_bucket_capacity must be positive");

  const int64_t batch = cache_slots.numel();
  TORCH_CHECK(input_table_ids.numel() == batch,
              "input_table_ids must match cache_slots");
  TORCH_CHECK(storage_src_rows.numel() == batch,
              "storage_src_rows must match cache_slots");
  TORCH_CHECK(storage_lookup_slots.numel() == batch,
              "storage_lookup_slots must match cache_slots");
  TORCH_CHECK(storage_founds.numel() == batch,
              "storage_founds must match cache_slots");
  TORCH_CHECK(evicted_storage_slots.numel() == batch,
              "evicted_storage_slots must match cache_slots");
  TORCH_CHECK(evicted_storage_dst_rows.numel() == batch,
              "evicted_storage_dst_rows must match cache_slots");
  TORCH_CHECK(evicted_mask.numel() == batch,
              "evicted_mask must match cache_slots");
  TORCH_CHECK(value_type == dyn_emb::DataType::Float32 ||
                  value_type == dyn_emb::DataType::Float16 ||
                  value_type == dyn_emb::DataType::BFloat16,
              "value_type must be Float32, Float16, or BFloat16");
  TORCH_CHECK(target_grid_size > 0,
              "target_grid_size must be initialized to a positive value");

  if (batch == 0)
    return;

  c10::cuda::CUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr int kBlockSize = 256;
  static_assert(kBlockSize ==
                    kExchangeWarpsPerBlock * kExchangeWarpSize,
                "exchange launch must match the per-warp shared layout");
  const int64_t needed_blocks =
      (batch + kExchangeInputsPerBlock - 1) / kExchangeInputsPerBlock;
  const int grid_size = static_cast<int>(
      needed_blocks < target_grid_size ? needed_blocks : target_grid_size);

  DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, ValueType, [&] {
    exchange_cache_storage_values_kernel<ValueType>
        <<<grid_size, kBlockSize, 0, stream>>>(
            batch, num_tables,
            get_pointer<int64_t>(cache_table_ptrs),
            get_pointer<int64_t>(cache_table_value_dims),
            get_pointer<int64_t>(storage_table_ptrs),
            get_pointer<int64_t>(storage_table_value_dims),
            get_pointer<int64_t>(storage_table_bucket_offsets),
            storage_bucket_capacity, get_pointer<int32_t>(storage_ref_counter),
            get_pointer<int64_t>(input_table_ids),
            get_pointer<int64_t>(cache_slots),
            get_pointer<int64_t>(storage_src_rows),
            get_pointer<int64_t>(storage_lookup_slots),
            get_pointer<bool>(storage_founds),
            get_pointer<int64_t>(evicted_storage_slots),
            get_pointer<int64_t>(evicted_storage_dst_rows),
            get_pointer<bool>(evicted_mask));
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

  m.def("reduce_grads", &reduce_grads, "reduce grads",
        py::arg("reverse_indices"), py::arg("grads"), py::arg("num_unique"),
        py::arg("batch_size"), py::arg("out_dim"),
        py::arg("offsets") = py::none(), py::arg("D_offsets") = py::none(),
        py::arg("combiner") = -1, py::arg("total_D") = 0);

  m.def("gather_embedding", &gather_embedding,
        "Gather embedding based on index.", py::arg("input"), py::arg("output"),
        py::arg("index"));

  m.def("gather_embedding_pooled", &gather_embedding_pooled,
        "Gather embedding with pooling (SUM/MEAN) based on index and offsets.",
        py::arg("input"), py::arg("output"), py::arg("index"),
        py::arg("offsets"), py::arg("combiner"), py::arg("total_D"),
        py::arg("batch_size"), py::arg("D_offsets") = py::none(),
        py::arg("max_D") = 0);

  m.def("load_from_flat_table_contiguous", &load_from_flat_table_contiguous,
        "Load from flat table: contiguous copy (NumRegions=0, single-table "
        "dump/load).",
        py::arg("table_ptrs"), py::arg("indices"), py::arg("table_id"),
        py::arg("output"), py::arg("table_value_dims"),
        py::arg("table_emb_dims"), py::arg("max_emb_dim"),
        py::arg("all_dims_vec4"));

  m.def("load_from_flat_table_emb", &load_from_flat_table_emb,
        "Load from flat table: emb-only copy (NumRegions=1, EMBEDDING mode).",
        py::arg("table_ptrs"), py::arg("indices"), py::arg("table_ids"),
        py::arg("output"), py::arg("table_value_dims"),
        py::arg("table_emb_dims"), py::arg("max_emb_dim"),
        py::arg("all_dims_vec4"));

  m.def("load_from_flat_table_value", &load_from_flat_table_value,
        "Load from flat table: 2-region copy (NumRegions=2, VALUE mode).",
        py::arg("table_ptrs"), py::arg("indices"), py::arg("table_ids"),
        py::arg("output"), py::arg("table_value_dims"),
        py::arg("table_emb_dims"), py::arg("max_emb_dim"),
        py::arg("all_dims_vec4"));

  m.def("store_to_flat_table_contiguous", &store_to_flat_table_contiguous,
        "Store to flat table: contiguous copy (NumRegions=0, single-table "
        "dump/load).",
        py::arg("table_ptrs"), py::arg("indices"), py::arg("table_id"),
        py::arg("input"), py::arg("table_value_dims"),
        py::arg("table_emb_dims"), py::arg("max_emb_dim"),
        py::arg("all_dims_vec4"));

  m.def("store_to_flat_table_value", &store_to_flat_table_value,
        "Store to flat table: 2-region copy (NumRegions=2, VALUE mode).",
        py::arg("table_ptrs"), py::arg("indices"), py::arg("table_ids"),
        py::arg("input"), py::arg("table_value_dims"),
        py::arg("table_emb_dims"), py::arg("max_emb_dim"),
        py::arg("all_dims_vec4"));

  m.def(
      "exchange_cache_storage_values", &exchange_cache_storage_values,
      "Exchange physical value rows directly between cache and mapped storage.",
      py::arg("cache_table_ptrs"), py::arg("cache_table_value_dims"),
      py::arg("storage_table_ptrs"), py::arg("storage_table_value_dims"),
      py::arg("storage_table_bucket_offsets"),
      py::arg("storage_bucket_capacity"), py::arg("storage_ref_counter"),
      py::arg("input_table_ids"), py::arg("cache_slots"),
      py::arg("storage_src_rows"), py::arg("storage_lookup_slots"),
      py::arg("storage_founds"),
      py::arg("evicted_storage_slots"),
      py::arg("evicted_storage_dst_rows"),
      py::arg("evicted_mask"), py::arg("target_grid_size"),
      py::arg("value_type"));

  m.def("select_insert_failed_values", &select_insert_failed_values,
        "select_insert_failed_values", py::arg("indices"),
        py::arg("input_values"), py::arg("evicted_values"));
}
