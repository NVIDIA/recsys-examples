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
#include "pooled_layout.cuh"
#include "torch_utils.h"
#include "utils.h"
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <limits>
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
    at::Tensor input, at::Tensor output, at::Tensor inverse_index,
    at::Tensor offsets, PoolingMode pooling_mode, int total_D, int batch_size,
    const std::optional<at::Tensor> &D_offsets = std::nullopt, int max_D = 0,
    const std::optional<at::Tensor> &weights = std::nullopt,
    bool feature_dims_vec4 = false) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int num_slots = static_cast<int>(offsets.size(0)) - 1;

  // Nothing to pool: no bags at all, or a batch with no samples in it. The
  // output already has no rows to fill, so there is no kernel to launch. Note
  // this is not the same as an empty *key* set: a batch whose bags are all
  // empty still has rows, and they have to be written as zeros, which the
  // kernel does on its own.  Mirrors the early return in reduce_grads.
  if (num_slots <= 0 || batch_size <= 0 || output.numel() == 0) {
    return;
  }
  // offsets is F*B+1 entries, so this is a malformed input rather than an
  // empty one.  Check it before deriving anything from the shapes.
  TORCH_CHECK(num_slots % batch_size == 0, "offsets holds ", num_slots,
              " bags, which is not divisible by batch_size (", batch_size, ")");

  const auto src_type = get_data_type(input);
  const auto dst_type = get_data_type(output);
  // One index type serves both: the descriptor reads the offsets and the
  // inverse indices through the same pointer type, so a mismatch here would
  // reinterpret one of them rather than convert it.  reduce_grads converts
  // instead; the forward has no reason to allocate, so it refuses.
  TORCH_CHECK(inverse_index.scalar_type() == offsets.scalar_type(),
              "inverse_index and offsets must have the same dtype, got ",
              inverse_index.scalar_type(), " vs ", offsets.scalar_type());
  const auto offset_type = get_data_type(offsets);

  int dim = D_offsets.has_value() ? max_D : static_cast<int>(input.size(1));
  int src_stride = static_cast<int>(input.stride(0));
  if (D_offsets.has_value()) {
    TORCH_CHECK(D_offsets.value().scalar_type() == at::kInt,
                "D_offsets must be int32, got ",
                D_offsets.value().scalar_type());
  }
  // The optional overload yields nullptr when there is none, which is exactly
  // what a uniform-dim layout wants.
  const int *d_D_offsets = get_pointer<const int>(D_offsets);
  const float *d_weights = nullptr;
  at::Tensor w;
  if (weights.has_value()) {
    w = weights.value().contiguous();
    TORCH_CHECK(w.scalar_type() == at::kFloat,
                "weights must be float32, got ", w.scalar_type());
    TORCH_CHECK(w.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(w.numel() == inverse_index.numel(),
                "weights.numel() (", w.numel(),
                ") must equal inverse_index.numel() (", inverse_index.numel(),
                ")");
    // MEAN divides the weighted sum by the pool size, which is neither a
    // weighted sum nor a weighted average -- reject it instead of silently
    // producing that.
    TORCH_CHECK(pooling_mode == PoolingMode::kSum,
                "weights require pooling_mode=SUM, got ",
                static_cast<int>(pooling_mode));
    d_weights = get_pointer<const float>(w);
  }
  // dim is the copy width and equals max_D either way: with D_offsets it is
  // max_D outright, without it it is the source row width, which is the
  // storage's max embedding dim -- the same number.
  const dyn_emb::PooledLayout layout{batch_size, num_slots / batch_size, total_D,
                                     dim, d_D_offsets};
  DISPATCH_INTEGER_DATATYPE_FUNCTION(offset_type, index_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(src_type, src_t, [&] {
      DISPATCH_FLOAT_DATATYPE_FUNCTION(dst_type, dst_t, [&] {
        dyn_emb::scatter_combine<src_t, dst_t, index_t>(
            get_pointer<const src_t>(input),
            get_pointer<dst_t>(output),
            get_pointer<const index_t>(offsets),
            get_pointer<const index_t>(inverse_index),
            pooling_mode,
            layout, src_stride, feature_dims_vec4, stream, d_weights);
      });
    });
  });
}
// For every key, the row of grads its gradient comes from, plus its pooling
// weight.
//
// offsets numbers bags in the input's order, grads in the output's, so the row
// is the transpose (see pooled_layout.cuh).  Doing it here, once per key, is
// what lets LocalReduce read grads directly instead of materializing a permuted
// copy of it.  One thread per bag; every key in a bag shares the bag's row but
// keeps its own weight.
template <typename index_t>
__global__ void
generate_grad_infos_pooled_kernel(const index_t *__restrict__ offsets,
                                  GradInfo<index_t> *__restrict__ grad_infos,
                                  PooledLayout layout,
                                  const float *__restrict__ weights) {
  const int num_bags = layout.num_bags();
  for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < num_bags;
       s += gridDim.x * blockDim.x) {
    const index_t row = static_cast<index_t>(layout.output_index_of_input(s));
    for (index_t j = offsets[s]; j < offsets[s + 1]; ++j) {
      grad_infos[j] = GradInfo<index_t>{row, weights ? weights[j] : 1.0f};
    }
  }
}

// Sequence lookups already have one grads row per key, so the row is the key's
// own index and nothing is pooled, hence no weight.
template <typename index_t>
__global__ void
generate_grad_infos_sequence_kernel(int64_t num_keys,
                                    GradInfo<index_t> *__restrict__ grad_infos) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < num_keys;
       i += (int64_t)gridDim.x * blockDim.x) {
    grad_infos[i] = GradInfo<index_t>{static_cast<index_t>(i), 1.0f};
  }
}

at::Tensor
reduce_grads(at::Tensor inverse_index, at::Tensor grads, int64_t num_unique,
             int batch_size, int64_t out_dim,
             const std::optional<at::Tensor> &offsets = std::nullopt,
             const std::optional<at::Tensor> &D_offsets = std::nullopt,
             PoolingMode pooling_mode = PoolingMode::kNone, int total_D = 0,
	     const std::optional<at::Tensor> &weights = std::nullopt,
             bool feature_dims_vec4 = false) {
  // Pooled (offsets present): grads is [B, total_D], numbered in the output's
  // order, while offsets numbers bags in the input's.  grad_infos bridges the
  // two -- one entry per key, holding the grads row that key's gradient comes
  // from -- so LocalReduce reads grads in place and no permuted copy of it is
  // ever materialized.  Mixed dims only change how a feature's columns are
  // found (layout.col_begin/col_width vs a uniform stride); MEAN scaling is
  // fused into the stage-1 kernel either way.  See pooled_layout.cuh.
  //
  // Sequence (offsets absent): one grads row per key already, so the row is
  // just the key's index.
  //
  // The pooling weight rides in the same GradInfo as the row, so the single
  // sort below carries it to the reduce; see GradInfo in lookup_backward.h for
  // why that beats sorting the weights separately.

  int64_t num_keys = inverse_index.size(0);

  if (!inverse_index.is_cuda() || !grads.is_cuda()) {
    throw std::runtime_error("All argument tensors should be on device");
  }

  auto device_ = inverse_index.device();
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto index_dtype = get_data_type(inverse_index);

  // Nothing to reduce.  No keys means no gradient reaches any row, and with no
  // unique rows there is nowhere to put one; num_unique is 0 whenever num_keys
  // is, so the answer really is an empty tensor rather than a shortcut.
  // Zeros rather than empty(): should a caller ever pass num_unique > 0 with no
  // keys, every row's gradient is genuinely zero, and handing back
  // uninitialized memory would feed garbage straight into the optimizer.
  // Mirrors the early return in gather_embedding_pooled.
  if (num_keys <= 0 || num_unique <= 0) {
    return at::zeros({num_unique, out_dim}, grads.options());
  }

  at::Tensor unique_grads = at::empty({num_unique, out_dim}, grads.options());

  // Validated before anything reads it, since the grad_infos kernel below is
  // the first consumer.
  const float *d_weights = nullptr;
  at::Tensor w;
  if (weights.has_value()) {
    w = weights.value().contiguous();
    TORCH_CHECK(w.scalar_type() == at::kFloat, "weights must be float32, got ",
                w.scalar_type());
    TORCH_CHECK(w.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(w.numel() == num_keys, "weights.numel() (", w.numel(),
                ") must equal inverse_index.numel() (", num_keys, ")");
    TORCH_CHECK(offsets.has_value(),
                "weights are only supported for pooled (offsets) reduce");
    // Must match the forward: see gather_embedding_pooled.
    TORCH_CHECK(pooling_mode == PoolingMode::kSum,
                "weights require pooling_mode=SUM, got ",
                static_cast<int>(pooling_mode));
    d_weights = get_pointer<const float>(w);
  }

  // GradInfo::grad_index, the offsets and the inverse indices are all
  // IndexType, so the offsets are converted when the caller's type disagrees.
  // This is F*B+1 elements against num_keys of everything else, and it buys a
  // halving of the reduce's instantiations.
  std::optional<at::Tensor> offs_opt;
  if (offsets.has_value()) {
    at::Tensor o = offsets.value();
    if (o.scalar_type() != inverse_index.scalar_type()) {
      o = o.to(inverse_index.scalar_type());
    }
    offs_opt = o;
  }

  constexpr int kBlockSize = 256;
  auto &device_prop = DeviceProp::getDeviceProp();
  const int max_grid_size =
      device_prop.num_sms * (device_prop.max_thread_per_sm / kBlockSize);

  // Only meaningful on the pooled path; the sequence path never reads it.
  dyn_emb::PooledLayout layout{0, 0, 0, 0, nullptr};
  int num_slots = 0;
  if (offs_opt.has_value()) {
    const at::Tensor &offs = offs_opt.value();
    // PooledLayout numbers bags with ints, so the bag count has to fit in one.
    // grad_index is IndexType and no longer constrains this.
    TORCH_CHECK(offs.numel() - 1 <= std::numeric_limits<int32_t>::max(),
                "reduce_grads: ", offs.numel() - 1,
                " bags exceeds what PooledLayout (int) can number");
    num_slots = static_cast<int>(offs.numel() - 1);
    TORCH_CHECK(batch_size > 0, "batch_size must be greater than 0");
    TORCH_CHECK(num_slots % batch_size == 0, "num_slots (", num_slots,
                ") must be divisible by batch_size (", batch_size, ")");
    int num_features = num_slots / batch_size;
    if (D_offsets.has_value()) {
      TORCH_CHECK(D_offsets.value().scalar_type() == at::kInt,
                  "D_offsets must be int32, got ",
                  D_offsets.value().scalar_type());
      TORCH_CHECK(D_offsets.value().numel() == num_features + 1,
                  "D_offsets.numel() (", D_offsets.value().numel(),
                  ") must equal num_features + 1 (", num_features + 1, ")");
    }
    // nullopt gives nullptr, which is how the kernels tell mixed dims from
    // uniform ones.
    const int *d_D_ptr = get_pointer<const int>(D_offsets);
    layout = {batch_size, num_features, total_D, static_cast<int>(out_dim),
              d_D_ptr};
  }

  // grad_infos is packed, so it is sized in bytes rather than viewed as an
  // integer pair.  Everything downstream of it is typed on index_t, hence the
  // single dispatch covering the build, the sort and the reduce.
  int end_bit =
      (num_unique > 1)
          ? (64 - __builtin_clzll(static_cast<uint64_t>(num_unique - 1)))
          : 1;
  DISPATCH_INTEGER_DATATYPE_FUNCTION(index_dtype, index_t, [&] {
    using Info = dyn_emb::GradInfo<index_t>;

    // --- Generate grad_infos ---
    at::Tensor grad_infos =
        at::empty({static_cast<int64_t>(num_keys * sizeof(Info))},
                  at::TensorOptions().dtype(at::kByte).device(device_));
    auto *d_grad_infos = get_pointer<Info>(grad_infos);

    if (offs_opt.has_value()) {
      // One thread per bag; see the kernel for why this is done up front.
      int slot_grid = static_cast<int>(
          std::min(((int64_t)num_slots + kBlockSize - 1) / kBlockSize,
                   (int64_t)max_grid_size));
      generate_grad_infos_pooled_kernel<index_t>
          <<<slot_grid, kBlockSize, 0, stream>>>(
              get_pointer<const index_t>(offs_opt.value()), d_grad_infos,
              layout, d_weights);
    } else {
      int key_grid = static_cast<int>(
          std::min((num_keys + kBlockSize - 1) / kBlockSize,
                   (int64_t)max_grid_size));
      generate_grad_infos_sequence_kernel<index_t>
          <<<key_grid, kBlockSize, 0, stream>>>(num_keys, d_grad_infos);
    }
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();

    // --- Sort (inverse_index, grad_infos) by inverse_index, so that every key
    //     landing on the same unique row ends up contiguous ---
    auto sorted_inverse_index = at::empty_like(inverse_index);
    auto sorted_grad_infos = at::empty_like(grad_infos);

    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes, get_pointer<index_t>(inverse_index),
        get_pointer<index_t>(sorted_inverse_index), d_grad_infos,
        get_pointer<Info>(sorted_grad_infos), num_keys, 0, end_bit, stream);
    auto temp_storage =
        at::empty({static_cast<int64_t>(temp_storage_bytes)},
                  at::TensorOptions().dtype(at::kByte).device(device_));
    cub::DeviceRadixSort::SortPairs(
        temp_storage.data_ptr(), temp_storage_bytes,
        get_pointer<index_t>(inverse_index),
        get_pointer<index_t>(sorted_inverse_index), d_grad_infos,
        get_pointer<Info>(sorted_grad_infos), num_keys, 0, end_bit, stream);

    // --- LocalReduce ---
    // MEAN scaling and the pooling weights are both fused inside the reduce
    // kernel, so no separate scaling pass is needed.
    LocalReduce<index_t> localReduceOp(device_, num_keys, out_dim,
                                       DataType::Float32);
    if (offs_opt.has_value()) {
      localReduceOp.local_reduce(grads, unique_grads, sorted_grad_infos,
                                 sorted_inverse_index, stream, layout,
                                 feature_dims_vec4, offs_opt, pooling_mode);
    } else {
      localReduceOp.local_reduce(grads, unique_grads, sorted_grad_infos,
                                 sorted_inverse_index, stream);
    }
  });

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

  // Single source of truth for the pooling mode: DynamicEmbPoolingMode on the
  // Python side takes its values from here, the same way
  // DynamicEmbEvictStrategy does for EvictStrategy.  Python imports it under
  // the alias BagPoolingMode, since fbgemm already exports a PoolingMode.
  // Single source of truth for the key -> rank rule's numbering: the codes in
  // dynamicemb/input_dist.py are read from here, so the kernel and the host
  // cannot drift apart on what a 1 or a 2 means.
  py::enum_<dyn_emb::DistType>(m, "DistType")
      .value("KBlock", dyn_emb::DistType::kBlock)
      .value("KCyclic", dyn_emb::DistType::kCyclic)
      .value("KHashedCyclic", dyn_emb::DistType::kHashedCyclic)
      .export_values();

  py::enum_<dyn_emb::PoolingMode>(m, "PoolingMode")
      .value("KSum", dyn_emb::PoolingMode::kSum)
      .value("KMean", dyn_emb::PoolingMode::kMean)
      .value("KNone", dyn_emb::PoolingMode::kNone)
      .export_values();
  // Keep plain ints working at the boundary so existing callers -- and
  // DynamicEmbPoolingMode, which is an IntEnum -- can be passed straight through.
  py::implicitly_convertible<py::int_, dyn_emb::PoolingMode>();

  m.def("reduce_grads", &reduce_grads, "reduce grads",
        py::arg("inverse_index"), py::arg("grads"), py::arg("num_unique"),
        py::arg("batch_size"), py::arg("out_dim"),
        py::arg("offsets") = py::none(), py::arg("D_offsets") = py::none(),
        py::arg("pooling_mode") = dyn_emb::PoolingMode::kNone,
        py::arg("total_D") = 0, py::arg("weights") = py::none(),
        py::arg("feature_dims_vec4") = false);

  m.def("gather_embedding", &gather_embedding,
        "Gather embedding based on index.", py::arg("input"), py::arg("output"),
        py::arg("index"));

  m.def("gather_embedding_pooled", &gather_embedding_pooled,
        "Gather embedding with pooling (SUM/MEAN) based on the dedup inverse "
        "index and offsets.",
        py::arg("input"), py::arg("output"), py::arg("inverse_index"),
        py::arg("offsets"), py::arg("pooling_mode"), py::arg("total_D"),
        py::arg("batch_size"), py::arg("D_offsets") = py::none(),
        py::arg("max_D") = 0, py::arg("weights") = py::none(),
        py::arg("feature_dims_vec4") = false);

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

  m.def("select_insert_failed_values", &select_insert_failed_values,
        "select_insert_failed_values", py::arg("indices"),
        py::arg("input_values"), py::arg("evicted_values"));
}
