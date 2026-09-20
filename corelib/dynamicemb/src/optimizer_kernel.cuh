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

#ifndef OPTIMIZER_KERNEL_H
#define OPTIMIZER_KERNEL_H

#include "lookup_kernel.cuh"
#include "utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace dyn_emb {

template <typename wgrad_t, typename weight_t> struct OptimizierInput {
  const wgrad_t *wgrad_ptr;
  weight_t *weight_ptr;
  // Per-row embedding width: number of weight/grad elements processed and the
  // width of every optimizer state region (e.g. Adam m and v are each `dim`).
  const uint32_t dim;
  // Offset (in elements) from weight_ptr to the first optimizer state region.
  // Compact flat-table layout packs states right after the embedding, so this
  // equals `dim`. The padded buffer reserves `max_emb_dim` for the embedding
  // region of every row, so states begin at `max_emb_dim` regardless of `dim`.
  const uint32_t state_offset;
};

template <typename wgrad_t, typename weight_t, int kWarpSize = 32>
struct SGDVecOptimizer {
  const float lr;

  DEVICE_INLINE void update4(const OptimizierInput<wgrad_t, weight_t> &input) {

    constexpr int VecSize = 4;
    const int lane_id = threadIdx.x % kWarpSize;
    const wgrad_t *wgrad_ptr = input.wgrad_ptr;
    weight_t *weight_ptr = input.weight_ptr;
    // If failed to insert into HKV, no need to update.
    if (weight_ptr == nullptr)
      return;
    for (int i = 0; VecSize * (kWarpSize * i + lane_id) < input.dim; ++i) {
      int idx4 = VecSize * (kWarpSize * i + lane_id);
      Vec4T<wgrad_t> grad_vec;
      Vec4T<float> weight_vec;
      grad_vec.load(wgrad_ptr + idx4);
      weight_vec.load(weight_ptr + idx4);
      weight_vec.accumulate_multiply(grad_vec, -lr);
      weight_vec.store(weight_ptr + idx4);
    }
  }

  DEVICE_INLINE void update(const OptimizierInput<wgrad_t, weight_t> &input) {

    const wgrad_t *wgrad_ptr = input.wgrad_ptr;
    weight_t *weight_ptr = input.weight_ptr;
    if (weight_ptr == nullptr)
      return;

    for (int i = threadIdx.x; i < input.dim; i += blockDim.x) {
      wgrad_t tmp_grad = wgrad_ptr[i];
      float tmp_weight = weight_ptr[i];
      tmp_weight -= ((float)(tmp_grad)*lr);
      weight_ptr[i] = (weight_t)tmp_weight;
    }
  }
};

template <typename wgrad_t, typename weight_t, int kWarpSize = 32>
struct AdamVecOptimizer {
  const float lr;
  const float beta1;
  const float beta2;
  const float eps;
  const float weight_decay;
  const uint32_t iter_num;

  DEVICE_INLINE void update4(const OptimizierInput<wgrad_t, weight_t> &input) {

    constexpr int VecSize = 4;
    const int lane_id = threadIdx.x % kWarpSize;
    const wgrad_t *wgrad_ptr = input.wgrad_ptr;
    weight_t *weight_ptr = input.weight_ptr;
    if (not weight_ptr)
      return;
    weight_t *m_ptr = weight_ptr + input.state_offset;
    weight_t *v_ptr = m_ptr + input.dim;

    Vec4T<float> weight_vec;
    Vec4T<float> m_vec;
    Vec4T<float> v_vec;
    for (int i = 0; VecSize * (kWarpSize * i + lane_id) < input.dim; ++i) {
      int idx4 = VecSize * (kWarpSize * i + lane_id);
      weight_vec.load(weight_ptr + idx4);
      m_vec.load(m_ptr + idx4);
      v_vec.load(v_ptr + idx4);

      // update m and v
      {
        Vec4T<float> grad_vec;
        grad_vec.load(wgrad_ptr + idx4);
        {
          m_vec.val.x = beta1 * m_vec.val.x + (1.0f - beta1) * grad_vec.val.x;
          m_vec.val.y = beta1 * m_vec.val.y + (1.0f - beta1) * grad_vec.val.y;
          m_vec.val.z = beta1 * m_vec.val.z + (1.0f - beta1) * grad_vec.val.z;
          m_vec.val.w = beta1 * m_vec.val.w + (1.0f - beta1) * grad_vec.val.w;
          m_vec.store(m_ptr + idx4);
        }

        {

          v_vec.val.x = beta2 * v_vec.val.x +
                        (1.0f - beta2) * grad_vec.val.x * grad_vec.val.x;
          v_vec.val.y = beta2 * v_vec.val.y +
                        (1.0f - beta2) * grad_vec.val.y * grad_vec.val.y;
          v_vec.val.z = beta2 * v_vec.val.z +
                        (1.0f - beta2) * grad_vec.val.z * grad_vec.val.z;
          v_vec.val.w = beta2 * v_vec.val.w +
                        (1.0f - beta2) * grad_vec.val.w * grad_vec.val.w;
          v_vec.store(v_ptr + idx4);
        }
      }

      // Get mhat and vhat
      {
        {
          m_vec.val.x = m_vec.val.x / (1.0f - __powf(beta1, iter_num));
          m_vec.val.y = m_vec.val.y / (1.0f - __powf(beta1, iter_num));
          m_vec.val.z = m_vec.val.z / (1.0f - __powf(beta1, iter_num));
          m_vec.val.w = m_vec.val.w / (1.0f - __powf(beta1, iter_num));
        }

        {
          v_vec.val.x = v_vec.val.x / (1.0f - __powf(beta2, iter_num));
          v_vec.val.y = v_vec.val.y / (1.0f - __powf(beta2, iter_num));
          v_vec.val.z = v_vec.val.z / (1.0f - __powf(beta2, iter_num));
          v_vec.val.w = v_vec.val.w / (1.0f - __powf(beta2, iter_num));
        }
        // Use m_vec as weight_update
        {
          m_vec.val.x = lr * ((m_vec.val.x / (sqrt(v_vec.val.x) + eps)) +
                              weight_vec.val.x * weight_decay);
          m_vec.val.y = lr * ((m_vec.val.y / (sqrt(v_vec.val.y) + eps)) +
                              weight_vec.val.y * weight_decay);
          m_vec.val.z = lr * ((m_vec.val.z / (sqrt(v_vec.val.z) + eps)) +
                              weight_vec.val.z * weight_decay);
          m_vec.val.w = lr * ((m_vec.val.w / (sqrt(v_vec.val.w) + eps)) +
                              weight_vec.val.w * weight_decay);
        }

        weight_vec.val.x -= m_vec.val.x;
        weight_vec.val.y -= m_vec.val.y;
        weight_vec.val.z -= m_vec.val.z;
        weight_vec.val.w -= m_vec.val.w;

        weight_vec.store(weight_ptr + idx4);
      }
    }
  }

  DEVICE_INLINE void update(const OptimizierInput<wgrad_t, weight_t> &input) {

    const wgrad_t *wgrad_ptr = input.wgrad_ptr;
    weight_t *weight_ptr = input.weight_ptr;
    if (not weight_ptr)
      return;
    weight_t *m_ptr = weight_ptr + input.state_offset;
    weight_t *v_ptr = m_ptr + input.dim;

    for (int i = threadIdx.x; i < input.dim; i += blockDim.x) {
      float tmp_grad = TypeConvertFunc<float, wgrad_t>::convert(wgrad_ptr[i]);
      float tmp_m = TypeConvertFunc<float, weight_t>::convert(m_ptr[i]);
      float tmp_v = TypeConvertFunc<float, weight_t>::convert(v_ptr[i]);
      float tmp_weight =
          TypeConvertFunc<float, weight_t>::convert(weight_ptr[i]);

      tmp_m = beta1 * tmp_m + (1.0f - beta1) * tmp_grad;
      tmp_v = beta2 * tmp_v + (1.0f - beta2) * tmp_grad * tmp_grad;

      float tmp_mhat = tmp_m / (1.0f - __powf(beta1, iter_num));
      float tmp_vhat = tmp_v / (1.0f - __powf(beta2, iter_num));

      tmp_weight -= lr * ((tmp_mhat / (sqrtf(tmp_vhat) + eps)) +
                          weight_decay * tmp_weight);
      weight_ptr[i] = TypeConvertFunc<weight_t, float>::convert(tmp_weight);
      m_ptr[i] = TypeConvertFunc<weight_t, float>::convert(tmp_m);
      v_ptr[i] = TypeConvertFunc<weight_t, float>::convert(tmp_v);
    }
  }
};

template <typename wgrad_t, typename weight_t, int kWarpSize = 32>
struct AdaGradVecOptimizer {
  const float lr;
  const float eps;

  DEVICE_INLINE void update4(const OptimizierInput<wgrad_t, weight_t> &input) {

    constexpr int VecSize = 4;
    const int lane_id = threadIdx.x % kWarpSize;
    const wgrad_t *wgrad_ptr = input.wgrad_ptr;
    weight_t *weight_ptr = input.weight_ptr;
    if (not weight_ptr)
      return;
    weight_t *gt_ptr = weight_ptr + input.state_offset;

    Vec4T<float> weight_vec;
    Vec4T<float> gt_vec;

    for (int i = 0; VecSize * kWarpSize * i + VecSize * lane_id < input.dim;
         ++i) {
      int idx4 = VecSize * kWarpSize * i + VecSize * lane_id;
      weight_vec.load(weight_ptr + idx4);
      gt_vec.load(gt_ptr + idx4);

      Vec4T<float> grad_vec;
      grad_vec.load(wgrad_ptr + idx4);
      {
        gt_vec.val.x += grad_vec.val.x * grad_vec.val.x;
        gt_vec.val.y += grad_vec.val.y * grad_vec.val.y;
        gt_vec.val.z += grad_vec.val.z * grad_vec.val.z;
        gt_vec.val.w += grad_vec.val.w * grad_vec.val.w;
        gt_vec.store(gt_ptr + idx4);
      }

      {
        grad_vec.val.x = lr * grad_vec.val.x / (sqrtf(gt_vec.val.x) + eps);
        grad_vec.val.y = lr * grad_vec.val.y / (sqrtf(gt_vec.val.y) + eps);
        grad_vec.val.z = lr * grad_vec.val.z / (sqrtf(gt_vec.val.z) + eps);
        grad_vec.val.w = lr * grad_vec.val.w / (sqrtf(gt_vec.val.w) + eps);
      }

      weight_vec.val.x -= grad_vec.val.x;
      weight_vec.val.y -= grad_vec.val.y;
      weight_vec.val.z -= grad_vec.val.z;
      weight_vec.val.w -= grad_vec.val.w;

      weight_vec.store(weight_ptr + idx4);
    }
  }

  DEVICE_INLINE void update(const OptimizierInput<wgrad_t, weight_t> &input) {

    const wgrad_t *wgrad_ptr = input.wgrad_ptr;
    weight_t *weight_ptr = input.weight_ptr;
    if (not weight_ptr)
      return;
    weight_t *gt_ptr = weight_ptr + input.state_offset;

    for (int i = threadIdx.x; i < input.dim; i += blockDim.x) {
      float tmp_grad = TypeConvertFunc<float, wgrad_t>::convert(wgrad_ptr[i]);
      float tmp_gt = TypeConvertFunc<float, weight_t>::convert(gt_ptr[i]);
      float tmp_weight =
          TypeConvertFunc<float, weight_t>::convert(weight_ptr[i]);

      tmp_gt = tmp_gt + tmp_grad * tmp_grad;
      tmp_grad = lr * tmp_grad / (sqrtf(tmp_gt) + eps);

      tmp_weight -= tmp_grad;
      weight_ptr[i] = TypeConvertFunc<weight_t, float>::convert(tmp_weight);
      gt_ptr[i] = TypeConvertFunc<weight_t, float>::convert(tmp_gt);
    }
  }
};

DEVICE_INLINE float warp_reduce_sum_xor(float val, int kWarpSize) {
  const unsigned full_mask = 0xFFFFFFFF;
#pragma unroll
  for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
    val += __shfl_xor_sync(full_mask, val, delta);
  }
  return val;
}

DEVICE_INLINE unsigned int nextPow2(unsigned int n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return ++n;
}

template <typename wgrad_t, typename weight_t, int kWarpSize = 32>
struct RowWiseAdaGradVecOptimizer {
  const float lr;
  const float eps;

  /// TODO: whether can load grad once like online-softmax.
  DEVICE_INLINE void update4(const OptimizierInput<wgrad_t, weight_t> &input) {

    constexpr int VecSize = 4;
    const int lane_id = threadIdx.x % kWarpSize;
    const wgrad_t *wgrad_ptr = input.wgrad_ptr;
    weight_t *weight_ptr = input.weight_ptr;

    if (not(weight_ptr))
      return;
    weight_t *gt_ptr = weight_ptr + input.state_offset;

    float tmp_gt = TypeConvertFunc<float, weight_t>::convert(*gt_ptr);
    float tmp_g_pow = 0;
    /// TODO: vectorize
    for (int i = lane_id; i < input.dim; i += kWarpSize) {
      float tmp_g = TypeConvertFunc<float, wgrad_t>::convert(wgrad_ptr[i]);
      tmp_g_pow += tmp_g * tmp_g;
    }

    tmp_g_pow = warp_reduce_sum_xor(tmp_g_pow, kWarpSize);
    tmp_g_pow /= input.dim;
    tmp_gt += tmp_g_pow;

    if (lane_id == 0) {
      *gt_ptr = TypeConvertFunc<weight_t, float>::convert(tmp_gt);
    }

    Vec4T<float> weight_vec;
    for (int i = 0; VecSize * (kWarpSize * i + lane_id) < input.dim; ++i) {
      int idx4 = VecSize * (kWarpSize * i + lane_id);
      Vec4T<float> grad_vec;

      grad_vec.load(wgrad_ptr + idx4);
      weight_vec.load(weight_ptr + idx4);

      {
        grad_vec.val.x = lr * grad_vec.val.x / (sqrtf(tmp_gt) + eps);
        grad_vec.val.y = lr * grad_vec.val.y / (sqrtf(tmp_gt) + eps);
        grad_vec.val.z = lr * grad_vec.val.z / (sqrtf(tmp_gt) + eps);
        grad_vec.val.w = lr * grad_vec.val.w / (sqrtf(tmp_gt) + eps);
      }

      weight_vec.val.x -= grad_vec.val.x;
      weight_vec.val.y -= grad_vec.val.y;
      weight_vec.val.z -= grad_vec.val.z;
      weight_vec.val.w -= grad_vec.val.w;

      weight_vec.store(weight_ptr + idx4);
    }
  }

  DEVICE_INLINE void update(const OptimizierInput<wgrad_t, weight_t> &input) {

    extern __shared__ float sdata[];
    const uint32_t tid = threadIdx.x;
    const uint32_t blockSize = blockDim.x;
    const unsigned int pow2_size = nextPow2(blockSize) >> 1;

    const wgrad_t *wgrad_ptr = input.wgrad_ptr;
    weight_t *weight_ptr = input.weight_ptr;

    if (not(weight_ptr))
      return;
    weight_t *gt_ptr = weight_ptr + input.state_offset;

    float tmp_gt = TypeConvertFunc<float, weight_t>::convert(*gt_ptr);
    float tmp_g_pow = 0;
    for (int i = tid; i < input.dim; i += blockSize) {
      float tmp_g = TypeConvertFunc<float, wgrad_t>::convert(wgrad_ptr[i]);
      tmp_g_pow += tmp_g * tmp_g;
    }

    sdata[tid] = tmp_g_pow;
    __syncthreads();

    if (pow2_size >= 1) {
      for (unsigned s = pow2_size; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < blockSize) {
          sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
      }
    }

    tmp_g_pow = sdata[0];
    tmp_g_pow /= input.dim;
    tmp_gt += tmp_g_pow;
    if (tid == 0) {
      *gt_ptr = TypeConvertFunc<weight_t, float>::convert(tmp_gt);
    }

    for (int i = tid; i < input.dim; i += blockSize) {
      float tmp_grad = TypeConvertFunc<float, wgrad_t>::convert(wgrad_ptr[i]);
      float tmp_weight =
          TypeConvertFunc<float, weight_t>::convert(weight_ptr[i]);

      tmp_grad = lr * tmp_grad / (sqrtf(tmp_gt) + eps);

      tmp_weight -= tmp_grad;
      weight_ptr[i] = TypeConvertFunc<weight_t, float>::convert(tmp_weight);
    }
  }
};

// FTRL-Proximal, Algorithm 1 of McMahan et al. 2013, generalized to an
// arbitrary learning-rate exponent the way TensorFlow's FtrlOptimizer does.
// Per coordinate the state is `linear` (the accumulated linear term, z in the
// paper) followed by `accum` (the sum of squared gradients, n), each `dim`
// wide -- the same two-region layout Adam uses for m and v.
//
// The weight is re-solved in closed form from (linear, accum) each step instead
// of being nudged from its previous value, which is what makes the L1 term
// produce exact zeros rather than merely small weights.
template <typename wgrad_t, typename weight_t, int kWarpSize = 32>
struct FTRLVecOptimizer {
  // alpha in the paper.
  const float lr;
  // The accumulator is raised to -learning_rate_power. The paper fixes that
  // exponent at a square root, which -0.5 recovers and `use_sqrt` selects; the
  // host decides, since the predicate is the same for every row.
  const float learning_rate_power;
  const bool use_sqrt;
  // beta, which keeps the per-coordinate learning rate finite while accum is
  // still small.
  const float beta;
  // lambda1 and lambda2 in the paper.
  const float l1_reg;
  const float l2_reg;

  DEVICE_INLINE float accum_pow(const float accum) const {
    return use_sqrt ? sqrtf(accum) : powf(accum, -learning_rate_power);
  }

  DEVICE_INLINE void update_one(float &weight, float &linear, float &accum,
                                const float grad) const {
    const float new_accum = accum + grad * grad;
    const float new_accum_pow = accum_pow(new_accum);
    linear += grad - (new_accum_pow - accum_pow(accum)) / lr * weight;
    accum = new_accum;

    // Computing the shrunk weight inside the branch keeps a 0/0 out of the
    // arithmetic when accum and beta are both still zero.
    if (fabsf(linear) > l1_reg) {
      const float sign_linear = (0.0f < linear) - (linear < 0.0f);
      weight = (l1_reg * sign_linear - linear) /
               ((beta + new_accum_pow) / lr + l2_reg);
    } else {
      weight = 0.0f;
    }
  }

  DEVICE_INLINE void update4(const OptimizierInput<wgrad_t, weight_t> &input) {

    constexpr int VecSize = 4;
    const int lane_id = threadIdx.x % kWarpSize;
    const wgrad_t *wgrad_ptr = input.wgrad_ptr;
    weight_t *weight_ptr = input.weight_ptr;
    if (not weight_ptr)
      return;
    weight_t *linear_ptr = weight_ptr + input.state_offset;
    weight_t *accum_ptr = linear_ptr + input.dim;

    Vec4T<float> weight_vec;
    Vec4T<float> linear_vec;
    Vec4T<float> accum_vec;
    Vec4T<float> grad_vec;

    for (int i = 0; VecSize * (kWarpSize * i + lane_id) < input.dim; ++i) {
      int idx4 = VecSize * (kWarpSize * i + lane_id);
      weight_vec.load(weight_ptr + idx4);
      linear_vec.load(linear_ptr + idx4);
      accum_vec.load(accum_ptr + idx4);
      grad_vec.load(wgrad_ptr + idx4);

      update_one(weight_vec.val.x, linear_vec.val.x, accum_vec.val.x,
                 grad_vec.val.x);
      update_one(weight_vec.val.y, linear_vec.val.y, accum_vec.val.y,
                 grad_vec.val.y);
      update_one(weight_vec.val.z, linear_vec.val.z, accum_vec.val.z,
                 grad_vec.val.z);
      update_one(weight_vec.val.w, linear_vec.val.w, accum_vec.val.w,
                 grad_vec.val.w);

      linear_vec.store(linear_ptr + idx4);
      accum_vec.store(accum_ptr + idx4);
      weight_vec.store(weight_ptr + idx4);
    }
  }

  DEVICE_INLINE void update(const OptimizierInput<wgrad_t, weight_t> &input) {

    const wgrad_t *wgrad_ptr = input.wgrad_ptr;
    weight_t *weight_ptr = input.weight_ptr;
    if (not weight_ptr)
      return;
    weight_t *linear_ptr = weight_ptr + input.state_offset;
    weight_t *accum_ptr = linear_ptr + input.dim;

    for (int i = threadIdx.x; i < input.dim; i += blockDim.x) {
      float tmp_grad = TypeConvertFunc<float, wgrad_t>::convert(wgrad_ptr[i]);
      float tmp_linear =
          TypeConvertFunc<float, weight_t>::convert(linear_ptr[i]);
      float tmp_accum = TypeConvertFunc<float, weight_t>::convert(accum_ptr[i]);
      float tmp_weight =
          TypeConvertFunc<float, weight_t>::convert(weight_ptr[i]);

      update_one(tmp_weight, tmp_linear, tmp_accum, tmp_grad);

      weight_ptr[i] = TypeConvertFunc<weight_t, float>::convert(tmp_weight);
      linear_ptr[i] = TypeConvertFunc<weight_t, float>::convert(tmp_linear);
      accum_ptr[i] = TypeConvertFunc<weight_t, float>::convert(tmp_accum);
    }
  }
};

template <typename wgrad_t, typename weight_t, typename index_t,
          typename OptimizerFunc>
__global__ void update4_with_index_flat_table_kernel(
    const uint32_t num_keys, const uint32_t grad_stride,
    const wgrad_t *grad_evs, const int64_t *table_ptrs, const index_t *indices,
    const int64_t *table_ids, const int64_t *table_value_dims,
    const int64_t *table_emb_dims, OptimizerFunc optimizer) {
  constexpr int kWarpSize = 32;
  const int warp_num_per_block = blockDim.x / kWarpSize;
  const int warp_id_in_block = threadIdx.x / kWarpSize;

  for (uint32_t ev_id = warp_num_per_block * blockIdx.x + warp_id_in_block;
       ev_id < num_keys; ev_id += gridDim.x * warp_num_per_block) {
    index_t const index = indices[ev_id];
    if (index == -1)
      continue;

    int64_t table_id = table_ids[ev_id];
    int64_t vdim = table_value_dims[table_id];
    int64_t edim = table_emb_dims[table_id];

    weight_t *weight_ptr = reinterpret_cast<weight_t *>(table_ptrs[table_id]) +
                           static_cast<int64_t>(index) * vdim;
    const wgrad_t *grad_ptr = grad_evs + ev_id * grad_stride;

    OptimizierInput<wgrad_t, weight_t> input{grad_ptr, weight_ptr,
                                             (uint32_t)edim, (uint32_t)edim};
    optimizer.update4(input);
  }
}

template <typename wgrad_t, typename weight_t, typename index_t,
          typename OptimizerFunc>
__global__ void update_with_index_flat_table_kernel(
    const uint32_t num_keys, const uint32_t grad_stride,
    const wgrad_t *grad_evs, const int64_t *table_ptrs, const index_t *indices,
    const int64_t *table_ids, const int64_t *table_value_dims,
    const int64_t *table_emb_dims, OptimizerFunc optimizer) {

  for (uint32_t ev_id = blockIdx.x; ev_id < num_keys; ev_id += gridDim.x) {
    index_t const index = indices[ev_id];
    if (index == -1)
      continue;

    int64_t table_id = table_ids[ev_id];
    int64_t vdim = table_value_dims[table_id];
    int64_t edim = table_emb_dims[table_id];

    weight_t *weight_ptr = reinterpret_cast<weight_t *>(table_ptrs[table_id]) +
                           static_cast<int64_t>(index) * vdim;
    const wgrad_t *grad_ptr = grad_evs + ev_id * grad_stride;

    OptimizierInput<wgrad_t, weight_t> input{grad_ptr, weight_ptr,
                                             (uint32_t)edim, (uint32_t)edim};
    optimizer.update(input);
  }
}

// The padded buffer reserves `max_emb_dim` for the embedding region of every
// row and packs the optimizer states immediately after it (see
// load_from_flat_table / store_to_flat_table). A row's embedding occupies
// [0, edim) and its states begin at `max_emb_dim`, so `dim` is the per-row
// embedding width while `state_offset` is the fixed `max_emb_dim`. When the
// table ids / per-table dims are unavailable (all rows share one dim) the
// caller passes nullptr and the uniform `max_emb_dim` is used as `dim`.
template <typename wgrad_t, typename weight_t, typename OptimizerFunc>
__global__ void update4_padded_buffer_kernel(
    const uint32_t num_rows, const uint32_t grad_stride,
    const uint32_t value_stride, const uint32_t max_emb_dim,
    const wgrad_t *grads, weight_t *values, const int64_t *table_ids,
    const int64_t *table_emb_dims, OptimizerFunc optimizer) {
  constexpr int kWarpSize = 32;
  const int warp_num_per_block = blockDim.x / kWarpSize;
  const int warp_id_in_block = threadIdx.x / kWarpSize;

  for (uint32_t row = warp_num_per_block * blockIdx.x + warp_id_in_block;
       row < num_rows; row += gridDim.x * warp_num_per_block) {
    uint32_t edim = table_emb_dims == nullptr
                        ? max_emb_dim
                        : (uint32_t)table_emb_dims[table_ids[row]];
    weight_t *weight_ptr = values + row * value_stride;
    const wgrad_t *grad_ptr = grads + row * grad_stride;
    OptimizierInput<wgrad_t, weight_t> input{grad_ptr, weight_ptr, edim,
                                             max_emb_dim};
    optimizer.update4(input);
  }
}

template <typename wgrad_t, typename weight_t, typename OptimizerFunc>
__global__ void update_padded_buffer_kernel(
    const uint32_t num_rows, const uint32_t grad_stride,
    const uint32_t value_stride, const uint32_t max_emb_dim,
    const wgrad_t *grads, weight_t *values, const int64_t *table_ids,
    const int64_t *table_emb_dims, OptimizerFunc optimizer) {
  for (uint32_t row = blockIdx.x; row < num_rows; row += gridDim.x) {
    uint32_t edim = table_emb_dims == nullptr
                        ? max_emb_dim
                        : (uint32_t)table_emb_dims[table_ids[row]];
    weight_t *weight_ptr = values + row * value_stride;
    const wgrad_t *grad_ptr = grads + row * grad_stride;
    OptimizierInput<wgrad_t, weight_t> input{grad_ptr, weight_ptr, edim,
                                             max_emb_dim};
    optimizer.update(input);
  }
}

} // namespace dyn_emb
#endif // OPTIMIZER_KERNEL_H
