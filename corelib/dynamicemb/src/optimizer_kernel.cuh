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
#include <stdexcept>

namespace dyn_emb {

template <typename wgrad_t, typename weight_t> struct OptimizierInput {
  const wgrad_t *wgrad_ptr;
  weight_t *weight_ptr;
  const uint32_t dim;
};

template <typename wgrad_t, typename weight_t, int kWarpSize = 32>
struct SgdVecOptimizer {
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
    weight_t *m_ptr = weight_ptr + input.dim;
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
    weight_t *m_ptr = weight_ptr + input.dim;
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
    weight_t *gt_ptr = weight_ptr + input.dim;

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
    weight_t *gt_ptr = weight_ptr + input.dim;

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
    weight_t *gt_ptr = weight_ptr + input.dim;

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
    weight_t *gt_ptr = weight_ptr + input.dim;

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

template <typename wgrad_t, typename weight_t, typename OptimizerFunc>
__global__ void update4_kernel(const uint32_t num_keys, const uint32_t dim,
                               const wgrad_t *grad_evs, weight_t **weight_evs,
                               const bool *masks, OptimizerFunc optimizer) {
  constexpr int kWarpSize = 32;
  const int warp_num_per_block = blockDim.x / kWarpSize;
  const int warp_id_in_block = threadIdx.x / kWarpSize;

  for (uint32_t ev_id = warp_num_per_block * blockIdx.x + warp_id_in_block;
       ev_id < num_keys; ev_id += gridDim.x * warp_num_per_block) {
    bool mask = masks ? masks[ev_id] : true;
    weight_t *weight_ptr = weight_evs[ev_id];
    const wgrad_t *grad_ptr = grad_evs + ev_id * dim;
    if ((!mask) or (weight_ptr == nullptr)) {
      continue;
    }
    OptimizierInput<wgrad_t, weight_t> input{grad_ptr, weight_ptr, dim};
    optimizer.update4(input);
  }
}

template <typename wgrad_t, typename weight_t, typename OptimizerFunc>
__global__ void update_kernel(const uint32_t num_keys, const uint32_t dim,
                              const wgrad_t *grad_evs, weight_t **weight_evs,
                              const bool *masks, OptimizerFunc optimizer) {
  constexpr int kWarpSize = 32;

  for (uint32_t ev_id = blockIdx.x; ev_id < num_keys; ev_id += gridDim.x) {
    bool mask = masks ? masks[ev_id] : true;
    weight_t *weight_ptr = weight_evs[ev_id];
    const wgrad_t *grad_ptr = grad_evs + ev_id * dim;
    if ((!mask) or (weight_ptr == nullptr)) {
      continue;
    }
    OptimizierInput<wgrad_t, weight_t> input{grad_ptr, weight_ptr, dim};
    optimizer.update(input);
  }
}

template <typename wgrad_t, typename weight_t, typename OptimizerFunc>
__global__ void update4_kernel_fused(const uint32_t num_keys,
                                     const uint32_t dim, const uint32_t val_dim,
                                     const wgrad_t *grad_evs,
                                     weight_t *weight_evs, const bool *masks,
                                     OptimizerFunc optimizer) {
  constexpr int kWarpSize = 32;
  const int warp_num_per_block = blockDim.x / kWarpSize;
  const int warp_id_in_block = threadIdx.x / kWarpSize;

  for (uint32_t ev_id = warp_num_per_block * blockIdx.x + warp_id_in_block;
       ev_id < num_keys; ev_id += gridDim.x * warp_num_per_block) {
    bool mask = masks ? masks[ev_id] : true;
    weight_t *weight_ptr = weight_evs + ev_id * val_dim;
    const wgrad_t *grad_ptr = grad_evs + ev_id * dim;
    if ((!mask) or (weight_ptr == nullptr)) {
      continue;
    }
    OptimizierInput<wgrad_t, weight_t> input{grad_ptr, weight_ptr, dim};
    optimizer.update4(input);
  }
}

template <typename wgrad_t, typename weight_t, typename OptimizerFunc>
__global__ void update_kernel_fused(const uint32_t num_keys, const uint32_t dim,
                                    const uint32_t val_dim,
                                    const wgrad_t *grad_evs,
                                    weight_t *weight_evs, const bool *masks,
                                    OptimizerFunc optimizer) {
  constexpr int kWarpSize = 32;

  for (uint32_t ev_id = blockIdx.x; ev_id < num_keys; ev_id += gridDim.x) {
    bool mask = masks ? masks[ev_id] : true;
    weight_t *weight_ptr = weight_evs + ev_id * val_dim;
    const wgrad_t *grad_ptr = grad_evs + ev_id * dim;
    if ((!mask) or (weight_ptr == nullptr)) {
      continue;
    }
    OptimizierInput<wgrad_t, weight_t> input{grad_ptr, weight_ptr, dim};
    optimizer.update(input);
  }
}

template <typename grad_t, typename emb_t> struct OptimizierInputV2 {
  const grad_t *grad_;
  const emb_t *emb_;
  emb_t *table_vector_;
  const uint32_t dim;
  bool initialized;

  DEVICE_INLINE const grad_t *grad_ptr() const { return grad_; }

  DEVICE_INLINE const emb_t *emb_input_ptr() const { return emb_; }

  DEVICE_INLINE emb_t *emb_output_ptr() const { return table_vector_; }

  DEVICE_INLINE bool state_initialized() const { return initialized; }

  DEVICE_INLINE emb_t *state_ptr() const { return table_vector_ + dim; }
};

template <typename grad_t, typename emb_t, int kWarpSize = 32>
struct SgdVecOptimizerV2 {
  const float lr;

  DEVICE_INLINE void update4(const OptimizierInputV2<grad_t, emb_t> &input) {

    constexpr int VecSize = 4;
    const int lane_id = threadIdx.x % kWarpSize;

    for (int i = 0; VecSize * (kWarpSize * i + lane_id) < input.dim; ++i) {
      int idx4 = VecSize * (kWarpSize * i + lane_id);
      Vec4T<grad_t> grad_vec;
      Vec4T<float> weight_vec;
      grad_vec.load(input.grad_ptr() + idx4);
      weight_vec.load(input.emb_input_ptr() + idx4);
      weight_vec.accumulate_multiply(grad_vec, -lr);
      weight_vec.store(input.emb_output_ptr() + idx4);
    }
  }

  DEVICE_INLINE void update(const OptimizierInputV2<grad_t, emb_t> &input) {

    for (int i = threadIdx.x; i < input.dim; i += blockDim.x) {
      grad_t tmp_grad = input.grad_ptr()[i];
      float tmp_weight = input.emb_input_ptr()[i];
      tmp_weight -= ((float)(tmp_grad)*lr);
      input.emb_output_ptr()[i] =
          TypeConvertFunc<emb_t, float>::convert(tmp_weight);
    }
  }
};

template <typename grad_t, typename emb_t, int kWarpSize = 32>
struct AdamVecOptimizerV2 {
  const float lr;
  const float beta1;
  const float beta2;
  const float eps;
  const float weight_decay;
  const uint32_t iter_num;

  DEVICE_INLINE void update4(const OptimizierInputV2<grad_t, emb_t> &input) {

    constexpr int VecSize = 4;
    const int lane_id = threadIdx.x % kWarpSize;
    emb_t *m_ptr = input.state_ptr();
    emb_t *v_ptr = m_ptr + input.dim;

    Vec4T<float> weight_vec;
    Vec4T<float> m_vec;
    Vec4T<float> v_vec;
    for (int i = 0; VecSize * (kWarpSize * i + lane_id) < input.dim; ++i) {
      int idx4 = VecSize * (kWarpSize * i + lane_id);
      weight_vec.load(input.emb_input_ptr() + idx4);
      if (input.state_initialized()) {
        m_vec.load(m_ptr + idx4);
        v_vec.load(v_ptr + idx4);
      } else {
        m_vec.reset();
        v_vec.reset();
      }

      // update m and v
      {
        Vec4T<float> grad_vec;
        grad_vec.load(input.grad_ptr() + idx4);
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

        weight_vec.store(input.emb_output_ptr() + idx4);
      }
    }
  }

  DEVICE_INLINE void update(const OptimizierInputV2<grad_t, emb_t> &input) {

    emb_t *m_ptr = input.state_ptr();
    emb_t *v_ptr = m_ptr + input.dim;

    for (int i = threadIdx.x; i < input.dim; i += blockDim.x) {
      float tmp_grad =
          TypeConvertFunc<float, grad_t>::convert(input.grad_ptr()[i]);
      float tmp_m = input.state_initialized()
                        ? TypeConvertFunc<float, emb_t>::convert(m_ptr[i])
                        : 0.0f;
      float tmp_v = input.state_initialized()
                        ? TypeConvertFunc<float, emb_t>::convert(v_ptr[i])
                        : 0.0f;
      float tmp_weight =
          TypeConvertFunc<float, emb_t>::convert(input.emb_input_ptr()[i]);

      tmp_m = beta1 * tmp_m + (1.0f - beta1) * tmp_grad;
      tmp_v = beta2 * tmp_v + (1.0f - beta2) * tmp_grad * tmp_grad;

      float tmp_mhat = tmp_m / (1.0f - __powf(beta1, iter_num));
      float tmp_vhat = tmp_v / (1.0f - __powf(beta2, iter_num));

      tmp_weight -= lr * ((tmp_mhat / (sqrtf(tmp_vhat) + eps)) +
                          weight_decay * tmp_weight);
      input.emb_output_ptr()[i] =
          TypeConvertFunc<emb_t, float>::convert(tmp_weight);
      m_ptr[i] = TypeConvertFunc<emb_t, float>::convert(tmp_m);
      v_ptr[i] = TypeConvertFunc<emb_t, float>::convert(tmp_v);
    }
  }
};

template <typename grad_t, typename emb_t, int kWarpSize = 32>
struct AdaGradVecOptimizerV2 {
  const float lr;
  const float eps;
  const float initial_accumulator;

  DEVICE_INLINE void update4(const OptimizierInputV2<grad_t, emb_t> &input) {

    constexpr int VecSize = 4;
    const int lane_id = threadIdx.x % kWarpSize;
    emb_t *gt_ptr = input.state_ptr();

    Vec4T<float> weight_vec;
    Vec4T<float> gt_vec;

    for (int i = 0; VecSize * kWarpSize * i + VecSize * lane_id < input.dim;
         ++i) {
      int idx4 = VecSize * kWarpSize * i + VecSize * lane_id;
      weight_vec.load(input.emb_input_ptr() + idx4);
      if (input.state_initialized()) {
        gt_vec.load(gt_ptr + idx4);
      } else {
        gt_vec.reset(initial_accumulator);
      }

      Vec4T<float> grad_vec;
      grad_vec.load(input.grad_ptr() + idx4);
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

      weight_vec.store(input.emb_output_ptr() + idx4);
    }
  }

  DEVICE_INLINE void update(const OptimizierInputV2<grad_t, emb_t> &input) {

    emb_t *gt_ptr = input.state_ptr();

    for (int i = threadIdx.x; i < input.dim; i += blockDim.x) {
      float tmp_grad =
          TypeConvertFunc<float, grad_t>::convert(input.grad_ptr()[i]);
      float tmp_gt = input.state_initialized()
                         ? TypeConvertFunc<float, emb_t>::convert(gt_ptr[i])
                         : initial_accumulator;
      float tmp_weight =
          TypeConvertFunc<float, emb_t>::convert(input.emb_input_ptr()[i]);

      tmp_gt = tmp_gt + tmp_grad * tmp_grad;
      tmp_grad = lr * tmp_grad / (sqrtf(tmp_gt) + eps);

      tmp_weight -= tmp_grad;
      input.emb_output_ptr()[i] =
          TypeConvertFunc<emb_t, float>::convert(tmp_weight);
      gt_ptr[i] = TypeConvertFunc<emb_t, float>::convert(tmp_gt);
    }
  }
};

template <typename grad_t, typename emb_t, int kWarpSize = 32>
struct RowWiseAdaGradVecOptimizerV2 {
  const float lr;
  const float eps;
  const float initial_accumulator;

  /// TODO: whether can load grad once like online-softmax.
  DEVICE_INLINE void update4(const OptimizierInputV2<grad_t, emb_t> &input) {

    constexpr int VecSize = 4;
    const int lane_id = threadIdx.x % kWarpSize;
    emb_t *gt_ptr = input.state_ptr();

    float tmp_gt = input.state_initialized()
                       ? TypeConvertFunc<float, emb_t>::convert(*gt_ptr)
                       : initial_accumulator;
    float tmp_g_pow = 0;
    /// TODO: vectorize
    for (int i = lane_id; i < input.dim; i += kWarpSize) {
      float tmp_g =
          TypeConvertFunc<float, grad_t>::convert(input.grad_ptr()[i]);
      tmp_g_pow += tmp_g * tmp_g;
    }

    tmp_g_pow = warp_reduce_sum_xor(tmp_g_pow, kWarpSize);
    tmp_g_pow /= input.dim;
    tmp_gt += tmp_g_pow;

    if (lane_id == 0) {
      *gt_ptr = TypeConvertFunc<emb_t, float>::convert(tmp_gt);
    }

    Vec4T<float> weight_vec;
    for (int i = 0; VecSize * (kWarpSize * i + lane_id) < input.dim; ++i) {
      int idx4 = VecSize * (kWarpSize * i + lane_id);
      Vec4T<float> grad_vec;

      grad_vec.load(input.grad_ptr() + idx4);
      weight_vec.load(input.emb_input_ptr() + idx4);

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

      weight_vec.store(input.emb_output_ptr() + idx4);
    }
  }

  DEVICE_INLINE void update(const OptimizierInputV2<grad_t, emb_t> &input) {

    extern __shared__ float sdata[];
    const uint32_t tid = threadIdx.x;
    const uint32_t blockSize = blockDim.x;
    const unsigned int pow2_size = nextPow2(blockSize) >> 1;

    emb_t *gt_ptr = input.state_ptr();

    float tmp_gt = input.state_initialized()
                       ? TypeConvertFunc<float, emb_t>::convert(*gt_ptr)
                       : initial_accumulator;
    float tmp_g_pow = 0;
    for (int i = tid; i < input.dim; i += blockSize) {
      float tmp_g =
          TypeConvertFunc<float, grad_t>::convert(input.grad_ptr()[i]);
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
      *gt_ptr = TypeConvertFunc<emb_t, float>::convert(tmp_gt);
    }

    for (int i = tid; i < input.dim; i += blockSize) {
      float tmp_grad =
          TypeConvertFunc<float, grad_t>::convert(input.grad_ptr()[i]);
      float tmp_weight =
          TypeConvertFunc<float, emb_t>::convert(input.emb_input_ptr()[i]);

      tmp_grad = lr * tmp_grad / (sqrtf(tmp_gt) + eps);

      tmp_weight -= tmp_grad;
      input.emb_output_ptr()[i] =
          TypeConvertFunc<emb_t, float>::convert(tmp_weight);
    }
  }
};

template <typename grad_t, typename emb_t, typename OptimizerFunc>
__global__ void update4_kernel_v2(const uint32_t num_keys, const uint32_t dim,
                                  const grad_t *grad_evs, emb_t **weight_evs,
                                  const emb_t *emb_evs, const bool *masks,
                                  OptimizerFunc optimizer) {
  constexpr int kWarpSize = 32;
  const int warp_num_per_block = blockDim.x / kWarpSize;
  const int warp_id_in_block = threadIdx.x / kWarpSize;

  for (uint32_t ev_id = warp_num_per_block * blockIdx.x + warp_id_in_block;
       ev_id < num_keys; ev_id += gridDim.x * warp_num_per_block) {
    bool mask = masks[ev_id];
    emb_t *weight_ptr = weight_evs[ev_id];
    const grad_t *grad_ptr = grad_evs + ev_id * dim;
    const emb_t *emb_ptr = emb_evs + ev_id * dim;
    if ((!mask) and (weight_ptr == nullptr)) {
      continue;
    }
    OptimizierInputV2<grad_t, emb_t> input{grad_ptr, emb_ptr, weight_ptr, dim,
                                           mask};
    optimizer.update4(input);
  }
}

template <typename grad_t, typename emb_t, typename OptimizerFunc>
__global__ void update_kernel_v2(const uint32_t num_keys, const uint32_t dim,
                                 const grad_t *grad_evs, emb_t **weight_evs,
                                 const emb_t *emb_evs, const bool *masks,
                                 OptimizerFunc optimizer) {
  constexpr int kWarpSize = 32;

  for (uint32_t ev_id = blockIdx.x; ev_id < num_keys; ev_id += gridDim.x) {
    bool mask = masks[ev_id];
    emb_t *weight_ptr = weight_evs[ev_id];
    const grad_t *grad_ptr = grad_evs + ev_id * dim;
    const emb_t *emb_ptr = emb_evs + ev_id * dim;
    if ((!mask) and (weight_ptr == nullptr)) {
      continue;
    }
    OptimizierInputV2<grad_t, emb_t> input{grad_ptr, emb_ptr, weight_ptr, dim,
                                           mask};
    optimizer.update(input);
  }
}

} // namespace dyn_emb
#endif // OPTIMIZER_KERNEL_H
