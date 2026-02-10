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
#include "lookup_backward.h"
#include "lookup_kernel.cuh"

using namespace dyn_emb;

namespace {

// Stage-1 no-vec reduce kernel.
// kMultiDim=false: uniform dim — source is in_grads[gather_id *
// max_vec_length]. kMultiDim=true:  multi-dim   — source is in_grads[b*total_D
// + D_offsets[f]],
//   vec_length = D_f.  Writes use max_vec_length stride.
// MEAN scaling (1/pool_size) is fused for both modes when combiner==1.
template <typename io_t, typename accum_t, typename id_t, int kWarpSize = 32,
          bool kMultiDim = false, typename offset_t = int64_t>
__global__ void multi_to_one_reduce_kernel1_no_vec(
    int64_t num_vec, int64_t max_vec_length, const io_t *__restrict__ in_grads,
    io_t *__restrict__ out_grads, const id_t *__restrict__ original_ids,
    const id_t *__restrict__ unique_ids, accum_t *__restrict__ partial_buffer,
    id_t *__restrict__ partial_unique_ids, const int *__restrict__ D_offsets,
    int total_D, int F, const offset_t *__restrict__ offsets, int B,
    int combiner) {

  const int block_id = blockIdx.x;
  int local_sample_num = kWarpSize;

  int global_index = block_id * local_sample_num;
  if (global_index >= num_vec)
    return;
  local_sample_num = local_sample_num < num_vec - global_index
                         ? local_sample_num
                         : num_vec - global_index;

  accum_t accum = 0;
  id_t tmp_dst_id;
  int vec_length = -1;
  for (int sp = 0; sp < local_sample_num; ++sp) {
    tmp_dst_id = unique_ids[global_index];
    id_t gather_id = original_ids[global_index];

    const io_t *tmp_src;
    float scale = 1.0f;
    if constexpr (kMultiDim) {
      int f = static_cast<int>(gather_id) % F;
      int b = static_cast<int>(gather_id) / F;
      vec_length = D_offsets[f + 1] - D_offsets[f];
      tmp_src = in_grads + (int64_t)b * total_D + D_offsets[f];
      if (combiner == 1) {
        int slot = f * B + b;
        offset_t pool_size = offsets[slot + 1] - offsets[slot];
        if (pool_size > 0)
          scale = 1.0f / (float)pool_size;
      }
    } else {
      tmp_src = in_grads + gather_id * max_vec_length;
      vec_length = max_vec_length;
      if (combiner == 1) {
        int f = static_cast<int>(gather_id) % F;
        int b = static_cast<int>(gather_id) / F;
        int slot = f * B + b;
        offset_t pool_size = offsets[slot + 1] - offsets[slot];
        if (pool_size > 0)
          scale = 1.0f / (float)pool_size;
      }
    }

    if (threadIdx.x < vec_length)
      accum += (accum_t)(tmp_src[threadIdx.x]) * (accum_t)scale;

    // when key changes, write to dst and reset
    if (sp < local_sample_num - 1) {
      id_t new_id = unique_ids[global_index + 1];
      if (new_id != tmp_dst_id) {
        io_t *tmp_dst = out_grads + tmp_dst_id * max_vec_length;
        if (threadIdx.x < max_vec_length)
          tmp_dst[threadIdx.x] = (io_t)accum;
        accum = 0;
      }
    }
    global_index++;
  }

  if (vec_length != -1) {
    bool is_last = true;
    if (global_index < num_vec) {
      auto next_id = unique_ids[global_index];
      if (tmp_dst_id == next_id)
        is_last = false;
    }

    if (is_last) {
      io_t *tmp_dst = out_grads + tmp_dst_id * max_vec_length;
      if (threadIdx.x < max_vec_length)
        tmp_dst[threadIdx.x] = (io_t)accum;
      if (threadIdx.x == 0) {
        partial_unique_ids[blockIdx.x] = num_vec + 1;
      }
    } else {
      accum_t *tmp_partial_ptr = partial_buffer + blockIdx.x * max_vec_length;
      if (threadIdx.x < max_vec_length)
        tmp_partial_ptr[threadIdx.x] = accum;
      if (threadIdx.x == 0) {
        partial_unique_ids[blockIdx.x] = tmp_dst_id;
      }
    }
  }
  return;
}

template <typename io_t,    // element type of input/output.
          typename accum_t, // element type of accumulator.
          typename id_t, int kWarpSize = 32>
__global__ void multi_to_one_reduce_kernel2_no_vec(
    int64_t partial_num_vec, int64_t num_vec, int local_sample_num,
    const accum_t *__restrict__ partial_buffer,
    const id_t *__restrict__ partial_unique_ids, io_t *__restrict__ out_grads,
    int vec_length) {

  const int block_id = blockIdx.x;
  const int block_num = gridDim.x;

  int global_index = local_sample_num * block_id;
  if (global_index >= partial_num_vec)
    return;
  local_sample_num = local_sample_num < partial_num_vec - global_index
                         ? local_sample_num
                         : partial_num_vec - global_index;
  if (local_sample_num < 0)
    local_sample_num = 0;

  accum_t accum = 0;
  id_t tmp_dst_id;
  bool if_accum = false;
  for (int sp = 0; sp < local_sample_num; ++sp) {

    tmp_dst_id = partial_unique_ids[global_index];
    if_accum = tmp_dst_id < num_vec;
    if (if_accum) {
      const accum_t *tmp_src = partial_buffer + global_index * vec_length;
      if (threadIdx.x < vec_length)
        accum = tmp_src[threadIdx.x];
      io_t *tmp_dst = out_grads + tmp_dst_id * vec_length;
      if (threadIdx.x < vec_length) {
        atomicAdd(tmp_dst + threadIdx.x, (io_t)accum);
      }
    }
    global_index++;
  }

  return;
}

// Stage-1 vec4 reduce kernel.
// kMultiDim=false: uniform dim — source is in_grads[gather_id *
// max_vec_length]. kMultiDim=true:  multi-dim   — source is in_grads[b*total_D
// + D_offsets[f]],
//   vec_length = D_f.  Writes use max_vec_length stride.
// MEAN scaling (1/pool_size) is fused for both modes when combiner==1.
template <typename io_t, typename accum_t, typename id_t, int kMaxElemPerThread,
          int kWarpSize = 32, bool kMultiDim = false,
          typename offset_t = int64_t>
__global__ void multi_to_one_reduce_kernel1_vec4(
    int64_t num_vec, int64_t max_vec_length, const io_t *__restrict__ in_grads,
    io_t *__restrict__ out_grads, const id_t *__restrict__ original_ids,
    const id_t *__restrict__ unique_ids, accum_t *__restrict__ partial_buffer,
    id_t *__restrict__ partial_unique_ids, const int *__restrict__ D_offsets,
    int total_D, int F, const offset_t *__restrict__ offsets, int B,
    int combiner) {

  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int warp_num = blockDim.x >> 5;
  int local_sample_num = kWarpSize;
  constexpr int copy_width = 4;

  int global_index = kWarpSize * (blockIdx.x * warp_num + warp_id);
  if (global_index >= num_vec)
    return;
  local_sample_num = local_sample_num < num_vec - global_index
                         ? local_sample_num
                         : num_vec - global_index;

  Vec4T<accum_t> accum[kMaxElemPerThread]; // init to 0 by constructor.
  id_t tmp_dst_id;
  int vec_length = -1;
  for (int sp = 0; sp < local_sample_num; ++sp) {
    tmp_dst_id = unique_ids[global_index];
    id_t gather_id = original_ids[global_index];

    const io_t *tmp_src;
    float scale = 1.0f;
    if constexpr (kMultiDim) {
      int f = static_cast<int>(gather_id) % F;
      int b = static_cast<int>(gather_id) / F;
      vec_length = D_offsets[f + 1] - D_offsets[f];
      tmp_src = in_grads + (int64_t)b * total_D + D_offsets[f];
      if (combiner == 1) {
        int slot = f * B + b;
        offset_t pool_size = offsets[slot + 1] - offsets[slot];
        if (pool_size > 0)
          scale = 1.0f / (float)pool_size;
      }
    } else {
      tmp_src = in_grads + gather_id * max_vec_length;
      vec_length = max_vec_length;
      if (combiner == 1) {
        int f = static_cast<int>(gather_id) % F;
        int b = static_cast<int>(gather_id) / F;
        int slot = f * B + b;
        offset_t pool_size = offsets[slot + 1] - offsets[slot];
        if (pool_size > 0)
          scale = 1.0f / (float)pool_size;
      }
    }

    for (int i = 0; i < kMaxElemPerThread &&
                    (4 * kWarpSize * i + 4 * lane_id) < vec_length;
         ++i) {
      Vec4T<io_t> src_elem;
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      src_elem.load(tmp_src + idx4, n);
      accum[i].accumulate_multiply(src_elem, scale);
    }

    // when key changes, write to dst and reset
    if (sp < local_sample_num - 1) {
      id_t new_id = unique_ids[global_index + 1];
      if (new_id != tmp_dst_id) {
        io_t *tmp_dst = out_grads + tmp_dst_id * max_vec_length;
        for (int i = 0; i < kMaxElemPerThread &&
                        (4 * kWarpSize * i + 4 * lane_id) < max_vec_length;
             ++i) {
          int idx4 = 4 * kWarpSize * i + 4 * lane_id;
          int n = min((int)max_vec_length - idx4, copy_width);
          accum[i].store(tmp_dst + idx4, n);
          accum[i].reset();
        }
      }
    }
    global_index++;
  }

  if (vec_length != -1) {
    bool is_last = true;
    if (global_index < num_vec) {
      auto next_id = unique_ids[global_index];
      if (tmp_dst_id == next_id)
        is_last = false;
    }

    if (is_last) {
      io_t *tmp_dst = out_grads + tmp_dst_id * max_vec_length;
      for (int i = 0; i < kMaxElemPerThread &&
                      (4 * kWarpSize * i + 4 * lane_id) < max_vec_length;
           ++i) {
        int idx4 = 4 * kWarpSize * i + 4 * lane_id;
        int n = min((int)max_vec_length - idx4, copy_width);
        accum[i].store(tmp_dst + idx4, n);
        accum[i].reset();
      }
      if (lane_id == 0) {
        partial_unique_ids[blockIdx.x * warp_num + warp_id] = num_vec + 1;
      }
    } else {
      for (int i = 0; i < kMaxElemPerThread &&
                      (4 * kWarpSize * i + 4 * lane_id) < max_vec_length;
           ++i) {
        int idx4 = 4 * kWarpSize * i + 4 * lane_id;
        int n = min((int)max_vec_length - idx4, copy_width);
        accum[i].store(partial_buffer +
                           (blockIdx.x * warp_num + warp_id) * max_vec_length +
                           idx4,
                       n);
        accum[i].reset();
      }
      if (lane_id == 0) {
        partial_unique_ids[blockIdx.x * warp_num + warp_id] = tmp_dst_id;
      }
    }
  }
  return;
}

template <typename io_t,    // element type of input/output.
          typename accum_t, // element type of accumulator.
          typename id_t, int kMaxElemPerThread, int kWarpSize = 32>
__global__ void
multi_to_one_reduce_kernel2(int64_t partial_num_vec, int64_t num_vec,
                            int local_sample_num,
                            const accum_t *__restrict__ partial_buffer,
                            const id_t *__restrict__ partial_unique_ids,
                            io_t *__restrict__ out_grads, int vec_length) {

  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int warp_num = blockDim.x >> 5;
  constexpr int copy_width = 4;
  int global_index = local_sample_num * (blockIdx.x * warp_num + warp_id);
  if (global_index >= partial_num_vec)
    return;
  local_sample_num = local_sample_num < partial_num_vec - global_index
                         ? local_sample_num
                         : partial_num_vec - global_index;
  if (local_sample_num < 0)
    local_sample_num = 0;

  Vec4T<accum_t> accum[kMaxElemPerThread];
  id_t tmp_dst_id;
  bool if_accum = false;
  for (int sp = 0; sp < local_sample_num; ++sp) {
    tmp_dst_id = partial_unique_ids[global_index];
    if_accum = tmp_dst_id < num_vec;
    if (if_accum) {
      const accum_t *tmp_src = partial_buffer + global_index * vec_length;
      for (int i = 0; i < kMaxElemPerThread &&
                      (4 * kWarpSize * i + 4 * lane_id) < vec_length;
           ++i) {
        Vec4T<accum_t> src_elem;
        int idx4 = 4 * kWarpSize * i + 4 * lane_id;
        int n = min(vec_length - idx4, copy_width);
        src_elem.load(tmp_src + idx4, n);
        accum[i].accumulate(src_elem);
      }

      // when key is change , write to dst
      if (sp < local_sample_num - 1) {
        id_t new_id = partial_unique_ids[global_index + 1];
        if (new_id != tmp_dst_id) {
          io_t *tmp_dst = out_grads + tmp_dst_id * vec_length;
          for (int i = 0; i < kMaxElemPerThread &&
                          (4 * kWarpSize * i + 4 * lane_id) < vec_length;
               ++i) {
            int idx4 = 4 * kWarpSize * i + 4 * lane_id;
            int n = min(vec_length - idx4, copy_width);
            accum[i].atomic_store_accum(tmp_dst + idx4, n);
            accum[i].reset();
          }
        }
      }
    }
    global_index++;
  }

  if (if_accum) {
    io_t *tmp_dst = out_grads + tmp_dst_id * vec_length;
    for (int i = 0; i < kMaxElemPerThread &&
                    (4 * kWarpSize * i + 4 * lane_id) < vec_length;
         ++i) {
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      accum[i].atomic_store_accum(tmp_dst + idx4, n);
    }
  }
  return;
}

template <int NUM_VECTOR_PER_WARP = 32>
inline void get_kernel_config_use_warp(
    const int num_sms, const int num_thread_per_sm, const int block_size,
    const int warp_size, const int num_vector, int *grid_size,
    int *num_vector_per_warp, const int multiple_num = 4) {
  int warp_num_per_sm = num_thread_per_sm / warp_size;
  int warp_num_per_block = block_size / warp_size;
  int saturate_num = num_sms * warp_num_per_sm * multiple_num;

  if (num_vector <= saturate_num) {
    *num_vector_per_warp = 1;
    *grid_size = (num_vector - 1) / warp_num_per_block + 1;
    return;
  }

  if (num_vector / saturate_num >= NUM_VECTOR_PER_WARP) {
    *num_vector_per_warp = NUM_VECTOR_PER_WARP;
    *grid_size =
        (num_vector - 1) / (NUM_VECTOR_PER_WARP * warp_num_per_block) + 1;
  } else {
    *num_vector_per_warp = num_vector / saturate_num + 1;
    *grid_size = (saturate_num - 1) / warp_num_per_block + 1;
  }
  return;
}

// Unified dispatch: when d_D_offsets is non-null, stage-1 uses kMultiDim=true
// addressing (source is grads[B, total_D], per-feature offsets via D_offsets).
// Otherwise kMultiDim=false (uniform-dim).
// MEAN scaling (1/pool_size) is fused in stage-1 for both modes when
// combiner==1 and d_offsets is provided.
// Stage 2 is identical for both modes.
template <typename io_t, typename accum_t, typename id_t,
          typename offset_t = int64_t, int kWarpSize = 32>
void multi_to_one_reduce(int64_t n, int64_t len_vec, const at::Tensor &in_grads,
                         at::Tensor &out_grads,
                         const at::Tensor &sorted_key_ids,
                         const at::Tensor &unique_key_ids,
                         at::Tensor &partial_buffer,
                         at::Tensor &partial_unique_ids, cudaStream_t &stream,
                         const int *d_D_offsets = nullptr, int total_D = 0,
                         int F = 0, const offset_t *d_offsets = nullptr,
                         int B = 0, int combiner = 0) {
  const bool multi_dim = (d_D_offsets != nullptr);
  auto &device_prop = DeviceProp::getDeviceProp(in_grads.device().index());
  const uint64_t first_stage_key_num = n;
  const uint64_t second_stage_key_num = (n - 1) / kWarpSize + 1;
  constexpr uint64_t WGRAD_REDUCE_BLOCK_SIZE = 64;

  int grid_size = (first_stage_key_num - 1) / WGRAD_REDUCE_BLOCK_SIZE + 1;
  int block_size = WGRAD_REDUCE_BLOCK_SIZE;
  bool aligned = len_vec % 4 == 0;
  bool small_than_256 = len_vec <= 256;

  // Tensors cast to the same pointer type must have matching dtypes.
  TORCH_CHECK(in_grads.dtype() == out_grads.dtype(),
              "in_grads and out_grads must have the same dtype, got ",
              in_grads.dtype(), " vs ", out_grads.dtype());
  TORCH_CHECK(
      sorted_key_ids.dtype() == unique_key_ids.dtype(),
      "sorted_key_ids and unique_key_ids must have the same dtype, got ",
      sorted_key_ids.dtype(), " vs ", unique_key_ids.dtype());
  TORCH_CHECK(
      sorted_key_ids.dtype() == partial_unique_ids.dtype(),
      "sorted_key_ids and partial_unique_ids must have the same dtype, got ",
      sorted_key_ids.dtype(), " vs ", partial_unique_ids.dtype());

  // Common kernel args (same for both multi_dim and uniform).
  auto *p_in = reinterpret_cast<const io_t *>(in_grads.data_ptr());
  auto *p_out = reinterpret_cast<io_t *>(out_grads.data_ptr());
  auto *p_sorted = reinterpret_cast<const id_t *>(sorted_key_ids.data_ptr());
  auto *p_unique = reinterpret_cast<const id_t *>(unique_key_ids.data_ptr());
  auto *p_partial = reinterpret_cast<accum_t *>(partial_buffer.data_ptr());
  auto *p_partial_ids = reinterpret_cast<id_t *>(partial_unique_ids.data_ptr());

  if (aligned && small_than_256) {
    if (len_vec <= 128) {
      // Stage 1
      if (multi_dim) {
        multi_to_one_reduce_kernel1_vec4<io_t, accum_t, id_t, 1, kWarpSize,
                                         true, offset_t>
            <<<grid_size, block_size, 0, stream>>>(
                n, len_vec, p_in, p_out, p_sorted, p_unique, p_partial,
                p_partial_ids, d_D_offsets, total_D, F, d_offsets, B, combiner);
      } else {
        multi_to_one_reduce_kernel1_vec4<io_t, accum_t, id_t, 1, kWarpSize,
                                         false, offset_t>
            <<<grid_size, block_size, 0, stream>>>(
                n, len_vec, p_in, p_out, p_sorted, p_unique, p_partial,
                p_partial_ids, nullptr, 0, F, d_offsets, B, combiner);
      }
      // Stage 2
      int second_grid_size =
          (second_stage_key_num - 1) / WGRAD_REDUCE_BLOCK_SIZE + 1;
      int second_local_sample = kWarpSize;
      get_kernel_config_use_warp(
          device_prop.num_sms, device_prop.max_thread_per_sm,
          WGRAD_REDUCE_BLOCK_SIZE, device_prop.warp_size, second_stage_key_num,
          &second_grid_size, &second_local_sample, 1);
      if (second_local_sample < 8)
        second_local_sample = 8;
      multi_to_one_reduce_kernel2<io_t, accum_t, id_t, 1, kWarpSize>
          <<<second_grid_size, block_size, 0, stream>>>(
              second_stage_key_num, n, second_local_sample, p_partial,
              p_partial_ids, p_out, len_vec);
    } else if (len_vec <= 256) {
      // Stage 1
      if (multi_dim) {
        multi_to_one_reduce_kernel1_vec4<io_t, accum_t, id_t, 2, kWarpSize,
                                         true, offset_t>
            <<<grid_size, block_size, 0, stream>>>(
                n, len_vec, p_in, p_out, p_sorted, p_unique, p_partial,
                p_partial_ids, d_D_offsets, total_D, F, d_offsets, B, combiner);
      } else {
        multi_to_one_reduce_kernel1_vec4<io_t, accum_t, id_t, 2, kWarpSize,
                                         false, offset_t>
            <<<grid_size, block_size, 0, stream>>>(
                n, len_vec, p_in, p_out, p_sorted, p_unique, p_partial,
                p_partial_ids, nullptr, 0, F, d_offsets, B, combiner);
      }
      // Stage 2
      int second_grid_size =
          (second_stage_key_num - 1) / WGRAD_REDUCE_BLOCK_SIZE + 1;
      int second_local_sample = kWarpSize;
      get_kernel_config_use_warp(
          device_prop.num_sms, device_prop.max_thread_per_sm,
          WGRAD_REDUCE_BLOCK_SIZE, device_prop.warp_size, second_stage_key_num,
          &second_grid_size, &second_local_sample, 1);
      if (second_local_sample < 8)
        second_local_sample = 8;
      multi_to_one_reduce_kernel2<io_t, accum_t, id_t, 2, kWarpSize>
          <<<second_grid_size, block_size, 0, stream>>>(
              second_stage_key_num, n, second_local_sample, p_partial,
              p_partial_ids, p_out, len_vec);
    } else {
      throw std::runtime_error("DynamicEmb aligned wgrad reduce does not "
                               "support emb vector size > 256");
    }
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    if (len_vec <= 1024) {
      int grid_size_unaligned = (first_stage_key_num - 1) / kWarpSize + 1;
      int block_size_unaligned = ((len_vec - 1) / kWarpSize + 1) * kWarpSize;

      // Stage 1
      if (multi_dim) {
        multi_to_one_reduce_kernel1_no_vec<io_t, accum_t, id_t, kWarpSize, true,
                                           offset_t>
            <<<grid_size_unaligned, block_size_unaligned, 0, stream>>>(
                n, len_vec, p_in, p_out, p_sorted, p_unique, p_partial,
                p_partial_ids, d_D_offsets, total_D, F, d_offsets, B, combiner);
      } else {
        multi_to_one_reduce_kernel1_no_vec<io_t, accum_t, id_t, kWarpSize,
                                           false, offset_t>
            <<<grid_size_unaligned, block_size_unaligned, 0, stream>>>(
                n, len_vec, p_in, p_out, p_sorted, p_unique, p_partial,
                p_partial_ids, nullptr, 0, F, d_offsets, B, combiner);
      }
      DEMB_CUDA_KERNEL_LAUNCH_CHECK();

      // Stage 2
      int second_grid_size = (second_stage_key_num - 1) / kWarpSize + 1;
      int second_local_sample = kWarpSize;
      multi_to_one_reduce_kernel2_no_vec<io_t, accum_t, id_t, kWarpSize>
          <<<second_grid_size, block_size_unaligned, 0, stream>>>(
              second_stage_key_num, n, second_local_sample, p_partial,
              p_partial_ids, p_out, len_vec);
      DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      throw std::runtime_error(
          "DynamicEmb does not support emb vector size > 1024");
    }
  }
}

} // namespace

namespace dyn_emb {

LocalReduce::LocalReduce(c10::Device &device, int64_t num_key, int64_t len_vec,
                         DataType id_type, DataType accum_type)
    : device_(device), num_key_(num_key), len_vec_(len_vec), id_type_(id_type),
      accum_type_(accum_type) {

  int64_t len_partial_buffer = (num_key - 1) / WarpSize + 1;
  DISPATCH_FLOAT_ACCUM_TYPE_FUNC(accum_type_, accum_t, [&] {
    partial_buffer =
        at::empty({static_cast<int64_t>(len_partial_buffer * len_vec)},
                  at::TensorOptions()
                      .dtype(datatype_to_scalartype(accum_type_))
                      .device(device_));
  });

  DISPATCH_INTEGER_DATATYPE_FUNCTION(id_type_, id_t, [&] {
    partial_unique_ids = at::empty({static_cast<int64_t>(len_partial_buffer)},
                                   at::TensorOptions()
                                       .dtype(datatype_to_scalartype(id_type_))
                                       .device(device_));
  });
}

void LocalReduce::local_reduce(const at::Tensor &in_grads,
                               at::Tensor &out_grads,
                               const at::Tensor &sorted_key_ids,
                               const at::Tensor &unique_key_ids,
                               cudaStream_t &stream,
                               const std::optional<at::Tensor> &D_offsets,
                               const std::optional<at::Tensor> &offsets, int B,
                               int F, int total_D, int combiner) {
  if (num_key_ == 0)
    return;
  auto scalar_type = out_grads.dtype().toScalarType();
  auto tmp_type = in_grads.dtype().toScalarType();
  if (scalar_type != tmp_type) {
    throw std::runtime_error(
        "Input grad's dtype mismatches with output grad's.");
  }
  auto grad_type = scalartype_to_datatype(scalar_type);

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, grad_t, [&] {
    DISPATCH_FLOAT_ACCUM_TYPE_FUNC(accum_type_, accum_t, [&] {
      DISPATCH_INTEGER_DATATYPE_FUNCTION(id_type_, id_t, [&] {
        if (offsets.has_value()) {
          auto offset_type =
              scalartype_to_datatype(offsets.value().dtype().toScalarType());
          const int *d_D_ptr = nullptr;
          if (D_offsets.has_value()) {
            TORCH_CHECK(D_offsets.value().scalar_type() == at::kInt,
                        "D_offsets must be int32, got ",
                        D_offsets.value().scalar_type());
            TORCH_CHECK(D_offsets.value().numel() == F + 1,
                        "D_offsets.numel() (", D_offsets.value().numel(),
                        ") must equal F + 1 (", F + 1, ")");
            d_D_ptr =
                reinterpret_cast<const int *>(D_offsets.value().data_ptr());
          }
          DISPATCH_INTEGER_DATATYPE_FUNCTION(offset_type, offset_t, [&] {
            multi_to_one_reduce<grad_t, accum_t, id_t, offset_t, WarpSize>(
                num_key_, len_vec_, in_grads, out_grads, sorted_key_ids,
                unique_key_ids, partial_buffer, partial_unique_ids, stream,
                d_D_ptr, total_D, F,
                reinterpret_cast<const offset_t *>(offsets.value().data_ptr()),
                B, combiner);
          });
        } else {
          multi_to_one_reduce<grad_t, accum_t, id_t, int64_t, WarpSize>(
              num_key_, len_vec_, in_grads, out_grads, sorted_key_ids,
              unique_key_ids, partial_buffer, partial_unique_ids, stream);
        }
      });
    });
  });
}

} // namespace dyn_emb
