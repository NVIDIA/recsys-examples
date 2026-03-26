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

template <typename CopyDesc>
__global__ void one_to_one_atomic_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  constexpr int kWarpSize = 32;

  const int vec_length = copy_desc.get_vec_length();
  for (int i_ev = blockIdx.x; i_ev < copy_desc.num_vec_; i_ev += gridDim.x) {
    const src_type *tmp_src = copy_desc.get_src_ptr(i_ev);
    dst_type *tmp_dst = copy_desc.get_dst_ptr(i_ev);
    for (int i = threadIdx.x; i < vec_length; i += blockDim.x) {
      atomicAdd(tmp_dst + i, (dst_type)tmp_src[i]);
    }
  }
  return;
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void one_to_one_atomic_vec4_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;

  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int warp_num = blockDim.x >> 5;

  constexpr int kWarpSize = 32;
  constexpr int copy_width = 4;

  for (int i_ev = blockIdx.x * warp_num + warp_id; i_ev < copy_desc.num_vec_;
       i_ev += gridDim.x * warp_num) {
    const src_type *tmp_src = copy_desc.get_src_ptr(i_ev);
    dst_type *tmp_dst = copy_desc.get_dst_ptr(i_ev);
    int vec_length = copy_desc.get_vec_length();
    for (int i = 0;
         i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length;
         ++i) {
      Vec4T<float> src_elem;
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      src_elem.load(tmp_src + idx4, n);
      src_elem.atomic_store_accum(tmp_dst + idx4, n);
    }
  }
  return;
}

template <typename io_t,    // element type of input/output.
          typename accum_t, // element type of accumulator.
          typename id_t,
          int kWarpSize = 32>
__global__ void multi_to_one_reduce_kernel1_no_vec(
    int64_t num_vec, // n
    int64_t max_vec_length,
    const io_t *__restrict__ in_grads,     // [n]
    io_t *__restrict__ out_grads,          // [m]
    const id_t *__restrict__ original_ids, // there exists shuffle.
    const id_t *__restrict__ unique_ids, // multi-to-one mapping. [n] ~(0, m-1)
    accum_t *__restrict__ partial_buffer,
    id_t *__restrict__ partial_unique_ids) {

  const int block_id = blockIdx.x;
  const int block_num = gridDim.x;
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
    const io_t *tmp_src =
        in_grads + original_ids[global_index] * max_vec_length;
    vec_length = max_vec_length;
    if (threadIdx.x < vec_length)
      accum += (accum_t)(tmp_src[threadIdx.x]);

    // when key is change , write to dst
    if (sp < local_sample_num - 1) {
      id_t new_id = unique_ids[global_index + 1];
      if (new_id != tmp_dst_id) {
        io_t *tmp_dst = out_grads + tmp_dst_id * max_vec_length;
        if (threadIdx.x < vec_length)
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
      if (threadIdx.x < vec_length)
        tmp_dst[threadIdx.x] = (io_t)accum;
      if (threadIdx.x == 0) {
        // max(unique ids) < num_vec, therefore (num_vec + 1) means no partial
        // gradient.
        partial_unique_ids[blockIdx.x] = num_vec + 1;
      }
    } else {
      accum_t *tmp_partial_ptr = partial_buffer + blockIdx.x * max_vec_length;
      if (threadIdx.x < vec_length)
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

template <typename io_t,    // element type of input/output.
          typename accum_t, // element type of accumulator.
          typename id_t, int kMaxElemPerThread,
          int kWarpSize = 32>
__global__ void multi_to_one_reduce_kernel1_vec4(
    int64_t num_vec, // n
    int64_t max_vec_length,
    const io_t *__restrict__ in_grads,     // [n]
    io_t *__restrict__ out_grads,          // [m]
    const id_t *__restrict__ original_ids, // there exists shuffle.
    const id_t *__restrict__ unique_ids, // multi-to-one mapping. [n] ~(0, m-1)
    accum_t *__restrict__ partial_buffer,
    id_t *__restrict__ partial_unique_ids) {

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
    const io_t *tmp_src =
        in_grads + original_ids[global_index] * max_vec_length;
    vec_length = max_vec_length;
    for (int i = 0; i < kMaxElemPerThread &&
                    (4 * kWarpSize * i + 4 * lane_id) < vec_length;
         ++i) {
      Vec4T<io_t> src_elem;
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      src_elem.load(tmp_src + idx4, n);
      accum[i].accumulate(src_elem);
    }

    // when key is change , write to dst
    if (sp < local_sample_num - 1) {
      id_t new_id = unique_ids[global_index + 1];
      if (new_id != tmp_dst_id) {
        io_t *tmp_dst = out_grads + tmp_dst_id * max_vec_length;
        for (int i = 0; i < kMaxElemPerThread &&
                        (4 * kWarpSize * i + 4 * lane_id) < vec_length;
             ++i) {
          int idx4 = 4 * kWarpSize * i + 4 * lane_id;
          int n = min(vec_length - idx4, copy_width);
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
                      (4 * kWarpSize * i + 4 * lane_id) < vec_length;
           ++i) {
        int idx4 = 4 * kWarpSize * i + 4 * lane_id;
        int n = min(vec_length - idx4, copy_width);
        accum[i].store(tmp_dst + idx4, n);
        accum[i].reset();
      }
      if (lane_id == 0) {
        // max(unique ids) < num_vec, therefore (num_vec + 1) means no partial
        // gradient.
        partial_unique_ids[blockIdx.x * warp_num + warp_id] = num_vec + 1;
      }
    } else {
      for (int i = 0; i < kMaxElemPerThread &&
                      (4 * kWarpSize * i + 4 * lane_id) < vec_length;
           ++i) {
        int idx4 = 4 * kWarpSize * i + 4 * lane_id;
        int n = min(vec_length - idx4, copy_width);
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

template <typename io_t, typename accum_t, typename id_t, int kWarpSize = 32>
void multi_to_one_reduce(int64_t n, int64_t len_vec, const at::Tensor &in_grads,
                         at::Tensor &out_grads,
                         const at::Tensor &sorted_key_ids,
                         const at::Tensor &unique_key_ids,
                         at::Tensor &partial_buffer,
                         at::Tensor &partial_unique_ids, cudaStream_t &stream) {
  auto &device_prop = DeviceProp::getDeviceProp(in_grads.device().index());
  const uint64_t first_stage_key_num = n;
  const uint64_t second_stage_key_num = (n - 1) / kWarpSize + 1;
  constexpr uint64_t WGRAD_REDUCE_BLOCK_SIZE = 64;

  int grid_size = (first_stage_key_num - 1) / WGRAD_REDUCE_BLOCK_SIZE + 1;
  int block_size = WGRAD_REDUCE_BLOCK_SIZE;
  bool aligned = len_vec % 4 == 0;
  bool small_than_256 = len_vec <= 256;

  if (aligned && small_than_256) {
    if (len_vec <= 128) {
      multi_to_one_reduce_kernel1_vec4<io_t, accum_t, id_t, 1, kWarpSize>
          <<<grid_size, block_size, 0, stream>>>(
              n, len_vec, reinterpret_cast<io_t *>(in_grads.data_ptr()),
              reinterpret_cast<io_t *>(out_grads.data_ptr()),
              reinterpret_cast<id_t *>(sorted_key_ids.data_ptr()),
              reinterpret_cast<id_t *>(unique_key_ids.data_ptr()),
              reinterpret_cast<accum_t *>(partial_buffer.data_ptr()),
              reinterpret_cast<id_t *>(partial_unique_ids.data_ptr()));

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
              second_stage_key_num, n, second_local_sample,
              reinterpret_cast<accum_t *>(partial_buffer.data_ptr()),
              reinterpret_cast<id_t *>(partial_unique_ids.data_ptr()),
              reinterpret_cast<io_t *>(out_grads.data_ptr()), len_vec);
    } else if (len_vec <= 256) {
      multi_to_one_reduce_kernel1_vec4<io_t, accum_t, id_t, 2, kWarpSize>
          <<<grid_size, block_size, 0, stream>>>(
              n, len_vec, reinterpret_cast<io_t *>(in_grads.data_ptr()),
              reinterpret_cast<io_t *>(out_grads.data_ptr()),
              reinterpret_cast<id_t *>(sorted_key_ids.data_ptr()),
              reinterpret_cast<id_t *>(unique_key_ids.data_ptr()),
              reinterpret_cast<accum_t *>(partial_buffer.data_ptr()),
              reinterpret_cast<id_t *>(partial_unique_ids.data_ptr()));

      int second_grid_size =
          (second_stage_key_num - 1) / WGRAD_REDUCE_BLOCK_SIZE + 1;
      int second_local_sample = kWarpSize;
      /// TODO: the last param is 2?
      get_kernel_config_use_warp(
          device_prop.num_sms, device_prop.max_thread_per_sm,
          WGRAD_REDUCE_BLOCK_SIZE, device_prop.warp_size, second_stage_key_num,
          &second_grid_size, &second_local_sample, 1);
      if (second_local_sample < 8)
        second_local_sample = 8;
      multi_to_one_reduce_kernel2<io_t, accum_t, id_t, 2, kWarpSize>
          <<<second_grid_size, block_size, 0, stream>>>(
              second_stage_key_num, n, second_local_sample,
              reinterpret_cast<accum_t *>(partial_buffer.data_ptr()),
              reinterpret_cast<id_t *>(partial_unique_ids.data_ptr()),
              reinterpret_cast<io_t *>(out_grads.data_ptr()), len_vec);
    } else {
      throw std::runtime_error("DynamicEmb aligned wgrad reduce does not "
                               "support emb vector size > 256");
    }
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  } else {

    if (len_vec <= 1024) {
      int grid_size_unaligned = (first_stage_key_num - 1) / kWarpSize + 1;
      int block_size_unaligned = ((len_vec - 1) / kWarpSize + 1) * kWarpSize;

      // sorted_key_ids_h

      multi_to_one_reduce_kernel1_no_vec<io_t, accum_t, id_t, kWarpSize>
          <<<grid_size_unaligned, block_size_unaligned, 0, stream>>>(
              n, len_vec, reinterpret_cast<io_t *>(in_grads.data_ptr()),
              reinterpret_cast<io_t *>(out_grads.data_ptr()),
              reinterpret_cast<id_t *>(sorted_key_ids.data_ptr()),
              reinterpret_cast<id_t *>(unique_key_ids.data_ptr()),
              reinterpret_cast<accum_t *>(partial_buffer.data_ptr()),
              reinterpret_cast<id_t *>(partial_unique_ids.data_ptr()));

      DEMB_CUDA_KERNEL_LAUNCH_CHECK();

      int second_grid_size = (second_stage_key_num - 1) / kWarpSize + 1;
      int second_local_sample = kWarpSize;
      multi_to_one_reduce_kernel2_no_vec<io_t, accum_t, id_t, kWarpSize>
          <<<second_grid_size, block_size_unaligned, 0, stream>>>(
              second_stage_key_num, n, second_local_sample,
              reinterpret_cast<accum_t *>(partial_buffer.data_ptr()),
              reinterpret_cast<id_t *>(partial_unique_ids.data_ptr()),
              reinterpret_cast<io_t *>(out_grads.data_ptr()), len_vec);
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
                               cudaStream_t &stream) {
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
        multi_to_one_reduce<grad_t, accum_t, id_t, WarpSize>(
            num_key_, len_vec_, in_grads, out_grads, sorted_key_ids,
            unique_key_ids, partial_buffer, partial_unique_ids, stream);
      });
    });
  });
}

template <typename Key_t, typename Value_t>
__global__ void wgrad_reduction_kernel(
    const Key_t *unique_indices, const Key_t *inverse_indices,
    const Key_t *biased_offset, const Value_t *grads, Value_t *unique_buffer,
    int dim, int batch_size, int feature_num, int num_key, int combiner) {
  const int warpsize = 32;
  int tid = threadIdx.x;

  for (int i_ev = blockIdx.x * 2 + (tid / 32); i_ev < num_key;
       i_ev += gridDim.x * 2) {

    Key_t src_id = bs_upper_bound_sub_one(
        biased_offset, batch_size * feature_num + 1, (Key_t)i_ev);
    Value_t pooling_factor = 1.0f;
    if (combiner == 1) {
      pooling_factor = Value_t(static_cast<float>(biased_offset[src_id + 1] -
                                                  biased_offset[src_id]));
    }

    const Value_t *src_ptr = grads + src_id * dim;
    Key_t dst_id = inverse_indices[i_ev];
    Value_t *dst_ptr = unique_buffer + dst_id * dim;

    for (int i = tid % warpSize; i < dim; i += warpsize) {
      Value_t value = atomicAdd(dst_ptr + i, src_ptr[i] / pooling_factor);
    }
  }
}

void backward(void *grads, void *unique_buffer, void *unique_indices,
              void *inverse_indices, void *biased_offset, const int dim,
              const int batch_size, const int feature_num, const int num_key,
              int combiner, DataType key_type, DataType value_type,
              cudaStream_t stream) {
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, key_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, value_t, [&] {
      int block_size = 64;
      int grid_size = (num_key - 1) / 2 + 1;
      wgrad_reduction_kernel<<<grid_size, block_size, 0, stream>>>(
          (key_t *)unique_indices, (key_t *)inverse_indices,
          (key_t *)biased_offset, (value_t *)grads, (value_t *)unique_buffer,
          dim, batch_size, feature_num, num_key, combiner);
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SrcType, typename DstType, typename rev_t>
struct BackwardDedupSequenceCopyDesc {

  using SrcT = SrcType;
  using DstT = DstType;
  HOST_DEVICE_INLINE int get_vec_length() {
    // TODO:now only have one size
    return ev_size;
  }
  HOST_DEVICE_INLINE const SrcType *get_src_ptr(int i) {
    return src_ptr + i * ev_size;
  }
  HOST_DEVICE_INLINE DstType *get_dst_ptr(int i) {
    rev_t idx = reverse_idx_ptr[i];
    return dst_ptr + idx * ev_size;
  }

  int64_t num_vec_;
  int ev_size;
  const rev_t *__restrict__ reverse_idx_ptr;
  const SrcType *__restrict__ src_ptr;
  DstType *dst_ptr;
};

void one_to_one_atomic(void *grads, void *unique_indices, void *reverse_indices,
                       void *unique_grads, const int ev_size,
                       const int64_t key_num, const int64_t unique_key_num,
                       DataType rev_idx_type, DataType grad_type,
                       DataType key_type, int num_sms, cudaStream_t stream) {

  if (key_num == 0)
    return;

  constexpr int WGRAD_REDUCE_BLOCK_SIZE = 64;

  DISPATCH_INTEGER_DATATYPE_FUNCTION(rev_idx_type, rev_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, grad_t, [&] {
      using CopyDesc = BackwardDedupSequenceCopyDesc<grad_t, grad_t, rev_t>;
      CopyDesc copy_desc{key_num, ev_size, (rev_t *)reverse_indices,
                         (grad_t *)grads, (grad_t *)unique_grads};

      constexpr int WGRAD_REDUCE_BLOCK_SIZE = 64;
      int grid_size = (key_num - 1) / WGRAD_REDUCE_BLOCK_SIZE + 1;
      int block_size = WGRAD_REDUCE_BLOCK_SIZE;
      constexpr int MAX_THREADS_PER_BLOCK = 1024;
      if (ev_size % 4 != 0) {
        //  need to optimize for small ev_size
        int grid_dim = copy_desc.num_vec_;
        int block_dim =
            ev_size < MAX_THREADS_PER_BLOCK ? ev_size : MAX_THREADS_PER_BLOCK;

        one_to_one_atomic_kernel<CopyDesc>
            <<<grid_dim, block_dim, 0, stream>>>(copy_desc);
      } else {
        if (ev_size <= 128) {
          int grid_size = num_sms * 32; // 2048/64 =32
          if (copy_desc.num_vec_ < grid_size)
            grid_size = copy_desc.num_vec_;
          int block_size = WGRAD_REDUCE_BLOCK_SIZE;
          one_to_one_atomic_vec4_kernel<CopyDesc, 1>
              <<<grid_size, block_size, 0, stream>>>(copy_desc);
        } else if (ev_size <= 256) {
          int grid_size = num_sms * 32; // 2048/64 =32
          if (copy_desc.num_vec_ < grid_size)
            grid_size = copy_desc.num_vec_;
          int block_size = WGRAD_REDUCE_BLOCK_SIZE;
          one_to_one_atomic_vec4_kernel<CopyDesc, 2>
              <<<grid_size, block_size, 0, stream>>>(copy_desc);
        } else if (ev_size <= 1024) {
          int grid_dim = copy_desc.num_vec_;
          int block_dim =
              ev_size < MAX_THREADS_PER_BLOCK ? ev_size : MAX_THREADS_PER_BLOCK;
          one_to_one_atomic_kernel<CopyDesc>
              <<<grid_dim, block_dim, 0, stream>>>(copy_desc);
        } else {
          throw std::runtime_error(
              "dynamic emb does not support emb vector size > 1024");
        }
      }
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace dyn_emb
