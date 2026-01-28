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
#include "index_calculation.h"
#include "utils.h"
#include <cuda/std/tuple>
#include <iostream>
#include <torch/extension.h>
#include <type_traits>

namespace { // anonymous namespace

template <typename T>
HOST_DEVICE_INLINE int64_t bs_upper_bound_sub_one(const T *const arr,
                                                  int64_t num, T target) {
  int64_t start = 0;
  int64_t end = num;
  while (start < end) {
    int64_t middle = start + (end - start) / 2;
    T value = arr[middle];
    if (value <= target) {
      start = middle + 1;
    } else {
      end = middle;
    }
  }
  return (start == num && arr[start - 1] != target) ? num : start - 1;
}

template <typename key_t, typename seg_id_t = int32_t> struct SegmentedKey {
  struct Decomposer {
    __host__ __device__ cuda::std::tuple<seg_id_t &, key_t &>
    operator()(SegmentedKey<key_t, seg_id_t> &segmented_key) const {
      return {segmented_key.segment_id, segmented_key.key};
    }
  };
  seg_id_t segment_id;
  key_t key;
} __attribute__((packed));

template <typename key_t, typename seg_id_t,
          typename ComposeKey = SegmentedKey<key_t, seg_id_t>>
__global__ void decompose_segmented_key_kernel(
    const ComposeKey *__restrict__ compose_arr, key_t *__restrict__ keys_out,
    seg_id_t *__restrict__ segment_ids_out, const int64_t num_keys) {
  CUDA_1D_KERNEL_LOOP(tid, num_keys) {
    ComposeKey compose = compose_arr[tid];
    keys_out[tid] = compose.key;
    if (segment_ids_out != nullptr) {
      segment_ids_out[tid] = compose.segment_id;
    }
  }
}

template <typename key_t, typename value_t, typename seg_id_t = int32_t,
          typename id_t = value_t,
          typename ComposeKey = SegmentedKey<key_t, seg_id_t>>
__global__ void segmented_sort_input_init_kernel(
    const key_t *__restrict__ keys, value_t *__restrict__ values,
    const id_t *__restrict__ segment_offsets, const int64_t num_keys,
    const int32_t num_segment_offsets, ComposeKey *__restrict__ compose_arr,
    seg_id_t *__restrict__ segment_ids_out) {
  CUDA_1D_KERNEL_LOOP(tid, num_keys) {
    key_t tmp_key = keys[tid];
    if (values != nullptr) {
      values[tid] = static_cast<value_t>(tid);
    }
    seg_id_t segment_id = bs_upper_bound_sub_one(
        segment_offsets, num_segment_offsets, static_cast<id_t>(tid));
    if (compose_arr != nullptr) {
      ComposeKey compose;
      compose.segment_id = segment_id;
      compose.key = tmp_key;
      compose_arr[tid] = compose;
    }
    if (segment_ids_out != nullptr) {
      segment_ids_out[tid] = segment_id;
    }
  }
}

template <typename key_t, typename seg_id_t>
__global__ void set_keys_flag(const key_t *__restrict__ sorted_keys,
                              const seg_id_t *__restrict__ sorted_segment_ids,
                              uint32_t *__restrict__ key_flag_buffer,
                              const int64_t num_key) {
  CUDA_1D_KERNEL_LOOP(tid, num_key) {
    key_t local_key = sorted_keys[tid];
    seg_id_t segment_id = sorted_segment_ids[tid];
    uint32_t is_first = 0;
    if ((tid == 0) ||
        ((tid > 0) && ((sorted_keys[tid - 1] != local_key) ||
                       (sorted_segment_ids[tid - 1] != segment_id)))) {
      is_first = 1;
    }
    key_flag_buffer[tid] = is_first;
  }
}

/// TODO:optimize with reschedule
template <typename key_t, typename id_t>
__global__ void set_unique_keys_and_unique_ids(
    const key_t *__restrict__ sorted_keys,
    const uint32_t
        *__restrict__ key_flag_buffer, /// TODO:upgrade to uint64_t/int64_t
    key_t *__restrict__ unique_keys, id_t *__restrict__ unique_key_ids,
    const int64_t key_num) {
  CUDA_1D_KERNEL_LOOP(tid, key_num) {
    uint32_t key_buffer = key_flag_buffer[tid];
    unique_key_ids[tid] = key_buffer - 1;
    if ((tid > 0 && key_flag_buffer[tid - 1] != key_buffer) || tid == 0) {
      unique_keys[key_buffer - 1] = sorted_keys[tid];
    }
  }
}

} // anonymous namespace

namespace dyn_emb {

SegmentedSortDevice::SegmentedSortDevice(c10::Device &device, int64_t key_num,
                                         int32_t segment_num, DataType key_type,
                                         DataType value_type)
    : device_(device), key_num_(key_num), segment_num_(segment_num),
      key_type_(key_type), value_type_(value_type) {

#if CUB_VERSION >= 200200
  need_compose_flag_ = true;
  set_out_segment_ids_ = false;
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type_, key_t, [&] {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(value_type_, value_t, [&] {
      using ComposeKey = SegmentedKey<key_t, seg_id_t>;
      this->segmented_keys_in_ =
          at::empty({static_cast<int64_t>(key_num_ * (sizeof(ComposeKey)))},
                    at::TensorOptions().dtype(torch::kChar).device(device_));
      this->segmented_keys_out_ =
          at::empty({static_cast<int64_t>(key_num_ * (sizeof(ComposeKey)))},
                    at::TensorOptions().dtype(torch::kChar).device(device_));
      cub::DeviceRadixSort::SortPairs<ComposeKey, value_t>(
          nullptr, cub_sort_temp_bytes_, nullptr, nullptr, nullptr, nullptr,
          key_num_, ComposeKey::Decomposer{}, 0, sizeof(ComposeKey) * 8);
      cub_sort_temp_buffer_ =
          at::empty({static_cast<int64_t>(cub_sort_temp_bytes_)},
                    at::TensorOptions().dtype(torch::kChar).device(device_));
    });
  });
#else
  need_compose_flag_ = false;
  set_out_segment_ids_ = true;
#endif
}

void SegmentedSortDevice::operator()(
    const at::Tensor &keys_in, at::Tensor &values_in,
    const at::Tensor &segment_offsets, at::Tensor &keys_out,
    at::Tensor &values_out, at::Tensor &segment_ids_out, cudaStream_t &stream,
    bool set_input_value_to_idx, bool set_output_segment_ids) {
  if (key_num_ == 0)
    return;
  /// TODO: to optimize the grid_size
  auto &device_prop = DeviceProp::getDeviceProp(keys_in.device().index());
  constexpr int block_size = 256;
  const int grid_size = device_prop.total_threads / block_size;

  // initialize input.
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type_, key_t, [&] {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(value_type_, value_t, [&] {
      using ComposeKey = SegmentedKey<key_t, seg_id_t>;
      segmented_sort_input_init_kernel<key_t, value_t, seg_id_t, value_t>
          <<<grid_size, block_size, 0, stream>>>(
              reinterpret_cast<key_t *>(keys_in.data_ptr()),
              set_input_value_to_idx
                  ? reinterpret_cast<value_t *>(values_in.data_ptr())
                  : nullptr,
              reinterpret_cast<value_t *>(segment_offsets.data_ptr()), key_num_,
              segment_num_ + 1,
              need_compose_flag_ ? reinterpret_cast<ComposeKey *>(
                                       segmented_keys_in_.data_ptr())
                                 : nullptr,
              set_out_segment_ids_
                  ? reinterpret_cast<seg_id_t *>(segment_ids_out.data_ptr())
                  : nullptr);
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  // sort
#if CUB_VERSION >= 200200
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type_, key_t, [&] {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(value_type_, value_t, [&] {
      using ComposeKey = SegmentedKey<key_t, seg_id_t>;
      cub::DeviceRadixSort::SortPairs(
          cub_sort_temp_buffer_.data_ptr(), cub_sort_temp_bytes_,
          reinterpret_cast<ComposeKey *>(segmented_keys_in_.data_ptr()),
          reinterpret_cast<ComposeKey *>(segmented_keys_out_.data_ptr()),
          reinterpret_cast<value_t *>(values_in.data_ptr()),
          reinterpret_cast<value_t *>(values_out.data_ptr()), key_num_,
          ComposeKey::Decomposer{}, 0, sizeof(ComposeKey) * 8, stream);
      decompose_segmented_key_kernel<<<grid_size, block_size, 0, stream>>>(
          reinterpret_cast<ComposeKey *>(segmented_keys_out_.data_ptr()),
          reinterpret_cast<key_t *>(keys_out.data_ptr()),
          set_output_segment_ids
              ? reinterpret_cast<seg_id_t *>(segment_ids_out.data_ptr())
              : nullptr,
          key_num_);
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
#else
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type_, key_t, [&] {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(value_type_, value_t, [&] {
      void *dummy_ptr = nullptr;
      cub::DeviceSegmentedRadixSort::SortPairs<key_t, value_t>(
          dummy_ptr, cub_sort_temp_bytes_,
          reinterpret_cast<key_t *>(keys_in.data_ptr()),
          reinterpret_cast<key_t *>(keys_out.data_ptr()),
          reinterpret_cast<value_t *>(values_in.data_ptr()),
          reinterpret_cast<value_t *>(values_out.data_ptr()), key_num_,
          segment_num_, reinterpret_cast<value_t *>(segment_offsets.data_ptr()),
          reinterpret_cast<value_t *>(segment_offsets.data_ptr()) + 1, 0,
          sizeof(key_t) * 8, stream);
      cub_sort_temp_buffer_ =
          at::empty({static_cast<int64_t>(cub_sort_temp_bytes_)},
                    at::TensorOptions().dtype(torch::kChar).device(device_));
      cub::DeviceSegmentedRadixSort::SortPairs(
          cub_sort_temp_buffer_.data_ptr(), cub_sort_temp_bytes_,
          reinterpret_cast<key_t *>(keys_in.data_ptr()),
          reinterpret_cast<key_t *>(keys_out.data_ptr()),
          reinterpret_cast<value_t *>(values_in.data_ptr()),
          reinterpret_cast<value_t *>(values_out.data_ptr()), key_num_,
          segment_num_, reinterpret_cast<value_t *>(segment_offsets.data_ptr()),
          reinterpret_cast<value_t *>(segment_offsets.data_ptr()) + 1, 0,
          sizeof(key_t) * 8, stream);
    });
  });
#endif
}

SegmentedUniqueDevice::SegmentedUniqueDevice(c10::Device &device,
                                             int64_t num_key, DataType key_type,
                                             DataType id_type)
    : device_(device), num_key_(num_key), key_type_(key_type),
      id_type_(id_type) {
  key_flag_buffer_ =
      at::empty({static_cast<int64_t>(num_key_)},
                at::TensorOptions().dtype(torch::kUInt32).device(device_));
  cub::DeviceScan::InclusiveSum<uint32_t *, uint32_t *>(
      nullptr, cub_scan_temp_bytes_, nullptr, nullptr, num_key_);
  cub_scan_temp_buffer_ =
      at::empty({static_cast<int64_t>(cub_scan_temp_bytes_)},
                at::TensorOptions().dtype(torch::kChar).device(device_));
}

void SegmentedUniqueDevice::operator()(
    const at::Tensor &sorted_keys, const at::Tensor &sorted_segment_ids,
    at::Tensor &unique_keys,
    at::Tensor &unique_key_ids, // mapping from sorted keys to unique keys
    cudaStream_t &stream) {

  if (num_key_ == 0)
    return;
  auto &device_prop = DeviceProp::getDeviceProp(sorted_keys.device().index());
  const int block_size = 256;
  const int grid_size = device_prop.total_threads / block_size;

  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type_, key_t, [&] {
    set_keys_flag<key_t, seg_id_t><<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<key_t *>(sorted_keys.data_ptr()),
        reinterpret_cast<seg_id_t *>(sorted_segment_ids.data_ptr()),
        reinterpret_cast<uint32_t *>(key_flag_buffer_.data_ptr()), num_key_);
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  cub::DeviceScan::InclusiveSum(
      cub_scan_temp_buffer_.data_ptr(), cub_scan_temp_bytes_,
      reinterpret_cast<uint32_t *>(key_flag_buffer_.data_ptr()),
      reinterpret_cast<uint32_t *>(key_flag_buffer_.data_ptr()), num_key_,
      stream);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type_, key_t, [&] {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(id_type_, id_t, [&] {
      set_unique_keys_and_unique_ids<key_t, id_t>
          <<<grid_size, block_size, 0, stream>>>(
              reinterpret_cast<key_t *>(sorted_keys.data_ptr()),
              reinterpret_cast<uint32_t *>(key_flag_buffer_.data_ptr()),
              reinterpret_cast<key_t *>(unique_keys.data_ptr()),
              reinterpret_cast<id_t *>(unique_key_ids.data_ptr()), num_key_);
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename InT, typename OutT>
__global__ void get_table_range_kernel(int64_t num_table,
                                       int64_t feature_x_batch,
                                       InT const *__restrict__ offsets,
                                       OutT const *__restrict__ feature_offsets,
                                       OutT *__restrict__ table_range) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_table + 1) {
    OutT num_feature = feature_offsets[num_table];
    int64_t batch = feature_x_batch / num_feature;
    OutT feature_offset = feature_offsets[tid];
    int64_t feature_x_batch_offset = feature_offset * batch;
    table_range[tid] = static_cast<OutT>(offsets[feature_x_batch_offset]);
  }
}

at::Tensor get_table_range(at::Tensor offsets, at::Tensor feature_offsets) {
  if (!offsets.is_cuda()) {
    throw std::runtime_error("Tensor <offsets> must be on CUDA device.");
  }
  if (!feature_offsets.is_cuda()) {
    throw std::runtime_error(
        "Tensor <feature_offsets> must be on CUDA device.");
  }
  int64_t feature_x_batch = offsets.size(0) - 1;
  int64_t num_table = feature_offsets.size(0) - 1;

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  at::Tensor table_range = at::empty_like(feature_offsets);

  int block_size = 128;
  if (num_table + 1 < block_size) {
    block_size = num_table + 1;
  }
  int grid_size = (num_table + block_size) / block_size;
  auto offset_type = scalartype_to_datatype(offsets.dtype().toScalarType());
  auto range_type =
      scalartype_to_datatype(feature_offsets.dtype().toScalarType());
  DISPATCH_OFFSET_INT_TYPE(offset_type, offset_t, [&] {
    DISPATCH_OFFSET_INT_TYPE(range_type, range_t, [&] {
      get_table_range_kernel<offset_t, range_t>
          <<<grid_size, block_size, 0, stream>>>(
              num_table, feature_x_batch,
              reinterpret_cast<offset_t *>(offsets.data_ptr()),
              reinterpret_cast<range_t *>(feature_offsets.data_ptr()),
              reinterpret_cast<range_t *>(table_range.data_ptr()));
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  return table_range;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
segmented_unique(
    at::Tensor keys, at::Tensor segment_range,
    std::shared_ptr<dyn_emb::UniqueOpBase> unique_op,
    const c10::optional<EvictStrategy> evict_strategy = c10::nullopt,
    const c10::optional<at::Tensor> frequency_counts_uint64 = c10::nullopt) {

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t num_total = keys.size(0);
  size_t unique_op_capacity = unique_op->get_capacity();
  if (num_total * 2 > unique_op_capacity) {
    at::Tensor new_keys = at::empty({num_total * 2}, keys.options());
    at::Tensor new_vals =
        at::empty({num_total * 2},
                  at::TensorOptions().dtype(at::kLong).device(keys.device()));
    unique_op->reset_capacity(new_keys, new_vals, num_total * 2, stream);
  }

  at::Tensor h_segment_range =
      at::empty(segment_range.sizes(),
                segment_range.options().device(at::kCPU).pinned_memory(true));
  h_segment_range.copy_(segment_range, /*non_blocking=*/true);

  int table_num = segment_range.size(0) - 1;

  // Determine score strategy
  bool is_lfu_enabled = false;
  if (evict_strategy.has_value()) {
    is_lfu_enabled = evict_strategy.value() == EvictStrategy::kLfu;
  }
  // Create vector of tensors for per-table frequency output
  bool need_frequency_output = false;
  need_frequency_output = is_lfu_enabled || frequency_counts_uint64.has_value();
  // Use single shared buffer instead of table_num separate buffers
  at::Tensor shared_unique_buffer = at::empty(num_total, keys.options());
  at::Tensor shared_frequency_buffer;
  if (need_frequency_output) {
    shared_frequency_buffer = at::zeros(
        num_total, at::TensorOptions().dtype(at::kLong).device(keys.device()));
  }
  at::Tensor d_unique_nums = at::empty(table_num, segment_range.options());
  at::Tensor d_unique_indices_table_range =
      at::zeros(table_num + 1, segment_range.options());

  auto unique_num_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(d_unique_nums.dtype()));
  auto unique_offset_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(d_unique_indices_table_range.dtype()));
  auto inverse_idx = at::empty(num_total, segment_range.options());

  // sync for h_segment_range
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  for (int i = 0; i < table_num; ++i) {
    int64_t indices_begin = h_segment_range[i].item<int64_t>();
    int64_t indices_end = h_segment_range[i + 1].item<int64_t>();
    int64_t indices_length = indices_end - indices_begin;

    if (indices_length == 0) {
      DEMB_CUDA_CHECK(cudaMemsetAsync(
          reinterpret_cast<int64_t *>(d_unique_nums.data_ptr()) + i, 0,
          sizeof(int64_t), stream));
      dyn_emb::add_offset(d_unique_nums.data_ptr(),
                          d_unique_indices_table_range.data_ptr(), i,
                          unique_num_type, unique_offset_type, stream);
    } else {
      at::Tensor tmp_indices = keys.slice(0, indices_begin, indices_end);
      at::Tensor tmp_inverse_idx =
          inverse_idx.slice(0, indices_begin, indices_end);
      at::Tensor tmp_d_unique_num = d_unique_nums.slice(0, i, table_num);

      at::Tensor previous_d_unique_num =
          d_unique_indices_table_range.slice(0, i, table_num + 1);

      at::Tensor tmp_unique_buffer_slice =
          shared_unique_buffer.slice(0, indices_begin, indices_end);

      at::Tensor tmp_frequency_counts_uint64;
      at::Tensor tmp_frequency_output_slice;
      if (frequency_counts_uint64.has_value()) {
        // LFU mode: use input frequency counts
        tmp_frequency_counts_uint64 = frequency_counts_uint64.value().slice(
            0, indices_begin, indices_end);
        tmp_frequency_output_slice =
            shared_frequency_buffer.slice(0, indices_begin, indices_end);
        unique_op->unique(tmp_indices, indices_length, tmp_inverse_idx,
                          tmp_unique_buffer_slice, tmp_d_unique_num, stream,
                          previous_d_unique_num, tmp_frequency_output_slice,
                          tmp_frequency_counts_uint64);
      } else if (need_frequency_output) {
        tmp_frequency_output_slice =
            shared_frequency_buffer.slice(0, indices_begin, indices_end);
        unique_op->unique(tmp_indices, indices_length, tmp_inverse_idx,
                          tmp_unique_buffer_slice, tmp_d_unique_num, stream,
                          previous_d_unique_num, tmp_frequency_output_slice);
      } else {
        // Non-LFU mode: call unique without frequency counting
        unique_op->unique(tmp_indices, indices_length, tmp_inverse_idx,
                          tmp_unique_buffer_slice, tmp_d_unique_num, stream,
                          previous_d_unique_num);
      }

      dyn_emb::add_offset(d_unique_nums.data_ptr(),
                          d_unique_indices_table_range.data_ptr(), i,
                          unique_num_type, unique_offset_type, stream);
    }
  }

  at::Tensor h_unique_indices_table_range =
      at::empty(table_num + 1, segment_range.options().device(at::kCPU));
  AT_CUDA_CHECK(cudaMemcpyAsync(h_unique_indices_table_range.data_ptr(),
                                d_unique_indices_table_range.data_ptr(),
                                (d_unique_indices_table_range.size(0)) *
                                    d_unique_indices_table_range.element_size(),
                                cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  int64_t unique_embs_offset = 0;
  int64_t num_unique_total =
      h_unique_indices_table_range[table_num].item<int64_t>();
  at::Tensor unique_keys = at::empty(num_unique_total, keys.options());
  at::Tensor output_scores;
  output_scores = at::Tensor();
  if (need_frequency_output) {
    output_scores = at::empty(num_unique_total, keys.options());
  }
  for (int i = 0; i < table_num; ++i) {
    int64_t tmp_unique_num =
        h_unique_indices_table_range[i + 1].item<int64_t>() -
        h_unique_indices_table_range[i].item<int64_t>();
    if (tmp_unique_num != 0) {
      int64_t indices_begin = h_segment_range[i].item<int64_t>();

      void *dst_ptr = reinterpret_cast<char *>(unique_keys.data_ptr()) +
                      unique_embs_offset * unique_keys.element_size();
      void *src_ptr =
          reinterpret_cast<char *>(shared_unique_buffer.data_ptr()) +
          indices_begin * shared_unique_buffer.element_size();
      size_t copy_size = tmp_unique_num * unique_keys.element_size();
      AT_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, copy_size,
                                    cudaMemcpyDeviceToDevice, stream));
      if (need_frequency_output) {
        void *dst_ptr = reinterpret_cast<char *>(output_scores.data_ptr()) +
                        unique_embs_offset * output_scores.element_size();
        void *src_ptr =
            reinterpret_cast<char *>(shared_frequency_buffer.data_ptr()) +
            indices_begin * shared_frequency_buffer.element_size();
        size_t copy_size = tmp_unique_num * output_scores.element_size();
        AT_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, copy_size,
                                      cudaMemcpyDeviceToDevice, stream));
      }
    }
    unique_embs_offset += tmp_unique_num;
  }
  return std::make_tuple(unique_keys, inverse_idx, d_unique_indices_table_range,
                         h_unique_indices_table_range, output_scores);
}

void select(at::Tensor flags, at::Tensor inputs, at::Tensor outputs,
            at::Tensor num_selected) {

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t num_total = inputs.size(0);
  auto scalar_type = inputs.dtype().toScalarType();
  auto key_type = scalartype_to_datatype(scalar_type);
  auto num_select_iter_type =
      scalartype_to_datatype(num_selected.dtype().toScalarType());

  if (num_total == 0) {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(
      num_select_iter_type, NumSelectedIteratorT, [&] {
        DEMB_CUDA_CHECK(cudaMemsetAsync(
          reinterpret_cast<NumSelectedIteratorT *>(num_selected.data_ptr()), 0,
          sizeof(NumSelectedIteratorT), stream));
    });
    return;
  }
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, KeyType, [&] {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(
        num_select_iter_type, NumSelectedIteratorT, [&] {
          select_async<KeyType, NumSelectedIteratorT>(
              num_total, flags.data_ptr<bool>(),
              reinterpret_cast<KeyType *>(inputs.data_ptr()),
              reinterpret_cast<KeyType *>(outputs.data_ptr()),
              reinterpret_cast<NumSelectedIteratorT *>(num_selected.data_ptr()),
              inputs.device(), stream);
        });
  });
}

void select_index(at::Tensor flags, at::Tensor output_indices,
                  at::Tensor num_selected) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t num_total = output_indices.size(0);
  auto scalar_type = output_indices.dtype().toScalarType();
  auto key_type = scalartype_to_datatype(scalar_type);
  auto num_select_iter_type =
      scalartype_to_datatype(num_selected.dtype().toScalarType());

  if (num_total == 0) {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(
      num_select_iter_type, NumSelectedIteratorT, [&] {
        DEMB_CUDA_CHECK(cudaMemsetAsync(
          reinterpret_cast<NumSelectedIteratorT *>(num_selected.data_ptr()), 0,
          sizeof(NumSelectedIteratorT), stream));
    });
    return;
  }
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, KeyType, [&] {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(
        num_select_iter_type, NumSelectedIteratorT, [&] {
          select_index_async<KeyType, NumSelectedIteratorT>(
              num_total, flags.data_ptr<bool>(),
              reinterpret_cast<KeyType *>(output_indices.data_ptr()),
              reinterpret_cast<NumSelectedIteratorT *>(num_selected.data_ptr()),
              output_indices.device(), stream);
        });
  });
}

} // namespace dyn_emb

void bind_index_calculation_op(py::module &m) {
  m.def("get_table_range", &dyn_emb::get_table_range,
        "Make offsets from <feature, batch> scope into <table> scope",
        py::arg("offsets"), py::arg("feature_offsets"));

  m.def("segmented_unique", &dyn_emb::segmented_unique,
        "Dose segmented unique operation on keys with segment_range, return "
        "tuple<unique_keys, inverse, unique_keys_table_range, "
        "h_unique_keys_table_range>",
        py::arg("keys"), py::arg("segment_range"), py::arg("unique_op"),
        py::arg("evict_strategy") = c10::nullopt,
        py::arg("frequency_counts_uint64") = c10::nullopt);

  m.def("select", &dyn_emb::select,
        "Select items in inputs which flags are true.", py::arg("flags"),
        py::arg("inputs"), py::arg("outputs"), py::arg("num_selected"));
  m.def("select_index", &dyn_emb::select_index,
        "Select items' indices where flags are true.", py::arg("flags"),
        py::arg("output_indices"), py::arg("num_selected"));
}
