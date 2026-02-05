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
              reinterpret_cast<NumSelectedIteratorT *>(num_selected.data_ptr()),
              0, sizeof(NumSelectedIteratorT), stream));
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
              reinterpret_cast<NumSelectedIteratorT *>(num_selected.data_ptr()),
              0, sizeof(NumSelectedIteratorT), stream));
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

  m.def("select", &dyn_emb::select,
        "Select items in inputs which flags are true.", py::arg("flags"),
        py::arg("inputs"), py::arg("outputs"), py::arg("num_selected"));
  m.def("select_index", &dyn_emb::select_index,
        "Select items' indices where flags are true.", py::arg("flags"),
        py::arg("output_indices"), py::arg("num_selected"));
}
