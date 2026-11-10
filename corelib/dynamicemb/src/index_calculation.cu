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
#include "counted_ops.h"
#include "index_calculation.h"
#include "utils.h"
#ifdef DEMB_USE_PYBIND11
#include <torch/extension.h>
#endif
#include <thrust/iterator/transform_iterator.h>

namespace dyn_emb {

namespace {

struct CastI32ToI64 {
  __host__ __device__ __forceinline__ int64_t operator()(int32_t x) const {
    return static_cast<int64_t>(x);
  }
};

// The flag of entry i in the counted compaction: set iff i < n and flags[i],
// with n = clamp(*num_valid, 0, capacity) read on the device (all of capacity
// when num_valid is null). Entries at or past n are never read, so whatever
// they hold selects nothing.
struct CountedFlag {
  bool const *flags;
  int64_t const *num_valid;
  int64_t capacity;
  __host__ __device__ __forceinline__ bool operator()(int64_t i) const {
    return i < counted_limit(num_valid, capacity) && flags[i];
  }
};

} // namespace

at::Tensor segmented_sum_cuda(at::Tensor data, at::Tensor offsets) {
  TORCH_CHECK(data.is_cuda(), "data must be on CUDA");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be on CUDA");
  TORCH_CHECK(offsets.dtype() == at::kLong, "offsets must be int64");
  TORCH_CHECK(data.dtype() == at::kInt, "data must be int32");

  int64_t num_segments = offsets.size(0) - 1;
  TORCH_CHECK(num_segments > 0,
              "offsets size must be at least 2 (num_segments >= 1)");

  auto device = data.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  at::Tensor output = torch::empty(
      {num_segments}, at::TensorOptions().dtype(at::kLong).device(device));

  const int64_t *d_offsets = offsets.data_ptr<int64_t>();
  const int32_t *d_data = data.data_ptr<int32_t>();
  int64_t *d_out = output.data_ptr<int64_t>();

  auto cast_iter = thrust::make_transform_iterator(d_data, CastI32ToI64{});

  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(nullptr, temp_storage_bytes, cast_iter, d_out,
                                  static_cast<int>(num_segments), d_offsets,
                                  d_offsets + 1, stream);

  at::Tensor temp_storage =
      torch::empty({static_cast<int64_t>(temp_storage_bytes)},
                   at::TensorOptions().dtype(at::kByte).device(device));

  cub::DeviceSegmentedReduce::Sum(
      temp_storage.data_ptr(), temp_storage_bytes, cast_iter, d_out,
      static_cast<int>(num_segments), d_offsets, d_offsets + 1, stream);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
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

// Compaction of capacity-N arrays whose live length is a device count: the
// entries of `inputs` where `flags` is set, over the entries i < n only,
// n = clamp(*num_valid, 0, N) (None: N). Nothing is read back (the temporary
// storage is an at::empty, which does not wait). Returns (count[1] int64 on
// the device, positions[N], outs[N] aligned with `inputs`, None staying None);
// entries from count on are undefined.
std::tuple<at::Tensor, at::Tensor, std::vector<c10::optional<at::Tensor>>>
flagged_compact(at::Tensor flags, std::vector<c10::optional<at::Tensor>> inputs,
                c10::optional<at::Tensor> num_valid) {

  TORCH_CHECK(flags.is_cuda(), "flags must be on CUDA");
  TORCH_CHECK(flags.dtype() == at::kBool, "flags must be bool");
  TORCH_CHECK(flags.dim() == 1 && flags.is_contiguous(),
              "flags must be a contiguous 1D tensor");
  auto device = flags.device();
  c10::cuda::CUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t num_total = flags.size(0);
  const int64_t *d_num_valid = counted_num_valid_ptr(num_valid, device);

  std::vector<at::Tensor> real_inputs;
  std::vector<size_t> real_positions;
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i].has_value()) {
      auto &t = inputs[i].value();
      TORCH_CHECK(t.is_cuda() && t.device() == device,
                  "all inputs must be on the device of flags");
      TORCH_CHECK(t.dim() == 1, "all inputs must be 1D");
      TORCH_CHECK(t.element_size() == 8,
                  "all inputs must be 8-byte elements (int64/uint64)");
      TORCH_CHECK(t.size(0) == num_total, "all inputs must match flags length");
      TORCH_CHECK(t.is_contiguous(), "all inputs must be contiguous");
      real_inputs.push_back(t);
      real_positions.push_back(i);
    }
  }
  size_t num_real = real_inputs.size();
  TORCH_CHECK(num_real <= 6,
              "flagged_compact supports at most 6 non-None input tensors");

  const auto i64 = at::TensorOptions().dtype(at::kLong).device(device);
  std::vector<c10::optional<at::Tensor>> full_outputs(inputs.size(),
                                                      c10::nullopt);

  if (num_total == 0) {
    for (size_t j = 0; j < num_real; j++) {
      full_outputs[real_positions[j]] =
          at::empty({0}, real_inputs[j].options());
    }
    return {at::zeros({1}, i64), at::empty({0}, i64), full_outputs};
  }

  auto num_selected = at::empty({1}, i64);
  auto out_indices = at::empty({num_total}, i64);
  std::vector<at::Tensor> real_outputs;
  real_outputs.reserve(num_real);
  for (const auto &inp : real_inputs) {
    real_outputs.push_back(at::empty_like(inp));
  }

  std::vector<int64_t const *> in_ptrs(num_real);
  std::vector<int64_t *> out_ptrs(num_real);
  for (size_t i = 0; i < num_real; i++) {
    in_ptrs[i] = reinterpret_cast<int64_t const *>(real_inputs[i].data_ptr());
    out_ptrs[i] = reinterpret_cast<int64_t *>(real_outputs[i].data_ptr());
  }

  // CUB runs over the capacity; the flag iterator reads the count and turns
  // every entry at or past it off.
  auto d_flags = thrust::make_transform_iterator(
      thrust::counting_iterator<int64_t>(0),
      CountedFlag{flags.data_ptr<bool>(), d_num_valid, num_total});
  auto *d_out_idx = out_indices.data_ptr<int64_t>();
  auto *d_num_sel = num_selected.data_ptr<int64_t>();

  switch (num_real) {
  case 0:
    select_index_async<int64_t, int64_t>(num_total, d_flags, d_out_idx,
                                         d_num_sel, device, stream);
    break;
  case 1:
    flagged_compact_cub<1>(num_total, d_flags, in_ptrs.data(), out_ptrs.data(),
                           d_out_idx, d_num_sel, device, stream);
    break;
  case 2:
    flagged_compact_cub<2>(num_total, d_flags, in_ptrs.data(), out_ptrs.data(),
                           d_out_idx, d_num_sel, device, stream);
    break;
  case 3:
    flagged_compact_cub<3>(num_total, d_flags, in_ptrs.data(), out_ptrs.data(),
                           d_out_idx, d_num_sel, device, stream);
    break;
  case 4:
    flagged_compact_cub<4>(num_total, d_flags, in_ptrs.data(), out_ptrs.data(),
                           d_out_idx, d_num_sel, device, stream);
    break;
  case 5:
    flagged_compact_cub<5>(num_total, d_flags, in_ptrs.data(), out_ptrs.data(),
                           d_out_idx, d_num_sel, device, stream);
    break;
  case 6:
    flagged_compact_cub<6>(num_total, d_flags, in_ptrs.data(), out_ptrs.data(),
                           d_out_idx, d_num_sel, device, stream);
    break;
  default:
    TORCH_CHECK(false,
                "flagged_compact supports at most 6 non-None input tensors");
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

  for (size_t j = 0; j < num_real; j++) {
    full_outputs[real_positions[j]] = real_outputs[j];
  }

  return {num_selected, out_indices, full_outputs};
}

} // namespace dyn_emb

#ifdef DEMB_USE_PYBIND11
void bind_index_calculation_op(py::module &m) {
  m.def("get_table_range", &dyn_emb::get_table_range,
        "Make offsets from <feature, batch> scope into <table> scope",
        py::arg("offsets"), py::arg("feature_offsets"));

  m.def("flagged_compact", &dyn_emb::flagged_compact,
        "CUB stream compaction with zipped iterators, with a device count and "
        "no host read. Only entries i < n (n = clamp(*num_valid, 0, "
        "len(flags)); None: len(flags)) are candidates. Accepts "
        "Optional[Tensor] in inputs; None passes through. Returns (count[1] "
        "int64 on device, positions[len(flags)], [opt_tensors[len(flags)]...]); "
        "entries from count on are undefined.",
        py::arg("flags"), py::arg("inputs"), py::arg("num_valid"));

  m.def("segmented_sum_cuda", &dyn_emb::segmented_sum_cuda,
        "Segmented sum: output[i] = sum(data[offsets[i]:offsets[i+1]]). data "
        "int32, offsets int64, output int64. Async.",
        py::arg("data"), py::arg("offsets"));
}
  #endif
