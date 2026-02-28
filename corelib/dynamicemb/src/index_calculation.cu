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
#include <torch/extension.h>

namespace dyn_emb {

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

std::tuple<int64_t, at::Tensor, std::vector<c10::optional<at::Tensor>>>
flagged_compact(
    at::Tensor flags,
    std::vector<c10::optional<at::Tensor>> inputs) {

    auto device = flags.device();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t num_total = flags.size(0);
    TORCH_CHECK(flags.dtype() == at::kBool, "flags must be bool");

    std::vector<at::Tensor> real_inputs;
    std::vector<size_t> real_positions;
    for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i].has_value()) {
            auto &t = inputs[i].value();
            TORCH_CHECK(t.dim() == 1, "all inputs must be 1D");
            TORCH_CHECK(t.element_size() == 8,
                         "all inputs must be 8-byte elements (int64/uint64)");
            TORCH_CHECK(t.size(0) == num_total, "all inputs must match flags length");
            real_inputs.push_back(t);
            real_positions.push_back(i);
        }
    }
    size_t num_real = real_inputs.size();

    if (num_total == 0) {
        auto empty_idx = at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device));
        std::vector<c10::optional<at::Tensor>> full_out(inputs.size(), c10::nullopt);
        for (size_t j = 0; j < num_real; j++) {
            auto s = real_inputs[j].sizes().vec(); s[0] = 0;
            full_out[real_positions[j]] = at::empty(s, real_inputs[j].options());
        }
        return {0, empty_idx, full_out};
    }

    auto num_selected = at::empty({1}, at::TensorOptions().dtype(at::kLong).device(device));
    auto out_indices = at::empty({num_total}, at::TensorOptions().dtype(at::kLong).device(device));
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

    auto *d_flags = flags.data_ptr<bool>();
    auto *d_out_idx = out_indices.data_ptr<int64_t>();
    auto *d_num_sel = num_selected.data_ptr<int64_t>();

    switch (num_real) {
        case 0:
            select_index_async<int64_t, int64_t>(
                num_total, d_flags, d_out_idx, d_num_sel, device, stream);
            break;
        case 1: flagged_compact_cub<1>(num_total, d_flags,
                    in_ptrs.data(), out_ptrs.data(), d_out_idx, d_num_sel,
                    device, stream); break;
        case 2: flagged_compact_cub<2>(num_total, d_flags,
                    in_ptrs.data(), out_ptrs.data(), d_out_idx, d_num_sel,
                    device, stream); break;
        case 3: flagged_compact_cub<3>(num_total, d_flags,
                    in_ptrs.data(), out_ptrs.data(), d_out_idx, d_num_sel,
                    device, stream); break;
        case 4: flagged_compact_cub<4>(num_total, d_flags,
                    in_ptrs.data(), out_ptrs.data(), d_out_idx, d_num_sel,
                    device, stream); break;
        case 5: flagged_compact_cub<5>(num_total, d_flags,
                    in_ptrs.data(), out_ptrs.data(), d_out_idx, d_num_sel,
                    device, stream); break;
        case 6: flagged_compact_cub<6>(num_total, d_flags,
                    in_ptrs.data(), out_ptrs.data(), d_out_idx, d_num_sel,
                    device, stream); break;
        default:
            TORCH_CHECK(false, "flagged_compact supports at most 6 non-None input tensors");
    }

    int64_t h_count = num_selected.cpu().item<int64_t>();
    auto sliced_indices = out_indices.slice(0, 0, h_count);

    std::vector<c10::optional<at::Tensor>> full_outputs(inputs.size(), c10::nullopt);
    for (size_t j = 0; j < num_real; j++) {
        full_outputs[real_positions[j]] = real_outputs[j].slice(0, 0, h_count);
    }

    return {h_count, sliced_indices, full_outputs};
}

} // namespace dyn_emb

void bind_index_calculation_op(py::module &m) {
  m.def("get_table_range", &dyn_emb::get_table_range,
        "Make offsets from <feature, batch> scope into <table> scope",
        py::arg("offsets"), py::arg("feature_offsets"));

  m.def("flagged_compact", &dyn_emb::flagged_compact,
        "CUB stream compaction with zipped iterators. "
        "Accepts Optional[Tensor] in inputs; None passes through. "
        "Returns (count, indices[count], [opt_tensors[count]...]).",
        py::arg("flags"), py::arg("inputs"));
}
