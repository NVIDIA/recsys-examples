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

#include "unique_op.h"

namespace dyn_emb {

std::shared_ptr<dyn_emb::UniqueOpBase>
UniqueOpFactory::create(at::Tensor keys, at::Tensor vals, at::Tensor counter,
                        const size_t capacity) {
  auto key_type = keys.dtype();
  auto val_type = vals.dtype();
  auto counter_type = counter.dtype();
  assert(val_type == counter_type);

  if (key_type == at::kLong) {
    if (val_type == at::kLong) {
      return std::make_shared<dyn_emb::HashUniqueOp<int64_t, int64_t>>(
          keys, vals, counter, capacity);
    } else if (val_type == at::kUInt64) {
      return std::make_shared<dyn_emb::HashUniqueOp<int64_t, uint64_t>>(
          keys, vals, counter, capacity);
    }
  } else if (key_type == at::kUInt64) {
    if (val_type == at::kLong) {
      return std::make_shared<dyn_emb::HashUniqueOp<uint64_t, int64_t>>(
          keys, vals, counter, capacity);
    } else if (val_type == at::kUInt64) {
      return std::make_shared<dyn_emb::HashUniqueOp<uint64_t, uint64_t>>(
          keys, vals, counter, capacity);
    }
  }

  throw std::invalid_argument("Invalid key type or value type of unique op.");
}
} // namespace dyn_emb

// PYTHON WARP
void bind_unique_op(py::module &m) {

  py::class_<dyn_emb::UniqueOpBase, std::shared_ptr<dyn_emb::UniqueOpBase>>(
      m, "UniqueOp")
      .def(py::init([](at::Tensor keys, at::Tensor vals, at::Tensor counter,
                       const size_t capacity) {
        return dyn_emb::UniqueOpFactory::create(keys, vals, counter, capacity);
      }))
      .def(
          "unique",
          [](dyn_emb::UniqueOpBase &self, const at::Tensor &d_key, uint64_t len,
             const at::Tensor &d_output_index, const at::Tensor &d_unique_key,
             const at::Tensor &d_output_counter, uint64_t stream = 0,
             const c10::optional<at::Tensor> &offset = c10::nullopt,
             const c10::optional<at::Tensor> &d_frequency_counters =
                 c10::nullopt,
             const c10::optional<at::Tensor> &d_input_frequencies =
                 c10::nullopt) {
            cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

            at::Tensor offset_tensor =
                offset.has_value() ? offset.value() : at::Tensor();
            at::Tensor frequency_counters_tensor =
                d_frequency_counters.has_value() ? d_frequency_counters.value()
                                                 : at::Tensor();
            at::Tensor input_frequencies_tensor =
                d_input_frequencies.has_value() ? d_input_frequencies.value()
                                                : at::Tensor();

            self.unique(d_key, len, d_output_index, d_unique_key,
                        d_output_counter, cuda_stream, offset_tensor,
                        frequency_counters_tensor, input_frequencies_tensor);
          },
          "Unique operation.", py::arg("d_key"), py::arg("len"),
          py::arg("d_output_index"), py::arg("d_unique_key"),
          py::arg("d_output_counter"), py::arg("stream") = 0,
          py::arg("offset") = c10::nullopt,
          py::arg("d_frequency_counters") = c10::nullopt,
          py::arg("d_input_frequencies") = c10::nullopt)

      .def(
          "reset_capacity",
          [](dyn_emb::UniqueOpBase &self, const at::Tensor &keys,
             const at::Tensor &vals, size_t capacity, uint64_t stream = 0) {
            cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
            self.reset_capacity(keys, vals, capacity, cuda_stream);
          },
          "Reset capacity.", py::arg("keys"), py::arg("vals"),
          py::arg("capacity"), py::arg("stream") = 0)
      .def("get_capacity", &dyn_emb::UniqueOpBase::get_capacity,
           "Get capacity");
}
