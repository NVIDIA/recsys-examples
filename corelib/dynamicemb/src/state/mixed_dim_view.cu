/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

#include "mixed_dim_view.h"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

// PYTHON WRAP
void bind_mixed_dim_view(py::module &m) {
  py::class_<dyn_emb::PaddedBufferView>(m, "PaddedBufferView")
      .def(py::init<at::Tensor, int64_t, int64_t, bool>(), py::arg("emb_dims"),
           py::arg("max_emb_dim"), py::arg("max_value_dim"),
           py::arg("all_dims_vec4"))
      .def_property_readonly("emb_dims", &dyn_emb::PaddedBufferView::emb_dims)
      .def_property_readonly("num_tables",
                             &dyn_emb::PaddedBufferView::num_tables)
      .def_property_readonly("max_emb_dim",
                             &dyn_emb::PaddedBufferView::max_emb_dim)
      .def_property_readonly("max_value_dim",
                             &dyn_emb::PaddedBufferView::max_value_dim)
      .def_property_readonly("all_dims_vec4",
                             &dyn_emb::PaddedBufferView::all_dims_vec4);

  py::class_<dyn_emb::FlatTablesView>(m, "FlatTablesView")
      .def(py::init<at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, bool,
                    at::ScalarType>(),
           py::arg("ptrs"), py::arg("value_dims"), py::arg("emb_dims"),
           py::arg("max_emb_dim"), py::arg("max_value_dim"),
           py::arg("all_dims_vec4"), py::arg("dtype"))
      .def_property_readonly("ptrs", &dyn_emb::FlatTablesView::ptrs)
      .def_property_readonly("value_dims", &dyn_emb::FlatTablesView::value_dims)
      .def_property_readonly("emb_dims", &dyn_emb::FlatTablesView::emb_dims)
      .def_property_readonly("num_tables", &dyn_emb::FlatTablesView::num_tables)
      .def_property_readonly("max_emb_dim",
                             &dyn_emb::FlatTablesView::max_emb_dim)
      .def_property_readonly("max_value_dim",
                             &dyn_emb::FlatTablesView::max_value_dim)
      .def_property_readonly("all_dims_vec4",
                             &dyn_emb::FlatTablesView::all_dims_vec4)
      .def_property_readonly("dtype", &dyn_emb::FlatTablesView::dtype);
}
