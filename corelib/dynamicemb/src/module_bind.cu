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

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_dyn_emb_op(py::module &m);
void bind_unique_op(py::module &m);
void bind_bucktiz_kernel_op(py::module &m);
void bind_optimizer_kernel_op(py::module &m);
void bind_utils(py::module &m);
void bind_index_calculation_op(py::module &m);
void bind_initializer_op(py::module &m);

PYBIND11_MODULE(dynamicemb_extensions, m) {
  m.doc() = "DYNAMICEMB"; // Optional

  bind_dyn_emb_op(m);
  bind_unique_op(m);
  bind_bucktiz_kernel_op(m);
  bind_optimizer_kernel_op(m);
  bind_index_calculation_op(m);
  bind_initializer_op(m);
  bind_utils(m);
}
