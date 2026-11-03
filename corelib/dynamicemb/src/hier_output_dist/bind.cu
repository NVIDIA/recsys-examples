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
// The Python bindings of the op, which take and give tensors and plain
// values only: dynamicemb_extensions.HierOutputDist (the op), and
// hier_output_dist_unique_id, hier_output_dist_unsupported,
// hier_output_dist_invalid and hier_output_dist_traced; the test-only ones
// are in testing.cu.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <string>
#include <vector>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/config.h"
#include "hier_output_dist/hier_output_dist.h"

namespace py = pybind11;
using dyn_emb::hier_output_dist::Communicator;
using dyn_emb::hier_output_dist::HierOutputDist;
using dyn_emb::hier_output_dist::kTrace;

namespace dyn_emb {
namespace hier_output_dist {
void bind_testing(py::module &py_module);  // testing.cu
}  // namespace hier_output_dist
}  // namespace dyn_emb

void bind_hier_output_dist(py::module &py_module) {
  py_module.def(
      "hier_output_dist_unique_id", []() { return py::bytes(Communicator::unique_id()); },
      "A new NCCL unique id for a hierarchical dist's communicator");
  py_module.def(
      "hier_output_dist_traced", []() { return kTrace; },
      "Whether the build keeps the hierarchical dist's trace (DYNAMICEMB_HIER_TRACE=1)");
  py_module.def("hier_output_dist_unsupported", &dyn_emb::hier_output_dist::unsupported,
                py::arg("world_size"), py::arg("local_size"), py::arg("num_features"),
                py::arg("row_bytes"),
                "Why the op does not apply to such a sharding, or an empty string");
  py_module.def(
      "hier_output_dist_invalid",
      [](int64_t rank, int64_t world_size, int64_t local_size,
         const std::vector<int64_t> &feature_tables, const at::Tensor &block_sizes,
         const at::Tensor &dist_types, int64_t dim, at::ScalarType dtype, bool weights,
         bool out_weights, bool node_dedup, int64_t lookup_keys, int64_t remote_keys,
         int64_t proxy_bytes, double timeout_s, int64_t ctas, int64_t reduce_ctas) {
        return dyn_emb::hier_output_dist::invalid(dyn_emb::hier_output_dist::config_of(
            rank, world_size, local_size, feature_tables, block_sizes, dist_types, dim, dtype,
            weights, out_weights, node_dedup, lookup_keys, remote_keys, proxy_bytes, timeout_s,
            ctas, reduce_ctas));
      },
      py::arg("rank"), py::arg("world_size"), py::arg("local_size"), py::arg("feature_tables"),
      py::arg("block_sizes"), py::arg("dist_types"), py::arg("dim"), py::arg("dtype"),
      py::arg("weights"), py::arg("out_weights"), py::arg("node_dedup"), py::arg("lookup_keys"),
      py::arg("remote_keys"), py::arg("proxy_bytes"), py::arg("timeout_s"), py::arg("ctas"),
      py::arg("reduce_ctas"),
      "Why the op refuses the configuration the constructor would take, or an empty string; "
      "builds nothing");

  py::class_<HierOutputDist, std::shared_ptr<HierOutputDist>>(
      py_module, "HierOutputDist",
      "The hierarchical input and output dist of a row-wise sequence sharding, as one static "
      "operator")
      .def(py::init<const std::string &, int64_t, int64_t, int64_t, const std::vector<int64_t> &,
                    const at::Tensor &, const at::Tensor &, int64_t, at::ScalarType, bool, bool,
                    bool, int64_t, int64_t, int64_t, double, int64_t, int64_t>(),
           py::arg("unique_id"), py::arg("rank"), py::arg("world_size"), py::arg("local_size"),
           py::arg("feature_tables"), py::arg("block_sizes"), py::arg("dist_types"), py::arg("dim"),
           py::arg("dtype"), py::arg("weights"), py::arg("out_weights"), py::arg("node_dedup"),
           py::arg("lookup_keys"), py::arg("remote_keys"), py::arg("proxy_bytes"),
           py::arg("timeout_s"), py::arg("ctas"), py::arg("reduce_ctas"),
           py::call_guard<py::gil_scoped_release>())  // the constructor's collectives
      .def("input_dist", &HierOutputDist::input_dist, py::arg("values"), py::arg("offsets"),
           py::arg("lengths"), py::arg("batch_size"), py::arg("weights"), py::arg("input_offsets"),
           py::arg("input_batch_size"), py::arg("reverse_indices"),
           py::call_guard<py::gil_scoped_release>(),
           "The input dist of a call on the current stream, no host sync: (values, weights, "
           "lengths, offsets, lookup_rows, status, fwd_plan, bwd_plan)")
      .def("output_dist_fwd", &HierOutputDist::output_dist_fwd, py::arg("fwd_plan"),
           py::arg("lookup_out"),
           "The output dist's forward kernel: the rows of this rank's output")
      .def("output_dist_bwd", &HierOutputDist::output_dist_bwd, py::arg("bwd_plan"),
           py::arg("counts"), py::arg("grad"), py::arg("divide"),
           "The output dist's backward kernel: the gradient of the lookup output")
      .def("quiesce", &HierOutputDist::quiesce, py::call_guard<py::gil_scoped_release>(),
           "Collective, after the last call: the teardown then frees the window at once")
      .def("memory", &HierOutputDist::memory)
      .def("trace", &HierOutputDist::trace)
      .def("trace_launches", &HierOutputDist::trace_launches);

  dyn_emb::hier_output_dist::bind_testing(py_module);
}
