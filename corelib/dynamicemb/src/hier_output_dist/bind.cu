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
// The Python bindings of the op: dynamicemb_extensions.HierOutputDist (the
// op), HierOutputDistConfig (what it is built from), HierOutputDistPlan (a
// call), hier_output_dist_unique_id and hier_output_dist_traced; the
// test-only ones are in testing.cu.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <map>
#include <string>

#include "hier_output_dist/common/trace_flag.h"
#include "hier_output_dist/hier_output_dist.h"

namespace py = pybind11;
using dyn_emb::hier_output_dist::Capacities;
using dyn_emb::hier_output_dist::Config;
using dyn_emb::hier_output_dist::HierOutputDist;
using dyn_emb::hier_output_dist::kTrace;
using dyn_emb::hier_output_dist::Plan;
using dyn_emb::hier_output_dist::PlanInput;

namespace dyn_emb {
namespace hier_output_dist {
void bind_testing(py::module &py_module);  // testing.cu
}  // namespace hier_output_dist
}  // namespace dyn_emb

namespace {

// The capacities by name: lookup_keys, remote_keys, proxy_bytes.
Capacities capacities_of(const std::map<std::string, int64_t> &named) {
  Capacities capacities;
  capacities.lookup_keys = named.at("lookup_keys");
  capacities.remote_keys = named.at("remote_keys");
  capacities.proxy_bytes = named.at("proxy_bytes");
  return capacities;
}

}  // namespace

void bind_hier_output_dist(py::module &py_module) {
  py_module.def(
      "hier_output_dist_unique_id", []() { return py::bytes(HierOutputDist::unique_id()); },
      "A new NCCL unique id for a hierarchical dist's communicator");
  py_module.def(
      "hier_output_dist_traced", []() { return kTrace; },
      "Whether the build keeps the hierarchical dist's trace (DYNAMICEMB_HIER_TRACE=1)");

  py::class_<Config>(py_module, "HierOutputDistConfig", "What a HierOutputDist is built from")
      .def(py::init([](int rank, int world_size, int local_size, std::vector<int> feature_tables,
                       at::Tensor block_sizes, at::Tensor dist_types, int64_t dim,
                       at::ScalarType dtype, bool weights, bool out_weights, bool node_dedup,
                       std::map<std::string, int64_t> capacities, double timeout_s, int ctas,
                       int reduce_ctas) {
             Config config;
             config.rank = rank;
             config.world_size = world_size;
             config.local_size = local_size;
             config.feature_tables = std::move(feature_tables);
             config.block_sizes = std::move(block_sizes);
             config.dist_types = std::move(dist_types);
             config.dim = dim;
             config.dtype = dtype;
             config.weights = weights;
             config.out_weights = out_weights;
             config.node_dedup = node_dedup;
             config.capacities = capacities_of(capacities);
             config.timeout_s = timeout_s;
             config.ctas = ctas;
             config.reduce_ctas = reduce_ctas;
             return config;
           }),
           py::arg("rank"), py::arg("world_size"), py::arg("local_size"), py::arg("feature_tables"),
           py::arg("block_sizes"), py::arg("dist_types"), py::arg("dim"), py::arg("dtype"),
           py::arg("weights"), py::arg("out_weights"), py::arg("node_dedup"), py::arg("capacities"),
           py::arg("timeout_s"), py::arg("ctas"), py::arg("reduce_ctas"));

  py::class_<Plan, std::shared_ptr<Plan>>(py_module, "HierOutputDistPlan",
                                          "One call of a HierOutputDist: its state between the "
                                          "op's steps")
      .def_property_readonly("values", &Plan::values)
      .def_property_readonly("weights", &Plan::weights)
      .def_property_readonly("lengths", &Plan::lengths)
      .def_property_readonly("offsets", &Plan::offsets)
      .def_property_readonly("num_sources", &Plan::num_sources)
      .def_property_readonly("length_per_key", &Plan::length_per_key)
      .def_property_readonly("num_recv", &Plan::num_recv)
      .def_property_readonly("sent", &Plan::sent)
      .def_property_readonly("input_length_per_key", &Plan::input_length_per_key)
      .def_property_readonly("num_out", &Plan::num_out)
      .def("needs", &Plan::needs)
      .def("overflows", &Plan::overflows)
      .def("record_stream", &Plan::record_stream, py::arg("stream"));

  py::class_<HierOutputDist, std::shared_ptr<HierOutputDist>>(
      py_module, "HierOutputDist",
      "The hierarchical input and output dist of a row-wise sequence sharding, as one static "
      "operator")
      .def(py::init([](py::bytes unique_id_bytes, const Config &config) {
             std::string unique_id = unique_id_bytes;
             Config copy = config;
             py::gil_scoped_release no_gil;  // the constructor's collectives
             return std::make_shared<HierOutputDist>(unique_id, std::move(copy));
           }),
           py::arg("unique_id"), py::arg("config"))
      .def_static("unsupported", &HierOutputDist::unsupported, py::arg("world_size"),
                  py::arg("local_size"), py::arg("num_features"), py::arg("row_bytes"),
                  "Why the op does not apply to such a sharding, or an empty string")
      .def_static("invalid", &HierOutputDist::invalid, py::arg("config"),
                  "Why the op refuses this configuration, or an empty string; builds nothing")
      .def(
          "plan",
          [](HierOutputDist &self, at::Tensor values, c10::optional<at::Tensor> offsets,
             c10::optional<at::Tensor> lengths, int64_t batch_size,
             c10::optional<at::Tensor> weights, c10::optional<at::Tensor> input_offsets,
             int64_t input_batch_size, c10::optional<at::Tensor> reverse_indices) {
            PlanInput input;
            input.values = std::move(values);
            input.offsets = std::move(offsets);
            input.lengths = std::move(lengths);
            input.batch_size = batch_size;
            input.weights = std::move(weights);
            input.input_offsets = std::move(input_offsets);
            input.input_batch_size = input_batch_size;
            input.reverse_indices = std::move(reverse_indices);
            return self.plan(std::move(input));
          },
          py::arg("values"), py::arg("offsets"), py::arg("lengths"), py::arg("batch_size"),
          py::arg("weights"), py::arg("input_offsets"), py::arg("input_batch_size"),
          py::arg("reverse_indices"), py::call_guard<py::gil_scoped_release>(),
          "Part 1 of a call on the current stream (local, no sync); returns its plan")
      .def("start", &HierOutputDist::start, py::arg("plan"),
           py::call_guard<py::gil_scoped_release>(),
           "The first wait: the rest of the call on the op's side stream (no sync)")
      .def("wait", &HierOutputDist::wait, py::arg("plan"), py::call_guard<py::gil_scoped_release>(),
           "The second wait, the call's one host sync: False when the call does not fit the "
           "capacities (on every rank alike)")
      .def("lookup_rows", &HierOutputDist::lookup_rows, py::arg("plan"),
           "The rows the call's lookup writes, [num_recv, dim]")
      .def("forward", &HierOutputDist::forward, py::arg("plan"), py::arg("lookup_out"),
           "The output dist's forward kernel: the rows, and the maps the backward takes")
      .def("backward", &HierOutputDist::backward, py::arg("plan"), py::arg("grad"), py::arg("maps"),
           py::arg("divide"),
           "The output dist's backward kernel: the gradient of the lookup output")
      .def("quiesce", &HierOutputDist::quiesce, py::call_guard<py::gil_scoped_release>(),
           "Collective, after the last call: the teardown then frees the windows at once")
      .def("capacities", &HierOutputDist::capacities)
      .def("memory", &HierOutputDist::memory)
      .def("gin_railed", &HierOutputDist::gin_railed)
      .def("trace", &HierOutputDist::trace)
      .def("trace_launches", &HierOutputDist::trace_launches)
      .def("in_trace", &HierOutputDist::in_trace)
      .def("in_calls", &HierOutputDist::in_calls);

  dyn_emb::hier_output_dist::bind_testing(py_module);
}
