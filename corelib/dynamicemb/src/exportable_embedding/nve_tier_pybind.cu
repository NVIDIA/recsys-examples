// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <pybind11/stl.h>
#include "dynamicemb/exportable_embedding/nve_runtime.h"
#include "dynamicemb/exportable_embedding/nve_tier_binding.h"

namespace py = pybind11;
namespace dynamicemb::exportable_embedding {

void bind_nve_tier_runtime(py::module_& module) {
  // Defer registration of derived pybind types until pynve.nve was imported.
  // Importing DynamicEmb indexers alone must not load NVE's Python runtime.
  module.def("register_nve_tier_bindings", [module]() mutable {
    if (py::hasattr(module, "TierGPUEmbedding")) return;
    py::module_::import("pynve.nve");
    const auto update = [](const std::shared_ptr<NveBinding>& binding, size_t count,
                           uintptr_t keys, uintptr_t values, int64_t tier, uint64_t stream) {
      update_nve_tier(binding, count, keys, values, tier, stream);
    };
    py::class_<TierGPUEmbedding, nve::GPUEmbedding<int64_t>,
               std::shared_ptr<TierGPUEmbedding>>(module, "TierGPUEmbedding")
        .def(py::init<size_t, size_t, nve::DataType_t, std::shared_ptr<nve::MemBlock>,
                      int, nve::EmbedLayerConfig>())
        .def("update_tier", update, py::call_guard<py::gil_scoped_release>());
    py::class_<TierUVMEmbedding, nve::LinearUVMEmbedding<int64_t>,
               std::shared_ptr<TierUVMEmbedding>>(module, "TierUVMEmbedding")
        .def(py::init<size_t, size_t, nve::DataType_t, std::shared_ptr<nve::MemBlock>,
                      size_t, bool, int, nve::EmbedLayerConfig>())
        .def("update_tier", update, py::call_guard<py::gil_scoped_release>());
    py::class_<TierHierarchicalEmbedding, nve::HierarchicalEmbedding<int64_t>,
               std::shared_ptr<TierHierarchicalEmbedding>>(module, "TierHierarchicalEmbedding")
        .def(py::init<size_t, nve::DataType_t, uint64_t, uint64_t, nve::table_ptr_t,
                      uint64_t, bool, int, nve::EmbedLayerConfig>())
        .def("update_tier", update, py::call_guard<py::gil_scoped_release>());
  });
  module.def("has_nve_tier_update", &has_nve_tier_update);
  py::class_<NveRuntimeLayer>(module, "NveRuntimeLayer")
      .def_readonly("_export_module_path", &NveRuntimeLayer::module_path)
      .def_readonly("emb_layer", &NveRuntimeLayer::binding)
      .def_readonly("marker_tensor", &NveRuntimeLayer::marker);
  py::class_<NveRuntimeDirectory>(module, "NveRuntimeDirectory")
      .def(py::init<const std::string&, int>())
      .def("layers", &NveRuntimeDirectory::layers)
      .def("markers", &NveRuntimeDirectory::markers);
}
}  // namespace dynamicemb::exportable_embedding
