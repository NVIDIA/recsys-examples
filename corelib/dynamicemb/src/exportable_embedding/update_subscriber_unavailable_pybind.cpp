// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace dynamicemb::exportable_embedding {
namespace {

class UnavailableUpdateSubscriber {
 public:
  UnavailableUpdateSubscriber(const std::string&, py::object, py::dict, int) {
    throw std::runtime_error(
        "Incremental embedding updates require NVE 26.06 or later; "
        "this build targets NVE 26.05");
  }
};

}  // namespace

void bind_update_subscriber(py::module_& module) {
  py::class_<UnavailableUpdateSubscriber>(
      module, "EmbeddingCollectionUpdateSubscriber")
      .def(py::init<const std::string&, py::object, py::dict, int>(),
           py::arg("package_dir"), py::arg("indexers"),
           py::arg("nve_layers"), py::arg("device_index"));
}

}  // namespace dynamicemb::exportable_embedding
