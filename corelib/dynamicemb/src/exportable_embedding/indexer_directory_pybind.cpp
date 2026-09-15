// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <pybind11/stl.h>

#include <stdexcept>
#include <unordered_set>

#include "dynamicemb/exportable_embedding/indexer_directory.h"

namespace py = pybind11;

namespace dynamicemb::exportable_embedding {

void bind_indexer_directory(py::module_& module) {
  py::class_<EmbeddingCollectionBinding>(module,
                                         "EmbeddingCollectionBinding")
      .def(py::init<>())
      .def_readwrite("table_names", &EmbeddingCollectionBinding::table_names)
      .def_readwrite("indexer_type", &EmbeddingCollectionBinding::indexer_type)
      .def_readwrite("indexer_module_path",
                     &EmbeddingCollectionBinding::indexer_module_path)
      .def_readwrite("nve_layer_module_path",
                     &EmbeddingCollectionBinding::nve_layer_module_path)
      .def_readwrite("snapshot_path",
                     &EmbeddingCollectionBinding::snapshot_path)
      .def_readwrite("feature_id_bits",
                     &EmbeddingCollectionBinding::feature_id_bits)
      .def_readwrite("marker_value", &EmbeddingCollectionBinding::marker_value);

  py::class_<EmbeddingCollectionIndexerDirectory,
             std::unique_ptr<EmbeddingCollectionIndexerDirectory>>(
      module, "EmbeddingCollectionIndexerDirectory")
      .def_static("create", &EmbeddingCollectionIndexerDirectory::create,
                  py::arg("bindings"), py::arg("markers"),
                  py::arg("snapshots"), py::arg("device_index"))
      .def_static("load", &EmbeddingCollectionIndexerDirectory::load,
                  py::arg("package_dir"), py::arg("device_index"))
      .def("bind_aoti",
           [](const EmbeddingCollectionIndexerDirectory& directory,
              const py::object& loader) {
             const auto names = loader.attr("get_constant_fqns")()
                                    .cast<std::vector<std::string>>();
             const std::unordered_set<std::string> available(names.begin(),
                                                              names.end());
             for (const auto& item : directory.marker_constants()) {
               if (available.find(item.first) == available.end()) {
                 throw std::runtime_error(
                     "Indexer marker is not an AOTI constant: " + item.first);
               }
               py::dict constants;
               constants[py::str(item.first)] = item.second;
               loader.attr("load_constants")(constants, false, false, true);
               loader.attr("load_constants")(constants, true, false, true);
             }
           })
      .def("wait_for_retirement",
           &EmbeddingCollectionIndexerDirectory::wait_for_retirement)
      .def("snapshot", &EmbeddingCollectionIndexerDirectory::snapshot)
      .def_property_readonly(
          "bindings",
          [](const EmbeddingCollectionIndexerDirectory& directory) {
            return directory.bindings();
          })
      .def_property_readonly("device_index",
                             &EmbeddingCollectionIndexerDirectory::device_index);
}

}  // namespace dynamicemb::exportable_embedding
