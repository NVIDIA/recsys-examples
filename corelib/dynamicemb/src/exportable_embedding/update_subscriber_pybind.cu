// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <pybind11/stl.h>

#include <utility>

#include "dynamicemb/exportable_embedding/incremental_update.h"
#include "python/pynve/bindings/binding_layers.hpp"

namespace py = pybind11;

namespace dynamicemb::exportable_embedding {

void bind_update_subscriber(py::module_& module) {
  py::class_<EmbeddingCollectionUpdateSubscriber>(
      module, "EmbeddingCollectionUpdateSubscriber")
      .def(
          py::init([](
                       const std::string& package_dir,
                       EmbeddingCollectionIndexerDirectory& indexers,
                       const py::dict& layers, int device_index) {
            std::unordered_map<
                std::string,
                std::shared_ptr<nve::NVEmbedBinding<int64_t>>>
                bindings;
            for (const auto& item : layers) {
              bindings.emplace(
                  py::cast<std::string>(item.first),
                  py::cast<std::shared_ptr<nve::NVEmbedBinding<int64_t>>>(
                      item.second));
            }
            return std::make_unique<EmbeddingCollectionUpdateSubscriber>(
                package_dir, indexers, std::move(bindings), device_index);
          }),
          py::arg("package_dir"), py::arg("indexers"), py::arg("nve_layers"),
          py::arg("device_index"), py::keep_alive<1, 3>())
      .def(
          "apply_incremental_load",
          [](EmbeddingCollectionUpdateSubscriber& subscriber,
             const std::string& update_json, uint64_t inference_stream) {
            subscriber.apply_incremental_load(
                EmbeddingCollectionUpdate::from_json(update_json),
                reinterpret_cast<cudaStream_t>(inference_stream));
          },
          py::arg("update_json"), py::arg("inference_stream"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "wait_for_retirement",
          [](EmbeddingCollectionUpdateSubscriber& subscriber,
             const std::string& collection_id, int64_t snapshot_id) {
            return subscriber.wait_for_retirement(collection_id, snapshot_id)
                .to_json();
          },
          py::arg("collection_id"), py::arg("snapshot_id"),
          py::call_guard<py::gil_scoped_release>());
}

}  // namespace dynamicemb::exportable_embedding
