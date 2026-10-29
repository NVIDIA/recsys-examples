// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

namespace py = pybind11;

namespace dynamicemb::exportable_embedding {

void bind_indexer_snapshot(py::module_& module);
void bind_indexer_directory(py::module_& module);
void bind_update_subscriber(py::module_& module);

}  // namespace dynamicemb::exportable_embedding

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  dynamicemb::exportable_embedding::bind_indexer_snapshot(module);
  dynamicemb::exportable_embedding::bind_indexer_directory(module);
  dynamicemb::exportable_embedding::bind_update_subscriber(module);
}
