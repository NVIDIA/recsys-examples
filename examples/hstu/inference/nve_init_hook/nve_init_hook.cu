// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// NVE model-init hook for the upstream Triton PyTorch (AOTI) backend.
//
// Built as a standalone shared library (libnve_init_hook.so). The Triton
// PyTorch backend dlopen()s it at model load when the model's config.pbtxt sets
//
//   parameters { key: "MODEL_INIT_LIBRARY"
//                value: { string_value: "/abs/path/to/libnve_init_hook.so" } }
//
// and calls triton_pytorch_model_init() once, before the model package is
// loaded. HSTU AOTI models call nve_ops::embedding_lookup(layer_id), which
// resolves the embedding table from a process-global NVELayerRegistry at
// execute time. The NVE weights live OUTSIDE model.pt2 (in
// <model_dir>/metadata.json + <model_dir>/weights/*.nve), so loading the
// package does not load them. This hook loads + registers them via
// nve::LayerDirectory -- exactly what the standalone C++ inference demo does.
//
// The Triton backend links nothing NVE-specific; only THIS library does. It is
// compiled with nvcc because nve_loader.hpp pulls in CUDA-heavy headers
// (cub, cuda_bf16) that a g++ translation unit cannot build.

#include <sys/stat.h>

#include <exception>
#include <iostream>
#include <string>

#include "python/pynve/torch_bindings/nve_loader.hpp"

namespace {
bool
FileExists(const std::string& path)
{
  struct stat st;
  return ::stat(path.c_str(), &st) == 0;
}
}  // namespace

extern "C" {

// Called once by the Triton PyTorch backend at model load, before the model
// package is loaded. `model_dir` is the versioned model directory (the one that
// contains model.pt2, metadata.json and weights/). `device_index` is the GPU
// ordinal, or -1 for CPU. Returns an opaque handle (the loaded LayerDirectory)
// that the backend hands back to triton_pytorch_model_release() on unload, or
// nullptr if there is nothing to load (no metadata.json) or loading failed.
void*
triton_pytorch_model_init(const char* model_dir, int device_index)
{
  const std::string package_dir(model_dir != nullptr ? model_dir : "");
  // Gate: only treat this as an NVE model if metadata.json is present next to
  // the package, so a misdirected hook is a harmless no-op.
  if (!FileExists(package_dir + "/metadata.json")) {
    std::cerr << "[nve-init-hook] no metadata.json in \"" << package_dir
              << "\"; nothing to load" << std::endl;
    return nullptr;
  }
  // NVE needs a valid device ordinal; map CPU (-1) to device 0.
  const int dev = device_index < 0 ? 0 : device_index;
  try {
    std::cerr << "[nve-init-hook] loading NVE layers from \"" << package_dir
              << "\" (device " << dev << ")" << std::endl;
    auto* layers = new nve::LayerDirectory(package_dir, dev);
    std::cerr << "[nve-init-hook] loaded " << layers->size()
              << " NVE layer(s) into the registry" << std::endl;
    return static_cast<void*>(layers);
  }
  catch (const std::exception& e) {
    // Must not let an exception cross the C/dlopen boundary. Log and return
    // nullptr; the model then fails at execute time with the original
    // "no binding for layer_id" error, which points back here.
    std::cerr << "[nve-init-hook] FAILED to load NVE layers from \""
              << package_dir << "\": " << e.what() << std::endl;
    return nullptr;
  }
}

// Called by the backend on model unload. Releases the layers, unregistering
// them from the global registry. Safe on nullptr.
void
triton_pytorch_model_fini(void* state)
{
  delete static_cast<nve::LayerDirectory*>(state);
}

}  // extern "C"
