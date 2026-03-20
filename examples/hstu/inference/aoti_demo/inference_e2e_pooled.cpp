// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file inference_e2e_pooled.cpp
 *
 * End-to-end C++ inference demo for sum/mean pooling mode.
 *
 * Loads an AOTInductor packaged model produced by
 * ``test_inference_emb_pooled_sum()`` and runs it with three inputs:
 *   - keys             : flat lookup key tensor (int64)
 *   - offsets          : per-table CSR boundary tensor (int64)
 *   - pooling_offsets  : per-bag CSR boundary tensor (int64)
 *
 * The model was exported from an ``InferenceEmbeddingTable`` with
 * ``pooling_mode=1`` (sum pooling).  The output shape is ``(B, D)`` where
 * ``B = pooling_offsets.numel() - 1`` is the number of pooling bags and
 * ``D`` is the embedding dimension.
 *
 * Usage:
 *   inference_embedding_aoti_e2e_pooled_demo \
 *     <package_path> \
 *     <keys_path> \
 *     <offsets_path> \
 *     <pooling_offsets_path> \
 *     <expected_pooled_embeddings_path> \
 *     [<custom_ops_path>] \
 *     [<nve_torch_path>]
 */

#include <dlfcn.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace {

bool load_shared_library(const std::string& label, const std::string& path) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    std::cerr << "Failed to load " << label << ": " << path << std::endl;
    std::cerr << "dlerror(): " << dlerror() << std::endl;
    return false;
  }
  std::cout << "Loaded " << label << ": " << path << std::endl;
  return true;
}

std::string infer_default_custom_ops_path(const char* argv0) {
  std::filesystem::path exe_path = std::filesystem::absolute(argv0);
  return (exe_path.parent_path() / "../lib/inference_emb_ops.so")
      .lexically_normal()
      .string();
}

std::string infer_default_nve_torch_path(const char* argv0) {
  std::filesystem::path exe_path = std::filesystem::absolute(argv0);
  return (exe_path.parent_path() / "../lib/libnve_torch.so")
      .lexically_normal()
      .string();
}

// Load a tensor saved from Python via _save_tensor_cpp_compatible().
// Python wraps the tensor as a buffer in a scripted nn.Module (torch.jit.save),
// so we load it back with torch::jit::load and pull out the "tensor" buffer.
torch::Tensor load_tensor_from_file(const std::string& path) {
  torch::jit::script::Module module = torch::jit::load(path);
  return module.attr("tensor").toTensor();
}

}  // namespace

int main(int argc, char** argv) {
  c10::InferenceMode mode;

  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is required for this pooled E2E inference demo." << std::endl;
    return 1;
  }

  const std::string package_path =
      (argc > 1) ? argv[1] : "model_pooled_sum.pt2";
  const std::string keys_path =
      (argc > 2) ? argv[2] : "keys.pt";
  const std::string offsets_path =
      (argc > 3) ? argv[3] : "offsets.pt";
  const std::string pooling_offsets_path =
      (argc > 4) ? argv[4] : "pooling_offsets.pt";
  const std::string expected_embeddings_path =
      (argc > 5) ? argv[5] : "pooled_embeddings.pt";
  const std::string custom_ops_path =
      (argc > 6) ? argv[6] : infer_default_custom_ops_path(argv[0]);
  const std::string nve_torch_path =
      (argc > 7) ? argv[7] : infer_default_nve_torch_path(argv[0]);

  try {
    if (!load_shared_library("inference_emb_ops library", custom_ops_path)) {
      return 5;
    }
    if (!load_shared_library("nve torch class library", nve_torch_path)) {
      return 8;
    }

    torch::inductor::AOTIModelPackageLoader loader(package_path);

    auto device = torch::Device(torch::kCUDA, 0);

    torch::Tensor keys_cpu = load_tensor_from_file(keys_path);
    torch::Tensor offsets_cpu = load_tensor_from_file(offsets_path);
    torch::Tensor pooling_offsets_cpu = load_tensor_from_file(pooling_offsets_path);
    torch::Tensor expected_embeddings_cpu =
        load_tensor_from_file(expected_embeddings_path);

    auto keys = keys_cpu.to(device, /*dtype=*/torch::kInt64).contiguous();
    auto offsets = offsets_cpu.to(device, /*dtype=*/torch::kInt64).contiguous();
    auto pooling_offsets =
        pooling_offsets_cpu.to(device, /*dtype=*/torch::kInt64).contiguous();

    // Run the packaged model with three inputs: keys, offsets, pooling_offsets.
    std::vector<torch::Tensor> outputs = loader.run({keys, offsets, pooling_offsets});

    if (outputs.empty()) {
      std::cerr << "The packaged model returned no outputs." << std::endl;
      return 2;
    }

    torch::Tensor output_cpu = outputs[0].to(torch::kCPU);
    torch::Tensor expected_cpu =
        expected_embeddings_cpu.to(output_cpu.dtype()).contiguous();

    if (!output_cpu.sizes().equals(expected_cpu.sizes())) {
      std::cerr << "Output shape mismatch. Expected " << expected_cpu.sizes()
                << ", got " << output_cpu.sizes() << std::endl;
      return 6;
    }

    torch::Tensor diff = (output_cpu - expected_cpu).abs();
    double max_abs_diff = diff.max().item<double>();
    bool allclose = torch::allclose(output_cpu, expected_cpu);

    std::cout << "Loaded package      : " << package_path << std::endl;
    std::cout << "Loaded keys         : " << keys_path << std::endl;
    std::cout << "Loaded offsets      : " << offsets_path << std::endl;
    std::cout << "Loaded pool offsets : " << pooling_offsets_path << std::endl;
    std::cout << "Loaded expected emb : " << expected_embeddings_path << std::endl;
    std::cout << "NVE torch library   : " << nve_torch_path << std::endl;
    std::cout << "Output shape (B, D) : " << output_cpu.sizes() << std::endl;
    std::cout << "First pooling-bag embedding, first 8 values: "
              << output_cpu[0].slice(/*dim=*/0, /*start=*/0, /*end=*/8)
              << std::endl;
    std::cout << "Max absolute diff vs expected: " << max_abs_diff << std::endl;

    if (!allclose) {
      std::cerr << "Output does not match expected pooled embeddings." << std::endl;
      return 7;
    }

    std::cout << "Pooled E2E comparison passed." << std::endl;
    return 0;

  } catch (const c10::Error& e) {
    std::cerr << "AOTInductor pooled E2E inference failed: " << e.what()
              << std::endl;
    return 3;
  } catch (const std::exception& e) {
    std::cerr << "Unexpected error: " << e.what() << std::endl;
    return 4;
  }
}
