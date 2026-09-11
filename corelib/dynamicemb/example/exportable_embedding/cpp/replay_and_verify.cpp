// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include "dynamicemb/exportable_embedding/indexer_directory.h"
#if !defined(DYNAMICEMB_NVE_2605)
#include "dynamicemb/exportable_embedding/incremental_update.h"
#endif
#include "python/pynve/torch_bindings/nve_loader.hpp"

namespace {

struct Arguments {
  std::string package_dir;
  std::string expected_sums;
  bool wait_for_updates{false};
  int device{0};
};

Arguments parse_arguments(int argc, char** argv) {
  Arguments args;
  for (int index = 1; index < argc; ++index) {
    const std::string name = argv[index];
    if (name == "--package-dir") {
      args.package_dir = argv[++index];
    } else if (name == "--expected-sums") {
      args.expected_sums = argv[++index];
    } else if (name == "--wait-for-updates") {
      args.wait_for_updates = true;
    } else if (name == "--device") {
      args.device = std::stoi(argv[++index]);
    } else {
      throw std::runtime_error("Unknown argument: " + name);
    }
  }
  if (args.package_dir.empty() || args.expected_sums.empty()) {
    throw std::runtime_error("--package-dir and --expected-sums are required");
  }
  return args;
}

std::vector<double> parse_sums(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    values.push_back(std::stod(item));
  }
  return values;
}

std::vector<torch::Tensor> make_inputs(bool after_update, int device) {
  const auto options = torch::TensorOptions()
                           .dtype(torch::kInt64)
                           .device(torch::kCUDA, device);
  if (after_update) {
    return {
        torch::tensor({1, 1}, options),
        torch::tensor({0, 1, 2}, options),
        torch::tensor({1, 256, 1, 256}, options),
        torch::tensor({0, 2, 4}, options),
    };
  }
  auto table_keys = torch::arange(256, options);
  auto keys = torch::cat({table_keys, table_keys});
  return {
      keys,
      torch::tensor({0, 256, 512}, options),
      keys.clone(),
      torch::tensor({0, 256, 512}, options),
  };
}

void verify_sums(const std::vector<torch::Tensor>& outputs,
                 const std::vector<double>& expected) {
  if (outputs.size() != expected.size()) {
    throw std::runtime_error("Unexpected output count");
  }
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const double actual = outputs[index].sum().item<double>();
    const double tolerance = std::max(0.01, std::abs(expected[index]) * 1e-6);
    if (std::abs(actual - expected[index]) > tolerance) {
      throw std::runtime_error("Output checksum mismatch");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse_arguments(argc, argv);
    c10::cuda::CUDAGuard guard(args.device);

#if !defined(DYNAMICEMB_NVE_2605)
    std::shared_ptr<nve::ResourceDirectory> nve_resources;
#endif
    std::unique_ptr<nve::LayerDirectory> nve_layers;
    std::unique_ptr<dynamicemb::exportable_embedding::
                        EmbeddingCollectionIndexerDirectory>
        indexers;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;
#if defined(DYNAMICEMB_NVE_2605)
    nve_layers = std::make_unique<nve::LayerDirectory>(args.package_dir,
                                                       args.device);
    loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        args.package_dir + "/model.pt2", "model", false, 1, args.device);
#else
    loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        args.package_dir + "/model.pt2", "model", false, 1, args.device);
    nve_resources = std::make_shared<nve::ResourceDirectory>();
    nve_layers = std::make_unique<nve::LayerDirectory>(
        args.package_dir, *loader, args.device, nve_resources);
#endif

    indexers = dynamicemb::exportable_embedding::
        EmbeddingCollectionIndexerDirectory::load(args.package_dir, args.device);
    indexers->bind(*loader);

    const auto run_and_verify = [&](bool after_update,
                                    const std::vector<double>& expected) {
      auto call = indexers->begin_inference();
      auto outputs = loader->run(make_inputs(after_update, args.device));
      call.record_complete(
          at::cuda::getCurrentCUDAStream(args.device).stream());
      verify_sums(outputs, expected);
    };

    run_and_verify(false, parse_sums(args.expected_sums));

#if !defined(DYNAMICEMB_NVE_2605)
    if (args.wait_for_updates) {
      dynamicemb::exportable_embedding::EmbeddingCollectionUpdateSubscriber
          subscriber(args.package_dir, *indexers, *nve_layers, args.device);
      std::vector<dynamicemb::exportable_embedding::EmbeddingCollectionUpdate>
          updates;
      std::vector<double> updated_expected;
      bool run_requested = false;

      std::cout << "READY incremental updates" << std::endl;
      std::string line;
      while (std::getline(std::cin, line)) {
        if (line.rfind("RUN ", 0) == 0) {
          updated_expected = parse_sums(line.substr(4));
          run_requested = true;
          break;
        }
        if (line.empty()) continue;
        updates.push_back(dynamicemb::exportable_embedding::
                              EmbeddingCollectionUpdate::from_json(line));
        subscriber.apply_incremental_load(updates.back());
      }
      if (!run_requested) {
        throw std::runtime_error("Update stream ended before RUN signal");
      }

      run_and_verify(true, updated_expected);
      for (const auto& update : updates) {
        const auto ack = subscriber.wait_for_retirement(
            update.collection_id, update.snapshot_id);
        std::cout << "ACK " << ack.to_json() << '\n';
      }
    }
#else
    if (args.wait_for_updates) {
      throw std::runtime_error("Incremental replay requires NVE 26.07");
    }
#endif
    std::cout << "verified C++ AOTI replay" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
