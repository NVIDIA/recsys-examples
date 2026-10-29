// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <atomic>
#include <thread>
#ifdef HOST_EXAMPLE_HAS_UPDATE
#include "dynamicemb/exportable_embedding/incremental_update.h"
#endif
#include <iostream>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include "dynamicemb/exportable_embedding/indexer_directory.h"
#include "python/pynve/torch_bindings/nve_loader.hpp"

int main(int argc, char** argv) {
  try {
    TORCH_CHECK(argc == 3, "Usage: replay_host PACKAGE_DIR DEVICE_INDEX (-1 for CPU)");
    const std::string package = argv[1];
    const int index = std::stoi(argv[2]);
    const auto device = index < 0 ? torch::Device(torch::kCPU) : torch::Device(torch::kCUDA, index);
    c10::cuda::OptionalCUDAGuard guard;
    if (index >= 0) guard.set_index(index);
    // Resource owners must outlive the AOTI loader's user-managed constants.
    auto resources = std::make_shared<nve::ResourceDirectory>();
    std::unique_ptr<nve::LayerDirectory> layers;
    std::unique_ptr<dynamicemb::exportable_embedding::EmbeddingCollectionIndexerDirectory> indexers;
    torch::inductor::AOTIModelPackageLoader loader(package + "/model.pt2", "model", false, 1, index);
    layers = std::make_unique<nve::LayerDirectory>(package, loader, index, resources);
    indexers = dynamicemb::exportable_embedding::EmbeddingCollectionIndexerDirectory::load(package, index);
    indexers->bind(loader);
    std::ifstream stream(package + "/replay_cases.json");
    auto cases = nlohmann::json::parse(stream);
    const auto verify = [&](const nlohmann::json& tests) {
    for (const auto& test : tests) {
      std::vector<at::Tensor> inputs;
      for (const auto& values : test.at("inputs"))
        inputs.push_back(torch::tensor(values.get<std::vector<int64_t>>(),
                                      torch::TensorOptions().dtype(torch::kLong)).to(device));
      const auto expected_rows = test.at("expected").get<std::vector<std::vector<float>>>();
      std::vector<float> flat;
      for (const auto& row : expected_rows) flat.insert(flat.end(), row.begin(), row.end());
      auto expected = torch::tensor(flat).reshape({static_cast<int64_t>(expected_rows.size()), 8});
      auto outputs = loader.run(inputs);
      TORCH_CHECK(outputs.size() == 1 && torch::allclose(outputs[0].cpu().to(torch::kFloat), expected, 1e-3, 1e-3),
                  "HostLayer replay differs from reference");
    }
    };
    verify(cases);
#ifdef HOST_EXAMPLE_HAS_UPDATE
    std::ifstream updates_file(package + "/host_updates.json");
    if (updates_file) {
      TORCH_CHECK(index < 0, "Incremental Host example requires CPU execution");
      const auto document = nlohmann::json::parse(updates_file);
      const bool paused = document.at("paused").get<bool>();
      dynamicemb::exportable_embedding::EmbeddingCollectionUpdateSubscriber
          subscriber(package, *indexers, *layers, -1);
      std::vector<at::Tensor> inputs;
      for (const auto& values : cases.at(0).at("inputs"))
        inputs.push_back(torch::tensor(values.get<std::vector<int64_t>>()));
      std::atomic<bool> ready{false}, stop{false};
      std::atomic<int64_t> lookups{0};
      std::exception_ptr reader_error;
      std::thread reader;
      if (!paused) {
        reader = std::thread([&] {
          try {
            while (!stop.load()) {
              auto output = loader.run(inputs)[0];
              TORCH_CHECK(torch::isfinite(output).all().item<bool>(), "Invalid concurrent result");
              ++lookups;
              ready.store(true);
            }
          } catch (...) { reader_error = std::current_exception(); ready.store(true); }
        });
        while (!ready.load()) std::this_thread::yield();
      }
      auto before = lookups.load();
      try {
        for (const auto& payload : document.at("updates")) {
          const auto update = dynamicemb::exportable_embedding::EmbeddingCollectionUpdate::from_json(payload.dump());
          subscriber.apply_incremental_load(update, nullptr, paused);
          subscriber.wait_for_retirement(update.collection_id, update.snapshot_id);
        }
      } catch (...) {
        stop.store(true);
        if (reader.joinable()) reader.join();
        throw;
      }
      stop.store(true);
      if (reader.joinable()) reader.join();
      if (reader_error) std::rethrow_exception(reader_error);
      TORCH_CHECK(paused || lookups.load() > before, "Inference did not progress during loading");
      verify(document.at("final_cases"));
      std::cout << "Verified C++ CPU Host updates, concurrent lookups=" << lookups.load() << std::endl;
    }
#endif
    std::cout << "Verified C++ HostLayer AOT replay on " << device << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
