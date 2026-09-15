// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include "dynamicemb/exportable_embedding/indexer_directory.h"
#if !defined(DYNAMICEMB_NVE_2605)
#include "dynamicemb/exportable_embedding/incremental_update.h"
#endif
#include "python/pynve/torch_bindings/nve_loader.hpp"

namespace {

constexpr int64_t kRowsPerTable = 256;
constexpr int kRequestsPerVerifiedRound = 100;
constexpr int kRoundThreeMinInferences = 1000;
constexpr int kRoundThreePostStageInferences = 100;

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

std::vector<torch::Tensor> make_inputs(bool after_update,
                                       int device,
                                       int64_t new_feature_id = kRowsPerTable) {
  const auto options = torch::TensorOptions()
                           .dtype(torch::kInt64)
                           .device(torch::kCUDA, device);
  if (after_update) {
    return {
        torch::tensor({1, 1}, options),
        torch::tensor({0, 1, 2}, options),
        torch::tensor({int64_t{1}, new_feature_id, int64_t{1}, new_feature_id},
                      options),
        torch::tensor({0, 2, 4}, options),
    };
  }
  auto table_keys = torch::arange(kRowsPerTable, options);
  auto keys = torch::cat({table_keys, table_keys});
  return {
      keys,
      torch::tensor({int64_t{0}, kRowsPerTable, 2 * kRowsPerTable}, options),
      keys.clone(),
      torch::tensor({int64_t{0}, kRowsPerTable, 2 * kRowsPerTable}, options),
  };
}

bool verify_sums(const std::vector<torch::Tensor>& outputs,
                 const std::vector<double>& expected,
                 std::string& error) {
  if (outputs.size() != expected.size()) {
    std::ostringstream message;
    message << "unexpected output count: actual=" << outputs.size()
            << " expected=" << expected.size();
    error = message.str();
    return false;
  }
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const double actual = outputs[index].sum().item<double>();
    const double tolerance = std::max(0.01, std::abs(expected[index]) * 1e-6);
    if (std::abs(actual - expected[index]) > tolerance) {
      std::ostringstream message;
      message << "output=" << index << " actual=" << actual
              << " expected=" << expected[index];
      error = message.str();
      return false;
    }
  }
  return true;
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
    const auto inference_cuda_stream =
        at::cuda::getDefaultCUDAStream(args.device);
    const auto inference_stream = inference_cuda_stream.stream();
    c10::cuda::CUDAStreamGuard inference_stream_guard(inference_cuda_stream);

    const auto run_and_verify = [&](const char* stage,
                                    bool after_update,
                                    int64_t new_feature_id,
                                    const std::vector<double>& expected) {
      std::vector<torch::Tensor> outputs;
      std::string error;
      for (int iteration = 0; iteration < kRequestsPerVerifiedRound;
           ++iteration) {
        outputs = loader->run(
            make_inputs(after_update, args.device, new_feature_id),
            inference_stream);
        if (!verify_sums(outputs, expected, error)) break;
      }
      cudaDeviceSynchronize();
      const bool verified = error.empty();
      if (!verified) {
        std::cout << "RESULT ERROR stage=" << stage << ' ' << error << '\n'
                  << std::flush;
      }
      return verified;
    };

    bool verification_ok = run_and_verify(
        "round_1_original",
        false,
        kRowsPerTable,
        parse_sums(args.expected_sums));

#if !defined(DYNAMICEMB_NVE_2605)
    if (args.wait_for_updates) {
      dynamicemb::exportable_embedding::EmbeddingCollectionUpdateSubscriber
          subscriber(args.package_dir, *indexers, *nve_layers, args.device);
      std::vector<dynamicemb::exportable_embedding::EmbeddingCollectionUpdate>
          updates;
      int update_round = 0;

      std::cout << "READY incremental updates" << std::endl;
      std::string line;
      while (std::getline(std::cin, line)) {
        if (line == "STOP") {
          break;
        }
        if (line.rfind("RUN ", 0) == 0) {
          const auto expected = parse_sums(line.substr(4));
          std::vector<std::string> acknowledgements;
          if (update_round == 0) {
            std::promise<void> update_staged;
            auto update_staged_future = update_staged.get_future();
            std::thread update_thread([&] {
              for (const auto& update : updates) {
                subscriber.apply_incremental_load(update, inference_stream);
              }
              update_staged.set_value();

              for (const auto& update : updates) {
                acknowledgements.push_back(
                    subscriber
                        .wait_for_retirement(
                            update.collection_id, update.snapshot_id)
                        .to_json());
              }
            });

            update_staged_future.wait();
            const bool round_ok = run_and_verify(
                "round_2_snapshot_1", true, kRowsPerTable, expected);
            verification_ok = round_ok && verification_ok;
            update_thread.join();
          } else {
            std::promise<void> inference_started;
            auto inference_started_future = inference_started.get_future();
            std::atomic<bool> update_staged{false};

            std::thread update_thread([&] {
              inference_started_future.wait();

              for (const auto& update : updates) {
                subscriber.apply_incremental_load(update, inference_stream);
              }
              update_staged.store(true, std::memory_order_release);

              for (const auto& update : updates) {
                acknowledgements.push_back(
                    subscriber
                        .wait_for_retirement(
                            update.collection_id, update.snapshot_id)
                        .to_json());
              }
            });

            for (int iteration = 0;
                 iteration < kRoundThreeMinInferences;
                 ++iteration) {
              loader->run(make_inputs(true, args.device, kRowsPerTable + 1),
                          inference_stream);
              if (iteration == 0) inference_started.set_value();
            }
            while (!update_staged.load(std::memory_order_acquire)) {
              loader->run(make_inputs(true, args.device, kRowsPerTable + 1),
                          inference_stream);
            }
            for (int iteration = 0;
                 iteration < kRoundThreePostStageInferences;
                 ++iteration) {
              loader->run(make_inputs(true, args.device, kRowsPerTable + 1),
                          inference_stream);
            }
            cudaDeviceSynchronize();
            update_thread.join();

            const bool round_ok = run_and_verify(
                "round_4_snapshot_2", true, kRowsPerTable + 1, expected);
            verification_ok = round_ok && verification_ok;
          }
          for (const auto& acknowledgement : acknowledgements) {
            std::cout << "ACK " << acknowledgement << '\n';
          }
          updates.clear();
          ++update_round;
          std::cout << "READY incremental updates" << std::endl;
          continue;
        }
        if (line.empty()) continue;
        updates.push_back(dynamicemb::exportable_embedding::
                              EmbeddingCollectionUpdate::from_json(line));
      }
    }
#else
    if (args.wait_for_updates) {
      throw std::runtime_error(
          "Incremental replay requires NVE 26.06 or later");
    }
#endif
    if (verification_ok) {
      std::cout << "verified C++ AOTI replay" << std::endl;
    } else {
      std::cout << "C++ AOTI replay completed with verification errors"
                << std::endl;
    }
    return verification_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
