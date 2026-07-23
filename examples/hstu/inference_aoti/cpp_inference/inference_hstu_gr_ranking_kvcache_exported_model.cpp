#include <dlfcn.h>

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "python/pynve/torch_bindings/nve_loader.hpp"

namespace {

#ifndef HSTU_KVCACHE_CPP_DEMO_DEBUG
#define HSTU_KVCACHE_CPP_DEMO_DEBUG 0
#endif

#ifndef HSTU_KVCACHE_CPP_DEMO_CUDA_CHECK
#define HSTU_KVCACHE_CPP_DEMO_CUDA_CHECK 0
#endif

void log_demo_debug(const std::string& message) {
#if HSTU_KVCACHE_CPP_DEMO_DEBUG
  std::cout << message << '\n';
#else
  (void)message;
#endif
}

bool load_shared_library(const std::string& label, const std::string& path) {
  dlerror();
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    const char* err = dlerror();
    std::cerr << "[WARN] Failed to load " << label << ": " << path;
    if (err != nullptr) {
      std::cerr << " | dlerror: " << err;
    }
    std::cerr << '\n';
    return false;
  }
  log_demo_debug("[INFO] Loaded " + label + ": " + path);
  return true;
}

std::string infer_repo_root(const char* argv0) {
  std::filesystem::path exe_path = std::filesystem::absolute(argv0);
  return (exe_path.parent_path() / "../../../../..").lexically_normal().string();
}

std::string infer_default_inference_emb_ops_path(const char* argv0) {
  return (std::filesystem::path(infer_repo_root(argv0)) /
          "corelib/dynamicemb/torch_binding_build/inference_emb_ops.so")
      .lexically_normal()
      .string();
}

std::string infer_default_kvcache_manager_ops_path(const char* argv0) {
  std::filesystem::path exe_path = std::filesystem::absolute(argv0);
  const std::vector<std::filesystem::path> candidates = {
      exe_path.parent_path() / "kvcache_manager_ops.so",
      "/usr/local/lib/python3.12/dist-packages/recsys_kvcache_manager/kvcache_manager_ops.so",
      std::filesystem::path(infer_repo_root(argv0)) /
          "corelib/recsys_kvcache_manager/build/kvcache_manager_ops.so",
      std::filesystem::path(infer_repo_root(argv0)) /
          "corelib/recsys_kvcache_manager/kvcache_manager_ops.so",
  };

  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      return candidate.lexically_normal().string();
    }
  }

  return candidates.front().string();
}

std::string infer_default_hstu_runtime_ops_path(const char* argv0) {
  std::filesystem::path exe_path = std::filesystem::absolute(argv0);
  return (exe_path.parent_path() / "libhstu_cuda_ops_runtime.so")
      .lexically_normal()
      .string();
}

std::string infer_default_paged_kvcache_ops_path(const char* argv0) {
  std::filesystem::path exe_path = std::filesystem::absolute(argv0);
  const std::vector<std::filesystem::path> candidates = {
      exe_path.parent_path() / "libpaged_kvcache_ops_runtime.so",
      "/usr/local/lib/python3.12/dist-packages/paged_kvcache_ops.cpython-312-x86_64-linux-gnu.so",
      std::filesystem::path(infer_repo_root(argv0)) /
          "examples/commons/paged_kvcache_ops.cpython-312-x86_64-linux-gnu.so",
      std::filesystem::path(infer_repo_root(argv0)) /
          "examples/commons/build/lib.linux-x86_64-cpython-312/paged_kvcache_ops.cpython-312-x86_64-linux-gnu.so",
  };

  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      return candidate.lexically_normal().string();
    }
  }

  return candidates.front().string();
}

void try_load_fbgemm_operators() {
#ifdef FBGEMM_GPU_AVAILABLE
  log_demo_debug("[INFO] FBGEMM GPU library is linked.");
#else
  const char* fbgemm_so_paths[] = {
      "/usr/local/lib/python3.12/dist-packages/fbgemm_gpu/fbgemm_gpu_py.so",
  };

  bool loaded = false;
  for (const auto& path : fbgemm_so_paths) {
    loaded = load_shared_library("fbgemm_gpu", path);
    if (loaded) {
      break;
    }
  }
  if (!loaded) {
    log_demo_debug("[WARN] Could not load fbgemm_gpu_py.so. fbgemm ops may be unavailable.");
  }
#endif
}

void try_load_fbgemm_hstu_experimental_operators() {
  const char* hstu_fbgemm_so_paths[] = {
      "/usr/local/lib/python3.12/dist-packages/hstu/fbgemm_gpu_experimental_hstu.so",
  };

  bool loaded = false;
  for (const auto& path : hstu_fbgemm_so_paths) {
    loaded = load_shared_library("fbgemm_gpu_experimental_hstu", path);
    if (loaded) {
      break;
    }
  }
  if (!loaded) {
    log_demo_debug(
        "[WARN] Could not load fbgemm_gpu_experimental_hstu.so. "
        "HSTU experimental fbgemm ops may be unavailable.");
  }
}

torch::Tensor load_tensor(const std::string& path) {
  auto module = torch::jit::load(path);
  return module.attr("tensor").toTensor();
}

bool file_exists(const std::string& path) {
  return std::filesystem::exists(std::filesystem::path(path));
}

bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  std::string text(value);
  return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

struct DemoConfig {
  std::string package_path;
  std::string dump_dir;
  std::string model_name = "model";
  int device_index = 0;
  int batch_index = -1;
  std::string inference_emb_ops_path;
  std::string hstu_runtime_ops_path;
  std::string paged_kvcache_ops_path;
  std::string kvcache_manager_ops_path;
};

DemoConfig parse_args(int argc, char** argv) {
  if (argc < 3) {
    throw std::invalid_argument(
        "Usage: inference_hstu_gr_ranking_kvcache_exported_model <hstu_gr_ranking_model_dir> <dump_dir> "
        "[model_name] [device_index] [batch_index] "
        "[inference_emb_ops.so] [libhstu_cuda_ops_runtime.so] [kvcache_manager_ops.so] [paged_kvcache_ops.so]\n"
      "Before running, start start_flexkv_server_for_kvcache_cpp.py with the shared kvcache_cpp_runtime.yaml.");
  }

  DemoConfig cfg;
  cfg.package_path = argv[1];
  cfg.dump_dir = argv[2];
  if (argc > 3) {
    cfg.model_name = argv[3];
  }
  if (argc > 4) {
    cfg.device_index = std::stoi(argv[4]);
  }
  if (argc > 5) {
    cfg.batch_index = std::stoi(argv[5]);
  }
  cfg.inference_emb_ops_path =
      (argc > 6) ? argv[6] : infer_default_inference_emb_ops_path(argv[0]);
  cfg.hstu_runtime_ops_path =
      (argc > 7) ? argv[7] : infer_default_hstu_runtime_ops_path(argv[0]);
  cfg.kvcache_manager_ops_path =
      (argc > 8) ? argv[8] : infer_default_kvcache_manager_ops_path(argv[0]);
  cfg.paged_kvcache_ops_path =
      (argc > 9) ? argv[9] : infer_default_paged_kvcache_ops_path(argv[0]);
  return cfg;
}

std::vector<int> discover_batch_indices(const std::string& dump_dir) {
  std::vector<int> indices;
  std::regex pattern(R"(batch_(\d+)_values\.pt)");
  for (const auto& entry : std::filesystem::directory_iterator(dump_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string filename = entry.path().filename().string();
    std::smatch match;
    if (std::regex_match(filename, match, pattern)) {
      indices.push_back(std::stoi(match[1].str()));
    }
  }
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  return indices;
}

std::string batch_file(const std::string& dump_dir, int batch_idx, const std::string& suffix) {
  char buffer[160];
  std::snprintf(buffer, sizeof(buffer), "batch_%06d_%s.pt", batch_idx, suffix.c_str());
  return (std::filesystem::path(dump_dir) / buffer).string();
}

std::filesystem::path find_kvcache_config_path() {
  const char* override_path = std::getenv("KVCACHE_MANAGER_CONFIG_FILE");
  if (override_path != nullptr && std::string(override_path).size() > 0) {
    return std::filesystem::path(override_path);
  }

  const std::vector<std::filesystem::path> candidates = {
      "inference_aoti/kvcache_cpp_runtime.yaml",
      "examples/hstu/inference_aoti/kvcache_cpp_runtime.yaml",
      "kvcache_cpp_runtime.yaml",
  };
  for (const auto& path : candidates) {
    if (std::filesystem::exists(path)) {
      return path;
    }
  }
  return candidates.front();
}

void check_required_kvcache_config() {
  auto config_path = find_kvcache_config_path();
  TORCH_CHECK(
      std::filesystem::exists(config_path),
      "Missing KV-cache config YAML: ",
      config_path.string(),
      ". Set KVCACHE_MANAGER_CONFIG_FILE to the shared kvcache_cpp_runtime.yaml path.");
  log_demo_debug(std::string("[INFO] KV-cache config=") + config_path.string());
}

void shutdown_kvcache_runtime() {
  try {
    auto dummy = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    c10::Dispatcher::singleton()
        .findSchemaOrThrow("kvcache_manager_ops::shutdown_runtime", "")
        .typed<void(at::Tensor)>()
        .call(dummy);
  } catch (const std::exception& e) {
    std::cerr << "[WARN] Failed to shutdown kvcache runtime cleanly: " << e.what() << '\n';
  }
}

void init_kvcache_runtime() {
  log_demo_debug("[INFO] Preflight kvcache_manager_ops::init_kvcache begin");
  auto dummy = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t result = c10::Dispatcher::singleton()
      .findSchemaOrThrow("kvcache_manager_ops::init_kvcache", "")
      .typed<int64_t(at::Tensor)>()
      .call(dummy);
  log_demo_debug(
      "[INFO] Preflight kvcache_manager_ops::init_kvcache done result="
      + std::to_string(result));
}

void check_cuda_after(const char* label) {
#if HSTU_KVCACHE_CPP_DEMO_CUDA_CHECK
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    std::cerr << "[ERROR] CUDA last error after " << label << ": "
              << cudaGetErrorString(last_error) << std::endl;
  }
  cudaError_t sync_error = cudaDeviceSynchronize();
  if (sync_error != cudaSuccess) {
    std::cerr << "[ERROR] CUDA synchronize error after " << label << ": "
              << cudaGetErrorString(sync_error) << std::endl;
    throw std::runtime_error(
        std::string("CUDA synchronize failed after ") + label + ": " + cudaGetErrorString(sync_error));
  }
  log_demo_debug(std::string("[INFO] CUDA synchronize ok after ") + label);
#else
  (void)label;
#endif
}

bool run_one_batch(
    torch::inductor::AOTIModelPackageLoader& loader,
    const std::string& dump_dir,
    int batch_idx,
    const c10::Device& device) {
  const std::string values_path = batch_file(dump_dir, batch_idx, "values");
  const std::string lengths_path = batch_file(dump_dir, batch_idx, "lengths");
  const std::string num_candidates_path = batch_file(dump_dir, batch_idx, "num_candidates");
  const std::string user_ids_path = batch_file(dump_dir, batch_idx, "user_ids");
  const std::string total_history_lengths_path = batch_file(dump_dir, batch_idx, "total_history_lengths");
  const std::string compiled_logits_path = batch_file(dump_dir, batch_idx, "compiled_logits");

  if (!file_exists(values_path) || !file_exists(lengths_path) ||
      !file_exists(num_candidates_path) || !file_exists(user_ids_path) ||
      !file_exists(total_history_lengths_path) || !file_exists(compiled_logits_path)) {
    std::cerr << "[WARN] Skip batch " << batch_idx << " because one or more dump files are missing.\n";
    return false;
  }

  auto values = load_tensor(values_path).to(device, torch::kInt64).contiguous();
  auto lengths = load_tensor(lengths_path).to(device, torch::kInt64).contiguous();
  auto num_candidates = load_tensor(num_candidates_path).to(device, torch::kInt64).contiguous();
  auto user_ids = load_tensor(user_ids_path).to(torch::kCPU, torch::kInt64).contiguous();
  auto total_history_lengths =
      load_tensor(total_history_lengths_path).to(torch::kCPU, torch::kInt64).contiguous();
  auto ref_logits_cpu = load_tensor(compiled_logits_path).to(torch::kCPU).contiguous();

    log_demo_debug("[INFO] Batch " + std::to_string(batch_idx) + " loader.run begin");
  std::vector<torch::Tensor> outputs = loader.run(
      {values, lengths, num_candidates, user_ids, total_history_lengths});
    log_demo_debug(
      "[INFO] Batch " + std::to_string(batch_idx)
      + " loader.run returned outputs=" + std::to_string(outputs.size()));
  check_cuda_after("loader.run");
  TORCH_CHECK(!outputs.empty(), "Model returned no outputs.");

    log_demo_debug("[INFO] Batch " + std::to_string(batch_idx) + " logits copy to CPU begin");
  torch::Tensor logits_cpu = outputs[0].to(torch::kCPU).contiguous();
    log_demo_debug("[INFO] Batch " + std::to_string(batch_idx) + " logits copy to CPU done");
  torch::Tensor ref_cpu = ref_logits_cpu.to(logits_cpu.dtype()).contiguous();
  if (!logits_cpu.sizes().equals(ref_cpu.sizes())) {
    std::cerr << "[ERROR] Batch " << batch_idx << " shape mismatch: logits="
              << logits_cpu.sizes() << ", ref=" << ref_cpu.sizes() << '\n';
    return false;
  }

  torch::Tensor diff = (logits_cpu.to(torch::kFloat32) - ref_cpu.to(torch::kFloat32)).abs();
  const double max_abs_diff = diff.max().item<double>();
  const bool pass = max_abs_diff <= 0.0625;

  torch::Tensor offload_task_ids;
  if (outputs.size() > 1 && outputs[1].defined()) {
    offload_task_ids = outputs[1].to(torch::kCPU, torch::kInt64).contiguous();
  }

  std::cout << "[INFO] Batch " << batch_idx
            << ": max_abs_diff=" << max_abs_diff
            << "; pass(max_abs_diff<=0.0625)=" << (pass ? "True" : "False");
  if (offload_task_ids.defined()) {
    std::cout << "; offload_task_ids_shape=" << offload_task_ids.sizes();
  }
  std::cout << '\n';
  return pass;
}

void load_required_libraries(const DemoConfig& cfg) {
  check_required_kvcache_config();
  try_load_fbgemm_operators();
  try_load_fbgemm_hstu_experimental_operators();
  TORCH_CHECK(load_shared_library("inference_emb_ops", cfg.inference_emb_ops_path),
              "inference_emb_ops is required.");
  TORCH_CHECK(load_shared_library("hstu_cuda_ops_runtime", cfg.hstu_runtime_ops_path),
              "hstu_cuda_ops_runtime is required.");
  TORCH_CHECK(load_shared_library("paged_kvcache_ops", cfg.paged_kvcache_ops_path),
              "paged_kvcache_ops is required.");
  TORCH_CHECK(load_shared_library("kvcache_manager_ops", cfg.kvcache_manager_ops_path),
              "kvcache_manager_ops is required.");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    c10::InferenceMode guard;
    DemoConfig cfg = parse_args(argc, argv);

    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for this demo.");
    TORCH_CHECK(cfg.device_index >= 0 && cfg.device_index < torch::cuda::device_count(),
                "Invalid CUDA device index: ", cfg.device_index);

    c10::Device device(torch::kCUDA, cfg.device_index);
    c10::cuda::CUDAGuard device_guard(device);

    load_required_libraries(cfg);
    init_kvcache_runtime();

    std::cout << "Loading NVE layers from " << cfg.package_path << '\n';
    nve::LayerDirectory dir(cfg.package_path, cfg.device_index);
    std::cout << "  Loaded " << dir.size() << " layer(s)\n";

    const bool run_single_threaded = env_flag_enabled("KVCACHE_CPP_RUN_SINGLE_THREADED");
    log_demo_debug(
      std::string("[INFO] AOTI run_single_threaded=")
      + (run_single_threaded ? "true" : "false"));

    torch::inductor::AOTIModelPackageLoader loader(
        cfg.package_path + "/model.pt2",
        cfg.model_name,
        /*run_single_threaded=*/run_single_threaded,
        /*num_runners=*/1,
        cfg.device_index);

    auto call_spec = loader.get_call_spec();
    std::cout << "Input call spec:\n" << call_spec[0] << "\n";
    std::cout << "Output call spec:\n" << call_spec[1] << "\n";

    std::vector<int> batch_indices;
    if (cfg.batch_index >= 0) {
      batch_indices = {cfg.batch_index};
    } else {
      batch_indices = discover_batch_indices(cfg.dump_dir);
    }
    TORCH_CHECK(!batch_indices.empty(), "No dumped batches found in ", cfg.dump_dir);

    int passed = 0;
    int total = 0;
    for (int idx : batch_indices) {
      std::cout << "[INFO] Running batch " << idx << "...\n";
      ++total;
      if (run_one_batch(loader, cfg.dump_dir, idx, device)) {
        ++passed;
      }
    }

    shutdown_kvcache_runtime();
    std::cout << "[INFO] max_abs_diff<=0.0625 passed " << passed << "/" << total << " batches.\n";
    return (passed == total) ? 0 : 2;
  } catch (const c10::Error& e) {
    std::cerr << "PyTorch error: " << e.what() << std::endl;
    shutdown_kvcache_runtime();
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    shutdown_kvcache_runtime();
    return 1;
  }
}
