#include <dlfcn.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>
#include <torch/script.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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
  std::cout << "[INFO] Loaded " << label << ": " << path << '\n';
  return true;
}

std::string infer_default_inference_emb_ops_path(const char* argv0) {
  std::filesystem::path exe_path = std::filesystem::absolute(argv0);
  return (exe_path.parent_path() / "../../aoti_demo/lib/inference_emb_ops.so")
      .lexically_normal()
      .string();
}

std::string infer_default_nve_torch_path(const char* argv0) {
  std::filesystem::path exe_path = std::filesystem::absolute(argv0);
  return (exe_path.parent_path() / "../../aoti_demo/lib/libnve_torch.so")
      .lexically_normal()
      .string();
}

std::string infer_default_hstu_runtime_ops_path(const char* argv0) {
  std::filesystem::path exe_path = std::filesystem::absolute(argv0);
  return (exe_path.parent_path() / "libhstu_cuda_ops_runtime.so")
      .lexically_normal()
      .string();
}

void try_load_fbgemm_operators() {
#ifdef FBGEMM_GPU_AVAILABLE
  std::cout << "[INFO] FBGEMM GPU library is linked.\n";
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
    std::cout << "[WARN] Could not load fbgemm_gpu_py.so. "
                 "fbgemm ops may be unavailable.\n";
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
    std::cout << "[WARN] Could not load fbgemm_gpu_experimental_hstu.so. "
                 "HSTU experimental fbgemm ops may be unavailable.\n";
  }
}

torch::Tensor load_tensor(const std::string& path) {
  torch::Tensor tensor;
  // Load as TorchScript module (wrapper contains tensor as buffer "tensor")
  auto module = torch::jit::load(path);
  tensor = module.attr("tensor").toTensor();
  return tensor;
}

bool file_exists(const std::string& path) {
  return std::filesystem::exists(std::filesystem::path(path));
}

struct DemoConfig {
  std::string package_path;
  std::string dump_dir;
  std::string model_name = "model";
  int device_index = 0;
  int batch_index = -1;  // -1 means run all dumped batches.
  std::string inference_emb_ops_path;
  std::string nve_torch_path;
  std::string hstu_runtime_ops_path;
};

DemoConfig parse_args(int argc, char** argv) {
  if (argc < 3) {
    throw std::invalid_argument(
        "Usage: inferece_hstu_gr_ranking_exported_model <hstu_gr_ranking_model.pt2> <dump_dir> "
        "[model_name] [device_index] [batch_index] "
        "[inference_emb_ops.so] [libnve_torch.so] [libhstu_cuda_ops_runtime.so]");
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
  cfg.nve_torch_path =
      (argc > 7) ? argv[7] : infer_default_nve_torch_path(argv[0]);
  cfg.hstu_runtime_ops_path =
      (argc > 8) ? argv[8] : infer_default_hstu_runtime_ops_path(argv[0]);

  if (cfg.package_path.empty() || cfg.dump_dir.empty()) {
    throw std::invalid_argument("package_path and dump_dir must not be empty");
  }
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
  char buffer[128];
  std::snprintf(buffer, sizeof(buffer), "batch_%06d_%s.pt", batch_idx, suffix.c_str());
  return (std::filesystem::path(dump_dir) / buffer).string();
}

bool run_one_batch(
    torch::inductor::AOTIModelPackageLoader& loader,
    const std::string& dump_dir,
    int batch_idx,
  const c10::Device& device) {
  const std::string values_path = batch_file(dump_dir, batch_idx, "values");
  const std::string lengths_path = batch_file(dump_dir, batch_idx, "lengths");
  const std::string num_candidates_path = batch_file(dump_dir, batch_idx, "num_candidates");
  const std::string ref_logits_path = batch_file(dump_dir, batch_idx, "ref_logits");

  if (!file_exists(values_path) || !file_exists(lengths_path) ||
      !file_exists(num_candidates_path) || !file_exists(ref_logits_path)) {
    std::cerr << "[WARN] Skip batch " << batch_idx
              << " because one or more dump files are missing.\n";
    return false;
  }

  torch::Tensor values_cpu = load_tensor(values_path);
  torch::Tensor lengths_cpu = load_tensor(lengths_path);
  torch::Tensor num_candidates_cpu = load_tensor(num_candidates_path);
  torch::Tensor ref_logits_cpu = load_tensor(ref_logits_path);

  auto values = values_cpu.to(device, /*dtype=*/torch::kInt64).contiguous();
  auto lengths = lengths_cpu.to(device, /*dtype=*/torch::kInt64).contiguous();
  auto num_candidates = num_candidates_cpu.to(device, /*dtype=*/torch::kInt64).contiguous();

  std::vector<torch::Tensor> outputs = loader.run({values, lengths, num_candidates});
  TORCH_CHECK(!outputs.empty(), "Model returned no outputs.");

  torch::Tensor logits_cpu = outputs[0].to(torch::kCPU).contiguous();
  torch::Tensor ref_cpu = ref_logits_cpu.to(logits_cpu.dtype()).contiguous();

  if (!logits_cpu.sizes().equals(ref_cpu.sizes())) {
    std::cerr << "[ERROR] Batch " << batch_idx << " shape mismatch: logits="
              << logits_cpu.sizes() << ", ref=" << ref_cpu.sizes() << '\n';
    return false;
  }

  // Compare only in bfloat16 on GPU.
  torch::Tensor logits_bf16 = outputs[0].to(device, torch::kBFloat16).contiguous();
  torch::Tensor ref_bf16 = ref_cpu.to(device, torch::kBFloat16).contiguous();

  torch::Tensor diff_bf16 = (logits_bf16.to(torch::kFloat32) - ref_bf16.to(torch::kFloat32)).abs();
  torch::Tensor ref_abs_bf16 = ref_bf16.to(torch::kFloat32).abs();
  torch::Tensor rel_diff_bf16 = diff_bf16 / (ref_abs_bf16 + 1e-8f);

  const double max_abs_diff = diff_bf16.max().item<double>();
  const double mean_rel_diff = rel_diff_bf16.mean().item<double>();
  const bool pass_abs_threshold = max_abs_diff <= 0.0625;

  std::cout << "[INFO] Batch " << batch_idx << ": [bf16 compare]"
            << " max_abs_diff=" << max_abs_diff << ";"
            << " mean_rel_diff=" << mean_rel_diff << ";"
            << " pass(max_abs_diff<=0.0625)=" << (pass_abs_threshold ? "True" : "False")
            << '\n';

  return pass_abs_threshold;
}

}  // namespace

void load_required_libraries(const DemoConfig& cfg) {
  try_load_fbgemm_operators();
  try_load_fbgemm_hstu_experimental_operators();

  TORCH_CHECK(
      load_shared_library("inference_emb_ops", cfg.inference_emb_ops_path),
      "inference_emb_ops is required.");
  TORCH_CHECK(
      load_shared_library("libnve_torch", cfg.nve_torch_path),
      "libnve_torch is required.");
  // TORCH_CHECK(
  //     load_shared_library("splitops", cfg.splitops_path),
  //     "splitops custom op library is required.");
  TORCH_CHECK(
      load_shared_library("hstu_cuda_ops_runtime", cfg.hstu_runtime_ops_path),
      "hstu_cuda_ops_runtime is required.");
}

int main(int argc, char** argv) {
  try {
    c10::InferenceMode guard;
    DemoConfig cfg = parse_args(argc, argv);

    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for this demo.");
    TORCH_CHECK(
        cfg.device_index >= 0 && cfg.device_index < torch::cuda::device_count(),
        "Invalid CUDA device index: ",
        cfg.device_index);

    c10::Device device(torch::kCUDA, cfg.device_index);
    c10::cuda::CUDAGuard device_guard(device);

    load_required_libraries(cfg);

    torch::inductor::AOTIModelPackageLoader loader(
        cfg.package_path,
        cfg.model_name,
        /*run_single_threaded=*/false,
        /*num_runners=*/1,
        cfg.device_index);

    auto call_spec = loader.get_call_spec();
    std::cout << "Input call spec:\n" << call_spec[0] << "\n\n";
    std::cout << "Output call spec:\n" << call_spec[1] << "\n\n";

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
      ++total;
      if (run_one_batch(loader, cfg.dump_dir, idx, device)) {
        ++passed;
      }
    }

    std::cout << "[INFO] max_abs_diff<=0.0625 passed " << passed << "/" << total << " batches.\n";
    return (passed == total) ? 0 : 2;
  } catch (const c10::Error& e) {
    std::cerr << "PyTorch error: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
