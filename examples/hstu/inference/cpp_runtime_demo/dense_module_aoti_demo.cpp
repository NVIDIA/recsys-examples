#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <dlfcn.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

namespace {

struct DemoConfig {
  std::string package_path;
  std::string model_name = "model";
  int device_index = 0;
  int batch_size = 1;
  int history_len = 64;
  int num_candidates = 10;
  int embedding_dim = 512;
  bool print_call_spec = true;
};

// Load FBGEMM operators at runtime
// This is necessary because torch.ops.fbgemm.asynchronous_complete_cumsum and other
// FBGEMM operators are not automatically registered unless the fbgemm_gpu library is loaded.
void load_fbgemm_operators() {
#ifdef FBGEMM_GPU_AVAILABLE
  // If compiled with FBGEMM support, it should already be linked
  std::cout << "FBGEMM GPU library is linked.\n";
#else
  // Try to dynamically load fbgemm_gpu library
  // Common installation paths for fbgemm_gpu
  const char* fbgemm_so_paths[] = {
      "/usr/local/lib/python3.12/dist-packages/fbgemm_gpu/fbgemm_gpu_py.so",  // only path for fbgemm_gpu installed via pip
  };
  
  bool loaded = false;
  for (const auto& path : fbgemm_so_paths) {
    dlerror();
    void* handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    if (handle != nullptr) {
      std::cout << "Successfully loaded FBGEMM operators from: " << path << '\n';
      loaded = true;
      break;
    } else {
      const char* err = dlerror();
      if (err != nullptr) {
        std::cout << "Failed to load " << path << ": " << err << '\n';
      }
    }
  }
  
  if (!loaded) {
    std::cout << "Warning: Could not find fbgemm_gpu_py.so library.\n"
              << "FBGEMM operators may not work if the model uses them.\n"
             << "FBGEMM operators (asynchronous_complete_cumsum) may not work.\n";
  }
#endif
}

// Load experimental HSTU FBGEMM operators at runtime.
// This is required for fbgemm::hstu_varlen_fwd_80 / fbgemm::hstu_varlen_fwd_90
// when these ops are exported into the AOT package.
void load_fbgemm_hstu_experimental_operators() {
  const char* hstu_fbgemm_so_paths[] = {
      "/usr/local/lib/python3.12/dist-packages/hstu/fbgemm_gpu_experimental_hstu.so",
  };

  bool loaded = false;
  for (const auto& path : hstu_fbgemm_so_paths) {
    dlerror();
    void* handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    if (handle != nullptr) {
      std::cout << "Successfully loaded experimental HSTU FBGEMM operators from: " << path << '\n';
      loaded = true;
      break;
    } else {
      const char* err = dlerror();
      if (err != nullptr) {
        std::cout << "Failed to load " << path << ": " << err << '\n';
      }
    }
  }

  if (!loaded) {
    std::cout << "Warning: Could not load fbgemm_gpu_experimental_hstu.so.\n"
              << "fbgemm::hstu_varlen_fwd_80 / fwd_90 schemas may be missing at runtime.\n";
  }
}

// Load HSTU CUDA operators at runtime
// This is necessary because torch.ops.hstu_cuda_ops.compute_block_workloads,
// concat_2D_jagged_tensors_forward/backward are not automatically registered
// unless the hstu_cuda_ops library is loaded.
void load_hstu_operators() {
// #ifdef HSTU_CUDA_OPS_AVAILABLE
//   // If compiled with hstu_cuda_ops support, it should already be linked
//   std::cout << "HSTU CUDA ops library is linked.\n";
// #else
  // Try to dynamically load hstu_cuda_ops library
  // Common installation paths for hstu_cuda_ops
  const char* hstu_so_paths[] = {
      "./build-release/libhstu_cuda_ops_runtime.so",
  };
  
  bool loaded = false;
  for (const auto& path : hstu_so_paths) {
    void* handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    if (handle != nullptr) {
      std::cout << "Successfully loaded HSTU CUDA operators from: " << path << '\n';
      loaded = true;
      break;
    }
  }
  
  if (!loaded) {
    std::cout << "Warning: Could not find hstu_cuda_ops.so library.\n"
              << "HSTU operators (compute_block_workloads, concat_2D_jagged_tensors_*) may not work.\n"
              << "Please ensure hstu_cuda_ops is installed from examples/commons/setup.py\n";
  }
// #endif
}
constexpr int64_t kContextualFeatureLen = 1;
constexpr int64_t kMaxHistoryLen = 4096;
constexpr int64_t kMaxNumCandidates = 100;

const std::vector<std::string> kBatchFeatureOrder = {
    "user_id",
    "user_active_degree",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "video_id",
    "action_weights",
};

// IMPORTANT:
// This order must match the flattened tensor order of the `embeddings` dict used
// during export. With the current Python code, `InferenceEmbedding.forward()`
// merges dynamic tables first, then static tables:
//   embeddings = {**dynamic_embeddings, **static_embeddings}
const std::vector<std::string> kEmbeddingFlattenOrder = {
    "user_id",
    "video_id",
    "user_active_degree",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "action_weights",
};

const std::unordered_map<std::string, int64_t> kFeatureVocabSizes = {
    {"user_id", 1000},
    {"user_active_degree", 8},
    {"follow_user_num_range", 9},
    {"fans_user_num_range", 9},
    {"friend_user_num_range", 8},
    {"register_days_range", 8},
    {"video_id", 1000000},
    {"action_weights", 233},
};

void validate_config(const DemoConfig& cfg) {
  if (cfg.package_path.empty()) {
    throw std::invalid_argument("package_path must not be empty");
  }
  if (cfg.batch_size <= 0) {
    throw std::invalid_argument("batch_size must be > 0");
  }
  if (cfg.history_len <= 0 || cfg.history_len > kMaxHistoryLen) {
    throw std::invalid_argument("history_len must be in [1, 4096]");
  }
  if (cfg.num_candidates <= 0 || cfg.num_candidates > kMaxNumCandidates) {
    throw std::invalid_argument("num_candidates must be in [1, 100]");
  }
  if (cfg.embedding_dim <= 0) {
    throw std::invalid_argument("embedding_dim must be > 0");
  }
}

DemoConfig parse_args(int argc, char** argv) {
  if (argc < 2) {
    throw std::invalid_argument(
        "Usage: dense_module_aoti_demo <dense_module.pt2> [model_name] [device_index] [batch_size] [history_len] [num_candidates] [embedding_dim]");
  }

  DemoConfig cfg;
  cfg.package_path = argv[1];
  if (argc > 2) {
    cfg.model_name = argv[2];
  }
  if (argc > 3) {
    cfg.device_index = std::stoi(argv[3]);
  }
  if (argc > 4) {
    cfg.batch_size = std::stoi(argv[4]);
  }
  if (argc > 5) {
    cfg.history_len = std::stoi(argv[5]);
  }
  if (argc > 6) {
    cfg.num_candidates = std::stoi(argv[6]);
  }
  if (argc > 7) {
    cfg.embedding_dim = std::stoi(argv[7]);
  }

  validate_config(cfg);
  return cfg;
}

at::Tensor make_random_id_values(
    int64_t total_values,
    int64_t vocab_size,
    const at::TensorOptions& options) {
  return torch::randint(/*high=*/vocab_size, {total_values}, options);
}

at::Tensor make_uniform_lengths(
    int64_t batch_size,
    int64_t seq_len,
    const at::TensorOptions& options) {
  return torch::full({batch_size}, seq_len, options);
}

std::vector<at::Tensor> build_inputs(const DemoConfig& cfg, const c10::Device& device) {
  const auto long_options = torch::TensorOptions().device(device).dtype(torch::kInt64);
  const auto embedding_options = torch::TensorOptions().device(device).dtype(torch::kFloat32);

  std::vector<at::Tensor> inputs;
  inputs.reserve(3 + static_cast<int>(kEmbeddingFlattenOrder.size()) * 2);

  // Flattened `batch.features.values()` and `batch.features.lengths()`.
  std::vector<at::Tensor> batch_feature_lengths;
  batch_feature_lengths.reserve(kBatchFeatureOrder.size());

  for (const auto& feature_name : kBatchFeatureOrder) {
    int64_t seq_len = kContextualFeatureLen;
    if (feature_name == "video_id") {
      seq_len = cfg.history_len + cfg.num_candidates;
    } else if (feature_name == "action_weights") {
      seq_len = cfg.history_len;
    }
    batch_feature_lengths.push_back(
        make_uniform_lengths(cfg.batch_size, seq_len, long_options));
  }

  auto batch_lengths = torch::cat(batch_feature_lengths, 0);
  auto batch_total_values = batch_lengths.sum().item<int64_t>();
  auto batch_values = make_random_id_values(
      batch_total_values,
      /*vocab_size=*/1000000,
      long_options);
  auto num_candidates = make_uniform_lengths(cfg.batch_size, cfg.num_candidates, long_options);

  inputs.push_back(batch_values);
  inputs.push_back(batch_lengths);
  inputs.push_back(num_candidates);

  // Flattened `embeddings` dict: each JaggedTensor contributes (values, lengths).
  for (const auto& feature_name : kEmbeddingFlattenOrder) {
    int64_t seq_len = kContextualFeatureLen;
    if (feature_name == "video_id") {
      seq_len = cfg.history_len + cfg.num_candidates;
    } else if (feature_name == "action_weights") {
      seq_len = cfg.history_len;
    }

    auto lengths = make_uniform_lengths(cfg.batch_size, seq_len, long_options);
    auto total_rows = lengths.sum().item<int64_t>();
    auto values = torch::randn({total_rows, cfg.embedding_dim}, embedding_options);
    inputs.push_back(values);
    inputs.push_back(lengths);
  }

  return inputs;
}

void print_tensor_summary(const at::Tensor& tensor, const std::string& name) {
  std::cout << name << ": dtype=" << tensor.dtype() << ", device=" << tensor.device()
            << ", sizes=" << tensor.sizes() << '\n';
}

} // namespace

int main(int argc, char** argv) {
  try {
    // Load FBGEMM operators before anything else
    load_fbgemm_operators();
    // Load experimental HSTU FBGEMM operators before package loader init.
    load_fbgemm_hstu_experimental_operators();
    // Load HSTU CUDA operators before anything else
    load_hstu_operators();
    
    c10::InferenceMode guard;
    DemoConfig cfg = parse_args(argc, argv);

    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for this demo.");
    TORCH_CHECK(
        cfg.device_index >= 0 && cfg.device_index < torch::cuda::device_count(),
        "Invalid CUDA device index: ",
        cfg.device_index);

    c10::Device device(torch::kCUDA, cfg.device_index);
    c10::cuda::CUDAGuard device_guard(device);

    std::cout << "Loading package: " << cfg.package_path << '\n';
    torch::inductor::AOTIModelPackageLoader loader(
        cfg.package_path,
        cfg.model_name,
        /*run_single_threaded=*/false,
        /*num_runners=*/1,
        cfg.device_index);

    if (cfg.print_call_spec) {
      auto call_spec = loader.get_call_spec();
      std::cout << "Input call spec:\n" << call_spec[0] << "\n\n";
      std::cout << "Output call spec:\n" << call_spec[1] << "\n\n";
    }

    auto metadata = loader.get_metadata();
    std::cout << "Metadata entries: " << metadata.size() << '\n';
    for (const auto& [key, value] : metadata) {
      std::cout << "  " << key << " = " << value << '\n';
    }
    std::cout << '\n';

    auto inputs = build_inputs(cfg, device);
    std::cout << "Prepared " << inputs.size() << " flattened tensor inputs." << '\n';
    for (size_t i = 0; i < inputs.size(); ++i) {
      print_tensor_summary(inputs[i], "input[" + std::to_string(i) + "]");
    }
    std::cout << '\n';

    auto outputs = loader.run(inputs);
    std::cout << "Model returned " << outputs.size() << " output tensor(s)." << '\n';
    for (size_t i = 0; i < outputs.size(); ++i) {
      print_tensor_summary(outputs[i], "output[" + std::to_string(i) + "]");
      std::cout << "output[" << i << "] sample:\n" << outputs[i].slice(/*dim=*/0, /*start=*/0, /*end=*/std::min<int64_t>(4, outputs[i].size(0))) << "\n\n";
    }

    return 0;
  } catch (const c10::Error& e) {
    std::cerr << "PyTorch error: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
