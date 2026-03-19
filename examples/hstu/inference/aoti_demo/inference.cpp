#include <dlfcn.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace {

bool load_custom_ops_library(const std::string& path) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    std::cerr << "Failed to load custom op library: " << path << std::endl;
    std::cerr << "dlerror(): " << dlerror() << std::endl;
    return false;
  }
  std::cout << "Loaded custom op library: " << path << std::endl;
  return true;
}

std::string infer_default_custom_ops_path(const char* argv0) {
  std::filesystem::path exe_path = std::filesystem::absolute(argv0);
  return (exe_path.parent_path() / "inference_emb_ops.so").string();
}

}  // namespace

int main(int argc, char** argv) {
  c10::InferenceMode mode;

  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is required for this inference demo." << std::endl;
    return 1;
  }

  const std::string package_path = (argc > 1) ? argv[1] : "model.pt2";
  const std::string custom_ops_path =
      (argc > 2) ? argv[2] : infer_default_custom_ops_path(argv[0]);

  try {
    if (!load_custom_ops_library(custom_ops_path)) {
      return 5;
    }

    torch::inductor::AOTIModelPackageLoader loader(package_path);

    auto device = torch::Device(torch::kCUDA, 0);
    auto options = torch::TensorOptions().device(device).dtype(torch::kInt64);

    // These inputs match the fixed export signature used in test_torch_export().
    torch::Tensor indices = torch::tensor({0, 1, 2, 3}, options);
    torch::Tensor offsets = torch::tensor({0, 2, 4}, options);

    std::vector<torch::Tensor> inputs{indices, offsets};
    std::vector<torch::Tensor> outputs = loader.run(inputs);

    if (outputs.empty()) {
      std::cerr << "The packaged model returned no outputs." << std::endl;
      return 2;
    }

    torch::Tensor output = outputs[0];
    torch::Tensor output_cpu = output.to(torch::kCPU);

    std::cout << "Loaded package: " << package_path << std::endl;
    std::cout << "Custom op library: " << custom_ops_path << std::endl;
    std::cout << "Input indices: " << indices.to(torch::kCPU) << std::endl;
    std::cout << "Input offsets: " << offsets.to(torch::kCPU) << std::endl;
    std::cout << "Output dtype: " << output_cpu.dtype() << std::endl;
    std::cout << "Output shape: " << output_cpu.sizes() << std::endl;
    std::cout << "First embedding row, first 8 values: "
              << output_cpu[0].slice(/*dim=*/0, /*start=*/0, /*end=*/8) << std::endl;

    return 0;
  } catch (const c10::Error& e) {
    std::cerr << "AOTInductor inference failed: " << e.what() << std::endl;
    return 3;
  } catch (const std::exception& e) {
    std::cerr << "Unexpected error: " << e.what() << std::endl;
    return 4;
  }
}
