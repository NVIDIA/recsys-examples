#include <iostream>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>


int main() {
    c10::InferenceMode mode;

    torch::inductor::AOTIModelPackageLoader loader("preprocessing.pt2");

    
    torch::inductor::AOTIModelPackageLoader loader("dense.pt2");
    
    // Assume running on CUDA
    std::vector<torch::Tensor> inputs = {torch::randn({8, 10}, at::kCUDA)};
    std::vector<torch::Tensor> outputs = loader.run(inputs);
    std::cout << "Result from the first inference:"<< std::endl;
    std::cout << outputs[0] << std::endl;

    // The second inference uses a different batch size and it works because we
    // specified that dimension as dynamic when compiling model.pt2.
    std::cout << "Result from the second inference:"<< std::endl;
    // Assume running on CUDA
    std::cout << loader.run({torch::randn({1, 10}, at::kCUDA)})[0] << std::endl;

    return 0;
}