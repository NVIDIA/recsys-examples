#include <iostream>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

// struct KeyedJaggedTensor {
//     std::vector<std::string> keys;
//     torch::Tensor values;
//     torch::Tensor lengths;
// };

// KeyedJaggedTensor create_example_sparse_features() {
//     auto values = torch::randint(0, 1000, {6}, torch::dtype(torch::kInt64));
//     auto lengths = torch::ones(6, torch::dtype(torch::kInt64));
//     return KeyedJaggedTensor({
//         "user_id",
//         "movie_id",
//         "gender",
//         "age",
//         "occupation",
//         "year"
//     }, values, lengths);
// }

int main() {
    c10::InferenceMode mode;

    torch::inductor::AOTIModelPackageLoader loader("model.pt2");
    // auto inputs = create_example_sparse_features();
    auto values = torch::randint(0, 1000, {6}, torch::dtype(torch::kInt64));
    auto lengths = torch::ones(6, torch::dtype(torch::kInt64));
    std::vector<torch::Tensor> inputs{values, lengths};
    // Assume running on CUDA
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