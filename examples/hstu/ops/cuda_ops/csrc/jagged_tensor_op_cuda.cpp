#include <pybind11/pybind11.h>
#include <vector>
#include <torch/extension.h>
#include <ATen/ATen.h>
void concat_2D_jagged_tensors_cuda_forward (
    const std::vector<at::Tensor>& values_list,
    const std::vector<at::Tensor>& offsets_list,
    at::Tensor merged_values,
    at::Tensor merged_offsets);

std::vector<torch::Tensor> concat_2D_jagged_tensors_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_lengths,
    const std::vector<torch::Tensor>& offsets_list,
    torch::Tensor merged_offsets);

void concat_2D_jagged_tensors_forward (
    const std::vector<at::Tensor>& values_list,
    const std::vector<at::Tensor>& offsets_list,
    at::Tensor merged_values,
    at::Tensor merged_offsets) {

    assert(merged_values.defined());
    assert(merged_values.dtype() == values_list[0].dtype());

    // // printf("merged_values.dtype() in cpp = %d\n", merged_values.dtype());
    // std::cout << "merged_values.dtype() in cpp = " << merged_values.dtype() << std::endl;
    concat_2D_jagged_tensors_cuda_forward(
        values_list, 
        offsets_list, 
        merged_values, 
        merged_offsets);
    return;
}

std::vector<torch::Tensor> concat_2D_jagged_tensors_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_lengths,
    const std::vector<torch::Tensor>& offsets_list,
    torch::Tensor merged_offsets) {
    return concat_2D_jagged_tensors_cuda_backward(
        grad_output, 
        grad_lengths,
        offsets_list,
        merged_offsets);
}

PYBIND11_MODULE(hstu_cuda_ops, m) {
  m.def("concat_2D_jagged_tensors_forward", &concat_2D_jagged_tensors_forward, "JaggedTensor concat forward (CUDA)");
  m.def("concat_2D_jagged_tensors_backward", &concat_2D_jagged_tensors_backward, "JaggedTensor concat backward (CUDA)");
}
