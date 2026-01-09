#include <pybind11/pybind11.h>
#include <vector>
#include <torch/extension.h>
#include <ATen/ATen.h>
void concat_2D_jagged_tensors_cuda_forward (
    const std::vector<at::Tensor>& values_list,
    const std::vector<at::Tensor>& offsets_list,
    int seqlen_per_block,
    int max_seqlen,
    int total_blocks,
    int blocks,
    int threads,
    at::Tensor workload_offset,
    at::Tensor merged_values,
    at::Tensor merged_offsets);

void concat_2D_jagged_tensors_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_lengths,
    int seqlen_per_block,
    int max_seqlen,
    int total_blocks,
    int blocks,
    int threads,
    torch::Tensor workload_offset,
    const std::vector<torch::Tensor>& grad_inputs,
    const std::vector<torch::Tensor>& offsets_list,
    torch::Tensor merged_offsets);

void compute_block_workloads_cuda(
    const std::vector<at::Tensor>& offsets_list,
    int seqlen_per_block,
    int max_seqlen,
    at::Tensor block_workloads);

void concat_2D_jagged_tensors_forward (
    const std::vector<at::Tensor>& values_list,
    const std::vector<at::Tensor>& offsets_list,
    int seqlen_per_block,
    int max_seqlen,
    int total_blocks,
    int blocks,
    int threads,
    at::Tensor workload_offset,
    at::Tensor merged_values,
    at::Tensor merged_offsets) {

    assert(merged_values.defined());
    assert(merged_values.dtype() == values_list[0].dtype());

    concat_2D_jagged_tensors_cuda_forward(
        values_list, 
        offsets_list, 
        seqlen_per_block,
        max_seqlen,
        total_blocks,
        blocks,
        threads,
        workload_offset,
        merged_values, 
        merged_offsets);
    return;
}

void concat_2D_jagged_tensors_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_lengths,
    int seqlen_per_block,
    int max_seqlen,
    int total_blocks,
    int blocks,
    int threads,
    torch::Tensor workload_offset,
    const std::vector<torch::Tensor>& grad_inputs,
    const std::vector<torch::Tensor>& offsets_list,
    torch::Tensor merged_offsets) {
    concat_2D_jagged_tensors_cuda_backward(
        grad_output, 
        grad_lengths,
        seqlen_per_block,
        max_seqlen,
        total_blocks,
        blocks,
        threads,
        workload_offset,
        grad_inputs,
        offsets_list,
        merged_offsets);
}

void compute_block_workloads(
    const std::vector<at::Tensor>& offsets_list,
    int seqlen_per_block,
    int max_seqlen,
    at::Tensor block_workloads) {
    compute_block_workloads_cuda(offsets_list, seqlen_per_block, max_seqlen, block_workloads);
    return;
}
PYBIND11_MODULE(hstu_cuda_ops, m) {
  m.def("concat_2D_jagged_tensors_forward", &concat_2D_jagged_tensors_forward, "JaggedTensor concat forward (CUDA)");
  m.def("concat_2D_jagged_tensors_backward", &concat_2D_jagged_tensors_backward, "JaggedTensor concat backward (CUDA)");
  m.def("compute_block_workloads", &compute_block_workloads, "Compute block workloads (CUDA)");
}
