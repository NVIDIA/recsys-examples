#ifdef WITH_PYBIND11
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#endif
#include <vector>
#include <torch/library.h>
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
    at::Tensor grad_output,
    at::Tensor grad_lengths,
    int seqlen_per_block,
    int max_seqlen,
    int total_blocks,
    int blocks,
    int threads,
    at::Tensor workload_offset,
    const std::vector<at::Tensor>& grad_inputs,
    const std::vector<at::Tensor>& offsets_list,
    at::Tensor merged_offsets);

void compute_block_workloads_cuda(
    const std::vector<at::Tensor>& offsets_list,
    int seqlen_per_block,
    int max_seqlen,
    at::Tensor block_workloads);

void concat_2D_jagged_tensors_forward (
    const std::vector<at::Tensor>& values_list,
    const std::vector<at::Tensor>& offsets_list,
    int64_t seqlen_per_block,
    int64_t max_seqlen,
    int64_t total_blocks,
    int64_t blocks,
    int64_t threads,
    at::Tensor workload_offset,
    at::Tensor merged_values,
    at::Tensor merged_offsets) {

    assert(merged_values.defined());
    assert(merged_values.dtype() == values_list[0].dtype());

    concat_2D_jagged_tensors_cuda_forward(
        values_list,
        offsets_list,
        (int)seqlen_per_block,
        (int)max_seqlen,
        (int)total_blocks,
        (int)blocks,
        (int)threads,
        workload_offset,
        merged_values,
        merged_offsets);
}

void concat_2D_jagged_tensors_backward(
    at::Tensor grad_output,
    at::Tensor grad_lengths,
    int64_t seqlen_per_block,
    int64_t max_seqlen,
    int64_t total_blocks,
    int64_t blocks,
    int64_t threads,
    at::Tensor workload_offset,
    std::vector<at::Tensor> grad_inputs,
    const std::vector<at::Tensor>& offsets_list,
    at::Tensor merged_offsets) {
    concat_2D_jagged_tensors_cuda_backward(
        grad_output,
        grad_lengths,
        (int)seqlen_per_block,
        (int)max_seqlen,
        (int)total_blocks,
        (int)blocks,
        (int)threads,
        workload_offset,
        grad_inputs,
        offsets_list,
        merged_offsets);
}

void compute_block_workloads(
    const std::vector<at::Tensor>& offsets_list,
    int64_t seqlen_per_block,
    int64_t max_seqlen,
    at::Tensor block_workloads) {
    compute_block_workloads_cuda(
        offsets_list,
        (int)seqlen_per_block,
        (int)max_seqlen,
        block_workloads);
}

TORCH_LIBRARY_FRAGMENT(hstu_cuda_ops, m) {
    m.def("concat_2D_jagged_tensors_forward(Tensor[] values_list, Tensor[] offsets_list, int seqlen_per_block, int max_seqlen, int total_blocks, int blocks, int threads, Tensor workload_offset, Tensor(a!) merged_values, Tensor(b!) merged_offsets) -> ()");
    m.def("concat_2D_jagged_tensors_backward(Tensor grad_output, Tensor grad_lengths, int seqlen_per_block, int max_seqlen, int total_blocks, int blocks, int threads, Tensor workload_offset, Tensor(a!)[] grad_inputs, Tensor[] offsets_list, Tensor merged_offsets) -> ()");
    m.def("compute_block_workloads(Tensor[] offsets_list, int seqlen_per_block, int max_seqlen, Tensor(a!) block_workloads) -> ()");
}

TORCH_LIBRARY_IMPL(hstu_cuda_ops, CUDA, m) {
    m.impl("concat_2D_jagged_tensors_forward", &concat_2D_jagged_tensors_forward);
    m.impl("concat_2D_jagged_tensors_backward", &concat_2D_jagged_tensors_backward);
    m.impl("compute_block_workloads", &compute_block_workloads);
}

// Keep a minimal pybind11 module so `import hstu_cuda_ops` continues to work
// as the mechanism to load this shared library and trigger TORCH_LIBRARY registration.
#ifdef WITH_PYBIND11
PYBIND11_MODULE(hstu_cuda_ops, m) {}
#endif
