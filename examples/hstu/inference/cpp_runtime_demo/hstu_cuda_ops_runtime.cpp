#include <vector>

#include <ATen/ATen.h>
#include <torch/library.h>

void concat_2D_jagged_tensors_cuda_forward(
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

void concat_2D_jagged_tensors_forward(
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
  TORCH_CHECK(merged_values.defined(), "merged_values must be defined");
  TORCH_CHECK(!values_list.empty(), "values_list must not be empty");
  TORCH_CHECK(merged_values.dtype() == values_list[0].dtype(), "dtype mismatch");

  concat_2D_jagged_tensors_cuda_forward(
      values_list,
      offsets_list,
      static_cast<int>(seqlen_per_block),
      static_cast<int>(max_seqlen),
      static_cast<int>(total_blocks),
      static_cast<int>(blocks),
      static_cast<int>(threads),
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
      static_cast<int>(seqlen_per_block),
      static_cast<int>(max_seqlen),
      static_cast<int>(total_blocks),
      static_cast<int>(blocks),
      static_cast<int>(threads),
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
      static_cast<int>(seqlen_per_block),
      static_cast<int>(max_seqlen),
      block_workloads);
}

TORCH_LIBRARY(hstu_cuda_ops, m) {
  m.def("concat_2D_jagged_tensors_forward(Tensor[] values_list, Tensor[] offsets_list, int seqlen_per_block, int max_seqlen, int total_blocks, int blocks, int threads, Tensor workload_offset, Tensor(a!) merged_values, Tensor(b!) merged_offsets) -> ()");
  m.def("concat_2D_jagged_tensors_backward(Tensor grad_output, Tensor grad_lengths, int seqlen_per_block, int max_seqlen, int total_blocks, int blocks, int threads, Tensor workload_offset, Tensor(a!)[] grad_inputs, Tensor[] offsets_list, Tensor merged_offsets) -> ()");
  m.def("compute_block_workloads(Tensor[] offsets_list, int seqlen_per_block, int max_seqlen, Tensor(a!) block_workloads) -> ()");
}

TORCH_LIBRARY_IMPL(hstu_cuda_ops, CUDA, m) {
  m.impl("concat_2D_jagged_tensors_forward", &concat_2D_jagged_tensors_forward);
  m.impl("concat_2D_jagged_tensors_backward", &concat_2D_jagged_tensors_backward);
  m.impl("compute_block_workloads", &compute_block_workloads);
}
