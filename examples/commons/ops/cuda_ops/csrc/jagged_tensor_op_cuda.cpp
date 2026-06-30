#ifdef WITH_PYBIND11
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#endif
#include <vector>
#include <tuple>
#include <torch/library.h>
#include <ATen/ATen.h>

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

void concat_2D_jagged_tensors_fwd_exportable(
    const std::vector<at::Tensor>& values_list,
    const std::vector<at::Tensor>& offsets_list,
    int64_t seqlen_per_block,
    int64_t max_seqlen,
    const at::Tensor& total_blocks,
    const at::Tensor& blocks,
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
        (int)total_blocks.item<int>(),
        (int)blocks.item<int>(),
        (int)threads,
        workload_offset,
        merged_values,
        merged_offsets);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_inference_preprocess(
    const at::Tensor& item_values,
    const at::Tensor& item_lengths,
    const at::Tensor& action_values,
    const at::Tensor& action_lengths,
    const at::Tensor& num_candidates) {
    TORCH_CHECK(item_values.is_cuda(), "item_values must be a CUDA tensor");
    TORCH_CHECK(item_lengths.is_cuda(), "item_lengths must be a CUDA tensor");
    TORCH_CHECK(action_values.is_cuda(), "action_values must be a CUDA tensor");
    TORCH_CHECK(action_lengths.is_cuda(), "action_lengths must be a CUDA tensor");
    TORCH_CHECK(num_candidates.is_cuda(), "num_candidates must be a CUDA tensor");
    TORCH_CHECK(item_values.dim() == 2, "item_values must be 2D");
    TORCH_CHECK(action_values.dim() == 2, "action_values must be 2D");
    TORCH_CHECK(action_lengths.dim() == 1, "action_lengths must be 1D");
    TORCH_CHECK(item_lengths.dim() == 1, "item_lengths must be 1D");
    TORCH_CHECK(num_candidates.dim() == 1, "num_candidates must be 1D");
    TORCH_CHECK(item_values.size(1) == action_values.size(1), "item/action embedding dims must match");
    TORCH_CHECK(item_lengths.size(0) == action_lengths.size(0), "item/action length batch sizes must match");
    TORCH_CHECK(item_lengths.size(0) == num_candidates.size(0), "num_candidates batch size must match");

    const auto item_lengths_i64 = item_lengths.to(at::kLong);
    const auto action_lengths_i64 = action_lengths.to(at::kLong);
    const auto num_candidates_i64 = num_candidates.to(at::kLong);
    const auto history_lengths_i64 = item_lengths_i64 - num_candidates_i64;
    const auto action_history_lengths_i64 = action_lengths_i64 - num_candidates_i64;
    const auto extra_action_lengths_i64 = action_lengths_i64 - item_lengths_i64;
    TORCH_CHECK(
        at::all(history_lengths_i64 >= 0).item<bool>(),
        "item history lengths must be non-negative after removing candidates");
    TORCH_CHECK(
        at::all(action_history_lengths_i64 >= 0).item<bool>(),
        "action history lengths must be non-negative after removing candidates");
    TORCH_CHECK(
        at::all((extra_action_lengths_i64 == 0) | (extra_action_lengths_i64 == 1)).item<bool>(),
        "each action length must equal item length or item length + 1");
    const auto output_lengths_i64 = history_lengths_i64 + action_history_lengths_i64 + num_candidates_i64;
    const auto output_lengths = output_lengths_i64; //.to(at::kInt);

    const auto zero_i64 = at::zeros({1}, item_lengths_i64.options());
    const auto item_offsets_i64 = at::cat({zero_i64, at::cumsum(item_lengths_i64, 0)}, 0);
    const auto action_offsets_i64 = at::cat({zero_i64, at::cumsum(action_lengths_i64, 0)}, 0);
    const auto output_offsets_i64 = at::cat({zero_i64, at::cumsum(output_lengths_i64, 0)}, 0);
    const auto candidate_offsets_i64 = at::cat({zero_i64, at::cumsum(num_candidates_i64, 0)}, 0);
    const auto output_offsets = output_offsets_i64; //.to(at::kInt);
    const auto candidate_offsets = candidate_offsets_i64.to(at::kInt);

    const int64_t output_num_tokens = output_lengths_i64.sum().item<int64_t>();
    auto output_values = at::empty(
        {output_num_tokens, item_values.size(1)},
        item_values.options());

    const auto item_offsets_cpu = item_offsets_i64.cpu();
    const auto action_offsets_cpu = action_offsets_i64.cpu();
    const auto output_offsets_cpu = output_offsets_i64.cpu();
    const auto history_lengths_cpu = history_lengths_i64.cpu();
    const auto action_history_lengths_cpu = action_history_lengths_i64.cpu();
    const auto extra_action_lengths_cpu = extra_action_lengths_i64.cpu();
    const auto num_candidates_cpu = num_candidates_i64.cpu();

    const auto* item_offsets_ptr = item_offsets_cpu.data_ptr<int64_t>();
    const auto* action_offsets_ptr = action_offsets_cpu.data_ptr<int64_t>();
    const auto* output_offsets_ptr = output_offsets_cpu.data_ptr<int64_t>();
    const auto* history_lengths_ptr = history_lengths_cpu.data_ptr<int64_t>();
    const auto* action_history_lengths_ptr = action_history_lengths_cpu.data_ptr<int64_t>();
    const auto* extra_action_lengths_ptr = extra_action_lengths_cpu.data_ptr<int64_t>();
    const auto* num_candidates_ptr = num_candidates_cpu.data_ptr<int64_t>();
    const int64_t batch_size = item_lengths.size(0);
    const int64_t embedding_dim = item_values.size(1);

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int64_t item_start = item_offsets_ptr[batch_idx];
        const int64_t action_start = action_offsets_ptr[batch_idx];
        const int64_t history_len = history_lengths_ptr[batch_idx];
        const int64_t action_history_len = action_history_lengths_ptr[batch_idx];
        const int64_t extra_action_len = extra_action_lengths_ptr[batch_idx];
        const int64_t num_candidate = num_candidates_ptr[batch_idx];
        int64_t dst_start = output_offsets_ptr[batch_idx];

        if (history_len > 0 || action_history_len > 0) {
            TORCH_CHECK(
                action_history_len == history_len || action_history_len == history_len + 1,
                "action history length must equal item history length or item history length + 1");
            auto history_item = item_values.narrow(0, item_start, history_len);
            auto history_action = action_values.narrow(0, action_start, action_history_len);
            if (extra_action_len == 0) {
                auto interleaved_history = at::cat({history_item, history_action}, 1).view({history_len * 2, embedding_dim});
                output_values.narrow(0, dst_start, history_len * 2).copy_(interleaved_history);
                dst_start += history_len * 2;
            } else {
                output_values.narrow(0, dst_start, 1).copy_(history_action.narrow(0, 0, 1));
                dst_start += 1;
                for (int64_t history_idx = 0; history_idx < history_len; ++history_idx) {
                    output_values.narrow(0, dst_start, 1).copy_(history_item.narrow(0, history_idx, 1));
                    output_values.narrow(0, dst_start + 1, 1).copy_(history_action.narrow(0, history_idx + 1, 1));
                    dst_start += 2;
                }
            }
        }
        if (num_candidate > 0) {
            output_values.narrow(0, dst_start, num_candidate).copy_(
                item_values.narrow(0, item_start + history_len, num_candidate));
        }
    }

    return std::make_tuple(output_values, output_lengths, output_offsets, candidate_offsets);
}

TORCH_LIBRARY_FRAGMENT(hstu_cuda_ops, m) {
    m.def("concat_2D_jagged_tensors_forward(Tensor[] values_list, Tensor[] offsets_list, int seqlen_per_block, int max_seqlen, int total_blocks, int blocks, int threads, Tensor workload_offset, Tensor(a!) merged_values, Tensor(b!) merged_offsets) -> ()");
    m.def("concat_2D_jagged_tensors_backward(Tensor grad_output, Tensor grad_lengths, int seqlen_per_block, int max_seqlen, int total_blocks, int blocks, int threads, Tensor workload_offset, Tensor(a!)[] grad_inputs, Tensor[] offsets_list, Tensor merged_offsets) -> ()");
    m.def("compute_block_workloads(Tensor[] offsets_list, int seqlen_per_block, int max_seqlen, Tensor(a!) block_workloads) -> ()");
    m.def("concat_2D_jagged_tensors_fwd_exportable(Tensor[] values_list, Tensor[] offsets_list, int seqlen_per_block, int max_seqlen, Tensor total_blocks, Tensor blocks, int threads, Tensor workload_offset, Tensor(a!) merged_values, Tensor(b!) merged_offsets) -> ()");
    m.def("hstu_inference_preprocess(Tensor item_values, Tensor item_lengths, Tensor action_values, Tensor action_lengths, Tensor num_candidates) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(hstu_cuda_ops, CUDA, m) {
    m.impl("concat_2D_jagged_tensors_forward", &concat_2D_jagged_tensors_forward);
    m.impl("concat_2D_jagged_tensors_backward", &concat_2D_jagged_tensors_backward);
    m.impl("compute_block_workloads", &compute_block_workloads);
    m.impl("concat_2D_jagged_tensors_fwd_exportable", &concat_2D_jagged_tensors_fwd_exportable);
    m.impl("hstu_inference_preprocess", &hstu_inference_preprocess);
}

// Keep a minimal pybind11 module so `import hstu_cuda_ops` continues to work
// as the mechanism to load this shared library and trigger TORCH_LIBRARY registration.
#ifdef WITH_PYBIND11
PYBIND11_MODULE(hstu_cuda_ops, m) {}
#endif
