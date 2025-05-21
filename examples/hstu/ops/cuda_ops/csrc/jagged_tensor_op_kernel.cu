#include <ATen/Functions.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include "../include/utils.h"
constexpr int kMaxNumTensors = 32;
template <typename T>
struct InputJaggedTensor {
	T* value_list[kMaxNumTensors];
	int32_t* offsets_list[kMaxNumTensors];
};


template <typename T>
__global__ void concat_2D_jagged_tensors_forward_kernel(
	const InputJaggedTensor<T> input_jagged_tensor,
	const int32_t num_tensors,
	const int32_t num_rows,
	const int32_t hidden_dim,
	T* merged_values,
    int* merged_offsets) {
	
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= num_rows) return;
	int out_idx = merged_offsets[row];

	for (int t = 0; t < num_tensors; ++t) {
		const T* values = input_jagged_tensor.value_list[t];
		const int32_t* offsets = input_jagged_tensor.offsets_list[t];
		int start = offsets[row];
		int end = offsets[row + 1];

		for (int i = start; i < end; ++i) {
			for (int h = 0; h < hidden_dim; ++h) {
				merged_values[out_idx * hidden_dim + h] = values[i * hidden_dim + h];
			}
			out_idx++;
		} 
	}
}

__global__ void concat_1D_jagged_tensor_kernel(
	const float** values_list,
	const int** offsets_list,
	int num_tensor,
	int num_rows,//total_length
	float* merged_values,
	int* merged_offsets){
	
    int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= num_rows) return;

	int out_idx = merged_offsets[row]; // data start from this row
	for(int i = 0; i < num_tensor; i++){
		const float* values = values_list[i];
		const int* offsets = offsets_list[i];
		int st = offsets[row];
		int end = offsets[row+1];
		for(int j = st; j < end; j++){
			merged_values[out_idx++] = values[j];
		}
	}
}

void concat_2D_jagged_tensors_cuda_forward (
    const std::vector<torch::Tensor>& values_list,
    const std::vector<torch::Tensor>& offsets_list,
    torch::Tensor merged_values,
    torch::Tensor merged_offsets){

    int num_tensors = values_list.size();
    if (num_tensors == 0) {
        return; 
    }
    int num_rows = offsets_list[0].size(0) - 1;
    int hidden_dim = values_list[0].size(-1);

    int threads = 128;
    int blocks = (num_rows + threads - 1) / threads;

    assert(merged_values.is_contiguous());

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // printf("values_list[0].scalar_type() = %d\n", values_list[0].scalar_type());
    DISPATCH_KERNEL_BY_TYPE(
        values_list[0].scalar_type(), 
        "concat_2D_jagged_tensors_forward_kernel",
        ([&] {
            InputJaggedTensor<scalar_t> input_jagged_tensor_typed;
            for (int i = 0; i < num_tensors; ++i) {
                TORCH_CHECK(i < kMaxNumTensors, "Number of tensors exceeds kMaxNumTensors");
                input_jagged_tensor_typed.value_list[i] = values_list[i].data_ptr<scalar_t>();
                input_jagged_tensor_typed.offsets_list[i] = offsets_list[i].data_ptr<int32_t>();
            }

            concat_2D_jagged_tensors_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                input_jagged_tensor_typed,
                num_tensors,
                num_rows,
                hidden_dim,
                merged_values.data_ptr<scalar_t>(),
                merged_offsets.data_ptr<int>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        })
    );

    return; 
}

template <typename T>
__global__ void concat_2D_jagged_tensors_backward_kernel(
    const InputJaggedTensor<T> grad_jagged_tensor,
    const int32_t num_tensors,
    const int32_t num_rows,
    const int32_t hidden_dim,
    const T* grad_output,
    int* merged_offsets) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    int out_idx = merged_offsets[row];

    for (int t = 0; t < num_tensors; ++t) {
        T* grad_values = grad_jagged_tensor.value_list[t];
        const int32_t* offsets = grad_jagged_tensor.offsets_list[t];
        int start = offsets[row];
        int end = offsets[row + 1];
        for (int i = start; i < end; ++i) {
            for (int h = 0; h < hidden_dim; ++h) {
                grad_values[i * hidden_dim + h] = grad_output[out_idx * hidden_dim + h];
            }
            out_idx++;
        }
    }
}

std::vector<torch::Tensor> concat_2D_jagged_tensors_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_lengths,
    const std::vector<torch::Tensor>& offsets_list,
    torch::Tensor merged_offsets) {

    int num_tensors = offsets_list.size();
    int num_rows = grad_lengths.size(0);
    int hidden_dim = grad_output.size(-1);

    std::vector<torch::Tensor> grad_inputs(num_tensors);
    for (int i = 0; i < num_tensors; ++i) {
        int tensor_size = offsets_list[i].index({offsets_list[i].size(0) - 1}).item<int>();
        grad_inputs[i] = torch::empty(
            {tensor_size, hidden_dim},
            grad_output.options()
        );
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int threads = 128;
    int blocks = (num_rows + threads - 1) / threads;


    DISPATCH_KERNEL_BY_TYPE(
        grad_output.scalar_type(), 
        "concat_2D_jagged_tensors_backward_kernel",
        ([&] {
            InputJaggedTensor<scalar_t> grad_jagged_tensor;
            for (int i = 0; i < num_tensors; ++i) {
                TORCH_CHECK(i < kMaxNumTensors, "Number of tensors exceeds kMaxNumTensors");
                grad_jagged_tensor.value_list[i] = grad_inputs[i].data_ptr<scalar_t>();
                grad_jagged_tensor.offsets_list[i] = offsets_list[i].data_ptr<int32_t>();
            }

            concat_2D_jagged_tensors_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                grad_jagged_tensor,
                num_tensors,
                num_rows,
                hidden_dim,
                grad_output.data_ptr<scalar_t>(),
                merged_offsets.data_ptr<int>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        })
    );
    return grad_inputs;
}
