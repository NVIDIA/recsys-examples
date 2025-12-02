#include <ATen/Functions.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <cuda_bf16.h>
constexpr int kMaxNumTensors = 128;
template <typename T>
struct InputJaggedTensor {
	T* value_list[kMaxNumTensors];
	int64_t* offsets_list[kMaxNumTensors];
};

// float4 vectorized copy function
template<typename T>
__device__ __forceinline__ void copy_float4(T* dst, const T* src) {
    *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
}

template<typename T>
__device__ __forceinline__ void copy_float2(T* dst, const T* src) {
    *reinterpret_cast<float2*>(dst) = *reinterpret_cast<const float2*>(src);
}

template<typename T>
__device__ __forceinline__ void copy(T* dst, const T* src) {
    *dst = *src;
}

template <typename T, auto copy_func, int vector_length>
__launch_bounds__(1024, 2) __global__ void concat_2D_jagged_tensors_forward_kernel(
    InputJaggedTensor<T> input_jagged_tensor,
    int32_t num_tensors,
    int32_t batch_size,
    int32_t hidden_dim,
    int32_t max_seqlen,
    int32_t total_blocks,
    int32_t seqlen_per_block,
    int64_t* workload_offset,
    T* merged_values,
    int64_t* merged_offsets
    ) {
    // each block processes workload_offset[i+1]-workload_offset[i] sequences
    for(int block_id = blockIdx.x; block_id < total_blocks; block_id += gridDim.x){
        if(workload_offset[block_id] == workload_offset[block_id + 1]) continue;
        
        int num_bucket_per_batch = (max_seqlen + seqlen_per_block - 1) / seqlen_per_block;
        int tensor_id = (block_id / num_bucket_per_batch) % num_tensors;
        int batch_id =  block_id / num_bucket_per_batch / num_tensors; 
        int idx = block_id % num_bucket_per_batch; // which bucket
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        const int64_t* offsets = input_jagged_tensor.offsets_list[tensor_id];
        const T* values = input_jagged_tensor.value_list[tensor_id];
        int seq_start = offsets[batch_id];
        
        // for merged_values, each block processes workload_offset[block_id+1]-workload_offset[block_id] sequences starting from workload_offset[block_id] 
        // for source_values, copy len sequences from input_jagged_tensor.value_list[tensor_id] starting from seq_start

        for(int seq_offset = warp_id; seq_offset < workload_offset[block_id+1]-workload_offset[block_id]; seq_offset += 32){
            int src_row = seq_start + seq_offset + idx*seqlen_per_block; // add bucket offset
            int dst_row = workload_offset[block_id] + seq_offset;
            int vec_count = hidden_dim / vector_length;
            for (int i = lane_id; i < vec_count; i += 32) {
                int h = i * vector_length;
                copy_func(&merged_values[dst_row * hidden_dim + h], &values[src_row * hidden_dim + h]);
            }
        }
    }
}

void concat_2D_jagged_tensors_cuda_forward (
    const std::vector<torch::Tensor>& values_list,
    const std::vector<torch::Tensor>& offsets_list,
    int seqlen_per_block,
    int max_seqlen,
    int total_blocks,
    int blocks,
    int threads,
    torch::Tensor workload_offset,
    torch::Tensor merged_values,
    torch::Tensor merged_offsets){

    int num_tensors = values_list.size();
    if (num_tensors == 0) {
        return; 
    }

    int hidden_dim = values_list[0].size(-1);

    int batch_size = offsets_list[0].size(0) - 1;


    assert(merged_values.is_contiguous());

    int device_id;
    cudaGetDevice(&device_id);
    c10::cuda::CUDAGuard guard(device_id);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        values_list[0].scalar_type(), 
        "concat_2D_jagged_tensors_forward_kernel",
        [&] {
            InputJaggedTensor<scalar_t> input_jagged_tensor_typed;
            for (int i = 0; i < num_tensors; ++i) {
                TORCH_CHECK(i < kMaxNumTensors, "Number of tensors exceeds kMaxNumTensors");
                input_jagged_tensor_typed.value_list[i] = values_list[i].data_ptr<scalar_t>();
                input_jagged_tensor_typed.offsets_list[i] = offsets_list[i].data_ptr<int64_t>();
            }

            dim3 opt_blocks(blocks);
            dim3 opt_threads(threads);
            void* kernel_func_ptr;
            if (std::is_same_v<scalar_t, __half> || std::is_same_v<scalar_t, __nv_bfloat16>) {
                if (hidden_dim % 8 == 0) {  
                    kernel_func_ptr = (void*)&concat_2D_jagged_tensors_forward_kernel<scalar_t, copy_float4<scalar_t>, 8>;
                } else if(hidden_dim % 4 == 0) {
                    kernel_func_ptr = (void*)&concat_2D_jagged_tensors_forward_kernel<scalar_t, copy_float2<scalar_t>, 4>;
                } else {
                    kernel_func_ptr = (void*)&concat_2D_jagged_tensors_forward_kernel<scalar_t, copy<scalar_t>, 1>;
                }
            } else if (std::is_same_v<scalar_t, float>){
                if (hidden_dim % 4 == 0) {
                    kernel_func_ptr = (void*)&concat_2D_jagged_tensors_forward_kernel<scalar_t, copy_float4<scalar_t>, 4>;
                } else {
                    kernel_func_ptr = (void*)&concat_2D_jagged_tensors_forward_kernel<scalar_t, copy<scalar_t>, 1>;
                }
            } else {
                kernel_func_ptr = reinterpret_cast<void*>(&concat_2D_jagged_tensors_forward_kernel<scalar_t, copy<scalar_t>, 1>);
            }
            
            auto workload_ptr = workload_offset.data_ptr<int64_t>();
            auto merged_values_ptr = merged_values.data_ptr<scalar_t>();
            auto merged_offsets_ptr = merged_offsets.data_ptr<int64_t>();
            
            void* args[] = {
                &input_jagged_tensor_typed,
                &num_tensors,
                &batch_size,
                &hidden_dim,
                &max_seqlen,
                &total_blocks,
                &seqlen_per_block,
                &workload_ptr,
                &merged_values_ptr,
                &merged_offsets_ptr
            };
            
            cudaLaunchKernel((void*)kernel_func_ptr, opt_blocks, opt_threads, args, 0, stream);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    return; 
}

template <typename T, auto copy_func, int vector_length>
__launch_bounds__(1024, 2) __global__ void concat_2D_jagged_tensors_backward_kernel(
    const InputJaggedTensor<T> grad_jagged_tensor,
    const int32_t num_tensors,
    const int32_t batch_size,
    const int32_t hidden_dim,
    const int32_t max_seqlen,
    const int32_t total_blocks,
    const int32_t seqlen_per_block,
    int64_t* workload_offset,
    T* grad_output,
    int64_t* merged_offsets) {
    // each block processes workload_offset[i+1]-workload_offset[i] sequences
    for(int block_id = blockIdx.x; block_id < total_blocks; block_id += gridDim.x){
        if(workload_offset[block_id] == workload_offset[block_id + 1]) continue;
        
        int num_bucket_per_batch = (max_seqlen + seqlen_per_block - 1) / seqlen_per_block;
        int tensor_id = (block_id / num_bucket_per_batch) % num_tensors;
        int batch_id =  block_id / num_bucket_per_batch / num_tensors; 
        int idx = block_id % num_bucket_per_batch; // which bucket
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        const int64_t* offsets = grad_jagged_tensor.offsets_list[tensor_id];
        T* values = grad_jagged_tensor.value_list[tensor_id];
        int seq_start = offsets[batch_id];
        
        // for backward pass: copy gradients from merged grad_output back to individual input gradients
        // dst_row in grad_output -> src_row in values (individual tensor gradients)

        for(int seq_offset = warp_id; seq_offset < workload_offset[block_id+1]-workload_offset[block_id]; seq_offset += 32){
            int src_row = seq_start + seq_offset + idx*seqlen_per_block; // add bucket offset
            int dst_row = workload_offset[block_id] + seq_offset;
            int vec_count = hidden_dim / vector_length;
            for (int i = lane_id; i < vec_count; i += 32) {
                int h = i * vector_length;
                copy_func(&values[src_row * hidden_dim + h], &grad_output[dst_row * hidden_dim + h]);
            }
        }
    }
}
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
    torch::Tensor merged_offsets) {

    int num_tensors = offsets_list.size();
    int batch_size = grad_lengths.size(0);
    int hidden_dim = grad_output.size(-1);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    int device_id;
    cudaGetDevice(&device_id);
    c10::cuda::CUDAGuard guard(device_id);
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        grad_output.scalar_type(), 
        "concat_2D_jagged_tensors_backward_kernel",
        [&] {
            InputJaggedTensor<scalar_t> grad_jagged_tensor;
            for (int i = 0; i < num_tensors; ++i) {
                TORCH_CHECK(i < kMaxNumTensors, "Number of tensors exceeds kMaxNumTensors");
                grad_jagged_tensor.value_list[i] = grad_inputs[i].data_ptr<scalar_t>();
                grad_jagged_tensor.offsets_list[i] = offsets_list[i].data_ptr<int64_t>();
            }
            
            dim3 opt_blocks(blocks);
            dim3 opt_threads(threads);
            void* kernel_func_ptr;
            if (std::is_same_v<scalar_t, __half> || std::is_same_v<scalar_t, __nv_bfloat16>) {
                if (hidden_dim % 8 == 0) {  
                    kernel_func_ptr = (void*)&concat_2D_jagged_tensors_backward_kernel<scalar_t, copy_float4<scalar_t>, 8>;
                } else if(hidden_dim % 4 == 0) {
                    kernel_func_ptr = (void*)&concat_2D_jagged_tensors_backward_kernel<scalar_t, copy_float2<scalar_t>, 4>;
                } else {
                    kernel_func_ptr = (void*)&concat_2D_jagged_tensors_backward_kernel<scalar_t, copy<scalar_t>, 1>;
                }
            } else if (std::is_same_v<scalar_t, float>){
                if (hidden_dim % 4 == 0) {
                    kernel_func_ptr = (void*)&concat_2D_jagged_tensors_backward_kernel<scalar_t, copy_float4<scalar_t>, 4>;
                } else {
                    kernel_func_ptr = (void*)&concat_2D_jagged_tensors_backward_kernel<scalar_t, copy<scalar_t>, 1>;
                }
            } else {
                kernel_func_ptr = reinterpret_cast<void*>(&concat_2D_jagged_tensors_backward_kernel<scalar_t, copy<scalar_t>, 1>);
            }
            
            auto workload_ptr = workload_offset.data_ptr<int64_t>();
            auto grad_output_ptr = grad_output.data_ptr<scalar_t>();
            auto merged_offsets_ptr = merged_offsets.data_ptr<int64_t>();
            
            void* args[] = {
                &grad_jagged_tensor,
                &num_tensors,
                &batch_size,
                &hidden_dim,
                &max_seqlen,
                &total_blocks,
                &seqlen_per_block,
                &workload_ptr,
                &grad_output_ptr,
                &merged_offsets_ptr
            };
            
            cudaLaunchKernel((void*)kernel_func_ptr, opt_blocks, opt_threads, args, 0, stream);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
    return;
}

__global__ void compute_block_workloads_cuda_kernel(
    const InputJaggedTensor<int64_t> input_jagged_tensor,
    const int32_t num_tensors,
    const int32_t batch_size,
    const int32_t seqlen_per_block,
    const int32_t max_seqlen,
    const int32_t total_blocks,
    int64_t* block_workloads) {
    
    int work_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_id >= total_blocks) return;
    
    int num_bucket_per_batch = (max_seqlen + seqlen_per_block - 1) / seqlen_per_block;

    int tensor_id = (work_id / num_bucket_per_batch) % num_tensors;
    int batch_id = work_id / num_bucket_per_batch / num_tensors; 
    int idx = work_id % num_bucket_per_batch; // which bucket
    const int64_t* offsets = input_jagged_tensor.offsets_list[tensor_id];
    int seq_start = offsets[batch_id];
    int seq_end = offsets[batch_id + 1];
    int len = seq_end - seq_start;

    int remaining_len = len - idx * seqlen_per_block;
    block_workloads[work_id] = max(0, min(remaining_len, seqlen_per_block));

    return;
}

/*
target: determine how many rows each block should process from merged_offsets
e.g. [4, 1, 4, 4, 4, 0, 4, 1, ...]

Working principle:
1. Each block is responsible for one bucket (sequence segment of seqlen_per_block size) of a batch
2. Calculate the total number of sequences for all tensors in that batch
3. Calculate the actual workload for that block based on bucket position
4. Output to block_workloads array for subsequent prefix sum calculation
*/
void compute_block_workloads_cuda(
    const std::vector<torch::Tensor>& offsets_list,
    int seqlen_per_block,
    int max_seqlen,
    torch::Tensor block_workloads) {

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int num_tensors = offsets_list.size();
    int batch_size = offsets_list[0].size(0) - 1;
    int blocks_per_batch = (max_seqlen + seqlen_per_block - 1) / seqlen_per_block;
    int total_blocks = batch_size * blocks_per_batch * num_tensors;

    InputJaggedTensor<int64_t> offsets_jagged_tensor;
    for (int i = 0; i < num_tensors; ++i) {
        TORCH_CHECK(i < kMaxNumTensors, "Number of tensors exceeds kMaxNumTensors");
        offsets_jagged_tensor.offsets_list[i] = offsets_list[i].data_ptr<int64_t>();
    }
    int threads_per_block = min(1024, total_blocks);
    int num_blocks = (total_blocks + threads_per_block - 1) / threads_per_block;
    dim3 blocks(num_blocks);
    dim3 threads(threads_per_block);
    
    compute_block_workloads_cuda_kernel<<<blocks, threads, 0, stream>>>(
        offsets_jagged_tensor,
        num_tensors,
        batch_size,
        seqlen_per_block,
        max_seqlen,
        total_blocks,
        block_workloads.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
}