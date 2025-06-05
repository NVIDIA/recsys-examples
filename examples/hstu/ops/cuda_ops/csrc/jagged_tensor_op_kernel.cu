#include <ATen/Functions.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include "../include/utils.h"
constexpr int kMaxNumTensors = 32;
template <typename T>
struct InputJaggedTensor {
	T* value_list[kMaxNumTensors];
	int32_t* offsets_list[kMaxNumTensors];
};

// float4 vectorized copy function
template<typename T>
__device__ __forceinline__ void copy_float4(T* dst, const T* src) {
    if (sizeof(T) == sizeof(float) && 
        reinterpret_cast<uintptr_t>(dst) % 16 == 0 && 
        reinterpret_cast<uintptr_t>(src) % 16 == 0) {
        // float4 copy
        *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
    } else if (sizeof(T) == sizeof(at::Half) && 
               reinterpret_cast<uintptr_t>(dst) % 16 == 0 && 
               reinterpret_cast<uintptr_t>(src) % 16 == 0) {
        // half8 copy (8 half = 1 float4)
        *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
    } else {
        // fallback to scalar copy
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dst[i] = src[i];
        }
    }
}

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

template <typename T>
__global__ void concat_2D_jagged_tensors_forward_kernel_opt(
	const InputJaggedTensor<T> input_jagged_tensor,
	const int32_t num_tensors,
	const int32_t num_rows,
	const int32_t hidden_dim,
	T* merged_values,
    int* merged_offsets) {
	
    // shared memory to cache offset lengths for current batch
    __shared__ int32_t shared_lens[kMaxNumTensors];
    
    int batch_id = blockIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (batch_id >= num_rows) return;
    
    // load offset lengths to shared memory
    if (threadIdx.x < num_tensors) {
        const int32_t* offsets = input_jagged_tensor.offsets_list[threadIdx.x];
        shared_lens[threadIdx.x] = offsets[batch_id + 1] - offsets[batch_id];
    }
    __syncthreads();
    
    // each warp processes one tensor
    if (warp_id < num_tensors) {
        const T* values = input_jagged_tensor.value_list[warp_id];
        const int32_t* offsets = input_jagged_tensor.offsets_list[warp_id];
        int start = offsets[batch_id];
        int end = offsets[batch_id + 1];
        int num_rows_in_tensor = end - start;
        
        // calculate output start position for current tensor
        int out_start = merged_offsets[batch_id];
        for (int t = 0; t < warp_id; ++t) {
            out_start += shared_lens[t];
        }

        //each thread handles different rows
        for (int row_offset = lane_id; row_offset < num_rows_in_tensor; row_offset += 32) {
            int i = start + row_offset;
            int out_row = out_start + row_offset;
            
            // each thread copies entire hidden_dim 
            if (hidden_dim % 4 == 0 && hidden_dim <= 256) {
                for (int h = 0; h < hidden_dim; h += 4) {
                    copy_float4(&merged_values[out_row * hidden_dim + h],
                               &values[i * hidden_dim + h]);
                }
            } else {
                for (int h = 0; h < hidden_dim; ++h) {
                    merged_values[out_row * hidden_dim + h] = values[i * hidden_dim + h];
                }
            }
        }
    }
}

template <typename T>
__global__ void concat_2D_jagged_tensors_forward_kernel_opt_v2(
	const InputJaggedTensor<T> input_jagged_tensor,
	const int32_t num_tensors,
	const int32_t num_rows,
	const int32_t hidden_dim,
    const int32_t max_seqlen,
    const int32_t seqlen_per_block,
	T* merged_values,
    int* merged_offsets) {
    // shared memory to cache offset lengths for current batch
    __shared__ int32_t shared_lens[kMaxNumTensors];
    
    int batch_id = blockIdx.x / (max_seqlen / seqlen_per_block);
    int block_id = blockIdx.x % (max_seqlen / seqlen_per_block); // range 0, max_seqlen/seqlen_per_block - 1
    //一个block处理seqlen_per_block个seq 如何划分给不同的warp？还是直接线程做？
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int seq_id = block_id * seqlen_per_block;
    //int seq_id = block_id * seqlen_per_block + threadIdx.x;不太对
    //这个batch里从这个block开始为空，直接返回
    if(block_id * seqlen_per_block > merged_offsets[batch_id+1] - merged_offsets[batch_id]) return;
    if (batch_id >= num_rows) return;

    // load offset lengths to shared memory
    if (threadIdx.x < num_tensors) {
        const int32_t* offsets = input_jagged_tensor.offsets_list[threadIdx.x];
        shared_lens[threadIdx.x] = offsets[batch_id + 1] - offsets[batch_id];
    }
    __syncthreads();
    
    // each warp processes one tensor
    if (warp_id < num_tensors) {
        const T* values = input_jagged_tensor.value_list[warp_id];
        const int32_t* offsets = input_jagged_tensor.offsets_list[warp_id];
        int start = offsets[batch_id];
        int end = offsets[batch_id + 1];
        int num_rows_in_tensor = end - start;
        
        // calculate output start position for current tensor
        int out_start = merged_offsets[batch_id];
        for (int t = 0; t < warp_id; ++t) {
            out_start += shared_lens[t];
        }

        //each thread handles different rows
        for (int row_offset = lane_id; row_offset < num_rows_in_tensor; row_offset += 32) {
            int i = start + row_offset;
            int out_row = out_start + row_offset;
            
            // each thread copies entire hidden_dim 
            if (hidden_dim % 4 == 0 && hidden_dim <= 256) {
                for (int h = 0; h < hidden_dim; h += 4) {
                    copy_float4(&merged_values[out_row * hidden_dim + h],
                               &values[i * hidden_dim + h]);
                }
            } else {
                for (int h = 0; h < hidden_dim; ++h) {
                    merged_values[out_row * hidden_dim + h] = values[i * hidden_dim + h];
                }
            }
        }
    }
}

template <typename T>
__global__ void concat_2D_jagged_tensors_forward_kernel_opt_v3(
    const InputJaggedTensor<T> input_jagged_tensor,
    const int32_t num_tensors,
    const int32_t num_rows,
    const int32_t hidden_dim,
    const int32_t max_seqlen,
    const int32_t seqlen_per_block,
    T* merged_values,
    int* merged_offsets) {
    
    // shared memory to cache offset lengths for current batch
    __shared__ int32_t shared_lens[kMaxNumTensors];
    __shared__ int32_t shared_cum_lens[kMaxNumTensors + 1];
    
    int batch_id = blockIdx.x / ((max_seqlen + seqlen_per_block - 1) / seqlen_per_block);
    int block_id = blockIdx.x % ((max_seqlen + seqlen_per_block - 1) / seqlen_per_block);
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;
    
    if (batch_id >= num_rows) return;
    
    // load offset lengths to shared memory
    if (threadIdx.x < num_tensors) {
        const int32_t* offsets = input_jagged_tensor.offsets_list[threadIdx.x];
        shared_lens[threadIdx.x] = offsets[batch_id + 1] - offsets[batch_id];
        //printf("batch_id: %d, threadIdx.x: %d, shared_lens[%d]: %d, offsets[%d]: %d, offsets[%d]: %d\n", batch_id, threadIdx.x, threadIdx.x, shared_lens[threadIdx.x], batch_id + 1, offsets[batch_id + 1], batch_id, offsets[batch_id]);
    }
    if (threadIdx.x == 0) {
        shared_cum_lens[0] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < num_tensors; ++i) {
            shared_cum_lens[i + 1] = shared_cum_lens[i] + shared_lens[i];
        }
    } 
    __syncthreads();

    int total_seqlen = shared_cum_lens[num_tensors];
    
    // calculate sequence range for this block
    int seq_start = block_id * seqlen_per_block;
    int seq_end = min(seq_start + seqlen_per_block, total_seqlen);
    
    // early return if no work
    if (seq_start >= total_seqlen) return;

    //total_seqlen print some strange value
    //printf("seq_start: %d, seq_end: %d, total_seqlen: %d\n", seq_start, seq_end, total_seqlen);


    // each warp processes different sequences in this block
    for (int seq_offset = warp_id; seq_offset < (seq_end - seq_start); seq_offset += num_warps) {
        int global_seq_idx = seq_start + seq_offset;

        // find which tensor this sequence belongs to
        int tensor_id = 0;
        while (tensor_id < num_tensors && global_seq_idx >= shared_cum_lens[tensor_id + 1]) {
            tensor_id++;
        }
        
        if (tensor_id >= num_tensors) continue;
        // calculate local sequence index within tensor
        int local_seq_idx = global_seq_idx - shared_cum_lens[tensor_id];
        
        const T* values = input_jagged_tensor.value_list[tensor_id];
        const int32_t* offsets = input_jagged_tensor.offsets_list[tensor_id];
        
        int tensor_start = offsets[batch_id];
        int tensor_end = offsets[batch_id + 1];
        //printf("tensor_start: %d, tensor_end: %d, local_seq_idx: %d\n", tensor_start, tensor_end, local_seq_idx);
        if (local_seq_idx < (tensor_end - tensor_start)) {
            int src_row = tensor_start + local_seq_idx;
            int dst_row = merged_offsets[batch_id] + global_seq_idx;
            //printf("src_row: %d, dst_row: %d  merged_offsets[batch_id]: %d  global_seq_idx: %d\n ", src_row, dst_row, merged_offsets[batch_id], global_seq_idx);
            // warp-level parallelization for hidden_dim
            // each thread in warp handles multiple hidden dimensions
            int elements_per_thread = (hidden_dim + 32 - 1) / 32;
            int thread_start = lane_id * elements_per_thread;
            int thread_end = min(thread_start + elements_per_thread, hidden_dim);
            
            // vectorized copy 
            if (hidden_dim % 4 == 0 && thread_start % 4 == 0) {
                int vectorized_start = (thread_start + 3) / 4 * 4;
                int vectorized_end = thread_end / 4 * 4;
                
                for (int h = thread_start; h < vectorized_start && h < thread_end; ++h) {
                    merged_values[dst_row * hidden_dim + h] = values[src_row * hidden_dim + h];
                }
                
                for (int h = vectorized_start; h < vectorized_end; h += 4) {
                    copy_float4(&merged_values[dst_row * hidden_dim + h],
                               &values[src_row * hidden_dim + h]);
                }
                
                // handle remaining elements
                for (int h = vectorized_end; h < thread_end; ++h) {
                    merged_values[dst_row * hidden_dim + h] = values[src_row * hidden_dim + h];
                }
            } else {
                // scalar copy
                for (int h = thread_start; h < thread_end; ++h) {
                    merged_values[dst_row * hidden_dim + h] = values[src_row * hidden_dim + h];
                }
            }
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
//先不管workload了
//等跑完ncu再说
// std::tuple<torch::Tensor, torch::Tensor, int> compute_workload_mappings(
//     torch::Tensor merged_offsets,
//     int num_rows, 
//     int seqlen_per_block) {
    
//     std::vector<int> batch_ids;
//     std::vector<int> seq_block_ids;


//     // 计算每个batch的实际workload
//     for (int batch = 0; batch < num_rows; ++batch) {
//         // 计算该batch的总序列长度
//         int total_seqlen_for_batch = 0;
//         for (const auto& offsets : offsets_list) {
//             auto offsets_ptr = offsets.data_ptr<int32_t>();
//             int len = offsets_ptr[batch + 1] - offsets_ptr[batch];
//             if (len > 0) {  // 只计算非空的tensor
//                 total_seqlen_for_batch += len;
//             }
//         }
        
//         // 跳过空batch
//         if (total_seqlen_for_batch <= 0) continue;
        
//         // 计算该batch需要的block数量
//         int blocks_needed = (total_seqlen_for_batch + seqlen_per_block - 1) / seqlen_per_block;
        
//         // 为每个需要的block生成workload映射
//         for (int seq_block = 0; seq_block < blocks_needed; ++seq_block) {
//             batch_ids.push_back(batch);
//             seq_block_ids.push_back(seq_block);
//         }
//     }
    
//     int total_workloads = batch_ids.size();
    
//     // 处理空workload情况
//     if (total_workloads == 0) {
//         return std::make_tuple(torch::empty({0}, torch::kInt32).cuda(), 
//                               torch::empty({0}, torch::kInt32).cuda(), 0);
//     }
    
//     // 安全的tensor创建方式 - 先拷贝数据，再转移到GPU
//     torch::Tensor workload_batch_ids = torch::tensor(batch_ids, torch::kInt32).cuda();
//     torch::Tensor workload_seq_block_ids = torch::tensor(seq_block_ids, torch::kInt32).cuda();
    
//     return std::make_tuple(workload_batch_ids, workload_seq_block_ids, total_workloads);
// }

void concat_2D_jagged_tensors_cuda_forward (
    const std::vector<torch::Tensor>& values_list,
    const std::vector<torch::Tensor>& offsets_list,
    torch::Tensor merged_values,
    torch::Tensor merged_offsets,
    int max_seqlen){

    int num_tensors = values_list.size();
    if (num_tensors == 0) {
        return; 
    }
    int num_rows = offsets_list[0].size(0) - 1;
    int hidden_dim = values_list[0].size(-1);

    assert(merged_values.is_contiguous());

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
                input_jagged_tensor_typed.offsets_list[i] = offsets_list[i].data_ptr<int32_t>();
            }
            
            // hidden_dim越大，seqlen_per_block越小
            int seqlen_per_block = (hidden_dim <= 128) ? 8 : 
                                  (hidden_dim <= 256) ? 4 : 
                                  (hidden_dim <= 512) ? 2 : 1;
            
            // 避免超过1024线程限制
            int blocks_per_batch = (max_seqlen + seqlen_per_block - 1) / seqlen_per_block;
            int total_blocks = num_rows * blocks_per_batch;
            
            // warp配置：保证不超过1024线程
            int target_warps = min(32, max(1, seqlen_per_block)); // 每个warp处理1个序列
            int threads = min(1024, target_warps * 32);
            
            dim3 opt_blocks(total_blocks);
            dim3 opt_threads(threads);
            
            concat_2D_jagged_tensors_forward_kernel_opt_v3<scalar_t><<<opt_blocks, opt_threads, 0, stream>>>(
                input_jagged_tensor_typed,
                num_tensors,
                num_rows,
                hidden_dim,
                max_seqlen,
                seqlen_per_block,
                merged_values.data_ptr<scalar_t>(),
                merged_offsets.data_ptr<int>()
            );
            
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
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

template <typename T>
__global__ void concat_2D_jagged_tensors_backward_kernel_opt(
    const InputJaggedTensor<T> grad_jagged_tensor,
    const int32_t num_tensors,
    const int32_t num_rows,
    const int32_t hidden_dim,
    const T* grad_output,
    int* merged_offsets) {
    
    __shared__ int32_t shared_lens[kMaxNumTensors];
    
    int batch_id = blockIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (batch_id >= num_rows) return;
    
    if (threadIdx.x < num_tensors) {
        const int32_t* offsets = grad_jagged_tensor.offsets_list[threadIdx.x];
        shared_lens[threadIdx.x] = offsets[batch_id + 1] - offsets[batch_id];
    }
    __syncthreads();
    
    if (warp_id < num_tensors) {
        T* grad_values = grad_jagged_tensor.value_list[warp_id];
        const int32_t* offsets = grad_jagged_tensor.offsets_list[warp_id];
        int start = offsets[batch_id];
        int end = offsets[batch_id + 1];
        int num_rows_in_tensor = end - start;
        
        int out_start = merged_offsets[batch_id];
        for (int t = 0; t < warp_id; ++t) {
            out_start += shared_lens[t];
        }
        
        for (int row_offset = lane_id; row_offset < num_rows_in_tensor; row_offset += 32) {
            int i = start + row_offset;
            int out_row = out_start + row_offset;
            
            if (hidden_dim % 4 == 0 && hidden_dim <= 256) {
                for (int h = 0; h < hidden_dim; h += 4) {
                    copy_float4(&grad_values[i * hidden_dim + h],
                               &grad_output[out_row * hidden_dim + h]);
                }
            } else {
                for (int h = 0; h < hidden_dim; ++h) {
                    grad_values[i * hidden_dim + h] = grad_output[out_row * hidden_dim + h];
                }
            }
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


    // DISPATCH_KERNEL_BY_TYPE(
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        grad_output.scalar_type(), 
        "concat_2D_jagged_tensors_backward_kernel",
        [&] {
            InputJaggedTensor<scalar_t> grad_jagged_tensor;
            for (int i = 0; i < num_tensors; ++i) {
                TORCH_CHECK(i < kMaxNumTensors, "Number of tensors exceeds kMaxNumTensors");
                grad_jagged_tensor.value_list[i] = grad_inputs[i].data_ptr<scalar_t>();
                grad_jagged_tensor.offsets_list[i] = offsets_list[i].data_ptr<int32_t>();
            }

            // choose kernel based on problem size
            if (hidden_dim <= 256 && num_tensors <= 16) {
                // use optimized kernel with shared memory and warp cooperation
                dim3 opt_blocks(num_rows);
                dim3 opt_threads(min(num_tensors * 32, 1024));  // each tensor gets one warp
                
                concat_2D_jagged_tensors_backward_kernel_opt<scalar_t><<<opt_blocks, opt_threads, 0, stream>>>(
                    grad_jagged_tensor,
                    num_tensors,
                    num_rows,
                    hidden_dim,
                    grad_output.data_ptr<scalar_t>(),
                    merged_offsets.data_ptr<int>()
                );
            } else {
                // use basic kernel for large problems
                concat_2D_jagged_tensors_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    grad_jagged_tensor,
                    num_tensors,
                    num_rows,
                    hidden_dim,
                    grad_output.data_ptr<scalar_t>(),
                    merged_offsets.data_ptr<int>()
                );
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
    return grad_inputs;
}
