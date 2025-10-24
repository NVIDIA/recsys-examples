// CUDA kernel for embedding pooling
// Optimized for one block per segment

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Warp-level primitives
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Vector load helper for coalesced memory access
template<typename T>
__device__ __forceinline__ float4 load_vec4(const T* ptr) {
    if constexpr (std::is_same_v<T, float>) {
        return *reinterpret_cast<const float4*>(ptr);
    } else if constexpr (std::is_same_v<T, __half>) {
        // Load 8 halfs as float4
        const half2* ptr_h2 = reinterpret_cast<const half2*>(ptr);
        half2 a = ptr_h2[0];
        half2 b = ptr_h2[1];
        half2 c = ptr_h2[2];
        half2 d = ptr_h2[3];
        return make_float4(
            __half2float(a.x) + __half2float(a.y),
            __half2float(b.x) + __half2float(b.y),
            __half2float(c.x) + __half2float(c.y),
            __half2float(d.x) + __half2float(d.y)
        );
    }
}

// Store vector helper
template<typename T>
__device__ __forceinline__ void store_vec4(T* ptr, float4 val) {
    if constexpr (std::is_same_v<T, float>) {
        *reinterpret_cast<float4*>(ptr) = val;
    } else if constexpr (std::is_same_v<T, __half>) {
        half2* ptr_h2 = reinterpret_cast<half2*>(ptr);
        ptr_h2[0] = __floats2half2_rn(val.x, val.y);
        ptr_h2[1] = __floats2half2_rn(val.z, val.w);
    }
}

/**
 * Embedding pooling kernel - one block per segment
 * 
 * Design decisions:
 * - Block size: 256 threads (8 warps) - good balance for most dims
 * - Each thread handles multiple dimensions (stride by blockDim.x)
 * - Vectorized loads (float4) when dimension is aligned
 * - Simple serial reduction over segment length (no complex sync needed)
 * 
 * @param embeddings: [total_embeddings, embedding_dim]
 * @param offsets: [num_segments + 1]
 * @param output: [num_segments, embedding_dim]
 * @param embedding_dim: dimension of embeddings
 * @param num_segments: number of segments
 * @param pooling_mode: 0=sum, 1=mean
 */
template<typename T, int VEC_SIZE>
__global__ void embedding_pooling_kernel(
    const T* __restrict__ embeddings,
    const int64_t* __restrict__ offsets,
    T* __restrict__ output,
    const int embedding_dim,
    const int num_segments,
    const int pooling_mode
) {
    // One block per segment
    const int seg_id = blockIdx.x;
    if (seg_id >= num_segments) return;
    
    // Get segment boundaries
    const int64_t start = offsets[seg_id];
    const int64_t end = offsets[seg_id + 1];
    const int64_t length = end - start;
    
    // Handle empty segments
    if (length == 0) {
        for (int d = threadIdx.x; d < embedding_dim; d += blockDim.x) {
            if constexpr (std::is_same_v<T, float>) {
                output[seg_id * embedding_dim + d] = 0.0f;
            } else {
                output[seg_id * embedding_dim + d] = __float2half(0.0f);
            }
        }
        return;
    }
    
    const int num_vec = embedding_dim / VEC_SIZE;
    const int tid = threadIdx.x;
    
    if constexpr (VEC_SIZE == 4) {
        // Vectorized path (4-element vectors)
        for (int vec_idx = tid; vec_idx < num_vec; vec_idx += blockDim.x) {
            float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            
            // Accumulate over all embeddings in segment
            for (int64_t i = 0; i < length; ++i) {
                const int64_t emb_idx = start + i;
                const T* emb_ptr = embeddings + emb_idx * embedding_dim + vec_idx * VEC_SIZE;
                
                // Load 4 elements
                float4 vals;
                if constexpr (std::is_same_v<T, float>) {
                    vals = *reinterpret_cast<const float4*>(emb_ptr);
                } else {
                    // For half precision
                    vals.x = __half2float(emb_ptr[0]);
                    vals.y = __half2float(emb_ptr[1]);
                    vals.z = __half2float(emb_ptr[2]);
                    vals.w = __half2float(emb_ptr[3]);
                }
                
                acc.x += vals.x;
                acc.y += vals.y;
                acc.z += vals.z;
                acc.w += vals.w;
            }
            
            // Apply pooling mode
            if (pooling_mode == 1) { // mean
                float inv_length = 1.0f / static_cast<float>(length);
                acc.x *= inv_length;
                acc.y *= inv_length;
                acc.z *= inv_length;
                acc.w *= inv_length;
            }
            
            // Store result
            T* out_ptr = output + seg_id * embedding_dim + vec_idx * VEC_SIZE;
            if constexpr (std::is_same_v<T, float>) {
                *reinterpret_cast<float4*>(out_ptr) = acc;
            } else {
                out_ptr[0] = __float2half(acc.x);
                out_ptr[1] = __float2half(acc.y);
                out_ptr[2] = __float2half(acc.z);
                out_ptr[3] = __float2half(acc.w);
            }
        }
        
        // Handle remaining elements (if dim not divisible by 4)
        const int remaining_start = num_vec * VEC_SIZE;
        for (int d = remaining_start + tid; d < embedding_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (int64_t i = 0; i < length; ++i) {
                const int64_t emb_idx = start + i;
                if constexpr (std::is_same_v<T, float>) {
                    acc += embeddings[emb_idx * embedding_dim + d];
                } else {
                    acc += __half2float(embeddings[emb_idx * embedding_dim + d]);
                }
            }
            
            if (pooling_mode == 1) {
                acc /= static_cast<float>(length);
            }
            
            if constexpr (std::is_same_v<T, float>) {
                output[seg_id * embedding_dim + d] = acc;
            } else {
                output[seg_id * embedding_dim + d] = __float2half(acc);
            }
        }
    } else {
        // Scalar path (no vectorization)
        for (int d = tid; d < embedding_dim; d += blockDim.x) {
            float acc = 0.0f;
            
            // Accumulate over segment
            for (int64_t i = 0; i < length; ++i) {
                const int64_t emb_idx = start + i;
                if constexpr (std::is_same_v<T, float>) {
                    acc += embeddings[emb_idx * embedding_dim + d];
                } else {
                    acc += __half2float(embeddings[emb_idx * embedding_dim + d]);
                }
            }
            
            // Apply pooling mode
            if (pooling_mode == 1) {
                acc /= static_cast<float>(length);
            }
            
            if constexpr (std::is_same_v<T, float>) {
                output[seg_id * embedding_dim + d] = acc;
            } else {
                output[seg_id * embedding_dim + d] = __float2half(acc);
            }
        }
    }
}

// Explicit instantiation for float and half
template __global__ void embedding_pooling_kernel<float, 4>(
    const float*, const int64_t*, float*, const int, const int, const int);
template __global__ void embedding_pooling_kernel<float, 1>(
    const float*, const int64_t*, float*, const int, const int, const int);

// Launcher function
extern "C" {

void launch_embedding_pooling_kernel(
    const void* embeddings,
    const int64_t* offsets,
    void* output,
    int embedding_dim,
    int num_segments,
    int pooling_mode,
    bool use_fp16,
    cudaStream_t stream
) {
    // Determine block size
    // 256 threads is a good balance for most dimensions
    const int block_size = 256;
    const int grid_size = num_segments;
    
    if (use_fp16) {
        // Half precision
        if (embedding_dim % 4 == 0) {
            embedding_pooling_kernel<__half, 4><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(embeddings),
                offsets,
                reinterpret_cast<__half*>(output),
                embedding_dim,
                num_segments,
                pooling_mode
            );
        } else {
            embedding_pooling_kernel<__half, 1><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(embeddings),
                offsets,
                reinterpret_cast<__half*>(output),
                embedding_dim,
                num_segments,
                pooling_mode
            );
        }
    } else {
        // Float precision
        if (embedding_dim % 4 == 0) {
            embedding_pooling_kernel<float, 4><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const float*>(embeddings),
                offsets,
                reinterpret_cast<float*>(output),
                embedding_dim,
                num_segments,
                pooling_mode
            );
        } else {
            embedding_pooling_kernel<float, 1><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const float*>(embeddings),
                offsets,
                reinterpret_cast<float*>(output),
                embedding_dim,
                num_segments,
                pooling_mode
            );
        }
    }
}

} // extern "C"

