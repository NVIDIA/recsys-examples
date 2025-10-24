// PyBind wrapper for embedding pooling CUDA kernel

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// Forward declaration of CUDA launcher
extern "C" void launch_embedding_pooling_kernel(
    const void* embeddings,
    const int64_t* offsets,
    void* output,
    int embedding_dim,
    int num_segments,
    int pooling_mode,
    bool use_fp16,
    cudaStream_t stream
);

torch::Tensor embedding_pooling_cuda(
    torch::Tensor embeddings,
    torch::Tensor offsets,
    const std::string& pooling_mode
) {
    // Validate inputs
    TORCH_CHECK(embeddings.is_cuda(), "embeddings must be a CUDA tensor");
    TORCH_CHECK(offsets.is_cuda(), "offsets must be a CUDA tensor");
    TORCH_CHECK(embeddings.dim() == 2, "embeddings must be 2D");
    TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D");
    TORCH_CHECK(embeddings.is_contiguous(), "embeddings must be contiguous");
    TORCH_CHECK(offsets.is_contiguous(), "offsets must be contiguous");
    TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
    
    const int64_t total_embeddings = embeddings.size(0);
    const int64_t embedding_dim = embeddings.size(1);
    const int64_t num_segments = offsets.size(0) - 1;
    
    // Validate offsets
    TORCH_CHECK(num_segments >= 0, "num_segments must be non-negative");
    
    // Determine pooling mode
    int mode_int;
    if (pooling_mode == "sum") {
        mode_int = 0;
    } else if (pooling_mode == "mean") {
        mode_int = 1;
    } else {
        TORCH_CHECK(false, "pooling_mode must be 'sum' or 'mean'");
    }
    
    // Create output tensor
    auto output = torch::empty(
        {num_segments, embedding_dim},
        embeddings.options()
    );
    
    // Get CUDA stream
    auto stream = c10::cuda::getCurrentCUDAStream();
    
    // Determine if using fp16
    bool use_fp16 = (embeddings.scalar_type() == torch::kFloat16);
    
    // Launch kernel
    launch_embedding_pooling_kernel(
        embeddings.data_ptr(),
        offsets.data_ptr<int64_t>(),
        output.data_ptr(),
        static_cast<int>(embedding_dim),
        static_cast<int>(num_segments),
        mode_int,
        use_fp16,
        stream
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "embedding_pooling_cuda",
        &embedding_pooling_cuda,
        "CUDA implementation of embedding pooling",
        py::arg("embeddings"),
        py::arg("offsets"),
        py::arg("pooling_mode")
    );
}

