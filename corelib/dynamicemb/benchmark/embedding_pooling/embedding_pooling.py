"""
Optimized embedding pooling with Triton and CUDA implementations
"""

import torch
import triton
import triton.language as tl

# Try to import CUDA extension
try:
    import embedding_pooling_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA extension not available. Compile with: python setup.py install")


# ============================================================================
# Triton Kernel: Parallel Reduction (Core Implementation)
# ============================================================================

@triton.jit
def pooling_parallel_reduce_kernel(
    embeddings_ptr,
    offsets_ptr,
    output_ptr,
    embedding_dim: tl.constexpr,
    num_segments: tl.constexpr,
    pooling_mode: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Parallel reduction pooling kernel.
    Each program processes one segment with parallel reduction over sequence length.
    
    Key optimization: Loads BLOCK_N embeddings at once and reduces in parallel.
    """
    seg_id = tl.program_id(0)
    if seg_id >= num_segments:
        return
    
    start = tl.load(offsets_ptr + seg_id)
    end = tl.load(offsets_ptr + seg_id + 1)
    length = end - start
    
    # Handle empty segments
    if length == 0:
        for d_off in range(0, embedding_dim, BLOCK_D):
            d_idx = d_off + tl.arange(0, BLOCK_D)
            mask = d_idx < embedding_dim
            tl.store(output_ptr + seg_id * embedding_dim + d_idx, 0.0, mask=mask)
        return
    
    # Process each dimension block
    for d_off in range(0, embedding_dim, BLOCK_D):
        d_idx = d_off + tl.arange(0, BLOCK_D)
        d_mask = d_idx < embedding_dim
        
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        
        # Parallel reduction: process BLOCK_N embeddings at once
        for n_off in range(0, length, BLOCK_N):
            n_idx = n_off + tl.arange(0, BLOCK_N)
            n_mask = n_idx < length
            
            # 2D load: [BLOCK_N, BLOCK_D]
            row_idx = start + n_idx
            indices = row_idx[:, None] * embedding_dim + d_idx[None, :]
            
            embs = tl.load(
                embeddings_ptr + indices,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0
            )
            
            # Parallel sum along sequence axis
            acc += tl.sum(embs, axis=0)
        
        if pooling_mode == 1:
            acc = acc / length.to(tl.float32)
        
        tl.store(output_ptr + seg_id * embedding_dim + d_idx, acc, mask=d_mask)


# ============================================================================
# Main Interface
# ============================================================================

def embedding_pooling(
    embeddings: torch.Tensor,
    offsets: torch.Tensor,
    pooling_mode: str = "mean"
) -> torch.Tensor:
    """
    High-performance embedding pooling with automatic strategy selection.
    
    Strategy:
    - For many small segments (>5000): Use PyTorch native (single kernel launch)
    - Otherwise: Use Triton parallel reduction (optimized for medium/long segments)
    
    Args:
        embeddings: [total_embeddings, embedding_dim]
        offsets: [num_segments + 1]
        pooling_mode: "sum" or "mean"
    
    Returns:
        pooled: [num_segments, embedding_dim]
    """
    assert embeddings.dim() == 2 and offsets.dim() == 1
    assert pooling_mode in ["sum", "mean"]
    assert embeddings.is_contiguous() and offsets.is_contiguous()
    assert embeddings.is_cuda and offsets.is_cuda
    
    num_segs = offsets.shape[0] - 1
    emb_dim = embeddings.shape[1]
    
    # For very large number of segments, PyTorch native is better
    # (avoids compilation overhead and benefits from single kernel launch)
    if num_segs > 5000:
        return embedding_pooling_torch(embeddings, offsets, pooling_mode)
    
    # Use Triton parallel reduction
    output = torch.empty(
        (num_segs, emb_dim),
        dtype=embeddings.dtype,
        device=embeddings.device
    )
    
    mode = 0 if pooling_mode == "sum" else 1
    
    # Adaptive block sizes based on embedding dimension
    if emb_dim <= 64:
        BLOCK_D = 64
    elif emb_dim <= 128:
        BLOCK_D = 128
    elif emb_dim <= 256:
        BLOCK_D = 256
    else:
        BLOCK_D = 512
    
    # BLOCK_N: number of embeddings to process in parallel
    BLOCK_N = 32
    
    grid = (num_segs,)
    
    pooling_parallel_reduce_kernel[grid](
        embeddings_ptr=embeddings,
        offsets_ptr=offsets,
        output_ptr=output,
        embedding_dim=emb_dim,
        num_segments=num_segs,
        pooling_mode=mode,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
    )
    
    return output


# ============================================================================
# CUDA Kernel Wrapper
# ============================================================================

def embedding_pooling_cuda_wrapper(
    embeddings: torch.Tensor,
    offsets: torch.Tensor,
    pooling_mode: str = "mean"
) -> torch.Tensor:
    """
    Wrapper for CUDA kernel implementation.
    
    Design:
    - One block per segment (block_size=256)
    - Vectorized loads (float4) when dim % 4 == 0
    - Simple and efficient for most cases
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA extension not compiled. Run: python setup.py install")
    
    assert embeddings.dim() == 2 and offsets.dim() == 1
    assert pooling_mode in ["sum", "mean"]
    assert embeddings.is_contiguous() and offsets.is_contiguous()
    assert embeddings.is_cuda and offsets.is_cuda
    
    return embedding_pooling_cuda.embedding_pooling_cuda(
        embeddings, offsets, pooling_mode
    )


# ============================================================================
# Reference Implementations (for testing and comparison)
# ============================================================================

def embedding_pooling_reference(
    embeddings: torch.Tensor,
    offsets: torch.Tensor,
    pooling_mode: str
) -> torch.Tensor:
    """
    Reference implementation (from leadership).
    Pure Python loop - easy to understand but slow.
    """
    assert pooling_mode in ["sum", "mean"]
    assert embeddings.dim() == 2, "embeddings must be a 2D tensor"
    assert offsets.dim() == 1, "offsets must be a 1D tensor"
    
    num_segments = offsets.numel() - 1
    dim = embeddings.size(1)
    ret = torch.empty(num_segments, dim, device=embeddings.device, dtype=embeddings.dtype)
    
    if pooling_mode == "sum":
        for i in range(num_segments):
            ret[i, :] = torch.sum(embeddings[offsets[i]:offsets[i+1], :], dim=0)
    elif pooling_mode == "mean":
        for i in range(num_segments):
            segment = embeddings[offsets[i]:offsets[i+1]]
            if segment.shape[0] > 0:
                ret[i, :] = torch.mean(segment, dim=0)
            else:
                ret[i, :] = 0.0
    else:
        raise ValueError(f"Invalid pooling mode: {pooling_mode}")
    
    return ret


def embedding_pooling_torch(
    embeddings: torch.Tensor,
    offsets: torch.Tensor,
    pooling_mode: str = "mean"
) -> torch.Tensor:
    """PyTorch native implementation using scatter."""
    num_segs = offsets.shape[0] - 1
    dim = embeddings.shape[1]
    
    # Create segment IDs
    lengths = offsets[1:] - offsets[:-1]
    seg_ids = torch.repeat_interleave(
        torch.arange(num_segs, device=embeddings.device),
        lengths
    )
    
    # Use scatter_add
    output = torch.zeros(num_segs, dim, dtype=embeddings.dtype, device=embeddings.device)
    
    if pooling_mode == "sum":
        output.scatter_add_(0, seg_ids.unsqueeze(1).expand(-1, dim), embeddings)
    elif pooling_mode == "mean":
        output.scatter_add_(0, seg_ids.unsqueeze(1).expand(-1, dim), embeddings)
        output = output / lengths.unsqueeze(1).clamp(min=1)
    
    return output


# ============================================================================
# Testing
# ============================================================================

def test_correctness():
    """Test correctness against reference implementations."""
    print("=" * 80)
    print("Correctness Testing")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    test_cases = [
        ("Small segments", 100, 128, 10),
        ("Medium segments", 1000, 256, 50),
        ("Large segments", 500, 512, 100),
        ("Many segments", 10000, 128, 20),
        ("Mixed lengths", 1000, 128, None),
    ]
    
    for name, batch_size, emb_dim, avg_len in test_cases:
        print(f"\n{name}:")
        print(f"  batch={batch_size}, dim={emb_dim}, avg_len={avg_len}")
        
        if avg_len is None:
            lengths = torch.randint(1, 100, (batch_size,), device='cuda')
        else:
            lengths = torch.randint(
                max(1, avg_len - 10),
                avg_len + 10,
                (batch_size,),
                device='cuda'
            )
        
        total_embs = lengths.sum().item()
        embeddings = torch.randn(total_embs, emb_dim, device='cuda', dtype=torch.float32)
        offsets = torch.cat([torch.tensor([0], device='cuda'), lengths.cumsum(0)])
        
        for mode in ["sum", "mean"]:
            # Reference (leadership's version)
            ref = embedding_pooling_reference(embeddings, offsets, mode)
            
            # Triton (our implementation)
            triton_out = embedding_pooling(embeddings, offsets, mode)
            
            # PyTorch
            torch_out = embedding_pooling_torch(embeddings, offsets, mode)
            
            # CUDA (if available)
            if CUDA_AVAILABLE:
                cuda_out = embedding_pooling_cuda_wrapper(embeddings, offsets, mode)
                diff_cuda = (cuda_out - ref).abs().max().item()
            else:
                diff_cuda = None
            
            # Compare
            diff_ref = (triton_out - ref).abs().max().item()
            diff_torch = (triton_out - torch_out).abs().max().item()
            
            status = "✓" if diff_ref < 1e-4 and diff_torch < 1e-4 else "✗"
            
            if CUDA_AVAILABLE:
                cuda_status = "✓" if diff_cuda < 1e-4 else "✗"
                print(f"  {mode:4s}: ref={diff_ref:.2e}, torch={diff_torch:.2e}, cuda={diff_cuda:.2e} {status} {cuda_status}")
            else:
                print(f"  {mode:4s}: ref={diff_ref:.2e}, torch={diff_torch:.2e} {status}")
    
    # Edge cases
    print(f"\nEdge cases (with empty segments):")
    lengths = torch.tensor([5, 0, 3, 0, 1], device='cuda')
    total_embs = lengths.sum().item()
    embeddings = torch.randn(total_embs, 64, device='cuda')
    offsets = torch.cat([torch.tensor([0], device='cuda'), lengths.cumsum(0)])
    
    ref = embedding_pooling_reference(embeddings, offsets, "mean")
    triton_out = embedding_pooling(embeddings, offsets, "mean")
    
    diff_triton = (triton_out - ref).abs().max().item()
    
    if CUDA_AVAILABLE:
        cuda_out = embedding_pooling_cuda_wrapper(embeddings, offsets, "mean")
        diff_cuda = (cuda_out - ref).abs().max().item()
        print(f"  Triton: diff={diff_triton:.2e} {'✓' if diff_triton < 1e-4 else '✗'}")
        print(f"  CUDA:   diff={diff_cuda:.2e} {'✓' if diff_cuda < 1e-4 else '✗'}")
    else:
        print(f"  Triton: diff={diff_triton:.2e} {'✓' if diff_triton < 1e-4 else '✗'}")


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark():
    """Benchmark against reference and PyTorch implementations."""
    print("\n" + "=" * 80)
    print("Performance Benchmarking")
    print("=" * 80)
    
    import time
    
    configs = [
        ("Small segs", 1000, 128, 10),
        ("Medium segs", 1000, 256, 50),
        ("Large segs", 500, 256, 150),
        ("Very large", 100, 512, 600),
        ("Many small", 10000, 128, 20),
        ("Huge batch", 20000, 128, 15),
    ]
    
    for name, batch, dim, avg_len in configs:
        lengths = torch.randint(
            max(1, avg_len - 10),
            avg_len + 10,
            (batch,),
            device='cuda'
        )
        total = lengths.sum().item()
        
        embeddings = torch.randn(total, dim, device='cuda', dtype=torch.float32)
        offsets = torch.cat([torch.tensor([0], device='cuda'), lengths.cumsum(0)])
        
        # Determine which implementation will be used
        strategy = "PyTorch" if batch > 5000 else "Triton"
        # strategy = "Triton"
        
        print(f"\n{name}: {batch} segs, dim={dim}, avg_len={avg_len:.0f}, total={total}")
        print(f"  → Strategy: {strategy}")
        
        # Warmup
        for _ in range(10):
            _ = embedding_pooling(embeddings, offsets, "mean")
        torch.cuda.synchronize()
        
        num_iters = 100 if batch <= 10000 else 50
        
        # Benchmark Triton (auto-selects strategy)
        start = time.time()
        for _ in range(num_iters):
            _ = embedding_pooling(embeddings, offsets, "mean")
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / num_iters * 1000
        
        # Benchmark CUDA kernel (if available)
        if CUDA_AVAILABLE:
            start = time.time()
            for _ in range(num_iters):
                _ = embedding_pooling_cuda_wrapper(embeddings, offsets, "mean")
            torch.cuda.synchronize()
            cuda_time = (time.time() - start) / num_iters * 1000
        else:
            cuda_time = None
        
        # Benchmark PyTorch explicitly
        start = time.time()
        for _ in range(num_iters):
            _ = embedding_pooling_torch(embeddings, offsets, "mean")
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / num_iters * 1000
        
        # Benchmark Reference (leadership's version) - skip for huge batches
        if batch <= 10000:
            start = time.time()
            for _ in range(num_iters):
                _ = embedding_pooling_reference(embeddings, offsets, "mean")
            torch.cuda.synchronize()
            ref_time = (time.time() - start) / num_iters * 1000
        else:
            ref_time = None
        
        print(f"\n  Results:")
        print(f"    Triton:       {triton_time:7.4f} ms")
        if CUDA_AVAILABLE:
            print(f"    CUDA kernel:  {cuda_time:7.4f} ms  (vs Triton: {triton_time/cuda_time:5.2f}x)")
        print(f"    PyTorch:      {torch_time:7.4f} ms  (vs Triton: {torch_time/triton_time:5.2f}x)")
        if ref_time:
            print(f"    Reference:    {ref_time:7.4f} ms  (speedup: {ref_time/triton_time:5.1f}x)")


if __name__ == "__main__":
    print("\n🚀 Embedding Pooling - Optimized Triton Implementation\n")
    test_correctness()
    benchmark()
    print("\n" + "=" * 80)
    print("✓ All tests completed")
    print("=" * 80 + "\n")
