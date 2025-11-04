"""
Optimized Triton backward kernel for embedding pooling
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Triton Kernel: Pooling Backward (Optimized)
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 64, 'BLOCK_N': 16}, num_warps=2),
        triton.Config({'BLOCK_D': 64, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_D': 64, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_D': 64, 'BLOCK_N': 128}, num_warps=4),
        
        triton.Config({'BLOCK_D': 128, 'BLOCK_N': 16}, num_warps=2),
        triton.Config({'BLOCK_D': 128, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_D': 128, 'BLOCK_N': 64}, num_warps=8),
        triton.Config({'BLOCK_D': 128, 'BLOCK_N': 128}, num_warps=8),
        
        triton.Config({'BLOCK_D': 256, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_D': 256, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_D': 256, 'BLOCK_N': 64}, num_warps=8),
        triton.Config({'BLOCK_D': 256, 'BLOCK_N': 128}, num_warps=8),
        
        triton.Config({'BLOCK_D': 512, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_D': 512, 'BLOCK_N': 32}, num_warps=8),
        triton.Config({'BLOCK_D': 512, 'BLOCK_N': 64}, num_warps=8),
        triton.Config({'BLOCK_D': 512, 'BLOCK_N': 128}, num_warps=8),
    ],
    key=['embedding_dim', 'num_segments'],
)
@triton.jit
def pooling_backward_kernel(
    grad_output_ptr,    # [num_segments, embedding_dim]
    offsets_ptr,        # [num_segments + 1]
    grad_input_ptr,     # [total_embeddings, embedding_dim]
    embedding_dim: tl.constexpr,
    num_segments: tl.constexpr,
    pooling_mode: tl.constexpr,  # 0=sum, 1=mean
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Segment-parallel backward kernel for pooling (scatter operation).
    
    Each program processes one segment and scatters gradient to all embeddings in that segment.
    This mirrors the forward kernel structure - no binary search needed!
    
    For mean pooling: grad_embedding = grad_pooled / length
    For sum pooling:  grad_embedding = grad_pooled
    """
    # Each program handles one segment (same as forward!)
    seg_id = tl.program_id(0)
    
    if seg_id >= num_segments:
        return
    
    # Directly read segment boundaries from offsets (no search needed!)
    start = tl.load(offsets_ptr + seg_id)
    end = tl.load(offsets_ptr + seg_id + 1)
    length = end - start
    
    # Handle empty segments
    if length == 0:
        return
    
    # Calculate scale for mean pooling
    if pooling_mode == 1:
        scale = 1.0 / length.to(tl.float32)
    else:
        scale = 1.0
    
    # Process each dimension block
    for d_off in range(0, embedding_dim, BLOCK_D):
        d_idx = d_off + tl.arange(0, BLOCK_D)
        d_mask = d_idx < embedding_dim
        
        # Load gradient from pooled output (once per dimension block)
        grad_offset = seg_id * embedding_dim + d_idx
        grad = tl.load(grad_output_ptr + grad_offset, mask=d_mask, other=0.0)
        
        # Apply scaling for mean pooling
        if pooling_mode == 1:
            grad = grad * scale
        
        # Parallel scatter: write to all embeddings in this segment
        # Process BLOCK_N embeddings at once
        for n_off in range(0, length, BLOCK_N):
            n_idx = n_off + tl.arange(0, BLOCK_N)
            n_mask = n_idx < length
            
            # 2D store: [BLOCK_N, BLOCK_D]
            row_idx = start + n_idx
            indices = row_idx[:, None] * embedding_dim + d_idx[None, :]
            
            # Broadcast grad to all embeddings in segment
            grad_broadcasted = grad[None, :]  # [1, BLOCK_D] -> broadcast to [BLOCK_N, BLOCK_D]
            
            tl.store(
                grad_input_ptr + indices,
                grad_broadcasted,
                mask=n_mask[:, None] & d_mask[None, :]
            )


# ============================================================================
# Python Interface
# ============================================================================

def embedding_pooling_backward_triton(
    grad_output: torch.Tensor,  # [num_segments, embedding_dim]
    offsets: torch.Tensor,      # [num_segments + 1]
    pooling_mode: str = "mean"
) -> torch.Tensor:
    """
    Triton implementation of pooling backward using segment-parallel scatter.
    
    Key optimization: Mirrors forward kernel structure - each program handles one segment.
    No binary search needed! Direct scatter operation is faster than searching.
    
    Args:
        grad_output: Gradient w.r.t. pooled output [num_segments, embedding_dim]
        offsets: Segment boundaries [num_segments + 1]
        pooling_mode: "sum" or "mean"
    
    Returns:
        grad_input: Gradient w.r.t. input embeddings [total_embeddings, embedding_dim]
    """
    assert grad_output.dim() == 2 and offsets.dim() == 1
    assert pooling_mode in ["sum", "mean"]
    assert grad_output.is_contiguous() and offsets.is_contiguous()
    assert grad_output.is_cuda and offsets.is_cuda
    
    num_segs = offsets.shape[0] - 1
    emb_dim = grad_output.shape[1]
    total_embs = offsets[-1].item()
    
    # Create output tensor
    grad_input = torch.empty(
        (total_embs, emb_dim),
        dtype=grad_output.dtype,
        device=grad_output.device
    )
    
    mode = 0 if pooling_mode == "sum" else 1
    
    # Grid size = num_segments (same as forward!)
    grid = (num_segs,)
    
    # Launch kernel - segment-parallel scatter, no search needed!
    pooling_backward_kernel[grid](
        grad_output_ptr=grad_output,
        offsets_ptr=offsets,
        grad_input_ptr=grad_input,
        embedding_dim=emb_dim,
        num_segments=num_segs,
        pooling_mode=mode,
    )
    
    return grad_input


def embedding_pooling_backward_torch(
    grad_output: torch.Tensor,
    offsets: torch.Tensor,
    pooling_mode: str = "mean"
) -> torch.Tensor:
    """
    PyTorch reference implementation of pooling backward.
    """
    num_segs = offsets.shape[0] - 1
    emb_dim = grad_output.shape[1]
    
    lengths = offsets[1:] - offsets[:-1]
    segment_ids = torch.repeat_interleave(
        torch.arange(num_segs, device=offsets.device),
        lengths
    )
    
    # Scatter: each embedding gets gradient from its segment
    grad_input = grad_output[segment_ids]
    
    # For mean pooling, divide by length
    if pooling_mode == "mean":
        lengths_expanded = lengths[segment_ids].unsqueeze(1).float()
        grad_input = grad_input / lengths_expanded
    
    return grad_input


# ============================================================================
# Correctness Testing
# ============================================================================

def test_correctness():
    """Test backward correctness."""
    print("=" * 80)
    print("Backward Correctness Testing")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    test_cases = [
        ("Small", 100, 128, 10),
        ("Medium", 1000, 256, 50),
        ("Large", 500, 512, 100),
        ("Many segs", 10000, 128, 20),
    ]
    
    for name, batch_size, emb_dim, avg_len in test_cases:
        print(f"\n{name}: batch={batch_size}, dim={emb_dim}, avg_len={avg_len}")
        
        lengths = torch.randint(
            max(1, avg_len - 10),
            avg_len + 10,
            (batch_size,),
            device='cuda'
        )
        total_embs = lengths.sum().item()
        offsets = torch.cat([torch.tensor([0], device='cuda'), lengths.cumsum(0)])
        
        grad_output = torch.randn(batch_size, emb_dim, device='cuda', dtype=torch.float32)
        
        for mode in ["sum", "mean"]:
            # Triton implementation
            grad_triton = embedding_pooling_backward_triton(grad_output, offsets, mode)
            
            # PyTorch reference
            grad_torch = embedding_pooling_backward_torch(grad_output, offsets, mode)
            
            # Compare
            diff = (grad_triton - grad_torch).abs().max().item()
            status = "✓" if diff < 1e-5 else "✗"
            
            print(f"  {mode:4s}: diff={diff:.2e} {status}")
    
    # Edge case: empty segments
    print(f"\nEdge case (empty segments):")
    lengths = torch.tensor([3, 0, 5, 0, 2], device='cuda')
    offsets = torch.cat([torch.tensor([0], device='cuda'), lengths.cumsum(0)])
    total_embs = lengths.sum().item()
    
    grad_output = torch.randn(5, 64, device='cuda')
    
    grad_triton = embedding_pooling_backward_triton(grad_output, offsets, "mean")
    grad_torch = embedding_pooling_backward_torch(grad_output, offsets, "mean")
    
    diff = (grad_triton - grad_torch).abs().max().item()
    print(f"  diff={diff:.2e} {'✓' if diff < 1e-5 else '✗'}")


def test_gradient_check():
    """Use PyTorch's gradcheck to verify backward."""
    print("\n" + "=" * 80)
    print("PyTorch Gradient Check")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    embeddings = torch.randn(20, 16, device='cuda', dtype=torch.float32, requires_grad=True)
    offsets = torch.tensor([0, 5, 12, 20], device='cuda')
    
    from embedding_pooling_autograd import PoolingFunction
    
    print("\nMean pooling:")
    def func_mean(emb):
        return PoolingFunction.apply(emb, offsets, "mean")
    
    result = torch.autograd.gradcheck(
        func_mean, embeddings,
        eps=1e-6, atol=1e-4,
        raise_exception=False
    )
    print(f"  {'✓ PASSED' if result else '✗ FAILED'}")
    
    print("\nSum pooling:")
    def func_sum(emb):
        return PoolingFunction.apply(emb, offsets, "sum")
    
    result = torch.autograd.gradcheck(
        func_sum, embeddings,
        eps=1e-6, atol=1e-4,
        raise_exception=False
    )
    print(f"  {'✓ PASSED' if result else '✗ FAILED'}")


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark():
    """Benchmark backward performance."""
    print("\n" + "=" * 80)
    print("Backward Performance Benchmarking")
    print("=" * 80)
    
    import time
    
    configs = [
        ("Small segs", 1000, 128, 10),
        ("Medium segs", 1000, 256, 50),
        ("Large segs", 500, 256, 150),
        ("Very large", 100, 512, 600),
        ("Many small", 10000, 128, 20),
    ]
    
    for name, batch, dim, avg_len in configs:
        lengths = torch.randint(
            max(1, avg_len - 10),
            avg_len + 10,
            (batch,),
            device='cuda'
        )
        total = lengths.sum().item()
        offsets = torch.cat([torch.tensor([0], device='cuda'), lengths.cumsum(0)])
        
        grad_output = torch.randn(batch, dim, device='cuda', dtype=torch.float32)
        
        print(f"\n{name}: {batch} segs, dim={dim}, avg_len={avg_len:.0f}, total={total}")
        
        # Warmup
        for _ in range(20):
            _ = embedding_pooling_backward_triton(grad_output, offsets, "mean")
        torch.cuda.synchronize()
        
        num_iters = 100 if batch <= 10000 else 50
        
        # Benchmark Triton
        start = time.time()
        for _ in range(num_iters):
            grad_triton = embedding_pooling_backward_triton(grad_output, offsets, "mean")
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / num_iters * 1000
        
        # Warmup PyTorch
        for _ in range(20):
            _ = embedding_pooling_backward_torch(grad_output, offsets, "mean")
        torch.cuda.synchronize()
        
        # Benchmark PyTorch
        start = time.time()
        for _ in range(num_iters):
            grad_torch = embedding_pooling_backward_torch(grad_output, offsets, "mean")
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / num_iters * 1000
        
        print(f"  Results:")
        print(f"    Triton:   {triton_time:7.4f} ms")
        print(f"    PyTorch:  {torch_time:7.4f} ms  (ratio: {torch_time/triton_time:5.2f}x)")
        
        # Verify they produce same result
        diff = (grad_triton - grad_torch).abs().max().item()
        print(f"    Diff: {diff:.2e} {'✓' if diff < 1e-5 else '✗'}")


def benchmark_forward_backward():
    """Benchmark complete forward + backward pass."""
    print("\n" + "=" * 80)
    print("Complete Forward + Backward Benchmarking")
    print("=" * 80)
    
    import time
    from embedding_pooling import embedding_pooling
    from embedding_pooling_autograd import PoolingFunction
    
    configs = [
        ("Medium", 1000, 256, 50),
        ("Large", 500, 512, 100),
    ]
    
    for name, batch, dim, avg_len in configs:
        lengths = torch.randint(
            max(1, avg_len - 10),
            avg_len + 10,
            (batch,),
            device='cuda'
        )
        total = lengths.sum().item()
        offsets = torch.cat([torch.tensor([0], device='cuda'), lengths.cumsum(0)])
        
        print(f"\n{name}: {batch} segs, dim={dim}, total={total}")
        
        # Warmup
        for _ in range(10):
            embeddings = torch.randn(total, dim, device='cuda', requires_grad=True)
            pooled = PoolingFunction.apply(embeddings, offsets, "mean")
            loss = pooled.sum()
            loss.backward()
        torch.cuda.synchronize()
        
        num_iters = 50
        
        # Benchmark with autograd
        start = time.time()
        for _ in range(num_iters):
            embeddings = torch.randn(total, dim, device='cuda', requires_grad=True)
            pooled = PoolingFunction.apply(embeddings, offsets, "mean")
            loss = pooled.sum()
            loss.backward()
        torch.cuda.synchronize()
        total_time = (time.time() - start) / num_iters * 1000
        
        # Measure forward only
        start = time.time()
        for _ in range(num_iters):
            embeddings = torch.randn(total, dim, device='cuda')
            pooled = embedding_pooling(embeddings, offsets, "mean")
        torch.cuda.synchronize()
        forward_time = (time.time() - start) / num_iters * 1000
        
        backward_time = total_time - forward_time
        
        print(f"  Forward:  {forward_time:7.4f} ms")
        print(f"  Backward: {backward_time:7.4f} ms")
        print(f"  Total:    {total_time:7.4f} ms")
        print(f"  Backward/Forward ratio: {backward_time/forward_time:.2f}x")


if __name__ == "__main__":
    print("\n🚀 Embedding Pooling Backward - Triton Implementation\n")
    
    test_correctness()
    # test_gradient_check()
    benchmark()
    benchmark_forward_backward()
    
    print("\n" + "=" * 80)
    print("✓ All backward tests completed")
    print("=" * 80 + "\n")
