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
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['embedding_dim'],
)
@triton.jit
def pooling_backward_kernel(
    grad_output_ptr,    # [num_segments, embedding_dim]
    offsets_ptr,        # [num_segments + 1]
    grad_input_ptr,     # [total_embeddings, embedding_dim]
    embedding_dim: tl.constexpr,
    num_segments: tl.constexpr,
    total_embeddings: tl.constexpr,
    pooling_mode: tl.constexpr,  # 0=sum, 1=mean
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward kernel for pooling with in-kernel binary search.
    
    Each program processes one embedding.
    Uses binary search to find which segment the embedding belongs to.
    Then scatters gradient from that segment.
    
    For mean pooling: grad_embedding = grad_pooled / length
    For sum pooling:  grad_embedding = grad_pooled
    """
    # Each program handles one embedding
    emb_id = tl.program_id(0)
    
    if emb_id >= total_embeddings:
        return
    
    # Binary search to find segment ID
    # Find seg_id such that: offsets[seg_id] <= emb_id < offsets[seg_id+1]
    left = 0
    right = num_segments
    
    # Binary search loop (Triton doesn't support break, use conditional)
    for _ in range(20):  # log2(1M) ~ 20, enough for most cases
        # Only continue search if not converged
        if left < right - 1:
            mid = (left + right) // 2
            mid_offset = tl.load(offsets_ptr + mid)
            if emb_id < mid_offset:
                right = mid
            else:
                left = mid
    
    seg_id = left
    
    # Get segment boundaries for length calculation
    start = tl.load(offsets_ptr + seg_id)
    end = tl.load(offsets_ptr + seg_id + 1)
    length = end - start
    
    # Calculate scale for mean pooling
    if pooling_mode == 1:
        scale = 1.0 / length.to(tl.float32)
    else:
        scale = 1.0
    
    # Process dimensions
    for d_off in range(0, embedding_dim, BLOCK_SIZE):
        d_idx = d_off + tl.arange(0, BLOCK_SIZE)
        mask = d_idx < embedding_dim
        
        # Load gradient from pooled output
        grad_offset = seg_id * embedding_dim + d_idx
        grad = tl.load(grad_output_ptr + grad_offset, mask=mask, other=0.0)
        
        # Apply scaling for mean pooling
        if pooling_mode == 1:
            grad = grad * scale
        
        # Store to grad_input
        output_offset = emb_id * embedding_dim + d_idx
        tl.store(grad_input_ptr + output_offset, grad, mask=mask)


# ============================================================================
# Python Interface
# ============================================================================

def embedding_pooling_backward_triton(
    grad_output: torch.Tensor,  # [num_segments, embedding_dim]
    offsets: torch.Tensor,      # [num_segments + 1]
    pooling_mode: str = "mean"
) -> torch.Tensor:
    """
    Triton implementation of pooling backward with in-kernel binary search.
    
    Optimization: No need to precompute segment_ids, saves memory and computation.
    
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
    grid = (total_embs,)
    
    # Launch kernel - no need for precomputed segment_ids!
    pooling_backward_kernel[grid](
        grad_output_ptr=grad_output,
        offsets_ptr=offsets,
        grad_input_ptr=grad_input,
        embedding_dim=emb_dim,
        num_segments=num_segs,
        total_embeddings=total_embs,
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
    test_gradient_check()
    benchmark()
    benchmark_forward_backward()
    
    print("\n" + "=" * 80)
    print("✓ All backward tests completed")
    print("=" * 80 + "\n")
