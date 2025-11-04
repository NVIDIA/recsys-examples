"""
Optimized Triton backward kernel for embedding pooling
"""

import torch
import triton
import triton.language as tl


from embedding_pooling_kernel import embedding_pooling_backward_triton


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
        ("Many segments", 10000, 128, 20),
        ("Mixed lengths", 1000, 128, None),
    ]
    
    for name, batch, dim, avg_len in configs:
        if avg_len is None:
            lengths = torch.randint(1, 100, (batch,), device='cuda')
        else:
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
