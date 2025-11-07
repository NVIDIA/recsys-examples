import torch
import triton
from embedding_pooling import embedding_pooling
from embedding_pooling_kernel import pooling_backward_kernel


@torch.fx.wrap
def prev_power_of_2(x: int) -> int:
    if torch.compiler.is_compiling():
        # Re-write to make Dynamo happy
        x_tensor = torch.scalar_tensor(x, dtype=torch.int64)  # type: ignore[arg-type]
        x_tensor_orig = x_tensor.clone()
        out = triton.next_power_of_2(x_tensor)  # type: ignore[arg-type]
        return int(torch.where(torch.lt(x_tensor_orig, out), out // 2, out).item())  # type: ignore[return-value]
    else:
        out = triton.next_power_of_2(x)
        return out // 2 if out > x else out


# ============================================================================
# Backward: Only for testing
# ============================================================================


def embedding_pooling_backward_triton(
    grad_output: torch.Tensor,  # [num_segments, embedding_dim]
    offsets: torch.Tensor,  # [num_segments + 1]
    pooling_mode: str = "mean",
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

    num_segs = offsets.shape[0] - 1
    emb_dim = grad_output.shape[1]
    total_embs = offsets[-1].item()

    # Create output tensor
    grad_input = torch.empty(
        (total_embs, emb_dim), dtype=grad_output.dtype, device=grad_output.device
    )

    mode = 0 if pooling_mode == "sum" else 1

    MIN_BLOCK_D = 64  # Minimum BLOCK_D across all autotune configs
    num_d_blocks = triton.cdiv(emb_dim, MIN_BLOCK_D)
    grid = (num_segs, num_d_blocks)
    autotune_num_segments = prev_power_of_2(num_segs)
    pooling_backward_kernel[grid](
        grad_output_ptr=grad_output,
        offsets_ptr=offsets,
        grad_input_ptr=grad_input,
        embedding_dim=emb_dim,
        num_segments=num_segs,
        pooling_mode=mode,
        autotune_num_segments=autotune_num_segments,
    )

    return grad_input


# ============================================================================
# Forward and backward: Reference Implementations (for testing and comparison)
# ============================================================================


def embedding_pooling_reference(
    embeddings: torch.Tensor, offsets: torch.Tensor, pooling_mode: str
) -> torch.Tensor:
    """
    Reference implementation.
    """
    assert pooling_mode in ["sum", "mean"]
    assert embeddings.dim() == 2, "embeddings must be a 2D tensor"
    assert offsets.dim() == 1, "offsets must be a 1D tensor"

    num_segments = offsets.numel() - 1
    dim = embeddings.size(1)
    ret = torch.empty(
        num_segments, dim, device=embeddings.device, dtype=embeddings.dtype
    )

    if pooling_mode == "sum":
        for i in range(num_segments):
            ret[i, :] = torch.sum(embeddings[offsets[i] : offsets[i + 1], :], dim=0)
    elif pooling_mode == "mean":
        for i in range(num_segments):
            segment = embeddings[offsets[i] : offsets[i + 1]]
            if segment.shape[0] > 0:
                ret[i, :] = torch.mean(segment, dim=0)
            else:
                ret[i, :] = 0.0
    else:
        raise ValueError(f"Invalid pooling mode: {pooling_mode}")

    return ret


def embedding_pooling_torch(
    embeddings: torch.Tensor, offsets: torch.Tensor, pooling_mode: str = "mean"
) -> torch.Tensor:
    """PyTorch reference implementation using scatter."""
    num_segs = offsets.shape[0] - 1
    dim = embeddings.shape[1]

    # Create segment IDs
    lengths = offsets[1:] - offsets[:-1]
    seg_ids = torch.repeat_interleave(
        torch.arange(num_segs, device=embeddings.device), lengths
    )

    # Use scatter_add
    output = torch.zeros(
        num_segs, dim, dtype=embeddings.dtype, device=embeddings.device
    )

    if pooling_mode == "sum":
        output.scatter_add_(0, seg_ids.unsqueeze(1).expand(-1, dim), embeddings)
    elif pooling_mode == "mean":
        output.scatter_add_(0, seg_ids.unsqueeze(1).expand(-1, dim), embeddings)
        output = output / lengths.unsqueeze(1).clamp(min=1)

    return output


def embedding_pooling_backward_torch(
    grad_output: torch.Tensor, offsets: torch.Tensor, pooling_mode: str = "mean"
) -> torch.Tensor:
    """
    PyTorch reference implementation of pooling backward.
    """
    num_segs = offsets.shape[0] - 1
    grad_output.shape[1]

    lengths = offsets[1:] - offsets[:-1]
    segment_ids = torch.repeat_interleave(
        torch.arange(num_segs, device=offsets.device), lengths
    )

    # Scatter: each embedding gets gradient from its segment
    grad_input = grad_output[segment_ids]

    # For mean pooling, divide by length
    if pooling_mode == "mean":
        lengths_expanded = lengths[segment_ids].unsqueeze(1).float()
        grad_input = grad_input / lengths_expanded

    return grad_input


# ============================================================================
# Forward: Correctness Testing
# ============================================================================


def test_forward_correctness():
    """Test forward correctness against reference implementations."""
    print("=" * 80)
    print("Forward Correctness Testing")
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
            lengths = torch.randint(1, 100, (batch_size,), device="cuda")
        else:
            lengths = torch.randint(
                max(1, avg_len - 10), avg_len + 10, (batch_size,), device="cuda"
            )

        total_embs = lengths.sum().item()
        embeddings = torch.randn(
            total_embs, emb_dim, device="cuda", dtype=torch.float32
        )
        offsets = torch.cat([torch.tensor([0], device="cuda"), lengths.cumsum(0)])

        for mode in ["sum", "mean"]:
            # Reference (leadership's version)
            ref = embedding_pooling_reference(embeddings, offsets, mode)

            # Triton (our implementation - official interface)
            triton_out = embedding_pooling(embeddings, offsets, mode)

            # PyTorch
            torch_out = embedding_pooling_torch(embeddings, offsets, mode)

            # Compare
            diff_ref = (triton_out - ref).abs().max().item()
            diff_torch = (triton_out - torch_out).abs().max().item()

            status = "✓" if diff_ref < 1e-4 and diff_torch < 1e-4 else "✗"
            print(f"  {mode:4s}: ref={diff_ref:.2e}, torch={diff_torch:.2e} {status}")

            # Assert correctness
            assert (
                diff_ref < 1e-4
            ), f"Forward failed: diff vs reference = {diff_ref:.2e} (mode={mode}, case={name})"
            assert (
                diff_torch < 1e-4
            ), f"Forward failed: diff vs PyTorch = {diff_torch:.2e} (mode={mode}, case={name})"

    # Edge cases
    print(f"\nEdge cases (with empty segments):")
    lengths = torch.tensor([5, 0, 3, 0, 1], device="cuda")
    total_embs = lengths.sum().item()
    embeddings = torch.randn(total_embs, 64, device="cuda")
    offsets = torch.cat([torch.tensor([0], device="cuda"), lengths.cumsum(0)])

    ref = embedding_pooling_reference(embeddings, offsets, "mean")
    triton_out = embedding_pooling(embeddings, offsets, "mean")

    diff_triton = (triton_out - ref).abs().max().item()
    print(f"  Triton: diff={diff_triton:.2e} {'✓' if diff_triton < 1e-4 else '✗'}")

    # Assert correctness
    assert diff_triton < 1e-4, f"Forward edge case failed: diff = {diff_triton:.2e}"


# ============================================================================
# Backward: Correctness Testing
# ============================================================================


def test_backward_correctness():
    """
    Test backward correctness by comparing against PyTorch autograd.
    """
    print("\n" + "=" * 80)
    print("Backward vs PyTorch Autograd")
    print("=" * 80)

    torch.manual_seed(42)

    from embedding_pooling import embedding_pooling

    test_cases = [
        ("Small", 100, 128, 10),
        ("Medium", 1000, 256, 50),
        ("Large", 500, 512, 100),
        ("Many segs", 5000, 128, 20),
    ]

    for name, batch_size, emb_dim, avg_len in test_cases:
        print(f"\n{name}: batch={batch_size}, dim={emb_dim}, avg_len={avg_len}")

        lengths = torch.randint(
            max(1, avg_len - 10), avg_len + 10, (batch_size,), device="cuda"
        )
        total_embs = lengths.sum().item()
        offsets = torch.cat([torch.tensor([0], device="cuda"), lengths.cumsum(0)])

        for mode in ["sum", "mean"]:
            embeddings_ref = torch.randn(
                total_embs,
                emb_dim,
                device="cuda",
                dtype=torch.float32,
                requires_grad=True,
            )

            # Reference: PyTorch
            num_segs = offsets.shape[0] - 1
            seg_ids = torch.repeat_interleave(
                torch.arange(num_segs, device="cuda"), lengths
            )

            pooled_ref = torch.zeros(num_segs, emb_dim, device="cuda")
            pooled_ref.scatter_add_(
                0, seg_ids.unsqueeze(1).expand(-1, emb_dim), embeddings_ref
            )
            if mode == "mean":
                pooled_ref = pooled_ref / lengths.unsqueeze(1).float().clamp(min=1)

            grad_output = torch.randn_like(pooled_ref)
            pooled_ref.backward(grad_output)
            grad_ref = embeddings_ref.grad.clone()

            # ===== Our Implementation:  =====
            embeddings_ours = embeddings_ref.detach().clone().requires_grad_(True)

            pooled_ours = embedding_pooling(embeddings_ours, offsets, mode)

            pooled_ours.backward(grad_output)
            grad_ours = embeddings_ours.grad.clone()

            diff = (grad_ours - grad_ref).abs().max().item()
            status = "✓" if diff < 1e-5 else "✗"

            print(f"  {mode:4s}: diff={diff:.2e} {status}")

            assert (
                diff < 1e-5
            ), f"Backward failed: diff vs PyTorch autograd = {diff:.2e} (mode={mode}, case={name})"

    # Edge case: empty segments
    print(f"\nEdge case (empty segments):")
    lengths = torch.tensor([3, 0, 5, 0, 2], device="cuda")
    total_embs = lengths.sum().item()
    offsets = torch.cat([torch.tensor([0], device="cuda"), lengths.cumsum(0)])

    # Ground truth
    embeddings_ref = torch.randn(total_embs, 64, device="cuda", requires_grad=True)
    num_segs = 5
    seg_ids = torch.repeat_interleave(torch.arange(num_segs, device="cuda"), lengths)

    pooled_ref = torch.zeros(num_segs, 64, device="cuda")
    pooled_ref.scatter_add_(0, seg_ids.unsqueeze(1).expand(-1, 64), embeddings_ref)
    pooled_ref = pooled_ref / lengths.unsqueeze(1).float().clamp(min=1)

    grad_output = torch.randn_like(pooled_ref)
    pooled_ref.backward(grad_output)
    grad_ref = embeddings_ref.grad.clone()

    # Our implementation
    embeddings_ours = embeddings_ref.detach().clone().requires_grad_(True)
    pooled_ours = embedding_pooling(embeddings_ours, offsets, "mean")
    pooled_ours.backward(grad_output)
    grad_ours = embeddings_ours.grad.clone()

    diff = (grad_ours - grad_ref).abs().max().item()
    print(f"  diff={diff:.2e} {'✓' if diff < 1e-5 else '✗'}")

    # Assert correctness
    assert diff < 1e-5, f"Backward edge case failed: diff = {diff:.2e}"

    print("\n✓ All backward tests passed against PyTorch autograd!")


# ============================================================================
# Forward: Benchmarking
# ============================================================================


def benchmark_forward():
    """Benchmark forward against reference and PyTorch implementations."""
    print("\n" + "=" * 80)
    print("Forward Performance Benchmarking")
    print("=" * 80)

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
            max(1, avg_len - 10), avg_len + 10, (batch,), device="cuda"
        )
        total = lengths.sum().item()

        embeddings = torch.randn(total, dim, device="cuda", dtype=torch.float32)
        offsets = torch.cat([torch.tensor([0], device="cuda"), lengths.cumsum(0)])

        print(
            f"\n{name}: {batch} segs, dim={dim}, avg_len={avg_len:.0f}, total={total}"
        )

        # Warmup
        for _ in range(20):
            _ = embedding_pooling(embeddings, offsets, "mean")
        torch.cuda.synchronize()

        num_iters = 100 if batch <= 10000 else 50

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Benchmark Triton (official interface)
        start_event.record()
        for _ in range(num_iters):
            _ = embedding_pooling(embeddings, offsets, "mean")
        end_event.record()
        torch.cuda.synchronize()
        triton_time = start_event.elapsed_time(end_event) / num_iters

        # Benchmark PyTorch
        # Warmup
        for _ in range(20):
            _ = embedding_pooling_torch(embeddings, offsets, "mean")
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(num_iters):
            _ = embedding_pooling_torch(embeddings, offsets, "mean")
        end_event.record()
        torch.cuda.synchronize()
        torch_time = start_event.elapsed_time(end_event) / num_iters

        # Benchmark Reference (skip for huge batches)
        if batch <= 10000:
            # Warmup
            for _ in range(20):
                _ = embedding_pooling_reference(embeddings, offsets, "mean")
            torch.cuda.synchronize()

            start_event.record()
            for _ in range(num_iters):
                _ = embedding_pooling_reference(embeddings, offsets, "mean")
            end_event.record()
            torch.cuda.synchronize()
            ref_time = start_event.elapsed_time(end_event) / num_iters
        else:
            ref_time = None

        print(f"\n  Results:")
        print(f"    Triton:       {triton_time:7.4f} ms")
        print(
            f"    PyTorch:      {torch_time:7.4f} ms  (vs Triton: {torch_time/triton_time:5.2f}x)"
        )
        if ref_time:
            print(
                f"    Reference:    {ref_time:7.4f} ms  (speedup: {ref_time/triton_time:5.1f}x)"
            )


# ============================================================================
# Backward: Benchmarking
# ============================================================================


def benchmark_backward():
    """Benchmark backward performance."""
    print("\n" + "=" * 80)
    print("Backward Performance Benchmarking")
    print("=" * 80)

    configs = [
        ("Small segs", 1000, 128, 10),
        ("Medium segs", 1000, 256, 50),
        ("Large segs", 500, 256, 150),
        ("Very large", 100, 512, 600),
        ("Many small", 10000, 128, 20),
    ]

    for name, batch, dim, avg_len in configs:
        lengths = torch.randint(
            max(1, avg_len - 10), avg_len + 10, (batch,), device="cuda"
        )
        total = lengths.sum().item()
        offsets = torch.cat([torch.tensor([0], device="cuda"), lengths.cumsum(0)])

        grad_output = torch.randn(batch, dim, device="cuda", dtype=torch.float32)

        print(
            f"\n{name}: {batch} segs, dim={dim}, avg_len={avg_len:.0f}, total={total}"
        )

        # Warmup
        for _ in range(20):
            _ = embedding_pooling_backward_triton(grad_output, offsets, "mean")
        torch.cuda.synchronize()

        num_iters = 100 if batch <= 10000 else 50

        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Benchmark Triton
        start_event.record()
        for _ in range(num_iters):
            grad_triton = embedding_pooling_backward_triton(
                grad_output, offsets, "mean"
            )
        end_event.record()
        torch.cuda.synchronize()
        triton_time = start_event.elapsed_time(end_event) / num_iters

        # Warmup PyTorch
        for _ in range(20):
            _ = embedding_pooling_backward_torch(grad_output, offsets, "mean")
        torch.cuda.synchronize()

        # Benchmark PyTorch
        start_event.record()
        for _ in range(num_iters):
            grad_torch = embedding_pooling_backward_torch(grad_output, offsets, "mean")
        end_event.record()
        torch.cuda.synchronize()
        torch_time = start_event.elapsed_time(end_event) / num_iters

        print(f"  Results:")
        print(f"    Triton:   {triton_time:7.4f} ms")
        print(
            f"    PyTorch:  {torch_time:7.4f} ms  (ratio: {torch_time/triton_time:5.2f}x)"
        )

        # Verify they produce same result
        diff = (grad_triton - grad_torch).abs().max().item()
        print(f"    Diff: {diff:.2e} {'✓' if diff < 1e-5 else '✗'}")


def benchmark_forward_backward():
    """Benchmark complete forward + backward pass."""
    print("\n" + "=" * 80)
    print("Complete Forward + Backward Benchmarking")
    print("=" * 80)

    from embedding_pooling import PoolingFunction

    configs = [
        ("Medium", 1000, 256, 50),
        ("Large", 500, 512, 100),
        ("Many segments", 10000, 128, 20),
        ("Mixed lengths", 1000, 128, None),
    ]

    for name, batch, dim, avg_len in configs:
        if avg_len is None:
            lengths = torch.randint(1, 100, (batch,), device="cuda")
        else:
            lengths = torch.randint(
                max(1, avg_len - 10), avg_len + 10, (batch,), device="cuda"
            )
        total = lengths.sum().item()
        offsets = torch.cat([torch.tensor([0], device="cuda"), lengths.cumsum(0)])

        print(f"\n{name}: {batch} segs, dim={dim}, total={total}")

        embeddings = torch.randn(total, dim, device="cuda", requires_grad=True)

        # Warmup
        for _ in range(10):
            embeddings.grad = None
            pooled = PoolingFunction.apply(embeddings, offsets, "mean")
            loss = pooled.sum()
            loss.backward()
        torch.cuda.synchronize()

        num_iters = 50

        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Benchmark with autograd (forward + backward)
        start_event.record()
        for _ in range(num_iters):
            embeddings.grad = None
            pooled = PoolingFunction.apply(embeddings, offsets, "mean")
            loss = pooled.sum()
            loss.backward()
        end_event.record()
        torch.cuda.synchronize()
        total_time = start_event.elapsed_time(end_event) / num_iters

        # Measure forward only
        embeddings_no_grad = torch.randn(total, dim, device="cuda")

        # Warmup forward
        for _ in range(10):
            _ = embedding_pooling(embeddings_no_grad, offsets, "mean")
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(num_iters):
            pooled = embedding_pooling(embeddings_no_grad, offsets, "mean")
        end_event.record()
        torch.cuda.synchronize()
        forward_time = start_event.elapsed_time(end_event) / num_iters

        backward_time = total_time - forward_time

        print(f"  Forward:  {forward_time:7.4f} ms")
        print(f"  Backward: {backward_time:7.4f} ms")
        print(f"  Total:    {total_time:7.4f} ms")
        print(f"  Backward/Forward ratio: {backward_time/forward_time:.2f}x")


if __name__ == "__main__":
    print("\n Embedding Pooling - Forward and Backward Testing\n")

    # Forward tests
    test_forward_correctness()
    benchmark_forward()

    # Backward tests
    test_backward_correctness()  # Ground truth validation using PyTorch autograd
    benchmark_backward()
    benchmark_forward_backward()

    print("\n" + "=" * 80)
    print("✓ All forward and backward tests completed")
    print("=" * 80 + "\n")
