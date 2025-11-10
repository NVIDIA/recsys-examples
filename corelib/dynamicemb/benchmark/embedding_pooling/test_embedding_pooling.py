import torch
import triton
from embedding_pooling import embedding_pooling


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


# ============================================================================
# Correctness Testing: Forward and Backward
# ============================================================================

# Unified tolerance for both forward and backward
TOLERANCE = 1e-4


def test_correctness():
    """
    Unified test for forward and backward correctness.
    Compares Triton implementation against PyTorch on the same data.
    """
    print("=" * 80)
    print("Forward and Backward Correctness Testing")
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
        offsets = torch.cat([torch.tensor([0], device="cuda"), lengths.cumsum(0)])

        # Generate embeddings with gradient tracking
        embeddings = torch.randn(
            total_embs, emb_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        embeddings_clone = embeddings.clone()

        for mode in ["sum", "mean"]:
            # warmup
            for i in range(10):
                triton_out = embedding_pooling(embeddings, offsets, mode)
                torch_out = embedding_pooling_torch(embeddings_clone, offsets, mode)
                grad_out_triton = torch.randn_like(triton_out)
                grad_out_torch = torch.randn_like(torch_out)

                _ = torch.autograd.grad(
                    triton_out, embeddings, grad_out_triton, retain_graph=True
                )
                _ = torch.autograd.grad(
                    torch_out, embeddings, grad_out_torch, retain_graph=True
                )
            torch.cuda.synchronize()
            # Triton forward
            with torch.cuda.nvtx.range("triton_forward"):
                triton_out = embedding_pooling(embeddings, offsets, mode)
            torch.cuda.synchronize()

            # PyTorch forward
            with torch.cuda.nvtx.range("torch_forward"):
                torch_out = embedding_pooling_torch(embeddings_clone, offsets, mode)
            torch.cuda.synchronize()

            # Compare forward outputs
            forward_diff = (triton_out - torch_out).abs().max().item()

            # ===== Backward Test =====
            # Generate gradient output
            grad_output = torch.randn_like(triton_out)
            torch.cuda.synchronize()

            # Triton backward
            with torch.cuda.nvtx.range("triton_backward"):
                triton_out.backward(grad_output, retain_graph=True)
            grad_triton = embeddings.grad.clone()
            embeddings.grad.zero_()
            torch.cuda.synchronize()

            # PyTorch backward
            with torch.cuda.nvtx.range("torch_backward"):
                torch_out.backward(grad_output)
            grad_torch = embeddings.grad.clone()
            embeddings.grad.zero_()
            torch.cuda.synchronize()

            # Compare backward gradients
            backward_diff = (grad_triton - grad_torch).abs().max().item()

            # Status
            forward_status = "✓" if forward_diff < TOLERANCE else "✗"
            backward_status = "✓" if backward_diff < TOLERANCE else "✗"

            print(
                f"  {mode:4s}: fwd={forward_diff:.2e} {forward_status}  "
                f"bwd={backward_diff:.2e} {backward_status}"
            )

            # Assert correctness
            assert (
                forward_diff < TOLERANCE
            ), f"Forward failed: diff = {forward_diff:.2e} (mode={mode}, case={name})"
            assert (
                backward_diff < TOLERANCE
            ), f"Backward failed: diff = {backward_diff:.2e} (mode={mode}, case={name})"

    # Edge case: empty segments
    print(f"\nEdge case (empty segments):")
    lengths = torch.tensor([5, 0, 3, 0, 1], device="cuda")
    total_embs = lengths.sum().item()
    embeddings = torch.randn(total_embs, 64, device="cuda", requires_grad=True)
    offsets = torch.cat([torch.tensor([0], device="cuda"), lengths.cumsum(0)])

    for mode in ["sum", "mean"]:
        # Forward
        triton_out = embedding_pooling(embeddings, offsets, mode)
        torch_out = embedding_pooling_torch(embeddings, offsets, mode)
        forward_diff = (triton_out - torch_out).abs().max().item()

        # Backward
        grad_output = torch.randn_like(triton_out)

        triton_out.backward(grad_output, retain_graph=True)
        grad_triton = embeddings.grad.clone()
        embeddings.grad.zero_()

        torch_out.backward(grad_output)
        grad_torch = embeddings.grad.clone()
        embeddings.grad.zero_()

        backward_diff = (grad_triton - grad_torch).abs().max().item()

        forward_status = "✓" if forward_diff < TOLERANCE else "✗"
        backward_status = "✓" if backward_diff < TOLERANCE else "✗"

        print(
            f"  {mode:4s}: fwd={forward_diff:.2e} {forward_status}  "
            f"bwd={backward_diff:.2e} {backward_status}"
        )

        assert (
            forward_diff < TOLERANCE
        ), f"Forward edge case failed: diff = {forward_diff:.2e} (mode={mode})"
        assert (
            backward_diff < TOLERANCE
        ), f"Backward edge case failed: diff = {backward_diff:.2e} (mode={mode})"

    print("\n✓ All forward and backward tests passed!")


if __name__ == "__main__":
    print("\n Embedding Pooling - Forward and Backward Testing\n")

    test_correctness()
