import torch
import triton
from embedding_pooling_kernel import (
    pooling_backward_kernel,
    pooling_parallel_reduce_kernel,
)


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


class PoolingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embeddings, offsets, pooling_mode):
        """
        Args:
            embeddings: [total_embeddings, embedding_dim] - All embeddings
            offsets: [num_segments + 1] - Segment boundaries
            pooling_mode: "sum" or "mean"
        Returns:
            pooled: [num_segments, embedding_dim] - Pooled embeddings

        Example:
            embeddings = [[e0], [e1], [e2], [e3], [e4]]  # 5 embeddings
            offsets = [0, 3, 5]  # 2 segments: [0:3] and [3:5]

            output[0] = mean([e0, e1, e2])
            output[1] = mean([e3, e4])
        """

        embeddings = embeddings.contiguous()
        offsets = offsets.contiguous()

        num_segs = offsets.shape[0] - 1
        emb_dim = embeddings.shape[1]

        output = torch.empty(
            (num_segs, emb_dim), dtype=embeddings.dtype, device=embeddings.device
        )

        mode = 0 if pooling_mode == "sum" else 1
        grid = (num_segs,)

        autotune_num_segments = prev_power_of_2(num_segs)

        pooling_parallel_reduce_kernel[grid](
            embeddings_ptr=embeddings,
            offsets_ptr=offsets,
            output_ptr=output,
            embedding_dim=emb_dim,
            num_segments=num_segs,
            pooling_mode=mode,
            autotune_num_segments=autotune_num_segments,
        )

        ctx.save_for_backward(offsets)
        ctx.pooling_mode = pooling_mode
        ctx.emb_dim = emb_dim
        ctx.total_embs = embeddings.size(0)
        ctx.num_segs = num_segs

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: Scatter gradients back to embeddings.

        Args:
            grad_output: [num_segments, embedding_dim] - Gradient w.r.t. pooled output

        Returns:
            grad_embeddings: [total_embeddings, embedding_dim] - Gradient w.r.t. embeddings
            None: No gradient for offsets
            None: No gradient for pooling_mode
        """
        (offsets,) = ctx.saved_tensors
        pooling_mode = ctx.pooling_mode
        emb_dim = ctx.emb_dim
        total_embs = ctx.total_embs
        num_segs = ctx.num_segs

        grad_output = grad_output.contiguous()
        offsets = offsets.contiguous()

        grad_embeddings = torch.empty(
            (total_embs, emb_dim), dtype=grad_output.dtype, device=grad_output.device
        )

        mode = 0 if pooling_mode == "sum" else 1

        autotune_num_segments = prev_power_of_2(num_segs)

        grid = (num_segs,)

        pooling_backward_kernel[grid](
            grad_output_ptr=grad_output,
            offsets_ptr=offsets,
            grad_input_ptr=grad_embeddings,
            embedding_dim=emb_dim,
            num_segments=num_segs,
            pooling_mode=mode,
            autotune_num_segments=autotune_num_segments,
        )
        return grad_embeddings, None, None


def embedding_pooling(
    embeddings: torch.Tensor, offsets: torch.Tensor, pooling_mode: str = "mean"
) -> torch.Tensor:
    """
    Args:
        embeddings: [total_embeddings, embedding_dim] - All embeddings
        offsets: [num_segments + 1] - Segment boundaries
        pooling_mode: "sum" or "mean"
    Returns:
        pooled: [num_segments, embedding_dim] - Pooled embeddings
    """
    assert pooling_mode in ["mean", "sum"]
    assert embeddings.dim() == 2 and offsets.dim() == 1
    assert embeddings.is_cuda and offsets.is_cuda

    return PoolingFunction.apply(embeddings, offsets, pooling_mode)
