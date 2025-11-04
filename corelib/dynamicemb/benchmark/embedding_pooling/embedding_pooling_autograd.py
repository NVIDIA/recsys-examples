import torch
import triton
from embedding_pooling_kernel import pooling_parallel_reduce_kernel
from embedding_pooling_kernel import pooling_backward_kernel
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
        assert embeddings.dim() == 2 and offsets.dim() == 1
        assert pooling_mode in ["sum", "mean"]
        assert embeddings.is_cuda and offsets.is_cuda
        
        if not embeddings.is_contiguous():
            embeddings = embeddings.contiguous()
        if not offsets.is_contiguous():
            offsets = offsets.contiguous()
        
        num_segs = offsets.shape[0] - 1
        emb_dim = embeddings.shape[1]
        
        output = torch.empty(
            (num_segs, emb_dim),
            dtype=embeddings.dtype,
            device=embeddings.device
        )
        
        mode = 0 if pooling_mode == "sum" else 1
        grid = (num_segs,)
        
        pooling_parallel_reduce_kernel[grid](
            embeddings_ptr=embeddings,
            offsets_ptr=offsets,
            output_ptr=output,
            embedding_dim=emb_dim,
            num_segments=num_segs,
            pooling_mode=mode,
        )
        
        ctx.save_for_backward(offsets)
        ctx.pooling_mode = pooling_mode
        ctx.emb_dim = emb_dim
        
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
        offsets, = ctx.saved_tensors
        pooling_mode = ctx.pooling_mode
        emb_dim = ctx.emb_dim
        
        assert grad_output.dim() == 2 and offsets.dim() == 1
        assert pooling_mode in ["sum", "mean"]
        assert grad_output.is_cuda and offsets.is_cuda
        
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        if not offsets.is_contiguous():
            offsets = offsets.contiguous()
        
        num_segs = offsets.shape[0] - 1
        emb_dim = grad_output.shape[1]
        total_embs = offsets[-1].item()
        
        grad_embeddings = torch.empty(
            (total_embs, emb_dim),
            dtype=grad_output.dtype,
            device=grad_output.device
        )
        
        mode = 0 if pooling_mode == "sum" else 1
        
        grid = (num_segs,)
        
        pooling_backward_kernel[grid](
            grad_output_ptr=grad_output,
            offsets_ptr=offsets,
            grad_input_ptr=grad_embeddings,
            embedding_dim=emb_dim,
            num_segments=num_segs,
            pooling_mode=mode,
        )  

        return grad_embeddings, None, None


def embedding_pooling_with_grad(
    embeddings: torch.Tensor,
    offsets: torch.Tensor,
    pooling_mode: str = "mean"
) -> torch.Tensor:
    """
    Args:
        embeddings: [total_embeddings, embedding_dim] - All embeddings
        offsets: [num_segments + 1] - Segment boundaries
        pooling_mode: "sum" or "mean"
    Returns:
        pooled: [num_segments, embedding_dim] - Pooled embeddings
    """
    return PoolingFunction.apply(embeddings, offsets, pooling_mode)


