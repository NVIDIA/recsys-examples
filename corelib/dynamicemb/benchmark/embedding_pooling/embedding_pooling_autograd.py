import torch
import triton
from embedding_pooling import pooling_parallel_reduce_kernel
from embedding_pooling_backward import embedding_pooling_backward_triton
class PoolingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embeddings, offsets, pooling_mode):
        """
        High-performance embedding pooling forward.
        
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
        assert embeddings.is_contiguous() and offsets.is_contiguous()
        assert embeddings.is_cuda and offsets.is_cuda
        
        num_segs = offsets.shape[0] - 1
        emb_dim = embeddings.shape[1]
        
        # Create output tensor
        output = torch.empty(
            (num_segs, emb_dim),
            dtype=embeddings.dtype,
            device=embeddings.device
        )
        
        mode = 0 if pooling_mode == "sum" else 1
        grid = (num_segs,)
        
        # Call Triton kernel for forward
        pooling_parallel_reduce_kernel[grid](
            embeddings_ptr=embeddings,
            offsets_ptr=offsets,
            output_ptr=output,
            embedding_dim=emb_dim,
            num_segments=num_segs,
            pooling_mode=mode,
        )
        
        # Save for backward
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
        
        # Calculate segment information
        lengths = offsets[1:] - offsets[:-1]  # [num_segments]
        num_segs = lengths.shape[0]
        total_embs = offsets[-1].item()
        
        # Create segment IDs for each embedding
        # seg_ids[i] tells which segment embedding i belongs to
        # Example: lengths=[3,2,5] → seg_ids=[0,0,0,1,1,2,2,2,2,2]
        seg_ids = torch.repeat_interleave(
            torch.arange(num_segs, device=offsets.device),
            lengths
        )
        # Scatter gradient: each embedding gets gradient from its segment
        # This is the inverse of pooling (expand instead of reduce)
        grad_embeddings = grad_output[seg_ids]  # [total_embeddings, embedding_dim]
        
        # For mean pooling, gradient needs to be divided by segment length
        if pooling_mode == "mean":
            # Get length for each embedding
            lengths_per_emb = lengths[seg_ids]  # [total_embeddings]
            # Divide gradient by length
            grad_embeddings = grad_embeddings / lengths_per_emb.unsqueeze(1).float()
        
        # Return gradients: (embeddings, offsets, pooling_mode)
        return grad_embeddings, None, None


def embedding_pooling_with_grad(
    embeddings: torch.Tensor,
    offsets: torch.Tensor,
    pooling_mode: str = "mean"
) -> torch.Tensor:
    """
    User-friendly wrapper for PoolingFunction.
    
    This function supports autograd (forward and backward).
    """
    return PoolingFunction.apply(embeddings, offsets, pooling_mode)


# ============================================================================
# Testing
# ============================================================================

def test():
    """Test forward correctness."""
    print("=" * 80)
    print("Test 1: Forward Correctness")
    print("=" * 80)
    
    torch.manual_seed(42)
    embeddings = torch.randn(10, 128, device='cuda', requires_grad=True)
    offsets = torch.tensor([0, 3, 7, 10], device='cuda')
    pooling_mode = "mean"
    output = PoolingFunction.apply(embeddings, offsets, pooling_mode)
    print(output.shape)
    print(output)
    output.backward(torch.randn_like(output))
    print(embeddings.grad.shape)
    print(embeddings.grad)

if __name__ == "__main__":
    test()