import triton
import triton.language as tl


# ============================================================================
# Forward Kernel
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64, "BLOCK_N": 16}, num_warps=2),
        triton.Config({"BLOCK_D": 64, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_D": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 64, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 128, "BLOCK_N": 16}, num_warps=2),
        triton.Config({"BLOCK_D": 128, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_D": 128, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_D": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_D": 256, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_D": 256, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_D": 256, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_D": 256, "BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_D": 512, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_D": 512, "BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_D": 512, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_D": 512, "BLOCK_N": 128}, num_warps=8),
    ],
    key=["embedding_dim", "autotune_num_segments"],
    # TODO: add prune_configs_by to prune obviously unreasonable configs
    # prune_configs_by={
    #     'early_config_prune': early_config_prune,
    # }
)
@triton.jit
def pooling_parallel_reduce_kernel(
    embeddings_ptr,
    offsets_ptr,
    output_ptr,
    embedding_dim: tl.constexpr,
    num_segments: tl.constexpr,
    autotune_num_segments: tl.constexpr,
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
                other=0.0,
            )

            # Parallel sum along sequence axis
            acc += tl.sum(embs, axis=0)

        if pooling_mode == 1:
            acc = acc / length.to(tl.float32)

        tl.store(output_ptr + seg_id * embedding_dim + d_idx, acc, mask=d_mask)


# ============================================================================
# Backward Kernel
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64, "BLOCK_N": 16}, num_warps=2),
        triton.Config({"BLOCK_D": 64, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_D": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 64, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 128, "BLOCK_N": 16}, num_warps=2),
        triton.Config({"BLOCK_D": 128, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_D": 128, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_D": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_D": 256, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_D": 256, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_D": 256, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_D": 256, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_D": 512, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_D": 512, "BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_D": 512, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_D": 512, "BLOCK_N": 128}, num_warps=8),
    ],
    key=["embedding_dim", "autotune_num_segments"],
)
@triton.jit
def pooling_backward_kernel(
    grad_output_ptr,  # [num_segments, embedding_dim]
    offsets_ptr,  # [num_segments + 1]
    grad_input_ptr,  # [total_embeddings, embedding_dim]
    embedding_dim: tl.constexpr,
    num_segments: tl.constexpr,
    autotune_num_segments: tl.constexpr,
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
    seg_id = tl.program_id(0)

    if seg_id >= num_segments:
        return

    start = tl.load(offsets_ptr + seg_id)
    end = tl.load(offsets_ptr + seg_id + 1)
    length = end - start

    if length == 0:
        return

    if pooling_mode == 1:
        scale = 1.0 / length.to(tl.float32)
    else:
        scale = 1.0

    # Process embeddings in outer loop, dimensions in inner loop, for memory coalescing
    for n_off in range(0, length, BLOCK_N):
        n_idx = n_off + tl.arange(0, BLOCK_N)
        n_mask = n_idx < length

        row_idx = start + n_idx  # [BLOCK_N]

        # Process each dimension block
        for d_off in range(0, embedding_dim, BLOCK_D):
            d_idx = d_off + tl.arange(0, BLOCK_D)
            d_mask = d_idx < embedding_dim

            # Load gradient from pooled output
            grad_offset = seg_id * embedding_dim + d_idx
            grad = tl.load(grad_output_ptr + grad_offset, mask=d_mask, other=0.0)

            if pooling_mode == 1:
                grad = grad * scale

            # 2D store: [BLOCK_N, BLOCK_D]
            indices = row_idx[:, None] * embedding_dim + d_idx[None, :]
            grad_broadcasted = grad[None, :]  # [1, BLOCK_D] -> [BLOCK_N, BLOCK_D]

            tl.store(
                grad_input_ptr + indices,
                grad_broadcasted,
                mask=n_mask[:, None] & d_mask[None, :],
            )
