from typing import List

import hstu_cuda_ops
import torch
from commons.ops.length_to_offsets import length_to_complete_offsets


class _JaggedTensorOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        offsets_list: List[torch.Tensor],
        max_seqlen: int,
        *values_list,
    ):
        # Early validation to prevent edge cases
        if len(offsets_list) == 0 or len(values_list) == 0:
            raise ValueError("offsets_list and values_list cannot be empty")

        # Check batch_size
        batch_size = offsets_list[0].size(0) - 1
        if batch_size <= 0:
            raise ValueError(
                f"Invalid batch_size: {batch_size}. offsets tensor size: {offsets_list[0].size()}"
            )

        if len(offsets_list) == 1:
            single_offsets = offsets_list[0]
            lengths = single_offsets[1:] - single_offsets[:-1]
            ctx.mark_non_differentiable(lengths)
            return values_list[0], lengths

        dim_list = [v.size(-1) for v in values_list]
        assert all(
            dim == dim_list[0] for dim in dim_list
        ), "All tensors must have the same value dimension"

        with torch.cuda.nvtx.range("Calculate merged offsets", color="purple"):
            merged_offsets = torch.sum(
                torch.stack(offsets_list), dim=0, dtype=offsets_list[0].dtype
            )

        with torch.cuda.nvtx.range("Calculate merged lengths", color="purple"):
            total_length = sum(v.size(0) for v in values_list)

            hidden_dim = values_list[0].size(-1)

            merged_lengths = torch.diff(merged_offsets)
            ctx.mark_non_differentiable(merged_lengths)

        with torch.cuda.nvtx.range("merged values mem alloc", color="purple"):
            merged_values = torch.empty(
                (total_length, hidden_dim),
                dtype=values_list[0].dtype,
                device=values_list[0].device,
            ).requires_grad_(True)

        device_properties = torch.cuda.get_device_properties(0)
        BLOCK_SIZE = 256
        GRID_SIZE = int(
            device_properties.multi_processor_count
            * (device_properties.max_threads_per_multi_processor / BLOCK_SIZE)
        )

        with torch.cuda.nvtx.range("calculate seqlen_per_block", color="purple"):
            # the larger hidden_dim is, the smaller seqlen_per_block becomes
            seqlen_per_block = (
                8
                if hidden_dim <= 128
                else 4
                if hidden_dim <= 256
                else 2
                if hidden_dim <= 512
                else 1
            )
            blocks_per_batch = (max_seqlen + seqlen_per_block - 1) // seqlen_per_block
            num_tensors = len(offsets_list)
            total_blocks = batch_size * blocks_per_batch * num_tensors
            block_workloads = torch.empty(
                total_blocks,
                dtype=torch.int64,
                device=values_list[0].device,
            )
            # warp configuration: ensure not exceeding 1024 threads, each warp processes 1 sequence
            target_warps = min(32, max(1, seqlen_per_block))
            threads = min(BLOCK_SIZE, target_warps * 32)
            blocks = min(GRID_SIZE, total_blocks)

        # Handle max_seqlen == 0 case to prevent division by zero in CUDA kernel
        if max_seqlen == 0:
            # All sequences are empty, so workloads should be zero
            block_workloads.zero_()
            workload_offset = length_to_complete_offsets(block_workloads)
            # merged_values is already correctly allocated as empty tensor when total_length == 0
        else:
            with torch.cuda.nvtx.range("calculate blocks workload", color="purple"):
                hstu_cuda_ops.compute_block_workloads(
                    offsets_list, seqlen_per_block, max_seqlen, block_workloads
                )
            with torch.cuda.nvtx.range("workload offset cumsum", color="orange"):
                workload_offset = length_to_complete_offsets(block_workloads)

            with torch.cuda.nvtx.range("Cpp part forward", color="purple"):
                hstu_cuda_ops.concat_2D_jagged_tensors_forward(
                    values_list,
                    offsets_list,
                    seqlen_per_block,
                    max_seqlen,
                    total_blocks,
                    blocks,
                    threads,
                    workload_offset,
                    merged_values,
                    merged_offsets,
                )

        # save tensor variables
        ctx.save_for_backward(merged_offsets, workload_offset, *offsets_list)
        # save non-tensor variables
        ctx.seqlen_per_block = seqlen_per_block
        ctx.max_seqlen = max_seqlen
        ctx.blocks = blocks
        ctx.threads = threads
        ctx.total_blocks = total_blocks
        ctx.input_shapes = [v.shape for v in values_list]
        return merged_values, merged_lengths

    @staticmethod
    def backward(ctx, grad_output, grad_lengths):
        # handle special case: when there's only one tensor, return gradient directly
        if len(ctx.saved_tensors) == 0:
            # len(offsets_list) == 1, return gradient directly
            return None, None, grad_output

        # get saved tensor variables
        merged_offsets, workload_offset, *offsets_list = ctx.saved_tensors
        # get saved non-tensor variables
        seqlen_per_block = ctx.seqlen_per_block
        max_seqlen = ctx.max_seqlen
        blocks = ctx.blocks
        threads = ctx.threads
        total_blocks = ctx.total_blocks
        grad_inputs = [
            torch.empty(shape, dtype=grad_output.dtype, device=grad_output.device)
            for shape in ctx.input_shapes
        ]

        # Handle max_seqlen == 0 case - return zero gradients
        if max_seqlen == 0:
            # Zero out all gradient inputs since no computation was done
            for grad_input in grad_inputs:
                grad_input.zero_()
        else:
            with torch.cuda.nvtx.range("CUDA Backward", color="red"):
                hstu_cuda_ops.concat_2D_jagged_tensors_backward(
                    grad_output,
                    grad_lengths,
                    seqlen_per_block,
                    max_seqlen,
                    total_blocks,
                    blocks,
                    threads,
                    workload_offset,
                    grad_inputs,
                    offsets_list,
                    merged_offsets,
                )
        return (None, None, *grad_inputs)


def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range (0, 10**9)
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


def jagged_2D_tensor_concat(
    values_list: List[torch.Tensor],
    offsets_list: List[torch.Tensor],
    max_seqlens: List[int],
):
    assert len(values_list) == len(offsets_list)
    assert all(values_list[0].dtype == v.dtype for v in values_list)
    assert all(values_list[0].device == v.device for v in values_list)
    # Todo: need to support non contiguous case in the future.
    # assert all(v.is_contiguous() for v in values_list)

    values_list = [switch_to_contiguous_if_needed(v) for v in values_list]

    max_seqlen = max(max_seqlens)
    # Handle case where values_list length > 128 by batching
    batch_size = 128
    if len(values_list) <= batch_size:
        return _JaggedTensorOpFunction.apply(offsets_list, max_seqlen, *values_list)

    result_values, result_lengths = None, None

    for i in range(0, len(values_list), batch_size):
        end_idx = min(i + batch_size, len(values_list))
        batch_values = values_list[i:end_idx]
        batch_offsets = offsets_list[i:end_idx]
        batch_max_seqlens = max_seqlens[i:end_idx]
        batch_max_seqlen = max(batch_max_seqlens)

        batch_result = _JaggedTensorOpFunction.apply(
            batch_offsets, batch_max_seqlen, *batch_values
        )

        if result_values is None:
            # First batch
            result_values, result_lengths = batch_result
            prev_max_seqlen = sum(batch_max_seqlens)
        else:
            # Merge with previous result using 2-tensor concat
            prev_offsets = length_to_complete_offsets(result_lengths)
            curr_offsets = length_to_complete_offsets(batch_result[1])

            max_seqlen = prev_max_seqlen + sum(batch_max_seqlens)
            # Merge two results
            result_values, result_lengths = _JaggedTensorOpFunction.apply(
                [prev_offsets, curr_offsets],
                max_seqlen,
                result_values,
                batch_result[0],
            )
            prev_max_seqlen = max_seqlen

    return result_values, result_lengths
