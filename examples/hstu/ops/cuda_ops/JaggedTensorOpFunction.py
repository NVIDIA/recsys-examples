import math
import torch
from typing import List, Tuple
import fbgemm_gpu
from torchrec.sparse.jagged_tensor import JaggedTensor

import hstu_cuda_ops

class _JaggedTensorOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, offsets_list: List[torch.Tensor], max_seqlens: List[int], *values_list):

        if len(offsets_list) == 1:
            single_offsets = offsets_list[0]
            lengths = single_offsets[1:] - single_offsets[:-1]
            ctx.mark_non_differentiable(lengths)
            return values_list[0], lengths
        
        dim_list = [v.size(-1) for v in values_list]
        assert all(dim == dim_list[0] for dim in dim_list), "All tensors must have the same value dimension"

        with torch.cuda.nvtx.range("Calculate merged offsets", color="purple"):
            merged_offsets = offsets_list[0].clone()
            for offset_tensor in offsets_list[1:]:
                merged_offsets.add_(offset_tensor)
        
        ctx.save_for_backward(merged_offsets, *offsets_list)
        # total_length = merged_offsets[-1].item()
        total_length = sum(v.size(0) for v in values_list)

        hidden_dim = values_list[0].size(-1)
        merged_lengths = []
        for offsets_tensor in offsets_list:
            lengths = offsets_tensor[1:] - offsets_tensor[:-1]
            merged_lengths.append(lengths)

        merged_lengths = torch.sum(
            torch.concat([lengths.view(-1, 1) for lengths in merged_lengths], dim=1), dim=1)
        ctx.mark_non_differentiable(merged_lengths)

        with torch.cuda.nvtx.range("merged values mem alloc", color="purple"):
            merged_values = (
                torch.empty(
                    (total_length, hidden_dim),
                    dtype=values_list[0].dtype,
                    device=values_list[0].device,
                )
                .requires_grad_(True)
            ) 
        # print(f"values_list = {values_list}")
        # print(f"values_list[0].dtype = {values_list[0].dtype}")
        # print(f"merged_values.dtype = {merged_values.dtype}")
        # import pdb; pdb.set_trace()
        with torch.cuda.nvtx.range("Cpp part forward", color="purple"):
            hstu_cuda_ops.concat_2D_jagged_tensors_forward(
                values_list, 
                offsets_list, 
                merged_values,
                merged_offsets
            )
        
        return merged_values, merged_lengths


    @staticmethod
    def backward(ctx, grad_output, grad_lengths):
        merged_offsets, *offsets_list = ctx.saved_tensors
        grad_input = hstu_cuda_ops.concat_2D_jagged_tensors_backward(grad_output, grad_lengths, offsets_list, merged_offsets)
        return None, None, *grad_input

def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range (0, 10**9)
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()

def jagged_2D_tensor_concat(values_list: List[torch.Tensor], offsets_list: List[torch.Tensor], max_seqlens: List[int]):
    assert len(values_list) == len(offsets_list)
    assert all(values_list[0].dtype == v.dtype for v in values_list)
    assert all(values_list[0].device == v.device for v in values_list)
    #Todo: need to support non contiguous case in the future.
    # assert all(v.is_contiguous() for v in values_list)
    values_list = [switch_to_contiguous_if_needed(v) for v in values_list]
    return _JaggedTensorOpFunction.apply(offsets_list, max_seqlens, *values_list)

