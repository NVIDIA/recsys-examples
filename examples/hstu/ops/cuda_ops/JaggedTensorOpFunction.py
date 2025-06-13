import math
import torch
from typing import List, Tuple
import fbgemm_gpu
from torchrec.sparse.jagged_tensor import JaggedTensor
from ops.length_to_offsets import length_to_complete_offsets
import hstu_cuda_ops

class _JaggedTensorOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, offsets_list: List[torch.Tensor], max_seqlens: List[int], max_seqlen: int, *values_list):

        if len(offsets_list) == 1:
            single_offsets = offsets_list[0]
            lengths = single_offsets[1:] - single_offsets[:-1]
            ctx.mark_non_differentiable(lengths)
            return values_list[0], lengths
        
        dim_list = [v.size(-1) for v in values_list]
        assert all(dim == dim_list[0] for dim in dim_list), "All tensors must have the same value dimension"
        if isinstance(max_seqlen, torch.Tensor):    
            max_seqlen = int(max_seqlen.item())
        with torch.cuda.nvtx.range("Calculate merged offsets", color="purple"):
            merged_offsets = torch.sum(torch.stack(offsets_list), dim=0, dtype=offsets_list[0].dtype)
            # merged_offsets = offsets_list[0].clone()
            # for offset_tensor in offsets_list[1:]:
            #     merged_offsets.add_(offset_tensor) 
        # import pdb; pdb.set_trace()

        with torch.cuda.nvtx.range("Calculate merged lengths", color="purple"):
            # total_length = merged_offsets[-1].item()
            total_length = sum(v.size(0) for v in values_list)

            hidden_dim = values_list[0].size(-1)

            merged_lengths = torch.diff(merged_offsets)
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
        batch_size = offsets_list[0].size(0) - 1

        with torch.cuda.nvtx.range("calculate seqlen_per_block", color="purple"):
        # hidden_dim越大，seqlen_per_block越小
            seqlen_per_block = (8 if hidden_dim <= 128 else 
                                    4 if hidden_dim <= 256 else 
                                    2 if hidden_dim <= 512 else 1)
            blocks_per_batch = (max_seqlen + seqlen_per_block - 1) // seqlen_per_block
            num_tensors = len(offsets_list)
            total_blocks = batch_size * blocks_per_batch * num_tensors
            block_workloads = torch.empty(
                total_blocks,
                dtype=torch.int32,
                device=values_list[0].device,
            )
        with torch.cuda.nvtx.range("calculate blocks workload", color="purple"):
            hstu_cuda_ops.compute_block_workloads(offsets_list, seqlen_per_block, max_seqlen, block_workloads)
        with torch.cuda.nvtx.range("workload offset cumsum", color="orange"):
            workload_offset = length_to_complete_offsets(block_workloads)


        with torch.cuda.nvtx.range("Cpp part forward", color="purple"):
            hstu_cuda_ops.concat_2D_jagged_tensors_forward(
                values_list, 
                offsets_list, 
                seqlen_per_block,
                max_seqlen,
                workload_offset,
                merged_values,
                merged_offsets,
            )
        
        # 保存张量变量
        ctx.save_for_backward(merged_offsets, workload_offset, *offsets_list)
        # 保存非张量变量
        ctx.seqlen_per_block = seqlen_per_block
        ctx.max_seqlen = max_seqlen
        return merged_values, merged_lengths


    @staticmethod
    def backward(ctx, grad_output, grad_lengths):
        # 处理特殊情况：只有一个张量时，直接返回梯度
        if len(ctx.saved_tensors) == 0:
            # 这是 len(offsets_list) == 1 的情况，直接返回梯度
            return None, None, None, grad_output
        
        # 获取保存的张量变量
        merged_offsets, workload_offset, *offsets_list = ctx.saved_tensors
        # 获取保存的非张量变量
        seqlen_per_block = ctx.seqlen_per_block
        max_seqlen = ctx.max_seqlen
        
        grad_input = hstu_cuda_ops.concat_2D_jagged_tensors_backward(grad_output, grad_lengths, seqlen_per_block, max_seqlen, workload_offset, offsets_list, merged_offsets)
        return None, None, None, *grad_input

def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range (0, 10**9)
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()

def jagged_2D_tensor_concat(values_list: List[torch.Tensor], offsets_list: List[torch.Tensor], max_seqlens: List[int], max_seqlen: int):
    assert len(values_list) == len(offsets_list)
    assert all(values_list[0].dtype == v.dtype for v in values_list)
    assert all(values_list[0].device == v.device for v in values_list)
    #Todo: need to support non contiguous case in the future.
    # assert all(v.is_contiguous() for v in values_list)

    values_list = [switch_to_contiguous_if_needed(v) for v in values_list]
    
    # Handle case where values_list length > 128 by batching
    batch_size = 128
    if len(values_list) <= batch_size:
        return _JaggedTensorOpFunction.apply(offsets_list, max_seqlens, max_seqlen, *values_list)
    
    result_values, result_lengths = None, None
     
    for i in range(0, len(values_list), batch_size):
        end_idx = min(i + batch_size, len(values_list))
        batch_values = values_list[i:end_idx]
        batch_offsets = offsets_list[i:end_idx] 
        batch_max_seqlens = max_seqlens[i:end_idx]
        batch_max_seqlen = max(batch_max_seqlens)
        
        batch_result = _JaggedTensorOpFunction.apply(batch_offsets, batch_max_seqlens, batch_max_seqlen, *batch_values)
        
        if result_values is None:
            # First batch
            result_values, result_lengths = batch_result
        else:
            # Merge with previous result using 2-tensor concat
            prev_offsets = torch.cat([torch.tensor([0], device=result_lengths.device, dtype=torch.int32), 
                                    torch.cumsum(result_lengths, dim=0, dtype=torch.int32)])
            curr_offsets = torch.cat([torch.tensor([0], device=batch_result[1].device, dtype=torch.int32), 
                                    torch.cumsum(batch_result[1], dim=0, dtype=torch.int32)])
            
            prev_max_seqlen = torch.max(result_lengths).item()
            curr_max_seqlen = torch.max(batch_result[1]).item()
            
            # Merge two results
            result_values, result_lengths = _JaggedTensorOpFunction.apply(
                [prev_offsets, curr_offsets], 
                [prev_max_seqlen, curr_max_seqlen], 
                max(prev_max_seqlen, curr_max_seqlen),
                result_values, batch_result[0]
            )
    
    return result_values, result_lengths

