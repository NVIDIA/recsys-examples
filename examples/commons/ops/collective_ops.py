# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional

import torch
import torch.distributed as dist
from commons.ops.length_to_offsets import length_to_complete_offsets
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


def _split_along_first_dim(input_, pg: Optional[dist.ProcessGroup] = None):
    """
    Split the tensor along its first dimension and keep the corresponding slice.

    Args:
        input_ (torch.Tensor): Input tensor to be split.
        pg (Optional[dist.ProcessGroup]): Process group for distributed operations.

    Returns:
        torch.Tensor: Sliced tensor.
    """

    world_size = dist.get_world_size(pg)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert (
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = dist.get_rank(pg)
    dim_offset = rank * local_dim_size

    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def _split_along_last_dim(input_, pg: Optional[dist.ProcessGroup] = None):
    """
    Split the tensor along its last dimension and keep the corresponding slice.

    Args:
        input_ (torch.Tensor): Input tensor to be split.
        pg (Optional[dist.ProcessGroup]): Process group for distributed operations.

    Returns:
        torch.Tensor: Sliced tensor.
    """

    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    dim_size = input_.size()[-1]
    assert (
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    dim_offset = rank * local_dim_size
    output = input_[..., dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def _splitv_along_first_dim(
    input_, split_offsets: List[int], pg: Optional[dist.ProcessGroup] = None
):
    """
    Split the tensor along its first dimension based on split offsets and keep the corresponding slice.

    Args:
        input_ (torch.Tensor): Input tensor to be split.
        split_offsets (List[int]): List of split offsets.
        pg (Optional[dist.ProcessGroup]): Process group for distributed operations.

    Returns:
        torch.Tensor: Sliced tensor.
    """

    world_size = dist.get_world_size(pg)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    # Split along first dimension.
    aggrated_dim = input_.size()[0]
    assert (
        len(split_offsets) == world_size + 1
    ), "number of split should match given split_offsets"
    assert (
        aggrated_dim == split_offsets[-1]
    ), "First dimension of the tensor should match the split_offsets[-1]"
    rank = dist.get_rank(pg)
    dim_start = split_offsets[rank]
    dim_end = split_offsets[rank + 1]

    output = input_[dim_start:dim_end, ...].contiguous()
    # the view changes the stride!!
    output = output.view(output.shape)

    return output


def _gather_along_first_dim(input_, pg: Optional[dist.ProcessGroup] = None):
    """
    Gather tensors and concatenate along the first dimension.

    Args:
        input_ (torch.Tensor): A tensor to be gathered.
        pg (Optional[dist.ProcessGroup]): Process group for distributed operations.

    Returns:
        torch.Tensor: Gathered tensor.
    """

    world_size = dist.get_world_size(pg)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=pg)
    return output


def _gather_along_last_dim(input_, pg: Optional[dist.ProcessGroup] = None):
    """
    Gather tensors and concatenate along the last dimension.
    """
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return input_
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    # allgather along the first dim and then concatenate along the last dim
    torch.distributed.all_gather(tensor_list, input_.contiguous(), group=pg)
    output = torch.cat(tensor_list, dim=-1)
    return output


def _gatherv_along_first_dim(input_, pg: Optional[dist.ProcessGroup] = None):
    """
    Gatherv tensors and concatenate along the first dimension.

    Args:
        input_ (torch.Tensor): A tensor to be gathered. The first dim of tensors on different rank may vary.
        pg (Optional[dist.ProcessGroup]): Process group for distributed operations.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Gathered tensor and offsets tensor.
    """

    world_size = dist.get_world_size(pg)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_, None
    input_dim0_tensor = torch.tensor(
        [input_.size(0)], device=input_.device, dtype=torch.int64
    )
    output_dim0_tensor = torch.empty(
        size=(world_size,),
        dtype=input_dim0_tensor.dtype,
        device=input_dim0_tensor.device,
    )
    torch.distributed.all_gather_into_tensor(
        output_dim0_tensor, input_dim0_tensor.contiguous(), group=pg
    )
    # No barrier needed: PyTorch's ProcessGroupNCCL already synchronizes the
    # default CUDA stream with the NCCL stream via cudaStreamWaitEvent after
    # each non-async collective.  The subsequent .cpu() (which runs on the
    # default stream) will therefore wait for all_gather_into_tensor to finish.
    variable_dim0 = output_dim0_tensor.cpu()
    dim_size = list(input_.size())
    tensor_list = []
    tensor_list = [
        torch.empty(
            size=([dim0] + dim_size[1:]), dtype=input_.dtype, device=input_.device
        )
        for dim0 in variable_dim0
    ]
    torch.distributed.all_gather(tensor_list, input_.contiguous(), group=pg)
    return torch.cat(tensor_list).contiguous(), length_to_complete_offsets(
        output_dim0_tensor
    )


class AllGatherFirstDimFromRegion(torch.autograd.Function):
    """
    All gather tensor from given process group and concatenate along the first dimension.

    Args:
        input_ (torch.Tensor): Input tensor. The first dim of tensors on different ranks must be the same.
        pg (torch.distributed.ProcessGroup): Process group for distributed operations.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend='nccl', world_size=2)
        >>> input_tensor = torch.rand(3, 4).cuda()
        >>> gathered_tensor = AllGatherFirstDimFromRegion.apply(input_tensor)
        >>> print(gathered_tensor.size())
        torch.Size([6, 4])
    """

    @staticmethod
    def symbolic(graph, input_, pg: torch.distributed.ProcessGroup):
        """"""
        return _gather_along_first_dim(input_, pg)

    @staticmethod
    def forward(ctx, input_, pg: torch.distributed.ProcessGroup):
        """"""
        ctx.pg = pg
        return _gather_along_first_dim(input_, pg)

    @staticmethod
    def backward(ctx, grad_output):
        """"""
        pg = ctx.pg
        return _split_along_first_dim(grad_output, pg), None


class AllGatherLastDimFromRegion(torch.autograd.Function):
    """
    All gather tensor from given process group and concatenate along the last dimension.

    Args:
        input_ (torch.Tensor): Input tensor. The last dim of tensors on different ranks must be the same.
        pg (torch.distributed.ProcessGroup): Process group for distributed operations.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend='nccl', world_size=2)
        >>> input_tensor = torch.rand(3, 4).cuda()
        >>> gathered_tensor = AllGatherLastDimFromRegion.apply(input_tensor)
        >>> print(gathered_tensor.size())
        torch.Size([3, 8])
    """

    @staticmethod
    def symbolic(graph, input_, pg: torch.distributed.ProcessGroup):
        """"""
        return _gather_along_last_dim(input_, pg)

    @staticmethod
    def forward(ctx, input_, pg: torch.distributed.ProcessGroup):
        """"""
        ctx.pg = pg
        return _gather_along_last_dim(input_, pg)

    @staticmethod
    def backward(ctx, grad_output):
        """"""
        pg = ctx.pg
        return _split_along_last_dim(grad_output, pg), None


class AllGathervFirstDimFromRegion(torch.autograd.Function):
    """
    All gatherv tensor from given process group and concatenate along the first dimension.
    This is specific to DMP to megatron DP.

    Args:
        input_ (torch.Tensor): Input tensor. The first dim of tensors on different ranks may vary.
        pg (torch.distributed.ProcessGroup): Process group for distributed operations.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend='nccl', world_size=2)
        >>> input_tensor = torch.rand(3, 4).cuda()
        >>> gathered_tensor, offsets = AllGathervFirstDimFromRegion.apply(input_tensor)
        >>> print(gathered_tensor.size())
        torch.Size([6, 4])
        >>> print(offsets)
        tensor([3, 3], device='cuda:0')
    """

    @staticmethod
    def symbolic(graph, input_, pg: torch.distributed.ProcessGroup):
        """"""
        value, offsets = _gatherv_along_first_dim(input_, pg)
        return value

    @staticmethod
    def forward(ctx, input_, pg: torch.distributed.ProcessGroup):
        """"""
        ctx.pg = pg
        value, offsets = _gatherv_along_first_dim(input_, pg)
        ctx.offsets = offsets
        return value

    @staticmethod
    def backward(ctx, grad_output):
        """"""
        pg = ctx.pg
        offsets = ctx.offsets
        ret_values = _splitv_along_first_dim(grad_output, offsets, pg)
        return ret_values, None


class SplitAlongFirstDimFromRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_, pg: torch.distributed.ProcessGroup):
        return _split_along_first_dim(input_, pg)

    @staticmethod
    def forward(ctx, input_, pg: torch.distributed.ProcessGroup):
        ctx.pg = pg
        return _split_along_first_dim(input_, pg)

    @staticmethod
    def backward(ctx, grad_output):
        pg = ctx.pg
        return _gather_along_first_dim(grad_output, pg), None


class SplitAlongLastDimFromRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_, pg: torch.distributed.ProcessGroup):
        return _split_along_last_dim(input_, pg)

    @staticmethod
    def forward(ctx, input_, pg: torch.distributed.ProcessGroup):
        ctx.pg = pg
        return _split_along_last_dim(input_, pg)

    @staticmethod
    def backward(ctx, grad_output):
        pg = ctx.pg
        return _gather_along_last_dim(grad_output, pg), None


split_along_last_dim = SplitAlongLastDimFromRegion.apply
split_along_first_dim = SplitAlongFirstDimFromRegion.apply
gather_along_first_dim = AllGatherFirstDimFromRegion.apply
gatherv_along_first_dim = AllGathervFirstDimFromRegion.apply
gather_along_last_dim = AllGatherLastDimFromRegion.apply


class ViewAsDtype(torch.autograd.Function):
    """
    View tensor as a different dtype. A differentiable datatype reinterpret cast function.

    Args:
        input_ (torch.Tensor): Input tensor.
        dst_dtype (torch.dtype): Destination dtype.

    Example:
        >>> import torch
        >>> input_tensor = torch.randn(3, 4).cuda().requires_grad_()
        >>> output_tensor = ViewAsDtype.apply(input_tensor, torch.float16)
        >>> print(output_tensor.dtype, out_tensor.size())
        torch.float, torch.Size([3, 8])
        >>> grad_tensor = torch.randn_like(output_tensor
        >>> output_tensor.backward(grad_tensor)
        >>> print(input_tensor.grad.dtype, input_tensor.grad.size())
        torch.float, torch.Size([3, 4])
    """

    @staticmethod
    def forward(ctx, input_, dst_dtype: torch.dtype = torch.float):
        """"""
        ctx.dtype = input_.dtype
        return input_.contiguous().view(dst_dtype)

    @staticmethod
    def backward(ctx, grad):
        """"""
        return grad.contiguous().view(grad.shape).view(ctx.dtype), None


view_as_dtype = ViewAsDtype.apply


def grouped_allgatherv_tensor_list(
    value_list: List[torch.Tensor],
    pg: Optional[dist.ProcessGroup] = None,
):
    """
    A differentiable allgatherv function. To reduce the collective calls, all tensors in `value_list`
    will be viewed as bfloat16 and then concatenated as single tensor. The input tensors on one rank must share
    the same seqlen_sum.

    Args:
        value_list (List[torch.Tensor]): List of tensors to be gathered.
        pg (Optional[dist.ProcessGroup]): Process group for distributed operations.

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor]: List of gathered tensors and output sequence length tensor.

    Example:
      >>> # We have 1 process groups, 2 ranks.
      >>> import torch
      >>> import torch.distributed as dist
      >>> from typing import List, Optional
      >>> dist.init_process_group(backend='nccl') # a
      >>> rank = torch.distributed.get_rank()
      >>> seqlen = torch.tensor([2, 2]) if rank == 0 else torch.tensor([1, 1])
      >>> T = seqlen.sum().item()
      >>> value_list = [torch.arange(0,T).cuda(), torch.arange(0,T).cuda() + 1]
      >>> gathered_tensors = grouped_allgatherv_tensor_list(value_list)
      >>> print(gathered_tensors)
      [tensor([0., 1. ,2. ,3., 0., 1.]), tensor([1., 2., 3., 4., 1., 2.])]
    """
    # 1. make sure all input tensors share the same dim0
    T0 = value_list[0].size(0)
    assert all(
        [T0 == value.size(0) for value in value_list]
    ), "grouped gatherv requires dim0 equal-size"
    input_dims = [value.dim() for value in value_list]
    value_list = [
        value.unsqueeze(-1) if value.dim() == 1 else value for value in value_list
    ]
    # 2. make sure all values are 2D tensors
    assert all(
        [value.dim() == 2 for value in value_list]
    ), "grouped gatherv only supports 2D tensors"
    assert all(
        [value.element_size() > 1 for value in value_list]
    ), "grouped gatherv only element size > 1 byte"

    element_dtypes = [value.dtype for value in value_list]
    last_dim_element_2bytes = [
        value.size(-1) * value.element_size() // 2 for value in value_list
    ]

    def pack_as_2byte_and_concat(value_list: List[torch.Tensor]):
        interpreted_tensor = [
            view_as_dtype(value, torch.bfloat16) for value in value_list
        ]
        tensor_container = torch.cat(interpreted_tensor, dim=-1)
        return tensor_container

    def split_and_unpack_dtype(
        tensor_container: torch.Tensor,
        split_sizes: List[int],
        element_dtypes: List[torch.dtype],
    ):
        split_2byte_tensors = list(torch.split(tensor_container, split_sizes, dim=-1))
        ret = [
            view_as_dtype(value, dtype)
            for value, dtype in zip(split_2byte_tensors, element_dtypes)
        ]
        return ret

    # 3. pack tensor as bfloat16 and concat all tensors
    untyped_tensor = pack_as_2byte_and_concat(value_list)
    # 4. allgatherv
    gathered_tensor = gatherv_along_first_dim(untyped_tensor, pg)

    # 5. unpack
    ret_value_list = split_and_unpack_dtype(
        gathered_tensor, last_dim_element_2bytes, element_dtypes
    )
    ret_value_list = [
        value.squeeze(-1) if dim == 1 else value
        for dim, value in zip(input_dims, ret_value_list)
    ]
    return ret_value_list


def grouped_allgather_tensor_list(
    value_list: List[torch.Tensor],
    pg: Optional[dist.ProcessGroup] = None,
):
    # allgather
    output_value_list = []
    for value in value_list:
        output_value_list.append(gather_along_first_dim(value, pg))
    return output_value_list


def jagged_tensor_allgather(
    jt: JaggedTensor,
    pg: Optional[dist.ProcessGroup] = None,
):
    values, lengths = jt.values(), jt.lengths()
    values = gatherv_along_first_dim(values, pg)
    lengths = gather_along_first_dim(lengths, pg)
    return JaggedTensor(values=values, lengths=lengths)


def keyed_jagged_tensor_list_allgather(
    kjt_list: List[KeyedJaggedTensor],
    pg: Optional[dist.ProcessGroup] = None,
) -> List[KeyedJaggedTensor]:
    """
    Fused AllGather for a **list** of KeyedJaggedTensors.

    All KJTs' lengths and values are concatenated before communication so that
    the entire list is gathered with only **2 NCCL calls** (1 AllGather for
    lengths, 1 AllGather for values) regardless of how many KJTs or keys
    there are. After gathering, the layout is transposed from rank-major
    [W, K_total, B] to key-major [K_total, W, B] using
    ``keyed_jagged_index_select_dim1``, then the result is split back into
    individual KJTs.

    Requirements:
      - All KJTs must share the same ``batch_size`` (stride).
      - All KJTs must share the same ``values`` dtype.

    Note: this api is not differentiable.

    Args:
        kjt_list: List of KeyedJaggedTensors to AllGather.
        pg: Process group for distributed operations.

    Returns:
        List of AllGathered KeyedJaggedTensors, one per input KJT.
    """
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return list(kjt_list)

    if not kjt_list:
        return []

    W = world_size
    device = kjt_list[0].lengths().device

    # --- Collect per-KJT metadata ---
    keys_list = [list(kjt.keys()) for kjt in kjt_list]
    K_list = [len(keys) for keys in keys_list]
    K_total = sum(K_list)
    B = kjt_list[0].lengths().numel() // K_list[0]

    # --- Concatenate all lengths and values across KJTs ---
    all_local_lengths = torch.cat([kjt.lengths() for kjt in kjt_list])  # [K_total * B]
    all_local_values = torch.cat([kjt.values() for kjt in kjt_list])  # [T_total_local]

    # === Step 1: Fused communication (2 NCCL calls for the entire list) ===
    # 1a. AllGather all lengths in one shot
    all_lengths = gather_along_first_dim(all_local_lengths, pg)  # [W * K_total * B]

    # 1b. AllGather all values — derive per-rank counts from gathered lengths
    #     to skip the redundant dim0 AllGather inside _gatherv_along_first_dim.
    per_rank_num_values = (
        all_lengths.view(W, K_total * B).to(torch.long).sum(dim=1)  # [W] on GPU
    )
    per_rank_num_values_cpu = per_rank_num_values.cpu()  # small D2H, [W] ints
    values_dim_tail = list(all_local_values.shape[1:])
    recv_buffers = [
        torch.empty(
            [cnt.item()] + values_dim_tail,
            dtype=all_local_values.dtype,
            device=device,
        )
        for cnt in per_rank_num_values_cpu
    ]
    dist.all_gather(recv_buffers, all_local_values.contiguous(), group=pg)
    all_values = torch.cat(recv_buffers).contiguous()  # [sum(T_all_ranks)]

    # === Step 2: Transpose [W, K_total, B] -> [K_total, W, B] ===
    NB = W * K_total * B  # fake batch size (1 fake key)
    perm = (
        torch.arange(NB, device=device)
        .view(W, K_total, B)
        .permute(1, 0, 2)
        .contiguous()
        .view(-1)
    )

    # === Step 3: Reorder via keyed_jagged_index_select_dim1 ===
    all_offsets = torch.zeros(NB + 1, dtype=all_lengths.dtype, device=device)
    all_offsets[1:] = all_lengths.cumsum(0)

    output = torch.ops.fbgemm.keyed_jagged_index_select_dim1(
        all_values,
        all_lengths,
        all_offsets,
        perm,
        NB,
    )
    reordered_values = output[0]
    reordered_lengths = output[1]

    # === Step 4: Split back into individual KJTs ===
    # Compute per-KJT value counts on GPU, then bring to CPU in one shot.
    per_key_value_counts = (
        reordered_lengths.view(K_total, W * B).to(torch.long).sum(dim=1)
    )  # [K_total]
    kjt_value_counts = []
    key_offset = 0
    for K in K_list:
        kjt_value_counts.append(per_key_value_counts[key_offset : key_offset + K].sum())
        key_offset += K
    kjt_value_counts_cpu = torch.stack(kjt_value_counts).cpu()  # single D2H

    result_kjts: List[KeyedJaggedTensor] = []
    lengths_offset = 0
    values_offset = 0
    for i, (keys, K) in enumerate(zip(keys_list, K_list)):
        kjt_num_lengths = K * W * B
        kjt_lengths = reordered_lengths[
            lengths_offset : lengths_offset + kjt_num_lengths
        ]
        kjt_num_values = kjt_value_counts_cpu[i].item()
        kjt_values = reordered_values[values_offset : values_offset + kjt_num_values]

        result_kjts.append(
            KeyedJaggedTensor(
                keys=keys,
                values=kjt_values,
                lengths=kjt_lengths,
                stride=W * B,
            )
        )
        lengths_offset += kjt_num_lengths
        values_offset += kjt_num_values

    return result_kjts


def keyed_jagged_tensor_allgather(
    kjt: KeyedJaggedTensor,
    pg: Optional[dist.ProcessGroup] = None,
) -> KeyedJaggedTensor:
    """
    AllGather a single KeyedJaggedTensor. Convenience wrapper around
    :func:`keyed_jagged_tensor_list_allgather`.
    """
    return keyed_jagged_tensor_list_allgather([kjt], pg)[0]
