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
from typing import List, Tuple

import fbgemm_gpu  # to load permute_2D_sparse_data
import torch
from torchrec.sparse.jagged_tensor import JaggedTensor


def jagged_tensors_shift_n(
    tensor_list: List[torch.Tensor], seqlen, seqlen_offsets, max_seqlen, shift_n=1
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, int]:
    """
    Right shift jagged tensors by shift_n elements. If the input sequence length is less than shift_n, the sequence length will be set to 0.

    Args:
        tensor_list (List[torch.Tensor]): List of value tensors to be shifted. They must share the same seqlen and seqlen_offsets.
        seqlen (torch.Tensor): Sequence lengths.
        seqlen_offsets (torch.Tensor): Sequence length offsets.
        max_seqlen (int): Maximum sequence length.
        shift_n (int): Number of positions to shift.

    Returns:
        Tuple: Shifted tensor list, shifted sequence lengths, shifted sequence length offsets, and new maximum sequence length.

    Example:
        >>> tensor_list = [torch.tensor([1, 2, 3, 4]).cuda(), torch.tensor([5, 6, 7, 8]).cuda()]
        >>> seqlen = torch.tensor([2, 2]).cuda()
        >>> seqlen_offsets = torch.tensor([0, 2, 4]).cuda()
        >>> max_seqlen = 2
        >>> shift_n = 1
        >>> jagged_tensors_shift_n(tensor_list, seqlen, seqlen_offsets, max_seqlen, shift_n)
        ([tensor([2, 4]), tensor([6, 8])], tensor([1, 1], tensor([0, 1, 2]), 1)
    """
    T0 = tensor_list[0].size(0)
    assert all(
        [T0 == value.size(0) for value in tensor_list]
    ), "jagged tensors shift n requires dim0 equal-size"
    input_dims = [value.dim() for value in tensor_list]
    tensor_list = [
        value.unsqueeze(-1) if value.dim() == 1 else value for value in tensor_list
    ]

    assert all(
        [value.dim() == 2 for value in tensor_list]
    ), "jagged_tensors_shift_n only support tensor with dim <= 2"
    seqlen_shift_n = torch.clamp(seqlen - shift_n, min=0)
    seqlen_offsets_shift_n = torch.ops.fbgemm.asynchronous_complete_cumsum(
        seqlen_shift_n
    )  # B + 1

    shift_n_tensor_list = []
    for t in tensor_list:
        dense_emb = torch.ops.fbgemm.jagged_2d_to_dense(
            values=t,
            offsets=seqlen_offsets,
            max_sequence_length=max_seqlen,
        )
        dense_history_emb = dense_emb[:, :-shift_n, ...]
        dense_supervision_emb = dense_emb[:, shift_n:, ...]

        history_emb = torch.ops.fbgemm.dense_to_jagged(
            dense_history_emb, [seqlen_offsets_shift_n]
        )[0]

        supervision_emb = torch.ops.fbgemm.dense_to_jagged(
            dense_supervision_emb, [seqlen_offsets_shift_n]
        )[0]
        shift_n_tensor_list.append((history_emb, supervision_emb))
    shift_n_tensor_list = [
        (v1.squeeze(-1), v2.squeeze(-1)) if dim == 1 else (v1, v2)
        for dim, (v1, v2) in zip(input_dims, shift_n_tensor_list)
    ]
    return (
        shift_n_tensor_list,
        seqlen_shift_n,
        seqlen_offsets_shift_n,
        max_seqlen - shift_n,
    )


def concat_2D_jagged_tensors(
    jagged_tensors: List[JaggedTensor],
    max_seqlens: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(jagged_tensors) == 0:
        raise ValueError("empty tensor list to concat")
    if len(jagged_tensors) == 1:
        return jagged_tensors[0].values(), jagged_tensors[0].lengths()
    padded_dense_list = []
    padded_mask_list = []
    for jt, max_seqlen in zip(jagged_tensors, max_seqlens):
        padded_dense = torch.ops.fbgemm.jagged_to_padded_dense(
            values=jt.values(),
            offsets=[jt.offsets()],
            max_lengths=[max_seqlen],
            padding_value=0.0,
        )
        padded_mask = torch.ops.fbgemm.jagged_to_padded_dense(
            values=torch.ones(
                (jt.values().numel(),), dtype=torch.long, device=jt.values().device
            ).view(-1, 1),
            offsets=[jt.offsets()],
            max_lengths=[max_seqlen],
            padding_value=0,
        ).to(torch.bool)
        padded_dense_list.append(padded_dense)
        padded_mask_list.append(padded_mask)

    concatted_dense = torch.cat(padded_dense_list, dim=1)
    concatted_mask = torch.cat(padded_mask_list, dim=1)
    return concatted_dense.flatten(0, 1)[concatted_mask.view(-1), :], torch.sum(
        torch.concat([jt.lengths().view(-1, 1) for jt in jagged_tensors], dim=1), dim=1
    )
