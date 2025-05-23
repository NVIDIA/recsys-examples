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
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# yapf: disable

#!/usr/bin/env python3
# pyre-strict

from typing import Optional, Tuple, Union

import fbgemm_gpu  # pylint: disable-unused-import
import torch
import torch.nn.functional as F

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass

@torch.fx.wrap
def _get_valid_attn_mask(
    device: torch.device,
    causal: bool,
    N: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: Optional[int] = None,
    num_contextuals: Union[int, torch.Tensor] = 0, # (bathsize,)
    min_full_attn_seq_len: int = 0,
    target_group_size: int = 1,
):
  ids = torch.arange(0, N, device=device).view(1, N)
  max_ids = seq_lengths.view(-1, 1, 1)
  if isinstance(num_contextuals, int):
    if num_contextuals > 0:
      ids = ids - num_contextuals + 1 # +1 to make sure contexutal token ids are 0-indexed and history token ids are originated from 0
      ids = torch.clamp(ids, min=0)
      max_ids = max_ids - num_contextuals + 1
  else:
    ids = ids - num_contextuals.view(-1, 1) + 1 # (batchsize, N)
    ids = torch.clamp(ids, min=0)
    max_ids = max_ids - num_contextuals.view(-1, 1, 1) + 1
  row_ids = ids.unsqueeze(-1).expand(-1, N, N)
  col_ids = row_ids.transpose(1, 2)
  row_col_dist = row_ids - col_ids
  valid_attn_mask = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
  if not causal:
      row_col_dist = torch.where(row_col_dist > 0, row_col_dist, -row_col_dist)
  valid_attn_mask = torch.logical_or(valid_attn_mask, row_col_dist > 0)
  
  if num_targets is not None:
    target_group_row_ids = torch.clamp(row_ids - max_ids + num_targets.view(-1, 1, 1), min=-1) // target_group_size
    target_group_col_ids = target_group_row_ids.transpose(1, 2)
    target_dist = target_group_row_ids - target_group_col_ids

    target_group_mask = torch.logical_or(target_dist == 0, (target_group_row_ids < 0) + (target_group_col_ids < 0))
    # preserve the intra-target-group attention and purge the inter-target-group attention
    valid_attn_mask = torch.logical_and(valid_attn_mask, target_group_mask)
    max_ids = max_ids - num_targets.view(-1, 1, 1)
    
  if max_attn_len is not None and max_attn_len > 0:
      if min_full_attn_seq_len > 0:
          valid_attn_mask = torch.logical_and(
              valid_attn_mask,
              torch.logical_or(
                  row_col_dist <= max_attn_len,
                  row_ids >= max_ids - min_full_attn_seq_len,
              ),
          )
      else:
          valid_attn_mask = torch.logical_and(
              valid_attn_mask, row_col_dist <= max_attn_len
          )
  if (isinstance(num_contextuals, int) and num_contextuals > 0) or isinstance(
      num_contextuals, torch.Tensor
  ):  # int or tensor
      valid_attn_mask = torch.logical_or(
          valid_attn_mask, torch.logical_and(row_ids == 0, col_ids < max_ids)
      )
  return valid_attn_mask

def _pad_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L, H, D = q.shape
    V = v.shape[2]
    padded_q = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=q.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )  # [B, H, N, A]
    padded_k = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )  # [B, H, N, A]
    padded_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(L, H * V),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, V)
        .transpose(1, 2)
    )  # [B, H, N, D]
    return padded_q, padded_k, padded_v


@torch.fx.wrap
def pytorch_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    max_attn_len: Optional[int] = None,
    target_group_size: int = 1,
) -> torch.Tensor:
    if num_contextuals is None:
      num_contextuals = 0

    L, H, _ = q.shape
    V = v.shape[2]
    q, k, v = _pad_qkv(
        q, k, v, seq_offsets, max_seq_len
    )  # [B, H, N, D) and [B, H, N, V]
    qk_attn = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha
    qk_attn = F.silu(qk_attn) / max_seq_len
    valid_attn_mask = _get_valid_attn_mask(
      device=q.device,
      causal=causal,
      N=max_seq_len,
      seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
      num_targets=num_targets,
      max_attn_len=max_attn_len,
      num_contextuals=num_contextuals,
      target_group_size=target_group_size,
    )
    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)
    if dropout_pr > 0.0:
        qk_attn = F.dropout(qk_attn, p=dropout_pr, training=training)
    attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v)  # [B, H, N, V]
    return torch.ops.fbgemm.dense_to_jagged(
        attn_dense.transpose(1, 2).flatten(2, 3),  # [B, N, H, V]->[B, N, H * V]
        [seq_offsets],
        L,
    )[0].view(L, H, V)


@torch.fx.wrap
def _get_delta_valid_attn_mask(
    max_seq_len: int,
    delta_x_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: Optional[int] = None,
) -> torch.Tensor:
    B = seq_lengths.size(0)
    ids = torch.arange(0, max_seq_len, device=delta_x_offsets.device)
    col_ids = ids.view(1, 1, max_seq_len)
    row_ids = delta_x_offsets.view(B, -1) - seq_offsets[:-1].view(-1, 1)
    row_ids = row_ids.view(B, -1, 1)
    valid_attn_mask = col_ids == row_ids
    if num_targets is not None:
        seq_lengths = seq_lengths.view(-1, 1, 1)
        num_targets = num_targets.view(-1, 1, 1)
        row_ids = torch.clamp(row_ids, max=seq_lengths - num_targets)
        col_ids = torch.clamp(col_ids, max=seq_lengths - num_targets)
    row_col_dist = row_ids - col_ids
    valid_attn_mask = torch.logical_or(valid_attn_mask, row_col_dist > 0)
    if max_attn_len is not None:
        valid_attn_mask = torch.logical_and(
            valid_attn_mask, row_col_dist <= max_attn_len
        )
    return valid_attn_mask.unsqueeze(1)


@torch.fx.wrap
def pytorch_cached_hstu_mha(
    N: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    delta_x_offsets: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    max_attn_len: Optional[int] = None,
) -> torch.Tensor:
    L, H, D = delta_q.shape
    _, _, V = v.shape
    B = seq_offsets.size(0) - 1
    delta_q = delta_q.view(B, -1, H, D).transpose(1, 2)
    full_k = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(-1, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(B, -1, H, D)
        .transpose(1, 2)
    )
    full_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(-1, H * V),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(B, -1, H, V)
        .transpose(1, 2)
    )
    qk_attn = torch.einsum("bhxa,bhya->bhxy", delta_q, full_k) * alpha
    if attn_bias is not None:
        qk_attn = qk_attn + attn_bias
    qk_attn = F.silu(qk_attn) / N
    valid_attn_mask = _get_delta_valid_attn_mask(
        max_seq_len=N,
        delta_x_offsets=delta_x_offsets,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
    )
    qk_attn = qk_attn * valid_attn_mask
    attn_output = torch.einsum("bhxd,bhdv->bhxv", qk_attn, full_v)

    return attn_output.transpose(1, 2).reshape(-1, H, V)
