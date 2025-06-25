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
import abc
from typing import Optional, Union

import torch
from commons.utils.nvtx_op import output_nvtx_hook
from configs import KernelBackend


class HSTUAttention(torch.nn.Module):
    """
    Base module interface for different HSTUAttention backends.

    """

    @abc.abstractmethod
    def forward(
        self,
        tq: torch.Tensor,  # (T, d)
        tk: torch.Tensor,  # (T, d)
        tv: torch.Tensor,  # (T, d)
        offsets: torch.Tensor,  # (batch_size, 1)
        max_seqlen: int,
        target_group_size: int = 1,  # target <=> candidates
        num_candidates: Optional[torch.Tensor] = None,
        num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:  # T, d
        """
        Abstract method for the forward pass of HSTUAttention.

        Args:
            tq (torch.Tensor): Query tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the query.
            tk (torch.Tensor): Key tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the key.
            tv (torch.Tensor): Value tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the value.
            offsets (torch.Tensor): Offsets tensor of shape (batch_size, 1), indicating the start position of each sequence in the batch.
            max_seqlen (int): The maximum sequence length across all batches.
            target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
            num_candidates (torch.Tensor): Tensor containing the number of candidates for each sequence.
            num_contextuals (int | torch.Tensor | None): The number of contextuals for each sequence, could be a single integer or a tensor of shape (batch_size,) when different sequences have different number of contextuals.
        Returns:
            torch.Tensor: Output tensor of shape (T, d).
        """


class TorchHSTUAttention(HSTUAttention):
    """
    Native HUST implementation. All jagged inputs are padded to the maximum length before computation.

    Args:
        num_heads (int): Number of attention heads.
        attention_dim (int): Dimension of the attention.
        linear_dim (int): Dimension of the linear layer.
        is_causal (bool): Whether the attention is causal.
    """

    def __init__(
        self,
        num_heads: int,
        attention_dim: int,
        linear_dim: int,
        is_causal: bool,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.is_causal = is_causal

    def forward(
        self,
        tq: torch.Tensor,  # (T, d)
        tk: torch.Tensor,  # (T, d)
        tv: torch.Tensor,  # (T, d)
        offsets: torch.Tensor,  # (batch_size, 1)
        max_seqlen: int,
        target_group_size: int = 1,  # target == candidates
        num_candidates: Optional[torch.Tensor] = None,
        num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:  # T, d
        """
        Forward pass of the TorchHSTUAttention module.

        Args:
            tq (torch.Tensor): Query tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the query.
            tk (torch.Tensor): Key tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the key.
            tv (torch.Tensor): Value tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the value.
            offsets (torch.Tensor): Offsets tensor of shape (batch_size, 1), indicating the start position of each sequence in the batch.
            max_seqlen (int): The maximum sequence length across all batches.
            target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
            num_candidates (torch.Tensor): Tensor containing the number of candidates for each sequence.
            num_contextuals (int | torch.Tensor | None): The number of contextuals for each sequence, could be a single integer or a tensor of shape (batch_size,) when different sequences have different number of contextuals.
        Returns:
            torch.Tensor: Output tensor of shape (T, d).
        """
        from ops.pt_ops.pt_hstu_attention import pytorch_hstu_mha

        if isinstance(num_contextuals, torch.Tensor):
            num_contextuals = num_contextuals.to(torch.int32)
        elif isinstance(num_contextuals, int):
            num_contextuals = (
                torch.tensor([num_contextuals], dtype=torch.int32, device=tq.device)
                .view(1)
                .expand(offsets.size(0) - 1)
                .contiguous()
            )

        return pytorch_hstu_mha(
            max_seq_len=max_seqlen,
            alpha=1.0 / (self.attention_dim**0.5),
            q=tq.view(-1, self.num_heads, self.attention_dim),
            k=tk.view(-1, self.num_heads, self.attention_dim),
            v=tv.view(-1, self.num_heads, self.linear_dim),
            seq_offsets=offsets,
            num_contextuals=num_contextuals,
            num_targets=num_candidates,
            causal=self.is_causal,
            dropout_pr=0.0,
            training=self.training,
            target_group_size=target_group_size,
        ).view(-1, self.num_heads * self.linear_dim)


class TritonHSTUAttention(HSTUAttention):
    """
    Triton-based HUST implementation.

    Args:
        num_heads (int): Number of attention heads.
        attention_dim (int): Dimension of the attention.
        linear_dim (int): Dimension of the linear layer.
        is_causal (bool): Whether the attention is causal.
    """

    def __init__(
        self,
        num_heads: int,
        attention_dim: int,
        linear_dim: int,
        is_causal: bool,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.is_causal = is_causal

    def forward(
        self,
        tq: torch.Tensor,  # (T, d)
        tk: torch.Tensor,  # (T, d)
        tv: torch.Tensor,  # (T, d)
        offsets: torch.Tensor,  # (batch_size, 1)
        max_seqlen: int,
        target_group_size: int = 1,  # target == candidates
        num_candidates: Optional[torch.Tensor] = None,
        num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:  # T, d
        """
        Forward pass of the TritonHSTUAttention module.

        Args:
             tq (torch.Tensor): Query tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the query.
            tk (torch.Tensor): Key tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the key.
            tv (torch.Tensor): Value tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the value.
            offsets (torch.Tensor): Offsets tensor of shape (batch_size, 1), indicating the start position of each sequence in the batch.
            max_seqlen (int): The maximum sequence length across all batches.
            target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
            num_candidates (torch.Tensor): Tensor containing the number of candidates for each sequence.
            num_contextuals (int | torch.Tensor | None): The number of contextuals for each sequence, could be a single integer or a tensor of shape (batch_size,) when different sequences have different number of contextuals.
        Returns:
            torch.Tensor: Output tensor of shape (T, d).
        """
        from ops.triton_ops.triton_hstu_attention import (  # type: ignore[attr-defined]
            triton_hstu_mha,
        )

        assert (
            target_group_size == 1
        ), "target_group_size is not supported in TritonHSTUAttention"
        if num_contextuals is None:
            num_contextuals = 0
        assert isinstance(
            num_contextuals, int
        ), "num_contextuals must be an integer in TritonHSTUAttention"
        return triton_hstu_mha(
            N=max_seqlen,
            alpha=1.0 / (self.attention_dim**0.5),
            q=tq.view(-1, self.num_heads, self.attention_dim),
            k=tk.view(-1, self.num_heads, self.attention_dim),
            v=tv.view(-1, self.num_heads, self.linear_dim),
            seq_offsets=offsets,
            num_targets=num_candidates,
            causal=self.is_causal,
            contextual_seq_len=num_contextuals,
        ).view(-1, self.num_heads * self.linear_dim)


# TODO, support packed qkv attention
class FusedHSTUAttention(HSTUAttention):
    """
    Cutlass-based HUST implementations. This is the default implementation on pre hopper GPU.

    Args:
        num_heads (int): Number of attention heads.
        attention_dim (int): Dimension of the attention.
        linear_dim (int): Dimension of the linear layer.
        is_causal (bool): Whether the attention is causal.
    """

    def __init__(
        self,
        num_heads: int,
        attention_dim: int,
        linear_dim: int,
        is_causal: bool,
    ):
        super().__init__()
        from hstu_attn import hstu_attn_varlen_func

        self._hstu_attn_varlen_func = hstu_attn_varlen_func
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.is_causal = is_causal
        assert (
            self.linear_dim == self.attention_dim
        ), "only support linear_dim and attention_dim"

    @output_nvtx_hook(nvtx_tag="FusedHSTUAttn")
    def forward(
        self,
        tq: torch.Tensor,  # (T, d)
        tk: torch.Tensor,  # (T, d)
        tv: torch.Tensor,  # (T, d)
        offsets: torch.Tensor,  # (batch_size, 1)
        max_seqlen: int,
        target_group_size: int = 1,  # target == candidates
        num_candidates: Optional[torch.Tensor] = None,
        num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the FusedHSTUAttention module.

        Args:
            tq (torch.Tensor): Query tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the query.
            tk (torch.Tensor): Key tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the key.
            tv (torch.Tensor): Value tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the value.
            offsets (torch.Tensor): Offsets tensor of shape (batch_size, 1), indicating the start position of each sequence in the batch.
            max_seqlen (int): The maximum sequence length across all batches.
            target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
            num_candidates (torch.Tensor): Tensor containing the number of candidates for each sequence.
            num_contextuals (int | torch.Tensor | None): The number of contextuals for each sequence, could be a single integer or a tensor of shape (batch_size,) when different sequences have different number of contextuals.

        Returns:
            torch.Tensor: Output tensor.
        """
        # TODO: remove once cutlass backend is ready
        assert (
            self.is_causal or num_contextuals is None
        ), "Only causal attention is supported when max_num_contextuals > 0 in cutlass backend"
        # (b * ~s, nh * hh)
        if isinstance(num_contextuals, torch.Tensor):
            num_contextuals = num_contextuals.to(torch.int32)
        elif isinstance(num_contextuals, int):
            num_contextuals = (
                torch.tensor([num_contextuals], dtype=torch.int32, device=tq.device)
                .view(1)
                .expand(offsets.size(0) - 1)
                .contiguous()
            )

        return self._hstu_attn_varlen_func(
            q=tq.view(-1, self.num_heads, self.attention_dim),
            k=tk.view(-1, self.num_heads, self.attention_dim),
            v=tv.view(-1, self.num_heads, self.linear_dim),
            seq_offsets_q=offsets.to(torch.int32),
            seq_offsets_k=offsets.to(torch.int32),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            num_contexts=num_contextuals,
            num_targets=num_candidates.to(torch.int32)
            if isinstance(num_candidates, torch.Tensor)
            else None,
            target_group_size=target_group_size,
            window_size=(-1, 0) if self.is_causal else (-1, -1),
            rab=None,
            alpha=1.0 / (self.attention_dim**0.5),
        ).view(-1, self.num_heads * self.linear_dim)


# TODO, support packed qkv attention
class FusedHSTUAttentionHopper(HSTUAttention):
    """
    Cutlass-based HUST implementation. This is the specialized implementation on hopper.

    Args:
        num_heads (int): Number of attention heads.
        attention_dim (int): Dimension of the attention.
        linear_dim (int): Dimension of the linear layer.
        is_causal (bool): Whether the attention is causal.
    """

    def __init__(
        self,
        num_heads: int,
        attention_dim: int,
        linear_dim: int,
        is_causal: bool,
    ):
        super().__init__()
        from hopper.hstu_attn_interface import hstu_attn_varlen_func

        self._hstu_attn_varlen_func = hstu_attn_varlen_func
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.is_causal = is_causal
        assert (
            self.linear_dim == self.attention_dim
        ), "only support linear_dim and attention_dim"

    @output_nvtx_hook(nvtx_tag="FusedHSTUAttnHopper")
    def forward(
        self,
        tq: torch.Tensor,  # (T, d)
        tk: torch.Tensor,  # (T, d)
        tv: torch.Tensor,  # (T, d)
        offsets: torch.Tensor,  # (batch_size, 1)
        max_seqlen: int,
        target_group_size: int = 1,  # target == candidates
        num_candidates: Optional[torch.Tensor] = None,
        num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the FusedHSTUAttentionHopper module.

        Args:
            tq (torch.Tensor): Query tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the query.
            tk (torch.Tensor): Key tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the key.
            tv (torch.Tensor): Value tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the value.
            offsets (torch.Tensor): Offsets tensor of shape (batch_size, 1), indicating the start position of each sequence in the batch.
            max_seqlen (int): The maximum sequence length across all batches.
            target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
            num_candidates (torch.Tensor): Tensor containing the number of candidates for each sequence.
            num_contextuals (int | torch.Tensor | None): The number of contextuals for each sequence, could be a single integer or a tensor of shape (batch_size,) when different sequences have different number of contextuals.

        Returns:
            torch.Tensor: Output tensor.
        """
        # TODO: remove once cutlass backend is ready
        assert (
            self.is_causal or num_contextuals is None
        ), "Only causal attention is supported when max_num_contextuals > 0 in cutlass backend"
        # (b * ~s, nh * hh)
        if isinstance(num_contextuals, torch.Tensor):
            num_contextuals = num_contextuals.to(torch.int32)
        elif isinstance(num_contextuals, int):
            num_contextuals = (
                torch.tensor([num_contextuals], dtype=torch.int32, device=tq.device)
                .view(1)
                .expand(offsets.size(0) - 1)
                .contiguous()
            )
        return self._hstu_attn_varlen_func(
            q=tq.view(-1, self.num_heads, self.attention_dim),
            k=tk.view(-1, self.num_heads, self.attention_dim),
            v=tv.view(-1, self.num_heads, self.linear_dim),
            seq_offsets_q=offsets.to(torch.int32),
            seq_offsets_k=offsets.to(torch.int32),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            num_contexts=num_contextuals,
            num_targets=num_candidates.to(torch.int32)
            if isinstance(num_candidates, torch.Tensor)
            else None,
            target_group_size=target_group_size,
            window_size=(-1, 0) if self.is_causal else (-1, -1),
            rab=None,
            alpha=1.0 / (self.attention_dim**0.5),
        ).view(-1, self.num_heads * self.linear_dim)


def create_hstu_attention(
    kernel_backend: KernelBackend,
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    is_causal: bool,
) -> HSTUAttention:
    """
    Factory function to create an HSTUAttention module based on the kernel backend.

    Args:
        kernel_backend (KernelBackend): The kernel backend to use.
        num_heads (int): Number of attention heads.
        attention_dim (int): Dimension of the attention.
        linear_dim (int): Dimension of the linear layer.
        is_causal (bool): Whether the attention is causal.

    Returns:
        HSTUAttention: The created HSTUAttention module.

    Raises:
        ValueError: If the kernel backend is not supported.
    """
    if kernel_backend == KernelBackend.CUTLASS:
        sm_major_version = torch.cuda.get_device_properties(0).major
        sm_minor_version = torch.cuda.get_device_properties(0).minor
        if sm_major_version == 9 and sm_minor_version == 0:
            return FusedHSTUAttentionHopper(
                num_heads,
                attention_dim,
                linear_dim,
                is_causal,
            )
        elif sm_major_version == 8 and sm_minor_version == 0:
            return FusedHSTUAttention(
                num_heads,
                attention_dim,
                linear_dim,
                is_causal,
            )
        print(
            "CUTLASS backend only support H100, H20 and A100, fallback to PyTorch backend"
        )
    elif kernel_backend == KernelBackend.TRITON:
        if is_causal:
            return TritonHSTUAttention(
                num_heads,
                attention_dim,
                linear_dim,
                is_causal,
            )
        else:
            print(
                "Triton backend does not support is_causal=False, fallback to PyTorch backend"
            )
    return TorchHSTUAttention(
        num_heads,
        attention_dim,
        linear_dim,
        is_causal,
    )
