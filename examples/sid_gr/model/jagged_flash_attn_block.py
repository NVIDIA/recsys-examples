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
"""
JaggedFlashAttnBlock: a self-contained GPT Transformer block that uses
jiayus's Flash Attention with arbitrary_func mask encoding.

This replaces Megatron-Core's TransformerBlock for inference with
Method A (Incremental Append) beam search.

Architecture per layer (standard pre-norm GPT):
  Input → LayerNorm → QKV Projection → Flash Attention (arbitrary mask)
        → Output Projection → Residual
        → LayerNorm → FFN → Residual → Output

Reference: examples/hstu/modules/native_hstu_layer.py
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# flash_attn imports are deferred to runtime (inside functions / __init__)
# so that the module can be imported even without flash_attn installed.


def build_block_sparsity(
    arbitrary_func: torch.Tensor,
    seqlen_q: int,
    seqlen_k: int,
    headdim: int,
) -> Tuple[Optional[object], Optional[object]]:
    """
    Build forward (Q2K) and backward (K2Q) block sparsity indices from an
    arbitrary_func tensor.

    Returns (linear_k, linear_q) — either can be None if the CUDA extension
    is not available (falls back to dense attention).
    """
    try:
        import create_block_mask_cuda
        from flash_attn.cute.block_sparsity import LinearBlockSparseTensorsTorch
        from flash_attn.utils.tile_size import get_arch, get_tile_sizes_by_backend
    except ImportError:
        return None, None

    arch = get_arch()
    fwd_q_block, fwd_kv_block = get_tile_sizes_by_backend(
        backend="dsl",
        pass_type="forward",
        arch=arch,
        headdim=headdim,
        is_causal=False,
        is_local=False,
        is_arbitrary=True,
    )
    bwd_q_block, bwd_kv_block = get_tile_sizes_by_backend(
        backend="dsl",
        pass_type="backward",
        arch=arch,
        headdim=headdim,
        is_causal=False,
        is_local=False,
        is_arbitrary=True,
    )

    (
        k_cnt,
        k_off,
        k_idx,
        k_fcnt,
        k_foff,
        k_fidx,
    ) = create_block_mask_cuda.create_q2k_csr_sparse_from_func(
        arbitrary_func,
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE=fwd_q_block,
        KV_BLOCK_SIZE=fwd_kv_block,
        check_q_boundary=True,
    )
    linear_k = LinearBlockSparseTensorsTorch(
        mask_block_cnt=k_cnt,
        mask_block_offset=k_off,
        mask_block_idx=k_idx,
        full_block_cnt=k_fcnt,
        full_block_offset=k_foff,
        full_block_idx=k_fidx,
    )

    (
        q_cnt,
        q_off,
        q_idx,
        q_fcnt,
        q_foff,
        q_fidx,
    ) = create_block_mask_cuda.create_k2q_csr_sparse_from_func(
        arbitrary_func,
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE=bwd_q_block,
        KV_BLOCK_SIZE=bwd_kv_block,
    )
    linear_q = LinearBlockSparseTensorsTorch(
        mask_block_cnt=q_cnt,
        mask_block_offset=q_off,
        mask_block_idx=q_idx,
        full_block_cnt=q_fcnt,
        full_block_offset=q_foff,
        full_block_idx=q_fidx,
    )

    return linear_k, linear_q


class JaggedGPTLayer(nn.Module):
    """
    One Transformer layer with jagged Flash Attention.

    Pre-norm GPT structure:
      x = x + Attn(LayerNorm(x))
      x = x + FFN(LayerNorm(x))

    Q/K/V are produced by a single fused linear (same pattern as HSTU's
    ``linear_uvqk``). Flash Attention is called with arbitrary_func for
    tree-shaped beam search masks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        layernorm_epsilon: float = 1e-5,
        hidden_dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.ffn_hidden_size = ffn_hidden_size

        # --- Attention sub-layers ---
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)
        # Fused QKV projection: hidden_size → 3 * hidden_size
        self.linear_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        # Output projection after attention
        self.linear_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(hidden_dropout)

        # --- FFN sub-layers ---
        self.pre_mlp_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.mlp_fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.mlp_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)
        self.mlp_dropout = nn.Dropout(hidden_dropout)

        if activation == "gelu":
            self.activation_fn = F.gelu
        elif activation == "silu":
            self.activation_fn = F.silu
        else:
            self.activation_fn = F.gelu

    def forward(
        self,
        hidden_states: torch.Tensor,
        arbitrary_func: Optional[torch.Tensor] = None,
        linear_k: Optional[object] = None,
        linear_q: Optional[object] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seqlen, hidden_size]
            arbitrary_func: [batch, 1, n_func, seqlen+256] int32 mask encoding.
            linear_k: forward block sparsity (Q2K).
            linear_q: backward block sparsity (K2Q).

        Returns:
            hidden_states: [batch, seqlen, hidden_size]
        """
        # ---- Attention block ----
        residual = hidden_states
        x = self.input_layernorm(hidden_states)

        # QKV projection: [B, S, H] → [B, S, 3*H]
        qkv = self.linear_qkv(x)
        # Reshape to [B, S, 3, num_heads, head_dim] and unbind
        B, S, _ = qkv.shape
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, S, num_heads, head_dim]

        # Flash Attention requires fp16/bf16 inputs
        from flash_attn.cute.interface import flash_attn_func

        input_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        if arbitrary_func is not None:
            attn_out, _ = flash_attn_func(
                q,
                k,
                v,
                softmax_scale=self.head_dim ** (-0.5),
                causal=False,
                arbitrary=True,
                linear_k_block_sparse_tensors=linear_k,
                linear_q_block_sparse_tensors=linear_q,
                aux_tensors=[arbitrary_func],
            )
        else:
            attn_out, _ = flash_attn_func(
                q,
                k,
                v,
                softmax_scale=self.head_dim ** (-0.5),
                causal=True,
            )

        if attn_out.dtype != input_dtype:
            attn_out = attn_out.to(input_dtype)

        # attn_out: [B, S, num_heads, head_dim] → [B, S, hidden_size]
        attn_out = attn_out.reshape(B, S, self.hidden_size)
        attn_out = self.linear_proj(attn_out)
        attn_out = self.attn_dropout(attn_out)
        hidden_states = residual + attn_out

        # ---- FFN block ----
        residual = hidden_states
        x = self.pre_mlp_layernorm(hidden_states)
        x = self.mlp_fc1(x)
        x = self.activation_fn(x)
        x = self.mlp_fc2(x)
        x = self.mlp_dropout(x)
        hidden_states = residual + x

        return hidden_states


class JaggedFlashAttnBlock(nn.Module):
    """
    A stack of JaggedGPTLayers — the GPT decoder block using jiayus's
    Flash Attention with arbitrary_func mask encoding.

    This module owns its own weights (not shared with Megatron-Core).
    It is used in place of Megatron's TransformerBlock for inference
    with Method A beam search.

    Usage::

        block = JaggedFlashAttnBlock(
            num_layers=4,
            hidden_size=256,
            num_attention_heads=4,
            ffn_hidden_size=1024,
        )
        # padded input: [B, S, D]
        output = block(hidden_states, arbitrary_func=af, seqlen=S)
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        layernorm_epsilon: float = 1e-5,
        hidden_dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads
        self.layers = nn.ModuleList(
            [
                JaggedGPTLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    ffn_hidden_size=ffn_hidden_size,
                    layernorm_epsilon=layernorm_epsilon,
                    hidden_dropout=hidden_dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        arbitrary_func: Optional[torch.Tensor] = None,
        seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seqlen, hidden_size] padded input.
            arbitrary_func: [batch, 1, n_func, seqlen+256] int32 mask tensor.
                If None, uses standard causal attention.
            seqlen: sequence length (used for block sparsity construction).
                If None, inferred from hidden_states.shape[1].

        Returns:
            hidden_states: [batch, seqlen, hidden_size]
        """
        if seqlen is None:
            seqlen = hidden_states.shape[1]

        # Build block sparsity from arbitrary_func (once per forward, shared by all layers)
        linear_k, linear_q = None, None
        if arbitrary_func is not None:
            linear_k, linear_q = build_block_sparsity(
                arbitrary_func, seqlen, seqlen, self.head_dim
            )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                arbitrary_func=arbitrary_func,
                linear_k=linear_k,
                linear_q=linear_q,
            )

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class JaggedTransformerBlock(nn.Module):
    """
    Wrapper that accepts jagged (variable-length) hidden states and a
    pre-built arbitrary_func tensor in the flattened (B=1) coordinate space.

    All batch sequences are concatenated into a single sequence of length
    *total_tokens* (no padding).  The arbitrary_func encodes both the
    block-diagonal batch isolation and the desired attention pattern
    (causal, target-grouped, etc.).

    Internally:
      1. Reshape jagged [total_tokens, D] → [1, total_tokens, D]
      2. Forward through JaggedFlashAttnBlock (FA with arbitrary mask)
      3. Reshape [1, total_tokens, D] → [total_tokens, D]

    This is intended to replace Megatron-Core's TransformerBlock in
    SIDGRDecoder.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        layernorm_epsilon: float = 1e-5,
        hidden_dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.block = JaggedFlashAttnBlock(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            ffn_hidden_size=ffn_hidden_size,
            layernorm_epsilon=layernorm_epsilon,
            hidden_dropout=hidden_dropout,
            activation=activation,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        arbitrary_func: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: jagged [total_tokens, hidden_size].
            arbitrary_func: [1, 1, n_func, total_tokens + pad] int32 tensor
                in flattened (B=1) coordinate space, encoding both batch
                isolation and the attention pattern.

        Returns:
            jagged output [total_tokens, hidden_size].
        """
        total_tokens = hidden_states.shape[0]

        # [total_tokens, D] → [1, total_tokens, D]
        flat_input = hidden_states.unsqueeze(0)

        output = self.block(
            flat_input, arbitrary_func=arbitrary_func, seqlen=total_tokens
        )

        # [1, total_tokens, D] → [total_tokens, D]
        return output.squeeze(0)
