# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashAttention prefill backend adapter."""

from __future__ import annotations

from gr_inference.gr_kernels.prefill.base import (
    PrefillAttentionInputs,
    PrefillAttentionOutput,
)


class FlashAttentionPrefillBackend:
    """Dense causal prefill via ``flash_attn.flash_attn_func`` when installed."""

    def __init__(self, softmax_scale: float | None = None) -> None:
        self.softmax_scale = softmax_scale

    def __call__(self, inputs: PrefillAttentionInputs) -> PrefillAttentionOutput:
        try:
            from flash_attn import flash_attn_func
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "FlashAttentionPrefillBackend requires flash-attn"
            ) from exc

        hidden = flash_attn_func(
            inputs.q,
            inputs.k,
            inputs.v,
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=inputs.causal,
        )
        return PrefillAttentionOutput(hidden=hidden)


class SGLangFlashAttentionPrefillBackend:
    """Dense causal prefill via SGLang's FA3/FA4 wrapper when available."""

    def __init__(
        self,
        softmax_scale: float | None = None,
        *,
        version: int = 3,
    ) -> None:
        self.softmax_scale = softmax_scale
        self.version = int(version)

    def __call__(self, inputs: PrefillAttentionInputs) -> PrefillAttentionOutput:
        try:
            import torch
            from sglang.jit_kernel.flash_attention import flash_attn_varlen_func
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "SGLangFlashAttentionPrefillBackend requires sglang.jit_kernel"
            ) from exc

        q = inputs.q
        k = inputs.k
        v = inputs.v
        batch, seq_len, num_q_heads, head_dim = q.shape
        _k_batch, _k_seq_len, num_kv_heads, _k_head_dim = k.shape

        q_flat = q.reshape(batch * seq_len, num_q_heads, head_dim)
        k_flat = k.reshape(batch * seq_len, num_kv_heads, head_dim)
        v_flat = v.reshape(batch * seq_len, num_kv_heads, head_dim)
        cu_seqlens = torch.arange(
            0,
            (batch + 1) * seq_len,
            seq_len,
            device=q.device,
            dtype=torch.int32,
        )
        hidden = flash_attn_varlen_func(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens,
            cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            softmax_scale=self.softmax_scale,
            causal=inputs.causal,
            ver=self.version,
        )
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return PrefillAttentionOutput(
            hidden=hidden.reshape(batch, seq_len, num_q_heads, head_dim)
        )
