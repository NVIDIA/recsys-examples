# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch SDPA prefill backend used as a correctness baseline."""

from __future__ import annotations

import os

from gr_inference.gr_kernels.prefill.base import (
    PrefillAttentionInputs,
    PrefillAttentionOutput,
)


class TorchSDPAPrefillBackend:
    """Dense causal prefill attention via ``torch.nn.functional.scaled_dot_product_attention``.

    This backend prioritizes availability and correctness. It gives us a stable
    reference before adding FlashAttention or TRT-LLM prefill backends.
    """

    def __call__(self, inputs: PrefillAttentionInputs) -> PrefillAttentionOutput:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("TorchSDPAPrefillBackend requires torch") from exc

        q = inputs.q
        k = inputs.k
        v = inputs.v

        if not hasattr(q, "permute"):
            raise TypeError("TorchSDPAPrefillBackend requires torch tensors")

        batch, seq_len, num_q_heads, head_dim = q.shape
        num_kv_heads = k.shape[2]
        qhead_per_kv = num_q_heads // num_kv_heads

        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                "num_q_heads must be divisible by num_kv_heads for GQA prefill"
            )

        # SDPA expects [B, H, S, D].
        q_sdpa = q.permute(0, 2, 1, 3)
        k_sdpa = k.permute(0, 2, 1, 3)
        v_sdpa = v.permute(0, 2, 1, 3)

        with torch.no_grad():
            if qhead_per_kv == 1:
                out = F.scaled_dot_product_attention(
                    q_sdpa,
                    k_sdpa,
                    v_sdpa,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=inputs.causal,
                )
            elif _sdpa_native_gqa_enabled():
                try:
                    out = F.scaled_dot_product_attention(
                        q_sdpa,
                        k_sdpa,
                        v_sdpa,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=inputs.causal,
                        enable_gqa=True,
                    )
                except TypeError:
                    expanded_k = k_sdpa.repeat_interleave(qhead_per_kv, dim=1)
                    expanded_v = v_sdpa.repeat_interleave(qhead_per_kv, dim=1)
                    out = F.scaled_dot_product_attention(
                        q_sdpa,
                        expanded_k,
                        expanded_v,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=inputs.causal,
                    )
            else:
                expanded_k = k_sdpa.repeat_interleave(qhead_per_kv, dim=1)
                expanded_v = v_sdpa.repeat_interleave(qhead_per_kv, dim=1)
                out = F.scaled_dot_product_attention(
                    q_sdpa,
                    expanded_k,
                    expanded_v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=inputs.causal,
                )
        hidden = out.permute(0, 2, 1, 3).contiguous()
        if hidden.shape != (batch, seq_len, num_q_heads, head_dim):
            raise RuntimeError(f"unexpected SDPA output shape: {tuple(hidden.shape)}")
        return PrefillAttentionOutput(hidden=hidden)


def _sdpa_native_gqa_enabled() -> bool:
    return os.environ.get("GR_INFERENCE_PREFILL_SDPA_ENABLE_GQA") == "1"
