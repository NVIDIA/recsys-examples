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
JaggedFlashAttnBlock: the default single-GPU GPT Transformer block for SID-GR.

Training and prefill use standard FA2 causal attention. Cached beam decode
uses the repo-vendored ``gr_decode_atten`` CuTe kernel.

Architecture per layer (standard pre-norm GPT):
  Input → LayerNorm → QKV Projection → Attention
        → Output Projection → Residual
        → LayerNorm → FFN → Residual → Output

Causal training and prefill use the standard FA2 operators shipped in the
base image. The optimized decode step calls the repo-vendored
``gr_decode_atten`` CuTe DSL kernel directly.

Reference: examples/hstu/modules/native_hstu_layer.py
"""

import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard flash_attn imports are deferred to runtime so this module can be
# imported in CPU-only development environments.

# beam_decode_attn kernel import — deferred so module loads without the kernel.
# Falls back to a pure-PyTorch reference implementation when the CuTe kernel
# is not installed (the real kernel requires ``cutlass`` and ``quack``).
_beam_decode_attn = None
_beam_decode_attn_import_error: Optional[ImportError] = None


def _ensure_gr_decode_atten_on_path() -> None:
    """Prefer the repo-vendored beam_decode_attn interface when present."""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    gr_decode_dir = os.path.join(repo_root, "corelib", "gr_decode_atten")
    if os.path.isdir(gr_decode_dir) and gr_decode_dir not in sys.path:
        sys.path.insert(0, gr_decode_dir)


def _beam_decode_attn_reference(
    q: torch.Tensor,
    k_context: torch.Tensor,
    v_context: torch.Tensor,
    k_beam: torch.Tensor,
    v_beam: torch.Tensor,
    topk_indices: torch.Tensor,
    decode_nums: int,
    softmax_scale: Optional[float] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pure-PyTorch reference for beam_decode_attn (single-pass).

    Shapes follow the CuTe kernel convention:
        q:            [B, Sq, W, Hq, D]
        k_context:    [B, Sk, Hkv, D]
        v_context:    [B, Sk, Hkv, D]
        k_beam:       [B, dn*W, Hkv, D]
        v_beam:       same
        topk_indices: [B, Sq, Hq, max_dn, W] int32
        seqused_k:    optional [B] int32; positions >= seqused_k[b] in
                      k_context are masked out of the softmax (matches the
                      CuTe kernel's seqused_k semantics).
        cu_seqlens_k: not supported in the reference path (jagged context K
                      would require a different layout). Raises if set.
    Returns:
        out: [B, Sq, W, Hq, D]  (same dtype as q)
        lse: None
    """
    import math

    if cu_seqlens_k is not None:
        # Jagged context K is a kernel-only optimization. The reference uses
        # dense expansion below, which doesn't have a sensible jagged form.
        # Fail explicitly so callers don't think this code path validates
        # use_jagged_kv=True.
        raise NotImplementedError(
            "_beam_decode_attn_reference does not implement jagged context "
            "K/V (cu_seqlens_k). Run the CuTe kernel for jagged validation."
        )

    B, Sq, W, Hq, D = q.shape
    Hkv = k_context.shape[2]
    ngroups = Hq // Hkv
    Sk = k_context.shape[1]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    q_f = q.float()
    k_ctx_f = k_context.float()
    v_ctx_f = v_context.float()
    k_beam_f = k_beam.float()
    v_beam_f = v_beam.float()

    if ngroups > 1:
        k_ctx_f = k_ctx_f.repeat_interleave(ngroups, dim=2)
        v_ctx_f = v_ctx_f.repeat_interleave(ngroups, dim=2)
        k_beam_f = k_beam_f.repeat_interleave(ngroups, dim=2)
        v_beam_f = v_beam_f.repeat_interleave(ngroups, dim=2)

    # Context KV → [B, 1, 1, Hq, Sk, D]
    k_ctx_exp = k_ctx_f.permute(0, 2, 1, 3).unsqueeze(1).unsqueeze(2)
    k_ctx_exp = k_ctx_exp.expand(B, Sq, W, Hq, Sk, D)
    v_ctx_exp = v_ctx_f.permute(0, 2, 1, 3).unsqueeze(1).unsqueeze(2)
    v_ctx_exp = v_ctx_exp.expand(B, Sq, W, Hq, Sk, D)

    if decode_nums > 0:
        idx = topk_indices[:, :, :, :decode_nums, :]  # [B, Sq, Hq, dn, W]
        idx = idx.permute(0, 1, 4, 2, 3).contiguous()  # [B, Sq, W, Hq, dn]
        b_idx = torch.arange(B, device=q.device)[:, None, None, None, None]
        h_idx = torch.arange(Hq, device=q.device)[None, None, None, :, None]
        k_beam_g = k_beam_f[b_idx, idx, h_idx]  # [B, Sq, W, Hq, dn, D]
        v_beam_g = v_beam_f[b_idx, idx, h_idx]
        k_all = torch.cat([k_ctx_exp, k_beam_g], dim=4)
        v_all = torch.cat([v_ctx_exp, v_beam_g], dim=4)
    else:
        k_all = k_ctx_exp
        v_all = v_ctx_exp

    scores = torch.einsum("bqwhd,bqwhsd->bqwhs", q_f * softmax_scale, k_all)

    if seqused_k is not None:
        # Mask out context K positions >= seqused_k[b] before softmax.
        # Beam K positions (concatenated to context K above) are always
        # valid, so they're not masked.
        ctx_pos = torch.arange(Sk, device=q.device)
        valid_ctx = ctx_pos[None, :] < seqused_k.to(torch.long)[:, None]  # [B, Sk]
        if decode_nums > 0:
            valid_beam = torch.ones(B, decode_nums, dtype=torch.bool, device=q.device)
            valid = torch.cat([valid_ctx, valid_beam], dim=1)  # [B, Sk + dn]
        else:
            valid = valid_ctx
        mask = ~valid[:, None, None, None, :]  # [B, 1, 1, 1, Sk(+dn)]
        scores = scores.masked_fill(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bqwhs,bqwhsd->bqwhd", attn, v_all)
    return out.to(q.dtype), None


def _get_beam_decode_attn():
    global _beam_decode_attn, _beam_decode_attn_import_error
    if _beam_decode_attn is None:
        _ensure_gr_decode_atten_on_path()
        try:
            from interface import beam_decode_attn

            _beam_decode_attn = beam_decode_attn
            _beam_decode_attn_import_error = None
        except ImportError as exc:
            _beam_decode_attn_import_error = exc
            _beam_decode_attn = _beam_decode_attn_reference
    return _beam_decode_attn


def _build_padded_context_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    seqused: torch.Tensor,
    max_seqlen: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Identity pass-through for padded context K/V.

    Padding-aware masking is handled by the kernel via the ``seqused_k``
    argument (added in our local interface.py extension). This helper
    exists for symmetry with the test-side construction and may grow
    additional logic (e.g. reshape) in the future.
    """
    return k, v


class JaggedGPTLayer(nn.Module):
    """
    One Transformer layer with jagged Flash Attention.

    Pre-norm GPT structure:
      x = x + Attn(LayerNorm(x))
      x = x + FFN(LayerNorm(x))

    Q/K/V are produced by a single fused linear (same pattern as HSTU's
    ``linear_uvqk``). Standard FA2 handles dense or varlen causal attention.

    Scope:
        This is the default single-GPU SID-GR transformer block. It owns its own
        ``nn.Linear`` weights for Q/K/V/output/MLP and is **not** a
        drop-in replacement for Megatron-Core's ``TransformerBlock``.
        In particular it does not support tensor parallelism, sequence
        parallelism, FP8 / Transformer Engine, or Megatron-shaped
        checkpoints.

        Existing SID-GR checkpoints trained against Megatron-Core need
        weight migration before this block can be substituted in. That
        migration is intentionally out of scope here. Select the Megatron
        reference backend when TP/SP/FP8 or Megatron checkpoint compatibility
        is required.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        kv_channels: Optional[int] = None,
        normalization: str = "LayerNorm",
        layernorm_epsilon: float = 1e-5,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = (
            kv_channels
            if kv_channels is not None
            else hidden_size // num_attention_heads
        )
        self.attention_size = self.num_heads * self.head_dim
        self.ffn_hidden_size = ffn_hidden_size
        if normalization == "RMSNorm":
            norm_cls = nn.RMSNorm
        elif normalization == "LayerNorm":
            norm_cls = nn.LayerNorm
        else:
            raise ValueError(f"Unsupported normalization: {normalization!r}")
        self.attention_dropout = attention_dropout

        # --- Attention sub-layers ---
        self.input_layernorm = norm_cls(hidden_size, eps=layernorm_epsilon)
        # Fused QKV projection: hidden_size → 3 * heads * head_dim
        self.linear_qkv = nn.Linear(
            hidden_size, 3 * self.attention_size, bias=False
        )
        # Output projection after attention
        self.linear_proj = nn.Linear(self.attention_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(hidden_dropout)

        # --- FFN sub-layers ---
        self.pre_mlp_layernorm = norm_cls(hidden_size, eps=layernorm_epsilon)
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
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: ``[1, total_tokens, hidden_size]`` when
                ``cu_seqlens`` is set, otherwise dense
                ``[batch, seqlen, hidden_size]``.
            cu_seqlens: ``[batch + 1]`` int32 offsets for varlen causal FA2.
            max_seqlen: maximum sequence length, required with ``cu_seqlens``.

        Returns:
            hidden_states: [batch, seqlen, hidden_size]
        """
        residual, q, k, v = self._qkv_projection(hidden_states)

        input_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        if cu_seqlens is not None:
            if q.shape[0] != 1:
                raise ValueError("cu_seqlens mode expects B=1 flattened input")
            if max_seqlen is None:
                raise ValueError("max_seqlen is required when cu_seqlens is set")
            from flash_attn.flash_attn_interface import flash_attn_varlen_func

            attn_out = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=self.head_dim ** (-0.5),
                causal=True,
            ).unsqueeze(0)
        else:
            from flash_attn.flash_attn_interface import flash_attn_func

            attn_out = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=self.head_dim ** (-0.5),
                causal=True,
            )

        if attn_out.dtype != input_dtype:
            attn_out = attn_out.to(input_dtype)

        return self._post_attention(residual, attn_out)

    def _qkv_projection(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared pre-attention: LayerNorm → QKV projection.

        Returns:
            residual, q, k, v — each of q/k/v is [..., num_heads, head_dim].
        """
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        qkv = self.linear_qkv(x)
        leading = qkv.shape[:-1]
        qkv = qkv.view(*leading, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=-3)
        return residual, q, k, v

    def _post_attention(
        self, residual: torch.Tensor, attn_out: torch.Tensor
    ) -> torch.Tensor:
        """Shared post-attention: output proj → residual → FFN."""
        leading = attn_out.shape[:-2]
        attn_out = attn_out.reshape(*leading, self.attention_size)
        attn_out = self.linear_proj(attn_out)
        attn_out = self.attn_dropout(attn_out)
        hidden_states = residual + attn_out

        residual = hidden_states
        x = self.pre_mlp_layernorm(hidden_states)
        x = self.mlp_fc1(x)
        x = self.activation_fn(x)
        x = self.mlp_fc2(x)
        x = self.mlp_dropout(x)
        return residual + x

    def prefill(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass that also returns the K/V cache for this layer.

        Two attention modes:
          1. ``cu_seqlens`` is set → flattened jagged input, plain
             per-sample causal. Uses upstream Tri Dao FA2
             ``flash_attn_varlen_func`` (``flash_attn.flash_attn_interface``),
             which is shipped pre-built in the nvcr base image and
             supports SM80+. ``max_seqlen`` must be supplied (FA2
             requires it; cute could infer but we don't use cute here).
          2. Otherwise → dense per-batch causal (``[B, S, ...]`` input,
             padded). Uses standard FA2 ``flash_attn_func``.

        Args:
            hidden_states: ``[1, total_tokens, hidden]`` for mode 1,
                ``[B, S, hidden]`` for mode 2.
            cu_seqlens: ``[B + 1]`` int32 offsets for mode 1.
            max_seqlen: max sequence length across the batch, required
                in mode 1.

        Returns:
            hidden_states: same leading shape as input.
            (k_cache, v_cache): each ``[..., num_heads, head_dim]``.
        """
        residual, q, k, v = self._qkv_projection(hidden_states)

        input_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        k_cache = k.clone()
        v_cache = v.clone()

        if cu_seqlens is not None:
            # Mode 1: jagged + plain causal via upstream FA2 varlen.
            # Squeeze B=1 leading dim so the call sees [total, H, D].
            assert q.shape[0] == 1, "cu_seqlens mode expects B=1 flattened input"
            assert max_seqlen is not None, (
                "max_seqlen is required when cu_seqlens is set "
                "(FA2 flash_attn_varlen_func needs it explicitly)"
            )
            from flash_attn.flash_attn_interface import flash_attn_varlen_func

            q_flat, k_flat, v_flat = q.squeeze(0), k.squeeze(0), v.squeeze(0)
            attn_flat = flash_attn_varlen_func(
                q_flat,
                k_flat,
                v_flat,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=self.head_dim ** (-0.5),
                causal=True,
            )
            attn_out = attn_flat.unsqueeze(0)
        else:
            # Mode 2: dense per-batch causal fast path.
            from flash_attn.flash_attn_interface import flash_attn_func

            attn_out = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=self.head_dim ** (-0.5),
                causal=True,
            )

        if attn_out.dtype != input_dtype:
            attn_out = attn_out.to(input_dtype)

        hidden_states = self._post_attention(residual, attn_out)
        return hidden_states, (k_cache, v_cache)

    def decode_beam(
        self,
        hidden_states: torch.Tensor,
        k_context: torch.Tensor,
        v_context: torch.Tensor,
        k_beam: Optional[torch.Tensor],
        v_beam: Optional[torch.Tensor],
        topk_indices: torch.Tensor,
        decode_nums: int,
        softmax_scale: Optional[float] = None,
        seqused_k: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        backend: str = "3kernel",
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode step using beam_decode_attn kernel.

        Args:
            hidden_states: [batch, beam_width, hidden_size]
            k_context: Dense  (cu_seqlens_k=None): [B, Sk, num_heads, head_dim]
                       Jagged (cu_seqlens_k set):  [total_k, num_heads, head_dim]
            v_context: same shape as k_context
            k_beam: [batch, prev_decode_nums * beam_width, num_heads, head_dim]
                or None if no previous decode steps.
            v_beam: same shape as k_beam, or None.
            topk_indices: [batch, 1, num_heads, decode_nums, beam_width] int32
            decode_nums: number of decode steps in beam KV (including self).
            seqused_k: [batch] int32 valid context length per sample (dense
                mode), or None.
            cu_seqlens_k: [batch+1] int32 jagged offsets for k_context /
                v_context, or None. Mutually exclusive with seqused_k. Only
                supported with backend="3kernel".
            backend: "3kernel" (default) or "dsl" (fused). The fused path
                does not support seqused_k or cu_seqlens_k; it silently
                ignores them and would produce wrong output on a padded
                batch. Use "3kernel" whenever the batch isn't uniform.

        Returns:
            hidden_states: [batch, beam_width, hidden_size]
            (k_new, v_new): each [batch, beam_width, num_heads, head_dim]
        """
        residual, q, k, v = self._qkv_projection(hidden_states)
        # q, k, v: [B, W, num_heads, head_dim]

        if softmax_scale is None:
            softmax_scale = self.head_dim ** (-0.5)

        B, W = q.shape[0], q.shape[1]
        k_new = k  # [B, W, num_heads, D]
        v_new = v

        # The kernel requires fp16/bf16 for q, k, v. We assume the caller
        # has already converted context_kv and beam_kv to a supported dtype
        # (generate_beam_decode does this once after prefill). We only
        # need to convert q/k_new/v_new if the layer was run in fp32
        # (e.g. unit tests with fp32 weights).
        input_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.bfloat16()
            k_new = k_new.bfloat16()
            v_new = v_new.bfloat16()
        # Sanity: cached tensors must already be fp16/bf16.
        assert k_context.dtype in (
            torch.float16,
            torch.bfloat16,
        ), f"k_context must be fp16/bf16, got {k_context.dtype}"
        assert v_context.dtype in (
            torch.float16,
            torch.bfloat16,
        ), f"v_context must be fp16/bf16, got {v_context.dtype}"

        if k_beam is not None:
            assert v_beam is not None, "k_beam and v_beam must be paired"
            assert k_beam.dtype in (
                torch.float16,
                torch.bfloat16,
            ), f"k_beam must be fp16/bf16, got {k_beam.dtype}"
            assert v_beam.dtype in (
                torch.float16,
                torch.bfloat16,
            ), f"v_beam must be fp16/bf16, got {v_beam.dtype}"
            k_beam_full = torch.cat([k_beam, k_new], dim=1)
            v_beam_full = torch.cat([v_beam, v_new], dim=1)
        else:
            k_beam_full = k_new
            v_beam_full = v_new

        # Reshape Q for beam_decode_attn: [B, 1, W, H, D]
        q_5d = q.unsqueeze(1)

        beam_decode_attn = _get_beam_decode_attn()
        # seqused_k / cu_seqlens_k are local kernel extensions on the
        # pipelined context-attention launch. The fused path doesn't
        # thread them through and would silently give wrong results on
        # padded batches — reject upfront.
        kernel_kwargs = {}
        if seqused_k is not None:
            if backend != "3kernel":
                raise ValueError(
                    f"seqused_k is only supported with backend='3kernel'; "
                    f"got backend={backend!r}"
                )
            kernel_kwargs["seqused_k"] = seqused_k
        if cu_seqlens_k is not None:
            if backend != "3kernel":
                raise ValueError(
                    f"cu_seqlens_k is only supported with backend='3kernel'; "
                    f"got backend={backend!r}"
                )
            if seqused_k is not None:
                raise ValueError("cu_seqlens_k and seqused_k are mutually exclusive")
            kernel_kwargs["cu_seqlens_k"] = cu_seqlens_k
        attn_out, _ = beam_decode_attn(
            q_5d,
            k_context,
            v_context,
            k_beam_full,
            v_beam_full,
            topk_indices,
            decode_nums,
            softmax_scale=softmax_scale,
            backend=backend,
            **kernel_kwargs,
        )
        # attn_out: [B, 1, W, H, D] → [B, W, H, D]
        attn_out = attn_out.squeeze(1)

        if attn_out.dtype != input_dtype:
            attn_out = attn_out.to(input_dtype)

        hidden_states = self._post_attention(residual, attn_out)
        return hidden_states, (k_new, v_new)


class JaggedFlashAttnBlock(nn.Module):
    """A stack of default FA2/CuTe SID-GR transformer layers.

    This module owns its own weights (not shared with Megatron-Core). Standard
    FA2 is used for training and prefill; ``gr_decode_atten`` is used for
    cached beam decode.

    Usage::

        block = JaggedFlashAttnBlock(
            num_layers=4,
            hidden_size=256,
            num_attention_heads=4,
            ffn_hidden_size=1024,
        )
        # padded input: [B, S, D]
        output = block(hidden_states)
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        kv_channels: Optional[int] = None,
        normalization: str = "LayerNorm",
        layernorm_epsilon: float = 1e-5,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = (
            kv_channels
            if kv_channels is not None
            else hidden_size // num_attention_heads
        )
        self.layers = nn.ModuleList(
            [
                JaggedGPTLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    ffn_hidden_size=ffn_hidden_size,
                    kv_channels=kv_channels,
                    normalization=normalization,
                    layernorm_epsilon=layernorm_epsilon,
                    hidden_dropout=hidden_dropout,
                    attention_dropout=attention_dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        norm_cls = nn.RMSNorm if normalization == "RMSNorm" else nn.LayerNorm
        self.final_layernorm = norm_cls(hidden_size, eps=layernorm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: flattened ``[1, total_tokens, hidden_size]`` when
                ``cu_seqlens`` is set, otherwise padded ``[B, S, hidden_size]``.
            cu_seqlens: offsets for standard FA2 varlen causal attention.
            max_seqlen: maximum sequence length, required with ``cu_seqlens``.

        Returns:
            hidden_states: [batch, seqlen, hidden_size]
        """
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states

    def prefill(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward through all layers, returning per-layer KV caches.

        Args:
            hidden_states: ``[1, total_tokens, hidden]`` when ``cu_seqlens``
                is set, otherwise ``[B, S, hidden]``.
            cu_seqlens: ``[B + 1]`` int32 offsets for the varlen + causal
                fast path (jagged input, per-sample causal). Standard
                FA2 ``flash_attn_varlen_func`` requires ``max_seqlen``
                to be supplied alongside.
            max_seqlen: max sequence length across the batch; required
                when ``cu_seqlens`` is set.

        Returns:
            hidden_states: same leading shape as input.
            kv_caches: list of (k, v) per layer.
        """
        kv_caches = []
        for layer in self.layers:
            hidden_states, kv = layer.prefill(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            kv_caches.append(kv)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, kv_caches

    def decode_beam(
        self,
        hidden_states: torch.Tensor,
        context_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        beam_kv_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
        topk_indices: torch.Tensor,
        decode_nums: int,
        seqused_k: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        backend: str = "3kernel",
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Decode one beam search step through all layers.

        Args:
            hidden_states: [batch, beam_width, hidden_size]
            context_kv_caches: per-layer (k, v) from prefill. In dense mode
                each (k, v) is [B, Sk, H, D]; in jagged mode (cu_seqlens_k
                set) each is [total_k, H, D].
            beam_kv_caches: per-layer (k_beam, v_beam) accumulated from
                previous decode steps, or None for each layer if no
                previous steps.
            topk_indices: [B, 1, H, decode_nums, W] int32
            decode_nums: total decode steps including self.
            seqused_k: [B] int32 valid context length per sample (dense mode),
                or None.
            cu_seqlens_k: [B+1] int32 jagged offsets for the per-layer
                k_context / v_context, or None. Mutually exclusive with
                seqused_k.
            backend: forwarded to the kernel; see JaggedGPTLayer.decode_beam.

        Returns:
            hidden_states: [batch, beam_width, hidden_size]
            new_beam_kvs: per-layer (k_new, v_new), each [B, W, H, D]
        """
        new_beam_kvs = []
        for i, layer in enumerate(self.layers):
            k_context, v_context = context_kv_caches[i]
            beam_cache_i = beam_kv_caches[i]
            k_beam = beam_cache_i[0] if beam_cache_i is not None else None
            v_beam = beam_cache_i[1] if beam_cache_i is not None else None
            hidden_states, kv_new = layer.decode_beam(
                hidden_states,
                k_context,
                v_context,
                k_beam,
                v_beam,
                topk_indices,
                decode_nums,
                seqused_k=seqused_k,
                cu_seqlens_k=cu_seqlens_k,
                backend=backend,
            )
            new_beam_kvs.append(kv_new)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, new_beam_kvs


class JaggedTransformerBlock(nn.Module):
    """Standard FA2 varlen wrapper for jagged ``[total_tokens, D]`` input."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        kv_channels: Optional[int] = None,
        normalization: str = "LayerNorm",
        layernorm_epsilon: float = 1e-5,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.block = JaggedFlashAttnBlock(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            ffn_hidden_size=ffn_hidden_size,
            kv_channels=kv_channels,
            normalization=normalization,
            layernorm_epsilon=layernorm_epsilon,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            activation=activation,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: jagged [total_tokens, hidden_size].
            cu_seqlens: [batch + 1] int32 cumulative offsets.
            max_seqlen: maximum sequence length in the batch.

        Returns:
            jagged output [total_tokens, hidden_size].
        """
        # [total_tokens, D] → [1, total_tokens, D]
        flat_input = hidden_states.unsqueeze(0)

        output = self.block(
            flat_input,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # [1, total_tokens, D] → [total_tokens, D]
        return output.squeeze(0)
