# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kept free of CuTe DSL imports so CPU-only tests can import it."""

from __future__ import annotations


def check_fused_inputs(
    q, topk_indices, decode_nums, seqused_k, cu_seqlens_k, backend
) -> None:
    """The rules ``interface.beam_decode_attn`` applies beyond ``_validate_inputs``."""
    import torch

    batch = q.shape[0]
    if q.shape[1] != 1:
        raise ValueError("q must be [batch, 1, width, heads, head_dim]")
    if not isinstance(decode_nums, int) or decode_nums < 0:
        raise ValueError("decode_nums must be a non-negative host int")
    if seqused_k is not None and (
        seqused_k.dtype != torch.int32 or seqused_k.shape != (batch,)
    ):
        raise ValueError("seqused_k must be int32 [batch]")
    if cu_seqlens_k is not None and (
        cu_seqlens_k.dtype != torch.int32 or cu_seqlens_k.shape != (batch + 1,)
    ):
        raise ValueError("cu_seqlens_k must be int32 [batch + 1]")
    if seqused_k is not None and cu_seqlens_k is not None:
        raise ValueError("cu_seqlens_k and seqused_k are mutually exclusive")
    if (seqused_k is not None or cu_seqlens_k is not None) and backend != "3kernel":
        raise ValueError("seqused_k and cu_seqlens_k need backend='3kernel'")
    for name, tensor in (
        ("topk_indices", topk_indices),
        ("seqused_k", seqused_k),
        ("cu_seqlens_k", cu_seqlens_k),
    ):
        if tensor is not None and tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")


def supports_vector_rows(q, k_beam, v_beam) -> bool:
    """Validated for head_dim 128 in 16-bit floats; every row moves as 128-bit accesses."""
    if q.shape[-1] != 128 or q.element_size() != 2:
        return False
    for tensor in (q, k_beam, v_beam):
        if tensor.stride(-1) != 1 or tensor.data_ptr() % 16:
            return False
        if any(
            stride * tensor.element_size() % 16
            for stride, size in zip(tensor.stride()[:-1], tensor.shape[:-1])
            if size > 1
        ):
            return False
    return True
