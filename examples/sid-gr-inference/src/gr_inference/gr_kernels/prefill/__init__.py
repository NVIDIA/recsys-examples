# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefill attention backend contracts."""

from gr_inference.gr_kernels.prefill.auto_backend import AutoPrefillBackend
from gr_inference.gr_kernels.prefill.base import (
    MissingPrefillBackend,
    PrefillAttention,
    PrefillAttentionInputs,
    PrefillAttentionOutput,
)
from gr_inference.gr_kernels.prefill.flash_attn_backend import (
    FlashAttentionPrefillBackend,
    SGLangFlashAttentionPrefillBackend,
)
from gr_inference.gr_kernels.prefill.torch_sdpa_backend import TorchSDPAPrefillBackend

__all__ = [
    "AutoPrefillBackend",
    "FlashAttentionPrefillBackend",
    "MissingPrefillBackend",
    "PrefillAttention",
    "PrefillAttentionInputs",
    "PrefillAttentionOutput",
    "SGLangFlashAttentionPrefillBackend",
    "TorchSDPAPrefillBackend",
]
