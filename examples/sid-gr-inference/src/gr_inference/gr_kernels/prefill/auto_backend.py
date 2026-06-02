# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto-selection prefill attention backend."""

from __future__ import annotations

from gr_inference.gr_kernels.prefill.base import (
    PrefillAttentionInputs,
    PrefillAttentionOutput,
)
from gr_inference.gr_kernels.prefill.flash_attn_backend import (
    FlashAttentionPrefillBackend,
)
from gr_inference.gr_kernels.prefill.torch_sdpa_backend import TorchSDPAPrefillBackend


class AutoPrefillBackend:
    """Prefer FlashAttention for GR prefill, fallback to PyTorch SDPA.

    The current benchmark points to FlashAttention as the better default for
    long-context GR shapes. SDPA remains useful for correctness and environments
    where flash-attn is not installed.
    """

    def __init__(
        self,
        prefer: tuple[str, ...] = ("flash_attn", "torch_sdpa"),
    ) -> None:
        self.prefer = prefer
        self.selected_backend: str | None = None
        self._backend = None

    def __call__(self, inputs: PrefillAttentionInputs) -> PrefillAttentionOutput:
        if self._backend is not None:
            return self._backend(inputs)

        errors: list[str] = []
        for name in self.prefer:
            backend = self._make_backend(name)
            try:
                output = backend(inputs)
            except Exception as exc:  # pragma: no cover - environment dependent
                errors.append(f"{name}: {exc}")
                continue
            self.selected_backend = name
            self._backend = backend
            return output

        raise RuntimeError("No prefill backend is available: " + "; ".join(errors))

    @staticmethod
    def _make_backend(name: str):
        if name == "flash_attn":
            return FlashAttentionPrefillBackend()
        if name == "torch_sdpa":
            return TorchSDPAPrefillBackend()
        raise ValueError(f"unknown prefill backend: {name}")
