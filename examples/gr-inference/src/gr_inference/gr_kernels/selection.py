"""Kernel backend selection policy."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from gr_inference.gr_kernels.backends import build_default_kernel_registry
from gr_inference.gr_kernels.profile import KernelProfile
from gr_inference.gr_kernels.registry import (
    CAP_FUSED_ADD_RMSNORM,
    CAP_FUSED_MLP,
    CAP_GR_DECODE_ATTENTION,
    CAP_PACKED_GEMM,
    CAP_PREFILL_ATTENTION,
    CAP_QK_NORM_ROPE,
    CAP_RMSNORM,
    CAP_ROPE,
    CAP_ROPE_WITH_CACHE,
    CAP_SAMPLING_TOPK,
    KernelBackendInfo,
    KernelBackendRegistry,
)

DEFAULT_CAPABILITY_ORDER: dict[str, tuple[str, ...]] = {
    CAP_RMSNORM: ("flashinfer", "torch"),
    CAP_FUSED_ADD_RMSNORM: ("flashinfer",),
    CAP_ROPE: ("flashinfer", "torch"),
    CAP_ROPE_WITH_CACHE: ("flashinfer", "torch"),
    # SGLang's inplace qk-norm/RoPE fast path is tried directly at the Qwen3
    # call site and falls back here when unavailable. The TRT-LLM-aligned
    # packed-QKV qk_norm_rope kernel remains an explicit experimental backend.
    CAP_QK_NORM_ROPE: ("flashinfer", "torch", "trtllm_aligned"),
    CAP_PREFILL_ATTENTION: ("flash_attn", "torch"),
    CAP_GR_DECODE_ATTENTION: ("gr_decode_atten",),
    CAP_PACKED_GEMM: ("torch", "trtllm", "cutlass_triton"),
    CAP_FUSED_MLP: (
        "trtllm_aligned",
        "torch",
        "sgl_kernel",
        "trtllm",
        "cutlass_triton",
    ),
    CAP_SAMPLING_TOPK: ("flashinfer", "torch"),
}

TORCH_ONLY_CAPABILITY_ORDER: dict[str, tuple[str, ...]] = {
    capability: ("torch",) for capability in DEFAULT_CAPABILITY_ORDER
}
TORCH_ONLY_CAPABILITY_ORDER[CAP_GR_DECODE_ATTENTION] = ("gr_decode_atten",)

FLASHINFER_FIRST_CAPABILITY_ORDER = dict(DEFAULT_CAPABILITY_ORDER)


@dataclass(frozen=True)
class KernelSelectionPolicy:
    """Capability-oriented backend selection policy."""

    capability_order: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_CAPABILITY_ORDER)
    )
    overrides: dict[str, str] = field(default_factory=dict)
    profile: KernelProfile | None = None

    @classmethod
    def from_env(cls) -> "KernelSelectionPolicy":
        preset = os.environ.get("GR_INFERENCE_KERNEL_PRESET", "auto").lower()
        if preset == "torch":
            capability_order = dict(TORCH_ONLY_CAPABILITY_ORDER)
        elif preset in {"auto", "flashinfer"}:
            capability_order = dict(FLASHINFER_FIRST_CAPABILITY_ORDER)
        else:
            capability_order = dict(DEFAULT_CAPABILITY_ORDER)

        overrides: dict[str, str] = {}
        for capability in DEFAULT_CAPABILITY_ORDER:
            env_name = "GR_INFERENCE_KERNEL_" + capability.upper()
            value = os.environ.get(env_name)
            if value:
                overrides[capability] = value
        profile_path = os.environ.get("GR_INFERENCE_KERNEL_PROFILE")
        profile = KernelProfile.load(profile_path) if profile_path else None
        return cls(
            capability_order=capability_order,
            overrides=overrides,
            profile=profile,
        )

    def select(
        self,
        capability: str,
        registry: KernelBackendRegistry | None = None,
    ) -> KernelBackendInfo | None:
        if registry is None:
            registry = _default_kernel_registry()
        override = self.overrides.get(capability)
        if override:
            backend = registry.get(override)
            if backend is not None and backend.supports(capability):
                return backend
            return None
        if self.profile is not None:
            selected = self.profile.selected.get(capability)
            if selected:
                backend = registry.get(selected)
                if backend is not None and backend.supports(capability):
                    return backend
        return registry.prefer(
            capability,
            self.capability_order.get(capability, ()),
        )


_DEFAULT_POLICY: KernelSelectionPolicy | None = None
_DEFAULT_REGISTRY: KernelBackendRegistry | None = None


def _default_kernel_registry() -> KernelBackendRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = build_default_kernel_registry()
    return _DEFAULT_REGISTRY


def default_kernel_selection_policy() -> KernelSelectionPolicy:
    global _DEFAULT_POLICY
    if _DEFAULT_POLICY is None:
        _DEFAULT_POLICY = KernelSelectionPolicy.from_env()
    return _DEFAULT_POLICY


def reset_default_kernel_selection_policy() -> None:
    global _DEFAULT_POLICY, _DEFAULT_REGISTRY
    _DEFAULT_POLICY = None
    _DEFAULT_REGISTRY = None
