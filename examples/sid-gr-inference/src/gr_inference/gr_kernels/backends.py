# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in kernel backend descriptors."""

from __future__ import annotations

from gr_inference.gr_kernels.package_imports import (
    distribution_available as _distribution_available,
)
from gr_inference.gr_kernels.package_imports import (
    load_trtllm_kernel_registration as _load_trtllm_kernel_registration,
)
from gr_inference.gr_kernels.package_imports import (
    module_attribute_available as _module_attribute_available,
)
from gr_inference.gr_kernels.package_imports import (
    module_available as _module_available,
)
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


def build_default_kernel_registry() -> KernelBackendRegistry:
    registry = KernelBackendRegistry()
    registry.register(torch_backend_info())
    registry.register(sgl_kernel_backend_info())
    registry.register(torch_compile_backend_info())
    registry.register(trtllm_aligned_backend_info())
    registry.register(flashinfer_backend_info())
    registry.register(flash_attn_backend_info())
    registry.register(gr_decode_atten_backend_info())
    return registry


def torch_backend_info() -> KernelBackendInfo:
    return KernelBackendInfo(
        name="torch",
        available=_module_available("torch"),
        capabilities=frozenset(
            {
                CAP_RMSNORM,
                CAP_ROPE,
                CAP_QK_NORM_ROPE,
                CAP_PREFILL_ATTENTION,
                CAP_PACKED_GEMM,
                CAP_FUSED_MLP,
            }
        ),
    )


def torch_compile_backend_info() -> KernelBackendInfo:
    has_torch = _module_available("torch")
    has_compile = False
    if has_torch:
        try:
            import torch  # type: ignore[import-not-found]

            has_compile = hasattr(torch, "compile")
        except Exception:
            has_compile = False
    return KernelBackendInfo(
        name="torch_compile",
        available=has_torch and has_compile,
        capabilities=frozenset({CAP_FUSED_MLP}),
        metadata={
            "torch": has_torch,
            "torch.compile": has_compile,
        },
    )


def sgl_kernel_backend_info() -> KernelBackendInfo:
    has_sgl_kernel = _module_available("sgl_kernel")
    has_silu_and_mul = _module_attribute_available("sgl_kernel", "silu_and_mul")
    capabilities = set()
    if has_silu_and_mul:
        capabilities.add(CAP_FUSED_MLP)
    return KernelBackendInfo(
        name="sgl_kernel",
        available=has_sgl_kernel and bool(capabilities),
        capabilities=frozenset(capabilities or {CAP_FUSED_MLP}),
        metadata={
            "sgl_kernel": has_sgl_kernel,
            "sgl_kernel.silu_and_mul": has_silu_and_mul,
        },
    )


def trtllm_aligned_backend_info() -> KernelBackendInfo:
    has_torch = _module_available("torch")
    has_gr_trtllm = _module_available("gr_inference_trtllm_kernels")
    has_tokenspeed_kernel_dist = _distribution_available("tokenspeed-trtllm-kernel")
    has_tokenspeed_kernel_module = _module_available("tokenspeed_trtllm_kernel")
    has_tensorrt_llm = _module_available("tensorrt_llm")
    has_gr_trtllm_namespace = False
    has_gr_fused_qk_norm_rope = False
    has_gr_gated_mlp = False
    has_gr_packed_gemm = False
    has_trtllm_namespace = False
    has_fused_qk_norm_rope = False
    has_gated_mlp = False
    has_cublas_mm = False
    loaded_package = None
    if has_torch:
        try:
            loaded_package = _load_trtllm_kernel_registration()
            import torch  # type: ignore[import-not-found]

            has_gr_trtllm_namespace = hasattr(torch.ops, "gr_trtllm")
            has_gr_fused_qk_norm_rope = has_gr_trtllm_namespace and hasattr(
                torch.ops.gr_trtllm, "fused_qk_norm_rope"
            )
            has_gr_gated_mlp = has_gr_trtllm_namespace and hasattr(
                torch.ops.gr_trtllm, "gated_mlp"
            )
            has_gr_packed_gemm = has_gr_trtllm_namespace and hasattr(
                torch.ops.gr_trtllm, "packed_gemm"
            )
            has_trtllm_namespace = hasattr(torch.ops, "trtllm")
            has_fused_qk_norm_rope = has_trtllm_namespace and hasattr(
                torch.ops.trtllm, "fused_qk_norm_rope"
            )
            has_gated_mlp = has_trtllm_namespace and hasattr(
                torch.ops.trtllm, "gated_mlp"
            )
            has_cublas_mm = has_trtllm_namespace and hasattr(
                torch.ops.trtllm, "cublas_mm"
            )
        except Exception:
            has_gr_trtllm_namespace = False
            has_gr_fused_qk_norm_rope = False
            has_gr_gated_mlp = False
            has_gr_packed_gemm = False
            has_trtllm_namespace = False
            has_fused_qk_norm_rope = False
            has_gated_mlp = False
            has_cublas_mm = False
    capabilities = set()
    if has_fused_qk_norm_rope or has_gr_fused_qk_norm_rope:
        capabilities.add(CAP_QK_NORM_ROPE)
    if has_gated_mlp or has_gr_gated_mlp:
        capabilities.add(CAP_FUSED_MLP)
    if has_cublas_mm or has_gr_packed_gemm:
        capabilities.add(CAP_PACKED_GEMM)
    available = bool(capabilities)
    advertised_capabilities = capabilities or {
        CAP_QK_NORM_ROPE,
        CAP_FUSED_MLP,
        CAP_PACKED_GEMM,
    }
    return KernelBackendInfo(
        name="trtllm_aligned",
        available=available,
        capabilities=frozenset(advertised_capabilities),
        metadata={
            "torch": has_torch,
            "gr_inference_trtllm_kernels": has_gr_trtllm,
            "torch.ops.gr_trtllm": has_gr_trtllm_namespace,
            "torch.ops.gr_trtllm.fused_qk_norm_rope": has_gr_fused_qk_norm_rope,
            "torch.ops.gr_trtllm.gated_mlp": has_gr_gated_mlp,
            "torch.ops.gr_trtllm.packed_gemm": has_gr_packed_gemm,
            "tokenspeed-trtllm-kernel": has_tokenspeed_kernel_dist,
            "tokenspeed_trtllm_kernel module": (
                "available" if has_tokenspeed_kernel_module else "missing"
            ),
            "tensorrt_llm": has_tensorrt_llm,
            "loaded_package": loaded_package,
            "torch.ops.trtllm": has_trtllm_namespace,
            "torch.ops.trtllm.fused_qk_norm_rope": has_fused_qk_norm_rope,
            "torch.ops.trtllm.gated_mlp": has_gated_mlp,
            "torch.ops.trtllm.cublas_mm": has_cublas_mm,
        },
    )


def flashinfer_backend_info() -> KernelBackendInfo:
    has_package = _module_available("flashinfer")
    has_norm_module = _module_available("flashinfer.norm")
    has_rope_module = _module_available("flashinfer.rope")
    has_rmsnorm = _module_attribute_available("flashinfer.norm", "rmsnorm")
    has_fused_add_rmsnorm = _module_attribute_available(
        "flashinfer.norm",
        "fused_add_rmsnorm",
    )
    has_rope = _module_attribute_available("flashinfer.rope", "apply_rope_pos_ids")
    available = (
        has_package and has_norm_module and has_rope_module and has_rmsnorm and has_rope
    )
    return KernelBackendInfo(
        name="flashinfer",
        available=available,
        capabilities=frozenset(
            {
                CAP_RMSNORM,
                CAP_FUSED_ADD_RMSNORM,
                CAP_ROPE,
                CAP_ROPE_WITH_CACHE,
                CAP_QK_NORM_ROPE,
                CAP_SAMPLING_TOPK,
            }
        ),
        metadata={
            "flashinfer": has_package,
            "flashinfer.norm": has_norm_module,
            "flashinfer.rope": has_rope_module,
            "flashinfer.norm.rmsnorm": has_rmsnorm,
            "flashinfer.norm.fused_add_rmsnorm": has_fused_add_rmsnorm,
            "flashinfer.rope.apply_rope_pos_ids": has_rope,
        },
    )


def flash_attn_backend_info() -> KernelBackendInfo:
    available = _module_available("flash_attn")
    return KernelBackendInfo(
        name="flash_attn",
        available=available,
        capabilities=frozenset({CAP_PREFILL_ATTENTION}),
    )


def gr_decode_atten_backend_info() -> KernelBackendInfo:
    # The external kernel is optional and discovered at runtime via
    # GR_DECODE_ATTEN_ROOT, so expose the backend descriptor as available.
    return KernelBackendInfo(
        name="gr_decode_atten",
        available=True,
        capabilities=frozenset({CAP_GR_DECODE_ATTENTION}),
    )
