"""Kernel wrapper boundaries."""

from gr_inference.gr_kernels.backends import (
    build_default_kernel_registry,
    flash_attn_backend_info,
    flashinfer_backend_info,
    gr_decode_atten_backend_info,
    torch_backend_info,
    torch_compile_backend_info,
    trtllm_aligned_backend_info,
)
from gr_inference.gr_kernels.fused_mlp import FusedMLP, TorchFusedMLPBackend
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
    KernelCapability,
)
from gr_inference.gr_kernels.selection import (
    KernelSelectionPolicy,
    default_kernel_selection_policy,
    reset_default_kernel_selection_policy,
)

__all__ = [
    "CAP_FUSED_ADD_RMSNORM",
    "CAP_FUSED_MLP",
    "CAP_GR_DECODE_ATTENTION",
    "CAP_PACKED_GEMM",
    "CAP_PREFILL_ATTENTION",
    "CAP_QK_NORM_ROPE",
    "CAP_RMSNORM",
    "CAP_ROPE",
    "CAP_ROPE_WITH_CACHE",
    "CAP_SAMPLING_TOPK",
    "KernelBackendInfo",
    "KernelBackendRegistry",
    "KernelCapability",
    "KernelProfile",
    "KernelSelectionPolicy",
    "FusedMLP",
    "TorchFusedMLPBackend",
    "build_default_kernel_registry",
    "flash_attn_backend_info",
    "default_kernel_selection_policy",
    "flashinfer_backend_info",
    "gr_decode_atten_backend_info",
    "reset_default_kernel_selection_policy",
    "torch_backend_info",
    "torch_compile_backend_info",
    "trtllm_aligned_backend_info",
]
