# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import pytest
from gr_inference.gr_kernels import (
    CAP_FUSED_MLP,
    CAP_GR_DECODE_ATTENTION,
    CAP_PACKED_GEMM,
    CAP_QK_NORM_ROPE,
    CAP_RMSNORM,
    CAP_ROPE,
    KernelBackendInfo,
    KernelBackendRegistry,
    KernelSelectionPolicy,
    build_default_kernel_registry,
    reset_default_kernel_selection_policy,
)


def test_kernel_backend_registry_selects_by_capability() -> None:
    registry = KernelBackendRegistry()
    registry.register(
        KernelBackendInfo(
            name="torch",
            capabilities=frozenset({CAP_RMSNORM, CAP_ROPE}),
        )
    )
    registry.register(
        KernelBackendInfo(
            name="flashinfer",
            capabilities=frozenset({CAP_RMSNORM}),
        )
    )

    assert registry.prefer(CAP_RMSNORM, ("flashinfer", "torch")).name == "flashinfer"
    assert registry.prefer(CAP_ROPE, ("flashinfer", "torch")).name == "torch"
    assert registry.prefer(CAP_GR_DECODE_ATTENTION, ("flashinfer",)) is None


def test_default_kernel_registry_contains_core_backends() -> None:
    summary = build_default_kernel_registry().summary()

    assert "torch" in summary
    assert "flashinfer" in summary
    assert "gr_decode_atten" in summary
    assert "torch_compile" in summary
    assert "trtllm_aligned" in summary
    assert "sgl_kernel" in summary
    assert CAP_QK_NORM_ROPE in summary["torch"]["capabilities"]
    assert CAP_QK_NORM_ROPE in summary["trtllm_aligned"]["capabilities"]
    assert CAP_PACKED_GEMM in summary["torch"]["capabilities"]
    assert CAP_FUSED_MLP in summary["torch"]["capabilities"]
    assert CAP_FUSED_MLP in summary["torch_compile"]["capabilities"]
    assert CAP_FUSED_MLP in summary["sgl_kernel"]["capabilities"]
    assert CAP_GR_DECODE_ATTENTION in summary["gr_decode_atten"]["capabilities"]


def test_kernel_selection_policy_prefers_capability_order() -> None:
    registry = KernelBackendRegistry()
    registry.register(
        KernelBackendInfo(
            name="torch",
            capabilities=frozenset({CAP_RMSNORM}),
        )
    )
    registry.register(
        KernelBackendInfo(
            name="flashinfer",
            capabilities=frozenset({CAP_RMSNORM}),
        )
    )
    policy = KernelSelectionPolicy(
        capability_order={CAP_RMSNORM: ("flashinfer", "torch")}
    )

    assert policy.select(CAP_RMSNORM, registry).name == "flashinfer"


def test_kernel_selection_policy_override_requires_capability() -> None:
    registry = KernelBackendRegistry()
    registry.register(
        KernelBackendInfo(
            name="flashinfer",
            capabilities=frozenset({CAP_RMSNORM}),
        )
    )
    policy = KernelSelectionPolicy(
        overrides={CAP_ROPE: "flashinfer"},
    )

    assert policy.select(CAP_ROPE, registry) is None


def test_kernel_selection_policy_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GR_INFERENCE_KERNEL_PRESET", "torch")
    reset_default_kernel_selection_policy()

    policy = KernelSelectionPolicy.from_env()

    assert policy.capability_order[CAP_RMSNORM] == ("torch",)


def test_default_qk_norm_rope_prefers_validated_path_before_trtllm() -> None:
    policy = KernelSelectionPolicy.from_env()

    assert policy.capability_order[CAP_QK_NORM_ROPE][:2] == ("flashinfer", "torch")
    assert "sgl_kernel" not in policy.capability_order[CAP_QK_NORM_ROPE]
    assert "trtllm_aligned" in policy.capability_order[CAP_QK_NORM_ROPE]


def test_default_fused_mlp_prefers_exact_trtllm_before_torch() -> None:
    policy = KernelSelectionPolicy.from_env()

    assert policy.capability_order[CAP_FUSED_MLP][:2] == ("trtllm_aligned", "torch")


def test_gr_trtllm_reference_backend_is_discovered_without_enable_flag(
    monkeypatch,
) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    reset_default_kernel_selection_policy()
    summary = build_default_kernel_registry().summary()

    assert summary["trtllm_aligned"]["metadata"]["gr_inference_trtllm_kernels"]
    assert summary["trtllm_aligned"]["available"]
    assert CAP_QK_NORM_ROPE in summary["trtllm_aligned"]["capabilities"]
    assert CAP_FUSED_MLP in summary["trtllm_aligned"]["capabilities"]
    assert CAP_PACKED_GEMM in summary["trtllm_aligned"]["capabilities"]
