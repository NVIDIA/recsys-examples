# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gr_inference.gr_kernels import CAP_RMSNORM, KernelProfile
from gr_inference.gr_kernels.registry import KernelBackendInfo, KernelBackendRegistry
from gr_inference.gr_kernels.selection import KernelSelectionPolicy


def test_kernel_profile_roundtrip(tmp_path) -> None:
    profile = KernelProfile(
        schema_version=1,
        model={"family": "qwen3"},
        target={"device": "cuda"},
        selected={CAP_RMSNORM: "flashinfer"},
        benchmarks={"decode_step_ms": 29.0},
    )
    path = tmp_path / "profile.json"

    profile.save(path)
    loaded = KernelProfile.load(path)

    assert loaded.selected[CAP_RMSNORM] == "flashinfer"
    assert loaded.benchmarks["decode_step_ms"] == 29.0


def test_kernel_selection_policy_uses_profile() -> None:
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
    profile = KernelProfile(
        schema_version=1,
        model={},
        target={},
        selected={CAP_RMSNORM: "flashinfer"},
    )
    policy = KernelSelectionPolicy(
        capability_order={CAP_RMSNORM: ("torch",)},
        profile=profile,
    )

    assert policy.select(CAP_RMSNORM, registry).name == "flashinfer"


def test_kernel_profile_can_store_benchmark_results(tmp_path) -> None:
    profile = KernelProfile(
        schema_version=1,
        model={"family": "qwen3"},
        target={"device": "cuda"},
        selected={CAP_RMSNORM: "flashinfer"},
        benchmarks={
            "real_decode_step_ms": 29.2,
            "context_len": 16,
            "beam_width": 128,
        },
    )
    path = tmp_path / "profile.json"

    profile.save(path)
    loaded = KernelProfile.load(path)

    assert loaded.benchmarks["real_decode_step_ms"] == 29.2
    assert loaded.benchmarks["beam_width"] == 128
