# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Phase-0 reference test (plan T0.1).

Locks the single-GPU baseline behaviour we use as the numerical oracle for
every later CP test:

  - the installed `hstu_attn_varlen_func` signature is pinned (Global
    rule 6 — runtime authority);
  - every entry in the canonical reference matrix runs without crashing
    and produces finite output / grads;
  - the same seed produces bit-identical output (oracle is deterministic).

If any of these regress, we want to know **before** any CP code runs.
"""

from __future__ import annotations

import pytest
import torch

from .conftest import (
    CANONICAL_HSTU_PARAMS,
    REFERENCE_MATRIX,
    get_signature_summary,
    random_varlen_batch,
    single_gpu_baseline_fwd,
    single_gpu_baseline_fwd_bwd,
)


# ----------------------------------------------------------------------------
# Signature pin — the most important guard in this file.
# ----------------------------------------------------------------------------
def test_hstu_signature_pinned() -> None:
    """Fail loudly if a `hstu` package upgrade changes the kernel signature.

    The canonical list lives in `conftest.CANONICAL_HSTU_PARAMS`. Any drift
    requires explicit re-pin (don't silently update the canonical list to
    match — the user should review what changed first).
    """
    live = get_signature_summary()
    assert live == CANONICAL_HSTU_PARAMS, (
        "Installed hstu.hstu_attn_varlen_func signature drifted from canonical pin.\n"
        f"  expected ({len(CANONICAL_HSTU_PARAMS)} params): {CANONICAL_HSTU_PARAMS}\n"
        f"  got      ({len(live)} params): {live}\n"
        "If this is intentional, update `conftest.CANONICAL_HSTU_PARAMS` AFTER "
        "auditing the signature change against SPEC §2 / plan T3.1 hard guards."
    )


# ----------------------------------------------------------------------------
# Matrix-driven baseline tests. Parametrised over REFERENCE_MATRIX via the
# `matrix_entry` fixture in conftest.py.
# ----------------------------------------------------------------------------
def _alpha_for(head_dim: int) -> float:
    return 1.0 / (head_dim**0.5)


def test_baseline_runs_fwd(matrix_entry: dict, cuda_device: torch.device) -> None:
    """Forward call returns the right shape, dtype, and finite values."""
    q, k, v, cu = random_varlen_batch(
        matrix_entry["seqlens"],
        matrix_entry["num_heads"],
        matrix_entry["head_dim"],
        device=cuda_device,
        seed=0,
    )
    max_seqlen = max(matrix_entry["seqlens"])
    out = single_gpu_baseline_fwd(
        q, k, v, cu, max_seqlen, alpha=_alpha_for(matrix_entry["head_dim"])
    )
    assert out.shape == q.shape, f"output shape {out.shape} != input shape {q.shape}"
    assert out.dtype == q.dtype, f"output dtype {out.dtype} != input dtype {q.dtype}"
    assert torch.isfinite(out).all().item(), (
        f"non-finite output for {matrix_entry['id']}: "
        f"min={out.min().item()}, max={out.max().item()}"
    )


def test_baseline_runs_fwd_bwd(matrix_entry: dict, cuda_device: torch.device) -> None:
    """Forward + backward both run; gradients are finite and correctly shaped."""
    q, k, v, cu = random_varlen_batch(
        matrix_entry["seqlens"],
        matrix_entry["num_heads"],
        matrix_entry["head_dim"],
        device=cuda_device,
        seed=0,
    )
    max_seqlen = max(matrix_entry["seqlens"])
    dout = torch.randn_like(q)
    out, dq, dk, dv = single_gpu_baseline_fwd_bwd(
        q, k, v, cu, max_seqlen, dout, alpha=_alpha_for(matrix_entry["head_dim"])
    )
    for name, t in [("out", out), ("dq", dq), ("dk", dk), ("dv", dv)]:
        assert t.shape == q.shape, f"{name} shape {t.shape} != q shape {q.shape}"
        assert torch.isfinite(t).all().item(), (
            f"non-finite {name} for {matrix_entry['id']}: "
            f"min={t.min().item()}, max={t.max().item()}"
        )


def test_baseline_self_consistent(
    matrix_entry: dict, cuda_device: torch.device
) -> None:
    """Same seed → bit-identical output (oracle is deterministic).

    Required so CP tests can re-run against the oracle in any order without
    spurious mismatches from non-deterministic kernels.
    """
    head_dim = matrix_entry["head_dim"]
    max_seqlen = max(matrix_entry["seqlens"])

    def _run() -> torch.Tensor:
        q, k, v, cu = random_varlen_batch(
            matrix_entry["seqlens"],
            matrix_entry["num_heads"],
            head_dim,
            device=cuda_device,
            seed=0,
        )
        return single_gpu_baseline_fwd(
            q, k, v, cu, max_seqlen, alpha=_alpha_for(head_dim)
        )

    a = _run()
    b = _run()
    assert torch.equal(a, b), (
        f"non-deterministic baseline for {matrix_entry['id']}: "
        f"max |a-b| = {(a.float() - b.float()).abs().max().item()}"
    )


# ----------------------------------------------------------------------------
# Matrix self-checks (cheap, run once, do not require CUDA).
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("entry", REFERENCE_MATRIX, ids=lambda e: e["id"])
def test_matrix_divisibility(entry: dict) -> None:
    """Every per-sample seqlen must be divisible by 2*cp_size (DualChunkSwap)."""
    bad = [s for s in entry["seqlens"] if s % (2 * entry["cp_size"]) != 0]
    assert (
        not bad
    ), f"{entry['id']}: seqlen(s) {bad} not divisible by 2*cp_size={2 * entry['cp_size']}"


@pytest.mark.parametrize("entry", REFERENCE_MATRIX, ids=lambda e: e["id"])
def test_matrix_head_dim_supported(entry: dict) -> None:
    assert entry["head_dim"] in {
        32,
        64,
        128,
        256,
    }, f"{entry['id']}: head_dim {entry['head_dim']} not in kernel-supported set"


def test_matrix_has_smallest_legal_per_cp_size() -> None:
    """For each cp_size in the matrix, there must be at least one entry whose
    seqlens equals `[2*cp_size]` (the smallest legal shape — single chunk per
    rank). Required by plan §Global rule 4 boundary coverage."""
    seen: dict[int, bool] = {}
    for e in REFERENCE_MATRIX:
        if e["seqlens"] == [2 * e["cp_size"]]:
            seen[e["cp_size"]] = True
    for cp in (2, 4, 8):
        assert seen.get(cp), f"no smallest-legal entry for cp_size={cp}"


def test_matrix_has_padding_heavy_case() -> None:
    """At least one entry where padding (per DualChunkSwap divisibility) is
    a substantial fraction of total tokens."""
    found = False
    for e in REFERENCE_MATRIX:
        seqlens = e["seqlens"]
        if not seqlens:
            continue
        ratio = max(seqlens) / min(seqlens)
        if ratio >= 16:
            found = True
            break
    assert found, "no padding-heavy entry in REFERENCE_MATRIX (max/min ratio ≥ 16)"


def test_matrix_has_head_dim_256() -> None:
    """At least one cell at head_dim=256 (full kernel-supported range)."""
    assert any(e["head_dim"] == 256 for e in REFERENCE_MATRIX)


def test_matrix_has_minimum_size() -> None:
    assert (
        len(REFERENCE_MATRIX) >= 12
    ), f"plan T0.1 AC requires ≥ 12 distinct tuples; got {len(REFERENCE_MATRIX)}"
