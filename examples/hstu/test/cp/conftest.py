# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Pytest fixtures + reusable helpers for HSTU CP tests.

Phase 0 deliverable (per `tasks/plan.md` T0.1). This file is the *single
source of truth* for:
  - the canonical reference matrix consumed by every CP test,
  - the deterministic input generator,
  - the single-GPU baseline call (the numerical oracle for every CP
    correctness test, per SPEC §6),
  - the canonical installed `hstu_attn_varlen_func` signature (pinned
    so a future package upgrade fails loudly before any CP test runs).

Used by `test_reference.py` and (later) by Slice 3 / 4 multi-GPU CP tests.
"""

from __future__ import annotations

import inspect
from typing import Sequence

import pytest
import torch

# The installed `hstu` package — runtime authority per plan §Global rule 6.
hstu_attn_varlen_func = pytest.importorskip("hstu").hstu_attn_varlen_func


# ----------------------------------------------------------------------------
# Canonical installed `hstu_attn_varlen_func` signature (Global rule 6).
#
# Captured 2026-04-27 against the production container
# `gitlab-master.nvidia.com:5005/devtech-compute/distributed-recommender:devel_latest`
# on an A100 80GB host. The pin test asserts the live `inspect.signature(...)`
# matches this exact list. If a future package upgrade changes the API,
# this assertion fails first — explicit re-pin required, not silent drift.
# ----------------------------------------------------------------------------
CANONICAL_HSTU_PARAMS: list[tuple[str, str]] = [
    ("q", "no-default"),
    ("k", "no-default"),
    ("v", "no-default"),
    ("cu_seqlens_q", "no-default"),
    ("cu_seqlens_k", "no-default"),
    ("seqused_q", "no-default"),
    ("seqused_k", "no-default"),
    ("max_seqlen_q", "no-default"),
    ("max_seqlen_k", "no-default"),
    ("scaling_seqlen", "no-default"),
    ("num_contexts", "no-default"),
    ("num_targets", "no-default"),
    ("target_group_size", "1"),
    ("window_size", "(-1, -1)"),
    ("alpha", "1.0"),
    ("rab", "None"),
    ("has_drab", "False"),
    ("kv_cache", "None"),
    ("page_offsets", "None"),
    ("page_ids", "None"),
    ("last_page_lens", "None"),
    ("func", "None"),
    ("quant_mode", "-1"),
]


# ----------------------------------------------------------------------------
# Canonical reference matrix (SPEC §3 Slice 2 + Phase-0 boundary cases).
#
# Each row: a `(cp_size, seqlens, num_heads, head_dim)` tuple plus a
# human-readable id. Mask is pure causal `window_size=(-1, 0)` for every
# entry (sliding-causal is out of v0 scope per SPEC §2).
#
# Constraints enforced:
#   - every per-sample seqlen divisible by `2 * cp_size` (DualChunkSwap);
#   - head_dim ∈ {32, 64, 128, 256} (kernel-supported set);
#   - boundary coverage: smallest legal shape per cp_size, padding-heavy,
#     one head_dim=256 cell.
# ----------------------------------------------------------------------------
REFERENCE_MATRIX: list[dict] = [
    # SPEC §3 Slice 2 matrix
    dict(id="cp2_eq_64", cp_size=2, seqlens=[64, 64, 64, 64], num_heads=2, head_dim=32),
    dict(
        id="cp2_varlen", cp_size=2, seqlens=[16, 32, 48, 64], num_heads=2, head_dim=32
    ),
    dict(
        id="cp4_padded_heavy",
        cp_size=4,
        seqlens=[8, 8, 8, 256],
        num_heads=2,
        head_dim=32,
    ),
    dict(
        id="cp4_eq", cp_size=4, seqlens=[128, 256, 384, 512], num_heads=4, head_dim=64
    ),
    dict(
        id="cp8_padded_heavy",
        cp_size=8,
        seqlens=[16] * 7 + [1024],
        num_heads=4,
        head_dim=128,
    ),
    dict(
        id="cp8_eq",
        cp_size=8,
        seqlens=[512, 1024, 1024, 2048],
        num_heads=4,
        head_dim=128,
    ),
    # Phase-0 boundary additions
    dict(id="cp1_passthrough", cp_size=1, seqlens=[64], num_heads=2, head_dim=32),
    dict(id="cp2_smallest", cp_size=2, seqlens=[4], num_heads=2, head_dim=32),
    dict(id="cp4_smallest", cp_size=4, seqlens=[8], num_heads=2, head_dim=32),
    dict(id="cp8_smallest", cp_size=8, seqlens=[16], num_heads=2, head_dim=32),
    dict(
        id="cp4_one_long",
        cp_size=4,
        seqlens=[16, 16, 16, 4096],
        num_heads=2,
        head_dim=32,
    ),
    dict(
        id="cp4_hd256",
        cp_size=4,
        seqlens=[128, 256, 384, 512],
        num_heads=4,
        head_dim=256,
    ),
]

# Required minimum per plan T0.1 acceptance.
assert (
    len(REFERENCE_MATRIX) >= 12
), f"REFERENCE_MATRIX must have ≥ 12 tuples (plan T0.1 AC); got {len(REFERENCE_MATRIX)}"

# Validate divisibility at import time so a bad matrix fails before any test runs.
for _entry in REFERENCE_MATRIX:
    _bad = [s for s in _entry["seqlens"] if s % (2 * _entry["cp_size"]) != 0]
    assert not _bad, (
        f"matrix entry {_entry['id']!r}: seqlen(s) {_bad} not divisible by "
        f"2*cp_size={2 * _entry['cp_size']}"
    )
    assert _entry["head_dim"] in {
        32,
        64,
        128,
        256,
    }, f"matrix entry {_entry['id']!r}: head_dim {_entry['head_dim']} not in {{32, 64, 128, 256}}"


# ----------------------------------------------------------------------------
# Deterministic varlen jagged batch generator.
# ----------------------------------------------------------------------------
def random_varlen_batch(
    seqlens: Sequence[int],
    num_heads: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same seed → bit-identical (q, k, v, cu_seqlens). Standard HSTU THD layout."""
    g = torch.Generator(device=device).manual_seed(seed)
    total = sum(seqlens)
    q = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    k = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    v = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    cu_seqlens = (
        torch.tensor([0] + list(seqlens), dtype=torch.int32, device=device)
        .cumsum(0)
        .int()
    )
    return q, k, v, cu_seqlens


# ----------------------------------------------------------------------------
# Single-GPU baseline (oracle) — every CP correctness test compares against
# this. Pure-causal `window_size=(-1, 0)` is the only v0 mask flavour.
# ----------------------------------------------------------------------------
def single_gpu_baseline_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    alpha: float,
    scaling_seqlen: int | None = None,
    window_size: tuple[int, int] = (-1, 0),
) -> torch.Tensor:
    """One forward call to the production CUTLASS kernel."""
    if scaling_seqlen is None:
        scaling_seqlen = max_seqlen
    return hstu_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        scaling_seqlen=scaling_seqlen,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=window_size,
        alpha=alpha,
    )


def single_gpu_baseline_fwd_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dout: torch.Tensor,
    *,
    alpha: float,
    scaling_seqlen: int | None = None,
    window_size: tuple[int, int] = (-1, 0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward + backward; returns (out, dq, dk, dv). Tensors are detached and
    independent so the caller can re-run if needed.
    """
    q_g = q.detach().clone().requires_grad_(True)
    k_g = k.detach().clone().requires_grad_(True)
    v_g = v.detach().clone().requires_grad_(True)
    out = single_gpu_baseline_fwd(
        q_g,
        k_g,
        v_g,
        cu_seqlens,
        max_seqlen,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        window_size=window_size,
    )
    out.backward(dout)
    return out.detach(), q_g.grad.detach(), k_g.grad.detach(), v_g.grad.detach()


# ----------------------------------------------------------------------------
# Tolerance helper for CP tests. Wraps `torch.testing.assert_close` with the
# bf16 thresholds locked in SPEC §3.
# ----------------------------------------------------------------------------
def assert_cp_close(
    actual: torch.Tensor,
    ref: torch.Tensor,
    *,
    fwd: bool = True,
    atol: float = 2e-2,
    rtol: float = 2e-2,
) -> None:
    """Assert `actual` matches `ref` at bf16 tolerance. `fwd` is currently
    cosmetic (kept in the signature for parity with existing
    `assert_hstu_close`); future bwd-specific loosening can hang off it."""
    del fwd  # reserved
    torch.testing.assert_close(actual.float(), ref.float(), atol=atol, rtol=rtol)


# ----------------------------------------------------------------------------
# Pytest fixtures
# ----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


@pytest.fixture(params=REFERENCE_MATRIX, ids=lambda e: e["id"])
def matrix_entry(request) -> dict:
    """Per-test parametrization over the reference matrix. Yields a dict with
    `cp_size`, `seqlens`, `num_heads`, `head_dim`, `id`."""
    return request.param


def get_signature_summary() -> list[tuple[str, str]]:
    """Introspect the live installed signature into the same shape as
    CANONICAL_HSTU_PARAMS. Used by the pin test."""
    sig = inspect.signature(hstu_attn_varlen_func)
    out: list[tuple[str, str]] = []
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            out.append((name, "no-default"))
        else:
            out.append((name, repr(param.default)))
    return out
