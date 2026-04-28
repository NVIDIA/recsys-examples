# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
T3.1 smoke test (single-GPU; no torchrun).

Covers:
  - cp_size == 1 short-circuit: bit-exact match with `hstu_attn_varlen_func`
    on a small random input;
  - the 13-item hard-guard battery (each guard fires on its bad input);
  - cp_size > 1 path raises NotImplementedError (until T3.3 lands).
  - the dispatch helper round-trips identity at cp_size=1 and produces
    the right per-rank shard shapes at cp_size>1.

This file does NOT need torchrun; it exercises the public API without
ever creating a real CP process group, by passing a `FakeProcessGroup`
that simply reports `world_size`.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch

import pytest
import torch
from hstu import (  # type: ignore[attr-defined]
    GuardError,
    gather_global_from_cp_rank,
    get_batch_on_this_cp_rank_for_hstu,
    hstu_attn_varlen_cp_func,
    hstu_attn_varlen_func,
)

from .conftest import random_varlen_batch


# ----------------------------------------------------------------------------
# A fake `torch.distributed.ProcessGroup` that just reports a world size.
# Used to fake `cp_size > 1` without an actual NCCL setup.
# ----------------------------------------------------------------------------
class _FakeCPGroup:
    def __init__(self, size: int):
        self._size = size

    # `torch.distributed.get_world_size(group)` calls `group.size()` (or the
    # equivalent on the C++ side). We patch `dist.get_world_size` directly
    # below to return `self._size`, so this object only needs identity.
    def __repr__(self) -> str:
        return f"<FakeCPGroup size={self._size}>"


@contextmanager
def fake_cp_group(size: int) -> Iterator[_FakeCPGroup]:
    grp = _FakeCPGroup(size)
    with patch("torch.distributed.get_world_size", return_value=size):
        yield grp


# ============================================================================
# 1. cp_size == 1 short-circuit: bit-exact passthrough
# ============================================================================
def test_cp1_passthrough_bit_exact(cuda_device: torch.device) -> None:
    """When cp_group is None, the CP wrapper must equal the underlying kernel."""
    q, k, v, cu = random_varlen_batch(
        [16, 32, 48, 64], num_heads=2, head_dim=32, device=cuda_device, seed=42
    )
    alpha = 1.0 / 32**0.5
    max_s = 64

    expected = hstu_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=max_s,
        max_seqlen_k=max_s,
        scaling_seqlen=max_s,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, 0),
        alpha=alpha,
    )
    got = hstu_attn_varlen_cp_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=max_s,
        max_seqlen_k=max_s,
        scaling_seqlen=max_s,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, 0),
        alpha=alpha,
        cp_group=None,
    )
    assert torch.equal(got, expected), (
        f"cp=1 passthrough not bit-exact: max |Δ| = "
        f"{(got.float() - expected.float()).abs().max().item()}"
    )


def test_cp1_passthrough_with_explicit_world_size_1(cuda_device: torch.device) -> None:
    """cp_group with size=1 should also short-circuit identically."""
    q, k, v, cu = random_varlen_batch(
        [64], num_heads=2, head_dim=32, device=cuda_device, seed=1
    )
    alpha = 1.0 / 32**0.5
    expected = hstu_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=64,
        max_seqlen_k=64,
        scaling_seqlen=64,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, 0),
        alpha=alpha,
    )
    with fake_cp_group(1) as grp:
        got = hstu_attn_varlen_cp_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=64,
            max_seqlen_k=64,
            scaling_seqlen=64,
            num_contexts=None,
            num_targets=None,
            target_group_size=1,
            window_size=(-1, 0),
            alpha=alpha,
            cp_group=grp,
        )
    assert torch.equal(got, expected)


def test_cp1_passthrough_does_not_invoke_chunking(cuda_device: torch.device) -> None:
    """cp=1 must NOT call `get_batch_on_this_cp_rank_for_hstu` or any
    distributed primitive. Monkeypatch the helper to fail-on-call."""
    q, k, v, cu = random_varlen_batch(
        [64], num_heads=2, head_dim=32, device=cuda_device, seed=0
    )
    alpha = 1.0 / 32**0.5

    def _boom(*a, **kw):
        raise AssertionError("cp=1 path leaked into chunking helper")

    with patch(
        "hstu.hstu_attn_cp.get_batch_on_this_cp_rank_for_hstu", side_effect=_boom
    ):
        with patch("torch.distributed.batch_isend_irecv", side_effect=_boom):
            hstu_attn_varlen_cp_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=64,
                max_seqlen_k=64,
                scaling_seqlen=64,
                num_contexts=None,
                num_targets=None,
                target_group_size=1,
                window_size=(-1, 0),
                alpha=alpha,
                cp_group=None,
            )


# ============================================================================
# 2. Hard-guard battery (cp_size > 1, fake group). Each guard fires.
# ============================================================================
def _bad_kw(**override) -> dict:
    """Build a kwargs dict that's valid except for one override."""
    base = dict(
        cu_seqlens_q=None,  # filled in
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=64,
        max_seqlen_k=64,
        scaling_seqlen=64,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, 0),
        alpha=1.0 / 32**0.5,
        rab=None,
        has_drab=False,
        kv_cache=None,
        page_offsets=None,
        page_ids=None,
        last_page_lens=None,
        func=None,
        quant_mode=-1,
    )
    base.update(override)
    return base


@pytest.mark.parametrize(
    "override",
    [
        dict(num_contexts="MARK_ME_TENSOR"),
        dict(num_targets="MARK_ME_TENSOR"),
        dict(target_group_size=2),
        dict(window_size=(16, 0)),  # sliding-causal
        dict(window_size=(-1, -1)),  # full attention (no causal)
        dict(rab="MARK_ME_TENSOR"),
        dict(has_drab=True),
        dict(kv_cache="MARK_ME_TENSOR"),
        dict(page_offsets="MARK_ME_TENSOR"),
        dict(page_ids="MARK_ME_TENSOR"),
        dict(last_page_lens="MARK_ME_TENSOR"),
        dict(func="MARK_ME_TENSOR"),
        dict(quant_mode=0),
        dict(seqused_q="MARK_ME_TENSOR"),
        dict(seqused_k="MARK_ME_TENSOR"),
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()),
)
def test_guard_fires(override: dict, cuda_device: torch.device) -> None:
    """Every guarded mode raises GuardError on the cp_size>1 path."""
    q, k, v, cu = random_varlen_batch(
        [64, 64], num_heads=2, head_dim=32, device=cuda_device, seed=0
    )
    kw = _bad_kw(**override)
    kw["cu_seqlens_q"] = cu
    kw["cu_seqlens_k"] = cu
    # Materialise placeholders into actual tensors where required.
    for k_name, v_val in list(kw.items()):
        if v_val == "MARK_ME_TENSOR":
            if k_name in ("seqused_q", "seqused_k", "page_offsets", "last_page_lens"):
                kw[k_name] = torch.zeros(2, dtype=torch.int32, device=cuda_device)
            elif k_name == "page_ids":
                kw[k_name] = torch.zeros(2, dtype=torch.int32, device=cuda_device)
            elif k_name in ("num_contexts", "num_targets"):
                kw[k_name] = torch.zeros(2, dtype=torch.int32, device=cuda_device)
            elif k_name == "rab":
                kw[k_name] = torch.zeros(
                    2, 64, 64, dtype=torch.bfloat16, device=cuda_device
                )
            elif k_name == "kv_cache":
                kw[k_name] = torch.zeros(
                    1, 2, 1, 2, 32, dtype=torch.bfloat16, device=cuda_device
                )
            elif k_name == "func":
                kw[k_name] = torch.zeros(1, dtype=torch.int32, device=cuda_device)

    with fake_cp_group(2) as grp:
        with pytest.raises(GuardError):
            hstu_attn_varlen_cp_func(q=q, k=k, v=v, cp_group=grp, **kw)


def test_guard_head_dim_unsupported(cuda_device: torch.device) -> None:
    """head_dim outside {32, 64, 128, 256} fails."""
    q = torch.randn(8, 2, 48, dtype=torch.bfloat16, device=cuda_device)  # head_dim=48
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu = torch.tensor([0, 8], dtype=torch.int32, device=cuda_device)
    with fake_cp_group(2) as grp, pytest.raises(GuardError, match="head_dim"):
        hstu_attn_varlen_cp_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=8,
            max_seqlen_k=8,
            scaling_seqlen=8,
            num_contexts=None,
            num_targets=None,
            target_group_size=1,
            window_size=(-1, 0),
            alpha=1.0 / 48**0.5,
            cp_group=grp,
        )


def test_guard_divisibility(cuda_device: torch.device) -> None:
    """seqlen not divisible by 2*cp_size fails."""
    # seqlen=10, cp_size=2 → 2*cp_size=4, 10%4=2 ≠ 0
    q = torch.randn(10, 2, 32, dtype=torch.bfloat16, device=cuda_device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu = torch.tensor([0, 10], dtype=torch.int32, device=cuda_device)
    with fake_cp_group(2) as grp, pytest.raises(GuardError, match="not divisible"):
        hstu_attn_varlen_cp_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=10,
            max_seqlen_k=10,
            scaling_seqlen=10,
            num_contexts=None,
            num_targets=None,
            target_group_size=1,
            window_size=(-1, 0),
            alpha=1.0 / 32**0.5,
            cp_group=grp,
        )


def test_guard_cu_seqlens_q_neq_k(cuda_device: torch.device) -> None:
    q, k, v, cu = random_varlen_batch(
        [16, 16], num_heads=2, head_dim=32, device=cuda_device, seed=0
    )
    cu_other = torch.tensor([0, 16, 24], dtype=torch.int32, device=cuda_device)
    with fake_cp_group(2) as grp, pytest.raises(
        GuardError, match="cu_seqlens_q must equal cu_seqlens_k"
    ):
        hstu_attn_varlen_cp_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu_other,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=16,
            max_seqlen_k=16,
            scaling_seqlen=16,
            num_contexts=None,
            num_targets=None,
            target_group_size=1,
            window_size=(-1, 0),
            alpha=1.0 / 32**0.5,
            cp_group=grp,
        )


# ============================================================================
# 3. Multi-GPU path attempts real comm (fails on a fake group).
#
# T3.3 implemented `_multi_gpu_forward`. With a FakeProcessGroup (no NCCL),
# `dist.batch_isend_irecv` (or earlier `dist.get_rank`) must raise. The
# important property is that the wrapper does NOT silently pass through —
# it engages the CP path, which requires a real process group.
# ============================================================================
def test_cp_path_requires_real_pg(cuda_device: torch.device) -> None:
    q, k, v, cu = random_varlen_batch(
        [16, 16], num_heads=2, head_dim=32, device=cuda_device, seed=0
    )
    with fake_cp_group(2) as grp:
        with pytest.raises(
            (RuntimeError, AttributeError, AssertionError, ValueError, TypeError)
        ):
            hstu_attn_varlen_cp_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=16,
                max_seqlen_k=16,
                scaling_seqlen=16,
                num_contexts=None,
                num_targets=None,
                target_group_size=1,
                window_size=(-1, 0),
                alpha=1.0 / 32**0.5,
                cp_group=grp,
            )


def test_cp_backward_not_implemented(cuda_device: torch.device) -> None:
    """Backward path stub is still NotImplementedError until T4.2.

    We can't actually trigger backward with a FakeProcessGroup (forward will
    fail first). This test asserts the marker string is in the source so a
    refactor that drops it is caught.
    """
    import inspect

    from hstu.hstu_attn_cp import _HSTUVarlenCPFunc  # type: ignore[attr-defined]

    src = inspect.getsource(_HSTUVarlenCPFunc.backward)
    assert "T4.2" in src, "backward stub must reference plan T4.2"
    assert "NotImplementedError" in src


# ============================================================================
# 4. Dispatch helper round-trip
# ============================================================================
def test_dispatch_helper_cp1_returns_global(cuda_device: torch.device) -> None:
    q, k, v, cu = random_varlen_batch(
        [64, 32, 16], num_heads=2, head_dim=32, device=cuda_device, seed=0
    )
    q_l, k_l, v_l, cu_l, l2g, chunk_sizes = get_batch_on_this_cp_rank_for_hstu(
        q, k, v, cu, cp_size=1, cp_rank=0
    )
    assert torch.equal(q_l, q)
    assert torch.equal(k_l, k)
    assert torch.equal(v_l, v)
    assert torch.equal(cu_l, cu)


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_dispatch_helper_round_trip(cp_size: int, cuda_device: torch.device) -> None:
    """gather(scatter(global)) == global when summed across all ranks."""
    chunks = 2 * cp_size
    seqlens = [chunks * 4, chunks * 6, chunks * 8]
    q, k, v, cu = random_varlen_batch(
        seqlens, num_heads=2, head_dim=32, device=cuda_device, seed=0
    )
    q.shape[0]
    reconstructed = torch.zeros_like(q)
    for rank in range(cp_size):
        q_l, _, _, _, l2g, _ = get_batch_on_this_cp_rank_for_hstu(
            q, k, v, cu, cp_size=cp_size, cp_rank=rank
        )
        reconstructed[l2g] += q_l
    assert torch.equal(reconstructed, q), (
        f"cp_size={cp_size}: round-trip max |Δ| = "
        f"{(reconstructed.float() - q.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_dispatch_helper_balanced(cp_size: int, cuda_device: torch.device) -> None:
    """Per-rank shard size is 1/cp_size of total tokens (balanced load)."""
    chunks = 2 * cp_size
    seqlens = [chunks * 4, chunks * 6, chunks * 8]
    q, k, v, cu = random_varlen_batch(
        seqlens, num_heads=2, head_dim=32, device=cuda_device, seed=0
    )
    total = q.shape[0]
    expected_per_rank = total // cp_size
    for rank in range(cp_size):
        q_l, _, _, _, _, _ = get_batch_on_this_cp_rank_for_hstu(
            q, k, v, cu, cp_size=cp_size, cp_rank=rank
        )
        assert (
            q_l.shape[0] == expected_per_rank
        ), f"cp_size={cp_size}, rank={rank}: shard size {q_l.shape[0]} != {expected_per_rank}"


def test_dispatch_helper_bad_divisibility(cuda_device: torch.device) -> None:
    q = torch.randn(10, 2, 32, dtype=torch.bfloat16, device=cuda_device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu = torch.tensor([0, 10], dtype=torch.int32, device=cuda_device)
    with pytest.raises(GuardError, match="not divisible"):
        get_batch_on_this_cp_rank_for_hstu(q, k, v, cu, cp_size=2, cp_rank=0)


def test_dispatch_helper_bad_rank(cuda_device: torch.device) -> None:
    q, k, v, cu = random_varlen_batch(
        [16], num_heads=2, head_dim=32, device=cuda_device, seed=0
    )
    with pytest.raises(GuardError, match="cp_rank must be in"):
        get_batch_on_this_cp_rank_for_hstu(q, k, v, cu, cp_size=2, cp_rank=5)


def test_gather_global_inverse(cuda_device: torch.device) -> None:
    """gather_global_from_cp_rank(get_batch_on_this_cp_rank_for_hstu) == identity."""
    q, k, v, cu = random_varlen_batch(
        [16, 32], num_heads=2, head_dim=32, device=cuda_device, seed=0
    )
    total = q.shape[0]
    out = torch.zeros_like(q)
    for rank in range(2):
        q_l, _, _, _, l2g, _ = get_batch_on_this_cp_rank_for_hstu(
            q, k, v, cu, cp_size=2, cp_rank=rank
        )
        gather_global_from_cp_rank(q_l, l2g, global_total_tokens=total, out=out)
    assert torch.equal(out, q)


# ============================================================================
# 5. Public-API signature shape (mirrors installed kernel + 4 CP args).
# ============================================================================
def test_cp_func_signature_has_cp_args() -> None:
    sig = inspect.signature(hstu_attn_varlen_cp_func)
    params = list(sig.parameters.keys())
    for name in ("cp_group", "cp_global_ranks", "cp_stream", "cp_comm_type"):
        assert name in params, f"public CP arg `{name}` missing from signature"
    # Must mirror the installed kernel's positional names.
    must_have_positional = [
        "q",
        "k",
        "v",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "seqused_q",
        "seqused_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "scaling_seqlen",
        "num_contexts",
        "num_targets",
    ]
    for name in must_have_positional:
        assert name in params, f"signature missing kernel-mirror arg `{name}`"
