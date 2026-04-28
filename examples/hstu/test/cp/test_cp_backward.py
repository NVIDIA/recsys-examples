# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
T4.2+T4.3 multi-GPU backward correctness test (torchrun pytest).

Compares `hstu_attn_varlen_cp_func.backward` (reverse-ring) against the
single-GPU baseline grads on the SPEC §3 Slice 4 matrix.

Run:
    bash examples/hstu/cp/run_cp_tests.sh --bwd

Driver:
  1. Build deterministic global batch on every rank with the same seed.
  2. Run single-GPU baseline forward+backward; record dq/dk/dv globally.
  3. Each rank dispatches its DualChunkSwap shard via the helper.
  4. Each rank invokes the CP wrapper with requires_grad=True.
  5. Each rank's local out + dout (= sliced global dout) drives backward.
  6. Each rank's q.grad / k.grad / v.grad (local shard) is scattered back
     into a globally-shaped buffer; all-reduced; compared to baseline grads.
  7. Tolerance: forward rtol/atol = 2e-2; gradient rtol/atol = 5e-2 (per
     existing `assert_hstu_close` convention — bwd has multiplier 5 vs
     fwd multiplier 2).
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest
import torch
import torch.distributed as dist
from hstu import (  # type: ignore[attr-defined]
    get_batch_on_this_cp_rank_for_hstu,
    hstu_attn_varlen_cp_func,
)

from .conftest import random_varlen_batch, single_gpu_baseline_fwd_bwd


def _alpha_for(head_dim: int) -> float:
    return 1.0 / (head_dim**0.5)


@pytest.fixture(scope="module")
def cp_world() -> Iterator[dict]:
    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        pytest.skip("multi-GPU test requires WORLD_SIZE >= 2 (run under torchrun)")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    cp_group = dist.new_group(list(range(world_size)), backend="nccl")
    cp_global_ranks = list(range(world_size))
    yield dict(
        cp_group=cp_group,
        cp_global_ranks=cp_global_ranks,
        rank=rank,
        world_size=world_size,
        device=torch.device(f"cuda:{rank}"),
    )
    dist.barrier()


SLICE4_MATRIX_CP2 = [
    dict(id="cp2_eq_64", seqlens=[64, 64, 64, 64], num_heads=2, head_dim=32),
    dict(id="cp2_varlen", seqlens=[16, 32, 48, 64], num_heads=2, head_dim=32),
]
SLICE4_MATRIX_CP4 = [
    dict(id="cp4_padded_heavy", seqlens=[8, 8, 8, 256], num_heads=2, head_dim=32),
    dict(id="cp4_eq", seqlens=[128, 256, 384, 512], num_heads=4, head_dim=64),
    dict(id="cp4_one_long", seqlens=[16, 16, 16, 4096], num_heads=2, head_dim=32),
    dict(id="cp4_hd256", seqlens=[128, 256, 384, 512], num_heads=4, head_dim=256),
]
SLICE4_MATRIX_CP8 = [
    dict(id="cp8_padded_heavy", seqlens=[16] * 7 + [1024], num_heads=4, head_dim=128),
    dict(id="cp8_eq", seqlens=[512, 1024, 1024, 2048], num_heads=4, head_dim=128),
]


def _run_one_correctness(entry: dict, cp_world: dict) -> None:
    cp_group = cp_world["cp_group"]
    cp_size = cp_world["world_size"]
    cp_rank = cp_world["rank"]
    device = cp_world["device"]

    seqlens = entry["seqlens"]
    head_dim = entry["head_dim"]
    num_heads = entry["num_heads"]
    alpha = _alpha_for(head_dim)
    max_seqlen = max(seqlens)

    # Step 1: build deterministic global batch on every rank with the same seed.
    q_global, k_global, v_global, cu_global = random_varlen_batch(
        seqlens, num_heads=num_heads, head_dim=head_dim, device=device, seed=0
    )

    # Step 2: deterministic dout on every rank (same seed).
    g = torch.Generator(device=device).manual_seed(101)
    dout_global = torch.randn(
        q_global.shape, generator=g, dtype=q_global.dtype, device=device
    )

    # Step 3: single-GPU baseline forward+backward (oracle).
    out_baseline, dq_baseline, dk_baseline, dv_baseline = single_gpu_baseline_fwd_bwd(
        q_global,
        k_global,
        v_global,
        cu_global,
        max_seqlen,
        dout_global,
        alpha=alpha,
        scaling_seqlen=max_seqlen,
        window_size=(-1, 0),
    )

    # Step 4: dispatch this rank's DualChunkSwap shard.
    q_loc, k_loc, v_loc, cu_loc, l2g, _ = get_batch_on_this_cp_rank_for_hstu(
        q_global,
        k_global,
        v_global,
        cu_global,
        cp_size=cp_size,
        cp_rank=cp_rank,
    )
    q_loc = q_loc.detach().clone().requires_grad_(True)
    k_loc = k_loc.detach().clone().requires_grad_(True)
    v_loc = v_loc.detach().clone().requires_grad_(True)
    dout_loc = dout_global[l2g]

    # Step 5: run CP forward + backward.
    out_loc = hstu_attn_varlen_cp_func(
        q=q_loc,
        k=k_loc,
        v=v_loc,
        cu_seqlens_q=cu_loc,
        cu_seqlens_k=cu_loc,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        scaling_seqlen=max_seqlen,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, 0),
        alpha=alpha,
        cp_group=cp_group,
        cp_global_ranks=cp_world["cp_global_ranks"],
    )
    out_loc.backward(dout_loc)
    dq_loc = q_loc.grad.detach()
    dk_loc = k_loc.grad.detach()
    dv_loc = v_loc.grad.detach()

    # Step 6: scatter local grads back into global shape; all-reduce SUM.
    def _scatter_allreduce(local: torch.Tensor) -> torch.Tensor:
        contrib = torch.zeros_like(q_global, dtype=torch.float32)
        contrib[l2g] = local.float()
        dist.all_reduce(contrib, op=dist.ReduceOp.SUM, group=cp_group)
        return contrib.to(q_global.dtype)

    dq_global = _scatter_allreduce(dq_loc)
    dk_global = _scatter_allreduce(dk_loc)
    dv_global = _scatter_allreduce(dv_loc)

    # Forward output gather (sanity).
    out_global = _scatter_allreduce(out_loc.detach())

    # Step 7: compare. fwd rtol/atol=2e-2; bwd looser at 5e-2.
    for name, t in [
        ("out", out_global),
        ("dq", dq_global),
        ("dk", dk_global),
        ("dv", dv_global),
    ]:
        assert (
            torch.isfinite(t).all().item()
        ), f"{name}: non-finite max={t.abs().max().item()}"
    torch.testing.assert_close(
        out_global.float(),
        out_baseline.float(),
        rtol=2e-2,
        atol=2e-2,
        msg=f"{entry['id']} fwd",
    )
    torch.testing.assert_close(
        dq_global.float(),
        dq_baseline.float(),
        rtol=5e-2,
        atol=5e-2,
        msg=f"{entry['id']} dq",
    )
    torch.testing.assert_close(
        dk_global.float(),
        dk_baseline.float(),
        rtol=5e-2,
        atol=5e-2,
        msg=f"{entry['id']} dk",
    )
    torch.testing.assert_close(
        dv_global.float(),
        dv_baseline.float(),
        rtol=5e-2,
        atol=5e-2,
        msg=f"{entry['id']} dv",
    )


@pytest.mark.parametrize("entry", SLICE4_MATRIX_CP2, ids=lambda e: e["id"])
def test_cp2(entry: dict, cp_world: dict) -> None:
    if cp_world["world_size"] != 2:
        pytest.skip(f"requires WORLD_SIZE=2; got {cp_world['world_size']}")
    _run_one_correctness(entry, cp_world)


@pytest.mark.parametrize("entry", SLICE4_MATRIX_CP4, ids=lambda e: e["id"])
def test_cp4(entry: dict, cp_world: dict) -> None:
    if cp_world["world_size"] != 4:
        pytest.skip(f"requires WORLD_SIZE=4; got {cp_world['world_size']}")
    _run_one_correctness(entry, cp_world)


@pytest.mark.parametrize("entry", SLICE4_MATRIX_CP8, ids=lambda e: e["id"])
def test_cp8(entry: dict, cp_world: dict) -> None:
    if cp_world["world_size"] != 8:
        pytest.skip(f"requires WORLD_SIZE=8; got {cp_world['world_size']}")
    _run_one_correctness(entry, cp_world)
