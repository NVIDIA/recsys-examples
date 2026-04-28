# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
T3.3+T3.4 multi-GPU forward correctness test (torchrun pytest).

Compares `hstu_attn_varlen_cp_func` against the single-GPU baseline
`hstu_attn_varlen_func` on the SPEC §3 Slice 3 matrix.

Run:
    torchrun --standalone --nproc-per-node=2 -m pytest \\
        examples/hstu/test/cp/test_cp_forward.py -v -k cp2

    torchrun --standalone --nproc-per-node=4 -m pytest \\
        examples/hstu/test/cp/test_cp_forward.py -v -k cp4

    torchrun --standalone --nproc-per-node=8 -m pytest \\
        examples/hstu/test/cp/test_cp_forward.py -v -k cp8

The wrapper `examples/hstu/cp/run_cp_tests.sh` covers all three.

Each test case:
  1. Materialises the global jagged batch on rank 0 with a deterministic seed;
     broadcasts it to all ranks.
  2. Each rank computes the single-GPU baseline on the global batch (every
     rank does it for simplicity — the result is bit-identical across ranks).
  3. Each rank dispatches via `get_batch_on_this_cp_rank_for_hstu` to extract
     its DualChunkSwap shard.
  4. Each rank invokes `hstu_attn_varlen_cp_func` on its shard.
  5. Each rank's local output is scattered back into a global output buffer
     via `gather_global_from_cp_rank`. The buffer is summed across ranks
     (all-reduce) so every rank can assert against the global baseline.
  6. Assert max |diff| within bf16 tolerance (rtol=2e-2, atol=2e-2).
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

from .conftest import random_varlen_batch, single_gpu_baseline_fwd


def _alpha_for(head_dim: int) -> float:
    return 1.0 / (head_dim**0.5)


@pytest.fixture(scope="module")
def cp_world() -> Iterator[dict]:
    """Initialise the CP process group from the torchrun env. Skip if not run
    under torchrun, or if WORLD_SIZE < 2.
    """
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


# Per-cp_size matrix entries. Each test below filters by cp_size via -k.
SLICE3_MATRIX_CP2 = [
    dict(id="cp2_eq_64", seqlens=[64, 64, 64, 64], num_heads=2, head_dim=32),
    dict(id="cp2_varlen", seqlens=[16, 32, 48, 64], num_heads=2, head_dim=32),
]
SLICE3_MATRIX_CP4 = [
    dict(id="cp4_padded_heavy", seqlens=[8, 8, 8, 256], num_heads=2, head_dim=32),
    dict(id="cp4_eq", seqlens=[128, 256, 384, 512], num_heads=4, head_dim=64),
    dict(id="cp4_one_long", seqlens=[16, 16, 16, 4096], num_heads=2, head_dim=32),
]
SLICE3_MATRIX_CP8 = [
    dict(id="cp8_padded_heavy", seqlens=[16] * 7 + [1024], num_heads=4, head_dim=128),
    dict(id="cp8_eq", seqlens=[512, 1024, 1024, 2048], num_heads=4, head_dim=128),
]


def _run_one_correctness(entry: dict, cp_world: dict) -> None:
    """Common driver: build batch, run baseline, run CP, assert match."""
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

    # Step 2: single-GPU baseline (identical on every rank with the same seed).
    out_baseline = single_gpu_baseline_fwd(
        q_global,
        k_global,
        v_global,
        cu_global,
        max_seqlen,
        alpha=alpha,
        scaling_seqlen=max_seqlen,
        window_size=(-1, 0),
    )

    # Step 3: dispatch this rank's DualChunkSwap shard from the global batch.
    q_loc, k_loc, v_loc, cu_loc, l2g, _ = get_batch_on_this_cp_rank_for_hstu(
        q_global,
        k_global,
        v_global,
        cu_global,
        cp_size=cp_size,
        cp_rank=cp_rank,
    )

    # Step 4: run the CP wrapper.
    out_local = hstu_attn_varlen_cp_func(
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

    # Step 5: scatter rank's shard back into a globally-shaped buffer; all-reduce
    # to get every rank's contribution accumulated. Each global token is owned
    # by exactly one rank, so SUM is identity.
    contrib = torch.zeros_like(q_global, dtype=torch.float32)
    contrib[l2g] = out_local.float()
    dist.all_reduce(contrib, op=dist.ReduceOp.SUM, group=cp_group)
    out_global = contrib.to(q_global.dtype)

    # Step 6: tolerance compare (bf16: rtol=2e-2, atol=2e-2).
    diff = (out_global.float() - out_baseline.float()).abs()
    max_abs = diff.max().item()
    base_max = out_baseline.float().abs().max().item()
    assert (
        torch.isfinite(out_global).all().item()
    ), f"non-finite CP output: max={max_abs}"
    assert torch.isfinite(out_baseline).all().item()
    torch.testing.assert_close(
        out_global.float(),
        out_baseline.float(),
        rtol=2e-2,
        atol=2e-2,
        msg=lambda m: (
            f"{entry['id']}: cp_size={cp_size} max_abs={max_abs:.3e} base_max={base_max:.3e}\n{m}"
        ),
    )


# ----------------------------------------------------------------------------
# Parametrised tests per cp_size group. Each test only runs when WORLD_SIZE
# matches; pytest -k filtering selects the right group at the runner level.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("entry", SLICE3_MATRIX_CP2, ids=lambda e: e["id"])
def test_cp2(entry: dict, cp_world: dict) -> None:
    if cp_world["world_size"] != 2:
        pytest.skip(f"requires WORLD_SIZE=2; got {cp_world['world_size']}")
    _run_one_correctness(entry, cp_world)


@pytest.mark.parametrize("entry", SLICE3_MATRIX_CP4, ids=lambda e: e["id"])
def test_cp4(entry: dict, cp_world: dict) -> None:
    if cp_world["world_size"] != 4:
        pytest.skip(f"requires WORLD_SIZE=4; got {cp_world['world_size']}")
    _run_one_correctness(entry, cp_world)


@pytest.mark.parametrize("entry", SLICE3_MATRIX_CP8, ids=lambda e: e["id"])
def test_cp8(entry: dict, cp_world: dict) -> None:
    if cp_world["world_size"] != 8:
        pytest.skip(f"requires WORLD_SIZE=8; got {cp_world['world_size']}")
    _run_one_correctness(entry, cp_world)
