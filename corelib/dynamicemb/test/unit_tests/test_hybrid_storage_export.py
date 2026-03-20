# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build HybridStorage directly (no BatchedDynamicEmbeddingTablesV2).
# Each step: unique + per-key counts on inputs; find; export check; insert missing keys; export again.
#
# Run (from corelib/dynamicemb):
#   torchrun --nnodes 1 --nproc_per_node 1 -m pytest test/unit_tests/test_hybrid_storage_export.py -v
# or:
#   ./test/unit_tests/test_hybrid_storage_export.sh

from __future__ import annotations

import os
import random
from copy import deepcopy
from typing import Dict, List, Tuple

import pytest
import torch
import torch.distributed as dist
from dynamicemb import (
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
    align_to_table_size,
)
from dynamicemb.dynamicemb_config import (
    DTYPE_NUM_BYTES,
    DynamicEmbEvictStrategy,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    get_optimizer_state_dim,
)
from dynamicemb.key_value_table import HybridStorage
from dynamicemb.optimizer import OptimizerArgs, SGDDynamicEmbeddingOptimizer
from dynamicemb.types import CopyMode
from dynamicemb_extensions import OptimizerType
from test_lfu_scores import validate_lfu_scores

TABLE_NAME = "t0"


def _require_cuda_dist() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not dist.is_initialized():
        pytest.skip(
            "Distributed not initialized; run with:\n"
            "  torchrun --nnodes 1 --nproc_per_node 1 -m pytest "
            "test/unit_tests/test_hybrid_storage_export.py -v"
        )


@pytest.fixture(scope="session", autouse=True)
def _session_dist_init():
    """torchrun sets RANK/WORLD_SIZE but does not call init_process_group for pytest."""
    if not torch.cuda.is_available():
        yield
        return
    if os.environ.get("WORLD_SIZE") is None:
        yield
        return
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _total_value_bytes_per_row(embedding_dim: int, optimizer_type: EmbOptimType) -> int:
    ot_cvt = {
        EmbOptimType.SGD: OptimizerType.SGD,
        EmbOptimType.EXACT_SGD: OptimizerType.SGD,
    }.get(optimizer_type, OptimizerType.Null)
    state_dim = get_optimizer_state_dim(ot_cvt, embedding_dim, torch.float32)
    value_dim = embedding_dim + state_dim
    return value_dim * DTYPE_NUM_BYTES[torch.float32]


def _make_sgd_optimizer(lr: float = 1e-3) -> SGDDynamicEmbeddingOptimizer:
    return SGDDynamicEmbeddingOptimizer(
        OptimizerArgs(
            learning_rate=lr,
            stochastic_rounding=False,
            gradient_clipping=False,
        )
    )


def _ensure_capacities_multiple_of_bucket(opt: DynamicEmbTableOptions) -> None:
    """Require init_capacity and max_capacity to be multiples of bucket_capacity."""
    bc = max(1, int(opt.bucket_capacity))

    def _floor_to_bucket(cap: int) -> int:
        c = max(1, int(cap))
        return max(bc, (c // bc) * bc)

    opt.init_capacity = _floor_to_bucket(opt.init_capacity)
    opt.max_capacity = _floor_to_bucket(opt.max_capacity)
    if opt.max_capacity < opt.init_capacity:
        opt.max_capacity = opt.init_capacity


def _full_table_options(
    num_embeddings: int,
    embedding_dim: int,
    score_strategy: DynamicEmbScoreStrategy,
    optimizer_type: EmbOptimType,
) -> DynamicEmbTableOptions:
    """Full-table footprint; local_hbm_for_values is set to total table bytes (matches Batched total_memory).

    Here init_capacity == max_capacity == bucket_capacity == aligned (already bucket-aligned).
    """
    aligned = align_to_table_size(num_embeddings)
    per_row = _total_value_bytes_per_row(embedding_dim, optimizer_type)
    total_memory = max(1, aligned * per_row)
    if score_strategy == DynamicEmbScoreStrategy.LFU:
        evict_strategy = DynamicEmbEvictStrategy.LFU
    elif score_strategy == DynamicEmbScoreStrategy.TIMESTAMP:
        evict_strategy = DynamicEmbEvictStrategy.LRU
    else:
        evict_strategy = DynamicEmbEvictStrategy.LRU
    return DynamicEmbTableOptions(
        dim=embedding_dim,
        embedding_dtype=torch.float32,
        index_type=torch.int64,
        init_capacity=aligned,
        max_capacity=aligned,
        bucket_capacity=aligned,
        score_strategy=score_strategy,
        evict_strategy=evict_strategy,
        caching=False,
        local_hbm_for_values=total_memory,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.CONSTANT,
            value=1e-1,
        ),
    )


def _build_storage(
    num_embeddings: int,
    embedding_dim: int,
    score_strategy: DynamicEmbScoreStrategy,
    local_hbm_budget_scale: float,
    optimizer_type: EmbOptimType,
) -> HybridStorage:
    """Hybrid split only: aligned with BatchedDynamicEmbeddingTablesV2 non-caching hybrid path."""
    full = _full_table_options(
        num_embeddings, embedding_dim, score_strategy, optimizer_type
    )
    aligned = full.init_capacity
    per_row = _total_value_bytes_per_row(embedding_dim, optimizer_type)
    total_memory = max(1, aligned * per_row)
    scaled_local_hbm = max(1, int(total_memory * local_hbm_budget_scale))
    full.local_hbm_for_values = scaled_local_hbm

    optimizer = _make_sgd_optimizer()

    cap_scale = scaled_local_hbm / total_memory if total_memory > 0 else 1.0
    hbm_options = deepcopy([full])
    for hbm_option in hbm_options:
        hbm_option.bucket_capacity = 1024
        cap = max(1, int(hbm_option.init_capacity * cap_scale))
        hbm_option.max_capacity = min(hbm_option.max_capacity, cap)
        hbm_option.init_capacity = min(hbm_option.init_capacity, cap)
        _ensure_capacities_multiple_of_bucket(hbm_option)

    host_options = deepcopy([full])
    storage_cap_scale = 1.0 - cap_scale
    for host_option in host_options:
        host_option.local_hbm_for_values = 0
        cap_h = max(1, int(host_option.init_capacity * storage_cap_scale))
        host_option.max_capacity = min(host_option.max_capacity, cap_h)
        host_option.init_capacity = min(host_option.init_capacity, cap_h)
        _ensure_capacities_multiple_of_bucket(host_option)

    return HybridStorage(hbm_options, host_options, optimizer)


def generate_batches_limited_keys(
    rank: int,
    world_size: int,
    batch_size: int,
    num_iterations: int,
    max_unique_key: int,
    num_embeddings: int,
    multi_hot_max: int,
    seed: int,
) -> Tuple[List[torch.Tensor], Dict[str, Dict[int, int]], List[List[int]],]:
    """Per step: 1D keys tensor for this rank; global expected LFU counts; per-step key order for LRU."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cap = min(max_unique_key, num_embeddings - 1)
    freq: Dict[int, int] = {}
    batches: List[torch.Tensor] = []
    batch_key_order: List[List[int]] = []

    batch_size_per_rank = max(1, batch_size // world_size)
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")

    for _ in range(num_iterations):
        keys_this_batch: List[int] = []

        for sample_id in range(batch_size):
            hotness = random.randint(1, multi_hot_max)
            indices = [random.randint(0, cap) for _ in range(hotness)]
            for idx in indices:
                freq[idx] = freq.get(idx, 0) + 1
            if sample_id // batch_size_per_rank == rank:
                keys_this_batch.extend(indices)

        batches.append(torch.tensor(keys_this_batch, dtype=torch.int64, device=dev))
        batch_key_order.append(keys_this_batch)

    return batches, {TABLE_NAME: freq}, batch_key_order


def export_scores_dict(storage: HybridStorage, device: torch.device) -> Dict[int, int]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])
    out: Dict[int, int] = {}
    for keys, _, _, scores in storage.export_keys_values(device, 65536, table_id=0):
        if keys.numel() == 0:
            continue
        for k, s in zip(keys.tolist(), scores.tolist()):
            out[int(k)] = int(s)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])
    return out


def _assert_export_matches_expected(
    act: Dict[int, int], expected: Dict[int, int], phase: str
) -> None:
    for k, sc in act.items():
        assert k in expected, f"{phase}: unexpected key {k} in export"
        assert (
            sc == expected[k]
        ), f"{phase}: key {k} score {sc} != expected {expected[k]}"


def validate_lru_timestamp_order(
    batch_key_order: List[List[int]],
    rank: int,
    world_size: int,
    key_to_score: Dict[int, int],
    admission_visited: Dict[str, set],
) -> None:
    """Reverse batch order monotonicity (same idea as test_embedding_dump_load)."""
    visited_keys = admission_visited.get(TABLE_NAME, set()).copy()
    min_score = float("inf")
    lasted_min_score = float("inf")
    for keys_flat in reversed(batch_key_order):
        for key in keys_flat:
            if key % world_size == rank and key not in visited_keys:
                assert (
                    key in key_to_score
                ), f"Key {key} missing in export for {TABLE_NAME}"
            else:
                continue
            assert (
                key_to_score[key] <= min_score
            ), f"{TABLE_NAME} key {key} score {key_to_score[key]} should be <= {min_score}"
            lasted_min_score = min(lasted_min_score, key_to_score[key])
            visited_keys.add(key)
        min_score = lasted_min_score
        lasted_min_score = min_score


def validate_embeddings_finite(
    embs: Dict[int, torch.Tensor],
    embedding_dim: int,
    init_value: float = 0.1,
    atol: float = 2.0,
) -> None:
    for key, vec in embs.items():
        assert vec.numel() >= embedding_dim
        emb_part = vec[:embedding_dim]
        assert torch.isfinite(emb_part).all(), f"key {key} non-finite emb"
        ref = torch.full_like(emb_part, init_value)
        assert torch.allclose(
            emb_part, ref, atol=atol, rtol=1.0
        ), f"key {key} emb deviates too much from init {init_value}"


def export_embeddings_dict(
    storage: HybridStorage, device: torch.device
) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    emb_dim = storage.embedding_dim(0)
    for keys, embeddings, _, scores in storage.export_keys_values(
        device, 65536, table_id=0
    ):
        if keys.numel() == 0:
            continue
        for k, vec in zip(keys.tolist(), embeddings):
            out[int(k)] = vec.detach().clone()[:emb_dim]
    return out


@pytest.mark.parametrize("local_hbm_budget_scale", [0.88, 0.55, 0.36, 0.1])
@pytest.mark.parametrize(
    "score_strategy",
    [DynamicEmbScoreStrategy.LFU, DynamicEmbScoreStrategy.TIMESTAMP],
)
@pytest.mark.parametrize("table_profile", ["small", "large"])
def test_hybrid_storage_export_scores_and_embeddings(
    local_hbm_budget_scale: float,
    score_strategy: DynamicEmbScoreStrategy,
    table_profile: str,
):
    """Exercise ``HybridStorage`` (two-tier HBM + host) without ``BatchedDynamicEmbeddingTablesV2``.

    **Goal**
        Drive the public ``Storage`` API manually---unique keys and per-key visit counts
        are computed outside the table, then ``find`` / ``insert`` / ``export_keys_values``
        are used to ensure exported scores and embeddings stay consistent with that
        ground truth, including under HBM pressure (eviction to host).

    **Parametrization**
        * ``local_hbm_budget_scale``: fraction of full table byte footprint reserved as
          ``local_hbm_for_values`` (0.88, 0.55, 0.36, 0.1). Always builds hybrid storage
          (no HBM-only / ``DynamicEmbStorage`` branch in this file).
        * ``score_strategy``: ``LFU`` or ``TIMESTAMP`` (LRU-style global timer).
        * ``table_profile``: ``small`` (256 logical slots) vs ``large`` (10000).

    **Per-step loop (per rank)**
        1. ``torch.unique`` + ``bincount`` -> per-key counts for this step.
        2. ``find(uniq, table_ids=0, CopyMode.VALUE, lfu_accumulated_frequency=counts)``
           for LFU; TIMESTAMP passes ``lfu_arg=None``.
        3. **LFU only**: ``export_keys_values`` and assert every exported key's score equals
           ``old_cum[key] + count_this_step[key]`` for keys present in export (keys not yet
           inserted do not appear).
        4. ``insert`` missing keys with constant value ``1e-1``; for LFU pass
           ``scores=miss_counts`` so new rows start at the batch frequency, not the default
           insert score.
        5. Update Python ``cum`` with this step's counts; **LFU only**: export again and
           assert scores match ``cum``.

    **Final checks**
        * Embeddings from export are finite and close to initializer constant ``1e-1``.
        * **LFU**: ``validate_lfu_scores`` vs ``expected_freq`` (strict, tolerance 0).
        * **TIMESTAMP**: ``validate_lru_timestamp_order`` on reversed batch order (no
          admission filter; empty visited set).
    """
    _require_cuda_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    num_embeddings = 256 if table_profile == "small" else 10000
    embedding_dim = 16
    batch_size = 16
    num_iterations = 5
    max_unique_key = 40
    multi_hot_max = 3

    storage = _build_storage(
        num_embeddings,
        embedding_dim,
        score_strategy,
        local_hbm_budget_scale,
        EmbOptimType.SGD,
    )
    storage.training = True

    batches, expected_freq, batch_key_order = generate_batches_limited_keys(
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        num_iterations=num_iterations,
        max_unique_key=max_unique_key,
        num_embeddings=num_embeddings,
        multi_hot_max=multi_hot_max,
        seed=(
            42
            + abs(
                hash((local_hbm_budget_scale, str(score_strategy), table_profile))
                % 100000
            )
        ),
    )

    cum: Dict[int, int] = {}
    value_dim = storage.value_dim(0)
    emb_dtype = storage.embedding_dtype()

    for keys_flat in batches:
        if keys_flat.numel() == 0:
            continue

        uniq, inv = torch.unique(keys_flat, sorted=True, return_inverse=True)
        counts = torch.bincount(inv, minlength=uniq.numel()).to(
            dtype=torch.int64, device=device
        )
        counts_by_key = {
            int(uniq[i].item()): int(counts[i].item()) for i in range(uniq.numel())
        }
        batch_key_set = set(counts_by_key.keys())

        tids = torch.zeros(uniq.numel(), dtype=torch.int64, device=device)

        old_cum = dict(cum)

        lfu_arg = counts if score_strategy == DynamicEmbScoreStrategy.LFU else None
        (
            _num_m,
            _mk,
            _mi,
            _mtid,
            _ms,
            founds,
            _out_scores,
            _vals,
        ) = storage.find(uniq, tids, CopyMode.VALUE, lfu_arg)

        if score_strategy == DynamicEmbScoreStrategy.LFU:
            torch.cuda.synchronize()
            act_find = export_scores_dict(storage, device)
            expected_after_find: Dict[int, int] = {}
            for k in act_find:
                extra = counts_by_key[k] if k in batch_key_set else 0
                expected_after_find[k] = old_cum.get(k, 0) + extra
            _assert_export_matches_expected(act_find, expected_after_find, "after find")

        miss = ~founds
        if miss.any():
            miss_keys = uniq[miss]
            miss_counts = counts[miss]
            n_miss = int(miss_keys.numel())
            vals = torch.full(
                (n_miss, value_dim),
                1e-1,
                dtype=emb_dtype,
                device=device,
            )
            miss_tids = torch.zeros(n_miss, dtype=torch.int64, device=device)
            ins_scores = (
                miss_counts.to(dtype=torch.int64)
                if score_strategy == DynamicEmbScoreStrategy.LFU
                else None
            )
            storage.insert(miss_keys, miss_tids, vals, scores=ins_scores)

        for k, c in counts_by_key.items():
            cum[k] = old_cum.get(k, 0) + c

        if score_strategy == DynamicEmbScoreStrategy.LFU:
            torch.cuda.synchronize()
            act_ins = export_scores_dict(storage, device)
            _assert_export_matches_expected(act_ins, cum, "after insert")

    act_scores = export_scores_dict(storage, device)
    act_embs = export_embeddings_dict(storage, device)
    validate_embeddings_finite(act_embs, embedding_dim)

    if score_strategy == DynamicEmbScoreStrategy.LFU:
        validate_lfu_scores(expected_freq, {TABLE_NAME: act_scores}, tolerance=0.0)
    else:
        # No admission_counter: do not filter to admission-only keys.
        validate_lru_timestamp_order(
            batch_key_order,
            rank,
            world_size,
            act_scores,
            {TABLE_NAME: set()},
        )
