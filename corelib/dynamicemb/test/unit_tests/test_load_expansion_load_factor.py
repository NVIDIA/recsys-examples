# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from dynamicemb import DynamicEmbCheckMode, DynamicEmbScoreStrategy
from dynamicemb.dynamicemb_config import DynamicEmbEvictStrategy, DynamicEmbTableOptions
from dynamicemb.key_value_table import (
    DynamicEmbStorage,
    create_table_state,
    get_expand_info,
)
from dynamicemb.optimizer import OptimizerArgs, SGDDynamicEmbeddingOptimizer
from dynamicemb.scored_hashtable import ScoreArg, ScorePolicy

DIM = 8
BUCKET_CAPACITY = 128
MAX_CAPACITY = 1 << 20
MAX_LOAD_FACTOR = 0.5
INIT_CAPACITY = 1024


def _lfu_option(init_capacity):
    return DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=torch.cuda.current_device(),
        dim=DIM,
        max_capacity=MAX_CAPACITY,
        init_capacity=init_capacity,
        bucket_capacity=BUCKET_CAPACITY,
        max_load_factor=MAX_LOAD_FACTOR,
        safe_check_mode=DynamicEmbCheckMode.IGNORE,
        local_hbm_for_values=1024**3,
        score_strategy=DynamicEmbScoreStrategy.LFU,
        evict_strategy=DynamicEmbEvictStrategy.LFU,
    )


def _lfu_storage(init_capacity):
    return DynamicEmbStorage(
        [_lfu_option(init_capacity)], SGDDynamicEmbeddingOptimizer(OptimizerArgs())
    )


@pytest.mark.parametrize("n_new", [10_000, 300_000, MAX_CAPACITY])
def test_expansion_target_respects_max_load_factor(n_new):
    """An expansion lands at or below max_load_factor, unless max_capacity caps it."""
    state = create_table_state(
        [_lfu_option(init_capacity=INIT_CAPACITY)],
        SGDDynamicEmbeddingOptimizer(OptimizerArgs()),
    )
    flags, targets = get_expand_info(state, torch.tensor([0]), torch.tensor([n_new]))
    assert flags == [True]
    (target,) = targets
    assert target % BUCKET_CAPACITY == 0
    assert target <= MAX_CAPACITY
    assert target == MAX_CAPACITY or n_new / target <= MAX_LOAD_FACTOR


@pytest.mark.parametrize("n", [INIT_CAPACITY, 2 * INIT_CAPACITY, 200_000])
def test_load_keeps_every_key_of_a_table_below_max_capacity(tmp_path, n):
    """load() expands a fresh table once to fit the whole checkpoint. Sized to just fit,
    ~3.5% of the keys overflow their bucket and the lowest-score ones are dropped;
    checkpoints near init_capacity are the case where load() undercounted its keys."""
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(0)
    keys = torch.unique(torch.randint(1, 2**62, (2 * n,), generator=generator))[:n]
    keys = keys.to(device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    values = torch.randn(n, DIM, generator=generator).to(device)
    scores = torch.randint(1, 1000, (n,), generator=generator).to(device)

    src = _lfu_storage(init_capacity=MAX_CAPACITY)
    src.insert(keys, tids, values, scores=scores.to(torch.uint64))
    assert src.size() == n

    d = str(tmp_path)
    paths = [f"{d}/meta.json", f"{d}/keys", f"{d}/emb", f"{d}/score", f"{d}/opt"]
    src.dump(0, *paths, include_optim=False)

    dst = _lfu_storage(init_capacity=INIT_CAPACITY)
    dst.load(0, *paths, include_optim=False)
    assert dst.size() == n, f"load dropped {n - dst.size()} of {n} keys"
    _, founds, _ = dst.key_index_map.lookup(
        keys,
        tids,
        ScoreArg(name=dst._state.score_policy.name, policy=ScorePolicy.CONST),
    )
    assert torch.all(founds)
