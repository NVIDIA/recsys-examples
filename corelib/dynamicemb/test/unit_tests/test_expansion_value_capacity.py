# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from dynamicemb import DynamicEmbScoreStrategy, DynamicEmbTableOptions
from dynamicemb.key_value_table import (
    _expand_tables_impl,
    create_table_state,
    get_expand_info,
    load_from_flat_single_table,
    store_to_flat_single_table,
)
from dynamicemb.optimizer import AdamDynamicEmbeddingOptimizer, OptimizerArgs
from dynamicemb.scored_hashtable import ScoreArg, ScorePolicy


@pytest.mark.parametrize("on_host", [True, False])
@pytest.mark.parametrize("max_capacity", [300, 384])
@pytest.mark.parametrize(
    "score_strategy",
    [DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.STEP],
)
def test_expansion_value_capacity(on_host, max_capacity, score_strategy):
    """Every slot in an expanded key map must have a writable value row."""
    device = torch.device("cuda", torch.cuda.current_device())
    option = DynamicEmbTableOptions(
        dim=128,
        init_capacity=256,
        max_capacity=max_capacity,
        bucket_capacity=128,
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=device.index,
        score_strategy=score_strategy,
        local_hbm_for_values=0 if on_host else 1024 * 1024,
    )
    state = create_table_state([option], AdamDynamicEmbeddingOptimizer(OptimizerArgs()))
    keys = torch.tensor([11, 22, 33], device=device)
    table_ids = torch.zeros_like(keys)
    score = ScoreArg(
        name=state.score_policy.name,
        value=torch.ones_like(keys, dtype=torch.uint64),
        policy=ScorePolicy.ASSIGN,
    )
    indices = state.key_index_map.insert(keys, table_ids, score)
    values = torch.arange(3 * 384, device=device, dtype=torch.float32).view(3, 384)
    store_to_flat_single_table(state, indices, 0, values)

    # Exercise the max_capacity clamp, including a non-bucket-aligned target
    # as produced when HybridStorage subtracts its HBM capacity budget.
    flags, targets = get_expand_info(state, torch.tensor([128]), torch.tensor([1]))
    assert flags == [True]
    assert targets == [max_capacity]
    _expand_tables_impl(state, flags, targets)

    key_rows = state.key_index_map.per_table_capacity_[0]
    assert key_rows == 384
    assert state.tables[0].shape[0] == key_rows

    _, found, new_indices = state.key_index_map.lookup(keys, table_ids, score)
    assert bool(found.all())
    torch.testing.assert_close(
        load_from_flat_single_table(state, new_indices, 0), values
    )

    # Cover the final slot, which was beyond the original value allocation.
    tail = torch.tensor([key_rows - 1], device=device)
    tail_value = torch.full((1, 384), 3.25, device=device)
    store_to_flat_single_table(state, tail, 0, tail_value)
    torch.testing.assert_close(load_from_flat_single_table(state, tail, 0), tail_value)
