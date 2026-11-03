# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference protection must be local to the logical table being modified."""

import pytest
import torch
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import InsertResult, ScorePolicy


@pytest.mark.parametrize("bucket_capacity", [128, 1024])
@pytest.mark.parametrize("key_type", [torch.int64, torch.uint64])
@pytest.mark.parametrize("policy", [ScorePolicy.ASSIGN, ScorePolicy.LRU_LFU])
@pytest.mark.parametrize("operation", ["insert", "collect", "evict", "overflow"])
@pytest.mark.parametrize("protected_table", [0, 1])
def test_multitable_reference_protection(
    bucket_capacity, key_type, policy, operation, protected_table
):
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()
    table = get_scored_table(
        capacity=[bucket_capacity, bucket_capacity],
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score", policy=policy)],
        enable_overflow=operation == "overflow",
    )

    def score(n, value=1):
        return ScoreArg(
            name="score",
            value=torch.full((n,), value, dtype=torch.uint64, device=device),
        )

    # One bucket per table makes the victim independent of the key hash. Both
    # tables are full, but only one table holds references to its resident keys.
    keys = torch.arange(1, 2 * bucket_capacity + 1, device=device).to(key_type)
    tids = torch.arange(2, device=device).repeat_interleave(bucket_capacity)
    results = torch.empty(keys.numel(), dtype=table.result_type, device=device)
    slots = table.insert(keys, tids, score(keys.numel()), results)
    assert (results == InsertResult.INSERT.value).all()
    assert ((slots >= 0) & (slots < bucket_capacity)).all()
    held = tids == protected_table
    table.increment_counter(slots[held], tids[held])
    main_counters = table._ref_counter[: 2 * bucket_capacity].reshape(2, -1)
    assert (main_counters[protected_table] == 1).all()
    assert (main_counters[1 - protected_table] == 0).all()

    new_key = torch.tensor([2 * bucket_capacity + 1], device=device).to(key_type)
    new_tid = torch.ones(1, dtype=torch.int64, device=device)
    result = torch.empty(1, dtype=table.result_type, device=device)
    if operation == "insert":
        new_slot = table.insert(new_key, new_tid, score(1, 2), result)
    elif operation == "collect":
        new_slot, count, victims, victim_tids = table.insert(
            new_key, new_tid, score(1, 2), result, collect_evicted=True
        )
        assert count.item() == (0 if protected_table == 1 else 1)
        if count.item():
            assert victim_tids[0].item() == 1
            assert bucket_capacity < victims[0].item() <= 2 * bucket_capacity
    else:
        insert = (
            table.insert_and_evict_with_counter_and_overflow
            if operation == "overflow"
            else table.insert_and_evict
        )
        new_slot, count, victims, _, _, victim_tids = insert(
            new_key, new_tid, score(1, 2), result
        )
        # The eviction API also collects rejected inputs (Busy). Only an Evict
        # result identifies a displaced resident key; count alone does not.
        if protected_table == 0:
            assert count == 1
            assert victim_tids[0].item() == 1
            assert bucket_capacity < victims[0].item() <= 2 * bucket_capacity
        elif operation == "overflow":
            assert count == 0
        else:
            assert count == 1
            assert victims[0].item() == new_key.item()

    expected = (
        InsertResult.EVICT
        if protected_table == 0
        else InsertResult.INSERT
        if operation == "overflow"
        else InsertResult.BUSY
    )
    assert result.item() == expected.value
    if expected == InsertResult.EVICT:
        assert 0 <= new_slot.item() < bucket_capacity
    elif operation == "overflow":
        assert bucket_capacity <= new_slot.item() < table.capacity(table_id=1)
    else:
        assert new_slot.item() == -1

    lookup_score = ScoreArg(name="score", policy=ScorePolicy.CONST)
    _, found, retained_slots = table.lookup(keys, tids, lookup_score)
    assert found[held].all(), "An in-use key was evicted from its own table"
    assert torch.equal(retained_slots[held], slots[held])
    assert found[:bucket_capacity].all(), "Inserting into table 1 changed table 0"
    assert found[bucket_capacity:].sum().item() == bucket_capacity - (
        protected_table == 0
    )
    assert (main_counters[protected_table] == 1).all()
    table.decrement_counter(slots[held], tids[held])
    assert (table._ref_counter == 0).all()

    # Releasing references restores the ability to evict from the main table.
    released_key = torch.tensor([2 * bucket_capacity + 2], device=device).to(key_type)
    table.insert(released_key, new_tid, score(1, 3), result)
    assert result.item() == InsertResult.EVICT.value
