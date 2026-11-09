# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The value buffer of a host table grows in place, inside the address range it reserved.

A host buffer used to be moved when it grew (``mremap``) and registered again as a whole,
after waiting for the kernels that use it. It now reserves the address space it can ever
need when the table is created and commits, locks and registers only the new pages when it
grows: its address never changes, so no kernel holds a stale pointer and nothing waits for
the device. What the table gets to reserve is decided here (``_host_value_rows_reserve``):
exactly what ``max_capacity`` can need for a table that evicts, a margin for one that never
does; growing past it is an error that names the knob.
"""

import os
import resource

import pytest
import torch
from dynamicemb import (
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    extendable_tensor,
)
from dynamicemb import key_value_table as kvt
from dynamicemb.optimizer import AdamDynamicEmbeddingOptimizer, OptimizerArgs
from dynamicemb.scored_hashtable import ScoreArg, ScorePolicy

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not extendable_tensor._HOST_IN_PLACE,
    reason="CUDA and an extension that grows host buffers in place are required",
)

DIM = 4
PAGE = os.sysconf("SC_PAGE_SIZE")


def host_options(init_capacity, max_capacity, no_eviction=False, bucket=128):
    return DynamicEmbTableOptions(
        dim=DIM,
        init_capacity=init_capacity,
        max_capacity=max_capacity,
        bucket_capacity=bucket,
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=torch.cuda.current_device(),
        score_strategy=DynamicEmbScoreStrategy.NO_EVICTION
        if no_eviction
        else DynamicEmbScoreStrategy.TIMESTAMP,
        local_hbm_for_values=0,
    )


def host_state(*args, use_rows=True, **kwargs):
    return kvt.create_table_state(
        [host_options(*args, **kwargs)],
        AdamDynamicEmbeddingOptimizer(OptimizerArgs()),
        use_rows=use_rows,
    )


def installed_memory() -> int:
    return os.sysconf("SC_PHYS_PAGES") * PAGE


def address_space_is_unlimited() -> bool:
    return resource.getrlimit(resource.RLIMIT_AS)[0] == resource.RLIM_INFINITY


def round_up(n: int) -> int:
    return (n + PAGE - 1) // PAGE * PAGE


# --- what a table reserves -----------------------------------------------------------


@pytest.mark.skipif(
    not address_space_is_unlimited(), reason="the reservation halves under RLIMIT_AS"
)
def test_a_table_that_evicts_reserves_exactly_what_max_capacity_can_need():
    state = host_state(256, 1000)
    row_bytes = state.table_value_dims_cpu[0] * 4
    # max_capacity rounded up to a bucket, and a bucket more for a rounded target
    rows = ((1000 + 127) // 128 + 1) * 128
    assert state.tables[0].reserved_bytes() == round_up(rows * row_bytes)


@pytest.mark.skipif(
    not address_space_is_unlimited(), reason="the reservation halves under RLIMIT_AS"
)
def test_a_table_that_cannot_grow_reserves_what_it_has():
    state = host_state(1024, 1024)
    assert state.size_snapshots is None
    assert state.tables[0].reserved_bytes() == state.tables[0].allocated_bytes()


@pytest.mark.skipif(
    not address_space_is_unlimited(), reason="the reservation halves under RLIMIT_AS"
)
def test_a_table_that_never_evicts_reserves_a_margin():
    state = host_state(256, 512, no_eviction=True)
    # not capped by max_capacity: at least 64 GiB of address space (less than the
    # installed memory is never useful), a margin of eight times max_capacity above it
    assert state.tables[0].reserved_bytes() >= min(installed_memory(), 64 << 30) - PAGE
    assert state.tables[0].reserved_bytes() <= round_up(installed_memory())


def test_the_margin_is_a_factor_of_at_least_one(monkeypatch):
    monkeypatch.setenv("DYNAMICEMB_HOST_RESERVE_FACTOR", "16")
    assert kvt._host_reserve_factor() == 16.0
    for bad in ("0.5", "-3", "lots", "nan"):
        monkeypatch.setenv("DYNAMICEMB_HOST_RESERVE_FACTOR", bad)
        with pytest.raises(ValueError, match="DYNAMICEMB_HOST_RESERVE_FACTOR"):
            kvt._host_reserve_factor()


# --- growth keeps the address -------------------------------------------------------


@pytest.mark.parametrize("use_rows", [True, False], ids=["rows", "slots"])
def test_a_host_table_keeps_the_address_of_its_value_buffer_through_its_growth(
    use_rows,
):
    state = host_state(256, 1 << 14, use_rows=use_rows)
    device = state.device
    keys = torch.arange(500, 560, dtype=torch.int64, device=device)
    tids = torch.zeros_like(keys)
    arg = ScoreArg(
        name=state.score_policy.name,
        value=torch.ones_like(keys, dtype=torch.uint64),
        policy=ScorePolicy.ASSIGN,
    )
    rows, _ = state.key_index_map.insert(keys, tids, arg, return_slots=True)
    vdim = state.table_value_dims_cpu[0]
    values = keys.to(torch.float32).unsqueeze(1).expand(-1, vdim).contiguous()
    kvt.store_to_flat_single_table(state, rows, 0, values)
    torch.cuda.synchronize()
    pointers = [b.tensor().data_ptr() for b in state.tables]
    table = state.table_ptrs_dev.tolist()
    assert state.table_ptrs_host == pointers == table

    for target in (1024, 4096, 1 << 14):
        kvt._expand_tables_impl(state, [True], [target])
        torch.cuda.synchronize()
        assert [b.tensor().data_ptr() for b in state.tables] == pointers
        assert state.table_ptrs_host == pointers
        assert state.table_ptrs_dev.tolist() == table
        assert state.tables[0].shape[0] == state.key_index_map.per_table_capacity_[0]

    arg_const = ScoreArg(
        name=state.score_policy.name, value=None, policy=ScorePolicy.CONST
    )
    _, founds, new_rows = state.key_index_map.lookup(keys, tids, arg_const)
    assert bool(founds.all())
    torch.testing.assert_close(
        kvt.load_from_flat_single_table(state, new_rows, 0), values
    )


def test_a_table_that_never_evicts_grows_far_past_its_max_capacity():
    """Not capped by ``max_capacity``: its value buffer goes to sixteen times it, which
    the in-place buffer's margin has room for (it would have been a hard stop at eight times
    if the margin were the whole reservation), and the rows it hands out keep their values.
    """
    max_capacity = 4096
    state = host_state(1 << 15, max_capacity, no_eviction=True)
    assert state.tables[0].shape[0] == 1 << 15  # eight times max_capacity already
    pointer = state.tables[0].tensor().data_ptr()
    key_slots = state.key_index_map.per_table_capacity_[0]  # twice the rows

    kvt._grow_values_only(state, [key_slots])
    torch.cuda.synchronize()

    rows = state.tables[0].shape[0]
    assert rows == key_slots and rows >= 16 * max_capacity
    assert state.tables[0].tensor().data_ptr() == pointer
    last = torch.tensor([rows - 1], dtype=torch.int64, device=state.device)
    value = torch.full((1, state.table_value_dims_cpu[0]), 7.5, device=state.device)
    kvt.store_to_flat_single_table(state, last, 0, value)
    torch.testing.assert_close(kvt.load_from_flat_single_table(state, last, 0), value)


def test_outgrowing_the_reservation_is_an_error_that_leaves_the_table_as_it_was():
    state = host_state(256, 300)  # evicts: 300 rounds to 3 buckets, reserves 4
    pointer = state.tables[0].tensor().data_ptr()
    rows = state.tables[0].shape[0]
    with pytest.raises(RuntimeError, match="address space"):
        state.tables[0].extend((1 << 20, state.table_value_dims_cpu[0]))
    assert state.tables[0].shape[0] == rows
    assert state.tables[0].tensor().data_ptr() == pointer
    # the regular growth to max_capacity still works
    kvt._expand_tables_impl(state, [True], [300])
    torch.cuda.synchronize()
    assert state.key_index_map.per_table_capacity_[0] == 384
    assert state.tables[0].shape[0] == 384
