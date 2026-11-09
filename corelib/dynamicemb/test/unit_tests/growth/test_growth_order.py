# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A growth queues its kernels on one stream and returns; the table is used from others.

``GrowthOrder`` (see ``dynamicemb/growth_order.py``) makes the growth come after the work
the other streams queued on the table before it, and the work they queue afterwards come
after the growth, without the host waiting. These tests put a long ``torch.cuda._sleep``
on the stream whose work the growth has to wait for, or on the growth stream itself, so
that a missing edge shows as a wrong result instead of as a race that is usually won.

What they can show depends on how much of a growth still blocks the host: while the
rehash reads back counts (``.item()``) the growth stream is idle when the call returns and
the host has waited for it, so the hazards of the late edges are hidden and the tests pass
without exercising them; they are written for the rehash that reads nothing back. The edge
of the growth towards work queued on another stream (``H1``) is visible in every case where
no device-wide wait happens before the rehash: a table that never evicts and only grows its
key map.
"""

import contextlib

import pytest
import torch
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb import key_value_table as kvt
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.key_value_table import DynamicEmbStorage
from dynamicemb.optimizer import AdamDynamicEmbeddingOptimizer, OptimizerArgs
from dynamicemb.scored_hashtable import LinearBucketTable, ScoreArg, ScorePolicy
from dynamicemb.types import CopyMode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

SLEEP = 200_000_000  # GPU cycles: a tenth of a second or so
DIM = 8


@pytest.fixture
def device():
    return torch.device("cuda", torch.cuda.current_device())


def options_of(device, *, no_eviction=False, init_capacity=1024, **overrides):
    kwargs = dict(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=device.index,
        dim=DIM,
        max_capacity=1 << 20,
        init_capacity=init_capacity,
        bucket_capacity=128,
        max_load_factor=0.5,
        safe_check_mode=DynamicEmbCheckMode.IGNORE,
        local_hbm_for_values=1024**3,
        score_strategy=DynamicEmbScoreStrategy.NO_EVICTION
        if no_eviction
        else DynamicEmbScoreStrategy.TIMESTAMP,
        caching=False,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.DEBUG
        ),
    )
    kwargs.update(overrides)
    return DynamicEmbTableOptions(**kwargs)


def build_storage(device, num_tables=1, **overrides):
    return DynamicEmbStorage(
        [options_of(device, **overrides) for _ in range(num_tables)],
        AdamDynamicEmbeddingOptimizer(OptimizerArgs()),
    )


def fill(storage, keys, table_id=0):
    """Insert ``keys`` into ``table_id`` with a value of their own; the slots they got."""
    state = storage._state
    tids = torch.full_like(keys, table_id)
    score = ScoreArg(
        name=state.score_policy.name,
        value=torch.ones_like(keys, dtype=torch.uint64),
        policy=ScorePolicy.ASSIGN,
    )
    rows, slots = state.key_index_map.insert(keys, tids, score, return_slots=True)
    values = (
        keys.to(torch.float32)
        .unsqueeze(1)
        .expand(-1, state.table_value_dims_cpu[table_id])
        .contiguous()
    )
    kvt.store_to_flat_single_table(state, rows, table_id, values)
    torch.cuda.synchronize()
    return slots


def double(state, table_id=0, num_tables=1):
    """Grow ``table_id`` to twice its capacity, on the current stream."""
    flags = [i == table_id for i in range(num_tables)]
    targets = [
        2 * state.key_index_map.per_table_capacity_[i] if flags[i] else -1
        for i in range(num_tables)
    ]
    kvt._expand_tables_impl(state, flags, targets)


@contextlib.contextmanager
def late_rehash(monkeypatch, table_id=1):
    """The rehash of ``table_id`` starts behind a sleep on the growth stream.

    ``copy_table_from`` is what the growth calls for a table it does not grow, and it
    comes before the table that does in table order, so a sleep put in front of it
    delays everything the growth launches for ``table_id``: every rehash is an
    asynchronous, late chain of kernels on the growth stream.
    """
    real = LinearBucketTable.copy_table_from

    def sleep_first(self, other, table_id_):
        torch.cuda._sleep(SLEEP)
        return real(self, other, table_id_)

    monkeypatch.setattr(LinearBucketTable, "copy_table_from", sleep_first)
    yield
    monkeypatch.setattr(LinearBucketTable, "copy_table_from", real)


# --- H1: the growth comes after what the other streams queued ------------------------


def test_the_growth_waits_for_work_queued_on_another_stream(device):
    """A table that never evicts grows its key map and nothing else, so no device-wide
    wait comes before the rehash. A backward on the forward stream that is still queued
    decrements the reference counters of the old key map: if the rehash copied the
    counters before it, the decrement would hit a map nobody uses any more and the new
    one would keep the key pinned for good."""
    storage = build_storage(device, no_eviction=True)
    state = storage._state
    keys = torch.arange(5_000, 5_300, dtype=torch.int64, device=device)
    slots = fill(storage, keys)
    tids = torch.zeros_like(keys)
    forward, prefetch = torch.cuda.Stream(), torch.cuda.Stream()
    capacity = state.key_index_map.per_table_capacity_[0]

    with torch.cuda.stream(forward):
        storage.increment_counter(slots, tids)  # the prefetch of the batch
        torch.cuda._sleep(SLEEP)  # the forward
        storage.decrement_counter(slots, tids)  # the backward, queued behind it
    with torch.cuda.stream(prefetch):
        # Nothing makes this stream wait for ``forward`` but the growth itself.
        double(state)
    torch.cuda.synchronize()

    km = state.key_index_map
    assert km.per_table_capacity_[0] == 2 * capacity
    assert int(km._ref_counter.sum()) == 0, "a decrement was lost in the rehash"
    _, founds, _ = km.lookup(
        keys,
        tids,
        ScoreArg(
            name=state.score_policy.name,
            value=torch.ones_like(keys, dtype=torch.uint64),
            policy=ScorePolicy.CONST,
        ),
    )
    assert bool(founds.all())


# --- H2: later work comes after the growth -------------------------------------------


def test_later_work_on_another_stream_sees_the_finished_growth(device, monkeypatch):
    storage = build_storage(device, num_tables=2)
    state = storage._state
    keys = torch.arange(7_000, 7_300, dtype=torch.int64, device=device)
    fill(storage, keys, table_id=1)
    tids = torch.ones_like(keys)
    prefetch = torch.cuda.Stream()

    with late_rehash(monkeypatch):
        with torch.cuda.stream(prefetch):
            double(state, table_id=1, num_tables=2)
        # The default stream, which nothing orders after ``prefetch``: the lookup is
        # ordered by the table's own edge, or reads the map the rehash has not built.
        found = storage.lookup_masked(keys, tids)
    torch.cuda.synchronize()
    assert bool(found.founds.all())
    assert state.key_index_map.per_table_capacity_ == [1024, 2048]


@pytest.mark.parametrize(
    "api",
    [
        "find",
        "incremental_dump",
        "export_keys_values",
        "erase_keys",
        "key_index_map",
        "size",
    ],
)
def test_a_public_api_on_another_stream_comes_after_the_growth(
    api, device, monkeypatch
):
    """The calls that are not part of a training step -- a dump, an export, an erase, a
    read of the key map -- run on whatever stream the caller is on. Each of them is
    ordered after a growth that is still running on the prefetch stream, and the result
    is what a ``torch.cuda.synchronize()`` before the call would have given."""
    storage = build_storage(device, num_tables=2)
    state = storage._state
    count = 300
    keys = torch.arange(9_000, 9_000 + count, dtype=torch.int64, device=device)
    fill(storage, keys, table_id=1)
    tids = torch.ones_like(keys)
    expected = set(keys.tolist())
    prefetch = torch.cuda.Stream()

    with late_rehash(monkeypatch):
        with torch.cuda.stream(prefetch):
            double(state, table_id=1, num_tables=2)
        # On the default stream, nothing ordering it after ``prefetch``:
        if api == "find":
            missing, *_ = storage.find(keys, tids, CopyMode.EMBEDDING)
            assert missing == 0
        elif api == "incremental_dump":
            dumped, *_ = storage.incremental_dump(1, 0, None)
            assert set(dumped.tolist()) == expected
        elif api == "export_keys_values":
            exported = set()
            for batch in storage.export_keys_values(device, table_id=1):
                exported |= set(batch[0].tolist())
            assert exported == expected
        elif api == "erase_keys":
            assert storage.erase_keys(1, keys) == count
        elif api == "key_index_map":
            assert storage.key_index_map.num_rows(1) == count
        else:
            assert int(storage.size()) == count
    torch.cuda.synchronize()


def test_flush_orders_the_storage_below_the_cache_after_its_growth(device, monkeypatch):
    """``flush_cache`` writes into the storage below the cache, which grows on the
    prefetch stream."""
    storage = build_storage(device)
    seen = []
    monkeypatch.setattr(
        DynamicEmbStorage, "wait_for_growth", lambda self: seen.append(self)
    )

    class NoCache:
        """What flush_cache needs of a cache that holds nothing."""

        class _State:
            num_tables = 0
            threads_in_wave = 1

        _state = _State()

    kvt.flush_cache(NoCache(), storage)
    assert seen == [storage]


# --- H3: the replaced key map is not handed out again early --------------------------


def test_the_replaced_key_map_is_not_reused_while_the_rehash_may_read_it(
    device, monkeypatch
):
    storage = build_storage(device, num_tables=2)
    state = storage._state
    fill(storage, torch.arange(100, 300, dtype=torch.int64, device=device), table_id=1)
    old = state.key_index_map
    block, size = old.table_storage_.data_ptr(), old.table_storage_.nbytes
    del old
    prefetch = torch.cuda.Stream()

    with late_rehash(monkeypatch):
        with torch.cuda.stream(prefetch):
            double(state, table_id=1, num_tables=2)
    if prefetch.query():
        pytest.skip("the growth had finished when the call returned: nothing to guard")
    # The block went back to the allocator when the growth returned. The stream it was
    # allocated on (the default one) would get it again at once; it must wait for the
    # growth stream.
    fresh = [torch.empty(size, dtype=torch.uint8, device=device) for _ in range(4)]
    for tensor in fresh:
        assert not (block <= tensor.data_ptr() < block + size), "the block was reused"
    torch.cuda.synchronize()


# --- who is registered ---------------------------------------------------------------


def test_a_state_that_cannot_grow_has_no_order(device):
    assert build_storage(device, init_capacity=1 << 20)._state.growth is None
    state = build_storage(device)._state
    assert state.growth is not None
    assert state.growth.device == state.device
    # Harmless on either
    build_storage(device, init_capacity=1 << 20).wait_for_growth()


def build_model(device, **overrides):
    return BatchedDynamicEmbeddingTablesV2(
        table_options=[options_of(device, **overrides)],
        output_dtype=torch.float32,
        table_names=["t_0"],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.SUM,
        use_index_dedup=False,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.0,
        stochastic_rounding=False,
    )


def batch_of(keys, device):
    keys = torch.tensor(keys, dtype=torch.int64, device=device)
    offsets = torch.arange(0, keys.numel() + 1, dtype=torch.int64, device=device)
    return keys, offsets


def test_forward_backward_and_eval_register_their_streams(device):
    model = build_model(device)
    order = model._storage._state.growth
    train, evaluate = torch.cuda.Stream(), torch.cuda.Stream()
    keys, offsets = batch_of(list(range(200, 260)), device)
    handles = lambda: {s.cuda_stream for s in order.users()}  # noqa: E731

    model.train()
    with torch.cuda.stream(train):
        out = model(keys, offsets)
        out.sum().backward()
    assert train.cuda_stream in handles()

    model.eval()
    with torch.cuda.stream(evaluate), torch.no_grad():
        model(keys, offsets)
    assert evaluate.cuda_stream in handles()
    torch.cuda.synchronize()
