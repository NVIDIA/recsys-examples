# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A key the cache cannot place stops the prefetch and leaves the storage intact.

The cache's insert refuses a key when its bucket and its table's overflow area
hold only keys that cannot be evicted (pinned by batches in flight, or being
inserted by the same call). Such a key comes back among the evicted ones with a
negative slot, and has no row for the forward to read. The insert kernel flags a
refused key (a variant of the kernel asked to) in pinned host memory, which the
read of the number of evicted keys the prefetch makes anyway waits for, and the
prefetch raises before it writes anything back: the refused key's evicted row was
never loaded, and written back it would have overwritten the key's storage row.

The cache here is one bucket (1024 slots, its buckets always hold 1024) plus an
overflow area of 3 buckets: one prefetch of more new keys than those 4096 slots
makes the insert refuse the rest; nothing checks the batch before the insert
does. Unwritten memory is filled with NaN so that a row stored without being
written shows.
"""

import pytest
import torch
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.key_value_table import get_insert_score_arg
from dynamicemb.types import CopyMode

DIM = 8
STORAGE_CAPACITY = 1 << 16
CACHE_ROWS = 1024  # one bucket of the cache
CACHE_SLOTS = 4 * CACHE_ROWS  # the bucket and the overflow area


@pytest.fixture(autouse=True)
def nan_for_unwritten_memory():
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    yield
    torch.utils.deterministic.fill_uninitialized_memory = fill
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def make_model():
    assert torch.cuda.is_available()
    options = [
        DynamicEmbTableOptions(
            dim=DIM,
            max_capacity=STORAGE_CAPACITY,
            init_capacity=STORAGE_CAPACITY,
            bucket_capacity=128,
            device_id=torch.cuda.current_device(),
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            # SGD keeps no optimizer state: a row is DIM floats.
            local_hbm_for_values=CACHE_ROWS * DIM * 4,
            caching=True,
            # A row holds its key: key % 100000 in every column.
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.DEBUG
            ),
        )
    ]
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=options,
        table_names=["t0"],
        feature_table_map=[0],
        output_dtype=torch.float32,
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.25,
    )
    cache = model._cache
    assert cache is not None
    km = cache._state.key_index_map
    # One bucket and an overflow area of three: the state's capacity counts both.
    assert km.capacity_ == CACHE_ROWS
    assert km.overflow_bucket_capacity_ == 3 * CACHE_ROWS
    assert cache._state.capacity == CACHE_SLOTS
    return model


def prefetch(model, keys):
    """The module's prefetch of one feature holding ``keys``; the state it queued."""
    offsets = torch.tensor([0, keys.numel()], dtype=torch.int64, device=keys.device)
    model.prefetch(keys, offsets)
    return model._prefetch_states[-1]


def test_refused_keys_stop_the_prefetch_and_keep_the_storage():
    model = make_model()
    # More new keys than the cache's 4096 slots (bucket + overflow), all distinct.
    keys = torch.arange(1, 6001, dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match="could not place"):
        prefetch(model, keys)
    torch.cuda.synchronize()

    # No pin is left behind.
    counter = model._cache._state.key_index_map._ref_counter
    assert torch.count_nonzero(counter).item() == 0

    # Every key went into the storage (it has room for all of them) with its
    # initial value, and kept it: the prefetch raised before it wrote back.
    table_ids = torch.zeros_like(keys)
    found = model._storage.find(keys, table_ids, copy_mode=CopyMode.EMBEDDING)
    founds, values = found[5], found[7]
    assert bool(founds.all())
    expected = (keys % 100000).float()[:, None].expand(-1, DIM)
    torch.testing.assert_close(values, expected, rtol=0, atol=0)


def test_a_batch_that_fits_is_not_refused():
    model = make_model()
    keys = torch.arange(1, CACHE_ROWS + 1, dtype=torch.int64, device="cuda")
    state = prefetch(model, keys)
    torch.cuda.synchronize()
    assert bool((state.slot_indices >= 0).all())
    counter = model._cache._state.key_index_map._ref_counter
    assert counter.sum().item() == keys.numel()


def test_the_insert_counts_the_keys_it_refuses():
    model = make_model()
    state = model._cache._state
    km = state.key_index_map
    keys = torch.arange(1, 6001, dtype=torch.int64, device="cuda")
    table_ids = torch.zeros_like(keys)
    score = get_insert_score_arg(state, keys.numel(), keys.device, None)
    (
        _,
        num_evicted,
        _,
        evicted_indices,
        _,
        _,
        num_refused,
    ) = km.insert_and_evict_with_counter_and_overflow(
        keys, table_ids, score, report_refused=True
    )
    # The empty cache evicts nothing: every "evicted" key is one it refused, the
    # keys past its slots, each with a negative index.
    assert num_refused == keys.numel() - CACHE_SLOTS
    assert num_evicted == num_refused
    assert int((evicted_indices < 0).sum()) == num_refused

    # Not asked for the count, the insert returns what it always did.
    more = torch.arange(10001, 10011, dtype=torch.int64, device="cuda")
    score = get_insert_score_arg(state, more.numel(), more.device, None)
    result = km.insert_and_evict_with_counter_and_overflow(
        more, torch.zeros_like(more), score
    )
    assert len(result) == 6
