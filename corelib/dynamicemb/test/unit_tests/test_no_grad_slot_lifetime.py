# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward pins live until backward, or only until a no-grad forward returns.

Covers every storage layout a training forward can run on:

* ``hbm``    -- values all in HBM (HBM_DIRECT): slots are pinned in the storage.
* ``cache``  -- HBM cache over a host storage (CACHE): slots are pinned in the cache.
* ``hybrid`` -- HBM tier + host tier (DEFAULT): pins no slot.
* ``host``   -- host-only storage (DEFAULT): pins no slot, has no ref counter.

and a key that admission rejects (slot -1, nothing pinned).
"""

import pytest
import torch
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.embedding_admission import FrequencyAdmissionStrategy, KVCounter

DIM = 8
CAPACITY = 4096
# Bytes of one table's values (SGD keeps no optimizer state).
TABLE_BYTES = CAPACITY * DIM * 4
POOLINGS = [DynamicEmbPoolingMode.NONE, DynamicEmbPoolingMode.SUM]
# Layouts whose slots are pinned, and so have a ref counter to look at.
PINNING_LAYOUTS = ["hbm", "cache"]
ALL_LAYOUTS = ["hbm", "cache", "hybrid", "host"]


def make_model(pooling, layout, admit=False):
    assert torch.cuda.is_available()
    caching = layout == "cache"
    # Each table's HBM budget against its bytes picks the layout.
    local_hbm = {
        "hbm": 1024**3,
        "cache": 1024 * DIM * 4,  # a cache smaller than the working set
        "hybrid": TABLE_BYTES // 2,  # half of the table in HBM
        "host": 0,
    }[layout]
    options = [
        DynamicEmbTableOptions(
            dim=DIM,
            max_capacity=CAPACITY,
            init_capacity=CAPACITY,
            bucket_capacity=128,
            device_id=torch.cuda.current_device(),
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            local_hbm_for_values=local_hbm,
            caching=caching,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.DEBUG
            ),
            # A key is rejected the first time it is seen and admitted the second.
            admit_strategy=FrequencyAdmissionStrategy(
                threshold=2, counter=KVCounter(CAPACITY)
            )
            if admit
            else None,
        )
        for _ in range(2)
    ]
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=options,
        table_names=["t0", "t1"],
        feature_table_map=[0, 1],
        output_dtype=torch.float32,
        pooling_mode=pooling,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.25,
    )
    assert model.training
    assert (model._cache is not None) == caching
    if layout in PINNING_LAYOUTS:
        owner = model._cache if caching else model._storage
        return model, owner._state.key_index_map._ref_counter
    return model, None


def batch(iteration):
    keys = torch.cat(
        [torch.arange(16) + 16 * iteration, torch.arange(16) + 10000 + 16 * iteration]
    ).cuda()
    offsets = torch.arange(0, 33, 8, dtype=torch.int64, device="cuda")
    return keys, offsets


def check_values(output, keys, pooling):
    # DEBUG initializer: a row holds its key, also for a key admission rejects.
    values = (keys % 100000).float()
    if pooling == DynamicEmbPoolingMode.NONE:
        expected = values[:, None].expand(-1, DIM)
    else:
        expected = values.reshape(2, 2, DIM).sum(-1).T.repeat_interleave(DIM, dim=1)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def pins(counter):
    return counter.sum().item()


def no_grad_forward(model, keys, offsets, pooling):
    with torch.no_grad():
        output = model(keys, offsets)
    assert not output.requires_grad
    check_values(output, keys, pooling)


@pytest.mark.parametrize("pooling", POOLINGS)
@pytest.mark.parametrize("layout", ALL_LAYOUTS)
@pytest.mark.parametrize("admit", [False, True])
def test_no_grad_forward_releases_slot_pins(pooling, layout, admit):
    model, counter = make_model(pooling, layout, admit)
    for iteration in range(8):
        keys, offsets = batch(iteration)
        no_grad_forward(model, keys, offsets, pooling)
        if counter is not None:
            assert torch.count_nonzero(counter).item() == 0
    if admit:
        # Second sight of every key: admitted now, so slots are taken and must be
        # released all the same.
        for iteration in range(8):
            keys, offsets = batch(iteration)
            no_grad_forward(model, keys, offsets, pooling)
            if counter is not None:
                assert torch.count_nonzero(counter).item() == 0


@pytest.mark.parametrize("pooling", POOLINGS)
@pytest.mark.parametrize("layout", ALL_LAYOUTS)
def test_grad_forward_keeps_pins_until_backward(pooling, layout):
    model, counter = make_model(pooling, layout)
    keys, offsets = batch(0)
    output = model(keys, offsets)
    assert output.requires_grad
    check_values(output, keys, pooling)
    if counter is not None:
        assert pins(counter) > 0
    output.backward(torch.zeros_like(output))
    if counter is not None:
        assert torch.count_nonzero(counter).item() == 0
    no_grad_forward(model, keys, offsets, pooling)
    if counter is not None:
        assert torch.count_nonzero(counter).item() == 0


@pytest.mark.parametrize("pooling", POOLINGS)
@pytest.mark.parametrize("layout", PINNING_LAYOUTS)
def test_admission_grad_forward_pins_only_admitted_keys(pooling, layout):
    model, counter = make_model(pooling, layout, admit=True)
    keys, offsets = batch(0)
    # First sight: every key is rejected (slot -1). Nothing is pinned, and backward
    # has nothing to release.
    output = model(keys, offsets)
    check_values(output, keys, pooling)
    assert pins(counter) == 0
    output.backward(torch.zeros_like(output))
    assert pins(counter) == 0
    # Second sight: admitted, pinned until backward.
    output = model(keys, offsets)
    check_values(output, keys, pooling)
    assert pins(counter) > 0
    output.backward(torch.zeros_like(output))
    assert torch.count_nonzero(counter).item() == 0


@pytest.mark.parametrize("pooling", POOLINGS)
@pytest.mark.parametrize("layout", PINNING_LAYOUTS)
@pytest.mark.parametrize("admit", [False, True])
@pytest.mark.parametrize("grad_first", [False, True])
def test_no_grad_release_leaves_other_batch_pinned(pooling, layout, admit, grad_first):
    """Two batches in flight: releasing one must leave the other's pins alone."""
    model, counter = make_model(pooling, layout, admit)
    batches = [batch(0), batch(1)]
    if admit:
        for keys, offsets in batches:  # first sight: rejected; admitted from now on
            no_grad_forward(model, keys, offsets, pooling)

    model.prefetch(*batches[0])
    one_batch = pins(counter)
    assert one_batch > 0
    model.prefetch(*batches[1])
    # The batches share no key, so their pins add up.
    assert pins(counter) == 2 * one_batch

    # Forwards pop the prefetched batches in order.
    modes = [not grad_first, grad_first]  # True: under no_grad
    outputs = []
    for (keys, offsets), no_grad in zip(batches, modes):
        with torch.set_grad_enabled(not no_grad):
            output = model(keys, offsets)
        assert output.requires_grad != no_grad
        check_values(output, keys, pooling)
        outputs.append(output)

    # Only the no-grad batch gave its pins back; the grad one waits for backward.
    assert pins(counter) == one_batch
    grad_output = outputs[1 if not grad_first else 0]
    grad_output.backward(torch.zeros_like(grad_output))
    assert torch.count_nonzero(counter).item() == 0


@pytest.mark.parametrize("grad_first", [False, True])
def test_no_grad_release_after_rehash_by_a_later_prefetch(grad_first):
    """The second batch's prefetch expands the tables (a rehash), which moves the
    first batch's keys to other slots. Whichever batch runs under no_grad gives
    its pins back at the slots its keys hold now: no counter goes below 0, the
    other batch keeps its pins, and every counter is 0 after its backward."""
    assert torch.cuda.is_available()
    names = ["t0", "t1"]
    options = [
        DynamicEmbTableOptions(
            dim=DIM,
            init_capacity=1024,
            max_capacity=16384,
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=torch.cuda.current_device(),
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            # The tables fit in HBM: the prefetch counts keys at their slots.
            local_hbm_for_values=1024**3,
            caching=False,
        )
        for _ in names
    ]
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=options,
        table_names=names,
        feature_table_map=[0, 1],
        pooling_mode=DynamicEmbPoolingMode.SUM,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.3,
    )

    # A rehash replaces the key map, and its ref counter with it: read both afresh.
    def capacity(table_id):
        return model.tables.key_index_map.capacity(table_id)

    def counter():
        return model.tables.key_index_map._ref_counter

    def make_batch(keys):
        # One key per sample, the same keys for both features.
        indices = torch.tensor(keys * 2, dtype=torch.int64, device="cuda")
        return indices, torch.arange(len(indices) + 1, dtype=torch.int64, device="cuda")

    # A table of 1024 slots holds 512 keys. The first batch's 256 keys fit; with the
    # second's, each table holds 640, so the second prefetch expands both tables.
    first_keys, second_keys = list(range(256)), list(range(128, 640))
    batches = [make_batch(first_keys), make_batch(second_keys)]
    counts = [len(names) * len(first_keys), len(names) * len(second_keys)]

    model.prefetch(*batches[0])
    capacities = [capacity(t) for t in range(len(names))]
    model.prefetch(*batches[1])
    assert all(
        capacity(t) > capacities[t] for t in range(len(names))
    ), "the second prefetch did not expand every table"
    assert pins(counter()) == sum(counts)

    outputs = []
    for no_grad in [not grad_first, grad_first]:
        with torch.set_grad_enabled(not no_grad):
            outputs.append(model(*batches[len(outputs)]))
        assert (counter() >= 0).all(), "a ref counter went below 0"
    # Only the no-grad batch gave its pins back.
    assert pins(counter()) == counts[0 if grad_first else 1]
    grad_output = outputs[0 if grad_first else 1]
    grad_output.sum().backward()
    assert (counter() == 0).all(), "a ref counter is not 0 after the backward"
