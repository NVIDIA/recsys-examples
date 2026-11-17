# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pins taken on one stream and released on another add up.

Under TorchRec's prefetch pipeline the prefetch of batch i+1 pins its keys on the
prefetch stream while the backward of batch i releases its own on the default
stream, and nothing orders the two: a key in both batches gets an increment and
a decrement at the same time. Here two streams pin and release the same slots
over and over with no ordering between them; every slot must end where it
started.

A call's kernel takes a small part of the GPU (one block per 256 slots), and
both streams wait before their first one, so that every launch is queued before
any runs: the GPU then runs kernels of the two streams side by side, on the same
slots.
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

DIM = 8
CACHE_ROWS = 1 << 16
ROUNDS = 1000
# About 0.2 s on the GPU: longer than queueing the rounds takes.
STALL_CYCLES = 400_000_000


def make_cache():
    assert torch.cuda.is_available()
    options = [
        DynamicEmbTableOptions(
            dim=DIM,
            max_capacity=4 * CACHE_ROWS,
            init_capacity=4 * CACHE_ROWS,
            bucket_capacity=128,
            device_id=torch.cuda.current_device(),
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            local_hbm_for_values=CACHE_ROWS * DIM * 4,
            caching=True,
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
    assert model._cache is not None
    assert model._cache._state.key_index_map.capacity_ == CACHE_ROWS
    return model._cache


@pytest.mark.parametrize("base", [0, 1])
def test_pins_and_releases_on_two_streams_add_up(base):
    cache = make_cache()
    counter = cache._state.key_index_map._ref_counter
    slots = torch.arange(CACHE_ROWS, dtype=torch.int64, device="cuda")
    table_ids = torch.zeros_like(slots)
    # Load the kernel first: loading a module lazily waits for the device, which
    # would run the stalled streams dry while the rounds are being queued.
    cache.increment_counter(slots, table_ids)
    cache.decrement_counter(slots, table_ids)
    for _ in range(base):
        cache.increment_counter(slots, table_ids)
    torch.cuda.synchronize()
    start = counter.clone()

    pin_stream, release_stream = torch.cuda.Stream(), torch.cuda.Stream()
    for stream in (pin_stream, release_stream):
        with torch.cuda.stream(stream):
            torch.cuda._sleep(STALL_CYCLES)
    for _ in range(ROUNDS):
        with torch.cuda.stream(pin_stream):
            cache.increment_counter(slots, table_ids)
        with torch.cuda.stream(release_stream):
            cache.decrement_counter(slots, table_ids)
    torch.cuda.synchronize()

    lost = torch.count_nonzero(counter != start).item()
    assert lost == 0, f"{lost} slots lost a pin or a release"
