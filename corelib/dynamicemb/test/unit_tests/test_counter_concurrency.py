# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from dynamicemb import DynamicEmbPoolingMode, EmbOptimType
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb_extensions import table_update_counter_with_layout
from test_no_grad_slot_lifetime import make_model, pins


def test_counter_updates_on_different_streams_preserve_each_reference():
    # Use a larger grid rather than thousands of queued launches: the host
    # launch queue must not fill and block before both streams reach the gate.
    size, steps = 65536, 128
    counter = torch.full((size,), steps, device="cuda", dtype=torch.int32)
    slots = torch.arange(size, device="cuda")
    offsets = torch.tensor([0, size // 128], device="cuda")

    def update(delta):
        return table_update_counter_with_layout(
            counter, slots, delta, offsets, 128, size, 1
        )

    source, consumer, control = [torch.cuda.Stream() for _ in range(3)]
    torch.cuda.synchronize()
    # Warm the exact kernels and each stream's output allocation before adding
    # the common start gate. Lazy loading/cudaMalloc can otherwise serialize it.
    for stream in [source, consumer]:
        with torch.cuda.stream(stream):
            update(1)
    with torch.cuda.stream(control):
        torch.cuda._sleep(1)
    torch.cuda.synchronize()
    counter.fill_(steps)
    torch.cuda.synchronize()
    start = torch.cuda.Event()
    with torch.cuda.stream(control):
        torch.cuda._sleep(1_000_000_000)
        start.record()
    source.wait_event(start)
    consumer.wait_event(start)
    with torch.cuda.stream(source):
        for _ in range(steps):
            update(1)
    with torch.cuda.stream(consumer):
        for _ in range(steps):
            update(-1)
    queued_before_start = not start.query()
    source.synchronize()
    consumer.synchronize()
    if not queued_before_start:
        pytest.skip("could not queue both streams before the common start gate")
    # The result is independent of scheduling/interleaving. Each call itself
    # has unique slots, so uniqueness within a batch cannot prevent this race.
    torch.testing.assert_close(counter, torch.full_like(counter, steps), rtol=0, atol=0)


def test_cache_backward_and_prefetch_keep_live_pins_across_streams():
    template, _ = make_model(DynamicEmbPoolingMode.NONE, "cache")
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=[
            replace(
                template._dynamicemb_options[0],
                init_capacity=16384,
                max_capacity=16384,
                local_hbm_for_values=8192 * 8 * 4,
            )
        ],
        table_names=["table"],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        output_dtype=torch.float32,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.25,
    )
    count = 4096
    keys = torch.arange(count, device="cuda")
    offsets = torch.tensor([0, count], device="cuda")
    gradients = torch.zeros((count, 8), device="cuda")
    consumer, producer, control = (
        torch.cuda.current_stream(),
        torch.cuda.Stream(),
        torch.cuda.Stream(),
    )
    cache = model._cache
    counter = cache._state.key_index_map._ref_counter
    model(keys, offsets).backward(gradients)
    with torch.cuda.stream(control):
        torch.cuda._sleep(1)
    torch.cuda.synchronize()
    gate = [None]
    increment, decrement = type(cache).increment_counter, type(cache).decrement_counter

    def wait_then_increment(owner, *args, **kwargs):
        if owner is cache and gate[0] is not None:
            torch.cuda.current_stream().wait_event(gate[0])
        return increment(owner, *args, **kwargs)

    def wait_then_decrement(owner, *args, **kwargs):
        if owner is cache and gate[0] is not None:
            torch.cuda.current_stream().wait_event(gate[0])
        return decrement(owner, *args, **kwargs)

    # Only align the real cache kernels; no counter/value result is mocked.
    with patch.object(
        type(cache), "increment_counter", new=wait_then_increment
    ), patch.object(type(cache), "decrement_counter", new=wait_then_decrement):
        for _ in range(3):
            output = model(keys, offsets)
            torch.cuda.synchronize()
            assert pins(counter) == count
            with torch.cuda.stream(control):
                torch.cuda._sleep(1_000_000_000)
                gate[0] = torch.cuda.Event()
                gate[0].record()
            output.backward(gradients)
            with torch.cuda.stream(producer):
                model.prefetch(keys, offsets, forward_stream=consumer)
            producer.synchronize()
            consumer.synchronize()
            state = model._prefetch_states[0]
            # The previous backward returned its pins; the queued batch owns
            # exactly one per key, despite CPU-ordered calls using two streams.
            torch.testing.assert_close(
                counter[state.slot_indices],
                torch.ones(count, device=keys.device, dtype=counter.dtype),
                rtol=0,
                atol=0,
            )
            assert pins(counter) == count
            gate[0] = None
            model(keys, offsets).backward(gradients)
            torch.cuda.synchronize()
            assert pins(counter) == 0
