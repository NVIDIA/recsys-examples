# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receiver-state lifetime and output-window contracts, using real GPU tables."""

import weakref
from dataclasses import replace

import pytest
import torch
from dynamicemb import DynamicEmbPoolingMode, EmbOptimType
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from test_no_grad_slot_lifetime import batch, check_values, make_model, pins


@pytest.mark.parametrize("layout", ["hbm", "cache", "host", "hybrid"])
@pytest.mark.parametrize("replace_values", [False, True])
def test_eval_state_survives_drop_on_producer_stream(layout, replace_values):
    model, _ = make_model(DynamicEmbPoolingMode.NONE, layout)
    keys, offsets = batch(0)
    model(keys, offsets).sum().backward()
    model.eval()
    source, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(source):
        warm_state = model.resolve_input(keys, offsets)
    source.synchronize()
    with torch.cuda.stream(consumer):
        warm_output = model(keys, offsets, prefetch_state=warm_state)
    consumer.synchronize()
    expected = warm_output.cpu()
    shape, dtype = warm_state.unique_values.shape, warm_state.unique_values.dtype
    del warm_state, warm_output

    # Warm both the exact lookup and allocator pool. Lazy CUDA loading or a new
    # cudaMalloc can synchronize the device and conceal missing registration.
    with torch.cuda.stream(source):
        warm_allocs = [
            torch.full(shape, 7777, device=keys.device, dtype=dtype) for _ in range(64)
        ]
    source.synchronize()
    del warm_allocs
    with torch.cuda.stream(source):
        state = model.resolve_input(keys, offsets)
    source.synchronize()
    if replace_values:
        state.record_stream(consumer)
        with torch.cuda.stream(source):
            state.unique_values = state.unique_values.clone()
        source.synchronize()
    # Keep index metadata alive to isolate value-buffer reuse and avoid unsafe
    # indices in a failing implementation. The state itself must be collectible.
    other_metadata = [
        value
        for name, value in vars(state).items()
        if isinstance(value, torch.Tensor) and name != "unique_values"
    ]
    state_ref, values_ref = weakref.ref(state), weakref.ref(state.unique_values)
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(1_000_000_000)
        actual = model(keys, offsets, prefetch_state=state)
        done = torch.cuda.Event()
        done.record()
    del state
    assert state_ref() is None and values_ref() is None
    pending = not done.query()
    with torch.cuda.stream(source):
        reuse = [
            torch.full(shape, 7777, device=keys.device, dtype=dtype) for _ in range(64)
        ]
    source.synchronize()
    overwritten_while_pending = not done.query()
    consumer.synchronize()
    if not pending or not overwritten_while_pending:
        pytest.skip("consumer finished before allocator reuse could be exercised")
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert len(other_metadata) > 0 and len(reuse) == 64


@pytest.mark.parametrize("layout", ["hbm", "cache", "host", "hybrid"])
def test_retained_backward_observes_intervening_update(layout):
    model, counter = make_model(DynamicEmbPoolingMode.NONE, layout)
    keys, offsets = batch(0)
    output = model(keys, offsets)
    output.sum().backward(retain_graph=True)
    if counter is not None:
        assert pins(counter) == keys.numel()
    model(keys, offsets).sum().backward()
    output.sum().backward()
    if counter is not None:
        assert pins(counter) == 0
    model.eval()
    actual = model(keys, offsets)
    expected = (keys % 100000).float()[:, None].expand_as(actual) - 0.75
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["hbm", "cache", "host", "hybrid"])
@pytest.mark.parametrize(
    "pooling", [DynamicEmbPoolingMode.NONE, DynamicEmbPoolingMode.SUM]
)
def test_resolved_batch_train_eval_transition(layout, pooling):
    model, counter = make_model(pooling, layout)
    keys, offsets = batch(0)
    state = model.resolve_input(keys, offsets)
    model.eval()
    output = model(keys, offsets, prefetch_state=state)
    assert not output.requires_grad
    check_values(output, keys, pooling)
    if counter is not None:
        assert pins(counter) == 0
    state = model.resolve_input(keys, offsets)
    model.train()
    output = model(keys, offsets, prefetch_state=state)
    assert output.requires_grad
    output.sum().backward()
    if counter is not None:
        assert pins(counter) == 0


@pytest.mark.parametrize("layout", ["hbm", "cache", "host", "hybrid"])
@pytest.mark.parametrize("reverse", [False, True])
def test_recomputed_resolved_batch_owns_separate_state(layout, reverse):
    model, counter = make_model(DynamicEmbPoolingMode.NONE, layout)
    keys, offsets = batch(0)
    state = model.resolve_input(keys, offsets)
    outputs = [model(keys, offsets, prefetch_state=state) for _ in range(2)]
    if counter is not None:
        assert pins(counter) == 2 * keys.numel()
    for output in reversed(outputs) if reverse else outputs:
        output.sum().backward()
    if counter is not None:
        assert pins(counter) == 0
    model.eval()
    actual = model(keys, offsets)
    expected = (keys % 100000).float()[:, None].expand_as(actual) - 0.5
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["hbm", "cache", "host", "hybrid"])
@pytest.mark.parametrize(
    "dtype,optimizer",
    [
        (torch.float32, EmbOptimType.SGD),
        (torch.float32, EmbOptimType.ADAM),
        (torch.float32, EmbOptimType.EXACT_ADAGRAD),
        (torch.float32, EmbOptimType.EXACT_ROWWISE_ADAGRAD),
        (torch.float16, EmbOptimType.EXACT_ROWWISE_ADAGRAD),
        (torch.bfloat16, EmbOptimType.EXACT_ROWWISE_ADAGRAD),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_prepared_batches_preserve_complete_optimizer_rows(
    layout, dtype, optimizer, reverse
):
    import struct

    from dynamicemb.key_value_table import load_from_flat
    from dynamicemb.types import CopyMode

    # Low 16 bits are a NaN in both half encodings. They are ordinary bits of
    # a finite FP32 accumulator, and copying optimizer rows must preserve them.
    initial = struct.unpack("<f", struct.pack("<I", 0x40007FFF))[0]

    def create():
        template, _ = make_model(DynamicEmbPoolingMode.NONE, layout)
        return BatchedDynamicEmbeddingTablesV2(
            table_options=[
                replace(option, embedding_dtype=dtype)
                for option in template._dynamicemb_options
            ],
            table_names=["t0", "t1"],
            feature_table_map=[0, 1],
            pooling_mode=DynamicEmbPoolingMode.NONE,
            output_dtype=torch.float32,
            optimizer=optimizer,
            initial_accumulator_value=initial
            if optimizer == EmbOptimType.EXACT_ROWWISE_ADAGRAD
            else 0.0,
            learning_rate=0.25,
            stochastic_rounding=False,
        )

    actual, reference = create(), create()
    keys, offsets = batch(0)
    states = [actual.resolve_input(keys, offsets) for _ in range(2)]
    outputs = [actual(keys, offsets, prefetch_state=state) for state in states]
    order = [1, 0] if reverse else [0, 1]
    for index in order:
        outputs[index].backward(torch.full_like(outputs[index], index + 1))
        expected = reference(keys, offsets)
        expected.backward(torch.full_like(expected, index + 1))

    def values(model):
        state = model.resolve_input(keys, offsets)
        if state.unique_values is not None:
            result = state.unique_values.clone()
        else:
            owner = model._cache or model._storage
            result = load_from_flat(
                owner._state,
                state.value_row_indices,
                state.unique_table_ids,
                copy_mode=CopyMode.VALUE,
            )
        state.release()
        return result

    actual_rows, reference_rows = values(actual), values(reference)
    if optimizer == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
        torch.testing.assert_close(
            actual_rows[:, :8], reference_rows[:, :8], rtol=0, atol=0
        )

        def accumulator(rows):
            return (
                rows[:, 8:]
                .contiguous()
                .view(torch.uint8)[:, :4]
                .contiguous()
                .view(torch.float32)
            )

        torch.testing.assert_close(
            accumulator(actual_rows), accumulator(reference_rows), rtol=0, atol=0
        )
        torch.testing.assert_close(
            accumulator(actual_rows),
            torch.full_like(accumulator(actual_rows), initial + 5),
            rtol=0,
            atol=0,
        )
    else:
        torch.testing.assert_close(actual_rows, reference_rows, rtol=1e-5, atol=1e-6)
    assert not actual._active_rows.states


def test_pre_erase_host_snapshot_cannot_restore_deleted_rows():
    from dynamicemb import (
        DynamicEmbInitializerArgs,
        DynamicEmbInitializerMode,
        DynamicEmbScoreStrategy,
        DynamicEmbTableOptions,
    )
    from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2

    device = torch.device("cuda", torch.cuda.current_device())
    model = BatchedDynamicEmbeddingTablesV2(
        table_names=["host"],
        table_options=[
            DynamicEmbTableOptions(
                dim=8,
                max_capacity=4096,
                index_type=torch.int64,
                embedding_dtype=torch.float32,
                device_id=device.index,
                score_strategy=DynamicEmbScoreStrategy.LFU,
                local_hbm_for_values=0,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.CONSTANT, value=2.0
                ),
            )
        ],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.25,
    )
    keys = torch.arange(32, device=device)
    offsets = torch.tensor([0, keys.numel()], device=device)
    model(keys, offsets).sum().backward()
    old = model.resolve_input(keys, offsets)
    torch.testing.assert_close(
        old.unique_values, torch.full_like(old.unique_values, 1.75)
    )
    handle = model.erase_by_score_async(DynamicEmbScoreStrategy.LFU, 1 << 20)
    assert handle.result() == (keys.numel(), 0)
    assert not old.released
    with torch.no_grad():
        fresh = model.resolve_input(keys, offsets)
        actual = model(keys, offsets, prefetch_state=fresh)
    torch.testing.assert_close(actual, torch.full_like(actual, 2.0))
    old.release()
