# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receiver-state lifetime and output-window contracts, using real GPU tables."""

import gc
import weakref
from dataclasses import replace
from unittest.mock import patch

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


@pytest.mark.parametrize("layout", ["hbm", "cache"])
@pytest.mark.parametrize("stage", ["input_only", "discard_graph"])
def test_abandoned_pinned_batch_releases_rows(layout, stage):
    model, counter = make_model(DynamicEmbPoolingMode.NONE, layout)
    keys, offsets = batch(0)
    state = model.resolve_input(keys, offsets)
    assert pins(counter) == keys.numel()
    if stage == "discard_graph":
        output = model(keys, offsets, prefetch_state=state)
        detached = output.detach()
        del output
    del state
    gc.collect()
    torch.cuda.synchronize()
    assert pins(counter) == 0
    assert model._prefetch_outstanding_keys.item() == 0
    if stage == "discard_graph":
        check_values(detached, keys, DynamicEmbPoolingMode.NONE)


@pytest.mark.parametrize("layout", ["hbm", "cache"])
def test_cancellation_inside_table_gate_defers_cleanup(layout):
    model, counter = make_model(DynamicEmbPoolingMode.NONE, layout)
    keys, offsets = batch(0)
    state = model.resolve_input(keys, offsets)
    with model._table_gate():
        del state
        gc.collect()
        # The finalizer must not recursively take the non-reentrant table lock.
        assert pins(counter) == keys.numel()
    torch.cuda.synchronize()
    assert torch.count_nonzero(counter).item() == 0
    assert model._prefetch_outstanding_keys.item() == 0


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


@pytest.mark.parametrize("layout", ["hbm", "cache"])
def test_discard_retained_graph_releases_rows(layout):
    model, counter = make_model(DynamicEmbPoolingMode.NONE, layout)
    keys, offsets = batch(0)
    output = model(keys, offsets)
    output.sum().backward(retain_graph=True)
    assert pins(counter) == keys.numel()
    del output
    gc.collect()
    torch.cuda.synchronize()
    assert pins(counter) == 0


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
@pytest.mark.parametrize("mapped", [False, True])
def test_output_allocator_preserves_values_and_gradient_rows(layout, mapped):
    model, counter = make_model(DynamicEmbPoolingMode.NONE, layout)
    keys, offsets = batch(0)
    rows, dim = keys.numel(), model.max_D
    window = torch.empty((rows, dim), device=keys.device)
    row_map = (
        torch.arange(rows - 1, -1, -1, device=keys.device, dtype=torch.int32)
        if mapped
        else None
    )
    calls = []

    def allocate(n, d, dtype):
        calls.append((n, d, dtype))
        assert (n, d, dtype) == (rows, dim, window.dtype)
        return window, row_map

    model.output_alloc = allocate
    output = model(keys, offsets)
    assert output.data_ptr() == window.data_ptr()
    assert len(calls) == 1
    in_order = output if row_map is None else output.index_select(0, row_map.long())
    check_values(in_order, keys, DynamicEmbPoolingMode.NONE)
    gradients = (
        torch.arange(1, rows + 1, device=keys.device).float()[:, None].expand_as(output)
    )
    (output * gradients).sum().backward()
    if counter is not None:
        assert pins(counter) == 0
    model.output_alloc = None
    model.eval()
    actual = model(keys, offsets)
    key_gradients = (
        gradients if row_map is None else gradients.index_select(0, row_map.long())
    )
    expected = (keys % 100000).float()[:, None].expand_as(actual) - 0.25 * key_gradients
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    model.output_alloc = allocate
    evaluated = model(keys, offsets)
    assert evaluated.data_ptr() == window.data_ptr()
    in_order = (
        evaluated if row_map is None else evaluated.index_select(0, row_map.long())
    )
    torch.testing.assert_close(in_order, expected, rtol=0, atol=0)


def test_shared_output_window_accepts_multiple_lookup_groups():
    from dynamicemb.shard.hier_output_dist import _LookupOutput

    models = [make_model(DynamicEmbPoolingMode.NONE, "hbm")[0] for _ in range(2)]
    keys, offsets = batch(0)
    count = keys.numel()
    window = torch.empty((2 * count, models[0].max_D), device=keys.device)
    outputs = []
    for i, model in enumerate(models):
        model.output_alloc = lambda n, d, dtype, i=i: (
            window.narrow(0, i * count, n),
            None,
        )
        outputs.append(model(keys, offsets))
    joined = _LookupOutput.apply(window, *outputs)
    joined.sum().backward()
    for model in models:
        model.output_alloc = None
        model.eval()
        actual = model(keys, offsets)
        expected = (keys % 100000).float()[:, None].expand_as(actual) - 0.25
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["hbm", "cache"])
def test_cross_stream_cancellation_keeps_release_metadata(layout):
    model, counter = make_model(DynamicEmbPoolingMode.NONE, layout)
    keys, offsets = batch(0)
    producer, canceller = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        state = model.resolve_input(keys, offsets)
        ready = torch.cuda.Event()
        ready.record()
    ready.synchronize()
    assert pins(counter) == keys.numel()
    with torch.cuda.stream(canceller):
        torch.cuda._sleep(100_000_000)
        del state
        gc.collect()
    with torch.cuda.stream(producer):
        reused = [
            torch.zeros((keys.numel(),), dtype=torch.int64, device=keys.device)
            for _ in range(200)
        ]
    torch.cuda.synchronize()
    assert torch.count_nonzero(counter).item() == 0
    assert model._prefetch_outstanding_keys.item() == 0
    assert len(reused) == 200


@pytest.mark.parametrize("layout", ["hbm", "cache"])
@pytest.mark.parametrize("optimizer", [EmbOptimType.SGD, EmbOptimType.ADAM])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_single_persistent_matrix_uses_physical_rows(layout, optimizer, dtype):
    template, _ = make_model(DynamicEmbPoolingMode.NONE, layout)
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=[replace(template._dynamicemb_options[0], embedding_dtype=dtype)],
        table_names=["t0"],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=optimizer,
        learning_rate=0.25,
        stochastic_rounding=False,
        output_dtype=torch.float32,
    )
    del template
    keys = torch.arange(1, 33, dtype=torch.int64, device="cuda").repeat(2)
    offsets = torch.tensor([0, keys.numel()], dtype=torch.int64, device="cuda")
    state = model.resolve_input(keys, offsets)
    from dynamicemb.shard.resolved_input import ResolvedEmbeddingInput
    from torchrec import KeyedJaggedTensor

    features = ResolvedEmbeddingInput.from_rows(
        KeyedJaggedTensor(keys=["f0"], values=keys, offsets=offsets), state
    )
    assert features.values().data_ptr() == state.occurrence_rows.data_ptr()
    with patch(
        "dynamicemb.batched_dynamicemb_function.load_from_flat",
        side_effect=AssertionError("unexpected compact workspace"),
    ):
        output = model(features.values(), offsets, prefetch_state=state)
    expected = (keys % 100000).float()[:, None].expand_as(output)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.sum().backward()
    model.eval()
    # A repeated key has two occurrences; SGD sums them, Adam normalizes them.
    delta = 0.5 if optimizer == EmbOptimType.SGD else 0.25
    torch.testing.assert_close(model(keys, offsets), expected - delta)


@pytest.mark.parametrize("optimizer", [EmbOptimType.SGD, EmbOptimType.ADAM])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_all_hit_sequence_projects_rows_with_unique_pins(optimizer, dtype):
    from dynamicemb.shard.resolved_input import ResolvedEmbeddingInput
    from dynamicemb.types import CopyMode
    from torchrec import KeyedJaggedTensor

    template, _ = make_model(DynamicEmbPoolingMode.NONE, "hbm")

    def create():
        return BatchedDynamicEmbeddingTablesV2(
            table_options=[
                replace(template._dynamicemb_options[0], embedding_dtype=dtype)
            ],
            table_names=["table"],
            feature_table_map=[0],
            pooling_mode=DynamicEmbPoolingMode.NONE,
            output_dtype=torch.float32,
            optimizer=optimizer,
            learning_rate=0.25,
        )

    models = [create(), create()]
    # The existing compact-workspace path is the control. Its native lookup is
    # unchanged, but it does not opt into persistent-row/fused pin preparation.
    models[1]._native_lookup = models[1]._native_lookup.__call__
    keys = torch.tensor([2, 2, 9, 1, 9], device="cuda")
    offsets = torch.tensor([0, 5], device="cuda")
    states = []
    for index, model in enumerate(models):
        cold = model.resolve_input(keys, offsets)
        assert (
            cold.occurrence_rows is None
        ), "misses must follow the original insertion path"
        output = model(keys, offsets, prefetch_state=cold)
        output.backward(torch.zeros_like(output))
        raw = model.resolve_input(keys, offsets)
        assert raw.occurrence_rows is None
        raw.release()
        first = model.resolve_input(keys, offsets, resolve_occurrence_rows=index == 0)
        second = model.resolve_input(keys, offsets, resolve_occurrence_rows=index == 0)
        assert (first.occurrence_rows is not None) == (index == 0)
        if index == 0:
            cached_rows = first.occurrence_rows
            torch.testing.assert_close(
                first.occurrence_rows, first.value_row_indices[first.reverse_indices]
            )
            features = KeyedJaggedTensor(keys=["f"], values=keys, offsets=offsets)
            resolved = ResolvedEmbeddingInput.from_rows(features, first)
            assert resolved.values() is cached_rows
        counter = model._storage._state.key_index_map._ref_counter
        assert pins(counter) == 6  # two batches, three unique keys, five occurrences
        states.append((first, second, counter))
    first_outputs = [
        model(keys, offsets, prefetch_state=state[0])
        for model, state in zip(models, states)
    ]
    torch.testing.assert_close(*first_outputs)
    for output, (_, _, counter) in zip(first_outputs, states):
        output.sum().backward(retain_graph=True)
        assert pins(counter) == 6
    second_outputs = [
        model(keys, offsets, prefetch_state=state[1])
        for model, state in zip(models, states)
    ]
    torch.testing.assert_close(*second_outputs)
    for first, second, (_, _, counter) in zip(first_outputs, second_outputs, states):
        second.sum().backward()
        assert pins(counter) == 3
        first.sum().backward()
        assert pins(counter) == 0
    unique_keys = keys.unique()
    values = [
        model._storage.find(
            unique_keys, torch.zeros_like(unique_keys), copy_mode=CopyMode.VALUE
        )[-1]
        for model in models
    ]
    # Compare the complete persisted rows, including Adam moments.
    torch.testing.assert_close(*values, rtol=0, atol=0)


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


@pytest.mark.parametrize("layout", ["hbm", "cache"])
def test_retained_backward_cancelled_on_another_stream_waits_for_update(layout):
    model, counter = make_model(DynamicEmbPoolingMode.NONE, layout)
    keys, offsets = batch(0)
    producer, canceller = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        output = model(keys, offsets)
        ready = torch.cuda.Event()
        ready.record()
    ready.synchronize()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(100_000_000)
        output.sum().backward(retain_graph=True)
    with torch.cuda.stream(canceller):
        del output
        gc.collect()
        model.eval()
        observed = model(keys, offsets)
    torch.cuda.synchronize()
    assert torch.count_nonzero(counter).item() == 0
    expected = (keys % 100000).float()[:, None].expand_as(observed) - 0.25
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["hbm", "cache"])
def test_final_backward_uses_forward_device_when_another_device_is_current(layout):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two visible CUDA devices")
    device = torch.cuda.current_device()
    model, counter = make_model(DynamicEmbPoolingMode.NONE, layout)
    keys, offsets = batch(0)
    output = model(keys, offsets)
    other = (device + 1) % torch.cuda.device_count()
    with torch.cuda.device(other):
        output.sum().backward()
        assert torch.cuda.current_device() == other
    torch.cuda.synchronize(device)
    assert pins(counter) == 0
    model.eval()
    expected = (keys % 100000).float()[:, None].expand_as(output) - 0.25
    torch.testing.assert_close(model(keys, offsets), expected, rtol=0, atol=0)


def test_rejected_cache_reservation_can_recover():
    model, counter = make_model(DynamicEmbPoolingMode.NONE, "cache")
    capacity = model._cache._state.capacity
    per_table = capacity // 4
    keys = torch.cat([torch.arange(per_table), torch.arange(per_table) + 10000]).cuda()
    offsets = torch.tensor(
        [0, per_table, 2 * per_table], dtype=torch.int64, device="cuda"
    )
    states = [model.resolve_input(keys, offsets) for _ in range(2)]
    before = model._prefetch_outstanding_keys.item()
    with pytest.raises(RuntimeError, match="exceed cache capacity"):
        model.resolve_input(keys, offsets)
    assert model._prefetch_outstanding_keys.item() == before
    for state in states:
        state.release()
    torch.cuda.synchronize()
    assert torch.count_nonzero(counter).item() == 0
    assert model._prefetch_outstanding_keys.item() == 0
    state = model.resolve_input(keys, offsets)
    state.release()
    assert model._prefetch_outstanding_keys.item() == 0
