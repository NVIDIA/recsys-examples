# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receiver-state lifetime and output-window contracts, using real GPU tables."""

import gc
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from dynamicemb import DynamicEmbPoolingMode, EmbOptimType
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.shard.resolved_input import ResolvedEmbeddingInput
from test_no_grad_slot_lifetime import batch, check_values, make_model, pins
from torchrec import KeyedJaggedTensor


@pytest.mark.parametrize("layout", ["hbm", "cache"])
@pytest.mark.parametrize(
    "optimizer,dtype,wrap_kjt",
    [
        (EmbOptimType.SGD, torch.float32, False),
        (EmbOptimType.ADAM, torch.float16, True),
        (EmbOptimType.ADAM, torch.bfloat16, True),
    ],
    ids=["contiguous-dense", "strided-dense", "ec-fallback"],
)
def test_single_persistent_matrix_uses_physical_rows(
    layout, optimizer, dtype, wrap_kjt
):
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
    features = KeyedJaggedTensor(keys=["f0"], values=keys, offsets=offsets)
    if wrap_kjt:
        features = ResolvedEmbeddingInput.from_rows(features, state)
        assert features.values().data_ptr() == state.occurrence_rows.data_ptr()
    else:
        assert state.occurrence_rows is None
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


@pytest.mark.parametrize(
    "dtype,free_slots",
    [(torch.float32, 1), (torch.bfloat16, 0)],
    ids=["dense-partial-refusal", "ec-all-new-keys-refused"],
)
def test_hbm_refused_insert_uses_initialized_temporary_rows(dtype, free_slots):
    template, _ = make_model(DynamicEmbPoolingMode.NONE, "hbm")
    options = replace(
        template._dynamicemb_options[0],
        max_capacity=128,
        init_capacity=128,
        embedding_dtype=dtype,
    )
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=[options],
        table_names=["t0"],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=EmbOptimType.SGD,
        learning_rate=1.0,
        output_dtype=torch.float32,
    )
    counter = model._storage._state.key_index_map._ref_counter
    assert model._storage._state.capacity == 128

    def lookup(keys):
        offsets = keys.new_tensor([0, keys.numel()])
        state = model.resolve_input(keys, offsets)
        features = ResolvedEmbeddingInput.from_rows(
            KeyedJaggedTensor(keys=["f0"], values=keys, offsets=offsets), state
        )
        return model(features.values(), offsets, prefetch_state=state), state

    owner_keys = torch.arange(128 - free_slots, device="cuda")
    owner, _ = lookup(owner_keys)
    keys = owner_keys.new_tensor([0, 128, 129])
    result, state = lookup(keys)
    expected = keys.float()[:, None].expand_as(result)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    refused = state.value_row_indices[state.reverse_indices] < 0
    assert not refused[0].item() and refused.sum().item() == 2 - free_slots
    assert pins(counter) == 129
    result.sum().backward()
    assert pins(counter) == owner_keys.numel()

    owner.sum().backward()
    assert pins(counter) == 0
    updated, _ = lookup(keys[:1])
    torch.testing.assert_close(updated, torch.full_like(updated, -2))
    updated.sum().backward()
    retried, retry_state = lookup(keys[refused])
    torch.testing.assert_close(retried, expected[refused], rtol=0, atol=0)
    assert (retry_state.value_row_indices >= 0).all().item()
    retried.sum().backward()
    assert pins(counter) == 0


@pytest.mark.parametrize(
    "layout,mapped",
    [
        ("hbm", True),
        ("cache", False),
    ],
)
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
    output = model(keys, offsets, prefetch_state=model.resolve_input(keys, offsets))
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
        outputs.append(
            model(keys, offsets, prefetch_state=model.resolve_input(keys, offsets))
        )
    joined = _LookupOutput.apply(window, *outputs)
    joined.sum().backward()
    for model in models:
        model.output_alloc = None
        model.eval()
        actual = model(keys, offsets)
        expected = (keys % 100000).float()[:, None].expand_as(actual) - 0.25
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "layout",
    [
        "hbm",
    ],
)
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
    assert len(reused) == 200


@pytest.mark.parametrize(
    "layout",
    [
        "cache",
    ],
)
def test_retained_backward_cancelled_on_another_stream_waits_for_update(layout):
    model, counter = make_model(DynamicEmbPoolingMode.NONE, layout)
    keys, offsets = batch(0)
    producer, canceller = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        output = model(keys, offsets, prefetch_state=model.resolve_input(keys, offsets))
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


@pytest.mark.parametrize("stage", ["input_only", "discard_graph"])
def test_abandoned_pinned_batch_releases_rows(stage):
    model, counter = make_model(DynamicEmbPoolingMode.NONE, "hbm")
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
    if stage == "discard_graph":
        check_values(detached, keys, DynamicEmbPoolingMode.NONE)


def test_all_hit_sequence_projects_rows_with_unique_pins():
    from dynamicemb.shard.resolved_input import ResolvedEmbeddingInput
    from dynamicemb.types import CopyMode
    from torchrec import KeyedJaggedTensor

    template, _ = make_model(DynamicEmbPoolingMode.NONE, "hbm")

    def create():
        return BatchedDynamicEmbeddingTablesV2(
            table_options=[
                replace(template._dynamicemb_options[0], embedding_dtype=torch.float32)
            ],
            table_names=["table"],
            feature_table_map=[0],
            pooling_mode=DynamicEmbPoolingMode.NONE,
            output_dtype=torch.float32,
            optimizer=EmbOptimType.ADAM,
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
        first = model.resolve_input(keys, offsets)
        second = model.resolve_input(keys, offsets)
        if index == 0:
            features = KeyedJaggedTensor(keys=["f"], values=keys, offsets=offsets)
            resolved = ResolvedEmbeddingInput.from_rows(features, first)
            torch.testing.assert_close(
                resolved.values(), first.value_row_indices[first.reverse_indices]
            )
            assert resolved.values() is first.occurrence_rows
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
