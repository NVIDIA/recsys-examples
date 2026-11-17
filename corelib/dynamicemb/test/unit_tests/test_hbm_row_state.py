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
from test_no_grad_slot_lifetime import batch, check_values, make_model, pins


@pytest.mark.parametrize("layout", ["hbm", "cache"])
@pytest.mark.parametrize("optimizer", [EmbOptimType.SGD, EmbOptimType.ADAM])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("wrap_kjt", [False, True])
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
    from dynamicemb.shard.resolved_input import ResolvedEmbeddingInput
    from torchrec import KeyedJaggedTensor

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


@pytest.mark.parametrize("layout", ["hbm", "cache"])
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
    assert len(reused) == 200


@pytest.mark.parametrize("layout", ["hbm", "cache"])
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
