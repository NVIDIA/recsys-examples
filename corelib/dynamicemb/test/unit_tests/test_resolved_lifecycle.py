# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receiver-state lifetime and output-window contracts, using real GPU tables."""

import gc
from dataclasses import replace

import pytest
import torch
from dynamicemb import DynamicEmbPoolingMode, EmbOptimType
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from test_no_grad_slot_lifetime import batch, check_values, make_model, pins


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


def test_repeated_cache_resolution_uses_actual_capacity():
    model, counter = make_model(DynamicEmbPoolingMode.NONE, "cache")
    capacity = model._cache._state.capacity
    per_table = capacity // 4
    keys = torch.cat([torch.arange(per_table), torch.arange(per_table) + 10000]).cuda()
    offsets = torch.tensor(
        [0, per_table, 2 * per_table], dtype=torch.int64, device="cuda"
    )
    states = [model.resolve_input(keys, offsets) for _ in range(3)]
    assert pins(counter) == keys.numel() * 3
    for state in states:
        state.release()
    torch.cuda.synchronize()
    assert torch.count_nonzero(counter).item() == 0
    state = model.resolve_input(keys, offsets)
    state.release()


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
