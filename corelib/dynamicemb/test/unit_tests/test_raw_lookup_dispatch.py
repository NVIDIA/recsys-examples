# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw/FIFO compatibility and explicit resolved lookup share the same optimizer."""
import pytest
import torch
from dynamicemb import DynamicEmbPoolingMode
from test_hbm_receiver import batch as sharded_batch  # noqa: F401
from test_no_grad_slot_lifetime import batch, check_values, make_model, pins


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    "layout,pooling",
    [
        ("hbm", DynamicEmbPoolingMode.NONE),
        ("cache", DynamicEmbPoolingMode.SUM),
        ("host", DynamicEmbPoolingMode.NONE),
        ("hybrid", DynamicEmbPoolingMode.SUM),
    ],
)
def test_only_explicit_resolved_state_uses_torchrec_lookup(prepared, layout, pooling):
    model, counter = make_model(pooling, layout)
    keys, offsets = batch(0)
    native = model._native_lookup
    calls = []

    def lookup(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    model._native_lookup = lookup
    state = model.resolve_input(keys, offsets) if prepared else None
    output = model(keys, offsets, prefetch_state=state)
    check_values(output, keys, pooling)
    assert len(calls) == int(prepared)
    output.sum().backward()
    if counter is not None:
        assert pins(counter) == 0


@pytest.mark.parametrize(
    "layout,mapped", [("hbm", True), ("host", False), ("hybrid", True)]
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
