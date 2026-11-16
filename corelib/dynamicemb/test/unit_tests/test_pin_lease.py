# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receiver-state lifetime and output-window contracts, using real GPU tables."""

import gc

import pytest
import torch
from dynamicemb import DynamicEmbPoolingMode
from test_no_grad_slot_lifetime import batch, check_values, make_model, pins


def _prepare(model, keys, offsets):
    model.prefetch(keys, offsets)
    return model._prefetch_states.popleft()


def _consume(model, keys, offsets, state):
    model._prefetch_states.append(state)
    return model(keys, offsets)


@pytest.mark.parametrize("layout", ["hbm", "cache"])
@pytest.mark.parametrize("stage", ["input_only", "discard_graph"])
def test_abandoned_pinned_batch_releases_rows(layout, stage):
    model, counter = make_model(DynamicEmbPoolingMode.NONE, layout)
    keys, offsets = batch(0)
    state = _prepare(model, keys, offsets)
    assert pins(counter) == keys.numel()
    if stage == "discard_graph":
        output = _consume(model, keys, offsets, state)
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
    state = _prepare(model, keys, offsets)
    with model._table_gate():
        del state
        gc.collect()
        # The finalizer must not recursively take the non-reentrant table lock.
        assert pins(counter) == keys.numel()
    torch.cuda.synchronize()
    assert torch.count_nonzero(counter).item() == 0
    assert model._prefetch_outstanding_keys.item() == 0


@pytest.mark.parametrize("layout", ["hbm", "cache"])
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


@pytest.mark.parametrize("layout", ["hbm", "cache"])
def test_cross_stream_cancellation_keeps_release_metadata(layout):
    model, counter = make_model(DynamicEmbPoolingMode.NONE, layout)
    keys, offsets = batch(0)
    producer, canceller = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        state = _prepare(model, keys, offsets)
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
    states = [_prepare(model, keys, offsets) for _ in range(2)]
    before = model._prefetch_outstanding_keys.item()
    with pytest.raises(RuntimeError, match="exceed cache capacity"):
        _prepare(model, keys, offsets)
    assert model._prefetch_outstanding_keys.item() == before
    for state in states:
        state.release()
    torch.cuda.synchronize()
    assert torch.count_nonzero(counter).item() == 0
    assert model._prefetch_outstanding_keys.item() == 0
    state = _prepare(model, keys, offsets)
    state.release()
    assert model._prefetch_outstanding_keys.item() == 0
