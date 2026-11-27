# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TorchRec dense kernels and EC/EBC fallback share the same row contract."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from dynamicemb import DynamicEmbPoolingMode
from dynamicemb.dense_lookup import DenseLookup
from dynamicemb.state.extendable_buffer import DeviceExtendableBuffer
from dynamicemb.torchrec_lookup import TorchRecLookup


@pytest.mark.parametrize(
    "dtype,padding",
    [(torch.float32, 0), (torch.float16, 16), (torch.bfloat16, 1)],
    ids=["contiguous", "whole-row-padding", "fallback-padding"],
)
def test_persistent_weight_view_tracks_growth_replacement_and_owner(dtype, padding):
    dim, rows = 8, 4
    device = torch.device("cuda", torch.cuda.current_device())
    buffer = DeviceExtendableBuffer((rows, dim + padding), dtype, device)
    lookup = TorchRecLookup([dim], [0], DynamicEmbPoolingMode.NONE, device)
    indices = torch.tensor([1, 3, 1], device=device)
    offsets = torch.tensor([0, 3], device=device)
    batch = SimpleNamespace(reverse_indices=indices)
    with torch.no_grad(), patch.object(buffer, "tensor", wraps=buffer.tensor) as read:
        first = lookup.persistent_weights(buffer)
        first.fill_(3)
        first[:, 0] = 9
        torch.testing.assert_close(
            lookup(first, batch, offsets, torch.float32), first[indices].float()
        )
        assert lookup.persistent_weights(buffer) is first
        assert read.call_count == 1
        # CUDA optimizer writes must be visible through the cached view.
        first.add_(2)
        torch.testing.assert_close(
            lookup(lookup.persistent_weights(buffer), batch, offsets, torch.float32),
            first[indices].float(),
        )
        assert read.call_count == 1
        buffer.extend((rows, dim + padding))
        grown = lookup.persistent_weights(buffer)
        assert read.call_count == 2 and grown.shape == (2 * rows, dim)
        assert grown.data_ptr() == first.data_ptr()
        grown[rows:].fill_(17)
        batch.reverse_indices = torch.tensor([6, 1, 7], device=device)
        torch.testing.assert_close(
            lookup(grown, batch, offsets, torch.float32),
            grown[batch.reverse_indices].float(),
        )
    del read
    owner_ref = weakref.ref(buffer)
    del buffer
    gc.collect()
    assert owner_ref() is None
    # A same-shape replacement cannot reuse the former owner's borrowed view.
    replacement = DeviceExtendableBuffer((2 * rows, dim + padding), dtype, device)
    with torch.no_grad():
        current = lookup.persistent_weights(replacement)
        assert current is not grown
        current.fill_(23)
        torch.testing.assert_close(
            lookup(current, batch, offsets, torch.float32),
            torch.full((3, dim), 23.0, device=device),
        )


# One case per dispatch boundary, not a cross product of independent knobs.
F32, F16, BF16 = torch.float32, torch.float16, torch.bfloat16
NONE, SUM, MEAN = (
    DynamicEmbPoolingMode.NONE,
    DynamicEmbPoolingMode.SUM,
    DynamicEmbPoolingMode.MEAN,
)


@pytest.mark.parametrize(
    "dtype,output_dtype,dim,padding,mode,weighted",
    [
        pytest.param(F32, F32, 4, 0, NONE, False, id="dense-sequence"),
        pytest.param(F16, BF16, 8, 16, NONE, False, id="dense-strided"),
        pytest.param(F16, F16, 4096, 0, NONE, False, id="dense-wide-sequence"),
        pytest.param(F32, BF16, 2048, 0, SUM, False, id="dense-pooled-limit"),
        pytest.param(F16, F32, 8, 16, SUM, False, id="dense-sum-half"),
        pytest.param(F32, F16, 12, 0, MEAN, False, id="dense-mean"),
        pytest.param(F16, F16, 8, 0, SUM, True, id="dense-weighted"),
        pytest.param(F32, BF16, 7, 0, NONE, False, id="ec-odd-width"),
        pytest.param(BF16, F32, 8, 0, NONE, False, id="ec-bfloat16"),
        pytest.param(F16, F32, 8, 1, NONE, False, id="ec-fractional-stride"),
        pytest.param(F32, F32, 2052, 0, SUM, False, id="ebc-over-dense-limit"),
        pytest.param(F16, BF16, 2052, 0, SUM, True, id="ebc-wide-weighted"),
        pytest.param(BF16, F16, 8, 0, MEAN, False, id="ebc-mean-bfloat16"),
        pytest.param(BF16, F32, 8, 0, SUM, True, id="ebc-weighted-bfloat16"),
        pytest.param(F16, BF16, 8, 1, SUM, False, id="ebc-fractional-stride"),
    ],
)
def test_dense_lookup_and_fallback_match_reference(
    dtype, output_dtype, dim, padding, mode, weighted
):
    device = torch.device("cuda", torch.cuda.current_device())
    full = (
        (torch.arange(9 * (dim + padding), device=device).float() / 64)
        .reshape(9, dim + padding)
        .to(dtype)
    )
    # Nonzero storage offset and distinct optimizer/padding columns.
    weights = full[1:7, :dim]
    before = full.clone()
    indices = torch.tensor([3, 1, 3, 2, 4, 1, 3], device=device)
    offsets = torch.tensor([0, 2, 4, 4, 7], device=device)
    sample_weights = torch.arange(1, 8, device=device).float() / 4 if weighted else None
    batch = SimpleNamespace(reverse_indices=indices)
    adapter = TorchRecLookup([dim], [0, 0], mode, device)
    dense = (
        dim % 4 == 0
        and padding % dim == 0
        and dtype != torch.bfloat16
        and (mode == DynamicEmbPoolingMode.NONE or dim <= 2048)
    )
    backend = adapter._dense_module(weighted) if dense else adapter._module(weighted)
    name = "_operation" if dense else "forward"
    with torch.no_grad(), patch.object(
        backend, name, wraps=getattr(backend, name)
    ) as call:
        actual = adapter(weights, batch, offsets, output_dtype, sample_weights)
    assert call.call_count == 1
    if mode == DynamicEmbPoolingMode.NONE:
        expected = weights[indices].to(output_dtype)
    else:
        expected = (
            F.embedding_bag(
                indices,
                weights.float(),
                offsets,
                mode="sum" if mode == DynamicEmbPoolingMode.SUM else "mean",
                per_sample_weights=sample_weights,
                include_last_offset=True,
            )
            .view(2, 2, dim)
            .transpose(0, 1)
            .reshape(2, 2 * dim)
            .to(output_dtype)
        )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(full, before, rtol=0, atol=0)


def test_dense_lookup_flat_view_stops_at_last_embedding():
    dim, rows, stride = 8, 6, 24
    flat = torch.arange((rows - 1) * stride + dim, device="cuda").float()
    weights = flat.as_strided((rows, dim), (stride, 1))
    indices = torch.tensor([rows - 1, 0], device="cuda")
    offsets = torch.tensor([0, 2], device="cuda")
    lookup = DenseLookup(dim, DynamicEmbPoolingMode.NONE, False, weights.device)
    actual = lookup.forward(weights, indices, offsets, torch.float32, None)
    torch.testing.assert_close(actual, weights[indices], rtol=0, atol=0)


@pytest.mark.parametrize(
    "output_dtype,weighted", [(torch.float32, False), (torch.float16, True)]
)
def test_dense_half_pooling_accumulates_before_output_cast(output_dtype, weighted):
    weights = torch.tensor([60000.0, -60000.0], device="cuda", dtype=torch.float16)
    weights = weights[:, None].repeat(1, 8)
    indices = torch.tensor([0, 0, 1, 1], device="cuda")
    offsets = torch.tensor([0, 4], device="cuda")
    sample_weights = torch.ones(4, device="cuda") if weighted else None
    lookup = TorchRecLookup([8], [0], DynamicEmbPoolingMode.SUM, weights.device)
    with torch.no_grad():
        actual = lookup(
            weights,
            SimpleNamespace(reverse_indices=indices),
            offsets,
            output_dtype,
            sample_weights,
        )
    # A half accumulator would overflow before the negative values cancel.
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)


def test_dense_lookup_waits_for_metadata_initialization():
    source, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    weights = torch.arange(4096, device="cuda").float().reshape(512, 8)
    indices = torch.tensor([0, 1], device="cuda")
    offsets = torch.tensor([0, 2], device="cuda")
    # Lazy loading a first CUDA kernel may synchronize the device and hide a
    # missing metadata dependency. Warm the exact operation before delaying it.
    warm = DenseLookup(8, DynamicEmbPoolingMode.NONE, False, weights.device)
    warm.forward(weights, indices, offsets, torch.float32, None)
    torch.cuda.synchronize()
    consumer.wait_stream(torch.cuda.current_stream())

    def delayed_zeros(shape, **kwargs):
        value = torch.full((shape,), 1024, **kwargs)
        source.synchronize()
        torch.cuda._sleep(100_000_000)
        value.zero_()
        return value

    with torch.cuda.stream(source), patch(
        "dynamicemb.dense_lookup.torch.zeros", new=delayed_zeros
    ):
        lookup = DenseLookup(8, DynamicEmbPoolingMode.NONE, False, weights.device)
    with torch.cuda.stream(consumer):
        actual = lookup.forward(weights, indices, offsets, torch.float32, None)
    consumer.synchronize()
    torch.testing.assert_close(actual, weights[indices], rtol=0, atol=0)


def test_dense_lookup_metadata_survives_drop_after_repeated_stream_use():
    source, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    weights = torch.arange(4096, device="cuda").float().reshape(512, 8)
    indices = torch.tensor([0, 1], device="cuda")
    offsets = torch.tensor([0, 2], device="cuda")
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(source):
        lookup = DenseLookup(8, DynamicEmbPoolingMode.NONE, False, weights.device)
    ref = weakref.ref(lookup)
    with torch.cuda.stream(consumer):
        first = lookup.forward(weights, indices, offsets, torch.float32, None)
        torch.cuda._sleep(100_000_000)
        second = lookup.forward(weights, indices, offsets, torch.float32, None)
    del lookup
    assert ref() is None
    with torch.cuda.stream(source):
        reuse = [
            torch.full((1,), 1024, device="cuda", dtype=torch.int64) for _ in range(32)
        ]
    consumer.synchronize()
    torch.testing.assert_close(first, weights[indices], rtol=0, atol=0)
    torch.testing.assert_close(second, weights[indices], rtol=0, atol=0)
    assert len(reuse) == 32


def test_concurrent_fallback_calls_use_their_own_rows():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread

    device = torch.device("cuda", torch.cuda.current_device())
    adapter = TorchRecLookup([8], [0], DynamicEmbPoolingMode.NONE, device)
    rows = [
        torch.full((2, 8), value, device=device, dtype=torch.bfloat16)
        for value in (11, 22)
    ]
    batch = SimpleNamespace(reverse_indices=torch.tensor([1, 0], device=device))
    offsets = torch.tensor([0, 2], device=device)
    expected = [adapter(w, batch, offsets, torch.float32).cpu() for w in rows]
    module = adapter._module(False)
    a_entered, a_done, b_started, b_enqueued = (Event() for _ in range(4))

    def before_lookup(_module, _args):
        if current_thread().name.startswith("request-a"):
            a_entered.set()
            assert b_started.wait(10)
            # On the old implementation B binds its rows and reaches its
            # post-hook while A is paused. With serialization B cannot enter.
            b_enqueued.wait(1)

    def after_lookup(_module, _args, _out):
        if current_thread().name.startswith("request-b"):
            b_enqueued.set()
            assert a_done.wait(10)

    def lookup(i):
        with torch.cuda.device(device):
            if i == 1:
                b_started.set()
            try:
                return adapter(rows[i], batch, offsets, torch.float32).cpu()
            finally:
                if i == 0:
                    a_done.set()

    with (
        module.register_forward_pre_hook(before_lookup),
        module.register_forward_hook(after_lookup),
        ThreadPoolExecutor(1, thread_name_prefix="request-a") as a_pool,
        ThreadPoolExecutor(1, thread_name_prefix="request-b") as b_pool,
    ):
        a = a_pool.submit(lookup, 0)
        assert a_entered.wait(10)
        b = b_pool.submit(lookup, 1)
        actual = [a.result(15), b.result(15)]
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference)


def test_fallback_restores_weights_when_offsets_fail_validation():
    device = torch.device("cuda", torch.cuda.current_device())
    adapter = TorchRecLookup([8], [0], DynamicEmbPoolingMode.SUM, device)
    table = adapter._module(False).embedding_bags["table_0"]
    original = table.weight.data, table.num_embeddings
    rows = torch.ones((2, 8), device=device)
    batch = SimpleNamespace(reverse_indices=torch.tensor([0, 1], device=device))
    bad_offsets = torch.tensor([0, 2], device=device, dtype=torch.float32)
    # BF16 routes to EBC. Promoted rows must not escape even if input validation fails.
    with pytest.raises(AssertionError, match="offsets must be of integer type"):
        adapter(rows.bfloat16(), batch, bad_offsets, torch.float32)
    assert table.weight.data_ptr() == original[0].data_ptr()
    assert table.num_embeddings == original[1]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dense_declines_a_misaligned_view(dtype):
    device = torch.device("cuda", torch.cuda.current_device())
    adapter = TorchRecLookup([8], [0], DynamicEmbPoolingMode.NONE, device)
    matrix = torch.arange(64, device=device, dtype=dtype).view(4, 16)
    indices = torch.tensor([3, 1, 0, 2], device=device)
    batch = SimpleNamespace(reverse_indices=indices)
    offsets = torch.tensor([0, 4], device=device)
    # Identical shapes and strides; only the starting column differs.
    for start in (0, 1):
        rows = matrix[:, start : start + 8]
        actual = adapter(rows, batch, offsets, dtype)
        torch.testing.assert_close(actual, rows[indices])
