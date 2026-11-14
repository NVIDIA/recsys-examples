# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TorchRec dense kernels and EC/EBC fallback share the same row contract."""

import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from dynamicemb import DynamicEmbPoolingMode
from dynamicemb.dense_lookup import DenseLookup
from dynamicemb.torchrec_lookup import TorchRecLookup


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim,padding", [(4, 0), (8, 16), (8, 1), (7, 0), (12, 0)])
@pytest.mark.parametrize(
    "mode,weighted",
    [
        (DynamicEmbPoolingMode.NONE, False),
        (DynamicEmbPoolingMode.SUM, False),
        (DynamicEmbPoolingMode.MEAN, False),
        (DynamicEmbPoolingMode.SUM, True),
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
    dense = dim % 4 == 0 and padding % dim == 0 and dtype != torch.bfloat16
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


@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weighted", [False, True])
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
