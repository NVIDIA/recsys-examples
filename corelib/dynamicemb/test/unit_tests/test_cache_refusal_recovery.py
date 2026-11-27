# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed cache insertion preserves dirty rows, initialized hits and pin owners."""
from dataclasses import replace

import pytest
import torch
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.key_value_table import load_from_flat
from dynamicemb.types import CopyMode
from dynamicemb.utils import decode_table_values
from test_cache_refused_insert import make_model


def read_rows(model, keys):
    tids = torch.zeros_like(keys)
    _, found, slots = model._cache.lookup(keys, tids)
    cached = load_from_flat(model._cache._state, slots, tids, copy_mode=CopyMode.VALUE)
    backing = model._storage.find(keys, tids, copy_mode=CopyMode.VALUE)[-1]
    return torch.where(
        found[:, None], cached.view(torch.uint8), backing.view(torch.uint8)
    ).view(cached.dtype)


@pytest.mark.parametrize(
    "dtype,optimizer,held",
    [
        pytest.param(torch.float32, EmbOptimType.SGD, False, id="sgd"),
        pytest.param(torch.float32, EmbOptimType.ADAM, True, id="adam-held"),
        pytest.param(torch.float16, EmbOptimType.ADAM, False, id="half-adam"),
        pytest.param(
            torch.float8_e4m3fn,
            EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            True,
            id="fp8-adagrad-held",
        ),
    ],
)
def test_refusal_preserves_dirty_rows_moments_new_keys_and_hit_pin_ownership(
    dtype, optimizer, held
):
    template = make_model()
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=[
            replace(
                option,
                embedding_dtype=dtype,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.CONSTANT, value=12 / 256
                ),
            )
            for option in template._dynamicemb_options
        ],
        table_names=["t0"],
        feature_table_map=[0],
        output_dtype=torch.float32,
        pooling_mode=template.pooling_mode,
        optimizer=optimizer,
        learning_rate=0.25,
    )
    del template
    old = torch.arange(512, device="cuda", dtype=torch.int64)
    offsets = torch.tensor([0, old.numel()], device="cuda", dtype=torch.int64)
    model(old, offsets).sum().backward()
    expected = read_rows(model, old).clone()
    hits = old[:32]
    hit_offsets = torch.tensor([0, hits.numel()], device="cuda", dtype=torch.int64)
    if held:
        model.prefetch(hits, hit_offsets)
    new = torch.arange(
        10000,
        10000 + model._cache._state.capacity + 512,
        device="cuda",
        dtype=torch.int64,
    )
    request = torch.cat([hits, new])
    with pytest.raises(RuntimeError, match="cache could not place"):
        model.prefetch(request, torch.tensor([0, request.numel()], device="cuda"))
    counter = model._cache._state.key_index_map._ref_counter
    assert counter.sum().item() == (hits.numel() if held else 0)
    torch.testing.assert_close(
        read_rows(model, old).view(torch.uint8),
        expected.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    inserted = read_rows(model, new)
    torch.testing.assert_close(
        decode_table_values(inserted[:, :8], torch.float32),
        torch.full((new.numel(), 8), 12 / 256, device="cuda"),
        rtol=0,
        atol=0,
    )
    tail = inserted[:, 8:].view(torch.uint8)
    torch.testing.assert_close(tail, torch.zeros_like(tail), rtol=0, atol=0)
    with torch.no_grad():
        if held:
            model(hits, hit_offsets)  # consume only the older FIFO owner
        recovered = model(old, offsets)
    torch.testing.assert_close(
        recovered, decode_table_values(expected[:, :8], torch.float32), rtol=0, atol=0
    )
    assert counter.sum().item() == 0
