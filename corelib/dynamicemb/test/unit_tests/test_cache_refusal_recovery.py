# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed cache insertion preserves dirty rows, initialized hits and pin owners."""
import pytest
import torch
from dynamicemb import EmbOptimType
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.key_value_table import load_from_flat
from dynamicemb.types import CopyMode
from test_cache_refused_insert import CACHE_SLOTS, make_model


def read_rows(model, keys):
    tids = torch.zeros_like(keys)
    _, found, slots = model._cache.lookup(keys, tids)
    cached = load_from_flat(model._cache._state, slots, tids, copy_mode=CopyMode.VALUE)
    backing = model._storage.find(keys, tids, copy_mode=CopyMode.VALUE)[-1]
    return torch.where(found[:, None], cached, backing)


@pytest.mark.parametrize("optimizer", [EmbOptimType.SGD, EmbOptimType.ADAM])
@pytest.mark.parametrize("held", [False, True])
def test_refusal_preserves_dirty_rows_moments_new_keys_and_hit_pin_ownership(
    optimizer, held
):
    template = make_model()
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=template._dynamicemb_options,
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
        10000, 10000 + CACHE_SLOTS + 512, device="cuda", dtype=torch.int64
    )
    request = torch.cat([hits, new])
    with pytest.raises(RuntimeError, match="cache could not place"):
        model.prefetch(request, torch.tensor([0, request.numel()], device="cuda"))
    counter = model._cache._state.key_index_map._ref_counter
    assert counter.sum().item() == (hits.numel() if held else 0)
    torch.testing.assert_close(read_rows(model, old), expected, rtol=0, atol=0)
    inserted = read_rows(model, new)
    torch.testing.assert_close(
        inserted[:, :8], new.float()[:, None].expand(-1, 8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        inserted[:, 8:], torch.zeros_like(inserted[:, 8:]), rtol=0, atol=0
    )
    with torch.no_grad():
        if held:
            model(hits, hit_offsets)  # consume only the older FIFO owner
        recovered = model(old, offsets)
    torch.testing.assert_close(recovered, expected[:, :8], rtol=0, atol=0)
    assert counter.sum().item() == 0
