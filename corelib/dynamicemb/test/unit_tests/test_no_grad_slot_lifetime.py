# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward pins live until backward, or only until a no-grad forward returns."""

import pytest
import torch
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2


def make_model(pooling, caching):
    assert torch.cuda.is_available()
    dim = 8
    options = [
        DynamicEmbTableOptions(
            dim=dim,
            max_capacity=4096,
            init_capacity=4096,
            bucket_capacity=128,
            device_id=torch.cuda.current_device(),
            embedding_dtype=torch.float32,
            local_hbm_for_values=1024 * dim * 4 if caching else 1024**3,
            caching=caching,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.DEBUG
            ),
        )
        for _ in range(2)
    ]
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=options,
        table_names=["t0", "t1"],
        feature_table_map=[0, 1],
        output_dtype=torch.float32,
        pooling_mode=pooling,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.25,
    )
    assert model.training
    assert (model._cache is not None) == caching
    owner = model._cache if caching else model._storage
    return model, owner._state.key_index_map._ref_counter


def batch(iteration):
    keys = torch.cat(
        [torch.arange(16) + 16 * iteration, torch.arange(16) + 10000 + 16 * iteration]
    ).cuda()
    offsets = torch.arange(0, 33, 8, dtype=torch.int64, device="cuda")
    return keys, offsets


def check_values(output, keys, pooling):
    values = (keys % 100000).float()
    if pooling == DynamicEmbPoolingMode.NONE:
        expected = values[:, None].expand(-1, 8)
    else:
        expected = values.reshape(2, 2, 8).sum(-1).T.repeat_interleave(8, dim=1)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "pooling", [DynamicEmbPoolingMode.NONE, DynamicEmbPoolingMode.SUM]
)
@pytest.mark.parametrize("caching", [False, True])
def test_no_grad_forward_releases_slot_pins(pooling, caching):
    model, counter = make_model(pooling, caching)
    for iteration in range(8):
        keys, offsets = batch(iteration)
        with torch.no_grad():
            output = model(keys, offsets)
        assert not output.requires_grad
        check_values(output, keys, pooling)
        assert torch.count_nonzero(counter).item() == 0


@pytest.mark.parametrize(
    "pooling", [DynamicEmbPoolingMode.NONE, DynamicEmbPoolingMode.SUM]
)
@pytest.mark.parametrize("caching", [False, True])
def test_grad_forward_keeps_pins_until_backward(pooling, caching):
    model, counter = make_model(pooling, caching)
    keys, offsets = batch(0)
    output = model(keys, offsets)
    assert output.requires_grad
    check_values(output, keys, pooling)
    assert counter.sum().item() > 0
    output.backward(torch.zeros_like(output))
    assert torch.count_nonzero(counter).item() == 0
    with torch.no_grad():
        output = model(keys, offsets)
    check_values(output, keys, pooling)
    assert torch.count_nonzero(counter).item() == 0
