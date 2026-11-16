# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused tables keep their own ``local_hbm_for_values``.

The layout (HBM-only, caching, hybrid) is chosen for the fused group from the
sums of its sizes and budgets. When the rows do not all fit, each table keeps
the share of its rows that its *own* budget buys in HBM: a 10% / 90% split from
the planner stays 10% / 90% rather than becoming one shared ratio for both.
"""

from typing import List

import pytest
import torch
from dynamicemb import (
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import (
    BatchedDynamicEmbeddingTablesV2,
    StorageLayout,
)
from dynamicemb.extendable_tensor import DeviceExtendableBuffer

DIM = 16
ROWS = 64 * 1024  # a multiple of every bucket width involved, so nothing rounds
TABLE_BYTES = ROWS * DIM * 4  # fp32, SGD keeps no optimizer state
CACHE_BUCKET = 1024  # the bucket width a cache / HBM tier is built with


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


def build(current_device: int, budgets: List[int], caching: bool):
    options = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=current_device,
            dim=DIM,
            max_capacity=ROWS,
            local_hbm_for_values=budget,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
        )
        for budget in budgets
    ]
    return BatchedDynamicEmbeddingTablesV2(
        table_options=options,
        table_names=[f"t_{i}" for i in range(len(budgets))],
        feature_table_map=list(range(len(budgets))),
        pooling_mode=DynamicEmbPoolingMode.SUM,
        optimizer=EmbOptimType.SGD,
    )


def hbm_tier(model):
    """The state holding the HBM rows of a model that splits its rows."""
    if model._storage_layout == StorageLayout.CACHING:
        return model._cache._state
    assert model._storage_layout == StorageLayout.HYBRID
    return model._storage._hbm


def assert_on_device(state):
    for i, buf in enumerate(state.tables):
        assert isinstance(
            buf, DeviceExtendableBuffer
        ), f"table {i}'s HBM rows were allocated as {type(buf).__name__}"


def train_one_step(model, current_device, bags=2):
    """Every table answers a lookup and takes a gradient, `bags` bags apiece."""
    n = len(model._dynamicemb_options)
    device = torch.device(f"cuda:{current_device}")
    indices = torch.arange(bags * n, dtype=torch.int64, device=device)
    offsets = torch.arange(bags * n + 1, dtype=torch.int64, device=device)
    out = model(indices, offsets)
    assert out.shape == (bags, n * DIM)
    out.sum().backward()
    torch.cuda.synchronize()


@pytest.mark.parametrize("caching", [True, False])
def test_each_table_keeps_its_own_share(current_device, caching):
    """1/16 and 3/4 of each table's own bytes. One ratio for the group --
    13/32 of the summed bytes -- would give both 26624 rows."""
    model = build(
        current_device, [TABLE_BYTES // 16, TABLE_BYTES * 3 // 4], caching=caching
    )
    assert model._storage_layout == (
        StorageLayout.CACHING if caching else StorageLayout.HYBRID
    )

    hbm = hbm_tier(model)
    assert hbm.key_index_map.per_table_capacity_ == [ROWS // 16, ROWS * 3 // 4]
    assert_on_device(hbm)
    if not caching:
        # The two tiers of a hybrid table are disjoint: the host holds the rest.
        host = model._storage._host
        assert host.key_index_map.per_table_capacity_ == [
            ROWS - ROWS // 16,
            ROWS - ROWS * 3 // 4,
        ]
    train_one_step(model, current_device)


def test_a_table_without_a_budget_does_not_take_its_groupmates(current_device):
    """A hybrid split gives the first table, which has no budget, the
    one-bucket floor and the second all of its own. Listed first, it must not
    move the group's HBM rows to the host either.

    The same budgets with caching=True are a different case: the cache that
    zero-budget table would get is refused, in the test below."""
    model = build(current_device, [0, TABLE_BYTES * 3 // 4], caching=False)

    hbm = hbm_tier(model)
    assert hbm.key_index_map.per_table_capacity_ == [CACHE_BUCKET, ROWS * 3 // 4]
    assert_on_device(hbm)
    train_one_step(model, current_device)


@pytest.mark.parametrize(
    "budget", [0, TABLE_BYTES * 3 // 4], ids=["zero_budget", "funded"]
)
def test_zero_hbm_budget_takes_caching_off_the_table(current_device, budget):
    """caching=True with a zero local_hbm_for_values is normalized off, with a
    warning, before the grouping check: a cache is an HBM tier, and the
    one-bucket cache the runtime would build cannot absorb a batch. A positive
    local budget keeps the cache with global_hbm_for_values left at its
    default, which direct construction never reads."""
    if budget == 0:
        with pytest.warns(UserWarning, match="local_hbm_for_values of zero"):
            model = build(current_device, [0], caching=True)
        assert model._cache is None
        assert model._storage_layout == StorageLayout.HOST_ONLY
        # More unique keys than the former one-bucket cache plus its overflow
        # (1024 + 3 * 1024 rows) could ever hold.
        train_one_step(model, current_device, bags=8192)
    else:
        model = build(current_device, [budget], caching=True)
        assert model._storage_layout == StorageLayout.CACHING
        assert model._cache is not None


def test_mixed_caching_left_by_normalization_is_refused(current_device):
    """The zero-budget tablemate loses its cache to normalization, the funded
    one keeps it, and one fused module cannot hold both layouts -- so the
    group is refused, by name, rather than built with a broken half-cache."""
    with pytest.warns(UserWarning, match="local_hbm_for_values of zero"), pytest.raises(
        ValueError, match="caching was disabled for t_0"
    ):
        build(current_device, [0, TABLE_BYTES * 3 // 4], caching=True)


def test_a_group_that_fits_its_summed_budget_stays_in_hbm(current_device):
    """The layout is still decided for the group: 2x of rows against 2x of
    budget is HBM-only, the table with no budget included -- and on the device
    although it is the first table."""
    model = build(current_device, [0, 2 * TABLE_BYTES], caching=False)

    assert model._storage_layout == StorageLayout.HBM_ONLY
    assert_on_device(model._storage._state)
    train_one_step(model, current_device)
