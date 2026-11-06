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

"""A kernel group takes frequency counters only if it evicts by LFU or admits.

The counters ride the KJT weights channel of a whole sharding, so every group of
the sharding is handed the slice for its own features -- including a group that
has no use for it, such as CUSTOMIZED tables next to LFU ones.
``BatchedDynamicEmbeddingTablesV2`` must drop them there, or a table that never
asked for frequencies would have them misread as scores.
"""

import dynamicemb.batched_dynamicemb_tables as tables_module
import pytest
import torch
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    ProbabilisticAdmissionStrategy,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed.types import BoundsCheckMode

CAPACITY = 1024
DIM = 8


@pytest.fixture
def device():
    torch.cuda.set_device(0)
    return torch.cuda.current_device()


def _make_module(score_strategy, device, admit_strategy=None) -> BatchedDynamicEmbeddingTablesV2:
    option = DynamicEmbTableOptions(
        dim=DIM,
        max_capacity=CAPACITY,
        init_capacity=CAPACITY,
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=device,
        bucket_capacity=128,
        safe_check_mode=DynamicEmbCheckMode.IGNORE,
        local_hbm_for_values=1024**3,
        score_strategy=score_strategy,
        admit_strategy=admit_strategy,
    )
    return BatchedDynamicEmbeddingTablesV2(
        table_options=[option],
        table_names=["t0"],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.1,
        stochastic_rounding=False,
        bounds_check_mode=BoundsCheckMode.NONE,
    )


def _batch(device):
    indices = torch.tensor([3, 5, 3, 9], dtype=torch.int64, device=device)
    offsets = torch.tensor([0, 4], dtype=torch.int64, device=device)
    # What the EC dedup attaches to the sharding's weights: float32, one per key.
    counters = torch.tensor([2.0, 1.0, 2.0, 7.0], dtype=torch.float32, device=device)
    return indices, offsets, counters


def _spy_prefetch(monkeypatch):
    """Record the ``frequency_counters`` each ``dynamicemb_prefetch`` call gets."""
    seen = []
    real = tables_module.dynamicemb_prefetch

    def spy(*args, **kwargs):
        # Positional order: indices, offsets, cache, storage, feature_offsets,
        # initializer, optimizer, forward_stream, evict_strategy, frequency_counters, ...
        seen.append(args[9])
        return real(*args, **kwargs)

    monkeypatch.setattr(tables_module, "dynamicemb_prefetch", spy)
    return seen


@pytest.mark.parametrize(
    "score_strategy",
    [
        DynamicEmbScoreStrategy.LFU,
        (DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU),
        (DynamicEmbScoreStrategy.LFU, DynamicEmbScoreStrategy.TIMESTAMP),
    ],
)
def test_lfu_group_receives_counters(device, monkeypatch, score_strategy):
    module = _make_module(score_strategy, device)
    indices, offsets, counters = _batch(device)
    seen = _spy_prefetch(monkeypatch)

    module(indices, offsets, frequency_counters=counters)

    assert len(seen) == 1
    assert seen[0] is not None
    assert torch.equal(seen[0].cpu(), counters.cpu())


@pytest.mark.parametrize(
    "score_strategy",
    [
        DynamicEmbScoreStrategy.TIMESTAMP,
        DynamicEmbScoreStrategy.STEP,
        DynamicEmbScoreStrategy.CUSTOMIZED,
    ],
)
def test_non_lfu_group_drops_counters(device, monkeypatch, score_strategy):
    module = _make_module(score_strategy, device)
    if score_strategy == DynamicEmbScoreStrategy.CUSTOMIZED:
        module.set_score({"t0": 5})
    indices, offsets, counters = _batch(device)
    seen = _spy_prefetch(monkeypatch)

    module(indices, offsets, frequency_counters=counters)

    assert seen == [None]


def test_counters_absent_stay_absent(device, monkeypatch):
    module = _make_module(DynamicEmbScoreStrategy.LFU, device)
    indices, offsets, _ = _batch(device)
    seen = _spy_prefetch(monkeypatch)

    module(indices, offsets)

    assert seen == [None]


def test_prefetch_entry_point_filters_too(device, monkeypatch):
    """``prefetch`` is public and called directly by pipelines, not only via forward."""
    module = _make_module(DynamicEmbScoreStrategy.TIMESTAMP, device)
    indices, offsets, counters = _batch(device)
    seen = _spy_prefetch(monkeypatch)

    module.prefetch(indices, offsets, frequency_counters=counters)

    assert seen == [None]


def test_admitting_group_receives_counters(device, monkeypatch):
    """A group that admits needs the counters even when it does not evict by LFU."""
    module = _make_module(
        DynamicEmbScoreStrategy.TIMESTAMP,
        device,
        admit_strategy=ProbabilisticAdmissionStrategy(1.0),
    )
    indices, offsets, counters = _batch(device)
    seen = _spy_prefetch(monkeypatch)

    module(indices, offsets, frequency_counters=counters)

    assert len(seen) == 1
    assert seen[0] is not None
    assert torch.equal(seen[0].cpu(), counters.cpu())
