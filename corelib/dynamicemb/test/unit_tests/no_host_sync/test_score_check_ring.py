# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The late check for a negative score never waits for the device in a step.

``test_check_scores_late.py`` pins what the check reports and when. This file counts the
host syncs of the check with the probe's instruments (the CUDA profiler), in the two cases
that used to wait:

* the host is a whole ring of checks ahead of the device (``queue_ahead``: a sleep on the
  stream before the checks, so that none of their answers has landed). The check used to
  block on the oldest answer to free a slot;
* a module whose check mode was switched on after construction, which has no ring yet. The
  check used to read the scores at once (``bool((scores < 0).any())``).

The scores are not negative: a report is not a sync, but the census windows are better
without one. Needs a GPU and the CUDA profiler (the ``instruments_ok`` fixture validates
it first).
"""
import pytest

try:
    import torch
except ImportError:
    pytest.skip("needs torch", allow_module_level=True)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

DIM = 8
KEYS = 64
QUEUE_AHEAD = 8  # twice the slots of the ring
SLEEP_CYCLES = 200_000_000


def _build(safe_check_mode):
    from dynamicemb import (
        DynamicEmbPoolingMode,
        DynamicEmbScoreStrategy,
        DynamicEmbTableOptions,
        EmbOptimType,
    )
    from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2

    options = DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=torch.cuda.current_device(),
        dim=DIM,
        max_capacity=1 << 16,
        safe_check_mode=safe_check_mode,
        local_hbm_for_values=1024**3,
        score_strategy=DynamicEmbScoreStrategy.PRECISE_CUSTOMIZED,
        caching=False,
    )
    return BatchedDynamicEmbeddingTablesV2(
        table_options=[options],
        table_names=["t_0"],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.SUM,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.0,
        stochastic_rounding=False,
    )


def _batch():
    device = torch.device("cuda", torch.cuda.current_device())
    indices = torch.arange(KEYS, dtype=torch.int64, device=device)
    scores = torch.arange(1, KEYS + 1, dtype=torch.int64, device=device)
    return indices, scores


@cuda_only
def test_a_ring_of_checks_ahead_of_the_device_makes_no_blocking_call(
    count_syncs, instruments_ok
):
    from dynamicemb import DynamicEmbCheckMode

    model = _build(DynamicEmbCheckMode.ERROR)
    indices, scores = _batch()

    def queue_ahead(i):
        # The device is busy: no answer lands while the host issues the checks, so every
        # slot of the ring is in flight from the fifth on.
        torch.cuda._sleep(SLEEP_CYCLES)
        for _ in range(QUEUE_AHEAD):
            model._check_scores(scores, indices)

    result = count_syncs(queue_ahead, iters=4, warmup=2)
    assert result.blocking == 0, result.describe()


@cuda_only
def test_a_check_mode_switched_on_later_makes_no_blocking_call(
    count_syncs, instruments_ok
):
    from dynamicemb import DynamicEmbCheckMode

    model = _build(DynamicEmbCheckMode.IGNORE)
    assert model._score_ring is None
    for option in model._dynamicemb_options:
        option.safe_check_mode = DynamicEmbCheckMode.ERROR
    indices, scores = _batch()

    # The warm-up calls make the ring (a pinned allocation, once); the measured ones
    # only launch kernels and copy into it.
    result = count_syncs(lambda i: model._check_scores(scores, indices), iters=4)
    assert model._score_ring is not None
    assert result.blocking == 0 and result.by_kind["al"] == 0, result.describe()
