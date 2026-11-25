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

"""Score columns of a dump or a replay: logical order, ages instead of timestamps."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from dynamicemb.score.strategy import (
    DynamicEmbScoreStrategy,
    ScoreStrategy,
    get_physical_score_order,
    score_dump_permutation,
    score_load_permutation,
)
from dynamicemb.types import SCORE_TYPE
from torch import Tensor

if TYPE_CHECKING:
    from dynamicemb.key_value_table import DynamicEmbTableState


def _timestamp_score_columns(score_strategy: ScoreStrategy) -> List[int]:
    """Logical score-column indices holding a device timestamp.

    ``%globaltimer`` is per-device and resets across boots, so a raw timestamp is
    meaningless anywhere but the rank that produced it. These columns are dumped
    as an *age* (``cur_ts - score``) -- the same convention the file checkpoint
    uses for single-column LRU tables. Every other column (LFU frequency, STEP,
    CUSTOMIZED) is carried verbatim.
    """
    if isinstance(score_strategy, tuple):
        return [
            i
            for i, s in enumerate(score_strategy)
            if s == DynamicEmbScoreStrategy.TIMESTAMP
        ]
    return [0] if score_strategy == DynamicEmbScoreStrategy.TIMESTAMP else []


def _scores_to_age(
    score_strategy: ScoreStrategy, scores: Tensor, cur_ts: int
) -> Tensor:
    """Convert timestamp columns to ages, for dumping.

    *scores* is a ``[N, W]`` block in logical (configured) column order. When
    there is nothing to convert (no timestamp column, or an empty block) the
    input is returned unchanged rather than copied -- callers must not mutate the
    result in place. Otherwise a clone is returned.
    """
    ts_cols = _timestamp_score_columns(score_strategy)
    if not ts_cols or scores.numel() == 0:
        return scores
    out = scores.clone()
    for c in ts_cols:
        out[:, c] = torch.clamp(cur_ts - scores[:, c], min=0)
    return out


def _age_to_scores(
    score_strategy: ScoreStrategy, ages: Tensor, target_ts: int
) -> Tensor:
    """Inverse of :func:`_scores_to_age`, for replay.

    Same aliasing note: returns *ages* itself when there is nothing to convert.
    """
    ts_cols = _timestamp_score_columns(score_strategy)
    if not ts_cols or ages.numel() == 0:
        return ages
    out = ages.clone()
    for c in ts_cols:
        out[:, c] = torch.clamp(target_ts - ages[:, c], min=0)
    return out


def _fresh_score_block(
    state: DynamicEmbTableState,
    table_id: int,
    n: int,
    timestamp: int,
    insert_score: int,
) -> torch.Tensor:
    """``[N, num_scores]`` physical score block for keys written by replay.

    Used when the caller did not ask for ``ReplayContent.SCORE``. The delta does
    carry the source's scores, but they were not loaded, so a replayed key is
    scored as if it had just been inserted here instead: recency words get
    *timestamp*, every other word gets *insert_score*. The replica then ranks its
    restored keys by when it received them rather than by how the source ranked
    them -- acceptable because the score only orders future evictions, and a
    replica that evicts on its own schedule still holds correct embeddings.
    Ask for ``SCORE`` to inherit the source's ranking instead.

    *insert_score* is passed in rather than read off ``state.score``: that field
    is populated by the forward pass, and replay is the one write path that can
    run before a target has ever done a forward. The caller owns the table's
    score bookkeeping and always has a value. *n* is the number of keys.
    """
    device = state.device
    num_scores = state.key_index_map.num_scores_
    # Zeros, not ``empty``: the loop below fills one column per configured
    # strategy, while the width comes from the score policy. The two agree for
    # every strategy there is, but they are derived from different places, so a
    # future divergence should leave a quiet zero rather than whatever the
    # allocator handed back.
    block = torch.zeros((n, num_scores), device=device, dtype=SCORE_TYPE)
    physical = get_physical_score_order(state.options_list[table_id].score_strategy)
    for word, strategy in enumerate(physical):
        block[:, word] = (
            timestamp if strategy == DynamicEmbScoreStrategy.TIMESTAMP else insert_score
        )
    return block


def _dump_score_block(
    state: DynamicEmbTableState,
    table_id: int,
    slot_indices: torch.Tensor,
    primary_scores: torch.Tensor,
    cur_ts: int,
) -> torch.Tensor:
    """``[N, num_scores]`` per-key score block for a dumped table.

    Columns are in the user's configured (logical) order -- the same order the
    file checkpoint uses -- and timestamp columns are converted to an age
    relative to *cur_ts* so they survive the trip to another device or host.
    Single-score tables reuse the score column the dump already thresholded on;
    multi-word layouts gather every word from the slots.
    """
    device = state.device
    num_scores = state.key_index_map.num_scores_
    score_strategy = state.options_list[table_id].score_strategy
    if num_scores > 1:
        block = state.key_index_map.gather_score_blocks(
            table_id, slot_indices.to(device=device, dtype=torch.int64)
        )
        perm = score_dump_permutation(score_strategy)
        if perm != list(range(block.size(1))):
            block = block[:, perm].contiguous()
    else:
        block = primary_scores.to(device=device, dtype=SCORE_TYPE).view(-1, 1)
    return _scores_to_age(score_strategy, block, cur_ts)


def _replay_score_block(
    state: DynamicEmbTableState,
    table_id: int,
    scores: torch.Tensor,
    target_ts: int,
) -> torch.Tensor:
    """Inverse of :func:`_dump_score_block`: logical ages -> physical scores.

    Takes the ``[N, num_scores]`` logical block carried in the delta, restores
    timestamp columns against this device's *target_ts*, and permutes back into
    the physical device layout.

    Both steps index by *column*, so a block that is not exactly as wide as this
    table's score layout would silently transform or permute the wrong words --
    e.g. writing a frequency into the timestamp column. The width is validated
    here rather than trusted.
    """
    device = state.device
    score_strategy = state.options_list[table_id].score_strategy
    num_scores = state.key_index_map.num_scores_
    block = scores.to(device=device, dtype=SCORE_TYPE)
    if block.dim() == 1:
        block = block.view(-1, 1)
    if block.dim() != 2 or block.size(1) != num_scores:
        raise ValueError(
            f"replay_increment: table {table_id} has {num_scores} score word(s) "
            f"per key, but the delta carries a score block of shape "
            f"{tuple(scores.shape)}."
        )
    block = _age_to_scores(score_strategy, block, target_ts)
    perm = score_load_permutation(score_strategy)
    if perm != list(range(num_scores)):
        block = block[:, perm]
    return block.contiguous()
