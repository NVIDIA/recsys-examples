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

"""What a find or an insert is given for the score: the input it comes from, and the ``ScoreArg``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from dynamicemb_extensions import EvictStrategy, ScorePolicy

if TYPE_CHECKING:
    from dynamicemb.state.table_state import DynamicEmbTableState


@dataclass
class ScoreArg:
    name: str
    value: Optional[torch.Tensor] = None
    policy: Optional[
        ScorePolicy
    ] = None  # How to set the new score, and providing this will override the default.


def score_source(
    evict_strategy: Optional[EvictStrategy],
    frequency: Optional[torch.Tensor] = None,
    key_scores: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """The per-key values a table's score is made from, or ``None`` if it has none.

    The two inputs a key can carry are separate channels -- ``frequency``, how
    often the batch counted it, and ``key_scores``, what the input scored it
    with -- and this is the one place that says which of them a table takes: an
    LFU table the frequency, a CUSTOMIZED table the score of the input. A caller
    chooses here, once, and hands the table's finds and inserts that one tensor
    (``score_input``); the tables and stores below it never see the other.

    ``None`` means "the table's own default" -- ``get_find_score_arg`` and
    ``get_insert_score_arg`` supply it.
    """
    if evict_strategy == EvictStrategy.KLfu:
        return frequency
    if evict_strategy == EvictStrategy.KCustomized:
        return key_scores
    return None


def get_find_score_arg(
    state: DynamicEmbTableState,
    num_keys: int,
    device: torch.device,
    score_input: Optional[torch.Tensor] = None,
    *,
    const_lookup: bool = False,
) -> ScoreArg:
    """Build a ScoreArg for find/lookup operations.

    When ``state.training`` is False (eval), returns CONST policy so that
    existing scores in the hash table are never modified.

    When *const_lookup* is True (e.g. flush-time verification), always use the
    same read-only CONST policy regardless of ``state.training``.

    When ``state.training`` is True:
      - LFU: ACCUMULATE with provided or default (ones) frequency.
      - LRU (GLOBAL_TIMER): no explicit value needed.
      - CUSTOMIZED / STEP: ASSIGN with ``state.score``; a CUSTOMIZED table that
        takes a score per key is given those scores instead, one per key, and
        assigns each key its own.

    ``score_input`` is what :func:`score_source` chose for the table -- the
    frequency of an LFU table, the scores of the input for a CUSTOMIZED one --
    so this takes one tensor, whichever it is.
    """
    # A table that keeps no score (NO_EVICTION alone) has nothing to update.
    if state.score_policy.policy == ScorePolicy.FLAG:
        return ScoreArg(
            name=state.score_policy.name, value=None, policy=ScorePolicy.CONST
        )

    if const_lookup or not state.training:
        return ScoreArg(
            name=state.score_policy.name, value=None, policy=ScorePolicy.CONST
        )

    # LFU: use provided frequency for ACCUMULATE; must have length num_keys for lookup.
    if state.evict_strategy == EvictStrategy.KLfu:
        if score_input is not None and score_input.numel() == num_keys:
            scores = score_input.contiguous()
        else:
            # Fallback: each key counts as 1 so score accumulates by 1 per lookup
            scores = torch.ones(num_keys, device=device, dtype=torch.long)
    elif state.evict_strategy == EvictStrategy.KCustomized:
        if score_input is not None:
            assert score_input.numel() == num_keys
            scores = score_input.contiguous()
        else:
            scores = torch.empty(num_keys, device=device, dtype=torch.long)
            scores.fill_(state.score)
    else:
        scores = None

    return ScoreArg(
        name=state.score_policy.name,
        value=scores,
        policy=state.score_policy.policy,
    )


def get_insert_score_arg(
    state: DynamicEmbTableState,
    num_keys: int,
    device: torch.device,
    scores: Optional[torch.Tensor] = None,
    preserve_existing: bool = False,
) -> ScoreArg:
    """Build a ScoreArg for insert operations (new keys).

    *preserve_existing* should be True when re-inserting keys that already
    exist in the table (e.g. backward embedding update) so that their scores
    are not overwritten.  This fixes the bug where backward re-inserts
    incorrectly assigned new scores.

    When *preserve_existing* is False (the common case for genuinely new keys):
      - LRU: GLOBAL_TIMER (no explicit value).
      - LFU / CUSTOMIZED / STEP: ASSIGN with provided *scores* or
        ``state.score`` as default.
      - ACCUMULATE policy is converted to ASSIGN for inserts.
    """
    if preserve_existing:
        return ScoreArg(
            name=state.score_policy.name, value=None, policy=ScorePolicy.CONST
        )

    if state.score_policy.policy == ScorePolicy.FLAG:
        # NO_EVICTION alone keeps no score a user could set, but a new key still
        # sets its flag, so that it ranks above the slots an erase left.
        return ScoreArg(
            name=state.score_policy.name, value=None, policy=ScorePolicy.FLAG
        )

    is_lru = state.evict_strategy == EvictStrategy.KLru
    if not is_lru and scores is None:
        scores = torch.empty(num_keys, device=device, dtype=torch.uint64)
        scores.fill_(state.score)

    policy = state.score_policy.policy
    if policy == ScorePolicy.ACCUMULATE:
        policy = ScorePolicy.ASSIGN
    if is_lru and scores is not None:
        policy = ScorePolicy.ASSIGN

    return ScoreArg(name=state.score_policy.name, value=scores, policy=policy)
