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

"""The physical score of a table (``ScoreSpec``) and the score handed to a call (``ScoreArg``)."""

from dataclasses import dataclass
from typing import Optional

import torch
from dynamicemb.score.strategy import DynamicEmbScoreStrategy, get_score_part
from dynamicemb_extensions import ScorePolicy


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    policy: ScorePolicy  # How to set the new score, this is the default behavior.
    dtype: torch.dtype = torch.uint64
    priority: int = 0  # If multiple scores exist, the one with lower priority will be reduced first.
    is_reduction: bool = True  # Whether it is reduced


def score_policy_num_scores(policy) -> int:
    """Number of contiguous score words a policy occupies per key. LRU_LFU stores
    a (timestamp, frequency) pair; every other policy is a single word. Mirrors
    ``score_policy_num_scores`` on the C++ side."""
    return 2 if policy == ScorePolicy.LRU_LFU else 1


def get_score_policy(score_strategy):
    """Build the physical :class:`ScoreSpec` for a (validated) score strategy.

    *score_strategy* is either a single :class:`DynamicEmbScoreStrategy` or a
    supported compound tuple (currently only ``{TIMESTAMP, LFU}`` in either order).
    A compound strategy maps to the ``LruLfu`` policy whose physical layout is
    fixed at ``(timestamp, frequency)`` regardless of the user's tuple order; the
    user's ordering only affects checkpoint column order (see
    :func:`score_dump_permutation`). The no-eviction mode adds no score of its own:
    ``NO_EVICTION`` alone keeps none: its one word is a ``FLAG`` (1 while a key
    occupies the slot, 0 after an erase), which is not a score a user sets but
    what lets a full bucket give up an erased slot before a key in use. In a
    compound the score is the other strategy's.
    """
    score_strategy = get_score_part(score_strategy)
    if score_strategy is None:
        return ScoreSpec(name="no_eviction", policy=ScorePolicy.FLAG)
    if isinstance(score_strategy, tuple):
        # Only the {TIMESTAMP, LFU} compound is supported (validated at config
        # construction). Two adjacent AoS score words per key -- word 0 =
        # last-access timestamp (selects keys for a time-based incremental dump),
        # word 1 = frequency (drives LFU eviction). num_scores is derived from the
        # policy. The spec name maps to word 0 (score_index 0), i.e. the timestamp
        # column incremental_dump thresholds on -- hence "lru_lfu".
        if frozenset(score_strategy) == frozenset(
            {DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU}
        ):
            return ScoreSpec(
                name="lru_lfu",
                policy=ScorePolicy.LRU_LFU,
                dtype=torch.uint64,
                is_reduction=True,
            )
        raise NotImplementedError(
            f"Unsupported compound score_strategy {score_strategy}."
        )
    if score_strategy == DynamicEmbScoreStrategy.TIMESTAMP:
        return ScoreSpec(name="timestamp", policy=ScorePolicy.GLOBAL_TIMER)
    elif score_strategy == DynamicEmbScoreStrategy.STEP:
        return ScoreSpec(name="step", policy=ScorePolicy.ASSIGN)
    elif score_strategy in (
        DynamicEmbScoreStrategy.CUSTOMIZED,
        DynamicEmbScoreStrategy.PRECISE_CUSTOMIZED,
    ):
        # Stored the same way: what differs is only where a score comes from.
        return ScoreSpec(name="customized", policy=ScorePolicy.ASSIGN)
    elif score_strategy == DynamicEmbScoreStrategy.LFU:
        return ScoreSpec(name="frequency", policy=ScorePolicy.ACCUMULATE)
    else:
        raise RuntimeError("Not supported score strategy.")
