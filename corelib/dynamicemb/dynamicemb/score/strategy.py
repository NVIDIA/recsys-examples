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

"""Score strategies: what a table's score is, and how it is laid out and dumped."""

import enum
from typing import Callable, List, Optional, Tuple, Union


class DynamicEmbScoreStrategy(enum.IntEnum):
    """
    Enumeration for different modes to set index-embedding's score.
    The index-embedding pair with smaller scores will be more likely to be evicted from the embedding table when the table is full.

    dynamicemb allows configuring scores by table.
    For a table, the scores in the subsequent forward passes are larger than those in the previous ones for modes TIMESTAMP and STEP.
    Users can also provide customized score(mode CUSTOMIZED) for each table's forward pass,
    or a score for every key (mode PRECISE_CUSTOMIZED).
    Attributes
    ----------
    TIMESTAMP:
        In a forward pass, embedding table's scores will be set to global nanosecond timer of device, and due to the timing of GPU scheduling,
          different scores may have slight differences.
        Users must not set scores under TIMESTAMP mode.
    STEP:
        Each embedding table has a member `step` which will increment for every forward pass.
        All scores in each forward pass are the same which is step's value.
        Users must not set scores under STEP mode.
    CUSTOMIZED:
        Each embedding table's score are managed by users.
        Users have to set the score before every forward pass using `set_score` interface.
    LFU:
        If there are not enough slots inside the bucket to store new keys, the least used key in the bucket will be evicted.
    PRECISE_CUSTOMIZED:
        Like CUSTOMIZED, but every key has a score of its own instead of one score per table: the module is
        called with a `ScoredKeyedJaggedTensor`, whose `scores` hold an int64 for each key, and a key's score
        is the one it came with. Users must not set scores with `set_score` under this mode.
        A key that repeats in a batch takes the largest of its scores; a later batch assigns, so the latest
        score of a key wins. While training the input must be a `ScoredKeyedJaggedTensor`; evaluation reads
        no score and takes either kind. Row-wise sharding only; no `admit_strategy`; no variable batch size
        per key; no HybridStorage; not on the hierarchical dist (`output_dist`). May be combined with
        NO_EVICTION as `(NO_EVICTION, PRECISE_CUSTOMIZED)`.
        Within one collection either every table uses it or none does, and a collection that does holds
        DynamicEmb tables only.
    NO_EVICTION:
        The table’s capacity doubles whenever there are not enough slots for new keys, and this continues until available memory is exhausted.
        Alone it keeps no score; it may be combined with CUSTOMIZED or PRECISE_CUSTOMIZED as `(NO_EVICTION, CUSTOMIZED)`.
        With such a combination do not use a score of 0: an erased key's slot is cleared to 0, and a full bucket gives up
        the slot with the smallest score, so a key scored 0 ties with an erased slot and can be the one given up.
        Incremental dump and replay are not supported for such tables.
        When the memory resources are insufficient, there will be a warning message, and training can continue but the accuracy of eviction cannot be guaranteed.
    """

    TIMESTAMP = 0
    STEP = 1
    CUSTOMIZED = 2
    LFU = 3
    NO_EVICTION = 4
    PRECISE_CUSTOMIZED = 5


# A score strategy is either a single :class:`DynamicEmbScoreStrategy` or a
# *compound* strategy expressed as a tuple. In a compound tuple the element that
# is ``TIMESTAMP`` provides the reduction score column used by
# ``incremental_dump`` (a device timestamp), and the remaining element drives
# eviction ranking. The user's tuple order is the *logical* order that
# checkpoints are written in; the on-device *physical* layout stays fixed (see
# :func:`get_physical_score_order`). The tuple form keeps the interface open for
# future combinations; only ``(TIMESTAMP, LFU)``, ``(NO_EVICTION, CUSTOMIZED)`` and
# ``(NO_EVICTION, PRECISE_CUSTOMIZED)`` (in either order) are supported for now.
ScoreStrategy = Union[DynamicEmbScoreStrategy, Tuple[DynamicEmbScoreStrategy, ...]]


# Supported compound strategies, as unordered element sets. Both orderings of a
# supported set are accepted; only the checkpoint column order differs.
SUPPORTED_COMPOUND_SCORE_STRATEGIES: Tuple[frozenset, ...] = (
    frozenset({DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU}),
    # NO_EVICTION is a mode rather than a score: the table never evicts and grows
    # instead. Alone it keeps no score; with CUSTOMIZED the score is CUSTOMIZED's.
    frozenset(
        {DynamicEmbScoreStrategy.NO_EVICTION, DynamicEmbScoreStrategy.CUSTOMIZED}
    ),
    frozenset(
        {
            DynamicEmbScoreStrategy.NO_EVICTION,
            DynamicEmbScoreStrategy.PRECISE_CUSTOMIZED,
        }
    ),
)


def has_no_eviction(score_strategy: "ScoreStrategy") -> bool:
    """Whether the table never evicts: ``NO_EVICTION`` alone, or in a compound."""
    if isinstance(score_strategy, tuple):
        return DynamicEmbScoreStrategy.NO_EVICTION in score_strategy
    return score_strategy == DynamicEmbScoreStrategy.NO_EVICTION


def get_score_part(score_strategy: "ScoreStrategy") -> Optional["ScoreStrategy"]:
    """The part of *score_strategy* that decides the stored score, without the
    no-eviction mode: ``None`` for ``NO_EVICTION`` alone (no score is kept),
    ``CUSTOMIZED`` for ``(NO_EVICTION, CUSTOMIZED)``, anything else unchanged."""
    if not has_no_eviction(score_strategy):
        return score_strategy
    if not isinstance(score_strategy, tuple):
        return None
    rest = tuple(s for s in score_strategy if s != DynamicEmbScoreStrategy.NO_EVICTION)
    return rest[0] if len(rest) == 1 else rest


def takes_score_per_key(score_strategy: Optional["ScoreStrategy"]) -> bool:
    """Whether the table takes a score for every key from its input: its score
    strategy, without the no-eviction mode, is ``PRECISE_CUSTOMIZED``."""
    return (
        score_strategy is not None
        and get_score_part(score_strategy) == DynamicEmbScoreStrategy.PRECISE_CUSTOMIZED
    )


def normalize_score_strategy(score_strategy: ScoreStrategy) -> ScoreStrategy:
    """Validate *score_strategy* and return it in canonical form.

    ``None`` is passed through unchanged: it is the sentinel used by the planner /
    sharding path for non-DynamicEmb (plain TorchRec) tables, which never build a
    score policy (see :class:`~dynamicemb.shard.embedding` handling of a ``None``
    score strategy).
    A single :class:`DynamicEmbScoreStrategy` is returned unchanged. A one-element
    tuple ``(X,)`` is unwrapped to the single strategy ``X`` (a single score column
    has no ordering to preserve). A multi-element tuple must form a supported
    compound set (``{TIMESTAMP, LFU}``, ``{NO_EVICTION, CUSTOMIZED}`` or
    ``{NO_EVICTION, PRECISE_CUSTOMIZED}``) and is returned as a tuple.

    Raises
    ------
    NotImplementedError
        If a tuple's element set is not a supported compound strategy.
    TypeError
        If *score_strategy* (or a tuple element) is not a ``DynamicEmbScoreStrategy``.
    """
    if score_strategy is None:
        return None
    if isinstance(score_strategy, tuple):
        for element in score_strategy:
            if not isinstance(element, DynamicEmbScoreStrategy):
                raise TypeError(
                    f"score_strategy tuple elements must be DynamicEmbScoreStrategy, "
                    f"got {type(element)}."
                )
        if len(score_strategy) == 0:
            raise NotImplementedError("score_strategy tuple must be non-empty.")
        if len(score_strategy) == 1:
            return score_strategy[0]
        # Reject duplicate elements: comparing the frozenset alone would collapse
        # duplicates (e.g. (TIMESTAMP, LFU, LFU) -> {TIMESTAMP, LFU}) and wrongly
        # accept an over-long tuple, so require the element count to match the
        # de-duplicated set exactly.
        if len(score_strategy) != len(frozenset(score_strategy)) or (
            frozenset(score_strategy) not in SUPPORTED_COMPOUND_SCORE_STRATEGIES
        ):
            raise NotImplementedError(
                f"Unsupported compound score_strategy {score_strategy}. Supported "
                f"compound strategies (any order): "
                f"{[tuple(sorted(s, key=lambda e: e.value)) for s in SUPPORTED_COMPOUND_SCORE_STRATEGIES]}."
            )
        return tuple(score_strategy)
    if not isinstance(score_strategy, DynamicEmbScoreStrategy):
        raise TypeError(
            f"score_strategy must be a DynamicEmbScoreStrategy or a tuple of them, "
            f"got {type(score_strategy)}."
        )
    return score_strategy


def validate_score_strategy(score_strategy: ScoreStrategy) -> None:
    """Raise if *score_strategy* is not a supported single or compound strategy."""
    normalize_score_strategy(score_strategy)


def get_eviction_score_strategy(
    score_strategy: ScoreStrategy,
) -> DynamicEmbScoreStrategy:
    """Return the strategy that drives eviction ranking.

    For a single strategy this is the strategy itself. For a compound strategy the
    ``TIMESTAMP`` element is only the incremental-dump reduction column, so eviction
    is driven by the remaining (non-``TIMESTAMP``) element (e.g. ``LFU`` for
    ``(TIMESTAMP, LFU)``).
    """
    if isinstance(score_strategy, tuple):
        non_timestamp = [
            s
            for s in score_strategy
            if s
            not in (
                DynamicEmbScoreStrategy.TIMESTAMP,
                DynamicEmbScoreStrategy.NO_EVICTION,
            )
        ]
        # Supported compounds always contain exactly one non-TIMESTAMP element.
        return non_timestamp[0] if non_timestamp else DynamicEmbScoreStrategy.TIMESTAMP
    return score_strategy


def score_strategy_has_timestamp_column(score_strategy: ScoreStrategy) -> bool:
    """Whether the score layout carries a device-timestamp reduction column.

    ``True`` for the single ``TIMESTAMP`` strategy and for compound strategies that
    include ``TIMESTAMP`` (e.g. ``(TIMESTAMP, LFU)``). Only such strategies produce a
    *time-based* ``incremental_dump`` (items touched since the threshold time);
    strategies without a timestamp column threshold on the absolute score value.
    """
    if isinstance(score_strategy, tuple):
        return DynamicEmbScoreStrategy.TIMESTAMP in score_strategy
    return score_strategy == DynamicEmbScoreStrategy.TIMESTAMP


def get_physical_score_order(
    score_strategy: ScoreStrategy,
) -> Tuple[DynamicEmbScoreStrategy, ...]:
    """Strategies in the order their score words are physically stored on device.

    A single strategy occupies one word. The compound ``{TIMESTAMP, LFU}`` maps to
    the underlying ``LruLfu`` policy whose physical layout is always
    ``(TIMESTAMP, LFU)`` -- word 0 is the last-access timestamp, word 1 the
    frequency -- regardless of the user's configured (logical) order. The
    no-eviction mode stores no word of its own: ``NO_EVICTION`` alone keeps one
    unused word, and in a compound the layout is that of the other strategy.
    """
    part = get_score_part(score_strategy)
    if part is None:
        return (DynamicEmbScoreStrategy.NO_EVICTION,)
    score_strategy = part
    if isinstance(score_strategy, tuple):
        if frozenset(score_strategy) == frozenset(
            {DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU}
        ):
            return (DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU)
        raise NotImplementedError(
            f"No physical score layout for compound score_strategy {score_strategy}."
        )
    return (score_strategy,)


def score_dump_permutation(score_strategy: ScoreStrategy) -> List[int]:
    """Column permutation to reorder a physical score block into logical order.

    Given a device-order block ``phys`` of shape ``[N, W]``, the checkpoint block in
    the user's configured order is ``phys[:, perm]``: ``out[:, j]`` is the physical
    column holding ``score_strategy[j]``. Identity (``[0, 1, ...]``) for single
    strategies and for compounds whose logical order already matches the physical
    layout.
    """
    score_strategy = get_score_part(score_strategy)
    if not isinstance(score_strategy, tuple):
        return [0]
    physical = get_physical_score_order(score_strategy)
    return [physical.index(s) for s in score_strategy]


def score_load_permutation(score_strategy: ScoreStrategy) -> List[int]:
    """Column permutation to reorder a logical (checkpoint) score block into physical order.

    Inverse of :func:`score_dump_permutation`: given a checkpoint block ``logical`` of
    shape ``[N, W]`` in the user's configured order, the device-order block is
    ``logical[:, perm]``, where ``perm[p]`` is the logical column holding the strategy
    stored at physical word ``p``.
    """
    score_strategy = get_score_part(score_strategy)
    if not isinstance(score_strategy, tuple):
        return [0]
    physical = get_physical_score_order(score_strategy)
    logical = list(score_strategy)
    return [logical.index(s) for s in physical]


def _score_function_group_key(fn: Optional[Callable]):
    """Stable, hashable identity for a score_function (for table grouping).

    (module, qualname, source-hash) so two references to the same function group
    together and different sources stay separate; falls back to object identity
    when the source cannot be read."""
    if fn is None:
        return None
    try:
        import hashlib
        import inspect

        digest = hashlib.md5(inspect.getsource(fn).encode("utf-8")).hexdigest()
        return (
            getattr(fn, "__module__", None),
            getattr(fn, "__qualname__", None),
            digest,
        )
    except (OSError, TypeError):
        return (
            getattr(fn, "__module__", None),
            getattr(fn, "__qualname__", None),
            id(fn),
        )
