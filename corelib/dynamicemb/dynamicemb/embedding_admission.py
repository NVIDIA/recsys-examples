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


import math
import warnings
from typing import List, Optional

import torch
from dynamicemb.initializer import MultiTableInitializer
from dynamicemb.scored_hashtable import (
    ScoreArg,
    ScorePolicy,
    ScoreSpec,
    get_scored_table,
)
from dynamicemb.types import (
    DEFAULT_UNIFORM_LOWER,
    DEFAULT_UNIFORM_UPPER,
    AdmissionStrategy,
    Counter,
    MultiTableAdmitter,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    MemoryType,
    group_key_of,
)
from dynamicemb_extensions import flagged_compact


class KVCounter:
    """Per-table counter configuration.

    Sizes one logical table's share of the counter. The strategy that counts
    carries one of these; when a module materializes that strategy, the shares
    of its tables become a single ``MultiTableKVCounter`` over one fused scored
    hash table.
    """

    def __init__(
        self,
        capacity: int,
        bucket_capacity: int = 1024,
        key_type: torch.dtype = torch.int64,
    ):
        self.capacity = capacity
        self.bucket_capacity = bucket_capacity
        self.key_type = key_type

    def get_grouped_key(self):
        """What has to match for two tables to share one fused counter.

        ``capacity`` is left out on purpose: the fused table takes one capacity
        per logical table, so tables are free to differ there. The bucket
        layout and the key type describe the single physical table they all
        share, so tables that disagree on those cannot be fused.
        """
        return (type(self).__name__, self.bucket_capacity, self.key_type)


class MultiTableKVCounter(Counter):
    """Multi-table counter backed by a single fused ``ScoredHashTable``.

    Accepts a list of per-table ``KVCounter`` configs and creates one hash
    table whose capacity list maps to the individual counters.
    """

    def __init__(
        self,
        kv_counters: List[KVCounter],
        device: torch.device,
    ):
        if not kv_counters:
            raise ValueError("kv_counters must be non-empty")

        capacities = [kv.capacity for kv in kv_counters]
        self.score_name_ = "counter"
        self.score_specs_ = [
            ScoreSpec(name=self.score_name_, policy=ScorePolicy.ACCUMULATE)
        ]
        self.score_arg_ = ScoreArg(name=self.score_name_)
        self.table_ = get_scored_table(
            capacities,
            kv_counters[0].bucket_capacity,
            kv_counters[0].key_type,
            self.score_specs_,
            device,
        )

    def add(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        self.score_arg_.value = frequencies
        scores_out = torch.empty(keys.numel(), dtype=torch.int64, device=keys.device)
        self.table_.insert(keys, table_ids, self.score_arg_, score_out=scores_out)
        return scores_out

    def erase(self, keys: torch.Tensor, table_ids: torch.Tensor) -> None:
        self.table_.erase(keys, table_ids)

    def memory_usage(self, mem_type=MemoryType.DEVICE) -> int:
        return self.table_.memory_usage(mem_type)

    def load(self, key_file, counter_file, table_id: int, dist_type: str) -> None:
        self.table_.load(
            key_file,
            {self.score_name_: counter_file},
            table_id=table_id,
            dist_type=dist_type,
        )

    def dump(self, key_file, counter_file, table_id: int) -> None:
        self.table_.dump(key_file, {self.score_name_: counter_file}, table_id=table_id)


def _check_non_admitted_initializer_args(initializer_args):
    """The initializer for rejected rows is optional, but not free-form."""
    if initializer_args is None:
        return
    if not isinstance(initializer_args, DynamicEmbInitializerArgs):
        raise TypeError(
            "initializer_args must be a DynamicEmbInitializerArgs, got "
            f"{type(initializer_args).__name__}"
        )
    if initializer_args.mode == DynamicEmbInitializerMode.UNIFORM and (
        initializer_args.lower is None or initializer_args.upper is None
    ):
        # A table's own bounds resolve to +/-sqrt(1 / num_embeddings) in the
        # planner, which these cannot: they belong to no one table. Say so,
        # since the fallback is a far wider interval than a row count gives.
        warnings.warn(
            "A UNIFORM initializer for non-admitted rows cannot take its bounds "
            "from a table's row count, so it falls back to "
            f"[{DEFAULT_UNIFORM_LOWER}, {DEFAULT_UNIFORM_UPPER}]. Give lower "
            "and upper to choose them, or drop initializer_args to let those "
            "rows use the table's own initializer.",
            UserWarning,
            stacklevel=3,
        )


def _non_admitted_initializer(table_strategies, device):
    """What writes the rows these tables reject, or None for the tables' own.

    Grouping made the modes agree; the parameters may still differ per table,
    which is what MultiTableInitializer carries.
    """
    if table_strategies[0].initializer_args is None:
        return None
    return MultiTableInitializer.create(
        [strategy.initializer_args for strategy in table_strategies], device
    )


class FrequencyAdmissionStrategy(AdmissionStrategy):
    """Admit a key once it has been seen often enough.

    Configuration only. The counter that does the seeing is opened by
    :meth:`create_admitter`, once per fused module, so one of these can be
    built before CUDA is up and handed to as many tables as one likes.

    Parameters
    ----------
    threshold : int
        Accumulated occurrences a key needs before it may enter the table.
    counter : Optional[KVCounter]
        How much room each table this serves gets for counting. A key is erased
        the moment it is admitted, so size this for the keys still waiting, not
        for the embedding table. Give tables different room by giving them
        separately configured strategies: capacity is not part of what they are
        grouped on, so they still share a module. Optional only so that the
        deprecated ``DynamicEmbTableOptions.admission_counter`` can still
        supply it; leaving it unset otherwise fails when the admitter is built.
    initializer_args : Optional[DynamicEmbInitializerArgs]
        How to initialize the rows this rejects. None -- the default -- leaves
        them to the table's own initializer, which is also the only way to get
        bounds derived from a table's row count.
    """

    def __init__(
        self,
        threshold: int,
        counter: Optional[KVCounter] = None,
        initializer_args: Optional[DynamicEmbInitializerArgs] = None,
    ):
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")
        if counter is not None and not isinstance(counter, KVCounter):
            raise TypeError(
                "Frequency admission counts occurrences, so it needs a "
                f"KVCounter to count them in, got {type(counter).__name__}"
            )
        _check_non_admitted_initializer_args(initializer_args)

        self.threshold = threshold
        self.counter = counter
        self.initializer_args = initializer_args

    def get_grouped_key(self):
        # The threshold decides for a whole batch at once, so tables sharing a
        # module must share it. Of the initializer only the mode has to match;
        # its parameters are resolved per table. The counter contributes its
        # own answer, which leaves out the capacity for the same reason.
        return (
            type(self).__name__,
            self.threshold,
            group_key_of(self.initializer_args),
            group_key_of(self.counter),
        )

    @classmethod
    def create_admitter(cls, table_strategies, device):
        first = cls.one_configuration(table_strategies)
        if any(strategy.counter is None for strategy in table_strategies):
            raise ValueError(
                "Frequency admission counts occurrences, so it needs a counter "
                "to count them in: FrequencyAdmissionStrategy(threshold=..., "
                "counter=KVCounter(capacity=...))."
            )
        return MultiTableFrequencyAdmitter(
            threshold=first.threshold,
            counter=MultiTableKVCounter(
                [strategy.counter for strategy in table_strategies], device
            ),
            non_admitted_initializer=_non_admitted_initializer(
                table_strategies, device
            ),
        )


class MultiTableFrequencyAdmitter(MultiTableAdmitter):
    """What :class:`FrequencyAdmissionStrategy` becomes for one fused module."""

    def __init__(
        self,
        threshold: int,
        counter: Counter,
        non_admitted_initializer=None,
    ):
        self._threshold = threshold
        self._counter = counter
        self._non_admitted_initializer = non_admitted_initializer

    def admit(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        frequencies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if frequencies is None:
            frequencies = torch.ones(
                keys.shape[0], dtype=torch.int64, device=keys.device
            )
        elif frequencies.shape[0] != keys.shape[0]:
            raise ValueError(
                "Keys and frequencies must have same length, got "
                f"{keys.shape[0]} and {frequencies.shape[0]}"
            )

        accumulated = self._counter.add(keys, table_ids, frequencies)
        admit_mask = accumulated >= self._threshold

        # A key that got in is no longer waiting, so it stops taking up room.
        # This repeats a compaction the caller also makes over the same mask;
        # one pass over the missing keys is worth keeping the counter's whole
        # lifecycle in the one place that knows the counter exists.
        _, _, (admitted_keys, admitted_table_ids) = flagged_compact(
            admit_mask, [keys, table_ids]
        )
        if admitted_keys.numel() > 0:
            self._counter.erase(admitted_keys, admitted_table_ids)
        return admit_mask

    def state(self) -> Optional[Counter]:
        return self._counter

    @property
    def non_admitted_initializer(self):
        return self._non_admitted_initializer


class ProbabilisticAdmissionStrategy(AdmissionStrategy):
    """Admit a key by a coin toss, once per appearance, until it gets in.

    Admission is consulted only for a key that is missing, so a key gets a
    fresh toss every time it turns up and is not yet in the table: it takes
    ``1 / probability`` appearances on average to be admitted. That filters by
    frequency without counting anything, which is why the admitter this builds
    keeps no state -- ``state()`` stays None and no counter is built for it.

    Parameters
    ----------
    probability : float
        Chance in [0, 1] that one appearance of a missing key admits it.
    initializer_args : Optional[DynamicEmbInitializerArgs]
        How to initialize the rows this rejects. None -- the default -- leaves
        them to the table's own initializer, which is also the only way to get
        bounds derived from a table's row count.

    Notes
    -----
    The draws come from ``torch.rand``, so they follow ``torch.manual_seed``
    and survive CUDA graph capture. A seeded run repeats, but the draws are
    consumed in the order keys go missing, so changing the batch size or the
    data order changes which keys get in.
    """

    def __init__(
        self,
        probability: float,
        initializer_args: Optional[DynamicEmbInitializerArgs] = None,
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        _check_non_admitted_initializer_args(initializer_args)

        self.probability = probability
        self.initializer_args = initializer_args

    def get_grouped_key(self):
        return (
            type(self).__name__,
            self.probability,
            group_key_of(self.initializer_args),
        )

    @classmethod
    def create_admitter(cls, table_strategies, device):
        first = cls.one_configuration(table_strategies)
        return MultiTableProbabilisticAdmitter(
            probability=first.probability,
            non_admitted_initializer=_non_admitted_initializer(
                table_strategies, device
            ),
        )


class MultiTableProbabilisticAdmitter(MultiTableAdmitter):
    """What :class:`ProbabilisticAdmissionStrategy` becomes for one module."""

    def __init__(self, probability: float, non_admitted_initializer=None):
        self._probability = probability
        # log(1 - probability), for compounding a batch's repeats below. Only
        # meaningful strictly inside (0, 1): log1p(-1) has no value, and at
        # either end the answer needs no arithmetic.
        self._log_miss = math.log1p(-probability) if 0.0 < probability < 1.0 else None
        self._non_admitted_initializer = non_admitted_initializer

    def admit(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        frequencies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_keys = keys.shape[0]
        if self._probability <= 0.0:
            return torch.zeros(num_keys, dtype=torch.bool, device=keys.device)
        if self._probability >= 1.0:
            return torch.ones(num_keys, dtype=torch.bool, device=keys.device)

        # torch.rand draws from [0, 1), so the comparison is strict: nothing
        # passes at probability 0 and everything does at 1. (curand_uniform,
        # which the initializers use, is (0, 1] and wants the opposite.)
        draws = torch.rand(num_keys, device=keys.device)
        if frequencies is None:
            return draws < self._probability

        # A key the batch holds k times deserves k tosses. Tossing k times and
        # taking any success is exactly one toss against 1 - (1 - p)^k, so that
        # is what this compares to -- through log1p/expm1, because float32
        # loses a small p outright in 1 - p (at p = 1e-8 it rounds to 1).
        occurrences = frequencies.to(torch.float32).clamp(min=1.0)
        return draws < -torch.expm1(occurrences * self._log_miss)

    @property
    def non_admitted_initializer(self):
        return self._non_admitted_initializer
