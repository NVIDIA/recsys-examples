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

from typing import Dict

from torchrec.distributed.embedding_types import ShardingType
from torchrec.distributed.types import ParameterSharding
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from ..scored_kjt import ScoredKeyedJaggedTensor


def collect_per_key_score(params: Dict[str, ParameterSharding]) -> bool:
    """Whether the tables of one collection take a score for every key.

    A collection takes its scores from one input, so it either gives every
    DynamicEmb table a score per key or none: a table left out would be handed
    scores it never asked for, and one that asked would be starved. A table that
    is not DynamicEmb's has no scores to take, so none may sit in such a
    collection: the lookup then hands scores to every table it holds, with no
    other kind of table to tell apart.

    Per-table requirements (no admission) are checked
    where the option is built, in ``DynamicEmbTableOptions``. What can only be
    seen from the whole collection is checked here:

    - every DynamicEmb table takes a score per key (score strategy
      ``PRECISE_CUSTOMIZED``), or none does;
    - when it is set, every table is a DynamicEmb table and is row-wise sharded,
      the one sharding whose input distribution carries the scores along.

    Raises:
        ValueError: some DynamicEmb tables take a score per key and some do not,
            or tables that are not DynamicEmb's share the collection with ones
            that do.
        NotImplementedError: a table that sets it is not row-wise sharded.
    """
    tables = {
        name: param_sharding
        for name, param_sharding in params.items()
        if getattr(param_sharding, "dynamicemb_options", None)
    }
    enabled = sorted(
        name for name, p in tables.items() if p.dynamicemb_options.takes_score_per_key
    )
    if not enabled:
        return False
    not_dynamicemb = sorted(set(params) - set(tables))
    if not_dynamicemb:
        raise ValueError(
            "A collection whose tables take a score per key (PRECISE_CUSTOMIZED) "
            f"may hold DynamicEmb tables only; {not_dynamicemb} are not. Put them "
            "in a collection of their own."
        )
    disabled = sorted(set(tables) - set(enabled))
    if disabled:
        raise ValueError(
            "Within one collection either every DynamicEmb table sets "
            f"PRECISE_CUSTOMIZED or none does; {enabled} set it and {disabled} do not. "
            "Put the tables that take a score per key in a collection of their own."
        )
    not_row_wise = sorted(
        name
        for name in enabled
        if tables[name].sharding_type != ShardingType.ROW_WISE.value
    )
    if not_row_wise:
        raise NotImplementedError(
            f"PRECISE_CUSTOMIZED is supported for row-wise sharding only; {not_row_wise} "
            "are sharded otherwise."
        )
    return True


def require_scores(features: KeyedJaggedTensor) -> None:
    """Check the input of a collection that takes a score per key.

    The scores come with the input, as a :class:`ScoredKeyedJaggedTensor`. A
    plain ``KeyedJaggedTensor`` is refused instead of being read as "no scores":
    TorchRec's ``to``, ``concat`` and the like hand back a plain one, and a
    collection that went on without scores would score its keys some other way
    without a word.

    Raises:
        ValueError: ``features`` is not a ``ScoredKeyedJaggedTensor``.
        NotImplementedError: ``features`` has a variable batch size per key,
            which the distribution of scores does not handle.
    """
    if not isinstance(features, ScoredKeyedJaggedTensor):
        raise ValueError(
            "This collection takes a score per key (PRECISE_CUSTOMIZED), so its "
            "input must be a ScoredKeyedJaggedTensor; got "
            f"{type(features).__name__}. Build it with "
            "ScoredKeyedJaggedTensor.from_kjt(kjt, scores) after the last "
            "operation that returns a plain KeyedJaggedTensor (e.g. to())."
        )
    if features.variable_stride_per_key():
        raise NotImplementedError(
            "PRECISE_CUSTOMIZED does not support a variable batch size per key "
            "(stride_per_key_per_rank)."
        )
