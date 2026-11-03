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

"""Which shardings of an EmbeddingCollection need frequency counters.

The EC counts frequency per sharding, so ``collect_frequency_requirements``
answers per sharding type from *all* the tables of that sharding. It used to read
the first table's options only, which starved an LFU or admitting table that was
not first.
"""

from types import SimpleNamespace

import pytest
from dynamicemb import DynamicEmbScoreStrategy as S
from dynamicemb.shard.embedding import collect_frequency_requirements

ADMIT = object()  # only its presence matters, never its type

RW = "row_wise"
TW = "table_wise"


def _table(sharding_type, score_strategy=S.TIMESTAMP, admit_strategy=None):
    return SimpleNamespace(
        sharding_type=sharding_type,
        dynamicemb_options=SimpleNamespace(
            score_strategy=score_strategy, admit_strategy=admit_strategy
        ),
    )


def test_no_table_needs_frequency():
    params = {"a": _table(RW), "b": _table(RW, S.CUSTOMIZED), "c": _table(TW, S.STEP)}
    assert collect_frequency_requirements(params) == {
        RW: (False, False),
        TW: (False, False),
    }


@pytest.mark.parametrize("lfu_first", [True, False])
def test_lfu_table_is_found_wherever_it_is(lfu_first):
    # Reading only the first table made the answer depend on the order.
    tables = [_table(RW, S.LFU), _table(RW, S.TIMESTAMP)]
    params = dict(zip("ab", tables if lfu_first else tables[::-1]))
    assert collect_frequency_requirements(params) == {RW: (True, False)}


@pytest.mark.parametrize("strategy", [(S.TIMESTAMP, S.LFU), (S.LFU, S.TIMESTAMP)])
def test_compound_strategy_with_lfu(strategy):
    params = {"a": _table(RW), "b": _table(RW, strategy)}
    assert collect_frequency_requirements(params) == {RW: (True, False)}


@pytest.mark.parametrize(
    "strategy",
    [S.TIMESTAMP, S.STEP, S.CUSTOMIZED, S.NO_EVICTION, (S.NO_EVICTION, S.CUSTOMIZED)],
)
def test_other_strategies_do_not_need_lfu(strategy):
    assert collect_frequency_requirements({"a": _table(RW, strategy)}) == {
        RW: (False, False)
    }


@pytest.mark.parametrize("admit_first", [True, False])
def test_admitting_table_is_found_wherever_it_is(admit_first):
    tables = [_table(RW, admit_strategy=ADMIT), _table(RW)]
    params = dict(zip("ab", tables if admit_first else tables[::-1]))
    assert collect_frequency_requirements(params) == {RW: (False, True)}


def test_lfu_and_admit_are_independent():
    params = {"a": _table(RW, S.LFU), "b": _table(RW, admit_strategy=ADMIT)}
    assert collect_frequency_requirements(params) == {RW: (True, True)}


def test_answer_is_per_sharding_type():
    # The LFU table needs counters on its own sharding only; the other sharding
    # must not be told to count.
    params = {
        "a": _table(RW, S.LFU),
        "b": _table(TW, S.TIMESTAMP),
        "c": _table(TW, S.CUSTOMIZED),
    }
    assert collect_frequency_requirements(params) == {
        RW: (True, False),
        TW: (False, False),
    }


def test_tables_without_dynamicemb_options_are_skipped():
    plain = SimpleNamespace(sharding_type=TW)  # a plain TorchRec table
    empty = SimpleNamespace(sharding_type=TW, dynamicemb_options=None)
    params = {"a": plain, "b": empty, "c": _table(RW, S.LFU)}
    # A sharding type with no DynamicEmb table is simply absent.
    assert collect_frequency_requirements(params) == {RW: (True, False)}


def test_score_strategy_none_is_not_lfu():
    params = {"a": _table(RW, score_strategy=None)}
    assert collect_frequency_requirements(params) == {RW: (False, False)}
