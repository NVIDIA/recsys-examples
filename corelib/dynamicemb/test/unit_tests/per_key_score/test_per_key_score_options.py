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

"""``per_key_score``: what a table may combine it with, and what a collection
must agree on.

A table's own requirements are checked where the option is built; what only the
whole collection can show -- every DynamicEmb table sets it or none does, and all
of them are row-wise sharded -- is checked by ``collect_per_key_score``.
"""

from types import SimpleNamespace

import pytest
import torch
from dynamicemb import (
    DynamicEmbScoreStrategy as S,
    DynamicEmbTableOptions,
    ProbabilisticAdmissionStrategy,
    ScoredKeyedJaggedTensor,
)
from dynamicemb.shard.per_key_score import collect_per_key_score, require_scores
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

RW = "row_wise"
TW = "table_wise"


def _options(**kwargs) -> DynamicEmbTableOptions:
    kwargs.setdefault("score_strategy", S.CUSTOMIZED)
    return DynamicEmbTableOptions(**kwargs)


@pytest.mark.parametrize(
    "score_strategy",
    [S.CUSTOMIZED, (S.NO_EVICTION, S.CUSTOMIZED), (S.CUSTOMIZED, S.NO_EVICTION)],
)
def test_customized_may_take_a_score_per_key(score_strategy):
    assert _options(per_key_score=True, score_strategy=score_strategy).per_key_score


@pytest.mark.parametrize(
    "score_strategy",
    [
        S.TIMESTAMP,
        S.STEP,
        S.LFU,
        (S.TIMESTAMP, S.LFU),
        (S.LFU, S.TIMESTAMP),
        S.NO_EVICTION,  # keeps no score at all
        None,
    ],
)
def test_other_strategies_cannot_take_a_score_per_key(score_strategy):
    with pytest.raises(ValueError, match="needs score_strategy CUSTOMIZED"):
        _options(per_key_score=True, score_strategy=score_strategy)


def test_cannot_be_combined_with_admission():
    with pytest.raises(ValueError, match="admit_strategy"):
        _options(
            per_key_score=True, admit_strategy=ProbabilisticAdmissionStrategy(0.5)
        )


def test_tables_differing_in_it_are_not_grouped():
    on, off = _options(per_key_score=True), _options(per_key_score=False)
    assert on.get_grouped_key() != off.get_grouped_key()
    assert on != off
    assert on == _options(per_key_score=True)


def _table(sharding_type=RW, **option_kwargs):
    return SimpleNamespace(
        sharding_type=sharding_type, dynamicemb_options=_options(**option_kwargs)
    )


def _plain_table(sharding_type=RW):
    return SimpleNamespace(sharding_type=sharding_type)  # no dynamicemb_options


def test_a_collection_with_no_dynamicemb_table_takes_none():
    assert collect_per_key_score({}) is False
    assert collect_per_key_score({"a": _plain_table()}) is False
    assert collect_per_key_score({"a": _plain_table(), "b": _table()}) is False


def test_a_collection_where_none_sets_it():
    params = {"a": _table(), "b": _table(sharding_type=TW)}
    assert collect_per_key_score(params) is False


def test_a_collection_where_every_table_sets_it():
    params = {"a": _table(per_key_score=True), "b": _table(per_key_score=True)}
    assert collect_per_key_score(params) is True


def test_a_collection_where_only_some_set_it_is_refused():
    params = {
        "emb_on_0": _table(per_key_score=True),
        "emb_off": _table(),
        "emb_on_1": _table(per_key_score=True),
    }
    with pytest.raises(ValueError) as error:
        collect_per_key_score(params)
    # It names the tables, so the one to fix can be found.
    for name in params:
        assert name in str(error.value)


@pytest.mark.parametrize("which", ["first", "last"])
def test_the_refusal_does_not_depend_on_the_order(which):
    tables = [_table(per_key_score=True), _table()]
    params = dict(zip("ab", tables if which == "first" else tables[::-1]))
    with pytest.raises(ValueError):
        collect_per_key_score(params)


def test_only_row_wise_tables_may_take_a_score_per_key():
    params = {
        "rw_table": _table(per_key_score=True),
        "tw_table": _table(sharding_type=TW, per_key_score=True),
    }
    with pytest.raises(NotImplementedError) as error:
        collect_per_key_score(params)
    # It names the table that is not row-wise, and only that one.
    assert "tw_table" in str(error.value)
    assert "rw_table" not in str(error.value)


def test_a_collection_with_per_key_score_holds_dynamicemb_tables_only():
    params = {"a": _table(per_key_score=True), "plain": _plain_table()}
    with pytest.raises(ValueError) as error:
        collect_per_key_score(params)
    # It names the table that is not DynamicEmb's, so the one to move can be found.
    assert "plain" in str(error.value)


def _kjt(lengths=(1, 2), **kwargs):
    # Two features, a batch of one, three keys; unless ``lengths`` says otherwise.
    return dict(
        keys=["f0", "f1"],
        values=torch.tensor([3, 1, 2], dtype=torch.int64),
        lengths=torch.tensor(lengths, dtype=torch.int64),
        **kwargs,
    )


def _scored(**kwargs) -> ScoredKeyedJaggedTensor:
    return ScoredKeyedJaggedTensor(scores=torch.arange(3, dtype=torch.int64), **kwargs)


def test_a_scored_input_is_accepted():
    require_scores(_scored(**_kjt()))


def test_a_plain_input_is_refused_rather_than_read_as_no_scores():
    with pytest.raises(ValueError, match="ScoredKeyedJaggedTensor"):
        require_scores(KeyedJaggedTensor(**_kjt()))


def test_an_input_that_lost_its_scores_in_an_operation_is_refused():
    # concat() is one of the operations that hands back a plain KJT.
    scored = _scored(**_kjt())
    require_scores(scored)
    concatenated = KeyedJaggedTensor.concat([scored])
    assert not isinstance(concatenated, ScoredKeyedJaggedTensor)
    with pytest.raises(ValueError):
        require_scores(concatenated)


def test_a_variable_batch_per_key_is_refused():
    scored = _scored(**_kjt(lengths=(1, 1, 1), stride_per_key_per_rank=[[1], [2]]))
    assert scored.variable_stride_per_key()
    with pytest.raises(NotImplementedError, match="variable batch"):
        require_scores(scored)
