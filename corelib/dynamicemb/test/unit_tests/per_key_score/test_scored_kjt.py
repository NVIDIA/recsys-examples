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

"""``ScoredKeyedJaggedTensor`` keeps one score per key through the operations a
batch goes through before it reaches a module.

Every score is derived from its key (``score = key * 10 + 1``), so after any
operation ``scores == values * 10 + 1`` says each score is still beside its key,
whichever way the keys were reordered or cut.
"""

import pytest
import torch
from dynamicemb import ScoredKeyedJaggedTensor
from dynamicemb.scored_kjt import get_scores
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

DEVICE = "cuda"


def _scores_of(values: torch.Tensor) -> torch.Tensor:
    return values * 10 + 1


def _make(with_weights: bool = False) -> ScoredKeyedJaggedTensor:
    # Four features, a batch of two; lengths vary, one bag is empty.
    keys = ["f0", "f1", "f2", "f3"]
    lengths = torch.tensor([2, 0, 1, 3, 2, 2, 0, 1], dtype=torch.int64, device=DEVICE)
    values = torch.arange(100, 100 + int(lengths.sum()), dtype=torch.int64, device=DEVICE)
    weights = (
        torch.arange(values.numel(), dtype=torch.float32, device=DEVICE)
        if with_weights
        else None
    )
    return ScoredKeyedJaggedTensor(
        keys=keys,
        values=values,
        weights=weights,
        lengths=lengths,
        scores=_scores_of(values),
    )


def _assert_aligned(kjt, expect_weights: bool = False):
    assert isinstance(kjt, ScoredKeyedJaggedTensor)
    assert torch.equal(kjt.scores(), _scores_of(kjt.values()))
    assert (kjt.weights_or_none() is not None) == expect_weights


def test_is_a_keyed_jagged_tensor():
    kjt = _make()
    assert isinstance(kjt, KeyedJaggedTensor)
    assert kjt.keys() == ["f0", "f1", "f2", "f3"]
    _assert_aligned(kjt)


def test_get_scores_is_none_for_a_plain_kjt():
    plain = KeyedJaggedTensor(
        keys=["f"],
        values=torch.tensor([1, 2], dtype=torch.int64, device=DEVICE),
        lengths=torch.tensor([2], dtype=torch.int64, device=DEVICE),
    )
    assert get_scores(plain) is None
    scored = _make()
    assert get_scores(scored) is scored.scores()


def test_from_kjt_shares_the_tensors():
    plain = _make()  # any KJT will do
    plain = KeyedJaggedTensor(
        keys=plain.keys(), values=plain.values(), lengths=plain.lengths()
    )
    scored = ScoredKeyedJaggedTensor.from_kjt(plain, _scores_of(plain.values()))
    assert scored.values().data_ptr() == plain.values().data_ptr()
    _assert_aligned(scored)


@pytest.mark.parametrize(
    "scores, message",
    [
        (lambda v: v.int(), "must be int64"),
        (lambda v: v.float(), "must be int64"),
        (lambda v: v.view(1, -1), "must be 1-D"),
        (lambda v: v[:-1], "must equal the number of values"),
        (lambda v: v.cpu(), "must be on the values' device"),
    ],
)
def test_rejects_malformed_scores(scores, message):
    values = torch.arange(6, dtype=torch.int64, device=DEVICE)
    with pytest.raises(ValueError, match=message):
        ScoredKeyedJaggedTensor(
            keys=["f"],
            values=values,
            lengths=torch.tensor([6], dtype=torch.int64, device=DEVICE),
            scores=scores(values),
        )


@pytest.mark.parametrize("with_weights", [False, True])
def test_to_the_same_device_keeps_the_scores(with_weights):
    # TorchRec's to() builds a new KeyedJaggedTensor even when nothing moves, so
    # this is the call that used to drop an attribute.
    moved = _make(with_weights).to(torch.device(DEVICE))
    _assert_aligned(moved, with_weights)


def test_to_cpu_and_back():
    cpu = _make().to(torch.device("cpu"))
    assert cpu.scores().device.type == "cpu"
    _assert_aligned(cpu)
    _assert_aligned(cpu.to(torch.device(DEVICE)))


def test_pin_memory_keeps_the_scores():
    pinned = _make().to(torch.device("cpu")).pin_memory()
    assert pinned.scores().is_pinned()
    _assert_aligned(pinned)


def test_record_stream_covers_the_scores():
    # Nothing to compare; it must at least accept the call and leave the data.
    kjt = _make()
    kjt.record_stream(torch.cuda.current_stream())
    _assert_aligned(kjt)


@pytest.mark.parametrize("with_weights", [False, True])
@pytest.mark.parametrize("segments", [[4], [2, 2], [1, 3], [3, 1], [1, 1, 2], [0, 4]])
def test_split_keeps_each_score_beside_its_key(segments, with_weights):
    kjt = _make(with_weights)
    pieces = kjt.split(segments)
    assert len(pieces) == len(segments)
    key_names = []
    for piece in pieces:
        _assert_aligned(piece, with_weights)
        key_names.extend(piece.keys())
    assert key_names == kjt.keys()
    assert torch.equal(
        torch.cat([p.values() for p in pieces]),
        kjt.values(),
    )


@pytest.mark.parametrize("with_weights", [False, True])
@pytest.mark.parametrize(
    "order", [[0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1], [1, 3], [0, 0, 2]]
)
def test_permute_keeps_each_score_beside_its_key(order, with_weights):
    kjt = _make(with_weights)
    permuted = kjt.permute(order)
    _assert_aligned(permuted, with_weights)
    assert permuted.keys() == [kjt.keys()[i] for i in order]


def test_permute_with_a_tensor_of_indices():
    kjt = _make()
    order = [2, 0, 3, 1]
    permuted = kjt.permute(order, torch.tensor(order, dtype=torch.int32, device=DEVICE))
    _assert_aligned(permuted)


def test_weights_stay_independent_of_scores():
    kjt = _make(with_weights=True)
    expected = kjt.permute([3, 1, 0, 2]).weights()
    plain = KeyedJaggedTensor(
        keys=kjt.keys(),
        values=kjt.values(),
        weights=kjt.weights(),
        lengths=kjt.lengths(),
    )
    assert torch.equal(kjt.permute([3, 1, 0, 2]).weights(), plain.permute([3, 1, 0, 2]).weights())
    assert torch.equal(kjt.permute([3, 1, 0, 2]).weights(), expected)


def test_permute_rejects_a_variable_stride_per_key():
    values = torch.arange(3, dtype=torch.int64, device=DEVICE)
    kjt = ScoredKeyedJaggedTensor(
        keys=["f0", "f1"],
        values=values,
        lengths=torch.tensor([1, 1, 1], dtype=torch.int64, device=DEVICE),
        stride_per_key_per_rank=[[1], [2]],
        scores=_scores_of(values),
    )
    assert kjt.variable_stride_per_key()
    with pytest.raises(NotImplementedError, match="variable stride"):
        kjt.permute([1, 0])


def test_permute_with_empty_values():
    kjt = ScoredKeyedJaggedTensor(
        keys=["f0", "f1"],
        values=torch.empty(0, dtype=torch.int64, device=DEVICE),
        lengths=torch.zeros(4, dtype=torch.int64, device=DEVICE),
        scores=torch.empty(0, dtype=torch.int64, device=DEVICE),
    )
    permuted = kjt.permute([1, 0])
    assert permuted.scores().numel() == 0
    assert permuted.keys() == ["f1", "f0"]
