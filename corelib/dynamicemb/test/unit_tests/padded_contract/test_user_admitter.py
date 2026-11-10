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

"""User-written admitters, counters and non-admitted-row initializers.

The padded bodies call ``admitter.admit_masked`` over arrays that keep the length of the
batch. An admitter that declared ``accepts_empty_keys = True`` in its own class body (and holds
only what declared it) is called with those arrays and costs nothing. Any other is promised the
missing keys only, dense and in order, as it always was: the number of live keys is read once per
call (``types._live_rows``, by its contract), ``admit`` is called with exactly those keys and the
answer is scattered back; with no live key ``admit`` is not called at all. The same holds for the
counter a frequency admitter holds (``erase`` takes the admitted keys compacted) and for a
``non_admitted_initializer`` (``skips_negative_index`` in its own body, else exact index lists).

The logic of the adapters runs on CPU tensors too; the flows through real tables need a GPU.
"""
import os
import sys

import pytest

try:
    import torch
    from dynamicemb import embedding_admission as adm
    from dynamicemb import sync_free as sf
    from dynamicemb.initializer import ConstantInitializer
    from dynamicemb.types import Counter, MultiTableAdmitter
except ImportError as e:  # pragma: no cover
    pytest.skip("needs torch and dynamicemb: %s" % (e,), allow_module_level=True)

TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA device")
DEVICES = ["cpu"] + (["cuda"] if HAS_CUDA else [])
EMPTY = -1


class AdmitOdd(MultiTableAdmitter):
    """Written against the dense contract: it must never see padding."""

    def __init__(self):
        self.calls = []

    def admit(self, keys, table_ids, frequencies=None):
        k = sf.as_i64(keys)
        assert bool((k != EMPTY).all()) and bool(
            (table_ids >= 0).all()
        ), "padding reached an admitter that did not declare accepts_empty_keys"
        if frequencies is not None:
            assert frequencies.shape == keys.shape
        self.calls.append(keys.numel())
        return k % 2 == 1


class AdmitOddPadded(MultiTableAdmitter):
    """The same decision, written for whole-batch arrays: it answers True for the reserved key too
    (``-1 % 2 == 1``), which the module ignores."""

    accepts_empty_keys = True

    def admit(self, keys, table_ids, frequencies=None):
        return sf.as_i64(keys) % 2 == 1


class AdmitOddSubclass(AdmitOddPadded):
    """Overrides ``admit`` without declaring the promise itself: not inherited, so it is dense."""

    def admit(self, keys, table_ids, frequencies=None):
        assert bool((sf.as_i64(keys) != EMPTY).all())
        return super().admit(keys, table_ids, frequencies)


class _Reads:
    def __init__(self):
        self.start = sf.READ_NOW_CALLS

    @property
    def delta(self):
        return sf.READ_NOW_CALLS - self.start


def _padded(device, live=(10, 11, 12, 13, 14, 15)):
    keys = torch.tensor(list(live) + [EMPTY] * 4, dtype=torch.int64, device=device)
    ids = torch.tensor([0] * len(live) + [-1] * 4, dtype=torch.int64, device=device)
    return keys, ids


# ---------------------------------------------------------------------------------------
# who qualifies
# ---------------------------------------------------------------------------------------
def test_an_admitter_qualifies_by_its_own_declaration_and_what_it_holds():
    class Held(Counter):
        accepts_empty_keys = True

        def add(self, keys, table_ids, frequencies):
            return frequencies

        def erase(self, keys, table_ids):
            pass

        def memory_usage(self, mem_type=None):
            return 0

        def load(self, *a, **k):
            pass

        def dump(self, *a, **k):
            pass

    class HeldDense(Held):  # inherits the flag, does not declare it
        pass

    class WithState(AdmitOddPadded):
        accepts_empty_keys = True

        def __init__(self, counter):
            self.counter = counter

        def state(self):
            return self.counter

    assert adm.admitter_accepts_empty_keys(None) is True
    assert adm.admitter_accepts_empty_keys(AdmitOddPadded()) is True
    assert adm.admitter_accepts_empty_keys(AdmitOdd()) is False
    assert (
        adm.admitter_accepts_empty_keys(AdmitOddSubclass()) is False
    ), "an inherited flag does not count"
    assert adm.admitter_accepts_empty_keys(WithState(Held())) is True
    assert (
        adm.admitter_accepts_empty_keys(WithState(HeldDense())) is False
    ), "its state did not declare it"
    assert adm.admitter_accepts_empty_keys(WithState(None)) is True
    freq = adm.MultiTableFrequencyAdmitter(2, Held.__new__(Held))
    assert adm.admitter_accepts_empty_keys(freq) is True

    class MyFrequency(adm.MultiTableFrequencyAdmitter):
        def admit(self, *a, **k):
            return super().admit(*a, **k)

    assert adm.admitter_accepts_empty_keys(MyFrequency(2, Held.__new__(Held))) is False
    assert (
        adm.admitter_accepts_empty_keys(adm.MultiTableProbabilisticAdmitter(0.5))
        is True
    )


def test_the_in_tree_initializers_declare_skips_negative_index_and_a_subclass_does_not_inherit_it():
    class Mine(ConstantInitializer):
        def __call__(self, buffer, keys, table_ids, indices):
            raise AssertionError

    class MineDeclared(Mine):
        skips_negative_index = True

    from dynamicemb.initializer import _INITIALIZERS

    assert all(adm.has_skips_negative_index(c) for c in _INITIALIZERS.values())
    assert adm.initializer_needs_dense_rows(None) is False
    assert (
        adm.initializer_needs_dense_rows(
            ConstantInitializer.__new__(ConstantInitializer)
        )
        is False
    )
    assert adm.initializer_needs_dense_rows(Mine.__new__(Mine)) is True
    assert adm.initializer_needs_dense_rows(MineDeclared.__new__(MineDeclared)) is False


# ---------------------------------------------------------------------------------------
# the dense adapter of an admitter
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_a_dense_admitter_is_called_with_the_live_keys_and_its_answer_scattered_back(
    device,
):
    keys, ids = _padded(device)
    admitter = AdmitOdd()
    freq = torch.arange(100, 110, device=device)
    reads = _Reads()

    mask = admitter.admit_masked(keys, ids, freq)

    assert reads.delta == 1, "the number of live keys, once per call"
    assert admitter.calls == [6]
    assert mask.tolist() == [False, True, False, True, False, True] + [False] * 4


@pytest.mark.parametrize("device", DEVICES)
def test_a_padded_admitter_is_called_as_it_is_and_reads_nothing(device):
    keys, ids = _padded(device)
    reads = _Reads()

    mask = AdmitOddPadded().admit_masked(keys, ids)

    assert reads.delta == 0
    # it answers for the padding too: the module ANDs it with its own mask
    assert mask.tolist() == [False, True, False, True, False, True] + [True] * 4


@pytest.mark.parametrize("device", DEVICES)
def test_a_dense_admitter_is_not_called_when_no_key_is_asked_about(device):
    keys = torch.full((6,), EMPTY, dtype=torch.int64, device=device)
    ids = torch.full((6,), -1, dtype=torch.int64, device=device)
    admitter = AdmitOdd()
    reads = _Reads()

    mask = admitter.admit_masked(keys, ids)

    assert admitter.calls == [] and not bool(mask.any()) and reads.delta == 1


@pytest.mark.parametrize("device", DEVICES)
def test_a_subclass_that_overrides_admit_is_served_dense(device):
    keys, ids = _padded(device)
    mask = AdmitOddSubclass().admit_masked(keys, ids)  # asserts it never sees padding
    assert mask.tolist() == [False, True, False, True, False, True] + [False] * 4


# ---------------------------------------------------------------------------------------
# the non-admitted-row initializer
# ---------------------------------------------------------------------------------------
class DenseInit:
    """Written for exact index lists: a negative index would be an out-of-bounds write."""

    def __init__(self):
        self.calls = []

    def __call__(self, buffer, keys, table_ids, indices):
        assert bool(
            (indices >= 0).all()
        ), "a negative index reached a dense initializer"
        self.calls.append(indices.tolist())
        for r in indices.tolist():
            buffer[r] = -7.0


@pytest.mark.parametrize("device", DEVICES)
def test_a_dense_initializer_gets_exact_indices_after_one_read(device):
    n = 8
    mask = torch.tensor([0, 1, 0, 0, 1, 1, 0, 0], dtype=torch.bool, device=device)
    keys = torch.arange(n, device=device)
    ids = torch.zeros(n, dtype=torch.int64, device=device)
    out = torch.zeros(n, 3, device=device)
    init = DenseInit()
    reads = _Reads()

    sf.initialize_masked(init, out, keys, ids, mask, dense=True)

    assert reads.delta == 1 and init.calls == [[1, 4, 5]]
    assert out[:, 0].tolist() == [0, -7, 0, 0, -7, -7, 0, 0]
    # nothing to initialize: not called
    sf.initialize_masked(init, out, keys, ids, torch.zeros_like(mask), dense=True)
    assert len(init.calls) == 1


@pytest.mark.parametrize("device", DEVICES)
def test_a_declared_initializer_is_called_natively(monkeypatch, device):
    """``skips_negative_index``: the call is the one of ``initialize_masked`` always, which
    marks the rows to skip with a negative index (and reads nothing)."""
    monkeypatch.setattr(sf, "INIT_SKIPS_NEGATIVE", True)
    seen = []

    def native(buffer, keys, table_ids, indices):
        seen.append(indices.tolist())

    mask = torch.tensor([0, 1, 0, 1], dtype=torch.bool, device=device)
    out = torch.zeros(4, 2, device=device)
    reads = _Reads()

    sf.initialize_masked(
        native,
        out,
        torch.arange(4, device=device),
        torch.zeros(4, dtype=torch.int64, device=device),
        mask,
    )

    assert reads.delta == 0 and seen == [[-1, 1, -1, 3]]


# ---------------------------------------------------------------------------------------
# the counter of a frequency admitter
# ---------------------------------------------------------------------------------------
class RecordingCounter(Counter):
    """Adds every key once and records what ``erase`` is given."""

    def __init__(self):
        self.erased = []

    def add(self, keys, table_ids, frequencies):
        return frequencies.to(torch.int64) * 10

    def erase(self, keys, table_ids):
        self.erased.append(keys.clone())

    def memory_usage(self, mem_type=None):
        return 0

    def load(self, *a, **k):
        pass

    def dump(self, *a, **k):
        pass


class RecordingCounterDeclared(RecordingCounter):
    accepts_empty_keys = True


@pytest.mark.parametrize("device", DEVICES)
def test_a_counter_that_declared_the_contract_erases_over_the_whole_batch(device):
    keys, ids = _padded(device)
    counter = RecordingCounterDeclared()
    admitter = adm.MultiTableFrequencyAdmitter(threshold=20, counter=counter)
    freq = torch.tensor([1, 2, 3, 1, 2, 3, 1, 1, 1, 1], device=device)
    reads = _Reads()

    mask = admitter.admit_masked(keys, ids, freq)

    assert reads.delta == 0
    assert mask[:6].tolist() == [False, True, True, False, True, True]
    (erased,) = counter.erased
    assert (
        erased.shape == keys.shape
    ), "the keys that stay are the reserved key: no compaction"
    assert erased.tolist()[:6] == [EMPTY, 11, 12, EMPTY, 14, 15]


@cuda_only
def test_a_counter_that_did_not_declare_it_is_given_dense_keys_only():
    device = torch.device("cuda", torch.cuda.current_device())
    keys, ids = _padded(device)
    counter = RecordingCounter()
    admitter = adm.MultiTableFrequencyAdmitter(threshold=20, counter=counter)
    freq = torch.tensor([1, 2, 3, 1, 2, 3, 1, 1, 1, 1], device=device)

    mask = admitter.admit_masked(keys, ids, freq)

    # one read of how many keys the admitter is asked about, and the admitted keys erased dense
    assert mask[:6].tolist() == [False, True, True, False, True, True]
    (erased,) = counter.erased
    assert erased.tolist() == [11, 12, 14, 15]


# ---------------------------------------------------------------------------------------
# through the lookup paths (real tables)
# ---------------------------------------------------------------------------------------
@cuda_only
def test_the_cache_path_gives_a_dense_and_a_padded_admitter_the_same_tables():
    import ps_standins
    from dynamicemb import batched_dynamicemb_function as bdf
    from dynamicemb.key_value_table import (
        DynamicEmbCache,
        DynamicEmbStorage,
        load_from_flat,
    )
    from dynamicemb.types import CopyMode

    device = torch.device("cuda", torch.cuda.current_device())
    results = {}
    for name, admitter, expected_reads in (
        ("dense", AdmitOdd(), 1),
        ("padded", AdmitOddPadded(), 0),
    ):
        cache = DynamicEmbCache(
            ps_standins.table_options(capacity=2048, bucket=1024, hbm=True),
            ps_standins.sgd_optimizer(),
        )
        storage = DynamicEmbStorage(
            ps_standins.table_options(capacity=256, bucket=128),
            ps_standins.sgd_optimizer(),
            use_ref_counter=False,
        )
        initializer = ps_standins.constant_initializer(device, 3.0)
        batch = list(range(10, 30))
        keys = torch.tensor(batch + [EMPTY] * 6, dtype=torch.int64, device=device)
        ids = torch.tensor([0] * 20 + [EMPTY] * 6, dtype=torch.int64, device=device)
        reads = _Reads()

        slots, refused = bdf._prefetch_cache_path_masked(
            cache,
            storage,
            keys,
            ids,
            ids >= 0,
            8,
            8,
            torch.float32,
            initializer,
            None,
            admitter,
        )

        assert reads.delta == expected_reads, name
        odd = [k % 2 == 1 for k in batch]
        assert refused[:20].tolist() == [
            not o for o in odd
        ], "the even keys are refused"
        assert (slots[:20] >= 0).tolist() == odd, "only the odd ones got a slot"
        rows = load_from_flat(cache._state, slots, ids, CopyMode.EMBEDDING)
        assert (rows[:20][torch.tensor(odd)] == 3.0).all()
        results[name] = (slots.tolist(), refused.tolist())
    assert results["dense"][1] == results["padded"][1]
    assert [s >= 0 for s in results["dense"][0]] == [
        s >= 0 for s in results["padded"][0]
    ]


@cuda_only
def test_the_generic_body_serves_a_dense_admitter_and_its_own_initializer_for_the_refused_rows():
    import ps_standins
    from dynamicemb import batched_dynamicemb_function as bdf
    from dynamicemb.key_value_table import DynamicEmbStorage

    device = torch.device("cuda", torch.cuda.current_device())

    class OddWithInit(AdmitOdd):
        init = DenseInit()

        @property
        def non_admitted_initializer(self):
            return self.init

    admitter = OddWithInit()
    storage = DynamicEmbStorage(
        ps_standins.table_options(capacity=256, bucket=128),
        ps_standins.sgd_optimizer(),
        use_ref_counter=False,
    )
    storage.training = True
    batch = list(range(10, 30))
    keys = torch.tensor(batch + [EMPTY] * 4, dtype=torch.int64, device=device)
    ids = torch.tensor([0] * 20 + [EMPTY] * 4, dtype=torch.int64, device=device)
    reads = _Reads()

    values, persisted = bdf._generic_forward_path_masked(
        storage,
        ps_standins.sgd_optimizer(),
        keys,
        ids,
        ids >= 0,
        8,
        8,
        torch.float32,
        ps_standins.constant_initializer(device, 3.0),
        None,
        admitter,
    )

    # the admit call (live keys) and the initializer of the refused rows (live rows): one read each
    assert reads.delta == 2
    assert admitter.calls == [20] and OddWithInit.init.calls == [
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    ]
    odd = torch.tensor([k % 2 == 1 for k in batch] + [False] * 4, device=device)
    assert (
        values[:20][odd[:20]] == 3.0
    ).all(), "the admitted rows are the table's initializer's"
    assert (
        values[:20][~odd[:20]] == -7.0
    ).all(), "the refused rows are the admitter's own initializer's"
    assert persisted[:20].tolist() == odd[:20].tolist()
