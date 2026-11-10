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

"""The padded ``Storage`` contract and the adapter that serves a class without it.

``Storage.find_masked`` / ``insert_masked`` take tensors of the whole batch's length whose
inert rows (the reserved key) are no-ops, with no count anywhere. A class that declares
``accepts_empty_keys = True`` in its own body is called through them and DynamicEmb reads
nothing for it; any other class is served by ``OldContractStorage``, which reads how many
rows are live once per call, compacts the tensors to that length, calls the class's own
``find`` / ``insert`` and scatters the answer back. The adapter logic runs on CPU tensors
too (``read_now`` passes them through); the stand-ins that live on the device, and the
in-tree storages, need a GPU.
"""
import os
import sys
import warnings
from types import SimpleNamespace

import pytest

try:
    import torch
    from dynamicemb import sync_free as sf
    from dynamicemb.types import (
        CopyMode,
        MaskedFind,
        OldContractStorage,
        PaddedFind,
        as_padded,
        has_padded_contract,
    )
except ImportError as e:  # pragma: no cover
    pytest.skip("needs torch and dynamicemb: %s" % (e,), allow_module_level=True)

TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)
import ps_standins  # noqa: E402
from ps_standins import StandInBase  # noqa: E402

HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA device")
DEVICES = ["cpu"] + (["cuda"] if HAS_CUDA else [])
KEY_DTYPES = [torch.int64, torch.uint64]
EMPTY = -1
DIM = 4


class _NoState:
    def get_state_dim(self, dim):
        return 0

    def get_initial_optimizer_state(self):
        return 0.0


def _options(dim=DIM, tables=1):
    return [
        SimpleNamespace(dim=dim, embedding_dtype=torch.float32) for _ in range(tables)
    ]


def keys_of(values, dtype, device):
    """int64 keys as ``dtype``; -1 is the reserved key either way (all ones)."""
    t = torch.tensor(values, dtype=torch.int64, device=device)
    return t if dtype == torch.int64 else t.view(torch.uint64)


def key_list(keys):
    t = keys.view(torch.int64) if keys.dtype == torch.uint64 else keys
    return t.tolist()


class RecordingOld(StandInBase):
    """A store written against the host-sized contract: exact-length ``find`` / ``insert``
    over a Python dict. It records every call: the keys and table ids it got and the names
    of the optional arguments that were passed."""

    def __init__(self, device, dim=DIM, tables=1):
        super().__init__(_options(dim, tables), _NoState(), device)
        self.dim = dim
        self.dict = {}
        self.calls = []

    def find(self, keys, table_ids, copy_mode, **kwargs):
        k, t = key_list(keys), table_ids.tolist()
        self.calls.append(("find", k, t, tuple(sorted(kwargs))))
        values = torch.zeros(len(k), self.dim, device=keys.device)
        found, missing = [], []
        for i, (key, tid) in enumerate(zip(k, t)):
            row = self.dict.get((tid, key))
            found.append(row is not None)
            if row is None:
                missing.append(i)
            else:
                values[i] = row.to(keys.device)
        where = torch.tensor(missing, dtype=torch.int64, device=keys.device)
        freq = kwargs.get("lfu_accumulated_frequency")
        return (
            len(missing),
            keys[where],
            where,
            table_ids[where],
            None if freq is None or not missing else freq[where],
            torch.tensor(found, dtype=torch.bool, device=keys.device),
            None,
            values,
        )

    def insert(self, keys, table_ids, values, *args, **kwargs):
        k, t = key_list(keys), table_ids.tolist()
        self.calls.append(("insert", k, t, len(args), tuple(sorted(kwargs))))
        for i, (key, tid) in enumerate(zip(k, t)):
            self.dict[(tid, key)] = values[i].clone().cpu()

    def put(self, keys, tid=0):
        for key in keys:
            self.dict[(tid, key)] = torch.full((self.dim,), float(key))


def _batch(device, dtype=torch.int64):
    """Twelve rows: live keys 10, 11, 12, 13, 14 (tables 0 and 1) among inert ones."""
    keys = [10, EMPTY, 11, 12, EMPTY, EMPTY, 13, 14, EMPTY, EMPTY, EMPTY, EMPTY]
    ids = [0, -1, 0, 1, -1, -1, 1, 0, -1, 3, -1, -1]  # an inert row may carry any id
    return keys_of(keys, dtype, device), torch.tensor(ids, device=device)


def _values(n, device):
    return torch.arange(n * DIM, dtype=torch.float32, device=device).reshape(n, DIM)


class _Reads:
    """How many ``sf.read_now`` calls ran since the object was made."""

    def __init__(self):
        self.start = sf.READ_NOW_CALLS

    @property
    def delta(self):
        return sf.READ_NOW_CALLS - self.start


# ---------------------------------------------------------------------------------------
# the adapter
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", KEY_DTYPES)
def test_an_old_contract_class_is_called_with_the_live_rows_only(device, dtype):
    store = RecordingOld(device, tables=2)
    keys, ids = _batch(device, dtype)
    values = _values(12, device)
    reads = _Reads()

    store.insert_masked(keys, ids, values)

    assert reads.delta == 1, "one read of the number of live rows per call"
    (name, got_keys, got_ids, n_args, kw) = store.calls[0]
    assert name == "insert" and got_keys == [10, 11, 12, 13, 14]  # ascending positions
    assert got_ids == [0, 0, 1, 1, 0]
    assert n_args == 0 and kw == (), "no scores and no flag: neither is passed"
    live = [0, 2, 3, 6, 7]
    for key, row in zip(got_keys, live):
        tid = ids[row].item()
        assert torch.equal(store.dict[(tid, key)].to(device), values[row])


@pytest.mark.parametrize("device", DEVICES)
def test_scores_and_the_preserve_flag_are_passed_only_when_given(device):
    store = RecordingOld(device, tables=2)
    keys, ids = _batch(device)
    values = _values(12, device)
    scores = torch.arange(100, 112, dtype=torch.int64, device=device)

    store.insert_masked(keys, ids, values, scores)
    store.insert_masked(keys, ids, values, preserve_existing=True)
    store.insert_masked(keys, ids, values, scores, preserve_existing=True)

    assert [c[3:] for c in store.calls] == [
        (1, ()),
        (0, ("preserve_existing",)),
        (1, ("preserve_existing",)),
    ]


@pytest.mark.parametrize("device", DEVICES)
def test_no_live_row_means_no_call_into_the_class(device):
    store = RecordingOld(device)
    keys = keys_of([EMPTY] * 6, torch.int64, device)
    ids = torch.full((6,), -1, dtype=torch.int64, device=device)
    reads = _Reads()

    store.insert_masked(keys, ids, _values(6, device))
    found = store.find_masked(keys, ids, CopyMode.VALUE)

    assert store.calls == [] and reads.delta == 2
    assert not bool(found.founds.any()) and found.values.shape == (6, DIM)
    # an empty batch needs no read at all
    empty = keys_of([], torch.int64, device)
    store.insert_masked(empty, empty.clone(), _values(0, device))
    assert reads.delta == 2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", KEY_DTYPES)
def test_the_answer_is_scattered_back_to_the_rows_that_asked(device, dtype):
    store = RecordingOld(device, tables=2)
    store.put([10, 12], tid=0)  # (0, 10), (0, 12): 12 is in table 0 only
    store.put([13], tid=1)
    keys, ids = _batch(device, dtype)

    found = store.find_masked(keys, ids, CopyMode.VALUE)

    # live rows: 0:(0,10) 2:(0,11) 3:(1,12) 6:(1,13) 7:(0,14)
    assert found.founds.tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert found.values.shape == (12, DIM) and found.values.dtype == torch.float32
    assert torch.equal(found.values[0].cpu(), torch.full((DIM,), 10.0))
    assert torch.equal(found.values[6].cpu(), torch.full((DIM,), 13.0))
    assert [c[1] for c in store.calls] == [[10, 11, 12, 13, 14]]


@pytest.mark.parametrize("device", DEVICES)
def test_the_lookup_and_the_load_are_two_steps_and_the_load_only_writes_found_lanes(
    device,
):
    store = RecordingOld(device)
    store.put([10, 13])
    keys, ids = _batch(device)
    ids = ids.clamp(max=0)
    mf = as_padded(store).lookup_masked(keys, ids)
    assert isinstance(mf, MaskedFind)
    out = torch.full((12, DIM), -5.0, device=device)

    result = as_padded(store).load_masked(mf, ids, CopyMode.VALUE, output=out)

    assert result is out
    expected = torch.full((12, DIM), -5.0)
    expected[0] = 10.0
    expected[6] = 13.0
    assert torch.equal(
        out.cpu(), expected
    ), "only the lanes that were found are written"
    # a column slice of a wider block is a fine destination: the other columns stay
    block = torch.full((12, DIM + 3), -7.0, device=device)
    as_padded(store).load_masked(mf, ids, CopyMode.VALUE, output=block[:, :DIM])
    expected[expected == -5.0] = -7.0
    assert torch.equal(block[:, :DIM].cpu(), expected)
    assert (block[:, DIM:] == -7.0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_a_class_that_returns_missing_scores_gets_them_back_n_aligned(device):
    """The scores a later insert takes are the class's own ``missing_scores``, as the
    host-sized body handed them on; a class that returned none gets none."""
    store = RecordingOld(device, tables=2)
    store.put([10])
    keys, ids = _batch(device)
    freq = torch.arange(1000, 1012, dtype=torch.int64, device=device)

    with_scores = as_padded(store).lookup_masked(keys, ids, freq)
    without = as_padded(store).lookup_masked(keys, ids)

    # missing live rows are 2, 3, 6, 7: the class's own missing_scores are their frequencies
    expected = torch.zeros(12, dtype=torch.int64)
    for row in (2, 3, 6, 7):
        expected[row] = 1000 + row
    assert torch.equal(with_scores.score_in.cpu(), expected)
    assert without.score_in is None
    assert store.calls[0][3] == ("lfu_accumulated_frequency",)
    assert store.calls[1][3] == ()


@pytest.mark.parametrize("device", DEVICES)
def test_the_view_forwards_every_other_attribute(device):
    store = RecordingOld(device, dim=6)
    view = OldContractStorage(store)
    assert view.inner is store and has_padded_contract(view)
    assert view.max_value_dim() == 6 and view.embedding_dtype() == torch.float32
    assert view.options is store.options
    assert as_padded(view) is view, "a view is not wrapped again"
    with pytest.raises(AttributeError):
        OldContractStorage.__new__(OldContractStorage).inner  # noqa: B018


@pytest.mark.parametrize("device", DEVICES)
def test_a_duck_typed_tier_without_the_storage_base_is_served(device):
    """The tier below a storage may be any object with an ``insert`` (it was before)."""

    class Tier:
        def __init__(self):
            self.calls = []

        def insert(self, keys, table_ids, values, scores=None, preserve_existing=False):
            self.calls.append((keys.tolist(), table_ids.tolist(), values.clone()))

    tier = Tier()
    keys, ids = _batch(device)
    values = _values(12, device)

    as_padded(tier).insert_masked(keys, ids, values)

    assert tier.calls[0][0] == [10, 11, 12, 13, 14]
    assert torch.equal(tier.calls[0][2], values[[0, 2, 3, 6, 7]])


# ---------------------------------------------------------------------------------------
# who is served how
# ---------------------------------------------------------------------------------------
def test_the_capability_is_the_class_own_declaration_never_an_inherited_one():
    class Native(StandInBase):
        accepts_empty_keys = True

        def find_masked(self, keys, table_ids, copy_mode, *a, **k):
            return PaddedFind(torch.zeros(keys.numel(), dtype=torch.bool), keys)

        def insert_masked(self, *a, **k):
            pass

    class Derived(Native):
        def find(self, *a, **k):  # overrides what an old caller would use
            raise AssertionError

    class Redeclared(Derived):
        accepts_empty_keys = True

    class Plain(StandInBase):
        pass

    assert has_padded_contract(Native) is True
    assert (
        has_padded_contract(Derived) is False
    ), "inherited: not the promise of this class"
    assert has_padded_contract(Redeclared) is True
    assert has_padded_contract(Plain) is False
    assert isinstance(as_padded(Plain.__new__(Plain)), OldContractStorage)
    native = Native.__new__(Native)
    assert as_padded(native) is native


@pytest.mark.parametrize("device", DEVICES)
def test_a_subclass_without_its_own_flag_is_served_through_its_find(device):
    """The subclass may override ``find``: the adapter calls it, the padded methods of
    the parent are not trusted."""

    class Parent(RecordingOld):
        accepts_empty_keys = True

        def find_masked(self, *a, **k):
            raise AssertionError("the parent's padded method must not be called")

        def insert_masked(self, *a, **k):
            raise AssertionError("the parent's padded method must not be called")

    class Child(Parent):
        pass

    store = Child(device)
    store.put([10, 13])
    keys, ids = _batch(device)
    found = as_padded(store).find_masked(keys, ids.clamp(max=0), CopyMode.VALUE)
    assert found.founds.tolist()[0] is True and len(store.calls) == 1


def test_defining_the_padded_methods_without_the_flag_warns_once_at_class_creation():
    with pytest.warns(UserWarning, match="accepts_empty_keys"):

        class Forgot(StandInBase):
            def find_masked(self, *a, **k):
                raise AssertionError

    assert has_padded_contract(Forgot) is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class Declared(StandInBase):
            accepts_empty_keys = True

            def find_masked(self, *a, **k):
                raise AssertionError

            def insert_masked(self, *a, **k):
                raise AssertionError

        class OnlyOld(StandInBase):
            def find(self, *a, **k):
                raise AssertionError


class _PaddedFindOnly(StandInBase):
    """A padded class that implements only ``find_masked``: the protocol the lookup code
    uses (``lookup_masked`` then ``load_masked``) is built on it by default."""

    accepts_empty_keys = True

    def __init__(self, device, dim=DIM):
        super().__init__(_options(dim), _NoState(), device)
        self.calls = []

    def find_masked(
        self,
        keys,
        table_ids,
        copy_mode,
        lfu_accumulated_frequency=None,
        key_scores=None,
    ):
        self.calls.append(copy_mode)
        width = (
            self.max_embedding_dim()
            if copy_mode == CopyMode.EMBEDDING
            else self.max_value_dim()
        )
        live = sf.as_i64(keys) != EMPTY
        values = torch.arange(1, keys.numel() + 1, device=keys.device).float()
        found = live & (sf.as_i64(keys) % 2 == 0)
        return (found, values.unsqueeze(1).repeat(1, width))  # a plain tuple


@pytest.mark.parametrize("device", DEVICES)
def test_the_default_lookup_and_load_are_built_on_find_masked(device):
    store = _PaddedFindOnly(device)
    keys, ids = _batch(device)
    freq = torch.arange(12, device=device)

    mf = store.lookup_masked(keys, ids, freq, copy_mode=CopyMode.VALUE)

    assert isinstance(mf, MaskedFind) and mf.score_in is freq
    live_even = [i for i, k in enumerate(key_list(keys)) if k != EMPTY and k % 2 == 0]
    assert mf.founds.nonzero().flatten().tolist() == live_even
    out = torch.full((12, DIM), -1.0, device=device)
    store.load_masked(mf, ids, CopyMode.VALUE, output=out)
    for row in range(12):
        expected = float(row + 1) if row in live_even else -1.0
        assert (out[row] == expected).all(), row
    # the same lookup serves the load of the embedding columns, without a second find
    embedding = store.load_masked(mf, ids, CopyMode.EMBEDDING)
    assert embedding.shape == (12, DIM)
    assert store.calls == [CopyMode.VALUE]


class _WrongFind(_PaddedFindOnly):
    """A padded class whose ``find_masked`` answers wrongly in one way (``mode``)."""

    # its own declaration: a subclass has to make the promise itself
    accepts_empty_keys = True

    def __init__(self, device, mode):
        super().__init__(device)
        self.mode = mode

    def find_masked(self, keys, table_ids, copy_mode, *args, **kwargs):
        found, values = super().find_masked(keys, table_ids, copy_mode, *args, **kwargs)
        mode = self.mode
        if mode == "uint8":
            found = found.to(torch.uint8)
        elif mode == "one_per_live_key":
            found = found[found]
        elif mode == "list":
            found = found.tolist()
        elif mode == "narrow":
            values = values[:, :-1]
        elif mode == "one_row_short":
            values = values[:-1]
        elif mode == "wider":
            values = torch.cat([values, values], dim=1)
        return found, values


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "mode, complaint",
    [
        ("uint8", "founds"),
        ("one_per_live_key", "founds"),
        ("list", "founds"),
        ("narrow", "values"),
        ("one_row_short", "values"),
    ],
)
def test_a_padded_result_that_does_not_line_up_with_the_keys_is_refused(
    device, mode, complaint
):
    """Host facts only, no read: such a result would be matched with the wrong rows, silently, by
    the load that follows."""
    store = _WrongFind(device, mode)
    keys, ids = _batch(device)
    reads = _Reads()

    with pytest.raises(ValueError, match=complaint) as error:
        store.lookup_masked(keys, ids, copy_mode=CopyMode.VALUE)

    assert "_WrongFind.find_masked" in str(error.value)
    assert reads.delta == 0


@pytest.mark.parametrize("device", DEVICES)
def test_a_padded_result_with_more_columns_than_asked_is_accepted(device):
    store = _WrongFind(device, "wider")
    keys, ids = _batch(device)

    mf = store.lookup_masked(keys, ids, copy_mode=CopyMode.VALUE)
    out = store.load_masked(mf, ids, CopyMode.VALUE)

    assert out.shape == (12, DIM)


# ---------------------------------------------------------------------------------------
# the stand-ins and the in-tree storages (GPU)
# ---------------------------------------------------------------------------------------
def _device_options(hbm=False):
    from dynamicemb.dynamicemb_config import DynamicEmbTableOptions

    return [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=torch.cuda.current_device(),
            dim=DIM,
            max_capacity=1024,
            init_capacity=1024,
            bucket_capacity=128,
            local_hbm_for_values=1024**3 if hbm else 0,
        )
        for _ in range(2)
    ]


def _sgd():
    from dynamicemb.optimizer import OptimizerArgs, SGDDynamicEmbeddingOptimizer

    return SGDDynamicEmbeddingOptimizer(OptimizerArgs())


def _random_padded_batch(generator, n, device, key_space=60000):
    keys = torch.randint(0, key_space, (n,), generator=generator).unique()
    keys = keys[torch.randperm(keys.numel(), generator=generator)]
    ids = torch.randint(0, 2, (keys.numel(),), generator=generator)
    # unique (key, table) pairs; 30% of the rows are made inert, their ids are garbage
    inert = torch.rand(keys.numel(), generator=generator) < 0.3
    keys = torch.where(inert, torch.full_like(keys, EMPTY), keys)
    ids = torch.where(inert, torch.full_like(ids, -1), ids)
    return keys.to(device), ids.to(device)


@cuda_only
def test_a_device_resident_store_costs_the_host_nothing_and_its_adapter_one_read():
    options, optimizer = _device_options(), _sgd()
    native = ps_standins.DeviceDictStorage(options, optimizer)
    old = ps_standins.OldContractDeviceDict(options, optimizer)
    device = native.device
    generator = torch.Generator().manual_seed(7)
    for _ in range(4):
        keys, ids = _random_padded_batch(generator, 200, device)
        values = torch.randn(keys.numel(), native.max_value_dim(), device=device)
        probe_keys, probe_ids = _random_padded_batch(generator, 200, device)
        # the probe batch shares keys with the stored one half of the time
        half = min(keys.numel(), probe_keys.numel()) // 2
        probe_keys[:half], probe_ids[:half] = keys[:half], ids[:half]
        torch.cuda.synchronize()

        with ps_standins.no_sync():  # the padded contract: nothing is read
            native.insert_masked(keys, ids, values)
            a = native.find_masked(probe_keys, probe_ids, CopyMode.VALUE)
        reads = _Reads()
        as_padded(old).insert_masked(keys, ids, values)  # through the adapter
        b = as_padded(old).find_masked(probe_keys, probe_ids, CopyMode.VALUE)
        assert reads.delta == 2, "one read per call of a class that has not adopted it"

        assert torch.equal(a.founds, b.founds)
        live = a.founds.unsqueeze(1)
        assert torch.equal(
            torch.where(live, a.values, 0.0), torch.where(live, b.values, 0.0)
        )
    assert native.size() == old.size() > 0


@cuda_only
def test_preserve_existing_refreshes_the_rows_a_store_has_and_adds_none():
    options, optimizer = _device_options(), _sgd()
    store = ps_standins.DeviceDictStorage(options, optimizer)
    device = store.device
    keys = torch.tensor([5, 6, EMPTY], dtype=torch.int64, device=device)
    ids = torch.tensor([0, 0, -1], dtype=torch.int64, device=device)
    store.insert_masked(
        keys[:1], ids[:1], torch.ones(1, store.max_value_dim(), device=device)
    )

    store.insert_masked(
        keys,
        ids,
        torch.full((3, store.max_value_dim()), 2.0, device=device),
        preserve_existing=True,
    )

    found = store.find_masked(keys, ids, CopyMode.VALUE)
    assert found.founds.tolist() == [True, False, False]
    assert (found.values[0] == 2.0).all() and store.size() == 1


@cuda_only
@pytest.mark.parametrize("kind", ["storage", "hybrid"])
def test_the_in_tree_storages_speak_the_padded_contract(kind):
    """``find_masked`` equals ``find`` on founds and values, ``insert_masked`` equals ``insert``,
    and neither reads anything back."""
    from dynamicemb.key_value_table import DynamicEmbStorage, HybridStorage

    options, optimizer = _device_options(), _sgd()
    if kind == "storage":
        a, b = (DynamicEmbStorage(options, optimizer) for _ in range(2))
    else:
        a, b = (
            HybridStorage(_device_options(hbm=True), _device_options(), optimizer)
            for _ in range(2)
        )
    assert has_padded_contract(a) and as_padded(a) is a
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(3)
    keys, ids = _random_padded_batch(generator, 300, device)
    values = torch.randn(keys.numel(), a.max_value_dim(), device=device)

    with ps_standins.no_sync():
        a.insert_masked(keys, ids, values)
    live = sf.valid_keys(keys)
    b.insert(keys[live], ids[live], values[live])
    probe_keys, probe_ids = _random_padded_batch(generator, 300, device)
    probe_keys[:100], probe_ids[:100] = keys[:100], ids[:100]
    torch.cuda.synchronize()

    with ps_standins.no_sync():
        masked = a.find_masked(probe_keys, probe_ids, CopyMode.VALUE)
    pk, pi = probe_keys[sf.valid_keys(probe_keys)], probe_ids[sf.valid_keys(probe_keys)]
    old = b.find(pk, pi, CopyMode.VALUE)
    on = sf.valid_keys(probe_keys)
    assert torch.equal(masked.founds[on], old[5])
    got = masked.values[on]
    assert torch.equal(got[old[5]], old[7][old[5]])
    assert not bool(masked.founds[~on].any())
