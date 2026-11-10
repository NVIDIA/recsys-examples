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

"""Which participants the padded bodies accept, decided by capability only.

One body serves every lookup group; what the participants are decides how they are talked to
(``bdf._participants``, ``types.as_padded``) and never which body runs:

* the cache and the storage of the cache path and the HBM-direct path must be the in-tree classes
  (the bodies read their table state), or a subclass that declares ``accepts_empty_keys = True`` in
  its own class body; anything else is a ``TypeError`` that says so, not a silent fall back to
  another body;
* the DEFAULT path takes any storage: one that declared the padded contract is called as it is, any
  other (a class written against ``find`` / ``insert``, a subclass of an in-tree storage that did
  not redeclare) through an :class:`OldContractStorage` that reads the live rows once per call;
* the declaration is read from the class's own ``__dict__``: a subclass of a class that declared
  it has not, so it has to say it itself;
* eval takes a ``DynamicEmbCache`` (or such a subclass) as its cache, and any storage.

The checks are on types, so they need neither a GPU nor a table: the instances are made without
running their constructors.
"""
import pytest

try:
    import torch
    from dynamicemb import batched_dynamicemb_function as bdf
    from dynamicemb import key_value_table as kvt
    from dynamicemb.batched_dynamicemb_function import StorageMode
    from dynamicemb.key_value_table import DynamicEmbCache, DynamicEmbStorage
    from dynamicemb.types import OldContractStorage, as_padded, has_padded_contract
except ImportError as e:  # pragma: no cover
    pytest.skip("needs torch and dynamicemb: %s" % (e,), allow_module_level=True)


def _bare(cls):
    """An instance of ``cls`` whose constructor did not run: enough for a check of its type."""
    return object.__new__(cls)


class _SubStorage(DynamicEmbStorage):
    """Overrides nothing and declares nothing: has not made the promise."""


class _DeclaringStorage(DynamicEmbStorage):
    accepts_empty_keys = True


class _SubOfDeclaring(_DeclaringStorage):
    """A subclass of a class that declared it: the declaration is not inherited."""


class _SubCache(DynamicEmbCache):
    pass


class _DeclaringCache(DynamicEmbCache):
    accepts_empty_keys = True


class _OldPS:
    """A parameter server written against the host-sized contract (``find`` / ``insert``); not even
    a ``Storage`` subclass."""

    def find(self, keys, table_ids, copy_mode, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def insert(self, keys, table_ids, values, **kwargs):  # pragma: no cover
        raise NotImplementedError


class _PaddedPS(_OldPS):
    accepts_empty_keys = True


def test_the_declaration_is_read_from_the_own_class_body():
    assert has_padded_contract(_bare(DynamicEmbStorage))
    assert has_padded_contract(_bare(DynamicEmbCache))
    assert not has_padded_contract(_bare(_SubStorage))
    assert has_padded_contract(_bare(_DeclaringStorage))
    assert not has_padded_contract(_bare(_SubOfDeclaring)), "not inherited"
    assert has_padded_contract(_PaddedPS) and not has_padded_contract(_OldPS)


def test_the_cache_path_takes_the_in_tree_cache_and_storage():
    cache, storage = _bare(DynamicEmbCache), _bare(DynamicEmbStorage)

    got_storage, next_tier = bdf._participants(StorageMode.CACHE, cache, storage, None)

    assert got_storage is storage and next_tier is None


@pytest.mark.parametrize(
    "cache, storage, role",
    [
        (_SubCache, DynamicEmbStorage, "cache"),
        (DynamicEmbCache, _SubStorage, "storage"),
        (DynamicEmbCache, _SubOfDeclaring, "storage"),
        (_SubCache, _SubStorage, "cache"),  # the cache is looked at first
    ],
)
def test_the_cache_path_refuses_a_subclass_that_did_not_declare_the_contract(
    cache, storage, role
):
    with pytest.raises(TypeError) as err:
        bdf._participants(StorageMode.CACHE, _bare(cache), _bare(storage), None)

    message = str(err.value)
    assert "CACHE lookup group" in message and "accepts_empty_keys = True" in message
    assert message.startswith("the %s of" % role), message


def test_the_cache_path_takes_a_subclass_that_declares_it():
    cache, storage = _bare(_DeclaringCache), _bare(_DeclaringStorage)

    got_storage, _ = bdf._participants(StorageMode.CACHE, cache, storage, None)

    assert got_storage is storage


def test_the_cache_path_refuses_what_is_not_a_cache_at_all():
    with pytest.raises(TypeError, match="must be a DynamicEmbCache"):
        bdf._participants(StorageMode.CACHE, object(), _bare(DynamicEmbStorage), None)
    with pytest.raises(TypeError, match="must be a DynamicEmbCache"):
        bdf._participants(StorageMode.CACHE, None, _bare(DynamicEmbStorage), None)


def test_the_tier_below_the_storage_is_served_by_what_it_declared():
    cache, storage = _bare(DynamicEmbCache), _bare(DynamicEmbStorage)
    old, padded = _OldPS(), _PaddedPS()

    _, below_old = bdf._participants(StorageMode.CACHE, cache, storage, old)
    _, below_padded = bdf._participants(StorageMode.CACHE, cache, storage, padded)

    assert isinstance(below_old, OldContractStorage) and below_old.inner is old
    assert below_padded is padded, "a declared class is called as it is"


def test_the_hbm_direct_path_takes_the_in_tree_storage_and_has_no_tier_below():
    storage = _bare(DynamicEmbStorage)
    assert bdf._participants(StorageMode.HBM_DIRECT, None, storage, None) == (
        storage,
        None,
    )
    # a PS only exists below a cache or as the only tier: the path never hands it a tier
    assert bdf._participants(StorageMode.HBM_DIRECT, None, storage, _OldPS())[1] is None
    declaring = _bare(_DeclaringStorage)
    assert (
        bdf._participants(StorageMode.HBM_DIRECT, None, declaring, None)[0] is declaring
    )
    for cls in (_SubStorage, _SubOfDeclaring):
        with pytest.raises(TypeError, match="HBM_DIRECT lookup group"):
            bdf._participants(StorageMode.HBM_DIRECT, None, _bare(cls), None)


def test_the_default_path_takes_any_storage_and_adapts_what_did_not_declare():
    for declared in (_bare(DynamicEmbStorage), _bare(_DeclaringStorage), _PaddedPS()):
        got, below = bdf._participants(StorageMode.DEFAULT, None, declared, None)
        assert got is declared and below is None
    for undeclared in (_bare(_SubStorage), _bare(_SubOfDeclaring), _OldPS()):
        got, below = bdf._participants(StorageMode.DEFAULT, None, undeclared, None)
        assert isinstance(got, OldContractStorage) and got.inner is undeclared
        assert below is None


def test_as_padded_wraps_any_object_that_has_find_and_insert():
    old = _OldPS()
    view = as_padded(old)
    assert isinstance(view, OldContractStorage) and view.inner is old
    assert view.accepts_empty_keys, "the view itself speaks the padded contract"
    assert as_padded(_PaddedPS()).__class__ is _PaddedPS


class _Marker:
    """What a stubbed body returns, so that a test sees which one the dispatch reached."""

    def __init__(self, name, args):
        self.name, self.args = name, args


@pytest.fixture
def eval_bodies(monkeypatch):
    monkeypatch.setattr(
        kvt, "_eval_lookup_storage", lambda *a, **k: _Marker("storage", a)
    )
    monkeypatch.setattr(
        kvt, "_eval_lookup_cached", lambda *a, **k: _Marker("cached", a)
    )


KEYS = torch.tensor([3, 4], dtype=torch.int64)
IDS = torch.zeros(2, dtype=torch.int64)


def test_eval_takes_a_dynamicemb_cache_or_a_subclass_that_declares_the_contract(
    eval_bodies,
):
    storage = _bare(DynamicEmbStorage)
    for cache in (_bare(DynamicEmbCache), _bare(_DeclaringCache)):
        got = kvt.eval_lookup(storage, KEYS, IDS, None, cache=cache)
        assert got.name == "cached" and got.args[0] is cache
    assert kvt.eval_lookup(storage, KEYS, IDS, None).name == "storage"


@pytest.mark.parametrize("cache", [object(), _bare(_SubCache)])
def test_eval_refuses_a_cache_that_is_not_one(eval_bodies, cache):
    with pytest.raises(
        TypeError, match="eval_lookup reads the table state of its cache"
    ):
        kvt.eval_lookup(_bare(DynamicEmbStorage), KEYS, IDS, None, cache=cache)


def test_eval_serves_any_storage_through_the_contract_it_declared(eval_bodies):
    for undeclared in (_bare(_SubStorage), _OldPS()):
        got = kvt.eval_lookup(undeclared, KEYS, IDS, None)
        assert isinstance(got.args[0], OldContractStorage)
        assert got.args[0].inner is undeclared
    declared = _PaddedPS()
    assert kvt.eval_lookup(declared, KEYS, IDS, None).args[0] is declared
