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

"""External-storage stand-ins for the tests of the ``Storage`` contract.

``external_storage`` of ``DynamicEmbTableOptions`` is a class that the tables module
builds as ``PS(options, optimizer)``. Which tensors DynamicEmb hands it, and what it
costs the host, depends on the contract the class implements (see ``types.Storage``):

``PyDictStorage``
    the demo of ``test_batched_dynamic_embedding_tables_v2.py``: the host-sized
    contract (``find`` / ``insert``), rows in a Python dict. DynamicEmb serves it by
    reading, once per call, how many rows of the call are real.
``PaddedPyDictStorage``
    the same dict behind the padded contract (``find_masked`` / ``insert_masked``):
    DynamicEmb reads nothing for it. The class itself reads CUDA tensors on the host,
    which is its own cost, not DynamicEmb's.
``DeviceDictStorage``
    a padded-contract store that never touches the host: a direct-address table on the
    device, written only with kernels and index operations. Its key domain is
    ``[0, KEY_SPACE)`` per table. It is the skeleton of the documentation's example
    and what the zero-read assertions run against.
``OldContractDeviceDict``
    the same store behind the host-sized contract, to compare a class before and after
    it adopted the padded one.

``checked(ps_class)`` wraps any of them with the rules of the contract: what is checked
on every call (CUDA tensors, lengths, table ids, uniqueness of the live pairs of an
insert, the key domain) needs host reads, so a wrapped class must not run under
``no_sync()``. It also scribbles over what a padded class promises not to look at (the
table id and the values of an inert row) and records the calls it got.

``no_sync()`` is the one of ``no_host_sync/conftest.py``, loaded by path (two conftest
modules of this tree share a name).
"""
import functools
import importlib.util
import os
from typing import List, Optional, Tuple

import torch
from dynamicemb import sync_free as sf
from dynamicemb.types import CopyMode, PaddedFind, Storage, has_padded_contract

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT_TESTS = os.path.join(HERE, "unit_tests")
NO_HOST_SYNC = os.path.join(UNIT_TESTS, "no_host_sync")


@functools.lru_cache(maxsize=None)
def _conftest():
    spec = importlib.util.spec_from_file_location(
        "no_host_sync_conftest", os.path.join(NO_HOST_SYNC, "conftest.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def no_sync():
    """The context manager that turns a host sync inside it into an error."""
    return _conftest().no_sync()


# ---------------------------------------------------------------------------------------
# the accessors every stand-in shares
# ---------------------------------------------------------------------------------------
class StandInBase(Storage):
    """The accessors of a ``Storage`` for a class built like a PS: ``PS(options,
    optimizer)``. ``find`` / ``insert`` / the checkpoint methods are not implemented."""

    def __init__(self, options, optimizer, device: Optional[torch.device] = None):
        self.options = options
        self.optimizer = optimizer
        self._emb_dims = [o.dim for o in options]
        self._value_dims = [o.dim + optimizer.get_state_dim(o.dim) for o in options]
        self._emb_dtype = options[0].embedding_dtype
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else device
        )
        self._emb_dims_dev = torch.tensor(
            self._emb_dims, dtype=torch.int64, device=self.device
        )
        self._emb_dims_host = torch.tensor(self._emb_dims, dtype=torch.int64)

    def find(self, *args, **kwargs):
        raise NotImplementedError("test stand-in")

    def insert(self, *args, **kwargs):
        raise NotImplementedError("test stand-in")

    def dump(self, *args, **kwargs):
        raise NotImplementedError("test stand-in")

    def load(self, *args, **kwargs):
        raise NotImplementedError("test stand-in")

    def export_keys_values(self, *args, **kwargs):
        raise NotImplementedError("test stand-in")

    def embedding_dtype(self) -> torch.dtype:
        return self._emb_dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._emb_dims[table_id]

    def value_dim(self, table_id: int) -> int:
        return self._value_dims[table_id]

    def max_embedding_dim(self) -> int:
        return max(self._emb_dims)

    def max_value_dim(self) -> int:
        return max(self._value_dims)

    def embedding_dims(self, on_device: bool = False) -> torch.Tensor:
        return self._emb_dims_dev if on_device else self._emb_dims_host

    def all_dims_vec4(self) -> bool:
        return all(d % 4 == 0 for d in self._emb_dims) and all(
            v % 4 == 0 for v in self._value_dims
        )

    def init_optimizer_state(self) -> float:
        return self.optimizer.get_initial_optimizer_state()


# ---------------------------------------------------------------------------------------
# a padded store that never touches the host
# ---------------------------------------------------------------------------------------
class DeviceDictStorage(StandInBase):
    """A padded-contract store with no host code at all: a direct-address table.

    The row of ``(table_id, key)`` is ``table_id * KEY_SPACE + key``, so the keys it can
    hold are ``[0, KEY_SPACE)`` of every table (``checked`` asserts it; a key outside
    would index past the table). An inert row is sent to one extra row, which absorbs
    the stores of every inert lane and is never read.
    """

    accepts_empty_keys = True
    KEY_SPACE = 1 << 16

    def __init__(self, options, optimizer, device: Optional[torch.device] = None):
        super().__init__(options, optimizer, device)
        rows = len(options) * self.KEY_SPACE
        self._values = torch.zeros(
            rows + 1,
            max(self._value_dims),
            dtype=self._emb_dtype,
            device=self.device,
        )
        self._present = torch.zeros(rows + 1, dtype=torch.bool, device=self.device)
        self._dump_row = torch.full((1,), rows, dtype=torch.int64, device=self.device)

    def _slot(self, keys: torch.Tensor, table_ids: torch.Tensor):
        k = sf.as_i64(keys)
        live = (k >= 0) & (table_ids >= 0)  # the reserved key and the padding are inert
        return torch.where(live, table_ids * self.KEY_SPACE + k, self._dump_row), live

    def find_masked(
        self,
        keys,
        table_ids,
        copy_mode,
        lfu_accumulated_frequency=None,
        key_scores=None,
    ) -> PaddedFind:
        slot, live = self._slot(keys, table_ids)
        width = (
            self.max_embedding_dim()
            if copy_mode == CopyMode.EMBEDDING
            else self.max_value_dim()
        )
        return PaddedFind(live & self._present[slot], self._values[slot][:, :width])

    def insert_masked(
        self, keys, table_ids, values, scores=None, preserve_existing=False
    ) -> None:
        slot, _ = self._slot(keys, table_ids)
        if preserve_existing:
            slot = torch.where(self._present[slot], slot, self._dump_row)
        # Duplicates only at the extra row, whose content nobody reads.
        self._values[slot, : values.size(1)] = values.to(self._values.dtype)
        # ``self._present[slot] = True`` would be a blocking host-to-device copy.
        self._present.index_fill_(0, slot, True)
        self._present.index_fill_(0, self._dump_row, False)

    def size(self) -> int:  # tests only: a host read
        return int(self._present.sum())


class OldContractDeviceDict(DeviceDictStorage):
    """The same store behind the host-sized contract: ``find`` / ``insert`` take
    exact-length tensors and ``find`` returns a host count. The class does not
    redeclare ``accepts_empty_keys``, so DynamicEmb serves it through the adapter."""

    def find(
        self,
        keys,
        table_ids,
        copy_mode,
        lfu_accumulated_frequency=None,
        key_scores=None,
    ):
        found, values = DeviceDictStorage.find_masked(
            self, keys, table_ids, copy_mode, lfu_accumulated_frequency, key_scores
        )
        miss = ~found
        positions = torch.arange(keys.numel(), dtype=torch.int64, device=keys.device)
        missing = positions[miss]  # its own host read
        scores = (
            None
            if lfu_accumulated_frequency is None
            else lfu_accumulated_frequency[miss]
        )
        return (
            int(missing.numel()),
            keys[miss],
            missing,
            table_ids[miss],
            scores,
            found,
            torch.zeros(keys.numel(), dtype=torch.int64, device=keys.device),
            values,
        )

    def insert(self, keys, table_ids, values, scores=None, preserve_existing=False):
        DeviceDictStorage.insert_masked(
            self, keys, table_ids, values, scores, preserve_existing
        )


# ---------------------------------------------------------------------------------------
# the dict demo behind the padded contract
# ---------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def padded_py_dict_storage():
    """``PaddedPyDictStorage``: the demo ``PyDictStorage`` behind the padded contract.

    A function, because the demo lives in a test module that needs fbgemm."""
    import sys

    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from test_batched_dynamic_embedding_tables_v2 import PyDictStorage

    class PaddedPyDictStorage(PyDictStorage):
        accepts_empty_keys = True

        def find_masked(
            self,
            keys,
            table_ids,
            copy_mode,
            lfu_accumulated_frequency=None,
            key_scores=None,
        ) -> PaddedFind:
            width = (
                self._max_emb_dim
                if copy_mode == CopyMode.EMBEDDING
                else self._max_value_dim
            )
            host_keys = sf.as_i64(keys).cpu().tolist()
            host_ids = table_ids.cpu().tolist()
            values = torch.zeros(len(host_keys), width, dtype=self._emb_dtype)
            found = [False] * len(host_keys)
            for i, (key, tid) in enumerate(zip(host_keys, host_ids)):
                if key == sf.EMPTY_KEY:
                    continue  # inert: its table id is not looked at
                row = self.dict.get((tid, key))
                if row is not None:
                    values[i] = row[:width].cpu()
                    found[i] = True
            return PaddedFind(
                torch.tensor(found, dtype=torch.bool, device=self.device),
                values.to(self.device),
            )

        def insert_masked(
            self, keys, table_ids, values, scores=None, preserve_existing=False
        ) -> None:
            host_keys = sf.as_i64(keys).cpu().tolist()
            host_ids = table_ids.cpu().tolist()
            host_scores = None if scores is None else scores.cpu().tolist()
            for i, (key, tid) in enumerate(zip(host_keys, host_ids)):
                if key == sf.EMPTY_KEY:
                    continue
                if preserve_existing and (tid, key) not in self.dict:
                    continue
                self.dict[(tid, key)] = values[i, :].clone()
                if host_scores is not None:
                    self.scores[(tid, key)] = host_scores[i]

    return PaddedPyDictStorage


# ---------------------------------------------------------------------------------------
# the checker
# ---------------------------------------------------------------------------------------
def _live_pairs(keys: torch.Tensor, table_ids: torch.Tensor) -> torch.Tensor:
    k = sf.as_i64(keys)
    live = k != sf.EMPTY_KEY
    return torch.stack([k[live], table_ids[live]], dim=1)


def check_padded_call(ps, keys, table_ids, values=None, *, unique: bool) -> None:
    """The rules of a padded call that a host read can check (test only)."""
    assert keys.is_cuda and table_ids.is_cuda, "the tensors live on the device"
    assert keys.dim() == 1 and table_ids.shape == keys.shape
    assert table_ids.dtype == torch.int64
    assert values is None or values.size(0) == keys.numel(), "values are N-aligned"
    num_tables = len(ps.options)
    live = sf.as_i64(keys) != sf.EMPTY_KEY
    assert bool((table_ids >= -1).all()), "a table id is -1 or a table"
    assert bool(
        ((table_ids[live] >= 0) & (table_ids[live] < num_tables)).all()
    ), "a live row has a real table"
    domain = getattr(ps, "KEY_SPACE", None)
    if domain is not None and bool(live.any()):
        assert int(sf.as_i64(keys)[live].max()) < domain, "key outside the domain"
    if unique:
        pairs = _live_pairs(keys, table_ids)
        assert torch.unique(pairs, dim=0).size(0) == pairs.size(
            0
        ), "live (key, table) pairs are unique within an insert"


def check_old_call(ps, keys, table_ids, values=None) -> None:
    """The rules of a host-sized call: exact lengths, nothing inert in it."""
    assert keys.is_cuda and table_ids.is_cuda and keys.dim() == 1
    assert table_ids.shape == keys.shape and table_ids.dtype == torch.int64
    assert values is None or values.size(0) == keys.numel()
    if keys.numel():
        assert bool(
            (sf.as_i64(keys) != sf.EMPTY_KEY).all()
        ), "a host-sized call has no padding"
        assert bool(
            ((table_ids >= 0) & (table_ids < len(ps.options))).all()
        ), "a host-sized call has real tables only"


def checked(ps_class, scribble: bool = True):
    """``ps_class`` that checks every call it gets against the contract it implements,
    records the calls (``self.calls``: ``(method, number of live rows)``), and, for a
    padded class with ``scribble``, replaces the table id and the values of every inert
    row by garbage first, to see that they are not looked at."""
    padded = has_padded_contract(ps_class)

    def scribbled(keys, table_ids, values=None):
        if not scribble:
            return table_ids, values
        inert = sf.as_i64(keys) == sf.EMPTY_KEY
        garbage = torch.full_like(table_ids, 7)
        table_ids = torch.where(inert, garbage, table_ids)
        if values is not None and values.is_floating_point():
            values = torch.where(
                inert.unsqueeze(1), torch.full_like(values, 1e30), values
            )
        return table_ids, values

    def init(self, *args, **kwargs):
        ps_class.__init__(self, *args, **kwargs)
        self.calls: List[Tuple[str, int]] = []

    def n_live(keys):
        return int((sf.as_i64(keys) != sf.EMPTY_KEY).sum())

    namespace = {"accepts_empty_keys": padded, "__init__": init}
    if padded:

        def find_masked(
            self,
            keys,
            table_ids,
            copy_mode,
            lfu_accumulated_frequency=None,
            key_scores=None,
        ):
            check_padded_call(self, keys, table_ids, unique=False)
            self.calls.append(("find_masked", n_live(keys)))
            table_ids, _ = scribbled(keys, table_ids)
            found, values = ps_class.find_masked(
                self, keys, table_ids, copy_mode, lfu_accumulated_frequency, key_scores
            )
            width = (
                self.max_embedding_dim()
                if copy_mode == CopyMode.EMBEDDING
                else self.max_value_dim()
            )
            assert found.dtype == torch.bool and found.shape == keys.shape
            assert not bool(
                found[sf.as_i64(keys) == sf.EMPTY_KEY].any()
            ), "inert: not found"
            assert values.dim() == 2 and values.size(0) == keys.numel()
            assert values.size(1) == width and values.dtype == self.embedding_dtype()
            assert values.device == keys.device
            return PaddedFind(found, values)

        def insert_masked(
            self, keys, table_ids, values, scores=None, preserve_existing=False
        ):
            check_padded_call(self, keys, table_ids, values, unique=True)
            self.calls.append(("insert_masked", n_live(keys)))
            table_ids, values = scribbled(keys, table_ids, values)
            return ps_class.insert_masked(
                self, keys, table_ids, values, scores, preserve_existing
            )

        namespace.update(find_masked=find_masked, insert_masked=insert_masked)
    else:

        def find(
            self, keys, table_ids, copy_mode, lfu_accumulated_frequency=None, **kwargs
        ):
            check_old_call(self, keys, table_ids)
            self.calls.append(("find", int(keys.numel())))
            return ps_class.find(
                self, keys, table_ids, copy_mode, lfu_accumulated_frequency, **kwargs
            )

        def insert(self, keys, table_ids, values, scores=None, preserve_existing=False):
            check_old_call(self, keys, table_ids, values)
            self.calls.append(("insert", int(keys.numel())))
            return ps_class.insert(
                self, keys, table_ids, values, scores, preserve_existing
            )

        namespace.update(find=find, insert=insert)
    return type("Checked" + ps_class.__name__, (ps_class,), namespace)
