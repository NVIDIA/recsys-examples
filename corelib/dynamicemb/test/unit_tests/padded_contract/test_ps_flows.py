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

"""An external storage below the cache (CACHING_PS), through the padded cache path.

The cache path asks the storage for what the cache lacks, the tier below it for what storage
lacks, hands storage the keys the tier below had, spills what storage evicts to the tier below
and writes back to it what the cache evicts that storage no longer has. The tier is any
``Storage``: one that declared the padded contract (``DeviceDictStorage``, ``PaddedPyDictStorage``,
the in-tree ``DynamicEmbStorage``) is called through ``find_masked`` / ``insert_masked`` and DynamicEmb
reads nothing for it; one that did not (``PyDictStorage``) is served through its ``find`` / ``insert``
after one read of how many rows each call has.

What is checked, at the level of the path function (real tables, exact expectations) and of the
module (a three-tier module must be transparent: its forward values are those of one table big
enough to hold every key, and the PS must have been used):

* every row of the batch ends up in the cache with its key's value, whichever tier had it, and
  no key is lost to the tiers (it is in storage, or the PS, or the cache);
* the contract the PS is promised (CUDA tensors, table ids of real tables, unique live keys of an
  insert, nothing looked at on an inert row: ``ps_standins.checked`` scribbles over those);
* what DynamicEmb reads: a padded tier, nothing at all (the path runs under ``no_sync()`` for the
  device-resident ones); a host-sized one, one read of the miss count and, when something missed,
  one per call that needs a count; a step that missed nothing calls it not at all;
* the padded path equals the legacy body it replaced (``legacy_oracle``) on the same world.
"""
import os
import sys

import pytest

try:
    import torch
    from dynamicemb import sync_free as sf
except ImportError as e:  # pragma: no cover
    pytest.skip("needs torch and dynamicemb: %s" % (e,), allow_module_level=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

UNIT_TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.dirname(UNIT_TESTS)
NHS = os.path.join(UNIT_TESTS, "no_host_sync")
for path in (TEST_DIR, UNIT_TESTS, NHS):
    if path not in sys.path:
        sys.path.insert(0, path)
import host_sync_probe as hsp  # noqa: E402
import legacy_oracle  # noqa: E402
import ps_standins  # noqa: E402
from dynamicemb import batched_dynamicemb_function as bdf  # noqa: E402
from dynamicemb.key_value_table import (  # noqa: E402
    DynamicEmbCache,
    DynamicEmbStorage,
    load_from_flat,
    store_to_flat,
)
from dynamicemb.types import CopyMode, MultiTableAdmitter, as_padded  # noqa: E402

DIM = 8
INIT = 3.0  # the value of a row the table's initializer makes
EMPTY = -1
# what each tier's copy of a key says: the key plus ...
IN_CACHE, IN_STORAGE, IN_PS = 0.75, 0.5, 0.25

PS_KINDS = ps_standins.PS_KINDS
ps_class = ps_standins.ps_class


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _rows(keys, offset):
    return (keys.to(torch.float32) + offset).unsqueeze(1).repeat(1, DIM)


def _t(values):
    return torch.tensor(values, dtype=torch.int64, device=_device())


class World:
    """A cache over a small storage over a PS, filled with known rows.

    The storage holds ``storage_keys`` (row ``key + IN_STORAGE``), the cache ``cache_keys`` (a
    subset of them, row ``key + IN_CACHE``: the cache's row is the fresher one), the PS
    ``ps_keys`` (row ``key + IN_PS``). The storage is small, so that the keys of a batch push
    others out of it and into the PS."""

    def __init__(self, kind, storage_keys, cache_keys, ps_keys, wrap=None):
        device = _device()
        cls = ps_class(kind)
        if wrap is not None:
            cls = wrap(cls)
        self.cache = DynamicEmbCache(
            ps_standins.table_options(capacity=2048, bucket=1024, hbm=True),
            ps_standins.sgd_optimizer(),
        )
        self.storage = DynamicEmbStorage(
            ps_standins.table_options(capacity=256, bucket=128),
            ps_standins.sgd_optimizer(),
            use_ref_counter=False,
        )
        self.ps = cls(
            ps_standins.table_options(capacity=4096), ps_standins.sgd_optimizer()
        )
        self.initializer = ps_standins.constant_initializer(device, INIT)
        zeros = lambda keys: torch.zeros_like(keys)  # noqa: E731
        keys = _t(storage_keys)
        self.storage.insert(keys, zeros(keys), _rows(keys, IN_STORAGE))
        keys = _t(cache_keys)
        slots = self.cache.insert_and_evict(keys, zeros(keys))[0]
        store_to_flat(self.cache._state, slots, zeros(keys), _rows(keys, IN_CACHE))
        keys = _t(ps_keys)
        as_padded(self.ps).insert_masked(keys, zeros(keys), _rows(keys, IN_PS))
        torch.cuda.synchronize()
        if hasattr(self.ps, "calls"):
            self.ps.calls.clear()

    def prefetch(self, keys, pad=8, admitter=None, **kw):
        n = len(keys)
        unique_keys = _t(list(keys) + [EMPTY] * pad)
        ids = _t([0] * n + [EMPTY] * pad)
        out = bdf._prefetch_cache_path_masked(
            self.cache,
            self.storage,
            unique_keys,
            ids,
            ids >= 0,
            DIM,
            DIM,
            torch.float32,
            self.initializer,
            None,
            admitter,
            None,
            external_storage=self.ps,
            **kw,
        )
        return out, unique_keys, ids

    def cache_rows(self, slots, ids):
        return load_from_flat(self.cache._state, slots, ids, CopyMode.EMBEDDING)

    def in_ps(self, keys):
        keys = _t(keys)
        found = as_padded(self.ps).find_masked(
            keys, torch.zeros_like(keys), CopyMode.VALUE
        )
        return found.founds.tolist(), found.values


def _scenario():
    """Keys per origin: storage holds 0..159 (160 keys of two buckets of 128), the cache the odd
    ones below 40, the PS 1000..1059; the batch asks for some of each and for new keys.
    """
    storage_keys = list(range(160))
    cache_keys = list(range(1, 40, 2))
    ps_keys = list(range(1000, 1060))
    batch = (
        list(range(1, 40, 4))  # in the cache (and storage)
        + list(range(100, 140))  # in storage only
        + list(range(1000, 1030))  # in the PS only
        + list(range(5000, 5150))  # nowhere
    )
    return storage_keys, cache_keys, ps_keys, batch


def _expected_row(key, storage_keys, cache_keys, ps_keys):
    if key in cache_keys:
        return key + IN_CACHE
    if key in storage_keys:
        return key + IN_STORAGE
    if key in ps_keys:
        return key + IN_PS
    return INIT


def _check_world_after(world, batch, slots, ids, storage_keys, cache_keys, ps_keys):
    n = len(batch)
    rows = world.cache_rows(slots, ids)
    assert bool((slots[:n] >= 0).all()), "every key of the batch is in the cache"
    assert bool((slots[n:] == -1).all()), "the padding is in no table"
    for i, key in enumerate(batch):
        want = _expected_row(key, set(storage_keys), set(cache_keys), set(ps_keys))
        assert (rows[i] == want).all(), "key %d: a row of %s, wanted %s" % (
            key,
            rows[i],
            want,
        )
    # no key is lost: the keys the storage held and the PS held are in storage or in the PS
    # (a storage key that the batch asked for lives in the cache too) with their own row
    kept = [k for k in storage_keys + ps_keys if k not in set(batch)]
    in_storage = {}
    for keys, _, _, _ in world.storage.export_keys_values(_device(), 4096, 0):
        for k in keys.tolist():
            in_storage[k] = True
    ps_found, ps_values = world.in_ps(kept)
    for key, in_ps_row, row in zip(kept, ps_found, ps_values):
        assert in_storage.get(key) or in_ps_row, "key %d is lost to every tier" % key
        if in_ps_row and not in_storage.get(key):
            want = key + (IN_STORAGE if key in storage_keys else IN_PS)
            assert (row == want).all(), "the PS holds another row for key %d" % key
    assert any(
        not in_storage.get(k) for k in storage_keys if k not in set(batch)
    ), "the scenario must push storage keys out of storage"


@pytest.mark.parametrize("kind", PS_KINDS)
@pytest.mark.parametrize("chunk", [None, 16])
def test_the_batch_ends_in_the_cache_with_its_values_and_no_key_is_lost(
    kind, chunk, monkeypatch
):
    storage_keys, cache_keys, ps_keys, batch = _scenario()
    world = World(kind, storage_keys, cache_keys, ps_keys)
    monkeypatch.setattr(sf, "CHUNK_ROWS", chunk)

    (slots, refused), _, ids = world.prefetch(batch)

    assert refused is None
    _check_world_after(world, batch, slots, ids, storage_keys, cache_keys, ps_keys)


@pytest.mark.parametrize("kind", PS_KINDS)
def test_a_key_the_ps_has_is_not_asked_of_the_admitter(kind):
    """The tier below the storage had the key, so it was admitted before: the admitter decides
    on the keys found nowhere only (here the odd ones are admitted, the even ones refused and
    given no slot), and a PS key is loaded into the cache whatever its parity."""
    storage_keys, cache_keys, ps_keys, batch = _scenario()
    world = World(kind, storage_keys, cache_keys, ps_keys)
    asked = []

    class Odd(MultiTableAdmitter):
        accepts_empty_keys = True

        def admit(self, keys, table_ids, frequencies=None):
            k = keys.view(torch.int64) if keys.dtype == torch.uint64 else keys
            asked.append(k[k != EMPTY].tolist())
            return k % 2 == 1

    (slots, refused), _, ids = world.prefetch(batch, admitter=Odd())

    assert len(asked) == 1 and len(asked[0]) == 150
    assert all(key >= 5000 for key in asked[0]), "only the new keys were asked about"
    rows = world.cache_rows(slots, ids)
    for i, key in enumerate(batch):
        if key >= 5000 and key % 2 == 0:
            assert int(slots[i]) == -1 and bool(refused[i]), "key %d is refused" % key
            continue
        assert int(slots[i]) >= 0 and not bool(refused[i]), "key %d is cached" % key
        want = (
            INIT
            if key >= 5000
            else _expected_row(key, set(storage_keys), set(cache_keys), set(ps_keys))
        )
        assert (rows[i] == want).all(), "key %d: a row of %s, wanted %s" % (
            key,
            rows[i],
            want,
        )


@pytest.mark.parametrize("kind", ["device", "intree"])
def test_a_device_resident_padded_ps_makes_the_path_read_nothing(kind):
    """The padded contract: not a synchronization, not a read of the device, for the whole miss
    path including the lookup in the PS, the spill and the write-back."""
    storage_keys, cache_keys, ps_keys, batch = _scenario()
    world = World(kind, storage_keys, cache_keys, ps_keys)
    n = len(batch)
    keys, ids = _t(batch + [EMPTY] * 8), _t([0] * n + [EMPTY] * 8)
    valid = ids >= 0
    torch.cuda.synchronize()
    reads = sf.READ_NOW_CALLS

    with ps_standins.no_sync():
        slots, _ = bdf._prefetch_cache_path_masked(
            world.cache,
            world.storage,
            keys,
            ids,
            valid,
            DIM,
            DIM,
            torch.float32,
            world.initializer,
            None,
            None,
            None,
            external_storage=world.ps,
        )

    assert sf.READ_NOW_CALLS == reads
    _check_world_after(world, batch, slots, ids, storage_keys, cache_keys, ps_keys)


def test_a_host_sized_ps_costs_one_read_of_the_miss_count_and_one_per_call_that_needs_a_count():
    storage_keys, cache_keys, ps_keys, batch = _scenario()
    world = World("old", storage_keys, cache_keys, ps_keys, wrap=ps_standins.checked)
    reads = sf.READ_NOW_CALLS

    (slots, _), _, ids = world.prefetch(batch)

    # the miss count (which also counts what the lookup of the PS asks), the spill of what the
    # storage evicted, the write-back of what the cache evicted (nothing: it has room, the call
    # still reads how many rows it has and, finding none, does not call the PS)
    assert sf.READ_NOW_CALLS - reads == 3
    assert [name for name, _ in world.ps.calls] == ["find", "insert"]
    assert world.ps.calls[0] == (
        "find",
        30 + 150,
    ), "asked for what storage lacks, exactly"
    _check_world_after(world, batch, slots, ids, storage_keys, cache_keys, ps_keys)


@pytest.mark.parametrize("kind", ["old", "padded"])
def test_a_step_that_missed_nothing_does_not_call_the_ps(kind):
    storage_keys, cache_keys, ps_keys, _ = _scenario()
    world = World(kind, storage_keys, cache_keys, ps_keys, wrap=ps_standins.checked)
    batch = list(cache_keys)  # every key is in the cache
    reads = sf.READ_NOW_CALLS

    (slots, refused), _, ids = world.prefetch(batch)

    assert refused is None and bool((slots[: len(batch)] >= 0).all())
    if kind == "old":
        assert sf.READ_NOW_CALLS - reads == 1, "the miss count, and nothing else"
        assert world.ps.calls == []
    else:  # no read, and the padded path may call the PS on inert rows
        assert sf.READ_NOW_CALLS - reads == 0
        assert all(n_live == 0 for _, n_live in world.ps.calls)


@pytest.mark.parametrize("kind", ["padded", "device", "intree"])
def test_the_ps_gets_its_calls_in_the_documented_order_and_keeps_to_the_contract(kind):
    """``checked`` asserts the contract on every call (CUDA tensors, real table ids on live rows,
    unique live keys of an insert, the key domain) and scribbles over the table id and the values of
    the inert rows first; the lookup is followed by the spill and then the write-back.
    """
    storage_keys, cache_keys, ps_keys, batch = _scenario()
    world = World(kind, storage_keys, cache_keys, ps_keys, wrap=ps_standins.checked)

    (slots, _), _, ids = world.prefetch(batch)

    names = [name for name, _ in world.ps.calls]
    assert names == ["find_masked", "insert_masked", "insert_masked"]
    spilled = world.ps.calls[1][1]
    assert spilled > 0, "the storage must have evicted something for the PS"
    _check_world_after(world, batch, slots, ids, storage_keys, cache_keys, ps_keys)


def test_the_padded_path_equals_the_legacy_body_it_replaced():
    """The same world twice: the legacy body (exact-length arrays, host counts) on one, the padded
    path (a padded batch) on the other. The rows the batch gets in the cache are equal, and so is
    what the tiers below hold: every key of the world, with the row of its origin.

    Where a key sits (storage or the PS) is not compared. Which victim a full bucket gives up is
    not specified: a table reads the timestamp that scores a key while its kernel runs, so two
    worlds built by the same calls rank their old keys differently and spill different ones.
    What no ranking can change is that a key a tier gave up is in the other, and that its row
    is the one it came in with.
    """
    storage_keys, cache_keys, ps_keys, batch = _scenario()
    old, new = (
        World("old", storage_keys, cache_keys, ps_keys),
        World("old", storage_keys, cache_keys, ps_keys),
    )
    n = len(batch)
    keys, ids = _t(batch), _t([0] * n)
    legacy_slots, _ = legacy_oracle.legacy_prefetch_cache_path(
        old.cache,
        old.storage,
        keys,
        ids,
        DIM,
        DIM,
        torch.float32,
        old.initializer,
        None,
        None,
        None,
        old.ps,
        None,
    )
    (slots, _), _, new_ids = new.prefetch(batch)

    assert torch.equal(
        old.cache_rows(legacy_slots, ids), new.cache_rows(slots[:n], new_ids[:n])
    )

    everything = sorted(set(storage_keys + ps_keys + batch))

    def tiers(world):
        """``(storage, PS)``: key -> row, of what each tier holds."""
        in_storage = {}
        for keys, values, _, _ in world.storage.export_keys_values(_device(), 4096, 0):
            in_storage.update(zip(keys.tolist(), values[:, 0].tolist()))
        found, values = world.in_ps(everything)
        in_ps = {
            key: row
            for key, is_found, row in zip(everything, found, values[:, 0].tolist())
            if is_found
        }
        return in_storage, in_ps

    def origin_row(key):
        if key in storage_keys:
            return key + IN_STORAGE
        if key in ps_keys:
            return key + IN_PS
        return INIT

    want = {key: origin_row(key) for key in everything}
    for name, world in (("legacy body", old), ("padded path", new)):
        in_storage, in_ps = tiers(world)
        for tier in (in_storage, in_ps):
            for key, row in tier.items():
                assert row == want[key], "%s: key %d holds a row of %s, wanted %s" % (
                    name,
                    key,
                    row,
                    want[key],
                )
        assert set(in_storage) | set(in_ps) == set(want), "%s lost a key" % name
        assert any(
            key not in in_storage for key in storage_keys if key not in set(batch)
        ), "the scenario must push storage keys out of storage"


# --------------------------------------------------------------------------
# the module: a three-tier module is transparent
# --------------------------------------------------------------------------
KEY_RANGE = 3000
STEPS = 20


def _step(mod, batch):
    out = mod(batch.indices, batch.offsets)
    out.backward(torch.ones_like(out))
    return out.detach().clone().cpu()


@pytest.mark.parametrize("kind", PS_KINDS)
def test_a_three_tier_module_gives_the_values_of_one_table_that_holds_every_key(kind):
    """Cache of 1024 rows, storage of 2048, a PS below, 3000 keys per table: keys are spilled to
    the PS and read back, written back and updated by the backward. The reference is a host-only
    table big enough to hold them all; with the same initializer (a function of the key) and the same
    updates its forward values are the values every step has to give."""
    plan = {"max_capacity": 2048, "hbm_rows": 1024, "init_capacity": None}
    _, mod, maker = ps_standins.probe_module(
        "cache_ps", plan, ps_standins.checked(ps_class(kind))
    )
    ref_plan = hsp.plan_of(hsp.CONFIGS["host_only"], "all_hit")
    _, ref, ref_maker = ps_standins.probe_module("host_only", ref_plan)
    batches = ps_standins.random_batches(maker, STEPS, KEY_RANGE)
    ref_batches = ps_standins.random_batches(ref_maker, STEPS, KEY_RANGE)
    ps = mod._external_storage
    reads = []

    for step, (batch, ref_batch) in enumerate(zip(batches, ref_batches)):
        start = sf.READ_NOW_CALLS
        out = _step(mod, batch)
        reads.append(sf.READ_NOW_CALLS - start)
        assert torch.equal(out, _step(ref, ref_batch)), (
            "step %d differs from the one big table" % step
        )

    torch.cuda.synchronize()
    assert any(
        n > 0 for name, n in ps.calls if name.startswith("insert")
    ), "the PS was never spilled to"
    assert any(
        n > 0 for name, n in ps.calls if name.startswith("find")
    ), "the PS was never asked"
    if kind == "old":
        assert set(reads) <= {1, 3}, (
            "a host-sized tier: the miss count, then two calls that need a count: %s"
            % reads
        )
    else:
        assert reads == [0] * STEPS, (
            "a padded tier costs DynamicEmb no read: %s" % reads
        )
