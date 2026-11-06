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

"""An erase on a module trained for real, asked for the way a user asks: the uses it is
for, the layouts it works on, and what it must not do.

Every training step checks that each key reads its own value (``helpers.train_step``), so
a row that an erase took from a key that was using it fails there, not later."""

import time

import torch
from dynamicemb import DynamicEmbScoreStrategy
from dynamicemb.async_erase import EraseHandle
from dynamicemb.batched_dynamicemb_tables import StorageLayout
from dynamicemb_extensions import device_timestamp
from helpers import DIM, build_de_table, found, size, train_step

S = DynamicEmbScoreStrategy
NO_EVICTION_PRECISE = (S.NO_EVICTION, S.PRECISE_CUSTOMIZED)


def key_range(first, count):
    return list(range(first, first + count))


# ---------------------------------------------------------------------------
# What it is for
# ---------------------------------------------------------------------------


def test_a_sliding_window_of_keys_does_not_grow_the_table(device):
    """The use the feature is for. A table that never evicts takes a thousand new keys a
    round, scored by the round; each round erases what is older than the last three. The
    table grows to hold what the window holds and then stays that size: the keys the
    erase frees are the room the next round's keys take."""
    per_round, keep, rounds = 1000, 3, 12
    module = build_de_table(device.index, NO_EVICTION_PRECISE)
    key_map = module._storage.key_index_map
    shapes = []  # (key map slots, value rows) after each round
    for r in range(1, rounds + 1):
        keys = key_range((r - 1) * per_round + 1, per_round)
        train_step(module, keys, device, scores=[r] * per_round)
        # Scores are the rounds: the rounds up to r - keep are below this.
        handle = module.erase_by_score_async(S.PRECISE_CUSTOMIZED, max(1, r - keep + 1))
        assert isinstance(handle, EraseHandle)
        erased, spared = handle.result()
        assert (erased, spared) == (per_round if r > keep else 0, 0), f"round {r}"
        assert size(module) == min(r, keep) * per_round, f"round {r}"
        shapes.append(
            (key_map.main_capacity(0), module._storage._state.tables[0].shape[0])
        )
        # Keys of the rounds the window has left are gone, those inside are there.
        first = max(1, r - keep + 1)
        left = key_range(1, (first - 1) * per_round)
        inside = key_range((first - 1) * per_round + 1, (r - first + 1) * per_round)
        if left:
            assert not bool(found(module, left, device).any()), f"round {r}"
        assert bool(found(module, inside, device).all()), f"round {r}"
        key_map = module._storage.key_index_map  # an expansion replaces it
    # Four rounds' keys are live at the peak (three kept and one new): after the table
    # has grown to hold them it never grows again.
    assert len(set(shapes[keep + 1 :])) == 1, shapes
    # What the window kept was not touched: it still reads right.
    live = key_range((rounds - keep) * per_round + 1, keep * per_round)
    train_step(module, live, device, scores=[rounds] * len(live))


def test_training_goes_on_while_the_call_erases_and_each_key_reads_its_own_value(device):
    """The call is made, and the training does not wait: new keys come in -- the table
    grows under the scan, which starts again -- and every output is right. When the erase
    is done, the old keys are gone, once each, and the new ones are all there."""
    module = build_de_table(device.index, NO_EVICTION_PRECISE)
    old = key_range(1, 6000)
    for start in range(0, len(old), 1000):
        train_step(module, old[start : start + 1000], device, scores=[5] * 1000)

    handle = module.erase_by_score_async(S.PRECISE_CUSTOMIZED, 100)
    new = []
    for step in range(30):
        batch = key_range(10_001 + step * 1000, 1000)
        new += batch
        train_step(module, batch, device, scores=[500] * 1000)

    assert handle.result() == (len(old), 0)
    assert not bool(found(module, old, device).any())
    assert bool(found(module, new, device).all())
    assert size(module) == len(new)


def test_erase_by_time_with_a_threshold_from_the_device_clock(device):
    """``TIMESTAMP`` scores are the device's clock: the threshold is read from it."""
    module = build_de_table(device.index, S.TIMESTAMP)
    old, new = key_range(1, 500), key_range(1001, 500)
    train_step(module, old, device)
    time.sleep(0.05)
    threshold = device_timestamp()
    time.sleep(0.05)
    train_step(module, new, device)

    assert module.erase_by_score_async(S.TIMESTAMP, threshold).result() == (500, 0)
    assert not bool(found(module, old, device).any())
    assert bool(found(module, new, device).all())


# ---------------------------------------------------------------------------
# What it must not do
# ---------------------------------------------------------------------------


def test_a_key_a_prefetched_batch_is_using_is_spared_until_the_batch_is_done(device):
    """The forward of a batch pins its keys until the backward. An erase asked for in
    between spares them whatever their score and says how many; once the backward has run
    they are erased like any other."""
    module = build_de_table(device.index, S.CUSTOMIZED)
    keys = key_range(1, 800)
    module.set_score({"t_0": 5})
    indices = torch.tensor(keys, dtype=torch.int64, device=device)
    offsets = torch.arange(0, len(keys) + 1, dtype=torch.int64, device=device)
    out = module(indices, offsets)  # pinned from here to the backward
    torch.cuda.synchronize()

    assert module.erase_by_score_async(S.CUSTOMIZED, 100).result() == (0, len(keys))
    assert bool(found(module, keys, device).all())

    out.sum().backward()  # lets go of the keys
    torch.cuda.synchronize()
    assert module.erase_by_score_async(S.CUSTOMIZED, 100).result() == (len(keys), 0)
    assert not bool(found(module, keys, device).any())


def test_the_same_key_in_two_tables_is_erased_from_the_one_whose_score_is_low(device):
    """Two tables of a module, the same key numbers in both, opposite ages: a key that is
    old in the first table is new in the second. Each table is erased by its own keys'
    scores -- not by the key number, not by the other table -- which fails if a slot is
    read in the wrong table's range."""
    module = build_de_table(device.index, S.CUSTOMIZED, num_tables=2)
    low, high = key_range(1, 500), key_range(501, 500)
    train_step(module, [low, high], device, score=10)  # old: low in t_0, high in t_1
    train_step(module, [high, low], device, score=200)  # new: high in t_0, low in t_1

    assert module.erase_by_score_async(S.CUSTOMIZED, 100).result() == (1000, 0)
    assert size(module, 0) == 500 and size(module, 1) == 500
    assert not bool(found(module, low, device, table_id=0).any())
    assert bool(found(module, high, device, table_id=0).all())
    assert not bool(found(module, high, device, table_id=1).any())
    assert bool(found(module, low, device, table_id=1).all())


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------


def test_with_a_cache_only_the_storage_behind_it_is_erased(device):
    """The cache is small, so keys spill to the storage behind it. The erase scans that
    storage; the cache is neither read nor changed -- a key in it stays there, and a key
    it holds that the storage no longer does is written back when the cache lets it go.
    (Spared is always zero: a cache pins its own slots, the storage keeps no counter.)

    The cache holds about thirteen hundred rows, so four thousand keys do not fit."""
    max_capacity = 1 << 16
    module = build_de_table(
        device.index,
        S.CUSTOMIZED,
        caching=True,
        max_capacity=max_capacity,
        init_capacity=max_capacity,
        local_hbm_for_values=int(max_capacity * DIM * 4 * 0.02),
    )
    assert module._storage_layout is StorageLayout.CACHING
    old, new = key_range(1, 2000), key_range(2001, 2000)
    for keys, score in ((old, 10), (new, 200)):
        for start in range(0, len(keys), 500):
            train_step(module, keys[start : start + 500], device, score=score)
    everything = old + new
    in_cache = found(module, everything, device, cache=True).clone()
    assert size(module) == len(everything), "every key reaches the storage"
    assert 0 < int(in_cache.sum()) < len(everything), "the cache is neither empty nor all"

    assert module.erase_by_score_async(S.CUSTOMIZED, 100).result() == (len(old), 0)

    assert not bool(found(module, old, device).any())
    assert bool(found(module, new, device).all())
    assert bool((found(module, everything, device, cache=True) == in_cache).all())
    train_step(module, everything[:500], device, score=300)  # and it still reads right
