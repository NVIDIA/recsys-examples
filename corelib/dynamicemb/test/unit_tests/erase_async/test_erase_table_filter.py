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

"""``table_names``: an erase by score narrowed to the tables named, on a module and on a
model, so that tables which share a module can be erased at thresholds of their own --
a TTL of an hour on one and of a hundred hours on another, one call each."""

import pytest
import torch
from dynamicemb import DynamicEmbScoreStrategy, erase_by_score_async
from dynamicemb_extensions import device_timestamp
from helpers import build_de_table, found, size, train_step
from torch import nn

S = DynamicEmbScoreStrategy
LRU_LFU = (S.TIMESTAMP, S.LFU)
HOUR = 3600 * 10**9
KEYS = list(range(1, 601))
AGES = [(0, 2 * HOUR, 200 * HOUR)[k % 3] for k in KEYS]


def two_table_module(device, names=None):
    """Two ``(TIMESTAMP, LFU)`` tables with the same options, in one module and one key
    map, each holding ``KEYS`` aged by ``AGES``; returns the module and the clock."""
    module = build_de_table(device.index, LRU_LFU, num_tables=2, names=names)
    assert module._storage.key_index_map.num_tables_ == 2, "the tables do not share"
    train_step(module, [KEYS, KEYS], device)
    now = device_timestamp()
    assert now > 200 * HOUR
    for table_id in (0, 1):
        age(module, table_id, now, device)
    return module, now


def age(module, table_id, now, device):
    """Give each key of ``KEYS`` in table ``table_id`` the last access its ``AGES`` says,
    keeping its frequency."""
    key_map = module._storage.key_index_map
    keys = torch.tensor(KEYS, dtype=torch.int64, device=device)
    slots = key_map.slots_of(keys, torch.full_like(keys, table_id))
    assert bool((slots >= 0).all())
    blocks = key_map.gather_score_blocks(table_id, slots).clone()
    blocks[:, 0] = torch.tensor([now - a for a in AGES], device=device)
    key_map.scatter_score_blocks(table_id, slots, blocks)


def older_than(max_age):
    return [k for k, a in zip(KEYS, AGES) if a > max_age]


def younger_than(max_age):
    return [k for k, a in zip(KEYS, AGES) if a <= max_age]


def test_only_the_named_table_is_erased(device):
    module, now = two_table_module(device)
    stale = older_than(HOUR)
    handle = module.erase_by_score_async(S.TIMESTAMP, now - HOUR, table_names=["t_0"])
    assert handle.result() == (len(stale), 0)
    assert not bool(found(module, stale, device, table_id=0).any())
    assert bool(found(module, younger_than(HOUR), device, table_id=0).all())
    assert bool(found(module, KEYS, device, table_id=1).all()), "t_1 was touched"
    assert size(module, 1) == len(KEYS)


def test_two_tables_at_two_thresholds_one_call_each(device):
    """Back to back on one module: each table loses the keys past its own age, and each
    handle has its own counts."""
    module, now = two_table_module(device)
    short = module.erase_by_score_async(S.TIMESTAMP, now - HOUR, table_names=["t_0"])
    long = module.erase_by_score_async(
        S.TIMESTAMP, now - 100 * HOUR, table_names=["t_1"]
    )
    assert short.result() == (len(older_than(HOUR)), 0)
    assert long.result() == (len(older_than(100 * HOUR)), 0)
    for table_id, max_age in ((0, HOUR), (1, 100 * HOUR)):
        assert not bool(found(module, older_than(max_age), device, table_id).any())
        assert bool(found(module, younger_than(max_age), device, table_id).all())


def test_no_names_erases_what_naming_every_table_erases(device):
    unfiltered, now = two_table_module(device)
    named, named_now = two_table_module(device)
    everything = unfiltered.erase_by_score_async(S.TIMESTAMP, now - HOUR).result()
    both = named.erase_by_score_async(
        S.TIMESTAMP, named_now - HOUR, table_names=["t_1", "t_0"]
    ).result()
    assert everything == both == (2 * len(older_than(HOUR)), 0)
    for table_id in (0, 1):
        assert size(unfiltered, table_id) == size(named, table_id)


@pytest.mark.parametrize(
    "table_names, error, match",
    [
        (["t_0", "nope"], ValueError, "not tables of this module"),
        ([], ValueError, "empty"),
        ("t_0", TypeError, "not a str"),
    ],
)
def test_a_bad_name_is_refused_before_anything_is_erased(
    device, table_names, error, match
):
    module, now = two_table_module(device)
    with pytest.raises(error, match=match):
        module.erase_by_score_async(S.TIMESTAMP, now - HOUR, table_names=table_names)
    assert size(module, 0) == size(module, 1) == len(KEYS)


def test_a_table_the_strategy_does_not_erase_is_refused(device):
    """``LFU`` tables keep no timestamp: naming one for ``TIMESTAMP`` is an error, where
    leaving the names out picks nothing and is not."""
    module = build_de_table(device.index, S.LFU, num_tables=2)
    train_step(module, [KEYS, KEYS], device)
    with pytest.raises(ValueError, match="not erased by"):
        module.erase_by_score_async(S.TIMESTAMP, 1 << 62, table_names=["t_0"])
    assert module.erase_by_score_async(S.TIMESTAMP, 1 << 62).result() == (0, 0)
    assert size(module, 0) == len(KEYS)


def two_module_model(device):
    """A model of two modules: tables ``a_0`` and ``a_1`` in one, ``b_0`` in the other,
    every key two hundred hours old but the ones ``AGES`` keeps young."""
    first, now = two_table_module(device, names=["a_0", "a_1"])
    second = build_de_table(device.index, LRU_LFU, names=["b_0"])
    train_step(second, KEYS, device)
    age(second, 0, now, device)
    return nn.ModuleList([first, second]), first, second, now


def test_on_a_model_each_module_gets_the_names_it_holds(device):
    model, first, second, now = two_module_model(device)
    stale = len(older_than(HOUR))
    handle = erase_by_score_async(
        model, S.TIMESTAMP, now - HOUR, table_names=["a_0", "b_0"]
    )
    assert handle.result() == (2 * stale, 0)
    assert size(first, 0) == size(second, 0) == len(KEYS) - stale
    assert size(first, 1) == len(KEYS)

    handle = erase_by_score_async(model, S.TIMESTAMP, now - HOUR, table_names=["a_1"])
    assert handle.result() == (stale, 0)
    assert size(first, 1) == len(KEYS) - stale
    assert size(second, 0) == len(KEYS) - stale


@pytest.mark.parametrize(
    "table_names, match",
    [
        (["a_0", "nope"], "not tables of any"),
        (["b_0", "a_0", "t_9"], "not tables of any"),
        ([], "empty"),
    ],
)
def test_on_a_model_a_bad_name_starts_no_module(device, table_names, match):
    model, first, second, now = two_module_model(device)
    with pytest.raises(ValueError, match=match):
        erase_by_score_async(model, S.TIMESTAMP, now - HOUR, table_names=table_names)
    torch.cuda.synchronize()
    assert size(first, 0) == size(first, 1) == size(second, 0) == len(KEYS)


def test_on_a_model_a_name_the_strategy_does_not_erase_starts_no_module(device):
    lru_lfu, now = two_table_module(device, names=["a_0", "a_1"])
    lfu = build_de_table(device.index, S.LFU, names=["c_0"])
    train_step(lfu, KEYS, device)
    model = nn.ModuleList([lru_lfu, lfu])
    with pytest.raises(ValueError, match="not erased by"):
        erase_by_score_async(model, S.TIMESTAMP, now - HOUR, table_names=["a_0", "c_0"])
    torch.cuda.synchronize()
    assert size(lru_lfu, 0) == len(KEYS)
