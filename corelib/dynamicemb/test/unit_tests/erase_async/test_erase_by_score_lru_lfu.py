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

"""An erase by score on a ``(TIMESTAMP, LFU)`` table, whose keys carry two score words:
word 0 the last-access timestamp, word 1 the frequency. The strategy named picks the word
compared -- ``TIMESTAMP`` an expiry, ``LFU`` a frequency floor -- and the other is not
looked at. The scores are written with ``scatter_score_blocks``, in the physical order
``[timestamp, frequency]``, so each key's age and frequency are the test's own."""

import pytest
import torch
from dynamicemb import DynamicEmbScoreStrategy, erase_by_score_async
from dynamicemb.key_ownership import murmur3_hash_64bits
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import ScorePolicy, device_timestamp
from helpers import (
    BUCKET,
    build_de_table,
    erased_and_spared,
    found,
    new_counts,
    train_step,
)
from torch import nn

S = DynamicEmbScoreStrategy
SCORE = "s"
RECLAIM_KEY = (1 << 64) - 2
TIMESTAMP_WORD, FREQUENCY_WORD = 0, 1

KEYS = list(range(1, 1001))
FRESH_TS, STALE_TS = 2_000_000, 1_000
TS_THRESHOLD = 1_000_000
FREQ_THRESHOLD = 30
TIMESTAMPS = [STALE_TS if k % 2 else FRESH_TS for k in KEYS]
FREQUENCIES = [1 + (k * 37) % 100 for k in KEYS]


def make_table(capacities):
    return get_scored_table(
        capacity=capacities,
        bucket_capacity=BUCKET,
        key_type=torch.int64,
        score_specs=[ScoreSpec(name=SCORE, policy=ScorePolicy.LRU_LFU)],
    )


def fill(table, keys, timestamps, frequencies, device, table_id=0):
    """Insert ``keys`` and give them their two score words; returns their slots."""
    keys = torch.as_tensor(keys, dtype=torch.int64, device=device)
    ones = torch.ones(keys.numel(), dtype=torch.uint64, device=device)
    slots = table.insert(
        keys,
        torch.full_like(keys, table_id),
        ScoreArg(name=SCORE, value=ones, policy=ScorePolicy.LRU_LFU),
    )
    assert bool((slots >= 0).all())
    blocks = torch.tensor(
        list(zip(timestamps, frequencies)), dtype=torch.int64, device=device
    )
    table.scatter_score_blocks(table_id, slots, blocks)
    return slots


def found_in(table, keys, device, table_id=0):
    keys = torch.as_tensor(keys, dtype=torch.int64, device=device)
    arg = ScoreArg(name=SCORE, value=None, policy=ScorePolicy.CONST)
    _, founds, _ = table.lookup(keys, torch.full_like(keys, table_id), arg)
    return founds


def blocks_at(table, slots, table_id=0):
    return table.gather_score_blocks(table_id, slots).to(torch.int64).tolist()


def digest_of(key):
    return (murmur3_hash_64bits(key) >> 32) & 0xFF


@pytest.mark.parametrize(
    "score_word, threshold, scores",
    [
        (TIMESTAMP_WORD, TS_THRESHOLD, TIMESTAMPS),
        (FREQUENCY_WORD, FREQ_THRESHOLD, FREQUENCIES),
    ],
    ids=["timestamp", "frequency"],
)
def test_the_word_named_is_compared_and_the_other_is_not(
    device, score_word, threshold, scores
):
    table = make_table([BUCKET * 64])
    slots = fill(table, KEYS, TIMESTAMPS, FREQUENCIES, device)
    below = [k for k, s in zip(KEYS, scores) if s < threshold]
    kept = [i for i, s in enumerate(scores) if s >= threshold]

    counts = new_counts(device)
    table.erase_by_score(0, threshold, counts, score_word=score_word)
    assert erased_and_spared(counts) == (len(below), 0)
    assert int(table.size()) == len(KEYS) - len(below)
    assert not bool(found_in(table, below, device).any())
    kept_slots = slots[torch.tensor(kept, device=device)]
    assert blocks_at(table, kept_slots) == [
        [TIMESTAMPS[i], FREQUENCIES[i]] for i in kept
    ]


def test_by_timestamp_a_stale_key_goes_however_frequent_and_a_fresh_one_stays(device):
    """The expiry: what decides is the age alone."""
    table = make_table([BUCKET * 8])
    stale_frequent, fresh_rare = [1, 2], [3, 4]
    fill(
        table,
        stale_frequent + fresh_rare,
        [STALE_TS] * 2 + [FRESH_TS] * 2,
        [1 << 40, 1 << 40, 1, 1],
        device,
    )
    counts = new_counts(device)
    table.erase_by_score(0, TS_THRESHOLD, counts, score_word=TIMESTAMP_WORD)
    assert erased_and_spared(counts) == (2, 0)
    assert found_in(table, stale_frequent + fresh_rare, device).tolist() == [
        False,
        False,
        True,
        True,
    ]


def test_by_frequency_a_rare_key_goes_however_fresh_and_a_frequent_one_stays(device):
    """The mirror: what decides is the frequency alone, so a key just seen once goes."""
    table = make_table([BUCKET * 8])
    fresh_rare, stale_frequent = [1, 2], [3, 4]
    fill(
        table,
        fresh_rare + stale_frequent,
        [FRESH_TS] * 2 + [STALE_TS] * 2,
        [1, 1, 50, 50],
        device,
    )
    counts = new_counts(device)
    table.erase_by_score(0, FREQ_THRESHOLD, counts, score_word=FREQUENCY_WORD)
    assert erased_and_spared(counts) == (2, 0)
    assert found_in(table, fresh_rare + stale_frequent, device).tolist() == [
        False,
        False,
        True,
        True,
    ]


@pytest.mark.parametrize("score_word", [TIMESTAMP_WORD, FREQUENCY_WORD])
def test_an_erased_slot_is_cleared_and_taken_before_any_live_key(device, score_word):
    """Both score words of an erased slot are cleared and it carries the digest of a
    reclaimed key, whichever word was compared. Then a full bucket with half its slots
    erased takes as many new keys without evicting a live one: the reclaimed slots rank
    below every key in use (frequency 1 at least)."""
    table = make_table([BUCKET])
    keys = list(range(1, BUCKET + 1))
    stale = [k % 2 == 1 for k in keys]
    timestamps = [STALE_TS if s else FRESH_TS for s in stale]
    frequencies = [1 if s else 2 for s in stale]
    threshold = TS_THRESHOLD if score_word == TIMESTAMP_WORD else 2
    slots = fill(table, keys, timestamps, frequencies, device)
    assert int(table.size()) == BUCKET

    counts = new_counts(device)
    table.erase_by_score(0, threshold, counts, score_word=score_word)
    assert erased_and_spared(counts) == (BUCKET // 2, 0)
    erased_slots = slots[torch.tensor(stale, device=device)]
    assert blocks_at(table, erased_slots) == [[0, 0]] * (BUCKET // 2)
    digests = table.digests_.cpu()
    for slot in erased_slots.tolist():
        assert int(digests[slot // BUCKET, slot % BUCKET]) == digest_of(RECLAIM_KEY)

    live = [k for k, s in zip(keys, stale) if not s]
    new_keys = list(range(10_001, 10_001 + BUCKET // 2))
    fill(table, new_keys, [FRESH_TS] * len(new_keys), [1] * len(new_keys), device)
    assert int(table.size()) == BUCKET
    assert bool(found_in(table, live, device).all()), "a live key was evicted"
    assert bool(found_in(table, new_keys, device).all())


@pytest.mark.parametrize("score_word", [TIMESTAMP_WORD, FREQUENCY_WORD])
def test_a_pinned_slot_is_spared_whatever_its_scores(device, score_word):
    table = make_table([BUCKET * 32])
    slots = fill(table, KEYS, TIMESTAMPS, FREQUENCIES, device)
    scores = TIMESTAMPS if score_word == TIMESTAMP_WORD else FREQUENCIES
    threshold = TS_THRESHOLD if score_word == TIMESTAMP_WORD else FREQ_THRESHOLD
    below = [i for i, s in enumerate(scores) if s < threshold]
    pinned = below[::4]
    pin_slots = slots[torch.tensor(pinned, device=device)]
    table.increment_counter(pin_slots, torch.zeros_like(pin_slots))

    counts = new_counts(device)
    table.erase_by_score(0, threshold, counts, score_word=score_word)
    assert erased_and_spared(counts) == (len(below) - len(pinned), len(pinned))
    assert bool(found_in(table, [KEYS[i] for i in pinned], device).all())


def test_the_word_is_checked(device):
    table = make_table([BUCKET * 4])
    counts = new_counts(device)
    for bad in (None, -1, 2):
        with pytest.raises(ValueError, match="score_word"):
            table.erase_by_score(0, 10, counts, score_word=bad)
    single = get_scored_table(
        capacity=[BUCKET * 4],
        bucket_capacity=BUCKET,
        key_type=torch.int64,
        score_specs=[ScoreSpec(name=SCORE, policy=ScorePolicy.ASSIGN)],
    )
    with pytest.raises(ValueError, match="score word 0 only"):
        single.erase_by_score(0, 10, counts, score_word=1)
    single.erase_by_score(0, 10, counts, score_word=0)


ORDERS = [(S.TIMESTAMP, S.LFU), (S.LFU, S.TIMESTAMP)]


def set_score_words(module, keys, timestamps, frequencies, device):
    """Give ``keys`` of table 0 these score words, in the physical order."""
    key_map = module._storage.key_index_map
    keys = torch.as_tensor(keys, dtype=torch.int64, device=device)
    slots = key_map.slots_of(keys, torch.zeros_like(keys))
    assert bool((slots >= 0).all())
    blocks = torch.tensor(
        list(zip(timestamps, frequencies)), dtype=torch.int64, device=device
    )
    key_map.scatter_score_blocks(0, slots, blocks)


@pytest.mark.parametrize("order", ORDERS, ids=["ts_first", "lfu_first"])
def test_the_word_a_strategy_names_is_the_same_in_either_tuple_order(device, order):
    module = build_de_table(device.index, order)
    assert module.erasable_table_ids(S.TIMESTAMP) == [0]
    assert module.erasable_table_ids(S.LFU) == [0]
    assert module.erase_score_word(S.TIMESTAMP, [0]) == TIMESTAMP_WORD
    assert module.erase_score_word(S.LFU, [0]) == FREQUENCY_WORD


@pytest.mark.parametrize("order", ORDERS, ids=["ts_first", "lfu_first"])
@pytest.mark.parametrize("strategy", [S.TIMESTAMP, S.LFU])
def test_a_model_is_erased_by_the_word_the_strategy_names(device, order, strategy):
    """Timestamps from the device clock: stale keys were last seen an hour before the
    threshold. The reference sits two hours past the clock, so no timestamp is negative
    however recently the GPU started."""
    module = build_de_table(device.index, order)
    train_step(module, KEYS, device)
    hour = 3600 * 10**9
    now = device_timestamp() + 2 * hour
    timestamps = [now - 2 * hour if t == STALE_TS else now for t in TIMESTAMPS]
    set_score_words(module, KEYS, timestamps, FREQUENCIES, device)
    if strategy == S.TIMESTAMP:
        threshold, scores = now - hour, timestamps
    else:
        threshold, scores = FREQ_THRESHOLD, FREQUENCIES
    below = [k for k, s in zip(KEYS, scores) if s < threshold]
    kept = [k for k, s in zip(KEYS, scores) if s >= threshold]

    handle = erase_by_score_async(nn.ModuleList([module]), strategy, threshold)
    assert handle.result() == (len(below), 0)
    assert not bool(found(module, below, device).any())
    assert bool(found(module, kept, device).all())
    train_step(module, kept, device)


def test_a_model_of_single_score_and_two_word_tables_is_erased_in_both(device):
    """``LFU`` takes ``LFU`` tables and ``(TIMESTAMP, LFU)`` ones; ``TIMESTAMP`` only the
    second. The frequencies come from training: once for ``rare``, three times for
    ``frequent``."""
    single = build_de_table(device.index, S.LFU)
    two_words = build_de_table(device.index, (S.TIMESTAMP, S.LFU))
    model = nn.ModuleList([single, two_words])
    rare, frequent = list(range(1, 301)), list(range(1001, 1301))
    for module in model:
        train_step(module, rare + frequent, device)
        train_step(module, frequent, device)
        train_step(module, frequent, device)

    assert erase_by_score_async(model, S.LFU, 2).result() == (2 * len(rare), 0)
    for module in model:
        assert not bool(found(module, rare, device).any())
        assert bool(found(module, frequent, device).all())

    later = device_timestamp() + 1
    assert erase_by_score_async(model, S.TIMESTAMP, later).result() == (
        len(frequent),
        0,
    )
    assert bool(found(single, frequent, device).all()), "a single LFU table was touched"
    assert not bool(found(two_words, frequent, device).any())
