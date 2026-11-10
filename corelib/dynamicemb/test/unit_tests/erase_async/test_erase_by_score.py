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

"""``LinearBucketTable.erase_by_score``: erase the keys whose score is below a
threshold, over a range of slots, sparing the slots a ref counter pins."""

import pytest
import torch
from dynamicemb.key_ownership import murmur3_hash_64bits
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import ScorePolicy
from helpers import BUCKET, erased_and_spared, new_counts
SCORE = "s"
EMPTY_KEY = (1 << 64) - 1
RECLAIM_KEY = (1 << 64) - 2


def make_table(capacities, use_ref_counter=True, policy=ScorePolicy.ASSIGN):
    return get_scored_table(
        capacity=capacities,
        bucket_capacity=BUCKET,
        key_type=torch.int64,
        score_specs=[ScoreSpec(name=SCORE, policy=policy)],
        use_ref_counter=use_ref_counter,
    )


def fill(table, keys, scores, table_id, device):
    """Insert ``keys`` into logical table ``table_id`` with ``scores``; returns the
    slots they landed in."""
    keys = torch.as_tensor(keys, dtype=torch.int64, device=device)
    scores = torch.as_tensor(scores, dtype=torch.uint64, device=device)
    tids = torch.full_like(keys, table_id)
    return table.insert(
        keys, tids, ScoreArg(name=SCORE, value=scores, policy=ScorePolicy.ASSIGN)
    )


def find(table, keys, table_id, device):
    """``(score, found)`` of ``keys``, without touching their scores."""
    keys = torch.as_tensor(keys, dtype=torch.int64, device=device)
    tids = torch.full_like(keys, table_id)
    score_out, founds, _ = table.lookup(
        keys, tids, ScoreArg(name=SCORE, value=None, policy=ScorePolicy.CONST)
    )
    return score_out, founds


@pytest.mark.parametrize(
    "threshold, use_ref_counter",
    [
        (30, True),  # in the middle: some below it, some equal to it, some above
        (0, True),  # nothing is below zero
        (1 << 62, True),  # everything is below
        (30, False),  # a table that keeps no ref counter spares nothing either
    ],
)
def test_erases_exactly_the_keys_below_the_threshold(device, threshold, use_ref_counter):
    table = make_table([BUCKET * 64], use_ref_counter=use_ref_counter)
    keys = list(range(1, 2001))
    scores = [k % 100 for k in keys]  # 0..99, twenty keys of each
    fill(table, keys, scores, 0, device)
    assert int(table.size()) == len(keys)

    counts = new_counts(device)
    table.erase_by_score(0, threshold, counts)
    below = [k for k, s in zip(keys, scores) if s < threshold]
    kept = [k for k, s in zip(keys, scores) if s >= threshold]
    equal = [k for k, s in zip(keys, scores) if s == threshold]
    assert erased_and_spared(counts) == (len(below), 0)
    assert int(table.size()) == len(kept)

    if below:
        assert not bool(find(table, below, 0, device)[1].any())
    if kept:
        score_out, founds = find(table, kept, 0, device)
        assert bool(founds.all())
        # What is kept is untouched, its score included.
        expected = torch.tensor([k % 100 for k in kept], dtype=torch.int64, device=device)
        assert torch.equal(score_out.to(torch.int64), expected)
    # Strictly below: a key scored exactly the threshold is among the kept.
    if equal:
        assert set(equal) <= set(kept)


@pytest.mark.parametrize("chunk", [BUCKET, 100, 1000])
def test_scanning_in_chunks_erases_the_same_keys(device, chunk):
    """A chunk of whole buckets, and a ragged one (100 and 1000 slots are not whole
    warps' worth of buckets, which takes the one-lane path)."""
    n = BUCKET * 64
    keys = list(range(1, 3001))
    scores = [(k * 7) % 100 for k in keys]

    whole = make_table([n])
    fill(whole, keys, scores, 0, device)
    counts_whole = new_counts(device)
    whole.erase_by_score(0, 40, counts_whole)

    chunked = make_table([n])
    fill(chunked, keys, scores, 0, device)
    counts_chunked = new_counts(device)
    for start in range(0, n, chunk):
        chunked.erase_by_score(0, 40, counts_chunked, begin=start, end=min(start + chunk, n))

    assert erased_and_spared(counts_chunked) == erased_and_spared(counts_whole)
    assert int(chunked.size()) == int(whole.size())
    assert torch.equal(find(chunked, keys, 0, device)[1], find(whole, keys, 0, device)[1])


def test_only_the_named_table_is_scanned(device):
    """Two tables share one storage array; a range of table 1 stays inside it."""
    table = make_table([BUCKET * 16, BUCKET * 32])
    keys = list(range(1, 501))
    scores = [k % 100 for k in keys]
    fill(table, keys, scores, 0, device)
    fill(table, keys, scores, 1, device)  # the same keys, in the other table

    counts = new_counts(device)
    table.erase_by_score(1, 50, counts)
    below = sum(1 for s in scores if s < 50)
    assert erased_and_spared(counts) == (below, 0)
    assert bool(find(table, keys, 0, device)[1].all()), "table 0 was touched"
    assert int(table.size(0)) == len(keys)
    assert int(table.size(1)) == len(keys) - below


@pytest.mark.parametrize("table_id", [0, 1])
def test_a_pinned_slot_is_spared_whatever_its_score(device, table_id):
    """The ref counters of all the tables are one array indexed by the global slot,
    so a pin on a slot of table 1 has to be found at its place in that array, not
    where it falls counted from the start of its own table (that is table 0's
    counters). Both tables, both ways round."""
    table = make_table([BUCKET * 32, BUCKET * 32])
    other = 1 - table_id
    keys = list(range(1, 801))
    scores = [k % 100 for k in keys]
    fill(table, keys, scores, other, device)  # the same keys in the other table, unpinned
    slots = fill(table, keys, scores, table_id, device)
    assert bool((slots >= 0).all())

    below = [i for i, s in enumerate(scores) if s < 50]
    pinned = below[::4]  # a quarter of what is below the threshold
    pin_slots = slots[torch.tensor(pinned, device=device)]
    table.increment_counter(pin_slots, torch.full_like(pin_slots, table_id))

    counts = new_counts(device)
    table.erase_by_score(table_id, 50, counts)
    assert erased_and_spared(counts) == (len(below) - len(pinned), len(pinned))
    pinned_keys = [keys[i] for i in pinned]
    assert bool(find(table, pinned_keys, table_id, device)[1].all()), "a pinned key was erased"
    assert bool(find(table, keys, other, device)[1].all()), "the other table was touched"

    # Released, the same keys go the next time, and nothing is spared. A fresh
    # ``counts``: they add up across calls, so this one says what this scan did.
    table.decrement_counter(pin_slots, torch.full_like(pin_slots, table_id))
    counts = new_counts(device)
    table.erase_by_score(table_id, 50, counts)
    assert erased_and_spared(counts) == (len(pinned), 0)
    assert not bool(find(table, pinned_keys, table_id, device)[1].any())

    # The other table has no pin of its own: the pins above shelter nothing there.
    counts = new_counts(device)
    table.erase_by_score(other, 50, counts)
    assert erased_and_spared(counts) == (len(below), 0)


def test_an_erased_slot_takes_a_new_key_and_the_sizes_stay_right(device):
    """A slot an erase left reclaimed takes a new key, and the bucket sizes follow:
    the table is as big as the keys inserted since, and those can be erased again."""
    table = make_table([BUCKET * 8])
    keys = list(range(1, 501))
    fill(table, keys, [k % 100 for k in keys], 0, device)
    counts = new_counts(device)
    table.erase_by_score(0, 100, counts)  # everything: every score is below 100
    assert erased_and_spared(counts) == (len(keys), 0)

    new_keys = list(range(10_001, 10_301))
    fill(table, new_keys, [5] * len(new_keys), 0, device)
    assert int(table.size()) == len(new_keys)
    assert bool(find(table, new_keys, 0, device)[1].all())

    # The new keys go too (a score of 5 is below 6). A fresh ``counts``: they add up
    # across calls, so this one says what this scan did.
    counts = new_counts(device)
    table.erase_by_score(0, 6, counts)
    assert erased_and_spared(counts) == (len(new_keys), 0)
    assert int(table.size()) == 0


def test_the_arguments_are_checked(device):
    table = make_table([BUCKET * 4])
    counts = new_counts(device)
    with pytest.raises(ValueError, match="negative"):
        table.erase_by_score(0, -1, counts)
    with pytest.raises(ValueError, match="not inside"):
        table.erase_by_score(0, 10, counts, begin=0, end=BUCKET * 4 + 1)
    with pytest.raises(ValueError, match="not inside"):
        table.erase_by_score(0, 10, counts, begin=5, end=4)
    for bad_table in (-1, 1):  # there is only table 0
        with pytest.raises(ValueError, match="not one of the 1 tables"):
            table.erase_by_score(bad_table, 10, counts)

    # ``counts`` is written as two neighbours from one pointer, on the table's device.
    not_contiguous = torch.zeros(4, dtype=torch.int64, device=device)[::2]
    with pytest.raises(RuntimeError, match="counts must be a contiguous int64"):
        table.erase_by_score(0, 10, not_contiguous)
    with pytest.raises(RuntimeError, match="counts must be a contiguous int64"):
        table.erase_by_score(0, 10, torch.zeros(2, dtype=torch.int32, device=device))
    with pytest.raises(RuntimeError, match="counts must be a contiguous int64"):
        table.erase_by_score(0, 10, torch.zeros(2, dtype=torch.int64))  # on the CPU

    lru_lfu = get_scored_table(
        capacity=[BUCKET * 4],
        bucket_capacity=BUCKET,
        key_type=torch.int64,
        score_specs=[ScoreSpec(name=SCORE, policy=ScorePolicy.LRU_LFU)],
    )
    with pytest.raises(ValueError, match="score_word"):
        lru_lfu.erase_by_score(0, 10, counts)

    cache_like = get_scored_table(
        capacity=[BUCKET * 4],
        bucket_capacity=BUCKET,
        key_type=torch.int64,
        score_specs=[ScoreSpec(name=SCORE, policy=ScorePolicy.ASSIGN)],
        enable_overflow=True,
    )
    with pytest.raises(NotImplementedError, match="overflow"):
        cache_like.erase_by_score(0, 10, counts)


def digest_of(key):
    """The digest a slot holding ``key`` carries: ``Bucket::key_to_digest`` (the
    high word of the mixed key, its low byte)."""
    return (murmur3_hash_64bits(key) >> 32) & 0xFF


def test_an_erased_slot_carries_the_digest_of_a_reclaimed_key(device):
    """A slot carries the digest of the key it holds. An erase leaves ReclaimKey in
    it, so the digest is that key's -- not an empty slot's, which a probe looking for
    a free slot would take for a candidate and read the key of, for nothing."""
    assert digest_of(RECLAIM_KEY) != digest_of(EMPTY_KEY), "nothing to tell apart"
    table = make_table([BUCKET * 8])
    keys = list(range(1, 401))
    slots = fill(table, keys, [k % 100 for k in keys], 0, device)
    assert bool((slots >= 0).all())

    table.erase_by_score(0, 40, new_counts(device))
    torch.cuda.synchronize()

    founds = find(table, keys, 0, device)[1].tolist()
    digests = table.digests_.cpu()
    slot_list = slots.tolist()
    n_erased = n_kept = 0
    for key, slot, found in zip(keys, slot_list, founds):
        got = int(digests[slot // BUCKET, slot % BUCKET])
        if found:
            assert got == digest_of(key), f"key {key}: not its own digest"
            n_kept += 1
        else:
            assert got == digest_of(RECLAIM_KEY), f"key {key}: not a reclaimed digest"
            n_erased += 1
    assert n_erased > 0 and n_kept > 0
    # A slot nothing ever held is still empty.
    used = set(slot_list)
    for slot in range(BUCKET * 8):
        if slot not in used:
            assert int(digests[slot // BUCKET, slot % BUCKET]) == digest_of(EMPTY_KEY)


def test_the_binding_checks_the_tensors_it_indexes(device):
    """The wrapper passes the right ones; the binding does not take that on faith,
    since a wrong one is an out-of-bounds write on the GPU, found much later."""
    from dynamicemb_extensions import table_erase_by_score

    table = make_table([BUCKET * 4])
    counts = new_counts(device)
    real_counter = table._ref_counter

    def erase(bucket_sizes=None, counter=real_counter):
        table_erase_by_score(
            table.table_storage_,
            table.key_type_,
            BUCKET,
            table.bucket_sizes if bucket_sizes is None else bucket_sizes,
            counter,
            10,
            0,
            BUCKET * 4,
            1,
            0,
            counts,
        )

    erase()  # the real ones are fine
    erase(counter=None)  # no counter: nothing is pinned
    zeros = lambda n, dtype: torch.zeros(n, dtype=dtype, device=device)  # noqa: E731
    with pytest.raises(RuntimeError, match="bucket_sizes must be a contiguous int32"):
        erase(bucket_sizes=zeros(3, torch.int32))  # a bucket short
    with pytest.raises(RuntimeError, match="bucket_sizes must be a contiguous int32"):
        erase(bucket_sizes=zeros(4, torch.int64))
    with pytest.raises(RuntimeError, match="counter must be a contiguous int32"):
        erase(counter=zeros(BUCKET * 4 - 1, torch.int32))  # a slot short
    with pytest.raises(RuntimeError, match="counter must be a contiguous int32"):
        erase(counter=zeros(BUCKET * 4, torch.int64))
