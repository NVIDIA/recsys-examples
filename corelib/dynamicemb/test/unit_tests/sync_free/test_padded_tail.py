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

"""The padded tail of the unique op's outputs is inert.

``segmented_unique_cuda`` returns arrays sized at the upper bound ``N =
len(keys)``, of which only the first ``U = unique_table_offsets[num_tables]``
rows are unique keys (``U`` lives on the device and is never read back by a
caller that wants to stay asynchronous). The rows ``[U, N)`` are made
self-describing, so that code consuming all ``N`` rows needs no count:

* ``unique_keys[U:]`` is ``EmptyKey``: all ones, as an int64 and as a uint64
  key, a key ``Bucket::is_valid`` rejects, so every table kernel skips the row;
* the frequency of such a row is 0;
* ``expand_table_ids_cuda`` gives such a row table id -1.

These tests check that for both key types, with and without frequencies,
when ``N == U`` and when ``N > U``, with repeated keys within a table and the
same keys in several tables; that the real rows are what a CPU reference
computes; that the table kernels do ignore the tail; and that the
``DYNAMICEMB_DEBUG`` validation of ``segmented_range`` runs on the device.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

dynamicemb_extensions = pytest.importorskip("dynamicemb_extensions")

from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import (
    InsertResult,
    ScorePolicy,
    expand_table_ids_cuda,
    segmented_unique_cuda,
)

KEY_DTYPES = [torch.int64, torch.uint64]
# (input_frequencies, reduce): no frequencies at all, every occurrence counts 1,
# per-occurrence weights summed, and the largest / smallest weight of a key.
FREQ_MODES = [
    pytest.param(None, "sum", id="no-freq"),
    pytest.param("count", "sum", id="count"),
    pytest.param("weights", "sum", id="weights-sum"),
    pytest.param("weights", "max", id="weights-max"),
    pytest.param("weights", "min", id="weights-min"),
]


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def test_extension_announces_the_inert_tail():
    """Callers that fall back without the marked tails ask for this attribute
    (``getattr(dynamicemb_extensions, NAME, False)``): it names the unique op's
    key / frequency tail and the table id tail of ``expand_table_ids_cuda``."""
    assert hasattr(dynamicemb_extensions, "segmented_unique_tail_inert")
    assert dynamicemb_extensions.segmented_unique_tail_inert is True


def _make_keys(sizes, key_dtype, duplicates, seed=0):
    """Keys sorted by table, ``sizes[t]`` of them in table ``t``.

    ``duplicates`` is False (the keys of a table are all different), True (a
    table repeats keys: it draws from 11 values and its second key is forced to
    equal its first) or "one" (a table is one key, repeated). The tables share
    their keys in every mode, so the same key shows up in several tables, and a
    table may be empty. The keys are int64 bit patterns; a uint64 run sets the
    top bit of all of them (keys from 2**63 up) and views them as uint64. None
    of them is one of the four reserved keys.
    """
    gen = torch.Generator().manual_seed(seed)
    parts = []
    for size in sizes:
        if duplicates == "one":
            part = torch.full((size,), 5)
        elif duplicates:
            part = torch.randint(0, 11, (size,), generator=gen)
            if size >= 2:
                part[1] = part[0]
        else:
            part = torch.arange(size) * 3
        parts.append(part)
    bits = torch.cat(parts).to(torch.int64)
    if key_dtype == torch.uint64:
        bits = bits + torch.iinfo(torch.int64).min
    else:
        bits = bits + (1 << 40)
    keys = bits.to(_device())
    segmented_range = torch.zeros(len(sizes) + 1, dtype=torch.int64)
    segmented_range[1:] = torch.cumsum(torch.tensor(sizes), dim=0)
    keys = keys.view(torch.uint64) if key_dtype == torch.uint64 else keys
    return keys, segmented_range.to(_device())


def _make_weights(num_keys, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(1, 9, (num_keys,), generator=gen).to(_device())


def _reference(key_bits, sizes, weights, reduce):
    """Per table, ``{key: frequency}`` the unique op has to produce."""
    combine = {"sum": lambda a, b: a + b, "max": max, "min": min}[reduce]
    tables = []
    start = 0
    for size in sizes:
        freq = {}
        for i in range(start, start + size):
            key = key_bits[i]
            weight = 1 if weights is None else weights[i]
            freq[key] = combine(freq[key], weight) if key in freq else weight
        tables.append(freq)
        start += size
    return tables


def _run_unique(keys, segmented_range, num_tables, freq_mode, reduce):
    """``segmented_unique_cuda`` + ``expand_table_ids_cuda`` as the table code
    calls them, with the table ids covering every one of the ``N`` rows."""
    input_frequencies = None
    if freq_mode == "count":
        input_frequencies = torch.empty(0, dtype=torch.int64, device=keys.device)
    elif freq_mode == "weights":
        input_frequencies = _make_weights(keys.numel())
    num_uniques, unique_keys, reverse_indices, offsets, freq = segmented_unique_cuda(
        keys, segmented_range, num_tables, input_frequencies, reduce
    )
    table_ids = expand_table_ids_cuda(offsets, keys.numel())
    return (
        num_uniques,
        unique_keys,
        reverse_indices,
        offsets,
        freq,
        table_ids,
        input_frequencies,
    )


@pytest.mark.parametrize("key_dtype", KEY_DTYPES)
@pytest.mark.parametrize("freq_mode, reduce", FREQ_MODES)
@pytest.mark.parametrize(
    "sizes, duplicates",
    [
        # N == U: every key of a table is different (but tables share keys).
        pytest.param([40, 0, 25, 1, 33], False, id="all-unique"),
        # N > U: repeats within a table, shared keys across tables, an empty
        # table, a one-key table.
        pytest.param([40, 0, 25, 1, 60], True, id="duplicates"),
        # A table that is a single key repeated, one that is a lone key, an
        # empty one in between.
        pytest.param([6, 1, 0, 4], "one", id="one-key-per-table"),
        pytest.param([7, 1], True, id="small"),
    ],
)
def test_unique_outputs_are_padded_with_inert_rows(
    key_dtype, freq_mode, reduce, sizes, duplicates
):
    num_tables = len(sizes)
    keys, segmented_range = _make_keys(sizes, key_dtype, duplicates)
    (
        num_uniques,
        unique_keys,
        reverse_indices,
        offsets,
        freq,
        table_ids,
        input_frequencies,
    ) = _run_unique(keys, segmented_range, num_tables, freq_mode, reduce)
    torch.cuda.synchronize()

    num_keys = keys.numel()
    key_bits = keys.view(torch.int64).cpu().tolist()
    weights = input_frequencies.cpu().tolist() if freq_mode == "weights" else None
    expected = _reference(key_bits, sizes, weights, reduce)
    counts = [len(table) for table in expected]
    total_unique = sum(counts)
    if duplicates:
        assert total_unique < num_keys, "the case must have a tail"
    else:
        assert total_unique == num_keys, "the case must have no tail"

    # Shapes and dtypes: everything is sized at the upper bound.
    assert unique_keys.dtype == key_dtype and unique_keys.numel() == num_keys
    assert table_ids.dtype == torch.int64 and table_ids.numel() == num_keys
    if freq_mode is None:
        assert freq.numel() == 0
    else:
        assert freq.dtype == torch.int64 and freq.numel() == num_keys

    offsets_cpu = offsets.cpu().tolist()
    assert offsets_cpu == [0] + torch.cumsum(torch.tensor(counts), 0).tolist()
    assert int(num_uniques) == total_unique

    # The real rows: per table the keys (and their frequency) of the reference,
    # each once. Their order within a table is not specified.
    unique_bits = unique_keys.view(torch.int64).cpu()
    freq_cpu = freq.cpu()
    for t, table in enumerate(expected):
        lo, hi = offsets_cpu[t], offsets_cpu[t + 1]
        rows = unique_bits[lo:hi].tolist()
        assert len(rows) == len(set(rows)) == len(table)
        assert set(rows) == set(table)
        if freq_mode is not None:
            assert dict(zip(rows, freq_cpu[lo:hi].tolist())) == table

    # reverse_indices: each input key maps to the row holding it, in its table.
    reverse_cpu = reverse_indices.cpu()
    assert reverse_cpu.numel() == num_keys
    assert unique_bits[reverse_cpu].tolist() == key_bits
    start = 0
    for t, size in enumerate(sizes):
        rows = reverse_cpu[start : start + size]
        assert bool(((rows >= offsets_cpu[t]) & (rows < offsets_cpu[t + 1])).all())
        start += size

    # The padded tail: EmptyKey, frequency 0, table id -1; the real rows have a
    # table id and no reserved key.
    table_ids_cpu = table_ids.cpu()
    expected_ids = torch.repeat_interleave(
        torch.arange(num_tables), torch.tensor(counts)
    ).tolist()
    assert table_ids_cpu[:total_unique].tolist() == expected_ids
    assert bool((table_ids_cpu[total_unique:] == -1).all())
    assert bool((unique_bits[total_unique:] == -1).all())
    if freq_mode is not None:
        assert bool((freq_cpu[total_unique:] == 0).all())
    # The three markers agree on which rows are padding.
    in_tail = torch.arange(num_keys) >= total_unique
    assert torch.equal(table_ids_cpu == -1, in_tail)
    assert torch.equal(unique_bits == -1, in_tail)


def test_unique_tail_is_rewritten_over_stale_memory():
    """The outputs are allocated, not zeroed: the tail must be written, not
    merely happen to be empty. Run once to fill the allocator's cache with
    garbage (a batch with no repeats, then one with many), then check a batch
    that has a long tail."""
    device = _device()
    num_tables = 2
    sizes = [500, 500]
    dirty_keys, segmented_range = _make_keys(sizes, torch.int64, False)
    for _ in range(3):
        segmented_unique_cuda(dirty_keys, segmented_range, num_tables)
        expand_table_ids_cuda(segmented_range, dirty_keys.numel())
    torch.cuda.synchronize()

    keys = torch.full((1000,), 7, dtype=torch.int64, device=device)
    _, unique_keys, _, offsets, freq = segmented_unique_cuda(
        keys,
        segmented_range,
        num_tables,
        torch.empty(0, dtype=torch.int64, device=device),
    )
    table_ids = expand_table_ids_cuda(offsets, keys.numel())
    torch.cuda.synchronize()
    assert offsets.tolist() == [0, 1, 2]
    assert unique_keys[:2].tolist() == [7, 7]
    assert bool((unique_keys[2:] == -1).all())
    assert freq[:2].tolist() == [500, 500]
    assert bool((freq[2:] == 0).all())
    assert table_ids[:2].tolist() == [0, 1]
    assert bool((table_ids[2:] == -1).all())


def test_expand_table_ids_marks_rows_past_the_offsets():
    device = _device()
    # Table 1 is empty; the tables hold 7 elements.
    offsets = torch.tensor([0, 3, 3, 7], dtype=torch.int64, device=device)

    def expand(num_elements):
        return expand_table_ids_cuda(offsets, num_elements).tolist()

    # Up to offsets[-1] the answer is the plain expansion of the offsets.
    assert expand(7) == [0, 0, 0, 2, 2, 2, 2]
    assert expand(5) == [0, 0, 0, 2, 2]
    # Past it a row belongs to no table.
    assert expand(10) == [0, 0, 0, 2, 2, 2, 2, -1, -1, -1]
    assert expand(8) == [0, 0, 0, 2, 2, 2, 2, -1]
    assert expand_table_ids_cuda(offsets, 0).numel() == 0

    # Empty trailing tables put offsets[num_tables] before the last offsets'
    # plateau; a long padded array crosses many blocks.
    offsets = torch.tensor([0, 5, 5, 9, 9, 9], dtype=torch.int64, device=device)
    ids = expand_table_ids_cuda(offsets, 100_000).cpu()
    assert ids[:9].tolist() == [0] * 5 + [2] * 4
    assert bool((ids[9:] == -1).all())

    # No real elements at all.
    offsets = torch.zeros(4, dtype=torch.int64, device=device)
    assert bool((expand_table_ids_cuda(offsets, 6) == -1).all())


POLICIES = ["assign", "lru_lfu"]
CAPACITY = 128  # one bucket per table: a full table evicts


def _score_table(key_dtype, num_tables, capacity=CAPACITY, policy="assign", **kwargs):
    """Tables of one bucket each. Under ``assign`` a key's score is the value it
    is given and a full table evicts its lowest; ``lru_lfu`` is the two-word
    (timestamp, frequency) layout, whose inserts run in a JIT-compiled kernel."""
    if policy == "assign":
        spec = ScoreSpec(name="score1", policy=ScorePolicy.ASSIGN)
    else:
        spec = ScoreSpec(
            name="frequency", policy=ScorePolicy.LRU_LFU, is_reduction=True
        )
    return get_scored_table(
        capacity=[capacity] * num_tables,
        bucket_capacity=capacity,
        key_type=key_dtype,
        score_specs=[spec],
        **kwargs,
    )


def _score_arg(policy, scores):
    """``scores`` is the score of each key under ``assign``; under ``lru_lfu``
    every key is accessed once."""
    if policy == "assign":
        return ScoreArg(name="score1", value=scores)
    return ScoreArg(
        name="frequency", value=torch.ones_like(scores), policy=ScorePolicy.LRU_LFU
    )


def _const_score_arg(policy):
    name = "score1" if policy == "assign" else "frequency"
    return ScoreArg(name=name, policy=ScorePolicy.CONST)


def _insert_results(table, n):
    return torch.empty(n, dtype=table.result_type, device=_device()).fill_(
        InsertResult.INIT.value
    )


def _fill_two_tables(table, policy, key_dtype):
    """100 keys in each of two tables, score == key under ``assign``. Returns
    what is stored, as a set of ``(key, table id)``."""
    keys = torch.cat([torch.arange(1, 101), torch.arange(1000, 1100)])
    table_ids = torch.cat(
        [torch.zeros(100, dtype=torch.int64), torch.ones(100, dtype=torch.int64)]
    ).to(_device())
    keys = keys.to(_device()).to(key_dtype)
    table.insert(
        keys,
        table_ids,
        _score_arg(policy, keys.to(torch.uint64)),
        _insert_results(table, keys.numel()),
    )
    torch.cuda.synchronize()  # what follows is accessed later than these
    return set(zip(keys.view(torch.int64).tolist(), table_ids.tolist()))


def _padded_batch(key_dtype, sizes):
    """Deduplicated and expanded as the table code does it: ``(unique_keys,
    table_ids, U)`` with ``unique_keys`` and ``table_ids`` of ``sum(sizes)`` rows.
    Each table gets 40 different new keys, each repeated, in random order."""
    gen = torch.Generator().manual_seed(3)
    parts = [
        (first + torch.arange(size) % 40)[torch.randperm(size, generator=gen)]
        for first, size in zip((5000, 6000), sizes)
    ]
    keys = torch.cat(parts).to(_device()).to(key_dtype)
    segmented_range = torch.tensor([0, sizes[0], sum(sizes)], device=_device())
    _, unique_keys, _, offsets, _ = segmented_unique_cuda(keys, segmented_range, 2)
    table_ids = expand_table_ids_cuda(offsets, keys.numel())
    total_unique = int(offsets[-1])
    assert total_unique == 2 * 40 < keys.numel(), "the batch must have a tail"
    return unique_keys, table_ids, total_unique


def _high_scores(num_rows):
    """Far above every stored score, so under ``assign`` every new key wins its
    slot. Given for the tail rows too, as the kernels read the score of a row
    before they look at its key."""
    return torch.full((num_rows,), 10_000_000, dtype=torch.uint64, device=_device())


@pytest.mark.parametrize("key_dtype", KEY_DTYPES)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("overflow", [False, True], ids=["plain", "overflow"])
@pytest.mark.parametrize(
    "sizes",
    [
        # 130 rows: the table kernels compact the evictions per thread.
        pytest.param([60, 70], id="130-rows"),
        # 128 rows, a multiple of 32: the kernels compact them per tile of 32
        # lanes, and the lanes of the padding rows take part in the ballot.
        pytest.param([64, 64], id="128-rows"),
    ],
)
def test_insert_and_evict_ignores_the_padded_tail(key_dtype, policy, overflow, sizes):
    """The padded arrays go straight to the table kernels: the rows past ``U``
    insert nothing, evict nothing and report index -1, and the real rows behave
    as they do when handed over alone (the exact-size run). ``overflow`` is the
    cache's variant, with an overflow tier behind each table."""
    padded = _score_table(key_dtype, 2, policy=policy, enable_overflow=overflow)
    exact = _score_table(key_dtype, 2, policy=policy, enable_overflow=overflow)
    for table in (padded, exact):
        _fill_two_tables(table, policy, key_dtype)
    unique_keys, table_ids, total_unique = _padded_batch(key_dtype, sizes)
    num_rows = unique_keys.numel()
    scores = _high_scores(num_rows)
    method = (
        "insert_and_evict_with_counter_and_overflow" if overflow else "insert_and_evict"
    )

    results_padded = _insert_results(padded, num_rows)
    results_exact = _insert_results(exact, total_unique)
    out_padded = getattr(padded, method)(
        unique_keys, table_ids, _score_arg(policy, scores), results_padded
    )
    out_exact = getattr(exact, method)(
        unique_keys[:total_unique],
        table_ids[:total_unique],
        _score_arg(policy, scores[:total_unique]),
        results_exact,
    )
    indices_p, num_evicted_p, evicted_keys_p, _, _, evicted_ids_p = out_padded
    _, num_evicted_e, evicted_keys_e, _, _, evicted_ids_e = out_exact
    torch.cuda.synchronize()

    # The tail: not inserted, index -1, "illegal"; the real rows all inserted.
    assert bool((indices_p[total_unique:] == -1).all())
    assert bool((indices_p[:total_unique] >= 0).all())
    assert bool((results_padded[total_unique:] == InsertResult.ILLEGAL.value).all())
    assert bool((results_padded[:total_unique] != InsertResult.ILLEGAL.value).all())

    # Same evictions as the exact-size run: 2 * (100 + 40 - 128) of them, and
    # none for the tail rows. Under ``assign`` they are the lowest scores of
    # each table, which makes them the same keys in both runs.
    assert num_evicted_p == num_evicted_e == 2 * (100 + 40 - CAPACITY)
    victims_p = set(
        zip(evicted_keys_p.view(torch.int64).tolist(), evicted_ids_p.tolist())
    )
    assert len(victims_p) == num_evicted_p
    assert all(key != -1 and tid in (0, 1) for key, tid in victims_p)
    if policy == "assign":
        victims_e = set(
            zip(evicted_keys_e.view(torch.int64).tolist(), evicted_ids_e.tolist())
        )
        assert victims_p == victims_e

    # And the tables hold the same number of keys afterwards, the new ones among
    # them. A padded lookup finds the real keys and nothing in the tail.
    assert padded.size() == exact.size()
    _, founds, indices = padded.lookup(unique_keys, table_ids, _const_score_arg(policy))
    assert bool(founds[:total_unique].all()) and not bool(founds[total_unique:].any())
    assert bool((indices[total_unique:] == -1).all())


@pytest.mark.parametrize("key_dtype", KEY_DTYPES)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("with_score_out", [False, True], ids=["no-score", "score"])
def test_collect_evicted_ignores_the_padded_tail(key_dtype, policy, with_score_out):
    """The plain insert that collects the evicted keys, with and without the
    optional score output (the JIT kernel decides at run time whether to write
    it)."""
    padded = _score_table(key_dtype, 2, policy=policy)
    exact = _score_table(key_dtype, 2, policy=policy)
    stored = _fill_two_tables(padded, policy, key_dtype)
    _fill_two_tables(exact, policy, key_dtype)
    unique_keys, table_ids, total_unique = _padded_batch(key_dtype, [60, 70])
    num_rows = unique_keys.numel()
    scores = _high_scores(num_rows)

    results = _insert_results(padded, num_rows)
    score_out = torch.empty(num_rows, dtype=torch.int64, device=_device())
    indices, num_evicted, evicted_keys, evicted_ids = padded.insert(
        unique_keys,
        table_ids,
        _score_arg(policy, scores),
        results,
        score_out=score_out if with_score_out else None,
        collect_evicted=True,
    )
    _, num_evicted_e, evicted_keys_e, evicted_ids_e = exact.insert(
        unique_keys[:total_unique],
        table_ids[:total_unique],
        _score_arg(policy, scores[:total_unique]),
        _insert_results(exact, total_unique),
        collect_evicted=True,
    )
    torch.cuda.synchronize()

    assert bool((indices[total_unique:] == -1).all())
    assert bool((indices[:total_unique] >= 0).all())
    assert bool((results[total_unique:] == InsertResult.ILLEGAL.value).all())
    count = int(num_evicted)
    assert count == int(num_evicted_e) == 2 * (100 + 40 - CAPACITY)
    assert count == int((results == InsertResult.EVICT.value).sum())
    # The collected victims are stored keys; the entries past them are marked.
    victims = set(
        zip(
            evicted_keys[:count].view(torch.int64).tolist(),
            evicted_ids[:count].tolist(),
        )
    )
    assert victims <= stored and len(victims) == count
    assert bool((evicted_keys.view(torch.int64)[count:] == -1).all())
    assert bool((evicted_ids[count:] == -1).all())
    if policy == "assign":
        victims_e = set(
            zip(
                evicted_keys_e[:count].view(torch.int64).tolist(),
                evicted_ids_e[:count].tolist(),
            )
        )
        assert victims == victims_e


@pytest.mark.parametrize("key_dtype", KEY_DTYPES)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("with_score_out", [False, True], ids=["no-score", "score"])
def test_plain_insert_ignores_the_padded_tail(key_dtype, policy, with_score_out):
    num_tables = 3
    table = _score_table(key_dtype, num_tables, 1024, policy)
    keys, segmented_range = _make_keys([30, 0, 50], key_dtype, True)
    _, unique_keys, _, offsets, _ = segmented_unique_cuda(
        keys, segmented_range, num_tables
    )
    table_ids = expand_table_ids_cuda(offsets, keys.numel())
    total_unique = int(offsets[-1])
    num_rows = keys.numel()
    assert total_unique < num_rows

    scores = torch.arange(1, num_rows + 1, device=_device()).to(torch.uint64)
    results = _insert_results(table, num_rows)
    score_out = torch.empty(num_rows, dtype=torch.int64, device=_device())
    indices = table.insert(
        unique_keys,
        table_ids,
        _score_arg(policy, scores),
        results,
        score_out=score_out if with_score_out else None,
    )
    torch.cuda.synchronize()
    assert bool((indices[:total_unique] >= 0).all())
    assert bool((indices[total_unique:] == -1).all())
    assert bool((results[:total_unique] == InsertResult.INSERT.value).all())
    assert bool((results[total_unique:] == InsertResult.ILLEGAL.value).all())
    assert table.size() == total_unique


@pytest.mark.parametrize("key_dtype", KEY_DTYPES)
@pytest.mark.parametrize("policy", POLICIES)
def test_lookup_erase_and_counters_ignore_the_padded_tail(key_dtype, policy):
    num_tables = 3
    table = _score_table(key_dtype, num_tables, 1024, policy)
    keys, segmented_range = _make_keys([30, 0, 50], key_dtype, True)
    _, unique_keys, _, offsets, _ = segmented_unique_cuda(
        keys, segmented_range, num_tables
    )
    table_ids = expand_table_ids_cuda(offsets, keys.numel())
    total_unique = int(offsets[-1])
    num_rows = keys.numel()
    assert total_unique < num_rows
    scores = torch.arange(1, num_rows + 1, device=_device()).to(torch.uint64)
    table.insert(
        unique_keys[:total_unique],
        table_ids[:total_unique],
        _score_arg(policy, scores[:total_unique]),
    )

    # A lookup that accesses the keys (it assigns the given score, or counts one
    # more access): found for the real rows only, -1 for the tail.
    access = (
        ScoreArg(name="score1", value=scores, policy=ScorePolicy.ASSIGN)
        if policy == "assign"
        else ScoreArg(
            name="frequency",
            value=torch.ones_like(scores),
            policy=ScorePolicy.LRU_LFU,
        )
    )
    score_out, founds, slots = table.lookup(unique_keys, table_ids, access)
    torch.cuda.synchronize()
    assert bool(founds[:total_unique].all()) and not bool(founds[total_unique:].any())
    assert bool((slots[:total_unique] >= 0).all())
    assert bool((slots[total_unique:] == -1).all())
    expected = scores[:total_unique].long() if policy == "assign" else 2
    assert bool((score_out[:total_unique] == expected).all())

    # The reference counter: the real rows are counted, the tail is not.
    table.increment_counter(slots, table_ids)
    assert int(table._ref_counter.sum()) == total_unique
    table.decrement_counter(slots, table_ids)
    assert int(table._ref_counter.abs().sum()) == 0

    # Erase: the real rows go, the tail reports -1.
    erased = table.erase(unique_keys, table_ids, return_indices=True)
    assert bool((erased[:total_unique] >= 0).all())
    assert bool((erased[total_unique:] == -1).all())
    assert table.size() == 0


@pytest.mark.parametrize("key_dtype", KEY_DTYPES)
def test_row_resolution_ignores_the_padded_tail(key_dtype):
    """A table that stores a row per key resolves slots to rows: -1 stays -1 and
    takes no row, whatever the table id of its padded row."""
    num_tables = 3
    table = _score_table(key_dtype, num_tables, 1024, "assign", use_rows=True)
    keys, segmented_range = _make_keys([30, 0, 50], key_dtype, True)
    _, unique_keys, _, offsets, _ = segmented_unique_cuda(
        keys, segmented_range, num_tables
    )
    table_ids = expand_table_ids_cuda(offsets, keys.numel())
    offsets_cpu = offsets.tolist()
    total_unique = offsets_cpu[-1]
    num_rows = keys.numel()
    scores = torch.arange(1, num_rows + 1, device=_device()).to(torch.uint64)

    rows = table.insert(unique_keys, table_ids, _score_arg("assign", scores))
    torch.cuda.synchronize()
    assert bool((rows[total_unique:] == -1).all())
    # Rows are handed out densely, per table, to the real keys only.
    for t in range(num_tables):
        count = offsets_cpu[t + 1] - offsets_cpu[t]
        assert table.num_rows(t) == count
        assert sorted(rows[offsets_cpu[t] : offsets_cpu[t + 1]].tolist()) == list(
            range(count)
        )


def _subprocess_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    env["DYNAMICEMB_DEBUG"] = "1"
    return env


def test_debug_validation_accepts_a_valid_range(monkeypatch):
    """With ``DYNAMICEMB_DEBUG`` set a correct ``segmented_range`` goes through
    unchanged: the check is a kernel, and costs no error."""
    monkeypatch.setenv("DYNAMICEMB_DEBUG", "1")
    sizes = [20, 0, 35]
    keys, segmented_range = _make_keys(sizes, torch.int64, True)
    num_uniques, unique_keys, reverse_indices, offsets, _ = segmented_unique_cuda(
        keys, segmented_range, len(sizes)
    )
    torch.cuda.synchronize()
    total_unique = int(num_uniques)
    assert total_unique == int(offsets[-1]) > 0
    assert unique_keys[reverse_indices].tolist() == keys.tolist()
    # An empty batch is a valid range too.
    empty = torch.empty(0, dtype=torch.int64, device=keys.device)
    segmented_unique_cuda(empty, torch.zeros(4, dtype=torch.int64, device="cuda"), 3)
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "segmented_range, message",
    [
        pytest.param([1, 4, 8], "segmented_range[0] must be 0", id="not-from-0"),
        pytest.param(
            [0, 4, 7], "segmented_range[num_tables] must equal num_keys", id="short"
        ),
        pytest.param([0, 6, 4, 8], "must be non-decreasing", id="decreasing"),
    ],
)
def test_debug_validation_asserts_on_the_device(segmented_range, message):
    """A wrong ``segmented_range`` (for 8 keys) trips a device-side assert that
    names the condition. The CUDA context is unusable afterwards, so this runs
    in a subprocess."""
    code = textwrap.dedent(
        f"""
        import torch
        from dynamicemb_extensions import segmented_unique_cuda

        keys = torch.arange(8, dtype=torch.int64, device="cuda")
        bounds = torch.tensor({segmented_range}, dtype=torch.int64, device="cuda")
        segmented_unique_cuda(keys, bounds, {len(segmented_range) - 1})
        torch.cuda.synchronize()
        print("NO ERROR")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, output
    assert "NO ERROR" not in proc.stdout
    assert "Assertion" in output and message in output, output
