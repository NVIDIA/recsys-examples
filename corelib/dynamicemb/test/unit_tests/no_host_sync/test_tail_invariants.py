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

"""The padded unique arrays of the sync-free lookup paths, and what must hold of their tail.

A sync-free step never reads the number U of unique keys. The unique arrays keep the length
N of the batch: rows [0, U) are the head (real), rows [U, N) the tail, and the tail is inert.
A key that must not take part in a step is the reserved all-ones key EMPTY (-1 as int64,
0xFF..FF as uint64), its table id is -1, and every table kernel treats a reserved key as
not found, index -1, InsertResult.ILLEGAL, WITHOUT reading its table id. This file tests that
convention:

* the head of ``segmented_unique_cuda`` and ``expand_table_ids_cuda`` against a CPU reference
  (int64 and uint64 keys, 1-4 tables, empty tables, duplicates, with and without frequency
  counts), on any tree;
* the tail those kernels write, where the extension says it makes the tail inert
  (``segmented_unique_tail_inert``; before that the tail is unspecified and this part skips);
* ``sync_free.canonical_unique`` making any tail inert, with ``TAIL_INERT`` on and off, and
  the cross-array invariant ``keys[i] == EMPTY <=> ids[i] == -1 <=> not valid[i] <=> i >= U``;
* the fallback of ``canonical_unique`` on the tail the real kernels leave, whether or not the
  extension made it inert;
* the TAIL-SCRIBBLE tests: reserved keys with garbage table ids (in range, past the last table,
  huge, negative) change nothing in a table, its counters or its results (the table kernels never
  read the id of a reserved key), and the masked lookup paths run correctly, and insert and pin
  nothing, when the tail they receive carries other table ids than -1 and, for the cache and
  HBM-direct paths, keys that look live and that no table holds (only ``valid`` says they are not
  real: a path that does not AND it into its miss mask inserts them).

The scribbled tail keeps the ids that the masked paths receive in range: the paths gather
per-table arrays by the ids of every row (``table_emb_dims[ids]``), which wraps -1 into range but
faults on an id past the last table. A tail key that exists in a table is not a case the design
supports (its lookup would pin the slot, and the id -1 that the canonical arrays carry for it
would then be read by the counter kernel), and the DEFAULT paths do not take the live-key
variant at all (see SCRIBBLES).
Tests that look at the kernels or at ``sync_free`` skip where those do not exist yet.
"""
import os
import sys
from typing import List, NamedTuple, Optional, Tuple

import pytest

try:
    import torch
except ImportError:
    pytest.skip("needs torch", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
try:
    import dynamicemb_extensions as ext
    from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
    from dynamicemb_extensions import InsertResult, ScorePolicy
except ImportError as e:
    pytest.skip("needs the built dynamicemb: %s" % (e,), allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import host_sync_probe as hsp  # noqa: E402

EMPTY = -1  # the reserved all-ones key, as int64
TAIL_INERT_EXT = bool(getattr(ext, "segmented_unique_tail_inert", False))
KEY_DTYPES = [torch.int64, torch.uint64]
# (number of tables, the ones that get no key)
TABLE_LAYOUTS = [(1, ()), (2, (1,)), (3, ()), (4, (0, 2))]
BATCH_SIZES = [1, 2, 17, 1000, 100000]
FREQ_MODES = [
    None,
    "count",
    "custom",
]  # no counts, every occurrence counts 1, caller's counts (2)


def as_i64(t: "torch.Tensor") -> "torch.Tensor":
    """A key tensor as int64 (uint64 has no gather or where on CUDA)."""
    return t.view(torch.int64) if t.dtype == torch.uint64 else t


class Batch(NamedTuple):
    keys: "torch.Tensor"  # [N] on the GPU, sorted by table
    segment_range: "torch.Tensor"  # [T+1] int64 on the GPU
    num_tables: int
    n: int
    key_dtype: "torch.dtype"
    tables: List[List[int]]  # the keys of each table, as int64 bit patterns
    freq_in: Optional["torch.Tensor"]
    freq_mult: int  # what each occurrence counts for in the unique key's frequency; 0: no counts


def make_batch(
    num_tables, n, key_dtype, seed, empty_tables=(), freq_mode=None
) -> Batch:
    """n keys of `num_tables` tables, sorted by table (the segmented_unique contract), with
    about half of the draws repeated and the same key space in every table. The keys of a
    uint64 batch have the top bit set, none is one of the four reserved keys."""
    g = torch.Generator().manual_seed(seed)
    weights = torch.rand(num_tables, generator=g) + 0.1
    for t in empty_tables:
        weights[t] = 0.0
    counts = torch.floor(weights / weights.sum() * n).to(torch.int64)
    counts[int(weights.argmax())] += n - int(counts.sum())
    tables, chunks = [], []
    for t in range(num_tables):
        c = int(counts[t])
        draws = torch.randint(
            0, max(1, (c + 1) // 2), (c,), generator=g, dtype=torch.int64
        )
        keys = draws * 2654435761 + 17  # < 2^62 for every batch size here
        if key_dtype == torch.uint64:
            keys = (
                keys + torch.iinfo(torch.int64).min
            )  # sets bit 63, stays clear of the reserved keys
        tables.append(keys.tolist())
        chunks.append(keys)
    keys = torch.cat(chunks).cuda().view(key_dtype)
    segment_range = torch.zeros(num_tables + 1, dtype=torch.int64)
    segment_range[1:] = torch.cumsum(counts, 0)
    freq_in = {
        None: None,
        "count": torch.empty(0, dtype=torch.int64, device="cuda"),
        "custom": torch.full((n,), 2, dtype=torch.int64, device="cuda"),
    }[freq_mode]
    freq_mult = {None: 0, "count": 1, "custom": 2}[freq_mode]
    return Batch(
        keys, segment_range.cuda(), num_tables, n, key_dtype, tables, freq_in, freq_mult
    )


class Raw(NamedTuple):
    num_uniques: "torch.Tensor"
    unique_keys: "torch.Tensor"
    reverse_indices: "torch.Tensor"
    offsets: "torch.Tensor"
    freq: "torch.Tensor"
    table_ids: "torch.Tensor"


def run_kernels(b: Batch) -> Raw:
    (
        num_uniques,
        unique_keys,
        reverse_indices,
        offsets,
        freq,
    ) = ext.segmented_unique_cuda(b.keys, b.segment_range, b.num_tables, b.freq_in)
    table_ids = ext.expand_table_ids_cuda(offsets, b.n)
    torch.cuda.synchronize()
    return Raw(num_uniques, unique_keys, reverse_indices, offsets, freq, table_ids)


def check_head(b: Batch, raw: Raw) -> int:
    """The head of the raw outputs against a CPU reference; returns U."""
    ref_unique = [sorted(set(t)) for t in b.tables]
    ref_offsets = [0]
    for u in ref_unique:
        ref_offsets.append(ref_offsets[-1] + len(u))
    offsets = raw.offsets.cpu().tolist()
    u = offsets[-1]
    assert offsets == ref_offsets, "per-table unique offsets"
    assert int(raw.num_uniques.cpu()[0]) == u
    unique_keys = as_i64(raw.unique_keys).cpu()
    assert unique_keys.numel() == b.n, "the unique arrays have the length of the batch"
    for t in range(b.num_tables):
        got = sorted(unique_keys[offsets[t] : offsets[t + 1]].tolist())
        assert got == ref_unique[t], "unique keys of table %d" % t
    reverse = raw.reverse_indices.cpu()
    assert reverse.numel() == b.n
    assert (
        int(reverse.min()) >= 0 and int(reverse.max()) < u
    ), "every reverse index points into the head"
    assert torch.equal(
        unique_keys[reverse], as_i64(b.keys).cpu()
    ), "unique_keys[reverse_indices] is the input"
    ids = raw.table_ids.cpu()
    assert ids.numel() == b.n
    counts = torch.tensor(offsets[1:]) - torch.tensor(offsets[:-1])
    assert torch.equal(
        ids[:u], torch.repeat_interleave(torch.arange(b.num_tables), counts)
    ), "table ids of the head"
    if b.freq_mult:
        assert raw.freq.numel() == b.n
        assert torch.equal(
            raw.freq.cpu()[:u], b.freq_mult * torch.bincount(reverse, minlength=u)
        ), "frequencies"
    else:
        assert raw.freq.numel() == 0
    return u


def canonical_by_hand(
    raw: Raw, u: int
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """The arrays of the padded convention built from the raw ones by the test (it knows U)."""
    valid = torch.arange(raw.unique_keys.numel(), device="cuda") < u
    keys = torch.where(valid, as_i64(raw.unique_keys), EMPTY).view(
        raw.unique_keys.dtype
    )
    return keys, torch.where(valid, raw.table_ids, EMPTY), valid


def check_cross_array_invariant(keys, ids, valid, u: int) -> None:
    """keys[i] == EMPTY <=> ids[i] == -1 <=> not valid[i] <=> i >= U."""
    tail = torch.arange(keys.numel(), device="cuda") >= u
    assert torch.equal(as_i64(keys) == EMPTY, tail)
    assert torch.equal(ids == EMPTY, tail)
    assert torch.equal(~valid, tail)


MATRIX = [
    pytest.param(
        t,
        n,
        d,
        f,
        id="%dT%s-n%d-%s-%s"
        % (t[0], "".join(map(str, t[1])), n, str(d).split(".")[1], f),
    )
    for t in TABLE_LAYOUTS
    for n in BATCH_SIZES
    for d in KEY_DTYPES
    for f in FREQ_MODES
]


# --------------------------------------------------------------------------
# the kernels
# --------------------------------------------------------------------------
@pytest.mark.parametrize("layout, n, key_dtype, freq_mode", MATRIX)
def test_the_head_of_the_unique_arrays_matches_a_cpu_reference(
    layout, n, key_dtype, freq_mode
):
    b = make_batch(
        layout[0],
        n,
        key_dtype,
        seed=n + layout[0],
        empty_tables=layout[1],
        freq_mode=freq_mode,
    )
    check_head(b, run_kernels(b))


@pytest.mark.skipif(
    not TAIL_INERT_EXT,
    reason="the extension does not make the tail inert yet (report mode)",
)
@pytest.mark.parametrize("layout, n, key_dtype, freq_mode", MATRIX)
def test_the_kernels_write_an_inert_tail(layout, n, key_dtype, freq_mode):
    """segmented_unique_tail_inert: unique_keys[U:] is EMPTY, the frequencies there are 0 and
    expand_table_ids writes -1, not the number of tables, on the tail."""
    b = make_batch(
        layout[0],
        n,
        key_dtype,
        seed=n + layout[0],
        empty_tables=layout[1],
        freq_mode=freq_mode,
    )
    raw = run_kernels(b)
    u = check_head(b, raw)
    assert (as_i64(raw.unique_keys).cpu()[u:] == EMPTY).all()
    assert (raw.table_ids.cpu()[u:] == EMPTY).all()
    if b.freq_mult:
        assert (raw.freq.cpu()[u:] == 0).all()
    keys, ids, valid = raw.unique_keys, raw.table_ids, raw.table_ids >= 0
    assert int(valid.sum()) == u
    check_cross_array_invariant(keys, ids, valid, u)


def test_expand_table_ids_over_exactly_u_rows_is_unchanged():
    """The eval sequence path and test_expand_table_ids call it with num_elements == U."""
    b = make_batch(3, 1000, torch.int64, seed=7, empty_tables=(1,))
    raw = run_kernels(b)
    u = check_head(b, raw)
    ids = ext.expand_table_ids_cuda(raw.offsets, u).cpu()
    assert torch.equal(ids, raw.table_ids.cpu()[:u])


# --------------------------------------------------------------------------
# sync_free.canonical_unique
# --------------------------------------------------------------------------
CANONICAL = [
    pytest.param(
        t,
        n,
        d,
        id="%dT%s-n%d-%s" % (t[0], "".join(map(str, t[1])), n, str(d).split(".")[1]),
    )
    for t in TABLE_LAYOUTS
    for n in (1, 17, 1000)
    for d in KEY_DTYPES
]


def scribble_like_an_unpatched_kernel(raw: Raw, u: int, num_tables: int):
    """The arrays an extension without the tail guard returns: live-looking keys past U (the
    output is at::empty), and `num_tables` as the table id there."""
    keys = as_i64(raw.unique_keys).clone()
    ids = raw.table_ids.clone()
    keys[u:] = torch.randint(
        0, 1 << 40, (keys.numel() - u,), dtype=torch.int64, device="cuda"
    )
    ids[u:] = num_tables
    return keys.view(raw.unique_keys.dtype), ids


@pytest.mark.parametrize("layout, n, key_dtype", CANONICAL)
def test_canonical_unique_makes_the_tail_of_any_kernel_output_inert(
    sf, monkeypatch, layout, n, key_dtype
):
    """TAIL_INERT False: the fallback rewrites the tail the unpatched kernels leave behind."""
    monkeypatch.setattr(sf, "TAIL_INERT", False)
    b = make_batch(layout[0], n, key_dtype, seed=n, empty_tables=layout[1])
    raw = run_kernels(b)
    u = check_head(b, raw)
    keys_in, ids_in = scribble_like_an_unpatched_kernel(raw, u, b.num_tables)
    keys, ids, valid = sf.canonical_unique(keys_in, ids_in, b.num_tables)
    assert (
        keys.dtype == key_dtype
        and keys.shape == keys_in.shape
        and ids.shape == ids_in.shape
    )
    assert torch.equal(
        as_i64(keys)[:u], as_i64(raw.unique_keys)[:u]
    ), "the head keys are kept"
    assert torch.equal(ids[:u], raw.table_ids[:u]), "the head table ids are kept"
    check_cross_array_invariant(keys, ids, valid, u)


@pytest.mark.parametrize("layout, n, key_dtype", CANONICAL)
def test_canonical_unique_trusts_a_kernel_that_made_the_tail_inert(
    sf, monkeypatch, layout, n, key_dtype
):
    """TAIL_INERT True: only `valid` is computed, from the table ids."""
    monkeypatch.setattr(sf, "TAIL_INERT", True)
    b = make_batch(layout[0], n, key_dtype, seed=n, empty_tables=layout[1])
    raw = run_kernels(b)
    u = check_head(b, raw)
    keys_in, ids_in, _ = canonical_by_hand(raw, u)
    keys, ids, valid = sf.canonical_unique(keys_in, ids_in, b.num_tables)
    assert torch.equal(as_i64(keys), as_i64(keys_in)) and torch.equal(ids, ids_in)
    check_cross_array_invariant(keys, ids, valid, u)


@pytest.mark.skipif(
    not TAIL_INERT_EXT,
    reason="the extension does not make the tail inert yet (report mode)",
)
@pytest.mark.parametrize("layout, n, key_dtype", CANONICAL)
def test_canonical_unique_of_the_real_kernel_output(sf, layout, n, key_dtype):
    assert (
        sf.TAIL_INERT
    ), "the extension announces an inert tail, sync_free must mirror it"
    b = make_batch(layout[0], n, key_dtype, seed=n, empty_tables=layout[1])
    raw = run_kernels(b)
    u = check_head(b, raw)
    keys, ids, valid = sf.canonical_unique(raw.unique_keys, raw.table_ids, b.num_tables)
    check_cross_array_invariant(keys, ids, valid, u)


REAL_KERNEL_RUNS = [
    pytest.param(
        t,
        n,
        d,
        id="%dT%s-n%d-%s" % (t[0], "".join(map(str, t[1])), n, str(d).split(".")[1]),
    )
    for t in TABLE_LAYOUTS
    for n in BATCH_SIZES
    for d in KEY_DTYPES
]


@pytest.mark.parametrize("layout, n, key_dtype", REAL_KERNEL_RUNS)
def test_the_fallback_of_canonical_unique_on_the_tail_the_kernels_really_leave(
    sf, monkeypatch, layout, n, key_dtype
):
    """TAIL_INERT False, on the arrays the kernels return, whichever tail that is: the live-looking
    keys and `num_tables` of an extension without the tail guard, or the EMPTY keys and -1 of one
    with it (a test, or a deployment, that turns the capability off on a rebuilt extension). The
    fallback has to class the id -1 of a patched tail as not a table too: `ids < num_tables` alone
    holds for -1. With a patched extension the result is bit-identical to the TAIL_INERT one.
    """
    b = make_batch(layout[0], n, key_dtype, seed=n + layout[0], empty_tables=layout[1])
    raw = run_kernels(b)
    u = check_head(b, raw)
    monkeypatch.setattr(sf, "TAIL_INERT", False)
    keys, ids, valid = sf.canonical_unique(
        raw.unique_keys.clone(), raw.table_ids.clone(), b.num_tables
    )
    assert keys.dtype == key_dtype and keys.shape == raw.unique_keys.shape
    assert torch.equal(
        as_i64(keys)[:u], as_i64(raw.unique_keys)[:u]
    ), "the head keys are kept"
    assert torch.equal(ids[:u], raw.table_ids[:u]), "the head table ids are kept"
    check_cross_array_invariant(keys, ids, valid, u)
    if not TAIL_INERT_EXT:
        return  # the kernels' tail is not inert: nothing to compare the fallback with
    monkeypatch.setattr(sf, "TAIL_INERT", True)
    keys_k, ids_k, valid_k = sf.canonical_unique(
        raw.unique_keys.clone(), raw.table_ids.clone(), b.num_tables
    )
    assert torch.equal(as_i64(keys), as_i64(keys_k)), "keys: fallback == kernel tail"
    assert torch.equal(ids, ids_k), "table ids: fallback == kernel tail"
    assert torch.equal(valid, valid_k), "valid: fallback == kernel tail"


# --------------------------------------------------------------------------
# the table kernels and the tail
# --------------------------------------------------------------------------
def build_cache_table(key_dtype, num_tables, largest_table: int):
    capacity = 4096
    while capacity < 3 * max(1, largest_table):
        capacity *= 2
    return get_scored_table(
        capacity=[capacity] * num_tables,
        bucket_capacity=128,
        key_type=key_dtype,
        score_specs=[ScoreSpec(name="score", policy=ScorePolicy.ASSIGN)],
        enable_overflow=True,
    )


def score_of(n: int) -> "ScoreArg":
    return ScoreArg(
        name="score", value=torch.ones(n, dtype=torch.uint64, device="cuda")
    )


def const_score() -> "ScoreArg":
    return ScoreArg(name="score", policy=ScorePolicy.CONST)


def pad_with(keys, ids, key_dtype, tail_ids):
    """keys/ids followed by EMPTY keys that carry `tail_ids` as their table ids."""
    tail_keys = torch.full((tail_ids.numel(),), EMPTY, dtype=torch.int64, device="cuda")
    return torch.cat([as_i64(keys), tail_keys]).view(key_dtype), torch.cat(
        [ids, tail_ids]
    )


def garbage_ids(num_tables: int, count: int):
    """Table ids that no reserved key may be read through: valid, the last, one past it, huge, negative."""
    pool = [0, num_tables - 1, num_tables, 1 << 40, -7, 12345, -(1 << 40)]
    return torch.tensor(
        [pool[i % len(pool)] for i in range(count)], dtype=torch.int64, device="cuda"
    )


@pytest.mark.parametrize("layout", [(1, ()), (3, (1,)), (4, (0, 2))])
@pytest.mark.parametrize("n", [17, 1000, 100000])
@pytest.mark.parametrize("key_dtype", KEY_DTYPES)
@pytest.mark.parametrize("source", ["by_hand", "sync_free"])
def test_the_tail_pins_nothing(request, layout, n, key_dtype, source):
    """cache.lookup + increment_counter over the padded arrays leave every counter where the
    exact arrays leave it, and decrementing with the same arrays brings them back to zero.
    """
    b = make_batch(layout[0], n, key_dtype, seed=n, empty_tables=layout[1])
    raw = run_kernels(b)
    u = check_head(b, raw)
    if source == "by_hand":
        keys, ids, _ = canonical_by_hand(raw, u)
    else:
        sf = request.getfixturevalue("sf")
        keys, ids, _ = sf.canonical_unique(raw.unique_keys, raw.table_ids, b.num_tables)
    table = build_cache_table(
        key_dtype, b.num_tables, max(len(set(t)) for t in b.tables)
    )
    results = torch.empty(u, dtype=table.result_type, device="cuda")
    table.insert(raw.unique_keys[:u], raw.table_ids[:u], score_of(u), results)
    assert (results == InsertResult.INSERT.value).all()

    _, found_ref, index_ref = table.lookup_with_overflow(
        raw.unique_keys[:u], raw.table_ids[:u], const_score()
    )
    assert found_ref.all()
    table.increment_counter(index_ref, raw.table_ids[:u])
    counters_ref = table._ref_counter.clone()
    table.decrement_counter(index_ref, raw.table_ids[:u])
    assert (table._ref_counter == 0).all()

    _, found, index = table.lookup_with_overflow(keys, ids, const_score())
    assert found[:u].all() and not found[u:].any(), "found: the head, never the tail"
    assert torch.equal(index[:u], index_ref) and (index[u:] == EMPTY).all()
    table.increment_counter(index, ids)
    assert torch.equal(
        table._ref_counter, counters_ref
    ), "the counters of the padded step"
    table.decrement_counter(index, ids)
    assert (table._ref_counter == 0).all()


@pytest.mark.parametrize("key_dtype", KEY_DTYPES)
def test_scribbled_table_ids_on_reserved_keys_change_nothing(key_dtype):
    """TAIL-SCRIBBLE at the table kernels: reserved keys never read their table id."""
    num_tables, head = 3, 60
    keys = (torch.arange(head, dtype=torch.int64, device="cuda") * 7919 + 5).view(
        key_dtype
    )
    ids = torch.arange(head, device="cuda") % num_tables
    table = build_cache_table(key_dtype, num_tables, head)
    results = torch.empty(head, dtype=table.result_type, device="cuda")
    table.insert(keys, ids, score_of(head), results)
    assert (results == InsertResult.INSERT.value).all()
    size = int(table.size())

    tail = 37
    padded_keys, padded_ids = pad_with(
        keys, ids, key_dtype, garbage_ids(num_tables, tail)
    )
    n = head + tail

    _, found_ref, index_ref = table.lookup_with_overflow(keys, ids, const_score())
    _, found, index = table.lookup_with_overflow(padded_keys, padded_ids, const_score())
    assert found[:head].all() and not found[head:].any()
    assert torch.equal(index[:head], index_ref) and (index[head:] == EMPTY).all()
    table.increment_counter(index, padded_ids)
    assert int(table._ref_counter.sum()) == head, "exactly the head rows are pinned"
    table.decrement_counter(index, padded_ids)
    assert (table._ref_counter == 0).all()

    # inserting the padded batch again: the head keys exist, the reserved ones are ILLEGAL
    insert_results = torch.empty(n, dtype=table.result_type, device="cuda")
    slots = table.insert(padded_keys, padded_ids, score_of(n), insert_results)
    assert (insert_results[:head] == InsertResult.ASSIGN.value).all()
    assert (insert_results[head:] == InsertResult.ILLEGAL.value).all()
    assert torch.equal(slots[:head], index_ref) and (slots[head:] == EMPTY).all()
    assert int(table.size()) == size

    # the cache's own insert: nothing is evicted, nothing of the tail is collected
    insert_results = torch.empty(n, dtype=table.result_type, device="cuda")
    out = table.insert_and_evict_with_counter_and_overflow(
        padded_keys, padded_ids, score_of(n), insert_results
    )
    indices, num_evicted = out[0], out[1]
    assert num_evicted == 0
    assert (insert_results[head:] == InsertResult.ILLEGAL.value).all()
    assert (indices[head:] == EMPTY).all()
    assert int(table.size()) == size

    # erasing reserved keys erases nothing
    reserved_keys, reserved_ids = pad_with(
        keys[:0], ids[:0], key_dtype, garbage_ids(num_tables, tail)
    )
    erased = table.erase(reserved_keys, reserved_ids, return_indices=True)
    assert (erased == EMPTY).all()
    assert int(table.size()) == size
    _, found_after, index_after = table.lookup_with_overflow(keys, ids, const_score())
    assert found_after.all() and torch.equal(index_after, index_ref)


# --------------------------------------------------------------------------
# the masked lookup paths
# --------------------------------------------------------------------------
# path function -> the config of the probe that takes it
PATHS = [
    ("_prefetch_cache_path_masked", "cache_storage"),
    ("_prefetch_hbm_direct_path_masked", "hbm_only"),
    ("_generic_forward_path_masked", "host_only"),
    ("_generic_forward_path_masked", "hybrid"),
]


# What the scribble puts into the tail of the arrays a masked path is handed. "ids": real table
# ids on rows that are not real, EMPTY keys (the canonical convention keeps them). "live_keys":
# in addition keys that look live and that no table holds, so only `valid` says that these rows
# are not real: a path that does not AND `valid` into its miss mask treats them as new keys and
# inserts them. The DEFAULT (host_only, hybrid) paths take only the first: without an admitter
# their backward re-inserts `ctx.unique_keys` as they are (the tail is EMPTY by the convention),
# so a live-looking key there is outside what that path promises.
SCRIBBLES = ["ids", "live_keys"]
PATHS_WITHOUT_LIVE_KEYS = {"_generic_forward_path_masked"}


@pytest.mark.parametrize("scribble", SCRIBBLES)
@pytest.mark.parametrize("path, config", PATHS, ids=["%s[%s]" % p for p in PATHS])
def test_the_masked_paths_ignore_the_tail(bdf, monkeypatch, path, config, scribble):
    """TAIL-SCRIBBLE end to end: what reaches a masked lookup path obeys the cross-array
    invariant, and a step still runs, hits everything, inserts and pins nothing for the rows
    that are not real, when the table ids of the tail it is handed are replaced by ids of real
    tables ("ids"), and when its keys are replaced by keys that no table holds ("live_keys").
    """
    import inspect

    if scribble == "live_keys" and path in PATHS_WITHOUT_LIVE_KEYS:
        pytest.skip(
            "the DEFAULT paths rely on the canonical EMPTY tail keys (see SCRIBBLES)"
        )
    original = getattr(bdf, path, None)
    if original is None:
        pytest.skip("this tree has no %s yet" % path)
    signature = inspect.signature(original)
    seen = {"padded": 0, "exact": 0}

    def scribbling(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        keys, ids, valid = (
            bound.arguments[k] for k in ("unique_keys", "unique_table_ids", "valid")
        )
        if valid is None:
            seen["exact"] += 1
            return original(*args, **kwargs)
        seen["padded"] += 1
        tail = ~valid
        assert torch.equal(
            as_i64(keys) == EMPTY, tail
        ), "the tail keys are EMPTY, and only they"
        assert torch.equal(
            ids == EMPTY, tail
        ), "the tail table ids are -1, and only they"
        assert not bool(
            (tail[:-1] & valid[1:]).any()
        ), "the head comes first: valid is a prefix"
        # ids of real tables on rows that are not: gathers by id stay in range
        other = torch.randint(0, int(ids.max()) + 1, ids.shape, device=ids.device)
        bound.arguments["unique_table_ids"] = torch.where(valid, ids, other)
        if scribble == "live_keys":
            # keys outside the 62 bits the probe's batches use and far from the four reserved
            # keys, one per row: no table has them, and `valid` is False for their rows
            live = (1 << 62) + torch.arange(
                keys.numel(), dtype=torch.int64, device=keys.device
            )
            bound.arguments["unique_keys"] = torch.where(tail, live, as_i64(keys)).view(
                keys.dtype
            )
        return original(*bound.args, **bound.kwargs)

    cfg = hsp.CONFIGS[config]
    device = torch.device("cuda", torch.cuda.current_device())
    mod = hsp.build_module(cfg, hsp.plan_of(cfg, "all_hit"), device.index)
    batch = hsp.BatchMaker(cfg, device, 1234).fresh_batch()
    step = hsp._make_step(mod, cfg, "plain", True)
    mod.train()
    for _ in range(3):  # inserts every key
        step(batch)
    torch.cuda.synchronize()
    before = hsp.tier_sizes(mod)

    monkeypatch.setattr(bdf, path, scribbling)
    for _ in range(2):
        step(batch)
    torch.cuda.synchronize()  # a read through a garbage table id faults here
    monkeypatch.undo()

    if not seen["padded"]:
        pytest.skip(
            "this tree hands %s exact arrays (valid=None): no tail to scribble yet"
            % path
        )
    assert (
        hsp.tier_sizes(mod) == before
    ), "an all-hit step inserted or evicted something"
    for holder in (getattr(mod, "_cache", None), getattr(mod, "_storage", None)):
        table = getattr(getattr(holder, "_state", None), "key_index_map", None)
        counters = getattr(table, "_ref_counter", None)
        if counters is not None:
            assert (counters == 0).all(), "a counter is still pinned after the backward"
