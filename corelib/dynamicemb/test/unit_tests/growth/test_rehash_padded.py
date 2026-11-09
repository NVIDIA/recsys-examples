# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rehash of a key map reads nothing back.

A key map grows by inserting its old key column, as it is, into a bigger one a chunk at a
time (``LinearBucketTable.rehash_table_from``): the empty, reclaimed and locked slots hold
reserved keys, which insert nothing and report slot ``-1``; two kernels move what is
keyed by the slot (the value row and the reference counter, ``table_carry_slot_state``)
and the rows of the reclaimed slots (``table_carry_reclaimed_rows``), whose count is
computed on the device. No count is exported, no key is picked out, nothing is waited for.

What this checks, per key against the old table (the specification, not a second
implementation): every live key of a table is found in the grown one with the row, every
score word and the reference count it had; the table has as many live keys, and the
reclaimed rows of the old table sit on reclaimed slots of the new one, each exactly once;
and that none of it depends on how many chunks the old column is cut into.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap
from typing import List

import dynamicemb  # noqa: F401  (imports torch before the extension)
import dynamicemb_extensions
import numpy as np
import pytest
import torch
from dynamicemb import DynamicEmbScoreStrategy, DynamicEmbTableOptions
from dynamicemb import key_value_table as kvt
from dynamicemb import sync_free as sf
from dynamicemb.optimizer import AdamDynamicEmbeddingOptimizer, OptimizerArgs
from dynamicemb.scored_hashtable import (
    LinearBucketTable,
    ScoreArg,
    ScoreSpec,
    get_scored_table,
)
from dynamicemb_extensions import ScorePolicy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

BUCKET = 16
SCORE = "s"
DEVICE = "cuda"


def _load_no_sync():
    """``no_sync`` of the no_host_sync tests, loaded by its path: a conftest of this
    directory's neighbours is not a module a test can import by name."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "no_host_sync", "conftest.py")
    spec = importlib.util.spec_from_file_location("no_host_sync_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.no_sync


# --- helpers -------------------------------------------------------------------------


def make_keys(first: int, count: int, key_type: torch.dtype) -> torch.Tensor:
    """``count`` distinct keys. A uint64 key has its top bit set, so that as an int64 it is
    negative and below the reserved range, which a rehash must not mistake for one."""
    values = np.arange(first, first + count, dtype=np.uint64)
    if key_type == torch.uint64:
        values = values + np.uint64(1 << 63)
    keys = torch.from_numpy(values.view(np.int64)).to(DEVICE)
    return keys.view(torch.uint64) if key_type == torch.uint64 else keys


def make_table(capacities, key_type=torch.int64, lru_lfu=False, use_rows=True):
    policy = ScorePolicy.LRU_LFU if lru_lfu else ScorePolicy.ASSIGN
    return get_scored_table(
        capacity=list(capacities),
        bucket_capacity=BUCKET,
        key_type=key_type,
        score_specs=[ScoreSpec(name=SCORE, policy=policy)],
        use_rows=use_rows,
    )


def populate(table, table_id, keys, lru_lfu):
    """Insert ``keys`` with scores of their own, and reference counts of 0, 1 or 2."""
    n = keys.numel()
    tids = torch.full((n,), table_id, dtype=torch.int64, device=DEVICE)
    ones = torch.ones(n, dtype=torch.uint64, device=DEVICE)
    if lru_lfu:
        arg = ScoreArg(name=SCORE, value=ones, policy=ScorePolicy.LRU_LFU)
    else:
        scores = (torch.arange(n, device=DEVICE) + 100).to(torch.uint64)
        arg = ScoreArg(name=SCORE, value=scores, policy=ScorePolicy.ASSIGN)
    _, slots = table.insert(keys, tids, arg, return_slots=True)
    if lru_lfu:
        timestamps = torch.arange(n, device=DEVICE, dtype=torch.int64) + 1_000
        frequencies = torch.arange(n, device=DEVICE, dtype=torch.int64) % 7 + 1
        table.scatter_score_blocks(
            table_id, slots, torch.stack([timestamps, frequencies], dim=1)
        )
    for step in (3, 6):  # every third key is pinned once, every sixth twice
        table.increment_counter(slots[::step].contiguous(), tids[::step].contiguous())
    return tids


def erase(table, table_id, keys, fraction):
    gone = keys[: int(fraction * keys.numel())]
    if gone.numel():
        table.erase(gone, torch.full((gone.numel(),), table_id, device=DEVICE))


def select(keys, mask):
    """``keys`` where ``mask``; uint64 has no advanced indexing, so through its int64 bits."""
    picked = sf.as_i64(keys)[mask.nonzero().reshape(-1)]
    return picked.view(keys.dtype) if keys.dtype == torch.uint64 else picked


def lookup(table, table_id, keys):
    if keys.numel() == 0:
        empty = torch.empty(0, dtype=torch.int64, device=DEVICE)
        return torch.empty(0, dtype=torch.bool, device=DEVICE), empty, empty
    tids = torch.full((keys.numel(),), table_id, dtype=torch.int64, device=DEVICE)
    arg = ScoreArg(name=SCORE, value=None, policy=ScorePolicy.CONST)
    _, founds, rows, slots = table.lookup(keys, tids, arg, return_slots=True)
    return founds, rows, slots


def state_of(table, table_id, keys):
    """What the table holds for ``keys``: found, row, every score word, reference count."""
    founds, rows, slots = lookup(table, table_id, keys)
    if keys.numel() == 0:
        return {"keys": [], "rows": [], "words": [], "refs": []}
    live = founds.nonzero().reshape(-1)
    slots = slots[live]
    base = table._flat_base(table_id)
    return {
        "keys": sf.as_i64(keys)[live].tolist(),
        "rows": rows[live].tolist() if table.use_rows_ else None,
        "words": table.gather_score_blocks(table_id, slots).to(torch.int64).tolist(),
        "refs": table._ref_counter[base + slots].tolist(),
    }


def reclaimed_rows(table, table_id) -> List[int]:
    """The rows that sit on the reclaimed slots of ``table_id``."""
    b0, b1 = (
        table.bucket_offsets_host_[table_id],
        table.bucket_offsets_host_[table_id + 1],
    )
    key_col = table.keys_.view(torch.int64)[b0:b1]
    rows = table.rows_.view(table.num_buckets_, BUCKET)[b0:b1]
    mask = (key_col == -2) & (rows >= 0)
    return sorted(rows[mask].tolist())


def live_keys(table, table_id, keys):
    founds, _, _ = lookup(table, table_id, keys)
    return select(keys, founds)


def rehash(old, new, table_id, chunk_buckets=None, with_rows=True):
    kwargs = {}
    if chunk_buckets is not None:
        kwargs["chunk_slots"] = chunk_buckets * BUCKET
    new.rehash_table_from(old, table_id, **kwargs)
    if with_rows and new.use_rows_:
        new.carry_rows_from(old, table_id)


# --- the rehash of a table, key for key ----------------------------------------------


@pytest.mark.parametrize("chunk_buckets", [1, 3, None], ids=["c1", "c3", "whole"])
@pytest.mark.parametrize("erased", [0.0, 0.3, 1.0], ids=["e0", "e30", "e100"])
@pytest.mark.parametrize("use_rows", [True, False], ids=["rows", "slots"])
@pytest.mark.parametrize("lru_lfu", [False, True], ids=["one_word", "lru_lfu"])
@pytest.mark.parametrize("key_type", [torch.int64, torch.uint64], ids=["i64", "u64"])
def test_every_live_key_keeps_its_state(
    key_type, lru_lfu, use_rows, erased, chunk_buckets
):
    # Three tables; the middle one grows from 10 to 20 buckets (so its buckets do not
    # start at 0 in either map), the others are copied.
    old = make_table([64, 160, 96], key_type, lru_lfu, use_rows)
    new = make_table([64, 320, 96], key_type, lru_lfu, use_rows)
    keys = [
        make_keys(1_000 * (t + 1), 60 if t == 1 else 20, key_type) for t in range(3)
    ]
    for table_id in range(3):
        populate(old, table_id, keys[table_id], lru_lfu)
    erase(old, 1, keys[1], erased)
    torch.cuda.synchronize()
    live = live_keys(old, 1, keys[1])
    expected = state_of(old, 1, live)
    reclaimed_before = reclaimed_rows(old, 1) if use_rows else []

    rehash(old, new, 1, chunk_buckets)
    new.copy_table_from(old, 0)
    new.copy_table_from(old, 2)
    torch.cuda.synchronize()

    founds, _, _ = lookup(new, 1, live)
    assert live.numel() == 0 or bool(founds.all())
    assert state_of(new, 1, live) == expected
    # table 1 starts at bucket 4 in both maps, and has 10 buckets in the old one, 20 in the new
    assert int(new.bucket_sizes[4:24].sum()) == live.numel()
    assert int(new.bucket_sizes[4:24].sum()) == int(old.bucket_sizes[4:14].sum())
    if use_rows:
        assert int(new.row_counters_[1]) == int(old.row_counters_[1])
        # the reclaimed rows moved to reclaimed slots, each once, and none is below
        # a live key
        assert reclaimed_rows(new, 1) == reclaimed_before
    # the tables that were copied are what they were
    for table_id in (0, 2):
        assert state_of(new, table_id, keys[table_id]) == state_of(
            old, table_id, keys[table_id]
        )


def test_the_result_does_not_depend_on_the_chunks():
    old = make_table([320])
    keys = make_keys(5_000, 100, torch.int64)
    populate(old, 0, keys, False)
    erase(old, 0, keys, 0.25)
    torch.cuda.synchronize()
    live = live_keys(old, 0, keys)
    results = []
    for chunk_buckets in (1, 2, 7, None):
        new = make_table([640])
        rehash(old, new, 0, chunk_buckets)
        torch.cuda.synchronize()
        results.append((state_of(new, 0, live), reclaimed_rows(new, 0)))
    assert all(r == results[0] for r in results)
    assert results[0][0] == state_of(old, 0, live)


@pytest.mark.parametrize("determinism", [False, True], ids=["default", "determinism"])
def test_the_rehash_under_the_determinism_mode(determinism, monkeypatch):
    """Reserved keys that are not the all-ones one (reclaimed and locked) are made that key
    before a mode that drops only that one inserts them."""
    if determinism:
        monkeypatch.setenv("DEMB_DETERMINISM_MODE", "1")
    old = make_table([160])
    new = make_table([320])
    keys = make_keys(9_000, 60, torch.int64)
    populate(old, 0, keys, False)
    erase(old, 0, keys, 0.5)
    torch.cuda.synchronize()
    live = live_keys(old, 0, keys)
    rehash(old, new, 0, 2)
    torch.cuda.synchronize()
    assert state_of(new, 0, live) == state_of(old, 0, live)
    assert reclaimed_rows(new, 0) == reclaimed_rows(old, 0)


def test_the_exact_length_api_still_carries_rows():
    """``insert_carrying_rows`` + ``carry_rows_from``, the pair the previous rehash used:
    the keys of an export, their rows given, the reclaimed rows after them."""
    old = make_table([160])
    keys = make_keys(300, 60, torch.int64)
    populate(old, 0, keys, False)
    erase(old, 0, keys, 0.4)
    torch.cuda.synchronize()
    live = live_keys(old, 0, keys)
    _, rows, _ = lookup(old, 0, live)
    new = make_table([320])
    arg = ScoreArg(
        name=SCORE,
        value=torch.ones(live.numel(), dtype=torch.uint64, device=DEVICE),
        policy=ScorePolicy.ASSIGN,
    )
    slots = new.insert_carrying_rows(live, 0, arg, rows)
    new.carry_rows_from(old, 0)
    torch.cuda.synchronize()
    assert bool((slots >= 0).all())
    _, new_rows, _ = lookup(new, 0, live)
    assert new_rows.tolist() == rows.tolist()
    assert reclaimed_rows(new, 0) == reclaimed_rows(old, 0)
    assert int(new.row_counters_[0]) == int(old.row_counters_[0])


# --- no count comes back, nothing is waited for --------------------------------------


class _NoDeviceBarrier:
    """``torch`` as ``extendable_tensor`` sees it, without the device-wide wait of
    ``ExtendableBuffer.extend``: that wait is not what is under test here."""

    def __init__(self):
        import types

        self.cuda = types.SimpleNamespace(synchronize=lambda *a, **k: None)

    def __getattr__(self, name):
        return getattr(torch, name)


def option(on_host, no_eviction=False, lru_lfu=False, init_capacity=256):
    strategy = (
        (DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU)
        if lru_lfu
        else DynamicEmbScoreStrategy.NO_EVICTION
        if no_eviction
        else DynamicEmbScoreStrategy.TIMESTAMP
    )
    return DynamicEmbTableOptions(
        dim=16,
        init_capacity=init_capacity,
        max_capacity=1 << 20,
        bucket_capacity=128,
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=torch.cuda.current_device(),
        score_strategy=strategy,
        local_hbm_for_values=0 if on_host else 1 << 30,
    )


@pytest.mark.parametrize(
    "kind",
    ["hbm_rows", "host_rows", "no_eviction", "lru_lfu_rows", "hybrid_host_tier"],
)
def test_a_growth_reads_nothing_back(kind, monkeypatch):
    """Every kind of growth, under ``no_sync`` (``.item()``, ``.cpu()``, a device or
    stream wait: errors) and PyTorch's sync-debug mode, grows the table, and every key
    keeps its value. ``sf.read_now`` is not called either."""
    from dynamicemb import extendable_tensor

    no_sync = _load_no_sync()
    monkeypatch.setattr(extendable_tensor, "torch", _NoDeviceBarrier())
    on_host = kind in ("host_rows", "hybrid_host_tier")
    opt = option(
        on_host, no_eviction=kind == "no_eviction", lru_lfu=kind == "lru_lfu_rows"
    )
    use_rows = kind != "hybrid_host_tier"
    state = kvt.create_table_state(
        [opt], AdamDynamicEmbeddingOptimizer(OptimizerArgs()), use_rows=use_rows
    )
    n = 120
    keys = make_keys(77, n, torch.int64)
    tids = torch.zeros_like(keys)
    ones = torch.ones(n, dtype=torch.uint64, device=DEVICE)
    lru = state.score_policy.policy == ScorePolicy.LRU_LFU
    arg = ScoreArg(
        name=state.score_policy.name,
        value=ones,
        policy=ScorePolicy.LRU_LFU if lru else ScorePolicy.ASSIGN,
    )
    rows, _ = state.key_index_map.insert(keys, tids, arg, return_slots=True)
    vdim = state.table_value_dims_cpu[0]
    values = keys.to(torch.float32).unsqueeze(1).expand(-1, vdim).contiguous()
    kvt.store_to_flat_single_table(state, rows, 0, values)
    erase(state.key_index_map, 0, keys, 0.25)
    torch.cuda.synchronize()
    before = live_keys(state.key_index_map, 0, keys)
    capacity = state.key_index_map.per_table_capacity_[0]
    reads = sf.READ_NOW_CALLS

    with no_sync():
        kvt._expand_tables_impl(state, [True], [4 * capacity])
    torch.cuda.synchronize()

    assert sf.READ_NOW_CALLS == reads
    assert state.key_index_map.per_table_capacity_[0] == 4 * capacity
    assert before.numel() == n - n // 4
    founds, new_rows, _ = lookup(state.key_index_map, 0, before)
    assert bool(founds.all())
    got = kvt.load_from_flat_single_table(state, new_rows, 0)
    assert got[:, 0].tolist() == before.to(torch.float32).tolist()


@pytest.mark.parametrize("chunk_buckets", [1, 2])
def test_values_follow_their_keys_across_chunks(chunk_buckets, monkeypatch):
    """A table whose value row is its slot moves the values of its live keys: each chunk
    loads, the stores come after the last load, so a store never lands on a row that was
    not read yet -- with chunks of one and two buckets, where it would."""
    real = LinearBucketTable.rehash_table_from
    monkeypatch.setattr(
        LinearBucketTable,
        "rehash_table_from",
        lambda self, other, table_id, on_chunk=None, chunk_slots=0: real(
            self, other, table_id, on_chunk, chunk_buckets * 128
        ),
    )
    state = kvt.create_table_state(
        [option(on_host=False, init_capacity=1024)],
        AdamDynamicEmbeddingOptimizer(OptimizerArgs()),
    )
    n = 100
    keys = make_keys(3_000, n, torch.int64)
    tids = torch.zeros_like(keys)
    arg = ScoreArg(
        name=state.score_policy.name,
        value=torch.ones(n, dtype=torch.uint64, device=DEVICE),
        policy=ScorePolicy.ASSIGN,
    )
    rows, _ = state.key_index_map.insert(keys, tids, arg, return_slots=True)
    vdim = state.table_value_dims_cpu[0]
    values = keys.to(torch.float32).unsqueeze(1).expand(-1, vdim).contiguous()
    kvt.store_to_flat_single_table(state, rows, 0, values)

    kvt._expand_tables_impl(state, [True], [4096])
    torch.cuda.synchronize()

    founds, new_rows, _ = lookup(state.key_index_map, 0, keys)
    assert bool(founds.all())
    got = kvt.load_from_flat_single_table(state, new_rows, 0)
    assert torch.equal(got, values)


# --- what goes wrong is a device assert ----------------------------------------------


def _fails_on_the_device(script: str, expect: str) -> None:
    """The assert kills the CUDA context, so it runs in a process of its own."""
    run = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True
    )
    assert run.returncode != 0, "the process did not fail"
    assert expect in run.stderr.lower(), run.stderr[-2000:]


_PRELUDE = """
import torch
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import ScorePolicy

def table(capacity):
    return get_scored_table(
        capacity=[capacity], bucket_capacity=16, key_type=torch.int64,
        score_specs=[ScoreSpec(name="s", policy=ScorePolicy.ASSIGN)], use_rows=True,
    )

def fill(t, n):
    keys = torch.arange(1000, 1000 + n, dtype=torch.int64, device="cuda")
    arg = ScoreArg(name="s", value=torch.arange(n, device="cuda").to(torch.uint64) + 1,
                   policy=ScorePolicy.ASSIGN)
    t.insert(keys, torch.zeros_like(keys), arg)
    return keys
"""


def test_a_table_that_cannot_hold_the_live_keys_is_asserted_on_the_device():
    """The rehash of 30 keys into one bucket of 16: a key refused, or a resident evicted
    by a newcomer (which reports a slot): either way a device assert, never a silently
    smaller table."""
    _fails_on_the_device(
        _PRELUDE
        + """
old, new = table(32), table(16)
fill(old, 30)
new.rehash_table_from(old, 0)
torch.cuda.synchronize()
""",
        "assert",
    )


def test_more_rows_than_the_counters_promise_are_asserted_on_the_device():
    _fails_on_the_device(
        _PRELUDE
        + """
old, new = table(160), table(320)
fill(old, 60)
old.reserve_rows(0, 500)  # 440 rows owned by no live key and no reclaimed slot
new.rehash_table_from(old, 0)
new.carry_rows_from(old, 0)
torch.cuda.synchronize()
""",
        "assert",
    )


def test_a_count_below_zero_carries_nothing_and_asserts_nothing():
    old, new = make_table([160]), make_table([320])
    keys = make_keys(40, 60, torch.int64)
    populate(old, 0, keys, False)
    erase(old, 0, keys, 0.5)
    old.row_counters_[0] = 1  # fewer rows than live keys: the count is below zero
    new.rehash_table_from(old, 0)
    new.carry_rows_from(old, 0)
    torch.cuda.synchronize()
    assert reclaimed_rows(new, 0) == []
    assert int(new.row_counters_[0]) == 1


def test_a_stale_extension_says_to_rebuild(monkeypatch):
    monkeypatch.setattr(
        dynamicemb_extensions, "rehash_device_count", False, raising=False
    )
    old, new = make_table([160]), make_table([320])
    for call in (
        lambda: new.rehash_table_from(old, 0),
        lambda: new.carry_rows_from(old, 0),
        lambda: new.insert_carrying_rows(
            make_keys(1, 4, torch.int64),
            0,
            ScoreArg(
                name=SCORE,
                value=torch.ones(4, dtype=torch.uint64, device=DEVICE),
                policy=ScorePolicy.ASSIGN,
            ),
            torch.arange(4, device=DEVICE),
        ),
    ):
        with pytest.raises(RuntimeError, match="rebuild"):
            call()


# --- a long old column is cut into many chunks without allocating per chunk ----------


def test_the_temporaries_of_a_chunk_are_reused_by_the_next_one():
    """40 chunks allocate what 2 do: a chunk's temporaries are freed to the caching
    allocator and handed out again, so a rehash allocates per growth, not per chunk."""
    old = make_table([BUCKET * 64])
    keys = make_keys(10_000, 300, torch.int64)
    populate(old, 0, keys, False)
    warm = make_table([BUCKET * 128])
    rehash(old, warm, 0, 32)  # two chunks: the allocator now holds their sizes
    torch.cuda.synchronize()

    def segments():
        return torch.cuda.memory_stats()["segment.all.allocated"]

    few, many = make_table([BUCKET * 128]), make_table([BUCKET * 128])
    torch.cuda.synchronize()
    before = segments()
    rehash(old, few, 0, 32)
    torch.cuda.synchronize()
    with_few = segments() - before
    before = segments()
    rehash(old, many, 0, 1)  # 64 chunks
    torch.cuda.synchronize()
    with_many = segments() - before
    assert with_many <= with_few + 2, (with_few, with_many)
