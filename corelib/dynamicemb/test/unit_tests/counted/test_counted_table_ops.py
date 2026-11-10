# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The table ops with a device-side count (``LinearBucketTable`` ``num_valid=``).

``num_valid``, a one-element int64 device tensor, says how many leading entries of the
per-key arrays are live; what follows is garbage no kernel may read or write. For every
op and n in {0, 1, N // 2, N - 1, N}, a call on capacity-N inputs with a garbage tail
must match the same call on the exact slice [:n] (made on a twin table where the op
changes the table), leave the caller's output buffers untouched past n, and leave the
table as if the tail did not exist: the tail names keys that are in the table or would
be inserted, and tables, slots and keys that are out of range or reserved.

The exact-slice call takes no count (``None``: every entry is live). It answers in the
same form as the counted one -- arrays of the capacity of its input, counts as device
tensors -- so what it is compared to is the live prefix of the counted result.
"""

import pytest
import torch
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import InsertResult, ScorePolicy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

N = 64
COUNTS = [0, 1, N // 2, N - 1, N]
BIG = 1 << 22  # more entries than a capped grid has threads: the kernels must stride
ASSIGN, CONST, LRU_LFU = ScorePolicy.ASSIGN, ScorePolicy.CONST, ScorePolicy.LRU_LFU
INSERT, EVICT = InsertResult.INSERT.value, InsertResult.EVICT.value
UNSET, UNSET_SCORE = 255, -(1 << 62)  # what the caller's output buffers start with

COUNT = pytest.mark.parametrize("n", COUNTS)
ROWS = pytest.mark.parametrize("use_rows", [False, True], ids=["slots", "rows"])
POLICY = pytest.mark.parametrize("policy", [ASSIGN, LRU_LFU], ids=["assign", "lru_lfu"])
BIG_COUNTS = pytest.mark.parametrize("n", [0, 1, BIG // 2 + 1, BIG])


def _nv(n):
    """The count as a view into a longer tensor, as ``table_offsets[T:T + 1]`` is."""
    return torch.tensor([-7, n, -7], dtype=torch.int64, device="cuda")[1:2]


def _ids(m, value=0):
    return torch.full((m,), value, dtype=torch.int64, device="cuda")


def _unset(m, dtype=torch.uint8, value=UNSET):
    return torch.full((m,), value, dtype=dtype, device="cuda")


def _all(x, value):
    return bool((x == value).all())


def _is_count(x):
    """A count as the ops return it: a one-element int64 CUDA tensor."""
    return (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and x.dtype == torch.int64
        and x.shape == (1,)
    )


def _table(policy=ASSIGN, caps=(1024,), bucket=128, **kw):
    return get_scored_table(
        capacity=list(caps),
        bucket_capacity=bucket,
        key_type=torch.int64,
        score_specs=[ScoreSpec(name="score", policy=policy)],
        **kw,
    )


def _arg(value, policy=ASSIGN):
    return ScoreArg(name="score", value=value, policy=policy)


def _found(table, keys, ids):
    return table.lookup(keys, ids, _arg(None, CONST))[1]


def _junk_keys(m):
    """Random keys (taken, they would be inserted), the reserved ones among them."""
    k = torch.randint(1 << 40, 1 << 60, (m,), dtype=torch.int64, device="cuda")
    k[::4], k[1::8], k[3::8] = -1, -2, -3  # empty, reclaimed, locked
    return k


def _junk_ids(m, num_tables=1):
    """Table ids, every other one far out of range (a read of it would fault)."""
    t = torch.randint(0, num_tables, (m,), dtype=torch.int64, device="cuda")
    wild = torch.randint(1 << 30, 1 << 40, (m,), dtype=torch.int64, device="cuda")
    wild[1::4] *= -1
    t[1::2] = wild[1::2]
    return t


def _junk_scores(m):
    k = torch.randint(1 << 40, 1 << 60, (m,), dtype=torch.int64, device="cuda")
    return k.to(torch.uint64)


def _dirty(x, n, junk):
    """``x`` with the entries from ``n`` on replaced by ``junk(count)``."""
    y = x.clone()
    y[n:] = junk(x.numel() - n)
    return y


def _dirty_batch(keys, scores, n, policy=ASSIGN):
    """(keys, table_ids, score) of a batch whose entries from ``n`` on are garbage."""
    return (
        _dirty(keys, n, _junk_keys),
        _dirty(_ids(keys.numel()), n, _junk_ids),
        _arg(_dirty(scores, n, _junk_scores), policy),
    )


def _check_lookup(size, n, use_rows):
    table = _table(caps=(max(1024, 2 * size),), use_rows=use_rows)
    keys = torch.arange(1, size + 1, device="cuda")
    ids = _ids(size)
    scores = (keys * 10).to(torch.uint64)
    present = keys[::2].contiguous()  # the table holds every other key
    pids = ids[: present.numel()]
    table.insert(present, pids, _arg(scores[::2].contiguous()))

    def present_keys(m):  # the tail asks for keys the table has, ...
        return present[torch.randint(0, present.numel(), (m,), device="cuda")]

    args = (_dirty(keys, n, present_keys), _dirty(ids, n, _junk_ids))
    junk = _arg(_dirty(scores, n, _junk_scores))  # ... with scores to replace theirs
    got = table.lookup(*args, junk, return_slots=True, num_valid=_nv(n))
    want = table.lookup(keys[:n], ids[:n], _arg(scores[:n]), return_slots=True)
    for g, w in zip(got, want):
        assert g.numel() == size and torch.equal(g[:n], w)
    assert torch.equal(table.slots_of(*args, num_valid=_nv(n))[:n], want[3])
    # An ASSIGN lookup rewrites the score of a key it finds; the tail's did not.
    score_out, founds, _ = table.lookup(present, pids, _arg(None, CONST))
    assert bool(founds.all()) and torch.equal(score_out, present * 10)


@COUNT
@ROWS
def test_lookup(n, use_rows):
    _check_lookup(N, n, use_rows)


@BIG_COUNTS
def test_lookup_big_batch(n):
    _check_lookup(BIG, n, False)


def _check_insert(size, n, use_rows, policy):
    caps = (max(1024, 2 * size),)
    a, b = (_table(policy, caps, use_rows=use_rows) for _ in range(2))
    keys = torch.arange(1, size + 1, device="cuda")
    ids = _ids(size)
    scores = (keys * 3).to(torch.uint64)
    results, score_out = _unset(size), _unset(size, torch.int64, UNSET_SCORE)
    rows, slots = a.insert(
        *_dirty_batch(keys, scores, n, policy),
        results,
        score_out,
        return_slots=True,
        num_valid=_nv(n),
    )
    results_b, score_b = _unset(n), _unset(n, torch.int64, UNSET_SCORE)
    b.insert(keys[:n], ids[:n], _arg(scores[:n], policy), results_b, score_b)
    # The outputs match the exact-slice run, and are untouched past n.
    assert rows.numel() == slots.numel() == size
    assert torch.equal(results[:n], results_b) and _all(results[:n], INSERT)
    assert torch.equal(score_out[:n], score_b)
    assert torch.equal(score_out[:n], scores[:n].to(torch.int64))
    assert _all(results[n:], UNSET) and _all(score_out[n:], UNSET_SCORE)
    # So is the table: the tail inserted nothing.
    assert torch.equal(a.bucket_sizes, b.bucket_sizes)
    assert int(a.bucket_sizes.sum()) == n
    found = _found(a, keys, ids)
    assert torch.equal(found, _found(b, keys, ids))
    assert bool(found[:n].all()) and not bool(found[n:].any())
    # The indices reported are where the keys are found.
    assert torch.equal(rows[:n], a.lookup(keys[:n], ids[:n], _arg(None, CONST))[2])
    assert torch.equal(slots[:n], a.slots_of(keys[:n], ids[:n]))
    if use_rows:  # the rows are dense: the tail took none
        assert torch.equal(torch.sort(rows[:n]).values, torch.arange(n, device="cuda"))
        assert a.num_rows(0) == n


@COUNT
@ROWS
@POLICY
def test_insert(n, use_rows, policy):
    _check_insert(N, n, use_rows, policy)


@BIG_COUNTS
def test_insert_big_batch(n):
    _check_insert(BIG, n, False, ASSIGN)


@pytest.mark.parametrize("count, live", [(-3, 0), (N + 9, N)])
def test_count_is_clamped(count, live):
    table = _table()
    keys = torch.arange(1, N + 1, device="cuda")
    results = _unset(N)
    score = _arg(keys.to(torch.uint64))
    table.insert(keys, _ids(N), score, results, num_valid=_nv(count))
    assert int(table.bucket_sizes.sum()) == live
    assert _all(results[:live], INSERT) and _all(results[live:], UNSET)


def _resident(policy, use_rows=False):
    """Twin tables of one 128-slot bucket holding keys 1..100, scored by themselves."""
    tables = [_table(policy, (128,), use_rows=use_rows) for _ in range(2)]
    old = torch.arange(1, 101, device="cuda")
    for t in tables:
        t.insert(old, _ids(100), _arg(old.to(torch.uint64), policy))
    return tables


def _newcomers():
    """N keys that outscore every resident: each one past the 28 free slots evicts the
    lowest-scored resident, so the victims are exactly the lowest keys."""
    new = torch.arange(1000, 1000 + N, device="cuda")
    return new, torch.full((N,), 10**6, dtype=torch.uint64, device="cuda")


def _evictions(n):
    return max(0, 100 + n - 128)


@COUNT
@ROWS
@POLICY
def test_insert_and_evict(n, use_rows, policy):
    a, b = _resident(policy, use_rows)
    new, high = _newcomers()
    m, results = _evictions(n), _unset(N)
    indices, num_evicted, *evicted = a.insert_and_evict(
        *_dirty_batch(new, high, n, policy), results, num_valid=_nv(n)
    )
    ev_keys, ev_idx, _, ev_ids = evicted
    # The counted form: the count stays on the device and nothing is sliced.
    assert isinstance(num_evicted, torch.Tensor) and int(num_evicted) == m
    assert all(x.numel() == N for x in (indices, *evicted))
    assert sorted(ev_keys[:m].tolist()) == list(range(1, m + 1))
    assert _all(ev_ids[:m], 0) and _all(results[n:], UNSET)
    head = results[:n]
    assert int((head == EVICT).sum()) == m and int((head == INSERT).sum()) == n - m
    # A newcomer takes the slot (the row, with use_rows) of the resident it evicted.
    assert sorted(indices[:n][head == EVICT].tolist()) == sorted(ev_idx[:m].tolist())
    # The exact-slice run on the twin agrees, table included. It has no count, so
    # all n entries are live, and it answers in the same form: arrays of its own
    # capacity n and a device count, of which the first m_b entries are live.
    idx_b, m_b, *ev_b = b.insert_and_evict(new[:n], _ids(n), _arg(high[:n], policy))
    assert _is_count(m_b) and all(x.numel() == n for x in (idx_b, *ev_b))
    assert int(m_b) == m
    assert sorted(ev_b[0][:m].tolist()) == sorted(ev_keys[:m].tolist())
    keys = torch.cat([torch.arange(1, 101, device="cuda"), new])
    ids = _ids(keys.numel())
    assert torch.equal(_found(a, keys, ids), _found(b, keys, ids))
    assert torch.equal(a.bucket_sizes, b.bucket_sizes)
    if use_rows:
        assert a.num_rows(0) == b.num_rows(0) == min(100 + n, 128)


@COUNT
@POLICY
def test_insert_collect_evicted(n, policy):
    a, _ = _resident(policy)
    new, high = _newcomers()
    m, results = _evictions(n), _unset(N)
    _, num_evicted, ev_keys, ev_ids = a.insert(
        *_dirty_batch(new, high, n, policy),
        results,
        collect_evicted=True,
        num_valid=_nv(n),
    )
    assert int(num_evicted) == m and ev_keys.numel() == ev_ids.numel() == N
    assert sorted(ev_keys[:m].tolist()) == list(range(1, m + 1)) and _all(ev_ids[:m], 0)
    assert _all(results[n:], UNSET) and bool((results[:n] != UNSET).all())


@COUNT
def test_overflow(n):
    a, b = (_table(caps=(128,), enable_overflow=True) for _ in range(2))
    old = torch.arange(1, 129, device="cuda")
    for t in (a, b):  # fill the bucket, pin every slot: newcomers go to the overflow
        slots = t.insert(old, _ids(128), _arg(old.to(torch.uint64)))
        t.increment_counter(slots, _ids(128))
    new, high = _newcomers()
    results = _unset(N)
    keys, ids, junk = _dirty_batch(new, high, n)
    indices, num_evicted, *evicted = a.insert_and_evict_with_counter_and_overflow(
        keys, ids, junk, results, num_valid=_nv(n)
    )
    assert int(num_evicted) == 0 and all(x.numel() == N for x in (indices, *evicted))
    assert _all(results[:n], INSERT) and _all(results[n:], UNSET)
    # The exact-slice run has no count: all n entries are live, and it answers in the
    # same form, at its own capacity n.
    idx_b, m_b, *ev_b = b.insert_and_evict_with_counter_and_overflow(
        new[:n], _ids(n), _arg(high[:n])
    )
    assert _is_count(m_b) and int(m_b) == 0
    assert all(x.numel() == n for x in (idx_b, *ev_b))
    assert torch.equal(a.overflow_bucket_sizes, b.overflow_bucket_sizes)
    assert int(a.overflow_bucket_sizes.sum()) == n
    look = _arg(None, CONST)
    _, found, where = a.lookup_with_overflow(keys, ids, look, num_valid=_nv(n))
    assert where.numel() == N and bool(found[:n].all())
    assert bool((where[:n] >= 128).all())
    assert int(a.lookup_with_overflow(new, _ids(N), look)[1].sum()) == n


@pytest.mark.parametrize("overflow", [False, True], ids=["main", "overflow"])
@pytest.mark.parametrize("op", ["increment_counter", "decrement_counter"])
@COUNT
def test_counter(n, op, overflow):
    a, b = (_table(caps=(512, 512), enable_overflow=overflow) for _ in range(2))
    flat = torch.randperm(1024, device="cuda")[:N]  # one counter per entry: no atomics
    ids = (flat >= 512).to(torch.int64)
    slots = flat - 512 * ids
    if op == "decrement_counter":  # something to take back
        for t in (a, b):
            t.increment_counter(slots, ids)

    def junk_slots(m):
        return torch.randint(-8, 2048, (m,), dtype=torch.int64, device="cuda")

    dirty_ids = _dirty(ids, n, lambda m: _junk_ids(m, 2))
    getattr(a, op)(_dirty(slots, n, junk_slots), dirty_ids, num_valid=_nv(n))
    getattr(b, op)(slots[:n], ids[:n])
    assert torch.equal(a._ref_counter, b._ref_counter)
    live = N - n if op == "decrement_counter" else n
    assert int(a._ref_counter.abs().sum()) == live


@COUNT
@ROWS
def test_erase(n, use_rows):
    a, b = (_table(use_rows=use_rows) for _ in range(2))
    keys, ids = torch.arange(1, N + 1, device="cuda"), _ids(N)
    for t in (a, b):
        t.insert(keys, ids, _arg(keys.to(torch.uint64)))

    def kept(m):  # the tail names keys that are in the table and have to stay
        return keys[n:][torch.randint(0, max(N - n, 1), (m,), device="cuda")]

    erase_keys, erase_ids = _dirty(keys, n, kept), _dirty(ids, n, _junk_ids)
    present = a.erase(erase_keys, erase_ids, return_indices=True, num_valid=_nv(n))
    b.erase(keys[:n], ids[:n])
    assert present.numel() == N and bool((present[:n] >= 0).all())
    again = a.erase(erase_keys, erase_ids, return_indices=True, num_valid=_nv(n))
    assert bool((again[:n] < 0).all())
    found = _found(a, keys, ids)
    assert torch.equal(found, _found(b, keys, ids)) and int(found.sum()) == N - n
    assert torch.equal(a.bucket_sizes, b.bucket_sizes)
    assert int(a.bucket_sizes.sum()) == N - n


@COUNT
@ROWS
def test_determinism_mode(n, use_rows, monkeypatch):
    """The host reads the count once (not at all without one), runs the old code on
    exact slices, and hands back the counted shapes: capacity-length arrays,
    ``num_evicted`` a device tensor."""
    monkeypatch.setenv("DEMB_DETERMINISM_MODE", "1")
    a, b = _resident(ASSIGN, use_rows)
    new, high = _newcomers()
    m = _evictions(n)
    indices, num_evicted, *evicted = a.insert_and_evict(
        *_dirty_batch(new, high, n), num_valid=_nv(n)
    )
    assert num_evicted.shape == (1,) and num_evicted.dtype == torch.int64
    assert int(num_evicted) == m and all(x.numel() == N for x in (indices, *evicted))
    assert sorted(evicted[0][:m].tolist()) == list(range(1, m + 1))
    # Without a count all n entries are live, and the form is the same, at capacity n.
    idx_b, m_b, *ev_b = b.insert_and_evict(new[:n], _ids(n), _arg(high[:n]))
    assert _is_count(m_b) and all(x.numel() == n for x in (idx_b, *ev_b))
    assert int(m_b) == m and sorted(ev_b[0][:m].tolist()) == list(range(1, m + 1))
    if use_rows:  # dense rows, none taken for an eviction, a victim's is a newcomer's
        for t, rows, victims in [
            (a, indices[:n], evicted[1][:m]),
            (b, idx_b, ev_b[1][:m]),
        ]:
            assert t.num_rows(0) == min(100 + n, 128)
            assert len(set(rows.tolist())) == n
            assert set(victims.tolist()) <= set(rows.tolist())
    c = _table()
    keys = torch.arange(1, N + 1, device="cuda")
    batch = _dirty_batch(keys, keys.to(torch.uint64), n)
    rows, slots = c.insert(*batch, return_slots=True, num_valid=_nv(n))
    assert rows.numel() == slots.numel() == N and int(c.bucket_sizes.sum()) == n
    assert torch.equal(slots[:n], c.slots_of(keys[:n], _ids(n)))
