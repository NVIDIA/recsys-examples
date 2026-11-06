# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Tests of the lightweight cache (docs/lightweight_cache_design.md): FIFO slot
# allocation that skips pinned slots, the guarantee that an insert succeeds
# whenever enough slots are unpinned, the hash index, erase, and the row pool.
#
# A slot is pinned by ``lookup`` (the ones it finds) and by ``insert_and_evict``
# (the ones it hands out); ``release`` takes a pin back.
#
# Run (from ``corelib/dynamicemb``)::
#   python -m pytest test/unit_tests/test_cache.py -q

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import pytest
import torch
from dynamicemb.cache import DynamicEmbCache
from dynamicemb.extendable_tensor import FixedDeviceBuffer
from dynamicemb.mixed_dim_buffers import MixedDimBuffers
from dynamicemb.types import CacheMetrics, CopyMode

EMPTY = -1

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_values(
    caps: List[int], dims: Optional[List[int]] = None, kind: str = "fixed"
) -> MixedDimBuffers:
    dims = dims or [8] * len(caps)
    return MixedDimBuffers(
        dims,  # embedding dims
        dims,  # value dims: no optimizer state
        caps,
        torch.float32,
        torch.device("cuda", torch.cuda.current_device()),
        kind,
    )


def make_cache(
    caps: List[int],
    dims: Optional[List[int]] = None,
    key_dtype: torch.dtype = torch.int64,
) -> DynamicEmbCache:
    return DynamicEmbCache(make_values(caps, dims), key_dtype)


def dev(x: List[int], dtype=torch.int64) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device="cuda")


def tids(n: int, t: int = 0) -> torch.Tensor:
    return torch.full((n,), t, dtype=torch.int64, device="cuda")


def refs(cache, t: int = 0) -> List[int]:
    return cache._core.refs_of(t).tolist()


def insert(cache, keys: List[int], t: int = 0):
    """``insert_and_evict`` of keys of one table; the new slots come back pinned."""
    return cache.insert_and_evict(dev(keys), tids(len(keys), t))


def insert_free(cache, keys: List[int], t: int = 0):
    """Like ``insert``, then release what it pinned."""
    out = insert(cache, keys, t)
    cache.release(out[0], tids(len(keys), t))
    return out


def peek(cache, keys: List[int], t: int = 0):
    """``lookup`` without leaving a pin behind: (founds, slots)."""
    founds, slots = cache.lookup(dev(keys), tids(len(keys), t))
    cache.release(slots, tids(len(keys), t))
    return founds, slots


class RefCache:
    """The cache's contract in plain Python: per table, FIFO over the unpinned
    slots from a cursor, a key -> slot map, reference counts."""

    def __init__(self, caps: List[int]):
        self.caps = caps
        self.keys = [[EMPTY] * c for c in caps]
        self.refs = [[0] * c for c in caps]
        self.cursor = [0] * len(caps)
        self.index: List[Dict[int, int]] = [dict() for _ in caps]

    def lookup(self, keys, tids_):
        """Slots of the keys (-1 if not cached); every slot found is pinned."""
        out = []
        for k, t in zip(keys, tids_):
            s = self.index[t].get(k, -1)
            if s >= 0:
                self.refs[t][s] += 1
            out.append(s)
        return out

    def release(self, slots, tids_):
        for s, t in zip(slots, tids_):
            if s >= 0:
                assert self.refs[t][s] > 0
                self.refs[t][s] -= 1

    def insert(self, keys, tids_, assign=None):
        """Insert new keys. The cache hands out the first unpinned slots of each
        table from its cursor, but which key gets which of them is up to it when
        a pinned slot is in the way, so ``assign`` (the slots the cache gave) is
        checked to be exactly that set and then used."""
        by_t: Dict[int, List[int]] = {}
        for i, t in enumerate(tids_):
            by_t.setdefault(t, []).append(i)
        plan = {}
        for t, idxs in sorted(by_t.items()):
            n, c = len(idxs), self.caps[t]
            chosen, last = [], -1
            for j in range(c):
                s = (self.cursor[t] + j) % c
                if self.refs[t][s] == 0:
                    chosen.append(s)
                    last = j
                    if len(chosen) == n:
                        break
            if len(chosen) < n:
                raise RuntimeError(f"table {t}: {len(chosen)} free < {n}")
            plan[t] = (chosen, last)
        slots, evicted = [None] * len(keys), [None] * len(keys)
        for t, (chosen, last) in plan.items():
            if assign is not None:
                given = [assign[i] for i in by_t[t]]
                assert sorted(given) == sorted(chosen), f"table {t}: wrong slots"
                chosen = given
            for i, s in zip(by_t[t], chosen):
                old = self.keys[t][s]
                if old != EMPTY:
                    del self.index[t][old]
                self.keys[t][s] = keys[i]
                self.index[t][keys[i]] = s
                self.refs[t][s] = 1
                slots[i], evicted[i] = s, old
            self.cursor[t] = (self.cursor[t] + last + 1) % self.caps[t]
        return slots, evicted

    def erase(self, keys, tids_):
        for k, t in zip(keys, tids_):
            s = self.index[t].pop(k, None)
            if s is not None:
                self.keys[t][s] = EMPTY


# ------------------------------------------------------------------ examples --


def test_design_doc_example():
    """The worked example of the design doc: pinned slots are skipped."""
    cache = make_cache([8])

    # Step 1: three new keys take slots 0, 1, 2 (pinned by the insert)
    k1 = dev([5, 9, 6])
    founds, _ = cache.lookup(k1, tids(3))
    assert not founds.any()
    s1, n_ev, *_ = cache.insert_and_evict(k1, tids(3))
    assert s1.tolist() == [0, 1, 2] and n_ev == 0
    assert refs(cache) == [1, 1, 1, 0, 0, 0, 0, 0]

    # Step 2: key 5 hits and is pinned a second time; four new keys take 3..6
    k2 = dev([5, 7, 12, 3, 10])
    founds, slots = cache.lookup(k2, tids(5))
    assert founds.tolist() == [True, False, False, False, False]
    assert slots[0].item() == 0
    assert refs(cache) == [2, 1, 1, 0, 0, 0, 0, 0]
    s2, n_ev, *_ = cache.insert_and_evict(k2[~founds], tids(4))
    assert s2.tolist() == [3, 4, 5, 6] and n_ev == 0
    assert refs(cache) == [2, 1, 1, 1, 1, 1, 1, 0]

    # Backward of step 1: slots 1 and 2 are free but keep their keys (still hit)
    cache.release(s1, tids(3))
    assert refs(cache) == [1, 0, 0, 1, 1, 1, 1, 0]
    founds, slots = peek(cache, [9, 6])
    assert founds.all() and slots.tolist() == [1, 2]
    assert refs(cache) == [1, 0, 0, 1, 1, 1, 1, 0]  # peek left nothing behind

    # Step 3: key 7 hits; two new keys: slot 7 (empty), slot 0 is pinned and
    # skipped, slot 1 is free and evicts key 9
    k3 = dev([7, 20, 26])
    founds, slots = cache.lookup(k3, tids(3))
    assert founds.tolist() == [True, False, False] and slots[0].item() == 3
    s3, n_ev, ev_keys, ev_slots, ev_tids = cache.insert_and_evict(
        k3[~founds], tids(2)
    )
    assert s3.tolist() == [7, 1]
    assert n_ev == 1 and ev_keys.tolist() == [9] and ev_slots.tolist() == [1]
    assert ev_tids.tolist() == [0]
    assert refs(cache) == [1, 1, 0, 2, 1, 1, 1, 1]
    assert cache._core.cursors().tolist() == [2]
    # the evicted key is gone, the new ones are found
    founds, slots = peek(cache, [9, 20, 26, 6])
    assert founds.tolist() == [False, True, True, True]
    assert slots.tolist() == [-1, 7, 1, 2]
    assert cache.check_errors() == 0


def test_lookup_pins_what_it_finds_and_only_that():
    cache = make_cache([8])
    insert_free(cache, [1, 2, 3])
    assert refs(cache)[:3] == [0, 0, 0]
    founds, slots = cache.lookup(dev([2, 99, 3]), tids(3))
    assert founds.tolist() == [True, False, True]
    assert slots.tolist() == [1, -1, 2]
    assert refs(cache) == [0, 1, 1, 0, 0, 0, 0, 0]
    cache.release(slots, tids(3))  # the -1 is skipped
    assert refs(cache) == [0] * 8
    assert cache.check_errors() == 0


def test_insert_pins_the_new_slots():
    cache = make_cache([4])
    s, *_ = insert(cache, [1, 2])
    assert refs(cache) == [1, 1, 0, 0]
    cache.release(s, tids(2))
    assert refs(cache) == [0, 0, 0, 0]


def test_unpinned_slots_are_taken_in_fifo_order():
    cache = make_cache([4])
    s, n_ev, *_ = insert_free(cache, [1, 2, 3, 4])
    assert s.tolist() == [0, 1, 2, 3] and n_ev == 0
    # nothing is pinned: the next keys evict in the order they came
    s, n_ev, ev_keys, ev_slots, _ = insert_free(cache, [10, 11])
    assert s.tolist() == [0, 1] and n_ev == 2
    assert ev_keys.tolist() == [1, 2] and ev_slots.tolist() == [0, 1]
    s, n_ev, ev_keys, _, _ = insert_free(cache, [12, 13, 14])
    assert s.tolist() == [2, 3, 0] and ev_keys.tolist() == [3, 4, 10]


def test_insert_fails_only_without_enough_unpinned_slots_and_changes_nothing():
    cache = make_cache([8])
    s, *_ = insert(cache, [1, 2, 3, 4, 5, 6])  # six pinned slots
    cursor_before = cache._core.cursors().tolist()
    refs_before = refs(cache)
    with pytest.raises(RuntimeError, match="free slots"):
        insert(cache, [100, 101, 102])  # 2 unpinned slots, 3 keys
    # nothing changed: same cursor, same counts, the old keys are all still there
    assert cache._core.cursors().tolist() == cursor_before
    assert refs(cache) == refs_before
    founds, slots = peek(cache, [1, 2, 3, 4, 5, 6, 100])
    assert founds.tolist() == [True] * 6 + [False]
    assert slots[:6].tolist() == [0, 1, 2, 3, 4, 5]
    # two keys fit, whichever slots are free
    s2, *_ = insert(cache, [100, 101])
    assert sorted(s2.tolist()) == [6, 7]
    # releasing pins makes room again
    cache.release(s[:3], tids(3))
    s3, n_ev, ev_keys, *_ = insert(cache, [200, 201, 202])
    assert sorted(ev_keys.tolist()) == [1, 2, 3]
    assert cache.check_errors() == 0


def test_a_scattered_pin_pattern_does_not_get_in_the_way():
    """Free slots are found wherever they are: every other slot is pinned."""
    cap = 4096
    cache = make_cache([cap])
    keys = list(range(cap))  # key i takes slot i
    insert_free(cache, keys)
    # pin the even slots by looking their keys up
    founds, pinned = cache.lookup(dev(keys[::2]), tids(cap // 2))
    assert founds.all() and (pinned % 2 == 0).all()
    n = cap // 2
    s2, n_ev, ev_keys, *_ = insert(cache, list(range(10_000, 10_000 + n)))
    assert n_ev == n
    # only the odd slots were free, and each is taken once (which key gets which
    # is not specified: the keys that found their slot pinned take later ones)
    assert sorted(s2.tolist()) == list(range(1, cap, 2))
    with pytest.raises(RuntimeError):
        insert(cache, [999_999])  # every slot is pinned now


def test_a_long_run_of_pinned_slots_is_skipped():
    """A few keys that find their slot pinned must get past a run much longer
    than a warp's window of 32, and the cursor ends just after the last slot."""
    cap = 4096
    cache = make_cache([cap])
    insert_free(cache, list(range(cap)))  # key i in slot i, the cursor wraps to 0
    run = 2000
    founds, pinned = cache.lookup(dev(list(range(run))), tids(run))  # pin 0 .. 1999
    assert founds.all()
    s, n_ev, ev_keys, *_ = insert(cache, [10_000 + i for i in range(10)])
    assert sorted(s.tolist()) == list(range(run, run + 10))
    assert sorted(ev_keys.tolist()) == list(range(run, run + 10))
    assert cache._core.cursors().tolist() == [run + 10]
    assert cache.check_errors() == 0


def test_many_keys_that_find_their_slot_pinned_each_get_a_different_one():
    """Hundreds of needy keys, in many warps, claim slots from the same counter."""
    cap = 20000
    cache = make_cache([cap])
    insert_free(cache, list(range(cap)))
    pinned = list(range(0, cap, 3))  # every third slot, the rest are free
    cache.lookup(dev(pinned), tids(len(pinned)))
    n = 5000
    s, n_ev, *_ = insert(cache, [100_000 + i for i in range(n)])
    got = s.tolist()
    assert len(set(got)) == n and not set(got) & set(pinned)
    free = [x for x in range(cap) if x % 3 != 0]
    assert sorted(got) == free[:n]  # exactly the first free slots from the cursor
    assert cache.check_errors() == 0


def test_erase_frees_a_slot_and_removes_the_key():
    cache = make_cache([4])
    insert_free(cache, [1, 2, 3, 4])
    cache.erase(dev([2, 777]), tids(2))  # 777 is not cached: ignored
    founds, _ = peek(cache, [1, 2, 3, 4])
    assert founds.tolist() == [True, False, True, True]
    # the hole is taken when the cursor reaches it, and evicts nothing
    s2, n_ev, ev_keys, *_ = insert(cache, [10])
    assert s2.tolist() == [0] and ev_keys.tolist() == [1]
    s3, n_ev, *_ = insert(cache, [11])
    assert s3.tolist() == [1] and n_ev == 0
    assert cache.size() == 4
    assert cache.check_errors() == 0


def test_erasing_a_pinned_key_keeps_its_slot_unavailable_until_released():
    cache = make_cache([2])
    s, *_ = insert(cache, [1, 2])  # both pinned
    cache.erase(dev([1]), tids(1))
    assert cache.check_errors() == 1  # erasing a slot a step still holds is reported
    assert not peek(cache, [1])[0].item()
    with pytest.raises(RuntimeError):
        insert(cache, [3])  # slot 0 is erased but still pinned
    cache.release(s[:1], tids(1))
    s2, *_ = insert(cache, [3])
    assert s2.tolist() == [0]


def test_releasing_a_slot_that_is_not_pinned_is_reported():
    cache = make_cache([4])
    s, *_ = insert(cache, [1])
    cache.release(s, tids(1))
    assert cache.check_errors() == 0
    cache.release(s, tids(1))  # one release too many
    assert cache.check_errors() == 1
    assert cache.check_errors() == 0  # reading clears it


def test_the_cache_reports_what_it_holds():
    caps = [16, 24]
    cache = make_cache(caps)
    assert cache.num_tables == 2 and cache.capacity() == sum(caps)
    assert cache.size() == 0 and cache.size(0) == 0 and cache.size(1) == 0
    insert(cache, [1, 2, 3], 0)
    insert(cache, [1, 2], 1)
    assert cache.size() == 5 and cache.size(0) == 3 and cache.size(1) == 2
    assert cache.capacity() == sum(caps)  # it does not change as keys come and go


def test_cache_metrics_is_a_snapshot_dataclass():
    cache = make_cache([8])
    assert cache.cache_metrics is None  # not recording
    cache.set_record_cache_metrics(True)

    insert_free(cache, [1, 2, 3])
    m1 = cache.cache_metrics
    assert isinstance(m1, CacheMetrics)
    assert (m1.num_inserts, m1.num_evictions) == (3, 0)
    assert (m1.num_lookups, m1.num_hits) == (0, 0)

    _, slots = cache.lookup(dev([2, 3, 9]), tids(3))
    cache.release(slots, tids(3))
    m2 = cache.cache_metrics
    assert (m2.num_lookups, m2.num_hits) == (3, 2)
    assert (m2.num_inserts, m2.num_evictions) == (3, 0)  # kept from the insert
    assert (m1.num_lookups, m1.num_hits) == (0, 0)  # an earlier snapshot did not change

    # eight new keys take every slot, so the three keys held so far are evicted
    insert_free(cache, list(range(11, 19)))
    m3 = cache.cache_metrics
    assert (m3.num_inserts, m3.num_evictions) == (8, 3)

    cache.set_record_cache_metrics(False)
    assert cache.cache_metrics is None


def test_reset_and_size():
    cache = make_cache([4, 6])
    insert(cache, [1, 2, 3], 0)
    insert(cache, [1, 2], 1)  # the same keys in another table are other entries
    assert cache.size() == 5 and cache.size(0) == 3 and cache.size(1) == 2
    cache.reset()
    assert cache.size() == 0
    assert refs(cache, 0) == [0] * 4
    assert not cache.lookup(dev([1, 2, 3]), tids(3))[0].any()
    s, n_ev, *_ = insert(cache, [7, 8, 9, 10], 0)
    assert s.tolist() == [0, 1, 2, 3] and n_ev == 0


def test_export_lists_every_cached_key_with_its_slot():
    cache = make_cache([20, 30])
    insert_free(cache, list(range(100, 115)), 0)
    insert_free(cache, list(range(200, 230)), 1)
    for t in range(2):
        got = {}
        for keys, slots in cache.export(t, batch_size=7):
            got.update(zip(keys.tolist(), slots.tolist()))
        sk = sorted(got)
        founds, slots = peek(cache, sk, t)
        assert founds.all()
        assert [got[k] for k in sk] == slots.tolist()
        assert len(got) == (15 if t == 0 else 30)


def test_uint64_keys_round_trip():
    cache = make_cache([8], key_dtype=torch.uint64)
    big = [2**63 + 5, 2**63 + 9, 17]
    keys = torch.tensor(
        [k - 2**64 if k >= 2**63 else k for k in big],
        dtype=torch.int64,
        device="cuda",
    ).view(torch.uint64)
    s, *_ = cache.insert_and_evict(keys, tids(3))
    founds, slots = cache.lookup(keys, tids(3))
    assert founds.all() and slots.tolist() == s.tolist()
    for k, _ in cache.export(0, 4):
        assert k.dtype == torch.uint64


# -------------------------------------------------------------- random model --


def _random_run(caps, depth, steps, seed, universe_factor=3):
    rng = random.Random(seed)
    cache = make_cache(caps)
    ref = RefCache(caps)
    T = len(caps)
    per_step = [max(1, c // (depth + 1)) for c in caps]  # fits while `depth` are in flight
    in_flight: List[Tuple[List[int], List[int]]] = []  # (slots, tids) of each step
    for step in range(steps):
        keys, tl = [], []
        for t in range(T):
            n = rng.randint(1, per_step[t])
            ks = rng.sample(range(universe_factor * caps[t]), n)
            keys += ks
            tl += [t] * n
        kt, tt = dev(keys), dev(tl)
        founds, slots = cache.lookup(kt, tt)  # pins what it finds
        ref_slots = ref.lookup(keys, tl)
        assert slots.tolist() == ref_slots, f"lookup differs at step {step}"
        assert founds.tolist() == [s >= 0 for s in ref_slots]

        miss = [i for i, s in enumerate(ref_slots) if s < 0]  # grouped by table already
        all_slots = list(ref_slots)
        if miss:
            mk, mt = [keys[i] for i in miss], [tl[i] for i in miss]
            got, n_ev, ev_k, ev_s, ev_t = cache.insert_and_evict(dev(mk), dev(mt))
            # pins the new slots too; checks the cache took the right slots
            exp_slots, exp_ev = ref.insert(mk, mt, assign=got.tolist())
            expected_ev = [(k, s, t) for k, s, t in zip(exp_ev, exp_slots, mt) if k != EMPTY]
            assert n_ev == len(expected_ev)
            assert list(zip(ev_k.tolist(), ev_s.tolist(), ev_t.tolist())) == expected_ev
            for i, s in zip(miss, exp_slots):
                all_slots[i] = s
        in_flight.append((all_slots, tl))
        if len(in_flight) >= depth:
            s_old, t_old = in_flight.pop(0)
            cache.release(dev(s_old), dev(t_old))
            ref.release(s_old, t_old)
        if step % 7 == 0 and rng.random() < 0.5:  # now and then an erase
            # the caller erases only keys no step holds (erasing a pinned slot is
            # an error, tested separately)
            idle = [k for k, s in ref.index[0].items() if ref.refs[0][s] == 0]
            victims = rng.sample(idle, min(3, len(idle)))
            if victims:
                cache.erase(dev(victims), tids(len(victims), 0))
                ref.erase(victims, [0] * len(victims))
        for t in range(T):  # the counts agree after every step
            assert refs(cache, t) == ref.refs[t], f"counts differ at step {step}"
    assert cache.check_errors() == 0
    for s_old, t_old in in_flight:  # drain
        cache.release(dev(s_old), dev(t_old))
        ref.release(s_old, t_old)
    for t in range(T):
        assert refs(cache, t) == [0] * caps[t]
        assert cache._core.keys_of(t).tolist() == ref.keys[t]
        assert cache.size(t) == len(ref.index[t])
    assert cache._core.cursors().tolist() == ref.cursor
    assert cache.check_errors() == 0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_steps_match_the_reference_model_one_table(seed):
    _random_run([97], depth=3, steps=120, seed=seed)


@pytest.mark.parametrize("seed", [1, 2])
def test_random_steps_match_the_reference_model_many_tables(seed):
    _random_run([64, 200, 37, 128], depth=2, steps=100, seed=seed)


def test_random_steps_with_the_table_almost_all_pinned():
    # depth 4 with per-step sizes cap // 5: up to 80% of every table is pinned
    _random_run([50, 120], depth=4, steps=150, seed=11, universe_factor=2)


def test_a_large_table_with_many_collisions_in_the_index():
    rng = random.Random(5)
    cap = 20000
    cache = make_cache([cap])
    ref = RefCache([cap])
    for step in range(6):
        keys = rng.sample(range(5 * cap), cap // 2)
        kt, tt = dev(keys), tids(len(keys))
        founds, slots = cache.lookup(kt, tt)
        ref_slots = ref.lookup(keys, [0] * len(keys))
        assert slots.tolist() == ref_slots
        miss = [k for k, s in zip(keys, ref_slots) if s < 0]
        got, n_ev, *_ = insert(cache, miss)
        exp, _ = ref.insert(miss, [0] * len(miss), assign=got.tolist())
        # the step is over: release everything it pinned
        all_slots = list(ref_slots)
        j = 0
        for i, s in enumerate(ref_slots):
            if s < 0:
                all_slots[i] = exp[j]
                j += 1
        cache.release(dev(all_slots), tt)
        ref.release(all_slots, [0] * len(keys))
    assert cache.check_errors() == 0


# ------------------------------------------------------------------ streams --


def test_releases_on_another_stream_never_free_a_slot_that_is_still_used():
    """Prefetch (lookup, insert) on one stream, releases on another that runs
    concurrently with the next prefetches. A release only makes slots takeable,
    so a slot a step still holds must never be the victim of an insert."""
    cap, depth, steps = 4000, 3, 200
    cache = make_cache([cap])
    rng = random.Random(3)
    main = torch.cuda.current_stream()
    back = torch.cuda.Stream()
    in_use: List[Tuple[int, List[int]]] = []  # (step, keys), not yet released
    held: Dict[int, torch.Tensor] = {}
    for step in range(steps):
        n = cap // (depth + 1) - 50
        keys = rng.sample(range(6 * cap), n)
        kt = dev(keys)
        founds, slots = cache.lookup(kt, tids(n))  # pins what it finds
        miss = kt[~founds]
        if miss.numel():
            got, n_ev, ev_k, *_ = cache.insert_and_evict(miss, tids(miss.numel()))
            # Keys that are surely held: those of the steps whose release has not
            # been issued yet (the release of step s - depth, issued at the end of
            # the last iteration, may be running right now and is not in in_use),
            # and this step's own hits.
            surely_held = {k for _, ks in in_use for k in ks} | set(kt[founds].tolist())
            assert not (set(ev_k.tolist()) & surely_held), "a held slot was taken"
            slots = slots.clone()
            slots[~founds] = got
        in_use.append((step, keys))
        held[step] = slots
        if len(in_use) >= depth:
            old_step, _ = in_use.pop(0)
            slots_old = held.pop(old_step)
            slots_old.record_stream(back)
            ev = torch.cuda.Event()
            ev.record(main)  # the step's rows are done on the main stream
            with torch.cuda.stream(back):
                back.wait_event(ev)
                cache.release(slots_old, tids(slots_old.numel()))
    torch.cuda.synchronize()
    assert cache.check_errors() == 0


# ----------------------------------------------------------------- row pool --


def test_the_values_work_with_load_and_store():
    cache = make_cache([16, 24], dims=[8, 8])
    assert cache.num_tables == 2 and cache.capacity() == 40
    values = cache.values
    keys = list(range(100, 110))
    s0, *_ = insert(cache, keys, 0)
    s1, *_ = insert(cache, keys[:6], 1)
    slots = torch.cat([s0, s1])
    tl = torch.cat([tids(len(keys), 0), tids(6, 1)])
    rows = torch.randn(slots.numel(), values.max_value_dim, device="cuda")
    values.store_padded(slots, tl, rows)
    torch.testing.assert_close(values.load_padded(slots, tl, CopyMode.VALUE), rows)
    # the rows are by slot: after an eviction the old value is still there until
    # the new one is written
    cache.release(s0, tids(len(keys), 0))
    pre = values.load_padded(s0[:2], tids(2, 0), CopyMode.VALUE)
    _, n_ev, _, ev_slots, ev_tids = insert(cache, list(range(500, 515)), 0)
    assert n_ev > 0
    assert (
        values.load_padded(ev_slots[:1], ev_tids[:1], CopyMode.VALUE).shape[1]
        == values.max_value_dim
    )
    torch.testing.assert_close(values.load_padded(s0[:2], tids(2, 0), CopyMode.VALUE), pre)


def test_the_row_pool_is_a_fixed_plain_buffer():
    """A cache's rows are a plain device tensor per table: they never grow."""
    caps = [16, 24]
    cache = make_cache(caps, dims=[8, 8])
    buffers = cache.values.buffers
    for rows, buf in zip(caps, buffers):
        assert isinstance(buf, FixedDeviceBuffer)
        assert buf.is_device_buffer() and buf.shape == (rows, 8)
        assert buf.tensor().is_cuda and tuple(buf.tensor().shape) == (rows, 8)
        assert buf.num_bytes() == buf.allocated_bytes() == rows * 8 * 4
        with pytest.raises(NotImplementedError):
            buf.extend((4, 8))
    # what the kernels address: the buffers' own pointers, which never move
    ptrs = [b.tensor().data_ptr() for b in buffers]
    assert cache.values.ptrs.tolist() == ptrs
    insert(cache, list(range(100, 110)), 0)
    assert [b.tensor().data_ptr() for b in buffers] == ptrs
    # the memory report: no optimizer state (SGD), and the peak is the size
    assert cache.value_bytes() == [
        (16 * 8 * 4, 16 * 8 * 4, 0, 16 * 8 * 4, 16 * 8 * 4, 0),
        (24 * 8 * 4, 24 * 8 * 4, 0, 24 * 8 * 4, 24 * 8 * 4, 0),
    ]


def test_the_constructor_checks_its_inputs():
    caps = [16, 24]
    # a cache never grows, so its values must be a fixed pool
    with pytest.raises(ValueError, match="fixed"):
        DynamicEmbCache(make_values(caps, kind="device"))
    with pytest.raises(ValueError, match="key_dtype"):
        DynamicEmbCache(make_values(caps), torch.int32)
    # the capacities are the rows of the values
    cache = DynamicEmbCache(make_values(caps))
    assert [cache.capacity(t) for t in range(len(caps))] == caps and cache.capacity() == sum(caps)
