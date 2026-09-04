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

"""Table-level tests for the retain-evicted-keys collection path
(``LinearBucketTable.insert(collect_evicted=True)`` -> C++
``table_insert_collect_evicted``): a plain insert on a full bucket evicts a
victim and, when collecting, retains the victim's (key, table_id).

Two kernels back this, both covered here:
  * LruLfu score policy -> the ``dyn_emb_insert_collect_entry`` cubin
    (ranked comparator);
  * a single-score policy (ASSIGN) -> the AoT ``table_insert_collect_kernel``
    (DefaultEvictor reduce).

The collection contract is Evict-only (a full-bucket victim), NOT Busy, so every
test drives the new keys to the winning score, making eviction deterministic and
Busy-free -- then ``num_evicted == occupied + new - capacity`` and a whole-set
oracle (mirroring test_lru_lfu.py) predicts the exact victim set.
"""

import numpy as np
import pytest
import torch
from dynamicemb import DynamicEmbCheckMode, DynamicEmbScoreStrategy, EvictedItemMode
from dynamicemb.dynamicemb_config import DynamicEmbEvictStrategy, DynamicEmbTableOptions
from dynamicemb.key_value_table import (
    DynamicEmbStorage,
    _append_evicted,
    _pop_state_evicted_keys,
)
from dynamicemb.optimizer import OptimizerArgs, SGDDynamicEmbeddingOptimizer
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import InsertResult, ScorePolicy

# A compound (TIMESTAMP, LFU) strategy: eviction ranks by the default Lex
# comparator (frequency asc, then oldest timestamp), same as the cubin tests.
LRU_LFU_TS_FIRST = (DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU)


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


def _lru_lfu_table(capacity, bucket_capacity=128):
    caps = [capacity] if isinstance(capacity, int) else list(capacity)
    return get_scored_table(
        capacity=caps,
        bucket_capacity=bucket_capacity,
        key_type=torch.int64,
        score_specs=[
            ScoreSpec(name="frequency", policy=ScorePolicy.LRU_LFU, is_reduction=True)
        ],
    )


def _assign_table(capacity, bucket_capacity=128):
    caps = [capacity] if isinstance(capacity, int) else list(capacity)
    return get_scored_table(
        capacity=caps,
        bucket_capacity=bucket_capacity,
        key_type=torch.int64,
        score_specs=[ScoreSpec(name="score1", policy=ScorePolicy.ASSIGN)],
    )


def _ir(n, table, device):
    return torch.empty(n, dtype=table.result_type, device=device).fill_(
        InsertResult.INIT.value
    )


def _plain_insert(table, keys, tids, value, policy, name):
    table.insert(
        keys,
        tids,
        ScoreArg(name=name, value=value, policy=policy),
        _ir(keys.numel(), table, keys.device),
    )


def _insert_collect(table, keys, tids, value, policy, name):
    """Plain insert with collection. Returns (indices, m, evicted_keys[:m],
    evicted_table_ids[:m]) -- the evicted_* buffers are already sliced to the
    number actually evicted."""
    indices, num_evicted, evicted_keys, evicted_tids = table.insert(
        keys,
        tids,
        ScoreArg(name=name, value=value, policy=policy),
        _ir(keys.numel(), table, keys.device),
        collect_evicted=True,
    )
    m = int(num_evicted)
    return indices, m, evicted_keys[:m], evicted_tids[:m]


def test_collect_evicted_lrulfu_cubin_matches_oracle(current_device):
    """LruLfu path (dyn_emb_insert_collect_entry cubin, default Lex comparator).
    All keys share frequency 1, so eviction ranks purely by the timestamp
    tiebreak (older first); sync-separated insert groups give strictly increasing
    timestamp tiers. A whole-set oracle asserts the collected victim set equals
    the oldest keys."""
    device = current_device
    bc = 128
    table = _lru_lfu_table(bc, bucket_capacity=bc)  # single bucket: all compete
    assert table.score_fn_key_ == 0, "default path must use the built-in Lex cubin"

    G, per = 10, 10
    n = G * per
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    ones = torch.ones(n, dtype=torch.uint64, device=device)
    idx_parts = []
    for g in range(G):
        s = g * per
        indices = table.insert(
            keys[s : s + per],
            tids[s : s + per],
            ScoreArg(
                name="frequency", value=ones[s : s + per], policy=ScorePolicy.LRU_LFU
            ),
            _ir(per, table, device),
        )
        idx_parts.append(indices)
        torch.cuda.synchronize()
    idx = torch.cat(idx_parts)

    blk = table.gather_score_blocks(0, idx).cpu().numpy()
    ts = blk[:, 0].astype(np.uint64)
    assert np.all(blk[:, 1].astype(np.uint64) == 1), "every key must be at frequency 1"

    # Overflow with newest-timestamp frequency-1 keys via plain insert + collect;
    # being newest they win, so victims are exactly the oldest pre-existing keys.
    n_new = 58
    new_keys = torch.arange(100000, 100000 + n_new, dtype=torch.int64, device=device)
    new_tids = torch.zeros(n_new, dtype=torch.int64, device=device)
    new_freq = torch.ones(n_new, dtype=torch.uint64, device=device)
    _, m, evicted_keys, evicted_tids = _insert_collect(
        table, new_keys, new_tids, new_freq, ScorePolicy.LRU_LFU, "frequency"
    )
    assert m == n + n_new - bc, "Evict-only count == occupied + new - capacity"

    order = np.argsort(ts, kind="stable")
    assert ts[order[m]] > ts[order[m - 1]], "eviction boundary must fall on a ts gap"
    keys_cpu = keys.cpu().numpy()
    predicted = set(int(keys_cpu[i]) for i in order[:m])
    actual = set(int(k) for k in evicted_keys.tolist())
    assert (
        actual == predicted
    ), "cubin insert-collect must retain exactly the oracle victim set"
    assert torch.all(evicted_tids == 0), "single table -> every evicted table_id is 0"


def test_collect_evicted_aot_assign_matches_oracle(current_device):
    """AoT path (table_insert_collect_kernel, single-score DefaultEvictor). Old
    keys carry distinct low scores (score == key); new keys carry a far higher
    score so they always win. Victims are the lowest-score old keys; score == key
    makes the oracle the m smallest keys."""
    device = current_device
    bc = 128
    table = _assign_table(bc, bucket_capacity=bc)  # single bucket, non-LruLfu -> AoT

    n = 100
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    scores = torch.arange(1, 1 + n, dtype=torch.int64, device=device).to(
        torch.uint64
    )  # score == key
    _plain_insert(table, keys, tids, scores, ScorePolicy.ASSIGN, "score1")
    torch.cuda.synchronize()

    n_new = 58
    new_keys = torch.arange(100000, 100000 + n_new, dtype=torch.int64, device=device)
    new_tids = torch.zeros(n_new, dtype=torch.int64, device=device)
    new_scores = torch.full((n_new,), 10_000_000, dtype=torch.uint64, device=device)
    _, m, evicted_keys, evicted_tids = _insert_collect(
        table, new_keys, new_tids, new_scores, ScorePolicy.ASSIGN, "score1"
    )
    assert m == n + n_new - bc, "Evict-only count == occupied + new - capacity"

    predicted = set(range(1, 1 + m))  # the m lowest scores == the m smallest keys
    actual = set(int(k) for k in evicted_keys.tolist())
    assert (
        actual == predicted
    ), "AoT insert-collect must retain exactly the lowest-score victims"
    assert torch.all(evicted_tids == 0)


def test_collect_evicted_table_id_routing(current_device):
    """Two logical tables share one storage (distinct bucket regions). A single
    mixed-table_id insert overflows both; every collected victim's table_id must
    match the table its key belongs to (disjoint key ranges make this checkable)."""
    device = current_device
    bc = 128
    table = _assign_table([bc, bc], bucket_capacity=bc)  # 2 tables, 1 bucket each

    # Fill table 0 with keys [1,100], table 1 with keys [1000,1099]; score == key.
    k0 = torch.arange(1, 101, dtype=torch.int64, device=device)
    k1 = torch.arange(1000, 1100, dtype=torch.int64, device=device)
    fill_keys = torch.cat([k0, k1])
    fill_tids = torch.cat(
        [
            torch.zeros(100, dtype=torch.int64, device=device),
            torch.ones(100, dtype=torch.int64, device=device),
        ]
    )
    fill_scores = fill_keys.to(torch.uint64)
    _plain_insert(
        table, fill_keys, fill_tids, fill_scores, ScorePolicy.ASSIGN, "score1"
    )
    torch.cuda.synchronize()

    # Overflow both tables in one mixed batch with far-higher scores.
    nn = 40
    nk0 = torch.arange(100000, 100000 + nn, dtype=torch.int64, device=device)
    nk1 = torch.arange(200000, 200000 + nn, dtype=torch.int64, device=device)
    new_keys = torch.cat([nk0, nk1])
    new_tids = torch.cat(
        [
            torch.zeros(nn, dtype=torch.int64, device=device),
            torch.ones(nn, dtype=torch.int64, device=device),
        ]
    )
    new_scores = torch.full((2 * nn,), 10_000_000, dtype=torch.uint64, device=device)
    _, m, evicted_keys, evicted_tids = _insert_collect(
        table, new_keys, new_tids, new_scores, ScorePolicy.ASSIGN, "score1"
    )
    # Each table: 100 old + 40 new - 128 = 12 victims -> 24 total.
    assert m == 2 * (100 + nn - bc)

    ek = evicted_keys.cpu().tolist()
    et = evicted_tids.cpu().tolist()
    for key, tid in zip(ek, et):
        if 1 <= key <= 100:
            assert tid == 0, f"key {key} belongs to table 0 but table_id={tid}"
        elif 1000 <= key <= 1099:
            assert tid == 1, f"key {key} belongs to table 1 but table_id={tid}"
        else:
            raise AssertionError(
                f"unexpected evicted key {key} (a new key must never be a victim)"
            )
    # Both tables actually contributed victims.
    assert 0 in et and 1 in et, "both tables must have evicted something"


def test_collect_evicted_no_eviction_empty(current_device):
    """No overflow -> nothing collected: num_evicted == 0 and empty buffers."""
    device = current_device
    table = _assign_table(4096, bucket_capacity=128)
    n = 200
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    scores = torch.arange(1, 1 + n, dtype=torch.int64, device=device).to(torch.uint64)
    _, m, evicted_keys, evicted_tids = _insert_collect(
        table, keys, tids, scores, ScorePolicy.ASSIGN, "score1"
    )
    assert m == 0
    assert evicted_keys.numel() == 0 and evicted_tids.numel() == 0


def test_collect_evicted_determinism_mode_raises(current_device, monkeypatch):
    """collect_evicted is unsupported under DEMB_DETERMINISM_MODE -> RuntimeError
    (raised before the deterministic-insert path, so num_scores is irrelevant)."""
    monkeypatch.setenv("DEMB_DETERMINISM_MODE", "1")
    device = current_device
    table = _assign_table(128, bucket_capacity=128)
    n = 8
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    scores = torch.arange(1, 1 + n, dtype=torch.int64, device=device).to(torch.uint64)
    with pytest.raises(RuntimeError, match="collect_evicted"):
        table.insert(
            keys,
            tids,
            ScoreArg(name="score1", value=scores, policy=ScorePolicy.ASSIGN),
            _ir(n, table, device),
            collect_evicted=True,
        )


# ---------------------------------------------------------------------------
# Storage-level tests: the end-to-end retain flow through DynamicEmbStorage
# (a last tier). insert() routes to _insert_key_values, which collects into the
# state's retain buffers; pop_evicted_keys() drains them (unique, per table).
# ---------------------------------------------------------------------------


def _retain_storage(dim=8, max_capacity=128, bucket_capacity=128, retain=True):
    """A single-tier DynamicEmbStorage. init_capacity == max_capacity disables
    rehash growth, so a full bucket evicts (rather than expanding) -- the last-tier
    eviction the retain hook targets. LruLfu score policy + LFU evict strategy
    mirror the validated _lru_lfu_storage fixture."""
    device_id = torch.cuda.current_device()
    opts = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=device_id,
            dim=dim,
            max_capacity=max_capacity,
            init_capacity=max_capacity,
            bucket_capacity=bucket_capacity,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=LRU_LFU_TS_FIRST,
            evict_strategy=DynamicEmbEvictStrategy.LFU,
            evicted_item_mode=(
                EvictedItemMode.RETAIN_KEY if retain else EvictedItemMode.DISCARD
            ),
        )
    ]
    return DynamicEmbStorage(opts, SGDDynamicEmbeddingOptimizer(OptimizerArgs()))


def _storage_insert(storage, keys, dim, device):
    n = keys.numel()
    storage.insert(
        keys,
        torch.zeros(n, dtype=torch.int64, device=device),
        torch.randn(n, dim, dtype=torch.float32, device=device),
        scores=torch.ones(n, dtype=torch.uint64, device=device),
    )
    torch.cuda.synchronize()


def test_erased_and_evicted_use_separate_buffers(current_device):
    """An erase and an eviction are recorded in different buffers.

    They mean different things to a consumer -- an eviction is the table running
    out of room, an erase is somebody asking for the key to go, and only the
    latter is a removal a replica has to perform for itself. Mixing them would
    make a replay erase keys that were merely evicted, dropping live data.
    """
    device = current_device
    dim = 8
    # Same full-bucket overflow as test_storage_retain_end_to_end: 128 keys fill
    # the single bucket, then 58 newer ones evict the 58 oldest.
    storage = _retain_storage(
        dim=dim, max_capacity=128, bucket_capacity=128, retain=True
    )
    _storage_insert(
        storage, torch.arange(1, 129, dtype=torch.int64, device=device), dim, device
    )
    _storage_insert(
        storage, torch.arange(1000, 1058, dtype=torch.int64, device=device), dim, device
    )

    evicted = storage.pop_evicted_keys(0)
    assert evicted.numel() > 0, "a full-bucket overflow must have evicted something"
    assert storage.pop_erased_keys(0).numel() == 0, "no erase happened yet"

    # Erase a key that is still resident, i.e. one that was NOT evicted.
    resident = torch.tensor([1000], dtype=torch.int64, device=device)
    assert not bool(torch.isin(resident, evicted).any())
    assert storage.erase_keys(0, resident, EvictedItemMode.RETAIN_KEY) == 1

    erased = storage.pop_erased_keys(0)
    assert erased.tolist() == resident.tolist()
    # The erase must not have leaked into the eviction buffer, and both buffers
    # are read-and-clear.
    assert storage.pop_evicted_keys(0).numel() == 0
    assert storage.pop_erased_keys(0).numel() == 0


def test_erase_retention_is_per_call_not_per_table(current_device):
    """Whether an erase is recorded is the call's decision, not the table's.

    Retaining evictions has to be configured up front (it swaps the insert
    kernel), but an erase already holds its keys, so nothing has to be prepared
    -- which is why the same table can record one erase and not the next, and
    why a table configured to DISCARD evictions can still report its erases.
    """
    device = current_device
    dim = 8
    storage = _retain_storage(dim=dim, retain=False)  # DISCARD: no eviction retention
    keys = torch.arange(1, 9, dtype=torch.int64, device=device)
    _storage_insert(storage, keys, dim, device)

    assert storage.erase_keys(0, keys[:3], EvictedItemMode.DISCARD) == 3
    assert storage.pop_erased_keys(0).numel() == 0, "DISCARD must record nothing"

    assert storage.erase_keys(0, keys[3:6], EvictedItemMode.RETAIN_KEY) == 3
    assert storage.pop_erased_keys(0).tolist() == keys[3:6].tolist()
    assert storage.pop_evicted_keys(0).numel() == 0


def test_storage_retain_end_to_end(current_device):
    """DynamicEmbStorage (a last tier) with evicted_item_mode=RETAIN_KEY: a full-bucket
    insert evicts; pop returns the unique evicted keys; those keys are gone from the
    table; a second pop is empty (read-and-clear / incremental)."""
    device = current_device
    dim = 8
    storage = _retain_storage(
        dim=dim, max_capacity=128, bucket_capacity=128, retain=True
    )
    _storage_insert(
        storage, torch.arange(1, 129, dtype=torch.int64, device=device), dim, device
    )
    # 58 new keys (freq 1, newer ts) evict the 58 oldest (Lex: freq tie -> ts asc).
    _storage_insert(
        storage, torch.arange(1000, 1058, dtype=torch.int64, device=device), dim, device
    )

    evk = storage.pop_evicted_keys(0)
    assert evk.numel() > 0, "a full-bucket overflow must have evicted something"
    assert evk.numel() == len(set(evk.tolist())), "pop must return unique keys"

    name = storage._state.score_policy.name
    _, founds, _ = storage.key_index_map.lookup(
        evk, torch.zeros_like(evk), ScoreArg(name=name, policy=ScorePolicy.CONST)
    )
    assert not bool(founds.any()), "evicted keys must no longer be in the table"

    assert (
        storage.pop_evicted_keys(0).numel() == 0
    ), "second pop is empty (read-and-clear)"


def test_storage_retain_disabled_empty(current_device):
    """evicted_item_mode=DISCARD: eviction still happens but nothing is retained."""
    device = current_device
    dim = 8
    storage = _retain_storage(
        dim=dim, max_capacity=128, bucket_capacity=128, retain=False
    )
    _storage_insert(
        storage, torch.arange(1, 129, dtype=torch.int64, device=device), dim, device
    )
    _storage_insert(
        storage, torch.arange(1000, 1058, dtype=torch.int64, device=device), dim, device
    )
    assert (
        storage.pop_evicted_keys(0).numel() == 0
    ), "retain disabled -> pop is always empty"


def test_pop_dedup_and_table_id_filter(current_device):
    """Unit test of the retain-buffer logic (append + pop) decoupled from eviction:
    a key appended in two batches de-duplicates on pop; pop(tid) returns only that
    table's keys and clears only them (the other table's survive)."""
    device = current_device
    storage = _retain_storage(retain=True)
    state = storage._state

    def _append(keys, tids):
        _append_evicted(
            state,
            torch.tensor(keys, dtype=torch.int64, device=device),
            torch.tensor(tids, dtype=torch.int64, device=device),
            torch.tensor([len(keys)], dtype=torch.int64, device=device),
        )

    _append([1, 2, 3], [0, 0, 0])
    _append([2, 3, 4], [0, 0, 0])  # 2, 3 repeat across batches
    _append([5, 6], [0, 1])  # table 0: key 5; table 1: key 6

    ev0 = _pop_state_evicted_keys(state, 0)
    assert sorted(ev0.tolist()) == [
        1,
        2,
        3,
        4,
        5,
    ], "table 0 pop must dedup and exclude table 1"
    ev1 = _pop_state_evicted_keys(state, 1)
    assert sorted(ev1.tolist()) == [6], "table 1 keys must survive table 0's pop"
    assert _pop_state_evicted_keys(state, 0).numel() == 0
    assert _pop_state_evicted_keys(state, 1).numel() == 0
