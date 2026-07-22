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

"""Table-level tests for the LruLfu score policy (2 AoS score words per key:
word 0 = last-access timestamp, word 1 = frequency; eviction ranks by frequency,
the timestamp serves time-based incremental dump). Scores are read back via the
gather_score_blocks primitive so the CUDA kernels are directly observable."""

import math

import numpy as np
import pytest
import torch
from dynamicemb import DynamicEmbCheckMode, DynamicEmbScoreStrategy
from dynamicemb.dynamicemb_config import (
    DynamicEmbEvictStrategy,
    DynamicEmbTableOptions,
    ScoreStrategy,
)
from dynamicemb.key_value_table import DynamicEmbStorage, _expand_tables_impl
from dynamicemb.optimizer import OptimizerArgs, SGDDynamicEmbeddingOptimizer
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import InsertResult, ScorePolicy


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


def _lru_lfu_table(capacity, bucket_capacity=128, key_type=torch.int64):
    return get_scored_table(
        capacity=[capacity],
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[
            ScoreSpec(name="frequency", policy=ScorePolicy.LRU_LFU, is_reduction=True)
        ],
    )


def _insert(table, keys, tids, freq, policy=ScorePolicy.LRU_LFU):
    n = keys.numel()
    score_out = torch.empty(n, dtype=torch.int64, device=keys.device)
    insert_results = torch.empty(n, dtype=table.result_type, device=keys.device).fill_(
        InsertResult.INIT.value
    )
    indices = table.insert(
        keys,
        tids,
        ScoreArg(name="frequency", value=freq, policy=policy),
        insert_results,
        score_out=score_out,
    )
    return indices, score_out


def test_lru_lfu_num_scores():
    """LruLfu occupies two physical score words per key."""
    table = _lru_lfu_table(4096)
    assert table.num_scores_ == 2


def test_lru_lfu_frequency_accumulation(current_device):
    """word 1 accumulates the frequency; word 0 is a (monotonic) timestamp; and a
    CONST lookup returns the frequency (word 1), not the timestamp (word 0)."""
    device = torch.cuda.current_device()
    table = _lru_lfu_table(4096)
    n = 100
    keys = torch.arange(1000, 1000 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    freq = torch.ones(n, dtype=torch.uint64, device=device)

    indices, score_out = _insert(table, keys, tids, freq)
    assert torch.all(score_out == 1), "frequency after first insert should be 1"

    blocks = table.gather_score_blocks(0, indices)  # [n, 2] = (timestamp, freq)
    assert tuple(blocks.shape) == (n, 2)
    ts0 = blocks[:, 0].clone()
    assert torch.all(blocks[:, 1] == 1), "word 1 (frequency) should be 1"
    assert torch.all(ts0 > 0), "word 0 (timestamp) should be a device timer value"

    # Each LruLfu lookup is an access: accumulate frequency and re-stamp time.
    out2, founds, _ = table.lookup(
        keys, tids, ScoreArg(name="frequency", value=freq, policy=ScorePolicy.LRU_LFU)
    )
    assert torch.all(founds)
    assert torch.all(out2 == 2), "frequency after one lookup should be 2"

    out3, _, _ = table.lookup(
        keys, tids, ScoreArg(name="frequency", value=freq, policy=ScorePolicy.LRU_LFU)
    )
    assert torch.all(out3 == 3)

    # CONST (eval) lookup must return the reduction score == frequency (word 1),
    # NOT word 0 (the timestamp).
    out_const, _, _ = table.lookup(
        keys, tids, ScoreArg(name="frequency", value=None, policy=ScorePolicy.CONST)
    )
    assert torch.all(
        out_const == 3
    ), "CONST lookup must return frequency (word 1), not the timestamp"

    blocks2 = table.gather_score_blocks(0, indices)
    assert torch.all(blocks2[:, 1] == 3)
    assert torch.all(
        blocks2[:, 0] >= ts0
    ), "timestamp (word 0) must be non-decreasing after further accesses"


def test_lru_lfu_eviction_by_frequency(current_device):
    """Eviction ranks by frequency (word 1): a small set of frequently-accessed
    keys survives even though newly inserted keys have more recent timestamps.
    Exercises the 2-score reduce() path. The hot set is kept well below capacity
    so it is never forced out just for lack of room -- only the frequency ranking
    should matter."""
    device = torch.cuda.current_device()
    bc = 128
    table = _lru_lfu_table(bc, bucket_capacity=bc)  # single bucket: all keys compete

    # Fill ~80% of the bucket with frequency-1 keys.
    n = 100
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    ones = torch.ones(n, dtype=torch.uint64, device=device)
    idx, _ = _insert(table, keys, tids, ones)

    # Boost a small "hot" subset (10 keys) far above the rest.
    n_hot = 10
    hot = keys[:n_hot]
    hot_tids = tids[:n_hot]
    hot_freq = torch.ones(n_hot, dtype=torch.uint64, device=device)
    for _ in range(20):
        table.lookup(
            hot,
            hot_tids,
            ScoreArg(name="frequency", value=hot_freq, policy=ScorePolicy.LRU_LFU),
        )

    # Sanity: the boost actually raised the hot keys' frequency (word 1).
    hot_blocks = table.gather_score_blocks(0, idx[:n_hot])
    assert torch.all(hot_blocks[:, 1] == 21), "hot keys should have frequency 1 + 20"

    # Insert enough brand-new (frequency 1) keys to overflow the bucket and force
    # eviction. There are many more frequency-1 keys (cold + new) than eviction
    # slots, so a correct LFU must evict only those, never the hot keys.
    n_new = 100
    new_keys = torch.arange(100000, 100000 + n_new, dtype=torch.int64, device=device)
    new_tids = torch.zeros(n_new, dtype=torch.int64, device=device)
    new_freq = torch.ones(n_new, dtype=torch.uint64, device=device)
    ir = torch.empty(n_new, dtype=table.result_type, device=device).fill_(
        InsertResult.INIT.value
    )
    _, num_evicted, evicted_keys, _, _, _ = table.insert_and_evict(
        new_keys,
        new_tids,
        ScoreArg(name="frequency", value=new_freq, policy=ScorePolicy.LRU_LFU),
        ir,
    )

    assert num_evicted > 0, "eviction should have occurred (bucket overflowed)"
    evicted_set = set(int(k) for k in evicted_keys.tolist())
    assert evicted_set.isdisjoint(
        set(int(k) for k in hot.tolist())
    ), "high-frequency (hot) keys must not be evicted by LFU"

    # The hot keys must still be present.
    _, founds, _ = table.lookup(
        hot, hot_tids, ScoreArg(name="frequency", value=None, policy=ScorePolicy.CONST)
    )
    assert torch.all(founds), "high-frequency keys must survive LFU eviction"


def test_lru_lfu_gather_scatter_roundtrip(current_device):
    """gather_score_blocks -> scatter_score_blocks -> gather_score_blocks preserves
    both score words (used by dump/load)."""
    device = torch.cuda.current_device()
    n = 50
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    freq = torch.arange(1, 1 + n, dtype=torch.int64, device=device).to(
        torch.uint64
    )  # varied

    src = _lru_lfu_table(4096)
    idx_src, _ = _insert(src, keys, tids, freq)
    blocks = src.gather_score_blocks(0, idx_src)

    dst = _lru_lfu_table(4096)
    # Place keys without touching scores (CONST), then restore the full block.
    idx_dst, _ = _insert(dst, keys, tids, None, policy=ScorePolicy.CONST)
    dst.scatter_score_blocks(0, idx_dst, blocks)
    assert torch.equal(
        dst.gather_score_blocks(0, idx_dst), blocks
    ), "scatter then gather must round-trip both score words"


def test_lru_lfu_copy_score_blocks_roundtrip(current_device):
    """copy_score_blocks_from copies all score words between tables of different
    capacity (the rehash primitive)."""
    device = torch.cuda.current_device()
    n = 50
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    freq = torch.arange(1, 1 + n, dtype=torch.int64, device=device).to(torch.uint64)

    src = _lru_lfu_table(4096)
    idx_src, _ = _insert(src, keys, tids, freq)
    blocks = src.gather_score_blocks(0, idx_src)

    # Rehash target: a larger table (keys re-hash into different buckets).
    dst = _lru_lfu_table(8192)
    idx_dst, _ = _insert(dst, keys, tids, None, policy=ScorePolicy.CONST)
    dst.copy_score_blocks_from(src, 0, idx_src, idx_dst)
    assert torch.equal(
        dst.gather_score_blocks(0, idx_dst), blocks
    ), "copy_score_blocks_from must preserve both score words across tables"


# ---------------------------------------------------------------------------
# Storage-level integration: dump/load round-trip and rehash preservation for
# the LruLfu layout (the (TIMESTAMP, LFU) compound score). These exercise the
# Python glue around the gather/scatter/copy kernels (export_keys_values_iter,
# _dump_table, _iter_batches_from_files, _load_key_values, _expand_tables_impl).
# ---------------------------------------------------------------------------

# The compound score in both column orders. The two are equivalent at runtime
# (identical physical LruLfu layout); they differ only in checkpoint column order.
LRU_LFU_TS_FIRST = (DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU)
LRU_LFU_LFU_FIRST = (DynamicEmbScoreStrategy.LFU, DynamicEmbScoreStrategy.TIMESTAMP)


def _lru_lfu_storage(
    dim=8,
    max_capacity=4096,
    init_capacity=None,
    score_strategy: ScoreStrategy = LRU_LFU_TS_FIRST,
    score_function=None,
):
    device_id = torch.cuda.current_device()
    opts = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=device_id,
            dim=dim,
            max_capacity=max_capacity,
            # Constructing DynamicEmbStorage directly bypasses the batched layer's
            # planner/_create_score, so set init_capacity and the LFU evict
            # strategy explicitly (batched normally derives these).
            init_capacity=init_capacity if init_capacity is not None else max_capacity,
            bucket_capacity=128,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=score_strategy,
            evict_strategy=DynamicEmbEvictStrategy.LFU,
            score_function=score_function,
        )
    ]
    return DynamicEmbStorage(opts, SGDDynamicEmbeddingOptimizer(OptimizerArgs()))


def _read_score_blocks(storage, keys, tids):
    """Look up keys and gather their full [N, num_scores] score blocks."""
    name = storage._state.score_policy.name
    _, founds, idx = storage.key_index_map.lookup(
        keys, tids, ScoreArg(name=name, policy=ScorePolicy.CONST)
    )
    assert torch.all(founds), "all keys must be present"
    return storage.key_index_map.gather_score_blocks(0, idx)


@pytest.mark.parametrize(
    "score_strategy", [LRU_LFU_TS_FIRST, LRU_LFU_LFU_FIRST], ids=["ts_first", "lfu_first"]
)
def test_lru_lfu_storage_dump_load_roundtrip(current_device, tmp_path, score_strategy):
    """DynamicEmbStorage dump -> load preserves BOTH LruLfu score words
    (timestamp and frequency), not just word 0. The checkpoint stores the two
    columns in the user's configured (logical) order, and load reorders them back
    into the physical device layout, so the round-trip is exact for either order."""
    device = torch.cuda.current_device()
    dim = 8
    n = 60
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    values = torch.randn(n, dim, dtype=torch.float32, device=device)
    freq_ref = torch.arange(1, 1 + n, dtype=torch.int64, device=device)  # varied

    src = _lru_lfu_storage(dim=dim, score_strategy=score_strategy)
    src.insert(keys, tids, values, scores=freq_ref.to(torch.uint64))

    # Device layout is always physical [timestamp, frequency] regardless of order.
    blocks_src = _read_score_blocks(src, keys, tids)
    assert torch.all(
        blocks_src[:, 1] == freq_ref
    ), "insert should set physical word 1 (frequency) to the provided value"

    d = str(tmp_path)
    paths = [f"{d}/meta.json", f"{d}/keys", f"{d}/emb", f"{d}/score", f"{d}/opt"]
    src.dump(0, *paths, include_optim=False)

    # The score file must be written in the user's logical column order. Read the
    # keys and scores from the SAME dump so rows stay aligned (a fresh
    # export_keys_values scan is a separate iteration and need not reproduce the
    # dump's row order). The frequency column must land at the position where the
    # user placed LFU in score_strategy.
    file_keys = torch.tensor(np.fromfile(f"{d}/keys", dtype=np.int64), dtype=torch.int64)
    file_scores = torch.tensor(
        np.fromfile(f"{d}/score", dtype=np.uint64), dtype=torch.uint64
    ).view(-1, 2)
    freq_col = list(score_strategy).index(DynamicEmbScoreStrategy.LFU)
    # Map dumped key -> its checkpoint frequency column value, compare to freq_ref.
    key_to_freq = {
        int(k): int(file_scores[i, freq_col]) for i, k in enumerate(file_keys.tolist())
    }
    for k, f in zip(keys.tolist(), freq_ref.tolist()):
        assert key_to_freq[k] == f, (
            f"checkpoint frequency column ({freq_col}) must match user's logical order"
        )

    dst = _lru_lfu_storage(dim=dim, score_strategy=score_strategy)
    dst.load(0, *paths, include_optim=False)

    blocks_dst = _read_score_blocks(dst, keys, tids)
    assert torch.equal(
        blocks_dst, blocks_src
    ), "dump/load must preserve both score words (timestamp + frequency)"


def test_lru_lfu_storage_rehash_preserves_frequency(current_device):
    """Growing the table (_expand_tables_impl re-hashes keys into more buckets)
    must preserve both LruLfu score words -- a single-value re-insert would reset
    the frequency to 0."""
    device = torch.cuda.current_device()
    dim = 8
    bc = 128
    # Start with a single bucket, leave room to grow.
    storage = _lru_lfu_storage(dim=dim, max_capacity=8 * bc, init_capacity=bc)
    n = 80
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    values = torch.randn(n, dim, dtype=torch.float32, device=device)
    freq_ref = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    storage.insert(keys, tids, values, scores=freq_ref.to(torch.uint64))

    before = _read_score_blocks(storage, keys, tids)
    assert torch.all(before[:, 1] == freq_ref)

    # Rehash: grow the single logical table to 8 buckets; keys re-hash and are
    # re-inserted, and copy_score_blocks must carry both words across.
    _expand_tables_impl(storage._state, [True], [8 * bc])

    after = _read_score_blocks(storage, keys, tids)
    assert torch.equal(
        after, before
    ), "rehash must preserve both score words (timestamp + frequency)"


# ---------------------------------------------------------------------------
# Custom score_function eviction (numba JIT + nvJitLink). Decay functions are
# written in the *logical* (tuple) order and remapped to physical device order by
# score_jit; the functions below use the (TIMESTAMP, LFU) order (scores[0]=ts,
# scores[1]=freq), which matches the physical layout, plus one written for the
# (LFU, TIMESTAMP) order to exercise the remap. float(...) keeps the module
# import numba-free (only register_score_function pulls numba in).
# ---------------------------------------------------------------------------


def _evict_high_freq_first(scores, cur_timestamp):
    # (TIMESTAMP, LFU): scores[1] = frequency. Eviction removes the LOWEST score,
    # so -frequency evicts the HIGHEST-frequency key -- opposite of default LFU.
    return -float(scores[1])


def _evict_high_freq_first_lfu_first(scores, cur_timestamp):
    # (LFU, TIMESTAMP): scores[0] = frequency. After the logical->physical remap
    # this reads the SAME physical frequency word as _evict_high_freq_first, so
    # the two evict identically despite the opposite tuple order.
    return -float(scores[0])


def _evict_oldest(scores, cur_timestamp):
    # Pure LRU by recency. age = cur_timestamp - scores[0] (non-negative uint64;
    # scores[0] - cur_timestamp would underflow). -age evicts the OLDEST key.
    return -float(cur_timestamp - scores[0])


def _lfu_decay(scores, cur_timestamp):
    # Reference decayed-LFU: log(freq) decayed by elapsed age. Exercises the whole
    # formula through numba+nvJitLink (both words, gamma, math.log/max, the safe
    # cur_timestamp - scores[0] subtraction).
    return math.log(max(scores[1], 1)) + (cur_timestamp - scores[0]) * math.log(0.9)


def _custom_table(score_fn, score_strategy, bc=128):
    from dynamicemb.jit import register_score_function

    cc = torch.cuda.get_device_capability(torch.cuda.current_device())
    key = register_score_function(score_fn, score_strategy, cc[0], cc[1])
    return get_scored_table(
        capacity=[bc],
        bucket_capacity=bc,
        key_type=torch.int64,
        score_specs=[
            ScoreSpec(name="frequency", policy=ScorePolicy.LRU_LFU, is_reduction=True)
        ],
        score_fn_key=key,
    )


@pytest.mark.parametrize(
    "score_fn, evicted, survivor",
    [
        (_evict_high_freq_first, "HF", "OLD"),
        (_evict_oldest, "OLD", "HF"),
    ],
    ids=["evict_high_freq", "evict_oldest"],
)
def test_lru_lfu_custom_score_function_ranks_by_its_dimension(
    current_device, score_fn, evicted, survivor
):
    """Each custom score_function evicts by ITS OWN score dimension: -frequency
    evicts the highest-frequency probe, -age evicts the oldest probe -- different
    keys from the same table, proving the JIT'd function actually drives ranking."""
    device = torch.cuda.current_device()
    table = _custom_table(score_fn, LRU_LFU_TS_FIRST)

    n = 120
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    ones = torch.ones(n, dtype=torch.uint64, device=device)
    _insert(table, keys, tids, ones)
    probes = {"OLD": (keys[:1], tids[:1]), "HF": (keys[1:2], tids[1:2])}

    table.lookup(
        keys[1:], tids[1:], ScoreArg(name="frequency", value=ones[1:], policy=ScorePolicy.LRU_LFU)
    )
    torch.cuda.synchronize()
    for _ in range(28):
        table.lookup(
            keys[1:2], tids[1:2], ScoreArg(name="frequency", value=ones[1:2], policy=ScorePolicy.LRU_LFU)
        )
    torch.cuda.synchronize()

    n_new = 40
    new_keys = torch.arange(100000, 100000 + n_new, dtype=torch.int64, device=device)
    new_tids = torch.zeros(n_new, dtype=torch.int64, device=device)
    new_freq = torch.ones(n_new, dtype=torch.uint64, device=device)
    ir = torch.empty(n_new, dtype=table.result_type, device=device).fill_(
        InsertResult.INIT.value
    )
    _, num_evicted, _, _, _, _ = table.insert_and_evict(
        new_keys,
        new_tids,
        ScoreArg(name="frequency", value=new_freq, policy=ScorePolicy.LRU_LFU),
        ir,
    )
    assert num_evicted > 0, "eviction should have occurred"

    ev_keys, ev_tids = probes[evicted]
    sv_keys, sv_tids = probes[survivor]
    _, ev_found, _ = table.lookup(
        ev_keys, ev_tids, ScoreArg(name="frequency", value=None, policy=ScorePolicy.CONST)
    )
    _, sv_found, _ = table.lookup(
        sv_keys, sv_tids, ScoreArg(name="frequency", value=None, policy=ScorePolicy.CONST)
    )
    assert not torch.any(ev_found), f"{score_fn.__name__} should evict the {evicted} probe"
    assert torch.all(sv_found), f"{score_fn.__name__} should keep the {survivor} probe"


def test_lru_lfu_decay_matches_python_oracle(current_device):
    """Whole-set oracle for the combined LFU+recency decay (_lfu_decay): snapshot
    every key's stored (ts, freq), recompute the exact score in NumPy, predict the
    victim set, and assert the kernel evicts exactly that set."""
    device = torch.cuda.current_device()
    gamma = 0.9
    bc = 128
    table = _custom_table(_lfu_decay, LRU_LFU_TS_FIRST, bc=bc)

    n = 90
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    ones = torch.ones(n, dtype=torch.uint64, device=device)
    idx, _ = _insert(table, keys, tids, ones)

    for i in range(n):
        reps = 1 + (i * 7) % 5
        one, one_tid, one_val = keys[i : i + 1], tids[i : i + 1], ones[i : i + 1]
        for _ in range(reps):
            table.lookup(
                one, one_tid, ScoreArg(name="frequency", value=one_val, policy=ScorePolicy.LRU_LFU)
            )
        torch.cuda.synchronize()

    blk = table.gather_score_blocks(0, idx).cpu().numpy()
    ts = blk[:, 0].astype(np.uint64)
    freq = blk[:, 1].astype(np.float64)

    n_new = 60
    new_keys = torch.arange(100000, 100000 + n_new, dtype=torch.int64, device=device)
    new_tids = torch.zeros(n_new, dtype=torch.int64, device=device)
    new_freq = torch.ones(n_new, dtype=torch.uint64, device=device)
    ir = torch.empty(n_new, dtype=table.result_type, device=device).fill_(
        InsertResult.INIT.value
    )
    _, num_evicted, evicted_keys, _, _, _ = table.insert_and_evict(
        new_keys,
        new_tids,
        ScoreArg(name="frequency", value=new_freq, policy=ScorePolicy.LRU_LFU),
        ir,
    )
    m = int(num_evicted)
    assert m == n + n_new - bc, "eviction count must be occupied + new - capacity"

    # _lfu_decay bakes gamma=0.9 as a float64 literal (math.log(0.9)), so the
    # oracle uses the same float64 constant (no float32 rounding).
    recency = (ts - ts.min()).astype(np.float64)
    rank = np.log(np.maximum(freq, 1.0)) + recency * (-math.log(gamma))
    order = np.argsort(rank, kind="stable")
    assert rank[order[m]] - rank[order[m - 1]] > 1e-6, "eviction boundary must not tie"

    keys_cpu = keys.cpu().numpy()
    predicted = set(int(keys_cpu[i]) for i in order[:m])
    actual = set(int(k) for k in evicted_keys.tolist())
    assert actual == predicted, "kernel must evict exactly the oracle-predicted set"


def test_lru_lfu_score_function_logical_order_remap(current_device):
    """A score_function is written in the table's LOGICAL (tuple) order; the
    logical->physical remap makes eviction identical regardless of order.
    _evict_high_freq_first (freq=scores[1], for TIMESTAMP-first) and
    _evict_high_freq_first_lfu_first (freq=scores[0], for LFU-first) both rank by
    physical frequency, so on identical tables they evict the SAME keys."""
    device = torch.cuda.current_device()

    def _run(fn, strat):
        table = _custom_table(fn, strat)
        n = 100
        keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
        tids = torch.zeros(n, dtype=torch.int64, device=device)
        ones = torch.ones(n, dtype=torch.uint64, device=device)
        _insert(table, keys, tids, ones)
        # Give each key a distinct frequency (key i looked up n-1-i extra times).
        for r in range(1, n):
            table.lookup(
                keys[:r], tids[:r], ScoreArg(name="frequency", value=ones[:r], policy=ScorePolicy.LRU_LFU)
            )
        torch.cuda.synchronize()
        n_new = 40
        nk = torch.arange(100000, 100000 + n_new, dtype=torch.int64, device=device)
        nt = torch.zeros(n_new, dtype=torch.int64, device=device)
        nf = torch.ones(n_new, dtype=torch.uint64, device=device)
        ir = torch.empty(n_new, dtype=table.result_type, device=device).fill_(
            InsertResult.INIT.value
        )
        _, _, evk, _, _, _ = table.insert_and_evict(
            nk, nt, ScoreArg(name="frequency", value=nf, policy=ScorePolicy.LRU_LFU), ir
        )
        return set(int(k) for k in evk.tolist())

    evicted_ts = _run(_evict_high_freq_first, LRU_LFU_TS_FIRST)
    evicted_lfu = _run(_evict_high_freq_first_lfu_first, LRU_LFU_LFU_FIRST)
    assert len(evicted_ts) > 0 and evicted_ts == evicted_lfu, (
        "logical->physical remap must make (TIMESTAMP,LFU) and (LFU,TIMESTAMP) "
        "score_functions of the same physical meaning evict identically"
    )


def test_lru_lfu_custom_score_function_via_create_table_state(current_device):
    """Production path: score_strategy=(TIMESTAMP, LFU) + score_function through
    create_table_state auto-registers the function (numba+nvJitLink) and threads a
    nonzero score_fn_key to the table; without it the table uses the default
    evictor (key 0)."""
    default = _lru_lfu_storage(score_strategy=LRU_LFU_TS_FIRST)
    assert default.key_index_map.num_scores_ == 2
    assert default.key_index_map.score_fn_key_ == 0, "no score_function => key 0"

    custom = _lru_lfu_storage(
        score_strategy=LRU_LFU_TS_FIRST, score_function=_evict_high_freq_first
    )
    assert custom.key_index_map.num_scores_ == 2
    assert custom.key_index_map.score_fn_key_ != 0, "score_function must register a key"


def test_score_function_requires_compound(current_device):
    """score_function is only valid for the compound {TIMESTAMP, LFU} strategy."""
    with pytest.raises(ValueError):
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=torch.cuda.current_device(),
            dim=8,
            max_capacity=128,
            score_strategy=DynamicEmbScoreStrategy.LFU,
            score_function=_evict_high_freq_first,
        )


def _bad_nonconst(scores, cur_timestamp):
    j = int(cur_timestamp) % 2
    return -float(scores[j])  # non-constant subscript -> must be rejected


def _bad_oob(scores, cur_timestamp):
    return -float(scores[2])  # out-of-bounds (only 0,1 exist) -> must be rejected


def test_score_function_nonconstant_subscript_rejected(current_device):
    from dynamicemb.jit import register_score_function

    cc = torch.cuda.get_device_capability(torch.cuda.current_device())
    with pytest.raises(ValueError):
        register_score_function(_bad_nonconst, LRU_LFU_TS_FIRST, cc[0], cc[1])


def test_score_function_out_of_bounds_subscript_rejected(current_device):
    from dynamicemb.jit import register_score_function

    cc = torch.cuda.get_device_capability(torch.cuda.current_device())
    with pytest.raises(IndexError):
        register_score_function(_bad_oob, LRU_LFU_LFU_FIRST, cc[0], cc[1])


def _lfu_decay_lfu_first(scores, cur_timestamp):
    # (LFU, TIMESTAMP) logical order: freq = scores[0], ts = scores[1]. After the
    # logical->physical remap this reads physical freq (word 1) and ts (word 0),
    # computing the SAME physical decayed-LFU as _lfu_decay does under
    # (TIMESTAMP, LFU). Indexes BOTH logical words, so it exercises both index
    # remaps (0->physical 1 AND 1->physical 0).
    return math.log(max(scores[0], 1)) + (cur_timestamp - scores[1]) * math.log(0.9)


def test_lru_lfu_custom_score_function_batched_v2(current_device):
    """End-to-end through BatchedDynamicEmbeddingTablesV2 with the compound
    (LFU, TIMESTAMP) strategy (logical order != physical) + a custom score_function
    that indexes BOTH logical words. Covers: the batched _create_score ->
    create_table_state -> register_score_function wiring, and the logical->physical
    remap of BOTH score indices under LFU-first. A whole-set oracle then verifies
    the kernel evicts exactly the decayed-LFU-predicted set."""
    from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2

    device = torch.cuda.current_device()
    bc = 128
    opts = [
        DynamicEmbTableOptions(
            dim=8,
            max_capacity=bc,
            init_capacity=bc,
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=device,
            bucket_capacity=bc,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=LRU_LFU_LFU_FIRST,  # logical != physical
            score_function=_lfu_decay_lfu_first,
        )
    ]
    bdebt = BatchedDynamicEmbeddingTablesV2(
        table_names=["t0"], table_options=opts, feature_table_map=[0]
    )
    table = bdebt._storage.key_index_map
    assert table.num_scores_ == 2
    assert (
        table.score_fn_key_ != 0
    ), "batched (LFU,TIMESTAMP)+score_function must auto-register a nonzero key"
    sname = table.score_names_[0]

    def _sa(value, policy=ScorePolicy.LRU_LFU):
        return ScoreArg(name=sname, value=value, policy=policy)

    n = 90
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    ones = torch.ones(n, dtype=torch.uint64, device=device)
    ir = torch.empty(n, dtype=table.result_type, device=device).fill_(
        InsertResult.INIT.value
    )
    idx = table.insert(keys, tids, _sa(ones), ir)

    for i in range(n):
        reps = 1 + (i * 7) % 5
        one, one_tid, one_val = keys[i : i + 1], tids[i : i + 1], ones[i : i + 1]
        for _ in range(reps):
            table.lookup(one, one_tid, _sa(one_val))
        torch.cuda.synchronize()

    blk = table.gather_score_blocks(0, idx).cpu().numpy()
    ts = blk[:, 0].astype(np.uint64)
    freq = blk[:, 1].astype(np.float64)

    n_new = 60
    new_keys = torch.arange(100000, 100000 + n_new, dtype=torch.int64, device=device)
    new_tids = torch.zeros(n_new, dtype=torch.int64, device=device)
    new_freq = torch.ones(n_new, dtype=torch.uint64, device=device)
    ir2 = torch.empty(n_new, dtype=table.result_type, device=device).fill_(
        InsertResult.INIT.value
    )
    _, num_evicted, evicted_keys, _, _, _ = table.insert_and_evict(
        new_keys, new_tids, _sa(new_freq), ir2
    )
    m = int(num_evicted)
    assert m == n + n_new - bc, "eviction count must be occupied + new - capacity"

    # Same physical oracle as the (TIMESTAMP, LFU) case: a correct remap makes the
    # LFU-first function compute the identical physical decayed-LFU score.
    gamma = 0.9
    recency = (ts - ts.min()).astype(np.float64)
    rank = np.log(np.maximum(freq, 1.0)) + recency * (-math.log(gamma))
    order = np.argsort(rank, kind="stable")
    assert rank[order[m]] - rank[order[m - 1]] > 1e-6, "eviction boundary must not tie"
    keys_cpu = keys.cpu().numpy()
    predicted = set(int(keys_cpu[i]) for i in order[:m])
    actual = set(int(k) for k in evicted_keys.tolist())
    assert actual == predicted, (
        "batched (LFU,TIMESTAMP) custom eviction must match the physical decayed-LFU "
        "oracle (proves both logical->physical index remaps are correct)"
    )


def test_lru_lfu_default_evictor_timestamp_tiebreak(current_device):
    """Default (Lex) evictor ranks by frequency, then breaks ties by timestamp:
    among equal-frequency keys the OLDER timestamp is evicted first. No custom
    score_function (score_fn_key == 0), so the built-in LexFreqTsComparator cubin
    runs. All keys share frequency 1, so only the timestamp tiebreak decides;
    inserting in sync-separated groups gives strictly increasing timestamp tiers,
    and a whole-set oracle asserts the exact evicted set = the oldest keys."""
    device = torch.cuda.current_device()
    bc = 128
    table = _lru_lfu_table(bc, bucket_capacity=bc)  # single bucket: all compete
    assert table.score_fn_key_ == 0, "default path must use the built-in Lex evictor"

    G, per = 10, 10
    n = G * per
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    ones = torch.ones(n, dtype=torch.uint64, device=device)
    # Insert group-by-group with a sync barrier so each group lands in a strictly
    # later timestamp tier than the previous (frequency stays 1 for every key).
    idx_parts = []
    for g in range(G):
        s = g * per
        idx_g, _ = _insert(table, keys[s : s + per], tids[s : s + per], ones[s : s + per])
        idx_parts.append(idx_g)
        torch.cuda.synchronize()
    idx = torch.cat(idx_parts)

    blk = table.gather_score_blocks(0, idx).cpu().numpy()
    ts = blk[:, 0].astype(np.uint64)
    freq = blk[:, 1].astype(np.uint64)
    assert np.all(freq == 1), "tiebreak test requires every key at frequency 1"

    # Overflow with new frequency-1 keys carrying the NEWEST timestamps. Being the
    # newest, they rank highest among the (all equal) frequencies and are never
    # evicted, so every eviction falls on the oldest pre-existing keys.
    n_new = 58
    new_keys = torch.arange(100000, 100000 + n_new, dtype=torch.int64, device=device)
    new_tids = torch.zeros(n_new, dtype=torch.int64, device=device)
    new_freq = torch.ones(n_new, dtype=torch.uint64, device=device)
    ir = torch.empty(n_new, dtype=table.result_type, device=device).fill_(
        InsertResult.INIT.value
    )
    _, num_evicted, evicted_keys, _, _, _ = table.insert_and_evict(
        new_keys,
        new_tids,
        ScoreArg(name="frequency", value=new_freq, policy=ScorePolicy.LRU_LFU),
        ir,
    )
    m = int(num_evicted)
    assert m == n + n_new - bc, "eviction count must be occupied + new - capacity"

    # Equal frequency -> pure timestamp ordering; the m oldest must be evicted.
    order = np.argsort(ts, kind="stable")
    assert ts[order[m]] > ts[order[m - 1]], "eviction boundary must fall on a ts gap"
    keys_cpu = keys.cpu().numpy()
    predicted = set(int(keys_cpu[i]) for i in order[:m])
    actual = set(int(k) for k in evicted_keys.tolist())
    assert (
        actual == predicted
    ), "default Lex evictor must break equal-frequency ties by evicting oldest timestamps"


def test_lru_lfu_plain_insert_evicts_via_cubin(current_device):
    """Plain table.insert() (NOT insert_and_evict) on a full LruLfu bucket must
    evict via the ranked comparator / custom score_function through the cubin, not
    the single-score DefaultEvictor reduce() (which cannot even read the 2-word
    layout). Uses _evict_high_freq_first (evict HIGHEST frequency): the default
    min-frequency reduce() would KEEP the hot keys, so the hot keys being gone
    proves the plain-insert path drove eviction through the custom cubin."""
    device = torch.cuda.current_device()
    bc = 128
    table = _custom_table(_evict_high_freq_first, LRU_LFU_TS_FIRST, bc=bc)
    assert table.score_fn_key_ != 0

    # Fill the single bucket exactly with frequency-1 keys (no eviction yet).
    n = bc
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    ones = torch.ones(n, dtype=torch.uint64, device=device)
    idx, _ = _insert(table, keys, tids, ones)  # _insert uses plain table.insert()

    # Boost a small hot subset to high frequency.
    n_hot = 10
    hot = keys[:n_hot]
    hot_tids = tids[:n_hot]
    hot_ones = ones[:n_hot]
    for _ in range(20):
        table.lookup(
            hot,
            hot_tids,
            ScoreArg(name="frequency", value=hot_ones, policy=ScorePolicy.LRU_LFU),
        )
    torch.cuda.synchronize()
    assert torch.all(
        table.gather_score_blocks(0, idx[:n_hot])[:, 1] == 21
    ), "hot keys should be at frequency 1 + 20"

    # Plain insert new keys: the bucket is full, so each insertion evicts a victim.
    # With _evict_high_freq_first the highest-frequency (hot) keys go first.
    n_new = 40
    new_keys = torch.arange(100000, 100000 + n_new, dtype=torch.int64, device=device)
    new_tids = torch.zeros(n_new, dtype=torch.int64, device=device)
    new_ones = torch.ones(n_new, dtype=torch.uint64, device=device)
    table.insert(
        new_keys,
        new_tids,
        ScoreArg(name="frequency", value=new_ones, policy=ScorePolicy.LRU_LFU),
    )
    torch.cuda.synchronize()

    _, founds, _ = table.lookup(
        hot, hot_tids, ScoreArg(name="frequency", value=None, policy=ScorePolicy.CONST)
    )
    assert not bool(
        founds.any()
    ), "plain insert must evict the high-freq keys via the custom cubin (DefaultEvictor would have kept them)"
