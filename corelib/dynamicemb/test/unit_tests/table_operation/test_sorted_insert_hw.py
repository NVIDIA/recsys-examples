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

"""Hardware checks of the sorted insert: the questions a CPU model cannot answer.

The sorted insert (src/table_operation/sorted_insert.cuh) lets ONE thread insert the keys of a
bucket one after another, so a key's plain stores (its digest, its score) are followed, in the
same thread, by the next key's probe (16-byte digest loads) and by the next eviction scan (an
asynchronous ``cp.async`` copy of the scores). Program order is what makes the stores visible; a
fence (``__threadfence_block``, on by default in the header) is there until these tests have
passed on every architecture the library is built for. The slots a run took stay locked until the
run is done, which is also what keeps a scan that misread one of them from choosing it. The file
also holds the check of the lock protocol with other work on the same table on a second stream, a
smoke test small enough for ``compute-sanitizer``, the bound on the serial overflow pass and the
cost of a batch of reserved keys, which is what every step of a cache hands the insert when all
its keys hit.

Each test is self-contained (no oracle needed for the sanitizer ones) and says PASS or FAIL.

    pytest -q -s test_sorted_insert_hw.py     # -s shows the times

    # the memory checkers see only what cudaMalloc made: switch the caching allocator off, and run the
    # smoke test (the oracle's blocking reads are not what is being checked)
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck  pytest -q -x -k sanitizer test_sorted_insert_hw.py
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool initcheck pytest -q -x -k sanitizer test_sorted_insert_hw.py
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool racecheck pytest -q -x -k sanitizer test_sorted_insert_hw.py
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool synccheck pytest -q -x -k sanitizer test_sorted_insert_hw.py

    # the out-of-bounds write of the compact bucketize that the segmented sort replaced for the keys of
    # a last table without buckets (a guard in BucketizeOutput): only a memory checker sees it
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck pytest -q -x -k last_table_without_buckets test_bucketize_segmented.py
"""

import os
import sys

import dynamicemb_extensions as ext
import pytest
import torch
from dynamicemb.scored_hashtable import ScoreArg
from dynamicemb_extensions import InsertResult, ScorePolicy, table_partition

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_sorted_insert import (  # noqa: E402
    GARBAGE,
    KEY_TYPES,
    POLICIES,
    TIMER,
    assert_tables_equal,
    i64,
    make_batch,
    make_pair,
    make_table,
    new_insert,
    new_insert_and_evict,
    records,
    run_new,
    run_old,
    u64,
)

pytestmark = pytest.mark.skipif(
    not getattr(ext, "sorted_insert", False),
    reason="this dynamicemb_extensions has no sorted insert: rebuild it "
    "(python setup.py build_ext --inplace)",
)

LOCKED = -3  # the LockedKey bit pattern, as an int64 key


def assert_no_locked_slot(table):
    """No slot holds the lock marker once a call is over (invariant I3 of sorted_insert.cuh)."""
    assert not bool((i64(table.keys_) == LOCKED).any()), "a slot of the table is locked"
    if table.enable_overflow_:
        ovf_keys, *_ = table_partition(
            table.overflow_table_storage_,
            table.fileds_type_,
            table.overflow_bucket_capacity_,
            table.overflow_num_buckets_,
        )
        assert not bool(
            (i64(ovf_keys) == LOCKED).any()
        ), "a slot of the overflow bucket is locked"


# ---------------------------------------------------------------------------
# E5: the same thread's store, then its own scan
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("seed", range(25))
def test_same_thread_store_then_scan_chain(key_type, seed):
    """One bucket of 128 slots, so that ONE thread inserts the whole batch: 300 new keys and 10
    residents of a bucket that holds 60 live keys, 30 reclaimed slots and 38 empty ones. The
    residents sort first and are assigned; then the new keys take the 38 empty slots, the 30
    reclaimed ones and 50 of the live ones (Insert, Reclaim and Evict one after another in that
    thread, each followed by the digest and score stores the next key's probe and eviction scan
    must see: random ASSIGN scores, so a victim chosen on a stale score is another key), and the
    rest of the batch is refused, every slot being locked by then. The result must be the one of
    the waves that keep what they took, which had a kernel boundary between two keys."""
    old, new = make_pair(key_type, ScorePolicy.ASSIGN, [128], 128)
    gen = torch.Generator().manual_seed(seed)
    resident = torch.randperm(10_000, generator=gen)[:90] + 1
    keys = resident.cuda().view(key_type)
    tids = torch.zeros(90, dtype=torch.int64, device="cuda")
    scores = u64(torch.randint(1, 1000, (90,), generator=gen))
    for table in (old, new):
        new_insert(table, keys, tids, scores, ScorePolicy.ASSIGN)
        table.erase(keys[:30], tids[:30])
    fresh = torch.randperm(10_000, generator=gen)[:300] + 20_001
    batch = torch.cat([fresh, resident[30:40]]).cuda()
    batch_keys = batch.view(key_type)
    batch_tids = torch.zeros(batch.numel(), dtype=torch.int64, device="cuda")
    batch_scores = u64(torch.randint(1, 100_000, (batch.numel(),), generator=gen))
    idx_old, rec_old = run_old(
        old,
        "insert_and_evict",
        batch_keys,
        batch_tids,
        batch_scores,
        ScorePolicy.ASSIGN,
    )
    results = torch.empty(batch.numel(), dtype=torch.uint8, device="cuda")
    idx_new, rec_new = run_new(
        new,
        "insert_and_evict",
        batch_keys,
        batch_tids,
        batch_scores,
        ScorePolicy.ASSIGN,
        results,
    )
    for what in ("INSERT", "RECLAIM", "EVICT", "ASSIGN"):
        assert bool((results == getattr(InsertResult, what).value).any()), (
            "the batch must run through " + what
        )
    assert_tables_equal(old, new, ScorePolicy.ASSIGN)
    assert torch.equal(idx_old, idx_new)
    assert records(rec_old, ScorePolicy.ASSIGN) == records(rec_new, ScorePolicy.ASSIGN)
    assert_no_locked_slot(new)


# ---------------------------------------------------------------------------
# the lock protocol with other work on the table
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_two_streams_the_lock_protocol(key_type):
    """The sorted insert on one stream, while a second stream stamps the scores of the
    residents (an ASSIGN lookup locks each slot it updates) and pins and unpins them (reference
    counters, read by the eviction scan). The interleaving decides which victim a locked slot
    leaves to its neighbour, so no exact state is asserted: no call hangs, no slot is left
    locked, no key is in the table twice or in two slots' worth of ``bucket_sizes``, and the
    table holds nothing but keys somebody inserted."""
    table = make_table(key_type, ScorePolicy.ASSIGN, [16 * 16], 16)
    resident = torch.arange(1, 161, dtype=torch.int64, device="cuda")
    res_keys = resident.view(key_type)
    res_tids = torch.zeros(160, dtype=torch.int64, device="cuda")
    res_scores = u64(torch.arange(1, 161, device="cuda"))
    new_insert(table, res_keys, res_tids, res_scores, ScorePolicy.ASSIGN)
    torch.cuda.synchronize()

    side = torch.cuda.Stream()
    stamp = ScoreArg(name="score", value=res_scores, policy=ScorePolicy.ASSIGN)
    with torch.cuda.stream(side):
        for _ in range(150):
            table.lookup(res_keys, res_tids, stamp)
            slots = table.slots_of(res_keys, res_tids)
            table.increment_counter(slots, res_tids)
            table.decrement_counter(slots, res_tids)

    inserted = [resident]
    for call in range(4):
        fresh = torch.arange(
            1000 + 1024 * call,
            1000 + 1024 * (call + 1),
            dtype=torch.int64,
            device="cuda",
        )
        inserted.append(fresh)
        run_new(
            table,
            "insert_and_evict",
            fresh.view(key_type),
            torch.zeros(1024, dtype=torch.int64, device="cuda"),
            u64(torch.arange(500, 500 + 1024, device="cuda")),
            ScorePolicy.ASSIGN,
        )
    torch.cuda.synchronize()

    assert_no_locked_slot(table)
    keys = i64(table.keys_).flatten()
    live = keys[keys >= 0]
    assert live.numel() == torch.unique(live).numel(), "a key is in the table twice"
    assert (
        int(table.bucket_sizes.sum()) == live.numel()
    ), "bucket_sizes counts the live keys"
    assert bool(torch.isin(live, torch.cat(inserted)).all()), "a key nobody inserted"
    assert bool((table._ref_counter == 0).all()), "every pin was released"


# ---------------------------------------------------------------------------
# a smoke test for compute-sanitizer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("op", ["insert", "insert_and_evict", "cache"])
def test_sanitizer_smoke(key_type, policy, op):
    """Small batches into a table of one bucket, a zero-capacity table and 2 more buckets, with
    padding, ties, a full pinned bucket that spills into the overflow bucket (cache), and nothing
    else: no oracle, so a memory checker reports only what the new kernels and their compaction
    do."""
    table = make_table(key_type, policy, [16, 0, 32], 16, overflow=op == "cache")
    gen = torch.Generator().manual_seed(2)
    for step in range(3):
        keys, tids, scores = make_batch(gen, 90, 3, key_type, policy, pad=0.2)
        if op == "cache" and step == 1:
            # pin every slot of table 0 (one bucket, full by now): its newcomers spill into the
            # overflow bucket from now on
            slots = torch.arange(16, dtype=torch.int64, device="cuda")
            table.increment_counter(slots, torch.zeros_like(slots))
        insert_results = torch.empty(keys.numel(), dtype=torch.uint8, device="cuda")
        score_out = torch.empty(keys.numel(), dtype=torch.int64, device="cuda")
        _, rec = run_new(
            table, op, keys, tids, scores, policy, insert_results, score_out
        )
        padded = i64(keys) == -1
        assert bool((insert_results[padded] == InsertResult.ILLEGAL.value).all())
        if scores is not None:
            assert bool((score_out[padded] == GARBAGE).all())
        assert_no_locked_slot(table)
    if op == "cache":
        assert (
            int(table.overflow_bucket_sizes.sum()) > 0
        ), "nothing spilled: the test is idle"


# ---------------------------------------------------------------------------
# the serial overflow pass
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_ten_thousand_spills_in_one_call(key_type):
    """One bucket of 16 slots that is full and pinned, and 10,000 new keys in one call: all of
    them spill into the overflow bucket of 48 slots, the first 48 are placed and the rest
    refused (every slot is locked by then). The run of the bucket and the overflow pass are one
    thread each, so the time of the call is the bound of the design: it has to stay far below a
    training step (tens of milliseconds are expected, 500 ms is what this test allows), and the
    outcome is the same on every run."""
    reference = None
    times = []
    for run in range(20):
        table = make_table(key_type, ScorePolicy.ASSIGN, [16], 16, overflow=True)
        keys = torch.arange(1, 17, dtype=torch.int64, device="cuda")
        tids = torch.zeros(16, dtype=torch.int64, device="cuda")
        new_insert(
            table,
            keys.view(key_type),
            tids,
            u64(torch.arange(1, 17, device="cuda")),
            ScorePolicy.ASSIGN,
        )
        table.increment_counter(table.slots_of(keys.view(key_type), tids), tids)
        spill = torch.arange(1000, 11_000, dtype=torch.int64, device="cuda")
        spill_tids = torch.zeros(10_000, dtype=torch.int64, device="cuda")
        spill_scores = u64(torch.arange(1, 10_001, device="cuda"))
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        ext.table_insert_and_evict_sorted(
            table.table_storage_,
            table.table_bucket_offsets_,
            table.bucket_capacity_,
            table.bucket_sizes,
            spill.view(key_type),
            spill_tids,
            spill_scores,
            ScorePolicy.ASSIGN,
            table._ref_counter,
            None,
            None,
            ovf_storage=table.overflow_table_storage_,
            ovf_bucket_capacity=table.overflow_bucket_capacity_,
            ovf_bucket_sizes=table.overflow_bucket_sizes,
            ovf_counter=table._ovf_counter,
            ovf_output_offsets=table.overflow_output_offsets_,
            num_scores=1,
        )
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        assert_no_locked_slot(table)
        assert table.overflow_bucket_sizes.tolist() == [48]
        state = table.overflow_table_storage_.clone()
        if reference is None:
            reference = state
        else:
            assert torch.equal(
                state, reference
            ), "the overflow bucket differs between runs"
    print(
        "overflow pass of 10,000 spills: %.2f ms (best %.2f ms)"
        % (max(times), min(times))
    )
    assert max(times) < 500.0, "the serial overflow pass takes %.1f ms" % max(times)


# ---------------------------------------------------------------------------
# the batch that takes no part
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("op", ["insert", "insert_and_evict", "cache"])
def test_a_batch_of_reserved_keys_costs_little(key_type, op):
    """With the default miss gate a cache launches its insert every step, and a step whose keys
    all hit hands it a batch of reserved keys: nothing takes part, but the sort, the kernels and
    the compaction run over every lane. The design expects a few tenths of a millisecond of GPU
    time for 65,536 lanes (a radix sort of 65,536 128-bit keys and about fifty small launches);
    10 ms is the bound, which a pass that visited the inert lanes one after another would exceed
    by far. The time spans the enqueue of the call, so a slow host counts too."""
    n = 65_536
    table = make_table(
        key_type, ScorePolicy.ASSIGN, [16 * 64, 16 * 64], 16, overflow=op == "cache"
    )
    keys = torch.full((n,), -1, dtype=torch.int64, device="cuda").view(key_type)
    tids = torch.full((n,), -1, dtype=torch.int64, device="cuda")
    scores = u64(torch.full((n,), GARBAGE, dtype=torch.int64, device="cuda"))
    call = new_insert if op == "insert" else new_insert_and_evict
    times = []
    for rep in range(14):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        call(table, keys, tids, scores, ScorePolicy.ASSIGN)
        end.record()
        end.synchronize()
        if rep >= 2:  # the first calls fill the allocator's pool
            times.append(start.elapsed_time(end))
    assert bool((table.bucket_sizes == 0).all()), "a lane that takes no part took one"
    print(
        "65,536 reserved keys, %s, %s: median %.3f ms, max %.3f ms"
        % (op, key_type, sorted(times)[len(times) // 2], max(times))
    )
    assert max(times) < 10.0, "a batch of reserved keys takes %.1f ms" % max(times)


# ---------------------------------------------------------------------------
# the floor of the timer in a long run
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_timer_floor_in_a_long_run(key_type):
    """4,096 keys into one bucket of 128 slots in one call (the skewed batch of the design: one
    thread works for a few milliseconds): the 128 smallest keys take the slots (the others are
    refused, a key of a call is never evicted by the call) and their timestamps strictly increase
    with the key, whatever the timer does in between."""
    table = make_table(key_type, TIMER, [128], 128)
    gen = torch.Generator().manual_seed(1)
    keys_cpu = torch.randperm(1_000_000, generator=gen)[:4096] + 1
    new_insert(
        table,
        keys_cpu.cuda().view(key_type),
        torch.zeros(4096, dtype=torch.int64, device="cuda"),
        None,
        TIMER,
    )
    torch.cuda.synchronize()
    flat = i64(table.keys_).flatten()
    present = flat[flat > 0]
    assert sorted(present.tolist()) == sorted(keys_cpu.tolist())[:128]
    by_key = i64(table.scores_list[0]).flatten()[flat > 0][torch.argsort(present)]
    assert bool((by_key[1:] > by_key[:-1]).all())


def test_the_fence_switch_is_documented_in_the_kernel_header():
    """The compile-time switch named in the module docstring exists, and defaults to ON."""
    path = os.path.join(
        HERE, "..", "..", "..", "src", "table_operation", "sorted_insert.cuh"
    )
    with open(path) as f:
        source = f.read()
    assert "#ifndef DEMB_SORTED_INSERT_THREADFENCE" in source
    assert "#define DEMB_SORTED_INSERT_THREADFENCE 1" in source
