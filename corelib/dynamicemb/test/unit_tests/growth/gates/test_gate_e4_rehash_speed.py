# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GATE E4 -- the rehash of a table is not slower than the one it replaced.

``kvt._expand_tables_impl`` doubles one table of N slots that holds 0.4 N keys (fp32 rows of
64, SGD, a value row per key, 0% or 30% of the keys erased first): a new key map carrying the
keys, scores, reference counters and rows, and the value buffer extended. The file uses
nothing that the tree before the lane lacks, so it runs on that tree too, where the growth
reads counts back and waits for the device, and on this one, where it does neither. Each
size prints one ``E4 INFO`` line with the time until the call returned (``host``) and until
the device had finished (``total``); the comparison is of ``total``, the wall time of a
growth.

The two runs are two processes, one per tree, joined by a file::

    # on a checkout of the tree before the lane (355a1f61), built the same way
    DEMB_GROWTH_GATES=1 GATE_E4_OUT=/tmp/e4_base.json python -m pytest -s -q \\
        corelib/dynamicemb/test/unit_tests/growth/gates/test_gate_e4_rehash_speed.py
    # on this tree
    DEMB_GROWTH_GATES=1 GATE_E4_BASELINE=/tmp/e4_base.json python -m pytest -s -q \\
        corelib/dynamicemb/test/unit_tests/growth/gates/test_gate_e4_rehash_speed.py

With ``GATE_E4_BASELINE`` the verdict is one printed line, ``E4 PASS ...`` or ``E4 FAIL ...``:
every case must take at most 1.2 times the baseline up to 2**22 slots and at most 1.0 times it
above (faster, where the old rehash read a count per batch of the export). Without it the run
only prints and records. Every case also reads back a sample of the keys and their values
from the grown table. ``GATE_E4_LOG2_SLOTS`` (default ``16,18,20,22,24``; 2**26 needs about
35 GB) and ``GATE_E4_REPS`` (the best of this many, default 3) change the grid.
"""

import gc
import json
import os
import sys
import time

import pytest
import torch
from dynamicemb import DynamicEmbScoreStrategy, DynamicEmbTableOptions
from dynamicemb import key_value_table as kvt
from dynamicemb.optimizer import OptimizerArgs, SGDDynamicEmbeddingOptimizer
from dynamicemb.scored_hashtable import ScoreArg
from dynamicemb_extensions import ScorePolicy

pytestmark = pytest.mark.skipif(
    os.environ.get("DEMB_GROWTH_GATES") != "1" or not torch.cuda.is_available(),
    reason="a hardware gate: set DEMB_GROWTH_GATES=1 on a GPU",
)

LOG2_SLOTS = [
    int(x) for x in os.environ.get("GATE_E4_LOG2_SLOTS", "16,18,20,22,24").split(",")
]
REPS = int(os.environ.get("GATE_E4_REPS", "3"))
OUT = os.environ.get("GATE_E4_OUT")
BASELINE = os.environ.get("GATE_E4_BASELINE")
LOAD = 0.4
DIM = 64
CHUNK = 1 << 20
SAMPLE = 1 << 16
SMALL_LOG2 = 22
MAX_RATIO_SMALL = 1.2
MAX_RATIO_LARGE = 1.0


def make_state(slots):
    options = DynamicEmbTableOptions(
        dim=DIM,
        init_capacity=slots,
        max_capacity=4 * slots,
        bucket_capacity=128,
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=torch.cuda.current_device(),
        score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
        local_hbm_for_values=1 << 40,
    )
    return kvt.create_table_state(
        [options], SGDDynamicEmbeddingOptimizer(OptimizerArgs()), use_rows=True
    )


def fill(state, count, erased):
    """``count`` keys with a value of their own, of which the first ``erased`` fraction is
    erased; the keys that are left."""
    device = state.device
    km = state.key_index_map
    vdim = state.table_value_dims_cpu[0]
    for lo in range(0, count, CHUNK):
        hi = min(count, lo + CHUNK)
        keys = torch.arange(lo + 1000, hi + 1000, dtype=torch.int64, device=device)
        arg = ScoreArg(
            name=state.score_policy.name,
            value=torch.ones(hi - lo, dtype=torch.uint64, device=device),
            policy=ScorePolicy.ASSIGN,
        )
        rows, _ = km.insert(keys, torch.zeros_like(keys), arg, return_slots=True)
        values = keys.to(torch.float32).unsqueeze(1).expand(-1, vdim).contiguous()
        kvt.store_to_flat_single_table(state, rows, 0, values)
    gone = int(erased * count)
    if gone:
        keys = torch.arange(1000, 1000 + gone, dtype=torch.int64, device=device)
        km.erase(keys, torch.zeros_like(keys))
    torch.cuda.synchronize()
    return torch.arange(1000 + gone, 1000 + count, dtype=torch.int64, device=device)


def check_sample(state, live):
    """The grown table finds a sample of the keys that were live, with their values."""
    step = max(1, live.numel() // SAMPLE)
    keys = live[::step].contiguous()
    arg = ScoreArg(name=state.score_policy.name, value=None, policy=ScorePolicy.CONST)
    _, founds, rows = state.key_index_map.lookup(keys, torch.zeros_like(keys), arg)
    if not bool(founds.all()):
        return "a live key is missing from the grown table"
    got = kvt.load_from_flat_single_table(state, rows, 0)
    if not torch.equal(got[:, 0], keys.to(torch.float32)):
        return "a live key has the value of another key in the grown table"
    return None


def one_growth(slots, erased):
    """(host ms, total ms, problem) of one doubling of a fresh table of ``slots``."""
    state = make_state(slots)
    live = fill(state, int(LOAD * slots), erased)
    capacity = state.key_index_map.per_table_capacity_[0]
    torch.cuda.synchronize()
    start = time.perf_counter()
    kvt._expand_tables_impl(state, [True], [2 * capacity])
    host = time.perf_counter() - start
    torch.cuda.synchronize()
    total = time.perf_counter() - start
    problem = None
    if state.key_index_map.per_table_capacity_[0] != 2 * capacity:
        problem = "the table did not double"
    problem = problem or check_sample(state, live)
    del state, live
    gc.collect()
    torch.cuda.empty_cache()
    return host * 1e3, total * 1e3, problem


def measure(log2, erased):
    slots = 1 << log2
    one_growth(1 << 14, 0.0)  # kernels and allocator warm
    best_host, best_total, problem = float("inf"), float("inf"), None
    for _ in range(REPS):
        host, total, problem = one_growth(slots, erased)
        best_host, best_total = min(best_host, host), min(best_total, total)
        if problem:
            break
    return {
        "log2": log2,
        "slots": slots,
        "erased": erased,
        "host_ms": best_host,
        "total_ms": best_total,
        "problem": problem,
    }


def test_e4_rehash_speed():
    records = []
    for log2 in LOG2_SLOTS:
        for erased in (0.0, 0.3):
            record = measure(log2, erased)
            records.append(record)
            print(
                "E4 INFO slots=2^%d erased=%d%% host=%.2fms total=%.2fms (%.1f ns/slot)%s"
                % (
                    log2,
                    round(100 * erased),
                    record["host_ms"],
                    record["total_ms"],
                    1e6 * record["total_ms"] / record["slots"],
                    ": " + record["problem"] if record["problem"] else "",
                )
            )
    if OUT:
        with open(OUT, "w") as f:
            json.dump(records, f, indent=1)
    problems = ["2^%d: %s" % (r["log2"], r["problem"]) for r in records if r["problem"]]
    if BASELINE:
        with open(BASELINE) as f:
            base = {(r["log2"], r["erased"]): r for r in json.load(f)}
        for r in records:
            old = base.get((r["log2"], r["erased"]))
            if old is None or old["problem"]:
                continue
            limit = MAX_RATIO_SMALL if r["log2"] <= SMALL_LOG2 else MAX_RATIO_LARGE
            ratio = r["total_ms"] / old["total_ms"]
            print(
                "E4 INFO slots=2^%d erased=%d%% total %.2fms against %.2fms: ratio %.2f (limit %.1f)"
                % (
                    r["log2"],
                    round(100 * r["erased"]),
                    r["total_ms"],
                    old["total_ms"],
                    ratio,
                    limit,
                )
            )
            if ratio > limit:
                problems.append(
                    "2^%d slots, %d%% erased: %.2f times the baseline, at most %.1f"
                    % (r["log2"], round(100 * r["erased"]), ratio, limit)
                )
        print("E4 %s %s" % ("FAIL" if problems else "PASS", "; ".join(problems)))
    elif problems:
        print("E4 FAIL %s" % "; ".join(problems))
    assert not problems, "; ".join(problems)


if __name__ == "__main__":
    os.environ.setdefault("DEMB_GROWTH_GATES", "1")
    sys.exit(pytest.main(["-s", "-q", __file__]))
