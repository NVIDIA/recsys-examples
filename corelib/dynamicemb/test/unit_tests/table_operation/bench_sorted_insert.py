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

"""Time of the sorted insert against the waves it replaced. Not a test: it prints a table.

    python bench_sorted_insert.py            # from this directory, on a GPU

The waves (``wave_oracle.py``) are the code the library used to run: one launch per rank of a
key in its bucket, with the pins the oracle adds, and several blocking reads. Every cell inserts
a fresh batch per call into tables of 128-slot buckets:

* ``uniform``: 65,536 keys over 8,192 buckets (a few dozen per bucket), nothing is evicted;
* ``evicting``: 65,536 keys into 256 buckets (about 250 per bucket: evictions and refusals);
* ``skewed``: 4,096 keys, all of one of 64 buckets (one thread works through the run);
* ``inert``: 65,536 lanes of the reserved key, the batch a cache's insert gets when all its
  keys hit.

The two columns are the time between two events on the current stream, in milliseconds, medians
of 7 calls after 2 warm-up calls (the waves block the host, so their time includes the waits).
A cell where the sorted insert is slower than the waves is marked.
"""

import os
import statistics
import sys

import dynamicemb_extensions as ext
import torch
from dynamicemb_extensions import ScorePolicy

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import wave_oracle as oracle  # noqa: E402
from test_sorted_insert import (  # noqa: E402
    make_table,
    new_insert,
    new_insert_and_evict,
    u64,
)

BUCKET = 128
WARMUP, REPS = 2, 7
# (cell, lanes, buckets of the table)
CELLS = [
    ("uniform", 65_536, 8192),
    ("evicting", 65_536, 256),
    ("skewed", 4_096, 64),
    ("inert", 65_536, 8192),
]


def make_batches(table, cell, lanes, gen):
    """``WARMUP + REPS`` batches of ``lanes`` keys, each of keys no other batch has."""
    batches = []
    tids = torch.zeros(lanes, dtype=torch.int64, device="cuda")
    if cell == "inert":
        keys = torch.full((lanes,), -1, dtype=torch.int64, device="cuda")
        scores = u64(torch.zeros(lanes, dtype=torch.int64, device="cuda"))
        return [(keys, torch.full_like(tids, -1), scores)] * (WARMUP + REPS)
    # one universe of keys cut into batches; the skewed cell keeps only the keys whose home
    # bucket is 0 (a 64th of them), so its universe is 96 times larger (a margin)
    per_batch = lanes * (96 if cell == "skewed" else 1)
    universe = (torch.randperm(per_batch * (WARMUP + REPS), generator=gen) + 1).cuda()
    for rep in range(WARMUP + REPS):
        keys = universe[rep * per_batch : (rep + 1) * per_batch]
        if cell == "skewed":
            all_tids = torch.zeros(keys.numel(), dtype=torch.int64, device="cuda")
            _, inverse, segments = ext.bucketize_keys_segmented(
                keys,
                all_tids,
                table.table_bucket_offsets_,
                table.num_buckets_,
                table.bucket_capacity_,
            )
            home = torch.empty_like(segments)
            home[inverse] = segments
            keys = keys[home == 0][:lanes]
            assert keys.numel() == lanes, "too few keys in bucket 0: rerun"
        scores = u64(torch.randint(1, 1000, (lanes,), generator=gen).cuda())
        batches.append((keys, tids, scores))
    return batches


def median_ms(call):
    """The median time of ``call(rep)`` over ``REPS`` calls after ``WARMUP`` ones."""
    times = []
    for rep in range(WARMUP + REPS):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        call(rep)
        end.record()
        end.synchronize()
        if rep >= WARMUP:
            times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    policy = ScorePolicy.ASSIGN
    print("%-9s %-17s %10s %10s" % ("cell", "op", "waves ms", "sorted ms"))
    for cell, lanes, buckets in CELLS:
        for op in ("insert", "insert_and_evict"):
            gen = torch.Generator().manual_seed(1)
            # the waves pin with reference counters: both tables have them
            old = make_table(torch.int64, policy, [BUCKET * buckets], BUCKET)
            new = make_table(torch.int64, policy, [BUCKET * buckets], BUCKET)
            batches = make_batches(new, cell, lanes, gen)

            def run_old(rep):
                keys, tids, scores = batches[rep]
                if op == "insert":
                    oracle.insert(old, keys, tids, scores, policy)
                else:
                    oracle.insert_and_evict(old, keys, tids, scores, policy)

            def run_new(rep):
                keys, tids, scores = batches[rep]
                if op == "insert":
                    new_insert(new, keys, tids, scores, policy)
                else:
                    new_insert_and_evict(new, keys, tids, scores, policy)

            t_old, t_new = median_ms(run_old), median_ms(run_new)
            note = "  <- slower than the waves" if t_new > t_old else ""
            print("%-9s %-17s %10.3f %10.3f%s" % (cell, op, t_old, t_new, note))


if __name__ == "__main__":
    main()
