# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""A training loop with an ``erase_by_score_async`` every few steps, at a size that is
close to a real job: batches of ~64k keys, a table that grows to hold a million keys or
more and is scanned in many chunks, and an erase that is still running when the next
steps begin.

The module is trained in one thread; the erases run on its worker. Nothing waits for an
erase except the end. Every step checks that each key reads its own value (``train_step``),
and when the loop is over the table is compared with a model kept on the host: the score
of each key is the step that last trained it, so which keys the last erase had to remove,
and which it had to leave, is known exactly.

What the workload does, per step ``s`` (a score is the step):

* ``HOT`` keys, trained every step -- never below any threshold;
* ``FRESH`` keys nobody has trained before, which make the table grow;
* revisits of keys introduced within the last ``WINDOW`` steps -- the live set;
* a few ``STRAGGLERS`` from anywhere in the past, many of them below the threshold of an
  erase that is running or has run: an erase must either have removed them and the
  step put them back, or the step must have saved them, and either way they are there.

Every ``ERASE_EVERY`` steps an erase of what is older than the window is asked for, and
the loop goes on.

Size: ``STRESS_STEPS`` (default 120) steps; ``STRESS_BATCH_SCALE`` (default 1) multiplies
the batch.
"""

import os
import time

import numpy as np
import pytest
import torch
from dynamicemb import DynamicEmbScoreStrategy

from helpers import build_de_table, found, size, train_step

S = DynamicEmbScoreStrategy

SCALE = int(os.environ.get("STRESS_BATCH_SCALE", "1"))
STEPS = int(os.environ.get("STRESS_STEPS", "120"))
ERASE_EVERY = 10
WINDOW = 30  # steps whose fresh keys a revisit may pick from, and the erase keeps
HOT = 8192 * SCALE
FRESH = 16384 * SCALE
REVISITS = 36864 * SCALE
STRAGGLERS = 2048 * SCALE
# A step's batch is 8192 + 16384 + 36864 + 2048 = 63488 keys, at most, per table.
CHUNK_SLOTS = 1 << 18  # a table of a few million slots is scanned in a dozen chunks or more
MAX_CAPACITY = 1 << 23
INIT_CAPACITY = 1 << 20  # smaller than what the window holds: the table grows under the scans


def _workload(num_steps: int):
    """The keys of every step, and the first step each key was trained. Keys are 1..N, in
    the order they were first drawn, so that "introduced at step s" is a key range."""
    rng = np.random.default_rng(1234)
    hot = np.arange(1, HOT + 1, dtype=np.int64)
    next_key = HOT + 1
    first_of_step = {}  # step -> (first fresh key, one past the last)
    batches = []
    for s in range(1, num_steps + 1):
        fresh = np.arange(next_key, next_key + FRESH, dtype=np.int64)
        first_of_step[s] = (next_key, next_key + FRESH)
        next_key += FRESH
        lo = first_of_step[max(1, s - WINDOW + 1)][0]
        revisit = rng.integers(lo, next_key, REVISITS, dtype=np.int64)
        straggler = rng.integers(HOT + 1, next_key, STRAGGLERS, dtype=np.int64)
        batch = np.unique(np.concatenate([hot, fresh, revisit, straggler]))
        rng.shuffle(batch)
        batches.append(batch)
    return batches, next_key


@pytest.mark.parametrize("num_tables", [1, 2])
def test_training_with_an_erase_every_few_steps_leaves_the_table_the_host_model_says(
    device, num_tables
):
    batches, universe = _workload(STEPS)
    module = build_de_table(
        device.index,
        S.CUSTOMIZED,
        num_tables=num_tables,
        max_capacity=MAX_CAPACITY,
        init_capacity=INIT_CAPACITY,
        local_hbm_for_values=8 << 30,
    )
    state = module._storage._state
    generation_before = state.key_map_generation

    # The host model: the step that last trained a key, 0 for never. Tables are fed
    # the same keys, so one array serves them all.
    last_step = np.zeros(universe + 1, dtype=np.int32)
    handles, thresholds = [], []  # thresholds[i] is that of handles[i], asked for at step 10 * (i + 1)
    erased_total = 0
    started = time.monotonic()

    def reap(wait: bool) -> None:
        nonlocal erased_total
        while handles and (wait or handles[0].done()):
            erased, spared = handles.pop(0).result()
            assert spared == 0, "nothing in this loop has a prefetch in flight"
            erased_total += erased

    for s, batch in enumerate(batches, start=1):
        keys = batch.tolist()
        train_step(module, [keys] * num_tables if num_tables > 1 else keys, device, score=s)
        last_step[batch] = s
        if s % ERASE_EVERY == 0 and s > WINDOW:
            threshold = s - WINDOW + 1  # what was last trained before this window
            thresholds.append((s, threshold))
            handles.append(
                module.erase_by_score_async(
                    S.CUSTOMIZED, threshold, chunk_slots=CHUNK_SLOTS
                )
            )
        reap(wait=False)
    reap(wait=True)
    elapsed = time.monotonic() - started

    assert thresholds, "the loop was too short for an erase"
    asked_at, threshold = thresholds[-1]
    # Present iff trained at or after the last threshold. Anything trained before it was
    # trained before the step the erase was asked for, so that erase had to remove it
    # (unless a step saved it, which gave it a newer score); the erases before it asked for
    # less. Nothing was evicted: the table is far from full.
    expect = last_step >= threshold
    expect[0] = False
    all_keys = torch.arange(universe + 1, dtype=torch.int64)
    for table_id in range(num_tables):
        present = np.zeros(universe + 1, dtype=bool)
        for start in range(0, universe + 1, 1 << 20):
            part = all_keys[start : start + (1 << 20)].to(device)
            present[start : start + len(part)] = (
                found(module, part, device, table_id).cpu().numpy()
            )
        wrong_gone = np.flatnonzero(expect & ~present)
        wrong_kept = np.flatnonzero(~expect & present)
        assert len(wrong_gone) == 0, (
            f"table {table_id}: {len(wrong_gone)} keys at or above the threshold {threshold} "
            f"were erased, e.g. {wrong_gone[:5].tolist()} (last steps "
            f"{last_step[wrong_gone[:5]].tolist()})"
        )
        assert len(wrong_kept) == 0, (
            f"table {table_id}: {len(wrong_kept)} keys below the threshold {threshold} "
            f"survived the erase asked for at step {asked_at}, e.g. {wrong_kept[:5].tolist()} "
            f"(last steps {last_step[wrong_kept[:5]].tolist()})"
        )
        assert size(module, table_id) == int(expect.sum()), f"table {table_id}"

    # The erases did work, and not on a table that never changed shape.
    assert erased_total > 0
    print(
        f"stress: {STEPS} steps x {num_tables} tables, {len(thresholds)} erases, "
        f"{erased_total} keys erased, {state.key_map_generation - generation_before} "
        f"expansions seen, {size(module)} keys left, {elapsed:.1f}s"
    )

    # What survived still reads right.
    live = np.flatnonzero(expect)[:65536].tolist()
    train_step(module, [live] * num_tables if num_tables > 1 else live, device, score=STEPS + 1)
