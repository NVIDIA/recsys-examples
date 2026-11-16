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
close to a real job, on a model TorchRec has wrapped: a ``DistributedModelParallel`` around
an ``EmbeddingCollection`` of dynamicemb tables, on ONE rank.

Run with ``torchrun --nproc_per_node 1``. Batches are ~64k keys, the tables grow to hold a
million keys or more and are scanned in many chunks, and an erase is still running when
the next steps begin. The keys go through the collection's forward -- input dist, lookup,
output dist -- and the backward the wrapper runs, and the erase is the one a user makes,
``erase_by_score_async(model, ...)``, which has to find the fused tables inside the wrapper.

The model is trained in one thread; the erases run on each module's worker. Nothing waits
for an erase except the end. Every step checks the output, and when the loop is over the
tables are compared with a model kept on the host: the score of each key is the step that
last trained it, so which keys the last erase had to remove, and which it had to leave,
is known exactly.

The tables start from a constant (``TABLE_INITIALIZER_VALUE``) and the learning rate is 0,
so every key must read exactly that constant: a row a scan reclaimed and a step took before
it was reset, or one that is not initialised, reads something else.

What the workload does, per step ``s`` (a score is the step):

* ``HOT`` keys, trained every step -- never below any threshold;
* ``FRESH`` keys nobody has trained before, which make the tables grow;
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
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from dynamicemb import DynamicEmbScoreStrategy, EmbOptimType, erase_by_score_async
from dynamicemb.dump_load import get_dynamic_emb_module
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# test_embedding_dump_load is the fixture module the tests in unit_tests share, so it
# stays a directory up and has to be put on the path from here.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_embedding_dump_load import (  # noqa: E402
    TABLE_INITIALIZER_VALUE,
    create_model,
    idx_to_name,
)

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
CHUNK_SLOTS = (
    1 << 18
)  # a table of a few million slots is scanned in a dozen chunks or more


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


NUM_EMBEDDINGS = 1 << 23  # per table: more than the keys of the whole run
INIT_CAPACITY = (
    1 << 20
)  # less than what the window holds: the tables grow under the scans
# Without it a table is one bucket the size of the whole table (MAX_BUCKET_CAPACITY, what
# create_model uses): it could not grow, and a scan could not be cut into chunks, which
# never split a bucket.
BUCKET_CAPACITY = 128
DIM = 8


def _device() -> torch.device:
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def _model(num_tables: int):
    names = [idx_to_name(0, i) for i in range(num_tables)]
    model = create_model(
        num_embedding_collections=1,
        num_embeddings=[NUM_EMBEDDINGS] * num_tables,
        embedding_dim=DIM,
        # The fused parameter is called "learning_rate": under "lr" (which the other tests of a
        # wrapped model use) it is not read, and the default 0.01 trains every key it sees.
        optimizer_kwargs={"optimizer": EmbOptimType.SGD, "learning_rate": 0.0},
        score_strategy=S.CUSTOMIZED,
        table_option_overrides={
            table: {
                "init_capacity": INIT_CAPACITY,
                "bucket_capacity": BUCKET_CAPACITY,
            }
            for _, table in names
        },
    )
    return model, [f for f, _ in names], [t for _, t in names]


def _step(model, feature_names, keys: np.ndarray, score: int) -> None:
    """Forward and backward over ``keys``, one key per bag, in every feature; every key
    must read the initial value."""
    for module in get_dynamic_emb_module(model):
        module.set_score({name: score for name in module.table_names})
    values = torch.from_numpy(np.tile(keys, len(feature_names))).to(_device())
    features = KeyedJaggedTensor(
        keys=feature_names,
        values=values,
        lengths=torch.ones(len(values), dtype=torch.int64, device=_device()),
    )
    out = model(features)
    torch.testing.assert_close(out, torch.full_like(out, TABLE_INITIALIZER_VALUE))
    out.sum().backward()
    torch.cuda.synchronize()


def _held(model, table_names, universe: int):
    """For each table, which keys of 1..universe it holds (the keys it exports)."""
    torch.cuda.synchronize()
    held = [np.zeros(universe + 1, dtype=bool) for _ in table_names]
    counts = [0] * len(table_names)
    for module in get_dynamic_emb_module(model):
        for table_id, name in enumerate(module.table_names):
            index = table_names.index(name)
            for keys, _, _, _ in module.tables.export_keys_values(
                _device(), 65536, table_id=table_id
            ):
                keys = keys.cpu().numpy()
                held[index][keys] = True
                counts[index] += len(keys)
    return held, counts


def check_stress(num_tables: int) -> None:
    batches, universe = _workload(STEPS)
    model, feature_names, table_names = _model(num_tables)
    modules = get_dynamic_emb_module(model)
    assert modules, "the call would find no table in the model"

    def expansions() -> int:
        return sum(m._storage._state.key_map_generation for m in modules)

    expansions_before = expansions()

    last_step = np.zeros(universe + 1, dtype=np.int32)
    handles, asked = [], []
    erased_total = 0
    started = time.monotonic()

    def reap(wait: bool) -> None:
        nonlocal erased_total
        while handles and (wait or handles[0].done()):
            erased, spared = handles.pop(0).result()
            assert spared == 0, "nothing in this loop has a prefetch in flight"
            erased_total += erased

    for s, batch in enumerate(batches, start=1):
        _step(model, feature_names, batch, s)
        last_step[batch] = s
        if s % ERASE_EVERY == 0 and s > WINDOW:
            threshold = s - WINDOW + 1
            asked.append((s, threshold))
            handles.append(
                erase_by_score_async(
                    model, S.CUSTOMIZED, threshold, chunk_slots=CHUNK_SLOTS
                )
            )
        reap(wait=False)
    reap(wait=True)
    elapsed = time.monotonic() - started

    assert asked, "the loop was too short for an erase"
    asked_at, threshold = asked[-1]
    expect = (
        last_step >= threshold
    )  # present iff trained at or after the last threshold
    expect[0] = False
    held, counts = _held(model, table_names, universe)
    for index, name in enumerate(table_names):
        wrong_gone = np.flatnonzero(expect & ~held[index])
        wrong_kept = np.flatnonzero(~expect & held[index])
        assert len(wrong_gone) == 0, (
            f"{name}: {len(wrong_gone)} keys at or above the threshold {threshold} were "
            f"erased, e.g. {wrong_gone[:5].tolist()}"
        )
        assert len(wrong_kept) == 0, (
            f"{name}: {len(wrong_kept)} keys below the threshold {threshold} survived "
            f"the erase asked for at step {asked_at}, e.g. {wrong_kept[:5].tolist()}"
        )
        assert counts[index] == int(
            expect.sum()
        ), f"{name}: exports {counts[index]} keys, a key twice?"
    assert erased_total > 0
    # The scans had tables that grew under them to restart on.
    assert expansions() > expansions_before, "the tables never grew: no restart tried"

    # What survived still reads right.
    _step(model, feature_names, np.flatnonzero(expect)[:65536], STEPS + 1)
    print(
        f"stress (DMP, {num_tables} tables): {STEPS} steps, {len(asked)} erases, "
        f"{erased_total} keys erased, {expansions() - expansions_before} expansions, "
        f"{int(expect.sum())} left per table, "
        f"{elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="nccl")
    assert dist.get_world_size() == 1, "this test is for one rank"
    for tables in (1, 2):
        check_stress(tables)
    dist.barrier()
    print("test_erase_stress: all cases passed", flush=True)
    dist.destroy_process_group()
