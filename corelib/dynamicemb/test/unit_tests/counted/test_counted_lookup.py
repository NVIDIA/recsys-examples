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

"""The lookup step that keeps its counts on the device.

* ``test_step_matches_the_oracle``: with the DEBUG initializer the row of a new key is a
  function of the key, so what a forward returns, and where a step of the optimizer (SGD
  or Adam, whose state rides in the row) moves the rows, follows from the batches alone
  and is worked out here in plain torch. Every layout and pooling mode, with and without
  admission. The batches repeat keys and ask for more keys than the HBM tier of a cache
  or a hybrid layout has slots, so rows are evicted or spilled to the host and read
  back; one of them holds a single key and one none.
* ``test_step_does_not_block``: a training step and an eval step make no blocking host
  sync (cudaStreamSynchronize, cudaDeviceSynchronize, cudaEventSynchronize, a
  synchronous or pageable copy to the host), with the HBM tier full.
* ``test_large_batch_matches_the_oracle`` and ``test_glue_ops_over_a_capped_grid``:
  sizes at which a launch is capped and the kernels have to stride.
"""

import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import pytest
import torch
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import (
    BatchedDynamicEmbeddingTablesV2,
    StorageLayout,
)
from dynamicemb.dynamicemb_config import DEBUG_EMB_INITIALIZER_MOD
from dynamicemb.embedding_admission import FrequencyAdmissionStrategy, KVCounter

NONE = DynamicEmbPoolingMode.NONE
SUM = DynamicEmbPoolingMode.SUM
MEAN = DynamicEmbPoolingMode.MEAN
# (caching, local_hbm_for_values) of each layout, as in the v2 table tests. The budget
# is per table, and 1024 bytes buy the HBM tier of a cache or a hybrid layout the least
# it can have: one bucket of 1024 slots for each table.
LAYOUTS = {
    "hbm": (False, 1024**3),
    "cache": (True, 1024),
    "hybrid": (False, 1024),
    "host": (False, 0),
}
LR = 0.3
OPTIMIZERS = {
    "sgd": (EmbOptimType.SGD, {"learning_rate": LR}),
    "adam": (
        EmbOptimType.ADAM,
        {
            "learning_rate": LR,
            "weight_decay": 0.06,
            "eps": 3e-5,
            "beta1": 0.8,
            "beta2": 0.888,
        },
    ),
}
# Sequence output needs one width; pooled output may mix them.
DIMS = {NONE: [8, 8, 8], SUM: [8, 16, 8], MEAN: [8, 16, 8]}
FEATURE_TABLE_MAP = [0, 0, 1, 2]
# A batch asks for at most 4 * 128 * 4 = 2048 keys, fewer than the 3072 slots of a cache
# (1024 for each of its 3 tables): the prefetch of a cache never has to count the keys
# it pins, which would wait for the device.
BATCH = 128  # samples
# Keys 0..HOT-1 turn up again and again. The COLD ones, more than the 1024 slots of an
# HBM tier, turn up now and then: they are evicted (cache) or spilled to the host
# (hybrid), and read back.
HOT = 64
COLD = 2048
KEYS = HOT + COLD
COLD_SHARE = 0.75  # of the keys of a bag
THRESHOLD = 2  # the admission strategy's
# Room for about four times the keys: no bucket of a table, or of the admission counter,
# fills up and evicts a key that the oracle still holds.
MAX_CAPACITY = 8192
NUM_BATCHES = 12  # enough for a table to hold more keys than its HBM tier has slots


def make_module(
    layout: str,
    optimizer: str,
    pooling: DynamicEmbPoolingMode,
    admission: bool,
    max_capacity: int = MAX_CAPACITY,
    local_hbm: Optional[int] = None,
) -> BatchedDynamicEmbeddingTablesV2:
    caching, layout_hbm = LAYOUTS[layout]
    local_hbm = layout_hbm if local_hbm is None else local_hbm
    opt_type, opt_params = OPTIMIZERS[optimizer]
    options = [
        DynamicEmbTableOptions(
            dim=dim,
            init_capacity=max_capacity,
            max_capacity=max_capacity,
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=0,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=local_hbm,
            # The row of a new key is a function of the key: the oracle knows it.
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.DEBUG
            ),
            admit_strategy=FrequencyAdmissionStrategy(
                threshold=THRESHOLD, counter=KVCounter(capacity=max_capacity)
            )
            if admission
            else None,
        )
        for dim in DIMS[pooling]
    ]
    return BatchedDynamicEmbeddingTablesV2(
        table_names=["t0", "t1", "t2"],
        table_options=options,
        feature_table_map=FEATURE_TABLE_MAP,
        pooling_mode=pooling,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )


def jagged(bags: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keys and offsets of ``bags``: one list of keys per (feature, sample), feature by
    feature."""
    lengths = torch.tensor([len(bag) for bag in bags], dtype=torch.int64)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    keys = torch.tensor([key for bag in bags for key in bag], dtype=torch.int64)
    return keys.cuda(), offsets.cuda()


def make_batch(gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random keys with repeats: ``BATCH`` bags of 0 to 4 keys for every feature, a key
    being a ``COLD`` one ``COLD_SHARE`` of the time, else one of the ``HOT`` ones."""
    lengths = torch.randint(0, 5, (len(FEATURE_TABLE_MAP) * BATCH,), generator=gen)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    num_keys = int(offsets[-1])
    hot = torch.randint(0, HOT, (num_keys,), generator=gen)
    cold = torch.randint(HOT, KEYS, (num_keys,), generator=gen)
    is_cold = torch.rand(num_keys, generator=gen) < COLD_SHARE
    return torch.where(is_cold, cold, hot).cuda(), offsets.cuda()


# --------------------------------------------------------------------------------------
# Against an oracle


class Oracle:
    """What the tables should hold, worked out in plain torch from the batches alone.

    The DEBUG initializer fills the row of a new key with the key modulo
    ``DEBUG_EMB_INITIALIZER_MOD`` in every column. A step of the optimizer moves the row
    of each key of the batch that a table holds, on the sum of the gradients of the key:
    one per occurrence, divided by the length of its bag for MEAN. The other rows stay,
    and so does a key that admission has not let in: it has no row to move.
    """

    def __init__(
        self,
        pooling: DynamicEmbPoolingMode,
        key_space: int = KEYS,
        threshold: int = 0,
        optimizer: str = "sgd",
    ):
        self.pooling, self.threshold, self.optimizer = pooling, threshold, optimizer
        keys = torch.arange(key_space, device="cuda")
        initial = (keys % DEBUG_EMB_INITIALIZER_MOD).float()[:, None]
        # Per table: the row of every key, whether the table holds the key, and how
        # often a batch has asked for it while the table did not.
        self.rows = [initial.repeat(1, dim) for dim in DIMS[pooling]]
        self.stored = [torch.zeros_like(keys, dtype=torch.bool) for _ in self.rows]
        self.asked = [torch.zeros_like(keys) for _ in self.rows]
        # Adam: the first and second moment of every row, and the steps taken so far.
        if optimizer == "adam":
            self.m = [torch.zeros_like(rows) for rows in self.rows]
            self.v = [torch.zeros_like(rows) for rows in self.rows]
        self.steps = 0

    @staticmethod
    def features(keys, offsets):
        """Per feature: its table, its keys, the bag of each key and the bag lengths."""
        bounds = offsets.tolist()
        batch = (len(bounds) - 1) // len(FEATURE_TABLE_MAP)
        for feature, table in enumerate(FEATURE_TABLE_MAP):
            first, last = feature * batch, (feature + 1) * batch
            lengths = offsets[first + 1 : last + 1] - offsets[first:last]
            bags = torch.repeat_interleave(torch.arange(batch, device="cuda"), lengths)
            yield table, keys[bounds[first] : bounds[last]], bags, lengths

    def admit(self, keys, offsets):
        """A key a table lacks gets in the ``threshold``-th time a batch asks for it."""
        wanted: List[List[torch.Tensor]] = [[] for _ in self.rows]
        for table, table_keys, _, _ in self.features(keys, offsets):
            wanted[table].append(table_keys)
        for stored, asked, chunks in zip(self.stored, self.asked, wanted):
            missing = torch.cat(chunks).unique()
            missing = missing[~stored[missing]]
            asked[missing] += 1
            stored[missing[asked[missing] >= self.threshold]] = True

    def forward(self, keys, offsets, training=True):
        """What the module returns. A training forward admits keys; an evaluation gives
        a key the tables lack zeros."""
        if training:
            self.admit(keys, offsets)
        outputs = []
        for table, table_keys, bags, lengths in self.features(keys, offsets):
            rows = self.rows[table][table_keys]
            if not training:
                rows = rows * self.stored[table][table_keys][:, None]
            if self.pooling != NONE:
                pooled = torch.zeros(lengths.numel(), rows.size(1), device="cuda")
                rows = pooled.index_add_(0, bags, rows)
                if self.pooling == MEAN:
                    rows = rows / lengths.clamp(min=1)[:, None]
            outputs.append(rows)
        return torch.cat(outputs, dim=0 if self.pooling == NONE else 1)

    def backward(self, keys, offsets, grad):
        """A step of the optimizer on the gradient of the output."""
        summed = [torch.zeros_like(rows) for rows in self.rows]
        in_batch = [torch.zeros_like(stored) for stored in self.stored]
        row = column = 0  # where the gradients of the next feature begin
        for table, table_keys, bags, lengths in self.features(keys, offsets):
            if self.pooling == NONE:
                key_grads = grad[row : row + table_keys.numel()]
                row += table_keys.numel()
            else:
                dim = DIMS[self.pooling][table]
                key_grads = grad[:, column : column + dim][bags]
                column += dim
                if self.pooling == MEAN:
                    key_grads = key_grads / lengths[bags][:, None]
            summed[table].index_add_(0, table_keys, key_grads)
            in_batch[table][table_keys] = True
        self.steps += 1
        for table, grads in enumerate(summed):
            live = (self.stored[table] & in_batch[table])[:, None]
            self.update(table, grads, live)

    def update(self, table, grads, live):
        """The optimizer's rule on the rows of ``table`` that ``live`` flags, one flag a
        row; the other rows, and their moments, stay."""
        rows, args = self.rows[table], OPTIMIZERS[self.optimizer][1]
        if self.optimizer == "sgd":
            new_rows = rows - args["learning_rate"] * grads
        else:
            beta1, beta2 = args["beta1"], args["beta2"]
            m = beta1 * self.m[table] + (1 - beta1) * grads
            v = beta2 * self.v[table] + (1 - beta2) * grads * grads
            # The bias is corrected by the steps of the module, not of the key.
            m_hat = m / (1 - beta1**self.steps)
            v_hat = v / (1 - beta2**self.steps)
            step = m_hat / (v_hat.sqrt() + args["eps"]) + args["weight_decay"] * rows
            new_rows = rows - args["learning_rate"] * step
            self.m[table] = torch.where(live, m, self.m[table])
            self.v[table] = torch.where(live, v, self.v[table])
        rows.copy_(torch.where(live, new_rows, rows))


def train_and_compare(module, oracle, keys, offsets, gen):
    """A training step: the output is the oracle's, and both take the same step of the
    optimizer."""
    out = module(keys, offsets)
    torch.testing.assert_close(out, oracle.forward(keys, offsets), rtol=1e-4, atol=1e-4)
    weight = torch.randn(out.shape, generator=gen).cuda()  # a random upstream gradient
    (out * weight).sum().backward()
    oracle.backward(keys, offsets, weight)


def eval_and_compare(module, oracle, keys, offsets):
    """An eval reads back what the steps wrote: the oracle's rows, zeros for a key the
    tables lack."""
    module.eval()
    with torch.no_grad():
        out = module(keys, offsets)
    expected = oracle.forward(keys, offsets, training=False)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def hbm_tier(module):
    """The state of the HBM tier of a cache or a hybrid layout, which holds only part of
    the table; None for a layout whose one tier holds it all."""
    if module._storage_layout == StorageLayout.CACHING:
        return module._cache._state
    if module._storage_layout == StorageLayout.HYBRID:
        return module._storage._hbm
    return None


def assert_tier_overflows(module, oracle):
    """Some table holds more keys than its HBM tier (of a cache or a hybrid layout) has
    slots: the steps so far evicted keys from the cache, or spilled them to the host
    tier, and found them again."""
    tier = hbm_tier(module)
    if tier is not None:
        slots = tier.key_index_map.per_table_capacity_
        held = [int(stored.sum()) for stored in oracle.stored]
        overflowed = any(n > cap for n, cap in zip(held, slots))
        assert overflowed, f"{held} keys held in tables of {slots} HBM slots"


# The optimizer and the pooling mode do not depend on each other: SGD with each pooling
# mode, and Adam, which keeps its moments in the row, with mixed widths.
@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize(
    "optimizer, pooling",
    [("sgd", NONE), ("sgd", SUM), ("sgd", MEAN), ("adam", SUM)],
    ids=["sgd-none", "sgd-sum", "sgd-mean", "adam-sum"],
)
@pytest.mark.parametrize("admission", [False, True])
def test_step_matches_the_oracle(layout, optimizer, pooling, admission):
    module = make_module(layout, optimizer, pooling, admission)
    oracle = Oracle(
        pooling, threshold=THRESHOLD if admission else 0, optimizer=optimizer
    )
    gen = torch.Generator().manual_seed(7)
    bags = len(FEATURE_TABLE_MAP) * BATCH
    batches = [
        make_batch(gen),
        make_batch(gen),
        jagged([[7, 7]] * bags),  # one key, in every bag, twice
        jagged([[42]] + [[]] * (bags - 1)),  # one key
        jagged([[]] * bags),  # no key
        *[make_batch(gen) for _ in range(NUM_BATCHES)],
    ]
    for keys, offsets in batches:
        train_and_compare(module, oracle, keys, offsets, gen)
    assert_tier_overflows(module, oracle)

    # Every key, one to a bag.
    keys = torch.arange(KEYS, device="cuda").repeat(len(FEATURE_TABLE_MAP))
    offsets = torch.arange(keys.numel() + 1, device="cuda")
    eval_and_compare(module, oracle, keys, offsets)


# --------------------------------------------------------------------------------------
# No blocking host sync in a step


_SYNC_RUNTIME_CALLS = (
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
)


def blocking_syncs(step, num_steps: int = 3) -> Dict[str, int]:
    """Run ``step`` ``num_steps`` times under the profiler and count the host syncs
    inside the steps: the runtime's synchronize calls, synchronous memcpys, and copies
    to pageable host memory."""
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("counted_steps"):
            for _ in range(num_steps):
                step()
        # Outside the window: it is this wait, not one of the steps', that ends them.
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as f:
            events: List[dict] = json.load(f)["traceEvents"]

    windows = [e for e in events if e.get("name") == "counted_steps"]
    assert windows, "the profiler did not record the step window"
    begin, end = windows[0]["ts"], windows[0]["ts"] + windows[0]["dur"]
    counts: Dict[str, int] = {}
    runtime_calls = 0
    for event in events:
        if not (begin <= event.get("ts", -1) <= end):
            continue
        name = event.get("name", "")
        category = event.get("cat", "")
        runtime_calls += category == "cuda_runtime"
        blocking = category == "cuda_runtime" and (
            name in _SYNC_RUNTIME_CALLS
            or (name.startswith("cudaMemcpy") and not name.endswith("Async"))
        )
        blocking = blocking or (
            category == "gpu_memcpy" and "DtoH" in name and "Pageable" in name
        )
        if blocking:
            counts[name] = counts.get(name, 0) + 1
    # A profiler that saw no launch would let every step pass.
    assert runtime_calls > 0, "the profiler recorded no CUDA runtime calls"
    return counts


@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("optimizer", list(OPTIMIZERS))
@pytest.mark.parametrize("pooling", [NONE, SUM])
@pytest.mark.parametrize("admission", [False, True])
def test_step_does_not_block(layout, optimizer, pooling, admission):
    module = make_module(layout, optimizer, pooling, admission)
    gen = torch.Generator().manual_seed(11)
    # The warm-up fills the HBM tier. The steps under the profiler follow it with
    # batches that still hold keys new to the tables: inserting them evicts rows (cache)
    # or spills them to the host (hybrid).
    warm, profiled = NUM_BATCHES, 3
    batches = [make_batch(gen) for _ in range(warm + profiled)]

    def train_step() -> None:
        keys, offsets = batches[train_step.i % len(batches)]
        train_step.i += 1
        module(keys, offsets).sum().backward()

    train_step.i = 0
    for _ in range(warm):  # misses, evictions or spills, and the first launches
        train_step()
    oracle = Oracle(pooling, threshold=THRESHOLD if admission else 0)
    for keys, offsets in batches[:warm]:
        oracle.admit(keys, offsets)
    assert_tier_overflows(module, oracle)
    torch.cuda.synchronize()
    assert blocking_syncs(train_step, profiled) == {}, "a training step blocks"

    def no_grad_step() -> None:
        # A forward of a module that trains, without an autograd graph: it has no
        # backward to release the pins of its keys, so it releases them itself.
        keys, offsets = batches[no_grad_step.i % len(batches)]
        no_grad_step.i += 1
        with torch.no_grad():
            module(keys, offsets)

    no_grad_step.i = 0
    no_grad_step()
    torch.cuda.synchronize()
    assert blocking_syncs(no_grad_step, profiled) == {}, "a no-grad forward blocks"

    module.eval()

    def eval_step() -> None:
        keys, offsets = batches[0]
        with torch.no_grad():
            module(keys, offsets)

    eval_step()
    torch.cuda.synchronize()
    assert blocking_syncs(eval_step) == {}, "an eval step blocks"


def test_outstanding_keys_are_decided_by_the_real_count():
    """A cache is over-subscribed by the unique keys that prefetched batches pin, not by
    the number of keys they look up: a batch of repeats larger than the cache is fine,
    one of distinct keys is an error, raised before the cache is changed."""
    module = make_module("cache", "sgd", NONE, False)
    capacity = module._cache._state.capacity
    n = (capacity // 4 + 100) * 4  # more input keys than the cache has rows
    offsets = torch.arange(0, n + 1, n // 4, device="cuda")  # 4 features, 1 sample

    repeats = torch.arange(n, device="cuda") % 8  # 8 distinct keys
    module.prefetch(repeats, offsets)
    module(repeats, offsets)  # consumes it: nothing is outstanding any more

    resident = module._cache._state.key_index_map.size()
    distinct = torch.arange(n, device="cuda")
    with pytest.raises(RuntimeError, match="Outstanding prefetched keys"):
        module.prefetch(distinct, offsets)
    assert module._cache._state.key_index_map.size() == resident


# --------------------------------------------------------------------------------------
# Batches large enough for the kernels to stride: a launch is capped at a few times the
# threads the GPU can hold, about a million entries on an H100.

BIG = 1 << 22


@pytest.mark.parametrize("n", [1, BIG // 2 + 1, BIG])
def test_glue_ops_over_a_capped_grid(n):
    from dynamicemb_extensions import (
        counted_count_true,
        counted_gather,
        counted_scatter,
        counted_scatter_fill,
        flagged_compact,
    )

    num_valid = torch.tensor([n], device="cuda")
    flags = torch.rand(BIG, device="cuda") < 0.3
    keys = torch.randint(0, 1 << 40, (BIG,), device="cuda")
    idx = torch.randperm(BIG, device="cuda")  # n live entries, all distinct, all valid

    count, positions, (selected,) = flagged_compact(flags, [keys], num_valid)
    expect = torch.nonzero(flags[:n]).flatten()
    assert int(count) == expect.numel()
    assert torch.equal(positions[: expect.numel()], expect)
    assert torch.equal(selected[: expect.numel()], keys[expect])
    assert int(counted_count_true(flags, num_valid)) == int(flags[:n].sum())

    gathered = counted_gather(keys, idx, num_valid)
    assert torch.equal(gathered[:n], keys[idx[:n]])

    scattered = torch.zeros_like(keys)
    counted_scatter(scattered, idx, keys, num_valid)
    expect = torch.zeros_like(keys)
    expect[idx[:n]] = keys[:n]
    assert torch.equal(scattered, expect)

    filled = torch.zeros_like(keys)
    counted_scatter_fill(filled, idx, 7, num_valid)
    expect = torch.zeros_like(keys)
    expect[idx[:n]] = 7
    assert torch.equal(filled, expect)


@pytest.mark.parametrize("layout", ["hbm", "cache", "host"])
def test_large_batch_matches_the_oracle(layout):
    """The whole step at a size where every launch is capped: more than a million keys
    per batch, new keys in every batch, and (cache) evictions in the second.
    """
    rows, key_space = 1 << 22, 2_000_000
    # The budget is per table: 24 MiB of the 128 MiB that a table takes keep 786 432
    # rows in the cache. That is room for the 625 000 or so keys that one batch asks
    # table 0 for, not for the 1 055 000 or so of two batches: the second one evicts.
    local_hbm = 24 << 20 if layout == "cache" else None
    module = make_module(layout, "sgd", NONE, False, rows, local_hbm)
    oracle = Oracle(NONE, key_space)
    per_feature = 375_000
    offsets = torch.arange(0, 4 * per_feature + 1, per_feature, device="cuda")
    gen = torch.Generator().manual_seed(3)

    for _ in range(2):
        keys = torch.randint(0, key_space, (4 * per_feature,), generator=gen).cuda()
        train_and_compare(module, oracle, keys, offsets, gen)
    assert_tier_overflows(module, oracle)

    sample = torch.randint(0, key_space, (100_000,), generator=gen).cuda()
    sample_offsets = torch.arange(0, 100_001, 25_000, device="cuda")
    eval_and_compare(module, oracle, sample, sample_offsets)
