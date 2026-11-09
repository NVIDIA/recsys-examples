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

"""The miss path in row chunks.

A step whose ``[rows, value_dim]`` buffers would not fit ``DYNAMICEMB_MISS_PATH_BUFFER_BYTES``
runs its miss path in consecutive row chunks (``sync_free.row_chunks``) instead of taking another
code path. The key-level work (the cache lookup and the pin of every hit, the scoring lookup of
storage, the gate) is done once for the whole batch; each chunk then looks up, loads, initializes,
inserts, evicts and writes back its own rows, and pins its new cache entries before the next one
starts.

What must hold, and is checked here with ``sf.CHUNK_ROWS`` (the test hook of the budget):

* with room in every tier, a chunked step is the unchunked step: the same outputs, the same rows
  in every tier, the same metrics, nothing pinned afterwards (the table kernels give no order
  between the keys of one launch, so running a chunk to completion first is one of the orders the
  single launch could have taken; with room no order can differ);
* under eviction pressure the plans differ, but a row is never another key's data. The
  modules are trained with a gradient of zero, so a key's row is its initial row wherever it
  sits (cache, storage, or initialized anew because a tier lost it) and any other row is another
  key's;
* a chunk that holds only padding (the unique arrays are as long as the batch) is a no-op;
* the memory the miss path adds is bounded by the budget, not by the batch.

The random initializers are not chunk invariant (the generator state of a thread depends on the
launch geometry), so every module here initializes with the DEBUG initializer, a function of the key.
"""
import inspect
import os
import sys

import pytest

try:
    import torch
    from dynamicemb import sync_free as sf
except ImportError as e:  # pragma: no cover
    pytest.skip("needs torch and dynamicemb: %s" % (e,), allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
NHS = os.path.join(HERE, "no_host_sync")
if NHS not in sys.path:
    sys.path.insert(0, NHS)
import host_sync_probe as hsp  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

# configs of the probe that take the chunked paths: a cache over a storage, HBM only, and the
# DEFAULT-mode storages (host only, hybrid), each with the variants that change the paths
ROOMY = [
    "cache_storage",
    "cache_storage_admit",
    "cache_storage_lfu",
    "cache_storage_retain",
    "hbm_only",
    "hbm_only_admit",
    "hbm_only_retain",
    "hybrid",
    "hybrid_lfu",
    "host_only",
]
# the tiers too small for the stream, the retaining ones included: every chunk then hands its
# victims to the pending list of retained keys
PRESSURE = [
    "cache_storage",
    "hbm_only",
    "hybrid",
    "cache_storage_retain",
    "hbm_only_retain",
]
POLL = ["cache_storage", "cache_storage_admit", "hbm_only", "hybrid"]
CHUNKS = [1, 7, 64]
PRESSURE_CHUNKS = [None, 3, 64]


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _build(config, scenario):
    """A module of the probe's config with the DEBUG initializer (a row is a function of the
    key) and its batch maker."""
    from dynamicemb.dynamicemb_config import (
        DynamicEmbInitializerArgs,
        DynamicEmbInitializerMode,
    )
    from dynamicemb.initializer import MultiTableInitializer

    cfg = hsp.CONFIGS[config]
    device = _device()
    mod = hsp.build_module(cfg, hsp.plan_of(cfg, scenario), device.index)
    args = DynamicEmbInitializerArgs(mode=DynamicEmbInitializerMode.DEBUG)
    mod._initializer = MultiTableInitializer.create([args, args], device)
    mod.train()
    return cfg, mod, hsp.BatchMaker(cfg, device, 1234)


def _step(mod, cfg, batch, gradient=1.0):
    kw = {"frequency_counters": batch.freq} if cfg.lfu else {}
    out = mod(batch.indices, batch.offsets, **kw)
    out.backward(torch.full_like(out, gradient))
    return out.detach().clone()


def _initial_rows(mod, batch):
    """What the DEBUG initializer makes of every key of the batch (a function of the key)."""
    n = batch.indices.numel()
    rows = torch.empty(n, hsp.DIM, device=batch.indices.device)
    tids = torch.zeros(n, dtype=torch.int64, device=batch.indices.device)
    mod._initializer(rows, batch.indices, tids, torch.arange(n, device=rows.device))
    return rows


def _trace(cfg, maker):
    """Fresh batches, the first two again (hits), and, for admission, batches that carry over
    half of the previous one's keys (a threshold of two admits them on the second sight).
    """
    if cfg.admit:
        return maker.overlap_stream(8)
    fresh = [maker.fresh_batch() for _ in range(5)]
    return fresh + [fresh[0], maker.fresh_batch(), fresh[1], fresh[0]]


def _table_state(holder):
    state = getattr(holder, "_state", None)
    return state if state is not None else getattr(holder, "_hbm", None)


def _export(mod):
    out = {}
    for name in ("t0", "t1"):
        keys, rows = mod.export_keys_values(name, _device())
        order = torch.argsort(keys)
        out[name] = (keys[order].cpu(), rows[order].cpu())
    return out


_BASELINES = {}


def _baseline(config):
    """The unchunked run of a config, taken once."""
    if config not in _BASELINES:
        _BASELINES[config] = _run_trace(config, None)
    return _BASELINES[config]


def _run_trace(config, chunk, gate=None):
    cfg, mod, maker = _build(config, "miss")
    batches = _trace(cfg, maker)
    if mod._cache is not None:
        mod.set_record_cache_metrics(True)
    outputs, metrics = [], []
    sf.CHUNK_ROWS = chunk  # the budget's test hook; always put back
    gate_before = sf.MISS_GATE
    if gate is not None:
        sf.MISS_GATE = gate
    try:
        for batch in batches:
            outputs.append(_step(mod, cfg, batch).cpu())
            if mod._cache is not None:
                metrics.append(mod._cache.cache_metrics.clone())
        torch.cuda.synchronize()
    finally:
        sf.CHUNK_ROWS = None
        sf.MISS_GATE = gate_before
    state = _export(mod)
    pinned = []
    for holder in (getattr(mod, "_cache", None), getattr(mod, "_storage", None)):
        table = getattr(_table_state(holder), "key_index_map", None)
        counters = getattr(table, "_ref_counter", None)
        if counters is not None:
            pinned.append(int(counters.abs().sum()))
    return outputs, state, metrics, pinned


@pytest.mark.parametrize("config", ROOMY)
@pytest.mark.parametrize("chunk", CHUNKS)
def test_a_chunked_step_is_the_unchunked_step_when_every_tier_has_room(config, chunk):
    outputs_a, state_a, metrics_a, pinned_a = _baseline(config)
    outputs_b, state_b, metrics_b, pinned_b = _run_trace(config, chunk)

    for step, (a, b) in enumerate(zip(outputs_a, outputs_b)):
        assert torch.equal(a, b), "%s: step %d differs for chunk=%d" % (
            config,
            step,
            chunk,
        )
    for name in state_a:
        assert torch.equal(state_a[name][0], state_b[name][0]), (
            "%s: the keys differ" % name
        )
        assert torch.equal(state_a[name][1], state_b[name][1]), (
            "%s: the rows differ" % name
        )
    assert all(
        torch.equal(a, b) for a, b in zip(metrics_a, metrics_b)
    ), "the cache metrics are the totals of the prefetch whatever its chunk plan"
    assert (
        pinned_a == pinned_b == [0] * len(pinned_a)
    ), "nothing is pinned after the backward"


@pytest.mark.parametrize("config", POLL)
@pytest.mark.parametrize("chunk", [7, 64])
def test_the_poll_gate_gives_the_results_of_the_always_gate_with_chunks(config, chunk):
    """``poll`` skips the miss path of a step when a count that has already landed says that
    nothing missed: a control flow that depends on timing, a chunk plan or not. The results are
    those of the always gate (the cache metrics are not: a skipped step does not record its
    insert)."""
    outputs_a, state_a, _, pinned_a = _baseline(config)
    outputs_b, state_b, _, pinned_b = _run_trace(config, chunk, gate="poll")

    for step, (a, b) in enumerate(zip(outputs_a, outputs_b)):
        assert torch.equal(a, b), "%s: step %d differs under poll, chunk=%d" % (
            config,
            step,
            chunk,
        )
    for name in state_a:
        assert torch.equal(state_a[name][0], state_b[name][0]), (
            "%s: the keys differ" % name
        )
        assert torch.equal(state_a[name][1], state_b[name][1]), (
            "%s: the rows differ" % name
        )
    assert pinned_a == pinned_b == [0] * len(pinned_a), "nothing is pinned afterwards"


@pytest.mark.parametrize("config", ["cache_storage", "hbm_only"])
def test_a_chunk_of_nothing_but_padding_is_a_no_op(config, monkeypatch):
    """The unique arrays are as long as the batch: with 90% repeated keys, most chunks hold
    padding only. The step equals the unchunked one and every chunk ran."""
    import dynamicemb.batched_dynamicemb_function as bdf

    cfg, mod, maker = _build(config, "miss")
    fresh = {
        t: maker.fresh(t, 40) for t in (0, 1)
    }  # 40 unique keys per table in ~576 slots
    batch = maker.build(fresh)
    n = batch.indices.numel()
    calls = []
    name = "_cache_miss_chunk" if config == "cache_storage" else "_hbm_miss_chunk"
    real = getattr(bdf, name)
    signature = inspect.signature(real)

    def spy(*args, **kwargs):
        # the chunk's keys by name: what comes before them differs between the two
        # functions, and between the tiers a chunk is called with
        calls.append(signature.bind(*args, **kwargs).arguments["keys"].numel())
        return real(*args, **kwargs)

    monkeypatch.setattr(bdf, name, spy)
    monkeypatch.setattr(sf, "CHUNK_ROWS", 64)
    chunked = _step(mod, cfg, batch).cpu()
    monkeypatch.setattr(sf, "CHUNK_ROWS", None)
    assert len(calls) == -(-n // 64) and sum(calls) == n

    cfg2, mod2, maker2 = _build(config, "miss")
    plain = _step(
        mod2, cfg2, maker2.build({t: maker2.fresh(t, 40) for t in (0, 1)})
    ).cpu()
    assert torch.equal(chunked, plain)


@pytest.mark.parametrize("config", PRESSURE)
@pytest.mark.parametrize("chunk", PRESSURE_CHUNKS)
def test_a_row_is_never_another_keys_data_under_eviction(config, chunk, monkeypatch):
    """Tiers too small for the stream: the cache evicts to the storage, the storage evicts for
    good, and a chunk may find that an earlier chunk took the row of a key it was going to load.
    The gradient is zero, so every row of a key is its initial row wherever it sits."""
    cfg, mod, maker = _build(config, "tier2_hit")
    plan = hsp.plan_of(cfg, "tier2_hit")
    pool = [maker.fresh_batch() for _ in range(6)]
    n_fill = hsp._fill_steps(cfg, "tier2_hit", plan)
    stream = pool + [maker.fresh_batch() for _ in range(n_fill)] + pool + pool[::-1]
    monkeypatch.setattr(sf, "CHUNK_ROWS", chunk)
    for step, batch in enumerate(stream):
        out = _step(mod, cfg, batch, gradient=0.0)
        assert torch.equal(
            out, _initial_rows(mod, batch)
        ), "%s chunk=%s: step %d read a row that is not its key's" % (
            config,
            chunk,
            step,
        )
    torch.cuda.synchronize()


def test_the_chunked_miss_path_adds_memory_in_proportion_to_the_budget(monkeypatch):
    """The peak memory a prefetch with new keys adds over its start: with one chunk it holds the
    whole batch's ``values`` and ``evicted_values``; with 64-row chunks only 64 rows' worth.
    Compared with each other, so that what the batch costs in index arrays cancels.

    The prefetch alone is measured: the peak of a whole step may well be the backward's (its
    output, the gradient the test hands it and the unique rows' gradients are three buffers of
    the batch's size, against the two of the miss path), and chunking would then show nothing
    in it."""
    monkeypatch.setattr(hsp, "DIM", 256)
    itemsize = 4

    def peak(chunk):
        cfg, mod, maker = _build("cache_storage", "miss")
        warm = maker.fresh_batch()
        batch = maker.fresh_batch()
        monkeypatch.setattr(sf, "CHUNK_ROWS", chunk)
        for _ in range(2):  # the allocator and the lazily built state settle
            _step(mod, cfg, warm)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        mod.prefetch(batch.indices, batch.offsets)
        torch.cuda.synchronize()
        grown = torch.cuda.max_memory_allocated() - base
        _step(
            mod, cfg, batch
        )  # its forward takes the prefetched state; the keys are unpinned
        torch.cuda.synchronize()
        monkeypatch.setattr(sf, "CHUNK_ROWS", None)
        return grown, batch.indices.numel()

    whole, n = peak(None)
    chunked, _ = peak(64)
    value_dim = hsp.DIM  # SGD: no optimizer state
    saved = (
        2 * (n - 64) * value_dim * itemsize
    )  # values and evicted_values, minus a chunk's
    print(
        "MEMORY prefetch: %d bytes over its start in one chunk, %d in 64-row chunks, %d expected "
        "to be saved" % (whole, chunked, saved)
    )
    assert whole - chunked >= 0.5 * saved, (
        "a prefetch with 64-row chunks peaked at %d bytes over its start, one chunk at %d: the "
        "miss path should add about %d bytes less" % (chunked, whole, saved)
    )
