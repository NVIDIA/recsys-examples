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

"""Benchmark the lookup of many separate DynamicEmb collections.

Every step runs K independent BatchedDynamicEmbeddingTablesV2 modules (each is its
own lookup group) on the next of a few pre-built inputs, taken in turn, and nothing
else: the modules' forward and their backward with the fused sparse update, driven by
gradients built beforehand.  There is no dense part, no loss and no data creation in
the timed region, so GPU work per step is small and CPU launch latency and blocking
host syncs dominate.  The inputs share no key: a cache that cannot hold all of them
is refreshed at every step.  Only public DynamicEmb APIs are used (plus a guarded read
of ``_storage_layout`` to check the layout): run the same script on each tree with
--json and print the before/after table with --compare.  One GPU; no torchrun,
process group or TorchRec.  See benchmark/README.md, section 3.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict, dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    import torch

Batch = List["torch.Tensor"]  # the keys of each collection
# The StorageLayout the library must pick for each --storage choice.
LAYOUTS = dict(hbm="HBM_ONLY", caching="CACHING", hybrid="HYBRID", host="HOST_ONLY")
GIB = 2**30
META_BYTES_PER_ROW = 32  # hash-table key, digest, score, value row, ref count (approx.)
SYNC_KINDS = ("stream_sync", "device_sync", "event_sync", "memcpy_sync")
SWEEP_COLLECTIONS, SWEEP_BATCH_SIZES = (1, 4, 16, 64), (256, 1024, 4096)  # --sweep grid
# Default --hbm-ratio of the layouts that take one: the HBM tier of a hybrid layout. The
# cache of a caching layout is sized from the inputs unless --hbm-ratio is given.
HBM_RATIO = dict(hybrid=0.1)
CACHE_BUCKET = 1024  # rows per bucket of a cache: its capacity is a multiple of this


@dataclass(frozen=True)
class Config:
    """Everything that defines one benchmark point."""

    storage: str = "hbm"
    collections: int = 32
    tables: int = 2
    batch_size: int = 1024
    hotness: int = 4
    dim: int = 64
    num_embeddings: int = 1_000_000
    alpha: float = 1.05
    optimizer: str = "sgd"
    hbm_ratio: Optional[float] = None  # caching: None sizes the cache from the inputs
    lr: float = 0.01
    inputs: int = 2
    seed: int = 0

    @property
    def row_bytes(self) -> int:
        """Bytes of one fp32 value row (embedding + optimizer state)."""
        return 4 * self.dim * (3 if self.optimizer == "adam" else 1)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step time of the lookup of K separate DynamicEmb collections "
        "(forward and backward, nothing else): shows what blocking host syncs inside "
        "DynamicEmb cost.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add = parser.add_argument
    add("--collections", type=int, default=32, help="K: separate modules")
    add("--tables", type=int, default=2, help="T: tables per collection")
    add("--dim", type=int, default=64, help="D: embedding dim")
    add("--batch-size", type=int, default=1024, help="B: samples per step")
    add("--hotness", type=int, default=4, help="H: keys per sample and table")
    add("--num-embeddings", type=int, default=1_000_000, help="rows per table")
    add("--alpha", type=float, default=1.05, help="pow-law exponent (more dups if big)")
    add("--optimizer", choices=["sgd", "adam"], default="sgd", help="sparse optimizer")
    add("--lr", type=float, default=0.01, help="learning rate of the sparse optimizer")
    add(
        "--inputs",
        type=int,
        default=2,
        help="pre-built inputs, used in turn; each draws its keys from its own slice of "
        "the rows, so no two share a key (1: the same input at every step)",
    )
    add("--seed", type=int, default=0, help="seed of the key and gradient generators")
    add(
        "--storage",
        choices=list(LAYOUTS),
        default="hbm",
        help="hbm: all in HBM; caching: HBM cache over host storage; hybrid: partial "
        "HBM without cache; host: host memory only",
    )
    add(
        "--hbm-ratio",
        type=float,
        default=argparse.SUPPRESS,
        help="HBM share of the rows for caching (the cache) and hybrid (the HBM tier). "
        f"Default: hybrid {HBM_RATIO['hybrid']}; caching, the smallest cache that holds "
        "the unique keys of an input, which misses on the other inputs' keys at every "
        "step (see the README)",
    )
    skip = argparse.SUPPRESS
    add("--warmup", type=int, default=skip, help="untimed steps (30; 20 with --sweep)")
    add("--steps", type=int, default=skip, help="timed steps (200; 100 with --sweep)")
    add("--windows", type=int, default=5, help="timed windows (median, min, max)")
    add(
        "--sync-count",
        action="store_true",
        help="profile a few extra steps with torch.profiler: blocking host syncs per "
        "step and GPU-busy fraction",
    )
    add("--sync-steps", type=int, default=3, help="steps profiled by --sync-count")
    add("--profile-steps", type=int, default=0, help="steps bracketed for nsys")
    add("--device", type=int, help="CUDA device index")
    add("--json", metavar="PATH", help="write parameters and metrics as JSON")
    add("--sweep", action="store_true", help="run the K x B grid for --storage")
    add(
        "--compare",
        nargs=2,
        metavar=("A_JSON", "B_JSON"),
        help="print the speedup of run B over run A and exit (no GPU needed)",
    )
    args = parser.parse_args(argv)
    args.hbm_ratio = getattr(args, "hbm_ratio", HBM_RATIO.get(args.storage))
    if min(args.collections, args.tables, args.inputs, args.windows) < 1:
        parser.error("--collections, --tables, --inputs and --windows must be >= 1")
    if args.num_embeddings // args.inputs < 2:
        parser.error("--num-embeddings must be at least 2 per input")
    if args.alpha == 1:
        parser.error("--alpha must not be 1: the pow-law helper divides by 1 - alpha")
    ratio = args.hbm_ratio
    if args.storage in ("caching", "hybrid") and ratio is not None:
        if not 0 < ratio < 1:
            parser.error("--hbm-ratio must be in (0, 1) for --storage caching/hybrid")
    return args


def git_info(path: str) -> Dict[str, Any]:
    """Commit and dirty flag of the git tree containing ``path`` ({} if none)."""

    def git(*cmd: str) -> str:
        done = subprocess.run(["git", "-C", path, *cmd], capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else ""

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "-suno"))
    except OSError:  # no git binary
        return {}
    return {"commit": commit, "dirty": dirty} if commit else {}


def run_meta(device: torch.device) -> Dict[str, Any]:
    """Where the numbers come from: tree commit, library and extension paths, versions, GPU."""
    import dynamicemb
    import dynamicemb_extensions
    import torch

    extension = os.path.abspath(dynamicemb_extensions.__file__)
    package = os.path.dirname(os.path.abspath(dynamicemb.__file__))
    here = os.path.dirname(os.path.abspath(__file__))
    return {
        **(git_info(package) or git_info(here)),
        "dynamicemb_path": package,
        "dynamicemb_extension": extension,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv[1:],
    }


def plan_memory(cfg: Config) -> Dict[str, float]:
    """Planned memory in GiB: table values per tier, hash-table metadata, inputs."""
    n_tables = cfg.collections * cfg.tables
    values = n_tables * cfg.num_embeddings * cfg.row_bytes
    ratio = cfg.hbm_ratio or 0.0  # a cache sized from the inputs is a few K rows: ~0
    hbm, host = {
        "hbm": (values, 0.0),
        "caching": (ratio * values, values),  # HBM is a cache; host holds every row
        "hybrid": (ratio * values, (1 - ratio) * values),
        "host": (0.0, values),
    }[cfg.storage]
    plan = {
        "hbm_values": hbm,
        "host_values": host,
        "metadata": n_tables * cfg.num_embeddings * META_BYTES_PER_ROW,
        # int64 keys of every input, fp32 gradient of every pooled output
        "data": cfg.batch_size
        * (cfg.inputs * n_tables * cfg.hotness * 8 + n_tables * cfg.dim * 4),
    }
    plan = {name: nbytes / GIB for name, nbytes in plan.items()}
    plan["hbm_total"] = plan["hbm_values"] + plan["metadata"] + plan["data"]
    return plan


def build_inputs(cfg: Config, device: torch.device) -> Tuple[torch.Tensor, List[Batch]]:
    """Pre-build ``cfg.inputs`` inputs on the GPU, plus the shared offsets.

    Every input draws its keys from the pow-law helper of dataset_generator.py over
    ``R = num_embeddings // inputs`` rows and shifts them by ``s * R``: the same
    distribution in its own slice ``[s * R + 1, (s + 1) * R]`` of the rows, so no two
    inputs share a key.  Bags are feature-major as in a KeyedJaggedTensor: bag
    ``table * B + sample`` owns keys ``[bag * H, (bag + 1) * H)`` of every
    collection's index tensor.
    """
    import torch
    from dataset_generator import PowerLaw

    k, t, b, h = cfg.collections, cfg.tables, cfg.batch_size, cfg.hotness
    span = cfg.num_embeddings // cfg.inputs
    offsets = torch.arange(t * b + 1, device=device, dtype=torch.int64) * h
    inputs: List[Batch] = []
    for s in range(cfg.inputs):
        keys = PowerLaw(1, span + 1, cfg.alpha, k * t * b * h, device) + s * span
        inputs.append(list(keys.view(k, t * b * h).unbind(0)))
    return offsets, inputs


def key_stats(cfg: Config, inputs: List[Batch]) -> Dict[str, float]:
    """Unique-key statistics of collection 0, table 0 over the inputs."""
    import torch

    n = cfg.batch_size * cfg.hotness
    per_input = [keys[0][:n] for keys in inputs]
    unique = [torch.unique(keys).numel() for keys in per_input]
    return {
        "keys_per_table": n,
        "unique_per_input": sum(unique) / len(unique),
        "unique_ratio": sum(unique) / (n * len(per_input)),
        "working_set": torch.unique(torch.cat(per_input)).numel(),
    }


def cache_rows(unique: int, num_embeddings: int) -> int:
    """The rows of the default cache of a caching layout: the smallest multiple of
    ``CACHE_BUCKET`` that holds ``unique`` keys.

    A cache as large as the table is the table: the library builds an HBM-only layout
    then, so a table that small is refused here, before any module is built.
    """
    rows = -(-unique // CACHE_BUCKET) * CACHE_BUCKET
    if rows >= num_embeddings:
        raise ValueError(
            f"--storage caching needs a cache smaller than the table: an input has "
            f"{unique} unique keys per table, which take a cache of {rows} rows (a "
            f"multiple of {CACHE_BUCKET}), and --num-embeddings is {num_embeddings}. "
            "Raise --num-embeddings, lower --batch-size or --hotness, or use another "
            "--storage."
        )
    return rows


def cache_ratio(cfg: Config, inputs: List[Batch]) -> float:
    """The ``--hbm-ratio`` that gives a caching layout its default cache: the smallest
    one that holds the unique keys of any input in any table of collection 0.

    A cache holds a multiple of ``CACHE_BUCKET`` rows.  The library raises when the
    unique keys of a step do not fit its cache, so a size taken from the inputs works
    at any batch size.
    """
    import torch

    n = cfg.batch_size * cfg.hotness
    unique = max(
        torch.unique(keys[0][t * n : (t + 1) * n]).numel()
        for keys in inputs
        for t in range(cfg.tables)
    )
    return cache_rows(unique, cfg.num_embeddings) / cfg.num_embeddings


def make_collection(
    cfg: Config, k: int, device: torch.device, hbm_ratio: Optional[float]
) -> Any:
    """Collection ``k``: one BatchedDynamicEmbeddingTablesV2 of T pooled tables."""
    import dynamicemb as de
    import torch
    from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2

    # The layout follows from the HBM budget (see _create_cache_storage): a budget
    # covering the table is HBM-only, none is host-only, in between is a cache
    # (caching=True) or a hybrid split (caching=False).
    ratio = {"hbm": 2.0, "host": 0.0}.get(cfg.storage, hbm_ratio)
    options = [
        de.DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            dim=cfg.dim,
            max_capacity=cfg.num_embeddings,
            local_hbm_for_values=int(ratio * cfg.num_embeddings * cfg.row_bytes),
            bucket_capacity=128,
            initializer_args=de.DynamicEmbInitializerArgs(
                mode=de.DynamicEmbInitializerMode.NORMAL, std_dev=0.05
            ),
            score_strategy=de.DynamicEmbScoreStrategy.TIMESTAMP,
            caching=cfg.storage == "caching",
        )
        for _ in range(cfg.tables)
    ]
    optimizer = {"sgd": de.EmbOptimType.EXACT_SGD, "adam": de.EmbOptimType.ADAM}
    return BatchedDynamicEmbeddingTablesV2(
        table_options=options,
        table_names=[f"c{k}_t{t}" for t in range(cfg.tables)],
        pooling_mode=de.DynamicEmbPoolingMode.SUM,
        output_dtype=torch.float32,
        device=device,
        optimizer=optimizer[cfg.optimizer],
        learning_rate=cfg.lr,
        stochastic_rounding=False,
    )


class Workload:
    """K DynamicEmb collections; ``step`` is the lookup of the next input."""

    def __init__(self, cfg: Config, device: torch.device, verbose: bool) -> None:
        import torch

        self.offsets, self.inputs = build_inputs(cfg, device)
        # The gradient of every pooled output, built once: the backward then has no
        # work but the library's own.
        self.grads = [
            torch.randn(cfg.batch_size, cfg.tables * cfg.dim, device=device) * 0.01
            for _ in range(cfg.collections)
        ]
        torch.cuda.empty_cache()  # the tables allocate outside the caching allocator
        self.hbm_ratio = cfg.hbm_ratio
        if self.hbm_ratio is None and cfg.storage == "caching":
            self.hbm_ratio = cache_ratio(cfg, self.inputs)
        self.modules = []
        for k in range(cfg.collections):
            show = verbose and k == 0  # every module prints a memory table when built
            with nullcontext() if show else redirect_stdout(io.StringIO()):
                self.modules.append(make_collection(cfg, k, device, self.hbm_ratio))
        layout = getattr(self.modules[0], "_storage_layout", None)
        self.layout = getattr(layout, "name", None)  # None if a tree lacks the field
        if self.layout not in (None, LAYOUTS[cfg.storage]):
            raise RuntimeError(f"--storage {cfg.storage} built layout {self.layout}")
        self.names = [f"collection_{k}" for k in range(cfg.collections)]
        self.count = 0  # steps so far: the inputs are taken in turn across all calls
        self._nvtx, self._backward = torch.cuda.nvtx, torch.autograd.backward

    def step(self) -> List[torch.Tensor]:
        """K lookups, then their K backwards with the fused sparse update; no sync."""
        nvtx = self._nvtx
        i = self.count
        self.count += 1
        nvtx.range_push(f"step_{i}")
        pooled = []
        for module, name, keys in zip(
            self.modules, self.names, self.inputs[i % len(self.inputs)]
        ):
            nvtx.range_push(name)
            pooled.append(module(keys, self.offsets))  # prefetch + lookup, pooled sum
            nvtx.range_pop()
        nvtx.range_push("backward")  # every collection's backward + fused update
        self._backward(pooled, self.grads)
        nvtx.range_pop()
        nvtx.range_pop()
        return pooled


def summarize(v: List[float]) -> Dict[str, Any]:
    return {"median": statistics.median(v), "min": min(v), "max": max(v), "windows": v}


def measure(wl: Workload, warmup: int, steps: int, windows: int) -> Dict[str, Any]:
    """Warm up, then time ``windows`` windows of ``steps // windows`` steps each.

    Per window: wall ms/step (steps launched back to back, one synchronize at the
    end) and host ms/step (time spent inside ``Workload.step``, no sync).
    """
    import torch

    for _ in range(warmup):
        wl.step()
    torch.cuda.synchronize()
    per = max(1, steps // windows)
    wall: List[float] = []
    host: List[float] = []
    for _ in range(windows):
        gc.collect()
        gc.disable()  # keep collector pauses out of the timed region
        try:
            torch.cuda.synchronize()
            inside = 0.0
            begin = time.perf_counter()
            for _ in range(per):
                enter = time.perf_counter()
                pooled = wl.step()
                inside += time.perf_counter() - enter
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
        finally:
            gc.enable()
        if not all(bool(torch.isfinite(p).all()) for p in pooled):  # outside the window
            raise RuntimeError(f"non-finite output after step {wl.count}")
        wall.append(elapsed / per * 1e3)
        host.append(inside / per * 1e3)
    return {
        "wall_ms": summarize(wall),
        "host_ms": summarize(host),
        "steps_per_window": per,
    }


def classify(name: str) -> Optional[str]:
    """Map a profiler event name to a blocking-sync kind (matching by substring)."""
    n = name.lower()
    if n.startswith("cuda"):  # runtime API calls
        for part, kind in (
            ("streamsynchronize", "stream_sync"),
            ("devicesynchronize", "device_sync"),
            ("eventsynchronize", "event_sync"),
        ):
            if part in n:
                return kind
        if n.startswith("cudamemcpy") and "async" not in n:
            return "memcpy_sync"
    elif "dtoh" in n and "pageable" in n:  # GPU-side "Memcpy DtoH (Device -> Pageable)"
        return "d2h_pageable"
    return None


def profile_steps(step: Callable[[], Any], n: int) -> Dict[str, Any]:
    """Profile ``n`` steps: blocking host syncs per step and the GPU-busy fraction.

    Only events inside the steps count (the profiler ends its own window with a
    device sync).  The GPU-busy fraction is the summed duration of GPU activities
    over the span from the first step to the last GPU activity; the profiler's
    CPU overhead lowers it, so compare it between trees, not with the timed runs.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            with record_function("bench_step"):
                step()
    events = list(prof.events())
    gpu = {id(e) for e in events if "cuda" in str(e.device_type).lower()}
    steps = [e for e in events if e.name == "bench_step" and id(e) not in gpu]
    if not steps or not any(e.name.lower().startswith("cuda") for e in events):
        return {"note": "torch.profiler returned no CUDA runtime events (CUPTI?)"}
    first = min(e.time_range.start for e in steps)
    last_cpu = max(e.time_range.end for e in steps)
    counts = dict.fromkeys(SYNC_KINDS + ("d2h_pageable",), 0)
    busy = last_gpu = 0.0
    for e in events:
        start, end = e.time_range.start, e.time_range.end
        on_gpu = id(e) in gpu
        if start < first or (start > last_cpu and not on_gpu):
            continue
        kind = classify(e.name)
        if kind:
            counts[kind] += 1
        if on_gpu and e.name != "bench_step":  # not the GPU-side copy of our range
            busy += end - start
            last_gpu = max(last_gpu, end)
    syncs: Dict[str, float] = {kind: count / n for kind, count in counts.items()}
    syncs["total"] = sum(syncs[kind] for kind in SYNC_KINDS)
    span = last_gpu - first
    return {
        "syncs_per_step": syncs,
        "gpu_busy_pct": min(100.0, 100.0 * busy / span) if span > 0 else None,
    }


def cache_round(wl: Workload) -> Optional[List[Dict[str, int]]]:
    """What the cache of collection 0 did at each step of one round over the inputs
    (None: no cache, or no metrics on this tree).

    One step per input with the cache metrics of the library on: the unique keys looked
    up, how many were found, how many were inserted (the misses) and how many evictions
    that made (or keys that could not be placed), all over the tables of the
    collection.  The metrics are zeroed before each step, so a step that inserts
    nothing reports 0, not the numbers of the step before.  Recording waits for the
    device, so this runs after the timed windows.
    """
    import torch

    module = wl.modules[0]
    cache = getattr(module, "cache", None)
    if cache is None or not hasattr(module, "set_record_cache_metrics"):
        return None
    module.set_record_cache_metrics(True)
    try:
        rounds = []
        for _ in range(len(wl.inputs)):
            torch.cuda.synchronize()
            metrics = cache.cache_metrics  # a live host tensor on both trees
            if metrics is None:
                return None
            metrics.zero_()
            index = wl.count % len(wl.inputs)
            wl.step()
            torch.cuda.synchronize()
            keys, found, inserted, evicted = (int(v) for v in cache.cache_metrics[:4])
            rounds.append(
                dict(
                    input=index,
                    keys=keys,
                    found=found,
                    inserted=inserted,
                    evicted=evicted,
                )
            )
    finally:
        module.set_record_cache_metrics(False)
    return rounds


def hit_rate(res: Dict[str, Any]) -> Optional[float]:
    """The share of the unique keys that a cache found, over the steps of its round."""
    rounds = res.get("cache")
    if not rounds:
        return None
    return sum(r["found"] for r in rounds) / max(1, sum(r["keys"] for r in rounds))


def describe(cfg: Config, res: Dict[str, Any]) -> None:
    """Print what was built: layout, planned memory, key statistics, sync breakdown."""
    plan, keys, syncs = res["plan_gib"], res["keys"], res.get("syncs_per_step")
    print(f"layout {res['layout']} (--storage {cfg.storage}); memory in GiB:")
    print(
        f"  planned HBM {plan['hbm_total']:.1f} = values {plan['hbm_values']:.1f} + "
        f"metadata {plan['metadata']:.1f} + inputs and gradients {plan['data']:.1f}; "
        f"planned host {plan['host_values']:.1f}; GPU in use {res['gpu_used_gib']:.1f}"
    )
    print(
        f"keys per table per step: {keys['keys_per_table']}, "
        f"{keys['unique_per_input']:.0f} unique ({100 * keys['unique_ratio']:.0f}%); "
        f"the {cfg.inputs} inputs hold {keys['working_set']} distinct keys per table"
    )
    ratio = res["hbm_ratio"]
    if cfg.storage == "caching":
        wanted = int(ratio * cfg.num_embeddings)
        rows = -(-wanted // CACHE_BUCKET) * CACHE_BUCKET
        cells = "; ".join(
            f"input {r['input']}: {100 * r['found'] / max(1, r['keys']):.0f}% of "
            f"{r['keys']} found, {r['inserted']} inserted, {r['evicted']} evicted"
            for r in res.get("cache") or []
        )
        print(
            f"cache of {rows} rows per table (--hbm-ratio {ratio:.6f}: a multiple of "
            f"{CACHE_BUCKET} rows); collection 0, unique keys over its {cfg.tables} "
            f"tables, one step per input: {cells or 'metrics not available'}"
        )
    elif cfg.storage == "hybrid":
        rows = int(ratio * cfg.num_embeddings)
        where = (
            "all in HBM" if rows >= keys["working_set"] else "the rest in host memory"
        )
        print(
            f"HBM tier of about {rows} rows per table for {keys['working_set']} "
            f"distinct keys: {where}"
        )
    if syncs:
        kinds = " ".join(f"{kind}={syncs[kind]:.1f}" for kind in SYNC_KINDS)
        print(
            f"syncs/step: {syncs['total']:.1f} ({syncs['total'] / cfg.collections:.2f} "
            f"per collection): {kinds}; d2h_pageable={syncs['d2h_pageable']:.1f} "
            "(blocking copies into pageable memory, normally also counted above)"
        )


HEADER = (
    f"{'storage':<8}{'K':>4}{'B':>7} | {'wall ms/step':^26} | {'host ms/step':^26} | "
    f"{'host/wall':>9} | {'gpu busy':>8} | {'syncs/step (per coll)':>22} | "
    f"{'cache hit':>9}"
)


def row_text(res: Dict[str, Any]) -> str:
    """One table row: median [min, max] over the windows, per point."""
    cfg = res["config"]
    head = f"{cfg['storage']:<8}{cfg['collections']:>4}{cfg['batch_size']:>7}"
    if "error" in res:
        return f"{head} | FAILED: {res['error']}"

    def cell(m: Dict[str, Any]) -> str:
        return f"{m['median']:8.2f} [{m['min']:7.2f},{m['max']:7.2f}]"

    wall, host = res["wall_ms"], res["host_ms"]
    busy, syncs = res.get("gpu_busy_pct"), res.get("syncs_per_step")
    busy_cell = "-" if busy is None else f"{busy:.0f}%"
    sync_cell = "-"
    if syncs:
        sync_cell = f"{syncs['total']:.1f} ({syncs['total'] / cfg['collections']:.2f})"
    hit = hit_rate(res)
    hit_cell = "-" if hit is None else f"{100 * hit:.0f}%"
    return (
        f"{head} | {cell(wall)} | {cell(host)} | "
        f"{100 * host['median'] / wall['median']:8.0f}% | "
        f"{busy_cell:>8} | {sync_cell:>22} | {hit_cell:>9}"
    )


def load_results(path: str) -> Tuple[Dict[str, Any], Dict[Tuple, Dict[str, Any]]]:
    """Meta and the good points of a --json file, keyed by their configuration."""
    with open(path) as f:
        data = json.load(f)
    rows = {
        tuple(v for k, v in sorted(r["config"].items()) if k != "seed"): r
        for r in data["results"]
        if "error" not in r
    }
    return data.get("meta", {}), rows


def compare(path_a: str, path_b: str) -> int:
    """Print the per-config speedup of run B over run A (times A / times B)."""
    (meta_a, rows_a), (meta_b, rows_b) = load_results(path_a), load_results(path_b)
    for tag, path, meta in (("A", path_a, meta_a), ("B", path_b, meta_b)):
        dirty = " (dirty)" if meta.get("dirty") else ""
        print(
            f"{tag}: {path}  commit {str(meta.get('commit'))[:12]}{dirty}  "
            f"{meta.get('gpu')}, torch {meta.get('torch')}, "
            f"{meta.get('dynamicemb_path')}"
        )

    def order(key: Tuple) -> Tuple:
        cfg = rows_a[key]["config"]
        return cfg["storage"], cfg["collections"], cfg["batch_size"]

    shared = sorted(rows_a.keys() & rows_b.keys(), key=order)
    if not shared:
        print("no configuration appears in both files")
        return 1
    lead, abr = f"{'storage':<8}{'K':>4}{'B':>7}", f"{'A':>8} {'B':>8} {'A/B':>7}"
    print(f"{lead} | {'wall ms/step':^25} | {'host ms/step':^25} |")
    print(f"{'':19} | {abr} | {abr} | syncs/step A -> B | cache hit A -> B")
    speedups: Dict[str, List[float]] = {"wall_ms": [], "host_ms": []}
    for key in shared:
        a, b = rows_a[key], rows_b[key]
        cells = []
        for metric in speedups:
            ma, mb = a[metric]["median"], b[metric]["median"]
            speedups[metric].append(ma / mb)
            cells.append(f"{ma:8.2f} {mb:8.2f} {ma / mb:6.2f}x")
        sa, sb = a.get("syncs_per_step"), b.get("syncs_per_step")
        syncs = f"{sa['total']:.1f} -> {sb['total']:.1f}" if sa and sb else "-"
        ha, hb = hit_rate(a), hit_rate(b)
        hits = (
            f"{100 * ha:.0f}% -> {100 * hb:.0f}%"
            if ha is not None and hb is not None
            else "-"
        )
        cfg = a["config"]
        head = f"{cfg['storage']:<8}{cfg['collections']:>4}{cfg['batch_size']:>7}"
        print(f"{head} | {cells[0]} | {cells[1]} | {syncs:<17} | {hits}")
    for metric, ratios in speedups.items():
        mean = math.exp(statistics.fmean(math.log(r) for r in ratios))
        print(f"geometric-mean speedup of B over A, {metric}: {mean:.2f}x")
    if len(rows_a.keys() ^ rows_b.keys()):
        print("configurations present in only one file were skipped")
    return 0


def run_point(
    cfg: Config, args: argparse.Namespace, device: torch.device, verbose: bool
) -> Dict[str, Any]:
    """Build the workload for ``cfg``, measure it and return its result record."""
    import torch

    plan = plan_memory(cfg)
    free = torch.cuda.mem_get_info(device)[0] / GIB
    if plan["hbm_total"] > 0.9 * free:
        raise MemoryError(
            f"planned {plan['hbm_total']:.1f} GiB of HBM exceeds 90% of the "
            f"{free:.1f} GiB free: lower --collections, --num-embeddings or "
            "--hbm-ratio, or use --storage host"
        )
    torch.manual_seed(cfg.seed)  # also seeds the CUDA generators
    wl = Workload(cfg, device, verbose)
    warmup = getattr(args, "warmup", 20 if args.sweep else 30)
    steps = getattr(args, "steps", 100 if args.sweep else 200)
    res = {
        "config": asdict(cfg),
        "hbm_ratio": wl.hbm_ratio,  # as used: a caching layout sizes its own default
        "layout": wl.layout,
        "plan_gib": plan,
    }
    res.update(measure(wl, warmup, steps, args.windows))
    res.update(keys=key_stats(cfg, wl.inputs), warmup_steps=warmup)
    res["cache"] = cache_round(wl)
    if args.sync_count:
        res.update(profile_steps(wl.step, args.sync_steps))
    if args.profile_steps:  # for nsys --capture-range=cudaProfilerApi
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for _ in range(args.profile_steps):
            wl.step()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    free_now, total = torch.cuda.mem_get_info(device)
    res["gpu_used_gib"] = (total - free_now) / GIB
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    import torch

    if args.device is not None:
        torch.cuda.set_device(args.device)
    index = torch.cuda.current_device()
    if index > 9:  # BatchedDynamicEmbeddingTablesV2 reads str(device)[-1] as the index
        raise SystemExit(f"cuda:{index} is not supported: use a device index <= 9")
    device = torch.device(f"cuda:{index}")
    base = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    points = [base]
    if args.sweep:
        points = [
            replace(base, collections=k, batch_size=b)
            for k in SWEEP_COLLECTIONS
            for b in SWEEP_BATCH_SIZES
        ]
    meta = run_meta(device)
    dirty = " (dirty)" if meta.get("dirty") else ""
    print(f"{meta['gpu']}  torch {meta['torch']}  cuda {meta['cuda']}")
    print(f"dynamicemb {meta['dynamicemb_path']}  commit {meta.get('commit')}{dirty}")
    print(f"extension {meta['dynamicemb_extension']}")
    print(
        f"T={base.tables} H={base.hotness} D={base.dim} alpha={base.alpha} "
        f"rows/table={base.num_embeddings} optimizer={base.optimizer} inputs={base.inputs}"
    )
    if args.sweep:
        print(HEADER)
    baseline = torch.cuda.mem_get_info(device)[0]  # free bytes before any module
    results: List[Dict[str, Any]] = []
    for cfg in points:
        try:
            res = run_point(cfg, args, device, verbose=not args.sweep)
        except Exception as exc:  # a failed sweep point must not end the sweep
            if not args.sweep:
                raise
            res = {"config": asdict(cfg), "error": f"{type(exc).__name__}: {exc}"}
        results.append(res)
        if not args.sweep:  # after the library's memory table of collection 0
            print(HEADER)
        print(row_text(res), flush=True)
        if res.get("note"):
            print(f"note: {res['note']}")
        if not args.sweep:
            describe(cfg, res)
        gc.collect()  # free the modules (and their VMM memory) before the next point
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        held = (baseline - torch.cuda.mem_get_info(device)[0]) / GIB
        if held > 1:
            print(f"warning: {held:.1f} GiB of GPU memory not freed after this point")
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"meta": meta, "results": results}, f, indent=2, default=str)
    if args.json:
        print(f"results -> {args.json}")
    return 1 if any("error" in r for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
