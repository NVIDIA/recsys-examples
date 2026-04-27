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

import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest
import torch
import torchrec
from benchmark_utils import GPUTimer
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
    RecordCacheMetrics,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)

try:
    from fbgemm_gpu.runtime_monitor import StdLogStatsReporterConfig

    _HAS_STATS_REPORTER = True
except ImportError:
    _HAS_STATS_REPORTER = False


# ── Constants ────────────────────────────────────────────────────────────────

REPORT_INTERVAL = 10
WARMUP_ITERS = 5

GPU_PEAK_BW_GB_S = {
    "H100 SXM": 3350,
    "H100 NVL": 3350,
    "H100 PCIe": 2039,
    "H100": 2039,
    "H200": 4800,
    "A100 SXM": 2039,
    "A100 PCIe": 2039,
    "A100": 2039,
    "L40": 864,
    "V100": 900,
}


# ── Utility helpers ──────────────────────────────────────────────────────────


def get_emb_precision(s):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[s]


def get_fbgemm_precision(s):
    return {"fp32": SparseType.FP32, "fp16": SparseType.FP16, "bf16": SparseType.BF16}[
        s
    ]


_DYN_OPT = {
    "sgd": EmbOptimType.EXACT_SGD,
    "exact_sgd": EmbOptimType.EXACT_SGD,
    "adam": EmbOptimType.ADAM,
    "exact_adagrad": EmbOptimType.EXACT_ADAGRAD,
    "exact_row_wise_adagrad": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
}

_FBGEMM_OPT = {
    "sgd": OptimType.EXACT_SGD,
    "exact_sgd": OptimType.EXACT_SGD,
    "adam": OptimType.ADAM,
    "exact_adagrad": OptimType.EXACT_ADAGRAD,
    "exact_row_wise_adagrad": OptimType.EXACT_ROWWISE_ADAGRAD,
}

_DYN_POOL = {
    "none": DynamicEmbPoolingMode.NONE,
    "sum": DynamicEmbPoolingMode.SUM,
    "mean": DynamicEmbPoolingMode.MEAN,
}

_FBGEMM_POOL = {
    "none": PoolingMode.NONE,
    "sum": PoolingMode.SUM,
    "mean": PoolingMode.MEAN,
}

_OPT_STATE_DIM = {
    "sgd": lambda d: 0,
    "adam": lambda d: 2 * d,
    "exact_adagrad": lambda d: d,
    "exact_row_wise_adagrad": lambda d: 1,
}


def table_idx_to_name(i):
    return f"t_{i}"


def feature_idx_to_name(i):
    return f"cate_{i}"


def dtype_size(dt):
    return torch.tensor([], dtype=dt).element_size()


def get_peak_bandwidth():
    name = torch.cuda.get_device_name()
    best_match, best_len = None, 0
    for k, bw in GPU_PEAK_BW_GB_S.items():
        if k.lower() in name.lower() and len(k) > best_len:
            best_match, best_len = bw, len(k)
    return best_match


# ── BenchmarkConfig ──────────────────────────────────────────────────────────


@dataclass
class BenchmarkConfig:
    batch_size: int = 65536
    num_embeddings_per_feature: List[int] = field(
        default_factory=lambda: [24 * 1024 * 1024]
    )
    embedding_dim: int = 128
    optimizer_type: str = "adam"
    caching: bool = False
    cache_algorithm: str = "lru"
    gpu_ratio: float = 1.0
    hbm_for_embeddings: List[int] = field(
        default_factory=lambda: [36 * (1024**3)]
    )
    feature_distribution: str = "pow-law"
    alpha: float = 1.05
    pooling_mode: str = "none"
    max_hotness: int = 10
    num_iterations: int = 100
    emb_precision: str = "fp32"
    output_dtype: str = "fp32"
    use_index_dedup: bool = False
    learning_rate: float = 0.1
    eps: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    seed: int = 42

    @property
    def num_tables(self):
        return len(self.num_embeddings_per_feature)

    @property
    def value_dim(self):
        opt_fn = _OPT_STATE_DIM.get(self.optimizer_type, lambda d: 0)
        return self.embedding_dim + opt_fn(self.embedding_dim)

    @property
    def mode(self):
        if self.caching:
            return "caching"
        if self.gpu_ratio >= 1.0:
            return "gpu"
        if abs(self.gpu_ratio) <= 1e-3:
            return "no_hbm"
        return "no_caching"

    def label(self):
        caps = "_".join(
            f"{e // (1024 * 1024)}M" for e in self.num_embeddings_per_feature
        )
        return (
            f"T{self.num_tables}_B{self.batch_size}_D{self.embedding_dim}_"
            f"{self.optimizer_type}_{self.mode}_"
            f"pool={self.pooling_mode}_cap={caps}"
        )


# ── GPU-accelerated data generation ─────────────────────────────────────────


def generate_sparse_features_gpu(cfg: BenchmarkConfig, device: torch.device):
    """Batch-generate all sparse features on GPU.

    All random number generation happens in bulk GPU calls.  Only the final
    KJT construction loops in Python (unavoidable since KJT is a Python object).
    """
    num_tables = cfg.num_tables
    num_iters = cfg.num_iterations
    bs = cfg.batch_size
    feature_names = [feature_idx_to_name(i) for i in range(num_tables)]
    is_pooling = cfg.pooling_mode != "none"

    if is_pooling:
        all_lengths = torch.randint(
            1,
            cfg.max_hotness + 1,
            (num_iters, bs * num_tables),
            device=device,
            dtype=torch.int64,
        )
    else:
        all_lengths = torch.ones(
            num_iters, bs * num_tables, device=device, dtype=torch.int64
        )

    if cfg.feature_distribution == "random":
        total_vals = int(all_lengths.sum().item())
        all_values = torch.randint(
            0, (2**63) - 1, (total_vals,), device=device, dtype=torch.int64
        )
    elif cfg.feature_distribution in ("pow-law", "zipf"):
        from dataset_generator import PowerLaw, zipf

        per_table_lengths = all_lengths.view(num_iters, num_tables, bs)
        per_table_totals = per_table_lengths.sum(dim=(0, 2))

        per_table_vals = []
        for t in range(num_tables):
            n_samples = int(per_table_totals[t].item())
            cap = cfg.num_embeddings_per_feature[t]
            if cfg.feature_distribution == "pow-law":
                vals = PowerLaw(1, cap, cfg.alpha, n_samples, device)
            else:
                vals = zipf(0, cap, cfg.alpha, n_samples, device)
            per_table_vals.append(vals.to(torch.int64))

        per_table_iter_counts = per_table_lengths.sum(dim=2)
        per_table_offsets = []
        for t in range(num_tables):
            cs = torch.zeros(num_iters + 1, device=device, dtype=torch.long)
            torch.cumsum(per_table_iter_counts[:, t], dim=0, out=cs[1:])
            per_table_offsets.append(cs)

        total_vals = int(all_lengths.sum().item())
        all_values = torch.empty(total_vals, device=device, dtype=torch.int64)
        pos = 0
        for i in range(num_iters):
            for t in range(num_tables):
                s = int(per_table_offsets[t][i].item())
                e = int(per_table_offsets[t][i + 1].item())
                cnt = e - s
                all_values[pos : pos + cnt] = per_table_vals[t][s:e]
                pos += cnt
    else:
        raise ValueError(f"Unsupported distribution: {cfg.feature_distribution}")

    iter_counts = all_lengths.sum(dim=1)
    iter_offsets = torch.zeros(num_iters + 1, device=device, dtype=torch.long)
    torch.cumsum(iter_counts, dim=0, out=iter_offsets[1:])

    res = []
    for i in range(num_iters):
        s = int(iter_offsets[i].item())
        e = int(iter_offsets[i + 1].item())
        res.append(
            torchrec.KeyedJaggedTensor(
                keys=feature_names,
                values=all_values[s:e],
                lengths=all_lengths[i],
            )
        )
    return res


# ── Model creation ───────────────────────────────────────────────────────────


def is_hybrid_storage(cfg: BenchmarkConfig) -> bool:
    """HybridStorage: not pure caching, not full HBM, not zero HBM."""
    if cfg.caching:
        return False
    if abs(cfg.gpu_ratio - 1.0) <= 1e-3:
        return False
    if abs(cfg.gpu_ratio - 0.0) <= 1e-3:
        return False
    return True


def create_dynamic_embedding_tables(cfg: BenchmarkConfig, device: torch.device):
    table_options = []
    for i in range(cfg.num_tables):
        table_options.append(
            DynamicEmbTableOptions(
                index_type=torch.int64,
                embedding_dtype=get_emb_precision(cfg.emb_precision),
                dim=cfg.embedding_dim,
                max_capacity=cfg.num_embeddings_per_feature[i],
                local_hbm_for_values=cfg.hbm_for_embeddings[i],
                bucket_capacity=128,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.NORMAL,
                ),
                score_strategy=(
                    DynamicEmbScoreStrategy.LFU
                    if cfg.cache_algorithm == "lfu"
                    else DynamicEmbScoreStrategy.TIMESTAMP
                ),
                caching=cfg.caching,
            )
        )

    var = BatchedDynamicEmbeddingTablesV2(
        table_options=table_options,
        table_names=[table_idx_to_name(i) for i in range(cfg.num_tables)],
        use_index_dedup=cfg.use_index_dedup,
        pooling_mode=_DYN_POOL[cfg.pooling_mode],
        output_dtype=get_emb_precision(cfg.output_dtype),
        device=device,
        optimizer=_DYN_OPT[cfg.optimizer_type],
        learning_rate=cfg.learning_rate,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        beta1=cfg.beta1,
        beta2=cfg.beta2,
    )

    if is_hybrid_storage(cfg):
        storage = var.tables
        num_tables = cfg.num_tables
        optstate_dim = storage.value_dim(0) - storage.embedding_dim(0)
        initial_accumulator = storage.init_optimizer_state()
        value_dim = cfg.embedding_dim + optstate_dim
        max_num_embeddings = max(cfg.num_embeddings_per_feature)
        fill_batch = 1024 * 1024

        i = 0
        while i < max_num_embeddings:
            start = i
            end = min(i + fill_batch, max_num_embeddings)
            chunk = end - start
            i += fill_batch

            keys = torch.arange(start, end, device=device, dtype=torch.int64).repeat(num_tables)
            table_ids = torch.arange(num_tables, device=device, dtype=torch.int64).repeat_interleave(chunk)
            total = num_tables * chunk

            emb = torch.rand(total, cfg.embedding_dim, device=device, dtype=torch.float32)
            if optstate_dim > 0:
                opt = torch.rand(total, optstate_dim, device=device, dtype=torch.float32) * initial_accumulator
                values = torch.cat((emb, opt), dim=1).contiguous()
            else:
                values = emb

            scores = (
                torch.ones(total, dtype=torch.uint64, device=device)
                if cfg.cache_algorithm == "lfu"
                else None
            )
            storage.insert(keys, table_ids, values, scores)
    else:
        var.fill_tables()

    return var


def create_split_table_batched_embeddings(cfg: BenchmarkConfig, device: torch.device):
    optimizer = _FBGEMM_OPT[cfg.optimizer_type]
    D = cfg.embedding_dim
    Es = cfg.num_embeddings_per_feature
    cache_alg = (
        CacheAlgorithm.LRU if cfg.cache_algorithm == "lru" else CacheAlgorithm.LFU
    )
    pooling = _FBGEMM_POOL[cfg.pooling_mode]

    if cfg.caching:
        kwargs = {}
        if _HAS_STATS_REPORTER:
            kwargs["stats_reporter_config"] = StdLogStatsReporterConfig(
                REPORT_INTERVAL
            )
        emb = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (e, D, EmbeddingLocation.MANAGED_CACHING, ComputeDevice.CUDA)
                for e in Es
            ],
            optimizer=optimizer,
            weights_precision=get_fbgemm_precision(cfg.emb_precision),
            stochastic_rounding=False,
            cache_load_factor=cfg.gpu_ratio,
            cache_algorithm=cache_alg,
            pooling_mode=pooling,
            output_dtype=get_fbgemm_precision(cfg.output_dtype),
            device=device,
            learning_rate=cfg.learning_rate,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
            bounds_check_mode=BoundsCheckMode.NONE,
            record_cache_metrics=RecordCacheMetrics(True, False),
            **kwargs,
        ).cuda()
    else:
        loc = (
            EmbeddingLocation.MANAGED
            if abs(cfg.gpu_ratio - 1.0) > 1e-3
            else EmbeddingLocation.DEVICE
        )
        emb = SplitTableBatchedEmbeddingBagsCodegen(
            [(e, D, loc, ComputeDevice.CUDA) for e in Es],
            optimizer=optimizer,
            weights_precision=get_fbgemm_precision(cfg.emb_precision),
            stochastic_rounding=False,
            pooling_mode=pooling,
            output_dtype=get_fbgemm_precision(cfg.output_dtype),
            device=device,
            learning_rate=cfg.learning_rate,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
            bounds_check_mode=BoundsCheckMode.NONE,
        ).cuda()
    return emb


# ── Benchmark execution ──────────────────────────────────────────────────────


def benchmark_train_eval(model, sparse_features, num_iterations):
    """Measure train / eval latencies (ms per iteration) using CUDA Events."""
    model.train()

    start_event = torch.cuda.Event(enable_timing=True)
    mid_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_forward_ms = 0.0
    total_backward_ms = 0.0
    total_iter_ms = 0.0

    for i in range(num_iterations):
        sf = sparse_features[i]
        torch.cuda.nvtx.range_push(f"train_iter_{i}")
        torch.cuda.nvtx.range_push("forward")
        start_event.record()
        output = model(sf.values(), sf.offsets())
        mid_event.record()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("backward")
        grad = torch.empty_like(output)
        output.backward(grad)
        end_event.record()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        total_forward_ms += start_event.elapsed_time(mid_event)
        total_backward_ms += mid_event.elapsed_time(end_event)
        total_iter_ms += start_event.elapsed_time(end_event)

    train_ms = total_iter_ms / num_iterations
    fwd_ms = total_forward_ms / num_iterations
    bwd_ms = total_backward_ms / num_iterations

    model.eval()
    eval_start = torch.cuda.Event(enable_timing=True)
    eval_end = torch.cuda.Event(enable_timing=True)
    total_eval_ms = 0.0
    for i in range(num_iterations):
        sf = sparse_features[i]
        eval_start.record()
        output = model(sf.values(), sf.offsets())
        eval_end.record()
        torch.cuda.synchronize()
        total_eval_ms += eval_start.elapsed_time(eval_end)

    eval_ms = total_eval_ms / num_iterations

    return {
        "train_ms": train_ms,
        "forward_ms": fwd_ms,
        "backward_ms": bwd_ms,
        "eval_ms": eval_ms,
    }


# ── Torch profiler integration ──────────────────────────────────────────────


def benchmark_with_torch_profiler(
    model, sparse_features, num_iterations, trace_prefix=""
):
    """Run benchmark under torch.profiler; export trace and return profiler."""
    from torch.profiler import ProfilerActivity, profile, schedule

    model.train()

    n_warm = min(WARMUP_ITERS, num_iterations)
    for i in range(n_warm):
        sf = sparse_features[i]
        output = model(sf.values(), sf.offsets())
        grad = torch.empty_like(output)
        output.backward(grad)
    torch.cuda.synchronize()

    if num_iterations >= 8:
        wait, warmup, active = 1, 2, num_iterations - 3
    else:
        wait, warmup, active = 0, 1, max(1, num_iterations - 1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for i in range(num_iterations):
            sf = sparse_features[i]
            torch.cuda.nvtx.range_push(f"iter_{i}")
            torch.cuda.nvtx.range_push("forward")
            output = model(sf.values(), sf.offsets())
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("backward")
            grad = torch.empty_like(output)
            output.backward(grad)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
            prof.step()

    trace_file = f"{trace_prefix}trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"  Chrome trace -> {trace_file}")
    print(
        prof.key_averages().table(sort_by="device_time_total", row_limit=40)
    )
    return prof


# ── Kernel pattern definitions ────────────────────────────────────────────────


KERNEL_NAME_PATTERNS = {
    "load_from_flat": [
        "load_from_flat_table_kernel",
        "load_from_flat_table", "load_from_flat",
    ],
    "store_to_flat": [
        "store_to_flat_table_kernel",
        "store_to_flat_table", "store_to_flat",
    ],
    "gather_embedding": [
        "one_to_one_warp",
        "forwardsequencefusedcopy", "forwardpooledfusedcopy",
        "gather_embedding",
    ],
    "reduce_grads": [
        "multi_to_one_reduce",
        "reduce_grads",
    ],
    "optimizer_update": [
        "update4_with_index_flat_table",
        "update_with_index_flat_table",
        "vecoptimizer",
        "sgd_update", "adam_update",
        "adagrad_update", "rowwise_adagrad",
        "update_for_flat_table", "update_for_padded_buffer",
    ],
    "segmented_unique": ["segmented_unique"],
    "hash_find": ["lookup", "find_kernel", "_find"],
    "hash_insert": ["insert_and_evict", "insert_kernel"],
}


# ── NCU (Nsight Compute) profiler integration ────────────────────────────────


def benchmark_with_ncu(model, sparse_features):
    """Run a single train iteration for NCU profiling.

    Meant to be launched externally under ``ncu --profile-from-start off ...``.
    Uses cudaProfilerStart/Stop to capture only the benchmark iteration,
    excluding setup kernels (embedding creation, data gen, segmented_unique, etc.).
    """
    model.train()

    sf = sparse_features[0]
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("ncu_iter")
    torch.cuda.nvtx.range_push("forward")
    output = model(sf.values(), sf.offsets())
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("backward")
    grad = torch.empty_like(output)
    output.backward(grad)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print("  NCU profiled iteration complete (1 fwd+bwd).")


def print_ncu_command(cfg: BenchmarkConfig):
    """Print the ncu command for profiling this config to stdout."""
    label = cfg.label()

    all_patterns: list[str] = []
    for patterns in KERNEL_NAME_PATTERNS.values():
        all_patterns.extend(patterns)
    regex = "|".join(f".*{p}.*" for p in all_patterns)

    output_file = os.path.join(os.getcwd(), f"ncu_{label}")
    k_parts = label.split("=")
    k_filter = " and ".join(k_parts)
    inner_cmd = (
        f"bash benchmark/benchmark_batched_dynamicemb_tables.sh"
        f" --profile ncu-run -k '{k_filter}'"
    )
    ncu_cmd = (
        f"ncu -f --target-processes all"
        f" --profile-from-start off"
        f" --kernel-name 'regex:{regex}'"
        f" --set full"
        f" --csv --page raw"
        f" -o {output_file}"
        f" {inner_cmd}"
    )
    print(ncu_cmd)


# ── Pre-compute N_unique via segmented_unique ────────────────────────────────


def precompute_unique_counts(sparse_features, num_tables, device):
    """Return list of N_unique per iteration (cheap GPU operation)."""
    from dynamicemb_extensions import (
        expand_table_ids_cuda,
        get_table_range,
        segmented_unique_cuda,
    )

    feature_offsets = torch.arange(num_tables + 1, device=device, dtype=torch.int64)
    counts = []
    for kjt in sparse_features:
        indices = kjt.values()
        offsets = kjt.offsets()
        table_range = get_table_range(offsets, feature_offsets)
        num_uniques, _, _, _, _, _, _ = segmented_unique_cuda(
            indices, table_range, num_tables, None
        )
        counts.append(num_uniques.item())
    return counts


# ── Bandwidth computation ────────────────────────────────────────────────────


def get_kernel_patterns(cfg: BenchmarkConfig, avg_n_unique, avg_n_total):
    """Return kernel-group dict with 'patterns' and 'bytes' per group."""
    emb_dim = cfg.embedding_dim
    elem = dtype_size(get_emb_precision(cfg.emb_precision))
    out_elem = dtype_size(get_emb_precision(cfg.output_dtype))
    vdim = cfg.value_dim
    bs = cfg.batch_size
    total_D = emb_dim * cfg.num_tables
    is_pooling = cfg.pooling_mode != "none"
    Nu = avg_n_unique
    Nt = avg_n_total

    byte_counts = {
        "load_from_flat": Nu * emb_dim * elem,
        "store_to_flat": Nu * vdim * elem,
        "gather_embedding": (
            (Nu * emb_dim * elem + bs * total_D * out_elem)
            if is_pooling
            else (Nu + Nt) * emb_dim * out_elem
        ),
        "reduce_grads": (Nt + Nu) * emb_dim * elem,
        "optimizer_update": Nu * (emb_dim + 2 * vdim) * elem,
        "segmented_unique": (2 * Nt + Nu) * 8,
        "hash_find": Nu * 16,
        "hash_insert": Nu * 32,
    }

    return {
        name: {"patterns": KERNEL_NAME_PATTERNS[name], "bytes": byte_counts[name]}
        for name in KERNEL_NAME_PATTERNS
    }


def compute_bandwidth_report(prof, avg_n_unique, avg_n_total, cfg: BenchmarkConfig):
    """Match profiler kernel events to known ops and compute achieved BW."""
    kernels = get_kernel_patterns(cfg, avg_n_unique, avg_n_total)

    peak_bw = get_peak_bandwidth()
    events = prof.key_averages()
    rows = []
    for name, info in kernels.items():
        matched = [
            e
            for e in events
            if e.self_device_time_total > 0
            and any(p in e.key.lower() for p in info["patterns"])
        ]
        if not matched:
            continue
        avg_us = sum(e.device_time_total / e.count for e in matched if e.count > 0)
        if avg_us <= 0:
            continue
        data_bytes = info["bytes"]
        bw = (data_bytes / 1e9) / (avg_us / 1e6)
        pct = f"{100 * bw / peak_bw:.1f}%" if peak_bw else "N/A"
        rows.append(
            {
                "kernel": name,
                "avg_time_us": avg_us,
                "data_mb": data_bytes / 1e6,
                "bw_gb_s": bw,
                "pct_peak": pct,
            }
        )
    return rows


# ── Summary tables ───────────────────────────────────────────────────────────


def _fmt(val, width):
    """Right-align a string to *width*."""
    return f"{val:>{width}}"


def format_summary_table(results):
    if not results:
        return "No results."
    cols = [
        ("label", 50),
        ("T", 3),
        ("batch", 9),
        ("optim", 8),
        ("cch", 3),
        ("pool", 4),
        ("dyn_fwd", 9),
        ("dyn_bwd", 9),
        ("dyn_trn", 9),
        ("dyn_evl", 9),
        ("trc_fwd", 9),
        ("trc_bwd", 9),
        ("trc_trn", 9),
        ("trc_evl", 9),
    ]
    header = " | ".join(_fmt(n, w) for n, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)
    lines = [header, sep]

    for r in results:
        vals = [
            (r.get("label", "")[:50], 50),
            (str(r.get("num_tables", "")), 3),
            (str(r.get("batch_size", "")), 9),
            (r.get("optimizer_type", ""), 8),
            ("Y" if r.get("caching") else "N", 3),
            (r.get("pooling_mode", ""), 4),
            (f"{r.get('dyn_forward_ms', 0):.3f}", 9),
            (f"{r.get('dyn_backward_ms', 0):.3f}", 9),
            (f"{r.get('dyn_train_ms', 0):.3f}", 9),
            (f"{r.get('dyn_eval_ms', 0):.3f}", 9),
            (f"{r.get('trc_forward_ms', 0):.3f}", 9),
            (f"{r.get('trc_backward_ms', 0):.3f}", 9),
            (f"{r.get('trc_train_ms', 0):.3f}", 9),
            (f"{r.get('trc_eval_ms', 0):.3f}", 9),
        ]
        lines.append(" | ".join(_fmt(v, w) for v, w in vals))
    return "\n".join(lines)


def format_bandwidth_table(rows):
    if not rows:
        return "  (no matching kernels found -- inspect full profiler output above)"
    cols = [
        ("kernel", 22),
        ("avg_us", 10),
        ("data_MB", 10),
        ("BW_GB/s", 10),
        ("%peak", 8),
    ]
    header = " | ".join(_fmt(n, w) for n, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)
    lines = [header, sep]
    for r in rows:
        lines.append(
            " | ".join(
                [
                    _fmt(r["kernel"], 22),
                    _fmt(f"{r['avg_time_us']:.1f}", 10),
                    _fmt(f"{r['data_mb']:.2f}", 10),
                    _fmt(f"{r['bw_gb_s']:.1f}", 10),
                    _fmt(r["pct_peak"], 8),
                ]
            )
        )
    return "\n".join(lines)


def write_results(results, json_path=None, csv_path=None):
    if json_path:
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4, default=str)
        print(f"Results -> {json_path}")
    if csv_path and results:
        flat = []
        for r in results:
            row = {k: v for k, v in r.items() if k != "bandwidth"}
            flat.append(row)
        keys = list(flat[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(flat)
        print(f"Results -> {csv_path}")


# ── Single benchmark run ─────────────────────────────────────────────────────


def run_single_benchmark(
    cfg: BenchmarkConfig,
    device: torch.device,
    timer: GPUTimer,
    profile_mode: Optional[str] = None,
) -> Dict[str, Any]:
    print(f"\n{'=' * 80}")
    print(f"Config: {cfg.label()}")
    print(f"{'=' * 80}")

    if profile_mode == "ncu-gen":
        print_ncu_command(cfg)
        return {"label": cfg.label(), "ncu_gen": True}

    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.empty_cache()

    timer.start()
    dynamic_emb = create_dynamic_embedding_tables(cfg, device)
    timer.stop()
    print(f"  DynamicEmb created in {timer.elapsed_time() / 1000:.3f} s")

    timer.start()
    torchrec_emb = create_split_table_batched_embeddings(cfg, device)
    timer.stop()
    print(f"  TorchRec created in {timer.elapsed_time() / 1000:.3f} s")

    timer.start()
    sparse_features = generate_sparse_features_gpu(cfg, device)
    timer.stop()
    print(f"  Data generated in {timer.elapsed_time() / 1000:.3f} s")

    unique_counts = precompute_unique_counts(sparse_features, cfg.num_tables, device)
    avg_n_unique = sum(unique_counts) / len(unique_counts)
    avg_n_total = sum(sf.values().numel() for sf in sparse_features) / len(
        sparse_features
    )
    print(f"  Avg N_unique={avg_n_unique:.0f}  Avg N_total={avg_n_total:.0f}")

    if cfg.caching:
        dynamic_emb.set_record_cache_metrics(True)
        # Not resetting cache states before benchmark to measure hit rate on warm cache
        # dynamic_emb.reset_cache_states()
        # torchrec_emb.reset_cache_states()

    bw_results: List[Dict] = []
    if profile_mode == "torch":
        print("\n  >> DynamicEmb profiler run")
        prof = benchmark_with_torch_profiler(
            dynamic_emb,
            sparse_features,
            cfg.num_iterations,
            trace_prefix=f"dynamicemb_{cfg.label()}_",
        )
        bw_results = compute_bandwidth_report(prof, avg_n_unique, avg_n_total, cfg)

        print("\n  >> TorchRec profiler run")
        benchmark_with_torch_profiler(
            torchrec_emb,
            sparse_features,
            cfg.num_iterations,
            trace_prefix=f"torchrec_{cfg.label()}_",
        )

        if cfg.caching:
            dynamic_emb.reset_cache_states()
            torchrec_emb.reset_cache_states()
    elif profile_mode == "nsys":
        print("  (NVTX annotations active -- run under nsys profile)")
    elif profile_mode == "ncu-run":
        benchmark_with_ncu(dynamic_emb, sparse_features)
        del dynamic_emb, torchrec_emb, sparse_features
        torch.cuda.empty_cache()
        return {"label": cfg.label(), "ncu_run": True}

    if cfg.caching:
        dynamic_emb.set_record_cache_metrics(False)
        dynamic_emb.reset_cache_states()
        torchrec_emb.reset_cache_states()

    dyn = benchmark_train_eval(dynamic_emb, sparse_features, cfg.num_iterations)
    trc = benchmark_train_eval(torchrec_emb, sparse_features, cfg.num_iterations)

    result = {
        "label": cfg.label(),
        "num_tables": cfg.num_tables,
        "batch_size": cfg.batch_size,
        "embedding_dim": cfg.embedding_dim,
        "optimizer_type": cfg.optimizer_type,
        "caching": cfg.caching,
        "pooling_mode": cfg.pooling_mode,
        "num_embeddings_per_feature": cfg.num_embeddings_per_feature,
        "feature_distribution": cfg.feature_distribution,
        "avg_n_unique": avg_n_unique,
        "avg_n_total": avg_n_total,
        "dyn_forward_ms": dyn["forward_ms"],
        "dyn_backward_ms": dyn["backward_ms"],
        "dyn_train_ms": dyn["train_ms"],
        "dyn_eval_ms": dyn["eval_ms"],
        "trc_forward_ms": trc["forward_ms"],
        "trc_backward_ms": trc["backward_ms"],
        "trc_train_ms": trc["train_ms"],
        "trc_eval_ms": trc["eval_ms"],
    }
    if bw_results:
        result["bandwidth"] = bw_results

    print(
        f"\n  DynamicEmb  train={dyn['train_ms']:.3f}  fwd={dyn['forward_ms']:.3f}"
        f"  bwd={dyn['backward_ms']:.3f}  eval={dyn['eval_ms']:.3f} ms"
    )
    print(
        f"  TorchRec    train={trc['train_ms']:.3f}  fwd={trc['forward_ms']:.3f}"
        f"  bwd={trc['backward_ms']:.3f}  eval={trc['eval_ms']:.3f} ms"
    )
    if bw_results:
        print("\n  Bandwidth (DynamicEmb):")
        print(format_bandwidth_table(bw_results))

    del dynamic_emb, torchrec_emb, sparse_features
    torch.cuda.empty_cache()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Test configuration and suites
# ═══════════════════════════════════════════════════════════════════════════════

_NUM_TABLES = 10
_CAP_PER_TABLE = 1 * 1024 * 1024  # 1M entries
_CAPS = [_CAP_PER_TABLE] * _NUM_TABLES
_DIM = 128

_BATCH_SIZES = [65536]
# _OPTIMIZERS = ["adam", "sgd"]
# _POOLING_MODES = ["none", "sum"]
_OPTIMIZERS = ["sgd"]
_POOLING_MODES = ["none"]


def _cache_hbm(gpu_ratio, cap_per_table, dim, optimizer_type):
    """HBM for caching mode: gpu_ratio fraction of the full table per table."""
    opt_fn = _OPT_STATE_DIM.get(optimizer_type, lambda d: 0)
    value_dim = dim + opt_fn(dim)
    per_table = int(gpu_ratio * cap_per_table * value_dim * 4)
    return [per_table] * _NUM_TABLES


def _gpu_configs():
    hbm = [sys.maxsize] * _NUM_TABLES
    return [
        BenchmarkConfig(
            batch_size=bs,
            num_embeddings_per_feature=_CAPS,
            embedding_dim=_DIM,
            hbm_for_embeddings=hbm,
            optimizer_type=opt,
            caching=False,
            gpu_ratio=1.0,
            pooling_mode=pool,
            max_hotness=10,
        )
        for bs in _BATCH_SIZES
        for opt in _OPTIMIZERS
        for pool in _POOLING_MODES
    ]


_CACHE_GPU_RATIO = 0.1

def _caching_configs():
    return [
        BenchmarkConfig(
            batch_size=bs,
            num_embeddings_per_feature=_CAPS,
            embedding_dim=_DIM,
            hbm_for_embeddings=_cache_hbm(_CACHE_GPU_RATIO, _CAP_PER_TABLE, _DIM, opt),
            optimizer_type=opt,
            caching=True,
            cache_algorithm="lru",
            gpu_ratio=_CACHE_GPU_RATIO,
            pooling_mode=pool,
            max_hotness=10,
        )
        for bs in _BATCH_SIZES
        for opt in _OPTIMIZERS
        for pool in _POOLING_MODES
    ]


def _no_caching_configs():
    hbm = [0] * _NUM_TABLES
    return [
        BenchmarkConfig(
            batch_size=bs,
            num_embeddings_per_feature=_CAPS,
            embedding_dim=_DIM,
            hbm_for_embeddings=hbm,
            optimizer_type=opt,
            caching=False,
            gpu_ratio=0.1,
            pooling_mode=pool,
            max_hotness=10,
        )
        for bs in _BATCH_SIZES
        for opt in _OPTIMIZERS
        for pool in _POOLING_MODES
    ]


def _no_hbm_configs():
    """No HBM, no caching: all embedding data in system memory (UVM)."""
    hbm = [0] * _NUM_TABLES
    return [
        BenchmarkConfig(
            batch_size=bs,
            num_embeddings_per_feature=_CAPS,
            embedding_dim=_DIM,
            hbm_for_embeddings=hbm,
            optimizer_type=opt,
            caching=False,
            gpu_ratio=0.0,
            pooling_mode=pool,
            max_hotness=10,
        )
        for bs in _BATCH_SIZES
        for opt in _OPTIMIZERS
        for pool in _POOLING_MODES
    ]


# ── Test suites ───────────────────────────────────────────────────────────────


class TestGpu:
    @pytest.mark.parametrize("cfg", _gpu_configs(), ids=lambda c: c.label())
    def test_gpu(self, cfg, device, timer, profile_mode):
        result = run_single_benchmark(cfg, device, timer, profile_mode)
        assert "error" not in result


class TestCaching:
    @pytest.mark.parametrize("cfg", _caching_configs(), ids=lambda c: c.label())
    def test_caching(self, cfg, device, timer, profile_mode):
        result = run_single_benchmark(cfg, device, timer, profile_mode)
        assert "error" not in result


class TestNoCaching:
    @pytest.mark.parametrize("cfg", _no_caching_configs(), ids=lambda c: c.label())
    def test_no_caching(self, cfg, device, timer, profile_mode):
        result = run_single_benchmark(cfg, device, timer, profile_mode)
        assert "error" not in result


class TestNoHbm:
    @pytest.mark.parametrize("cfg", _no_hbm_configs(), ids=lambda c: c.label())
    def test_no_hbm(self, cfg, device, timer, profile_mode):
        result = run_single_benchmark(cfg, device, timer, profile_mode)
        assert "error" not in result
