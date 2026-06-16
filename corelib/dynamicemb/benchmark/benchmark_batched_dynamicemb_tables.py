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
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest
import torch
from benchmark_utils import GPUTimer
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
    get_table_value_bytes,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.optimizer import get_optimizer_state_dim
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

import torchrec
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.types import DataType

try:
    from fbgemm_gpu.runtime_monitor import StdLogStatsReporterConfig

    _HAS_STATS_REPORTER = True
except ImportError:
    _HAS_STATS_REPORTER = False


# ── Constants ────────────────────────────────────────────────────────────────

REPORT_INTERVAL = 10
WARMUP_ITERS = 5
RESULTS_FILE = os.environ.get("BENCHMARK_RESULTS_FILE", "benchmark_results.json")

# Chunk size for any storage.insert population call.  256K rows keeps the
# per-chunk workspace (init_w + opt_state + cat result) below ~400 MiB even
# for Adam (D + 2D layout), which matters on H200 where dyn+trc Adam tables
# already eat ~72 GiB of the 80 GiB HBM budget.
_INSERT_BATCH = 256 * 1024

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

_PRECISION_TO_DATATYPE = {
    "fp32": DataType.FP32,
    "fp16": DataType.FP16,
    "bf16": DataType.BF16,
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
    hbm_for_embeddings: List[int] = field(default_factory=lambda: [36 * (1024**3)])
    # When set together with ``caching=True``, the cache is re-sized at run
    # time to ``per_table_unique_count * cache_footprint_ratio`` rows
    # (per-table HBM bytes for DynamicEmb, pooled cache_load_factor for
    # FBGEMM).  Overrides ``hbm_for_embeddings`` / ``gpu_ratio`` set at
    # construction time.  ``None`` keeps the construction-time values.
    cache_footprint_ratio: Optional[float] = None
    feature_distribution: str = "pow-law"
    alpha: float = 1.05
    pooling_mode: str = "none"
    max_hotness: int = 10
    num_iterations: int = 100
    # Profiling-only: which iterations --profile ncu captures (raw spec string,
    # parsed by parse_ncu_iterations).  None = capture iter 0.  Not in cfg.label().
    ncu_iterations: Optional[str] = None
    # Profiling-only: user-supplied kernel-name regex for the generated ncu
    # command (--ncu-kernel-regex).  Required for --profile ncu-gen.
    ncu_kernel_regex: Optional[str] = None
    emb_precision: str = "fp32"
    output_dtype: str = "fp32"
    use_index_dedup: bool = False
    learning_rate: float = 0.1
    eps: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    seed: int = 42
    # When True the profile=none path runs a forward-output equality check
    # against the TorchRec/FBGEMM TBE baseline inside benchmark_train_eval.
    # Sampling is restricted to ``[0, cap/2)`` and DynamicEmb is populated by
    # mirroring TorchRec's ``[0, cap/2)`` slice (its constructor's random
    # init), so every lookup hits the same value on both backends.  The
    # default fill_tables / hybrid-storage population is skipped so the
    # mirrored values aren't overwritten.
    correctness: bool = False
    # Sparse-key sampling range: when set, each table draws indices from its
    # distribution over ``[.., sparse_key_range)`` instead of ``[.., per-table
    # cap)``, decoupling the duplicate rate from table capacity (a smaller range
    # yields more duplicates).  ``None`` keeps the per-table cap.
    sparse_key_range: Optional[int] = None
    # When False the TorchRec/FBGEMM TBE baseline is neither built nor run; only
    # DynamicEmb is exercised and the ``trc_*`` metrics are reported as None.
    run_torchrec: bool = True

    @property
    def num_tables(self):
        return len(self.num_embeddings_per_feature)

    @property
    def value_dim(self):
        dtype = get_emb_precision(self.emb_precision)
        optstate = get_optimizer_state_dim(
            _DYN_OPT[self.optimizer_type], self.embedding_dim, dtype
        )
        return self.embedding_dim + optstate

    @property
    def mode(self):
        if self.caching:
            return "caching"
        if self.gpu_ratio >= 1.0:
            return "gpu"
        if abs(self.gpu_ratio) <= 1e-3:
            return "no_hbm"
        return "no_caching"

    @property
    def total_batch_size(self):
        return self.batch_size * self.num_tables

    def label(self):
        # Uniform caps collapse to a single token (T{nt} already encodes the
        # count); heterogeneous caps fall back to underscore-joined per-table
        # values so the label remains lossless.  Sub-MB caps (e.g. 16K with many
        # tables) render in K rather than rounding to "0M".
        def _fmt_cap(e):
            if e >= 1024 * 1024 and e % (1024 * 1024) == 0:
                return f"{e // (1024 * 1024)}M"
            if e >= 1024 and e % 1024 == 0:
                return f"{e // 1024}K"
            return str(e)

        caps_fmt = [_fmt_cap(e) for e in self.num_embeddings_per_feature]
        caps = caps_fmt[0] if len(set(caps_fmt)) == 1 else "_".join(caps_fmt)
        s = (
            f"T{self.num_tables}_totalB{self.total_batch_size}_D{self.embedding_dim}_"
            f"{self.optimizer_type}_{self.mode}_"
            f"pool={self.pooling_mode}_cap={caps}"
        )
        if self.cache_footprint_ratio is not None:
            s += f"_cfr={self.cache_footprint_ratio}"
        return s


# ── GPU-accelerated data generation ─────────────────────────────────────────


def generate_sparse_features_gpu(
    cfg: BenchmarkConfig,
    device: torch.device,
    num_embeddings_override: Optional[List[int]] = None,
):
    """Batch-generate all sparse features on GPU.

    All random number generation happens in bulk GPU calls.  Only the final
    KJT construction loops in Python (unavoidable since KJT is a Python object).

    ``num_embeddings_override`` shrinks the per-table sampling range.  Used by
    the correctness path to sample indices from ``[0, cap/2)`` so every lookup
    hits the populated half of the table.
    """
    num_tables = cfg.num_tables
    num_iters = cfg.num_iterations
    bs = cfg.batch_size
    feature_names = [feature_idx_to_name(i) for i in range(num_tables)]
    is_pooling = cfg.pooling_mode != "none"
    caps = (
        num_embeddings_override
        if num_embeddings_override is not None
        else cfg.num_embeddings_per_feature
    )
    assert len(caps) == num_tables

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

    # Optional override of the per-table sampling range (decouples duplicate
    # rate from table capacity).  Applied on top of caps for every table.
    skr = cfg.sparse_key_range

    if cfg.feature_distribution == "random":
        total_vals = int(all_lengths.sum().item())
        hi = skr if skr is not None else (2**63) - 1
        all_values = torch.randint(
            0, hi, (total_vals,), device=device, dtype=torch.int64
        )
    elif cfg.feature_distribution in ("pow-law", "zipf"):
        from dataset_generator import PowerLaw, zipf

        per_table_lengths = all_lengths.view(num_iters, num_tables, bs)
        per_table_totals = per_table_lengths.sum(dim=(0, 2))

        per_table_vals = []
        for t in range(num_tables):
            n_samples = int(per_table_totals[t].item())
            cap = skr if skr is not None else caps[t]
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


def create_dynamic_embedding_tables(
    cfg: BenchmarkConfig, device: torch.device, populate: bool = True
):
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
        # Align with TorchRec TBE construction so numerical differences are
        # purely kernel-level: same rounding policy, same out-of-bounds policy.
        stochastic_rounding=False,
        bounds_check_mode=BoundsCheckMode.NONE,
    )

    if not populate:
        return var

    # Hybrid storage needs an explicit per-key populate; full-HBM, caching,
    # and zero-HBM modes use fill_tables() to stamp random keys into the
    # hash map (faster, leaves value slots zero so first lookup triggers
    # admit + initializer -- accepted for timing baselines).
    if is_hybrid_storage(cfg):
        storage = var.tables
        num_tables = cfg.num_tables
        optstate_dim = storage.value_dim(0) - storage.embedding_dim(0)
        initial_accumulator = storage.init_optimizer_state()
        caps_per_table = cfg.num_embeddings_per_feature
        max_num_embeddings = max(caps_per_table)

        i = 0
        while i < max_num_embeddings:
            start = i
            end_global = min(i + _INSERT_BATCH, max_num_embeddings)
            i += _INSERT_BATCH

            # Per-table end is clamped to that table's own capacity so a
            # heterogeneous cfg.num_embeddings_per_feature doesn't push
            # keys past a smaller table's cap (cur configs are uniform so
            # this is latent, but the function is general).
            keys_list = []
            tids_list = []
            for t in range(num_tables):
                cap_t = caps_per_table[t]
                if start >= cap_t:
                    continue
                end_t = min(end_global, cap_t)
                keys_list.append(
                    torch.arange(start, end_t, device=device, dtype=torch.int64)
                )
                tids_list.append(
                    torch.full((end_t - start,), t, dtype=torch.int64, device=device)
                )
            if not keys_list:
                break

            keys = torch.cat(keys_list)
            table_ids = torch.cat(tids_list)
            total = keys.numel()

            emb = torch.rand(
                total, cfg.embedding_dim, device=device, dtype=torch.float32
            )
            if optstate_dim > 0:
                opt = (
                    torch.rand(total, optstate_dim, device=device, dtype=torch.float32)
                    * initial_accumulator
                )
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
            kwargs["stats_reporter_config"] = StdLogStatsReporterConfig(REPORT_INTERVAL)
        emb = SplitTableBatchedEmbeddingBagsCodegen(
            [(e, D, EmbeddingLocation.MANAGED_CACHING, ComputeDevice.CUDA) for e in Es],
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


def _forward_allclose(out_dyn, out_trc, atol, rtol):
    """Memory-light replacement for torch.allclose on full-size embedding outs.

    Returns ``(passed, max_diff, mean_diff)``.  ``torch.allclose`` internally
    recomputes ``(a-b).abs()`` and ``rtol*|b|`` -- with 512 MiB outputs this
    stacked ~2.5 GiB of transients on top of the 72 GiB tables and tipped
    TestGpu adam over the 80 GiB H200 budget.  This version reuses ``diff``
    in-place and frees intermediates as soon as possible so the per-iter
    peak stays around 1 GiB.
    """
    with torch.no_grad():
        diff = (out_dyn - out_trc).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
        # Manual element-wise allclose: |a-b| <= atol + rtol*|b|.
        # diff <- |a-b| - rtol*|b|, then check max() <= atol.
        scaled = out_trc.abs().mul_(rtol)
        diff.sub_(scaled)
        del scaled
        passed = bool(diff.max().item() <= atol)
        del diff
    return passed, max_diff, mean_diff


def benchmark_train_eval(
    dynamic_emb,
    torchrec_emb,
    sparse_features,
    num_iterations,
    cfg: BenchmarkConfig,
    check_forward: bool = False,
):
    """Measure train / eval latencies for both backends using CUDA Events.

    When ``check_forward`` is True the per-iter dyn/trc forward outputs are
    compared with ``torch.allclose`` (precision-aware tolerance). The shared
    backward gradient is allocated once outside the timed loop and reused for
    every iter — ``backward`` does not mutate ``grad``, and using a valid
    (non-uninitialized) tensor keeps the optimizer-driven weight evolution
    deterministic across the two backends.
    """
    has_trc = torchrec_emb is not None
    dynamic_emb.train()
    if has_trc:
        torchrec_emb.train()

    # check_forward needs the baseline to compare against.
    check_forward = check_forward and has_trc
    atol, rtol = _CORRECTNESS_TOL.get(cfg.emb_precision, (1e-4, 1e-3))
    failures: List[Dict[str, Any]] = []

    dyn_fwd = dyn_bwd = dyn_total = 0.0
    trc_fwd = trc_bwd = trc_total = 0.0

    # CUDA events are reusable across record() calls; allocate once.
    dyn_s = torch.cuda.Event(enable_timing=True)
    dyn_m = torch.cuda.Event(enable_timing=True)
    dyn_e = torch.cuda.Event(enable_timing=True)
    trc_s = torch.cuda.Event(enable_timing=True)
    trc_m = torch.cuda.Event(enable_timing=True)
    trc_e = torch.cuda.Event(enable_timing=True)

    # grad shape is fixed across iters (pooling: [bs, D*nt]; non-pooling has
    # lengths=1 so [bs*nt, D]).  Allocate once so torch.randn_like doesn't
    # land inside dyn_bwd timing.
    device = sparse_features[0].values().device
    grad_dtype = get_emb_precision(cfg.output_dtype)
    if cfg.pooling_mode != "none":
        grad_shape = (cfg.batch_size, cfg.embedding_dim * cfg.num_tables)
    else:
        grad_shape = (cfg.batch_size * cfg.num_tables, cfg.embedding_dim)
    grad = torch.randn(grad_shape, device=device, dtype=grad_dtype)

    for i in range(num_iterations):
        sf = sparse_features[i]
        torch.cuda.nvtx.range_push(f"train_iter_{i}")

        # ── dyn ──
        torch.cuda.nvtx.range_push("dyn")
        dyn_s.record()
        out_dyn = dynamic_emb(sf.values(), sf.offsets())
        dyn_m.record()
        out_dyn.backward(grad)
        dyn_e.record()
        torch.cuda.nvtx.range_pop()

        # ── trc ──
        out_trc = None
        if has_trc:
            torch.cuda.nvtx.range_push("trc")
            trc_s.record()
            out_trc = torchrec_emb(sf.values(), sf.offsets())
            trc_m.record()
            out_trc.backward(grad)
            trc_e.record()
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        dyn_fwd += dyn_s.elapsed_time(dyn_m)
        dyn_bwd += dyn_m.elapsed_time(dyn_e)
        dyn_total += dyn_s.elapsed_time(dyn_e)
        if has_trc:
            trc_fwd += trc_s.elapsed_time(trc_m)
            trc_bwd += trc_m.elapsed_time(trc_e)
            trc_total += trc_s.elapsed_time(trc_e)

        if check_forward:
            passed, max_diff, mean_diff = _forward_allclose(
                out_dyn, out_trc, atol, rtol
            )
            if not passed:
                failures.append(
                    {
                        "phase": "train",
                        "iter": i,
                        "max_diff": max_diff,
                        "mean_diff": mean_diff,
                    }
                )

    dynamic_emb.eval()
    if has_trc:
        torchrec_emb.eval()
    dyn_eval = trc_eval = 0.0
    for i in range(num_iterations):
        sf = sparse_features[i]
        dyn_s.record()
        out_dyn = dynamic_emb(sf.values(), sf.offsets())
        dyn_e.record()
        torch.cuda.synchronize()
        dyn_eval += dyn_s.elapsed_time(dyn_e)

        out_trc = None
        if has_trc:
            trc_s.record()
            out_trc = torchrec_emb(sf.values(), sf.offsets())
            trc_e.record()
            torch.cuda.synchronize()
            trc_eval += trc_s.elapsed_time(trc_e)

        if check_forward:
            passed, max_diff, mean_diff = _forward_allclose(
                out_dyn, out_trc, atol, rtol
            )
            if not passed:
                failures.append(
                    {
                        "phase": "eval",
                        "iter": i,
                        "max_diff": max_diff,
                        "mean_diff": mean_diff,
                    }
                )

    if failures:
        raise AssertionError(f"forward mismatch: {failures}")

    return {
        "dyn_train_ms": dyn_total / num_iterations,
        "dyn_forward_ms": dyn_fwd / num_iterations,
        "dyn_backward_ms": dyn_bwd / num_iterations,
        "dyn_eval_ms": dyn_eval / num_iterations,
        "trc_train_ms": trc_total / num_iterations if has_trc else None,
        "trc_forward_ms": trc_fwd / num_iterations if has_trc else None,
        "trc_backward_ms": trc_bwd / num_iterations if has_trc else None,
        "trc_eval_ms": trc_eval / num_iterations if has_trc else None,
    }


# ── Per-iteration reporting ───────────────────────────────────────────────────


def benchmark_one_iteration(model, sparse_feature):
    start_event = torch.cuda.Event(enable_timing=True)
    mid_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    output = model(sparse_feature.values(), sparse_feature.offsets())
    mid_event.record()
    # Deterministic grad: empty_like is uninitialized and NaN/denormals here
    # would be written into the embedding table by the fused optimizer update,
    # then carried into the subsequent measured/profiled run.
    grad = torch.ones_like(output)
    output.backward(grad)
    end_event.record()

    torch.cuda.synchronize()
    return (
        start_event.elapsed_time(mid_event),
        mid_event.elapsed_time(end_event),
        start_event.elapsed_time(end_event),
    )


def run_reporting_loop(dynamic_emb, torchrec_emb, sparse_features, cfg):
    """Run num_iterations and print per-iteration latency and cache hit rate."""
    print("\n  >> Reporting run (per-iteration latency)")
    cache_miss_counter_trc = None

    for i in range(cfg.num_iterations):
        sf = sparse_features[i]
        fwd, bwd, total = benchmark_one_iteration(dynamic_emb, sf)
        cache_info = ""
        if cfg.caching:
            cache_metrics = dynamic_emb.cache.cache_metrics
            unique_num = cache_metrics[0].item()
            cache_hit = cache_metrics[1].item()
            hit_rate = cache_hit / unique_num if unique_num > 0 else 0.0
            cache_miss = unique_num - cache_hit
            cache_info = (
                f"  hit_rate={hit_rate:.4f} unique={unique_num} miss={int(cache_miss)}"
            )
        print(
            f"    dyn iter {i:3d}: fwd={fwd:.3f} bwd={bwd:.3f} total={total:.3f} ms{cache_info}"
        )

    if torchrec_emb is None:
        return

    print()
    for i in range(cfg.num_iterations):
        sf = sparse_features[i]
        fwd, bwd, total = benchmark_one_iteration(torchrec_emb, sf)
        cache_info = ""
        if cfg.caching:
            cnt = torchrec_emb.get_cache_miss_counter().clone()
            if cache_miss_counter_trc is not None:
                miss = int((cnt - cache_miss_counter_trc)[1].item())
            else:
                miss = 0
            cache_miss_counter_trc = cnt
            cache_info = f"  cache_miss={miss}"
        print(
            f"    trc iter {i:3d}: fwd={fwd:.3f} bwd={bwd:.3f} total={total:.3f} ms{cache_info}"
        )


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
        grad = torch.ones_like(output)
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
            grad = torch.ones_like(output)
            output.backward(grad)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
            prof.step()

    trace_file = f"{trace_prefix}trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"  Chrome trace -> {trace_file}")
    print(prof.key_averages().table(sort_by="device_time_total", row_limit=40))
    return prof


# ── Kernel pattern definitions ────────────────────────────────────────────────


KERNEL_NAME_PATTERNS = {
    "load_from_flat": [
        "load_from_flat_table_kernel",
        "load_from_flat_table",
        "load_from_flat",
    ],
    "store_to_flat": [
        "store_to_flat_table_kernel",
        "store_to_flat_table",
        "store_to_flat",
    ],
    "gather_embedding": [
        "one_to_one_warp",
        "forwardsequencefusedcopy",
        "forwardpooledfusedcopy",
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
        "sgd_update",
        "adam_update",
        "adagrad_update",
        "rowwise_adagrad",
        "update_for_flat_table",
        "update_for_padded_buffer",
    ],
    "segmented_unique": ["segmented_unique"],
    "hash_find": ["lookup", "find_kernel", "_find"],
    "hash_insert": ["insert_and_evict", "insert_kernel"],
}


# ── Nsight Systems profiler integration ──────────────────────────────────────


def benchmark_with_nsys(dynamic_emb, torchrec_emb, sparse_features, cfg):
    """Run fwd+bwd on both backends inside a cudaProfilerStart/Stop window.

    One Start/Stop pair per config.  When pytest runs N configs under
    ``--profile nsys`` this function is called N times, producing N
    independent windows in the same process.  How those windows show up
    in the trace depends on the nsys launcher flags:

    * ``--capture-range=cudaProfilerApi --capture-range-end=stop``
      (single-config) -- only the first window is recorded; use a
      ``-k <label>`` pytest filter to pick which config that is.
    * ``--capture-range=cudaProfilerApi --capture-range-end=repeat-shutdown``
      (multi-config) -- every window is recorded into the same .nsys-rep,
      the gaps between windows (setup, reporting loop, next-config build)
      are skipped.  The outermost NVTX range ``cfg.label()`` lets nsys-ui
      attribute each segment to its config.

    Setup kernels (table alloc, sparse-feature gen, populate) and
    ``run_reporting_loop`` warmup stay out of every window in both modes.

    Both backends share each iter so dyn and trc can be compared
    side-by-side in the same trace.  NVTX layout:
    ``<cfg.label()>`` -> ``nsys_iter_i`` -> ``dyn``/``trc`` -> ``forward``/``backward``.
    """
    has_trc = torchrec_emb is not None
    dynamic_emb.train()
    if has_trc:
        torchrec_emb.train()

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(cfg.label())
    for i, sf in enumerate(sparse_features):
        torch.cuda.nvtx.range_push(f"nsys_iter_{i}")

        torch.cuda.nvtx.range_push("dyn")
        torch.cuda.nvtx.range_push("forward")
        out_dyn = dynamic_emb(sf.values(), sf.offsets())
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("backward")
        # Deterministic grad (empty_like is uninitialized -> NaN/denormal noise).
        grad_dyn = torch.ones_like(out_dyn)
        out_dyn.backward(grad_dyn)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()  # dyn

        if has_trc:
            torch.cuda.nvtx.range_push("trc")
            torch.cuda.nvtx.range_push("forward")
            out_trc = torchrec_emb(sf.values(), sf.offsets())
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("backward")
            grad_trc = torch.ones_like(out_trc)
            out_trc.backward(grad_trc)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()  # trc

        torch.cuda.nvtx.range_pop()  # nsys_iter_i
    torch.cuda.nvtx.range_pop()  # cfg.label()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(f"  Nsys profiled {len(sparse_features)} iters (dyn + trc, fwd+bwd each).")


# ── NCU (Nsight Compute) profiler integration ────────────────────────────────


def parse_ncu_iterations(spec: Optional[str], num_iterations: int) -> set:
    """Parse --ncu-iterations into a set of 0-based iteration indices to profile.

    Two accepted forms:
      * comma-separated list, e.g. ``"0,3,7"``
      * Python-style slice ``begin:end:step`` (end exclusive, any part optional),
        e.g. ``":10"``, ``"5:"``, ``"::2"``, ``"2:20:3"``

    Indices are clamped to ``[0, num_iterations)``.  ``None`` means "not
    specified by the caller"; ``print_ncu_command`` treats it as iteration 0
    only (the default capture target).  Raises ValueError on malformed input.
    """
    if spec is None:
        return set(range(num_iterations))
    spec = spec.strip()
    if not spec:
        return set(range(num_iterations))

    if ":" in spec:
        parts = spec.split(":")
        if len(parts) > 3:
            raise ValueError(f"--ncu-iterations slice has too many ':' parts: {spec!r}")
        begin = int(parts[0]) if parts[0].strip() else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1].strip() else num_iterations
        step = int(parts[2]) if len(parts) > 2 and parts[2].strip() else 1
        if step <= 0:
            raise ValueError(f"--ncu-iterations step must be positive: {spec!r}")
        return {i for i in range(begin, end, step) if 0 <= i < num_iterations}

    selected = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        idx = int(tok)
        if 0 <= idx < num_iterations:
            selected.add(idx)
    return selected


def benchmark_with_ncu(model, sparse_features, cfg):
    """Run all iterations for NCU profiling under a single capture window.

    Meant to be launched externally under ``ncu --profile-from-start off
    ...``.  A *single* cudaProfilerStart/Stop pair wraps the whole loop (ncu
    reliably honors one cudaProfiler window), so setup kernels (table create,
    data gen, warmup) stay outside it.

    Every iteration runs and is tagged with its own ``iter_{i}`` NVTX range
    nested under ``ncu_iter``.  *Which* iterations are actually captured is
    decided in the ncu command itself: ``--ncu-iterations`` turns into per-iter
    ``--nvtx-include 'ncu_iter/iter_{i}/'`` filters (see print_ncu_command), so
    the unselected iterations still run here (advancing the table / keeping the
    cache warm) but their kernels are filtered out by NVTX.

    ncu is per-kernel-launch (not timeline) so multi-config "Just Works": one
    ncu command without ``-k`` profiles every config's iters into the same
    report.  The outer NVTX range ``cfg.label()`` attributes kernels per config.
    """
    model.train()

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(cfg.label())
    torch.cuda.nvtx.range_push("ncu_iter")
    for i, sf in enumerate(sparse_features):
        torch.cuda.nvtx.range_push(f"iter_{i}")
        torch.cuda.nvtx.range_push("forward")
        output = model(sf.values(), sf.offsets())
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("backward")
        # Deterministic grad: empty_like is uninitialized and may hold
        # NaN/denormals that propagate through reduce_grads -> fused_update and
        # add FP-counter / timing noise to the profiled kernels.
        grad = torch.ones_like(output)
        output.backward(grad)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()  # iter_i
    torch.cuda.nvtx.range_pop()  # ncu_iter
    torch.cuda.nvtx.range_pop()  # cfg.label()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(f"  NCU ran {len(sparse_features)} iters (capture scoped by --nvtx-include).")


def print_ncu_command(cfg: BenchmarkConfig):
    """Print two ncu commands to stdout: single-config and multi-config.

    Both wrap the same shell launcher; the difference is whether pytest
    is filtered to one config via ``-k`` or runs the whole suite.  ncu is
    per-kernel-launch so the multi-config form profiles every config's
    wrapped iter into the same report; the outermost NVTX range
    ``cfg.label()`` separates them.
    """
    label = cfg.label()

    # Optional kernel-name filter (--ncu-kernel-regex), emitted verbatim as
    # ncu's --kernel-name 'regex:...'.  If omitted, no name filter is emitted so
    # every kernel within the --nvtx-include scope is profiled -- the way to
    # capture all kernels of an op range (e.g. op:segmented_unique), since one
    # op launches several differently-named kernels that a name regex can't all
    # match.
    regex = cfg.ncu_kernel_regex
    kernel_part = f"--kernel-name 'regex:{regex}'" if regex else ""

    # NVTX-range filter: belt-and-suspenders to the cudaProfilerStart/Stop gate
    # (only kernels under ncu_iter are profiled, excluding anything that leaked
    # into the cudaProfiler window).  --ncu-iterations narrows this to specific
    # iterations by OR-ing one --nvtx-include per selected iter
    # (ncu_iter/iter_{i}/), instead of toggling the single cudaProfiler window.
    selected = parse_ncu_iterations(cfg.ncu_iterations, cfg.num_iterations)
    if cfg.ncu_iterations is None:
        # Default to iteration 0 only -- profiling every iteration with the full
        # metric set is prohibitively expensive.  Use --ncu-iterations to widen.
        nvtx_part = "--nvtx --nvtx-include 'ncu_iter/iter_0/'"
    elif selected:
        includes = " ".join(
            f"--nvtx-include 'ncu_iter/iter_{i}/'" for i in sorted(selected)
        )
        nvtx_part = f"--nvtx {includes}"
    else:
        print(
            f"# WARNING: --ncu-iterations={cfg.ncu_iterations!r} selects no "
            f"iteration in [0, {cfg.num_iterations}); defaulting to iter_0."
        )
        nvtx_part = "--nvtx --nvtx-include 'ncu_iter/iter_0/'"

    # pytest -k is an expression grammar that REJECTS '=', which cfg.label()
    # embeds (pool=none, cap=1M); passing the whole label fails at runtime with
    # "Wrong expression passed to '-k': ... unexpected character '='".  Split on
    # '=' and AND-join the remaining chunks into a valid -k expression that
    # still uniquely selects this config -- the leading chunk already encodes
    # tables/batch/dim/opt/mode, so the conjunction is unambiguous.  (The '='
    # is only a problem for -k; it is fine inside the -o report filename below.)
    k_expr = " and ".join(part for part in label.split("=") if part)
    single_inner = (
        f"bash benchmark/benchmark_batched_dynamicemb_tables.sh"
        f" --profile ncu -k '{k_expr}'"
    )
    single_out = os.path.join(os.getcwd(), f"ncu_{label}")
    single_cmd = (
        f"ncu -f --target-processes all"
        f" --profile-from-start off"
        f" {nvtx_part}"
        f" {kernel_part}"
        f" --set full"
        # Embed CUDA source into the report so the Source page works offline /
        # on a machine without the sources (requires -lineinfo at build time).
        f" --import-source=yes"
        f" --csv --page raw"
        f" -o {single_out}"
        f" {single_inner}"
    )

    multi_inner = "bash benchmark/benchmark_batched_dynamicemb_tables.sh --profile ncu"
    multi_out = os.path.join(os.getcwd(), "ncu_all_configs")
    multi_cmd = (
        f"ncu -f --target-processes all"
        f" --profile-from-start off"
        f" {kernel_part}"
        f" --set full"
        f" {nvtx_part}"
        # Embed CUDA source into the report (requires -lineinfo at build time).
        f" --import-source=yes"
        f" --csv --page raw"
        f" -o {multi_out}"
        f" {multi_inner}"
    )

    print(f"# single-config (this cfg only):\n{single_cmd}")
    print(f"\n# multi-config (every config of the suite into one report):\n{multi_cmd}")


# ── Pre-compute N_unique via segmented_unique ────────────────────────────────


def precompute_unique_counts(sparse_features, num_tables, device):
    """Return list of N_unique per iteration (cheap GPU operation)."""
    from dynamicemb_extensions import get_table_range, segmented_unique_cuda

    feature_offsets = torch.arange(num_tables + 1, device=device, dtype=torch.int64)
    counts = []
    for kjt in sparse_features:
        offsets = kjt.offsets()
        table_range = get_table_range(offsets, feature_offsets)
        # New dynamicemb API: second arg is per-table segment range, not the
        # expanded per-key table_ids; no expand_table_ids_cuda needed here.
        num_uniques, _, _, _, _ = segmented_unique_cuda(
            kjt.values(), table_range, num_tables, None
        )
        counts.append(int(num_uniques.item()))
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


def append_result(result):
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(result)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


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


# ── Correctness init alignment ───────────────────────────────────────────────

# torch.allclose tolerances picked per emb_precision.  fp32 gets a relatively
# loose atol because fused vs split kernels can reorder reductions; fp16/bf16
# loses much more precision per update, so we open the gates wider.
_CORRECTNESS_TOL = {
    "fp32": (1e-4, 1e-3),
    "fp16": (5e-2, 5e-2),
    "bf16": (5e-2, 5e-2),
}


def _per_table_unique_keys(sparse_features, num_tables, batch_size, device):
    """Return list of per-table unique key tensors (int64) across all iters."""
    per_table_chunks: List[List[torch.Tensor]] = [[] for _ in range(num_tables)]
    for kjt in sparse_features:
        vals = kjt.values()
        offs = kjt.offsets()
        for t in range(num_tables):
            s = int(offs[t * batch_size].item())
            e = int(offs[(t + 1) * batch_size].item())
            per_table_chunks[t].append(vals[s:e])

    unique_keys: List[torch.Tensor] = []
    for t in range(num_tables):
        chunks = per_table_chunks[t]
        if chunks:
            unique_keys.append(torch.unique(torch.cat(chunks)))
        else:
            unique_keys.append(torch.empty(0, dtype=torch.int64, device=device))
    return unique_keys


def _populate_correctness_tables(
    dynamic_emb, torchrec_emb, cfg: BenchmarkConfig, device
):
    """Mirror TorchRec's first-half weights into DynamicEmb.

    TorchRec already has random init from its constructor; we read its
    ``[0, cap/2)`` slice in ``_INSERT_BATCH``-sized chunks and ``insert`` the
    same ``(key, value)`` pairs into the dynamicemb storage, with
    optimizer-state slots zeroed to mirror the fused-optimizer default initial
    state.  Sparse features (generated with ``cap/2`` as the upper bound) only
    look up keys in this populated range, so every lookup hits the same value
    on both backends.
    """
    nt = cfg.num_tables
    D = cfg.embedding_dim
    storage = dynamic_emb.tables
    optstate_dim = storage.value_dim(0) - storage.embedding_dim(0)
    emb_dtype = get_emb_precision(cfg.emb_precision)

    trc_weights = torchrec_emb.split_embedding_weights()
    total = 0
    for t in range(nt):
        cap = cfg.num_embeddings_per_feature[t]
        half = cap // 2
        if half == 0:
            continue
        trc_w_t = trc_weights[t]
        assert (
            trc_w_t.shape[1] == D
        ), f"trc_weights[{t}] dim {trc_w_t.shape[1]} != cfg.embedding_dim {D}"

        for start in range(0, half, _INSERT_BATCH):
            end = min(start + _INSERT_BATCH, half)
            chunk = end - start
            keys = torch.arange(start, end, device=device, dtype=torch.int64)
            init_w = trc_w_t[start:end].to(device=device, dtype=torch.float32)

            if optstate_dim > 0:
                opt_zero = torch.zeros(
                    chunk, optstate_dim, device=device, dtype=torch.float32
                )
                values = torch.cat([init_w, opt_zero], dim=1).contiguous()
            else:
                values = init_w.contiguous()
            values = values.to(emb_dtype)

            table_ids = torch.full((chunk,), t, dtype=torch.int64, device=device)
            scores = (
                torch.ones(chunk, dtype=torch.uint64, device=device)
                if cfg.cache_algorithm == "lfu"
                else None
            )
            storage.insert(keys, table_ids, values, scores)
            total += chunk

    return total


def _resize_cache_to_footprint(
    cfg: BenchmarkConfig, per_table_unique_counts: List[int]
) -> None:
    """Resize the cache to hold ``footprint * cache_footprint_ratio`` rows.

    Mutates ``cfg.hbm_for_embeddings`` (per-table HBM bytes consumed by
    DynamicEmb when ``caching=True``) and ``cfg.gpu_ratio`` (FBGEMM's
    single global ``cache_load_factor``).  Per-row bytes use
    ``cfg.value_dim`` so the budget covers values *and* optimizer state.
    """
    ratio = cfg.cache_footprint_ratio
    elem = dtype_size(get_emb_precision(cfg.emb_precision))
    bytes_per_row = cfg.value_dim * elem

    cfg.hbm_for_embeddings = [
        int(uc * ratio * bytes_per_row) for uc in per_table_unique_counts
    ]

    total_cap = sum(cfg.num_embeddings_per_feature)
    total_target = sum(int(uc * ratio) for uc in per_table_unique_counts)
    cfg.gpu_ratio = (
        max(min(total_target / total_cap, 1.0), 0.0) if total_cap > 0 else 0.0
    )

    print(
        f"  cache_footprint_ratio={ratio}: "
        f"hbm_for_embeddings={cfg.hbm_for_embeddings} "
        f"gpu_ratio={cfg.gpu_ratio:.6f}"
    )


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

    # Profile modes capture/profile workloads, not validate them.  cap/2
    # sampling and the populate-then-reporting drift would yield a
    # misleading trace, so force-disable correctness whenever any profile
    # mode is active.  Warn if the user (cfg flag or --correctness CLI)
    # explicitly asked for it so the override is not silent.
    if profile_mode is not None and cfg.correctness:
        warnings.warn(
            f"profile_mode={profile_mode!r} is incompatible with "
            f"cfg.correctness=True; disabling correctness for this run.",
            UserWarning,
            stacklevel=2,
        )
        cfg.correctness = False

    # The correctness check compares against the TorchRec baseline, so it cannot
    # run with the baseline disabled.
    if cfg.correctness and not cfg.run_torchrec:
        warnings.warn(
            "cfg.correctness=True requires the TorchRec baseline; "
            "re-enabling run_torchrec for this run.",
            UserWarning,
            stacklevel=2,
        )
        cfg.run_torchrec = True

    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.empty_cache()

    # Sparse features are generated before table creation: when
    # cfg.cache_footprint_ratio is set we size the cache from the actual
    # unique-key footprint of the workload, which can only be known after
    # the features have been sampled.
    #
    # In correctness mode the sampler is restricted to the populated half
    # of each table so every lookup hits a key we mirrored from TorchRec
    # into DynamicEmb.  All other modes sample over the full cap.
    half_caps = (
        [c // 2 for c in cfg.num_embeddings_per_feature] if cfg.correctness else None
    )

    timer.start()
    sparse_features = generate_sparse_features_gpu(
        cfg, device, num_embeddings_override=half_caps
    )
    timer.stop()
    print(f"  Data generated in {timer.elapsed_time() / 1000:.3f} s")

    # Per-table count of distinct keys seen across all iterations -- the
    # workload's true HBM footprint.  Used to size the cache when
    # cfg.cache_footprint_ratio is set.
    per_table_unique_counts = [
        int(t.numel())
        for t in _per_table_unique_keys(
            sparse_features, cfg.num_tables, cfg.batch_size, device
        )
    ]
    print(f"  Per-table unique footprint: {per_table_unique_counts}")

    if cfg.caching and cfg.cache_footprint_ratio is not None:
        _resize_cache_to_footprint(cfg, per_table_unique_counts)

    timer.start()
    # When correctness is requested we skip the default fill so
    # _populate_correctness_tables can mirror TorchRec's first-half weights.
    dynamic_emb = create_dynamic_embedding_tables(
        cfg, device, populate=not cfg.correctness
    )
    timer.stop()
    print(f"  DynamicEmb created in {timer.elapsed_time() / 1000:.3f} s")

    if cfg.run_torchrec:
        timer.start()
        torchrec_emb = create_split_table_batched_embeddings(cfg, device)
        timer.stop()
        print(f"  TorchRec created in {timer.elapsed_time() / 1000:.3f} s")
    else:
        torchrec_emb = None
        print("  TorchRec baseline disabled (run_torchrec=False)")

    if cfg.correctness:
        # PowerLaw/zipf already produce values in [min, max-1] / [min, max),
        # so passing half_caps as the upper bound guarantees every index is
        # strictly less than cap/2 -- the populated range on DynamicEmb.
        n_keys = _populate_correctness_tables(dynamic_emb, torchrec_emb, cfg, device)
        print(
            f"  Correctness: mirrored {n_keys} keys "
            f"(TorchRec[:cap/2] -> DynamicEmb)"
        )

    unique_counts = precompute_unique_counts(sparse_features, cfg.num_tables, device)
    avg_n_unique = sum(unique_counts) / len(unique_counts)
    avg_n_total = sum(sf.values().numel() for sf in sparse_features) / len(
        sparse_features
    )
    print(f"  Avg N_unique={avg_n_unique:.0f}  Avg N_total={avg_n_total:.0f}")

    if cfg.caching:
        dynamic_emb.set_record_cache_metrics(True)

    # Warm up before timing/profiling for every mode except correctness.  The
    # reporting loop populates the cache (so measured runs see steady-state hit
    # rate instead of a cold start), warms the caching allocator, and triggers
    # first-launch autotune / lazy init.  Correctness is excluded because the
    # loop advances each backend with independent gradients, drifting their
    # weights apart and breaking the forward-output comparison.
    #
    # Profile windows are bounded separately (cudaProfilerStart/Stop inside
    # benchmark_with_nsys / benchmark_with_ncu, and torch.profiler's own
    # schedule), so this warmup is never captured.
    if not cfg.correctness:
        print("  (run_reporting_loop warmup)")
        run_reporting_loop(dynamic_emb, torchrec_emb, sparse_features, cfg)

    if cfg.caching:
        # Keep the warm cache contents/counters; only stop *recording* hit-rate
        # metrics so the measured/profiled run reflects steady state.
        dynamic_emb.set_record_cache_metrics(False)

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

        if torchrec_emb is not None:
            print("\n  >> TorchRec profiler run")
            benchmark_with_torch_profiler(
                torchrec_emb,
                sparse_features,
                cfg.num_iterations,
                trace_prefix=f"torchrec_{cfg.label()}_",
            )

        del dynamic_emb, torchrec_emb, sparse_features
        torch.cuda.empty_cache()
        result = {"label": cfg.label(), "torch_profile": True}
        if bw_results:
            result["bandwidth"] = bw_results
        return result
    elif profile_mode == "nsys":
        benchmark_with_nsys(dynamic_emb, torchrec_emb, sparse_features, cfg)
        del dynamic_emb, torchrec_emb, sparse_features
        torch.cuda.empty_cache()
        return {"label": cfg.label(), "nsys": True}
    elif profile_mode == "ncu":
        benchmark_with_ncu(dynamic_emb, sparse_features, cfg)
        del dynamic_emb, torchrec_emb, sparse_features
        torch.cuda.empty_cache()
        return {"label": cfg.label(), "ncu": True}

    metrics = benchmark_train_eval(
        dynamic_emb,
        torchrec_emb,
        sparse_features,
        cfg.num_iterations,
        cfg=cfg,
        check_forward=cfg.correctness,
    )

    result = {
        "label": cfg.label(),
        "gpu_name": torch.cuda.get_device_name(device),
        "num_tables": cfg.num_tables,
        "batch_size": cfg.batch_size,
        "embedding_dim": cfg.embedding_dim,
        "optimizer_type": cfg.optimizer_type,
        "caching": cfg.caching,
        "cache_footprint_ratio": cfg.cache_footprint_ratio,
        "pooling_mode": cfg.pooling_mode,
        "num_embeddings_per_feature": cfg.num_embeddings_per_feature,
        "feature_distribution": cfg.feature_distribution,
        "alpha": cfg.alpha,
        "max_hotness": cfg.max_hotness,
        "avg_n_unique": avg_n_unique,
        "avg_n_total": avg_n_total,
        "dyn_forward_ms": metrics["dyn_forward_ms"],
        "dyn_backward_ms": metrics["dyn_backward_ms"],
        "dyn_train_ms": metrics["dyn_train_ms"],
        "dyn_eval_ms": metrics["dyn_eval_ms"],
        "trc_forward_ms": metrics["trc_forward_ms"],
        "trc_backward_ms": metrics["trc_backward_ms"],
        "trc_train_ms": metrics["trc_train_ms"],
        "trc_eval_ms": metrics["trc_eval_ms"],
    }
    if bw_results:
        result["bandwidth"] = bw_results

    print(
        f"\n  DynamicEmb  train={metrics['dyn_train_ms']:.3f}"
        f"  fwd={metrics['dyn_forward_ms']:.3f}"
        f"  bwd={metrics['dyn_backward_ms']:.3f}"
        f"  eval={metrics['dyn_eval_ms']:.3f} ms"
    )
    if metrics["trc_train_ms"] is not None:
        print(
            f"  TorchRec    train={metrics['trc_train_ms']:.3f}"
            f"  fwd={metrics['trc_forward_ms']:.3f}"
            f"  bwd={metrics['trc_backward_ms']:.3f}"
            f"  eval={metrics['trc_eval_ms']:.3f} ms"
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

# Total embedding-table capacity *per suite* (across all tables in a config).
# At parametrize time we hand each table ``total_cap // nt`` rows so the
# overall HBM (and TorchRec TBE mirror) stays fixed as ``num_tables`` grows;
# without that scaling a 10-table TestGpu config would request ~240 GiB and
# OOM on 80 GB H100.  See the discussion above _gpu_configs for the 16M
# choice on TestGpu and 256M on the cache modes.
_GPU_TOTAL_CAP = 16 * 1024 * 1024
_CACHE_TOTAL_CAP = 256 * 1024 * 1024

_DIM = 128

_BATCH_SIZES = [1048576]
_NUM_TABLES = [10]
_OPTIMIZERS = ["adam", "sgd"]
# _POOLING_MODES = ["none", "sum"]
_POOLING_MODES = ["none"]


def _cache_hbm(
    gpu_ratio, cap_per_table, dim, optimizer_type, num_tables, emb_precision="fp32"
):
    """HBM for caching mode: gpu_ratio fraction of the full table value bytes per table."""
    emb_cfg = EmbeddingConfig(
        num_embeddings=cap_per_table,
        embedding_dim=dim,
        name="t",
        feature_names=["f"],
        data_type=_PRECISION_TO_DATATYPE.get(emb_precision, DataType.FP32),
    )
    table_bytes = get_table_value_bytes(emb_cfg, _DYN_OPT[optimizer_type], world_size=1)
    return [int(gpu_ratio * table_bytes)] * num_tables


def _gpu_configs():
    # _GPU_TOTAL_CAP=16M is chosen so the dual-backend TestGpu path (DynamicEmb
    # table + TorchRec TBE + per-iter activations on both + a pre-allocated
    # grad) fits in 80 GB HBM.  At 24M total with Adam (D + 2D state in fp32)
    # this peaked > 78 GiB and OOM'd inside benchmark_train_eval.  We hand
    # each of ``nt`` tables ``_GPU_TOTAL_CAP // nt`` rows so the total table
    # footprint stays at ~25 GiB per backend regardless of nt.
    return [
        BenchmarkConfig(
            batch_size=bs // nt,
            num_embeddings_per_feature=[_GPU_TOTAL_CAP // nt] * nt,
            embedding_dim=_DIM,
            hbm_for_embeddings=[sys.maxsize] * nt,
            optimizer_type=opt,
            caching=False,
            gpu_ratio=1.0,
            pooling_mode=pool,
            max_hotness=10,
        )
        for bs in _BATCH_SIZES
        for nt in _NUM_TABLES
        for opt in _OPTIMIZERS
        for pool in _POOLING_MODES
    ]


_CACHE_GPU_RATIO = 0.1
_CACHE_FOOTPRINT_RATIOS = [0.8, 1.0]


def _caching_configs():
    return [
        BenchmarkConfig(
            batch_size=bs // nt,
            num_embeddings_per_feature=[_CACHE_TOTAL_CAP // nt] * nt,
            embedding_dim=_DIM,
            # Placeholders -- both are overwritten at runtime by
            # _resize_cache_to_footprint based on the actual workload.
            hbm_for_embeddings=_cache_hbm(
                _CACHE_GPU_RATIO, _CACHE_TOTAL_CAP // nt, _DIM, opt, nt
            ),
            optimizer_type=opt,
            caching=True,
            cache_algorithm="lru",
            gpu_ratio=_CACHE_GPU_RATIO,
            pooling_mode=pool,
            max_hotness=10,
            cache_footprint_ratio=cfr,
        )
        for bs in _BATCH_SIZES
        for nt in _NUM_TABLES
        for opt in _OPTIMIZERS
        for pool in _POOLING_MODES
        for cfr in _CACHE_FOOTPRINT_RATIOS
    ]


def _no_caching_configs():
    return [
        BenchmarkConfig(
            batch_size=bs // nt,
            num_embeddings_per_feature=[_CACHE_TOTAL_CAP // nt] * nt,
            embedding_dim=_DIM,
            hbm_for_embeddings=_cache_hbm(
                _CACHE_GPU_RATIO, _CACHE_TOTAL_CAP // nt, _DIM, opt, nt
            ),
            optimizer_type=opt,
            caching=False,
            gpu_ratio=0.1,
            pooling_mode=pool,
            max_hotness=10,
        )
        for bs in _BATCH_SIZES
        for nt in _NUM_TABLES
        for opt in _OPTIMIZERS
        for pool in _POOLING_MODES
    ]


def _no_hbm_configs():
    """No HBM, no caching: all embedding data in system memory (UVM)."""
    return [
        BenchmarkConfig(
            batch_size=bs // nt,
            num_embeddings_per_feature=[_CACHE_TOTAL_CAP // nt] * nt,
            embedding_dim=_DIM,
            hbm_for_embeddings=[0] * nt,
            optimizer_type=opt,
            caching=False,
            gpu_ratio=0.0,
            pooling_mode=pool,
            max_hotness=10,
        )
        for bs in _BATCH_SIZES
        for nt in _NUM_TABLES
        for opt in _OPTIMIZERS
        for pool in _POOLING_MODES
    ]


# ── Test suites ───────────────────────────────────────────────────────────────


def _retable(cfg: BenchmarkConfig, new_nt: int) -> None:
    """Rebuild a config for ``new_nt`` tables in place, holding the per-suite
    total capacity and total batch fixed (per-table cap = total // new_nt,
    batch_size = total_batch // new_nt).  The per-table HBM entry is preserved
    and replicated to the new length."""
    old_nt = cfg.num_tables
    total_cap = sum(cfg.num_embeddings_per_feature)
    total_batch = cfg.batch_size * old_nt
    cfg.num_embeddings_per_feature = [max(1, total_cap // new_nt)] * new_nt
    cfg.batch_size = max(1, total_batch // new_nt)
    hbm0 = cfg.hbm_for_embeddings[0] if cfg.hbm_for_embeddings else sys.maxsize
    cfg.hbm_for_embeddings = [hbm0] * new_nt


def _apply_overrides(
    cfg: BenchmarkConfig,
    correctness_flag: bool,
    num_iterations: Optional[int] = None,
    ncu_iterations: Optional[str] = None,
    ncu_kernel_regex: Optional[str] = None,
    num_tables: Optional[int] = None,
    sparse_key_range: Optional[int] = None,
    no_torchrec: bool = False,
) -> BenchmarkConfig:
    """Apply session-wide CLI overrides onto a parametrized config."""
    if correctness_flag:
        cfg.correctness = True
    if num_iterations is not None:
        cfg.num_iterations = num_iterations
    if ncu_iterations is not None:
        cfg.ncu_iterations = ncu_iterations
    if ncu_kernel_regex is not None:
        cfg.ncu_kernel_regex = ncu_kernel_regex
    if num_tables is not None and num_tables != cfg.num_tables:
        _retable(cfg, num_tables)
    if sparse_key_range is not None:
        cfg.sparse_key_range = sparse_key_range
    if no_torchrec:
        cfg.run_torchrec = False
    return cfg


class TestGpu:
    @pytest.mark.parametrize("cfg", _gpu_configs(), ids=lambda c: c.label())
    def test_gpu(
        self,
        cfg,
        device,
        timer,
        profile_mode,
        correctness_flag,
        num_iterations,
        ncu_iterations,
        ncu_kernel_regex,
        num_tables,
        sparse_key_range,
        no_torchrec,
    ):
        cfg = _apply_overrides(
            cfg,
            correctness_flag,
            num_iterations,
            ncu_iterations,
            ncu_kernel_regex,
            num_tables,
            sparse_key_range,
            no_torchrec,
        )
        result = run_single_benchmark(cfg, device, timer, profile_mode)
        append_result(result)
        assert "error" not in result


class TestCaching:
    @pytest.mark.parametrize("cfg", _caching_configs(), ids=lambda c: c.label())
    def test_caching(
        self,
        cfg,
        device,
        timer,
        profile_mode,
        correctness_flag,
        num_iterations,
        ncu_iterations,
        ncu_kernel_regex,
        num_tables,
        sparse_key_range,
        no_torchrec,
    ):
        cfg = _apply_overrides(
            cfg,
            correctness_flag,
            num_iterations,
            ncu_iterations,
            ncu_kernel_regex,
            num_tables,
            sparse_key_range,
            no_torchrec,
        )
        result = run_single_benchmark(cfg, device, timer, profile_mode)
        append_result(result)
        assert "error" not in result


class TestNoCaching:
    @pytest.mark.parametrize("cfg", _no_caching_configs(), ids=lambda c: c.label())
    def test_no_caching(
        self,
        cfg,
        device,
        timer,
        profile_mode,
        correctness_flag,
        num_iterations,
        ncu_iterations,
        ncu_kernel_regex,
        num_tables,
        sparse_key_range,
        no_torchrec,
    ):
        cfg = _apply_overrides(
            cfg,
            correctness_flag,
            num_iterations,
            ncu_iterations,
            ncu_kernel_regex,
            num_tables,
            sparse_key_range,
            no_torchrec,
        )
        result = run_single_benchmark(cfg, device, timer, profile_mode)
        append_result(result)
        assert "error" not in result


class TestNoHbm:
    @pytest.mark.parametrize("cfg", _no_hbm_configs(), ids=lambda c: c.label())
    def test_no_hbm(
        self,
        cfg,
        device,
        timer,
        profile_mode,
        correctness_flag,
        num_iterations,
        ncu_iterations,
        ncu_kernel_regex,
        num_tables,
        sparse_key_range,
        no_torchrec,
    ):
        cfg = _apply_overrides(
            cfg,
            correctness_flag,
            num_iterations,
            ncu_iterations,
            ncu_kernel_regex,
            num_tables,
            sparse_key_range,
            no_torchrec,
        )
        result = run_single_benchmark(cfg, device, timer, profile_mode)
        append_result(result)
        assert "error" not in result
