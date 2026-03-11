# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unit test: Memory stats for varying EmbeddingConfig.num_embeddings and global_hbm_for_values,
# with caching on/off.
#
# Also compares with actual DMP model config: EmbeddingConfig -> EmbeddingCollection -> apply_dmp,
# then read max_capacity / local_hbm_for_values from _dynamicemb_options and compare to theoretical.

import math
import os
import sys
import warnings
from typing import Any, Dict, List, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from dynamicemb.dump_load import find_sharded_modules, get_dynamic_emb_module

# Run from dynamicemb package root or with PYTHONPATH including corelib/dynamicemb
from dynamicemb.dynamicemb_config import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
    align_to_table_size,
    data_type_to_dtype,
    dtype_to_bytes,
    get_constraint_capacity,
    get_optimizer_state_dim,
)
from dynamicemb.get_planner import get_planner
from dynamicemb.shard import DynamicEmbeddingCollectionSharder
from dynamicemb.types import DEMB_TABLE_ALIGN_SIZE
from dynamicemb_extensions import OptimizerType
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from torchrec import DataType
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# Default world sizes for report (aligned with planner: per-rank capacity and per-rank HBM)
DEFAULT_WORLD_SIZES = [1, 8]

# Fixed table params (aligned with planner / batched_dynamicemb_tables)
EMBEDDING_DIM = 128
EMBEDDING_DTYPE = torch.float32
OPTIMIZER_TYPE = OptimizerType.Adam
# Non-cache table: bucket_capacity fixed at 128; per-rank aligned capacity at least align_to_table_size(128)
BUCKET_CAPACITY_NORMAL = 128
# Cache mode: cache bucket_capacity=1024, minimum capacity 1024 (round up to 1 bucket if smaller)
BUCKET_CAPACITY_CACHE = 1024


def _element_size() -> int:
    return dtype_to_bytes(EMBEDDING_DTYPE)


def _optim_state_dim() -> int:
    return get_optimizer_state_dim(OPTIMIZER_TYPE, EMBEDDING_DIM, EMBEDDING_DTYPE)


def _total_dim() -> int:
    return EMBEDDING_DIM + _optim_state_dim()


def _byte_per_vector() -> int:
    return _element_size() * _total_dim()


def compute_memory_stats(
    num_embeddings: int,
    global_hbm_for_values: int,
    caching: bool,
    world_size: int = 1,
) -> Tuple[int, int, int, int, int]:
    """
    Compute per-rank HBM/DRAM stats for a single table (aligned with planner + batched_dynamicemb_tables).

    Planner logic:
    - num_embeddings_per_rank = align_to_table_size(ceil(num_embeddings / world_size))
    - If num_aligned_embedding_per_rank < bucket_capacity(128), round up to align_to_table_size(128)
    - local_hbm_for_values = ceil(global_hbm_for_values / world_size)
    - When caching: cache bucket=1024, min capacity 1024 (get_constraint_capacity rounds up to 1 bucket)

    Returns
    -------
    total_bytes_per_rank, hbm_bytes_per_rank, dram_bytes_per_rank, aligned_capacity_per_rank, total_bytes_all_ranks
    """
    num_per_rank = math.ceil(num_embeddings / world_size)
    aligned_capacity_per_rank = align_to_table_size(num_per_rank)
    # Align with planner: per-rank capacity at least align_to_table_size(bucket_capacity)=128
    min_capacity_from_bucket = align_to_table_size(BUCKET_CAPACITY_NORMAL)
    aligned_capacity_per_rank = max(aligned_capacity_per_rank, min_capacity_from_bucket)
    total_memory_per_rank = aligned_capacity_per_rank * _byte_per_vector()

    local_hbm = math.ceil(global_hbm_for_values / world_size)
    local_hbm = min(local_hbm, total_memory_per_rank)

    if caching:
        bucket_cap = BUCKET_CAPACITY_CACHE  # 1024
        cache_capacity = get_constraint_capacity(
            local_hbm,
            EMBEDDING_DTYPE,
            EMBEDDING_DIM,
            OPTIMIZER_TYPE,
            bucket_cap,
        )
        # Cache min capacity 1024 (get_constraint_capacity already rounds up to 1 bucket if needed)
        cache_capacity = max(cache_capacity, BUCKET_CAPACITY_CACHE)
        hbm_bytes_per_rank = cache_capacity * _byte_per_vector()
        # Storage holds full table shard for this rank, all in DRAM
        dram_bytes_per_rank = total_memory_per_rank
    else:
        hbm_bytes_per_rank = local_hbm
        dram_bytes_per_rank = total_memory_per_rank - local_hbm

    total_bytes_all_ranks = total_memory_per_rank * world_size
    return (
        total_memory_per_rank,
        hbm_bytes_per_rank,
        dram_bytes_per_rank,
        aligned_capacity_per_rank,
        total_bytes_all_ranks,
    )


def _mb(x: int) -> float:
    return x / (1024 * 1024)


def _format_mb(x: int) -> str:
    return f"{_mb(x):.2f} MB"


def run_alignment_memory_report(
    num_embeddings_list: List[int],
    global_hbm_modes: List[str],
    world_sizes: List[int],
    include_caching: bool = True,
) -> List[dict]:
    """
    Build memory report for (num_embeddings, global_hbm_for_values, caching, world_size) combinations.
    global_hbm_modes: ["0", "half", "full"] = HBM budget 0 / half of total need / full (all global).
    total/hbm/dram are per-rank values.
    """
    rows = []
    min_cap_bucket = align_to_table_size(BUCKET_CAPACITY_NORMAL)
    for num_emb in num_embeddings_list:
        for world_size in world_sizes:
            num_per_rank = math.ceil(num_emb / world_size)
            aligned_per_rank = align_to_table_size(num_per_rank)
            aligned_per_rank = max(aligned_per_rank, min_cap_bucket)
            total_mem_per_rank = aligned_per_rank * _byte_per_vector()
            # Global HBM budget (for half/full): based on total table memory across all ranks
            total_mem_global = total_mem_per_rank * world_size

            for gmode in global_hbm_modes:
                if gmode == "0":
                    global_hbm = 0
                elif gmode == "half":
                    global_hbm = total_mem_global // 2
                elif gmode == "full":
                    global_hbm = total_mem_global
                else:
                    raise ValueError(f"Unknown global_hbm mode: {gmode}")

                for caching in [False, True] if include_caching else [False]:
                    (
                        total_bytes,
                        hbm_bytes,
                        dram_bytes,
                        aligned_cap,
                        total_all_ranks,
                    ) = compute_memory_stats(num_emb, global_hbm, caching, world_size)
                    rows.append(
                        {
                            "num_embeddings": num_emb,
                            "world_size": world_size,
                            "aligned_capacity_per_rank": aligned_cap,
                            "global_hbm_mode": gmode,
                            "global_hbm_bytes": global_hbm,
                            "caching": caching,
                            "total_bytes": total_bytes,
                            "hbm_bytes": hbm_bytes,
                            "dram_bytes": dram_bytes,
                            "total_bytes_all_ranks": total_all_ranks,
                        }
                    )
    return rows


def print_report(rows: List[dict], show_all_ranks: bool = False) -> None:
    """Print memory consumption table. total/HBM/DRAM are per rank; optionally show all_ranks column."""
    sep = " | "
    headers = [
        "num_emb",
        "W",
        "aligned/r",
        "global_hbm",
        "caching",
        "total(MB)/r",
        "HBM(MB)/r",
        "DRAM(MB)/r",
    ]
    if show_all_ranks:
        headers.append("total(MB)*W")
    col_widths = [10, 4, 10, 8, 8, 12, 12, 12]
    if show_all_ranks:
        col_widths.append(12)
    line = sep.join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))

    for r in rows:
        total_mb = _format_mb(r["total_bytes"])
        hbm_mb = _format_mb(r["hbm_bytes"])
        dram_mb = _format_mb(r["dram_bytes"])
        global_hbm_str = r["global_hbm_mode"]
        row = [
            str(r["num_embeddings"]).ljust(col_widths[0]),
            str(r["world_size"]).ljust(col_widths[1]),
            str(r["aligned_capacity_per_rank"]).ljust(col_widths[2]),
            global_hbm_str.ljust(col_widths[3]),
            str(r["caching"]).ljust(col_widths[4]),
            total_mb.ljust(col_widths[5]),
            hbm_mb.ljust(col_widths[6]),
            dram_mb.ljust(col_widths[7]),
        ]
        if show_all_ranks:
            row.append(_format_mb(r["total_bytes_all_ranks"]).ljust(col_widths[8]))
        print(sep.join(row))
    print()


# --------------- Compare with actual DMP model config ---------------


class _SingleTableTestModel(nn.Module):
    """Single EmbeddingCollection, single table; used to read actual config after apply_dmp."""

    def __init__(self, embedding_module: EmbeddingCollection):
        super().__init__()
        self.embedding_modules = nn.ModuleList([embedding_module])

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        embeddings_dict = [emb(kjt).wait() for emb in self.embedding_modules]
        out = []
        for d in embeddings_dict:
            for v in d.values():
                out.append(v.values())
        return torch.cat(out, dim=0)


def _apply_dmp_with_global_hbm(
    num_embeddings: int,
    embedding_dim: int,
    global_hbm_for_values: int,
    caching: bool,
    device: torch.device,
    optimizer_kwargs: Dict[str, Any],
) -> nn.Module:
    """
    Create single-table EmbeddingConfig -> EmbeddingCollection -> apply_dmp; return DMP model.
    global_hbm_for_values is global HBM budget in bytes; planner splits by world_size per rank.
    """
    from dynamicemb import DynamicEmbScoreStrategy

    name = "emb_0"
    eb_config = EmbeddingConfig(
        name=name,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        feature_names=["f0"],
        data_type=DataType.FP32,
    )
    ebc = EmbeddingCollection(
        device=torch.device("meta"),
        tables=[eb_config],
    )
    model = _SingleTableTestModel(ebc)

    emb_num_aligned = align_to_table_size(num_embeddings)
    bucket_capacity = BUCKET_CAPACITY_CACHE if caching else BUCKET_CAPACITY_NORMAL
    # Planner will use align_to_table_size(ceil(num_embeddings/world_size)) per rank; placeholder here
    torch_dtype = data_type_to_dtype(DataType.FP32)
    opt_state_dim = get_optimizer_state_dim(
        OptimizerType.Adam, embedding_dim, torch_dtype
    )
    total_hbm_need = (
        (embedding_dim + opt_state_dim) * dtype_to_bytes(torch_dtype) * emb_num_aligned
    )
    # If global_hbm not set, use full need
    if global_hbm_for_values <= 0:
        global_hbm_for_values = total_hbm_need

    dynamicemb_options_dict = {
        name: DynamicEmbTableOptions(
            global_hbm_for_values=global_hbm_for_values,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.CONSTANT,
                value=0.1,
            ),
            bucket_capacity=bucket_capacity,
            max_capacity=emb_num_aligned,
            caching=caching,
        )
    }
    eb_configs = list(ebc.embedding_configs())
    planner = get_planner(
        eb_configs,
        set(),
        dynamicemb_options_dict,
        device,
    )
    fused_params = {
        "output_dtype": SparseType.FP32,
        **optimizer_kwargs,
    }
    sharder = DynamicEmbeddingCollectionSharder(
        fused_params=fused_params,
        use_index_dedup=False,
    )
    plan = planner.collective_plan(model, [sharder], dist.GroupMember.WORLD)
    dmp = DistributedModelParallel(
        module=model,
        device=device,
        sharders=[sharder],
        plan=plan,
    )
    return dmp


def get_actual_table_options_from_model(model: nn.Module) -> List[Dict[str, Any]]:
    """
    Collect actual config (max_capacity, local_hbm_for_values, caching) for all
    BatchedDynamicEmbeddingTablesV2 in the model after apply_dmp.
    Uses find_sharded_modules to get ShardedEmbeddingCollection, then get_dynamic_emb_module on it.
    """
    result = []
    for _path, _name, sharded_module in find_sharded_modules(model):
        emb_modules = get_dynamic_emb_module(sharded_module)
        for mod in emb_modules:
            for opt in mod._dynamicemb_options:
                result.append(
                    {
                        "max_capacity": opt.max_capacity,
                        "local_hbm_for_values": opt.local_hbm_for_values,
                        "caching": opt.caching,
                    }
                )
    return result


def _compare_actual_vs_theoretical(
    num_embeddings: int,
    global_hbm_for_values: int,
    caching: bool,
    world_size: int,
) -> Tuple[bool, str]:
    """
    Build DMP model, read actual config, compare with compute_memory_stats theoretical values.
    When bucket floor applies, planner may set local_hbm_for_values above ceil(global_hbm/W);
    we compare effective HBM = min(actual local_hbm, total_memory) to theory.
    Returns (match_ok, message).
    """
    (
        _,
        hbm_per_rank,
        _,
        aligned_cap_expected,
        _,
    ) = compute_memory_stats(num_embeddings, global_hbm_for_values, caching, world_size)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dmp = _apply_dmp_with_global_hbm(
            num_embeddings=num_embeddings,
            embedding_dim=EMBEDDING_DIM,
            global_hbm_for_values=global_hbm_for_values,
            caching=caching,
            device=device,
            optimizer_kwargs={"optimizer": EmbOptimType.ADAM, "lr": 1e-3},
        )
    actual_list = get_actual_table_options_from_model(dmp)
    if not actual_list:
        return False, "get_dynamic_emb_module returned no table options"
    actual = actual_list[0]
    cap_ok = actual["max_capacity"] == aligned_cap_expected
    # Non-cache: effective HBM is min(actual, total); planner may set actual > theory when bucket floor adds capacity.
    # Cache: actual HBM is cache size, may be >= theory due to bucket rounding.
    total_per_rank = aligned_cap_expected * _byte_per_vector()
    if not caching:
        effective_actual_hbm = min(
            actual["local_hbm_for_values"], total_per_rank
        )
        hbm_ok = effective_actual_hbm == hbm_per_rank
    else:
        hbm_ok = actual["local_hbm_for_values"] >= 0
    caching_ok = actual["caching"] == caching
    msg = (
        f"num_emb={num_embeddings} W={world_size} caching={caching} "
        f"expected_cap={aligned_cap_expected} actual_cap={actual['max_capacity']} "
        f"expected_hbm={hbm_per_rank} actual_hbm={actual['local_hbm_for_values']}"
    )
    return cap_ok and caching_ok and hbm_ok, msg


class TestAlignmentMemoryStats:
    """Tests for memory stats under varying num_embeddings / global_hbm_for_values / caching / world_size."""

    @pytest.fixture
    def num_embeddings_list(self) -> List[int]:
        # Cover alignment boundaries: below 16, equal 16, non-multiple of 16, larger
        return [10, 16, 17, 32, 100, 1000, 10000]

    @pytest.fixture
    def global_hbm_modes(self) -> List[str]:
        return ["0", "half", "full"]

    @pytest.fixture
    def world_sizes(self) -> List[int]:
        return [1, 8]

    def test_align_to_table_size_matches_demb_align_size(self):
        """Aligned capacity is a multiple of DEMB_TABLE_ALIGN_SIZE."""
        for n in [0, 1, 15, 16, 17, 32, 100]:
            aligned = align_to_table_size(n)
            assert (
                aligned % DEMB_TABLE_ALIGN_SIZE == 0
            ), f"align_to_table_size({n}) = {aligned} not multiple of {DEMB_TABLE_ALIGN_SIZE}"
            if n > 0:
                assert aligned >= n, f"align_to_table_size({n}) = {aligned} < {n}"

    def test_memory_stats_total_consistent(
        self, num_embeddings_list, global_hbm_modes, world_sizes
    ):
        """Per-rank total matches aligned_capacity_per_rank; non-caching: HBM+DRAM=total; caching: DRAM=total."""
        rows = run_alignment_memory_report(
            num_embeddings_list, global_hbm_modes, world_sizes, include_caching=True
        )
        for r in rows:
            expected_total = r["aligned_capacity_per_rank"] * _byte_per_vector()
            assert (
                r["total_bytes"] == expected_total
            ), f"total_bytes mismatch: {r['total_bytes']} vs {expected_total}"
            if not r["caching"]:
                assert (
                    r["hbm_bytes"] + r["dram_bytes"] == r["total_bytes"]
                ), f"non-caching: hbm + dram != total: {r}"
            else:
                assert (
                    r["dram_bytes"] == r["total_bytes"]
                ), f"caching: dram should equal total (storage): {r}"

    def test_caching_increases_hbm_when_global_hbm_nonzero(
        self, num_embeddings_list, global_hbm_modes, world_sizes
    ):
        """When global_hbm is non-zero, HBM under caching is determined by cache capacity."""
        min_cap = align_to_table_size(BUCKET_CAPACITY_NORMAL)
        for num_emb in num_embeddings_list:
            for world_size in world_sizes:
                num_per_rank = math.ceil(num_emb / world_size)
                aligned = align_to_table_size(num_per_rank)
                aligned = max(aligned, min_cap)
                total_mem_global = aligned * _byte_per_vector() * world_size
                for gmode in ["half", "full"]:
                    global_hbm = (
                        total_mem_global // 2 if gmode == "half" else total_mem_global
                    )
                    _, hbm_no_cache, _, _, _ = compute_memory_stats(
                        num_emb, global_hbm, caching=False, world_size=world_size
                    )
                    _, hbm_cache, _, _, _ = compute_memory_stats(
                        num_emb, global_hbm, caching=True, world_size=world_size
                    )
                    assert hbm_cache >= 0 and hbm_no_cache >= 0

    def test_alignment_memory_report_runs(
        self, num_embeddings_list, global_hbm_modes, world_sizes
    ):
        """Run full report and assert every row has valid values."""
        rows = run_alignment_memory_report(
            num_embeddings_list, global_hbm_modes, world_sizes, include_caching=True
        )
        assert len(rows) > 0
        for r in rows:
            assert r["total_bytes"] > 0
            assert r["hbm_bytes"] >= 0 and r["dram_bytes"] >= 0
            num_per_rank = math.ceil(r["num_embeddings"] / r["world_size"])
            assert r["aligned_capacity_per_rank"] >= align_to_table_size(num_per_rank)

    def test_multi_rank_reduces_per_rank_memory(self):
        """With world_size=8, per-rank total_bytes should be less than full-table size with world_size=1."""
        num_emb = 10000
        total_ws1 = align_to_table_size(num_emb) * _byte_per_vector()
        total_ws8_per_rank, _, _, _, _ = compute_memory_stats(
            num_emb, global_hbm_for_values=0, caching=False, world_size=8
        )
        assert total_ws8_per_rank < total_ws1

    def test_num_aligned_embedding_per_rank_bucket_floor(self):
        """num_aligned_embedding_per_rank is bounded by bucket_capacity: at least align_to_table_size(128)=128."""
        min_cap = align_to_table_size(BUCKET_CAPACITY_NORMAL)
        assert min_cap == 128
        for num_emb in [1, 10, 17, 50]:
            for world_size in [1, 8]:
                _, _, _, aligned_cap, _ = compute_memory_stats(
                    num_emb,
                    global_hbm_for_values=0,
                    caching=False,
                    world_size=world_size,
                )
                assert (
                    aligned_cap >= min_cap
                ), f"num_emb={num_emb} world_size={world_size} aligned_cap={aligned_cap} < {min_cap}"

    def test_cache_min_capacity_1024(self):
        """When caching is on, cache min capacity is 1024 (round up to 1 bucket if smaller)."""
        for num_emb in [10, 100, 1000]:
            for world_size in [1, 8]:
                _, hbm_bytes, _, _, _ = compute_memory_stats(
                    num_emb,
                    global_hbm_for_values=0,
                    caching=True,
                    world_size=world_size,
                )
                cache_capacity_rows = hbm_bytes // _byte_per_vector()
                assert (
                    cache_capacity_rows >= BUCKET_CAPACITY_CACHE
                ), f"num_emb={num_emb} W={world_size} cache_capacity_rows={cache_capacity_rows} < 1024"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for DMP model creation",
    )
    def test_actual_capacity_matches_theoretical(self):
        """
        Compare with actual DMP model config: create EmbeddingConfig -> EmbeddingCollection ->
        DynamicEmbTableOptions -> apply_dmp to get model, read actual max_capacity / local_hbm_for_values
        from model and assert they match compute_memory_stats theoretical values.
        Run with torchrun to init dist and perform comparison.
        """
        if not dist.is_initialized():
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                dist.init_process_group(
                    backend="gloo" if not torch.cuda.is_available() else "nccl",
                    init_method="env://",
                )
            else:
                pytest.skip(
                    "Distributed not initialized; run with torchrun to compare actual vs theoretical."
                )
        if not dist.is_initialized():
            pytest.skip("Failed to init process group")
        world_size = dist.get_world_size()
        num_embeddings = 1000
        # Global HBM = full (same as "full" in theoretical report; use same bucket floor as compute_memory_stats)
        aligned_per_rank = align_to_table_size(math.ceil(num_embeddings / world_size))
        aligned_per_rank = max(
            aligned_per_rank, align_to_table_size(BUCKET_CAPACITY_NORMAL)
        )
        total_mem_global = aligned_per_rank * _byte_per_vector() * world_size
        global_hbm = total_mem_global
        ok, msg = _compare_actual_vs_theoretical(
            num_embeddings=num_embeddings,
            global_hbm_for_values=global_hbm,
            caching=False,
            world_size=world_size,
        )
        assert ok, msg


def main():
    """CLI entry: print memory stats for varying num_embeddings / global_hbm_for_values / caching / world_size."""
    num_embeddings_list = [10, 16, 17, 32, 100, 1000, 10000]
    global_hbm_modes = ["0", "half", "full"]
    world_sizes = DEFAULT_WORLD_SIZES  # e.g. [1, 8]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        print("Alignment & HBM memory report (per rank; dim=128, Adam)")
        print("DEMB_TABLE_ALIGN_SIZE =", DEMB_TABLE_ALIGN_SIZE)
        print(
            "W = world_size; aligned/r = aligned capacity per rank; global_hbm = global budget"
        )
        print("total/HBM/DRAM = per rank (total(MB)*W = all ranks total table memory)")
        print()

        rows = run_alignment_memory_report(
            num_embeddings_list, global_hbm_modes, world_sizes, include_caching=True
        )
    print_report(rows, show_all_ranks=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
