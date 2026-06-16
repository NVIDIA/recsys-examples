# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Parametric tests: EmbeddingCollection + per-table ``global_hbm_for_values`` from
# ``get_table_value_bytes * hbm_ratio``, planner + DMP, then per-table checks that
# ``max_capacity`` matches ``get_sharded_table_capacity``, ``local_hbm_for_values``
# matches ``ceil(global/world_size)``, and cache/storage value tensors match
# hashtable ``per_table_capacity_`` (+ cache overflow slots when applicable).
#
# Run (from ``corelib/dynamicemb``)::
#   torchrun --nnodes 1 --nproc_per_node 1 -m pytest test/unit_tests/test_alignment.py -q
# ``world_size`` is ``int(os.environ.get("WORLD_SIZE", "1"))``; it must match
# ``dist.get_world_size()`` after init (``torchrun`` sets ``WORLD_SIZE``; default is 1).
# Set ``DYNAMICEMB_ALIGNMENT_FULL=1`` to add extra single-table ``num_embeddings`` values.
# ``bucket_capacity`` is parametrized over ``DEFAULT_BUCKET_CAPACITY``, ``1024``, and
# ``MAX_BUCKET_CAPACITY``.

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from dynamicemb import DynamicEmbScoreStrategy
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.dump_load import find_sharded_modules, get_dynamic_emb_module
from dynamicemb.dynamicemb_config import (
    DEFAULT_BUCKET_CAPACITY,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
    get_sharded_table_capacity,
    get_table_value_bytes,
)
from dynamicemb.get_planner import get_planner
from dynamicemb.key_value_table import DynamicEmbCache, DynamicEmbStorage, HybridStorage
from dynamicemb.optimizer import get_optimizer_state_dim
from dynamicemb.shard import DynamicEmbeddingCollectionSharder
from dynamicemb.types import MAX_BUCKET_CAPACITY
from dynamicemb.utils import DTYPE_NUM_BYTES
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from torchrec import DataType
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def _alignment_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _num_embeddings_per_table_params() -> List[Tuple[int, ...]]:
    singles = [(n,) for n in (7, 13, 127, 1001)]
    return singles + [(17, 99)]


def _require_cuda_dist() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not dist.is_initialized():
        pytest.skip(
            "Distributed not initialized; run with e.g.\n"
            "  torchrun --nnodes 1 --nproc_per_node 1 -m pytest "
            "test/unit_tests/test_alignment.py -v"
        )


@pytest.fixture(scope="session", autouse=True)
def _session_dist_init() -> None:
    if not torch.cuda.is_available():
        yield
        return
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


class _EmbeddingCollectionWrapper(nn.Module):
    def __init__(self, embedding_module: EmbeddingCollection) -> None:
        super().__init__()
        self.embedding_modules = nn.ModuleList([embedding_module])

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        embeddings_dict = [emb(kjt).wait() for emb in self.embedding_modules]
        out = []
        for d in embeddings_dict:
            for v in d.values():
                out.append(v.values())
        return torch.cat(out, dim=0)


def build_dmp_for_alignment_test(
    eb_configs: List[EmbeddingConfig],
    optimizer_type: EmbOptimType,
    bucket_capacity: int,
    hbm_ratio: float,
    caching: bool,
    training: bool,
    device: torch.device,
) -> DistributedModelParallel:
    world_size = dist.get_world_size()
    bc = int(bucket_capacity)
    dynamicemb_options_dict: Dict[str, DynamicEmbTableOptions] = {}
    for ec in eb_configs:
        total_bytes = get_table_value_bytes(ec, optimizer_type, world_size, bc)
        global_hbm = int(hbm_ratio * total_bytes)
        dynamicemb_options_dict[ec.name] = DynamicEmbTableOptions(
            global_hbm_for_values=global_hbm,
            caching=caching,
            training=training,
            bucket_capacity=bc,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.CONSTANT,
                value=0.1,
            ),
        )
    ebc = EmbeddingCollection(
        device=torch.device("meta"),
        tables=list(eb_configs),
    )
    model = _EmbeddingCollectionWrapper(ebc)
    eb_configs_list = list(ebc.embedding_configs())
    planner = get_planner(
        eb_configs_list,
        set(),
        dynamicemb_options_dict,
        device,
    )
    fused_params = {
        "output_dtype": SparseType.FP32,
        "optimizer": optimizer_type,
        "learning_rate": 1e-3,
    }
    sharder = DynamicEmbeddingCollectionSharder(
        fused_params=fused_params,
        use_index_dedup=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plan = planner.collective_plan(model, [sharder], dist.GroupMember.WORLD)
        dmp = DistributedModelParallel(
            module=model,
            device=device,
            sharders=[sharder],
            plan=plan,
        )
    return dmp


def _overflow_rows_for_cache(bucket_capacity: int) -> int:
    # dynamicemb.scored_hashtable: overflow_bucket_capacity_ = 3 * bucket_capacity_
    return 3 * int(bucket_capacity)


def _assert_value_buffer_matches_hashtable(
    state: object,
    table_id: int,
    *,
    include_overflow: bool,
) -> None:
    """Value tensor rows must match hashtable main capacity plus optional overflow slots."""
    km = state.key_index_map
    main = int(km.per_table_capacity_[table_id])
    ovf = int(km.overflow_bucket_capacity_) if include_overflow else 0
    if include_overflow:
        bc = int(state.options_list[0].bucket_capacity)
        assert ovf == _overflow_rows_for_cache(
            bc
        ), f"table {table_id}: overflow_bucket_capacity_={ovf} != 3 * bucket ({bc})"
    expected_rows = main + ovf
    t = state.tables[table_id].tensor()
    vd = int(state.table_value_dims_cpu[table_id])
    assert t.shape == (expected_rows, vd), (
        f"table {table_id}: value buffer shape {tuple(t.shape)} != "
        f"expected ({expected_rows}, {vd}) (main={main}, overflow_extra={ovf})"
    )


def assert_cache_and_storage_shapes(
    batched: BatchedDynamicEmbeddingTablesV2,
    eb_configs_by_name: Dict[str, EmbeddingConfig],
    bucket_capacity: int,
) -> None:
    """Planner capacities vs ``get_sharded_table_capacity``; value buffers vs hashtable metadata."""
    options = list(batched._dynamicemb_options)
    names = list(batched.table_names)
    world_size = dist.get_world_size()
    bc = int(bucket_capacity)

    for opt, name in zip(options, names):
        ec = eb_configs_by_name[name]
        expected_cap = get_sharded_table_capacity(ec, world_size, bc)
        assert int(opt.max_capacity) == expected_cap, (
            f"{name}: max_capacity={opt.max_capacity} != get_sharded_table_capacity "
            f"{expected_cap} (bucket_capacity={bc})"
        )
        gh = int(opt.global_hbm_for_values)
        assert int(opt.local_hbm_for_values) == (gh + world_size - 1) // world_size

    value_dims = [
        int(opt.dim) + int(batched._optimizer.get_state_dim(int(opt.dim)))
        for opt in options
    ]
    total_memory = sum(
        int(opt.max_capacity) * int(DTYPE_NUM_BYTES[opt.embedding_dtype]) * vd
        for opt, vd in zip(options, value_dims)
    )
    local_hbm = sum(int(opt.local_hbm_for_values) for opt in options)

    if batched._caching and total_memory > local_hbm:
        assert batched._cache is not None
        assert isinstance(batched._cache, DynamicEmbCache)
        cstate = batched._cache._state
        for tid in range(len(names)):
            _assert_value_buffer_matches_hashtable(cstate, tid, include_overflow=True)
    else:
        assert batched._cache is None

    storage = batched._storage
    if isinstance(storage, DynamicEmbStorage):
        for tid in range(len(names)):
            _assert_value_buffer_matches_hashtable(
                storage._state, tid, include_overflow=False
            )
    elif isinstance(storage, HybridStorage):
        assert not batched._caching
        assert total_memory > local_hbm
        for tid in range(len(names)):
            _assert_value_buffer_matches_hashtable(
                storage._hbm, tid, include_overflow=False
            )
            _assert_value_buffer_matches_hashtable(
                storage._host, tid, include_overflow=False
            )
    else:
        raise AssertionError(f"Unexpected storage type {type(storage)}")


@pytest.mark.parametrize("caching", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize(
    "bucket_capacity",
    [DEFAULT_BUCKET_CAPACITY, 1024, MAX_BUCKET_CAPACITY],
)
@pytest.mark.parametrize(
    "optimizer_type",
    [
        EmbOptimType.SGD,
        EmbOptimType.ADAM,
        EmbOptimType.EXACT_ROWWISE_ADAGRAD,
    ],
)
@pytest.mark.parametrize("hbm_ratio", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("embedding_dim", [16, 128])
@pytest.mark.parametrize("num_embeddings_per_table", _num_embeddings_per_table_params())
def test_alignment_cache_storage_shapes(
    caching: bool,
    training: bool,
    bucket_capacity: int,
    optimizer_type: EmbOptimType,
    hbm_ratio: float,
    embedding_dim: int,
    num_embeddings_per_table: Tuple[int, ...],
) -> None:
    if caching and hbm_ratio == 0.0:
        pytest.skip("caching with zero global HBM is invalid")
    world_size = _alignment_world_size()
    _require_cuda_dist()
    assert (
        dist.get_world_size() == world_size
    ), f"dist.world_size={dist.get_world_size()} != WORLD_SIZE={world_size}"

    tables: List[EmbeddingConfig] = []
    for i, nemb in enumerate(num_embeddings_per_table):
        name = f"t{i}"
        tables.append(
            EmbeddingConfig(
                name=name,
                embedding_dim=embedding_dim,
                num_embeddings=nemb,
                feature_names=[f"f{i}"],
                data_type=DataType.FP32,
            )
        )
    by_name = {t.name: t for t in tables}
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dmp = build_dmp_for_alignment_test(
                tables,
                optimizer_type,
                bucket_capacity,
                hbm_ratio,
                caching,
                training,
                device,
            )
    except ValueError as e:
        pytest.fail(f"DMP build failed: {e}")

    emb_modules: List[nn.Module] = []
    for _, _, sharded in find_sharded_modules(dmp):
        emb_modules.extend(get_dynamic_emb_module(sharded))
    assert emb_modules, "no BatchedDynamicEmbeddingTablesV2 under DMP"
    batched = emb_modules[0]
    assert isinstance(batched, BatchedDynamicEmbeddingTablesV2)
    assert batched._caching == caching

    assert_cache_and_storage_shapes(batched, by_name, bucket_capacity)

    if training:
        torch_dtype = batched._dynamicemb_options[0].embedding_dtype
        assert torch_dtype is not None
        st_dim = get_optimizer_state_dim(optimizer_type, embedding_dim, torch_dtype)
        if isinstance(batched._storage, HybridStorage):
            vd_state = batched._storage._hbm
        else:
            vd_state = batched._storage._state
        for tid, name in enumerate(batched.table_names):
            ec = by_name[name]
            assert int(batched._dynamicemb_options[tid].dim) == ec.embedding_dim
            assert vd_state.table_value_dims_cpu[tid] == int(ec.embedding_dim) + int(
                st_dim
            )
