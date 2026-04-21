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

import json
import os
import random
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

import numpy as np
import pytest
import torch
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
    get_sharded_table_capacity,
    get_table_value_bytes,
)
from dynamicemb.batched_dynamicemb_tables import (
    BatchedDynamicEmbeddingTablesV2,
    encode_checkpoint_file_path,
)
from dynamicemb.dynamicemb_config import (
    DEBUG_EMB_INITIALIZER_MOD,
    _sharded_table_bucket_layout,
)
from dynamicemb.key_value_table import (
    DynamicEmbCache,
    DynamicEmbStorage,
    DynamicEmbTableState,
    HybridStorage,
    Storage,
    _flat_row_indices_from_slots_and_scores,
    export_keys_values_iter,
    load_from_flat,
    load_from_flat_single_table,
)
from dynamicemb.optimizer import (
    BaseDynamicEmbeddingOptimizer,
    get_optimizer_ckpt_state_dim,
    pad_optimizer_states_from_checkpoint,
    truncate_optimizer_states_for_checkpoint,
)
from dynamicemb.types import EMBEDDING_TYPE, KEY_TYPE, OPT_STATE_TYPE, CopyMode
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)

POOLING_MODE: Dict[DynamicEmbPoolingMode, PoolingMode] = {
    DynamicEmbPoolingMode.NONE: PoolingMode.NONE,
    DynamicEmbPoolingMode.MEAN: PoolingMode.MEAN,
    DynamicEmbPoolingMode.SUM: PoolingMode.SUM,
}

# Strict table expansion: modest max_capacity so we can insert every slot exactly once.
_CACHE_RATIO = 0.25
_WORLD_SIZE = 1
_EMB_DIM_EXPANSION = 8
_BUCKET_CAPACITY_EXP = 128
# Global num_embeddings for single-table strict expansion (need not be multiples of _BUCKET_CAPACITY_EXP).
_STRICT_SINGLE_NUM_EMB = 20_000
# Per-table global num_embeddings (need not be multiples of _BUCKET_CAPACITY_EXP).
_STRICT_MULTI_NUM_EMB = (20_000, 30_000, 40_000)
# Same as ``DEBUG_EMB_INITIALIZER_MOD`` / ``debug_init`` (initializer.cu).
_DEBUG_EMB_MOD = DEBUG_EMB_INITIALIZER_MOD
# Expansion tests assume key_index_map rehash threshold at load factor 0.5.
_STRICT_MAX_LOAD_FACTOR = 0.5


def _init_capacity_strict(max_capacity: int, bucket_cap: int) -> int:
    """init_capacity ≈ max_capacity // 16, rounded down to a multiple of ``bucket_cap``; always < max_capacity."""
    if max_capacity <= bucket_cap:
        return bucket_cap
    q = max_capacity // 16
    init_cap = (q // bucket_cap) * bucket_cap
    if init_cap < bucket_cap:
        init_cap = bucket_cap
    if init_cap >= max_capacity:
        init_cap = ((max_capacity - 1) // bucket_cap) * bucket_cap
        if init_cap < bucket_cap:
            init_cap = bucket_cap
    return init_cap


def _embedding_configs_for_strict_expansion_test(
    multi_table: bool,
) -> Tuple[List[Any], List[str], List[int]]:
    from torchrec import DataType
    from torchrec.modules.embedding_configs import EmbeddingConfig

    if not multi_table:
        names = ["t0"]
        num_embs = [_STRICT_SINGLE_NUM_EMB]
        feature_table_map = [0]
    else:
        names = ["t0", "t1", "t2"]
        num_embs = list(_STRICT_MULTI_NUM_EMB)
        feature_table_map = [0, 1, 2]
    configs = [
        EmbeddingConfig(
            name=names[i],
            embedding_dim=_EMB_DIM_EXPANSION,
            num_embeddings=num_embs[i],
            feature_names=[f"f_{names[i]}"],
            data_type=DataType.FP32,
        )
        for i in range(len(names))
    ]
    return configs, names, feature_table_map


def _iter_dynamic_emb_table_states(
    bdebt: BatchedDynamicEmbeddingTablesV2,
) -> List[Tuple[str, DynamicEmbTableState]]:
    """All table states (cache + backing, or hybrid tiers)."""
    out: List[Tuple[str, DynamicEmbTableState]] = []
    cache = bdebt.cache
    if isinstance(cache, DynamicEmbCache):
        out.append(("cache", cache._state))
    st = bdebt.tables
    if isinstance(st, DynamicEmbStorage):
        out.append(("storage", st._state))
    elif isinstance(st, HybridStorage):
        out.append(("hbm", st.tables[0]))
        out.append(("host", st.tables[1]))
    return out


def _lfu_per_sample_weights(
    indices: torch.Tensor, device: torch.device
) -> torch.Tensor:
    return torch.ones(indices.numel(), dtype=torch.int64, device=device)


# Chunk size when iterating keys 0..cmp_capacity-1; align with table ``bucket_capacity``.
_EXPANSION_STBE_TRAIN_BATCH = _BUCKET_CAPACITY_EXP


def _init_stbe_debug_embedding_weights(
    stbe: SplitTableBatchedEmbeddingBagsCodegen,
) -> None:
    """Match DynamicEmb ``DEBUG`` init: row ``i`` is filled with ``i % _DEBUG_EMB_MOD`` in every dim."""
    with torch.no_grad():
        for w in stbe.split_embedding_weights():
            n, d = w.shape
            if n == 0:
                continue
            vals = (
                torch.arange(n, device=w.device, dtype=torch.float32)
                % float(_DEBUG_EMB_MOD)
            ).view(n, 1)
            w.copy_(vals.expand(-1, d))


def _bdebt_forward_maybe_lfu(
    bdebt: BatchedDynamicEmbeddingTablesV2,
    score_strategy: DynamicEmbScoreStrategy,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if score_strategy == DynamicEmbScoreStrategy.LFU:
        return bdebt(
            indices,
            offsets,
            per_sample_weights=_lfu_per_sample_weights(indices, device),
        )
    return bdebt(indices, offsets)


def _iter_compare_batches_single(
    cmp_cap: int,
    batch_size: int,
    device: torch.device,
    key_type: torch.dtype,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    for s in range(0, cmp_cap, batch_size):
        e = min(s + batch_size, cmp_cap)
        idx = torch.arange(s, e, device=device, dtype=key_type)
        off = torch.tensor([0, e - s], device=device, dtype=key_type)
        yield idx, off


def _iter_compare_batches_multi_three(
    cmp_caps: List[int],
    batch_size: int,
    device: torch.device,
    key_type: torch.dtype,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Cover local keys ``0 .. cmp_caps[t]-1`` on each of three tables using anchored triples.

    ``BatchedDynamicEmbeddingTablesV2`` / FBGEMM use the same layout as ``get_table_range``:
    for ``batch_size==1`` and three features (tables 0,1,2), ``indices`` must be **feature-major**
    — all keys for table 0, then all for table 1, then all for table 2 — with
    ``offsets = [0, n0, n0+n1, n0+n1+n2]``. Interleaved ``[a0,b0,c0,a1,...]`` with
    ``[0,n,2n,3n]`` is wrong: table 0 would incorrectly receive other tables' keys.
    """
    c0, c1, c2 = cmp_caps[0], cmp_caps[1], cmp_caps[2]
    min_c = min(cmp_caps)

    def pack_triples(
        rows: List[Tuple[int, int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack one micro-batch: rows are (key_table0, key_table1, key_table2) per sample."""
        n = len(rows)
        if n == 0:
            z = torch.tensor([], device=device, dtype=key_type)
            return z, torch.tensor([0, 0, 0, 0], device=device, dtype=key_type)
        col0 = [r[0] for r in rows]
        col1 = [r[1] for r in rows]
        col2 = [r[2] for r in rows]
        flat = col0 + col1 + col2
        idx = torch.tensor(flat, device=device, dtype=key_type)
        n0, n1 = len(col0), len(col1)
        off = torch.tensor(
            [0, n0, n0 + n1, n0 + n1 + len(col2)],
            device=device,
            dtype=key_type,
        )
        return idx, off

    for s in range(0, min_c, batch_size):
        e = min(s + batch_size, min_c)
        rows = [(k, k, k) for k in range(s, e)]
        yield pack_triples(rows)

    a1 = max(min_c - 1, 0)
    a2 = max(min_c - 1, 0)
    for s in range(min_c, c0, batch_size):
        e = min(s + batch_size, c0)
        rows = [(k, a1, a2) for k in range(s, e)]
        yield pack_triples(rows)

    a0 = max(c0 - 1, 0)
    for s in range(min_c, c1, batch_size):
        e = min(s + batch_size, c1)
        rows = [(a0, k, a2) for k in range(s, e)]
        yield pack_triples(rows)

    a1b = max(c1 - 1, 0)
    for s in range(min_c, c2, batch_size):
        e = min(s + batch_size, c2)
        rows = [(a0, a1b, k) for k in range(s, e)]
        yield pack_triples(rows)


def _print_hybrid_hbm_host_sizes(
    storage: Storage,
    table_names: List[str],
    *,
    phase: str,
    only_table_id: Optional[int] = None,
) -> None:
    """Print ``key_index_map.size()`` (occupied) and ``capacity()`` per tier for Hybrid backing."""
    if not isinstance(storage, HybridStorage):
        return
    hbm_state, host_state = storage.tables
    kim_h, kim_o = hbm_state.key_index_map, host_state.key_index_map
    indices = (
        [only_table_id] if only_table_id is not None else list(range(len(table_names)))
    )
    for tid in indices:
        if tid < 0 or tid >= len(table_names):
            continue
        name = table_names[tid]
        hbm_n, host_n = kim_h.size(tid), kim_o.size(tid)
        hbm_cap, host_cap = kim_h.capacity(tid), kim_o.capacity(tid)
        print(
            f"[bdebt vs stbe] {phase} table {name!r} (table_id={tid}):\n"
            f"  Hybrid HBM:   size()={hbm_n}, capacity={hbm_cap}\n"
            f"  Hybrid host:  size()={host_n}, capacity={host_cap}\n"
            f"  sum size()={hbm_n + host_n}",
            flush=True,
        )


def _bdebt_storage_entry_count_for_table(
    storage: Storage, table_id: int
) -> Tuple[int, str]:
    """Count occupied slots and capacity in backing storage for one logical table (post-flush)."""
    if isinstance(storage, HybridStorage):
        hbm_state, host_state = storage.tables
        kim_h, kim_o = hbm_state.key_index_map, host_state.key_index_map
        hbm_n, host_n = kim_h.size(table_id), kim_o.size(table_id)
        hbm_cap, host_cap = kim_h.capacity(table_id), kim_o.capacity(table_id)
        total = hbm_n + host_n
        return total, (
            f"HybridStorage table_id={table_id}: "
            f"HBM size={hbm_n}, capacity={hbm_cap}; "
            f"host size={host_n}, capacity={host_cap}; sum_size={total}"
        )
    if isinstance(storage, DynamicEmbStorage):
        kim = storage.key_index_map
        n, cap = kim.size(table_id), kim.capacity(table_id)
        return n, f"DynamicEmbStorage table_id={table_id}: size={n}, capacity={cap}"
    return -1, f"unknown storage type {type(storage).__name__}"


def _kim_occupied_sizes_per_table(km: Any, num_tables: int) -> List[int]:
    """``key_index_map.size(t)`` per logical table (handles int or 0-dim tensor)."""
    out: List[int] = []
    for t in range(num_tables):
        n_t = km.size(t)
        out.append(int(n_t.item() if isinstance(n_t, torch.Tensor) else n_t))
    return out


def _kim_capacities_per_table(km: Any, num_tables: int) -> List[int]:
    """``key_index_map.capacity(t)`` per logical table (handles int or 0-dim tensor)."""
    out: List[int] = []
    for t in range(num_tables):
        c_t = km.capacity(t)
        out.append(int(c_t.item() if isinstance(c_t, torch.Tensor) else c_t))
    return out


def _print_test_expansion_storage_cache_sizes_after_train_step(
    bdebt: BatchedDynamicEmbeddingTablesV2,
    *,
    outer_iter: int,
    batch_index: int,
) -> None:
    """Print storage/cache KIM ``size`` and ``capacity`` per table after one forward+backward."""
    tag = (
        f"[test_table_expansion_capacity_growth] outer_iter={outer_iter} "
        f"train_batch={batch_index}"
    )
    storage = bdebt.tables
    if isinstance(storage, DynamicEmbStorage):
        km = storage.key_index_map
        nt = storage._state.num_tables
        sz = _kim_occupied_sizes_per_table(km, nt)
        cap = _kim_capacities_per_table(km, nt)
        print(
            f"{tag} storage KIM size (per table): {sz}, capacity (per table): {cap}",
            flush=True,
        )
    elif isinstance(storage, HybridStorage):
        hbm, host = storage.tables
        hkm, okm = hbm.key_index_map, host.key_index_map
        nt_h, nt_o = hbm.num_tables, host.num_tables
        hs, os_ = _kim_occupied_sizes_per_table(
            hkm, nt_h
        ), _kim_occupied_sizes_per_table(okm, nt_o)
        hc, oc = _kim_capacities_per_table(hkm, nt_h), _kim_capacities_per_table(
            okm, nt_o
        )
        print(
            f"{tag} storage HBM KIM size: {hs}, capacity: {hc}; "
            f"host KIM size: {os_}, capacity: {oc}",
            flush=True,
        )
    else:
        print(f"{tag} storage type: {type(storage).__name__}", flush=True)

    cache = bdebt.cache
    if isinstance(cache, DynamicEmbCache):
        ckm = cache.key_index_map
        nt_c = cache._state.num_tables
        csz = _kim_occupied_sizes_per_table(ckm, nt_c)
        ccap = _kim_capacities_per_table(ckm, nt_c)
        print(
            f"{tag} cache KIM size (per table): {csz}, "
            f"capacity (per table): {ccap}",
            flush=True,
        )
    else:
        print(f"{tag} cache: disabled", flush=True)


def _stbe_size_capacity_for_table(
    stbe: SplitTableBatchedEmbeddingBagsCodegen, logical_table_id: int
) -> Tuple[int, int]:
    """Static TBE: allocated rows (``size``) and declared ``rows_per_table`` (``capacity``).

    For dense split tables both are the number of embedding rows; ``rows_per_table`` comes
    from the first feature mapped to ``logical_table_id``.
    """
    w = stbe.split_embedding_weights()[logical_table_id]
    weight_rows = int(w.shape[0])
    ftm = stbe.feature_table_map
    try:
        f_idx = next(i for i, t in enumerate(ftm) if t == logical_table_id)
        declared = int(stbe.rows_per_table[f_idx].item())
    except (StopIteration, AttributeError):
        declared = weight_rows
    return weight_rows, declared


def _dump_bde_stbe_embedding_samples(
    *,
    table_name: str,
    by_key: Dict[int, torch.Tensor],
    w_stbe: torch.Tensor,
    cmp_cap: int,
    max_keys: int = 5,
) -> None:
    """Print a few key rows for BDE export vs STBE (debug: single-table + caching)."""
    candidates = [0, 1, 2, cmp_cap // 2, cmp_cap - 1]
    seen: List[int] = []
    for k in candidates:
        if k in seen or k < 0 or k >= cmp_cap:
            continue
        seen.append(k)
        if len(seen) >= max_keys:
            break
    print(
        f"[bde vs stbe dump] table {table_name!r} sample embeddings "
        f"(cmp_cap={cmp_cap}, showing up to {len(seen)} keys):",
        flush=True,
    )
    for k in seen:
        stbe_row = w_stbe[k].tolist()
        if k in by_key:
            bde_row = by_key[k].tolist()
            print(
                f"  key={k}\n    BDE export: {bde_row}\n    STBE row:   {stbe_row}",
                flush=True,
            )
        else:
            print(
                f"  key={k}\n    BDE export: <missing>\n    STBE row:   {stbe_row}",
                flush=True,
            )


def _assert_cache_export_matches_lookup_embeddings(
    bdebt: BatchedDynamicEmbeddingTablesV2,
    device: torch.device,
    *,
    batch_size: int = 65536,
) -> None:
    """Check GPU cache: exported (key, slot) rows match ``lookup`` + ``load_from_flat``.

    Uses ``training=False`` on cache state so lookup does not mutate scores.
    No-op when ``bdebt.cache`` is None.
    """
    cache = bdebt.cache
    if not isinstance(cache, DynamicEmbCache):
        return
    state = cache._state
    saved_training = state.training
    cache.training = False
    try:
        for tid in range(state.num_tables):
            for keys, emb_exp, _, _ in export_keys_values_iter(
                state, device, batch_size=batch_size, table_id=tid
            ):
                if keys.numel() == 0:
                    continue
                tids = torch.full(
                    (keys.numel(),), tid, dtype=torch.int64, device=keys.device
                )
                _, founds, indices = cache.lookup(keys, tids, None)
                assert bool(
                    founds.all()
                ), f"cache table_id={tid}: every exported key must be found via lookup"
                emb_slots = load_from_flat(
                    state, indices, tids, copy_mode=CopyMode.EMBEDDING
                )
                torch.testing.assert_close(
                    emb_exp.to(dtype=emb_slots.dtype),
                    emb_slots,
                    rtol=1e-5,
                    atol=1e-5,
                )
    finally:
        cache.training = saved_training


def _assert_storage_export_matches_find_embeddings(
    bdebt: BatchedDynamicEmbeddingTablesV2,
    device: torch.device,
    *,
    batch_size: int = 65536,
) -> None:
    """After flush/insert into backing ``DynamicEmbStorage``, check key↔embedding consistency.

    For each export batch, ``find(keys)`` must return the same embeddings as
    ``export_keys_values_iter`` (read-only lookup via ``training=False``).
    If the table uses ``DynamicEmbInitializerMode.DEBUG``, also checks that both
    export and find embeddings equal ``key % DEBUG_EMB_INITIALIZER_MOD`` per dim.
    Skips non-``DynamicEmbStorage`` backends (e.g. ``HybridStorage``).
    """
    storage = bdebt.tables
    if not isinstance(storage, DynamicEmbStorage):
        return
    state = storage._state
    saved_training = state.training
    storage.training = False
    try:
        for tid in range(state.num_tables):
            debug_table = (
                state.options_list[tid].initializer_args.mode
                == DynamicEmbInitializerMode.DEBUG
            )
            emb_dim_t = state.table_emb_dims_cpu[tid]
            for keys, emb_exp, _, _ in storage.export_keys_values(
                device, batch_size, tid
            ):
                if keys.numel() == 0:
                    continue
                tids = torch.full(
                    (keys.numel(),), tid, dtype=torch.int64, device=keys.device
                )
                (
                    h_miss,
                    _,
                    _,
                    _,
                    _,
                    founds,
                    _,
                    emb_find,
                ) = storage.find(keys, tids, CopyMode.EMBEDDING)
                assert h_miss == 0, (
                    f"storage table_id={tid}: find reports {h_miss} missing keys "
                    "that export still lists"
                )
                assert bool(
                    founds.all()
                ), f"storage table_id={tid}: find must locate every exported key"
                expected_dbg: Optional[torch.Tensor] = None
                if debug_table:
                    mod = torch.tensor(
                        float(DEBUG_EMB_INITIALIZER_MOD),
                        device=keys.device,
                        dtype=torch.float32,
                    )
                    kf = keys.to(dtype=torch.float32)
                    expected_dbg = (kf % mod).unsqueeze(1).expand(-1, emb_dim_t)
                if os.environ.get("DYNAMICEMB_PRINT_STORAGE_EXPORT_FIND") == "1":
                    ze = emb_exp.to(dtype=emb_find.dtype)
                    parts = [
                        "[storage export vs find]",
                        f"table_id={tid}",
                        f"batch_n={keys.numel()}",
                        f"max_abs(exp-find)={(ze - emb_find).abs().max().item()}",
                    ]
                    if expected_dbg is not None:
                        ee32 = emb_exp.to(device=keys.device, dtype=torch.float32)
                        ef32 = emb_find.to(device=keys.device, dtype=torch.float32)
                        parts.append(
                            f"max_abs(exp-key%mod)={(ee32 - expected_dbg).abs().max().item()}"
                        )
                        parts.append(
                            f"max_abs(find-key%mod)={(ef32 - expected_dbg).abs().max().item()}"
                        )
                    print(*parts, flush=True)

                torch.testing.assert_close(
                    emb_exp.to(dtype=emb_find.dtype),
                    emb_find,
                    rtol=1e-5,
                    atol=1e-5,
                )
                torch.testing.assert_close(
                    emb_find.to(dtype=emb_find.dtype),
                    expected_dbg,
                    rtol=1e-5,
                    atol=1e-5,
                )
                torch.testing.assert_close(
                    emb_exp.to(dtype=emb_find.dtype),
                    expected_dbg,
                    rtol=1e-5,
                    atol=1e-5,
                )

                if debug_table:
                    assert expected_dbg is not None
                    torch.testing.assert_close(
                        emb_exp.to(device=keys.device, dtype=torch.float32),
                        expected_dbg,
                        rtol=0.0,
                        atol=0.0,
                        msg=lambda m: (
                            f"storage table_id={tid}: export embedding != key % "
                            f"{DEBUG_EMB_INITIALIZER_MOD}: {m}"
                        ),
                    )
                    torch.testing.assert_close(
                        emb_find.to(device=keys.device, dtype=torch.float32),
                        expected_dbg,
                        rtol=0.0,
                        atol=0.0,
                        msg=lambda m: (
                            f"storage table_id={tid}: find embedding != key % "
                            f"{DEBUG_EMB_INITIALIZER_MOD}: {m}"
                        ),
                    )
    finally:
        storage.training = saved_training


def _kim_scan_keys_embeddings_dict(
    state: DynamicEmbTableState,
    device: torch.device,
    *,
    batch_size: int = 65536,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Scan KIM + value buffer: (table_id, key) -> embedding.

    Does not call ``Storage.export_keys_values`` / ``export_keys_values_iter``;
    uses ``_batched_export_keys_scores`` and :func:`load_from_flat_single_table` only.
    """
    out: Dict[Tuple[int, int], torch.Tensor] = {}
    score_name = state.score_policy.name
    for tid in range(state.num_tables):
        emb_dim_t = state.table_emb_dims_cpu[tid]
        for (
            keys,
            named_scores,
            indices,
        ) in state.key_index_map._batched_export_keys_scores(
            [score_name],
            state.device,
            batch_size=batch_size,
            return_index=True,
            table_id=tid,
        ):
            if keys.numel() == 0:
                continue
            scores = named_scores[score_name]
            flat_rows = _flat_row_indices_from_slots_and_scores(state, indices, scores)
            values = load_from_flat_single_table(state, flat_rows, tid)
            embeddings = (
                values[:, :emb_dim_t].to(dtype=EMBEDDING_TYPE).contiguous().to(device)
            )
            keys_dev = keys.to(device=device)
            for i in range(keys.numel()):
                k_int = int(keys_dev[i].item())
                out[(tid, k_int)] = embeddings[i].detach().clone()
    return out


def _assert_bdebt_storage_cache_merged_export_matches_lookup(
    bdebt: BatchedDynamicEmbeddingTablesV2,
    device: torch.device,
    *,
    batch_size: int = 65536,
    verify_batch_size: int = 4096,
) -> None:
    """No flush: merge storage then cache embeddings into a dict; check vs live lookup/find.

    - Fills a dict from backing ``DynamicEmbStorage`` KIM scan, then overlays cache scan
      (cache wins on key collision).
    - Verifies every merged (table_id, key) matches ``cache.lookup`` + ``load_from_flat``
      when present in cache, else ``storage.find`` (embedding column).
    """
    storage = bdebt.tables
    cache = bdebt.cache
    if not isinstance(storage, DynamicEmbStorage) or not isinstance(
        cache, DynamicEmbCache
    ):
        return

    saved_st = storage.training
    saved_ca = cache.training
    storage.training = False
    cache.training = False
    try:
        storage_dict = _kim_scan_keys_embeddings_dict(
            storage._state, device, batch_size=batch_size
        )
        cache_dict = _kim_scan_keys_embeddings_dict(
            cache._state, device, batch_size=batch_size
        )
        merged: Dict[Tuple[int, int], torch.Tensor] = {**storage_dict, **cache_dict}

        for k, v in cache_dict.items():
            assert k in merged, f"cache key {k} missing from merged dict"
            torch.testing.assert_close(merged[k], v, rtol=1e-5, atol=1e-5)
        for k, v in storage_dict.items():
            if k in cache_dict:
                torch.testing.assert_close(
                    merged[k], cache_dict[k], rtol=1e-5, atol=1e-5
                )
            else:
                torch.testing.assert_close(merged[k], v, rtol=1e-5, atol=1e-5)

        st = storage._state
        ca = cache._state
        key_type = st.key_index_map.key_type
        by_tid: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
        for (tid, k_int), emb in merged.items():
            by_tid.setdefault(tid, []).append((k_int, emb))

        for tid, pairs in by_tid.items():
            for s in range(0, len(pairs), verify_batch_size):
                chunk = pairs[s : s + verify_batch_size]
                ks = [p[0] for p in chunk]
                ref = torch.stack([p[1] for p in chunk])
                key_t = torch.tensor(ks, device=device, dtype=key_type)
                tid_t = torch.full((len(ks),), tid, dtype=torch.int64, device=device)
                _, founds_c, ix_c = cache.lookup(key_t, tid_t, None)
                emb_live = torch.empty_like(ref)
                if bool(founds_c.any()):
                    emb_live[founds_c] = load_from_flat(
                        ca,
                        ix_c[founds_c],
                        tid_t[founds_c],
                        copy_mode=CopyMode.EMBEDDING,
                    )
                miss = ~founds_c
                if bool(miss.any()):
                    (
                        h_miss,
                        _,
                        _,
                        _,
                        _,
                        sf,
                        _,
                        sv,
                    ) = storage.find(
                        key_t[miss],
                        tid_t[miss],
                        copy_mode=CopyMode.EMBEDDING,
                    )
                    assert h_miss == 0, (
                        f"table_id={tid}: storage.find reports {h_miss} missing "
                        f"among keys not in cache"
                    )
                    assert bool(
                        sf.all()
                    ), f"table_id={tid}: storage.find must hit all keys absent from cache"
                    emb_live[miss] = sv
                torch.testing.assert_close(ref, emb_live, rtol=1e-5, atol=1e-5)
    finally:
        storage.training = saved_st
        cache.training = saved_ca


def _compare_bdebt_export_to_stbe_weights(
    bdebt: BatchedDynamicEmbeddingTablesV2,
    stbe: SplitTableBatchedEmbeddingBagsCodegen,
    table_names: List[str],
    cmp_caps: List[int],
    device: torch.device,
    *,
    dump_sample_embeddings: bool = False,
) -> None:
    # Match export_keys_values: flush so storage counts reflect the same backing state.
    bdebt.flush()
    torch.cuda.synchronize()
    # Key↔embedding mapping: cache tier (still hot after flush) then backing storage.
    _assert_cache_export_matches_lookup_embeddings(bdebt, device, batch_size=65536)
    _assert_storage_export_matches_find_embeddings(bdebt, device, batch_size=65536)
    _print_hybrid_hbm_host_sizes(
        bdebt.tables,
        table_names,
        phase="Hybrid tier sizes (after flush, before export)",
    )
    for tid, name in enumerate(table_names):
        _, st_detail = _bdebt_storage_entry_count_for_table(bdebt.tables, tid)
        if bdebt.cache is not None:
            ckim = bdebt.cache.key_index_map
            cache_sz, cache_cap = ckim.size(tid), ckim.capacity(tid)
            cache_line = (
                f"  Cache:    table_id={tid} size={cache_sz}, capacity={cache_cap}\n"
            )
        else:
            cache_line = "  Cache:    (none)\n"
        stbe_sz, stbe_cap = _stbe_size_capacity_for_table(stbe, tid)
        print(
            f"[bdebt vs stbe] table {name!r} (after flush, before export):\n"
            f"{cache_line}"
            f"  Storage:  {st_detail}\n"
            f"  STBE:     size={stbe_sz}, capacity={stbe_cap} "
            f"(split_embedding_weights rows={stbe_sz}, rows_per_table={stbe_cap})\n"
            f"  cmp_cap={cmp_caps[tid]}",
            flush=True,
        )

    for tid, name in enumerate(table_names):
        cmp_cap = cmp_caps[tid]
        keys_t, emb_t = bdebt.export_keys_values(name, device, batch_size=65536)
        exported_n = int(keys_t.numel())
        stbe_sz, stbe_cap = _stbe_size_capacity_for_table(stbe, tid)
        print(
            f"[bdebt vs stbe] table {name!r} (after export_keys_values):\n"
            f"  BDE export: size={exported_n} (unique keys in backing export)\n"
            f"  STBE:       size={stbe_sz}, capacity={stbe_cap}\n"
            f"  cmp_cap={cmp_cap}",
            flush=True,
        )
        _print_hybrid_hbm_host_sizes(
            bdebt.tables,
            table_names,
            phase="Hybrid tier sizes (after export_keys_values)",
            only_table_id=tid,
        )
        by_key: Dict[int, torch.Tensor] = {}
        for i in range(keys_t.numel()):
            by_key[int(keys_t[i].item())] = emb_t[i].detach().cpu()
        w = stbe.split_embedding_weights()[tid].detach().cpu()
        if dump_sample_embeddings:
            _dump_bde_stbe_embedding_samples(
                table_name=name,
                by_key=by_key,
                w_stbe=w,
                cmp_cap=cmp_cap,
                max_keys=5,
            )
        for k in range(cmp_cap):
            assert k in by_key, f"missing key {k} in exported table {name}"
            torch.testing.assert_close(by_key[k], w[k], rtol=1e-5, atol=1e-5)


def _assert_expected_storage_backend(
    bdebt: BatchedDynamicEmbeddingTablesV2,
    *,
    caching: bool,
    hbm_budget_ratio: Optional[float],
) -> None:
    st = bdebt.tables
    if caching:
        assert bdebt.cache is not None
        assert isinstance(st, DynamicEmbStorage)
    else:
        assert bdebt.cache is None
        if hbm_budget_ratio is not None and hbm_budget_ratio < 1.0:
            assert isinstance(st, HybridStorage)
        else:
            assert isinstance(st, DynamicEmbStorage)


def _split_table_batched_optimizer_for_compare(opt_type: EmbOptimType) -> EmbOptimType:
    """Optimizer passed to SplitTableBatchedEmbeddingBagsCodegen for TorchREC comparison.

    DynamicEmb may use ``SGD``; STBE uses ``EXACT_SGD`` so updates match for the test.
    All other optimizers match ``opt_type``."""
    return EmbOptimType.EXACT_SGD if opt_type == EmbOptimType.SGD else opt_type


class PyDictStorage(Storage[DynamicEmbTableOptions, BaseDynamicEmbeddingOptimizer]):
    def __init__(
        self,
        options: List[DynamicEmbTableOptions],
        optimizer: BaseDynamicEmbeddingOptimizer,
    ):
        self.options = options
        self.dict: Dict[Tuple[int, int], torch.Tensor] = {}
        self.scores: Dict[Tuple[int, int], int] = {}
        self.capacity = max(o.max_capacity for o in options)
        self.optimizer = optimizer

        self._emb_dims = [o.dim for o in options]
        self._emb_dtype = options[0].embedding_dtype
        self._value_dims = [o.dim + optimizer.get_state_dim(o.dim) for o in options]
        self._optstate_dims = [optimizer.get_state_dim(o.dim) for o in options]
        self._max_emb_dim = max(self._emb_dims)
        self._max_value_dim = max(self._value_dims)
        self._max_optstate_dim = max(self._optstate_dims)
        self._initial_optim_state = optimizer.get_initial_optim_states()

        device_idx = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{device_idx}")

    def size(self) -> int:
        return len(self.dict)

    def find_impl(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        unique_embs: torch.Tensor,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        h_unique_keys = unique_keys.cpu()
        h_table_ids = table_ids.cpu()
        lookup_dim = unique_embs.size(1)
        results = []
        missing_keys = []
        missing_indices = []
        missing_scores_list = []
        founds_ = []
        for i in range(h_unique_keys.size(0)):
            key = h_unique_keys[i].item()
            tid = h_table_ids[i].item()
            composite_key = (tid, key)
            if composite_key in self.dict:
                results.append(self.dict[composite_key][0:lookup_dim])
                founds_.append(True)
            else:
                missing_keys.append(key)
                missing_indices.append(i)
                if input_scores is not None:
                    missing_scores_list.append(input_scores[i].item())
                founds_.append(False)
        founds_ = torch.tensor(founds_, dtype=torch.bool, device=self.device)
        if len(results) > 0:
            unique_embs[founds_, :] = torch.cat(
                [t.unsqueeze(0) for t in results], dim=0
            )

        num_missing = torch.tensor(
            [len(missing_keys)], dtype=torch.long, device=self.device
        )
        missing_keys = torch.tensor(
            missing_keys, dtype=unique_keys.dtype, device=self.device
        )
        missing_indices = torch.tensor(
            missing_indices, dtype=torch.long, device=self.device
        )

        if input_scores is not None and len(missing_scores_list) > 0:
            missing_scores = torch.tensor(
                missing_scores_list, dtype=input_scores.dtype, device=self.device
            )
        else:
            missing_scores = torch.empty(0, dtype=torch.uint64, device=self.device)

        # output_scores: scores for all keys (0 for missing, input_scores for found)
        output_scores = torch.zeros(
            unique_keys.size(0), dtype=torch.int64, device=self.device
        )
        if input_scores is not None:
            output_scores[founds_] = input_scores[founds_].to(torch.int64)

        return (
            num_missing,
            missing_keys,
            missing_indices,
            missing_scores,
            founds_,
            output_scores,
        )

    def find_embeddings(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        unique_embs: torch.Tensor,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[
        int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        return self.find_impl(unique_keys, table_ids, unique_embs, input_scores)

    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
        *,
        find_debug_context: Optional[str] = None,
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if copy_mode == CopyMode.EMBEDDING:
            out_dim = self._max_emb_dim
        else:
            out_dim = self._max_value_dim
        unique_vals = torch.zeros(
            unique_keys.size(0), out_dim, dtype=self._emb_dtype, device=self.device
        )
        (
            num_missing,
            missing_keys,
            missing_indices,
            missing_scores,
            founds,
            output_scores,
        ) = self.find_impl(
            unique_keys, table_ids, unique_vals, lfu_accumulated_frequency
        )

        h_table_ids = table_ids.cpu()
        missing_table_ids_list = []
        for idx in missing_indices.cpu().tolist():
            missing_table_ids_list.append(h_table_ids[idx].item())
        missing_table_ids = torch.tensor(
            missing_table_ids_list, dtype=torch.int64, device=self.device
        )

        return (
            num_missing.item()
            if isinstance(num_missing, torch.Tensor)
            else num_missing,
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores if missing_scores.numel() > 0 else None,
            founds,
            output_scores,
            unique_vals,
        )

    def insert(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        preserve_existing: bool = False,
    ) -> None:
        h_keys = keys.cpu()
        h_table_ids = table_ids.cpu()
        h_scores = scores.cpu() if scores is not None and scores.numel() > 0 else None
        for i in range(h_keys.size(0)):
            key = h_keys[i].item()
            tid = h_table_ids[i].item()
            composite_key = (tid, key)
            self.dict[composite_key] = values[i, :].clone()
            if h_scores is not None:
                self.scores[composite_key] = h_scores[i].item()

    def update(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        grads: torch.Tensor,
        return_missing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise ValueError("Can't call update of PyDictStorage")

    def enable_update(self) -> bool:
        return False

    def dump(
        self,
        table_id: int,
        meta_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool = False,
        include_meta: bool = False,
        timestamp: int = 0,
    ) -> None:
        if include_meta:
            meta_data = {}
            meta_data.update(self.optimizer.get_opt_args())
            with open(meta_file_path, "w") as f:
                json.dump(meta_data, f)

        fkey = open(emb_key_path, "wb")
        fembedding = open(embedding_file_path, "wb")
        fscore = open(score_file_path, "wb") if score_file_path else None
        fopt_states = open(opt_file_path, "wb") if include_optim else None

        for keys, embeddings, opt_states, scores_out in self.export_keys_values(
            table_id, self.device
        ):
            fkey.write(keys.cpu().numpy().tobytes())
            if fscore is not None:
                fscore.write(scores_out.cpu().numpy().tobytes())
            fembedding.write(embeddings.cpu().numpy().tobytes())
            if fopt_states is not None and opt_states is not None:
                to_write = truncate_optimizer_states_for_checkpoint(
                    self.optimizer,
                    self._emb_dims[table_id],
                    opt_states,
                )
                fopt_states.write(to_write.cpu().numpy().tobytes())

        fkey.close()
        fembedding.close()
        if fscore:
            fscore.close()
        if fopt_states:
            fopt_states.close()

    def load(
        self,
        table_id: int,
        meta_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool = False,
        timestamp: int = 0,
    ) -> None:
        if meta_file_path and os.path.exists(meta_file_path):
            with open(meta_file_path, "r") as f:
                meta_data = json.load(f)
            opt_type = meta_data.get("opt_type", None)
            if (
                opt_type
                and self.optimizer.get_opt_args().get("opt_type", None) != opt_type
            ):
                include_optim = False
            if include_optim:
                self.optimizer.set_opt_args(meta_data)

        if not opt_file_path or not os.path.exists(opt_file_path):
            include_optim = False

        dim = self._emb_dims[table_id]
        runtime_optstate_dim = self._optstate_dims[table_id]
        value_dim = self._value_dims[table_id]
        ckpt_dim = self.optimizer.get_ckpt_state_dim(dim)

        num_keys = os.path.getsize(emb_key_path) // 8  # int64

        file_opt_dim = 0
        if (
            include_optim
            and opt_file_path
            and os.path.exists(opt_file_path)
            and runtime_optstate_dim > 0
        ):
            if num_keys == 0:
                if os.path.getsize(opt_file_path) != 0:
                    raise ValueError("Non-empty optimizer file but zero keys.")
            else:
                opt_sz = os.path.getsize(opt_file_path)
                row_block = 4 * num_keys
                if opt_sz % row_block != 0:
                    raise ValueError(
                        f"Optimizer file size {opt_sz} not divisible by {row_block}."
                    )
                file_opt_dim = opt_sz // row_block
                if file_opt_dim != ckpt_dim:
                    raise ValueError(
                        f"Checkpoint optimizer width {file_opt_dim}; expected {ckpt_dim}."
                    )

        fkey = open(emb_key_path, "rb")
        fembedding = open(embedding_file_path, "rb")
        fscore = (
            open(score_file_path, "rb")
            if score_file_path and os.path.exists(score_file_path)
            else None
        )
        fopt_states = open(opt_file_path, "rb") if include_optim else None

        batch_size = 65536
        for start in range(0, num_keys, batch_size):
            n = min(num_keys - start, batch_size)

            keys_bytes = fkey.read(8 * n)
            keys = torch.tensor(
                np.frombuffer(keys_bytes, dtype=np.int64).copy(),
                dtype=torch.int64,
                device=self.device,
            )

            emb_bytes = fembedding.read(4 * dim * n)
            embeddings = torch.tensor(
                np.frombuffer(emb_bytes, dtype=np.float32).copy(),
                dtype=torch.float32,
                device=self.device,
            ).view(-1, dim)

            opt_states = None
            if fopt_states is not None and file_opt_dim > 0 and n > 0:
                opt_bytes = fopt_states.read(4 * file_opt_dim * n)
                opt_states = torch.tensor(
                    np.frombuffer(opt_bytes, dtype=np.float32).copy(),
                    dtype=torch.float32,
                    device=self.device,
                ).view(-1, file_opt_dim)
                opt_states = pad_optimizer_states_from_checkpoint(
                    self.optimizer,
                    dim,
                    opt_states,
                    self._initial_optim_state,
                    torch.float32,
                    self.device,
                )

            scores = None
            if fscore:
                score_bytes = fscore.read(8 * n)
                scores = torch.tensor(
                    np.frombuffer(score_bytes, dtype=np.int64).copy(),
                    dtype=torch.int64,
                    device=self.device,
                )

            # Build full values tensor [N, value_dim]
            if opt_states is not None:
                values = torch.cat([embeddings, opt_states], dim=1)
            else:
                if value_dim > dim:
                    values = torch.empty(
                        n, value_dim, dtype=torch.float32, device=self.device
                    )
                    values[:, :dim] = embeddings
                    values[:, dim:] = self._initial_optim_state
                else:
                    values = embeddings

            tid = torch.full(
                (keys.numel(),), table_id, dtype=torch.int64, device=keys.device
            )
            self.insert(keys, tid, values, scores)

        fkey.close()
        fembedding.close()
        if fscore:
            fscore.close()
        if fopt_states:
            fopt_states.close()

    def export_keys_values(
        self,
        table_id: int,
        device: torch.device,
        batch_size: int = 65536,
    ) -> Iterator[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]
    ]:
        """Yield (keys, embeddings, opt_states, scores) batches for a given table."""
        emb_dim = self._emb_dims[table_id]
        optstate_dim = self._optstate_dims[table_id]
        all_keys = [ck for ck in self.dict.keys() if ck[0] == table_id]
        for start in range(0, len(all_keys), batch_size):
            batch_keys = all_keys[start : start + batch_size]

            keys_t = torch.tensor(
                [ck[1] for ck in batch_keys], dtype=torch.int64, device=device
            )
            values_t = torch.stack(
                [self.dict[ck].to(device) for ck in batch_keys], dim=0
            )
            embeddings = values_t[:, :emb_dim].contiguous()

            opt_states = None
            if optstate_dim > 0:
                opt_states = values_t[:, emb_dim : emb_dim + optstate_dim].contiguous()

            scores_list = [self.scores.get(ck, 0) for ck in batch_keys]
            scores_t = torch.tensor(scores_list, dtype=torch.int64, device=device)

            yield keys_t, embeddings, opt_states, scores_t

    def embedding_dtype(
        self,
    ) -> torch.dtype:
        return self._emb_dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._emb_dims[table_id]

    def value_dim(self, table_id: int) -> int:
        return self._value_dims[table_id]

    def max_embedding_dim(self) -> int:
        return self._max_emb_dim

    def max_value_dim(self) -> int:
        return self._max_value_dim

    def init_optimizer_state(
        self,
    ) -> float:
        return self._initial_optim_state


def create_split_table_batched_embedding(
    table_names,
    feature_table_map,
    optimizer_type,
    opt_params,
    dims,
    num_embs,
    pooling_mode,
    device,
):
    emb = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                e,
                d,
                EmbeddingLocation.DEVICE,
                ComputeDevice.CUDA,
            )
            for (e, d) in zip(num_embs, dims)
        ],
        optimizer=optimizer_type,
        weights_precision=SparseType.FP32,
        stochastic_rounding=False,
        pooling_mode=pooling_mode,
        output_dtype=SparseType.FP32,
        device=device,
        table_names=table_names,
        feature_table_map=feature_table_map,
        **opt_params,
        bounds_check_mode=BoundsCheckMode.FATAL,
    ).cuda()
    return emb


def init_embedding_tables(stbe, bdet):
    stbe.init_embedding_weights_uniform(0, 1)
    storage = bdet.tables
    optimizer = bdet.optimizer
    for table_idx, split in enumerate(stbe.split_embedding_weights()):
        num_emb = split.size(0)
        emb_dim = split.size(1)
        opt_state_dim = optimizer.get_state_dim(emb_dim)
        val_dim = emb_dim + opt_state_dim
        indices = torch.arange(num_emb, device=split.device, dtype=torch.long)
        table_ids = torch.full(
            (num_emb,), table_idx, dtype=torch.int64, device=split.device
        )
        if isinstance(storage, DynamicEmbStorage):
            max_emb_dim = storage.max_embedding_dim()
            max_value_dim = storage.max_value_dim()
            values = torch.zeros(
                num_emb, max_value_dim, dtype=split.dtype, device=split.device
            )
            values[:, :emb_dim] = split
            if opt_state_dim > 0:
                values[
                    :, max_emb_dim : max_emb_dim + opt_state_dim
                ] = storage.init_optimizer_state()
            storage.set_score(1)
            storage.insert(indices, table_ids, values)
        elif isinstance(storage, PyDictStorage):
            pydict = cast(PyDictStorage, storage)
            values = torch.empty(
                num_emb, val_dim, dtype=split.dtype, device=split.device
            )
            values[:, :emb_dim] = split
            if val_dim > emb_dim:
                values[:, emb_dim:] = pydict.init_optimizer_state()
            pydict.insert(indices, table_ids, values)
        else:
            raise ValueError("Not support table type")


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.ADAM,
            {
                "learning_rate": 0.3,
                "weight_decay": 0.06,
                "eps": 3e-5,
                "beta1": 0.8,
                "beta2": 0.888,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "deterministic",
    [True, False],
)
@pytest.mark.parametrize(
    "caching, PS, local_hbm_for_values",
    [
        (True, None, 1024),  # Caching + DynamicEmbStorage backend
        (True, PyDictStorage, 1024),  # Caching + PS backend
        (False, None, 1024**3),  # HBM-only
        (False, None, 1024),  # HybridStorage
        (False, None, 0),  # No HBM budget, no cache -> DynamicEmbStorage (host-only)
    ],
)
@pytest.mark.parametrize(
    "pooling_mode, dims",
    [
        (DynamicEmbPoolingMode.NONE, [8, 8, 8]),
        (DynamicEmbPoolingMode.NONE, [7, 7, 7]),
        (DynamicEmbPoolingMode.SUM, [8, 8, 8]),
        (DynamicEmbPoolingMode.MEAN, [8, 8, 8]),
        (DynamicEmbPoolingMode.SUM, [8, 16, 32]),
        (DynamicEmbPoolingMode.MEAN, [8, 16, 32]),
        (DynamicEmbPoolingMode.SUM, [7, 11, 13]),
        (DynamicEmbPoolingMode.MEAN, [7, 11, 13]),
    ],
)
def test_forward_train_eval(
    opt_type,
    opt_params,
    caching,
    deterministic,
    PS,
    local_hbm_for_values,
    pooling_mode,
    dims,
):
    print(
        f"step in test_forward_train_eval , opt_type = {opt_type} opt_params = {opt_params}"
        f" caching = {caching} PS = {PS} local_hbm_for_values = {local_hbm_for_values}"
        f" pooling_mode = {pooling_mode} dims = {dims}"
    )

    if deterministic:
        os.environ["DEMB_DETERMINISM_MODE"] = "ON"

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    table_names = ["table0", "table1", "table2"]
    feature_table_map = [0, 0, 1, 2]
    key_type = torch.int64
    value_type = torch.float32

    max_capacity = 2048

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=local_hbm_for_values,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    bdebt = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=pooling_mode,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )

    if caching:
        assert bdebt._cache is not None, "Caching mode should create a cache"
    elif local_hbm_for_values == 1024**3:
        assert bdebt._cache is None, "HBM-only mode should have no cache"
        assert isinstance(
            bdebt._storage, DynamicEmbStorage
        ), f"Expected DynamicEmbStorage, got {type(bdebt._storage)}"
    elif local_hbm_for_values == 0:
        assert bdebt._cache is None, "Host-only mode should have no cache"
        assert isinstance(
            bdebt._storage, DynamicEmbStorage
        ), f"Expected DynamicEmbStorage when local_hbm=0 without cache, got {type(bdebt._storage)}"
    else:
        assert bdebt._cache is None, "HybridStorage mode should have no cache"
        assert isinstance(
            bdebt._storage, HybridStorage
        ), f"Expected HybridStorage, got {type(bdebt._storage)}"

    """
    feature number = 4, batch size = 2

    f0  [0,1],      [12],
    f1  [64,8],     [12],
    f2  [15, 2],    [7,105],
    f3  [],         [0]
    """
    indices = torch.tensor(
        [0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], dtype=key_type, device=device
    )
    offsets = torch.tensor(
        [0, 2, 3, 5, 6, 8, 10, 10, 11], dtype=key_type, device=device
    )
    batch_size = 2

    embs_train = bdebt(indices, offsets)
    torch.cuda.synchronize()

    # Verify output shape
    if pooling_mode == DynamicEmbPoolingMode.NONE:
        assert embs_train.shape == (indices.numel(), dims[0])
    else:
        total_D = sum(dims[feature_table_map[f]] for f in range(len(feature_table_map)))
        assert embs_train.shape == (batch_size, total_D)

    with torch.no_grad():
        bdebt.eval()
        embs_eval = bdebt(indices, offsets)

    # corner case when all keys missed in eval.
    indices_ne_all = indices + 1024
    bdebt(indices_ne_all, offsets)

    torch.cuda.synchronize()

    # Train and eval should produce identical results for the same keys
    torch.testing.assert_close(embs_train, embs_eval)

    # non-exist key: replace index 0 (key=0) with key=777
    indices_ne = torch.tensor(
        [777, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device
    ).to(key_type)
    offsets_ne = torch.tensor([0, 2, 3, 5, 6, 8, 10, 10, 11], device=device).to(
        key_type
    )
    embs_non_exist = bdebt(indices_ne, offsets_ne)
    torch.cuda.synchronize()

    # train
    bdebt.train()
    embs_train_non_exist = bdebt(indices_ne, offsets_ne)
    torch.cuda.synchronize()

    if pooling_mode == DynamicEmbPoolingMode.NONE:
        # Sequence mode: row 0 is the embedding for key 777
        # In eval, non-exist key -> zero embedding
        torch.testing.assert_close(embs_train[1:, :], embs_non_exist[1:, :])
        assert torch.all(embs_non_exist[0, :] == 0)
        # In train, non-exist key gets initialized -> non-zero
        assert torch.all(embs_train_non_exist[0, :] != 0)
        torch.testing.assert_close(embs_train_non_exist[1:, :], embs_non_exist[1:, :])
    else:
        # Pooled mode: sample 1 is unaffected by the non-exist key (key 777
        # only appears in sample 0's f0 bag).
        torch.testing.assert_close(embs_non_exist[1, :], embs_train[1, :])
        torch.testing.assert_close(embs_train_non_exist[1, :], embs_non_exist[1, :])
        # Sample 0 should differ from the original because key 777 replaced
        # key 0 in f0's bag.  In eval the missing key contributes zero, so
        # the pooled result for sample 0 changes compared to embs_train.
        assert not torch.allclose(embs_non_exist[0, :], embs_train[0, :])

    if deterministic:
        del os.environ["DEMB_DETERMINISM_MODE"]

    print("all check passed")


"""
For torchrec's adam optimizer, it will increment the optimizer_step in every forward,
    which will affect the weights update, pay attention to it or try to use `set_optimizer_step()`
    to control(not verified) it.
"""


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.ADAM,
            {
                "learning_rate": 0.3,
                "weight_decay": 0.06,
                "eps": 3e-5,
                "beta1": 0.8,
                "beta2": 0.888,
            },
        ),
        (
            EmbOptimType.EXACT_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
        (
            EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "caching, pooling_mode, dims",
    [
        (True, DynamicEmbPoolingMode.NONE, [8, 8, 8]),
        (False, DynamicEmbPoolingMode.NONE, [16, 16, 16]),
        (False, DynamicEmbPoolingMode.NONE, [17, 17, 17]),
        (False, DynamicEmbPoolingMode.SUM, [128, 32, 16]),
        (False, DynamicEmbPoolingMode.MEAN, [4, 8, 16]),
    ],
)
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("PS", [None, PyDictStorage])
def test_backward(opt_type, opt_params, caching, pooling_mode, dims, deterministic, PS):
    print(f"step in test_backward , opt_type = {opt_type} opt_params = {opt_params}")

    if deterministic:
        os.environ["DEMB_DETERMINISM_MODE"] = "ON"

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    table_names = ["table0", "table1", "table2"]
    key_type = torch.int64
    value_type = torch.float32

    max_capacity = 2048

    dyn_emb_table_options_list = []
    cmp_with_torchrec = True
    for dim in dims:
        if dim % 4 != 0:
            cmp_with_torchrec = False
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=1024**3,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    feature_table_map = [0, 0, 1, 2]
    bdeb = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=pooling_mode,
        optimizer=opt_type,
        **opt_params,
    )
    num_embs = [max_capacity // 2 for d in dims]

    if cmp_with_torchrec:
        stbe = create_split_table_batched_embedding(
            table_names,
            feature_table_map,
            _split_table_batched_optimizer_for_compare(opt_type),
            opt_params,
            dims,
            num_embs,
            POOLING_MODE[pooling_mode],
            device,
        )
        init_embedding_tables(stbe, bdeb)
        """
        feature number = 4, batch size = 2

        f0  [0,1],      [12],
        f1  [64,8],     [12],
        f2  [15, 2, 7], [105],
        f3  [],         [0]
        """
        for i in range(10):
            indices = torch.tensor(
                [0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device
            ).to(key_type)
            offsets = torch.tensor([0, 2, 3, 5, 6, 9, 10, 10, 11], device=device).to(
                key_type
            )

            embs_bdeb = bdeb(indices, offsets)
            embs_stbe = stbe(indices, offsets)

            torch.cuda.synchronize()
            with torch.no_grad():
                torch.testing.assert_close(embs_bdeb, embs_stbe, rtol=1e-06, atol=1e-06)

            loss = embs_bdeb.mean()
            loss.backward()
            loss_stbe = embs_stbe.mean()
            loss_stbe.backward()

            torch.cuda.synchronize()
            torch.testing.assert_close(loss, loss_stbe)

            print(f"Passed iteration {i}")
    else:
        # This scenario will not test correctness, but rather test whether it functions correctly.
        for i in range(10):
            indices = torch.tensor(
                [0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device
            ).to(key_type)
            offsets = torch.tensor([0, 2, 3, 5, 6, 9, 10, 10, 11], device=device).to(
                key_type
            )

            embs_bdeb = bdeb(indices, offsets)
            loss = embs_bdeb.mean()
            loss.backward()

            torch.cuda.synchronize()

            print(f"Passed iteration {i}")

    if deterministic:
        del os.environ["DEMB_DETERMINISM_MODE"]


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.ADAM,
            {
                "learning_rate": 0.3,
                "weight_decay": 0.06,
                "eps": 3e-5,
                "beta1": 0.8,
                "beta2": 0.888,
            },
        ),
        (
            EmbOptimType.EXACT_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
        (
            EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
    ],
)
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("PS", [None, PyDictStorage])
def test_prefetch_flush_in_cache(opt_type, opt_params, deterministic, PS):
    print(
        f"step in test_prefetch_flush , opt_type = {opt_type} opt_params = {opt_params}"
    )
    if deterministic:
        os.environ["DEMB_DETERMINISM_MODE"] = "ON"

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    table_names = ["table0", "table1", "table2"]
    key_type = torch.int64
    value_type = torch.float32

    max_capacity = 2048
    dims = [8, 8, 8]

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.STEP,
            caching=True,
            local_hbm_for_values=1024**3,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    feature_table_map = [0, 0, 1, 2]
    bdeb = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        enable_prefetch=False,
        **opt_params,
    )
    bdeb.enable_prefetch = True
    bdeb.set_record_cache_metrics(True)

    num_embs = [max_capacity // 2 for d in dims]
    stbe = create_split_table_batched_embedding(
        table_names,
        feature_table_map,
        _split_table_batched_optimizer_for_compare(opt_type),
        opt_params,
        dims,
        num_embs,
        POOLING_MODE[DynamicEmbPoolingMode.NONE],
        device,
    )
    init_embedding_tables(stbe, bdeb)

    forward_stream = torch.cuda.Stream()
    pretch_stream = torch.cuda.Stream()

    # 1. Prepare input
    # Input A
    """
    feature number = 4, batch size = 2

    f0  [0, 1],      [12],
    f1  [64,8],     [12],
    f2  [15, 2],    [7,105],
    f3  [],         [0]
    """
    indicesA = torch.tensor([0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device).to(
        key_type
    )
    offsetsA = torch.tensor([0, 2, 3, 5, 6, 8, 10, 10, 11], device=device).to(key_type)

    # Input B
    # A intersection B is not none
    """
    feature number = 4, batch size = 2

    f0  [4, 12],        [55],
    f1  [2, 17],        [1],
    f2  [],             [5, 13, 105],
    f3  [0, 23],        [42]
    """
    indicesB = torch.tensor(
        [4, 12, 55, 2, 17, 1, 5, 13, 105, 0, 23, 42], device=device
    ).to(key_type)
    offsetsB = torch.tensor([0, 2, 3, 5, 6, 6, 9, 11, 12], device=device).to(key_type)

    # stream capture will bring a cudaMalloc.
    with torch.cuda.stream(forward_stream):
        indicesB + 1
    with torch.cuda.stream(pretch_stream):
        indicesB + 1

    # 2. Test prefetch works when Cache empty
    with torch.cuda.stream(pretch_stream):
        assert list(bdeb.get_score().values()) == [1] * len(dims)
        bdeb.prefetch(indicesA, offsetsA, forward_stream)
        assert list(bdeb.get_score().values()) == [2] * len(dims)

    with torch.cuda.stream(forward_stream):
        torch.cuda.current_stream().wait_stream(pretch_stream)
        embs_bdeb_A = bdeb(indicesA, offsetsA)
        loss_bdet_A = embs_bdeb_A.mean()
        loss_bdet_A.backward()

    embs_stbe_A = stbe(indicesA, offsetsA)
    loss_stbe_A = embs_stbe_A.mean()
    loss_stbe_A.backward()

    with torch.no_grad():
        torch.cuda.synchronize()
        torch.testing.assert_close(embs_bdeb_A, embs_stbe_A, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(loss_bdet_A, loss_stbe_A, rtol=1e-06, atol=1e-06)

        cache = bdeb.cache
        if cache is not None:
            metrics = cache.cache_metrics
            assert metrics[0].item() == metrics[1].item()

    with torch.no_grad():
        bdeb.flush()
        bdeb.reset_cache_states()
        # bdeb.set_score({table_name:1 for table_name in table_names})

    # 3. Test prefetch works when Cache not empty
    with torch.cuda.stream(pretch_stream):
        bdeb.prefetch(indicesA, offsetsA, forward_stream)
        assert list(bdeb.get_score().values()) == [3] * len(dims)
        bdeb.prefetch(indicesB, offsetsB, forward_stream)
        assert list(bdeb.get_score().values()) == [4] * len(dims)

    with torch.cuda.stream(forward_stream):
        torch.cuda.current_stream().wait_stream(pretch_stream)
        embs_bdeb_A = bdeb(indicesA, offsetsA)
        loss_bdet_A = embs_bdeb_A.mean()
        loss_bdet_A.backward()
        embs_bdeb_B = bdeb(indicesB, offsetsB)
        loss_bdet_B = embs_bdeb_B.mean()
        loss_bdet_B.backward()

    embs_stbe_A = stbe(indicesA, offsetsA)
    loss_stbe_A = embs_stbe_A.mean()
    loss_stbe_A.backward()
    embs_stbe_B = stbe(indicesB, offsetsB)
    loss_stbe_B = embs_stbe_B.mean()
    loss_stbe_B.backward()

    with torch.no_grad():
        torch.cuda.synchronize()
        torch.testing.assert_close(embs_bdeb_A, embs_stbe_A, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(loss_bdet_A, loss_stbe_A, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(embs_bdeb_B, embs_stbe_B, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(loss_bdet_B, loss_stbe_B, rtol=1e-06, atol=1e-06)

        cache = bdeb.cache
        if cache is not None:
            metrics = cache.cache_metrics
            assert metrics[0].item() == metrics[1].item()

    if deterministic:
        del os.environ["DEMB_DETERMINISM_MODE"]


def random_indices(batch, min_index, max_index):
    result = set({})
    while len(result) < batch:
        result.add(random.randint(min_index, max_index))
    return result


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
    ],
)
@pytest.mark.parametrize("caching", [False, True])
@pytest.mark.parametrize("PS", [None])
@pytest.mark.parametrize("iteration", [16])
@pytest.mark.parametrize("batch_size", [2048, 65536])  # ,[])
def test_deterministic_insert(opt_type, opt_params, caching, PS, iteration, batch_size):
    print(
        f"step in test_deterministic_insert , opt_type = {opt_type} opt_params = {opt_params}"
    )

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    dims = [8]
    table_names = ["table0"]
    key_type = torch.int64
    value_type = torch.float32

    init_capacity = iteration * batch_size
    max_capacity = init_capacity

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=init_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=init_capacity * dim * 4,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    bdebt_x = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )

    bdebt_y = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )

    print(
        f"Test deterministic insert with batch={batch_size}, iteration={iteration}, capacity={init_capacity}"
    )
    os.environ["DEMB_DETERMINISM_MODE"] = "ON"

    for i in range(iteration):
        indices = torch.tensor(
            list(random_indices(batch_size, 0, 2**63 - 1)),
            dtype=key_type,
            device=device,
        )
        offsets = torch.arange(0, batch_size + 1, dtype=key_type, device=device)

        bdebt_x(indices, offsets)
        bdebt_y(indices, offsets)

        torch.cuda.synchronize()

        storage_x = bdebt_x.tables
        storage_y = bdebt_y.tables
        map_x = storage_x.key_index_map
        map_y = storage_y.key_index_map

        assert torch.equal(map_x.keys_, map_y.keys_)

        print(
            f"Iteration {i} passed for deterministic insertion with table_x's size({map_x.size()}), table_y's size({map_y.size()}), totoal({map_x.capacity()})"
        )
        cache_x = bdebt_x.cache
        cache_y = bdebt_y.cache
        if cache_x is not None:
            map_x = cache_x.key_index_map
            map_y = cache_y.key_index_map

            assert torch.equal(map_x.keys_, map_y.keys_)

            print(
                f"Iteration {i} passed for deterministic insertion with cache_x's size({map_x.size()}), cache_y's size({map_y.size()}), totoal({map_x.capacity()})"
            )

    del os.environ["DEMB_DETERMINISM_MODE"]
    print("all check passed")


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
    ],
)
@pytest.mark.parametrize("dim", [7, 8])
@pytest.mark.parametrize("caching", [True, False])
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("PS", [None])
def test_empty_batch(opt_type, opt_params, dim, caching, deterministic, PS):
    print(
        f"step in test_forward_train_eval_empty_batch , opt_type = {opt_type} opt_params = {opt_params}"
    )

    if deterministic:
        os.environ["DEMB_DETERMINISM_MODE"] = "ON"

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    dims = [dim, dim, dim]
    table_names = ["table0", "table1", "table2"]
    key_type = torch.int64
    value_type = torch.float32

    init_capacity = 1024
    max_capacity = 2048

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=init_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=1024**3,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    bdebt = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=[0, 0, 1, 2],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )
    bdebt.enable_prefetch = True
    """
    feature number = 4, batch size = 1

    f0  [],     
    f1  [],  
    f2  [],  
    f3  [],       
    """
    indices = torch.tensor([], dtype=key_type, device=device)
    offsets = torch.tensor([0, 0, 0, 0, 0], dtype=key_type, device=device)

    pretch_stream = torch.cuda.Stream()
    forward_stream = torch.cuda.Stream()

    if caching:
        with torch.cuda.stream(pretch_stream):
            bdebt.prefetch(indices, offsets, forward_stream)
            torch.cuda.synchronize()

    with torch.cuda.stream(forward_stream):
        res = bdebt(indices, offsets)
        torch.cuda.synchronize()

        res.mean().backward()

        with torch.no_grad():
            bdebt.eval()
            bdebt(indices, offsets)
        torch.cuda.synchronize()

    if deterministic:
        del os.environ["DEMB_DETERMINISM_MODE"]

    print("all check passed")


def test_export_keys_values_empty_table():
    """export_keys_values() on a never-used table must return empty tensors
    (not crash on torch.cat([])) -- covers the empty keys_list guard."""
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    opt = DynamicEmbTableOptions(
        dim=8,
        init_capacity=1024,
        max_capacity=1024,
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=0,
        score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
        caching=False,
        local_hbm_for_values=1024**3,
    )
    bdebt = BatchedDynamicEmbeddingTablesV2(
        table_names=["t0"],
        table_options=[opt],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.SUM,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.1,
    )

    keys, values = bdebt.export_keys_values("t0", device)

    assert keys.shape == (0,)
    assert keys.dtype == torch.int64
    assert values.shape[0] == 0


# ---------------------------------------------------------------------------
# Multi-table tests: mixed dims, caching, admission, dump/load
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.ADAM,
            {
                "learning_rate": 0.3,
                "weight_decay": 0.06,
                "eps": 3e-5,
                "beta1": 0.8,
                "beta2": 0.888,
            },
        ),
    ],
)
@pytest.mark.parametrize("caching", [True, False])
@pytest.mark.parametrize(
    "pooling_mode, dims",
    [
        (DynamicEmbPoolingMode.SUM, [8, 16, 32]),
        (DynamicEmbPoolingMode.MEAN, [4, 8, 16]),
    ],
)
def test_multi_table_mixed_dims_forward_backward(
    opt_type, opt_params, caching, pooling_mode, dims
):
    """Multi-table forward/backward with mixed embedding dimensions."""
    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    table_names = ["table0", "table1", "table2"]
    feature_table_map = [0, 1, 2]
    key_type = torch.int64
    value_type = torch.float32
    max_capacity = 2048

    dyn_emb_table_options_list = []
    for dim in dims:
        opt = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=max_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=1024**3,
        )
        dyn_emb_table_options_list.append(opt)

    bdeb = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=pooling_mode,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )

    assert isinstance(bdeb.tables, DynamicEmbStorage)
    assert bdeb.tables.num_tables == 3

    batch_size = 4
    indices = torch.tensor(
        [0, 1, 2, 3, 10, 11, 12, 13, 100, 101, 102, 103],
        dtype=key_type,
        device=device,
    )
    offsets = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        dtype=key_type,
        device=device,
    )

    for i in range(5):
        embs = bdeb(indices, offsets)
        torch.cuda.synchronize()

        total_D = sum(dims[feature_table_map[f]] for f in range(len(feature_table_map)))
        assert embs.shape == (batch_size, total_D)

        loss = embs.mean()
        loss.backward()
        torch.cuda.synchronize()

    print("test_multi_table_mixed_dims_forward_backward passed")


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
    ],
)
@pytest.mark.parametrize("caching", [True, False])
def test_multi_table_with_admission(opt_type, opt_params, caching):
    """Multi-table with frequency-based admission counter."""
    from dynamicemb.embedding_admission import FrequencyAdmissionStrategy, KVCounter

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    dims = [8, 16]
    table_names = ["table0", "table1"]
    feature_table_map = [0, 1]
    key_type = torch.int64
    value_type = torch.float32
    max_capacity = 2048

    admit_strategy = FrequencyAdmissionStrategy(threshold=2)

    dyn_emb_table_options_list = []
    for dim in dims:
        counter = KVCounter(capacity=max_capacity)
        opt = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=max_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=1024**3,
            admit_strategy=admit_strategy,
            admission_counter=counter,
        )
        dyn_emb_table_options_list.append(opt)

    bdeb = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=DynamicEmbPoolingMode.SUM,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )

    assert bdeb._admission_counter is not None

    batch_size = 2
    indices = torch.tensor([0, 1, 10, 11], dtype=key_type, device=device)
    offsets = torch.tensor([0, 1, 2, 3, 4], dtype=key_type, device=device)

    for i in range(5):
        embs = bdeb(indices, offsets)
        torch.cuda.synchronize()

        total_D = sum(dims[f] for f in feature_table_map)
        assert embs.shape == (batch_size, total_D)

        loss = embs.mean()
        loss.backward()
        torch.cuda.synchronize()

    print("test_multi_table_with_admission passed")


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
    ],
)
def test_multi_table_dump_load(opt_type, opt_params, tmp_path):
    """Multi-table dump and load using fused storage."""
    import torch.distributed as dist

    assert torch.cuda.is_available()
    device_id = 0
    torch.cuda.set_device(device_id)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )

    device = torch.device(f"cuda:{device_id}")

    dims = [8, 16]
    table_names = ["table0", "table1"]
    feature_table_map = [0, 1]
    key_type = torch.int64
    value_type = torch.float32
    max_capacity = 1024

    def make_bdeb():
        opts = []
        for dim in dims:
            opt = DynamicEmbTableOptions(
                dim=dim,
                init_capacity=max_capacity,
                max_capacity=max_capacity,
                index_type=key_type,
                embedding_dtype=value_type,
                device_id=device_id,
                score_strategy=DynamicEmbScoreStrategy.STEP,
                caching=False,
                local_hbm_for_values=1024**3,
            )
            opts.append(opt)
        return BatchedDynamicEmbeddingTablesV2(
            table_names=table_names,
            table_options=opts,
            feature_table_map=feature_table_map,
            pooling_mode=DynamicEmbPoolingMode.SUM,
            optimizer=opt_type,
            use_index_dedup=True,
            **opt_params,
        )

    bdeb_src = make_bdeb()

    indices = torch.tensor([0, 1, 2, 3, 10, 11, 12, 13], dtype=key_type, device=device)
    offsets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=key_type, device=device)

    for _ in range(3):
        embs = bdeb_src(indices, offsets)
        loss = embs.mean()
        loss.backward()
        torch.cuda.synchronize()

    save_dir = str(tmp_path)
    bdeb_src.dump(save_dir)

    keys0_src, vals0_src = bdeb_src.export_keys_values("table0", device)
    keys1_src, vals1_src = bdeb_src.export_keys_values("table1", device)

    bdeb_dst = make_bdeb()
    bdeb_dst.load(save_dir)

    keys0_dst, vals0_dst = bdeb_dst.export_keys_values("table0", device)
    keys1_dst, vals1_dst = bdeb_dst.export_keys_values("table1", device)

    idx0_src = keys0_src.argsort()
    idx0_dst = keys0_dst.argsort()
    torch.testing.assert_close(keys0_src[idx0_src], keys0_dst[idx0_dst])
    torch.testing.assert_close(vals0_src[idx0_src], vals0_dst[idx0_dst])

    idx1_src = keys1_src.argsort()
    idx1_dst = keys1_dst.argsort()
    torch.testing.assert_close(keys1_src[idx1_src], keys1_dst[idx1_dst])
    torch.testing.assert_close(vals1_src[idx1_src], vals1_dst[idx1_dst])

    print("test_multi_table_dump_load passed")


def _dtype_element_size(dtype: torch.dtype) -> int:
    return int(torch.empty(0, dtype=dtype).element_size())


def _assert_opt_values_ckpt_row_width(
    save_dir: str,
    table_name: str,
    emb_dim: int,
    value_type: torch.dtype,
    opt_type: EmbOptimType,
) -> None:
    """``opt_values`` checkpoint uses ``get_ckpt_state_dim`` elements per key row."""
    import torch.distributed as dist

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    keys_path = encode_checkpoint_file_path(
        save_dir, table_name, rank, world_size, "keys"
    )
    opt_path = encode_checkpoint_file_path(
        save_dir, table_name, rank, world_size, "opt_values"
    )
    key_el = _dtype_element_size(KEY_TYPE)
    opt_el = _dtype_element_size(OPT_STATE_TYPE)
    num_keys = os.path.getsize(keys_path) // key_el
    opt_sz = os.path.getsize(opt_path)
    ckpt_elems = get_optimizer_ckpt_state_dim(opt_type, emb_dim, value_type)

    if opt_type == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
        assert ckpt_elems == 1
    elif opt_type == EmbOptimType.EXACT_ADAGRAD:
        assert ckpt_elems == emb_dim
    elif opt_type == EmbOptimType.ADAM:
        assert ckpt_elems == emb_dim * 2
    elif opt_type == EmbOptimType.SGD:
        assert ckpt_elems == 0
    else:
        raise AssertionError(f"unexpected optimizer type: {opt_type}")

    if ckpt_elems == 0:
        assert opt_sz == 0, (
            f"{table_name}: SGD has no optimizer state in checkpoint; "
            f"expected empty opt_values, got {opt_sz} bytes"
        )
    else:
        assert opt_sz == num_keys * opt_el * ckpt_elems, (
            f"{table_name}: opt_values size={opt_sz}, num_keys={num_keys}, "
            f"expected {num_keys * opt_el * ckpt_elems} (ckpt_elems={ckpt_elems})"
        )


@pytest.mark.parametrize(
    "multi_table", [False, True], ids=["single_table", "multi_table"]
)
@pytest.mark.parametrize("emb_dim", [1, 16, 1023], ids=lambda d: f"dim_{d}")
@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (
            EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            {"learning_rate": 0.01, "eps": 1e-8, "initial_accumulator_value": 0.0},
        ),
        (
            EmbOptimType.EXACT_ADAGRAD,
            {"learning_rate": 0.01, "eps": 1e-8, "initial_accumulator_value": 0.0},
        ),
        (
            EmbOptimType.ADAM,
            {
                "learning_rate": 0.01,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        ),
        (EmbOptimType.SGD, {"learning_rate": 0.01}),
    ],
    ids=["rowwise_adagrad", "adagrad", "adam", "sgd"],
)
def test_dump_optimizer_states_ckpt_width(
    tmp_path,
    multi_table: bool,
    emb_dim: int,
    opt_type: EmbOptimType,
    opt_params: Dict[str, Any],
) -> None:
    """After train + dump(optim=True), ``opt_values`` row width matches checkpoint layout."""
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device_id = 0
    torch.cuda.set_device(device_id)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )

    device = torch.device(f"cuda:{device_id}")
    key_type = torch.int64
    value_type = torch.float32
    max_capacity = 8192

    if multi_table:
        table_names = ["table0", "table1"]
        feature_table_map = [0, 1]
        dims = [emb_dim, emb_dim]
        indices = torch.tensor(
            [0, 1, 2, 3, 10, 11, 12, 13], dtype=key_type, device=device
        )
        offsets = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=key_type, device=device
        )
    else:
        table_names = ["table0"]
        feature_table_map = [0, 0]
        dims = [emb_dim]
        indices = torch.tensor(
            [0, 1, 12, 64, 8, 12, 15, 2], dtype=key_type, device=device
        )
        offsets = torch.tensor([0, 2, 4, 6, 8], dtype=key_type, device=device)

    opts: List[DynamicEmbTableOptions] = []
    for dim in dims:
        opts.append(
            DynamicEmbTableOptions(
                dim=dim,
                init_capacity=max_capacity,
                max_capacity=max_capacity,
                index_type=key_type,
                embedding_dtype=value_type,
                device_id=device_id,
                score_strategy=DynamicEmbScoreStrategy.STEP,
                caching=False,
                local_hbm_for_values=1024**3,
            )
        )

    bdebt = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=opts,
        feature_table_map=feature_table_map,
        pooling_mode=DynamicEmbPoolingMode.SUM,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )

    torch.manual_seed(emb_dim + (1001 if multi_table else 0))
    for _ in range(4):
        embs = bdebt(indices, offsets)
        loss = embs.mean()
        loss.backward()
        torch.cuda.synchronize()

    save_dir = str(tmp_path)
    bdebt.dump(save_dir, optim=True)

    for tid, name in enumerate(table_names):
        _assert_opt_values_ckpt_row_width(
            save_dir, name, dims[tid], value_type, opt_type
        )

    print(
        f"test_dump_optimizer_states_ckpt_width passed "
        f"(multi_table={multi_table}, emb_dim={emb_dim}, opt={opt_type})"
    )


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
    ],
)
def test_multi_table_caching_flush(opt_type, opt_params):
    """Multi-table caching mode: prefetch, forward, flush, verify cache metrics.

    ``DynamicEmbCache`` updates ``cache_metrics[0]`` (last lookup unique count) and
    ``[1]`` (last lookup hit count) only inside ``lookup()``, overwriting prior values.
    Prefetch calls ``lookup``; training forward with cache loads from flat via
    ``slot_indices`` and does not call ``lookup`` again. The first prefetch is thus
    cold (all misses); a second prefetch on the same keys should record all hits.
    """
    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    dims = [8, 16]
    table_names = ["table0", "table1"]
    feature_table_map = [0, 1]
    key_type = torch.int64
    value_type = torch.float32
    max_capacity = 2048

    # Caching tier + backing storage is only created when total value bytes exceed
    # the summed per-table local_hbm_for_values (see _create_cache_storage).
    # Keep HBM budget below table footprint so DynamicEmbCache is instantiated.
    local_hbm_per_table = 64 * 1024  # 64 KiB each -> 128 KiB total < ~192 KiB data

    dyn_emb_table_options_list = []
    for dim in dims:
        opt = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=max_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.STEP,
            caching=True,
            local_hbm_for_values=local_hbm_per_table,
        )
        dyn_emb_table_options_list.append(opt)

    bdeb = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=DynamicEmbPoolingMode.SUM,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )
    bdeb.enable_prefetch = True
    bdeb.set_record_cache_metrics(True)

    indices = torch.tensor([0, 1, 10, 11], dtype=key_type, device=device)
    offsets = torch.tensor([0, 1, 2, 3, 4], dtype=key_type, device=device)

    forward_stream = torch.cuda.Stream()
    prefetch_stream = torch.cuda.Stream()

    with torch.cuda.stream(forward_stream):
        indices + 1
    with torch.cuda.stream(prefetch_stream):
        indices + 1

    with torch.cuda.stream(prefetch_stream):
        bdeb.prefetch(indices, offsets, forward_stream)

    with torch.cuda.stream(forward_stream):
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        embs = bdeb(indices, offsets)
        loss = embs.mean()
        loss.backward()

    torch.cuda.synchronize()

    cache = bdeb.cache
    assert cache is not None
    metrics = cache.cache_metrics
    assert metrics[0].item() == 4
    assert metrics[1].item() == 0

    with torch.cuda.stream(prefetch_stream):
        bdeb.prefetch(indices, offsets, forward_stream)

    with torch.cuda.stream(forward_stream):
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        embs2 = bdeb(indices, offsets)
        _ = embs2.mean()

    torch.cuda.synchronize()
    assert metrics[0].item() == metrics[1].item() == 4

    bdeb.flush()
    bdeb.reset_cache_states()

    print("test_multi_table_caching_flush passed")


@pytest.mark.parametrize(
    "score_strategy",
    [
        DynamicEmbScoreStrategy.TIMESTAMP,
        DynamicEmbScoreStrategy.LFU,
        DynamicEmbScoreStrategy.STEP,
        DynamicEmbScoreStrategy.NO_EVICTION,
    ],
)
@pytest.mark.parametrize(
    "caching,hbm_budget_ratio",
    [
        (True, None),
        (False, 0.25),
        (False, 1.0),
    ],
)
@pytest.mark.parametrize("multi_table", [False, True])
@pytest.mark.parametrize(
    "optimizer_type,opt_params",
    [
        (EmbOptimType.SGD, {}),
        (
            EmbOptimType.ADAM,
            {
                "weight_decay": 0.06,
                "eps": 3e-5,
                "beta1": 0.8,
                "beta2": 0.888,
            },
        ),
    ],
)
def test_table_expansion_capacity_growth(
    score_strategy: DynamicEmbScoreStrategy,
    caching: bool,
    hbm_budget_ratio: Optional[float],
    multi_table: bool,
    optimizer_type: EmbOptimType,
    opt_params: Dict[str, Any],
) -> None:
    """Train keys ``0..CMP_CAPACITY-1`` twice vs STBE (half max rows), then export/compare embeddings.

    After each forward pair, runs backward with ``learning_rate=0`` and ``weight_decay=0`` so
    fused backward/optimizer steps do not move embeddings while still exercising the training path
    (DEBUG ``key % mod`` pattern preserved for export/compare).
    """
    pytest.importorskip("torchrec")

    assert torch.cuda.is_available()
    device_id = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_id}")
    key_type = torch.int64

    (
        emb_cfgs,
        table_names,
        feature_table_map,
    ) = _embedding_configs_for_strict_expansion_test(multi_table)
    memory_ratio = _CACHE_RATIO if caching else float(hbm_budget_ratio)

    table_options: List[DynamicEmbTableOptions] = []
    user_max_by_table: List[int] = []
    dims_list: List[int] = []
    for ec in emb_cfgs:
        _, bucket_cap = _sharded_table_bucket_layout(
            ec, _WORLD_SIZE, _BUCKET_CAPACITY_EXP
        )
        max_capacity = get_sharded_table_capacity(ec, _WORLD_SIZE, _BUCKET_CAPACITY_EXP)
        user_max_by_table.append(max_capacity)
        dims_list.append(_EMB_DIM_EXPANSION)
        init_capacity = _init_capacity_strict(max_capacity, bucket_cap)
        value_bytes = get_table_value_bytes(
            ec, optimizer_type, _WORLD_SIZE, _BUCKET_CAPACITY_EXP
        )
        local_hbm = max(1, int(value_bytes * memory_ratio))
        table_options.append(
            DynamicEmbTableOptions(
                dim=_EMB_DIM_EXPANSION,
                max_capacity=max_capacity,
                init_capacity=init_capacity,
                max_load_factor=_STRICT_MAX_LOAD_FACTOR,
                bucket_capacity=bucket_cap,
                index_type=key_type,
                embedding_dtype=torch.float32,
                device_id=device_id,
                score_strategy=score_strategy,
                caching=caching,
                local_hbm_for_values=local_hbm,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.DEBUG,
                ),
                eval_initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.CONSTANT,
                    value=0.0,
                ),
            )
        )

    train_opt_params = dict(opt_params)
    train_opt_params["learning_rate"] = 0.0
    train_opt_params["weight_decay"] = 0.0

    bdebt = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=table_options,
        feature_table_map=feature_table_map,
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=optimizer_type,
        use_index_dedup=True,
        **train_opt_params,
    )
    _assert_expected_storage_backend(
        bdebt, caching=caching, hbm_budget_ratio=hbm_budget_ratio
    )

    storage = bdebt.tables
    if isinstance(storage, DynamicEmbStorage):
        km = storage.key_index_map
        nt = storage._state.num_tables
        storage_sz = _kim_occupied_sizes_per_table(km, nt)
        storage_caps = _kim_capacities_per_table(km, nt)
        print(
            f"[test_table_expansion_capacity_growth] storage KIM size (per table): "
            f"{storage_sz}, capacity (per table): {storage_caps}",
            flush=True,
        )
    elif isinstance(storage, HybridStorage):
        hbm, host = storage.tables
        hkm, okm = hbm.key_index_map, host.key_index_map
        nt_h, nt_o = hbm.num_tables, host.num_tables
        h_sz = _kim_occupied_sizes_per_table(hkm, nt_h)
        o_sz = _kim_occupied_sizes_per_table(okm, nt_o)
        h_caps = _kim_capacities_per_table(hkm, nt_h)
        o_caps = _kim_capacities_per_table(okm, nt_o)
        print(
            f"[test_table_expansion_capacity_growth] storage HBM KIM size: {h_sz}, "
            f"capacity: {h_caps}; host KIM size: {o_sz}, capacity: {o_caps}",
            flush=True,
        )
    else:
        print(
            f"[test_table_expansion_capacity_growth] storage type: {type(storage).__name__}",
            flush=True,
        )
    cache = bdebt.cache
    if isinstance(cache, DynamicEmbCache):
        ckm = cache.key_index_map
        nt_c = cache._state.num_tables
        cache_sz = _kim_occupied_sizes_per_table(ckm, nt_c)
        cache_caps = _kim_capacities_per_table(ckm, nt_c)
        print(
            f"[test_table_expansion_capacity_growth] cache KIM size (per table): "
            f"{cache_sz}, capacity (per table): {cache_caps}",
            flush=True,
        )
    else:
        print(
            "[test_table_expansion_capacity_growth] cache: disabled (no DynamicEmbCache)",
            flush=True,
        )

    cmp_caps = [max(1, m // 2) for m in user_max_by_table]
    stbe_opt = _split_table_batched_optimizer_for_compare(optimizer_type)
    stbe = create_split_table_batched_embedding(
        table_names,
        feature_table_map,
        stbe_opt,
        train_opt_params,
        dims_list,
        cmp_caps,
        PoolingMode.NONE,
        device,
    )
    _init_stbe_debug_embedding_weights(stbe)

    stbe_parts: List[str] = []
    for tid, name in enumerate(table_names):
        w_rows, rpt = _stbe_size_capacity_for_table(stbe, tid)
        stbe_parts.append(f"{name!r}:weight_rows={w_rows},rows_per_table={rpt}")
    print(
        f"[test_table_expansion_capacity_growth] STBE table size (per logical table): "
        + "; ".join(stbe_parts),
        flush=True,
    )

    bdebt.train()
    stbe.train()
    # we need to do backward to unlock the keys in the cache, otherwise overflow buffer will be also full.
    for i in range(2):
        if not multi_table:
            for j, (idx, off) in enumerate(
                _iter_compare_batches_single(
                    cmp_caps[0], _EXPANSION_STBE_TRAIN_BATCH, device, key_type
                )
            ):
                res_b = _bdebt_forward_maybe_lfu(
                    bdebt, score_strategy, idx, off, device
                )
                res_s = stbe(idx, off)
                res_b.mean().backward()
                res_s.mean().backward()
                torch.cuda.synchronize()
                _print_test_expansion_storage_cache_sizes_after_train_step(
                    bdebt, outer_iter=i, batch_index=j
                )
        else:
            for j, (idx, off) in enumerate(
                _iter_compare_batches_multi_three(
                    cmp_caps, _EXPANSION_STBE_TRAIN_BATCH, device, key_type
                )
            ):
                res_b = _bdebt_forward_maybe_lfu(
                    bdebt, score_strategy, idx, off, device
                )
                # Merged storage+cache export (no flush) runs once after the train loop;
                # see ``_assert_bdebt_storage_cache_merged_export_matches_lookup`` below.
                res_s = stbe(idx, off)
                res_b.mean().backward()
                res_s.mean().backward()
                torch.cuda.synchronize()
                _print_test_expansion_storage_cache_sizes_after_train_step(
                    bdebt, outer_iter=i, batch_index=j
                )
        torch.cuda.synchronize()

    if caching:
        _assert_bdebt_storage_cache_merged_export_matches_lookup(bdebt, device)

    _compare_bdebt_export_to_stbe_weights(
        bdebt,
        stbe,
        table_names,
        cmp_caps,
        device,
        dump_sample_embeddings=(not multi_table and caching),
    )


@pytest.mark.parametrize("load_factor", [0.1, 1.0])
@pytest.mark.parametrize(
    "max_capacity",
    [
        pytest.param(8_192, id="cap_8k"),
        pytest.param(1_048_576, id="cap_1M"),
        pytest.param(64 * 1024 * 1024, id="cap_64M"),
    ],
)
@pytest.mark.parametrize(
    "score_strategy",
    [DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU],
    ids=lambda s: s.name,
)
def test_fill_tables(
    score_strategy: DynamicEmbScoreStrategy,
    load_factor: float,
    max_capacity: int,
) -> None:
    """
    ``fill_tables`` timing and correctness across table scales (up to 64M slots
    requested), load factors, and score strategies (``TIMESTAMP`` / ``LFU``).
    Physical ``capacity`` may be bucket-aligned above ``max_capacity``.

    Uses ``tolerance=0`` (integer entry target only; no load-factor early exit).
    ``effective_lf = min(requested_load_factor, 0.95)`` (``fill_tables`` caps at
    ``0.95``). For ``effective_lf < 0.95`` asserts exact
    ``size == min(cap, int(effective_lf * cap))``.
    For ``effective_lf == 0.95`` (e.g. requested ``1.0`` clamped), inserts can evict
    while ``remaining`` counts batch inserts; assert ``size`` in a loose band vs target.

    At 64M scale, ``dim=1`` keeps HBM footprint modest; smaller capacities use
    ``dim=8``.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device_id = 0
    device = torch.device(f"cuda:{device_id}")
    dim = 1 if max_capacity >= 16 * 1024 * 1024 else 8
    local_hbm = max(
        512 * 1024 * 1024, int(max_capacity * dim * 4 * 1.25) + 64 * 1024 * 1024
    )

    opt = DynamicEmbTableOptions(
        dim=dim,
        max_capacity=max_capacity,
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=device_id,
        score_strategy=score_strategy,
        caching=False,
        local_hbm_for_values=local_hbm,
        bucket_capacity=128,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.NORMAL,
        ),
    )
    m = BatchedDynamicEmbeddingTablesV2(
        table_options=[opt],
        table_names=["t0"],
        device=device,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.01,
    )
    assert isinstance(m._storage, DynamicEmbStorage)
    km = m._storage.key_index_map

    fill_tol = 0.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    m.fill_tables(load_factor, tolerance=fill_tol)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    cap0 = km.capacity(0)
    size0 = km.size(0)
    size_i = int(size0.item()) if torch.is_tensor(size0) else int(size0)
    effective_lf = min(load_factor, 0.95)
    actual_lf = float(size_i) / float(cap0)
    expected = min(cap0, int(effective_lf * cap0))
    print(
        f"\nfill_tables score_strategy={score_strategy.name} load_factor={load_factor} "
        f"max_capacity(requested)={max_capacity} dim={dim} cap0={cap0} size0={size_i} "
        f"effective_lf={effective_lf} actual_lf={actual_lf:.6f} expected={expected} "
        f"tol={fill_tol} time_sec={elapsed:.4f}"
    )
    if effective_lf >= 0.95:
        assert size_i <= cap0
        assert size_i >= int(0.92 * expected), (
            f"fill_tables({score_strategy.name}, {load_factor} -> effective {effective_lf}): "
            f"size={size_i} below 0.92 * int_target={expected} capacity={cap0}"
        )
    else:
        assert size_i == expected, (
            f"fill_tables({score_strategy.name}, {load_factor} -> effective {effective_lf}): "
            f"size={size_i} expected={expected} capacity={cap0}"
        )
