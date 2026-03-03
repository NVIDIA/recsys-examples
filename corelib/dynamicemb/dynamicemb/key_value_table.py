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
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch  # usort:skip
import torch.distributed as dist
from dynamicemb.dynamicemb_config import DynamicEmbScoreStrategy, DynamicEmbTableOptions
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.scored_hashtable import (
    ScoreArg,
    ScorePolicy,
    ScoreSpec,
    get_scored_table,
)
from dynamicemb.types import (
    EMBEDDING_TYPE,
    KEY_TYPE,
    OPT_STATE_TYPE,
    SCORE_TYPE,
    Cache,
    CopyMode,
    Storage,
    torch_dtype_to_np_dtype,
)
from dynamicemb_extensions import EvictStrategy, device_timestamp, flagged_compact
from dynamicemb_extensions import load_from_flat_table_contiguous as _load_contiguous
from dynamicemb_extensions import load_from_flat_table_emb as _load_emb
from dynamicemb_extensions import load_from_flat_table_value as _load_value
from dynamicemb_extensions import select_insert_failed_values
from dynamicemb_extensions import store_to_flat_table_contiguous as _store_contiguous
from dynamicemb_extensions import store_to_flat_table_value as _store_value
from torch import Tensor, nn  # usort:skip

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def save_to_json(data: Dict[str, Any], file_path: str) -> None:
    try:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        raise RuntimeError(f"Error saving data to JSON file: {e}")


def load_from_json(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading data from JSON file: {e}")


def get_score_policy(score_strategy):
    if score_strategy == DynamicEmbScoreStrategy.TIMESTAMP:
        return ScoreSpec(name="timestamp", policy=ScorePolicy.GLOBAL_TIMER)
    elif score_strategy == DynamicEmbScoreStrategy.STEP:
        return ScoreSpec(name="step", policy=ScorePolicy.ASSIGN)
    elif score_strategy == DynamicEmbScoreStrategy.CUSTOMIZED:
        return ScoreSpec(name="customized", policy=ScorePolicy.ASSIGN)
    elif score_strategy == DynamicEmbScoreStrategy.LFU:
        return ScoreSpec(name="frequency", policy=ScorePolicy.ACCUMULATE)
    else:
        raise RuntimeError("Not supported score strategy.")


def get_uvm_tensor(dim, dtype, device, is_managed=False):
    return torch.zeros(
        dim,
        out=torch.ops.fbgemm.new_unified_tensor(
            torch.zeros(1, device=device, dtype=dtype),
            [dim],
            is_host_mapped=(not is_managed),
        ),
    )


# ---------------------------------------------------------------------------
# DynamicEmbTableState – shared state dataclass
# ---------------------------------------------------------------------------


@dataclass
class DynamicEmbTableState:
    options_list: List[DynamicEmbTableOptions]
    num_tables: int
    device: torch.device
    score_policy: ScoreSpec
    evict_strategy: EvictStrategy
    key_index_map: Any
    capacity: int
    tables: List[torch.Tensor]
    table_ptrs: torch.Tensor
    table_emb_dims: torch.Tensor
    table_value_dims: torch.Tensor
    table_emb_dims_cpu: List[int]
    table_value_dims_cpu: List[int]
    max_emb_dim: int
    emb_dim: int
    value_dim: int
    emb_dtype: torch.dtype
    all_dims_vec4: bool
    optimizer: BaseDynamicEmbeddingOptimizer
    initial_optim_state: float
    threads_in_wave: int
    score: Optional[int] = None
    training: bool = False
    timestamp: int = 0
    # Overflow region fields (per-table, only set when overflow is enabled)
    overflow_caps: Optional[List[int]] = None
    overflow_offsets: Optional[List[int]] = None


def create_table_state(
    options: List[DynamicEmbTableOptions],
    optimizer: BaseDynamicEmbeddingOptimizer,
    enable_overflow: bool = False,
) -> DynamicEmbTableState:
    if not options:
        raise ValueError("options must be non-empty")

    base_opt = options[0]
    num_tables = len(options)

    device_idx = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_idx}")
    score_policy = get_score_policy(base_opt.score_strategy)
    evict_strategy = base_opt.evict_strategy.value

    capacities = [opt.init_capacity for opt in options]
    key_index_map = get_scored_table(
        capacity=capacities,
        bucket_capacity=base_opt.bucket_capacity,
        key_type=base_opt.index_type,
        score_specs=[score_policy],
        device=device,
        enable_overflow=enable_overflow,
    )
    capacity = key_index_map.capacity()

    dims = [opt.dim for opt in options]
    max_emb_dim = max(dims)
    emb_dtype = base_opt.embedding_dtype
    emb_dim = max(dims)

    optim_state_dims = [optimizer.get_state_dim(d) for d in dims]
    value_dims = [d + s for d, s in zip(dims, optim_state_dims)]
    value_dim = max(value_dims)
    all_dims_vec4 = all((d % 4) == 0 for d in dims) and all(
        (v % 4) == 0 for v in value_dims
    )

    table_emb_dims = torch.tensor(dims, dtype=torch.int64, device=device)
    table_value_dims = torch.tensor(value_dims, dtype=torch.int64, device=device)

    actual_caps = key_index_map.per_table_capacity_

    overflow_caps_list: Optional[List[int]] = None
    overflow_offsets_list: Optional[List[int]] = None

    if enable_overflow:
        ovf_cap = key_index_map.overflow_bucket_capacity_
        overflow_caps_list = [ovf_cap] * num_tables
        overflow_offsets_list = list(actual_caps)

    tables: List[torch.Tensor] = []
    for i, (cap, vd) in enumerate(zip(actual_caps, value_dims)):
        total_cap = cap
        if enable_overflow:
            total_cap += overflow_caps_list[i]
        size = total_cap * vd
        if base_opt.local_hbm_for_values == 0:
            tables.append(get_uvm_tensor(size, dtype=emb_dtype, device=device))
        else:
            tables.append(torch.zeros(size, dtype=emb_dtype, device=device))
    table_ptrs = torch.tensor(
        [t.data_ptr() for t in tables], dtype=torch.int64, device=device
    )

    props = torch.cuda.get_device_properties(device_idx)
    threads_in_wave = (
        props.multi_processor_count * props.max_threads_per_multi_processor
    )

    return DynamicEmbTableState(
        options_list=options,
        num_tables=num_tables,
        device=device,
        score_policy=score_policy,
        evict_strategy=evict_strategy,
        key_index_map=key_index_map,
        capacity=capacity,
        tables=tables,
        table_ptrs=table_ptrs,
        table_emb_dims=table_emb_dims,
        table_value_dims=table_value_dims,
        table_emb_dims_cpu=dims,
        table_value_dims_cpu=value_dims,
        max_emb_dim=max_emb_dim,
        emb_dim=emb_dim,
        value_dim=value_dim,
        emb_dtype=emb_dtype,
        all_dims_vec4=all_dims_vec4,
        optimizer=optimizer,
        initial_optim_state=optimizer.get_initial_optim_states(),
        threads_in_wave=threads_in_wave,
        score=None,
        training=False,
        timestamp=device_timestamp(),
        overflow_caps=overflow_caps_list,
        overflow_offsets=overflow_offsets_list,
    )


# ---------------------------------------------------------------------------
# Free functions operating on DynamicEmbTableState
# ---------------------------------------------------------------------------


def load_from_flat(
    state: DynamicEmbTableState,
    indices: torch.Tensor,
    table_ids: torch.Tensor,
    copy_mode: CopyMode,
) -> torch.Tensor:
    N = indices.numel()
    if copy_mode == CopyMode.EMBEDDING:
        max_dim = state.emb_dim
        _load = _load_emb
    else:
        max_dim = state.value_dim
        _load = _load_value
    output = torch.empty(N, max_dim, dtype=state.emb_dtype, device=state.device)
    if N > 0:
        _load(
            state.table_ptrs,
            indices,
            table_ids,
            output,
            state.table_value_dims,
            state.table_emb_dims,
            state.emb_dim,
            state.all_dims_vec4,
        )
    return output


def store_to_flat(
    state: DynamicEmbTableState,
    indices: torch.Tensor,
    table_ids: torch.Tensor,
    values: torch.Tensor,
) -> None:
    if values.dim() == 1:
        values = values.unsqueeze(1)
    _store_value(
        state.table_ptrs,
        indices,
        table_ids,
        values.to(state.emb_dtype),
        state.table_value_dims,
        state.table_emb_dims,
        state.emb_dim,
        state.all_dims_vec4,
    )


def load_from_flat_single_table(
    state: DynamicEmbTableState,
    indices: torch.Tensor,
    table_id: int,
) -> torch.Tensor:
    """Load full values for a single table. Returns compact [N, value_dim_t]."""
    N = indices.numel()
    vdim = state.table_value_dims_cpu[table_id]
    output = torch.empty(N, vdim, dtype=state.emb_dtype, device=state.device)
    if N > 0:
        _load_contiguous(
            state.table_ptrs,
            indices,
            table_id,
            output,
            state.table_value_dims,
            state.table_emb_dims,
            state.emb_dim,
            state.all_dims_vec4,
        )
    return output


def store_to_flat_single_table(
    state: DynamicEmbTableState,
    indices: torch.Tensor,
    table_id: int,
    values: torch.Tensor,
) -> None:
    """Store full values for a single table. Expects compact [N, value_dim_t]."""
    N = indices.numel()
    if N == 0:
        return
    if values.dim() == 1:
        values = values.unsqueeze(1)
    _store_contiguous(
        state.table_ptrs,
        indices,
        table_id,
        values.to(state.emb_dtype),
        state.table_value_dims,
        state.table_emb_dims,
        state.emb_dim,
        state.all_dims_vec4,
    )


def get_find_score_arg(
    state: DynamicEmbTableState,
    num_keys: int,
    device: torch.device,
    lfu_accumulated_frequency: Optional[torch.Tensor] = None,
) -> ScoreArg:
    """Build a ScoreArg for find/lookup operations.

    When ``state.training`` is False (eval), returns CONST policy so that
    existing scores in the hash table are never modified.

    When ``state.training`` is True:
      - LFU: ACCUMULATE with provided or default (ones) frequency.
      - LRU (GLOBAL_TIMER): no explicit value needed.
      - CUSTOMIZED / STEP: ASSIGN with ``state.score``.
    """
    if not state.training:
        return ScoreArg(
            name=state.score_policy.name, value=None, policy=ScorePolicy.CONST
        )

    if lfu_accumulated_frequency is not None:
        assert state.evict_strategy == EvictStrategy.KLfu
        scores = lfu_accumulated_frequency
    elif state.evict_strategy == EvictStrategy.KLfu:
        scores = torch.ones(num_keys, device=device, dtype=torch.long)
    elif state.evict_strategy == EvictStrategy.KCustomized:
        scores = torch.empty(num_keys, device=device, dtype=torch.long)
        scores.fill_(state.score)
    else:
        scores = None

    return ScoreArg(
        name=state.score_policy.name,
        value=scores,
        policy=state.score_policy.policy,
    )


def get_insert_score_arg(
    state: DynamicEmbTableState,
    num_keys: int,
    device: torch.device,
    scores: Optional[torch.Tensor] = None,
    preserve_existing: bool = False,
) -> ScoreArg:
    """Build a ScoreArg for insert operations (new keys).

    *preserve_existing* should be True when re-inserting keys that already
    exist in the table (e.g. backward embedding update) so that their scores
    are not overwritten.  This fixes the bug where backward re-inserts
    incorrectly assigned new scores.

    When *preserve_existing* is False (the common case for genuinely new keys):
      - LRU: GLOBAL_TIMER (no explicit value).
      - LFU / CUSTOMIZED / STEP: ASSIGN with provided *scores* or
        ``state.score`` as default.
      - ACCUMULATE policy is converted to ASSIGN for inserts.
    """
    if preserve_existing:
        return ScoreArg(
            name=state.score_policy.name, value=None, policy=ScorePolicy.CONST
        )

    is_lru = state.evict_strategy == EvictStrategy.KLru
    if not is_lru and scores is None:
        scores = torch.empty(num_keys, device=device, dtype=torch.uint64)
        scores.fill_(state.score)

    policy = state.score_policy.policy
    if policy == ScorePolicy.ACCUMULATE:
        policy = ScorePolicy.ASSIGN
    if is_lru and scores is not None:
        policy = ScorePolicy.ASSIGN

    return ScoreArg(name=state.score_policy.name, value=scores, policy=policy)


def _find_keys(
    state: DynamicEmbTableState,
    unique_keys: torch.Tensor,
    table_ids: torch.Tensor,
    lfu_accumulated_frequency: Optional[torch.Tensor] = None,
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
    """Key-only find: lookup in hash table, return missing info + slot indices."""
    if unique_keys.dtype != state.key_index_map.key_type:
        unique_keys = unique_keys.to(state.key_index_map.key_type)

    batch = unique_keys.size(0)
    device = unique_keys.device

    score_arg = get_find_score_arg(state, batch, device, lfu_accumulated_frequency)

    if batch == 0:
        return (
            0,
            torch.empty_like(unique_keys),
            torch.empty(batch, dtype=torch.long, device=device),
            torch.empty_like(table_ids),
            torch.empty(batch, dtype=torch.uint64, device=device)
            if score_arg.value is not None
            else None,
            torch.empty(batch, dtype=torch.bool, device=device),
            torch.empty(batch, dtype=torch.int64, device=device),
            torch.empty(batch, dtype=torch.int64, device=device),
        )

    score_out, founds, indices = state.key_index_map.lookup(
        unique_keys, table_ids, score_arg
    )

    missing = torch.logical_not(founds)
    (
        h_num_missing,
        missing_indices,
        (missing_keys, missing_table_ids, missing_scores),
    ) = flagged_compact(
        missing,
        [unique_keys, table_ids, score_arg.value],
    )

    return (
        h_num_missing,
        missing_keys,
        missing_indices,
        missing_table_ids,
        missing_scores,
        founds,
        score_out,
        indices,
    )


def _insert_key_values(
    state: DynamicEmbTableState,
    unique_keys: torch.Tensor,
    table_ids: torch.Tensor,
    unique_values: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    preserve_existing: bool = False,
) -> None:
    score_arg = get_insert_score_arg(
        state, unique_keys.numel(), unique_keys.device, scores, preserve_existing
    )
    indices = state.key_index_map.insert(unique_keys, table_ids, score_arg)
    store_to_flat(state, indices, table_ids, unique_values)


def _insert_and_evict_keys(
    state: DynamicEmbTableState,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Key-only insert_and_evict. Returns (indices, num_evicted, evicted_keys,
    evicted_table_ids, evicted_indices, evicted_scores).
    Caller is responsible for loading evicted values and storing new values."""
    score_arg = get_insert_score_arg(
        state, keys.numel(), keys.device, scores
    )
    (
        indices,
        num_evicted,
        evicted_keys,
        evicted_indices,
        evicted_scores,
        evicted_table_ids,
    ) = state.key_index_map.insert_and_evict(keys, table_ids, score_arg)

    return (
        indices,
        num_evicted,
        evicted_keys,
        evicted_table_ids,
        evicted_indices,
        evicted_scores,
    )


def export_keys_values_iter(
    state: DynamicEmbTableState,
    device: torch.device,
    batch_size: int = 65536,
    table_id: int = 0,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
    """Export keys, embeddings, opt_states, scores for a logical table."""
    emb_dim_t = state.table_emb_dims_cpu[table_id]
    vdim = state.table_value_dims_cpu[table_id]
    optim_state_dim = vdim - emb_dim_t

    for (
        keys,
        named_scores,
        indices,
    ) in state.key_index_map._batched_export_keys_scores(
        [state.score_policy.name],
        state.device,
        batch_size=batch_size,
        return_index=True,
        table_id=table_id,
    ):
        scores = named_scores[state.score_policy.name]
        values = load_from_flat_single_table(state, indices, table_id)
        embeddings = values[:, :emb_dim_t].to(dtype=EMBEDDING_TYPE).contiguous()
        if optim_state_dim != 0:
            opt_states = (
                values[:, -optim_state_dim:].to(dtype=OPT_STATE_TYPE).contiguous()
            ).to(device)
        else:
            opt_states = None
        yield (
            keys.to(device),
            embeddings.to(device),
            opt_states,
            scores.to(SCORE_TYPE).to(device),
        )


def _dump_table(
    state: DynamicEmbTableState,
    table_id: int,
    meta_json_file_path: str,
    emb_key_path: str,
    embedding_file_path: str,
    score_file_path: str,
    opt_file_path: str,
    include_optim: bool,
    include_meta: bool,
    current_score: Optional[int] = None,
    append: bool = False,
) -> None:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    if not append and include_meta:
        meta_data = {}
        meta_data.update(state.optimizer.get_opt_args())
        meta_data["evict_strategy"] = str(state.evict_strategy)

        if current_score is not None:
            meta_data["step_score"] = current_score

        save_to_json(meta_data, meta_json_file_path)

    mode = "ab" if append else "wb"
    fkey = open(emb_key_path, mode)
    fembedding = open(embedding_file_path, mode)
    fscore = open(score_file_path, mode)
    fopt_states = open(opt_file_path, mode) if include_optim else None

    for keys, embeddings, opt_states_batch, scores in export_keys_values_iter(
        state, device=device, table_id=table_id
    ):
        fkey.write(keys.cpu().numpy().tobytes())
        fembedding.write(embeddings.cpu().numpy().tobytes())
        if state.evict_strategy == EvictStrategy.KLru:
            scores = state.timestamp - scores
        fscore.write(scores.cpu().numpy().tobytes())
        if fopt_states and opt_states_batch is not None:
            fopt_states.write(opt_states_batch.cpu().numpy().tobytes())

    fkey.close()
    fembedding.close()
    if fscore:
        fscore.close()
    if fopt_states:
        fopt_states.close()


def _iter_batches_from_files(
    emb_key_path: str,
    embedding_file_path: str,
    score_file_path: Optional[str],
    opt_file_path: Optional[str],
    dim: int,
    optstate_dim: int,
    device: torch.device,
    batch_size: int = 65536,
) -> Iterator[Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]]:
    """Yield (keys, embeddings, scores, opt_states) batches from checkpoint files.

    Handles file I/O, numpy deserialization, and distributed world_size filtering.
    Pass *score_file_path* / *opt_file_path* as ``None`` to skip those files.
    """
    fkey = open(emb_key_path, "rb")
    fembedding = open(embedding_file_path, "rb")
    fscore = (
        open(score_file_path, "rb")
        if score_file_path and os.path.exists(score_file_path)
        else None
    )
    fopt = open(opt_file_path, "rb") if opt_file_path else None
    num_keys = os.path.getsize(emb_key_path) // KEY_TYPE.itemsize

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    try:
        for start in range(0, num_keys, batch_size):
            n = min(num_keys - start, batch_size)

            keys_bytes = fkey.read(KEY_TYPE.itemsize * n)
            keys = torch.tensor(
                np.frombuffer(keys_bytes, dtype=torch_dtype_to_np_dtype[KEY_TYPE]),
                dtype=KEY_TYPE,
                device=device,
            )

            emb_bytes = fembedding.read(EMBEDDING_TYPE.itemsize * dim * n)
            embeddings = torch.tensor(
                np.frombuffer(emb_bytes, dtype=torch_dtype_to_np_dtype[EMBEDDING_TYPE]),
                dtype=EMBEDDING_TYPE,
                device=device,
            ).view(-1, dim)

            scores = None
            if fscore:
                score_bytes = fscore.read(SCORE_TYPE.itemsize * n)
                scores = torch.tensor(
                    np.frombuffer(
                        score_bytes, dtype=torch_dtype_to_np_dtype[SCORE_TYPE]
                    ),
                    dtype=SCORE_TYPE,
                    device=device,
                )

            opt_states = None
            if fopt:
                opt_bytes = fopt.read(OPT_STATE_TYPE.itemsize * optstate_dim * n)
                opt_states = torch.tensor(
                    np.frombuffer(
                        opt_bytes, dtype=torch_dtype_to_np_dtype[OPT_STATE_TYPE]
                    ),
                    dtype=OPT_STATE_TYPE,
                    device=device,
                ).view(-1, optstate_dim)

            if world_size > 1:
                masks = keys % world_size == rank
                keys = keys[masks]
                embeddings = embeddings[masks]
                if scores is not None:
                    scores = scores[masks]
                if opt_states is not None:
                    opt_states = opt_states[masks]

            yield keys, embeddings, scores, opt_states
    finally:
        fkey.close()
        fembedding.close()
        if fscore:
            fscore.close()
        if fopt:
            fopt.close()


@dataclass
class _LoadParams:
    meta_data: Dict[str, Any]
    dim: int
    optstate_dim: int
    include_optim: bool


def _validate_load_meta(
    state: DynamicEmbTableState,
    table_id: int,
    meta_json_file_path: str,
    emb_key_path: str,
    embedding_file_path: str,
    score_file_path: Optional[str],
    opt_file_path: Optional[str],
    include_optim: bool,
) -> _LoadParams:
    """Shared validation for checkpoint loading.

    Reads meta JSON, validates opt_type / evict_strategy, resolves
    include_optim, and checks file-size consistency.
    """
    meta_data = load_from_json(meta_json_file_path)
    opt_type = meta_data.get("opt_type", None)
    if opt_type and state.optimizer.get_opt_args().get("opt_type", None) != opt_type:
        include_optim = False
        print(
            f"Optimizer type mismatch: {opt_type} != {state.optimizer.get_opt_args().get('opt_type')}. Will not load optimizer states."
        )

    evict_strategy = meta_data.get("evict_strategy", None)
    if evict_strategy and str(state.evict_strategy) != evict_strategy:
        raise ValueError(
            f"Evict strategy mismatch: {evict_strategy} != {state.evict_strategy}"
        )

    if score_file_path is None:
        print(
            f"Score file {score_file_path} does not exist. Will not load score states."
        )

    if not opt_file_path or not os.path.exists(opt_file_path):
        include_optim = False
        print(
            f"Optimizer file {opt_file_path} does not exist. Will not load optimizer states."
        )

    dim = state.table_emb_dims_cpu[table_id]
    optstate_dim = state.table_value_dims_cpu[table_id] - dim

    if optstate_dim == 0:
        include_optim = False

    if include_optim:
        state.optimizer.set_opt_args(meta_data)

    num_keys = os.path.getsize(emb_key_path) // KEY_TYPE.itemsize
    num_embeddings = (
        os.path.getsize(embedding_file_path) // EMBEDDING_TYPE.itemsize // dim
    )
    if num_keys != num_embeddings:
        raise ValueError(
            f"The number of keys in {emb_key_path} does not match with number of embeddings in {embedding_file_path}."
        )
    if score_file_path and os.path.exists(score_file_path):
        num_scores = os.path.getsize(score_file_path) // SCORE_TYPE.itemsize
        if num_keys != num_scores:
            raise ValueError(
                f"The number of keys in {emb_key_path} does not match with number of scores in {score_file_path}."
            )
    if include_optim and opt_file_path:
        num_opt_states = (
            os.path.getsize(opt_file_path) // OPT_STATE_TYPE.itemsize // optstate_dim
        )
        if num_keys != num_opt_states:
            raise ValueError(
                f"The number of keys in {emb_key_path} does not match with number of opt_states in {opt_file_path}."
            )

    return _LoadParams(
        meta_data=meta_data,
        dim=dim,
        optstate_dim=optstate_dim,
        include_optim=include_optim,
    )


def _load_table(
    state: DynamicEmbTableState,
    table_id: int,
    meta_json_file_path: str,
    emb_key_path: str,
    embedding_file_path: str,
    score_file_path: Optional[str],
    opt_file_path: Optional[str],
    include_optim: bool,
) -> Optional[int]:
    params = _validate_load_meta(
        state,
        table_id,
        meta_json_file_path,
        emb_key_path,
        embedding_file_path,
        score_file_path,
        opt_file_path,
        include_optim,
    )
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    for keys, embeddings, scores, opt_states in _iter_batches_from_files(
        emb_key_path,
        embedding_file_path,
        score_file_path,
        opt_file_path if params.include_optim else None,
        params.dim,
        params.optstate_dim,
        device,
    ):
        if scores is not None and state.evict_strategy == EvictStrategy.KLru:
            scores = torch.clamp(state.timestamp - scores, min=0)
        _load_key_values(state, keys, embeddings, scores, opt_states, table_id=table_id)

    return params.meta_data.get("step_score", None)


def _load_key_values(
    state: DynamicEmbTableState,
    keys: torch.Tensor,
    embeddings: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    opt_states: Optional[torch.Tensor] = None,
    table_id: int = 0,
) -> None:
    dim = embeddings.size(1)
    optstate_dim = (
        state.table_value_dims_cpu[table_id] - state.table_emb_dims_cpu[table_id]
    )
    if not keys.is_cuda:
        raise RuntimeError("Keys must be on GPU")
    if not embeddings.is_cuda:
        raise RuntimeError("Embeddings must be on GPU")
    if scores is not None and not scores.is_cuda:
        raise RuntimeError("Scores must be on GPU")
    if opt_states is not None and not opt_states.is_cuda:
        raise RuntimeError("Opt states must be on GPU")

    if opt_states is None and optstate_dim > 0:
        opt_states = (
            torch.ones(
                keys.numel(),
                optstate_dim,
                dtype=state.emb_dtype,
                device=embeddings.device,
            )
            * state.initial_optim_state
        )

    values = (
        torch.cat([embeddings.view(-1, dim), opt_states.view(-1, optstate_dim)], dim=-1)
        if opt_states is not None
        else embeddings
    )

    policy = ScorePolicy.ASSIGN

    if scores is None:
        assert (
            state.evict_strategy == EvictStrategy.KLru
        ), "scores is None for KLru evict strategy is allowed but will be deprecated in future."
        policy = ScorePolicy.GLOBAL_TIMER
    else:
        scores = scores.to(SCORE_TYPE)

    score_arg_insert = ScoreArg(
        name=state.score_policy.name,
        value=scores,
        policy=policy,
    )

    tid_tensor = torch.full(
        (keys.numel(),), table_id, dtype=torch.int64, device=keys.device
    )
    indices = state.key_index_map.insert(keys, tid_tensor, score_arg_insert)
    store_to_flat_single_table(state, indices, table_id, values)


# ---------------------------------------------------------------------------
# DynamicEmbCache – Cache interface (key-only find / insert_and_evict)
# ---------------------------------------------------------------------------


class DynamicEmbCache(Cache):
    def __init__(
        self,
        options: List[DynamicEmbTableOptions],
        optimizer: BaseDynamicEmbeddingOptimizer,
    ):
        self._state = create_table_state(options, optimizer, enable_overflow=True)
        self._cache_metrics = torch.zeros(10, dtype=torch.long, device="cpu")
        self._record_cache_metrics = False

    # -- Cache interface --

    def increment_counter(self, slot_indices: torch.Tensor) -> None:
        self._state.key_index_map.increment_counter(slot_indices)

    def decrement_counter(self, slot_indices: torch.Tensor) -> None:
        self._state.key_index_map.decrement_counter(slot_indices)

    def lookup(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lookup with overflow fallback. Returns (score_out, founds, indices)."""
        state = self._state
        score_arg = get_find_score_arg(
            state, unique_keys.size(0), unique_keys.device, lfu_accumulated_frequency
        )
        result = state.key_index_map.lookup_with_overflow(
            unique_keys, table_ids, score_arg
        )
        if self._record_cache_metrics:
            self._cache_metrics[0] = unique_keys.size(0)
            founds = result[1]
            self._cache_metrics[1] = founds.sum().item()
        return result

    def insert_and_evict(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Insert with counter-aware eviction and overflow fallback."""
        state = self._state
        score_arg = get_insert_score_arg(
            state, keys.numel(), keys.device, scores
        )
        result = state.key_index_map.insert_and_evict_with_counter_and_overflow(
            keys, table_ids, score_arg
        )
        if self._record_cache_metrics:
            self._cache_metrics[2] = keys.numel()
            self._cache_metrics[3] = result[1]  # num_evicted
        return result

    def reset(self) -> None:
        self._state.key_index_map.reset()

    @property
    def cache_metrics(self) -> Optional[torch.Tensor]:
        return self._cache_metrics if self._record_cache_metrics else None

    def set_record_cache_metrics(self, record: bool) -> None:
        self._record_cache_metrics = record

    # -- Score management --

    def set_score(self, score: int) -> None:
        self._state.score = score

    @property
    def training(self) -> bool:
        return self._state.training

    @training.setter
    def training(self, value: bool) -> None:
        self._state.training = value

    # -- Convenience accessors --

    @property
    def num_tables(self) -> int:
        return self._state.num_tables

    def embedding_dtype(self) -> torch.dtype:
        return self._state.emb_dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._state.table_emb_dims_cpu[table_id]

    def value_dim(self, table_id: int) -> int:
        return self._state.table_value_dims_cpu[table_id]

    def max_embedding_dim(self) -> int:
        return self._state.emb_dim

    def max_value_dim(self) -> int:
        return self._state.value_dim

    def init_optimizer_state(self) -> float:
        return self._state.initial_optim_state

    def evict_strategy(self) -> EvictStrategy:
        return self._state.evict_strategy

    def size(self) -> int:
        return self._state.key_index_map.size()

    @property
    def key_index_map(self):
        return self._state.key_index_map


# ---------------------------------------------------------------------------
# DynamicEmbStorage – Storage interface (find with values, insert, dump, load)
# ---------------------------------------------------------------------------


class DynamicEmbStorage(Storage):
    def __init__(
        self,
        options: List[DynamicEmbTableOptions],
        optimizer: BaseDynamicEmbeddingOptimizer,
    ):
        is_hbm = options[0].local_hbm_for_values > 0
        self._state = create_table_state(
            options,
            optimizer,
            enable_overflow=is_hbm,
        )

    @property
    def key_index_map(self):
        return self._state.key_index_map

    # -- Storage interface --

    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
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
        result = _find_keys(
            self._state, unique_keys, table_ids, lfu_accumulated_frequency
        )
        indices = result[-1]
        values = load_from_flat(self._state, indices, table_ids, copy_mode=copy_mode)
        return (*result[:-1], values)

    def increment_counter(self, slot_indices: torch.Tensor) -> None:
        self._state.key_index_map.increment_counter(slot_indices)

    def decrement_counter(self, slot_indices: torch.Tensor) -> None:
        self._state.key_index_map.decrement_counter(slot_indices)

    def insert(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        unique_values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        preserve_existing: bool = False,
    ) -> None:
        _insert_key_values(
            self._state,
            unique_keys,
            table_ids,
            unique_values,
            scores,
            preserve_existing,
        )

    def dump(
        self,
        table_id: int,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: str,
        opt_file_path: str,
        include_optim: bool = True,
        include_meta: bool = True,
        current_score: Optional[int] = None,
    ) -> None:
        if self._state.evict_strategy == EvictStrategy.KLru:
            if dist.is_initialized():
                dist.barrier()
            self._state.timestamp = device_timestamp()

        _dump_table(
            self._state,
            table_id,
            meta_json_file_path,
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path,
            include_optim,
            include_meta,
            current_score,
        )

    def load(
        self,
        table_id: int,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool = True,
    ) -> Optional[int]:
        if self._state.evict_strategy == EvictStrategy.KLru:
            if dist.is_initialized():
                dist.barrier()
            self._state.timestamp = device_timestamp()

        return _load_table(
            self._state,
            table_id,
            meta_json_file_path,
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path,
            include_optim,
        )

    # -- Export --

    def export_keys_values(
        self,
        device: torch.device,
        batch_size: int = 65536,
        table_id: int = 0,
    ) -> Iterator[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]
    ]:
        yield from export_keys_values_iter(self._state, device, batch_size, table_id)

    # -- Property accessors --

    def embedding_dtype(self) -> torch.dtype:
        return self._state.emb_dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._state.table_emb_dims_cpu[table_id]

    def value_dim(self, table_id: int) -> int:
        return self._state.table_value_dims_cpu[table_id]

    def max_embedding_dim(self) -> int:
        return self._state.emb_dim

    def max_value_dim(self) -> int:
        return self._state.value_dim

    def init_optimizer_state(self) -> float:
        return self._state.initial_optim_state

    # -- Score management --

    def set_score(self, score: int) -> None:
        self._state.score = score

    @property
    def training(self) -> bool:
        return self._state.training

    @training.setter
    def training(self, value: bool) -> None:
        self._state.training = value

    def evict_strategy(self) -> EvictStrategy:
        return self._state.evict_strategy

    @property
    def num_tables(self) -> int:
        return self._state.num_tables

    def size(self) -> int:
        return self._state.key_index_map.size()


# ---------------------------------------------------------------------------
# HybridStorage – two-tier storage using two DynamicEmbTableState instances
# ---------------------------------------------------------------------------


class HybridStorage(Storage):
    """Two-tier storage: HBM (GPU) table + host table, disjoint partitions."""

    def __init__(
        self,
        hbm_options: List[DynamicEmbTableOptions],
        host_options: List[DynamicEmbTableOptions],
        optimizer: BaseDynamicEmbeddingOptimizer,
    ):
        self._hbm = create_table_state(hbm_options, optimizer)
        self._host = create_table_state(host_options, optimizer)
        self.optimizer = optimizer

    @property
    def tables(self) -> List[DynamicEmbTableState]:
        return [self._hbm, self._host]

    # -- Score management --

    @property
    def training(self) -> bool:
        return self._hbm.training

    @training.setter
    def training(self, value: bool) -> None:
        self._hbm.training = value
        self._host.training = value

    def set_score(self, score: int) -> None:
        self._hbm.score = score
        self._host.score = score

    def evict_strategy(self) -> EvictStrategy:
        return self._hbm.evict_strategy

    # -- Storage property accessors --

    def embedding_dtype(self) -> torch.dtype:
        return self._hbm.emb_dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._hbm.table_emb_dims_cpu[table_id]

    def value_dim(self, table_id: int) -> int:
        return self._hbm.table_value_dims_cpu[table_id]

    def max_embedding_dim(self) -> int:
        return self._hbm.emb_dim

    def max_value_dim(self) -> int:
        return self._hbm.value_dim

    def init_optimizer_state(self) -> float:
        return self._hbm.initial_optim_state

    @property
    def num_tables(self) -> int:
        return self._hbm.num_tables

    # -- Two-tier find (with values) --

    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
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
        result_hbm = _find_keys(
            self._hbm, unique_keys, table_ids, lfu_accumulated_frequency
        )
        (
            h_num_missing_hbm,
            missing_keys_hbm,
            missing_indices_hbm,
            missing_table_ids_hbm,
            missing_scores_hbm,
            founds_hbm,
            scores_hbm,
            indices_hbm,
        ) = result_hbm

        values = load_from_flat(self._hbm, indices_hbm, table_ids, copy_mode=copy_mode)

        if h_num_missing_hbm == 0:
            return (
                0,
                missing_keys_hbm,
                missing_indices_hbm,
                missing_table_ids_hbm,
                missing_scores_hbm,
                founds_hbm,
                scores_hbm,
                values,
            )

        result_host = _find_keys(
            self._host, missing_keys_hbm, missing_table_ids_hbm, missing_scores_hbm,
        )
        (
            h_num_missing_both,
            missing_keys_both,
            missing_indices_both,
            missing_table_ids_both,
            missing_scores_both,
            founds_host,
            scores_host,
            indices_host,
        ) = result_host

        host_vals = load_from_flat(
            self._host, indices_host, missing_table_ids_hbm, copy_mode=copy_mode
        )

        host_found_mask = founds_host
        if host_found_mask.any():
            values[missing_indices_hbm[host_found_mask]] = host_vals[host_found_mask]

        founds_combined = founds_hbm.clone()
        founds_combined[missing_indices_hbm[host_found_mask]] = True

        global_missing_indices = missing_indices_hbm[missing_indices_both]
        global_missing_scores = missing_scores_both

        return (
            h_num_missing_both,
            missing_keys_both,
            global_missing_indices,
            missing_table_ids_both,
            global_missing_scores,
            founds_combined,
            scores_hbm,
            values,
        )

    # -- Insert: HBM first, evictions to host --

    def insert(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        unique_values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        preserve_existing: bool = False,
    ) -> None:
        (
            indices,
            num_evicted,
            evicted_keys,
            evicted_table_ids,
            evicted_indices,
            evicted_scores,
        ) = _insert_and_evict_keys(self._hbm, unique_keys, table_ids, scores)

        evicted_values = load_from_flat(
            self._hbm, evicted_indices, evicted_table_ids, copy_mode=CopyMode.VALUE
        )
        select_insert_failed_values(evicted_indices, unique_values, evicted_values)
        store_to_flat(self._hbm, indices, table_ids, unique_values)

        if num_evicted != 0:
            _insert_key_values(
                self._host,
                evicted_keys,
                evicted_table_ids,
                evicted_values,
                evicted_scores,
            )

    # -- Dump: write host first, then append HBM --

    def dump(
        self,
        table_id: int,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: str,
        opt_file_path: str,
        include_optim: bool = True,
        include_meta: bool = True,
        current_score: Optional[int] = None,
    ) -> None:
        is_lru_hbm = self._hbm.evict_strategy == EvictStrategy.KLru
        is_lru_host = self._host.evict_strategy == EvictStrategy.KLru
        if is_lru_hbm or is_lru_host:
            if dist.is_initialized():
                dist.barrier()
            ts = device_timestamp()
            if is_lru_host:
                self._host.timestamp = ts
            if is_lru_hbm:
                self._hbm.timestamp = ts

        _dump_table(
            self._host,
            table_id,
            meta_json_file_path,
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path,
            include_optim=include_optim,
            include_meta=include_meta,
            current_score=current_score,
        )

        _dump_table(
            self._hbm,
            table_id,
            meta_json_file_path,
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path,
            include_optim=include_optim,
            include_meta=False,
            append=True,
        )

    # -- Load: route through HBM, evictions to host --

    def load(
        self,
        table_id: int,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool = True,
    ) -> Optional[int]:
        params = _validate_load_meta(
            self._hbm,
            table_id,
            meta_json_file_path,
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path,
            include_optim,
        )

        is_lru_hbm = self._hbm.evict_strategy == EvictStrategy.KLru
        is_lru_host = self._host.evict_strategy == EvictStrategy.KLru
        if is_lru_hbm or is_lru_host:
            ts = device_timestamp()
            if is_lru_hbm:
                self._hbm.timestamp = ts
            if is_lru_host:
                self._host.timestamp = ts

        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        for keys, embeddings, file_scores, opt_states in _iter_batches_from_files(
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path if params.include_optim else None,
            params.dim,
            params.optstate_dim,
            device,
        ):
            if keys.numel() == 0:
                continue

            if opt_states is None and params.optstate_dim > 0:
                opt_states = (
                    torch.ones(
                        keys.numel(),
                        params.optstate_dim,
                        dtype=self._hbm.emb_dtype,
                        device=device,
                    )
                    * self._hbm.initial_optim_state
                )

            vtype = self._hbm.emb_dtype
            values = (
                torch.cat(
                    [embeddings.to(vtype), opt_states.to(vtype)],
                    dim=-1,
                )
                if opt_states is not None
                else embeddings.to(vtype)
            )

            tids = torch.full(
                (keys.numel(),), table_id, dtype=torch.int64, device=device
            )

            (
                ins_indices,
                num_evicted,
                evicted_keys,
                evicted_table_ids,
                evicted_indices,
                evicted_scores,
            ) = _insert_and_evict_keys(self._hbm, keys, tids, file_scores)

            evicted_values = load_from_flat(
                self._hbm,
                evicted_indices,
                evicted_table_ids,
                copy_mode=CopyMode.VALUE,
            )
            select_insert_failed_values(evicted_indices, values, evicted_values)
            store_to_flat_single_table(self._hbm, ins_indices, table_id, values)

            if num_evicted != 0:
                _insert_key_values(
                    self._host,
                    evicted_keys,
                    evicted_table_ids,
                    evicted_values,
                    evicted_scores,
                )

        return params.meta_data.get("step_score", None)

    # -- Export: yield from both tiers --

    def export_keys_values(
        self,
        device: torch.device,
        batch_size: int = 65536,
        table_id: int = 0,
    ) -> Iterator[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]
    ]:
        yield from export_keys_values_iter(self._hbm, device, batch_size, table_id)
        yield from export_keys_values_iter(self._host, device, batch_size, table_id)


# ---------------------------------------------------------------------------
# Higher-level free functions
# ---------------------------------------------------------------------------


def flush_cache(cache: DynamicEmbCache, storage: Storage) -> None:
    state = cache._state
    batch_size = state.threads_in_wave
    state.value_dim

    for t in range(state.num_tables):
        for (
            keys,
            named_scores,
            indices,
        ) in state.key_index_map._batched_export_keys_scores(
            [state.score_policy.name],
            state.device,
            batch_size=batch_size,
            return_index=True,
            table_id=t,
        ):
            scores = named_scores[state.score_policy.name]
            tid = torch.full((keys.numel(),), t, dtype=torch.int64, device=keys.device)
            values = load_from_flat(state, indices, tid, copy_mode=CopyMode.VALUE)
            storage.insert(keys, tid, values, scores)


# ---------------------------------------------------------------------------
# eval_lookup – unified eval path for storage-only and cache+storage
# ---------------------------------------------------------------------------


def _eval_lookup_storage(
    storage: Storage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    initializer: Callable,
) -> torch.Tensor:
    (
        h_num_missing,
        _,
        missing_indices,
        _,
        _,
        _,
        _,
        embs,
    ) = storage.find(
        keys,
        table_ids,
        copy_mode=CopyMode.EMBEDDING,
    )

    if h_num_missing > 0:
        initializer(embs, missing_indices, keys)

    return embs


def _eval_lookup_cached(
    cache: Cache,
    storage: Storage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    initializer: Callable,
) -> torch.Tensor:
    _, founds, cache_indices = cache.lookup(keys, table_ids)

    embs = load_from_flat(
        cache._state, cache_indices, table_ids, copy_mode=CopyMode.EMBEDDING
    )

    missing_mask = ~founds
    h_num_miss, miss_compact_idx, (missing_keys, missing_table_ids) = flagged_compact(
        missing_mask, [keys, table_ids]
    )

    if h_num_miss == 0:
        return embs

    (
        h_num_missing_in_storage,
        _,
        missing_indices_in_storage,
        _,
        _,
        _,
        _,
        storage_embs,
    ) = storage.find(
        missing_keys,
        missing_table_ids,
        copy_mode=CopyMode.EMBEDDING,
    )

    if h_num_missing_in_storage > 0:
        initializer(storage_embs, missing_indices_in_storage, missing_keys)

    embs[miss_compact_idx, :] = storage_embs

    return embs


def eval_lookup(
    storage: Storage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    initializer: Callable,
    cache: Optional[Cache] = None,
) -> torch.Tensor:
    """Eval-only lookup (no insertion, no admission, no backward).

    When *cache* is ``None``, looks up directly from *storage*.
    When *cache* is provided, looks up from cache first, then falls back to
    *storage* for cache misses.  Only embedding columns are copied
    (``CopyMode.EMBEDDING``); optimizer states are never touched.

    Returns the embedding tensor of shape ``[len(keys), emb_dim]``.
    """
    assert keys.dim() == 1
    if keys.numel() == 0:
        return torch.empty(
            0,
            storage.max_embedding_dim(),
            dtype=storage.embedding_dtype(),
            device=keys.device,
        )

    if cache is None:
        return _eval_lookup_storage(storage, keys, table_ids, initializer)

    return _eval_lookup_cached(
        cache,
        storage,
        keys,
        table_ids,
        initializer,
    )
