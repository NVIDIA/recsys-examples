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

"""Loading a ``DynamicEmbTableState`` from a table's checkpoint files: validate the meta, then insert the keys, values and scores."""

from __future__ import annotations

import contextlib
import os
import threading
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
from dynamicemb.checkpoint.files import load_from_json, resolve_checkpoint_value_layout
from dynamicemb.dynamicemb_config import DynamicEmbTableOptions
from dynamicemb.key_ownership import OwnershipRule
from dynamicemb.score.args import ScoreArg
from dynamicemb.types import (
    EMBEDDING_TYPE,
    KEY_TYPE,
    OPT_STATE_TYPE,
    SCORE_TYPE,
    torch_dtype_to_np_dtype,
)
from dynamicemb.utils import checkpoint_dtype, encode_table_values
from dynamicemb_extensions import EvictStrategy, ScorePolicy

if TYPE_CHECKING:
    from dynamicemb.state.table_state import DynamicEmbTableState


def _resolve_shard(
    shard_index: Optional[int], num_shards: Optional[int]
) -> Tuple[int, int]:
    """The table's own shard geometry, or the global rank and world size (the
    row-wise answer) when either half is left ``None``."""
    if num_shards is None or shard_index is None:
        num_shards = dist.get_world_size() if dist.is_initialized() else 1
        shard_index = dist.get_rank() if dist.is_initialized() else 0
    return shard_index, num_shards


def _count_owned_keys(
    emb_key_path: str,
    device: torch.device,
    dist_type: str,
    batch_size: int = 65536,
    shard_index: Optional[int] = None,
    num_shards: Optional[int] = None,
    global_capacity: Optional[int] = None,
    bucket_capacity: Optional[int] = None,
) -> int:
    """How many of a checkpoint file's keys this rank keeps.

    Reads only the key file, with the filter :func:`_iter_batches_from_files`
    applies. A checkpoint can hold many ranks' keys while a rank keeps only its
    own share, so the file's size is not what a rank has to make room for.
    """
    num_keys = os.path.getsize(emb_key_path) // KEY_TYPE.itemsize
    shard_index, num_shards = _resolve_shard(shard_index, num_shards)
    if num_shards <= 1:
        return num_keys
    rule = OwnershipRule(dist_type, num_shards, global_capacity, bucket_capacity)
    owned = 0
    with open(emb_key_path, "rb") as fkey:
        for start in range(0, num_keys, batch_size):
            n = min(num_keys - start, batch_size)
            keys = torch.tensor(
                np.frombuffer(
                    fkey.read(KEY_TYPE.itemsize * n),
                    dtype=torch_dtype_to_np_dtype[KEY_TYPE],
                ),
                dtype=KEY_TYPE,
                device=device,
            )
            mask = rule.mask(keys, shard_index)
            owned += int(mask.sum().item()) if mask is not None else n
    return owned


@dataclass
class _LoadParams:
    meta_data: Dict[str, Any]
    dim: int
    runtime_optstate_dim: int
    file_optstate_dim: int
    include_optim: bool
    num_keys: int
    # The table's key -> rank rule, already checked against the checkpoint's,
    # with the parameters hash_bucket needs (None for the other rules).
    dist_type: str
    global_capacity: Optional[int] = None
    bucket_capacity: Optional[int] = None
    # Precisions the value files were written at, resolved from meta (fp32 for
    # checkpoints predating it). Pass both to ``_iter_batches_from_files``.
    emb_dtype: torch.dtype = EMBEDDING_TYPE
    opt_state_dtype: torch.dtype = OPT_STATE_TYPE


def check_dist_type_migration(
    ckpt_dist_type: str, runtime_dist_type: str, meta_json_file_path: str
) -> None:
    """Refuse a load across ``dist_type`` when either side is ``continuous``.

    A ``continuous`` table stores shard-local ids (the bucketizer rewrites a key
    to ``key % block_size``), so two source shards can hold the same id for
    different rows: re-partitioning those ids by another rule would merge
    distinct rows and attach values to the wrong keys even with no bucket
    overflowing. The other rules store the global key, so a key says which
    rank owns it under any of them, and a migration between them is allowed
    (lossy only where a bucket overflows).
    """
    if ckpt_dist_type == runtime_dist_type:
        return
    if "continuous" in (ckpt_dist_type, runtime_dist_type):
        raise ValueError(
            f"{meta_json_file_path}: checkpoint was dumped with dist_type "
            f"{ckpt_dist_type!r} but the table uses {runtime_dist_type!r}. A "
            "'continuous' table stores shard-local ids, which no other rule can "
            "re-partition. Load it into a table of the same dist_type."
        )


# Layout warnings are said once per checkpoint table in a process: a load reads
# one file per source shard, and the warning is about the table, not the file.
_LAYOUT_WARNED: Set[str] = set()


_layout_warnings_suppressed = threading.local()


@contextlib.contextmanager
def suppress_layout_warnings():
    """Silence the per-file layout warnings of ``Storage.load``.

    For a caller that has already said it once -- ``BatchedDynamicEmbeddingTablesV2.load``
    warns per table on shard 0 before it hands the files to the storage.
    """
    previous = getattr(_layout_warnings_suppressed, "depth", 0)
    _layout_warnings_suppressed.depth = previous + 1
    try:
        yield
    finally:
        _layout_warnings_suppressed.depth = previous


def _warn_layout_once(meta_json_file_path: str, message: str) -> None:
    if getattr(_layout_warnings_suppressed, "depth", 0) > 0:
        return
    if meta_json_file_path in _LAYOUT_WARNED:
        return
    _LAYOUT_WARNED.add(meta_json_file_path)
    warnings.warn(message, UserWarning, stacklevel=3)


def _validate_load_meta(
    state: DynamicEmbTableState,
    table_id: int,
    meta_json_file_path: str,
    emb_key_path: str,
    embedding_file_path: str,
    score_file_path: Optional[str],
    opt_file_path: Optional[str],
    include_optim: bool,
    routing_options: Optional[DynamicEmbTableOptions] = None,
) -> _LoadParams:
    """Shared validation for checkpoint loading.

    Reads meta JSON, validates opt_type / evict_strategy / dist_type, resolves
    the value files' precision and row width, resolves include_optim, and checks
    file-size consistency.

    *routing_options* are the options the key -> rank rule is read from; they
    default to this state's own, and a storage whose state is a rescaled tier
    (``HybridStorage``) passes the tier that kept the shard's bucket capacity.
    """
    meta_data = load_from_json(meta_json_file_path)
    opt_type = meta_data.get("opt_type", None)
    # The optimizer checks only matter when the caller asked for its states.
    if (
        include_optim
        and opt_type
        and state.optimizer.get_opt_args().get("opt_type", None) != opt_type
    ):
        include_optim = False
        print(
            f"Optimizer type mismatch: {opt_type} != {state.optimizer.get_opt_args().get('opt_type')}. Will not load optimizer states."
        )

    evict_strategy = meta_data.get("evict_strategy", None)
    if evict_strategy and str(state.evict_strategy) != evict_strategy:
        raise ValueError(
            f"Evict strategy mismatch: {evict_strategy} != {state.evict_strategy}"
        )
    if routing_options is None:
        routing_options = state.options_list[table_id]
    ckpt_dist_type = meta_data.get("dist_type", "roundrobin")
    runtime_dist_type = routing_options.dist_type
    runtime_global_capacity = getattr(routing_options, "global_capacity", None)
    if runtime_dist_type != ckpt_dist_type:
        check_dist_type_migration(
            ckpt_dist_type, runtime_dist_type, meta_json_file_path
        )
        # Ownership is recomputed from the key by the *table's* rule, so a
        # checkpoint written under another rule still loads -- provided every
        # rank reads every file (BatchedDynamicEmbeddingTablesV2.load reads
        # all files when the rules differ). Keys then land in buckets the
        # source never grouped them by, and a full bucket evicts: a one-time
        # loss on migration, which is why this is a warning and not silent.
        _warn_layout_once(
            meta_json_file_path,
            f"{meta_json_file_path}: checkpoint was dumped with dist_type "
            f"{ckpt_dist_type!r} but the table uses {runtime_dist_type!r}; keys "
            "are re-partitioned by the table's rule, and rows may be lost where "
            "a bucket overflows. Every rank must read every shard file.",
        )
    elif runtime_dist_type == "hash_bucket" and (
        meta_data.get("global_capacity") != runtime_global_capacity
        or meta_data.get("bucket_capacity") != routing_options.bucket_capacity
    ):
        _warn_layout_once(
            meta_json_file_path,
            f"{meta_json_file_path}: checkpoint has global bucket layout "
            f"(global_capacity={meta_data.get('global_capacity')}, "
            f"bucket_capacity={meta_data.get('bucket_capacity')}) but the table "
            f"has ({runtime_global_capacity}, {routing_options.bucket_capacity}); "
            "keys are re-partitioned by the table's layout, and rows may be lost "
            "where a bucket overflows. Every rank must read every shard file.",
        )

    if score_file_path is None:
        print(
            f"Score file {score_file_path} does not exist. Will not load score states."
        )

    if include_optim and (not opt_file_path or not os.path.exists(opt_file_path)):
        include_optim = False
        print(
            f"Optimizer file {opt_file_path} does not exist. Will not load optimizer states."
        )

    dim = state.buffers.emb_dims_list[table_id]
    runtime_optstate_dim = state.optimizer.get_state_dim(dim)

    if runtime_optstate_dim == 0:
        include_optim = False

    if include_optim:
        state.optimizer.set_opt_args(meta_data)

    num_keys = os.path.getsize(emb_key_path) // KEY_TYPE.itemsize
    emb_dtype, opt_state_dtype, ckpt_emb_dim = resolve_checkpoint_value_layout(
        meta_data, embedding_file_path, num_keys, dim
    )
    if ckpt_emb_dim != dim:
        raise ValueError(
            f"Embedding dim mismatch: {embedding_file_path} holds rows of "
            f"{ckpt_emb_dim} but the runtime table is configured with dim {dim}."
        )
    if emb_dtype not in (state.buffers.dtype, checkpoint_dtype(state.buffers.dtype)):
        # Not fatal: the insert path casts, so this is a deliberate precision
        # migration as often as it is a mistake. Say which way it goes, since
        # narrowing loses bits that the checkpoint still has.
        warnings.warn(
            f"Checkpoint {embedding_file_path} stores {emb_dtype} embeddings but "
            f"the runtime table is {state.buffers.dtype}; values will be converted on "
            "load.",
            UserWarning,
            stacklevel=2,
        )
    if score_file_path and os.path.exists(score_file_path):
        # The score file holds num_scores words per key (e.g. LruLfu = 2:
        # timestamp + frequency), so compare against num_keys * num_scores.
        per_key = state.key_index_map.num_scores_
        num_score_words = os.path.getsize(score_file_path) // SCORE_TYPE.itemsize
        if num_score_words != num_keys * per_key:
            raise ValueError(
                f"The number of keys in {emb_key_path} does not match with number of scores in {score_file_path}."
            )

    file_optstate_dim = 0
    if include_optim and opt_file_path and os.path.exists(opt_file_path):
        file_bytes = os.path.getsize(opt_file_path)
        if num_keys == 0:
            if file_bytes != 0:
                raise ValueError(
                    f"Optimizer state file {opt_file_path} is non-empty but key file has no keys."
                )
        else:
            row_block = num_keys * opt_state_dtype.itemsize
            if file_bytes % row_block != 0:
                raise ValueError(
                    f"Optimizer state file {opt_file_path} size {file_bytes} is not divisible by "
                    f"{row_block} (num_keys={num_keys}, dtype itemsize={opt_state_dtype.itemsize})."
                )
            file_optstate_dim = file_bytes // row_block
            ckpt_dim = state.optimizer.get_ckpt_state_dim(dim)
            if file_optstate_dim != ckpt_dim:
                raise ValueError(
                    f"Optimizer state width in checkpoint is {file_optstate_dim}; expected "
                    f"{ckpt_dim}."
                )

    return _LoadParams(
        meta_data=meta_data,
        dim=dim,
        runtime_optstate_dim=runtime_optstate_dim,
        file_optstate_dim=file_optstate_dim,
        include_optim=include_optim,
        num_keys=num_keys,
        dist_type=runtime_dist_type,
        global_capacity=runtime_global_capacity,
        bucket_capacity=routing_options.bucket_capacity,
        emb_dtype=emb_dtype,
        opt_state_dtype=opt_state_dtype,
    )


def _load_key_values(
    state: DynamicEmbTableState,
    keys: torch.Tensor,
    embeddings: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    opt_states: Optional[torch.Tensor] = None,
    table_id: int = 0,
) -> None:
    dim = embeddings.size(1)
    emb_dim_cfg = state.buffers.emb_dims_list[table_id]
    runtime_optstate_dim = state.optimizer.get_state_dim(emb_dim_cfg)
    if not keys.is_cuda:
        raise RuntimeError("Keys must be on GPU")
    if not embeddings.is_cuda:
        raise RuntimeError("Embeddings must be on GPU")
    if scores is not None and not scores.is_cuda:
        raise RuntimeError("Scores must be on GPU")
    if opt_states is not None and not opt_states.is_cuda:
        raise RuntimeError("Opt states must be on GPU")

    # Normalize the embeddings to the table's precision up front. A checkpoint
    # may carry a different precision than the table (load converts -- see
    # _validate_load_meta). The optimizer state is deliberately left alone:
    # optimizer.states_from_checkpoint owns its conversion, because it need not
    # be the table's dtype (rowwise Adagrad keeps an fp32 accumulator), and an
    # elementwise cast here would round that accumulator away. The branch below
    # only calls it when the state is non-empty, which is the case it casts.
    embeddings = encode_table_values(embeddings, state.buffers.dtype)

    if opt_states is None and runtime_optstate_dim > 0:
        opt_states = torch.empty(
            keys.numel(),
            runtime_optstate_dim,
            dtype=state.buffers.dtype,
            device=embeddings.device,
        )
        state.optimizer.reset_optimizer_states(opt_states, emb_dims=emb_dim_cfg)
    elif opt_states is not None and runtime_optstate_dim > 0:
        opt_states = state.optimizer.states_from_checkpoint(
            opt_states,
            emb_dim_cfg,
            state.buffers.dtype,
            embeddings.device,
        )

    # Both halves are already state.buffers.dtype (normalized above), so this cat
    # cannot silently widen the row via type promotion.
    values = (
        torch.cat(
            [embeddings.view(-1, dim), opt_states.view(-1, runtime_optstate_dim)],
            dim=-1,
        )
        if opt_states is not None
        else embeddings
    )

    # Multi-word score layouts (LruLfu: timestamp + frequency) cannot be restored
    # by a single-value insert (which only writes word 0). Place the keys, then
    # scatter the full [N, num_scores] block. Placement uses the table's own
    # policy with a zero frequency delta so that a full bucket evicts through the
    # ranked LruLfu evictor (honoring a custom score_function); the scatter then
    # overwrites both words.
    num_scores = state.key_index_map.num_scores_
    if num_scores > 1:
        # Validate the score-block shape before scatter_score_blocks writes it
        # into device memory: a malformed checkpoint (e.g. a 1-D score tensor)
        # would otherwise produce out-of-bounds / wrong writes. Use ValueError
        # (not assert) so the check survives `python -O`.
        if (
            scores is None
            or scores.dim() != 2
            or scores.size(1) != num_scores
            or scores.size(0) != keys.numel()
        ):
            raise ValueError(
                f"multi-word load expects [{keys.numel()}, {num_scores}] scores, "
                f"got {None if scores is None else tuple(scores.shape)}"
            )
        tid_tensor = torch.full(
            (keys.numel(),), table_id, dtype=torch.int64, device=keys.device
        )
        place_arg = ScoreArg(
            name=state.score_policy.name,
            value=torch.zeros(keys.numel(), dtype=SCORE_TYPE, device=keys.device),
            policy=state.score_policy.policy,
        )
        indices = state.key_index_map.insert(keys, tid_tensor, place_arg)
        # The score blocks are keyed by slot, the values by row (the same thing
        # unless the table stores rows).
        state.key_index_map.scatter_score_blocks(
            table_id,
            state.key_index_map.slots_of(keys, tid_tensor),
            scores.to(SCORE_TYPE).contiguous(),
        )
        state.buffers.store(table_id, indices, values)
        return

    policy = ScorePolicy.ASSIGN
    tid_tensor = torch.full(
        (keys.numel(),), table_id, dtype=torch.int64, device=keys.device
    )

    if state.score_policy.policy == ScorePolicy.FLAG:
        # A table that keeps no score (NO_EVICTION alone): none is restored, but the
        # keys loaded set their flag like any new key.
        policy, scores = ScorePolicy.FLAG, None
    elif scores is None:
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

    indices = state.key_index_map.insert(keys, tid_tensor, score_arg_insert)
    state.buffers.store(table_id, indices, values)
