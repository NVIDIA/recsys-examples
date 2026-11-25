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

"""Dumping a ``DynamicEmbTableState``: iterate its keys, values and scores, and write a table's checkpoint files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from dynamicemb.checkpoint.files import (
    META_EMBEDDING_DIM,
    META_EMBEDDING_DTYPE,
    META_OPT_STATE_DTYPE,
    _raw_bytes,
    dtype_to_meta_name,
    save_to_json,
)
from dynamicemb.score.strategy import score_dump_permutation
from dynamicemb.state.value_rows import load_from_flat_single_table
from dynamicemb.types import SCORE_TYPE
from dynamicemb.utils import checkpoint_dtype, decode_table_values
from dynamicemb_extensions import EvictStrategy
from torch import Tensor

if TYPE_CHECKING:
    from dynamicemb.key_value_table import DynamicEmbTableState


def export_keys_values_iter(
    state: DynamicEmbTableState,
    device: torch.device,
    batch_size: int = 65536,
    table_id: int = 0,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
    """Export keys, embeddings, opt_states, scores for a logical table.

    Embeddings and optimizer states come out at the table's own precision
    (``state.emb_dtype``), matching what ``incremental_dump`` yields, so a
    checkpoint stores what the table actually holds instead of an fp32 widening
    of it. Callers that want fp32 cast on their own.
    """
    emb_dim_t = state.table_emb_dims_cpu[table_id]
    vdim = state.table_value_dims_cpu[table_id]
    optim_state_dim = vdim - emb_dim_t

    for (
        keys,
        named_scores,
        indices,
        slots,
    ) in state.key_index_map._batched_export_keys_scores(
        [state.score_policy.name],
        state.device,
        batch_size=batch_size,
        return_index=True,
        return_slots=True,
        table_id=table_id,
    ):
        scores = named_scores[state.score_policy.name]
        flat_rows = indices
        values = load_from_flat_single_table(state, flat_rows, table_id)
        embeddings = decode_table_values(
            values[:, :emb_dim_t], checkpoint_dtype(state.emb_dtype)
        ).contiguous()
        if optim_state_dim != 0:
            opt_states = values[:, -optim_state_dim:].contiguous().to(device)
        else:
            opt_states = None
        # Multi-word score layouts (e.g. LruLfu: timestamp + frequency) must
        # persist ALL words, not just the exported column, so the checkpoint can
        # restore them exactly. Gather the full [N, num_scores] block at the
        # exported slots; single-score tables keep the [N] shape unchanged.
        if state.key_index_map.num_scores_ > 1:
            # Score blocks are keyed by slot; ``indices`` are rows for a table
            # that stores them.
            out_scores = state.key_index_map.gather_score_blocks(table_id, slots)
            # Reorder the physical device layout into the user's configured
            # (logical) score-column order so both the checkpoint and callers of
            # export_keys_values see scores in the order they configured. Identity
            # when the logical order matches the physical layout.
            dump_perm = score_dump_permutation(
                state.options_list[table_id].score_strategy
            )
            if dump_perm != list(range(out_scores.size(1))):
                out_scores = out_scores[:, dump_perm].contiguous()
        else:
            out_scores = scores.to(SCORE_TYPE)
        yield (
            keys.to(device),
            embeddings.to(device),
            opt_states,
            out_scores.to(device),
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
    timestamp: int,
    current_score: Optional[int] = None,
    append: bool = False,
) -> None:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    if not append and include_meta:
        meta_data = {}
        meta_data.update(state.optimizer.get_opt_args())
        meta_data["evict_strategy"] = str(state.evict_strategy)
        meta_data["dist_type"] = state.options_list[table_id].dist_type
        # How to read the value files back. They carry no header, so without
        # these a loader can only assume the historical fp32 and guess the row
        # width from the file length. Note this dict is rebuilt and rewritten
        # whole on every dump (by rank 0 only, and not by the appending second
        # tier of a hybrid dump): anything a loader needs has to be contributed
        # here, because there is no later pass that merges into the file.
        meta_data[META_EMBEDDING_DTYPE] = dtype_to_meta_name(
            checkpoint_dtype(state.emb_dtype)
        )
        meta_data[META_EMBEDDING_DIM] = int(state.table_emb_dims_cpu[table_id])
        # Recorded separately from the embedding dtype because the two can
        # differ: rowwise Adagrad keeps its accumulator in fp32 in any table.
        meta_data[META_OPT_STATE_DTYPE] = dtype_to_meta_name(
            state.optimizer.get_state_dtype(state.emb_dtype)
        )

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
        fembedding.write(_raw_bytes(embeddings))
        if state.evict_strategy == EvictStrategy.KLru:
            scores = timestamp - scores
        fscore.write(scores.cpu().numpy().tobytes())
        if fopt_states and opt_states_batch is not None:
            to_write = state.optimizer.states_for_checkpoint(
                opt_states_batch,
                state.table_emb_dims_cpu[table_id],
            )
            fopt_states.write(_raw_bytes(to_write))

    fkey.close()
    fembedding.close()
    if fscore:
        fscore.close()
    if fopt_states:
        fopt_states.close()


def _all_gather_dumped_columns(
    columns: List[Tensor],
    pg: dist.ProcessGroup,
) -> List[Tensor]:
    """Gather a set of row-aligned dump columns from all ranks into CPU tensors.

    Every tensor in *columns* must live on the same device and share ``size(0)``
    (the per-rank dumped-key count); trailing dimensions may differ (e.g. keys
    ``(N,)``, values ``(N, D)``, scores ``(N, W)``). Returns the same columns with
    all ranks' rows concatenated in rank order, on host and still row-aligned.

    NCCL has no variable-length gather, so this is one size all_gather followed by
    a padded all_gather per column.
    """
    assert columns, "_all_gather_dumped_columns needs at least one column"
    device = columns[0].device
    world_size = dist.get_world_size(group=pg)
    n = columns[0].size(0)
    d_count = torch.tensor([n], dtype=torch.long, device=device)
    gathered_counts = [torch.empty_like(d_count) for _ in range(world_size)]
    dist.all_gather(gathered_counts, d_count, group=pg)
    counts = [int(c.item()) for c in gathered_counts]
    max_n = max(counts)

    out: List[Tensor] = []
    for col in columns:
        padded = torch.zeros(
            (max_n,) + tuple(col.shape[1:]), dtype=col.dtype, device=device
        )
        if n > 0:
            padded[:n] = col
        gathered = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded, group=pg)
        out.append(
            torch.cat(
                [gathered[i][: counts[i]] for i in range(world_size)], dim=0
            ).cpu()
        )
    return out
