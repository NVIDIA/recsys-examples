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

"""Replaying a delta into a ``DynamicEmbTableState``: write the delta's keys, values and scores back."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from dynamicemb.dynamicemb_config import ReplayContent
from dynamicemb.score.checkpoint_columns import _fresh_score_block, _replay_score_block
from dynamicemb.state.value_rows import (
    _row_differs_from_slot,
    store_to_flat_single_table,
)
from dynamicemb.types import ReplayStats
from dynamicemb.utils import encode_table_values

if TYPE_CHECKING:
    from dynamicemb.key_value_table import DynamicEmbTableState


def _replay_write_values(
    state: DynamicEmbTableState,
    table_id: int,
    rows: torch.Tensor,
    embeddings: torch.Tensor,
    optimizer_states: Optional[torch.Tensor],
    keep_optimizer: torch.Tensor,
    content: ReplayContent,
) -> None:
    """Write a replayed row's value columns, per :class:`ReplayContent`.

    A value row is an embedding followed by the optimizer state, and the two are
    written together or not at all -- ``store_to_flat_single_table`` copies from
    the row base, so there is no way to write the tail without the head. The
    embedding is therefore always written, and the only choice is what the tail
    gets:

    - **With ``OPTIMIZER_STATE``**, the delta carries the whole row; write it as
      it was dumped. Nothing is inferred.
    - **Without it**, rows that already belonged to this same key keep their
      state, because writing just the embedding columns leaves the tail
      untouched. Every other row is new to this key, so its tail still holds the
      previous occupant's moments and has to be reset by the optimizer.
    """
    if rows.numel() == 0:
        return
    emb_dim_cfg = state.table_emb_dims_cpu[table_id]
    optstate_dim = state.optimizer.get_state_dim(emb_dim_cfg)
    embeddings = encode_table_values(embeddings, state.emb_dtype)
    want_opt = ReplayContent.OPTIMIZER_STATE in content and optstate_dim > 0

    if optstate_dim == 0:
        store_to_flat_single_table(state, rows, table_id, embeddings)
        return

    if want_opt:
        if optimizer_states is None:
            raise ValueError(
                f"replay_increment: table {table_id} keeps optimizer state per "
                "row, but the delta carries none. Drop "
                "ReplayContent.OPTIMIZER_STATE, or replay a delta dumped from a "
                "table with the same optimizer."
            )
        ckpt_dim = state.optimizer.get_ckpt_state_dim(emb_dim_cfg)
        if optimizer_states.dim() != 2 or optimizer_states.size(1) != ckpt_dim:
            raise ValueError(
                f"replay_increment: table {table_id} dumps {ckpt_dim} "
                f"optimizer-state column(s) per row, but the delta carries a "
                f"block of shape {tuple(optimizer_states.shape)}."
            )
        # Back to the runtime width the fused value row expects.
        opt = state.optimizer.states_from_checkpoint(
            optimizer_states.to(device=embeddings.device),
            emb_dim_cfg,
            state.emb_dtype,
            embeddings.device,
        )
        store_to_flat_single_table(
            state, rows, table_id, torch.cat([embeddings, opt], dim=-1)
        )
        return

    keep = keep_optimizer
    if bool(keep.any()):
        store_to_flat_single_table(state, rows[keep], table_id, embeddings[keep])
    fresh = torch.logical_not(keep)
    if bool(fresh.any()):
        fresh_emb = embeddings[fresh]
        opt_states = torch.empty(
            (fresh_emb.size(0), optstate_dim),
            dtype=state.emb_dtype,
            device=fresh_emb.device,
        )
        state.optimizer.reset_optimizer_states(opt_states, emb_dims=emb_dim_cfg)
        store_to_flat_single_table(
            state,
            rows[fresh],
            table_id,
            torch.cat([fresh_emb, opt_states], dim=-1),
        )


def _replay_at_slots(
    state: DynamicEmbTableState,
    table_id: int,
    keys: torch.Tensor,
    embeddings: torch.Tensor,
    optimizer_states: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
    slot_index: torch.Tensor,
    timestamp: int,
    insert_score: int,
    content: ReplayContent,
) -> int:
    """Write keys at the exact slots they held in the source table.

    Every key must land: the caller has already established that this table's
    layout matches the source's, so a slot that does not fall inside its key's
    home bucket means the two have diverged anyway and the replay would silently
    lose that key. Raises rather than writing a partial result.

    Returns the number of keys written (always all of them, or it raised).
    """
    n = keys.numel()
    device = state.device
    if n == 0:
        return 0
    tids = torch.full((n,), table_id, dtype=torch.int64, device=device)

    if _row_differs_from_slot(state):
        # ``use_rows`` tables carry (key slot, value row) as ``[n, 2]``; every
        # other table uses one index for both.
        key_slots = slot_index[:, 0].contiguous()
        value_rows = slot_index[:, 1].contiguous()
    else:
        key_slots = slot_index
        value_rows = slot_index

    # The score words are written with the key, so they are chosen here: either
    # restored from the delta or minted as if the key had just been inserted.
    if ReplayContent.SCORE in content and scores is not None:
        score_block = _replay_score_block(state, table_id, scores, timestamp)
    else:
        score_block = _fresh_score_block(state, table_id, n, timestamp, insert_score)
    status, same_key = state.key_index_map.scatter_keys_at_slots(
        keys, tids, key_slots, score_block
    )
    placed = status >= 0
    num_unplaced = int(n - placed.sum().item())
    if num_unplaced:
        example = int(keys[torch.logical_not(placed)][0].item())
        raise RuntimeError(
            f"replay_increment: {num_unplaced} of {n} keys could not be written "
            f"at their source slot on table {table_id} (e.g. key {example}). "
            "Either the slot falls outside its key's home bucket -- where no "
            "lookup would ever probe it, meaning the target's hash layout "
            "differs from the source's despite matching metadata -- or the slot "
            "was held by another writer, meaning something ran against this "
            "table concurrently with the replay. Replaying past either would "
            "silently drop those keys."
        )

    if state.key_index_map.use_rows_:
        # The key now sits at the source's slot; make it own the source's row.
        # The row counter is set by the caller from the delta's metadata -- the
        # rows named here are only the ones that changed.
        state.key_index_map.set_rows(key_slots, table_id, value_rows)

    _replay_write_values(
        state, table_id, value_rows, embeddings, optimizer_states, same_key, content
    )
    return n


def _replay_state_increment(
    state: DynamicEmbTableState,
    table_id: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    optimizer_states: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
    slot_index: torch.Tensor,
    timestamp: int,
    insert_score: int,
    content: ReplayContent,
) -> ReplayStats:
    """Replay one delta batch into a single table state (one storage tier)."""
    stats = ReplayStats()
    n = keys.numel()
    if n == 0:
        return stats
    device = state.device
    keys = keys.to(device=device, dtype=state.key_index_map.key_type)
    embeddings = encode_table_values(values.to(device=device), state.emb_dtype)
    slots = slot_index.to(device=device, dtype=torch.int64)
    stats.upserted = _replay_at_slots(
        state,
        table_id,
        keys,
        embeddings,
        optimizer_states,
        scores,
        slots,
        timestamp,
        insert_score,
        content,
    )
    return stats
