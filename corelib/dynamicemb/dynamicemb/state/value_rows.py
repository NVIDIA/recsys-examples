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

"""Reading and writing the value rows of a ``DynamicEmbTableState``.

A row is addressed by its slot (row number) in a table's value buffer, the "flat"
layout the kernels use."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from dynamicemb.types import CopyMode
from dynamicemb.utils import checkpoint_dtype, decode_table_values, encode_table_values
from dynamicemb_extensions import load_from_flat_table_contiguous as _load_contiguous
from dynamicemb_extensions import load_from_flat_table_emb as _load_emb
from dynamicemb_extensions import load_from_flat_table_value as _load_value
from dynamicemb_extensions import store_to_flat_table_contiguous as _store_contiguous
from dynamicemb_extensions import store_to_flat_table_value as _store_value

if TYPE_CHECKING:
    from dynamicemb.state.table_state import DynamicEmbTableState


def _row_differs_from_slot(state: DynamicEmbTableState) -> bool:
    """Whether a key's value row is not its hash slot, i.e. the table stores a row
    per key (``use_rows``). A dumped ``slot_index`` then has to carry both."""
    return getattr(state.key_index_map, "use_rows_", False)


def load_from_flat(
    state: DynamicEmbTableState,
    indices: torch.Tensor,
    table_ids: torch.Tensor,
    copy_mode: CopyMode,
) -> torch.Tensor:
    # The key map hands rows back on the host after a dump.
    indices = indices.to(device=state.device, dtype=torch.int64)
    N = indices.numel()
    if copy_mode == CopyMode.EMBEDDING:
        max_dim = state.buffers.max_emb_dim
        _load = _load_emb
    else:
        max_dim = state.buffers.max_value_dim
        _load = _load_value
    output = torch.empty(N, max_dim, dtype=state.buffers.dtype, device=state.device)
    if N > 0:
        _load(
            state.buffers.ptrs,
            indices,
            table_ids,
            output,
            state.buffers.value_dims,
            state.buffers.emb_dims,
            state.buffers.max_emb_dim,
            state.buffers.all_dims_vec4,
        )
    return output


def load_from_flat_by(
    state: DynamicEmbTableState,
    indices: torch.Tensor,
    table_ids: torch.Tensor,
    entry_of_row: torch.Tensor,
    out: torch.Tensor,
    out_rows: Optional[torch.Tensor] = None,
) -> None:
    """Into row i of ``out`` (``[entry_of_row.numel(), columns]``, of the
    tables' dtype; row ``out_rows[i]`` with ``out_rows``, int32) the embedding
    of row ``indices[entry_of_row[i]]`` of table ``table_ids[entry_of_row[i]]``;
    rows whose index is negative are left as they are."""
    _load_emb(
        state.buffers.ptrs,
        indices,
        table_ids,
        out,
        state.buffers.value_dims,
        state.buffers.emb_dims,
        state.buffers.max_emb_dim,
        state.buffers.all_dims_vec4,
        entry_of_row,
        out_rows,
    )


def store_to_flat(
    state: DynamicEmbTableState,
    indices: torch.Tensor,
    table_ids: torch.Tensor,
    values: torch.Tensor,
) -> None:
    if values.dim() == 1:
        values = values.unsqueeze(1)
    _store_value(
        state.buffers.ptrs,
        indices,
        table_ids,
        encode_table_values(values, state.buffers.dtype),
        state.buffers.value_dims,
        state.buffers.emb_dims,
        state.buffers.max_emb_dim,
        state.buffers.all_dims_vec4,
    )


def load_from_flat_single_table(
    state: DynamicEmbTableState,
    indices: torch.Tensor,
    table_id: int,
) -> torch.Tensor:
    """Load full values for a single table. Returns compact [N, value_dim_t]."""
    indices = indices.to(device=state.device, dtype=torch.int64)
    N = indices.numel()
    vdim = state.buffers.value_dims_list[table_id]
    output = torch.empty(N, vdim, dtype=state.buffers.dtype, device=state.device)
    if N > 0:
        _load_contiguous(
            state.buffers.ptrs,
            indices,
            table_id,
            output,
            state.buffers.value_dims,
            state.buffers.emb_dims,
            state.buffers.max_emb_dim,
            state.buffers.all_dims_vec4,
        )
    return output


def store_to_flat_single_table(
    state: DynamicEmbTableState,
    indices: torch.Tensor,
    table_id: int,
    values: torch.Tensor,
) -> None:
    """Store full values for a single table. Expects compact [N, value_dim_t]."""
    indices = indices.to(device=state.device, dtype=torch.int64)
    N = indices.numel()
    if N == 0:
        return
    if values.dim() == 1:
        values = values.unsqueeze(1)
    _store_contiguous(
        state.buffers.ptrs,
        indices,
        table_id,
        encode_table_values(values, state.buffers.dtype),
        state.buffers.value_dims,
        state.buffers.emb_dims,
        state.buffers.max_emb_dim,
        state.buffers.all_dims_vec4,
    )


def _encode_slot_index(
    state: DynamicEmbTableState,
    slot_indices: torch.Tensor,
    flat_rows: torch.Tensor,
    tier: Optional[int] = None,
) -> torch.Tensor:
    """int64 ``slot_index`` for a dumped table (for precise replay_increment).

    Single-tier storage (``tier is None``):
      - normal policies: the key_index_map slot IS the value flat row, so the
        single slot value locates both key and value -> return the slot as-is.
      - a table that stores its rows (``use_rows``): key slot and value row are
        independent, so ``slot_index`` is an ``[N, 2]`` int64 tensor, column 0 =
        key slot, column 1 = value row.

    HybridStorage (``tier in {0, 1}``, called once per tier, HBM=0 / host=1):
      layout ``bit 63 = tier | bits 0..62 = key_slot``. HybridStorage does NOT
      support NO_EVICTION (rejected at construction), so every tier is a normal
      policy with key_slot == value_row -- one value locates both. Only the tier
      bit is added so replay can tell the two tiers' independent slot spaces apart.
    """
    slot_indices = slot_indices.to(device=state.device, dtype=torch.int64)
    if tier is not None:  # HybridStorage: bit 63 = tier, bits 0..62 = key_slot
        if slot_indices.numel() > 0:
            assert int(slot_indices.max()) < (
                1 << 63
            ), "HybridStorage key_slot exceeds 63 bits"
        if int(tier):
            return slot_indices | (torch.ones_like(slot_indices) << 63)
        return slot_indices
    if not _row_differs_from_slot(state):
        return slot_indices  # key slot == value row
    flat_rows = flat_rows.to(device=state.device, dtype=torch.int64)
    return torch.stack([slot_indices, flat_rows], dim=1)


def _split_value_row(
    state: DynamicEmbTableState, table_id: int, values: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Split a loaded ``[N, value_dim]`` row into (embedding, optimizer state).

    ``load_from_flat_single_table`` returns the table's own compact row, so the
    optimizer state is the trailing ``value_dim - emb_dim`` columns. That runtime
    width is not all payload: rowwise Adagrad reserves a fixed 16 bytes per row
    in the fused FBGEMM layout but only ever fills one accumulator scalar. The
    dumped block is therefore narrowed to the width the file checkpoint uses --
    the same ``optimizer.states_for_checkpoint`` ``_dump_table``
    applies -- so a delta and a checkpoint describe a row identically, and the
    padding is not paid for on every dump. Replay expands it back with
    ``optimizer.states_from_checkpoint``.

    Tables whose optimizer keeps no per-row state (e.g. plain SGD) get ``None``
    rather than a zero-width tensor, matching :func:`export_keys_values_iter`.
    """
    emb_dim = state.buffers.emb_dims_list[table_id]
    optim_state_dim = state.buffers.value_dims_list[table_id] - emb_dim
    emb = decode_table_values(
        values[:, :emb_dim], checkpoint_dtype(state.buffers.dtype)
    )
    if optim_state_dim <= 0:
        return emb, None
    opt = values[:, -optim_state_dim:]
    return (
        emb,
        state.optimizer.states_for_checkpoint(opt, emb_dim).contiguous(),
    )
