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

from typing import TYPE_CHECKING, Optional

import torch
from dynamicemb.types import CopyMode
from dynamicemb.utils import encode_table_values
from dynamicemb_extensions import load_from_flat_table_contiguous as _load_contiguous
from dynamicemb_extensions import load_from_flat_table_emb as _load_emb
from dynamicemb_extensions import load_from_flat_table_value as _load_value
from dynamicemb_extensions import store_to_flat_table_contiguous as _store_contiguous
from dynamicemb_extensions import store_to_flat_table_value as _store_value

if TYPE_CHECKING:
    from dynamicemb.key_value_table import DynamicEmbTableState


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
        max_dim = state.emb_dim
        _load = _load_emb
    else:
        max_dim = state.value_dim
        _load = _load_value
    output = torch.empty(N, max_dim, dtype=state.emb_dtype, device=state.device)
    if N > 0:
        _load(
            state.table_ptrs_dev,
            indices,
            table_ids,
            output,
            state.table_value_dims,
            state.table_emb_dims,
            state.emb_dim,
            state.all_dims_vec4,
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
        state.table_ptrs_dev,
        indices,
        table_ids,
        out,
        state.table_value_dims,
        state.table_emb_dims,
        state.emb_dim,
        state.all_dims_vec4,
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
        state.table_ptrs_dev,
        indices,
        table_ids,
        encode_table_values(values, state.emb_dtype),
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
    indices = indices.to(device=state.device, dtype=torch.int64)
    N = indices.numel()
    vdim = state.table_value_dims_cpu[table_id]
    output = torch.empty(N, vdim, dtype=state.emb_dtype, device=state.device)
    if N > 0:
        _load_contiguous(
            state.table_ptrs_dev,
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
    indices = indices.to(device=state.device, dtype=torch.int64)
    N = indices.numel()
    if N == 0:
        return
    if values.dim() == 1:
        values = values.unsqueeze(1)
    _store_contiguous(
        state.table_ptrs_dev,
        indices,
        table_id,
        encode_table_values(values, state.emb_dtype),
        state.table_value_dims,
        state.table_emb_dims,
        state.emb_dim,
        state.all_dims_vec4,
    )
