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

"""The evicted and erased keys a state keeps for ``pop_evicted_keys`` and ``pop_erased_keys``."""

from typing import List, Optional, Tuple

import torch
from dynamicemb.state.table_state import DynamicEmbTableState


def _append_evicted(
    state: DynamicEmbTableState,
    evicted_keys: torch.Tensor,
    evicted_table_ids: torch.Tensor,
    num_evicted: torch.Tensor,
) -> None:
    """Append this insert's evicted (key, table_id) to the state's retain chunks.

    ``evicted_keys`` / ``evicted_table_ids`` are sized at the batch upper bound;
    only the first ``num_evicted`` entries are valid, so we slice + clone (freeing
    the large per-batch buffers and keeping only the compact evicted rows). No
    de-duplication here -- that happens on pop. One D2H sync per evicting insert
    to read the count, acceptable since retain is opt-in.
    """
    n = int(num_evicted.item())
    if n <= 0:
        return
    state.evicted_key_chunks.append(evicted_keys[:n].clone())
    state.evicted_tid_chunks.append(evicted_table_ids[:n].clone())


def _append_erased(
    state: DynamicEmbTableState,
    keys: torch.Tensor,
    table_id: int,
    slot_index: Optional[torch.Tensor] = None,
) -> None:
    """Append an explicit erase's keys to the state's erased-key buffer.

    Unlike :func:`_append_evicted` there is no count to read back: the caller has
    already masked the keys down to the ones the erase actually removed, so this
    costs no extra device sync. ``slot_index`` is the ``[M, 2]`` (slot, row) each
    erased key left behind, for a table that stores rows.
    """
    if keys.numel() == 0:
        return
    state.erased_key_chunks.append(keys.clone())
    state.erased_tid_chunks.append(
        torch.full((keys.numel(),), table_id, dtype=torch.int64, device=keys.device)
    )
    if slot_index is not None:
        state.erased_slot_chunks.append(slot_index.clone())
        state.erased_slot_tid_chunks.append(
            torch.full(
                (slot_index.size(0),), table_id, dtype=torch.int64, device=keys.device
            )
        )


def _pop_chunked_keys(
    key_chunks: List[torch.Tensor],
    tid_chunks: List[torch.Tensor],
    table_id: int,
    state: DynamicEmbTableState,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Drain one table's rows out of a (key, table_id) chunk buffer.

    Concatenates the chunks, selects this table's rows, de-duplicates, and
    returns the rebuilt chunk lists holding only the OTHER tables' rows -- so
    this table's keys are cleared while the rest stay for their own pop.
    """
    if not key_chunks:
        # Match the non-empty path's dtype: chunks carry key_index_map.key_type
        # (== the table's index_type, which may be int32/uint32), so a hardcoded
        # int64 here would make callers that concat/compare across empty and
        # non-empty pops hit a dtype mismatch.
        empty = torch.empty(
            0,
            dtype=state.key_index_map.key_type if dtype is None else dtype,
            device=state.device,
        )
        return empty, [], []
    keys = torch.cat(key_chunks)
    tids = torch.cat(tid_chunks)
    mask = tids == table_id
    out = torch.unique(keys[mask], dim=0 if keys.dim() > 1 else None)
    keep = ~mask
    if bool(keep.any()):
        return out, [keys[keep]], [tids[keep]]
    return out, [], []


def _pop_state_evicted_keys(state: DynamicEmbTableState, table_id: int) -> torch.Tensor:
    """Pop (return + clear) the unique keys ``table_id`` had evicted."""
    out, state.evicted_key_chunks, state.evicted_tid_chunks = _pop_chunked_keys(
        state.evicted_key_chunks, state.evicted_tid_chunks, table_id, state
    )
    return out


def _pop_state_erased_keys(state: DynamicEmbTableState, table_id: int) -> torch.Tensor:
    """Pop (return + clear) the unique keys an explicit erase removed."""
    out, state.erased_key_chunks, state.erased_tid_chunks = _pop_chunked_keys(
        state.erased_key_chunks, state.erased_tid_chunks, table_id, state
    )
    return out


def _pop_state_erased_slot_index(
    state: DynamicEmbTableState, table_id: int
) -> torch.Tensor:
    """Pop (return + clear) the (slot, row) pair each explicit erase left behind,
    one per slot: a slot erased, refilled and erased again is one reclaimed slot
    with one row, however many keys passed through it."""
    out, state.erased_slot_chunks, state.erased_slot_tid_chunks = _pop_chunked_keys(
        state.erased_slot_chunks,
        state.erased_slot_tid_chunks,
        table_id,
        state,
        dtype=torch.int64,
    )
    return out.reshape(-1, 2)
