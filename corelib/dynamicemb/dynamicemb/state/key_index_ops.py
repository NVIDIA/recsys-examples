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

"""Operations on the key index map of a state: find keys, insert keys, insert and evict."""

from typing import Optional, Tuple

import torch
from dynamicemb.dynamicemb_config import EvictedItemMode
from dynamicemb.score.args import get_find_score_arg, get_insert_score_arg
from dynamicemb.state.evicted_items import _append_evicted
from dynamicemb.state.table_state import DynamicEmbTableState
from dynamicemb.state.value_rows import store_to_flat
from dynamicemb_extensions import flagged_compact


def _find_keys(
    state: DynamicEmbTableState,
    unique_keys: torch.Tensor,
    table_ids: torch.Tensor,
    score_input: Optional[torch.Tensor] = None,
    *,
    const_lookup: bool = False,
    return_slots: bool = False,
) -> Tuple:
    """Key-only find: lookup in hash table, return missing info + indices.

    The indices are value rows for a table that stores them, hash slots otherwise.
    With ``return_slots`` the hash slots follow as a ninth element: the reference
    counters are keyed by slot.

    ``score_input`` is the one tensor a table's score is made from, as
    :func:`score_source` chose it (see :func:`get_find_score_arg`).
    """
    if unique_keys.dtype != state.key_index_map.key_type:
        unique_keys = unique_keys.to(state.key_index_map.key_type)

    batch = unique_keys.size(0)
    device = unique_keys.device

    score_arg = get_find_score_arg(
        state,
        batch,
        device,
        score_input,
        const_lookup=const_lookup,
    )

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
            *((torch.empty(batch, dtype=torch.int64, device=device),) * return_slots),
        )

    km = state.key_index_map
    slots = None
    if getattr(km, "enable_overflow_", False):
        score_out, founds, indices = km.lookup_with_overflow(
            unique_keys, table_ids, score_arg
        )
        slots = indices  # a cache keeps slot == row
    elif return_slots:
        score_out, founds, indices, slots = km.lookup(
            unique_keys, table_ids, score_arg, return_slots=True
        )
    else:
        score_out, founds, indices = km.lookup(unique_keys, table_ids, score_arg)

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
        *((slots,) * return_slots),
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
        state,
        unique_keys.numel(),
        unique_keys.device,
        scores,
        preserve_existing,
    )
    if EvictedItemMode.RETAIN_KEY in state.evicted_item_mode:
        (
            indices,
            num_evicted,
            evicted_keys,
            evicted_table_ids,
        ) = state.key_index_map.insert(
            unique_keys,
            table_ids,
            score_arg,
            collect_evicted=True,
        )
        _append_evicted(state, evicted_keys, evicted_table_ids, num_evicted)
    else:
        indices = state.key_index_map.insert(unique_keys, table_ids, score_arg)
    store_to_flat(state, indices, table_ids, unique_values)


def _insert_and_evict_keys(
    state: DynamicEmbTableState,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    preserve_existing: bool = False,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Key-only insert_and_evict. Returns (indices, num_evicted, evicted_keys,
    evicted_table_ids, evicted_indices, evicted_scores).
    Caller is responsible for loading evicted values and storing new values.

    *preserve_existing* is forwarded to :func:`get_insert_score_arg` (e.g. backward
    re-insert should use True so existing slot scores are not overwritten).
    """
    score_arg = get_insert_score_arg(
        state,
        keys.numel(),
        keys.device,
        scores,
        preserve_existing,
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
