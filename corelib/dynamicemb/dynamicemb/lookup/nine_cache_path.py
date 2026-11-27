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

"""The NINE (non-inclusive, non-exclusive) cache prefetch path: HBM cache + an external PS."""

from typing import Optional, Tuple

import torch
from dynamicemb.batched_dynamicemb_function import _bool_item, _initialize_rows
from dynamicemb.initializer import MultiTableInitializer
from dynamicemb.key_value_table import (
    DynamicEmbCache,
    Storage,
    _select,
    score_source,
)
from dynamicemb.types import CopyMode, MultiTableAdmitter
from dynamicemb_extensions import EvictStrategy, PaddedBufferView, flagged_compact


def prefetch_nine_cache_path(
    cache: DynamicEmbCache,
    storage: Storage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    padded_view: PaddedBufferView,
    emb_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Cache prefetch with counter protection and overflow buffer (CachePolicy.NINE).

    ``storage`` is the external store (e.g. a PS), the only tier below the
    cache and opaque to it. The cache owns the scores -- one word per key, the table's
    ``score_strategy`` (a compound one is refused at construction) -- so a key
    enters it with a score, and leaves it for the store with one.

    A key is in the cache, in the store, or in neither (new); a key can also be in
    both, since the store is not asked to drop what the cache took. Only admitted
    new keys are inserted. Non-admitted keys get slot_indices = -1 and their
    positions (in the unique_keys array) are returned so the forward pass can
    lazily initialize their embeddings.

    Returns (slot_indices, non_admitted_positions).
    """
    with torch.cuda.nvtx.range("prefetch_nine_cache_path"):
        device = unique_keys.device
        state = cache._state

        if unique_keys.numel() == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, None

        # ``score_input`` is chosen once: the cache's score, and the store's, are
        # made from it. The frequency is only for admission below.
        score_input = score_source(
            state.evict_strategy, accumulated_frequency, key_scores
        )

        # 1. Lookup with overflow fallback; pin what is found (the counter kernel
        # skips the misses' slots, -1).
        with torch.cuda.nvtx.range("op:cache_lookup"):
            _, founds, cache_indices = cache.lookup(
                unique_keys, unique_table_ids, score_input
            )
        slot_indices = cache_indices.clone()
        cache.increment_counter(cache_indices, unique_table_ids)

        # 2. The misses. ``score_input`` is what the cache's score is made from;
        # the frequency is kept for admission, which counts by it.
        with torch.cuda.nvtx.range("op:flagged_compact"):
            (
                num_miss,
                miss_pos,
                (miss_keys, miss_tids, miss_freq, miss_score_input),
            ) = flagged_compact(
                ~founds,
                [unique_keys, unique_table_ids, accumulated_frequency, score_input],
            )
        if num_miss == 0:
            return slot_indices, None

        # 3. The store has the rest, or it is new
        with torch.cuda.nvtx.range("op:storage_find"):
            (_, _, _, _, _, in_store, store_scores, values) = storage.find(
                miss_keys,
                miss_tids,
                CopyMode.VALUE,
                miss_score_input,  # positionally: a store names the parameter itself
            )

        # 4. Admit the new ones (a key the store has was admitted before)
        is_new = ~in_store
        admitted = is_new.clone()
        non_admitted_positions: Optional[torch.Tensor] = None
        if admitter is not None:
            with torch.cuda.nvtx.range("op:flagged_compact"):
                (
                    num_new,
                    new_rows,
                    (new_keys, new_tids, new_freq, new_pos),
                ) = flagged_compact(is_new, [miss_keys, miss_tids, miss_freq, miss_pos])
            if num_new > 0:
                admit_mask = admitter.admit(new_keys, new_tids, new_freq)
                if (
                    miss_score_input is not None
                    and new_freq is not None
                    and state.evict_strategy == EvictStrategy.KLfu
                ):
                    # An LFU score is a frequency: the admitter hands back each
                    # key's running total in the frequencies it was given, so a
                    # key that waited several batches enters with its full count.
                    miss_score_input[new_rows] = new_freq
                admitted[new_rows] = admit_mask
                with torch.cuda.nvtx.range("op:flagged_compact"):
                    num_refused, _, (refused_pos,) = flagged_compact(
                        ~admit_mask, [new_pos]
                    )
                if num_refused > 0:
                    non_admitted_positions = refused_pos
                    slot_indices[refused_pos] = -1  # a refused key has no slot

        to_cache = in_store | admitted
        with torch.cuda.nvtx.range("op:flagged_compact"):
            (
                num_cache,
                into_cache,
                (cache_keys, cache_tids, cache_pos),
            ) = flagged_compact(to_cache, [miss_keys, miss_tids, miss_pos])
        if num_cache == 0:
            return slot_indices, non_admitted_positions

        # 5. Initialize the admitted-new rows
        with torch.cuda.nvtx.range("op:flagged_compact"):
            num_adm, adm_rows, (adm_keys, adm_tids) = flagged_compact(
                admitted, [miss_keys, miss_tids]
            )
        if num_adm > 0:
            values[adm_rows] = _initialize_rows(
                state,
                initializer,
                adm_keys,
                adm_tids,
                padded_view,
                emb_dtype,
            ).values

        # 6. The score a key enters the cache with. An LFU key the store has comes
        # with the count the store kept for it (its find has added this batch's).
        insert_score_input = miss_score_input
        if state.evict_strategy == EvictStrategy.KLfu and _bool_item(in_store.any()):
            if insert_score_input is None:
                insert_score_input = torch.ones(
                    num_miss, dtype=store_scores.dtype, device=device
                )
            insert_score_input = torch.where(
                in_store, store_scores.to(insert_score_input.dtype), insert_score_input
            )
        insert_scores = (
            None
            if insert_score_input is None
            else _select(insert_score_input, into_cache, num_cache == num_miss)
        )

        # 7. Bring the keys into the cache. The insert also flags the keys it could
        # not place, in pinned host memory that the read of num_evicted waits for.
        with torch.cuda.nvtx.range("op:cache_insert_evict"):
            (
                cache_slots,
                num_evicted,
                evicted_keys,
                evicted_slots,
                evicted_scores,
                evicted_tids,
                num_refused,
            ) = cache.insert_and_evict(
                cache_keys, cache_tids, insert_scores, report_refused=True
            )
        if num_refused > 0:
            raise RuntimeError(
                f"The cache could not place {num_refused} keys of this batch: their "
                "buckets and their tables' overflow areas held only keys it may not "
                "evict (pinned by batches in flight, or placed by this batch). "
                "Increase the cache size, or reduce the prefetch pipeline depth or "
                "the batch size."
            )

        # 8. What it evicted: read before the slots are overwritten, then write
        # back to the store with the scores the cache kept for it
        with torch.cuda.nvtx.range("op:load_from_flat"):
            evicted_values = state.buffers.load_to_padded(
                evicted_slots, evicted_tids, copy_mode=CopyMode.VALUE
            )
        with torch.cuda.nvtx.range("op:store_to_flat"):
            state.buffers.store_from_padded(
                cache_slots,
                cache_tids,
                _select(values, into_cache, num_cache == num_miss),
            )
        if num_evicted > 0:
            with torch.cuda.nvtx.range("op:storage_insert"):
                storage.insert(
                    evicted_keys, evicted_tids, evicted_values, evicted_scores
                )

        # 9. Counter & slot mapping
        cache.increment_counter(cache_slots, cache_tids)
        slot_indices[cache_pos] = cache_slots
        return slot_indices, non_admitted_positions
