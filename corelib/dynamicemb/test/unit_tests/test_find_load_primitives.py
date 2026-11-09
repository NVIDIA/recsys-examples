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

"""The two primitives a path without host counts is made of.

``_find_keys(compact=False)`` is the lookup and nothing else: the misses are not
compacted, so no count is read back. ``load_from_flat(output=)`` loads into an existing
buffer and writes only the rows whose index is not negative, so a caller can put defaults
in the buffer first and let what the table has overwrite them.
"""

import pytest
import torch
from dynamicemb import DynamicEmbCheckMode, DynamicEmbScoreStrategy, EvictedItemMode
from dynamicemb.dynamicemb_config import DynamicEmbEvictStrategy, DynamicEmbTableOptions
from dynamicemb.key_value_table import (
    _find_keys,
    create_table_state,
    get_insert_score_arg,
    load_from_flat,
    store_to_flat_single_table,
)
from dynamicemb.optimizer import AdamDynamicEmbeddingOptimizer, OptimizerArgs
from dynamicemb.types import CopyMode

DIM = 16
POISON = 7.0
KEYS = [11, 22, 33]


@pytest.fixture
def device():
    assert torch.cuda.is_available()
    return torch.device("cuda", torch.cuda.current_device())


def _state(device, customized=False, retain=False):
    """A table of Adam state (``value_dim == 3 * DIM``) holding ``KEYS`` with known rows."""
    option = DynamicEmbTableOptions(
        dim=DIM,
        init_capacity=1024,
        max_capacity=1024,
        bucket_capacity=128,
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=device.index,
        safe_check_mode=DynamicEmbCheckMode.IGNORE,
        score_strategy=(
            DynamicEmbScoreStrategy.CUSTOMIZED
            if customized
            else DynamicEmbScoreStrategy.TIMESTAMP
        ),
        evict_strategy=(
            DynamicEmbEvictStrategy.CUSTOMIZED
            if customized
            else DynamicEmbEvictStrategy.LRU
        ),
        evicted_item_mode=(
            EvictedItemMode.RETAIN_KEY if retain else EvictedItemMode.DISCARD
        ),
        local_hbm_for_values=1024**3,
    )
    state = create_table_state(
        [option],
        AdamDynamicEmbeddingOptimizer(OptimizerArgs()),
        evicted_item_mode=option.evicted_item_mode,
    )
    keys = torch.tensor(KEYS, dtype=torch.int64, device=device)
    tids = torch.zeros_like(keys)
    scores = (
        torch.as_tensor([5] * len(KEYS), dtype=torch.uint64, device=device)
        if customized
        else None
    )
    score_arg = get_insert_score_arg(state, keys.numel(), device, scores)
    indices = state.key_index_map.insert(keys, tids, score_arg)
    assert state.value_dim == 3 * DIM
    store_to_flat_single_table(state, indices, 0, _stored_rows(device))
    torch.cuda.synchronize()
    return state


def _stored_rows(device):
    """The row of each key in ``KEYS``: ``100 * (i + 1) + column``."""
    rows = torch.arange(3 * DIM, dtype=torch.float32, device=device).repeat(3, 1)
    return rows + 100 * torch.arange(
        1, 4, dtype=torch.float32, device=device
    ).unsqueeze(1)


def _query(device):
    """Two keys the table has, one it has not, the reserved all-ones key, one more it
    has (the table id of the reserved one is -1, as in an inert entry)."""
    keys = torch.tensor([11, 99, 33, -1, 22], dtype=torch.int64, device=device)
    tids = torch.tensor([0, 0, 0, -1, 0], dtype=torch.int64, device=device)
    return keys, tids


def test_a_table_state_has_the_pieces_of_a_path_without_counts(device):
    plain = _state(device)
    assert plain.miss_gate is not None
    assert plain.retain_ring is None and plain.evicted_pending == []
    retaining = _state(device, retain=True)
    assert retaining.miss_gate is not None
    assert retaining.retain_ring is not None and retaining.evicted_pending == []
    assert plain.miss_gate is not retaining.miss_gate


@pytest.mark.parametrize("customized", [False, True])
def test_a_find_without_compaction_is_the_lookup_alone(device, customized):
    state = _state(device, customized=customized)
    state.training = customized  # a training CUSTOMIZED lookup carries a score tensor
    keys, tids = _query(device)
    key_scores = (
        torch.as_tensor([1, 2, 3, 4, 5], dtype=torch.uint64, device=device)
        if customized
        else None
    )

    full = _find_keys(state, keys, tids, key_scores, return_slots=True)
    lean = _find_keys(state, keys, tids, key_scores, return_slots=True, compact=False)

    assert len(lean) == len(full) == 9
    assert lean[:4] == (None, None, None, None)
    assert lean[5].tolist() == [True, False, True, False, True]
    assert torch.equal(lean[5], full[5])
    assert torch.equal(lean[6], full[6])
    assert torch.equal(lean[7], full[7])
    assert torch.equal(lean[8], full[8])
    assert bool((lean[7][~lean[5]] == -1).all())
    if customized:
        # the full-length score tensor of the lookup, kept for a later insert
        assert lean[4].numel() == keys.numel()
        assert lean[4].tolist() == key_scores.tolist()
    else:
        assert lean[4] is None  # an LRU table has no score tensor

    # without return_slots the tuple is one shorter, the lookup the same
    assert len(_find_keys(state, keys, tids, key_scores, compact=False)) == 8


def test_a_find_of_nothing_is_unchanged(device):
    state = _state(device)
    empty = torch.empty(0, dtype=torch.int64, device=device)
    for compact in (True, False):
        result = _find_keys(state, empty, empty, compact=compact)
        assert result[0] == 0 and result[5].numel() == 0 and result[7].numel() == 0


@pytest.mark.parametrize("mode", [CopyMode.VALUE, CopyMode.EMBEDDING])
def test_a_load_into_a_buffer_writes_only_the_rows_the_table_has(device, mode):
    state = _state(device)
    keys, tids = _query(device)
    rows = _find_keys(state, keys, tids, compact=False)[7]
    present = rows >= 0
    fresh = load_from_flat(state, rows, tids, mode)

    width = state.value_dim if mode == CopyMode.VALUE else state.emb_dim
    buffer = torch.full((keys.numel(), width), POISON, device=device)
    loaded = load_from_flat(state, rows, tids, mode, output=buffer)

    assert loaded is buffer
    assert torch.equal(buffer[present], fresh[present])
    assert bool(
        (buffer[~present] == POISON).all()
    ), "a row without an index is left alone"
    # the rows stored for the keys came back whole: embedding and optimizer state
    stored = _stored_rows(device)
    expected = {11: stored[0], 33: stored[2], 22: stored[1]}
    for i, key in enumerate(keys.tolist()):
        if key in expected:
            assert torch.equal(buffer[i], expected[key][:width])


def test_a_load_into_a_column_slice_leaves_the_other_columns(device):
    """The embedding goes into the first columns of a wider buffer, whose other
    columns stay as they are (the row stride of the buffer, not of the slice's width).
    """
    state = _state(device)
    keys, tids = _query(device)
    rows = _find_keys(state, keys, tids, compact=False)[7]
    wide = torch.full((keys.numel(), state.value_dim), POISON, device=device)

    load_from_flat(
        state, rows, tids, CopyMode.EMBEDDING, output=wide[:, : state.emb_dim]
    )

    present = rows >= 0
    assert bool((wide[:, state.emb_dim :] == POISON).all())
    assert bool((wide[~present] == POISON).all())
    assert torch.equal(
        wide[present][:, : state.emb_dim],
        load_from_flat(state, rows, tids, CopyMode.EMBEDDING)[present],
    )


def test_a_load_refuses_a_buffer_of_the_wrong_shape_or_type(device):
    state = _state(device)
    keys, tids = _query(device)
    rows = _find_keys(state, keys, tids, compact=False)[7]
    n = keys.numel()
    with pytest.raises(AssertionError):
        load_from_flat(
            state,
            rows,
            tids,
            CopyMode.VALUE,
            output=torch.empty(n, state.value_dim - 1, device=device),
        )
    with pytest.raises(AssertionError):
        load_from_flat(
            state,
            rows,
            tids,
            CopyMode.EMBEDDING,
            output=torch.empty(n + 1, state.emb_dim, device=device),
        )
    with pytest.raises(AssertionError):
        load_from_flat(
            state,
            rows,
            tids,
            CopyMode.EMBEDDING,
            output=torch.empty(n, state.emb_dim, dtype=torch.float16, device=device),
        )
