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

"""A storage keeps a value row per key, apart from the key's hash slot: rows are
handed out densely per table, and the key map of a table that never evicts
(NO_EVICTION) is twice the size of its value buffer. Addressing the value buffer
with a slot therefore reads the wrong row -- or past the end of the buffer
entirely.

These tests drive the forward path over keys that are *already resident*, which is
what makes a lookup (rather than an insert) supply the index.
"""

import tempfile

import pytest
import torch
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.key_value_table import collect_table_sizes_to_device, get_expand_info

# The DEBUG initializer writes `key % DEBUG_EMB_INITIALIZER_MOD` into every
# element, so a row can be checked against the key that should own it.
DEBUG_MOD = 100_000
DIM = 8
TABLE = "t_0"


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


def build_model(
    current_device: int, max_capacity: int, init_capacity=None, **options_kwargs
):
    options = DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=current_device,
        dim=DIM,
        max_capacity=max_capacity,
        init_capacity=init_capacity,
        **options_kwargs,
        bucket_capacity=128,
        safe_check_mode=DynamicEmbCheckMode.IGNORE,
        local_hbm_for_values=1024**3,
        score_strategy=DynamicEmbScoreStrategy.NO_EVICTION,
        caching=False,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.DEBUG
        ),
    )
    return BatchedDynamicEmbeddingTablesV2(
        table_options=[options],
        output_dtype=torch.float32,
        table_names=[TABLE],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.SUM,
        use_index_dedup=False,
    )


def build_caching_model(
    current_device: int,
    max_capacity: int,
    cache_fraction: float,
    optimizer: EmbOptimType = EmbOptimType.SGD,
    **opt_params,
):
    """A CACHING-layout model: small GPU cache in front of a host-resident
    NO_EVICTION backing store.

    The cache size follows ``local_hbm_for_values / total_value_bytes``
    (``batched_dynamicemb_tables.py`` picks ``cap_scale`` from that ratio), so a
    small *cache_fraction* forces real spill traffic between the two tiers.
    """
    value_bytes = max_capacity * torch.finfo(torch.float32).bits // 8 * DIM
    options = DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=current_device,
        dim=DIM,
        max_capacity=max_capacity,
        bucket_capacity=128,
        safe_check_mode=DynamicEmbCheckMode.IGNORE,
        local_hbm_for_values=int(value_bytes * cache_fraction),
        score_strategy=DynamicEmbScoreStrategy.NO_EVICTION,
        caching=True,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.DEBUG
        ),
    )
    return BatchedDynamicEmbeddingTablesV2(
        table_options=[options],
        output_dtype=torch.float32,
        table_names=[TABLE],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.SUM,
        use_index_dedup=False,
        optimizer=optimizer,
        **opt_params,
    )


def next_row_counter(model, table_id: int = 0) -> int:
    """How many value rows the backing store has handed out."""
    return model._storage._state.key_index_map.num_rows(table_id)


def lookup(model, keys, device):
    """One SUM-pooled feature, one key per bag -> out[i] is keys[i]'s embedding."""
    indices = torch.tensor(keys, dtype=torch.int64, device=device)
    offsets = torch.arange(0, len(keys) + 1, dtype=torch.int64, device=device)
    out = model(indices, offsets)
    torch.cuda.synchronize()
    return out


def assert_debug_values(out: torch.Tensor, keys):
    """Every row must carry its own key's DEBUG pattern, not another row's."""
    expected = torch.tensor(
        [k % DEBUG_MOD for k in keys], dtype=out.dtype, device=out.device
    )
    torch.testing.assert_close(out, expected.unsqueeze(1).expand(-1, out.size(1)))


def test_resident_keys_reread_correctly(current_device):
    """A second forward over the same keys must return the same embeddings.

    Regression: the HBM-direct prefetch returned the *hash slot* for keys that hit
    in the table (``h_num_missing == 0`` early return), and the forward fed that
    straight into ``load_from_flat``. For NO_EVICTION the key map is twice the
    value buffer, so roughly half those slots addressed rows past the end of the
    buffer -- an illegal memory access.
    """
    device = torch.device(f"cuda:{current_device}")
    model = build_model(current_device, max_capacity=65536 * 2)
    keys = list(range(1001, 1301))

    first = lookup(model, keys, device)  # all miss -> rows come from the insert
    assert_debug_values(first, keys)

    second = lookup(model, keys, device)  # all hit -> rows must come from the score
    assert_debug_values(second, keys)
    torch.testing.assert_close(first, second)


def test_mixed_hit_and_miss_batch(current_device):
    """A batch that both hits and misses: found keys take the lookup index, new
    keys the freshly assigned row, and the two must not be crossed."""
    device = torch.device(f"cuda:{current_device}")
    model = build_model(current_device, max_capacity=65536 * 2)

    resident = list(range(2001, 2201))
    lookup(model, resident, device)

    fresh = list(range(9001, 9201))
    mixed = resident + fresh
    out = lookup(model, mixed, device)
    assert_debug_values(out, mixed)

    # Everything is resident now; one more pass must be stable.
    assert_debug_values(lookup(model, mixed, device), mixed)


def test_survives_repeated_passes(current_device):
    """Rows stay stable across many passes (the ref counter is indexed by slot,
    the values by row -- mixing them up corrupts one or the other over time)."""
    device = torch.device(f"cuda:{current_device}")
    model = build_model(current_device, max_capacity=65536 * 2)
    keys = list(range(3001, 3201))
    for _ in range(5):
        assert_debug_values(lookup(model, keys, device), keys)


@pytest.mark.parametrize(
    "init_capacity, n_keys, expect_growth",
    [
        # init == max_capacity: the value buffer is allocated to exactly the
        # logical row count, so an out-of-range slot runs off the end of the
        # allocation -- the loud, crashing form of the bug.
        (None, 800, False),
        # A small init_capacity leaves the buffer's *reserved* region far larger
        # than its logical one, so an out-of-range slot lands in reserved-but-
        # unused memory: no fault, just the wrong row. This is the silent form,
        # and the reason the bug survived -- the repo's own NO_EVICTION test uses
        # a small init_capacity and so only ever hit this variant.
        (4096, 800, False),
        # Same small buffer, but enough keys to cross max_load_factor (0.5 of a
        # 2*4096 key map) and actually rehash. Expansion grows the key map and the
        # value buffer by different amounts, so the two spaces stay mismatched
        # across the resize.
        (4096, 6000, True),
    ],
)
def test_explicit_init_capacity(current_device, init_capacity, n_keys, expect_growth):
    """Resident-key re-reads across the buffer-sizing regimes that decide whether
    a slot/row mix-up faults or corrupts silently."""
    device = torch.device(f"cuda:{current_device}")
    model = build_model(
        current_device, max_capacity=65536 * 2, init_capacity=init_capacity
    )
    cap_before = model._storage._state.key_index_map.capacity(0)

    keys = list(range(4001, 4001 + n_keys))
    assert_debug_values(lookup(model, keys, device), keys)  # insert
    assert_debug_values(lookup(model, keys, device), keys)  # all resident

    cap_after = model._storage._state.key_index_map.capacity(0)
    if expect_growth:
        assert cap_after > cap_before, (
            f"expected a rehash with {n_keys} keys in a {cap_before}-slot map, "
            "but the table never grew -- this case no longer covers expansion"
        )
        # One more pass after the rehash: slots moved, rows did not.
        assert_debug_values(lookup(model, keys, device), keys)
    else:
        assert cap_after == cap_before


# ---------------------------------------------------------------------------
# CACHING layout: GPU cache (forcibly TIMESTAMP/LRU) + host NO_EVICTION backing
# ---------------------------------------------------------------------------


def test_caching_downgrades_cache_but_not_backing(current_device):
    """The cache tier cannot be NO_EVICTION (it enables an overflow buffer, which
    ``create_table_state`` rejects for NO_EVICTION), so it is forced to
    TIMESTAMP/LRU while the backing store keeps NO_EVICTION.

    Pinning this down matters for users: with ``caching=True`` the *cache* still
    evicts -- only the backing store never drops a key.
    """
    model = build_caching_model(
        current_device, max_capacity=16384, cache_fraction=0.125
    )
    assert model._cache is not None, "expected a CACHING layout"
    assert (
        model._cache._state.options_list[0].score_strategy
        == DynamicEmbScoreStrategy.TIMESTAMP
    )
    assert (
        model._storage._state.options_list[0].score_strategy
        == DynamicEmbScoreStrategy.NO_EVICTION
    )
    assert model._storage._state.no_eviction


def test_caching_spill_does_not_leak_rows(current_device):
    """Spilling a key from cache to backing must reuse its existing logical row.

    NO_EVICTION rows come from a per-table auto-increment counter. If the spill
    path re-inserted an already-resident key without ``preserve_existing``, that
    key would be handed a *fresh* row on every eviction: the old row is orphaned
    and the counter climbs with spill traffic rather than with distinct keys --
    eventually past the value buffer, which is sized from the key count.

    Backward is required, not decorative: a prefetched key is pinned by the
    cache's ref counter until ``decrement_counter`` runs in the backward pass, so
    a forward-only loop cannot evict anything and nothing would ever spill.
    ``learning_rate=0`` keeps the DEBUG pattern intact for the value checks.
    """
    device = torch.device(f"cuda:{current_device}")
    model = build_caching_model(
        current_device,
        max_capacity=16384,
        cache_fraction=0.125,
        learning_rate=0.0,
    )
    keys = list(range(1001, 5001))  # 4000 distinct keys, ~2x the cache

    for _ in range(4):  # repeated sweeps -> repeated evict/refill of the same keys
        for start in range(0, len(keys), 500):
            chunk = keys[start : start + 500]
            out = lookup(model, chunk, device)
            out.sum().backward()  # releases the cache pins so eviction can happen
            torch.cuda.synchronize()
            assert_debug_values(out, chunk)

    rows = next_row_counter(model)
    assert rows > 0, "nothing ever spilled to the backing store -- test is vacuous"
    # One row per distinct key, no matter how much spill traffic happened. A leak
    # would push this ABOVE the distinct-key count.
    assert rows == len(keys), (
        f"expected {len(keys)} rows for {len(keys)} distinct keys, got {rows} -- "
        "rows are leaking on spill"
    )


def test_caching_backing_retains_every_key(current_device):
    """NO_EVICTION's core promise: the backing store never drops a key, however
    hard the cache in front of it evicts. It is the full set, so it holds the
    keys the cache holds as well.

    This has to be a *structural* check. The DEBUG initializer returns
    ``key % DEBUG_MOD``, so a key that was dropped and re-initialised reads back
    exactly like one that was retained -- value assertions cannot detect a drop.
    Counting live entries in the two tiers can.
    """
    device = torch.device(f"cuda:{current_device}")
    model = build_caching_model(
        current_device,
        max_capacity=16384,
        cache_fraction=0.125,
        learning_rate=0.0,
    )
    keys = list(range(2001, 6001))

    for start in range(0, len(keys), 500):
        chunk = keys[start : start + 500]
        out = lookup(model, chunk, device)
        out.sum().backward()  # unpins cache entries so eviction can proceed
        torch.cuda.synchronize()

    cached = int(model._cache._state.key_index_map.size(0))
    spilled = int(model._storage._state.key_index_map.size(0))
    # The cache is a fraction of the key set, so it must have evicted...
    assert cached < len(keys), (
        f"cache holds all {len(keys)} keys -- it never evicted, so this test "
        "says nothing about the backing store"
    )
    # ...and the backing store is the full key set: a new key goes into it
    # before it reaches the cache, so cache-resident keys are in it too and
    # nothing the cache evicted was dropped.
    assert spilled == len(keys), (
        f"{spilled} keys in the backing store != {len(keys)} -- some key was "
        "dropped between the tiers"
    )

    # Push the cache down so the backing store must hold the complete key set.
    model.flush()
    backing = int(model._storage._state.key_index_map.size(0))
    assert backing == len(keys), (
        f"backing store holds {backing} of {len(keys)} keys after flush -- "
        "NO_EVICTION must never drop one"
    )


def test_hbm_direct_backward_writes_the_right_row(current_device):
    """The backward half of the slot/row split, on the layout where they differ.

    ``fused_update_for_flat_table`` writes gradients into the **value buffer**
    (rows) while ``decrement_counter`` touches the **ref counter** (slots). Every
    other test here is forward-only, so this is the only coverage of the backward
    index space -- and NO_EVICTION on HBM-direct is the one configuration where
    feeding it a slot instead of a row would corrupt a different key's row.

    ``loss = out.sum()`` gives every element a gradient of exactly 1, so plain SGD
    moves each embedding from ``key % DEBUG_MOD`` to ``key % DEBUG_MOD - lr``. A
    write that landed on the wrong row shows up as one key un-updated and another
    double-updated.
    """
    device = torch.device(f"cuda:{current_device}")
    lr = 0.5
    options = DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=current_device,
        dim=DIM,
        max_capacity=65536 * 2,
        bucket_capacity=128,
        safe_check_mode=DynamicEmbCheckMode.IGNORE,
        local_hbm_for_values=1024**3,
        score_strategy=DynamicEmbScoreStrategy.NO_EVICTION,
        caching=False,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.DEBUG
        ),
    )
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=[options],
        output_dtype=torch.float32,
        table_names=[TABLE],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.SUM,
        use_index_dedup=False,
        optimizer=EmbOptimType.SGD,
        learning_rate=lr,
        stochastic_rounding=False,
    )
    assert model._cache is None, "expected the HBM-direct layout"
    keys = list(range(5001, 5301))

    out = lookup(model, keys, device)  # insert; rows come from the counter
    assert_debug_values(out, keys)
    out.sum().backward()
    torch.cuda.synchronize()

    # Re-read: this lookup takes the resident path, where the row must come from
    # the stored score rather than the hash slot.
    after = lookup(model, keys, device)
    expected = torch.tensor(
        [k % DEBUG_MOD - lr for k in keys], dtype=after.dtype, device=after.device
    )
    torch.testing.assert_close(after, expected.unsqueeze(1).expand(-1, after.size(1)))

    # A second step must move every row again by exactly lr -- no row left behind,
    # none hit twice.
    after.sum().backward()
    torch.cuda.synchronize()
    final = lookup(model, keys, device)
    expected2 = torch.tensor(
        [k % DEBUG_MOD - 2 * lr for k in keys], dtype=final.dtype, device=final.device
    )
    torch.testing.assert_close(final, expected2.unsqueeze(1).expand(-1, final.size(1)))


# ---------------------------------------------------------------------------
# Score strategy: NO_EVICTION is a mode, not a score
# ---------------------------------------------------------------------------


def test_no_eviction_alone_keeps_no_score(current_device):
    """``NO_EVICTION`` by itself stores no score: the key map's one score word is a
    constant that nothing writes, and the rows come from the row column."""
    from dynamicemb_extensions import ScorePolicy

    model = build_model(current_device, max_capacity=65536 * 2)
    state = model._storage._state
    assert state.no_eviction
    assert state.score_policy.policy == ScorePolicy.CONST
    assert state.key_index_map.use_rows_


@pytest.mark.parametrize(
    "strategy",
    [
        (DynamicEmbScoreStrategy.NO_EVICTION, DynamicEmbScoreStrategy.CUSTOMIZED),
        (DynamicEmbScoreStrategy.CUSTOMIZED, DynamicEmbScoreStrategy.NO_EVICTION),
    ],
)
def test_no_eviction_with_customized_keeps_the_customized_score(
    current_device, strategy
):
    """``(NO_EVICTION, CUSTOMIZED)``, in either order, never evicts and scores like
    CUSTOMIZED: the user sets the score before a forward and the keys it touches
    take it."""
    from dynamicemb.scored_hashtable import ScoreArg
    from dynamicemb_extensions import ScorePolicy

    device = torch.device(f"cuda:{current_device}")
    options = DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=current_device,
        dim=DIM,
        max_capacity=65536 * 2,
        bucket_capacity=128,
        safe_check_mode=DynamicEmbCheckMode.IGNORE,
        local_hbm_for_values=1024**3,
        score_strategy=strategy,
        caching=False,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.DEBUG
        ),
    )
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=[options],
        output_dtype=torch.float32,
        table_names=[TABLE],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.SUM,
        use_index_dedup=False,
    )
    state = model._storage._state
    assert state.no_eviction
    assert state.score_policy.policy == ScorePolicy.ASSIGN
    assert state.key_index_map.num_scores_ == 1

    keys = list(range(1001, 1301))
    key_tensor = torch.tensor(keys, dtype=torch.int64, device=device)
    tids = torch.zeros_like(key_tensor)
    read_only = ScoreArg(
        name=state.score_policy.name, value=None, policy=ScorePolicy.CONST
    )

    def stored_scores():
        scores, founds, _ = state.key_index_map.lookup(key_tensor, tids, read_only)
        assert bool(founds.all())
        return scores

    model.set_score({TABLE: 7})
    assert_debug_values(lookup(model, keys, device), keys)  # insert
    assert (stored_scores() == 7).all()
    model.set_score({TABLE: 9})
    assert_debug_values(lookup(model, keys, device), keys)  # all resident
    assert (stored_scores() == 9).all()
    assert next_row_counter(model) == len(keys)


def test_key_map_stays_twice_the_value_buffer_through_expansion(current_device):
    """The key map is exactly twice the value buffer when the table is made, and
    an expansion keeps it so: the value buffer grows with the key map, not by the
    key map's whole growth. The expansion happens with keys already in the table,
    so it rehashes them, and each must still find its own row afterwards."""
    device = torch.device(f"cuda:{current_device}")
    model = build_model(current_device, max_capacity=65536 * 2, init_capacity=4096)
    state = model._storage._state

    def ratio() -> float:
        return state.key_index_map.capacity(0) / state.tables[0].shape[0]

    assert ratio() == 2.0

    # Fits the 4096-row buffer, so nothing expands yet.
    first = list(range(4001, 4001 + 3000))
    assert_debug_values(lookup(model, first, device), first)
    assert state.key_index_map.capacity(0) == 8192
    assert next_row_counter(model) == len(first)

    # 6000 keys in all is past the buffer: this batch expands the table first,
    # rehashing the 3000 above into the larger map.
    second = list(range(20001, 20001 + 3000))
    assert_debug_values(lookup(model, second, device), second)
    assert state.key_index_map.capacity(0) > 8192, "the table never expanded"
    assert ratio() == 2.0
    assert next_row_counter(model) == len(first) + len(second)

    # The keys that were rehashed still own the rows their values sit in.
    assert_debug_values(lookup(model, first, device), first)
    assert_debug_values(lookup(model, second, device), second)


def test_unaligned_init_capacity_leaves_a_row_for_every_slot_to_fill(current_device):
    """An init_capacity that is not a whole number of buckets still gets a value
    buffer half the (bucket-rounded) key map, so the load factor at which the
    table expands is the load at which the buffer is full, not past it."""
    device = torch.device(f"cuda:{current_device}")
    model = build_model(current_device, max_capacity=65536 * 2, init_capacity=100)
    state = model._storage._state
    assert state.key_index_map.capacity(0) == 256
    assert state.tables[0].shape[0] == 128

    # Half the key map: fills the buffer. A single batch, since a later forward
    # counts every key in it as possibly new, resident or not, and would expand.
    keys = list(range(4001, 4001 + 128))
    assert_debug_values(lookup(model, keys, device), keys)
    assert state.key_index_map.capacity(0) == 256, "expanded before the buffer filled"
    assert next_row_counter(model) == len(keys)


def test_rows_handed_out_count_toward_expansion_not_just_live_keys(current_device):
    """An erased key's slot keeps its row, so rows can outnumber live keys, and the
    value buffer has to hold the rows. Expansion must follow them."""
    model = build_model(current_device, max_capacity=65536 * 2, init_capacity=4096)
    state = model._storage._state
    km = state.key_index_map
    assert state.tables[0].shape[0] == 4096

    km.reserve_rows(0, 3000)  # three thousand rows out, no live key at all
    sizes = collect_table_sizes_to_device(state).cpu()
    assert sizes.tolist() == [3000]

    flags, targets = get_expand_info(state, sizes, torch.tensor([500]))
    assert flags == [False]  # 3500 rows fit the 4096 the buffer has
    flags, targets = get_expand_info(state, sizes, torch.tensor([1500]))
    assert flags == [True]  # 4500 do not, whatever the 0 live keys say
    assert targets[0] >= 2 * (3000 + 1500)


def test_erase_and_reinsert_never_outgrows_the_value_buffer(current_device):
    """Keys that churn through a table near half full, each round erasing the
    oldest and inserting new ones, take slots no erased key left behind, so rows
    keep being handed out while the live count stays put. They must never pass the
    value buffer."""
    device = torch.device(f"cuda:{current_device}")
    model = build_model(current_device, max_capacity=65536 * 2, init_capacity=4096)
    state = model._storage._state
    km = state.key_index_map

    window, step = 3900, 1900
    live = list(range(10_001, 10_001 + window))
    assert_debug_values(lookup(model, live, device), live)
    next_key = 10_001 + window
    for _ in range(10):
        gone, live = live[:step], live[step:]
        model._storage.erase_keys(0, torch.tensor(gone, dtype=torch.int64, device=device))
        new = list(range(next_key, next_key + step))
        next_key += step
        live += new
        assert_debug_values(lookup(model, new, device), new)
        assert km.num_rows(0) <= state.tables[0].shape[0]
    assert_debug_values(lookup(model, live, device), live)


@pytest.mark.parametrize(
    "num_keys, expect_expansion",
    [
        # Fits the 1024 rows the destination starts with.
        (1024, False),
        # More than the rows, but no more than twice them: sizing the table by only
        # the part that overflows the buffer would see 476 keys in a 2048-slot map,
        # well under half full, and never expand.
        (1500, True),
        (2000, True),
    ],
)
def test_load_expands_for_every_key_in_the_checkpoint(
    current_device, num_keys, expect_expansion
):
    """Loading a checkpoint is sized by the keys in it, not by how far they overrun
    the value buffer: each of them takes a row."""
    device = torch.device(f"cuda:{current_device}")
    keys = list(range(20001, 20001 + num_keys))

    source = build_model(current_device, max_capacity=65536 * 2, init_capacity=4096)
    assert_debug_values(lookup(source, keys, device), keys)

    with tempfile.TemporaryDirectory() as save_dir:
        source.dump(save_dir)
        torch.cuda.synchronize()

        target = build_model(current_device, max_capacity=65536 * 2, init_capacity=1024)
        state = target._storage._state
        assert (state.key_index_map.capacity(0), state.tables[0].shape[0]) == (2048, 1024)
        target.load(save_dir)
        torch.cuda.synchronize()

    assert (state.key_index_map.capacity(0) > 2048) == expect_expansion
    assert state.key_index_map.capacity(0) == 2 * state.tables[0].shape[0]
    assert next_row_counter(target) == num_keys
    assert next_row_counter(target) <= state.tables[0].shape[0]
    assert_debug_values(lookup(target, keys, device), keys)


def test_load_reserves_room_only_for_the_keys_this_rank_keeps(current_device):
    """A checkpoint that holds every rank's keys is read whole by each rank, which
    keeps its own share. The table has to be sized for that share: a table that
    never evicts has no max_capacity to stop it growing for the whole file."""
    from dynamicemb.key_ownership import owned_key_mask

    device = torch.device(f"cuda:{current_device}")
    num_shards, shard_index = 8, 3
    keys = list(range(30001, 30001 + 4000))

    # Round-robin ownership, named rather than left to the default: consecutive
    # keys then fall on consecutive shards, so each of the 8 owns exactly 500 of
    # the 4000. A checkpoint is only loaded into a table of the same dist_type.
    source = build_model(
        current_device,
        max_capacity=65536 * 2,
        init_capacity=4096,
        dist_type="roundrobin",
    )
    assert_debug_values(lookup(source, keys, device), keys)

    with tempfile.TemporaryDirectory() as save_dir:
        source.dump(save_dir)
        torch.cuda.synchronize()

        target = build_model(
            current_device,
            max_capacity=65536 * 2,
            init_capacity=1024,
            dist_type="roundrobin",
        )
        state = target._storage._state
        assert (state.key_index_map.capacity(0), state.tables[0].shape[0]) == (2048, 1024)
        # This module is one rank of a larger group as far as the load is concerned.
        original = (target._shard_index, target._num_shards)
        target._shard_index, target._num_shards = shard_index, num_shards
        try:
            target.load(save_dir)
        finally:
            target._shard_index, target._num_shards = original
        torch.cuda.synchronize()

    key_tensor = torch.tensor(keys, dtype=torch.int64, device=device)
    mask = owned_key_mask(key_tensor, shard_index, num_shards, "roundrobin")
    mine = key_tensor[mask].tolist()
    # 500 fit the 1024 rows the table has, so it has no reason to grow.
    assert len(mine) == 500

    assert next_row_counter(target) == len(mine)
    assert state.key_index_map.capacity(0) == 2048, "grew for keys it does not keep"
    assert state.tables[0].shape[0] == 1024
    assert_debug_values(lookup(target, mine, device), mine)
