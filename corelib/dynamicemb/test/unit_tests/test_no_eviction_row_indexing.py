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

"""NO_EVICTION is the only score strategy whose value row differs from its hash
slot: rows come from a per-table auto-increment counter, and the key map is sized
``1 / max_load_factor`` times the value buffer. Addressing the value buffer with a
slot therefore reads the wrong row -- or past the end of the buffer entirely.

These tests drive the forward path over keys that are *already resident*, which is
what makes a lookup (rather than an insert) supply the index.
"""

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

# The DEBUG initializer writes `key % DEBUG_EMB_INITIALIZER_MOD` into every
# element, so a row can be checked against the key that should own it.
DEBUG_MOD = 100_000
DIM = 8
TABLE = "t_0"


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


def build_model(current_device: int, max_capacity: int, init_capacity=None):
    options = DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=current_device,
        dim=DIM,
        max_capacity=max_capacity,
        init_capacity=init_capacity,
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
    """The backing store's NO_EVICTION auto-increment row counter."""
    return int(model._storage._state.no_eviction_next_index_dev[table_id].item())


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
    assert model._storage._state.no_eviction_next_index is not None


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
    hard the cache in front of it evicts.

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
    # ...and everything it evicted landed in the backing store rather than being
    # dropped. The rest is still cache-resident and has not spilled yet.
    assert spilled + cached == len(keys), (
        f"{spilled} spilled + {cached} cached != {len(keys)} keys -- some key was "
        "dropped between the tiers"
    )

    # Push the cache down so the backing store must hold the complete key set.
    model.flush()
    backing = int(model._storage._state.key_index_map.size(0))
    assert backing == len(keys), (
        f"backing store holds {backing} of {len(keys)} keys after flush -- "
        "NO_EVICTION must never drop one"
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
@pytest.mark.parametrize("dim", [8, 6], ids=["vec4", "scalar"])
@pytest.mark.parametrize(
    "optimizer", [EmbOptimType.SGD, EmbOptimType.ADAM], ids=["sgd", "adam"]
)
def test_hbm_direct_backward_writes_the_right_row(
    current_device, dtype, dim, optimizer
):
    """Both optimizer paths must address values by row and counters by hash slot.

    Use small, exactly representable DEBUG values even though the full keys are
    large and distinct. At the old 5001..5300 initial values a BF16 update of 0.5
    rounds away, so an unchanged row could pass without exercising a write.

    With beta1 = beta2 = 0.5, eps = 0 and constant gradients, Adam's bias-corrected
    moments give an update of lr * sign(grad). The first three stored moments are
    exactly (1 - 2**-step) * grad and (1 - 2**-step) * grad**2 in every dtype.
    This hand-computable oracle checks the physical m/v rows as well as weights.
    """
    device = torch.device(f"cuda:{current_device}")
    lr = 0.5
    options = DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=dtype,
        device_id=current_device,
        dim=dim,
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
        optimizer=optimizer,
        learning_rate=lr,
        beta1=0.5,
        beta2=0.5,
        eps=0.0,
        weight_decay=0.0,
        stochastic_rounding=False,
    )
    assert model._cache is None, "expected the HBM-direct layout"
    assert model._storage.embedding_dtype() == dtype
    assert model._storage._state.tables[0].tensor().dtype == dtype
    keys = [DEBUG_MOD * (i + 1) + i % 16 + 1 for i in range(300)]
    # Keep every key's gradient nonzero: eps=0 would make Adam's zero-gradient
    # case 0/0. Different magnitudes distinguish SGD from Adam's normalization.
    coeffs = torch.tensor([1.0, -1.0, 2.0], device=device).repeat(100)
    initial = torch.tensor(
        [k % DEBUG_MOD for k in keys], dtype=torch.float32, device=device
    )
    indices = torch.tensor(keys, dtype=torch.int64, device=device)
    offsets = torch.arange(len(keys) + 1, dtype=torch.int64, device=device)
    update = coeffs if optimizer == EmbOptimType.SGD else coeffs.sign()

    out = model(indices, offsets)
    torch.testing.assert_close(out, initial[:, None].expand(-1, dim), rtol=0, atol=0)
    for step in range(1, 4):
        (out * coeffs[:, None]).sum().backward()
        torch.cuda.synchronize()

        # Export independently from the training forward/backward path; checking
        # only another lookup could let a paired wrong-row read/write hide a bug.
        exported_keys, exported_weights, exported_states = [], [], []
        for stored_keys, weights, states, _ in model._storage.export_keys_values(
            device
        ):
            exported_keys.extend(stored_keys.tolist())
            exported_weights.append(weights)
            if states is not None:
                exported_states.append(states)
        assert len(exported_keys) == len(keys)
        assert set(exported_keys) == set(keys)
        input_positions = {key: i for i, key in enumerate(keys)}
        positions = torch.tensor(
            [input_positions[key] for key in exported_keys], device=device
        )
        weights = torch.cat(exported_weights)
        expected = (initial - step * lr * update)[positions].to(dtype)
        torch.testing.assert_close(
            weights, expected[:, None].expand(-1, dim), rtol=0, atol=0
        )
        if optimizer == EmbOptimType.ADAM:
            assert model._optimizer.get_opt_args()["iters"] == step
            states = torch.cat(exported_states)
            scale = 1 - 2.0**-step
            expected_m = (scale * coeffs[positions]).to(dtype)
            expected_v = (scale * coeffs[positions] ** 2).to(dtype)
            torch.testing.assert_close(
                states[:, :dim], expected_m[:, None].expand(-1, dim), rtol=0, atol=0
            )
            torch.testing.assert_close(
                states[:, dim:], expected_v[:, None].expand(-1, dim), rtol=0, atol=0
            )

        # Reuse the resident-key lookup for the next training iteration; an
        # extra forward-only probe would pin another set of counter entries.
        out = model(indices, offsets)
        expected = initial - step * lr * update
        torch.testing.assert_close(
            out, expected[:, None].expand(-1, dim), rtol=0, atol=0
        )
