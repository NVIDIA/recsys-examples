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

"""The fan-out-invariant global bucket layout of ``hash_bucket`` tables.

A key's home bucket used to be ``hash % shard_capacity // bc``, so the keys
that shared a bucket depended on how many shards the table was split over, and
a checkpoint reloaded at another world size re-dealt them: buckets that were
full at 8 shards overflowed at 1, and ``insert`` evicted the lowest-score
resident -- about 1-2 % of the rows of a near-full table, every time the
topology changed. Under the new ``hash_bucket`` rule the bucket is
``(hash % global_capacity) // bc``, a number that knows nothing about shards,
and the shard is that bucket's position, ``g // ceil(G / S)``. These tests pin
down the three places the rule is written (table kernels, routing kernel, host
twin) and then do the thing it is for: fill 8 shards to 97 % of every bucket,
dump, load into 1, and lose nothing. All on one GPU, by building each shard's
module in turn (``_shard_index`` / ``_num_shards`` name the files).

Run (from corelib/dynamicemb)::

    pytest -svv test/unit_tests/test_global_bucket_layout.py
"""

import json
import math
import tempfile
import warnings
from typing import List, Tuple

import numpy as np
import pytest
import torch
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamicemb import batched_dynamicemb_tables as bdt_module
from dynamicemb import get_global_table_capacity, get_sharded_table_capacity
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.dynamicemb_config import (
    _global_bucket_layout_rows,
    _sharded_table_bucket_layout,
)
from dynamicemb.key_ownership import (
    INT64_MAX,
    OwnershipRule,
    hashed_shard_capacity,
    murmur3_fmix64,
    owning_shard,
)
from dynamicemb.key_value_table import HybridStorage
from dynamicemb.planner.plan import TableKey
from dynamicemb.planner.planners import (
    DynamicEmbParameterConstraints,
    _prepare_dynemb_table_options,
)
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb.shard.input_dist import (
    bucketize_kjt_before_all2all,
    dist_type_codes,
    routing_param_tensors,
)
from dynamicemb.types import MAX_BUCKET_CAPACITY
from dynamicemb_extensions import ScorePolicy, block_bucketize_sparse_features
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

DIM = 8
TABLE = "t_0"
# The DEBUG initializer writes key % DEBUG_MOD into every element of a row.
DEBUG_MOD = 100_000
BC = 128
NUM_EMBEDDINGS = 1_000_000
NUM_GLOBAL_BUCKETS, _ = _global_bucket_layout_rows(NUM_EMBEDDINGS, BC)
GLOBAL_CAPACITY = NUM_GLOBAL_BUCKETS * BC  # 7813 buckets, 1,000,064 rows
# 97 % of every bucket: the fill at which the old layout lost 1-2 % on a fold.
KEYS_PER_BUCKET = int(0.97 * BC)  # 124


@pytest.fixture
def current_device():
    return torch.cuda.current_device()


# ---------------------------------------------------------------------------
# Oracles, spelled out independently of the production code
# ---------------------------------------------------------------------------


def oracle_hash(keys: np.ndarray) -> np.ndarray:
    """``murmur3(key) & INT64_MAX`` -- what LinearBucket::hash computes."""
    return (murmur3_fmix64(keys) & np.uint64(INT64_MAX)).astype(np.int64)


def oracle_global_bucket(keys: np.ndarray) -> np.ndarray:
    return (oracle_hash(keys) % GLOBAL_CAPACITY) // BC


def buckets_per_shard(num_shards: int) -> int:
    return math.ceil(NUM_GLOBAL_BUCKETS / num_shards)


def oracle_shard(keys: np.ndarray, num_shards: int) -> np.ndarray:
    return oracle_global_bucket(keys) // buckets_per_shard(num_shards)


def keys_filling_every_bucket(per_bucket: int, seed: int = 0) -> np.ndarray:
    """``per_bucket`` distinct random keys for every one of the G global buckets."""
    rng = np.random.default_rng(seed)
    candidates = np.unique(
        rng.integers(
            1, 1 << 62, size=4 * NUM_GLOBAL_BUCKETS * per_bucket, dtype=np.int64
        )
    )
    g = oracle_global_bucket(candidates)
    order = np.argsort(g, kind="stable")
    candidates, g = candidates[order], g[order]
    starts = np.searchsorted(g, np.arange(NUM_GLOBAL_BUCKETS))
    rank_in_bucket = np.arange(candidates.size) - starts[g]
    chosen = candidates[rank_in_bucket < per_bucket]
    assert (
        chosen.size == NUM_GLOBAL_BUCKETS * per_bucket
    ), "a bucket got too few candidates"
    return chosen


# ---------------------------------------------------------------------------
# 1. The table kernels: a key's bucket is the global one, folded into the shard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_shards", [1, 2, 8])
def test_table_places_keys_in_their_global_bucket(current_device, num_shards):
    device = torch.device(f"cuda:{current_device}")
    P = buckets_per_shard(num_shards)
    keys = keys_filling_every_bucket(per_bucket=16)
    owner = oracle_shard(keys, num_shards)
    g = oracle_global_bucket(keys)
    for shard in range(num_shards):
        mine = torch.from_numpy(keys[owner == shard]).to(device)
        table = get_scored_table(
            capacity=[P * BC],
            bucket_capacity=BC,
            key_type=torch.int64,
            score_specs=[ScoreSpec(name="s", policy=ScorePolicy.ASSIGN)],
            device=device,
            global_capacities=[GLOBAL_CAPACITY],
        )
        assert table.per_table_num_buckets_ == [P]
        score = ScoreArg(
            name="s", value=torch.ones(mine.numel(), dtype=torch.uint64, device=device)
        )
        slots = table.insert(mine, torch.zeros_like(mine), score)
        assert (slots >= 0).all(), "every key fits: 16 of 128 slots per bucket"
        local_bucket = (slots // BC).cpu().numpy()
        expected = g[owner == shard] - shard * P
        assert (local_bucket == expected).all()
        # Lookup probes the same bucket, so every key is found again.
        _, founds, _ = table.lookup(
            mine,
            torch.zeros_like(mine),
            ScoreArg(
                name="s",
                value=torch.zeros(mine.numel(), dtype=torch.uint64, device=device),
                policy=ScorePolicy.CONST,
            ),
        )
        assert founds.all()


def test_table_without_global_capacity_keeps_the_old_buckets(current_device):
    """Unset, or equal to the table's own capacity, the formula is the old one
    bit for bit -- roundrobin tables and caches must not move a single key."""
    device = torch.device(f"cuda:{current_device}")
    capacity = 4096 * BC
    keys = torch.randint(1, 1 << 62, (200_000,), dtype=torch.int64, device=device)
    keys[:500] = -keys[:500]
    h = oracle_hash(keys.cpu().numpy())
    old_bucket = (h % capacity) // BC
    for global_capacities in (None, [capacity], [0]):
        table = get_scored_table(
            capacity=[capacity],
            bucket_capacity=BC,
            key_type=torch.int64,
            score_specs=[ScoreSpec(name="s", policy=ScorePolicy.ASSIGN)],
            device=device,
            global_capacities=global_capacities,
        )
        assert table.table_global_capacities_ is None
        score = ScoreArg(
            name="s", value=torch.ones(keys.numel(), dtype=torch.uint64, device=device)
        )
        slots = table.insert(keys, torch.zeros_like(keys), score)
        placed = (slots >= 0).cpu().numpy()
        assert placed.mean() > 0.9
        assert ((slots // BC).cpu().numpy()[placed] == old_bucket[placed]).all()


def test_table_rejects_a_global_capacity_that_cannot_hold_its_buckets(current_device):
    device = torch.device(f"cuda:{current_device}")
    spec = [ScoreSpec(name="s", policy=ScorePolicy.ASSIGN)]
    with pytest.raises(ValueError, match="not a multiple"):
        get_scored_table(
            capacity=[BC * 4],
            bucket_capacity=BC,
            score_specs=spec,
            device=device,
            global_capacities=[BC * 4 + 1],
        )
    with pytest.raises(ValueError, match="smaller than the table"):
        get_scored_table(
            capacity=[BC * 4],
            bucket_capacity=BC,
            score_specs=spec,
            device=device,
            global_capacities=[BC * 2],
        )
    with pytest.raises(ValueError, match="entries for"):
        get_scored_table(
            capacity=[BC * 4, BC * 4],
            bucket_capacity=BC,
            score_specs=spec,
            device=device,
            global_capacities=[BC * 4],
        )


# ---------------------------------------------------------------------------
# 2. The routing twins: kernel, host rule and oracle agree key for key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("num_shards", [1, 2, 3, 8])
def test_routing_kernel_and_host_rule_agree(current_device, num_shards, key_dtype):
    """Kernel, host rule and oracle agree key for key -- for int32 keys too. A
    negative int32 key must widen with its sign before hashing (the table and
    the host rule see it as int64), not as the kernel's unsigned index."""
    device = torch.device(f"cuda:{current_device}")
    if key_dtype == torch.int64:
        keys = torch.randint(1, 1 << 62, (300_000,), dtype=torch.int64, device=device)
        keys[:1000] = -keys[:1000]
    else:
        keys = torch.randint(
            -(1 << 31), 1 << 31, (300_000,), dtype=torch.int32, device=device
        )
        assert (keys < 0).sum() > 100_000
    expected = oracle_shard(keys.to(torch.int64).cpu().numpy(), num_shards)
    shard_capacity = buckets_per_shard(num_shards) * BC
    assert hashed_shard_capacity(GLOBAL_CAPACITY, BC, num_shards) == shard_capacity

    host = owning_shard(
        keys,
        num_shards,
        "hash_bucket",
        global_capacity=GLOBAL_CAPACITY,
        bucket_capacity=BC,
    )
    assert (host.cpu().numpy() == expected).all()
    rule = OwnershipRule("hash_bucket", num_shards, GLOBAL_CAPACITY, BC)
    assert rule.shard_capacity == shard_capacity
    assert torch.equal(rule.owner(keys), host)
    if key_dtype == torch.int32:
        # The rule is about the key's value, whatever width it arrives in.
        assert torch.equal(rule.owner(keys.to(torch.int64)), host)

    global_caps, shard_caps = routing_param_tensors(
        ["hash_bucket"], [(GLOBAL_CAPACITY, BC, shard_capacity)], num_shards, device
    )
    # The shard capacity is derived, the way the host rule derives it.
    assert global_caps.tolist() == [GLOBAL_CAPACITY]
    assert shard_caps.tolist() == [shard_capacity]
    new_lengths, _, _, _, permute = block_bucketize_sparse_features(
        torch.tensor([keys.numel()], dtype=torch.int64, device=device),
        keys,
        bucketize_pos=False,
        sequence=True,
        dist_type_per_feature=dist_type_codes(["hash_bucket"], device=device),
        global_capacity_per_feature=global_caps,
        shard_capacity_per_feature=shard_caps,
        # The kernel reads block_sizes in the keys' dtype (the distributor
        # converts them, _fx_wrap_tensor_to_device_dtype).
        block_sizes=torch.tensor(
            [GLOBAL_CAPACITY // num_shards + 1], dtype=key_dtype, device=device
        ),
        my_size=num_shards,
        max_B=1,
    )
    counts = new_lengths.cpu().numpy()[:num_shards]
    kernel = np.repeat(np.arange(num_shards), counts)[permute.cpu().numpy()]
    assert (kernel == expected).all()


def test_routing_without_parameters_is_refused():
    device = torch.device("cuda")
    with pytest.raises(ValueError, match="global_capacity"):
        owning_shard(torch.arange(10, device=device), 2, "hash_bucket")
    with pytest.raises(ValueError, match="needs a positive"):
        routing_param_tensors(["hash_bucket"], [(0, 0, 0)], 2, device)
    with pytest.raises(ValueError, match="another number of shards"):
        # A plan made for 8 shards (max_capacity is one 8th of the table) on a
        # 2-shard distributor: the route would use another shard capacity than
        # the host rule derives for 2 shards, and routing and loading would
        # silently disagree.
        routing_param_tensors(
            ["hash_bucket"],
            [(GLOBAL_CAPACITY, BC, buckets_per_shard(8) * BC)],
            2,
            device,
        )
    with pytest.raises(ValueError, match="multiple"):
        # hashed_shard_capacity refuses a global capacity that is not a whole
        # number of buckets.
        routing_param_tensors(
            ["hash_bucket"], [(GLOBAL_CAPACITY + 1, BC, BC)], 2, device
        )
    # Every shard count the plan can be made for is accepted with its own plan.
    for shards in (1, 2, 3, 8):
        g, s = routing_param_tensors(
            ["hash_bucket"],
            [(GLOBAL_CAPACITY, BC, buckets_per_shard(shards) * BC)],
            shards,
            device,
        )
        assert g.tolist() == [GLOBAL_CAPACITY]
        assert s.tolist() == [buckets_per_shard(shards) * BC]
    # Other rules carry no parameters and are left alone.
    g, s = routing_param_tensors(["roundrobin", "continuous"], None, 2, device)
    assert g.tolist() == [0, 0] and s.tolist() == [0, 0]


@pytest.mark.parametrize("dist_type", ["roundrobin", "hash_roundrobin", "hash_bucket"])
def test_row_positions_are_refused_for_a_rule_that_is_not_continuous(dist_type):
    """``block_bucketize_row_pos`` makes the kernel place keys by row range and
    ignore the feature's dist_type; for anything but ``continuous`` that would
    route a key by a rule the table does not follow, with nothing raising."""
    device = torch.device("cuda")
    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=["f0", "f1"],
        values=torch.arange(8, dtype=torch.int64, device=device),
        lengths=torch.tensor([2, 2, 2, 2], dtype=torch.int32, device=device),
    )
    positions = [torch.tensor([0, 5], dtype=torch.int32, device=device)] * 2
    block_sizes = torch.full((2,), 5, dtype=torch.int64, device=device)

    # One feature that is not continuous is enough, and it is named.
    mixed = dist_type_codes(["continuous", dist_type], device=device)
    with pytest.raises(ValueError, match=r"\['f1'\]"):
        bucketize_kjt_before_all2all(
            kjt,
            num_buckets=2,
            block_sizes=block_sizes,
            block_bucketize_row_pos=positions,
            dist_type_per_feature=mixed,
            global_capacity_per_feature=torch.tensor(
                [0, GLOBAL_CAPACITY], dtype=torch.int64, device=device
            ),
            shard_capacity_per_feature=torch.tensor(
                [0, buckets_per_shard(2) * BC], dtype=torch.int64, device=device
            ),
        )
    # Without positions the same call is fine.
    bucketize_kjt_before_all2all(
        kjt,
        num_buckets=2,
        block_sizes=block_sizes,
        dist_type_per_feature=mixed,
        global_capacity_per_feature=torch.tensor(
            [0, GLOBAL_CAPACITY], dtype=torch.int64, device=device
        ),
        shard_capacity_per_feature=torch.tensor(
            [0, buckets_per_shard(2) * BC], dtype=torch.int64, device=device
        ),
    )


# ---------------------------------------------------------------------------
# 3. The planner derives the layout; the per-shard size is what it always was
# ---------------------------------------------------------------------------


def _plan(
    num_embeddings: int, world_size: int, local_size: int = None, **option_kwargs
):
    options = DynamicEmbTableOptions(**option_kwargs)
    key = TableKey("ebc", "t")
    constraint = DynamicEmbParameterConstraints(
        sharding_types=[ShardingType.ROW_WISE.value],
        use_dynamicemb=True,
        dynamicemb_options=options,
    )
    config = EmbeddingConfig(
        name="t", embedding_dim=DIM, num_embeddings=num_embeddings, feature_names=["f"]
    )
    _prepare_dynemb_table_options(
        {key: constraint}, {key: config}, world_size, local_size or world_size
    )
    return options


@pytest.mark.parametrize("world_size", [1, 2, 3, 5, 8])
def test_planner_sets_the_global_capacity_only_for_hash_bucket(world_size):
    config = EmbeddingConfig(
        name="t", embedding_dim=DIM, num_embeddings=NUM_EMBEDDINGS, feature_names=["f"]
    )
    hashed = _plan(
        NUM_EMBEDDINGS, world_size, dist_type="hash_bucket", bucket_capacity=BC
    )
    plain = _plan(
        NUM_EMBEDDINGS, world_size, dist_type="roundrobin", bucket_capacity=BC
    )

    assert (
        hashed.global_capacity
        == GLOBAL_CAPACITY
        == get_global_table_capacity(config, BC)
    )
    assert plain.global_capacity is None
    # The shard keeps the size it always had: ceil(ceil(N/bc)/S) == ceil(ceil(N/S)/bc).
    expected_shard = buckets_per_shard(world_size) * BC
    assert hashed.max_capacity == plain.max_capacity == expected_shard
    assert expected_shard == get_sharded_table_capacity(config, world_size, BC)
    assert hashed.init_capacity == hashed.max_capacity
    assert hashed.max_capacity == hashed_shard_capacity(GLOBAL_CAPACITY, BC, world_size)
    # The sum over shards may exceed the global count, never fall short.
    assert world_size * hashed.max_capacity >= GLOBAL_CAPACITY
    assert (world_size - 1) * hashed.max_capacity < GLOBAL_CAPACITY + world_size * BC


def test_planner_keeps_a_small_init_capacity_for_hash_bucket():
    """A hash_bucket table may start below max_capacity and grow like any other:
    its keys go to a rank by the whole table's bucket map, which does not change with
    its size, so the planner no longer raises init_capacity (nor warns)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hashed = _plan(
            NUM_EMBEDDINGS,
            4,
            dist_type="hash_bucket",
            bucket_capacity=BC,
            init_capacity=4096,
        )
        plain = _plan(
            NUM_EMBEDDINGS,
            4,
            dist_type="roundrobin",
            bucket_capacity=BC,
            init_capacity=4096,
        )
    assert hashed.init_capacity == plain.init_capacity == 4096 < hashed.max_capacity
    # the bucket map stays the whole table's, whatever the table starts at
    assert hashed.global_capacity == GLOBAL_CAPACITY


def test_planner_one_bucket_per_shard_routes_over_the_shards_laid_end_to_end():
    """MAX_BUCKET_CAPACITY has no whole-table form: the rule's modulus is then
    world_size * shard rows, which routes uniformly but is not fan-out invariant."""
    for world_size in (1, 2, 8):
        # Routed consistently, but the user is told it is not fan-out invariant.
        with pytest.warns(UserWarning, match="not fan-out invariant"):
            opts = _plan(
                NUM_EMBEDDINGS,
                world_size,
                dist_type="hash_bucket",
                bucket_capacity=MAX_BUCKET_CAPACITY,
            )
        assert opts.global_capacity == world_size * opts.max_capacity
        assert (
            hashed_shard_capacity(
                opts.global_capacity, opts.bucket_capacity, world_size
            )
            == opts.max_capacity
        )


def test_planner_keeps_each_fused_table_its_own_global_capacity():
    a = _plan(NUM_EMBEDDINGS, 8, dist_type="hash_bucket", bucket_capacity=BC)
    b = _plan(NUM_EMBEDDINGS // 3, 8, dist_type="hash_bucket", bucket_capacity=BC)
    assert a.bucket_capacity == b.bucket_capacity == BC
    assert a.global_capacity != b.global_capacity
    assert b.global_capacity == get_global_table_capacity(
        EmbeddingConfig(
            name="b",
            embedding_dim=DIM,
            num_embeddings=NUM_EMBEDDINGS // 3,
            feature_names=["f"],
        ),
        BC,
    )


# ---------------------------------------------------------------------------
# 4. The point: dump at one fan-out, load at another, lose nothing
# ---------------------------------------------------------------------------


def build_model(
    current_device: int,
    num_shards: int,
    dist_type: str = "hash_bucket",
    caching: bool = False,
    score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
    hybrid: bool = False,
    init_fraction: float = 1.0,
) -> BatchedDynamicEmbeddingTablesV2:
    """One shard of a NUM_EMBEDDINGS-row table split over ``num_shards``, sized
    the way the planner sizes it (``_sharded_table_bucket_layout``). ``hybrid``
    gives it an HBM budget of half its rows and no cache: a HYBRID table, two
    tiers that are each a rescaled part of the shard. ``init_fraction`` starts it
    at that fraction of its rows (a whole number of buckets), to grow from."""
    assert not (caching and hybrid)
    config = EmbeddingConfig(
        name=TABLE,
        embedding_dim=DIM,
        num_embeddings=NUM_EMBEDDINGS,
        feature_names=["f"],
    )
    num_buckets, _ = _sharded_table_bucket_layout(config, num_shards, BC)
    shard_capacity = num_buckets * BC
    value_bytes = shard_capacity * DIM * 4
    options = DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=current_device,
        dim=DIM,
        max_capacity=shard_capacity,
        init_capacity=max(BC, int(shard_capacity * init_fraction) // BC * BC),
        global_capacity=GLOBAL_CAPACITY if dist_type == "hash_bucket" else None,
        bucket_capacity=BC,
        dist_type=dist_type,
        safe_check_mode=DynamicEmbCheckMode.IGNORE,
        # A caching table keeps a GPU cache half the size of its host store (it
        # must hold a forward's worth of keys, INSERT_CHUNK, with room to spare).
        local_hbm_for_values=value_bytes // 2 if (caching or hybrid) else 1024**3,
        caching=caching,
        score_strategy=score_strategy,
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
    if hybrid:
        assert isinstance(model._storage, HybridStorage), type(model._storage)
    return model


def as_shard(
    model: BatchedDynamicEmbeddingTablesV2, shard: int, num_shards: int
) -> None:
    """Make a module built in one process stand for shard ``shard`` of ``num_shards``:
    dump names its files by these, and load filters by them."""
    model._shard_index = shard
    model._num_shards = num_shards
    model._shard_ranks = list(range(num_shards))
    model._shard_world_size = num_shards


INSERT_CHUNK = 16384

# The compound strategy is the LruLfu policy (two score words per key). Its
# insert and eviction do not run in the AoT table kernel but in a driver-launched
# cubin (src/jit/evict_lrulfu.cu), which rebuilds the table from EvictParams and
# carries its own copy of the home-bucket arithmetic -- the one place
# table_global_capacities reaches by a struct field rather than a kernel
# argument. STEP, LFU alone and TIMESTAMP all take the AoT kernel.
LRU_LFU = (DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU)


def insert_keys(model: BatchedDynamicEmbeddingTablesV2, keys: torch.Tensor) -> None:
    """A training forward inserts the keys it does not find; one key per bag."""
    for start in range(0, keys.numel(), INSERT_CHUNK):
        chunk = keys[start : start + INSERT_CHUNK]
        offsets = torch.arange(
            0, chunk.numel() + 1, dtype=torch.int64, device=keys.device
        )
        model(chunk, offsets)
    torch.cuda.synchronize()


def owner_of(keys: np.ndarray, num_shards: int, dist_type: str) -> np.ndarray:
    if dist_type == "hash_bucket":
        return oracle_shard(keys, num_shards)
    if dist_type == "hash_roundrobin":
        return (murmur3_fmix64(keys) % np.uint64(num_shards)).astype(np.int64)
    return keys % num_shards  # keys are positive


def dump_shards(
    current_device: int,
    keys: np.ndarray,
    num_shards: int,
    save_dir: str,
    dist_type: str,
    caching: bool = False,
    score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
    hybrid: bool = False,
    init_fraction: float = 1.0,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Insert ``keys`` into ``num_shards`` modules, dump them as one checkpoint,
    and return the keys the shards actually hold (what the dump wrote) with
    their embedding rows."""
    device = torch.device(f"cuda:{current_device}")
    owner = owner_of(keys, num_shards, dist_type)
    held, values = [], []
    for shard in range(num_shards):
        model = build_model(
            current_device,
            num_shards,
            dist_type,
            caching=caching,
            score_strategy=score_strategy,
            hybrid=hybrid,
            init_fraction=init_fraction,
        )
        insert_keys(model, torch.from_numpy(keys[owner == shard]).to(device))
        as_shard(model, shard, num_shards)
        model.dump(save_dir)
        k, v = model.export_keys_values(TABLE, device)
        held.append(k.cpu().numpy())
        values.append(v.cpu())
        del model
        torch.cuda.empty_cache()
    held = np.concatenate(held)
    assert np.unique(held).size == held.size
    return held, torch.cat(values)


def load_shards(
    current_device: int,
    num_shards: int,
    save_dir: str,
    dist_type: str,
    caching: bool = False,
    score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
    hybrid: bool = False,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Load the checkpoint into ``num_shards`` fresh modules; return every key
    they hold and the corresponding embedding rows."""
    device = torch.device(f"cuda:{current_device}")
    keys, values = [], []
    for shard in range(num_shards):
        model = build_model(
            current_device,
            num_shards,
            dist_type,
            caching=caching,
            score_strategy=score_strategy,
            hybrid=hybrid,
        )
        as_shard(model, shard, num_shards)
        model.load(save_dir)
        k, v = model.export_keys_values(TABLE, device)
        keys.append(k.cpu().numpy())
        values.append(v.cpu())
        del model
        torch.cuda.empty_cache()
    return np.concatenate(keys), torch.cat(values)


def assert_debug_rows(keys: np.ndarray, values: torch.Tensor) -> None:
    expected = torch.from_numpy(keys % DEBUG_MOD).to(values.dtype)
    torch.testing.assert_close(values, expected.unsqueeze(1).expand(-1, values.size(1)))


def assert_same_rows(
    src_keys: np.ndarray,
    src_values: torch.Tensor,
    keys: np.ndarray,
    values: torch.Tensor,
) -> None:
    """Every loaded key carries the row the source held for it, bit for bit."""
    src_order = np.argsort(src_keys)
    order = np.argsort(keys)
    assert np.array_equal(src_keys[src_order], keys[order])
    torch.testing.assert_close(
        values[order], src_values[src_order], rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.parametrize("src_shards,dst_shards", [(8, 1), (1, 2), (2, 1), (8, 3)])
@pytest.mark.parametrize("caching", [False, True])
def test_hash_bucket_checkpoint_reloads_losslessly_at_another_fanout(
    current_device, src_shards, dst_shards, caching
):
    keys = keys_filling_every_bucket(KEYS_PER_BUCKET)
    with tempfile.TemporaryDirectory() as save_dir:
        held, held_values = dump_shards(
            current_device, keys, src_shards, save_dir, "hash_bucket", caching
        )
        # 97 % of every bucket fits at the source: nothing was evicted on the way in.
        assert np.array_equal(np.sort(held), np.sort(keys))
        loaded, values = load_shards(
            current_device, dst_shards, save_dir, "hash_bucket", caching
        )
    lost = np.setdiff1d(keys, loaded)
    assert (
        lost.size == 0
    ), f"{lost.size} of {keys.size} rows lost on {src_shards} -> {dst_shards}"
    assert loaded.size == keys.size, "no row duplicated across shards"
    # The rows arrive as they were written. The HBM-only table also still holds
    # every key's DEBUG pattern; a caching table at this fill does not (its
    # cache spills rows to the host store without their values -- a pre-existing
    # CACHING behaviour, unrelated to the bucket layout, so it is not asserted).
    assert_same_rows(held, held_values, loaded, values)
    if not caching:
        assert_debug_rows(loaded, values)


@pytest.mark.parametrize("src_shards,dst_shards", [(8, 1), (2, 1), (1, 2)])
def test_hash_bucket_lru_lfu_checkpoint_reloads_losslessly_at_another_fanout(
    current_device, src_shards, dst_shards
):
    """The same fold as above for the compound (TIMESTAMP, LFU) strategy, whose
    inserts run in the LruLfu cubin. A cubin that ignored table_global_capacities
    -- a fatbin built before EvictParams grew the field -- would place keys by the
    per-shard layout: buckets that fit under the global one overflow, so rows are
    evicted on the way in (``held`` falls short of ``keys``) or on the way back."""
    keys = keys_filling_every_bucket(KEYS_PER_BUCKET)
    with tempfile.TemporaryDirectory() as save_dir:
        held, held_values = dump_shards(
            current_device,
            keys,
            src_shards,
            save_dir,
            "hash_bucket",
            score_strategy=LRU_LFU,
        )
        assert np.array_equal(np.sort(held), np.sort(keys))
        loaded, values = load_shards(
            current_device,
            dst_shards,
            save_dir,
            "hash_bucket",
            score_strategy=LRU_LFU,
        )
    lost = np.setdiff1d(keys, loaded)
    assert (
        lost.size == 0
    ), f"{lost.size} of {keys.size} rows lost on {src_shards} -> {dst_shards}"
    assert loaded.size == keys.size, "no row duplicated across shards"
    assert_same_rows(held, held_values, loaded, values)
    assert_debug_rows(loaded, values)


def random_keys(n: int, seed: int = 1) -> np.ndarray:
    """``n`` distinct random keys -- the realistic case: no layout was consulted,
    so buckets fill unevenly and the fullest overflow during training."""
    keys = np.unique(
        np.random.default_rng(seed).integers(
            1, 1 << 62, size=int(n * 1.02), dtype=np.int64
        )
    )
    assert keys.size >= n
    return keys[:n]


@pytest.mark.parametrize(
    "score_strategy",
    [DynamicEmbScoreStrategy.TIMESTAMP, LRU_LFU],
    ids=["lru", "lru_lfu"],
)
def test_hash_bucket_training_survivors_reload_losslessly(
    current_device, score_strategy
):
    """Random keys at 97 % overall fill: the fullest global buckets overflow while
    training and evict, as they would in production. What the 8 shards still
    hold after that fits its buckets by construction -- and the same buckets at
    1 shard, so the fold keeps every surviving row. Under LRU_LFU the evictions
    are the LruLfu cubin's: it has to choose its victim inside the *global*
    bucket, or the survivors would not fit the buckets they are reloaded into."""
    keys = random_keys(int(0.97 * GLOBAL_CAPACITY))
    with tempfile.TemporaryDirectory() as save_dir:
        held, held_values = dump_shards(
            current_device,
            keys,
            8,
            save_dir,
            "hash_bucket",
            score_strategy=score_strategy,
        )
        assert (
            held.size < keys.size
        ), "97 % random fill should have evicted at the source"
        loaded, values = load_shards(
            current_device, 1, save_dir, "hash_bucket", score_strategy=score_strategy
        )
    assert np.setdiff1d(held, loaded).size == 0, "survivors lost on the 8 -> 1 fold"
    assert loaded.size == held.size
    assert_same_rows(held, held_values, loaded, values)
    assert_debug_rows(loaded, values)


@pytest.mark.parametrize("num_shards", [1, 2, 8])
def test_hash_bucket_table_grows_and_keeps_its_keys(current_device, num_shards):
    """A hash_bucket table that starts at a thirty-second of its rows grows as keys
    arrive: the whole table's bucket map does not change with its size, the table
    folds it into the buckets it has (``g % num_buckets``), and more buckets split
    the old ones. Every key must still be in the table after the growth, a second
    pass over them must find them rather than insert them again, and the table must
    not outgrow its max_capacity."""
    device = torch.device(f"cuda:{current_device}")
    keys = keys_filling_every_bucket(per_bucket=16)  # 12.5 % of every global bucket
    owner = oracle_shard(keys, num_shards)
    for shard in range(num_shards):
        mine = keys[owner == shard]
        model = build_model(
            current_device, num_shards, "hash_bucket", init_fraction=1 / 32
        )
        capacity = model._storage._state.key_index_map.per_table_capacity_
        start = capacity[0]
        insert_keys(model, torch.from_numpy(mine).to(device))
        grown = model._storage._state.key_index_map.per_table_capacity_[0]
        assert grown > start, "the table should have grown"
        assert grown <= model._dynamicemb_options[0].max_capacity
        held = model.export_keys_values(TABLE, device)[0].cpu().numpy()
        assert np.array_equal(np.sort(held), np.sort(mine)), "a key was lost"
        # the keys are found where they were put
        insert_keys(model, torch.from_numpy(mine).to(device))
        again = model.export_keys_values(TABLE, device)[0].cpu().numpy()
        assert np.array_equal(np.sort(again), np.sort(mine)), "a key was inserted twice"
        del model
        torch.cuda.empty_cache()


def test_hash_bucket_growth_does_not_stop_on_the_uneven_fold(current_device):
    """Where the number of buckets does not divide the shard's P (977 over 8 shards),
    the fold ``g % num_buckets`` puts floor(P / num_buckets) or ceil(P / num_buckets)
    global buckets in a bucket, the first P mod num_buckets of them the extra one.
    Between P / 2 and P that is one against two: at 960 buckets the first 17 expect
    about 126 keys of 128 at the usual load of 0.5, and evict (about 59 of 61 440
    keys). So a table with a global capacity whose next size would pass P / 2 grows to
    P (get_expand_info), where the fold is the identity. Random keys fill the first and
    the last shard to 45 % of its rows: the table must end at its max_capacity and
    nothing may be lost."""
    num_shards = 8
    device = torch.device(f"cuda:{current_device}")
    keys = random_keys(int(0.45 * GLOBAL_CAPACITY))
    owner = oracle_shard(keys, num_shards)
    buckets = buckets_per_shard(num_shards)
    for shard in (0, num_shards - 1):
        mine = keys[owner == shard]
        model = build_model(
            current_device, num_shards, "hash_bucket", init_fraction=1 / 32
        )
        state = model._storage._state
        start = state.key_index_map.per_table_capacity_[0] // BC
        assert buckets % start != 0, "it starts on a fold that does not divide P"
        insert_keys(model, torch.from_numpy(mine).to(device))
        capacity = state.key_index_map.per_table_capacity_[0]
        assert capacity == model._dynamicemb_options[0].max_capacity == buckets * BC
        held = model.export_keys_values(TABLE, device)[0].cpu().numpy()
        assert (
            held.size == mine.size
        ), f"{mine.size - held.size} of {mine.size} keys evicted at 45 % fill"
        del model
        torch.cuda.empty_cache()


@pytest.mark.parametrize("src_shards,dst_shards", [(2, 1), (8, 1), (1, 2)])
def test_hash_bucket_grown_checkpoint_reloads_losslessly_at_another_fanout(
    current_device, src_shards, dst_shards
):
    """The tables that were dumped grew from a thirty-second of their rows; the
    checkpoint holds keys, not buckets, so it reloads at another fan-out like one
    of a table that started full."""
    keys = keys_filling_every_bucket(per_bucket=16)
    with tempfile.TemporaryDirectory() as save_dir:
        held, held_values = dump_shards(
            current_device,
            keys,
            src_shards,
            save_dir,
            "hash_bucket",
            init_fraction=1 / 32,
        )
        assert np.array_equal(np.sort(held), np.sort(keys)), "lost while growing"
        loaded, values = load_shards(
            current_device, dst_shards, save_dir, "hash_bucket"
        )
    lost = np.setdiff1d(keys, loaded)
    assert (
        lost.size == 0
    ), f"{lost.size} of {keys.size} rows lost on {src_shards} -> {dst_shards}"
    assert loaded.size == keys.size, "no row duplicated across shards"
    assert_same_rows(held, held_values, loaded, values)


def test_roundrobin_checkpoint_loses_rows_at_another_fanout(current_device):
    """The control: the same random keys, the same fill, the old per-shard
    layout. What the 8 shards hold fits *their* buckets; folded onto 1 shard the
    keys are re-dealt into other buckets and the fullest overflow -- the loss
    this change removes, kept here so the lossless cases are seen to test
    something."""
    keys = random_keys(int(0.97 * GLOBAL_CAPACITY))
    with tempfile.TemporaryDirectory() as save_dir:
        held, _ = dump_shards(current_device, keys, 8, save_dir, "roundrobin")
        loaded, _ = load_shards(current_device, 1, save_dir, "roundrobin")
    lost = np.setdiff1d(held, loaded)
    assert (
        lost.size > 0
    ), "the per-shard layout should overflow buckets on an 8 -> 1 fold"
    print(
        f"\nroundrobin control: {lost.size} of {held.size} rows lost "
        f"({lost.size / held.size:.2%})"
    )


@pytest.mark.parametrize("num_shards", [1, 2])
def test_hash_bucket_hybrid_checkpoint_roundtrips_at_the_same_fanout(
    current_device, num_shards
):
    """A HYBRID table is accepted under ``hash_bucket`` but gets no fan-out
    guarantee: its two tiers are rescaled parts of the shard and keep their own
    per-tier hashing, so there is no lossless-across-fan-outs case here. What it
    does have is the global routing rule, so a dump and a load at the *same* fan-out
    must give every shard exactly its keys back. That goes through
    ``HybridStorage.load``, which reads the key -> rank rule from the *host* tier's
    options (the HBM tier's copy rescaled ``bucket_capacity``); a rule read from
    the wrong tier would filter keys away, or keep another shard's. The fill is low
    (16 of 128 slots per global bucket), so nothing is evicted and every key has to
    come back."""
    keys = keys_filling_every_bucket(per_bucket=16)
    with tempfile.TemporaryDirectory() as save_dir:
        held, held_values = dump_shards(
            current_device, keys, num_shards, save_dir, "hash_bucket", hybrid=True
        )
        assert np.array_equal(np.sort(held), np.sort(keys)), "nothing evicted at 12 %"
        loaded, values = load_shards(
            current_device, num_shards, save_dir, "hash_bucket", hybrid=True
        )
    lost = np.setdiff1d(keys, loaded)
    assert (
        lost.size == 0
    ), f"{lost.size} of {keys.size} rows lost at {num_shards} shards"
    assert loaded.size == keys.size, "no row duplicated across shards"
    assert_same_rows(held, held_values, loaded, values)


# ---------------------------------------------------------------------------
# 5. Checkpoints written under another layout are re-partitioned, not trusted
# ---------------------------------------------------------------------------


def _meta_path(save_dir: str) -> str:
    return bdt_module.encode_meta_json_file_path(save_dir, TABLE)


def _load_with_spy(monkeypatch, current_device, num_shards, save_dir, dist_type):
    """Load and record the ``read_all`` each get_loading_files call was given."""
    seen: List[bool] = []
    real = bdt_module.get_loading_files

    def spy(*args, **kwargs):
        seen.append(bool(kwargs.get("read_all", False)))
        return real(*args, **kwargs)

    monkeypatch.setattr(bdt_module, "get_loading_files", spy)
    loaded, values = load_shards(current_device, num_shards, save_dir, dist_type)
    return loaded, values, seen


def test_checkpoint_of_another_layout_is_read_whole_and_refiltered(
    monkeypatch, current_device
):
    """A hash_bucket checkpoint whose recorded layout differs from the table's
    (here: another bucket_capacity) may have grouped the keys differently, so
    every rank reads every file -- and says so once -- instead of trusting its
    own shard file."""
    keys = keys_filling_every_bucket(per_bucket=32)
    with tempfile.TemporaryDirectory() as save_dir:
        dump_shards(current_device, keys, 2, save_dir, "hash_bucket")
        meta_path = _meta_path(save_dir)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["global_capacity"] == GLOBAL_CAPACITY
        assert meta["bucket_capacity"] == BC
        assert meta["dist_type"] == "hash_bucket"

        # Same layout, same shard count: each rank reads only its own file.
        _, _, seen = _load_with_spy(
            monkeypatch, current_device, 2, save_dir, "hash_bucket"
        )
        assert seen == [False, False]

        meta["bucket_capacity"] = BC * 2
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        with pytest.warns(UserWarning, match="global bucket layout"):
            loaded, values, seen = _load_with_spy(
                monkeypatch, current_device, 2, save_dir, "hash_bucket"
            )
    assert seen == [True, True]
    # The files were in fact written under this layout, so re-filtering is exact.
    assert np.array_equal(np.sort(loaded), np.sort(keys))
    assert_debug_rows(loaded, values)


@pytest.mark.parametrize("source", ["roundrobin", "hash_roundrobin"])
def test_old_rule_checkpoint_loads_into_a_hash_bucket_table(
    monkeypatch, current_device, source
):
    """Migration: ownership is recomputed from the key by the table's rule, so a
    roundrobin or hash_roundrobin checkpoint loads into a hash_bucket table --
    every rank reading every file, with a warning that full buckets may lose
    rows. At this fill nothing overflows, so the keys all arrive; the layout
    they land in is the invariant one from here on."""
    keys = keys_filling_every_bucket(per_bucket=32)
    with tempfile.TemporaryDirectory() as save_dir:
        held, _ = dump_shards(current_device, keys, 2, save_dir, source)
        with pytest.warns(UserWarning, match="dist_type") as record:
            loaded, values, seen = _load_with_spy(
                monkeypatch, current_device, 2, save_dir, "hash_bucket"
            )
    assert seen == [True, True]
    assert np.array_equal(np.sort(loaded), np.sort(held))
    assert_debug_rows(loaded, values)
    # Said once, by shard 0, for the table -- not once per shard file per rank.
    layout_warnings = [w for w in record if "re-partition" in str(w.message)]
    assert len(layout_warnings) == 1, [str(w.message) for w in layout_warnings]


def _shard_files(save_dir: str, shard: int, num_shards: int) -> Tuple[str, ...]:
    return tuple(
        bdt_module.encode_checkpoint_file_path(save_dir, TABLE, shard, num_shards, kind)
        for kind in ("keys", "values", "scores", "opt_values")
    )


def test_direct_storage_load_warns_once_per_table(current_device):
    """A Storage.load outside the module has nobody else to say it, so it warns
    itself -- once for the table, however many shard files it is handed."""
    keys = keys_filling_every_bucket(per_bucket=16)
    with tempfile.TemporaryDirectory() as save_dir:
        dump_shards(current_device, keys, 2, save_dir, "roundrobin")
        model = build_model(current_device, 2, "hash_bucket")
        meta = _meta_path(save_dir)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            for shard in range(2):
                key_file, value_file, score_file, opt_file = _shard_files(
                    save_dir, shard, 2
                )
                model._storage.load(
                    0,
                    meta,
                    key_file,
                    value_file,
                    score_file,
                    opt_file,
                    include_optim=False,
                    shard_index=0,
                    num_shards=2,
                )
        del model
        torch.cuda.empty_cache()
    layout_warnings = [w for w in record if "re-partition" in str(w.message)]
    assert len(layout_warnings) == 1, [str(w.message) for w in layout_warnings]


def test_continuous_checkpoint_is_refused(monkeypatch, current_device):
    """A continuous table's keys are shard-local ids (the bucketizer rewrote
    them), so two source shards can hold the same id for different rows; no
    rule can re-partition them. The mismatch stays an error, in the module path
    and in a direct Storage.load."""
    keys = keys_filling_every_bucket(per_bucket=16)
    with tempfile.TemporaryDirectory() as save_dir:
        dump_shards(current_device, keys, 2, save_dir, "roundrobin")
        meta_path = _meta_path(save_dir)
        with open(meta_path) as f:
            meta = json.load(f)
        meta["dist_type"] = "continuous"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        with pytest.raises(ValueError, match="continuous"):
            _load_with_spy(monkeypatch, current_device, 2, save_dir, "hash_bucket")

        model = build_model(current_device, 2, "hash_bucket")
        key_file, value_file, score_file, opt_file = _shard_files(save_dir, 0, 2)
        with pytest.raises(ValueError, match="continuous"):
            model._storage.load(
                0,
                meta_path,
                key_file,
                value_file,
                score_file,
                opt_file,
                include_optim=False,
                shard_index=0,
                num_shards=2,
            )
        del model
        torch.cuda.empty_cache()
