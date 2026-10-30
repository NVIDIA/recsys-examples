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
"""The hierarchical dist (``output_dist="fused"``) against TorchRec's dists.

Each case builds the same EmbeddingCollection of DynamicEmb tables twice, on
TorchRec's dists and on the hierarchical dist, and trains both on the same
batches. Every step checks:

* the forward outputs: bit for bit in the first iteration, and in every
  iteration when every sum is exact (else within :func:`close`'s tolerances);
* the lookup's gradient per (table, key), the tables, the LFU scores and the
  admission counts;
* the input dist's host syncs, and in the second step an eval forward,
  which takes TorchRec's dists.

Before the cases: the sharding shapes, the ranks' agreement and the options.
After them: an input dist that waits for the device nowhere but in its
second wait, the output dist's call orders, the backward run to run, the
windows and the trace APIs, failed constructions, across nodes a proxy rows
cap below a round's rows, and the teardown.

Run it from ``corelib/dynamicemb``, one process per GPU (on one node, or
across nodes under Slurm)::

    torchrun --nproc_per_node 8 test/unit_tests/test_hier_dist.py
    srun --ntasks-per-node 8 python test/unit_tests/test_hier_dist.py

Across nodes, run it as is, and with ``DYNAMICEMB_GIN_PROXY_MB=1`` in the
environment (a proxy rows cap of 1 MiB: the larger calls move their rows in
several rounds).
"""

import contextlib
import gc
import os
import sys
import tempfile
import time
import warnings
import weakref
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torchrec
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
    EvictedItemMode,
    FrequencyAdmissionStrategy,
    KVCounter,
    get_table_value_bytes,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.dynamicemb_config import DIST_TYPE_CODES, OUTPUT_DIST_CAPACITY_NAMES
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingCollectionSharder
from dynamicemb.shard.embedding import ShardedDynamicEmbeddingCollection
from dynamicemb.shard.hier_output_dist import HierSharding
from dynamicemb.shard.rw_sharding import counts_frequencies
from dynamicemb_extensions import (
    block_bucketize_sparse_features,
    hier_output_dist_fail_next_construction,
    hier_output_dist_live_views,
    hier_output_dist_lookup_grad_live,
    hier_output_dist_traced,
)
from fbgemm_gpu.split_embedding_configs import SparseType
from torch.profiler import ProfilerActivity, profile, record_function
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.types import BoundsCheckMode, ShardingType
from torchrec.modules.embedding_configs import DataType
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# The capacities are counted as the benchmark counts them.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmark")
)
from benchmark_utils import hier_capacities, hier_needs, init_distributed  # noqa: E402

HEAVY = 64  # kHeavy in src/hier_output_dist/common/constants.h
NUM_EMBEDDINGS = 1 << 20
# Nodes of the run (main sets it): across nodes a requester's keys for an
# owner on another node reach it once per (node, table, key), and so do the
# rows of the output dist.
NODES = 1
DTYPES = {
    "fp32": (DataType.FP32, SparseType.FP32),
    "bf16": (DataType.BF16, SparseType.BF16),
    "fp16": (DataType.FP16, SparseType.FP16),
}
# Every hierarchical dist built so far, to check its teardown ran.
SHARDINGS: List[weakref.ref] = []
SYNC_CALLS = (
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaDeviceSynchronize",
    "cudaMemcpy",
)


def all_true(ok: bool) -> bool:
    flag = torch.tensor([int(ok)], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


# ── ops outside a model ──────────────────────────────────────────────────────


def bucketize_rules(num_features: int, world_size: int):
    """The bucketize's rule of each of ``num_features`` features, roundrobin
    over ``world_size`` ranks, as an op is built with them: block sizes
    (int64) and dist types (int32), on the GPU."""
    block_sizes = torch.full(
        (num_features,), -(-NUM_EMBEDDINGS // world_size), dtype=torch.int64
    )
    dist_types = torch.full(
        (num_features,), DIST_TYPE_CODES["roundrobin"], dtype=torch.int32
    )
    return block_sizes.cuda(), dist_types.cuda()


def standalone(group) -> HierSharding:
    """A hierarchical dist of 3 features of 32 fp32 columns on ``group``,
    outside any model, with capacities of 65536 entries and a proxy rows cap
    of 16 MiB (it holds the rounds of every node count here); its op is
    built by ``setup``."""
    capacity = dict.fromkeys(OUTPUT_DIST_CAPACITY_NAMES, 1 << 16)
    with environ("DYNAMICEMB_GIN_PROXY_MB", 16):
        hier = HierSharding(group, 3, 32, torch.float32, capacity)
    SHARDINGS.append(weakref.ref(hier))
    return hier


def check_group_shapes(rank: int, world_size: int) -> bool:
    """A sharding whose LOCAL_WORLD_SIZE splits each node in two (as if each
    half were a node) takes TorchRec's dists with a warning, the same on
    every rank: NCCL's LSA team holds every GPU of the node, not the half the
    sharding takes for a node. One on fewer ranks than a node, here a 2-rank
    group of the first node, runs with those 2 ranks as its node."""
    ok = True
    local_world_size = os.environ["LOCAL_WORLD_SIZE"]
    os.environ["LOCAL_WORLD_SIZE"] = str(max(int(local_world_size) // 2, 1))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hier = standalone(dist.group.WORLD)
            ok = hier.setup(*bucketize_rules(3, world_size), False) is False and any(
                "using TorchRec" in str(warning.message) for warning in caught
            )
        del hier
    finally:
        os.environ["LOCAL_WORLD_SIZE"] = local_world_size
    if world_size > 2:
        pair = dist.new_group([0, 1])
        if rank < 2:
            hier = standalone(pair)
            ok &= hier._arguments["local_size"] == 2 and hier.setup(
                *bucketize_rules(3, 2), False
            )
            del hier
    return ok


# ── models ───────────────────────────────────────────────────────────────────


@dataclass
class Case:
    name: str
    dtype: str = "fp32"  # table rows and lookup output
    # "uniform", "pow-law", or "hot": uniform, and one key in 1 of 16 places
    keys: str = "uniform"
    dedup: bool = False  # index dedup
    # DynamicEmb tables, row-wise: features per table (f0 and f1 share t0)
    tables: Tuple[Tuple[str, ...], ...] = (("f0", "f1"), ("f2",), ("f3",))
    # plain TorchRec tables, one feature each: in the row-wise sharding (which
    # then falls back to TorchRec's dists), or table-wise (TorchRec's dists)
    plain: Tuple[str, ...] = ()
    table_wise: Tuple[str, ...] = ()
    extra: Tuple[str, ...] = ()  # input features after the others, no table's
    empty: Tuple[str, ...] = ()  # features without keys
    empty_last: Tuple[str, ...] = ()  # features without keys on the last rank
    uneven: bool = False  # local batch size differs per rank
    skew: int = 1  # rank 0's local batch is this many times the batch
    # the first node's keys go to no owner of the last node (across nodes:
    # those owners get nothing from that node's proxies)
    avoid: bool = False
    int32: bool = False  # KJT values and lengths
    indices: bool = False  # the EC returns the indices too (need_indices)
    lfu: bool = False  # LFU scores (with index dedup the counters ride on the keys)
    admit: bool = False  # FrequencyAdmissionStrategy, threshold 2
    # the first table LFU, the others FrequencyAdmissionStrategy without LFU
    mixed: bool = False
    # the last DynamicEmb table in a lookup group of its own (it retains
    # evicted keys; nothing is evicted here)
    groups: bool = False
    clip: bool = False  # gradient clipping (across nodes: no node dedup)
    node_dedup: bool = True  # the tables' node_dedup option
    dist_type: str = "roundrobin"  # the DynamicEmb tables' key -> rank rule
    rules: Tuple[str, ...] = ()  # a rule per DynamicEmb table, instead of dist_type
    num_embeddings: int = NUM_EMBEDDINGS  # rows of each table
    batch: int = 128
    max_len: int = 8
    key_range: int = 4000
    dim: int = 32
    # host syncs of an input dist on the hierarchical path (None: not counted,
    # TorchRec's dists of another sharding add theirs)
    syncs: Optional[int] = 1

    @property
    def features(self) -> List[str]:
        """The features the EC serves, in the order of its shardings."""
        return (
            [feature for table in self.tables for feature in table]
            + list(self.plain)
            + list(self.table_wise)
        )

    @property
    def inputs(self) -> List[str]:
        return self.features + list(self.extra)

    @property
    def table_of(self) -> Dict[str, int]:
        groups = list(self.tables) + [
            (feature,) for feature in self.plain + self.table_wise
        ]
        return {
            feature: table for table, feats in enumerate(groups) for feature in feats
        }

    def exact(self, world_size: int) -> bool:
        # Sums of multiples of 1/8, scaled by 1/world_size, are exact in any
        # order when world_size is a power of two: in fp32, and in 16 bits on
        # one node without index dedup (which adds sums before the lookup,
        # TorchRec's in 16 bits, ours in fp32); across nodes the owner's lookup
        # backward adds a key's 16-bit rows in another order than TorchRec's
        # (its KJT's), and with node dedup a node's sum of them is rounded once
        # more. Clipped gradients are no such multiples.
        sixteen = not self.dedup and NODES == 1
        return (
            (self.dtype == "fp32" or sixteen)
            and not self.clip
            and (world_size & (world_size - 1)) == 0
        )


# Hot keys repeat hundreds of times on a rank: heavy rows of many sub-tasks.
HEAVY_ROWS = Case(
    "fp32 pow-law dedup, heavy rows",
    keys="pow-law",
    dedup=True,
    batch=1024,
    max_len=16,
    key_range=20000,
)
# 16-bit with index dedup takes uniform keys: TorchRec's index_add sums a hot
# key's gradients in 16 bits, so the reference itself would be far off.
CASES = [
    Case("fp32"),
    Case("fp32 dedup", dedup=True),
    Case("bf16", dtype="bf16"),
    Case("bf16 dedup", dtype="bf16", dedup=True),
    Case("fp16", dtype="fp16"),
    Case("bf16 pow-law", dtype="bf16", keys="pow-law"),
    Case("fp32 pow-law dedup", keys="pow-law", dedup=True),
    Case("fp32 pow-law uneven empty", keys="pow-law", uneven=True, empty=("f3",)),
    Case(
        "int32 uneven empty indices",
        int32=True,
        uneven=True,
        empty=("f1",),
        indices=True,
    ),
    Case("fp32 pow-law dedup lfu", keys="pow-law", dedup=True, lfu=True),
    Case("fp32 admission", admit=True),
    Case(
        "fp32 dedup, no keys on the last rank",
        dedup=True,
        empty_last=("f0", "f1", "f2", "f3"),
    ),
    Case("fp32 + an input feature no table serves", extra=("x0",), syncs=2),
    Case(
        "fp32 pow-law dedup lfu + table-wise table",
        keys="pow-law",
        dedup=True,
        lfu=True,
        table_wise=("g0",),
        empty_last=("g0",),
        syncs=None,
    ),
    Case("fp32 + plain table: falls back", plain=("p0",), syncs=None),
    Case("fp32 large", batch=2048, max_len=16, key_range=60000, dim=64),
    Case("fp32 large dedup", batch=2048, max_len=16, key_range=60000, dedup=True),
    Case("fp32 two lookup groups", groups=True),
    Case("bf16 dedup two lookup groups", dtype="bf16", dedup=True, groups=True),
    # LFU without index dedup: across nodes the owner's KJT carries each
    # node-unique key's occurrences as its weight
    Case("fp32 pow-law lfu, no index dedup", keys="pow-law", lfu=True),
    # gradient clipping clamps each looked-up row: across nodes every
    # occurrence crosses (no node dedup), so the clamp is TorchRec's
    Case("fp32 hot keys, gradient clipping", keys="hot", clip=True),
    # node dedup off by the tables' option: every occurrence crosses, and each
    # 16-bit gradient row is TorchRec's
    Case(
        "bf16 pow-law, node dedup off", dtype="bf16", keys="pow-law", node_dedup=False
    ),
    # continuous: a key's owner is its range, and the key is rewritten to its
    # place there, so different owners' keys collide at a proxy
    Case("fp32 continuous", dist_type="continuous", key_range=NUM_EMBEDDINGS),
    Case(
        "fp32 pow-law hash_roundrobin dedup",
        keys="pow-law",
        dedup=True,
        dist_type="hash_roundrobin",
    ),
    # LFU on one table, admission on the others: across nodes the owner's
    # KJT carries occurrence counts, which only the LFU table may see
    Case("fp32 pow-law, LFU and admission tables", keys="pow-law", mixed=True),
    HEAVY_ROWS,
    # every rank's features without keys: calls of no keys at all
    Case("fp32, no keys on any rank", empty=("f0", "f1", "f2", "f3")),
    Case("fp32 dedup, no keys on any rank", dedup=True, empty=("f0", "f1", "f2", "f3")),
    # 4 keys in all: owners 4 and up get no keys, and the 4 keys are very hot
    # (heavy rows under index dedup, heavy keys of a node across nodes)
    Case("fp32 a key range of 4 keys", key_range=4),
    Case("fp32 dedup, a key range of 4 keys", dedup=True, key_range=4),
    # across nodes: the first node's keys avoid the last node's owners
    Case(
        "fp32 one rank with most keys, a node avoiding an owner node",
        skew=8,
        empty_last=("f0", "f1", "f2"),
        avoid=True,
        batch=1024,
        max_len=16,
        key_range=200_000,
    ),
]


def build_model(
    case: Case,
    output_dist: str,
    world_size: int,
    local_size: int,
    capacity=None,
) -> DistributedModelParallel:
    """The case's EC, its DynamicEmb tables on ``output_dist`` with
    ``output_dist_capacity`` ``capacity`` (a list: that of each table)."""
    data_type, output_dtype = DTYPES[case.dtype]

    def config(name: str, feats) -> torchrec.EmbeddingConfig:
        return torchrec.EmbeddingConfig(
            name=name,
            embedding_dim=case.dim,
            num_embeddings=case.num_embeddings,
            feature_names=list(feats),
            data_type=data_type,
        )

    tables = [config(f"t{i}", feats) for i, feats in enumerate(case.tables)]
    others = [
        (f"p_{feature}", feature, ShardingType.ROW_WISE) for feature in case.plain
    ] + [
        (f"tw_{feature}", feature, ShardingType.TABLE_WISE)
        for feature in case.table_wise
    ]
    ec = torchrec.EmbeddingCollection(
        tables=tables + [config(name, (feature,)) for name, feature, _ in others],
        device=torch.device("meta"),
        need_indices=case.indices,
    )
    opt = EmbOptimType.EXACT_SGD

    def strategies(table_idx: int) -> Dict:
        extra = {}
        if case.lfu or (case.mixed and table_idx == 0):
            extra["score_strategy"] = DynamicEmbScoreStrategy.LFU
        if case.admit or (case.mixed and table_idx > 0):
            extra["admit_strategy"] = FrequencyAdmissionStrategy(
                threshold=2, counter=KVCounter(capacity=1 << 16)
            )
        return extra

    def evicted(table_idx: int) -> EvictedItemMode:
        last = case.groups and table_idx == len(tables) - 1
        return EvictedItemMode.RETAIN_KEY if last else EvictedItemMode.DISCARD

    constraints = {
        table.name: DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            enforce_hbm=True,
            bounds_check_mode=BoundsCheckMode.NONE,
            use_dynamicemb=True,
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=get_table_value_bytes(table, opt, world_size),
                caching=False,
                dist_type=case.rules[table_idx] if case.rules else case.dist_type,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.DEBUG
                ),
                output_dist=output_dist,
                output_dist_capacity=capacity[table_idx]
                if isinstance(capacity, list)
                else capacity,
                node_dedup=case.node_dedup,
                evicted_item_mode=evicted(table_idx),
                **strategies(table_idx),
            ),
        )
        for table_idx, table in enumerate(tables)
    }
    for name, _, sharding_type in others:
        constraints[name] = DynamicEmbParameterConstraints(
            sharding_types=[sharding_type.value],
            compute_kernels=[EmbeddingComputeKernel.FUSED.value],
            bounds_check_mode=BoundsCheckMode.NONE,
            use_dynamicemb=False,
        )
    topology = Topology(
        local_world_size=local_size,
        world_size=world_size,
        compute_device="cuda",
        hbm_cap=80 * 1024**3,
        ddr_cap=1024**4 // local_size,
    )
    planner = DynamicEmbeddingShardingPlanner(
        topology=topology, constraints=constraints, batch_size=case.batch
    )
    fused_params = {
        "optimizer": opt,
        "learning_rate": 1.0,
        "output_dtype": output_dtype,
    }
    if case.clip:
        fused_params.update(gradient_clipping=True, max_gradient=0.05)
    sharder = DynamicEmbeddingCollectionSharder(
        fused_params=fused_params, use_index_dedup=case.dedup
    )
    plan = planner.collective_plan(ec, [sharder], dist.GroupMember.WORLD)
    return DistributedModelParallel(
        module=ec,
        device=torch.device("cuda", torch.cuda.current_device()),
        sharders=[sharder],
        plan=plan,
    )


def rw_index(ec: ShardedDynamicEmbeddingCollection) -> int:
    """Position of the row-wise sharding (its lookup, dists and context)."""
    return list(ec._sharding_type_to_sharding).index(ShardingType.ROW_WISE.value)


def rw_sharding(ec: ShardedDynamicEmbeddingCollection):
    return list(ec._sharding_type_to_sharding.values())[rw_index(ec)]


def hier_of(ec: ShardedDynamicEmbeddingCollection) -> Optional[HierSharding]:
    """The row-wise sharding's hierarchical dist (None: TorchRec's)."""
    return getattr(rw_sharding(ec), "_hier", None)


def sharded_ec(model: DistributedModelParallel) -> ShardedDynamicEmbeddingCollection:
    ec = next(
        module
        for module in model.modules()
        if isinstance(module, ShardedDynamicEmbeddingCollection)
    )
    if hier_of(ec) is not None:
        SHARDINGS.append(weakref.ref(hier_of(ec)))
    return ec


def bucketized_owners(
    keys: torch.Tensor, world_size: int, dist_type: str, block_size: int
) -> torch.Tensor:
    """The rank TorchRec's path in DynamicEmb sends each key to: DynamicEmb's
    bucketize run on the keys as given (one sample of one feature), so by the
    rule of their own width, with the block size converted to their dtype as
    the input dist converts it. A key's owner is the bucket its place in the
    bucketized keys falls in."""
    if keys.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=keys.device)
    lengths, _, _, _, places = block_bucketize_sparse_features(
        torch.tensor([keys.numel()], dtype=torch.int64, device=keys.device),
        keys,
        bucketize_pos=False,
        sequence=True,
        dist_type_per_feature=torch.tensor(
            [DIST_TYPE_CODES[dist_type]], dtype=torch.int32, device=keys.device
        ),
        block_sizes=torch.tensor([block_size], device=keys.device).to(keys.dtype),
        my_size=world_size,
        weights=None,
        batch_size_per_feature=None,
        max_B=-1,
    )
    return torch.searchsorted(torch.cumsum(lengths, 0), places.long(), right=True)


def owner_rule(case: Case, ec: Optional[ShardedDynamicEmbeddingCollection]) -> Callable:
    """``owner_of(keys, feature)`` of the case's row-wise sharding: the rank
    its tables' ``dist_type`` gives each key on TorchRec's path
    (:func:`bucketized_owners`; ``continuous``: by the blocks of rows of the
    sizes TorchRec's input dist takes, from the sharding of ``ec``, the
    case's EC)."""
    world_size = dist.get_world_size()
    blocks: Dict[str, int] = {}
    if case.dist_type == "continuous":
        sharding = rw_sharding(ec)
        blocks = {
            feature: -(-hash_size // world_size)
            for feature, hash_size in zip(
                sharding.feature_names(), sharding._get_feature_hash_sizes()
            )
        }
    return lambda keys, feature: bucketized_owners(
        keys, world_size, case.dist_type, blocks.get(feature, 1)
    )


def capacities_for(
    case: Case,
    calls: Sequence[KeyedJaggedTensor],
    local_size: int,
    ec: Optional[ShardedDynamicEmbeddingCollection] = None,
) -> Dict:
    """The case's ``output_dist_capacity`` for its ``calls`` (this rank's KJT
    of each; ``ec`` gives the blocks of ``continuous`` keys): the most any
    rank needs (``hier_needs``, collective), times 1.25 and at least 4096
    more."""
    tables = {
        feature: table for table, feats in enumerate(case.tables) for feature in feats
    }
    needs = hier_needs(
        calls,
        tables,
        owner_rule(case, ec),
        local_size,
        index_dedup=case.dedup,
        node_dedup=case.node_dedup and not case.clip,
    )
    return hier_capacities(needs)


@contextlib.contextmanager
def environ(name: str, value: Optional[object]):
    """Environment variable ``name`` set to ``value`` meanwhile (None: as it
    is): the hierarchical dist reads its knobs when the model is sharded."""
    saved = os.environ.get(name)
    if value is not None:
        os.environ[name] = str(value)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved


def fused_model(
    case: Case, world_size: int, local_size: int, calls: Sequence[KeyedJaggedTensor]
) -> DistributedModelParallel:
    """The case's EC on the hierarchical dist alone, with the capacities its
    ``calls`` need."""
    capacity = capacities_for(case, calls, local_size)
    return build_model(case, "fused", world_size, local_size, capacity)


class Probe:
    """What a model's row-wise lookup saw in the last iteration: its KJT and
    the gradient of its output (a copy; with ``hold``, also that gradient
    itself: on the hierarchical path, ``lookup_grad``, a view of
    ``lookup_rows``)."""

    def __init__(self, ec: ShardedDynamicEmbeddingCollection, dim: int) -> None:
        self.kjt: Optional[KeyedJaggedTensor] = None
        self.grad: Optional[torch.Tensor] = None
        self.hold = False
        self.held: Optional[torch.Tensor] = None
        lookup = ec._lookups[rw_index(ec)]
        orig = lookup.forward

        def forward(features):
            out = orig(features)
            self.kjt = features
            if out.requires_grad:
                out.register_hook(keep)
            return out

        def keep(grad):
            self.grad = grad.detach().view(-1, dim).clone()
            if self.hold:
                self.held = grad

        lookup.forward = forward


def local_batch(case: Case, rank: int) -> int:
    return case.batch * (case.skew if rank == 0 else 1) + (
        7 * rank if case.uneven else 0
    )


def batch_of(
    case: Case, case_idx: int, iteration: int, rank: int, world_size: int
) -> KeyedJaggedTensor:
    gen = torch.Generator().manual_seed(1_000_003 * case_idx + 1009 * iteration + rank)
    inputs = case.inputs
    batch_size = local_batch(case, rank)
    lengths = torch.randint(
        0, case.max_len + 1, (len(inputs), batch_size), generator=gen
    )
    for feature in case.empty + (case.empty_last if rank == world_size - 1 else ()):
        lengths[inputs.index(feature)] = 0
    num_keys = int(lengths.sum())
    if case.keys == "pow-law":
        probs = 1.0 / torch.arange(1, case.key_range + 1, dtype=torch.float64) ** 1.2
        keys = torch.multinomial(probs, num_keys, replacement=True, generator=gen)
    else:
        keys = torch.randint(0, case.key_range, (num_keys,), generator=gen)
        if case.keys == "hot":
            keys[torch.rand(num_keys, generator=gen) < 1 / 16] = 7
    local_size = world_size // NODES
    if case.avoid and NODES > 1 and rank < local_size:
        # roundrobin: a key's owner is key % world_size; the last node's go to
        # the node before
        keys -= local_size * (keys % world_size >= world_size - local_size).long()
    index_dtype = torch.int32 if case.int32 else torch.int64
    return KeyedJaggedTensor(
        keys=inputs,
        values=keys.to(index_dtype).cuda(),
        lengths=lengths.view(-1).to(index_dtype).cuda(),
    )


def output_grads(
    case: Case, case_idx: int, iteration: int, rank: int, out
) -> Dict[str, torch.Tensor]:
    """Gradient of every output row, different per row so a row sent to the
    wrong key shows: multiples of 1/8, in [-1/4, 1/4] for 16-bit rows (exact
    there, and so are their sums), in [-1, 1] for fp32 ones. The latter include
    5/8 and 7/8, whose fp32 product with 1/6 rounds otherwise than their
    quotient by 6: at world_size = 6 the per-row check tells TorchRec's
    scaling (a multiplication by 1/world_size) from a division."""
    gen = torch.Generator().manual_seed(7_000_003 * case_idx + 1013 * iteration + rank)
    limit = 8 if case.dtype == "fp32" else 2
    return {
        feature: (
            torch.randint(-limit, limit + 1, out[feature].values().shape, generator=gen)
            / 8.0
        ).cuda()
        for feature in case.features
    }


def jagged_input(ec, ctx) -> KeyedJaggedTensor:
    """The KJT construct_jagged_tensors reads for the row-wise sharding."""
    rw = rw_index(ec)
    if ec._use_index_dedup:
        return ctx.input_features[rw]
    return ctx.sharding_contexts[rw].features_before_input_dist


class Run(NamedTuple):
    out: Dict
    ctx: object
    syncs: int  # host syncs in the input dist (-1: not counted)
    totals: bool  # the KJT construct_jagged_tensors reads had length_per_key
    plan: object  # the call's plan on the op (None: TorchRec's dists)


def counted(fn: Callable):
    """fn() with the host syncs the profiler records of it counted."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("counted_window"):
            result = fn()
    events = prof.events()
    window = next(
        event for event in events if event.name == "counted_window"
    ).time_range
    syncs = sum(
        event.name in SYNC_CALLS
        and window.start <= event.time_range.start <= window.end
        for event in events
    )
    return result, syncs


def run_model(ec, kjt, sync_count: bool) -> Run:
    """One forward: input dist and both waits (with the host syncs counted,
    if asked), then lookup and output dist."""
    ctx = ec.create_context()
    syncs = -1

    def input_dist():
        return ec.input_dist(ctx, kjt).wait().wait()

    if sync_count:
        dist_input, syncs = counted(input_dist)
    else:
        dist_input = input_dist()
    totals = jagged_input(ec, ctx)._length_per_key is not None
    plan = getattr(ctx.sharding_contexts[rw_index(ec)], "plan", None)
    out = ec.compute_and_output_dist(ctx, dist_input).wait()
    return Run(out, ctx, syncs, totals, plan)


def totals_right(kjt: KeyedJaggedTensor) -> bool:
    """The per-feature totals kept on a KJT are those of its lengths."""
    length_per_key = kjt.lengths().view(len(kjt.keys()), -1).sum(dim=1).tolist()
    offsets = [0] + torch.tensor(length_per_key).cumsum(0).tolist()
    return kjt._length_per_key == length_per_key and kjt._offset_per_key in (
        None,
        offsets,
    )


def per_key(case, kjt: KeyedJaggedTensor, grad: torch.Tensor):
    """(table, key) ids and their summed gradient rows (float64)."""
    table_of = case.table_of
    tables = torch.repeat_interleave(
        torch.tensor([table_of[feature] for feature in kjt.keys()], device=grad.device),
        torch.tensor(kjt.length_per_key(), device=grad.device),
    )
    ids = tables * (1 << 40) + kjt.values().long()
    uniq, inv = torch.unique(ids, return_inverse=True)
    sums = torch.zeros(
        uniq.numel(), grad.size(1), dtype=torch.float64, device=grad.device
    )
    sums.index_add_(0, inv, grad.double())
    return uniq, sums


def close(case: Case, want: torch.Tensor, got: torch.Tensor) -> bool:
    want, got = want.double(), got.double()
    if case.dtype == "fp32":
        return torch.allclose(got, want, rtol=1e-5, atol=1e-5)
    scale = (
        want.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        if want.dim() == 2
        else want.abs().max()
    )
    return bool(((want - got).abs() <= 2e-2 * scale).all())


def tables_of(ec) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Every row of the row-wise sharding's DynamicEmb tables."""
    out = {}
    for module in ec._lookups[rw_index(ec)]._emb_modules:
        emb = module._emb_module
        if not isinstance(emb, BatchedDynamicEmbeddingTablesV2):
            continue
        for name in emb._table_names:
            keys, values = emb.export_keys_values(name, torch.device("cuda"))
            order = torch.argsort(keys)
            out[name] = (keys[order], values[order])
    return out


def same_outputs(case: Case, ref_out: Dict, hier_out: Dict, exact: bool) -> bool:
    for feature in case.features:
        want, got = ref_out[feature], hier_out[feature]
        if not torch.equal(want.lengths(), got.lengths()):
            return False
        if not (
            torch.equal(want.values(), got.values())
            if exact
            else close(case, want.values(), got.values())
        ):
            return False
        if case.indices and not (
            want.weights().dtype == got.weights().dtype
            and torch.equal(want.weights(), got.weights())
        ):
            return False
    return True


class Pair:
    """A case built twice: on TorchRec's dists (``ref``) and on the
    hierarchical dist (``hier``, its sharding's ``state``), with a probe on
    each lookup. The hierarchical dist's capacities are what ``calls`` (this
    rank's KJT of each call the pair will run) need (:func:`capacities_for`),
    and its op is built before the first call."""

    def __init__(
        self,
        case: Case,
        case_idx: int,
        world_size: int,
        local_size: int,
        calls: Sequence[KeyedJaggedTensor],
    ) -> None:
        torch.manual_seed(case_idx)
        self.ref_model = build_model(case, "torchrec", world_size, local_size)
        self.ref = sharded_ec(self.ref_model)
        capacity = capacities_for(case, calls, local_size, self.ref)
        torch.manual_seed(case_idx)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.hier_model = build_model(
                case, "fused", world_size, local_size, capacity
            )
        self.fell_back = any(
            "does not apply here" in str(warning.message) for warning in caught
        )
        self.hier = sharded_ec(self.hier_model)
        self.state = hier_of(self.hier)
        # The first input_dist() makes the EC's input dists and builds the op
        # (collective): a part 1 nobody waits for, before the calls.
        self.hier.input_dist(self.hier.create_context(), calls[0])
        self.ref_probe = Probe(self.ref, case.dim)
        self.hier_probe = Probe(self.hier, case.dim)


def loss_of(
    case: Case, case_idx: int, iteration: int, rank: int, out: Dict
) -> torch.Tensor:
    """The loss whose gradient of every output row is ``output_grads``."""
    grads = output_grads(case, case_idx, iteration, rank, out)
    return sum(
        (out[feature].values().float() * grads[feature]).sum()
        for feature in case.features
    )


def same_keys_grads(case: Case, pair: Pair) -> bool:
    """The gradient of the lookup output the last backward handed each model's
    lookup: summed per (table, key), and per row on one node without index
    dedup (across nodes the owner has a row per node-unique key)."""
    ref_uniq, ref_sum = per_key(case, pair.ref_probe.kjt, pair.ref_probe.grad)
    hier_uniq, hier_sum = per_key(case, pair.hier_probe.kjt, pair.hier_probe.grad)
    ok = torch.equal(ref_uniq, hier_uniq) and close(case, ref_sum, hier_sum)
    if not case.dedup and NODES == 1:
        ok = ok and torch.equal(pair.ref_probe.grad, pair.hier_probe.grad)
    return ok


def same_tables(case: Case, pair: Pair, world_size: int) -> Tuple[bool, bool]:
    """Every table row of both models: equal (bit for bit when the case is
    exact at this world size, else within the tolerances), and whether bit
    for bit."""
    exact = case.exact(world_size)
    ref_tables, hier_tables = tables_of(pair.ref), tables_of(pair.hier)
    ok = ref_tables.keys() == hier_tables.keys()
    bitwise = True
    for name in ref_tables:
        (ref_keys, ref_values), (hier_keys, hier_values) = (
            ref_tables[name],
            hier_tables[name],
        )
        same = (
            torch.equal(ref_keys, hier_keys) and ref_values.shape == hier_values.shape
        )
        ok = (
            ok
            and same
            and (
                torch.equal(ref_values, hier_values)
                if exact
                else close(case, ref_values, hier_values)
            )
        )
        bitwise = bitwise and same and torch.equal(ref_values, hier_values)
    return ok, bitwise


def frequencies_of(ec) -> Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]]:
    """Per table of the row-wise sharding, by key: its LFU scores (an LFU
    table), and its admission counter's counts (a table with an admission
    strategy)."""
    out = {}
    dev = torch.device("cuda")
    for module in ec._lookups[rw_index(ec)]._emb_modules:
        emb = module._emb_module
        if not isinstance(emb, BatchedDynamicEmbeddingTablesV2):
            continue
        lfu = counts_frequencies(emb._dynamicemb_options[0])
        for table_id, name in enumerate(emb._table_names):
            parts = {}
            if emb._admitter is not None:
                counter = emb._admission_counter.table_
                parts["admit"] = [
                    (keys, scores["counter"])
                    for keys, scores, _ in counter._batched_export_keys_scores(
                        ["counter"], dev, table_id
                    )
                ]
            if lfu:
                parts["lfu"] = [
                    (keys, scores)
                    for keys, _, _, scores in emb._storage.export_keys_values(
                        dev, 65536, table_id
                    )
                ]
            for kind, got in parts.items():
                keys = (
                    torch.cat([part for part, _ in got])
                    if got
                    else torch.empty(0, device=dev)
                )
                values = (
                    torch.cat([scores for _, scores in got])
                    if got
                    else torch.empty(0, device=dev)
                )
                order = torch.argsort(keys)
                out[(kind, name)] = (keys[order], values[order])
    return out


def same_frequencies(pair: Pair) -> bool:
    """The LFU scores and the admission counts of both models' tables: the
    hierarchical owner counts every occurrence TorchRec's does (across
    nodes, a node-unique key's weight carries its node's, for LFU tables;
    an admission strategy counts a key once per batch on both)."""
    ref_freq, hier_freq = frequencies_of(pair.ref), frequencies_of(pair.hier)
    return ref_freq.keys() == hier_freq.keys() and all(
        torch.equal(ref_freq[key][0], hier_freq[key][0])
        and torch.equal(ref_freq[key][1], hier_freq[key][1])
        for key in ref_freq
    )


def same_everywhere(value) -> bool:
    every = [None] * dist.get_world_size()
    dist.all_gather_object(every, value)
    return all(other == value for other in every)


def check_step(
    case: Case,
    case_idx: int,
    iteration: int,
    rank: int,
    world_size: int,
    pair: Pair,
    ref_out: Dict,
    hier_out: Dict,
    fwd_exact: bool,
) -> Tuple[Dict[str, bool], bool]:
    """After the forwards of one batch on both models: their backwards, and
    then the forward outputs, the backward per (table, key) (and per row on
    one node without index dedup), the tables after the step and their LFU
    scores or admission counts, against TorchRec's, and that no view of
    ``lookup_rows`` outlived the step (lookup outputs, ``lookup_grad``).
    Returns the checks, and whether the tables are bitwise equal."""
    ok_fwd = same_outputs(case, ref_out, hier_out, fwd_exact)
    for out in (ref_out, hier_out):
        loss_of(case, case_idx, iteration, rank, out).backward()
    ok_tab, bitwise = same_tables(case, pair, world_size)
    checks = {
        "fwd": ok_fwd,
        "bwd": same_keys_grads(case, pair),
        "tables": ok_tab,
    }
    if case.lfu or case.admit or case.mixed:
        checks["freq"] = same_frequencies(pair)
    if pair.state is not None and pair.state.takes_hier_dist:
        checks["views"] = (
            hier_output_dist_live_views(pair.state.op) == 0
            and hier_output_dist_lookup_grad_live(pair.state.op) == 0
        )
    return checks, bitwise


def report(name: str, checks: Dict[str, bool], log: Callable, extra: str = "") -> int:
    """Each check, true on every rank or not; logs them, and returns the
    failures (0 or 1)."""
    flags = torch.tensor(
        [int(bool(ok)) for ok in checks.values()], dtype=torch.int32, device="cuda"
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    flags = flags.tolist()
    good = min(flags) == 1
    log(
        f"{name}: "
        + " ".join(f"{label}={flag}" for label, flag in zip(checks, flags))
        + extra
        + f" [{'ok' if good else 'FAIL'}]"
    )
    return int(not good)


def run_case(
    case_idx: int,
    case: Case,
    rank: int,
    world_size: int,
    local_size: int,
    log: Callable,
) -> int:
    calls = [
        batch_of(case, case_idx, iteration, rank, world_size) for iteration in range(3)
    ]
    pair = Pair(case, case_idx, world_size, local_size, calls)
    ref, hier, state = pair.ref, pair.hier, pair.state
    exact = case.exact(world_size)
    failures = 0
    for iteration in range(3):
        if iteration == 1:
            # A part 1 nobody waits for, and a first wait whose result is
            # dropped (TorchRec's prefetch and eval pipelines do both).
            hier.input_dist(
                hier.create_context(),
                batch_of(case, case_idx, iteration, rank, world_size),
            )
            hier.input_dist(
                hier.create_context(),
                batch_of(case, case_idx, iteration, rank, world_size),
            ).wait()
        ref_run = run_model(
            ref, batch_of(case, case_idx, iteration, rank, world_size), False
        )
        hier_run = run_model(
            hier,
            batch_of(case, case_idx, iteration, rank, world_size),
            sync_count=True,
        )
        checks, bitwise_tab = check_step(
            case,
            case_idx,
            iteration,
            rank,
            world_size,
            pair,
            ref_run.out,
            hier_run.out,
            iteration == 0 or exact,
        )
        ok_sync = case.syncs is None or hier_run.syncs == case.syncs
        ok_totals = (
            ref_run.totals
            and hier_run.totals
            and totals_right(jagged_input(hier, hier_run.ctx))
        )
        groups = len(hier._lookups[rw_index(hier)]._emb_modules)
        ok_path = pair.fell_back == bool(case.plain) and (
            state is None
            if case.plain
            else state is not None
            and state.takes_hier_dist is True
            and groups == (2 if case.groups or case.mixed else 1)
        )
        ok_op = (hier_run.plan is None) == bool(case.plain) and (
            state is None or state.fallbacks == 0
        )
        ok_eval = True
        if iteration == 1:
            # Eval runs call by call on TorchRec's dists; then training goes on.
            for model in (pair.ref_model, pair.hier_model):
                model.eval()
            with torch.no_grad():
                eval_ref = run_model(
                    ref, batch_of(case, case_idx, iteration, rank, world_size), False
                )
                eval_hier = run_model(
                    hier, batch_of(case, case_idx, iteration, rank, world_size), False
                )
            ok_eval = (
                same_outputs(case, eval_ref.out, eval_hier.out, exact)
                and eval_hier.plan is None
            )
            for model in (pair.ref_model, pair.hier_model):
                model.train()
        checks.update(
            op=ok_op, syncs=ok_sync, totals=ok_totals, path=ok_path, eval=ok_eval
        )
        flags = torch.tensor(
            [bool(ok) for ok in checks.values()] + [bitwise_tab],
            dtype=torch.int32,
            device="cuda",
        )
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        flags = flags.tolist()
        good = min(flags[: len(checks)]) == 1
        failures += int(not good)
        log(
            f"case {case_idx} '{case.name}' iteration {iteration}: "
            + " ".join(f"{label}={flag}" for label, flag in zip(checks, flags))
            + f" tables_bitwise={flags[-1]} [{'ok' if good else 'FAIL'}]"
        )
    torch.cuda.synchronize()
    dist.barrier()
    return failures


# ── the windows at the given capacities ──────────────────────────────────────

# Uniform keys over 4 features: an owner looks up about 4 * batch * max_len / 2
# keys of a call: 512, 32K, 196K, 524K and 524K keys per call here, under the
# capacities of the largest.
WINDOWS = Case("windows", key_range=200_000)
WINDOWS_SIZES = ((64, 4), (1024, 16), (4096, 24), (8192, 32), (8192, 32))


def trace_matches_build(op) -> bool:
    """The op's trace APIs match the build: without DYNAMICEMB_HIER_TRACE=1
    empty tensors and no launches; with it, after calls, the last call's row
    of the in channel's step times (its first step to its last, in order)
    and each direction's last kernel row (its start stamp)."""
    trace, steps = op.trace(), op.in_trace()
    if not hier_output_dist_traced():
        return (
            trace.numel() == 0 and steps.numel() == 0 and op.trace_launches() == [0, 0]
        )
    launches = op.trace_launches()
    started = all(
        trace[direction, (launches[direction] - 1) % trace.size(1), 0].item() > 0
        for direction in (0, 1)
    )
    row = steps[op.in_calls() % steps.size(0)].tolist()
    return min(launches) > 0 and started and 0 <= row[0] <= row[-1]


def run_windows(rank: int, world_size: int, local_size: int, log: Callable) -> int:
    """Calls of five sizes under one set of capacities, those of the largest:
    each takes the op and matches TorchRec's, and the op keeps the windows it
    made when it was built. Then its trace APIs match the build."""
    case_idx = 100
    cases = [
        replace(WINDOWS, batch=batch_size, max_len=max_len)
        for batch_size, max_len in WINDOWS_SIZES
    ]
    calls = [
        batch_of(case, case_idx, iteration, rank, world_size)
        for iteration, case in enumerate(cases)
    ]
    pair = Pair(WINDOWS, case_idx, world_size, local_size, calls)
    memory = pair.state.op.memory()
    failures = 0
    for iteration, case in enumerate(cases):
        ref_run = run_model(
            pair.ref, batch_of(case, case_idx, iteration, rank, world_size), False
        )
        hier_run = run_model(
            pair.hier, batch_of(case, case_idx, iteration, rank, world_size), False
        )
        checks, _ = check_step(
            case,
            case_idx,
            iteration,
            rank,
            world_size,
            pair,
            ref_run.out,
            hier_run.out,
            iteration == 0 or case.exact(world_size),
        )
        checks["op"] = hier_run.plan is not None
        checks["same windows"] = pair.state.op.memory() == memory
        failures += report(
            f"windows, call {iteration} (batch {case.batch}, len {case.max_len})",
            checks,
            log,
        )
    failures += report(
        f"trace APIs ({'traced' if hier_output_dist_traced() else 'untraced'} build)",
        {"match": trace_matches_build(pair.state.op)},
        log,
    )
    torch.cuda.synchronize()
    dist.barrier()
    return failures


def run_proxy_rows(rank: int, world_size: int, local_size: int, log: Callable) -> int:
    """Across nodes, a proxy rows cap below the rows a round needs: "fused"
    raises the same ValueError on every rank, naming the cap's knob, and
    "auto" warns and takes TorchRec's dists."""
    least_rows = 20 * HEAVY * (NODES - 1)  # min_proxy_rows_for in formats.h
    cap_mb = least_rows * 4 * 32 / 2 / (1 << 20)  # half their fp32 rows of 32 columns
    capacity = dict.fromkeys(OUTPUT_DIST_CAPACITY_NAMES, 1 << 16)
    checks: Dict[str, bool] = {}
    for asked in ("fused", "auto"):
        with environ("DYNAMICEMB_GIN_PROXY_MB", cap_mb):
            hier = HierSharding(
                dist.group.WORLD, 3, 32, torch.float32, capacity, asked=asked
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if asked == "fused":
                msg = expect_error(
                    ValueError, hier.setup, *bucketize_rules(3, world_size), False
                )
                named = "DYNAMICEMB_GIN_PROXY_MB" in msg
                checks["fused raised"] = named and same_everywhere(msg)
            else:
                took = hier.setup(*bucketize_rules(3, world_size), False)
                checks["auto takes TorchRec's dists"] = took is False and any(
                    "using TorchRec" in str(warning.message) for warning in caught
                )
        del hier
    gc.collect()
    return report("a proxy rows cap below a round's rows", checks, log, f": {msg}")


def expect_error(kind, fn: Callable, *args) -> str:
    """fn(*args), which must raise `kind`: its message ("" if it did not)."""
    try:
        fn(*args)
    except kind as err:
        return str(err)
    return ""


def option_error(**kwargs) -> str:
    """The ValueError DynamicEmbTableOptions(**kwargs) raises ("" if none)."""
    return expect_error(ValueError, lambda: DynamicEmbTableOptions(**kwargs))


def run_options(rank: int, world_size: int, local_size: int, log: Callable) -> int:
    """``output_dist`` and ``output_dist_capacity``: DynamicEmbTableOptions
    raises ValueError for ``"fused"`` without a capacity and for an output
    dist it does not know (naming those it knows); tables of one sharding
    that ask for different capacities raise it when the model is sharded;
    ``"auto"`` without a capacity takes TorchRec's dists with a warning, and
    trains as they do."""
    every_capacity = dict.fromkeys(OUTPUT_DIST_CAPACITY_NAMES, 1 << 16)
    unknown = option_error(output_dist="hierarchical", output_dist_capacity=4)
    checks = {
        "fused without a capacity": "output_dist" in option_error(output_dist="fused"),
        "an unknown output dist": all(
            f"'{name}'" in unknown for name in ("torchrec", "fused", "auto")
        ),
    }
    case = Case("options")
    msg = expect_error(
        ValueError,
        build_model,
        case,
        "fused",
        world_size,
        local_size,
        [every_capacity, {**every_capacity, "lookup_keys": 1 << 17}, every_capacity],
    )
    checks["tables that disagree"] = "must agree on output_dist" in msg
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        auto = sharded_ec(build_model(case, "auto", world_size, local_size))
    ref = sharded_ec(build_model(case, "torchrec", world_size, local_size))
    checks["auto without a capacity"] = hier_of(auto) is None and any(
        'output_dist="auto" without output_dist_capacity' in str(warning.message)
        for warning in caught
    )
    auto_run = run_model(auto, batch_of(case, 140, 0, rank, world_size), False)
    ref_run = run_model(ref, batch_of(case, 140, 0, rank, world_size), False)
    checks["auto trains as TorchRec"] = same_outputs(
        case, ref_run.out, auto_run.out, True
    )
    failures = report("options", checks, log, f": {msg}; {unknown}")
    torch.cuda.synchronize()  # before the tables' memory goes with the models
    del auto, ref, auto_run, ref_run
    return failures + gc_models(log)


def run_agreement(rank: int, world_size: int, local_size: int, log: Callable) -> int:
    """Ranks whose arguments differ raise the same ValueError on every rank,
    naming the first rank and argument that differ, and none hangs: an
    environment variable set on one rank (under ``"fused"``), and a capacity
    that differs on one rank (under ``"auto"``). Then an op on the same group
    builds."""
    checks: Dict[str, bool] = {}
    last = world_size - 1
    messages = []
    with environ("DYNAMICEMB_GIN_TIMEOUT_S", 60 if rank == last else None):
        hier = standalone(dist.group.WORLD)
    msg = expect_error(ValueError, hier.setup, *bucketize_rules(3, world_size), False)
    checks["an environment variable on one rank"] = (
        f"rank {last} has timeout_s=60.0" in msg
        and same_everywhere(msg)
        and hier.op is None
    )
    messages.append(msg)
    del hier
    capacity = dict.fromkeys(OUTPUT_DIST_CAPACITY_NAMES, 1 << 16)
    if rank == last:
        capacity["lookup_keys"] = 1 << 15
    with environ("DYNAMICEMB_GIN_PROXY_MB", 16):
        hier = HierSharding(
            dist.group.WORLD, 3, 32, torch.float32, capacity, asked="auto"
        )
    SHARDINGS.append(weakref.ref(hier))
    msg = expect_error(ValueError, hier.setup, *bucketize_rules(3, world_size), False)
    checks["a capacity on one rank"] = (
        f"rank {last} has capacities=" in msg
        and same_everywhere(msg)
        and hier.op is None
    )
    messages.append(msg)
    del hier
    hier = standalone(dist.group.WORLD)
    checks["built after them"] = (
        hier.setup(*bucketize_rules(3, world_size), False) is True
    )
    del hier
    torch.cuda.synchronize()
    dist.barrier()
    return report("op agreement", checks, log, f": {messages}")


def run_construction_failure(
    rank: int, world_size: int, local_size: int, log: Callable
) -> int:
    """A rank that cannot allocate the op's windows when it is built (the
    test hook, on that rank alone), 24 times in a row on an op outside a
    model: every rank warns once, naming that rank, and takes TorchRec's
    dists, and none hangs (every rank ends the op's communicator before it
    goes on)."""
    failing, attempts = world_size - 1, 24
    fell_back = named = 0
    for _ in range(attempts):
        hier = standalone(dist.group.WORLD)
        if rank == failing:
            hier_output_dist_fail_next_construction(fail_alloc=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fell_back += hier.setup(*bucketize_rules(3, world_size), False) is False
        said = [
            str(warning.message)
            for warning in caught
            if "using TorchRec" in str(warning.message)
        ]
        named += (
            len(said) == 1
            and f"rank {failing}: " in said[0]
            and "could not allocate" in said[0]
        )
        del hier
        torch.cuda.synchronize()
        dist.barrier()
    checks = {"every one fell back": fell_back == attempts, "warned": named == attempts}
    return report(f"op construction, {attempts} failures in a row", checks, log)


# ── no sync before the second wait ───────────────────────────────────────────


def run_no_sync(rank: int, world_size: int, local_size: int, log: Callable) -> int:
    """The input dist and its first wait wait for the device nowhere (the
    second wait is the call's one host sync): they run under the sync debug
    mode "error", where c10 raises on an op that syncs, inside C++ too. With
    index dedup, keys with weights, three owner rules, and a call that does
    not fit the capacities (it then takes TorchRec's dists)."""
    failures = 0
    for case_idx, case, capacity in (
        (130, Case("no sync, dedup", dedup=True), 1 << 16),
        (
            131,
            Case("no sync, pow-law dedup lfu", keys="pow-law", dedup=True, lfu=True),
            1 << 16,
        ),
        (
            132,
            Case(
                "no sync, three owner rules",
                rules=("roundrobin", "hash_roundrobin", "continuous"),
            ),
            1 << 16,
        ),
        (133, Case("no sync, a call that does not fit"), 64),
    ):
        ec = sharded_ec(build_model(case, "fused", world_size, local_size, capacity))
        kjt = batch_of(case, case_idx, 0, rank, world_size)
        # The first input_dist() builds the op (collective).
        ec.input_dist(ec.create_context(), kjt)
        ctx = ec.create_context()
        torch.cuda.set_sync_debug_mode("error")
        try:
            first = ec.input_dist(ctx, kjt).wait()
        finally:
            torch.cuda.set_sync_debug_mode("default")
        out = ec.compute_and_output_dist(ctx, first.wait()).wait()
        on_op = ctx.sharding_contexts[rw_index(ec)].plan is not None
        fits = capacity > 64
        checks = {
            "path": on_op == fits and hier_of(ec).fallbacks == int(not fits),
            "rows": all(feature in out for feature in case.features),
        }
        failures += report(case.name, checks, log)
        del ec, first, out
        failures += gc_models(log)
    return failures


# ── the output dist's kernels in every order, lookup_grad guard, determinism ──


def run_orders(rank: int, world_size: int, local_size: int, log: Callable) -> int:
    """The output dist's kernels in each order a training loop meets, against
    TorchRec's dists on the same batches, each followed by a checked step: a
    no-grad forward; two forwards, then one backward of both or their
    backwards in order; a second backward of one graph (``retain_graph``);
    and ``lookup_grad`` kept alive, where the next backward and the next
    lookup raise the same RuntimeError on every rank. Without index dedup,
    and with index dedup and heavy rows."""
    failures = 0
    for case_idx, case in (
        (110, Case("orders")),
        (
            111,
            Case(
                "orders, pow-law dedup",
                keys="pow-law",
                dedup=True,
                batch=512,
                max_len=16,
                key_range=20000,
            ),
        ),
    ):
        failures += orders_of(case_idx, case, rank, world_size, local_size, log)
        failures += gc_models(log)
    return failures


def orders_of(
    case_idx: int,
    case: Case,
    rank: int,
    world_size: int,
    local_size: int,
    log: Callable,
) -> int:
    """run_orders for one case (its models and outputs go when it returns)."""
    failures = 0
    calls = [batch_of(case, case_idx, index, rank, world_size) for index in range(14)]
    pair = Pair(case, case_idx, world_size, local_size, calls)
    exact = case.exact(world_size)
    iteration = 0

    def batch(index: int) -> KeyedJaggedTensor:
        return batch_of(case, case_idx, index, rank, world_size)

    def step(name: str) -> int:
        nonlocal iteration
        ref_run = run_model(pair.ref, batch(iteration), False)
        hier_run = run_model(pair.hier, batch(iteration), False)
        checks, _ = check_step(
            case,
            case_idx,
            iteration,
            rank,
            world_size,
            pair,
            ref_run.out,
            hier_run.out,
            iteration == 0 or exact,
        )
        checks["op"] = hier_run.plan is not None
        iteration += 1
        return report(f"{case.name}: step after {name}", checks, log)

    failures += step("setup")

    checks: Dict[str, bool] = {}
    with torch.no_grad():
        ref_run = run_model(pair.ref, batch(iteration), False)
        hier_run = run_model(pair.hier, batch(iteration), False)
    checks["no-grad training forward"] = (
        same_outputs(case, ref_run.out, hier_run.out, exact)
        and hier_run.plan is not None
    )
    failures += report(f"{case.name}: forward after forward", checks, log)
    failures += step("forward after forward")

    first, second = iteration, iteration + 1
    runs = {}
    for side, ec in (("ref", pair.ref), ("hier", pair.hier)):
        run_first, run_second = run_model(ec, batch(first), False), run_model(
            ec, batch(second), False
        )
        runs[side] = (run_first.out, run_second.out)
        (
            loss_of(case, case_idx, first, rank, run_first.out)
            + loss_of(case, case_idx, second, rank, run_second.out)
        ).backward()
    checks = {
        "first fwd": same_outputs(case, runs["ref"][0], runs["hier"][0], exact),
        "second fwd": same_outputs(case, runs["ref"][1], runs["hier"][1], exact),
        "tables": same_tables(case, pair, world_size)[0],
    }
    failures += report(f"{case.name}: two forwards, one backward", checks, log)
    iteration += 2
    failures += step("two forwards, one backward")

    losses = {}
    for side, ec in (("ref", pair.ref), ("hier", pair.hier)):
        run = run_model(ec, batch(iteration), False)
        losses[side] = loss_of(case, case_idx, iteration, rank, run.out)
        losses[side].backward(retain_graph=True)
        losses[side].backward()
    checks = {
        "second bwd": same_keys_grads(case, pair),
        "tables": same_tables(case, pair, world_size)[0],
    }
    del losses
    failures += report(f"{case.name}: a second backward (retain_graph)", checks, log)
    iteration += 1
    failures += step("a second backward")

    first, second = iteration, iteration + 1
    runs = {}
    for side, ec in (("ref", pair.ref), ("hier", pair.hier)):
        run_first, run_second = run_model(ec, batch(first), False), run_model(
            ec, batch(second), False
        )
        runs[side] = (run_first.out, run_second.out)
        loss_of(case, case_idx, first, rank, run_first.out).backward()
        loss_of(case, case_idx, second, rank, run_second.out).backward()
    checks = {
        "first fwd": same_outputs(case, runs["ref"][0], runs["hier"][0], exact),
        "second fwd": same_outputs(case, runs["ref"][1], runs["hier"][1], exact),
        "second bwd": same_keys_grads(case, pair),
        "tables": same_tables(case, pair, world_size)[0],
    }
    failures += report(
        f"{case.name}: two forwards, then their backwards in order", checks, log
    )
    iteration += 2
    failures += step("two forwards, then their backwards")

    pair.hier_probe.hold = True
    losses = {}
    for side, ec in (("ref", pair.ref), ("hier", pair.hier)):
        run = run_model(ec, batch(iteration), False)
        losses[side] = loss_of(case, case_idx, iteration, rank, run.out)
        losses[side].backward(retain_graph=True)
    held = hier_output_dist_lookup_grad_live(pair.state.op) == 1
    msg_bwd = expect_error(RuntimeError, losses["hier"].backward)
    msg_fwd = expect_error(
        RuntimeError, run_model, pair.hier, batch(iteration + 1), False
    )
    pair.hier_probe.hold, pair.hier_probe.held = False, None
    del losses
    checks = {
        "held": held,
        "backward raised": "still alive" in msg_bwd,
        "lookup raised": "still alive" in msg_fwd,
        "released": hier_output_dist_lookup_grad_live(pair.state.op) == 0,
    }
    failures += report(
        f"{case.name}: lookup gradient kept alive", checks, log, f": {msg_fwd}"
    )
    iteration += 2
    failures += step("lookup gradient kept alive")
    return failures


def by_cell(case: Case, kjt: KeyedJaggedTensor, grad: torch.Tensor) -> torch.Tensor:
    """The rows of the gradient of an owner's lookup output, ordered by
    (table, key, source column), and in KJT order within one (each comes
    once under index dedup, and across nodes in another node's column)."""
    num_features, num_sources = len(kjt.keys()), kjt.stride()
    per_cell = kjt.lengths().view(-1)
    feature_tables = torch.tensor(
        [case.table_of[feature] for feature in kjt.keys()], device=grad.device
    )
    tables = torch.repeat_interleave(
        feature_tables.repeat_interleave(num_sources), per_cell
    )
    cols = torch.repeat_interleave(
        torch.arange(num_sources, device=grad.device).repeat(num_features), per_cell
    )
    ids = (tables * (1 << 40) + kjt.values().long()) * num_sources + cols
    return grad[torch.argsort(ids, stable=True)]


def run_determinism(rank: int, world_size: int, local_size: int, log: Callable) -> int:
    """The backward is the same bit for bit from run to run: the same batch
    twice, with random fp32 output gradients (whose sums depend on their
    order), hands each owner's lookup the same gradient for every (table,
    key, source). Heavy index-dedup rows included; across nodes also without
    index dedup."""
    failures = 0
    case_idx = 108
    heavy_case = replace(HEAVY_ROWS, name="determinism")
    cases = [heavy_case]
    if NODES > 1:
        # without index dedup a requester's heavy groups hold hundreds of rows
        cases.append(
            replace(heavy_case, name="determinism, no index dedup", dedup=False)
        )
    for case in cases:
        torch.manual_seed(case_idx)
        calls = [batch_of(case, case_idx, 0, rank, world_size)]
        ec = sharded_ec(fused_model(case, world_size, local_size, calls))
        probe = Probe(ec, case.dim)
        grads, on_op = [], True
        for _ in range(2):
            run = run_model(ec, batch_of(case, case_idx, 0, rank, world_size), False)
            on_op = on_op and run.plan is not None
            gen = torch.Generator().manual_seed(rank)  # the same both times
            values = [run.out[feature].values().float() for feature in case.features]
            loss = sum(
                (value * torch.randn(value.shape, generator=gen).cuda()).sum()
                for value in values
            )
            loss.backward()
            grads.append(by_cell(case, probe.kjt, probe.grad))
        checks = {
            "op": on_op,
            "same gradient": grads[0].shape == grads[1].shape and torch.equal(*grads),
        }
        failures += report(
            f"backward bit for bit run to run ({case.name})", checks, log
        )
        del ec, probe, run, loss, values
        failures += gc_models(log)
    return failures


@contextlib.contextmanager
def fd2_into(out: List[str]):
    """Collects what the process writes to file descriptor 2 meanwhile (what
    the C++ teardown prints included)."""
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)
            out.append(tmp.read().decode(errors="replace"))


def run_teardown(rank: int, world_size: int, local_size: int, log: Callable) -> int:
    """The teardown's exit round, on an op of its own (built, no call). A
    rank that drops its hierarchical dist 3 s after the others is waited
    for by its node peers and (across nodes) its rail peers: nobody leaks,
    and their drop lasts those 3 s. One that keeps it past the limit of the
    wait (here 2 s, the timeout) is not: its node and rail peers warn, keep
    their windows and go on, and the late one frees its own when it drops
    it. The other ranks wait for nobody late."""
    failures = 0
    last = rank == world_size - 1
    # a node peer of the late one
    node_peer = rank // local_size == (world_size - 1) // local_size and not last
    # a rail peer of it
    rail_peer = rank % local_size == (world_size - 1) % local_size and not last
    peer = node_peer or rail_peer
    for name, timeout_s, late in (
        ("a peer drops it 3 s late", 20.0, 3.0),
        ("a peer keeps it past the limit", 2.0, 6.0),
    ):
        with environ("DYNAMICEMB_GIN_TIMEOUT_S", timeout_s):
            hier = standalone(dist.group.WORLD)
        ok = hier.setup(*bucketize_rules(3, world_size), False)
        dist.barrier()
        if last:
            time.sleep(late)
        err: List[str] = []
        with fd2_into(err):
            start = time.monotonic()
            del hier
            gc.collect()
            took = time.monotonic() - start
        kept = "[hier_output_dist]" in err[0]
        if late < timeout_s:
            ok = ok and not kept and (not peer or took > late - 1.0)
        else:
            ok = ok and kept == peer and (not peer or took < late)
        good = all_true(ok)
        failures += int(not good)
        log(
            f"teardown, {name}: rank 0 dropped it in {took:.1f} s, kept its windows:"
            f" {kept} [{'ok' if good else 'FAIL'}]"
        )
    return failures


def run_teardown_credits(
    rank: int, world_size: int, local_size: int, log: Callable
) -> int:
    """Every rank drops its model right after a forward, whose out kernel
    sent the credits for the next kernel at its end (across nodes, to its
    rail peers, which nobody waits for): the teardown's exit round still
    frees everything, without a warning and without a hang."""
    case = Case("teardown with credits out")
    calls = [batch_of(case, 121, 0, rank, world_size)]
    model = fused_model(case, world_size, local_size, calls)
    run = run_model(sharded_ec(model), batch_of(case, 121, 0, rank, world_size), False)
    on_op = run.plan is not None
    torch.cuda.synchronize()
    err: List[str] = []
    with fd2_into(err):
        del run, model
        gc.collect()
    good = all_true(
        on_op
        and "[hier_output_dist]" not in err[0]
        and all(ref() is None for ref in SHARDINGS)
    )
    log(f"teardown, right after a forward: [{'ok' if good else 'FAIL'}]")
    return int(not good)


def gc_models(log: Callable) -> int:
    """Drop a case's models on every rank at the same point (the hierarchical
    dist's teardown waits for its node peers'), and check it ran: no
    HierSharding is left."""
    torch.cuda.synchronize()
    dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()
    good = all_true(all(ref() is None for ref in SHARDINGS))
    if not good:
        log("teardown: a HierSharding outlived its model [FAIL]")
    return int(not good)


def main() -> None:
    global NODES
    rank, world_size, _, local_size = init_distributed()
    NODES = world_size // local_size

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    good = all_true(check_group_shapes(rank, world_size))
    failures = int(not good)
    log(f"op sharding groups: [{'ok' if good else 'FAIL'}]")
    failures += gc_models(log)
    failures += run_agreement(rank, world_size, local_size, log)
    failures += gc_models(log)
    failures += run_options(rank, world_size, local_size, log)
    for case_idx, case in enumerate(CASES):
        failures += run_case(case_idx, case, rank, world_size, local_size, log)
        failures += gc_models(log)
    failures += run_no_sync(rank, world_size, local_size, log)
    failures += run_orders(rank, world_size, local_size, log)
    for run in (
        run_determinism,
        run_windows,
        run_construction_failure,
    ):
        failures += run(rank, world_size, local_size, log)
        failures += gc_models(log)
    if NODES > 1:
        failures += run_proxy_rows(rank, world_size, local_size, log)
    failures += run_teardown_credits(rank, world_size, local_size, log)
    failures += run_teardown(rank, world_size, local_size, log)
    dist.barrier()
    log("ALL PASSED" if failures == 0 else f"{failures} FAILED")
    dist.destroy_process_group()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
