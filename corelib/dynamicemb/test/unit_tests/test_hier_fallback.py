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
"""A call over the hierarchical dist's capacities (``output_dist="fused"``)
runs on TorchRec's dists, on every rank alike.

Each case trains three models on the same batches: on TorchRec's dists, on
the hierarchical dist with ample capacities (whose calls give the needs),
and on the hierarchical dist with one capacity that the case's large calls
do not fit. The last one must give TorchRec's outputs, bit for bit at a
power-of-two world size. Its large calls alone take TorchRec's dists, on
every rank, and the sharding counts them and warns. The ``placement`` cases
also check that each table holds exactly the keys DynamicEmb's bucketize
gives its rank.

Run it from ``corelib/dynamicemb``, one process per GPU (on one node, or
across nodes under Slurm)::

    torchrun --nproc_per_node 8 test/unit_tests/test_hier_fallback.py
    srun --ntasks-per-node 8 python test/unit_tests/test_hier_fallback.py
"""

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    get_table_value_bytes,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.dynamicemb_config import OUTPUT_DIST_CAPACITY_NAMES
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingCollectionSharder, quiesce
from dynamicemb.shard.embedding import ShardedDynamicEmbeddingCollection
from dynamicemb.shard.hier_output_dist import RwSequenceHierEmbeddingDist
from fbgemm_gpu.split_embedding_configs import SparseType
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.types import BoundsCheckMode, ShardingType
from torchrec.modules.embedding_configs import DataType

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmark")
)
from benchmark_utils import init_distributed  # noqa: E402

AMPLE = 1 << 20  # entries: far above every batch here
DIM = 32
NUM_EMBEDDINGS = 1 << 20
TABLES = (("f0", "f1"), ("f2",))  # f0 and f1 share table t0
FEATURES = [feature for table in TABLES for feature in table]
BESIDE = "g0"  # the feature of the table-wise TorchRec table beside the sharding
SMALL, LARGE = (64, 4), (512, 16)  # (samples, longest sequence) of a batch


@dataclass
class Case:
    name: str
    capacity: str  # the tight one: a name of OUTPUT_DIST_CAPACITY_NAMES, or "int"
    # the calls, "s" small and "l" large (those do not fit the tight capacity)
    schedule: str = "sls"
    dedup: bool = False  # index dedup
    hot: float = 0.0  # share of the keys drawn from 4 hot keys (heavy rows)
    lfu: bool = False  # LFU scores (frequency counts as weights)
    int32: bool = False  # KJT values and lengths
    # int32 keys of placement_keys(), and the checks of where they went
    # (check_placement)
    placement: bool = False
    dist_type: str = "roundrobin"  # the tables' key -> rank rule
    vbe: bool = False  # variable batch size per feature
    groups: bool = False  # the last table in a lookup group of its own
    beside: bool = False  # a table-wise TorchRec table beside the sharding
    # "plain": each call's forward, then its backward; "interleaved": two
    # calls' forwards, then their backwards in order
    order: str = "plain"
    multi_node: bool = False  # across nodes only
    expected: Tuple[int, ...] = field(init=False)  # the calls that fall back

    def __post_init__(self) -> None:
        self.expected = tuple(
            step for step, kind in enumerate(self.schedule) if kind == "l"
        )

    @property
    def features(self) -> List[str]:
        return FEATURES + ([BESIDE] if self.beside else [])

    def size(self, step: int) -> Tuple[int, int]:
        return LARGE if self.schedule[step] == "l" else SMALL


CASES = [
    Case("lookup_keys", "lookup_keys"),
    Case("int", "int"),
    Case("first call", "lookup_keys", schedule="lss"),
    Case("many calls", "lookup_keys", schedule="sllslls"),
    Case("dedup heavy", "lookup_keys", dedup=True, hot=0.5),
    Case("dedup lfu", "lookup_keys", dedup=True, lfu=True),
    Case("lfu counts", "int", lfu=True),
    Case("int32 keys", "lookup_keys", int32=True),
    *(
        Case(
            f"int32 placement, {dist_type}{', dedup' if dedup else ''}",
            "lookup_keys",
            schedule="slss",
            int32=True,
            placement=True,
            dist_type=dist_type,
            dedup=dedup,
        )
        for dist_type in ("roundrobin", "hash_roundrobin", "continuous")
        for dedup in (False, True)
    ),
    Case("variable batch per feature", "lookup_keys", vbe=True),
    Case("two lookup groups", "lookup_keys", groups=True, dedup=True),
    Case("a table-wise table beside", "lookup_keys", beside=True),
    Case(
        "between a forward and its backward",
        "lookup_keys",
        schedule="slssls",
        order="interleaved",
        dedup=True,
        hot=0.5,
    ),
    Case("remote_keys", "remote_keys", multi_node=True),
]


def weighted_sum(out, weight: torch.Tensor, features: List[str]) -> torch.Tensor:
    return sum((out[name].values() * weight).sum() for name in features)


def build_model(
    case: Case, output_dist: str, capacity, world_size: int, local_size: int
) -> DistributedModelParallel:
    def config(name: str, features) -> torchrec.EmbeddingConfig:
        return torchrec.EmbeddingConfig(
            name=name,
            embedding_dim=DIM,
            num_embeddings=NUM_EMBEDDINGS,
            feature_names=list(features),
            data_type=DataType.FP32,
        )

    # the table-wise TorchRec table's rows: the same in every model
    torch.manual_seed(13)
    tables = [config(f"t{i}", features) for i, features in enumerate(TABLES)]
    beside = [config(f"tw_{BESIDE}", (BESIDE,))] if case.beside else []
    ec = torchrec.EmbeddingCollection(
        tables=tables + beside, device=torch.device("meta")
    )
    opt = EmbOptimType.EXACT_SGD
    extra = {"score_strategy": DynamicEmbScoreStrategy.LFU} if case.lfu else {}

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
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.DEBUG
                ),
                output_dist=output_dist,
                output_dist_capacity=capacity,
                dist_type=case.dist_type,
                evicted_item_mode=evicted(table_idx),
                **extra,
            ),
        )
        for table_idx, table in enumerate(tables)
    }
    for table in beside:
        constraints[table.name] = DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.TABLE_WISE.value],
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
        topology=topology, constraints=constraints, batch_size=LARGE[0]
    )
    sharder = DynamicEmbeddingCollectionSharder(
        fused_params={
            "optimizer": opt,
            "learning_rate": 1.0,
            "output_dtype": SparseType.FP32,
        },
        # TorchRec's dists in DynamicEmb take no int32 keys under index dedup
        # (segmented_unique takes int64 keys): the reference then runs
        # without it, which gives the same rows
        use_index_dedup=case.dedup and not (output_dist == "torchrec" and case.int32),
    )
    plan = planner.collective_plan(ec, [sharder], dist.GroupMember.WORLD)
    return DistributedModelParallel(
        module=ec,
        device=torch.device("cuda", torch.cuda.current_device()),
        sharders=[sharder],
        plan=plan,
    )


def batches_of(case: Case, rank: int) -> List[torchrec.KeyedJaggedTensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1009 * (rank + 1) + len(case.name))
    features = case.features
    out = []
    for step in range(len(case.schedule)):
        samples, longest = case.size(step)
        strides = [samples] * len(features)
        if case.vbe:  # each feature its own batch size, and its map to the batch
            strides = [
                max(1, samples - 13 * feature) for feature in range(len(features))
            ]
        lengths = torch.cat(
            [
                torch.randint(
                    1, longest + 1, (stride,), device="cuda", generator=generator
                )
                for stride in strides
            ]
        )
        total = int(lengths.sum())
        keys = torch.randint(
            0, NUM_EMBEDDINGS, (total,), device="cuda", generator=generator
        )
        if case.placement:
            pool = placement_keys()
            keys = pool[
                torch.randint(
                    0, pool.numel(), (total,), device="cuda", generator=generator
                )
            ]
        if case.hot:
            hot = torch.rand(total, device="cuda", generator=generator) < case.hot
            keys = torch.where(hot, keys % 4, keys)
        kwargs = {}
        if case.vbe:
            inverse = torch.stack(
                [
                    torch.randint(
                        0, stride, (samples,), device="cuda", generator=generator
                    )
                    for stride in strides
                ]
            )
            kwargs = dict(
                stride_per_key_per_rank=[[stride] for stride in strides],
                inverse_indices=(features, inverse),
            )
        index_dtype = torch.int32 if case.int32 else torch.int64
        out.append(
            torchrec.KeyedJaggedTensor(
                keys=features,
                values=keys.to(index_dtype),
                lengths=lengths.to(index_dtype),
                **kwargs,
            )
        )
    return out


def sharded_ec(model) -> ShardedDynamicEmbeddingCollection:
    return next(
        module
        for module in model.modules()
        if isinstance(module, ShardedDynamicEmbeddingCollection)
    )


def rw_index(ec: ShardedDynamicEmbeddingCollection) -> int:
    """Position of the row-wise sharding (its dists and context)."""
    return list(ec._sharding_type_to_sharding).index(ShardingType.ROW_WISE.value)


def placement_keys() -> torch.Tensor:
    """The keys of the ``placement`` cases, the same on every rank: 4096 from
    the whole int32 range, 512 below NUM_EMBEDDINGS (inside ``continuous``'s
    blocks), and both ends of the range, but the four keys the tables reserve
    (-4 to -1: their int64 keys are the hash tables' empty, locked and
    reclaimed slots)."""
    generator = torch.Generator().manual_seed(4093)
    keys = torch.cat(
        [
            torch.randint(-(1 << 31), 1 << 31, (4096,), generator=generator),
            torch.randint(0, NUM_EMBEDDINGS, (512,), generator=generator),
            torch.tensor([-(1 << 31), -(1 << 31) + 1, (1 << 31) - 2, (1 << 31) - 1]),
            torch.tensor([-6, -5, 0, 1]),
        ]
    )
    reserved = (keys >= -4) & (keys <= -1)
    return torch.unique(keys[~reserved]).cuda()


def table_keys(model) -> Dict[str, torch.Tensor]:
    """This rank's keys of each DynamicEmb table of the row-wise sharding,
    sorted."""
    ec = sharded_ec(model)
    out = {}
    for module in ec._lookups[rw_index(ec)]._emb_modules:
        emb = module._emb_module
        if not isinstance(emb, BatchedDynamicEmbeddingTablesV2):
            continue
        for name in emb._table_names:
            keys, _ = emb.export_keys_values(name, torch.device("cuda"))
            out[name] = torch.sort(keys).values
    return out


def bucketized_keys(model, case: Case, rank: int) -> Dict[str, torch.Tensor]:
    """The keys each table of this rank holds after the case's calls on
    TorchRec's path: the keys of every rank's KJTs that DynamicEmb's bucketize
    (the sharding's own, on the KJT as it is) gives this rank, as it writes
    them, widened to int64 with their sign; per table, sorted."""
    ec = sharded_ec(model)
    input_dist = ec._input_dists[rw_index(ec)]
    world_size = dist.get_world_size()
    # the sharding's features, in its order (a placement case has no table
    # beside it)
    features = list(ec._sharding_type_to_sharding.values())[
        rw_index(ec)
    ].feature_names()
    table_of = {
        feature: f"t{table}" for table, feats in enumerate(TABLES) for feature in feats
    }
    parts: Dict[str, List[torch.Tensor]] = {name: [] for name in table_of.values()}
    for source in range(world_size):
        for kjt in batches_of(case, source):
            kjt = kjt.permute([kjt.keys().index(feature) for feature in features])
            bucketized, _ = input_dist._bucketize(kjt)
            # keys by [owner][feature][sample]
            cells = bucketized.lengths().view(world_size, len(features), -1).sum(2)
            ends = torch.cumsum(cells.flatten(), 0).view_as(cells).tolist()
            values = bucketized.values().long()
            for index, feature in enumerate(features):
                end = ends[rank][index]
                begin = end - int(cells[rank][index])
                parts[table_of[feature]].append(values[begin:end])
    return {name: torch.unique(torch.cat(keys)) for name, keys in parts.items()}


def check_placement(case: Case, checks: "Checks", model, rank: int, which: str):
    """Each table of this rank holds exactly the keys the bucketize gives the
    rank (:func:`bucketized_keys`)."""
    want, got = bucketized_keys(model, case, rank), table_keys(model)
    placed = sorted(got) == sorted(want) and all(
        torch.equal(got[name].cpu(), want[name].cpu()) for name in want
    )
    if not placed:
        print(
            f"[rank {rank}] {case.name}: {which} tables hold "
            + ", ".join(
                f"{name} {got[name].numel() if name in got else None}" for name in want
            )
            + " keys, the bucketize gives "
            + ", ".join(f"{name} {keys.numel()}" for name, keys in want.items()),
            flush=True,
        )
    checks.check(f"{case.name}: {which} placed", placed)


def hier_of(model):
    for module in model.modules():
        for output_dist in getattr(module, "_output_dists", None) or []:
            if isinstance(output_dist, RwSequenceHierEmbeddingDist):
                return output_dist._hier
    return None


@dataclass
class Step:
    out: Dict[str, torch.Tensor]  # the output rows of each feature
    plan: Optional[object]  # the call's plan, None on TorchRec's dists


def rows_of(case: Case, out) -> Dict[str, torch.Tensor]:
    return {name: out[name].values().detach().clone() for name in case.features}


def input_dist(ec, kjt: torchrec.KeyedJaggedTensor):
    """A call's input dist, both waits: its context, the lookup's input, and
    its plan (None on TorchRec's dists)."""
    ctx = ec.create_context()
    dist_input = ec.input_dist(ctx, kjt).wait().wait()
    return ctx, dist_input, getattr(ctx.sharding_contexts[rw_index(ec)], "plan", None)


def plain_step(model, case: Case, kjt: torchrec.KeyedJaggedTensor, weight) -> Step:
    """One call: its forward, then its backward."""
    ec = sharded_ec(model)
    ctx, dist_input, plan = input_dist(ec, kjt)
    out = ec.compute_and_output_dist(ctx, dist_input).wait()
    rows = rows_of(case, out)
    weighted_sum(out, weight, case.features).backward()
    return Step(rows, plan)


def interleaved(model, case: Case, batches, weight) -> List[Step]:
    """Two calls' forwards, then their backwards in order, pair by pair."""
    ec = sharded_ec(model)
    steps = []
    for first in range(0, len(batches), 2):
        pending = []
        for kjt in batches[first : first + 2]:
            ctx, dist_input, plan = input_dist(ec, kjt)
            out = ec.compute_and_output_dist(ctx, dist_input).wait()
            pending.append((out, Step(rows_of(case, out), plan)))
        for out, step in pending:
            weighted_sum(out, weight, case.features).backward()
            steps.append(step)
    return steps


def run(model, case: Case, batches, weight) -> List[Step]:
    """The case's calls on ``model``, in the case's order; their kernels have
    ended when it returns (the caller may drop the model, and its tables'
    memory with it)."""
    if case.order == "interleaved":
        steps = interleaved(model, case, batches, weight)
    else:
        steps = [plain_step(model, case, kjt, weight) for kjt in batches]
    torch.cuda.synchronize()
    return steps


def same_everywhere(values: List[int]) -> bool:
    every = [None] * dist.get_world_size()
    dist.all_gather_object(every, values)
    return all(other == values for other in every)


class Checks:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.failed = 0

    def check(self, label: str, ok: bool) -> None:
        flag = torch.tensor([int(ok)], device="cuda")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        ok = bool(flag.item())
        self.failed += not ok
        if self.rank == 0:
            print(f"{label}={int(ok)} [{'ok' if ok else 'FAIL'}]", flush=True)


def tight_capacity(case: Case, needs: List[Dict[str, int]]):
    """The capacities of ``tight``: ample, but ``case.capacity`` at the most
    the case's small calls need (their needs, as ``ample``'s calls counted
    them)."""
    small = [need for step, need in enumerate(needs) if step not in case.expected]
    if case.capacity == "int":
        return max(
            max(need[name] for name in OUTPUT_DIST_CAPACITY_NAMES) for need in small
        )
    capacity = dict.fromkeys(OUTPUT_DIST_CAPACITY_NAMES, AMPLE)
    capacity[case.capacity] = max(need[case.capacity] for need in small)
    return capacity


def run_case(case: Case, checks: Checks, rank: int, world_size: int, local_size: int):
    exact = world_size & (world_size - 1) == 0
    torch.manual_seed(7)
    weight = torch.randn(DIM, device="cuda").sign()
    batches = batches_of(case, rank)
    # TorchRec's dists, then the ample capacities, whose calls give the needs
    ample_capacity = dict.fromkeys(OUTPUT_DIST_CAPACITY_NAMES, AMPLE)
    reference_model = build_model(case, "torchrec", None, world_size, local_size)
    reference = run(reference_model, case, batches, weight)
    del reference_model
    ample_model = build_model(case, "fused", ample_capacity, world_size, local_size)
    ample = run(ample_model, case, batches, weight)
    if case.placement:
        check_placement(case, checks, ample_model, rank, "ample's")
    needs = [step.plan.needs() for step in ample]
    quiesce(ample_model)
    del ample_model
    capacity = tight_capacity(case, needs)
    tight_model = build_model(case, "fused", capacity, world_size, local_size)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tight = run(tight_model, case, batches, weight)
    hier = hier_of(tight_model)
    label = case.name

    def same_rows(steps: List[Step]) -> bool:
        if len(steps) != len(reference):
            return False
        for index, (step, ref) in enumerate(zip(steps, reference)):
            for name in case.features:
                expected, got = ref.out[name], step.out[name]
                same = (
                    torch.equal(expected, got)
                    if exact
                    else torch.allclose(expected, got, 1e-4, 1e-4)
                )
                if not same:
                    print(
                        f"[rank {rank}] {label}: feature {name} differs at step {index}",
                        flush=True,
                    )
                    return False
        return True

    checks.check(f"{label}: fwd", same_rows(tight))
    fell = [step for step, done in enumerate(tight) if done.plan is None]
    checks.check(f"{label}: fallback {fell}", fell == list(case.expected))
    checks.check(f"{label}: same calls everywhere", same_everywhere(fell))
    said = [
        str(warning.message)
        for warning in caught
        if "does not fit the capacities" in str(warning.message)
    ]
    named = case.capacity if case.capacity != "int" else ""
    # at the first call that falls back, then at every power of two of them
    warnings_due = sum(
        1 for count in range(1, hier.fallbacks + 1) if count & (count - 1) == 0
    )
    checks.check(
        f"{label}: fallbacks {hier.fallbacks}", hier.fallbacks == len(case.expected)
    )
    checks.check(
        f"{label}: warned {len(said)} times",
        len(said) == warnings_due and (not said or named in said[0]),
    )
    if case.placement:
        check_placement(case, checks, tight_model, rank, "tight's")
    quiesce(tight_model)
    del tight_model
    torch.cuda.synchronize()
    dist.barrier()


def main() -> None:
    rank, world_size, _, local_size = init_distributed()
    os.environ.setdefault("DYNAMICEMB_GIN_TIMEOUT_S", "120")
    multi_node = world_size // local_size > 1
    checks = Checks(rank)
    for case in CASES:
        if case.multi_node and not multi_node:
            continue
        run_case(case, checks, rank, world_size, local_size)
    if rank == 0:
        print(
            "ALL PASSED" if not checks.failed else f"{checks.failed} FAILED", flush=True
        )
    dist.destroy_process_group()
    sys.exit(1 if checks.failed else 0)


if __name__ == "__main__":
    main()
