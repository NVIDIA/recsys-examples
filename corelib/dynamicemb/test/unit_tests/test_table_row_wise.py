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

"""Table-row-wise DynamicEmb tables, end to end.

A node here is ``LOCAL_WORLD_SIZE`` ranks, which both TorchRec's
``intra_and_cross_node_pg`` and the planner's Topology read from the
environment -- so N nodes are emulated on one box by setting it to
``world_size // N``, and these need no second machine.

Every mode puts a table-row-wise table and a row-wise one in the same
collection, which is the arrangement a real model has and the one §7 F5 is
about. The row-wise table is the control: with ``DEBUG`` initialization a row is
a function of its key alone, so the two tables must return the same embedding
for the same key however their shards are spread.

Modes:

``plan``
    What the planner decided, asserted directly. The only one of these that
    catches a planner that silently stopped planning (§8.5): such a plan still
    trains, it just trains a TorchRec table.
``parity``
    Forward, table-row-wise against row-wise. Covers DMP construction, the two
    output collectives and the MEAN downgrade.
``pooling``
    MEAN against SUM. Two MEAN tables agreeing says nothing about MEAN -- a
    double application makes both of them ``sum/len^2`` equally -- so this is
    the one absolute check here.
``jagged``
    The same forward with bags of differing length, empty ones included. Not
    §7 F4: true variable batch needs a KJT carrying ``stride_per_key_per_rank``,
    and F4 stays open.
``dump`` / ``load``
    A checkpoint round trip. The sharding is an argument so the driver can dump
    under one and load under the other.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torchrec
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
)
from dynamicemb.dump_load import DynamicEmbDump, DynamicEmbLoad
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingBagCollectionSharder
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from torch.distributed.elastic.multiprocessing.errors import record
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.types import BoundsCheckMode, ShardingType

TRW = ShardingType.TABLE_ROW_WISE.value
RW = ShardingType.ROW_WISE.value

# The table-row-wise table and its row-wise control, in that order. Same rows,
# same dim, same features per row -- only the sharding differs, which is what
# makes their outputs comparable.
TRW_TABLE, RW_TABLE = "t_trw", "t_rw"
# A third, table-row-wise and SUM-pooled. The two above are MEAN, and comparing
# them to each other cannot see a MEAN that was applied twice -- both would be
# wrong the same way, which is exactly how sum/len^2 survived until TRW made
# someone look at `sharding_type` (§8.5). This one gives the absolute check.
SUM_TABLE = "t_trw_sum"
NUM_EMBEDDINGS = 1_000_000
EMBEDDING_DIM = 16


def _feature(table: str) -> str:
    return f"f_{table}"


# name -> (sharding type, pooling). The sharding of the first is an argument so
# a checkpoint can be written under one geometry and read under the other.
def _tables(trw_sharding: str):
    return {
        TRW_TABLE: (trw_sharding, torchrec.PoolingType.MEAN),
        RW_TABLE: (RW, torchrec.PoolingType.MEAN),
        SUM_TABLE: (trw_sharding, torchrec.PoolingType.SUM),
    }


def _eb_configs(trw_sharding: str) -> List[torchrec.EmbeddingBagConfig]:
    return [
        torchrec.EmbeddingBagConfig(
            name=name,
            embedding_dim=EMBEDDING_DIM,
            num_embeddings=NUM_EMBEDDINGS,
            feature_names=[_feature(name)],
            pooling=pooling,
        )
        for name, (_, pooling) in _tables(trw_sharding).items()
    ]


def _constraints(
    trw_sharding: str, dist_type: str
) -> Dict[str, DynamicEmbParameterConstraints]:
    """Every table DynamicEmb's, differing only in sharding type and pooling.

    ``host_index`` is left unset: the planner's HostPlacer picks the node, which
    is the path a user takes and so the one worth testing.
    """

    def one(sharding: str) -> DynamicEmbParameterConstraints:
        return DynamicEmbParameterConstraints(
            sharding_types=[sharding],
            bounds_check_mode=BoundsCheckMode.NONE,
            enforce_hbm=True,
            use_dynamicemb=True,
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=1024**3,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.DEBUG
                ),
                safe_check_mode=DynamicEmbCheckMode.WARNING,
                dist_type=dist_type,
            ),
        )

    return {
        name: one(sharding) for name, (sharding, _) in _tables(trw_sharding).items()
    }


def _planner(device: torch.device, constraints) -> DynamicEmbeddingShardingPlanner:
    local_size = get_local_size()
    return DynamicEmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=local_size,
            world_size=dist.get_world_size(),
            compute_device=device.type,
            hbm_cap=80 * 1024**3,
            # A node's host memory, not a rank's: Topology replicates what it is
            # given to every device rather than dividing it.
            ddr_cap=1024**4 // local_size,
        ),
        constraints=constraints,
        batch_size=64,
        debug=False,
    )


def _build(device: torch.device, trw_sharding: str, dist_type: str):
    """An EBC sharded with one table-row-wise table and one row-wise, and its plan."""
    ebc = torchrec.EmbeddingBagCollection(
        device=torch.device("meta"), tables=_eb_configs(trw_sharding)
    )
    constraints = _constraints(trw_sharding, dist_type)
    sharder = DynamicEmbeddingBagCollectionSharder(
        fused_params={
            "output_dtype": SparseType.FP32,
            "optimizer": EmbOptimType.EXACT_SGD,
            "learning_rate": 0.1,
        }
    )
    plan = _planner(device, constraints).collective_plan(
        ebc, [sharder], dist.GroupMember.WORLD
    )
    model = DistributedModelParallel(
        module=ebc, device=device, sharders=[sharder], plan=plan
    )
    return model, plan


def _kjt(keys: List[int], device: torch.device, bags: Optional[List[int]] = None):
    """One KJT giving both tables the same keys, so their outputs can be compared.

    ``bags`` is the length of each bag, defaulting to one key per bag. Both
    features carry the same values, which is the whole point: a key lands on a
    different rank for each table, and must come back the same.
    """
    lengths = bags if bags is not None else [1] * len(keys)
    names = [_feature(n) for n in (TRW_TABLE, RW_TABLE, SUM_TABLE)]
    return torchrec.KeyedJaggedTensor(
        keys=names,
        values=torch.tensor(keys * len(names), dtype=torch.int64, device=device),
        lengths=torch.tensor(lengths * len(names), dtype=torch.int64, device=device),
    )


def _rank_keys(rank: int, world_size: int, count: int = 64) -> List[int]:
    """Keys this rank feeds in. Disjoint per rank, so every rank contributes."""
    return [rank + world_size * i for i in range(count)]


# --- modes -----------------------------------------------------------------


def check_plan(args, device, rank, world_size) -> None:
    """The plan says what we think it says.

    A DynamicEmb table that TorchRec planned instead would still train -- it
    would just not be a DynamicEmb table. Nothing downstream raises, so this is
    where it has to be caught (§8.5).
    """
    _, plan = _build(device, TRW, args.dist_type)
    [module_plan] = plan.plan.values()
    local_size = get_local_size()
    num_nodes = world_size // local_size

    for name, expected_type, expected_shards in (
        (TRW_TABLE, TRW, local_size),
        (RW_TABLE, RW, world_size),
        (SUM_TABLE, TRW, local_size),
    ):
        ps = module_plan[name]
        assert ps.sharding_type == expected_type, (
            f"{name}: planned {ps.sharding_type}, wanted {expected_type}. A "
            "DynamicEmb table TorchRec planned is still a plan, just not ours."
        )
        assert ps.compute_kernel == "customized_kernel", (
            f"{name}: compute_kernel is {ps.compute_kernel}; a DynamicEmb table "
            "reaches its hash table through CUSTOMIZED_KERNEL."
        )
        assert getattr(ps, "dynamicemb_options", None) is not None, (
            f"{name}: no DynamicEmbTableOptions on the ParameterSharding. They "
            "are reattached per rank after the broadcast, so missing here means "
            "this rank never settled the table."
        )
        shards = ps.sharding_spec.shards
        assert (
            len(shards) == expected_shards
        ), f"{name}: {len(shards)} shards, wanted {expected_shards}"
        assert len(ps.ranks) == expected_shards

    # The table-row-wise table is on one node's ranks, contiguously.
    trw_ranks = module_plan[TRW_TABLE].ranks
    host = trw_ranks[0] // local_size
    assert 0 <= host < num_nodes, f"host {host} outside {num_nodes} nodes"
    assert trw_ranks == list(
        range(host * local_size, (host + 1) * local_size)
    ), f"table-row-wise ranks {trw_ranks} are not node {host}'s ranks"

    # Capacity follows the fan-out: one node's worth of rows, not the world's.
    trw_rows = module_plan[TRW_TABLE].sharding_spec.shards[0].shard_sizes[0]
    rw_rows = module_plan[RW_TABLE].sharding_spec.shards[0].shard_sizes[0]
    assert trw_rows * local_size >= NUM_EMBEDDINGS
    assert rw_rows * world_size >= NUM_EMBEDDINGS
    if num_nodes > 1:
        assert trw_rows > rw_rows, (
            f"table-row-wise rows per rank ({trw_rows}) should exceed row-wise's "
            f"({rw_rows}) by about {num_nodes}x -- that is R1."
        )

    if rank == 0:
        print(
            f"plan ok: {num_nodes} nodes x {local_size}, trw on node {host} "
            f"({trw_rows} rows/rank), rw on all {world_size} ({rw_rows} rows/rank)"
        )


def _compare_forward(args, device, rank, world_size, bags=None) -> None:
    model, _ = _build(device, TRW, args.dist_type)
    keys = _rank_keys(rank, world_size)
    out = model(_kjt(keys, device, bags)).to_dict()

    trw_out = out[_feature(TRW_TABLE)]
    rw_out = out[_feature(RW_TABLE)]
    assert trw_out.shape == rw_out.shape, f"{trw_out.shape} vs {rw_out.shape}"
    torch.testing.assert_close(
        trw_out,
        rw_out,
        msg=lambda m: (
            "A table-row-wise table and a row-wise one, same rows and same keys, "
            "returned different embeddings. With DEBUG initialization a row is a "
            "function of its key alone, so the sharding is the only difference.\n" + m
        ),
    )
    # Not vacuous: DEBUG rows are non-zero, so an all-zero result would mean
    # both lookups missed rather than that both agreed.
    assert trw_out.abs().sum() > 0, "both tables returned zeros; nothing was compared"


def check_parity(args, device, rank, world_size) -> None:
    _compare_forward(args, device, rank, world_size)
    if rank == 0:
        print("parity ok: table-row-wise matches row-wise")


def check_jagged(args, device, rank, world_size) -> None:
    """Bags of differing length, including empty ones.

    This is **not** the variable-batch path of §7 F4. True VBE is a KJT carrying
    `stride_per_key_per_rank`, where different features have different batch
    sizes per rank; this is an ordinary jagged batch. What it does cover is the
    empty bag, which is where an off-by-one in the bucketized offsets shows up,
    and it covers it for table-row-wise where `num_buckets` is `local_size`.

    F4 stays open. Writing a test for it means building a KJT with inverse
    indices, and getting that wrong would look like a passing test.
    """
    keys = _rank_keys(rank, world_size)
    bags = [(i % 4) for i in range(len(keys))]  # 0..3, so some bags are empty
    # Keep the key count equal to the bag lengths' sum.
    keys = keys[: sum(bags)] if sum(bags) <= len(keys) else keys
    bags = _fit_bags(bags, len(keys))
    _compare_forward(args, device, rank, world_size, bags=bags)
    if rank == 0:
        print(f"jagged ok: bags {bags[:8]}... sum {sum(bags)}")


def _fit_bags(bags: List[int], total: int) -> List[int]:
    """Trim or pad the bag lengths so they sum to exactly `total` keys."""
    out, used = [], 0
    for b in bags:
        take = min(b, total - used)
        out.append(take)
        used += take
    out[-1] += total - used
    return out


def check_dump(args, device, rank, world_size) -> None:
    model, _ = _build(device, args.trw_sharding, args.dist_type)
    keys = _rank_keys(rank, world_size)
    model(_kjt(keys, device))  # insert the rows before writing them
    dist.barrier()
    DynamicEmbDump(args.save_path, model, optim=False)
    dist.barrier()
    if rank == 0:
        print(f"dump ok: {args.trw_sharding} -> {args.save_path}")


def check_load(args, device, rank, world_size) -> None:
    """Reload and check the values, not just that it did not raise.

    The failure this is for is silent: a loader applying the wrong fan-out keeps
    the keys that satisfy both rules and drops the rest (§8.1), so the tables
    come back smaller rather than broken.
    """
    model, _ = _build(device, args.trw_sharding, args.dist_type)
    DynamicEmbLoad(args.save_path, model, optim=False)
    dist.barrier()

    keys = _rank_keys(rank, world_size)
    out = model(_kjt(keys, device)).to_dict()
    torch.testing.assert_close(
        out[_feature(TRW_TABLE)],
        out[_feature(RW_TABLE)],
        msg=lambda m: (
            f"after loading a checkpoint into {args.trw_sharding}, the two tables "
            "disagree. A key kept by one fan-out and dropped by another comes "
            "back as a freshly initialized row, not as an error.\n" + m
        ),
    )
    if rank == 0:
        print(f"load ok: {args.save_path} -> {args.trw_sharding}")


def check_pooling(args, device, rank, world_size) -> None:
    """MEAN is applied once, checked against SUM rather than against MEAN.

    A MEAN table and a row-wise MEAN table agreeing proves nothing about MEAN:
    the kernel pooling MEAN *and* the EBC's callback dividing again makes both
    of them `sum/len^2`, equally. Against a SUM table of the same rows and keys,
    `mean * len` has to be `sum`, and a second division breaks it by `len`.
    """
    model, _ = _build(device, TRW, args.dist_type)
    keys = _rank_keys(rank, world_size)
    length = 4
    bags = [length] * (len(keys) // length)
    keys = keys[: sum(bags)]
    out = model(_kjt(keys, device, bags)).to_dict()

    torch.testing.assert_close(
        out[_feature(TRW_TABLE)] * length,
        out[_feature(SUM_TABLE)],
        msg=lambda m: (
            f"mean * {length} != sum over bags of {length}. If the ratio is off "
            f"by {length} the pooling happened twice: the kernel averaged the "
            "rows and the EBC's mean callback divided by the bag length again.\n" + m
        ),
    )
    assert out[_feature(SUM_TABLE)].abs().sum() > 0, "sum table returned zeros"
    if rank == 0:
        print(f"pooling ok: mean * {length} == sum")


MODES = {
    "plan": check_plan,
    "parity": check_parity,
    "pooling": check_pooling,
    "jagged": check_jagged,
    "dump": check_dump,
    "load": check_load,
}


@record
def main(argv) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument(
        "--trw-sharding",
        default=TRW,
        choices=[TRW, RW],
        help="what the first table is sharded as, so a checkpoint can be "
        "written under one geometry and read under the other",
    )
    parser.add_argument(
        "--dist-type", default="roundrobin", choices=["roundrobin", "hash_roundrobin"]
    )
    parser.add_argument("--save-path", default="/tmp/dynamicemb_trw_ckpt")
    args = parser.parse_args(argv)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank, world_size = dist.get_rank(), dist.get_world_size()

    if rank == 0:
        print(
            f"[{args.mode}] world {world_size}, node size {get_local_size()}, "
            f"dist_type {args.dist_type}, trw table sharded {args.trw_sharding}"
        )
    MODES[args.mode](args, device, rank, world_size)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
