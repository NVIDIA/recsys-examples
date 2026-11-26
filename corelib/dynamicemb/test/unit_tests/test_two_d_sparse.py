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

"""DynamicEmb tables under TorchRec 2D sparse parallelism (DMPCollection).

``DMPCollection`` shards each table over a sharding group and replicates the
group across the world. Each replica trains on its own keys here, some shared
and some its own, so the replicas' tables differ before a sync.

Modes:

``sync``
    ``DynamicEmbReplicaSync`` gives every replica the union of the keys, each the
    average of the replicas' rows (a fresh row standing in for a replica that
    lacked the key), and the replicas agree exactly. A second sync with no
    lookup in between covers no key.
``checkpoint``
    ``DynamicEmbDump`` of a ``DMPCollection`` is written by the first replica
    alone: replicas that disagree dump, and a fresh model's every replica loads
    the first replica's rows.
``reject``
    A table without a timestamp score cannot be synced, and says so.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import torch
import torch.distributed as dist
import torchrec
from dynamicemb import (
    DynamicEmbDump,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbLoad,
    DynamicEmbReplicaSync,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamicemb.dump_load import get_dynamic_emb_module
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingBagCollectionSharder
from dynamicemb.utils import decode_table_values, encode_table_values
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from torchrec.distributed.model_parallel import DMPCollection
from torchrec.distributed.planner import Topology
from torchrec.distributed.types import BoundsCheckMode, ShardingType

TABLES = ["t0", "t1"]
EMBEDDING_DIM = 8
NUM_EMBEDDINGS = 1 << 16
INIT_VALUE = 0.125
KEYS_PER_REPLICA = 4
# --grow: a table starts at GROW_CAPACITY and doubles past a load factor of 0.5.
# Each replica trains GROW_KEYS_PER_SHARD keys a shard of its own once, which
# fits, and the sync brings it the other replicas' as well, which does not.
GROW_CAPACITY = 1024
GROW_KEYS_PER_SHARD = 300
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "fp8": torch.float8_e4m3fn,
}
SCORES = {
    "lru_lfu": (DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU),
    "lru": DynamicEmbScoreStrategy.TIMESTAMP,
    "step": DynamicEmbScoreStrategy.STEP,
}
OPTIMIZERS = {
    "adagrad": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
    "adam": EmbOptimType.ADAM,
    "sgd": EmbOptimType.EXACT_SGD,
}
# What one replica holds of one table: key -> (embedding, optimizer state, frequency).
Rows = Dict[int, Tuple[List[float], List[float], int]]


def _feature(table: str) -> str:
    return f"f_{table}"


def _layout(args, table):
    # Different physical layouts create separate kernels in the same DMP.
    if args.mixed_tables and table == TABLES[1]:
        return 2 * EMBEDDING_DIM, torch.float16, "lru"
    return EMBEDDING_DIM, DTYPES[args.embedding_dtype], args.score_strategy


def _build(args) -> DMPCollection:
    ebc = torchrec.EmbeddingBagCollection(
        device=torch.device("meta"),
        tables=[
            torchrec.EmbeddingBagConfig(
                name=table,
                embedding_dim=_layout(args, table)[0],
                num_embeddings=NUM_EMBEDDINGS,
                feature_names=[_feature(table)],
                pooling=torchrec.PoolingType.SUM,
            )
            for table in TABLES
        ],
    )
    constraints = {
        table: DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            bounds_check_mode=BoundsCheckMode.NONE,
            enforce_hbm=True,
            use_dynamicemb=True,
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=1024**3,
                init_capacity=GROW_CAPACITY if args.grow else None,
                embedding_dtype=_layout(args, table)[1],
                index_type=torch.uint64 if args.uint64_keys else torch.int64,
                score_strategy=SCORES[_layout(args, table)[2]],
                initializer_args=DynamicEmbInitializerArgs(
                    mode=(
                        DynamicEmbInitializerMode.NORMAL
                        if args.random_initializer
                        else DynamicEmbInitializerMode.CONSTANT
                    ),
                    value=INIT_VALUE,
                ),
            ),
        )
        for table in TABLES
    }
    sharder = DynamicEmbeddingBagCollectionSharder(
        fused_params={
            "output_dtype": SparseType.FP32,
            "optimizer": OPTIMIZERS[args.optimizer],
            "learning_rate": 0.05,
            "initial_accumulator_value": 0.375 if args.optimizer == "adagrad" else 0.0,
        }
    )
    # The plan is for one sharding group; DMPCollection remaps it to each replica.
    group_size = args.sharding_group_size
    planner = DynamicEmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=group_size,
            world_size=group_size,
            compute_device="cuda",
            hbm_cap=80 * 1024**3,
        ),
        constraints=constraints,
        batch_size=KEYS_PER_REPLICA,
        debug=False,
    )
    plan = planner.collective_plan(ebc, [sharder], dist.GroupMember.WORLD)
    return DMPCollection(
        module=ebc,
        device=torch.device("cuda", torch.cuda.current_device()),
        plan=plan,
        world_size=dist.get_world_size(),
        sharding_group_size=group_size,
        global_pg=dist.GroupMember.WORLD,
        node_group_size=group_size,
        sharders=[sharder],
        use_inter_host_allreduce=not args.strided,
    )


def _replica_group(model: DMPCollection) -> dist.ProcessGroup:
    return model.device_mesh.get_group("replicate")


def _replica_keys(args, replica: int) -> List[int]:
    if args.grow:
        count = GROW_KEYS_PER_SHARD * args.sharding_group_size
        return [10 + replica * count + i for i in range(count)]
    # Consecutive replicas share half their keys, so every key is held by one
    # replica or two, and each shard gets some of each.
    return [10 + 2 * replica + i for i in range(KEYS_PER_REPLICA)]


def _train(
    args, model: DMPCollection, keys: Optional[List[int]] = None, active_tables=TABLES
) -> None:
    # Every rank of a replica feeds the replica's keys; the fused optimizer
    # steps inside backward. One step under --grow: the table counts a batch's
    # keys toward its load before it knows how many are new.
    steps = 1 if args.grow else 2
    if keys is None:
        keys = _replica_keys(args, dist.get_rank(_replica_group(model)))
    device = torch.device("cuda", torch.cuda.current_device())
    kjt = torchrec.KeyedJaggedTensor(
        keys=[_feature(table) for table in TABLES],
        values=torch.tensor(
            keys * len(active_tables), dtype=torch.int64, device=device
        ),
        lengths=torch.tensor(
            [int(t in active_tables) for t in TABLES for _ in keys],
            dtype=torch.int64,
            device=device,
        ),
    )
    for _ in range(steps):
        model(kjt).values().square().sum().backward()
    torch.cuda.synchronize()


def _rows(model: DMPCollection) -> Dict[str, Rows]:
    """What this rank holds, per table, at checkpoint precision and width."""
    rows: Dict[str, Rows] = {}
    for kernel in get_dynamic_emb_module(model):
        for table_id, name in enumerate(kernel.table_names):
            table = rows.setdefault(name, {})
            for keys, embeddings, opt, scores in kernel.tables.export_keys_values(
                torch.device("cuda"), 1024, table_id
            ):
                opt_rows = (
                    kernel.optimizer.states_for_checkpoint(opt, embeddings.size(1))
                    .float()
                    .tolist()
                    if opt is not None
                    else [[] for _ in range(keys.numel())]
                )
                # An LRU_LFU table exports (timestamp, frequency), as configured.
                frequency = (
                    scores[:, 1].tolist() if scores.dim() == 2 else [0] * keys.numel()
                )
                for i, key in enumerate(keys.tolist()):
                    table[int(key)] = (
                        embeddings[i].float().tolist(),
                        opt_rows[i],
                        int(frequency[i]),
                    )
    return rows


def _gather_replicas(model: DMPCollection, rows) -> list:
    """``rows`` of every replica's rank that holds this rank's shard."""
    gathered: list = [None] * dist.get_world_size(_replica_group(model))
    dist.all_gather_object(gathered, rows, group=_replica_group(model))
    return gathered


def _at_table_precision(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return decode_table_values(encode_table_values(values.cuda(), dtype), torch.float32)


def _check_rows(args, model, before, after, stats, touched=None) -> None:
    replica_count = len(before)
    assert set(stats) == set(TABLES)
    for name in TABLES:
        dim, dtype, score_strategy = _layout(args, name)
        kernel = next(k for k in get_dynamic_emb_module(model) if name in k.table_names)
        fresh_emb = _at_table_precision(torch.tensor([INIT_VALUE]), dtype).item()
        fresh_opt = [
            kernel.optimizer.get_initial_optimizer_state()
        ] * kernel.optimizer.get_ckpt_state_dim(dim)
        tolerance = {
            torch.float32: 1e-6,
            torch.float16: 1e-3,
            torch.float8_e4m3fn: 1 / 32,
        }[dtype]
        union = sorted(set().union(*(replica[name] for replica in before)))
        assert union, f"{name}: this shard holds no key; nothing was compared"
        mine = before[dist.get_rank(_replica_group(model))][name]
        merged = set(union) if touched is None else set(union) & touched[name]
        assert stats[name].union_keys == len(merged), (name, stats[name], merged)
        assert stats[name].local_keys == len(merged.intersection(mine)), (
            name,
            stats[name],
            mine,
        )
        for replica in after:
            assert sorted(replica[name]) == union, (name, sorted(replica[name]))
        for key in union:
            if key not in merged:
                for prev, cur in zip(before, after):
                    assert cur[name][key] == prev[name][key], (name, key)
                continue
            held = [replica[name][key] for replica in before if key in replica[name]]
            if not args.random_initializer:
                fresh_row = ([fresh_emb] * dim, fresh_opt, 0)
                held += [fresh_row] * (replica_count - len(held))
            emb = torch.tensor([row[0] for row in held]).mean(0)
            opt = torch.tensor([row[1] for row in held]).mean(0)
            opt = opt.to(kernel.optimizer.get_state_dtype(dtype)).float()
            frequency = max(1, sum(row[2] for row in held) // replica_count)
            got_emb, got_opt, got_frequency = after[0][name][key]
            torch.testing.assert_close(
                torch.tensor(got_emb).cuda(),
                _at_table_precision(emb, dtype),
                rtol=0,
                atol=tolerance,
            )
            torch.testing.assert_close(torch.tensor(got_opt), opt, rtol=1e-6, atol=1e-9)
            if score_strategy == "lru_lfu":
                assert got_frequency == frequency, (name, key, got_frequency, held)
            for replica in after[1:]:
                assert replica[name][key] == after[0][name][key], (name, key)


def check_sync(args) -> None:
    model = _build(args)
    rank = dist.get_rank(_replica_group(model))
    # The second table receives keys only on one replica in the mixed case.
    _train(
        args,
        model,
        active_tables=TABLES[:1] if args.mixed_tables and rank == 0 else TABLES,
    )
    sync = DynamicEmbReplicaSync(model)
    before = _gather_replicas(model, _rows(model))
    states = [kernel.tables.tables[0] for kernel in get_dynamic_emb_module(model)]
    generations = [state.key_map_generation for state in states]
    stats = {table.table_name: table for table in sync.sync()}
    if args.grow:
        grew = [state.key_map_generation for state in states] != generations
        assert grew, "no table grew during the sync; --grow tested nothing"
    after = _gather_replicas(model, _rows(model))
    _check_rows(args, model, before, after, stats)
    if args.resync:
        # All replicas touch the entire union twice to populate and reuse
        # the dense route. Then just a subset changes, with a new private
        # key on each replica, so a stale route would lose updates.
        replicas = dist.get_world_size(_replica_group(model))
        shared = sorted(set().union(*(_replica_keys(args, r) for r in range(replicas))))
        for keys in (shared, shared, [10 + rank, 1000 + rank]):
            _train(args, model, keys)
            touched = set().union(*_gather_replicas(model, keys))
            before = _gather_replicas(model, _rows(model))
            stats = {table.table_name: table for table in sync.sync()}
            after = _gather_replicas(model, _rows(model))
            _check_rows(
                args, model, before, after, stats, dict.fromkeys(TABLES, touched)
            )

    if args.mixed_tables:
        # Only one table gets touched after a nonempty sync. Check the other
        # table retains its exact rows, frequency counts and optimizer state.
        keys = [10000 + rank]
        _train(args, model, keys, active_tables=TABLES[:1])
        touched = set().union(*_gather_replicas(model, keys))
        before = _gather_replicas(model, _rows(model))
        # No expansion this time. Both kernels, including the untouched
        # table, must reach the C++ operator in one call.
        with patch.object(
            torch.ops.dyn_emb,
            "replica_sync_batch_",
            wraps=torch.ops.dyn_emb.replica_sync_batch_,
        ) as dispatch:
            result = sync.sync()
        dispatch.assert_called_once()
        assert len(dispatch.call_args.args[0]) == len(TABLES)
        assert len(result) == len(TABLES)
        stats = {item.table_name: item for item in result}
        after = _gather_replicas(model, _rows(model))
        _check_rows(
            args, model, before, after, stats, {TABLES[0]: touched, TABLES[1]: set()}
        )

    again = sync.sync()
    assert all(table.union_keys == 0 for table in again), again


def check_checkpoint(args) -> None:
    model = _build(args)
    _train(args, model)
    # No sync: the replicas disagree, so the rows a load returns say who wrote.
    first = _gather_replicas(model, _rows(model))[0]
    DynamicEmbDump(args.save_path, model, optim=True)
    if dist.get_rank() == 0:
        key_files = [
            name
            for _, _, names in os.walk(args.save_path)
            for name in names
            if "_emb_keys." in name
        ]
        expected = len(TABLES) * args.sharding_group_size
        assert len(key_files) == expected, (expected, sorted(key_files))

    loaded = _build(args)
    DynamicEmbLoad(args.save_path, loaded, optim=True)
    got = _rows(loaded)
    for name in TABLES:
        assert sorted(got[name]) == sorted(first[name]), (name, sorted(got[name]))
        for key, (emb, opt, _) in first[name].items():
            assert got[name][key][:2] == (emb, opt), (name, key, got[name][key])
    sync = DynamicEmbReplicaSync(loaded)
    sync.mark_synced()
    assert all(table.union_keys == 0 for table in sync.sync())


def check_reject(args) -> None:
    model = _build(args)
    try:
        DynamicEmbReplicaSync(model)
    except ValueError as error:
        assert "timestamp" in str(error), error
    else:
        raise AssertionError("a table without a timestamp score was accepted")


MODES = {"sync": check_sync, "checkpoint": check_checkpoint, "reject": check_reject}


def main(argv) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--sharding-group-size", type=int, default=1)
    parser.add_argument(
        "--strided",
        action="store_true",
        help="sharding groups of strided ranks (use_inter_host_allreduce=False) "
        "rather than contiguous ones",
    )
    parser.add_argument(
        "--mixed-tables",
        action="store_true",
        help="Exercise different dimensions, dtypes, scores and touched sets across tables",
    )
    parser.add_argument("--embedding-dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--score-strategy", choices=sorted(SCORES), default="lru_lfu")
    parser.add_argument("--uint64-keys", action="store_true")
    parser.add_argument("--random-initializer", action="store_true")
    parser.add_argument("--resync", action="store_true")
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZERS), default="adagrad")
    parser.add_argument(
        "--grow",
        action="store_true",
        help="small tables that the sync's inserts make grow",
    )
    parser.add_argument("--save-path", default="/tmp/dynamicemb_two_d_ckpt")
    args = parser.parse_args(argv)

    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size // args.sharding_group_size >= 2, "needs two replicas or more"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if rank == 0:
        print(
            f"[{args.mode}] world {world_size} = "
            f"{world_size // args.sharding_group_size} replica(s) x "
            f"{args.sharding_group_size}, {'strided' if args.strided else 'contiguous'}"
            f", {args.embedding_dtype}, {args.score_strategy}, {args.optimizer}"
            f"{', growing' if args.grow else ''}"
        )
    MODES[args.mode](args)
    dist.barrier()
    if rank == 0:
        print(f"[{args.mode}] passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
