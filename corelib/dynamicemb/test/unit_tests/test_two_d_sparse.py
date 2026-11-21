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
from typing import Dict, List, Tuple

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
    "sgd": EmbOptimType.EXACT_SGD,
}
# What one replica holds of one table: key -> (embedding, optimizer state, frequency).
Rows = Dict[int, Tuple[List[float], List[float], int]]


def _feature(table: str) -> str:
    return f"f_{table}"


def _build(args) -> DMPCollection:
    ebc = torchrec.EmbeddingBagCollection(
        device=torch.device("meta"),
        tables=[
            torchrec.EmbeddingBagConfig(
                name=table,
                embedding_dim=EMBEDDING_DIM,
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
                embedding_dtype=DTYPES[args.embedding_dtype],
                score_strategy=SCORES[args.score_strategy],
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.CONSTANT, value=INIT_VALUE
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


def _replica_keys(replica: int) -> List[int]:
    # Consecutive replicas share half their keys, so every key is held by one
    # replica or two, and each shard gets some of each.
    return [10 + 2 * replica + i for i in range(KEYS_PER_REPLICA)]


def _train(model: DMPCollection, steps: int = 2) -> None:
    # Every rank of a replica feeds the replica's keys; the fused optimizer
    # steps inside backward.
    keys = _replica_keys(dist.get_rank(_replica_group(model)))
    device = torch.device("cuda", torch.cuda.current_device())
    kjt = torchrec.KeyedJaggedTensor(
        keys=[_feature(table) for table in TABLES],
        values=torch.tensor(keys * len(TABLES), dtype=torch.int64, device=device),
        lengths=torch.ones(len(keys) * len(TABLES), dtype=torch.int64, device=device),
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
                    kernel.optimizer.states_for_checkpoint(opt, EMBEDDING_DIM)
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


def check_sync(args) -> None:
    model = _build(args)
    _train(model)
    sync = DynamicEmbReplicaSync(model)
    before = _gather_replicas(model, _rows(model))
    stats = {table.table_name: table for table in sync.sync()}
    after = _gather_replicas(model, _rows(model))
    replica_count = len(before)
    dtype = DTYPES[args.embedding_dtype]
    fresh_emb = _at_table_precision(torch.tensor([INIT_VALUE]), dtype).item()
    kernel = get_dynamic_emb_module(model)[0]
    opt_width = len(next(iter(before[0][TABLES[0]].values()))[1])
    fresh_opt = [kernel.optimizer.get_initial_optimizer_state()] * opt_width
    tolerance = {"fp32": 1e-6, "fp16": 1e-3, "fp8": 1 / 32}[args.embedding_dtype]

    for name in TABLES:
        union = sorted(set().union(*(replica[name] for replica in before)))
        assert union, f"{name}: this shard holds no key; nothing was compared"
        mine = before[dist.get_rank(_replica_group(model))][name]
        assert stats[name].union_keys == len(union), (name, stats[name], union)
        assert stats[name].local_keys == len(mine), (name, stats[name], mine)
        for replica in after:
            assert sorted(replica[name]) == union, (name, sorted(replica[name]))
        for key in union:
            held = [
                replica[name].get(key, ([fresh_emb] * EMBEDDING_DIM, fresh_opt, 0))
                for replica in before
            ]
            emb = torch.tensor([row[0] for row in held]).sum(0) / replica_count
            opt = torch.tensor([row[1] for row in held]).sum(0) / replica_count
            frequency = max(1, sum(row[2] for row in held) // replica_count)
            got_emb, got_opt, got_frequency = after[0][name][key]
            torch.testing.assert_close(
                torch.tensor(got_emb).cuda(),
                _at_table_precision(emb, dtype),
                rtol=0,
                atol=tolerance,
            )
            torch.testing.assert_close(torch.tensor(got_opt), opt, rtol=1e-6, atol=1e-9)
            if args.score_strategy == "lru_lfu":
                assert got_frequency == frequency, (name, key, got_frequency, held)
            for replica in after[1:]:
                assert replica[name][key] == after[0][name][key], (name, key)

    again = sync.sync()
    assert all(table.union_keys == 0 for table in again), again


def check_checkpoint(args) -> None:
    model = _build(args)
    _train(model)
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
    parser.add_argument("--embedding-dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--score-strategy", choices=sorted(SCORES), default="lru_lfu")
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZERS), default="adagrad")
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
        )
    MODES[args.mode](args)
    dist.barrier()
    if rank == 0:
        print(f"[{args.mode}] passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
