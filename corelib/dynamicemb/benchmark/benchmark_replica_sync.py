# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Effective bandwidth of DynamicEmbReplicaSync on two replicas of S shards."""

import argparse
import json
import math
import socket
import statistics
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torchrec
from benchmark_utils import init_distributed
from dynamicemb import (
    DynamicEmbReplicaSync,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    get_table_value_bytes,
)
from dynamicemb.dump_load import get_dynamic_emb_module
from dynamicemb.key_value_table import (
    DynamicEmbStorage,
    get_find_score_arg,
    load_from_flat_single_table,
)
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingBagCollectionSharder
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed.model_parallel import DMPCollection
from torchrec.distributed.planner import Topology
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import DataType


def build_model(rows, dim, dtype, shard_count, local_world_size, device, num_tables):
    tables = [
        torchrec.EmbeddingBagConfig(
            name=f"t{i}",
            feature_names=[f"f{i}"],
            embedding_dim=dim,
            num_embeddings=2 * rows * shard_count,
            data_type=DataType.FP32 if dtype == torch.float32 else DataType.FP16,
        )
        for i in range(num_tables)
    ]
    module = torchrec.EmbeddingBagCollection(tables=tables, device=torch.device("meta"))
    optimizer = EmbOptimType.EXACT_SGD
    sharder = DynamicEmbeddingBagCollectionSharder(
        fused_params={"optimizer": optimizer}
    )
    constraints = {
        table.name: DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            use_dynamicemb=True,
            enforce_hbm=True,
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=get_table_value_bytes(
                    table, optimizer, shard_count
                ),
                score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
                embedding_dtype=dtype,
                dist_type="roundrobin",
            ),
        )
        for table in tables
    }
    planner = DynamicEmbeddingShardingPlanner(
        topology=Topology(
            world_size=shard_count,
            local_world_size=min(shard_count, local_world_size),
            compute_device="cuda",
            hbm_cap=torch.cuda.get_device_properties(device).total_memory,
        ),
        constraints=constraints,
        batch_size=1,
        debug=False,
    )
    return DMPCollection(
        module=module,
        plan=planner.collective_plan(module, [sharder], dist.group.WORLD),
        device=device,
        sharders=[sharder],
        world_size=dist.get_world_size(),
        sharding_group_size=shard_count,
        node_group_size=min(shard_count, local_world_size),
        global_pg=dist.group.WORLD,
        use_inter_host_allreduce=True,
    )


def timed(device, operation, *args, **kwargs):
    torch.cuda.synchronize()
    dist.barrier(device_ids=[device.index])
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = operation(*args, **kwargs)
    torch.cuda.synchronize()
    # Include CPU dispatch, count reads and all kernels on the critical rank.
    seconds = torch.tensor(
        time.perf_counter() - start, dtype=torch.float64, device=device
    )
    dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
    return result, seconds.item()


@dataclass
class TableFixture:
    storage: DynamicEmbStorage
    table_id: int
    keys: torch.Tensor
    ids: torch.Tensor
    insert_keys: torch.Tensor
    insert_values: torch.Tensor
    expected: torch.Tensor

    def write(self):
        self.insert_values.neg_()
        self.expected.neg_()
        self.storage.insert(self.insert_keys, self.ids, self.insert_values)

    def check(self):
        state = self.storage.tables[0]
        for start in range(0, self.keys.numel(), 8192):
            batch = slice(start, start + 8192)
            keys = self.keys[batch]
            score_arg = get_find_score_arg(
                state, keys.numel(), keys.device, const_lookup=True
            )
            _, found, rows = state.key_index_map.lookup(
                keys, self.ids[batch], score_arg
            )
            if not found.all().item():
                raise RuntimeError(f"Missing rows in table {self.table_id}")
            actual = load_from_flat_single_table(state, rows, self.table_id)
            torch.testing.assert_close(actual, self.expected[batch], rtol=0, atol=0)


def fixtures(model, args, dtype, shard, shard_count, replica, device):
    # Every table has the same keys but distinct row values. Reversing insertion
    # order on one replica also tests the correspondence of physical slots.
    result = {}
    generator = torch.Generator(device=device)
    for kernel in get_dynamic_emb_module(model):
        storage = kernel.tables
        state = storage.tables[0]
        keys = (
            torch.arange(args.rows, dtype=state.key_index_map.key_type, device=device)
            * shard_count
            + shard
        )
        for table_id, name in enumerate(kernel.table_names):
            seed = 2 * (shard * args.tables + int(name[1:]))
            local, peer = [
                torch.rand(
                    (args.rows, args.dim),
                    generator=generator.manual_seed(seed + r),
                    dtype=dtype,
                    device=device,
                )
                for r in (replica, 1 - replica)
            ]
            expected = ((local.float() + peer.float()) * 0.5).to(dtype)
            del peer
            result[name] = TableFixture(
                storage,
                table_id,
                keys,
                torch.full_like(keys, table_id, dtype=torch.int64),
                keys.flip(0) if replica else keys,
                local.flip(0) if replica else local,
                expected,
            )
            del local
    if len(result) != args.tables:
        raise RuntimeError(f"Expected {args.tables} tables, found {list(result)}")
    return result


def run(args):
    rank, world, local_rank, local_world = init_distributed()
    if world < 2 or world % 2:
        raise ValueError("Use two replicas of equal size (an even world size).")
    device = torch.device("cuda", local_rank)
    shard_count = world // 2
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = build_model(
        args.rows, args.dim, dtype, shard_count, local_world, device, args.tables
    )
    sync = (
        DynamicEmbReplicaSync(model, max_staging_bytes=args.staging_mib * 1024**2)
        if args.staging_mib is not None
        else DynamicEmbReplicaSync(model)
    )
    group = model.device_mesh.get_group("replicate")
    replica = dist.get_rank(group)
    shard = dist.get_rank(model.device_mesh.get_group("shard"))
    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())
    pairs = [None] * world
    dist.all_gather_object(pairs, dist.get_process_group_ranks(group))
    pairs = sorted({tuple(pair) for pair in pairs})
    if args.fabric == "nvlink" and any(hosts[a] != hosts[b] for a, b in pairs):
        raise ValueError(
            f"This NVLink benchmark needs each pair on one host: {pairs}, {hosts}"
        )
    if args.fabric == "ib" and any(hosts[a] == hosts[b] for a, b in pairs):
        raise ValueError(
            f"IB needs each replica pair on separate hosts: {pairs}, {hosts}"
        )

    # Only embedding values are useful wire payload in this SGD/TIMESTAMP case.
    # Both ranks hold all touched keys and send one row for each to their peer.
    local = torch.rand(args.tables * args.rows, args.dim, dtype=dtype, device=device)
    payload_bytes = local.numel() * local.element_size()
    output = torch.empty(2 * local.size(0), args.dim, dtype=dtype, device=device)
    baseline = []
    for step in range(args.warmup + args.iterations):
        _, elapsed = timed(
            device, dist.all_gather_into_tensor, output, local, group=group
        )
        if step >= args.warmup:
            baseline.append(elapsed)
    del output, local

    tables = fixtures(model, args, dtype, shard, shard_count, replica, device)
    samples = []
    for step in range(args.warmup + args.iterations):
        for table in tables.values():
            table.write()
        stats, elapsed = timed(device, sync.sync)
        if step >= args.warmup:
            samples.append(elapsed)
        counts = {stat.table_name: (stat.union_keys, stat.local_keys) for stat in stats}
        if len(stats) != len(tables) or counts != dict.fromkeys(
            tables, (args.rows, args.rows)
        ):
            raise RuntimeError(f"rank {rank}: incomplete sync at step {step}: {stats}")
        for table in tables.values():
            table.check()

    elapsed_s = statistics.mean(samples)
    effective = payload_bytes / elapsed_s / 1e9
    utilization = effective / args.peak_gbps
    if rank == 0:
        print(
            json.dumps(
                {
                    "fabric": args.fabric,
                    "hosts_by_rank": hosts,
                    "replica_pairs": pairs,
                    "device": torch.cuda.get_device_name(device),
                    "dtype": args.dtype,
                    "rows_per_shard": args.rows,
                    "tables": args.tables,
                    "staging_mib_override": args.staging_mib,
                    "embedding_dim": args.dim,
                    "useful_bytes_per_gpu_one_direction": payload_bytes,
                    "sync_max_rank_ms": [s * 1e3 for s in samples],
                    "payload_only_nccl_GB_s": payload_bytes
                    / statistics.mean(baseline)
                    / 1e9,
                    "sync_effective_GB_s": effective,
                    "physical_peak_per_gpu_one_direction_GB_s": args.peak_gbps,
                    "sync_utilization": utilization,
                    "passed": utilization > 0.80,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()
    if utilization <= 0.80:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=int, default=131072, help="Touched keys per shard"
    )
    parser.add_argument(
        "--staging-mib", type=int, help="Maximum payload staging budget per GPU in MiB"
    )
    parser.add_argument(
        "--tables",
        type=int,
        default=1,
        help="Number of tables in each measured sync call",
    )
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--fabric", choices=["nvlink", "ib"], required=True)
    parser.add_argument(
        "--peak-gbps",
        type=float,
        required=True,
        help="Physical one-direction GB/s per GPU (decimal gigabytes/s)",
    )
    args = parser.parse_args()
    if min(args.rows, args.tables, args.dim, args.iterations, args.warmup) <= 0:
        parser.error("rows, tables, dim, iterations, and warmup must be positive")
    if args.staging_mib is not None and args.staging_mib <= 0:
        parser.error("staging-mib must be positive")
    if not math.isfinite(args.peak_gbps) or args.peak_gbps <= 0:
        parser.error("peak-gbps must be positive and finite")
    run(args)
