# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weighted EmbeddingBagCollection end-to-end with DynamicEmb tables (torchrun).

The DEBUG initializer fills row i with i % 100_000, so the weighted-sum output
has an exact reference computed from the local pre-all2all features."""

import argparse
import os
import random
from typing import List, Tuple

import torch
import torch.distributed as dist
import torchrec
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
)
from dynamicemb.dynamicemb_config import DEBUG_EMB_INITIALIZER_MOD
from dynamicemb.planner import (
    DynamicEmbeddingEnumerator,
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingBagCollectionSharder
from torch.distributed.elastic.multiprocessing.errors import record
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.model_parallel import (
    DefaultDataParallelWrapper,
    DistributedModelParallel,
)
from torchrec.distributed.planner import Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.types import BoundsCheckMode, ShardingType
from torchrec.modules.embedding_configs import PoolingType


def table_idx_to_name(i):
    return f"t_{i}"


def feature_idx_to_name(i):
    return f"cate_{i}"


def generate_sparse_feature_with_weights(
    feature_num, num_embeddings_list, multi_hot_sizes, local_batch_size
) -> torchrec.KeyedJaggedTensor:
    feature_batch = feature_num * local_batch_size
    indices, lengths, weights = [], [], []
    for i in range(feature_batch):
        f = i // local_batch_size
        cur_bag_size = random.randint(1, multi_hot_sizes[f])
        cur_bag = set({})
        while len(cur_bag) < cur_bag_size:
            cur_bag.add(random.randint(0, num_embeddings_list[f] - 1))
        for idx in cur_bag:
            indices.append(idx)
            weights.append(random.uniform(0.5, 2.0))
        lengths.append(cur_bag_size)
    keys = [feature_idx_to_name(f) for f in range(feature_num)]
    return torchrec.KeyedJaggedTensor(
        keys=keys,
        values=torch.tensor(indices, dtype=torch.int64).cuda(),
        weights=torch.tensor(weights, dtype=torch.float32).cuda(),
        lengths=torch.tensor(lengths, dtype=torch.int64).cuda(),
    )

def reference_weighted_sum(features, local_batch_size):
    """Reference per local slot: slot s = f*B + b -> sum w * (idx % MOD)."""
    F = len(features.keys())
    out = torch.zeros(local_batch_size, F, device=features.values().device)
    offsets = features.offsets().view(-1)
    w_values = features.weights()
    for s in range(F * local_batch_size):
        f, b = s // local_batch_size, s % local_batch_size
        acc = 0.0
        for p in range(int(offsets[s]), int(offsets[s + 1])):
            acc += float(w_values[p]) * float(
                int(features.values()[p]) % DEBUG_EMB_INITIALIZER_MOD
            )
        out[b, f] = acc
    return out

@record
def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Weighted EmbeddingBagCollection example with DynamicEmb"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=8)
    parser.add_argument("--num_embedding_table", type=int, default=2)
    parser.add_argument("--multi_hot_sizes", type=str, default="4,4")
    parser.add_argument("--num_embeddings_per_feature", type=str, default="1024,2048")
    parser.add_argument("--num_iterations", type=int, default=2)
    args = parser.parse_args(argv)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    multi_hot_sizes = [int(x) for x in args.multi_hot_sizes.split(",")]
    num_embeddings_per_feature = [
        int(x) for x in args.num_embeddings_per_feature.split(",")
    ]
    assert args.num_embedding_table == len(multi_hot_sizes) == len(num_embeddings_per_feature)

    # NOTE: `weighted=True` is the torchrec EmbeddingBagConfig field; weighted
    # tables require pooling=SUM. Adjust the flag name if the installed
    # torchrec release differs.
    eb_configs = [
        torchrec.EmbeddingBagConfig(
            name=table_idx_to_name(f),
            embedding_dim=args.embedding_dim,
            num_embeddings=num_embeddings_per_feature[f],
            feature_names=[feature_idx_to_name(f)],
            pooling=PoolingType.SUM,
        )
        for f in range(args.num_embedding_table)
    ]
    ebc = torchrec.EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))

    dict_const = {}
    for i in range(args.num_embedding_table):
        dict_const[table_idx_to_name(i)] = DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            pooling_factors=[multi_hot_sizes[i]],
            num_poolings=[1],
            enforce_hbm=True,
            bounds_check_mode=BoundsCheckMode.NONE,
            use_dynamicemb=True,
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=1024**3,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.DEBUG
                ),
                safe_check_mode=DynamicEmbCheckMode.IGNORE,
            ),
        )

    topology = Topology(
        local_world_size=get_local_size(),
        world_size=world_size,
        compute_device=device.type,
        hbm_cap=80 * 1024**3,
        ddr_cap=1024**4,
        intra_host_bw=300e9,
        inter_host_bw=25e9,
    )
    enumerator = DynamicEmbeddingEnumerator(
        topology=topology, batch_size=args.batch_size, constraints=dict_const
    )
    planner = DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=topology,
        constraints=dict_const,
        batch_size=args.batch_size,
        enumerator=enumerator,
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        debug=True,
    )

    sharder = DynamicEmbeddingBagCollectionSharder(fused_params={})
    plan = planner.collective_plan(ebc, [sharder], dist.GroupMember.WORLD)
    model = DistributedModelParallel(
        module=ebc,
        device=device,
        sharders=[sharder],
        plan=plan,
        data_parallel_wrapper=DefaultDataParallelWrapper(),
    )

    local_batch_size = args.batch_size // world_size
    prefix_dims = [0] + [
        args.embedding_dim * i for i in range(1, args.num_embedding_table + 1)
    ]

    for _ in range(args.num_iterations):
        features = generate_sparse_feature_with_weights(
            args.num_embedding_table,
            num_embeddings_per_feature,
            multi_hot_sizes,
            local_batch_size,
        )
        ret = model(features)
        kt = ret.values()  # [B_local, F * embedding_dim]
        ref = reference_weighted_sum(features, local_batch_size)

        for f in range(args.num_embedding_table):
            col = prefix_dims[f]
            for b in range(local_batch_size):
                assert torch.allclose(
                    kt[b, col].float(), ref[b, f], rtol=1e-4, atol=1e-2
                ), (
                    f"rank {local_rank} f={f} b={b}: "
                    f"{kt[b, col].item()} vs {ref[b, f].item()}"
                )

        loss = kt.sum()
        loss.backward()
        torch.cuda.synchronize()

        if local_rank == 0:
            print(f"Weighted DynamicEmb iteration {_ + 1} Passed")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
