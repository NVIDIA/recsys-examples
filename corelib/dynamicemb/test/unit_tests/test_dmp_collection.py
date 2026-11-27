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

"""A DynamicEmb table under TorchRec 2D sparse parallelism (DMPCollection).

DMPCollection shards each table over a sharding group and replicates that group
across the world. Its plan names shards by global rank, while the kernels get
the sharding group as their process group, so a rank's place in that group is
not its placement. Run on two GPUs with a sharding group of one: the two ranks
are two replicas of the same table, and rank 1's group rank is 0.
"""

import torch
import torch.distributed as dist
import torchrec
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
)
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingBagCollectionSharder
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from torch.distributed.elastic.multiprocessing.errors import record
from torchrec.distributed.model_parallel import DMPCollection
from torchrec.distributed.planner import Topology
from torchrec.distributed.types import BoundsCheckMode, ShardingType

TABLE = "t"
FEATURE = "f_t"
NUM_EMBEDDINGS = 1_000_000
EMBEDDING_DIM = 16
NUM_KEYS = 64


def _build(device: torch.device) -> DMPCollection:
    ebc = torchrec.EmbeddingBagCollection(
        device=torch.device("meta"),
        tables=[
            torchrec.EmbeddingBagConfig(
                name=TABLE,
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=NUM_EMBEDDINGS,
                feature_names=[FEATURE],
                pooling=torchrec.PoolingType.SUM,
            )
        ],
    )
    constraints = {
        TABLE: DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            bounds_check_mode=BoundsCheckMode.NONE,
            enforce_hbm=True,
            use_dynamicemb=True,
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=1024**3,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.DEBUG
                ),
                safe_check_mode=DynamicEmbCheckMode.WARNING,
            ),
        )
    }
    sharder = DynamicEmbeddingBagCollectionSharder(
        fused_params={
            "output_dtype": SparseType.FP32,
            "optimizer": EmbOptimType.EXACT_SGD,
            "learning_rate": 0.1,
        }
    )
    # The plan is for one sharding group, here of one rank; DMPCollection
    # replicates it and remaps its placements to each replica's global ranks.
    planner = DynamicEmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=1,
            world_size=1,
            compute_device=device.type,
            hbm_cap=80 * 1024**3,
        ),
        constraints=constraints,
        batch_size=NUM_KEYS,
        debug=False,
    )
    plan = planner.collective_plan(ebc, [sharder], dist.GroupMember.WORLD)
    return DMPCollection(
        module=ebc,
        device=device,
        plan=plan,
        world_size=dist.get_world_size(),
        sharding_group_size=1,
        global_pg=dist.GroupMember.WORLD,
        node_group_size=1,
        sharders=[sharder],
        use_inter_host_allreduce=True,
    )


def check_replicas_agree(model: DMPCollection, device: torch.device) -> None:
    # Both replicas look up the same keys. With DEBUG initialization a row is a
    # function of its key alone, so two working replicas return the same rows.
    kjt = torchrec.KeyedJaggedTensor(
        keys=[FEATURE],
        values=torch.arange(NUM_KEYS, dtype=torch.int64, device=device),
        lengths=torch.ones(NUM_KEYS, dtype=torch.int64, device=device),
    )
    out = model(kjt).to_dict()[FEATURE]  # (NUM_KEYS, EMBEDDING_DIM)
    assert out.abs().sum() > 0, "the lookup returned zeros; nothing was compared"
    gathered = [torch.empty_like(out) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, out)
    for replica, other in enumerate(gathered[1:], start=1):
        torch.testing.assert_close(
            other, gathered[0], msg=lambda m: f"replica {replica} vs 0:\n{m}"
        )


@record
def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() >= 2, "needs two ranks: one would be no replica"
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Before the fix, every replica but the first raised here: get_state_dict
    # looked its shard up by its rank in the sharding group.
    model = _build(device)
    check_replicas_agree(model, device)

    dist.barrier()
    if rank == 0:
        print("DMPCollection: every replica builds and serves the same rows")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
