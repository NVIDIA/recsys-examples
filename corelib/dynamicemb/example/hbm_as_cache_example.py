import warnings
from typing import Any, Dict

# Filter FBGEMM warning, make notebook clean
warnings.filterwarnings(
    "ignore", message=".*torch.library.impl_abstract.*", category=FutureWarning
)
import numpy as np
import torch
import torch.distributed as dist
import torchrec
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
)
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingCollectionSharder
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.types import BoundsCheckMode, ShardingType

backend = "nccl"
dist.init_process_group(backend=backend)

local_rank = dist.get_rank()  # for one node
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
np.random.seed(1024 + local_rank)
torch.manual_seed(1024)  # same seed for all processes

# Define the configuration parameters for the embedding table,
# including its name, embedding dimension, total number of embeddings, and feature name.
embedding_table_name = "table_0"
embedding_table_dim = 128
total_num_embedding = 1024 * 1024  # 1M
embedding_feature_name = "cate_0"
batch_size = 16

eb_configs = [
    torchrec.EmbeddingConfig(
        name=embedding_table_name,
        embedding_dim=embedding_table_dim,
        num_embeddings=total_num_embedding,
        feature_names=[embedding_feature_name],
    )
]

embedding_collection = torchrec.EmbeddingCollection(
    device=torch.device("meta"),
    tables=eb_configs,
)


# Use a function to wrap all the Planner code
def get_planner(device, eb_configs, batch_size):
    dict_const = {}
    cache_ratio = 0.05  # assume we will use 5% of the HBM for cache
    cache_size_in_bytes = (
        embedding_table_dim * total_num_embedding * cache_ratio * 3
    )  # 1 embedding + 2 optimizer states
    const = DynamicEmbParameterConstraints(
        sharding_types=[
            ShardingType.ROW_WISE.value,
        ],
        bounds_check_mode=BoundsCheckMode.FATAL,
        use_dynamicemb=True,  # This is a must to enable dynamic embedding.
        dynamicemb_options=DynamicEmbTableOptions(
            global_hbm_for_values=cache_size_in_bytes,  # when caching is True, global_hbm_for_values refers to the cache size
            caching=True,
            training=True,  # Setting training to True to allocate optimizer states if existing
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.NORMAL
            ),
        ),
    )
    dict_const[embedding_table_name] = const
    return DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        constraints=dict_const,
        batch_size=batch_size,
    )


planner = get_planner(device, eb_configs, batch_size)
fused_params: Dict[str, Any] = {}

# specify the embedding optimizer type
fused_params["optimizer"] = EmbOptimType.ADAM
fused_params["learning_rate"] = 1e-3
fused_params["prefetch_pipeline"] = True  # enable prefetch

sharder = DynamicEmbeddingCollectionSharder(
    use_index_dedup=True, fused_params=fused_params
)
plan = planner.collective_plan(embedding_collection, [sharder], dist.GroupMember.WORLD)
model = DistributedModelParallel(
    module=embedding_collection,
    device=device,
    sharders=[sharder],
    plan=plan,
)

# Generate test data
dp_rank = dist.get_rank()

if dp_rank == 0:
    values = torch.randint(0, 1024, (2**10,)).cuda()
    lengths = torch.zeros(2**28, dtype=torch.int64).cuda()
    lengths[0 : 2**10].fill_(1)
else:
    values = torch.randint(0, 1024, (2**10 + 1,)).cuda()
    lengths = torch.zeros(2**28 + 1, dtype=torch.int64).cuda()
    lengths[0 : 2**10 + 1].fill_(1)

torch.cuda.synchronize()
torch.distributed.barrier()

mb = torchrec.KeyedJaggedTensor(
    keys=[embedding_feature_name],
    values=values,  # key [] on rank0, [2] on rank 1
    lengths=lengths,  # length [] on rank0, 1 on rank 1
)
sdbe = model.module

# sdbe.prefetch([mb]) # prefetch is done locally. After dist a2a.

ret = model(mb)  # this is awaitable
print(f"rank {dp_rank} ret = {ret[embedding_feature_name].values().shape}")
