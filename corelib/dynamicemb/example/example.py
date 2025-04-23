import warnings

# Filter FBGEMM warning, make notebook clean
warnings.filterwarnings(
    "ignore", message=".*torch.library.impl_abstract.*", category=FutureWarning
)

import numpy as np
import torch
import torch.distributed as dist
import torchrec

backend = "nccl"
dist.init_process_group(backend=backend)

local_rank = dist.get_rank()  # for one node
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
np.random.seed(1024 + local_rank)

# Define the configuration parameters for the embedding table,
# including its name, embedding dimension, total number of embeddings, and feature name.
embedding_table_name = "table_0"
embedding_table_dim = 128
total_num_embedding = 1000
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

ebc = torchrec.EmbeddingCollection(
    device=torch.device("meta"),
    tables=eb_configs,
)

import math

from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
)
from dynamicemb.planner import (
    DynamicEmbeddingEnumerator,
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from torchrec import DataType
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner import Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.types import BoundsCheckMode, ShardingType


# use a function warp all the Planner code
def get_planner(device, eb_configs, batch_size):
    DATA_TYPE_NUM_BITS: Dict[DataType, int] = {
        DataType.FP32: 32,
        DataType.FP16: 16,
        DataType.BF16: 16,
    }

    # For HVK  embedding table , need to calculate how many bytes of embedding vector store in GPU HBM
    # In this case , we will put all the embedding vector into GPU HBM
    eb_config = eb_configs[0]
    dim = eb_config.embedding_dim
    tmp_type = eb_config.data_type

    embedding_type_bytes = DATA_TYPE_NUM_BITS[tmp_type] / 8
    emb_num_embeddings = eb_config.num_embeddings
    emb_num_embeddings_next_power_of_2 = 2 ** math.ceil(
        math.log2(emb_num_embeddings)
    )  # HKV need embedding vector num is power of 2
    total_hbm_need = embedding_type_bytes * dim * emb_num_embeddings_next_power_of_2

    hbm_cap = 80 * 1024 * 1024 * 1024  # H100's HBM bytes per GPU
    ddr_cap = 512 * 1024 * 1024 * 1024  # Assume a Node have 512GB memory
    intra_host_bw = 450e9  # Nvlink bandwidth
    inter_host_bw = 25e9  # NIC bandwidth

    dict_const = {}

    const = DynamicEmbParameterConstraints(
        sharding_types=[
            ShardingType.ROW_WISE.value,
        ],
        enforce_hbm=True,
        bounds_check_mode=BoundsCheckMode.NONE,
        use_dynamicemb=True,  # from here , is all the HKV options , default use_dynamicemb is False , if it is False , it will fallback to raw TorchREC ParameterConstraints
        dynamicemb_options=DynamicEmbTableOptions(
            global_hbm_for_values=total_hbm_need,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.NORMAL
            ),
        ),
    )

    dict_const[embedding_table_name] = const
    topology = Topology(
        local_world_size=get_local_size(),
        world_size=dist.get_world_size(),
        compute_device=device.type,
        hbm_cap=hbm_cap,
        ddr_cap=ddr_cap,  # For HVK  , if we need to put embedding vector into Host memory , it is important set ddr capacity
        intra_host_bw=intra_host_bw,
        inter_host_bw=inter_host_bw,
    )

    # Same usage of  TorchREC's EmbeddingEnumerator
    enumerator = DynamicEmbeddingEnumerator(
        topology=topology,
        constraints=dict_const,
    )

    # Almost same usage of  TorchREC's EmbeddingShardingPlanner , but we need to input eb_configs, so we can plan every GPU's HKV object.
    return DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=topology,
        constraints=dict_const,
        batch_size=batch_size,
        enumerator=enumerator,
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        debug=True,
    )


planner = get_planner(device, eb_configs, batch_size)

from dynamicemb.shard import DynamicEmbeddingCollectionSharder
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    QCommsConfig,
    get_qcomm_codecs_registry,
)
from torchrec.distributed.model_parallel import (
    DefaultDataParallelWrapper,
    DistributedModelParallel,
)

# set optimizer args
learning_rate = 0.1
beta1 = 0.9
beta2 = 0.999
weight_decay = 0
eps = 0.001

# Put args into a optimizer kwargs , which is same usage of TorchREC
optimizer_kwargs = {
    "optimizer": EmbOptimType.ADAM,
    "learning_rate": learning_rate,
    "beta1": beta1,
    "beta2": beta2,
    "weight_decay": weight_decay,
    "eps": eps,
}

fused_params = {}
fused_params["output_dtype"] = SparseType.FP32
fused_params.update(optimizer_kwargs)

qcomm_codecs_registry = (
    get_qcomm_codecs_registry(
        qcomms_config=QCommsConfig(
            # pyre-ignore
            forward_precision=CommType.FP32,
            # pyre-ignore
            backward_precision=CommType.FP32,
        )
    )
    if backend == "nccl"
    else None
)

# Create a sharder , same usage with TorchREC , but need Use DynamicEmb function, because for index_dedup
# DynamicEmb overload this process to fit HKV
sharder = DynamicEmbeddingCollectionSharder(
    qcomm_codecs_registry=qcomm_codecs_registry,
    fused_params=fused_params,
    use_index_dedup=True,
)

# Same usage of TorchREC
plan = planner.collective_plan(ebc, [sharder], dist.GroupMember.WORLD)

data_parallel_wrapper = DefaultDataParallelWrapper(allreduce_comm_precision="fp16")

# Same usage of TorchREC
model = DistributedModelParallel(
    module=ebc,
    device=device,
    # pyre-ignore
    sharders=[sharder],
    plan=plan,
    data_parallel_wrapper=data_parallel_wrapper,
)

print(model)

import numpy as np

num_iterations = 10


# This function generate a random indice to lookup
def generate_sparse_feature(
    feature_num, num_embeddings_list, max_sequence_size, local_batch_size=50
):
    prefix_sums = np.zeros(feature_num, dtype=int)
    for f in range(1, feature_num):
        prefix_sums[f] = prefix_sums[f - 1] + num_embeddings_list[f - 1]

    indices = []
    lengths = []

    for f in range(feature_num):
        unique_indices = np.random.choice(
            num_embeddings_list[f],
            size=(local_batch_size, max_sequence_size[f]),
            replace=True,
        )
        adjusted_indices = unique_indices
        indices.extend(adjusted_indices.flatten())
        lengths.extend([max_sequence_size[f]] * local_batch_size)

    return torchrec.KeyedJaggedTensor(
        keys=[f"cate_{feature_idx}" for feature_idx in range(feature_num)],
        values=torch.tensor(indices, dtype=torch.int64).cuda(),
        lengths=torch.tensor(lengths, dtype=torch.int64).cuda(),
    )


sparse_features = []
for i in range(num_iterations):
    sparse_features.append(
        generate_sparse_feature(
            feature_num=1,
            num_embeddings_list=[total_num_embedding],
            max_sequence_size=[10],
            local_batch_size=batch_size // world_size,
        )
    )

for i in range(num_iterations):
    sparse_feature = sparse_features[i]
    ret = model(sparse_feature)

    feature_names = []
    tensors = []
    for k, v in ret.items():
        feature_names.append(k)
        tensors.append(v.values())

    cat_tensor = torch.cat(tensors, dim=1)
    print(f"iter : {i} , cat_tensor = {cat_tensor}")
    loss = cat_tensor.sum()
    loss.backward()
