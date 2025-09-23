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

import argparse
import os
import random
import sys
from typing import List

import torch
import torch.distributed as dist
import torchrec
from debug import Debugger
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
)
from dynamicemb.planner import (
    DynamicEmbeddingEnumerator,
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingCollectionSharder
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    QCommsConfig,
    get_qcomm_codecs_registry,
)
from torchrec.distributed.model_parallel import (
    DefaultDataParallelWrapper,
    DistributedModelParallel,
)
from torchrec.distributed.planner import ParameterConstraints, Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.types import BoundsCheckMode, ShardingType


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def table_idx_to_name(i):
    return f"t_{i}"


def feature_idx_to_name(i):
    return f"cate_{i}"


def get_comm_precission(precision_str):
    if precision_str == "fp32":
        return CommType.FP32
    elif precision_str == "fp16":
        return CommType.FP16
    elif precision_str == "bf16":
        return CommType.BF16
    elif precision_str == "fp8":
        return CommType.FP8
    else:
        raise ValueError("unknown comm precision type")


def get_planner(args, device, eb_configs):
    dict_const = {}
    for i in range(args.num_embedding_table):
        if (
            args.data_parallel_embeddings is not None
            and i in args.data_parallel_embeddings
        ):
            const = ParameterConstraints(
                sharding_types=[ShardingType.DATA_PARALLEL.value],
                pooling_factors=[args.multi_hot_sizes[i]],
                num_poolings=[1],
                enforce_hbm=True,
                bounds_check_mode=BoundsCheckMode.NONE,
            )
        else:
            use_dynamicemb = True if i < args.num_dynamicemb else False
            const = DynamicEmbParameterConstraints(
                sharding_types=[
                    ShardingType.ROW_WISE.value,
                ],
                pooling_factors=[args.multi_hot_sizes[i]],
                num_poolings=[1],
                enforce_hbm=True,
                bounds_check_mode=BoundsCheckMode.NONE,
                use_dynamicemb=use_dynamicemb,
                dynamicemb_options=DynamicEmbTableOptions(
                    global_hbm_for_values=1024**3,
                    initializer_args=DynamicEmbInitializerArgs(
                        mode=DynamicEmbInitializerMode.DEBUG
                    ),
                    safe_check_mode=DynamicEmbCheckMode.WARNING,
                ),
            )
        dict_const[table_idx_to_name(i)] = const

    topology = Topology(
        local_world_size=get_local_size(),
        world_size=dist.get_world_size(),
        compute_device=device.type,
        hbm_cap=args.hbm_cap,
        ddr_cap=1024 * 1024 * 1024 * 1024,
        intra_host_bw=args.intra_host_bw,
        inter_host_bw=args.inter_host_bw,
    )

    enumerator = DynamicEmbeddingEnumerator(
        topology=topology,
        constraints=dict_const,
    )

    return DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=topology,
        constraints=dict_const,
        batch_size=args.batch_size,
        enumerator=enumerator,
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        debug=True,
    )


def init_fn(x: torch.Tensor):
    with torch.no_grad():
        x.fill_(2.0)


def generate_sparse_feature(
    feature_num, num_embeddings_list, multi_hot_sizes, local_batch_size=50, offset=0
):
    feature_batch = feature_num * local_batch_size

    indices = []
    lengths = []

    for i in range(feature_batch):
        f = i // local_batch_size
        cur_bag_size = random.randint(0, multi_hot_sizes[f])
        cur_bag = set({})
        while len(cur_bag) < cur_bag_size:
            cur_bag.add(random.randint(0, num_embeddings_list[f] - 1))

        indices.extend(list(cur_bag))
        lengths.append(cur_bag_size)

    return torchrec.KeyedJaggedTensor(
        keys=[
            feature_idx_to_name(feature_idx + offset)
            for feature_idx in range(feature_num)
        ],
        values=torch.tensor(
            indices, dtype=torch.int64
        ).cuda(),  # key [0,1] on rank0, [2] on rank 1
        lengths=torch.tensor(lengths, dtype=torch.int64).cuda(),
    )


class MyModel(torch.nn.Module):
    def __init__(
        self, eb_configs: List[torchrec.EmbeddingConfig], num_collection: int = 2
    ):
        super().__init__()

        mid = len(eb_configs) // 2
        first_half = eb_configs[:mid]
        second_half = eb_configs[mid:]
        self.ebc1 = torchrec.EmbeddingCollection(
            device=torch.device("meta"),
            tables=first_half,
        )
        self.ebc2 = torchrec.EmbeddingCollection(
            device=torch.device("meta"),
            tables=second_half,
        )

    def forward(
        self,
        features1: torchrec.KeyedJaggedTensor,
        features2: torchrec.KeyedJaggedTensor,
    ):
        return [self.ebc1(features1), self.ebc2(features2)]


def run(args):
    backend = "nccl"
    dist.init_process_group(backend=backend)

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    eb_configs = [
        torchrec.EmbeddingConfig(
            name=table_idx_to_name(feature_idx),
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings_per_feature[feature_idx],
            feature_names=[feature_idx_to_name(feature_idx)],
        )
        for feature_idx in range(args.num_embedding_table)
    ]
    ebcs = MyModel(eb_configs, args.num_collection)

    if args.use_torch_opt:
        optimizer_kwargs = {
            "lr": args.learning_rate,
            "betas": (args.beta1, args.beta2),
            "weight_decay": args.weight_decay,
            "eps": args.eps,
        }
        if args.optimizer_type == "sgd":
            embedding_optimizer = torch.optim.SGD
        elif args.optimizer_type == "adam":
            embedding_optimizer = torch.optim.Adam
        else:
            raise ValueError("unknown optimizer type")
    else:
        optimizer_kwargs = {
            "learning_rate": args.learning_rate,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "weight_decay": args.weight_decay,
            "eps": args.eps,
        }
        if args.optimizer_type == "sgd":
            optimizer_kwargs["optimizer"] = EmbOptimType.EXACT_SGD
        elif args.optimizer_type == "exact_sgd":
            optimizer_kwargs["optimizer"] = EmbOptimType.EXACT_SGD
        elif args.optimizer_type == "adam":
            optimizer_kwargs["optimizer"] = EmbOptimType.ADAM
        elif args.optimizer_type == "exact_adagrad":
            optimizer_kwargs["optimizer"] = EmbOptimType.EXACT_ADAGRAD
        elif args.optimizer_type == "exact_row_wise_adagrad":
            optimizer_kwargs["optimizer"] = EmbOptimType.EXACT_ROWWISE_ADAGRAD
        else:
            raise ValueError("unknown optimizer type")

    planner = get_planner(args, device, eb_configs)

    qcomm_forward_precision = get_comm_precission(args.fwd_a2a_precision)
    qcomm_backward_precision = get_comm_precission(args.fwd_a2a_precision)
    qcomm_codecs_registry = (
        get_qcomm_codecs_registry(
            qcomms_config=QCommsConfig(
                # pyre-ignore
                forward_precision=qcomm_forward_precision,
                # pyre-ignore
                backward_precision=qcomm_backward_precision,
            )
        )
        if backend == "nccl"
        else None
    )
    # TODO: parse args.output_dtype
    fused_params = {"output_dtype": SparseType.FP32}
    if not args.use_torch_opt:
        fused_params.update(optimizer_kwargs)
    sharder = DynamicEmbeddingCollectionSharder(
        qcomm_codecs_registry=qcomm_codecs_registry,
        fused_params=fused_params,
        use_index_dedup=args.use_index_dedup,
    )
    plan = planner.collective_plan(ebcs, [sharder], dist.GroupMember.WORLD)

    print(f"dynamicemb plan = {plan}")
    if args.use_torch_opt:
        apply_optimizer_in_backward(
            embedding_optimizer,
            ebcs.ebc1.parameters(),
            optimizer_kwargs,
        )
        apply_optimizer_in_backward(
            embedding_optimizer,
            ebcs.ebc2.parameters(),
            optimizer_kwargs,
        )

    data_parallel_wrapper = DefaultDataParallelWrapper(
        allreduce_comm_precision=args.allreduce_precision
    )
    model = DistributedModelParallel(
        module=ebcs,
        device=device,
        # pyre-ignore
        sharders=[sharder],
        plan=plan,
        data_parallel_wrapper=data_parallel_wrapper,
    )

    if local_rank == 0 and args.print_sharding_plan:
        for collectionkey, plans in model._plan.plan.items():
            print(collectionkey)
            for table_name, plan in plans.items():
                print(table_name, "\n", plan, "\n")

    debugger = Debugger()

    for i in range(args.num_iterations):
        assert args.num_embedding_table == len(args.num_embeddings_per_feature)
        assert args.num_embedding_table == len(args.multi_hot_sizes)
        mid = args.num_embedding_table // 2
        feature1 = generate_sparse_feature(
            feature_num=mid,
            num_embeddings_list=args.num_embeddings_per_feature[:mid],
            multi_hot_sizes=args.multi_hot_sizes[:mid],
            local_batch_size=args.batch_size // world_size,
        )
        feature2 = generate_sparse_feature(
            feature_num=args.num_embedding_table - mid,
            num_embeddings_list=args.num_embeddings_per_feature[:mid],
            multi_hot_sizes=args.multi_hot_sizes[:mid],
            local_batch_size=args.batch_size // world_size,
            offset=mid,
        )
        rets = model(feature1, feature2)  # => this is awaitable
        ret1, ret2 = rets[0], rets[1]
        feature1_names, feature2_names = [], []
        jagged_tensors1, jagged_tensors2 = [], []
        for k, v in ret1.items():
            feature1_names.append(k)
            jagged_tensors1.append(v)
        for k, v in ret2.items():
            feature2_names.append(k)
            jagged_tensors2.append(v)

        ret1.values()
        ret2.values()
        feature1_num = len(ret1.keys())
        feature2_num = len(ret2.keys())
        dims1 = [args.embedding_dim for _ in range(feature1_num)]
        dims2 = [args.embedding_dim for _ in range(feature2_num)]

        dyn_emb_features = [feature_idx_to_name(i) for i in range(args.num_dynamicemb)]

        debugger.feature_before_all2all(feature1)
        debugger.sequence_embds_after_all2all(
            jagged_tensors1, feature1_names, dyn_emb_features, dims1[0]
        )
        debugger.feature_before_all2all(feature2)
        debugger.sequence_embds_after_all2all(
            jagged_tensors2, feature2_names, dyn_emb_features, dims2[0]
        )
        print(f"DynamicEmb iteration {i+1} Passed")
    dist.barrier()
    dist.destroy_process_group()

    dict_ = model.state_dict()
    model.load_state_dict(dict_)


@record
def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(
        description="TorchRec EmbeddingCollection example with DynamicEmb"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="number of iterations",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default="65536,32768,4096,8192",
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        default="16,8,20,1",
        # default="20,20,20,20",
        help="Comma separated multihot size per sparse feature. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--print_sharding_plan",
        action="store_true",
        help="Print the sharding plan used for each embedding table.",
    )
    parser.add_argument(
        "--fwd_a2a_precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16", "fp8"],
    )
    parser.add_argument(
        "--bck_a2a_precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16", "fp8"],
    )
    parser.add_argument(
        "--allreduce_precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--dense_in_features",
        type=int,
        default=13,
        help="dense_in_features.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,128",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--dcn_num_layers",
        type=int,
        default=3,
        help="Number of DCN layers in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--dcn_low_rank_dim",
        type=int,
        default=512,
        help="Low rank dimension for DCN in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adam",
        choices=["sgd", "adam", "exact_adagrad", "row_wise_adagrad"],
        help="optimizer type.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate.",
    )

    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="beta1.",
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="beta1.",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=0.001,
        help="eps.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="weight_decay.",
    )
    parser.add_argument(
        "--use_torch_opt",
        action="store_false",
        help="if is true ,use torch register optimizer , or use torchrec",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TensorFloat-32 mode for matrix multiplications on A100 (or newer) GPUs.",
    )
    parser.add_argument(
        "--data_parallel_embeddings",
        type=str,
        default=None,
        help="Comma separated data parallel embedding table ids.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="a100",
        choices=["a100", "h100", "h200"],
        help="Platform, has different system spec",
    )
    parser.add_argument(
        "--bmlp_overlap",
        action="store_true",
        help="overlap bottom mlp",
    )
    parser.add_argument(
        "--enable_cuda_graph",
        action="store_true",
        help="enable cuda_graph",
    )
    parser.add_argument(
        "--skip_h2d",
        action="store_true",
        help="no input to the training pipeline",
    )
    parser.add_argument(
        "--skip_input_dist",
        action="store_true",
        help="skip the input distribution",
    )
    parser.add_argument(
        "--disable_pipeline",
        action="store_true",
        help="disable pipeline",
    )
    parser.add_argument(
        "--output_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16", "fp8"],
    )
    parser.add_argument(
        "--num_dynamicemb",
        type=int,
        default=4,
        help="Number of dynamic embedding tables.",
    )
    parser.add_argument(
        "--use_index_dedup",
        type=str2bool,
        default=True,
        help="Use index deduplication (default: True).",
    )
    parser.add_argument(
        "--num_collection",
        type=int,
        default=2,
        help="Number of torchrec.EmbeddingCollection in DMP",
    )
    args = parser.parse_args()

    args.num_embeddings_per_feature = [
        int(v) for v in args.num_embeddings_per_feature.split(",")
    ]
    args.multi_hot_sizes = [int(v) for v in args.multi_hot_sizes.split(",")]
    args.dense_arch_layer_sizes = [
        int(v) for v in args.dense_arch_layer_sizes.split(",")
    ]
    args.over_arch_layer_sizes = [int(v) for v in args.over_arch_layer_sizes.split(",")]
    args.data_parallel_embeddings = (
        None
        if args.data_parallel_embeddings is None
        else [int(v) for v in args.data_parallel_embeddings.split(",")]
    )

    args.num_embedding_table = len(args.num_embeddings_per_feature)

    if args.embedding_dim % 4 != 0:
        print(
            f"INFO: args.embedding_dim = {args.embedding_dim} is not aligned with 4, which can't use TorchREC raw embedding table , so all embedding table is dynamic embedding table"
        )
        args.num_dynamicemb = args.num_embedding_table

    if args.platform == "a100":
        args.intra_host_bw = 300e9
        args.inter_host_bw = 25e9
        args.hbm_cap = 80 * 1024 * 1024 * 1024
    elif args.platform == "h100":
        args.intra_host_bw = 450e9
        args.inter_host_bw = 25e9  # TODO: need check
        args.hbm_cap = 80 * 1024 * 1024 * 1024
    elif args.platform == "h200":
        args.intra_host_bw = 450e9
        args.inter_host_bw = 450e9
        args.hbm_cap = 140 * 1024 * 1024 * 1024

    # Print all arguments
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
