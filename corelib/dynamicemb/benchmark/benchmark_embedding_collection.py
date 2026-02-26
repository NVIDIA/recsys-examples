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
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
import torchrec
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
    align_to_table_size,
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
from torchrec import DataType
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    QCommsConfig,
    get_qcomm_codecs_registry,
)
from torchrec.distributed.model_parallel import (
    DefaultDataParallelWrapper,
    DistributedModelParallel,
)
from torchrec.distributed.planner import Topology
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


DATA_TYPE_NUM_BITS: Dict[DataType, int] = {
    DataType.FP32: 32,
    DataType.FP16: 16,
    DataType.BF16: 16,
    DataType.INT8: 8,
    DataType.UINT8: 8,
    DataType.INT4: 4,
    DataType.INT2: 2,
}


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
    # fuse all table in one table , and use feature in table
    use_dynamicemb = True if args.num_dyn_emb_table > 0 else False
    if use_dynamicemb:
        eb_config = eb_configs[0]
        dim = eb_config.embedding_dim
        tmp_type = eb_config.data_type

        embedding_type_bytes = DATA_TYPE_NUM_BITS[tmp_type] / 8
        eb_num_embeddings = eb_config.num_embeddings
        eb_num_embeddings_aligned = align_to_table_size(eb_num_embeddings)
        total_hbm_need = embedding_type_bytes * dim * eb_num_embeddings_aligned

        const = DynamicEmbParameterConstraints(
            sharding_types=[
                ShardingType.ROW_WISE.value,
            ],
            enforce_hbm=True,
            bounds_check_mode=BoundsCheckMode.NONE,
            use_dynamicemb=use_dynamicemb,
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=total_hbm_need,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.NORMAL
                ),
            ),
        )

        dict_const[table_idx_to_name(0)] = const
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
    else:
        for i in range(args.num_embedding_table):
            const = DynamicEmbParameterConstraints(
                sharding_types=[
                    ShardingType.ROW_WISE.value,
                ],
                pooling_factors=[args.multi_hot_sizes[i]],
                num_poolings=[1],
                enforce_hbm=True,
                bounds_check_mode=BoundsCheckMode.NONE,
                use_dynamicemb=False,
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


def generate_sparse_feature(
    feature_num,
    num_embeddings_list,
    max_sequence_size,
    local_batch_size=50,
    fuse_table=True,
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
        if fuse_table:
            adjusted_indices = unique_indices + prefix_sums[f]
        else:
            adjusted_indices = unique_indices

        indices.extend(adjusted_indices.flatten())

        lengths.extend([max_sequence_size[f]] * local_batch_size)
    return torchrec.KeyedJaggedTensor(
        keys=[feature_idx_to_name(feature_idx) for feature_idx in range(feature_num)],
        values=torch.tensor(indices, dtype=torch.int64).cuda(),
        lengths=torch.tensor(lengths, dtype=torch.int64).cuda(),
    )


def run(args):
    backend = "nccl"
    dist.init_process_group(backend=backend)

    local_rank = dist.get_rank()  # for one node
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    np.random.seed(1024 + local_rank)
    warm_iters = 10
    sparse_features = []
    for i in range(args.num_iterations + warm_iters):
        sparse_features.append(
            generate_sparse_feature(
                feature_num=args.num_embedding_table,
                num_embeddings_list=args.num_embeddings_per_feature,
                max_sequence_size=args.multi_hot_sizes,
                local_batch_size=args.batch_size // world_size,
                fuse_table=args.use_dynamic_embedding,
            )
        )

    total_num_embedding = sum(args.num_embeddings_per_feature)

    if args.use_dynamic_embedding:
        # fuse all embedding table at one table
        eb_configs = [
            torchrec.EmbeddingConfig(
                name=table_idx_to_name(0),
                embedding_dim=args.embedding_dim,
                num_embeddings=total_num_embedding,
                feature_names=[
                    feature_idx_to_name(feature_idx)
                    for feature_idx in range(args.num_embedding_table)
                ],
            )
        ]

    else:
        eb_configs = [
            torchrec.EmbeddingConfig(
                name=table_idx_to_name(feature_idx),
                embedding_dim=args.embedding_dim,
                num_embeddings=args.num_embeddings_per_feature[feature_idx],
                feature_names=[feature_idx_to_name(feature_idx)],
            )
            for feature_idx in range(args.num_embedding_table)
        ]

    ebc = torchrec.EmbeddingCollection(
        device=torch.device("meta"),
        tables=eb_configs,
    )

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
    fused_params = {}
    if args.fwd_output_precision == "fp32":
        fused_params["output_dtype"] = SparseType.FP32
    elif args.fwd_output_precision == "fp16":
        fused_params["output_dtype"] = SparseType.FP16
    elif args.fwd_output_precision == "bf16":
        fused_params["output_dtype"] = SparseType.BF16

    if not args.use_torch_opt:
        fused_params.update(optimizer_kwargs)

    if args.use_dynamic_embedding:
        sharder = DynamicEmbeddingCollectionSharder(
            qcomm_codecs_registry=qcomm_codecs_registry,
            fused_params=fused_params,
            use_index_dedup=args.use_index_dedup,
        )
    else:
        sharder = EmbeddingCollectionSharder(
            qcomm_codecs_registry=qcomm_codecs_registry,
            fused_params=fused_params,
            use_index_dedup=args.use_index_dedup,
        )
    plan = planner.collective_plan(ebc, [sharder], dist.GroupMember.WORLD)

    if args.use_torch_opt:
        apply_optimizer_in_backward(
            embedding_optimizer,
            ebc.parameters(),
            optimizer_kwargs,
        )

    data_parallel_wrapper = DefaultDataParallelWrapper(
        allreduce_comm_precision=args.allreduce_precision
    )

    model = DistributedModelParallel(
        module=ebc,
        device=device,
        # pyre-ignore
        sharders=[sharder],
        plan=plan,
        data_parallel_wrapper=data_parallel_wrapper,
    )

    for i in range(warm_iters):
        sparse_feature = sparse_features[i]
        ret = model(sparse_feature)  # => this is awaitable
        feature_names = []
        tensors = []
        for k, v in ret.items():
            feature_names.append(k)
            tensors.append(v.values())
        torch.cat(tensors, dim=1).sum().backward()

    torch.cuda.synchronize()

    start_time = time.perf_counter()
    # torch.cuda.profiler.start()
    for i in range(args.num_iterations):
        sparse_feature = sparse_features[i + warm_iters]
        ret = model(sparse_feature)  # => this is awaitable

        feature_names = []
        tensors = []
        for k, v in ret.items():
            feature_names.append(k)
            tensors.append(v.values())

        cat_tensor = torch.cat(tensors, dim=1)
        loss = cat_tensor.sum()
        loss.backward()

    # torch.cuda.profiler.stop()
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    average_iteration_time = (end_time - start_time) / args.num_iterations * 1000
    print(f"Total time taken: {end_time - start_time:.4f} seconds")
    print(f"Average time per iteration: {average_iteration_time:.4f} ms")


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
        default=8192,
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
        default="45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35",
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )

    hot_num = "10"
    feature_num = 26

    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        default=",".join([hot_num] * feature_num),
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
        "--fwd_output_precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
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
        action="store_true",
        help="if is true , use torch register optimizer , or use torchrec",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="a100",
        choices=["a100", "h100", "h200"],
        help="Platform, has different system spec",
    )
    parser.add_argument(
        "--use_index_dedup",
        type=str2bool,
        default=True,
        help="Use index deduplication (default: True).",
    )
    parser.add_argument(
        "--use_dynamic_embedding",
        type=str2bool,
        default=True,
        help="use dynamic embedding for embedding , if False ,will use TorchREC raw embedding table.",
    )

    args = parser.parse_args()

    args.num_embeddings_per_feature = [
        int(v) for v in args.num_embeddings_per_feature.split(",")
    ]
    args.multi_hot_sizes = [int(v) for v in args.multi_hot_sizes.split(",")]

    args.num_embedding_table = len(args.num_embeddings_per_feature)
    if args.use_dynamic_embedding:
        args.num_dyn_emb_table = args.num_embedding_table
    else:
        args.num_dyn_emb_table = 0

    if args.embedding_dim % 4 != 0:
        print(
            f"INFO: args.embedding_dim = {args.embedding_dim} is not aligned with 4, which can't use TorchREC raw embedding table , so all embedding table is dynamic embedding table"
        )
        args.num_dyn_emb_table = args.num_embedding_table

    if args.platform == "a100":
        args.intra_host_bw = 300e9
        args.inter_host_bw = 25e9
        args.hbm_cap = 80 * 1024 * 1024 * 1024
    elif args.platform == "h100":
        args.intra_host_bw = 450e9
        args.inter_host_bw = 25e9
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
