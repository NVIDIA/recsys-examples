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
import shutil
import sys
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from dynamicemb import (
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    FrequencyAdmissionStrategy,
)
from dynamicemb.dump_load import (
    DynamicEmbDump,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbLoad,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamicemb.incremental_dump import get_score, set_score
from dynamicemb.planner import (
    DynamicEmbeddingEnumerator,
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.embedding_admission import KVCounter
from dynamicemb.get_planner import get_planner
from dynamicemb.key_value_table import batched_export_keys_values
from dynamicemb.scored_hashtable import ScoreArg, ScorePolicy
from dynamicemb.shard import DynamicEmbeddingCollectionSharder
from dynamicemb.types import AdmissionStrategy
from dynamicemb.utils import TORCHREC_TYPES
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from torchrec import DataType
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


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


class CustomizedScore:
    def __init__(self, table_names: List[int]):
        self.table_names_ = table_names
        self.steps_: Dict[str, int] = {table_name: 1 for table_name in table_names}

    def get(self, table_name: str):
        assert table_name in self.table_names_
        ret = self.steps_[table_name]
        self.steps_[table_name] += 1
        return ret


def get_planner(args, device, eb_configs):
    dict_const = {}
    for i in range(args.num_embedding_table):
        if (
            args.data_parallel_embeddings is not None
            and i in args.data_parallel_embeddings
        ):
            const = ParameterConstraints(
                sharding_types=[ShardingType.DATA_PARALLEL.value],
                # min_partition=2,
                pooling_factors=[args.multi_hot_sizes[i]],
                num_poolings=[1],
                enforce_hbm=True,
                bounds_check_mode=BoundsCheckMode.NONE,
            )
        else:
            use_dynamicemb = True if i < args.dynamicemb_num else False
            const = DynamicEmbParameterConstraints(
                sharding_types=[
                    ShardingType.ROW_WISE.value,
                    # ShardingType.COLUMN_WISE.value,
                    # ShardingType.ROW_WISE.value,
                    # ShardingType.TABLE_ROW_WISE.value,
                    #  ShardingType.TABLE_COLUMN_WISE.value,
                ],
                # min_partition=2,
                pooling_factors=[args.multi_hot_sizes[i]],
                num_poolings=[1],
                enforce_hbm=True,
                bounds_check_mode=BoundsCheckMode.NONE,
                use_dynamicemb=use_dynamicemb,
                dynamicemb_options=DynamicEmbTableOptions(
                    global_hbm_for_values=1024**3,
                    initializer_args=DynamicEmbInitializerArgs(
                        mode=DynamicEmbInitializerMode.DEBUG,
                    ),
                    safe_check_mode=DynamicEmbCheckMode.WARNING,
                    score_strategy=args.score_strategies,
                ),
            )
        dict_const[table_idx_to_name(i)] = const

    topology = Topology(
        local_world_size=get_local_size(),
        world_size=dist.get_world_size(),
        compute_device=device.type,
        hbm_cap=args.hbm_cap,
        ddr_cap=1024 * 1024 * 1024 * 1024,
        # simulate DynamicEmb table is big and have other table
        # hbm_cap=int(340000000/dist.get_world_size()),
        # ddr_cap=1,
        intra_host_bw=args.intra_host_bw,
        inter_host_bw=args.inter_host_bw,
    )

    enumerator = DynamicEmbeddingEnumerator(
        topology=topology,
        # batch_size=args.batch_size,
        constraints=dict_const,
    )

    return DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=topology,
        constraints=dict_const,
        batch_size=args.batch_size,
        enumerator=enumerator,
        # # If experience OOM, increase the percentage. see
        # # https://pytorch.org/torchrec/torchrec.distributed.planner.html#torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        debug=True,
    )


def init_fn(x: torch.Tensor):
    with torch.no_grad():
        x.fill_(2.0)


def update_scores(
    score_strategy: str,
    expect_scores: Dict[int, int],
    key: int,
    step: int,
):
    if score_strategy == "step":
        expect_scores[key] = step
    elif score_strategy == "lfu":
        if key not in expect_scores:
            expect_scores[key] = 1
        else:
            expect_scores[key] = expect_scores[key] + 1
    else:
        return


def generate_sparse_feature(
    num_embedding_collections: int,
    num_embeddings: List[int],
    multi_hot_sizes: List[int],
    rank: int,
    world_size: int,
    batch_size: int,
    num_iterations: int,
    score_strategy: str,
    scores_collection: Dict[str, Dict[int, int]],
    seed: int = 42,
):
    feature_batch = feature_num * local_batch_size

    batch_size_per_rank = batch_size // world_size
    kjts = []
    all_kjts = []
    for embedding_collection_id in range(num_embedding_collections):
        for embedding_id, _ in enumerate(num_embeddings):
            _, table_name = idx_to_name(embedding_collection_id, embedding_id)
            scores_collection[table_name] = {}
    step = 0
    for _ in range(num_iterations):
        step += 1
        cur_indices, cur_lengths = [], []
        all_indices, all_lengths = [], []
        keys = []
        for embedding_collection_id in range(num_embedding_collections):
            for embedding_id, num_embedding in enumerate(num_embeddings):
                feature_name, table_name = idx_to_name(
                    embedding_collection_id, embedding_id
                )
                expected_scores: Dict[int, int] = scores_collection[table_name]
                for sample_id in range(batch_size):
                    hotness = random.randint(
                        0, multi_hot_sizes[embedding_collection_id]
                    )
                    indices = [random.randint(0, (1 << 63) - 1) for _ in range(hotness)]
                    all_indices.extend(indices)
                    all_lengths.append(hotness)
                    if sample_id // batch_size_per_rank == rank:
                        cur_indices.extend(indices)
                        cur_lengths.append(hotness)
                    for index in indices:
                        update_scores(score_strategy, expected_scores, index, step)
                keys.append(feature_name)
        kjts.append(
            KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.tensor(cur_indices, dtype=torch.int64).cuda(),
                lengths=torch.tensor(cur_lengths, dtype=torch.int64).cuda(),
            )
        )
        all_kjts.append(
            KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.tensor(all_indices, dtype=torch.int64).cuda(),
                lengths=torch.tensor(all_lengths, dtype=torch.int64).cuda(),
            )
        )
    return kjts, keys, all_kjts

    for i in range(feature_batch):
        f = i // local_batch_size
        cur_bag_size = random.randint(0, multi_hot_sizes[f])
        cur_bag = set({})
        while len(cur_bag) < cur_bag_size:
            cur_bag.add(random.randint(0, num_embeddings_list[f] - 1))

        indices.extend(list(cur_bag))
        lengths.append(cur_bag_size)

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        embeddings_dict = [
            embedding_module(kjt).wait() for embedding_module in self.embedding_modules
        ]
        embeddings = []
        for embedding_dict in embeddings_dict:
            for embedding in embedding_dict.values():
                embeddings.append(embedding.values())
        return torch.cat(embeddings, dim=0)


DATA_TYPE_NUM_BITS: Dict[DataType, int] = {
    DataType.FP32: 32,
    DataType.FP16: 16,
    DataType.BF16: 16,
}


def apply_dmp(
    model: torch.nn.Module,
    optimizer_kwargs: Dict[str, Any],
    device: torch.device,
    score_strategy: DynamicEmbScoreStrategy = DynamicEmbScoreStrategy.LFU,
    use_index_dedup: bool = False,
    caching: bool = False,
    cache_capacity_ratio: float = 0.5,
    admit_strategy: AdmissionStrategy = None,
):
    eb_configs = []
    dynamicemb_options_dict = {}
    for n, m in model.named_modules():
        if type(m) in TORCHREC_TYPES:
            eb_configs.extend(m.embedding_configs())
            for eb_config in eb_configs:
                dim = eb_config.embedding_dim
                tmp_type = eb_config.data_type

                embedding_type_bytes = DATA_TYPE_NUM_BITS[tmp_type] / 8
                emb_num_embeddings = (
                    eb_config.num_embeddings * cache_capacity_ratio
                    if caching
                    else eb_config.num_embeddings
                )
                emb_num_embeddings_next_power_of_2 = 2 ** math.ceil(
                    math.log2(emb_num_embeddings)
                )  # HKV need embedding vector num is power of 2

                # Calculate optimizer state dimension
                from dynamicemb.dynamicemb_config import (
                    data_type_to_dtype,
                    get_optimizer_state_dim,
                )
                from dynamicemb_extensions import OptimizerType

                # Map fbgemm EmbOptimType to dynamicemb OptimizerType
                emb_opt_type = (
                    optimizer_kwargs.get("optimizer") if optimizer_kwargs else None
                )
                opt_type_map = {
                    EmbOptimType.EXACT_ROWWISE_ADAGRAD: OptimizerType.RowWiseAdaGrad,
                    EmbOptimType.SGD: OptimizerType.SGD,
                    EmbOptimType.EXACT_SGD: OptimizerType.SGD,
                    EmbOptimType.ADAM: OptimizerType.Adam,
                    EmbOptimType.EXACT_ADAGRAD: OptimizerType.AdaGrad,
                }
                opt_type = opt_type_map.get(emb_opt_type) if emb_opt_type else None
                # Convert torchrec DataType to torch.dtype
                torch_dtype = data_type_to_dtype(tmp_type)
                optimizer_state_dim = (
                    get_optimizer_state_dim(opt_type, dim, torch_dtype)
                    if opt_type
                    else 0
                )

                # Include optimizer state in HBM calculation
                total_hbm_need = (
                    embedding_type_bytes
                    * (dim + optimizer_state_dim)
                    * emb_num_embeddings_next_power_of_2
                )

                admission_counter = KVCounter(
                    max(1024 * 1024, emb_num_embeddings_next_power_of_2 // 4)
                )
                dynamicemb_options_dict[eb_config.name] = DynamicEmbTableOptions(
                    global_hbm_for_values=total_hbm_need,
                    score_strategy=score_strategy,
                    initializer_args=DynamicEmbInitializerArgs(
                        mode=DynamicEmbInitializerMode.CONSTANT,
                        value=1e-1,
                    ),
                    bucket_capacity=emb_num_embeddings_next_power_of_2,
                    max_capacity=emb_num_embeddings_next_power_of_2,
                    caching=caching,
                    local_hbm_for_values=1024**3,
                    admit_strategy=admit_strategy,
                    admission_counter=admission_counter,
                )
    planner = get_planner(
        eb_configs,
        {},
        dynamicemb_options_dict,
        device,
    )


def run(args):
    backend = "nccl"
    dist.init_process_group(backend=backend)

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    all_table_names = [
        table_idx_to_name(feature_idx)
        for feature_idx in range(args.num_embedding_table)
    ]

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

    if not args.use_torch_opt:
        sharder = DynamicEmbeddingCollectionSharder(
            qcomm_codecs_registry=qcomm_codecs_registry,
            fused_params=optimizer_kwargs,
            use_index_dedup=args.use_index_dedup,
        )
    else:
        sharder = DynamicEmbeddingCollectionSharder(
            qcomm_codecs_registry=qcomm_codecs_registry,
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

    customized_scores = CustomizedScore(all_table_names)
    ret: Dict[str, Dict[str, int]] = get_score(model)
    prefix_path = "model"

def create_model(
    num_embedding_collections: int,
    num_embeddings: List[int],
    embedding_dim: int,
    optimizer_kwargs: Dict[str, Any],
    score_strategy: DynamicEmbScoreStrategy = DynamicEmbScoreStrategy.LFU,
    use_index_dedup: bool = False,
    caching: bool = False,
    cache_capacity_ratio: float = 0.5,
    admit_strategy: AdmissionStrategy = None,
):
    ebc_list = []
    for embedding_collection_id in range(num_embedding_collections):
        eb_configs = []
        for embedding_id, num_embedding in enumerate(num_embeddings):
            feature_name, embedding_name = idx_to_name(
                embedding_collection_id, embedding_id
            )
    set_score(model, {prefix_path: scores_to_set})

    if local_rank == 0 and args.print_sharding_plan:
        for collectionkey, plans in model._plan.plan.items():
            print(collectionkey)
            for table_name, plan in plans.items():
                print(table_name, "\n", plan, "\n")

    def optimizer_with_params():
        if args.optimizer_type == "sgd":
            return lambda params: torch.optim.SGD(params, lr=args.learning_rate)
        elif args.optimizer_type == "adagrad":
            return lambda params: torch.optim.Adagrad(
                params, lr=args.learning_rate, eps=args.eps
            )
        elif args.optimizer_type == "rowwise_adagrad":
            return lambda params: torch.optim.Adagrad(
                params, lr=args.learning_rate, eps=args.eps
            )
        else:
            raise ValueError("unknown optimizer type")

    Debugger()

    for i in range(args.num_iterations):
        sparse_feature = generate_sparse_feature(
            feature_num=args.num_embedding_table,
            num_embeddings_list=args.num_embeddings_per_feature,
            multi_hot_sizes=args.multi_hot_sizes,
            local_batch_size=args.batch_size // world_size,
        )
        ret = model(sparse_feature)  # => this is awaitable

        feature_names = []
        jagged_tensors = []
        for k, v in ret.items():
            feature_names.append(k)
            jagged_tensors.append(v.values())

        concatenated_tensor = torch.cat(jagged_tensors, dim=0)
        reduced_tensor = concatenated_tensor.sum()
        reduced_tensor.backward()

        scores_to_set: Dict[str, int] = {}
        for i in range(args.num_embedding_table):
            if args.score_strategies == DynamicEmbScoreStrategy.CUSTOMIZED:
                scores_to_set[all_table_names[i]] = customized_scores.get(
                    all_table_names[i]
                )

        set_score(model, {prefix_path: scores_to_set})

    DynamicEmbDump("debug_weight", model, optim=True)
    DynamicEmbLoad("debug_weight", model, optim=True)

    table_names = {"model": ["t_0"]}
    DynamicEmbDump("debug_weight_t0", model, table_names=table_names, optim=True)
    DynamicEmbLoad("debug_weight_t0", model, table_names=table_names, optim=False)

    table_names = {"model": ["t_1"]}
    DynamicEmbDump("debug_weight_t1", model, table_names=table_names, optim=False)
    DynamicEmbLoad("debug_weight_t1", model, table_names=table_names, optim=False)

    dist.barrier()

    if local_rank == 0:
        shutil.rmtree("debug_weight")
        shutil.rmtree("debug_weight_t0")
        shutil.rmtree("debug_weight_t1")

    dist.barrier()
    dist.destroy_process_group()


@record
def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(description="DynamicEmb dump load test")
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
        default="65536,32768,409600,81920",
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        default="16,8,20,1",
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

    model = apply_dmp(
        model,
        optimizer_kwargs,
        torch.device(f"cuda:{torch.cuda.current_device()}"),
        score_strategy=score_strategy,
        use_index_dedup=use_index_dedup,
        caching=caching,
        cache_capacity_ratio=cache_capacity_ratio,
        admit_strategy=admit_strategy,
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
        "--dynamicemb_num",
        type=int,
        default=2,
        help="Number of dynamic embedding tables.",
    )
    parser.add_argument(
        "--use_index_dedup",
        type=str2bool,
        default=True,
        help="Use index deduplication (default: True).",
    )

    parser.add_argument(
        "--score_type",
        type=str,
        default="timestamp",
        choices=["timestamp", "step", "custimized"],
        help="score type string",
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
        args.dynamicemb_num = args.num_embedding_table

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

    if args.score_type == "timestamp":
        args.score_strategies = DynamicEmbScoreStrategy.TIMESTAMP
    elif args.score_type == "step":
        args.score_strategies = DynamicEmbScoreStrategy.STEP
    elif args.score_type == "custimized":
        args.score_strategies = DynamicEmbScoreStrategy.CUSTOMIZED

def check_counter_table_checkpoint(x, y):
    device = torch.cuda.current_device()
    tables_x = get_dynamic_emb_module(x)
    tables_y = get_dynamic_emb_module(y)

    for table_x, table_y in zip(tables_x, tables_y):
        for cnt_tx, cnt_ty in zip(
            table_x._admission_counter, table_y._admission_counter
        ):
            assert cnt_tx.table_.size() == cnt_ty.table_.size()

            for keys, named_scores in cnt_tx._batched_export_keys_scores(
                cnt_tx.table_.score_names_, torch.device(f"cuda:{device}")
            ):
                if keys.numel() == 0:
                    continue
                freq_name = cnt_tx.table_.score_names_[0]
                frequencies = named_scores[freq_name]

                score_args_lookup = [
                    ScoreArg(
                        name=freq_name,
                        value=torch.zeros_like(frequencies),
                        policy=ScorePolicy.CONST,
                        is_return=True,
                    )
                ]
                founds = torch.empty(
                    keys.numel(), dtype=torch.bool, device=device
                ).fill_(False)

                cnt_ty.lookup(keys, score_args_lookup, founds)

                assert torch.equal(frequencies, score_args_lookup)


@click.command()
@click.option("--num-embedding-collections", type=int, required=True)
@click.option("--num-embeddings", type=str, required=True)
@click.option("--multi-hot-sizes", type=str, required=True)
@click.option("--embedding-dim", type=int, required=True)
@click.option("--save-path", type=str, required=True)
@click.option(
    "--optimizer-type",
    type=click.Choice(["sgd", "adam", "adagrad", "rowwise_adagrad"]),
    required=True,
)
@click.option("--mode", type=click.Choice(["load", "dump"]), required=True)
@click.option(
    "--score-strategy",
    type=click.Choice(["timestamp", "step", "lfu"]),
    required=True,
)
@click.option("--optim", type=bool, required=True)
@click.option("--counter", type=bool, required=True)
def test_model_load_dump(
    num_embedding_collections: int,
    num_embeddings: str,
    multi_hot_sizes: str,
    embedding_dim: int,
    optimizer_type: str,
    score_strategy: str,
    mode: str,
    save_path: str,
    optim: bool,
    counter: bool,
    batch_size: int = 128,
    num_iterations: int = 10,
):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    num_embeddings = [int(v) for v in num_embeddings.split(",")]
    multi_hot_sizes = [int(v) for v in multi_hot_sizes.split(",")]

    for num_embedding, multi_hot_size in zip(num_embeddings, multi_hot_sizes):
        if batch_size * num_iterations * multi_hot_size > num_embedding:
            raise ValueError(
                "batch_size * num_iterations * multi_hot_size > num_embedding, this may lead to eviction of dynamicemb and cause test fail"
            )

    optimizer_kwargs = get_optimizer_kwargs(optimizer_type)
    score_strategy_ = get_score_strategy(score_strategy)

    ref_model = create_model(
        num_embedding_collections=num_embedding_collections,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        optimizer_kwargs=optimizer_kwargs,
        score_strategy=score_strategy_,
        admit_strategy=FrequencyAdmissionStrategy(
            threshold=2 if counter else 1,
        ),
    )

    expect_scores_collection: Dict[str, Dict[int, int]] = {}
    kjts, feature_names, all_kjts = generate_sparse_feature(
        num_embedding_collections=num_embedding_collections,
        num_embeddings=num_embeddings,
        multi_hot_sizes=multi_hot_sizes,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        batch_size=batch_size,
        num_iterations=num_iterations,
        score_strategy=score_strategy,
        scores_collection=expect_scores_collection,
    )

    for kjt in kjts:
        ret = ref_model(kjt)
        loss = (
            ret.sum() * dist.get_world_size()
        )  # scale the loss by world size to make the gradients consistent between different gpu settings
        loss.backward()

    if mode == "dump":
        shutil.rmtree(save_path, ignore_errors=True)
        DynamicEmbDump(save_path, ref_model, optim=optim, counter=counter)

    if mode == "load":
        model = create_model(
            num_embedding_collections=num_embedding_collections,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            optimizer_kwargs=optimizer_kwargs,
            score_strategy=score_strategy_,
            admit_strategy=FrequencyAdmissionStrategy(
                threshold=2 if counter else 1,
            ),
        )

        DynamicEmbLoad(save_path, model, optim=optim, counter=counter)

        if counter:
            check_counter_table_checkpoint(model, ref_model)

        table_name_to_key_score_dict = {}
        table_name_to_visited_key_dict = {}
        for _, _, sharded_module in find_sharded_modules(model):
            dynamic_emb_modules = get_dynamic_emb_module(sharded_module)
            for dynamic_emb_module in dynamic_emb_modules:
                for table_name, table, counter_table in zip(
                    dynamic_emb_module.table_names,
                    dynamic_emb_module.tables,
                    dynamic_emb_module._admission_counter,
                ):
                    key_to_score = {}
                    visited_keys = set({})
                    for batched_key, _, _, batched_score in batched_export_keys_values(
                        table.table, torch.device(f"cpu")
                    ):
                        for key, score in zip(
                            batched_key.tolist(), batched_score.tolist()
                        ):
                            key_to_score[key] = score

                    for (
                        keys,
                        named_scores,
                    ) in counter_table.table_._batched_export_keys_scores(
                        counter_table.table_.score_names_, torch.device(f"cpu")
                    ):
                        if keys.numel() == 0:
                            continue
                        for key in keys.tolist():
                            visited_keys.add(key)

                    table_name_to_key_score_dict[table_name] = key_to_score
                    table_name_to_visited_key_dict[table_name] = visited_keys

        for embedding_collection_idx, embedding_idx in product(
            range(num_embedding_collections), range(len(num_embeddings))
        ):
            feature_name, table_name = idx_to_name(
                embedding_collection_idx, embedding_idx
            )
            key_to_score_dict = table_name_to_key_score_dict[table_name].copy()
            expect_scores = expect_scores_collection[table_name]
            visited_keys = table_name_to_visited_key_dict[table_name]

            if score_strategy == "step" or score_strategy == "lfu":
                for kjt in reversed(all_kjts):
                    keys = kjt[feature_name].values().tolist()
                    for key in keys:
                        if key % world_size == rank and key not in visited_keys:
                            assert (
                                key in key_to_score_dict
                            ), f"Key {key} must exist in table of rank {rank}."
                            assert (
                                key_to_score_dict[key] == expect_scores[key]
                            ), f"Expect {key_to_score_dict[key]} = {expect_scores[key]}"
            # The idea is that the score of a newer key is greater than that of an older key. Therefore, I iterate through the input in reverse order and track the minimum score encountered. For each batch, the score should be lower than the minimum score from the previous batch. To avoid issues caused by duplicate keys, every time I access a key, I set its score to -inf. This ensures that if that key appears again, its score will be sufficiently small to remain below the minimum score.
            elif score_strategy == "timestamp":
                min_score = float("inf")
                lasted_min_score = float("inf")
                for kjt in reversed(all_kjts):
                    keys = kjt[feature_name].values().tolist()
                    for key in keys:
                        if key % world_size == rank and key not in visited_keys:
                            assert (
                                key in key_to_score_dict
                            ), f"Key {key} must exist in table of rank {rank}."
                        else:
                            continue

                        assert (
                            key_to_score_dict[key] <= min_score
                        ), f"key {key} score {key_to_score_dict[key]} should be < min_score {min_score}"
                        lasted_min_score = min(lasted_min_score, key_to_score_dict[key])
                        visited_keys.add(key)

                    min_score = lasted_min_score
                    lasted_min_score = min_score

            else:
                raise RuntimeError("Not supported score strategy.")

        if optim:
            for kjt in kjts:
                ret = model(kjt)
                ret.sum().backward()
                ref_ret = ref_model(kjt)
                ref_ret.sum().backward()

        ref_model = ref_model.eval()
        model = model.eval()

        with torch.inference_mode():
            for kjt in kjts:
                ret = model(kjt)
                ref_ret = ref_model(kjt)
                assert torch.allclose(ret, ref_ret)


if __name__ == "__main__":
    main(sys.argv[1:])
