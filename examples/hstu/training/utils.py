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
import sys
from typing import Dict, List, Optional, Tuple, Union

import configs
import dataset
import torch  # pylint: disable-unused-import
import torch.distributed as dist
from configs import (
    HSTUConfig,
    HSTULayerType,
    HSTUPreprocessingConfig,
    KernelBackend,
    OptimizerParam,
    PositionEncodingConfig,
    get_hstu_config,
)
from dynamicemb import DynamicEmbTableOptions
from modules.embedding import ShardedEmbeddingConfig
from training.gin_config_args import (
    BenchmarkDatasetArgs,
    DatasetArgs,
    DynamicEmbeddingArgs,
    EmbeddingArgs,
    NetworkArgs,
    OptimizerArgs,
    TensorModelParallelArgs,
    TrainerArgs,
)

_OPTIMIZER_TYPE_TO_STORAGE_MULTIPLIER = {
    "sgd": 1,
    "adam": 3,
}


def get_embedding_vector_storage_multiplier(optimizer_type: str) -> int:
    global _OPTIMIZER_TYPE_TO_STORAGE_MULTIPLIER
    return _OPTIMIZER_TYPE_TO_STORAGE_MULTIPLIER.get(optimizer_type, 1)


def cal_flops_single_rank(
    hstu_config: HSTUConfig,
    seqlens: torch.Tensor,
    num_contextuals: Optional[torch.Tensor],
    num_candidates: Optional[torch.Tensor],
    has_bwd: bool = True,
) -> torch.Tensor:
    num_layers = hstu_config.num_layers
    hidden_size = hstu_config.hidden_size
    num_heads = hstu_config.num_attention_heads
    dim_per_head = hstu_config.kv_channels
    if num_contextuals is None:
        num_contextuals = torch.zeros_like(seqlens)
    if num_candidates is None:
        num_candidates = torch.zeros_like(seqlens)
    with torch.inference_mode():
        seqlens = seqlens.to(torch.float)
        num_contextuals = num_contextuals.to(torch.float)
        num_candidates = num_candidates.to(torch.float)
        num_history = seqlens - num_contextuals - num_candidates
        # reference: https://github.com/Dao-AILab/flash-attention/blob/9c0e9ee86d0e0022b60deddb405c20ab77481582/benchmarks/benchmark_flash_attention.py#L27-L30
        # flops between seq and contextual + history
        attn_flops_per_layer = (
            4 * num_heads * seqlens * (num_contextuals + num_history) * dim_per_head
        )
        if hstu_config.is_causal:
            # remove upper triangular flops between history and history
            attn_flops_per_layer -= (
                2 * num_heads * num_history * num_history * dim_per_head
            )
        # flops between candidates
        attn_flops_per_layer += 4 * num_heads * num_candidates * dim_per_head
        if has_bwd:
            attn_flops_per_layer *= 3.5

        gemm_flops_per_layer = (
            2 * seqlens * 4 * num_heads * dim_per_head * hidden_size
        )  # qkvu proj fwd
        gemm_flops_per_layer += 2 * seqlens * num_heads * hidden_size  # proj fwd
        if has_bwd:
            gemm_flops_per_layer *= 3

        other_ops_flops_per_layer = seqlens * num_heads * dim_per_head  # mul fwd
        if has_bwd:
            other_ops_flops_per_layer *= 2  # bwd
        if hstu_config.residual:
            other_ops_flops_per_layer += (
                seqlens * num_heads * hidden_size
            )  # add fwd, bwd is no-op

        return (
            torch.sum(
                gemm_flops_per_layer + attn_flops_per_layer + other_ops_flops_per_layer
            )
            * num_layers
        )


def cal_flops(
    hstu_config: HSTUConfig,
    seqlens: List[torch.Tensor],
    num_contextuals: List[torch.Tensor],
    num_candidates: List[torch.Tensor],
) -> int:
    seqlens_tensor = torch.cat(seqlens)
    world_size = torch.distributed.get_world_size()
    gathered_seqlens = (
        [torch.empty_like(seqlens_tensor) for _ in range(world_size)]
        if torch.distributed.get_rank() == 0
        else None
    )
    num_contextuals_tensor = torch.cat(num_contextuals)
    num_candidates_tensor = torch.cat(num_candidates)

    gathered_num_contextuals = (
        [torch.empty_like(num_contextuals_tensor) for _ in range(world_size)]
        if torch.distributed.get_rank() == 0
        else None
    )
    gathered_num_candidates = (
        [torch.empty_like(num_candidates_tensor) for _ in range(world_size)]
        if torch.distributed.get_rank() == 0
        else None
    )
    torch.distributed.gather(seqlens_tensor, gathered_seqlens, dst=0)
    torch.distributed.gather(num_contextuals_tensor, gathered_num_contextuals, dst=0)
    torch.distributed.gather(num_candidates_tensor, gathered_num_candidates, dst=0)
    if torch.distributed.get_rank() == 0:
        flops = (
            cal_flops_single_rank(
                hstu_config,
                torch.cat(gathered_seqlens),
                torch.cat(gathered_num_contextuals),
                torch.cat(gathered_num_candidates),
            )
            .cpu()
            .item()
        )
    else:
        flops = 0
    return flops


def create_hstu_config(
    network_args: NetworkArgs, tensor_model_parallel_args: TensorModelParallelArgs
):
    dtype = None
    if network_args.dtype_str == "bfloat16":
        dtype = torch.bfloat16
    if network_args.dtype_str == "float16":
        dtype = torch.float16
    assert dtype is not None, "dtype not selected. Check your input."

    kernel_backend = None
    if network_args.kernel_backend == "cutlass":
        kernel_backend = KernelBackend.CUTLASS
    elif network_args.kernel_backend == "triton":
        kernel_backend = KernelBackend.TRITON
    elif network_args.kernel_backend == "pytorch":
        kernel_backend = KernelBackend.PYTORCH
    else:
        raise ValueError(
            f"Kernel backend {network_args.kernel_backend} is not supported."
        )
    layer_type = None
    if tensor_model_parallel_args.tensor_model_parallel_size == 1:
        layer_type = HSTULayerType.FUSED
    else:
        layer_type = HSTULayerType.NATIVE

    position_encoding_config = PositionEncodingConfig(
        num_position_buckets=network_args.num_position_buckets,
        num_time_buckets=2048,
        use_time_encoding=False,
    )
    if network_args.item_embedding_dim > 0 or network_args.contextual_embedding_dim > 0:
        hstu_preprocessing_config = HSTUPreprocessingConfig(
            item_embedding_dim=network_args.item_embedding_dim,
            contextual_embedding_dim=network_args.contextual_embedding_dim,
        )
    else:
        hstu_preprocessing_config = None
    return get_hstu_config(
        hidden_size=network_args.hidden_size,
        kv_channels=network_args.kv_channels,
        num_attention_heads=network_args.num_attention_heads,
        num_layers=network_args.num_layers,
        hidden_dropout=network_args.hidden_dropout,
        norm_epsilon=network_args.norm_epsilon,
        is_causal=network_args.is_causal,
        dtype=dtype,
        kernel_backend=kernel_backend,
        hstu_preprocessing_config=hstu_preprocessing_config,
        position_encoding_config=position_encoding_config,
        target_group_size=network_args.target_group_size,
        hstu_layer_type=layer_type,
        recompute_input_layernorm=network_args.recompute_input_layernorm,
        recompute_input_silu=network_args.recompute_input_silu,
    )


def get_data_loader(
    task_type: str,
    dataset_args: Union[DatasetArgs, BenchmarkDatasetArgs],
    trainer_args: TrainerArgs,
    num_tasks: int,
):
    assert task_type in [
        "ranking",
        "retrieval",
    ], f"task type should be ranking or retrieval not {task_type}"
    if isinstance(dataset_args, BenchmarkDatasetArgs):
        from dataset.utils import FeatureConfig

        assert (
            trainer_args.max_train_iters is not None
            and trainer_args.max_eval_iters is not None
        ), "Benchmark dataset expects max_train_iters and max_eval_iters as num_batches"
        feature_name_to_max_item_id = {}
        for e in dataset_args.embedding_args:
            for feature_name in e.feature_names:
                feature_name_to_max_item_id[feature_name] = (
                    sys.maxsize
                    if isinstance(e, DynamicEmbeddingArgs)
                    else e.item_vocab_size_or_capacity
                )
        feature_configs = []
        for f in dataset_args.feature_args:
            feature_configs.append(
                FeatureConfig(
                    feature_names=f.feature_names,
                    max_item_ids=[
                        feature_name_to_max_item_id[n] for n in f.feature_names
                    ],
                    max_sequence_length=f.max_sequence_length,
                    is_jagged=f.is_jagged,
                )
            )

        kwargs = dict(
            feature_configs=feature_configs,
            item_feature_name=dataset_args.item_feature_name,
            contextual_feature_names=dataset_args.contextual_feature_names,
            action_feature_name=dataset_args.action_feature_name,
            max_num_candidates=dataset_args.max_num_candidates,
            num_generated_batches=100,
            num_tasks=num_tasks,
        )
        train_dataset = dataset.dummy_dataset.DummySequenceDataset(
            batch_size=trainer_args.train_batch_size, **kwargs
        )
        test_dataset = dataset.dummy_dataset.DummySequenceDataset(
            batch_size=trainer_args.eval_batch_size, **kwargs
        )
    else:
        assert isinstance(dataset_args, DatasetArgs)
        (
            train_dataset,
            test_dataset,
        ) = dataset.sequence_dataset.get_dataset(
            dataset_name=dataset_args.dataset_name,
            dataset_path=dataset_args.dataset_path,
            max_sequence_length=dataset_args.max_sequence_length,
            max_num_candidates=dataset_args.max_num_candidates,
            num_tasks=num_tasks,
            batch_size=trainer_args.train_batch_size,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            shuffle=dataset_args.shuffle,
            random_seed=trainer_args.seed,
            eval_batch_size=trainer_args.eval_batch_size,
        )
    return dataset.get_data_loader(train_dataset), dataset.get_data_loader(test_dataset)  # type: ignore[attr-defined]


def create_optimizer_params(optimizer_args: OptimizerArgs):
    return OptimizerParam(
        optimizer_str=optimizer_args.optimizer_str,
        learning_rate=optimizer_args.learning_rate,
        adam_beta1=optimizer_args.adam_beta1,
        adam_beta2=optimizer_args.adam_beta2,
        adam_eps=optimizer_args.adam_eps,
    )


def create_embedding_config(
    hidden_size: int, embedding_args: EmbeddingArgs
) -> ShardedEmbeddingConfig:
    if isinstance(embedding_args, DynamicEmbeddingArgs):
        return configs.ShardedEmbeddingConfig(
            feature_names=embedding_args.feature_names,
            table_name=embedding_args.table_name,
            vocab_size=embedding_args.item_vocab_size_or_capacity,
            dim=hidden_size,
            sharding_type="model_parallel",
        )
    return configs.ShardedEmbeddingConfig(
        feature_names=embedding_args.feature_names,
        table_name=embedding_args.table_name,
        vocab_size=embedding_args.item_vocab_size_or_capacity,
        dim=hidden_size,
        sharding_type=embedding_args.sharding_type,
    )


def create_embedding_configs(
    dataset_args: Union[DatasetArgs, BenchmarkDatasetArgs],
    network_args: NetworkArgs,
    embedding_args: List[EmbeddingArgs],
) -> List[ShardedEmbeddingConfig]:
    if (
        network_args.item_embedding_dim <= 0
        or network_args.contextual_embedding_dim <= 0
    ):
        return [
            create_embedding_config(network_args.hidden_size, arg)
            for arg in embedding_args
        ]
    if isinstance(dataset_args, DatasetArgs):
        from preprocessor import get_common_preprocessors

        common_preprocessors = get_common_preprocessors()
        dp = common_preprocessors[dataset_args.dataset_name]
        item_feature_name = dp._item_feature_name
        contextual_feature_names = dp._contextual_feature_names
        action_feature_name = dp._action_feature_name
    elif isinstance(dataset_args, BenchmarkDatasetArgs):
        item_feature_name = dataset_args.item_feature_name
        contextual_feature_names = dataset_args.contextual_feature_names
        action_feature_name = dataset_args.action_feature_name
    else:
        raise ValueError(f"Dataset args type {type(dataset_args)} not supported")

    embedding_configs = []
    for arg in embedding_args:
        if (
            item_feature_name in arg.feature_names
            or action_feature_name in arg.feature_names
        ):
            emb_config = create_embedding_config(network_args.item_embedding_dim, arg)
        else:
            if len(set(arg.feature_names) & set(contextual_feature_names)) != len(
                arg.feature_names
            ):
                raise ValueError(
                    f"feature name {arg.feature_name} not match with contextual feature names {contextual_feature_names}"
                )
            emb_config = create_embedding_config(
                network_args.contextual_embedding_dim, arg
            )
        embedding_configs.append(emb_config)
    return embedding_configs


def create_dynamic_optitons_dict(
    embedding_args_list: List[Union[EmbeddingArgs, DynamicEmbeddingArgs]],
    hidden_size: int,
    training: bool = True,
    embedding_dim_multiplier: int = 1,  # for training, we store the optimizer states together with original embedding vectors.
) -> Dict[str, DynamicEmbTableOptions]:
    dynamic_options_dict: Dict[str, DynamicEmbTableOptions] = {}
    for embedding_args in embedding_args_list:
        if isinstance(embedding_args, DynamicEmbeddingArgs):
            from dynamicemb import DynamicEmbCheckMode, DynamicEmbEvictStrategy

            embedding_args.calculate_and_reset_global_hbm_for_values(
                hidden_size, embedding_dim_multiplier
            )
            dynamic_options_dict[embedding_args.table_name] = DynamicEmbTableOptions(
                global_hbm_for_values=embedding_args.global_hbm_for_values,
                evict_strategy=DynamicEmbEvictStrategy.LRU
                if embedding_args.evict_strategy == "lru"
                else DynamicEmbEvictStrategy.LFU,
                safe_check_mode=DynamicEmbCheckMode.IGNORE,
                bucket_capacity=128,
                training=training,
                caching=embedding_args.caching,
            )
    return dynamic_options_dict


def get_dataset_and_embedding_args() -> (
    Tuple[
        Union[DatasetArgs, BenchmarkDatasetArgs],
        List[Union[DynamicEmbeddingArgs, EmbeddingArgs]],
    ]
):
    try:
        dataset_args = DatasetArgs()  # type: ignore[call-arg]
    except:
        benchmark_dataset_args = BenchmarkDatasetArgs()  # type: ignore[call-arg]
        return benchmark_dataset_args, benchmark_dataset_args.embedding_args
    assert isinstance(dataset_args, DatasetArgs)
    HASH_SIZE = 10_000_000
    if dataset_args.dataset_name == "kuairand-pure":
        return dataset_args, [
            EmbeddingArgs(
                feature_names=["user_active_degree"],
                table_name="user_active_degree",
                item_vocab_size_or_capacity=10,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["follow_user_num_range"],
                table_name="follow_user_num_range",
                item_vocab_size_or_capacity=9,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["fans_user_num_range"],
                table_name="fans_user_num_range",
                item_vocab_size_or_capacity=10,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["friend_user_num_range"],
                table_name="friend_user_num_range",
                item_vocab_size_or_capacity=8,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["register_days_range"],
                table_name="register_days_range",
                item_vocab_size_or_capacity=8,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["action_weights"],
                table_name="action_weights",
                item_vocab_size_or_capacity=226,
                sharding_type="data_parallel",
            ),
            DynamicEmbeddingArgs(
                feature_names=["video_id"],
                table_name="video_id",
                item_vocab_size_or_capacity=HASH_SIZE,
                item_vocab_gpu_capacity_ratio=0.5,
            ),
            DynamicEmbeddingArgs(
                feature_names=["user_id"],
                table_name="user_id",
                item_vocab_size_or_capacity=HASH_SIZE,
                item_vocab_gpu_capacity_ratio=0.5,
            ),
        ]
    elif dataset_args.dataset_name == "kuairand-1k":
        return dataset_args, [
            EmbeddingArgs(
                feature_names=["user_active_degree"],
                table_name="user_active_degree",
                item_vocab_size_or_capacity=8,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["follow_user_num_range"],
                table_name="follow_user_num_range",
                item_vocab_size_or_capacity=9,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["fans_user_num_range"],
                table_name="fans_user_num_range",
                item_vocab_size_or_capacity=9,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["friend_user_num_range"],
                table_name="friend_user_num_range",
                item_vocab_size_or_capacity=8,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["register_days_range"],
                table_name="register_days_range",
                item_vocab_size_or_capacity=8,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["action_weights"],
                table_name="action_weights",
                item_vocab_size_or_capacity=233,
                sharding_type="data_parallel",
            ),
            DynamicEmbeddingArgs(
                feature_names=["video_id"],
                table_name="video_id",
                item_vocab_size_or_capacity=HASH_SIZE,
                item_vocab_gpu_capacity_ratio=0.5,
            ),
            DynamicEmbeddingArgs(
                feature_names=["user_id"],
                table_name="user_id",
                item_vocab_size_or_capacity=HASH_SIZE,
                item_vocab_gpu_capacity_ratio=0.5,
            ),
        ]
    elif dataset_args.dataset_name == "kuairand-27k":
        return dataset_args, [
            EmbeddingArgs(
                feature_names=["user_active_degree"],
                table_name="user_active_degree",
                item_vocab_size_or_capacity=10,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["follow_user_num_range"],
                table_name="follow_user_num_range",
                item_vocab_size_or_capacity=9,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["fans_user_num_range"],
                table_name="fans_user_num_range",
                item_vocab_size_or_capacity=10,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["friend_user_num_range"],
                table_name="friend_user_num_range",
                item_vocab_size_or_capacity=8,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["register_days_range"],
                table_name="register_days_range",
                item_vocab_size_or_capacity=8,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["action_weights"],
                table_name="action_weights",
                item_vocab_size_or_capacity=246,
                sharding_type="data_parallel",
            ),
            DynamicEmbeddingArgs(
                feature_names=["video_id"],
                table_name="video_id",
                item_vocab_size_or_capacity=32038725,
                item_vocab_gpu_capacity_ratio=0.5,
            ),
            DynamicEmbeddingArgs(
                feature_names=["user_id"],
                table_name="user_id",
                item_vocab_size_or_capacity=HASH_SIZE,
                item_vocab_gpu_capacity_ratio=0.5,
            ),
        ]
    elif dataset_args.dataset_name == "ml-1m":
        return dataset_args, [
            EmbeddingArgs(
                feature_names=["sex"],
                table_name="sex",
                item_vocab_size_or_capacity=3,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["age_group"],
                table_name="age_group",
                item_vocab_size_or_capacity=8,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["occupation"],
                table_name="occupation",
                item_vocab_size_or_capacity=22,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["zip_code"],
                table_name="zip_code",
                item_vocab_size_or_capacity=3440,
                sharding_type="data_parallel",
            ),
            EmbeddingArgs(
                feature_names=["rating"],
                table_name="action_weights",
                item_vocab_size_or_capacity=11,
                sharding_type="data_parallel",
            ),
            DynamicEmbeddingArgs(
                feature_names=["movie_id"],
                table_name="movie_id",
                item_vocab_size_or_capacity=HASH_SIZE,
                item_vocab_gpu_capacity_ratio=0.5,
            ),
            DynamicEmbeddingArgs(
                feature_names=["user_id"],
                table_name="user_id",
                item_vocab_size_or_capacity=HASH_SIZE,
                item_vocab_gpu_capacity_ratio=0.5,
            ),
        ]
    elif dataset_args.dataset_name == "ml-20m":
        return dataset_args, [
            EmbeddingArgs(
                feature_names=["rating"],
                table_name="action_weights",
                item_vocab_size_or_capacity=11,
                sharding_type="data_parallel",
            ),
            DynamicEmbeddingArgs(
                feature_names=["movie_id"],
                table_name="movie_id",
                item_vocab_size_or_capacity=HASH_SIZE,
                item_vocab_gpu_capacity_ratio=0.5,
                caching=True,
            ),
            DynamicEmbeddingArgs(
                feature_names=["user_id"],
                table_name="user_id",
                item_vocab_size_or_capacity=HASH_SIZE,
                item_vocab_gpu_capacity_ratio=0.5,
                caching=True,
            ),
        ]
    else:
        raise ValueError(f"dataset {dataset_args.dataset_name} is not supported")
