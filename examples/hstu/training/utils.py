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
from modules.embedding import ShardedEmbeddingConfig, ShardedEmbedding
from training.gin_config_args import (
    BenchmarkDatasetArgs,
    DatasetArgs,
    DynamicEmbeddingArgs,
    EmbeddingArgs,
    NetworkArgs,
    OptimizerArgs,
    TensorModelParallelArgs,
    MixedPrecisionArgs,
    TrainerArgs,
)
from commons.utils.logger import print_rank_0

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
    network_args: NetworkArgs, tensor_model_parallel_args: TensorModelParallelArgs, mp_args: MixedPrecisionArgs
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

    # Define HSTU attn quantization separate from TE FP8
    hstu_attn_quantization_mode = mp_args.hstu_attn_quantization_map[mp_args.hstu_attn_quantization_mode]

    if mp_args.enabled:
        # Matching Megatron FP8 arguments
        fp8 = mp_args.linear_scaling_precision  # Flag to set both te linear and precision https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py
        fp8_recipe = mp_args.linear_recipe
        fp8_alignment_mode = mp_args.fp8_alignment_mode
    else:
        fp8 = None
        fp8_recipe = None
        fp8_alignment_mode = None

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
        hstu_attn_quantization_mode=hstu_attn_quantization_mode,
        fp8_alignment_mode=fp8_alignment_mode,
        hstu_preprocessing_config=hstu_preprocessing_config,
        position_encoding_config=position_encoding_config,
        target_group_size=network_args.target_group_size,
        hstu_layer_type=layer_type,
        recompute_input_layernorm=network_args.recompute_input_layernorm,
        recompute_input_silu=network_args.recompute_input_silu,
        fp8 = fp8,
        fp8_recipe = fp8_recipe,
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

def inspect_sharded_embedding_tables(embedding_collection: ShardedEmbedding, table_name_filter: str = None, tracking_state: dict = None) -> dict:
    """
    Helper function to inspect all embedding tables in a ShardedEmbedding collection.
    Works with both regular embeddings and dynamic embeddings.
    
    Args:
        embedding_collection: The ShardedEmbedding instance to inspect
        table_name_filter: Optional filter to only show tables containing this string
        tracking_state: Optional dict to track IDs across iterations {table_name: {'all_ids': set(), 'nan_ids': set()}}
    
    Returns:
        Updated tracking_state dict
    """
    if tracking_state is None:
        tracking_state = {}
    print_rank_0("=" * 80)
    print_rank_0("EMBEDDING COLLECTION INSPECTION")
    print_rank_0("=" * 80)
    
    # Check model-parallel embeddings
    if embedding_collection._model_parallel_embedding_collection is not None:
        print_rank_0("\n[MODEL-PARALLEL EMBEDDINGS]")
        mp_collection = embedding_collection._model_parallel_embedding_collection
        
        # Try to get dynamic embedding modules
        try:
            from dynamicemb.dump_load import get_dynamic_emb_module
            dynamic_modules = get_dynamic_emb_module(mp_collection)
            
            if len(dynamic_modules) > 0:
                print_rank_0(f"Found {len(dynamic_modules)} dynamic embedding module(s)")
                for module_idx, dyn_module in enumerate(dynamic_modules):
                    print_rank_0(f"\n  Dynamic Module {module_idx}:")
                    for table_idx, (table_name, table) in enumerate(zip(dyn_module.table_names, dyn_module.tables)):
                        if table_name_filter and table_name_filter not in table_name:
                            continue
                        print_rank_0(f"    Table '{table_name}':")
                        print_rank_0(f"      Type: Dynamic Embedding (KeyValueTable)")
                        capacity = table.capacity()
                        used_size = table.size()
                        emb_dim = table.embedding_dim()
                        opt_dim = table.optim_state_dim()
                        
                        print_rank_0(f"      Capacity: {capacity}")
                        print_rank_0(f"      Size (used): {used_size}")
                        print_rank_0(f"      Embedding dim: {emb_dim}")
                        print_rank_0(f"      Optimizer state dim: {opt_dim}")
                        print_rank_0(f"      Effective shape: [{used_size}, {emb_dim}] (sparse)")
                        
                        # Check ALL embeddings for NaNs (sparse hash table requires full scan)
                        if used_size > 0:
                            from dynamicemb.dump_load import export_keys_values
                            device = torch.device(f"cuda:{torch.cuda.current_device()}")
                            
                            try:
                                total_checked = 0
                                total_nan_count = 0
                                all_embedding_ids = []
                                ids_with_nan = []
                                emb_min = float('inf')
                                emb_max = float('-inf')
                                has_any_nan = False
                                
                                # Scan through entire hash table to find all stored embeddings
                                for keys, embeddings, opt_states, scores in export_keys_values(table, device, batch_size=65536):
                                    batch_size_actual = embeddings.shape[0]
                                    total_checked += batch_size_actual
                                    
                                    # Track all embedding IDs
                                    all_embedding_ids.extend(keys.tolist())
                                    
                                    # Check for NaNs and get the actual movie IDs with NaN
                                    if torch.isnan(embeddings).any():
                                        has_any_nan = True
                                        nan_mask = embeddings.isnan().any(dim=1)  # Which embeddings have NaN
                                        nan_per_embedding = embeddings.isnan().sum(dim=1)  # Count NaNs per embedding
                                        total_nan_count += torch.isnan(embeddings).sum().item()
                                        ids_with_nan_in_batch = keys[nan_mask].tolist()  # Get actual movie IDs
                                        ids_with_nan.extend(ids_with_nan_in_batch)
                                        
                                        # Store NaN pattern for detailed analysis
                                        if 'nan_patterns' not in tracking_state:
                                            tracking_state['nan_patterns'] = {}
                                        if table_name not in tracking_state['nan_patterns']:
                                            # For each ID with NaN, store how many dimensions have NaN
                                            nan_pattern = {
                                                int(keys[i].item()): int(nan_per_embedding[i].item()) 
                                                for i in range(len(keys)) if nan_mask[i]
                                            }
                                            tracking_state['nan_patterns'][table_name] = nan_pattern
                                    else:
                                        emb_min = min(emb_min, embeddings.min().item())
                                        emb_max = max(emb_max, embeddings.max().item())
                                
                                print_rank_0(f"      Scanned {total_checked}/{used_size} embeddings")
                                print_rank_0(f"      Has NaN: {has_any_nan}")
                                if has_any_nan:
                                    print_rank_0(f"      Total NaN count: {total_nan_count}")
                                    total_elements = total_checked * emb_dim
                                    print_rank_0(f"      NaN percentage: {100.0 * total_nan_count / total_elements:.2f}%")
                                    print_rank_0(f"      Number of embeddings with NaN: {len(ids_with_nan)}")
                                    print_rank_0(f"      IDs with NaN: {ids_with_nan}")
                                    
                                    # Print NaN pattern analysis
                                    if 'nan_patterns' in tracking_state and table_name in tracking_state['nan_patterns']:
                                        nan_pattern = tracking_state['nan_patterns'][table_name]
                                        fully_nan = sum(1 for count in nan_pattern.values() if count == emb_dim)
                                        partially_nan = len(nan_pattern) - fully_nan
                                        print_rank_0(f"      NaN Pattern:")
                                        print_rank_0(f"        Fully NaN embeddings: {fully_nan}/{len(nan_pattern)} (all {emb_dim} dims)")
                                        print_rank_0(f"        Partially NaN embeddings: {partially_nan}/{len(nan_pattern)}")
                                        if partially_nan > 0:
                                            # Show examples of partial NaN
                                            partial_examples = [(id_, count) for id_, count in list(nan_pattern.items())[:5] if count < emb_dim]
                                            if partial_examples:
                                                print_rank_0(f"        Partial NaN examples (ID: NaN_count): {partial_examples}")
                                    
                                    # Store for comparison across iterations
                                    if table_name not in tracking_state:
                                        tracking_state[table_name] = {'all_ids': set(), 'nan_ids': set()}
                                    
                                    prev_all_ids = tracking_state[table_name]['all_ids']
                                    prev_nan_ids = tracking_state[table_name]['nan_ids']
                                    current_all_ids = set(all_embedding_ids)
                                    current_nan_ids = set(ids_with_nan)
                                    
                                    new_ids = current_all_ids - prev_all_ids
                                    new_ids_with_nan = current_nan_ids & new_ids
                                    old_ids_with_nan = current_nan_ids - new_ids
                                    
                                    print_rank_0(f"      New embeddings this iteration: {len(new_ids)}")
                                    print_rank_0(f"      New embeddings with NaN: {len(new_ids_with_nan)} (IDs: {list(new_ids_with_nan)[:10]})")
                                    print_rank_0(f"      Old embeddings corrupted: {len(old_ids_with_nan)} (IDs: {list(old_ids_with_nan)[:10]})")
                                    
                                    # Update tracking
                                    tracking_state[table_name]['all_ids'] = current_all_ids
                                    tracking_state[table_name]['nan_ids'] = current_nan_ids
                                else:
                                    print_rank_0(f"      Min/Max: {emb_min:.4f} / {emb_max:.4f}")
                                    
                                    # Track IDs even when no NaN for future comparison
                                    if table_name not in tracking_state:
                                        tracking_state[table_name] = {'all_ids': set(), 'nan_ids': set()}
                                    tracking_state[table_name]['all_ids'] = set(all_embedding_ids)
                            except Exception as e:
                                print_rank_0(f"      Error scanning embeddings: {e}")
        except ImportError:
            print_rank_0("  Dynamic embeddings module not available")
        
        # Check regular (non-dynamic) embeddings via state_dict
        print_rank_0("\n  Regular embeddings in state_dict:")
        for name, tensor in mp_collection.state_dict().items():
            if table_name_filter and table_name_filter not in name:
                continue
            print_rank_0(f"    '{name}':")
            
            if hasattr(tensor, "local_shards"):
                print_rank_0(f"      Type: ShardedTensor (model-parallel)")
                for shard_idx, shard in enumerate(tensor.local_shards()):
                    shard_tensor = shard.tensor
                    print_rank_0(f"      Shard {shard_idx}:")
                    print_rank_0(f"        Shape: {shard_tensor.shape}")
                    print_rank_0(f"        Offsets: {shard.metadata.shard_offsets}")
                    print_rank_0(f"        Sizes: {shard.metadata.shard_sizes}")
                    print_rank_0(f"        Has NaN: {torch.isnan(shard_tensor).any()}")
                    if torch.isnan(shard_tensor).any():
                        print_rank_0(f"        NaN count: {torch.isnan(shard_tensor).sum()}")
            else:
                print_rank_0(f"      Type: Regular Tensor")
                print_rank_0(f"      Shape: {tensor.shape}")
                print_rank_0(f"      Has NaN: {torch.isnan(tensor).any()}")
                if torch.isnan(tensor).any():
                    print_rank_0(f"      NaN count: {torch.isnan(tensor).sum()}")
    
    # Check data-parallel embeddings
    if embedding_collection._data_parallel_embedding_collection is not None:
        print_rank_0("\n[DATA-PARALLEL EMBEDDINGS]")
        dp_collection = embedding_collection._data_parallel_embedding_collection
        
        if hasattr(dp_collection, 'embedding_weights'):
            for table_name, weight_tensor in dp_collection.embedding_weights.items():
                if table_name_filter and table_name_filter not in table_name:
                    continue
                print_rank_0(f"  Table '{table_name}':")
                print_rank_0(f"    Shape: {weight_tensor.shape}")
                print_rank_0(f"    Has NaN: {torch.isnan(weight_tensor).any()}")
                if torch.isnan(weight_tensor).any():
                    print_rank_0(f"    NaN count: {torch.isnan(weight_tensor).sum()}")
    
    print_rank_0("=" * 80)
    return tracking_state