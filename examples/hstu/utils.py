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
import os
import sys
from dataclasses import dataclass
from functools import partial  # pylint: disable-unused-import
from itertools import islice
from typing import Dict, List, Optional, Tuple, Union

import commons.checkpoint as checkpoint
import configs
import dataset
import gin
import torch  # pylint: disable-unused-import
import torch.distributed as dist
from commons.checkpoint import get_unwrapped_module
from commons.utils.distributed_utils import collective_assert
from commons.utils.gpu_timer import GPUTimer
from commons.utils.logging import print_rank_0
from commons.utils.stringify import stringify_dict
from configs import (
    HSTULayerType,
    KernelBackend,
    OptimizerParam,
    PositionEncodingConfig,
    get_hstu_config,
)
from dynamicemb import DynamicEmbTableOptions
from megatron.core import parallel_state
from megatron.core.distributed import finalize_model_grads
from model import RankingGR, RetrievalGR
from modules.embedding import ShardedEmbeddingConfig
from torchrec.distributed.model_parallel import DistributedModelParallel


@gin.configurable
@dataclass
class TrainerArgs:
    # below batchsize is batchsize_per_gpu
    train_batch_size: int
    eval_batch_size: int

    eval_interval: int = 100
    log_interval: int = 100

    seed: int = 1234
    # ==nsys args==
    profile: bool = False
    profile_step_start: int = 100
    profile_step_end: int = 200
    # ==nsys args==
    max_train_iters: Optional[int] = None
    max_eval_iters: Optional[int] = None

    # ckpt args
    ckpt_save_interval: int = -1  # -1 means only save at the end
    ckpt_save_dir: str = "./checkpoints"
    ckpt_load_dir: str = ""

    def __post_init__(self):
        if isinstance(self.max_train_iters, str):
            self.max_train_iters = int(self.max_train_iters)


@dataclass
class BaseEmbeddingArgs:
    # for dynamic emb, it serves as capacity, while for static emb, it serves as vocab size
    feature_names: List[str]
    table_name: str
    item_vocab_size_or_capacity: int


@gin.configurable
@dataclass
class EmbeddingArgs(BaseEmbeddingArgs):
    sharding_type: str = "None"

    def __post_init__(self):
        assert self.sharding_type.lower() in [
            "data_parallel",
            "model_parallel",
        ]


@gin.configurable
@dataclass
class DynamicEmbeddingArgs(EmbeddingArgs):
    # the precedence is global_hbm_for_values > item_vocab_gpu_capacity > item_vocab_gpu_capacity_ratio
    # without optimizer consideration
    global_hbm_for_values: Optional[int] = None
    item_vocab_gpu_capacity: Optional[float] = None
    item_vocab_gpu_capacity_ratio: Optional[float] = None

    evict_strategy: str = "lru"

    def __post_init__(self):
        assert self.evict_strategy.lower() in ["lru", "lfu"]

    def calculate_and_reset_global_hbm_for_values(self, hidden_size):
        if self.global_hbm_for_values is not None:
            return
        assert (
            self.item_vocab_gpu_capacity_ratio is not None
            or self.item_vocab_gpu_capacity is not None
        ), "Please provide either item_vocab_gpu_capacity_ratio or item_vocab_gpu_capacity"
        if self.item_vocab_gpu_capacity is None:
            self.item_vocab_gpu_capacity = int(
                self.item_vocab_size_or_capacity * self.item_vocab_gpu_capacity_ratio
            )
        self.global_hbm_for_values = self.item_vocab_gpu_capacity * hidden_size * 4


@gin.configurable
@dataclass
class DatasetArgs:
    dataset_name: str
    max_sequence_length: int
    max_num_candidates: int = 0
    shuffle: bool = False


@gin.configurable
@dataclass
class FeatureArgs:
    feature_names: List[str]
    max_sequence_length: int
    is_jagged: bool = False


@gin.configurable
@dataclass
class BenchmarkDatasetArgs:
    feature_args: List[FeatureArgs]
    embedding_args: List[Union[EmbeddingArgs, DynamicEmbeddingArgs]]
    item_feature_name: str
    contextual_feature_names: List[str]
    action_feature_name: Optional[str] = None
    max_num_candidates: int = 0


@gin.configurable
@dataclass
class NetworkArgs:
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    kv_channels: int

    hidden_dropout: float = 0.2
    norm_epsilon: float = 1e-5
    is_causal: bool = True

    dtype_str: str = "bfloat16"

    kernel_backend: str = "cutlass"
    layer_type: str = "fused"
    target_group_size: int = 1

    num_position_buckets: int = 8192

    def __post_init__(self):
        assert self.dtype_str in [
            "bfloat16",
            "float16",
        ], "Only support bfloat16 and float16 precision for Network."

        assert self.kernel_backend.lower() in ["cutlass", "triton", "pytorch"]
        assert self.layer_type.lower() in ["fused", "native"]


@gin.configurable
@dataclass
class OptimizerArgs:
    optimizer_str: str
    learning_rate: float
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


@gin.configurable
@dataclass
class TensorModelParallelArgs:
    tensor_model_parallel_size: int = 1


def create_hstu_config(network_args: NetworkArgs):
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
    if network_args.layer_type == "fused":
        layer_type = HSTULayerType.FUSED
    elif network_args.layer_type == "native":
        layer_type = HSTULayerType.NATIVE
    else:
        raise ValueError(f"Layer type {network_args.layer_type} is not supported.")
    position_encoding_config = PositionEncodingConfig(
        num_position_buckets=network_args.num_position_buckets,
        num_time_buckets=2048,
        use_time_encoding=False,
    )
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
        position_encoding_config=position_encoding_config,
        target_group_size=network_args.target_group_size,
        hstu_layer_type=layer_type,
    )


def get_data_loader(
    task_type: str,
    dataset_args: Union[DatasetArgs, BenchmarkDatasetArgs],
    trainer_args: TrainerArgs,
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
            num_tasks=1 if task_type == "ranking" else None,
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
            max_sequence_length=dataset_args.max_sequence_length,
            max_num_candidates=dataset_args.max_num_candidates,
            num_tasks=1 if task_type == "ranking" else 0,
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


def create_dynamic_optitons_dict(
    embedding_args_list: List[Union[EmbeddingArgs, DynamicEmbeddingArgs]],
    hidden_size: int,
) -> Dict[str, DynamicEmbTableOptions]:
    dynamic_options_dict: Dict[str, DynamicEmbTableOptions] = {}
    for embedding_args in embedding_args_list:
        if isinstance(embedding_args, DynamicEmbeddingArgs):
            from dynamicemb import DynamicEmbCheckMode, DynamicEmbEvictStrategy

            embedding_args.calculate_and_reset_global_hbm_for_values(hidden_size)
            dynamic_options_dict[embedding_args.table_name] = DynamicEmbTableOptions(
                global_hbm_for_values=embedding_args.global_hbm_for_values,
                evict_strategy=DynamicEmbEvictStrategy.LRU
                if embedding_args.evict_strategy == "lru"
                else DynamicEmbEvictStrategy.LFU,
                safe_check_mode=DynamicEmbCheckMode.IGNORE,
                bucket_capacity=128,
            )
    return dynamic_options_dict


def evaluate(
    model: Union[RankingGR, RetrievalGR],
    trainer_args: TrainerArgs,
    eval_loader: torch.utils.data.DataLoader,
    max_eval_iters: Optional[int] = None,
):
    eval_iter = 0
    torch.cuda.nvtx.range_push(f"#evaluate")
    with torch.no_grad():
        # drop last batch
        for batch in islice(eval_loader, len(eval_loader)):
            batch = batch.to("cuda")
            eval_iter += 1
            model.evaluate_one_batch(batch)
            if max_eval_iters is not None and eval_iter == max_eval_iters:
                break

        eval_metric_dict = model.compute_metric()
        dp_size = parallel_state.get_data_parallel_world_size()
    # TODO, fix the samples when there is incomplete batch
    print_rank_0(
        f"[eval] [eval {eval_iter * dp_size * trainer_args.eval_batch_size} samples]:\n    "
        + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
    )
    torch.cuda.nvtx.range_pop()


def maybe_load_ckpts(
    ckpt_load_dir: str,
    model: Union[RankingGR, RetrievalGR],
    dense_optimizer: Optional[torch.optim.Optimizer] = None,
):
    if ckpt_load_dir == "":
        return

    assert os.path.exists(
        ckpt_load_dir
    ), f"ckpt_load_dir {ckpt_load_dir} does not exist"

    print_rank_0(f"Loading checkpoints from {ckpt_load_dir}")
    checkpoint.load(ckpt_load_dir, model, dense_optimizer=dense_optimizer)
    print_rank_0(f"Checkpoints loaded!!")


def save_ckpts(
    ckpt_save_dir: str,
    model: Union[RankingGR, RetrievalGR],
    dense_optimizer: Optional[torch.optim.Optimizer] = None,
):
    print_rank_0(f"Saving checkpoints to {ckpt_save_dir}")
    import shutil

    if dist.get_rank() == 0:
        if os.path.exists(ckpt_save_dir):
            shutil.rmtree(ckpt_save_dir)
        try:
            os.makedirs(ckpt_save_dir, exist_ok=True)
        except Exception as e:
            raise Exception("can't build path:", ckpt_save_dir) from e
    dist.barrier(device_ids=[torch.cuda.current_device()])
    checkpoint.save(ckpt_save_dir, model, dense_optimizer=dense_optimizer)
    print_rank_0(f"Checkpoints saved!!")


def train(
    model: DistributedModelParallel,
    trainer_args: TrainerArgs,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    dense_optimizer: torch.optim.Optimizer,
):
    gpu_timer = GPUTimer()
    max_train_iters = trainer_args.max_train_iters or len(train_loader)
    dp_size = parallel_state.get_data_parallel_world_size() * 1.0
    gpu_timer.start()
    last_td = 0
    # using a tensor on gpu to avoid d2h copy
    tokens_logged = torch.zeros(1).cuda().float()
    train_loader_iter = iter(train_loader)
    for train_iter in range(max_train_iters):
        if trainer_args.profile and train_iter == trainer_args.profile_step_start:
            torch.cuda.profiler.start()
        if (
            train_iter * trainer_args.ckpt_save_interval > 0
            and train_iter % trainer_args.ckpt_save_interval == 0
        ):
            save_path = os.path.join(trainer_args.ckpt_save_dir, f"iter{train_iter}")
            save_ckpts(save_path, model, dense_optimizer)

        torch.cuda.nvtx.range_push(f"step {train_iter}")

        try:
            batch = next(train_loader_iter)
        except StopIteration:
            # Reset iterator when we reach the end
            train_loader_iter = iter(train_loader)
            batch = next(train_loader_iter)

        batch = batch.to("cuda")
        model.module.zero_grad_buffer()
        dense_optimizer.zero_grad()

        # shape = [T, num_event]
        losses, (_, logits, labels) = model(batch)
        collective_assert(not torch.isnan(losses).any(), "loss has nan value")
        jagged_size = logits.size(0)
        local_tokens = torch.tensor(jagged_size).cuda().float()

        losses = torch.sum(losses, dim=0)
        local_loss = torch.cat([torch.sum(losses).view(1), local_tokens.view(1)])
        reporting_loss = local_loss.clone().detach()
        # [allreduced_sum_loss, allreduced_sum_tokens]
        torch.distributed.all_reduce(
            reporting_loss, group=parallel_state.get_data_parallel_group()
        )
        tokens_logged += reporting_loss[1]
        if train_iter >= 0 and train_iter % trainer_args.log_interval == 0:
            gpu_timer.stop()
            cur_td = gpu_timer.elapsed_time() - last_td
            print_rank_0(
                f"[train] [iter {train_iter}, tokens {int(tokens_logged.item())}, elapsed_time {cur_td:.2f} ms]: loss {reporting_loss[0] / reporting_loss[1]:.6f}"
            )
            last_td = cur_td + last_td
            tokens_logged.zero_()

        # backward, loss_sum / total_tokens * mcore_dp_size
        local_loss_average = local_loss[0] / reporting_loss[1] * dp_size
        local_loss_average.backward()

        # dense gradient allreduce
        finalize_model_grads([model.module], None)
        torch.cuda.nvtx.range_push(f"#dense opt")
        dense_optimizer.step()
        torch.cuda.nvtx.range_pop()
        if train_iter > 0 and train_iter % trainer_args.eval_interval == 0:
            model.eval()
            evaluate(
                get_unwrapped_module(model),
                trainer_args=trainer_args,
                eval_loader=eval_loader,
                max_eval_iters=None,
            )
            model.train()
        torch.cuda.nvtx.range_pop()

        if trainer_args.profile and train_iter == trainer_args.profile_step_end:
            torch.cuda.profiler.stop()

    if trainer_args.ckpt_save_interval == -1:
        save_ckpts(trainer_args.ckpt_save_dir, model, dense_optimizer)


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
                item_vocab_size_or_capacity=7583,
                item_vocab_gpu_capacity_ratio=1.0,
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
                item_vocab_size_or_capacity=4371900,
                item_vocab_gpu_capacity_ratio=1.0,
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
                item_vocab_gpu_capacity_ratio=1.0,
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
                item_vocab_size_or_capacity=3953,
                item_vocab_gpu_capacity_ratio=1.0,
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
                item_vocab_size_or_capacity=131263,
                item_vocab_gpu_capacity_ratio=1.0,
            ),
        ]
    else:
        raise ValueError(f"dataset {dataset_args.dataset_name} is not supported")
