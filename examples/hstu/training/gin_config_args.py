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
from dataclasses import dataclass
from typing import List, Optional, Union

import gin


@gin.configurable
@dataclass
class TrainerArgs:
    # below batchsize is batchsize_per_gpu
    # when TP is enabled, the theoratical batchsize is (train_batch_size * tp_size)
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
    ckpt_save_interval: int = -1  # -1 means not save ckpt
    ckpt_save_dir: str = "./checkpoints"
    ckpt_load_dir: str = ""
    pipeline_type: str = "native"  # none, native, prefetch

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
        self.sharding_type = "model_parallel"
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
    dataset_path: Optional[str] = None
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

    recompute_input_layernorm: bool = False
    recompute_input_silu: bool = False

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
