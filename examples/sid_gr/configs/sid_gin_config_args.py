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
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import gin


@gin.configurable
@dataclass
class TrainerArgs:
    """Trainer Configuration.

    Training-related parameters and settings.

    Attributes:
        train_batch_size (int): **Required**. Batch size per GPU. When TP is enabled,
            the theoretical batch size is (train_batch_size × tp_size).
        eval_batch_size (int): **Required**. Evaluation batch size.
        eval_interval (int): Evaluation interval in iterations. Default: 100.
        log_interval (int): Logging interval in iterations. Default: 100.
        top_k_for_generation (int): Top K items to generate(retrieve) during evaluation. Default: 10.
        eval_metrics (Tuple[str, ...]): Evaluation metrics (e.g., "HR@2", "NDCG@10").
            Default: ("HR@2", "NDCG@10").
        max_train_iters (Optional[int]): Maximum training iterations. Default: None.
        max_eval_iters (Optional[int]): Maximum evaluation iterations. Default: None.
        seed (int): Random seed. Default: 1234.
        profile (bool): Enable profiling. Default: False.
        profile_step_start (int): Profiling start step. Default: 100.
        profile_step_end (int): Profiling end step. Default: 200.
        ckpt_save_interval (int): Checkpoint save interval, -1 means no checkpoint saving.
            Default: -1.
        ckpt_save_dir (str): Checkpoint save directory. Default: "./checkpoints".
        ckpt_load_dir (str): Checkpoint load directory. Default: "".
        log_dir (str): Log directory. Default: "./logs".
        pipeline_type (str): Pipeline overlap type: 'none' (no overlap), 'native'
            (overlap h2d, input dist, fwd+bwd), 'prefetch' (includes prefetch overlap).
            Default: "native".
    """

    # below batchsize is batchsize_per_gpu
    # when TP is enabled, the theoratical batchsize is (train_batch_size * tp_size)
    train_batch_size: int
    eval_batch_size: int

    eval_interval: int = 100
    log_interval: int = 100

    top_k_for_generation: int = 10
    eval_metrics: Tuple[str, ...] = field(
        default_factory=lambda: ("HitRate@2", "NDCG@10")
    )

    max_train_iters: Optional[int] = None
    max_eval_iters: Optional[int] = None
    seed: int = 1234

    # ==profile args==
    profile: bool = False
    profile_step_start: int = 100
    profile_step_end: int = 200
    # ==ckpt args==
    ckpt_save_interval: int = -1  # -1 means not save ckpt
    ckpt_save_dir: str = "./checkpoints"
    ckpt_load_dir: str = ""

    # log_dir
    log_dir: str = "./logs"

    # overlap pipeline type
    # - none -> no overlap
    # - native -> overlap [h2d, input dist, fwd+bwd]
    # - prefetch -> overlap [h2d, input dist, prefetch, fwd+bwd]
    pipeline_type: str = "native"  # none, native, prefetch

    def __post_init__(self):
        if isinstance(self.max_train_iters, str):
            self.max_train_iters = int(self.max_train_iters)
        for metric_spec in self.eval_metrics:
            metric_name, top_k = metric_spec.split("@")
            assert metric_name.lower() in [
                "ndcg",
                "recall",
                "hitrate",
            ], "invalid metric name"
            assert (
                int(top_k) <= self.top_k_for_generation
            ), "top_k for evaluation should be less than top_k for generation"


@gin.configurable
@dataclass
class EmbeddingArgs:
    """Embedding Configuration.

    Base embedding layer configuration parameters.

    Attributes:
        feature_names (List[str]): **Required**. List of feature names.
        table_name (str): **Required**. Embedding table name.
        item_vocab_size_or_capacity (int): **Required**. For dynamic embedding: capacity;
            for static embedding: vocabulary size.
        sharding_type (str): Sharding type, must be "data_parallel" or "model_parallel".
            Default: "None".

    Note:
        A table could be only one of type `EmbeddingArgs`.
        When movielen* or kuairand* datasets are used, `EmbeddingArgs`
        are predefined. Setting the proper DatasetArgs.dataset_name in the gin config file will automatically set the proper EmbeddingArgs.
        See `examples/sid_gr/data/sid_data_loader.py::get_train_and_test_data_loader()` for more details.
    """

    feature_names: List[str]
    table_name: str
    item_vocab_size_or_capacity: int

    sharding_type: str = "data_parallel"

    def __post_init__(self):
        assert self.sharding_type.lower() in [
            "data_parallel",
            "model_parallel",
        ]


class DatasetType(Enum):
    """
    Dataset type:
    - InMemoryRandomDataset: in-memory random dataset, used for debugging and testing.
    - DiskSequenceDataset: disk-based sequence dataset, used for training and evaluation.
    """

    InMemoryRandomDataset = "in_memory_random_dataset"
    DiskSequenceDataset = "disk_sequence_dataset"


@gin.configurable
@dataclass
class DatasetArgs:
    """Dataset Configuration.

    Dataset-related configuration parameters.

    Attributes:
        dataset_name (str): **Required**. Dataset name.
        max_history_length (int): **Required**. Maximum history length.
        dataset_type (DatasetType): Dataset type. Default: DatasetType.InMemoryRandomDataset.
        dataset_type_str (str): Dataset type string. Default: "in_memory_random_dataset".
        sequence_features_training_data_path (Optional[str]): Path to training data. Default: None.
        sequence_features_testing_data_path (Optional[str]): Path to testing data. Default: None.
        shuffle (bool): Whether to shuffle data. Default: False.
        item_to_sid_mapping_path (Optional[str]): Path to item to sid mapping. Default: None.
        num_hierarchies (int): Number of hierarchies. Default: 4.
        codebook_sizes (List[int]): Codebook sizes. Default: [500] * 4.
        max_candidate_length (int): Maximum candidate length. Default: 1.
        deduplicate_label_across_hierarchy (bool): Whether to deduplicate label across hierarchy. User should not set this explicitly. This is equal to share_lm_head_across_hierarchies.
    """

    dataset_name: str
    max_history_length: int
    dataset_type: DatasetType = DatasetType.InMemoryRandomDataset
    dataset_type_str: str = "in_memory_random_dataset"
    sequence_features_training_data_path: Optional[
        str
    ] = None  # None when dataset_type is InMemoryRandomDataset
    sequence_features_testing_data_path: Optional[
        str
    ] = None  # None when dataset_type is InMemoryRandomDataset
    shuffle: bool = False

    # below are used to describe the sid features
    item_to_sid_mapping_path: Optional[
        str
    ] = None  # None when dataset_type is InMemoryRandomDataset or the dataset is already sid features
    num_hierarchies: int = 4
    codebook_sizes: List[int] = field(default_factory=lambda: [500] * 4)
    max_candidate_length: int = 1

    # below are used to describe the sid features in the dataset batch
    # and the embedding feature names should match the dataset batch feature names
    _history_sid_feature_name: str = "hist_sids"
    _candidate_sid_feature_name: str = "cand_sids"
    deduplicate_label_across_hierarchy: bool = False

    def __post_init__(self):
        assert (
            len(self.codebook_sizes) == self.num_hierarchies
        ), "codebook_sizes should have the same length as num_hierarchies"
        assert self.dataset_type_str.lower() in [
            "in_memory_random_dataset",
            "disk_sequence_dataset",
        ], "dataset_type_str should be in ['in_memory_random_dataset', 'disk_sequence_dataset']"
        if self.dataset_type_str == "in_memory_random_dataset":
            self.dataset_type = DatasetType.InMemoryRandomDataset
        elif self.dataset_type_str == "disk_sequence_dataset":
            self.dataset_type = DatasetType.DiskSequenceDataset
        else:
            raise ValueError(f"Invalid dataset type: {self.dataset_type_str}")


@gin.configurable
@dataclass
class NetworkArgs:
    """Network Architecture Configuration.

    Neural network architecture parameters.

    Attributes:
        num_layers (int): **Required**. Number of layers.
        hidden_size (int): **Required**. Hidden layer size.
        num_attention_heads (int): **Required**. Number of attention heads.
        kv_channels (int): **Required**. Key-value channels.
        hidden_dropout (float): Hidden layer dropout rate. Default: 0.2.
        norm_epsilon (float): Normalization epsilon. Default: 1e-5.
        is_causal (bool): Use causal attention mask. Default: True.
        dtype_str (str): Data type: "bfloat16" or "float16". Default: "bfloat16".
        share_lm_head_across_hierarchies (bool): Whether to share language model head
            across hierarchies. Default: True.
    """

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    kv_channels: int

    hidden_dropout: float = 0.2
    norm_epsilon: float = 1e-5
    is_causal: bool = True

    dtype_str: str = "bfloat16"
    share_lm_head_across_hierarchies: bool = True


@gin.configurable
@dataclass
class OptimizerArgs:
    """Optimizer Configuration.

    Optimizer-related parameters.

    Attributes:
        optimizer_str (str): **Required**. Optimizer name.
        learning_rate (float): **Required**. Learning rate.
        adam_beta1 (float): Adam optimizer beta1 parameter. Default: 0.9.
        adam_beta2 (float): Adam optimizer beta2 parameter. Default: 0.999.
        adam_eps (float): Adam optimizer epsilon parameter. Default: 1e-8.
        weight_decay (float): Weight decay parameter. Default: 0.01.
    """

    optimizer_str: str
    learning_rate: float
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.01


@gin.configurable
@dataclass
class TensorModelParallelArgs:
    """Tensor Model Parallelism Configuration.

    Tensor model parallelism settings.

    Attributes:
        tensor_model_parallel_size (int): Tensor model parallel size (number of GPUs
            for model sharding). Default: 1.

    Note:
        The data parallel size is deduced based on the world_size and
        tensor_model_parallel_size.
    """

    tensor_model_parallel_size: int = 1
