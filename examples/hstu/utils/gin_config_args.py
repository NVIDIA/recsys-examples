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
from typing import List, Optional, Tuple, Union, cast

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

    # overlap pipeline type
    # - none -> no overlap
    # - native -> overlap [h2d, input dist, fwd+bwd]
    # - prefetch -> overlap [h2d, input dist, prefetch, fwd+bwd]
    pipeline_type: str = "native"  # none, native, prefetch

    def __post_init__(self):
        if isinstance(self.max_train_iters, str):
            self.max_train_iters = int(self.max_train_iters)


@dataclass
class BaseEmbeddingArgs:
    """Base Embedding Arguments.

    Base class for embedding configuration parameters.

    Attributes:
        feature_names (List[str]): List of feature names.
        table_name (str): Embedding table name.
        item_vocab_size_or_capacity (int): For dynamic embedding: capacity;
            for static embedding: vocabulary size.
    """

    # for dynamic emb, it serves as capacity, while for static emb, it serves as vocab size
    feature_names: List[str]
    table_name: str
    item_vocab_size_or_capacity: int


@gin.configurable
@dataclass
class EmbeddingArgs(BaseEmbeddingArgs):
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
        A table could be only one of type `EmbeddingArgs` or `DynamicEmbeddingArgs`.
        When movielen* or kuairand* datasets are used, `DynamicEmbeddingArgs`/`EmbeddingArgs`
        are predefined. Setting the proper DatasetArgs.dataset_name in the gin config file will automatically set the proper EmbeddingArgs/DynamicEmbeddingArgs.
        See `examples/hstu/training/trainer/utils.py::get_dataset_and_embedding_args()` for more details.
    """

    sharding_type: str = "None"

    def __post_init__(self):
        assert self.sharding_type.lower() in [
            "data_parallel",
            "model_parallel",
        ]


@gin.configurable
@dataclass
class DynamicEmbeddingArgs(EmbeddingArgs):
    """Dynamic Embedding Configuration.

    Extends EmbeddingArgs with dynamic embedding-specific parameters.

    Attributes:
        global_hbm_for_values (Optional[int]): Global HBM size in bytes (highest priority).
            Default: None.
        item_vocab_gpu_capacity (Optional[float]): Item vocabulary GPU capacity
            (second priority). Default: None.
        item_vocab_gpu_capacity_ratio (Optional[float]): Item vocabulary GPU capacity ratio
            (lowest priority). Default: None.
        evict_strategy (str): Eviction strategy: "lru" or "lfu". Default: "lru".
        caching (bool): Enable caching on HBM. When caching is enabled, the
            global_hbm_for_values indicates the cache size. Default: False.

    Note:
        - sharding_type is automatically set to "model_parallel".
        - Precedence: The first 3 params can be used for setting the HBM size for dynamic
          embedding, with precedence: `global_hbm_for_values` > `item_vocab_gpu_capacity` >
          item_vocab_gpu_capacity_ratio. When only item_vocab_gpu_capacity_ratio is given,
          `item_vocab_gpu_capacity` = `item_vocab_gpu_capacity_ratio` * `item_vocab_size_or_capacity`
          and `global_hbm_for_values` are deduced based on the optimizer and embedding dims.
    """

    # the precedence is `global_hbm_for_values` > `item_vocab_gpu_capacity` > `item_vocab_gpu_capacity_ratio`
    # without optimizer consideration
    global_hbm_for_values: Optional[int] = None
    item_vocab_gpu_capacity: Optional[float] = None
    item_vocab_gpu_capacity_ratio: Optional[float] = None

    evict_strategy: str = "lru"
    caching: bool = False

    def __post_init__(self):
        self.sharding_type = "model_parallel"
        assert self.evict_strategy.lower() in ["lru", "lfu"]

    def calculate_and_reset_global_hbm_for_values(self, hidden_size, multiplier=1):
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
        self.global_hbm_for_values = (
            self.item_vocab_gpu_capacity * hidden_size * 4 * multiplier
        )  # we assume the embedding vector storage precision is fp32


@gin.configurable
@dataclass
class DatasetArgs:
    """Dataset Configuration.

    Dataset-related configuration parameters.

    Attributes:
        dataset_name (str): **Required**. Dataset name.
        max_sequence_length (int): **Required**. Maximum sequence length.
        dataset_path (Optional[str]): Path to dataset. Default: None.
        max_num_candidates (int): Maximum number of candidates. Default: 0.
        shuffle (bool): Whether to shuffle data. Default: False.

    Note:
        dataset_path could be None if your dataset is preprocessed and moved under
        <root-to-repo>/hstu/tmp_data folder or you're running with BenchmarkDatasetArgs
        which is an in-memory random data generator.
    """

    dataset_name: str
    max_sequence_length: int
    dataset_path: Optional[str] = None
    max_num_candidates: int = 0
    shuffle: bool = False


@gin.configurable
@dataclass
class FeatureArgs:
    """Feature Configuration.

    Feature-specific configuration parameters.

    Attributes:
        feature_names (List[str]): **Required**. List of feature names.
        max_sequence_length (int): **Required**. Maximum sequence length.
        is_jagged (bool): Whether features are jagged (variable length). Default: False.

    Note:
        `FeatureArgs` are only used when the dataset is of `BenchmarkDatasetArgs` type.
    """

    feature_names: List[str]
    max_sequence_length: int
    is_jagged: bool = False


@gin.configurable
@dataclass
class BenchmarkDatasetArgs:
    """Benchmark Dataset Configuration.

    Configuration for benchmark datasets combining features and embeddings.

    Attributes:
        feature_args (List[FeatureArgs]): **Required**. List of feature arguments.
        embedding_args (List[Union[EmbeddingArgs, DynamicEmbeddingArgs]]): **Required**.
            List of embedding arguments.
        item_feature_name (str): **Required**. Item feature name.
        contextual_feature_names (List[str]): **Required**. List of contextual feature names.
        action_feature_name (Optional[str]): Action feature name. Default: None.
        max_num_candidates (int): Maximum number of candidates. Default: 0.
    """

    feature_args: List[FeatureArgs]
    embedding_args: List[Union[EmbeddingArgs, DynamicEmbeddingArgs]]
    item_feature_name: str
    contextual_feature_names: List[str]
    action_feature_name: Optional[str] = None
    max_num_candidates: int = 0


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
        kernel_backend (str): Kernel backend: "cutlass", "triton", or "pytorch".
            Default: "cutlass".
        target_group_size (int): Target group size. Default: 1.
        num_position_buckets (int): Number of position buckets. Default: 8192.
        recompute_input_layernorm (bool): Recompute input layer normalization. Default: False.
        recompute_input_silu (bool): Recompute input SiLU activation. Default: False.
        item_embedding_dim (int): Item embedding dimension. Default: -1.
        contextual_embedding_dim (int): Contextual embedding dimension. Default: -1.
    """

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    kv_channels: int

    hidden_dropout: float = 0.2
    norm_epsilon: float = 1e-5
    is_causal: bool = True

    dtype_str: str = "bfloat16"

    kernel_backend: str = "cutlass"
    target_group_size: int = 1

    num_position_buckets: int = 8192

    recompute_input_layernorm: bool = False
    recompute_input_silu: bool = False

    item_embedding_dim: int = -1
    contextual_embedding_dim: int = -1

    scaling_seqlen: int = -1

    def __post_init__(self):
        assert self.dtype_str in [
            "bfloat16",
            "float16",
        ], "Only support bfloat16 and float16 precision for Network."

        assert self.kernel_backend.lower() in ["cutlass", "triton", "pytorch"]


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
    """

    optimizer_str: str
    learning_rate: float
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


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


@gin.configurable
@dataclass
class RankingArgs:
    """Ranking Task Configuration.

    Configuration specific to ranking tasks.

    Attributes:
        prediction_head_arch (List[int]): **Required**. Prediction head architecture
            (list of layer sizes). Default: None.
        prediction_head_act_type (str): Prediction head activation type: "relu" or "gelu".
            Default: "relu".
        prediction_head_bias (bool): Whether to use bias in prediction head. Default: True.
        num_tasks (int): Number of tasks (for multi-task learning). Default: 1.
        eval_metrics (Tuple[str, ...]): Evaluation metrics tuple. Default: ("AUC",).
    """

    prediction_head_arch: List[int] = cast(List[int], None)
    prediction_head_act_type: str = "relu"
    prediction_head_bias: bool = True
    num_tasks: int = 1
    eval_metrics: Tuple[str, ...] = ("AUC",)

    def __post_init__(self):
        assert (
            self.prediction_head_arch is not None
        ), "Please provide prediction head arch for ranking model"
        if isinstance(self.prediction_head_act_type, str):
            assert self.prediction_head_act_type.lower() in [
                "relu",
                "gelu",
            ], "prediction_head_act_type should be in ['relu', 'gelu']"


@gin.configurable
@dataclass
class RetrievalArgs:
    """Retrieval Task Configuration.

    Configuration specific to retrieval tasks.

    Attributes:
        num_negatives (int): Number of negative samples. Default: -1.
        temperature (float): Temperature parameter for similarity scoring. Default: 0.05.
        l2_norm_eps (float): Epsilon value for L2 normalization. Default: 1e-6.
        eval_metrics (Tuple[str, ...]): Evaluation metrics tuple (Hit Rate, NDCG).
            Default: ("HR@10", "NDCG@10").
    """

    ### retrieval
    num_negatives: int = -1
    temperature = 0.05
    l2_norm_eps = 1e-6
    eval_metrics: Tuple[str, ...] = ("HR@10", "NDCG@10")
