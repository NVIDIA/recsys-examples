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
from typing import List, Tuple, Union, cast


@dataclass
class OptimizerParam:
    """
    Configuration for the embedding optimizer.

    Args:
        optimizer_str (str): The optimizer type as a string: ``'adam'`` | ``'sgd'``.
        learning_rate (float): The learning rate for the optimizer.
        adam_beta1 (float, optional): The beta1 parameter for the Adam optimizer. Defaults to 0.9.
        adam_beta2 (float, optional): The beta2 parameter for the Adam optimizer. Defaults to 0.95.
        adam_eps (float, optional): The epsilon parameter for the Adam optimizer. Defaults to 1e-08.
    """

    optimizer_str: str
    learning_rate: float
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-08


@dataclass
class ShardedEmbeddingConfig:
    """
    Configuration for sharded embeddings with sharding type. Inherits from BaseShardedEmbeddingConfig.

    Args:
        config (EmbeddingConfig): The embedding configuration.
        sharding_type (str): The type of sharding, ``'data_parallel'`` | ``'model_parallel'``.
    """

    """
    Base configuration for sharded embeddings.

    Args:
        feature_names (List[str]): The name of the features in this embedding.
        table_name (str): The name of the table.
        vocab_size (int): The size of the vocabulary.
        dim (int): The dimension size of the embeddings.
        sharding_type (str): The type of sharding, ``'data_parallel'`` | ``'model_parallel'``.
    """

    feature_names: List[str]
    table_name: str
    vocab_size: int
    dim: int
    sharding_type: str

    def __post_init__(self):
        assert self.sharding_type in [
            "data_parallel",
            "model_parallel",
        ], "sharding type should be data_parallel or model_parallel"


@dataclass
class BaseTaskConfig:
    """
    Base configuration for tasks.

    Args:
        embedding_configs (List[ShardedEmbeddingConfig]): A list of embedding configurations.
        user_embedding_norm (str, optional): Normalization for user embeddings. ``'layer_norm'`` | ``'l2_norm'``. Defaults to ``'l2_norm'``.
        item_l2_norm (bool, optional): Whether to apply L2 normalization to item embeddings. Defaults to False.
    """

    embedding_configs: List[ShardedEmbeddingConfig]

    user_embedding_norm: str = "l2_norm"
    item_l2_norm: bool = False

    def __post_init__(self):
        table_names = [emb_config.table_name for emb_config in self.embedding_configs]
        assert len(set(table_names)) == len(
            table_names
        ), f"duplicate table_names in embedding {table_names}"


@dataclass
class RankingConfig(BaseTaskConfig):
    """
    Configuration for ranking tasks.

    Args:
        prediction_head_arch (List[List[int]]): Architecture of the prediction head.
        prediction_head_act_type (Union[List[str], str]): Activation types for the prediction head. Defaults to ``'relu'``
        prediction_head_bias (Union[List[bool], bool]): Bias flags for the prediction head. Defaults to ``True``
        eval_metrics (Tuple[str], optional): Tuple of evaluation metric type str during training. Refer to :obj:`~modules.metrics.metric_modules.MetricType`
          for available metrics. Defaults to ``'AUC'``.
    """

    prediction_head_arch: List[List[int]] = cast(List[List[int]], None)
    prediction_head_act_type: Union[List[str], str] = "relu"
    prediction_head_bias: Union[List[bool], bool] = True
    # one head per event
    # [binary cross entropy or multicross]
    # number of tasks/events
    eval_metrics: Tuple[str, ...] = ("AUC",)

    def __post_init__(self):
        assert (
            self.prediction_head_arch is not None
        ), "Please provide prediction head arch"
        if isinstance(self.prediction_head_act_type, str):
            self.prediction_head_act_type = [
                self.prediction_head_act_type
                for _ in range(len(self.prediction_head_arch))
            ]
        if isinstance(self.prediction_head_bias, bool):
            self.prediction_head_bias = [
                self.prediction_head_bias for _ in range(len(self.prediction_head_arch))
            ]
        assert len(self.prediction_head_act_type) == len(
            self.prediction_head_arch
        ), "prediction head arch number should match with act_type"
        assert len(self.prediction_head_bias) == len(
            self.prediction_head_arch
        ), "prediction head arch number should match with bias number"


@dataclass
class RetrievalConfig(BaseTaskConfig):
    """
    Configuration for retrieval tasks.

    Args:
        temperature (float, optional): Temperature for softmax in loss computation. Defaults to 0.05.
        l2_norm_eps (float, optional): Epsilon for L2 normalization. Defaults to 1e-6.
        num_negatives (int, optional): Number of negative samples for loss computation. Defaults to -1.
        eval_metrics (Tuple[str, ...], optional): Tuple of evaluation metric type str during training. Refer to :obj:`~modules.metrics.metric_modules.MetricType`
          for available metrics. Defaults to ``'NDCG@10'``. Note that for retrieval tasks, a eval metric type str is composed of <MetricTypeStr>+'@'+<k> where k is designated as the top-k retrieval.
    """

    temperature: float = 0.05
    l2_norm_eps: float = 1e-6  # sampled item embedding l2norm eps

    num_negatives: int = -1  # number of negative samples used for loss computation
    eval_metrics: Tuple[str, ...] = ("NDCG@10",)
