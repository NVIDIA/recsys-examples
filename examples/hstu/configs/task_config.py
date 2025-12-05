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
from typing import List, Tuple, cast

from commons.modules.embedding import ShardedEmbeddingConfig


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
        prediction_head_arch (List[int]): Architecture of the prediction head.
        prediction_head_act_type (str): Activation function type for the prediction head layers. Must be one of: ``'relu'`` | ``'gelu'``. Defaults to ``'relu'``.
        prediction_head_bias (bool): Whether to use bias terms in the prediction head layers. Defaults to ``True``.
        num_tasks (int): Number of tasks. Defaults to ``1``.
        eval_metrics (Tuple[str], optional): Tuple of evaluation metric type str during training. Refer to :obj:`~modules.metrics.metric_modules.MetricType`
          for available metrics. Defaults to ``'AUC'``.
    """

    prediction_head_arch: List[int] = cast(List[int], None)
    prediction_head_act_type: str = "relu"
    prediction_head_bias: bool = True
    num_tasks: int = 1
    eval_metrics: Tuple[str, ...] = ("AUC",)

    def __post_init__(self):
        assert (
            self.prediction_head_arch is not None
        ), "Please provide prediction head arch"
        assert isinstance(
            self.prediction_head_arch, list
        ), "prediction_head_arch should be a list"
        assert isinstance(
            self.prediction_head_act_type, str
        ), "prediction_head_act_type should be a str"
        assert isinstance(
            self.prediction_head_bias, bool
        ), "prediction_head_bias should be a bool"


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
