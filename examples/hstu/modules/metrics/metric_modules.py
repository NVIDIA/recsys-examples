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
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from enum import Enum
from functools import partial

# pyre-strict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchmetrics.classification as classification_metrics
from commons.ops.collective_ops import grouped_allgatherv_tensor_list
from commons.utils.nvtx_op import output_nvtx_hook
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner as DynamicEmbeddingShardingPlanner,
)

try:
    from megatron.core import parallel_state
except ImportError:
    print("megatron.core is not installed, training is not supported.")

# from torchrec.distributed import ModuleShardingPlan
from tqdm import tqdm


class MetricType(Enum):
    """
    MetricType is an enumeration of various metrics used for evaluating ranking, classification, and retrieval tasks.

    Attributes:
      ACC: Accuracy metric for classification tasks.
      AUC: Area Under the Curve metric for classification tasks.
      RECALL: Recall metric for classification tasks.
      PRECISION: Precision metric for classification tasks, averaged as macro.
      F1Score: F1 Score metric for classification tasks.
      AP: Average Precision metric for classification tasks.
      MRR: Mean Reciprocal Rank metric for retrieval tasks.
      HR: Hit Rate metric for retrieval tasks.
      NDCG: Normalized Discounted Cumulative Gain metric for retrieval tasks.
    """

    # ranking/classification
    ACC = "ACC"
    AUC = "AUC"
    RECALL = "RECALL"
    PRECISION = "PRECISION"  # average = macro
    F1Score = "F1Score"
    AP = "AVERAGE PRECISION"

    # retrieval
    MRR = "MRR"
    HR = "HR"
    NDCG = "NDCG"
    # RetrievalAUROC = "RetrievalAUROC"


def _get_ndcg(eval_ranks, topk):
    ndcg = torch.where(
        eval_ranks <= topk,
        1.0 / torch.log2(eval_ranks + 1),
        torch.zeros(1, dtype=torch.float32, device=eval_ranks.device),
    )
    return ndcg


def _get_hr(eval_ranks, topk):
    return eval_ranks < topk


def _get_mrr(eval_ranks, topk):  # topk is not used
    return 1.0 / eval_ranks


_metric_type_to_cls_map = {
    # Ranking
    MetricType.ACC: classification_metrics.Accuracy,
    MetricType.AUC: classification_metrics.AUROC,
    MetricType.RECALL: classification_metrics.Recall,
    MetricType.PRECISION: classification_metrics.Precision,
    MetricType.F1Score: classification_metrics.F1Score,
    MetricType.AP: classification_metrics.AveragePrecision,
    # Retrieval
    MetricType.MRR: _get_mrr,
    MetricType.HR: _get_hr,
    MetricType.NDCG: _get_ndcg,
}


class BaseTaskMetric(ABC, torch.nn.Module):
    """
    Abstract base class for task metrics.
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class MultiClassificationTaskMetric(BaseTaskMetric):
    """
    This module is intended for evaluation, and the forward will return nothing.
    One forward step corresponds to one batch, where internally results get cached
    by torchmetrics objects.

    Use compute() to get the final metric scores after all eval batches get forward done.

    Requires a process group to perform the sync across DP.

    Args:
        number_of_tasks (int): Number of tasks.
        logit_dim_per_event (List[int]): List of logit dimensions per event.
        metric_type (Union[str, MetricType]): Type of metric to use : ``'ACC'``, ``'AUC'``, ``'RECALL'``, ``'PRECISION'``, ``'F1Score'``, ``'AP'``.
        process_group (torch.distributed.ProcessGroup, optional): Process group for synchronization.
        task_names (Optional[List[str]], optional): List of task names.
    example:
        >>> metric = MultiClassificationTaskMetric(
        ...     number_of_tasks=2,
        ...     logit_dim_per_event=[2, 3],
        ...     metric_type="AUC",
        ...     process_group=None,
        ...     task_names=["task1", "task2"],
        ... )
        >>> multi_task_logits = torch.randn(10, 5)
        >>> targets = torch.randint(0, 2, (10, 2))
        >>> metric(multi_task_logits, targets)
        >>> metric.compute()
        {''task1.AUC': 0.5, 'task2.AUC': 0.6}
    """

    def __init__(
        self,
        num_classes: int,
        number_of_tasks: int,
        metric_types: Tuple[str, ...] = ("AUC",),
        process_group: torch.distributed.ProcessGroup = None,
        task_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self._num_classes = num_classes
        self._number_of_tasks = number_of_tasks

        assert len(metric_types) == 1, "Only one ranking metric type is supported now"
        global _metric_type_to_cls_map
        mtype = (
            metric_types[0]
            if isinstance(metric_types[0], MetricType)
            else MetricType(metric_types[0])
        )
        self._metric_type = mtype
        metric_factory = _metric_type_to_cls_map[mtype]

        if task_names is None:
            task_names = [f"task{i}" for i in range(number_of_tasks)]
        else:
            assert (
                len(task_names) == number_of_tasks
            ), " please specify task names of same size as number of tasks"
        self._task_names = task_names
        self._eval_metrics_modules: torch.nn.ModuleList = torch.nn.ModuleList()
        self._logit_preprocessors = []
        if num_classes == number_of_tasks:
            for _ in range(number_of_tasks):
                module = metric_factory(task="binary", process_group=process_group)
                self._eval_metrics_modules.append(module)

                self._logit_preprocessors.append(torch.nn.functional.sigmoid)
        else:
            assert (
                number_of_tasks == 1
            ), "num_tasks should be 1 for multi-class classification"
            self._eval_metrics_modules.append(
                metric_factory(
                    task="multiclass",
                    num_classes=num_classes,
                    process_group=process_group,
                )
            )
            self._logit_preprocessors.append(
                partial(torch.nn.functional.softmax, dim=-1)
            )
        self.training = False

    # return a
    @output_nvtx_hook("ranking metrics", backward=False)
    def forward(self, multi_task_logits, targets):
        """
        Forward one eval batch, this forward returns None object.

        Args:
            multi_task_logits (torch.Tensor): Multi-task logits of shape [T, sum(logit_dim)].
            targets (torch.Tensor): Targets of shape [T] (E is the number of tasks).

        Returns:
            None
        """
        if self._num_classes == self._number_of_tasks:
            for task_id, task_name in enumerate(self._task_names):
                logit = multi_task_logits[..., task_id]
                target = (torch.bitwise_and(targets, 1 << task_id) > 0).long()
                pred = self._logit_preprocessors[task_id](logit)
                # target must be long
                # Metric forward returns the metric of current batch
                if pred.numel() > 0:
                    _ = self._eval_metrics_modules[task_id](pred, target)
        else:
            pred = self._logit_preprocessors[0](multi_task_logits)
            if pred.numel() > 0:
                _ = self._eval_metrics_modules[0](pred, targets)
        return None

    def compute(self):
        """
        Compute the final metric scores after all eval batches are done.

        Returns:
            OrderedDict: Dictionary containing the final metric scores.
        """
        ret_dict = OrderedDict()
        for task_id, eval_module in enumerate(self._eval_metrics_modules):
            ret_dict[
                self._task_names[task_id] + "." + self._metric_type.value
            ] = eval_module.compute()
        for eval_module in self._eval_metrics_modules:
            eval_module.reset()
        return ret_dict


# TODO, use torchmetrics instead
class RetrievalTaskMetricWithSampling(BaseTaskMetric):
    """
    This module is intended for retrieval task evaluation with sampling.

    Args:
        metric_types (Tuple[str]): Tuple of metric types. A eval metric type str is composed of <MetricTypeStr>+'@'+<k> where k is designated as the top-k retrieval. Available values for <MetricTypeStr> : ``'NDCG'`` | ``'HR'`` | ``'MRR'``.
        MAX_K (int): Maximum value of K for top-K calculations.

    Example:
        >>> metric = RetrievalTaskMetricWithSampling(metric_types=("NDCG@10", "HR@10", "MRR"), MAX_K=2500)
        >>> query_embeddings = torch.randn(10, 128)
        >>> target_ids = torch.randint(0, 100, (10,))
        >>> metric(query_embeddings, target_ids)
        >>> metric.compute()
        {'NDCG@10': 0.5, 'HR@10': 0.6, 'MRR': 0.7}
    """

    def __init__(
        self,
        metric_types: Tuple[str, ...] = ("NDCG@10",),
        MAX_K: int = 2500,
    ):
        super().__init__()
        self._max_k = MAX_K
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._mtype_topk = []
        for metric_type in metric_types:
            tmp_str = metric_type.split("@")
            mtype, topk = MetricType(tmp_str[0]), 10 if len(tmp_str) < 2 else int(
                tmp_str[1]
            )
            self._mtype_topk.append((mtype, topk))
        self._cache_query_embeddings: List[torch.Tensor] = []
        self._cache_target_ids: List[torch.Tensor] = []
        self._chunk_size = 512

    @output_nvtx_hook("retrieval metrics", backward=False)
    def forward(
        self,
        query_embeddings: torch.Tensor,  # preds, dense embedding tensor
        target_ids: torch.Tensor,  # targets,  dense id tensor
    ):
        """
        Forward pass for the retrieval task metric.

        Args:
            query_embeddings (torch.Tensor): Query embeddings (predictions).
            target_ids (torch.Tensor): Target IDs (ground truth).
        """
        self._cache_query_embeddings.append(query_embeddings)
        self._cache_target_ids.append(target_ids)

    def compute(
        self, keys_array: np.ndarray, values_array: np.ndarray
    ) -> Tuple[Dict[Any, Any], Any, Any]:
        """
        Compute the final retrieval metrics after all eval batches are done.

        Args:
            item_embedding (ShardedEmbedding): Sharded embedding for items.
            table_name (str): The item embedding table name

        Returns:
            Tuple[dict, torch.Tensor, torch.Tensor]: Dictionary containing the final metric scores,
                                                     global top-K logits, and global top-K keys.
        """
        # 1. export local embedding
        local_shard_rows = keys_array.size
        if local_shard_rows == 0:
            raise ValueError(
                f"No local shard on rank {torch.distributed.get_rank()}. Evaluation failed."
            )
        eval_dict_all: Dict[str, List] = defaultdict(list)
        for query_embeddings, target_ids in tqdm(
            zip(self._cache_query_embeddings, self._cache_target_ids)
        ):
            # 2. allgatherv query_embeddings and target_ids
            (
                global_query_embeddings,
                global_target_ids,
            ) = grouped_allgatherv_tensor_list(
                [query_embeddings, target_ids],
                pg=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )
            # 3. calc topk largest in a streaming way
            local_topk_logits: torch.Tensor = None
            local_topk_keys: torch.Tensor = None
            for start in range(0, local_shard_rows, self._chunk_size):
                stop = min(local_shard_rows, start + self._chunk_size)
                chunk_keys = torch.tensor(
                    keys_array[start:stop], device=self._device
                )  # (T)
                chunk_embedding = torch.tensor(
                    values_array[start:stop, :], device=self._device
                )
                logits = torch.mm(global_query_embeddings, chunk_embedding.T)  # (Q, T)
                if local_topk_logits is None:
                    local_topk_logits = logits
                    local_topk_keys = chunk_keys.unsqueeze(0).expand(
                        local_topk_logits.size(0), -1
                    )  # (Q, T)
                else:
                    local_topk_logits = torch.cat([local_topk_logits, logits], dim=1)
                    chunk_keys = chunk_keys.unsqueeze(0).expand(
                        local_topk_keys.size(0), -1
                    )
                    local_topk_keys = torch.cat([local_topk_keys, chunk_keys], dim=1)
                k = min(self._max_k, logits.size(1))
                local_topk_logits, local_topk_indices = torch.topk(
                    local_topk_logits,
                    k=k,
                    dim=1,
                    sorted=False,
                    largest=True,
                )
                local_topk_keys = torch.gather(
                    local_topk_keys, dim=1, index=local_topk_indices
                )

            # 4. allgatherv local_topk_* results
            t_local_topk_logits = local_topk_logits.T
            t_local_topk_keys = local_topk_keys.T
            (
                t_global_n_topk_logits,
                t_global_n_topk_keys,
            ) = grouped_allgatherv_tensor_list(
                [t_local_topk_logits, t_local_topk_keys],
                pg=torch.distributed.group.WORLD,
            )

            global_n_topk_logits = t_global_n_topk_logits.T
            global_n_topk_keys = t_global_n_topk_keys.T

            # 5. calc global topk
            k = min(self._max_k, global_n_topk_logits.size(1))
            global_topk_logits, global_topk_indices = torch.topk(
                global_n_topk_logits,
                k=k,
                dim=1,
                sorted=True,
                largest=True,
            )

            global_topk_keys = torch.gather(
                global_n_topk_keys, dim=1, index=global_topk_indices
            )
            _, eval_rank_indices = torch.max(
                torch.cat([global_topk_keys, global_target_ids.unsqueeze(-1)], dim=1)
                == global_target_ids.unsqueeze(-1),
                dim=1,
            )
            eval_ranks = torch.where(
                eval_rank_indices == k, self._max_k + 1, eval_rank_indices + 1
            )
            output_dict: Dict[str, torch.Tensor] = defaultdict()

            for mtype, topk in self._mtype_topk:
                if topk is not None:
                    output_dict[f"{mtype.value}@{topk}"] = _metric_type_to_cls_map[
                        mtype
                    ](eval_ranks, topk)
            for k, v in output_dict.items():
                eval_dict_all[k] = eval_dict_all[k] + [v]

        final_eval_dict: Dict[str, float] = dict()
        for k, v in eval_dict_all.items():
            res = torch.cat(v, dim=-1)
            final_eval_dict[k] = res.sum() / res.numel()

        self._cache_query_embeddings.clear()
        self._cache_target_ids.clear()
        return final_eval_dict, global_topk_logits, global_topk_keys
