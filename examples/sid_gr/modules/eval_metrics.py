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
from typing import Any, Dict, Tuple

import torch
import torchmetrics


class BaseMeanDistributedReductionMetric(torchmetrics.Metric, ABC):
    """
    Computes a metric using mean reduction (average across queries) aggregated across distributed workers.
    Note that we suppose the parallelism is along the query dimension, that is, each worker only processes a subset of queries.
    We allow batch_size per worker to be different.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.metric_values: float = 0.0
        self.num_queries: int = 0

    @abstractmethod
    def update(
        self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor, **kwargs
    ) -> None:
        """
        To be implemented by child classes: update stats from current batch.
        """

    def compute(self) -> torch.Tensor:
        """
        Returns: torch.Tensor, the aggregated metric averaged over all queries in all distributed workers.
        """
        metric_values_tensor = torch.tensor(
            self.metric_values, device=self.device, dtype=torch.float32
        )
        num_queries_tensor = torch.tensor(
            self.num_queries, device=self.device, dtype=torch.int64
        )
        if self.num_queries == 0:
            return torch.tensor(0.0, device=self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                metric_values_tensor, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                num_queries_tensor, op=torch.distributed.ReduceOp.SUM
            )
        return metric_values_tensor.sum() / num_queries_tensor.sum()

    def reset(self) -> None:
        self.metric_values = 0.0
        self.num_queries = 0


class DistributedRetrievalMetric(BaseMeanDistributedReductionMetric, ABC):
    """
    Base for distributed retrieval ranking metrics (@K metrics), per query.
    """

    def __init__(self, top_k: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor, **kwargs
    ) -> None:
        """
        Args:
            preds, target: [batch_size * num_candidates], all flattened.
            indexes: [batch_size * num_candidates], each element is the index of the query (0 ... batch_size-1).
        """
        # Determine batch and candidate shapes by how index is structured.
        # Each group of (num_candidate_per_query) consecutive elements is a query.
        # We assume indexes is constructed in row-major order.
        # If all queries are contiguous, then the number of zeros in indexes is the batch_size.
        num_candidate_per_query = (indexes == 0).sum().item()
        batch_size = indexes.numel() // num_candidate_per_query

        preds = preds.view(batch_size, num_candidate_per_query)
        target = target.view(batch_size, num_candidate_per_query).int()

        metric_result = self._metric_impl(preds, target)
        # sum over batch dimension
        self.metric_values += metric_result.sum().item()
        self.num_queries += batch_size

    @abstractmethod
    def _metric_impl(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Returns tensor of shape [batch_size]
        """


class DistributedRetrievalNDCG(DistributedRetrievalMetric):
    """
    Normalized Discounted Cumulative Gain@K metric for retrieval (per query/batch).
    """

    def _metric_impl(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # topk predicted indices by score
        topk_indices = torch.topk(preds, self.top_k, dim=1).indices
        topk_true = target.gather(1, topk_indices)

        # DCG
        denom = torch.log2(
            torch.arange(2, self.top_k + 2, device=target.device).float()
        ).unsqueeze(0)
        dcg = (topk_true / denom).sum(dim=1)

        # Ideal DCG (use ideal ranking)
        ideal_indices = torch.topk(target, self.top_k, dim=1).indices
        ideal_dcg = (target.gather(1, ideal_indices) / denom).sum(dim=1)

        # Avoid div by zero
        ndcg = dcg / torch.where(ideal_dcg == 0, torch.ones_like(ideal_dcg), ideal_dcg)

        return ndcg


class DistributedRetrievalRecall(DistributedRetrievalMetric):
    """
    Recall@K metric for retrieval (per query/batch).
    """

    def _metric_impl(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ """
        # preds (sorted by score) [1, 2', 3, 4, 5, 6', 7', 8']
        # target (ground truth) [False, True, False, False, False, True, True, True]

        # top2 preds  [1, 2']
        # top2 target [False, True]
        # recall: |2'| / |1, 2'| = 1 / 2 = 0.5
        # How many relevant items are in the topk predicted items?
        topk_indices = torch.topk(preds, self.top_k, dim=1).indices
        topk_target = target.gather(1, topk_indices)  # [batch, top_k]
        num_hit_in_topk = topk_target.sum(dim=1)  # [batch], total recalled samples
        # for sid, total_relevant <= 1. Because the labels for each query contain single item.
        total_relevant = target.sum(dim=1)
        # denorm is different from standard torchmetrics. We use the min
        denom = total_relevant.minimum(
            torch.tensor(self.top_k, device=target.device)
        ).clamp(min=1)
        recall = num_hit_in_topk / denom
        return recall


class DistributedRetrievalHitRate(DistributedRetrievalMetric):
    """
    Recall@K metric for retrieval (per query/batch).
    """

    def _metric_impl(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        preds: [batchsize, num_candidates] ( for sid, the num_candidates is the beam_width)
        target: [batchsize, num_candidates] ( for sid, the target is the ground truth)
        """

        # 1. get the topk result
        topk_indices = torch.topk(preds, self.top_k, dim=1).indices
        topk_target = target.gather(1, topk_indices)  # [batch, top_k]

        # 2. check if topk results hit the ground truth
        hit = topk_target.any(dim=1)  # [batch]
        return hit


_metric_str_to_object = {
    "ndcg": DistributedRetrievalNDCG,
    "recall": DistributedRetrievalRecall,
    "hitrate": DistributedRetrievalHitRate,
}


class SIDRetrievalEvaluator(torch.nn.Module):
    """
    Helper for evaluating retrieval metrics for semantic ID tasks.
    """

    def __init__(self, eval_metrics: Tuple[str, ...], sid_prefix_length: int = -1):
        super().__init__()
        self.metrics = torch.nn.ModuleDict()
        for metric_spec in eval_metrics:
            metric_name, top_k = metric_spec.split("@")
            metric_class = _metric_str_to_object[metric_name.lower()]
            self.metrics[metric_spec] = metric_class(
                top_k=int(top_k), sync_on_compute=False, compute_with_cache=False
            )
        self.sid_prefix_length = sid_prefix_length

    def state_dict(self):
        # Metrics not checkpointed.
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass

    def forward(
        self,
        log_probs: torch.Tensor,
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ):
        """
        Args:
            log_probs: [batch, num_candidates]
            generated_ids: [batch, num_candidates, num_hierarchies]
            labels: [batch, num_hierarchies]
        """
        batch_size, num_candidates, num_hierarchies = generated_ids.shape
        # Reshape for matching
        labels = labels.view(batch_size, 1, num_hierarchies)
        generated_ids = generated_ids[:, :, : self.sid_prefix_length]
        labels = labels[:, :, : self.sid_prefix_length]
        preds = log_probs.reshape(-1)
        # Match each candidate's IDs to groundtruth: [batch, num_candidates]
        matched_id_coord = torch.all(generated_ids == labels, dim=2).nonzero(
            as_tuple=True
        )
        target = torch.zeros(
            batch_size, num_candidates, dtype=torch.bool, device=generated_ids.device
        )

        target[matched_id_coord] = True
        target = target.view(-1)
        # indexes is not used. Assign a dummy value.
        expanded_indexes = (
            torch.arange(batch_size, device=log_probs.device)
            .unsqueeze(-1)
            .expand(batch_size, num_candidates)
            .reshape(-1)
        )

        for metric_obj in self.metrics.values():
            metric_obj.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes.to(preds.device),
            )

    def compute(self):
        return {
            metric_name: metric.compute()
            for metric_name, metric in self.metrics.items()
        }

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()


class MultipleEvaluatorWrapper(torch.nn.Module):
    """
    Wrapper for multiple evaluators.
    """

    def __init__(self, evaluators: Dict[str, SIDRetrievalEvaluator]):
        super().__init__()
        self.evaluators = torch.nn.ModuleDict(evaluators)

    def forward(
        self, log_probs: torch.Tensor, generated_ids: torch.Tensor, labels: torch.Tensor
    ):
        for evaluator in self.evaluators.values():
            evaluator(log_probs, generated_ids, labels)

    def compute(self):
        ret = {}
        for evaluator_name, evaluator in self.evaluators.items():
            ret[evaluator_name] = evaluator.compute()
        return ret

    def reset(self):
        for evaluator in self.evaluators.values():
            evaluator.reset()
