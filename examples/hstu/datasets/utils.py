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
from typing import Dict, List, Optional, Union

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable


@dataclass
class FeatureConfig:
    """
    Configuration for features in a dataset. A FeatureConfig is a collection of features that share the same seqlen (also the same max_seqlence_length).
    For example, in HSTU based models, an item is always associated with a timestamp token.

    Attributes:
      feature_names (List[str]): List of names for the features.
      max_item_ids (List[int]): List of maximum item IDs for each feature.
      max_sequence_length (int): The maximum length of sequences in the dataset.
      is_jagged (bool): Whether the sequences are jagged (i.e., have varying lengths).
    """

    feature_names: List[str]
    max_item_ids: List[int]
    max_sequence_length: int
    is_jagged: bool


@dataclass
class Batch:
    """
    A class used to represent a Batch of data for a distributed recommender system.

    Attributes:
      features (KeyedJaggedTensor): A tensor containing the features for the batch, where each key corresponds to a specific feature.
      feature_to_max_seqlen (Dict[str, int]): A dictionary mapping each feature to its maximum sequence length.
      batch_size (int): The number of elements in the batch.
      contextual_feature_names (List[str]): List of names for the contextual features.
      item_feature_name (str): The name of the item feature.
      action_feature_name (Optional[str]): The name of the action feature, if applicable.
      max_num_candidates (int): The maximum number of candidate items.
      num_candidates (Optional[torch.Tensor]): A tensor containing the number of candidates for each batch element.
    """

    features: KeyedJaggedTensor
    feature_to_max_seqlen: Dict[str, int]

    batch_size: int

    contextual_feature_names: List[str]
    item_feature_name: str
    action_feature_name: Optional[str]

    max_num_candidates: int
    num_candidates: Optional[torch.Tensor]

    def __post_init__(self):
        if len(set(self.features.keys())) != len(list(self.features.keys())):
            raise ValueError(f"duplicate features keys {list(self.features.keys())}")
        assert isinstance(self.contextual_feature_names, list)
        assert isinstance(self.item_feature_name, str)
        assert self.action_feature_name is None or isinstance(
            self.action_feature_name, str
        )
        assert isinstance(self.max_num_candidates, int)

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":  # type: ignore
        """
        Move the batch to the specified device.

        Args:
            device (torch.device): The device to move the batch to.
            non_blocking (bool, optional): Whether to perform the move asynchronously. Defaults to False.

        Returns:
            RankingBatch: The batch on the specified device.
        """
        return Batch(
            features=self.features.to(device=device, non_blocking=non_blocking),
            batch_size=self.batch_size,
            feature_to_max_seqlen=self.feature_to_max_seqlen,
            contextual_feature_names=self.contextual_feature_names,
            item_feature_name=self.item_feature_name,
            action_feature_name=self.action_feature_name,
            max_num_candidates=self.max_num_candidates,
            num_candidates=self.num_candidates.to(
                device=device, non_blocking=non_blocking
            )
            if self.num_candidates is not None
            else None,
        )

    @staticmethod
    def random(
        batch_size: int,
        feature_configs: List[FeatureConfig],
        item_feature_name: str,
        contextual_feature_names: List[str] = [],
        action_feature_name: Optional[str] = None,
        max_num_candidates: int = 0,
        *,
        device: torch.device,
    ) -> "Batch":
        """
        Generate a random Batch.

        Args:
            batch_size (int): The number of elements in the batch.
            feature_configs (List[FeatureConfig]): List of configurations for each feature.
            item_feature_name (str): The name of the item feature.
            contextual_feature_names (List[str], optional): List of names for the contextual features. Defaults to [].
            action_feature_name (Optional[str], optional): The name of the action feature. Defaults to None.
            max_num_candidates (int, optional): The maximum number of candidate items. Defaults to 0.
            device (torch.device): The device on which the batch will be generated.

        Returns:
            Batch: The generated random Batch.
        """
        keys = []
        values = []
        lengths = []
        num_candidates = None
        feature_to_max_seqlen = {}
        for fc in feature_configs:
            if fc.is_jagged:
                seqlen = torch.randint(
                    fc.max_sequence_length, (batch_size,), device=device
                )
            else:
                seqlen = torch.full(
                    (batch_size,), fc.max_sequence_length, device=device
                )
            cur_seqlen_sum = torch.sum(seqlen).item()

            for feature_name, max_item_id in zip(fc.feature_names, fc.max_item_ids):
                value = torch.randint(max_item_id, (cur_seqlen_sum,), device=device)
                keys.append(feature_name)
                values.append(value)
                lengths.append(seqlen)
                if max_num_candidates > 0 and feature_name == item_feature_name:
                    non_candidates_seqlen = torch.clamp(
                        seqlen - max_num_candidates, min=0
                    )
                    num_candidates = seqlen - non_candidates_seqlen
                feature_to_max_seqlen[feature_name] = fc.max_sequence_length
        return Batch(
            features=KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.concat(values).to(device),
                lengths=torch.concat(lengths).to(device).long(),
            ),
            batch_size=batch_size,
            feature_to_max_seqlen=feature_to_max_seqlen,
            contextual_feature_names=contextual_feature_names,
            item_feature_name=item_feature_name,
            action_feature_name=action_feature_name,
            max_num_candidates=max_num_candidates,
            num_candidates=num_candidates.to(device)
            if num_candidates is not None
            else None,
        )


@dataclass
class RankingBatch(Batch, Pipelineable):
    labels: torch.Tensor  # shape: (T, num_tasks)

    def to(self, device: torch.device, non_blocking: bool = False) -> "RankingBatch":  # type: ignore
        """
        Move the batch to the specified device.

        Args:
            device (torch.device): The device to move the batch to.
            non_blocking (bool, optional): Whether to perform the move asynchronously. Defaults to False.

        Returns:
            RankingBatch: The batch on the specified device.
        """
        return RankingBatch(
            features=self.features.to(device=device, non_blocking=non_blocking),
            batch_size=self.batch_size,
            feature_to_max_seqlen=self.feature_to_max_seqlen,
            contextual_feature_names=self.contextual_feature_names,
            item_feature_name=self.item_feature_name,
            action_feature_name=self.action_feature_name,
            max_num_candidates=self.max_num_candidates,
            num_candidates=self.num_candidates.to(
                device=device, non_blocking=non_blocking
            )
            if self.num_candidates is not None
            else None,
            labels=self.labels.to(device=device, non_blocking=non_blocking),
        )

    def record_stream(self, stream: torch.cuda.Stream):
        self.features.record_stream(stream)
        if self.num_candidates is not None:
            self.num_candidates.record_stream(stream)
        self.labels.record_stream(stream)

    def pin_memory(self) -> "RankingBatch":
        """
        Pin the memory of the batch.

        Returns:
            RankingBatch: The batch with pinned memory.
        """
        return RankingBatch(
            features=self.features.pin_memory(),
            batch_size=self.batch_size,
            feature_to_max_seqlen=self.feature_to_max_seqlen,
            contextual_feature_names=self.contextual_feature_names,
            item_feature_name=self.item_feature_name,
            action_feature_name=self.action_feature_name,
            max_num_candidates=self.max_num_candidates,
            num_candidates=self.num_candidates.pin_memory()
            if self.num_candidates is not None
            else None,
            labels=self.labels.pin_memory(),
        )

    @staticmethod
    def random(
        batch_size: int,
        feature_configs: List[FeatureConfig],
        item_feature_name: str,
        contextual_feature_names: List[str] = [],
        action_feature_name: Optional[str] = None,
        max_num_candidates: int = 0,
        num_tasks: int = 1,
        *,
        device: torch.device,
    ) -> "RankingBatch":
        """
        Generate a random RankingBatch.

        Args:
            batch_size (int): The number of elements in the batch.
            feature_configs (List[FeatureConfig]): List of configurations for each feature.
            item_feature_name (str): The name of the item feature.
            contextual_feature_names (List[str], optional): List of names for the contextual features. Defaults to [].
            action_feature_name (Optional[str], optional): The name of the action feature. Defaults to None.
            max_num_candidates (int, optional): The maximum number of candidate items. Defaults to 0.
            num_tasks (int): The number of tasks. Defaults to 1.
            device (torch.device): The device on which the batch will be generated.

        Returns:
            RankingBatch: The generated random RankingBatch.
        """
        assert num_tasks is not None, "num_tasks is required for RankingBatch"
        batch = Batch.random(
            batch_size=batch_size,
            feature_configs=feature_configs,
            item_feature_name=item_feature_name,
            contextual_feature_names=contextual_feature_names,
            action_feature_name=action_feature_name,
            max_num_candidates=max_num_candidates,
            device=device,
        )
        if batch.num_candidates is not None:
            total_num_labels = torch.sum(batch.num_candidates)
        else:
            total_num_labels = torch.sum(batch.features[item_feature_name].lengths())
        labels = torch.randint(1 << num_tasks, (total_num_labels,), device=device)
        return RankingBatch(
            features=batch.features,
            batch_size=batch.batch_size,
            feature_to_max_seqlen=batch.feature_to_max_seqlen,
            contextual_feature_names=batch.contextual_feature_names,
            item_feature_name=batch.item_feature_name,
            action_feature_name=batch.action_feature_name,
            max_num_candidates=batch.max_num_candidates,
            num_candidates=batch.num_candidates,
            labels=labels,
        )


@dataclass
class RetrievalBatch(Batch, Pipelineable):
    def to(self, device: torch.device, non_blocking: bool = False) -> "RetrievalBatch":  # type: ignore
        """
        Move the batch to the specified device.

        Args:
            device (torch.device): The device to move the batch to.
            non_blocking (bool, optional): Whether to perform the move asynchronously. Defaults to False.

        Returns:
            RetrievalBatch: The batch on the specified device.
        """
        return RetrievalBatch(
            features=self.features.to(device=device, non_blocking=non_blocking),
            batch_size=self.batch_size,
            feature_to_max_seqlen=self.feature_to_max_seqlen,
            contextual_feature_names=self.contextual_feature_names,
            item_feature_name=self.item_feature_name,
            action_feature_name=self.action_feature_name,
            max_num_candidates=self.max_num_candidates,
            num_candidates=self.num_candidates.to(
                device=device, non_blocking=non_blocking
            )
            if self.num_candidates is not None
            else None,
        )

    def record_stream(self, stream: torch.cuda.Stream):
        self.features.record_stream(stream)
        if self.num_candidates is not None:
            self.num_candidates.record_stream(stream)

    def pin_memory(self) -> "RetrievalBatch":
        """
        Pin the memory of the batch.

        Returns:
            RetrievalBatch: The batch with pinned memory.
        """
        return RetrievalBatch(
            features=self.features.pin_memory(),
            batch_size=self.batch_size,
            feature_to_max_seqlen=self.feature_to_max_seqlen,
            contextual_feature_names=self.contextual_feature_names,
            item_feature_name=self.item_feature_name,
            action_feature_name=self.action_feature_name,
            max_num_candidates=self.max_num_candidates,
            num_candidates=self.num_candidates.pin_memory()
            if self.num_candidates is not None
            else None,
        )

    @staticmethod
    def random(
        batch_size: int,
        feature_configs: List[FeatureConfig],
        item_feature_name: str,
        contextual_feature_names: List[str] = [],
        action_feature_name: Optional[str] = None,
        max_num_candidates: int = 0,
        *,
        device: torch.device,
    ) -> "RetrievalBatch":
        """
        Generate a random RetrievalBatch.

        Args:
            batch_size (int): The number of elements in the batch.
            feature_configs (List[FeatureConfig]): List of configurations for each feature.
            item_feature_name (str): The name of the item feature.
            contextual_feature_names (List[str], optional): List of names for the contextual features. Defaults to [].
            action_feature_name (Optional[str], optional): The name of the action feature. Defaults to None.
            max_num_candidates (int, optional): The maximum number of candidate items. Defaults to 0.
            device (torch.device): The device on which the batch will be generated.

        Returns:
            RetrievalBatch: The generated random RetrievalBatch.
        """
        batch = Batch.random(
            batch_size=batch_size,
            feature_configs=feature_configs,
            item_feature_name=item_feature_name,
            contextual_feature_names=contextual_feature_names,
            action_feature_name=action_feature_name,
            max_num_candidates=max_num_candidates,
            device=device,
        )
        return RetrievalBatch(
            features=batch.features,
            batch_size=batch.batch_size,
            feature_to_max_seqlen=batch.feature_to_max_seqlen,
            contextual_feature_names=batch.contextual_feature_names,
            item_feature_name=batch.item_feature_name,
            action_feature_name=batch.action_feature_name,
            max_num_candidates=batch.max_num_candidates,
            num_candidates=batch.num_candidates,
        )


def is_batch_valid(
    batch: Union[RankingBatch, RetrievalBatch],
):
    """
    Validates a batch of data to ensure it meets the necessary criteria.

    Args:
        batch (Union[RankingBatch, RetrievalBatch]): The batch to validate.

    Raises:
        AssertionError: If any of the validation checks fail.
    """
    assert isinstance(batch, RankingBatch) or isinstance(
        batch, RetrievalBatch
    ), "batch type should be RankingBatch or RetrievalBatch"

    assert (
        batch.item_feature_name in batch.features.keys()
    ), "batch must have item_feature_name in features"
    assert (
        batch.features[batch.item_feature_name].lengths().numel() == batch.batch_size
    ), "item_feature shape is not equal to batch_size"
    if batch.action_feature_name is not None:
        assert (
            batch.action_feature_name in batch.features.keys()
        ), "action_feature_name is configured, but not in features"
        assert (
            batch.features[batch.action_feature_name].lengths().numel()
            == batch.batch_size
        ), "action_feature shape is not equal to batch_size"
        assert torch.allclose(
            batch.features[batch.item_feature_name].lengths(),
            batch.features[batch.action_feature_name].lengths(),
        ), "item_feature and action_feature shape should equal"
    if batch.num_candidates is not None:
        assert (
            batch.max_num_candidates > 0
        ), "max_num_candidates should > 0 when num_candidates configured"
        assert torch.all(
            batch.features[batch.item_feature_name].lengths() - batch.num_candidates
            >= 0
        ), "num_candidates is larger than item_feature seqlen"
        expected_label_size = torch.sum(batch.num_candidates).cpu().item()
    else:
        expected_label_size = (
            torch.sum(batch.features[batch.item_feature_name].lengths()).cpu().item()
        )

    if isinstance(batch, RankingBatch):
        assert (
            batch.labels.dim() == 1
        ), f"label dim() should equal to 1 but got {batch.labels.dim()}"
        assert (
            batch.labels.size(0) == expected_label_size
        ), f"label seqlen sum should be {expected_label_size}, but got {batch.labels.size(0)}"
    for contextual_feature_name in batch.contextual_feature_names:
        assert (
            contextual_feature_name in batch.features.keys()
        ), f"contextual_feature {contextual_feature_name} is configured, but not in features"
        assert (
            batch.features[contextual_feature_name].lengths().numel()
            == batch.batch_size
        ), f"contextual_feature {contextual_feature_name} shape is not equal to batch_size"
