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
import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch
from commons.sequence_batch.batch import BaseBatch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


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
class HSTUBatch(BaseBatch):
    """
    HSTU Batch class for ranking and retrieval tasks.

    Inherits from BaseBatch which provides:
      - features (KeyedJaggedTensor)
      - batch_size (int)
      - feature_to_max_seqlen (Dict[str, int])
      - contextual_feature_names (List[str])
      - labels (Optional[torch.Tensor])
      - to(), pin_memory(), record_stream() methods

    Additional HSTU-specific attributes:
      item_feature_name (str): The name of the item feature.
      action_feature_name (Optional[str]): The name of the action feature, if applicable.
      max_num_candidates (int): The maximum number of candidate items.
      num_candidates (Optional[torch.Tensor]): A tensor containing the number of candidates for each batch element.
    """

    # HSTU-specific fields (BaseBatch fields are inherited)
    item_feature_name: str = "item_id"
    action_feature_name: Optional[str] = None
    max_num_candidates: int = 0
    num_candidates: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Call parent __post_init__ first
        super().__post_init__()

        # HSTU-specific validations
        assert isinstance(
            self.item_feature_name, str
        ), "item_feature_name must be a string"
        assert self.action_feature_name is None or isinstance(
            self.action_feature_name, str
        ), "action_feature_name must be None or a string"
        assert isinstance(
            self.max_num_candidates, int
        ), "max_num_candidates must be an int"

    # to(), pin_memory(), record_stream() are inherited from BaseBatch
    @staticmethod
    def random(
        batch_size: int,
        feature_configs: List[FeatureConfig],
        item_feature_name: str,
        contextual_feature_names: List[str] = [],
        action_feature_name: Optional[str] = None,
        max_num_candidates: int = 0,
        num_tasks: Optional[int] = None,  # used for ranking task
        actual_batch_size: Optional[int] = None,  # for incomplete batch
        *,
        device: torch.device,
    ) -> "HSTUBatch":
        """
        Generate a random Batch.

        Args:
            batch_size (int): The target batch size (for padding).
            feature_configs (List[FeatureConfig]): List of configurations for each feature.
            item_feature_name (str): The name of the item feature.
            contextual_feature_names (List[str], optional): List of names for the contextual features. Defaults to [].
            action_feature_name (Optional[str], optional): The name of the action feature. Defaults to None.
            max_num_candidates (int, optional): The maximum number of candidate items. Defaults to 0.
            num_tasks (Optional[int], optional): Number of tasks for ranking. Defaults to None.
            actual_batch_size (Optional[int], optional): Actual number of samples (< batch_size for incomplete batch).
                If None, equals to batch_size. Defaults to None.
            device (torch.device): The device on which the batch will be generated.

        Returns:
            HSTUBatch: The generated random Batch.
        """
        # Use actual_batch_size for data generation, batch_size for padding
        if actual_batch_size is None:
            actual_batch_size = batch_size

        assert (
            actual_batch_size <= batch_size
        ), f"actual_batch_size ({actual_batch_size}) must be <= batch_size ({batch_size})"

        keys = []
        values = []
        lengths = []
        num_candidates = None
        feature_to_max_seqlen = {}
        labels_numel = 0
        history_seqlen = 0

        for fc in feature_configs:
            # Generate data for actual_batch_size samples
            if fc.is_jagged:
                seqlen = torch.randint(
                    fc.max_sequence_length, (actual_batch_size,), device=device
                )
            else:
                seqlen = torch.full(
                    (actual_batch_size,), fc.max_sequence_length, device=device
                )

            if actual_batch_size < batch_size:
                padded_size = batch_size - actual_batch_size
                seqlen = torch.cat(
                    [
                        seqlen,
                        torch.zeros(padded_size, dtype=seqlen.dtype, device=device),
                    ]
                )

            cur_seqlen_sum = torch.sum(seqlen).item()

            for feature_name, max_item_id in zip(fc.feature_names, fc.max_item_ids):
                if feature_name in contextual_feature_names and fc.is_jagged:
                    warnings.warn(f"contextual feature {feature_name} is jagged")
                value = torch.randint(max_item_id, (cur_seqlen_sum,), device=device)
                keys.append(feature_name)
                values.append(value)
                lengths.append(seqlen)
                if feature_name == item_feature_name:
                    labels_numel = cur_seqlen_sum
                    history_seqlen = seqlen
                if max_num_candidates > 0 and feature_name == item_feature_name:
                    non_candidates_seqlen = torch.clamp(
                        seqlen - max_num_candidates, min=0
                    )
                    num_candidates = seqlen - non_candidates_seqlen
                    labels_numel = num_candidates.sum()
                feature_to_max_seqlen[feature_name] = fc.max_sequence_length

        if num_tasks is not None:
            label_values = torch.randint(1 << num_tasks, (labels_numel,), device=device)
            # when no candidates, we use the history seqlen as the label length.
            label_lengths = history_seqlen if num_candidates is None else num_candidates
            labels = KeyedJaggedTensor.from_lengths_sync(
                keys=["label"],
                values=label_values,
                lengths=label_lengths,
            )
        else:
            labels = None
        return HSTUBatch(
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
            labels=labels,
            actual_batch_size=actual_batch_size,
        )


def is_batch_valid(
    batch: HSTUBatch,
):
    """
    Validates a batch of data to ensure it meets the necessary criteria.

    Args:
        batch (HSTUBatch): The batch to validate.

    Raises:
        AssertionError: If any of the validation checks fail.
    """
    assert isinstance(batch, HSTUBatch), "batch type should be HSTUBatch"

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

    # Validate labels if present
    if batch.labels is not None:
        assert isinstance(
            batch.labels, KeyedJaggedTensor
        ), "labels should be a KeyedJaggedTensor"
        batchsize = batch.labels.lengths().numel()
        assert (
            batchsize == batch.batch_size
        ), "label batchsize should be equal to batch_size"

    for contextual_feature_name in batch.contextual_feature_names:
        assert (
            contextual_feature_name in batch.features.keys()
        ), f"contextual_feature {contextual_feature_name} is configured, but not in features"
        assert (
            batch.features[contextual_feature_name].lengths().numel()
            == batch.batch_size
        ), f"contextual_feature {contextual_feature_name} shape is not equal to batch_size"
