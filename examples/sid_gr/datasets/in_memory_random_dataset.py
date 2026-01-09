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
import sys
from typing import Iterator, List, Optional, cast

import fbgemm_gpu  # pylint: disable-unused-import
import torch
from torch.utils.data.dataset import IterableDataset

from .gpt_sid_batch import FeatureConfig, GPTSIDBatch


class InMemoryRandomDataset(IterableDataset[GPTSIDBatch]):
    """
    InMemoryRandomDataset is an iterable dataset for generating random batches of data.

    Args:
        batch_size (int): The batchsize per rank.
        feature_configs (List[FeatureConfig]): A list of configurations for different features.
        item_feature_name (str): The name of the item feature.
        contextual_feature_names (List[str], optional): A list of names for contextual features. Defaults to an empty list.
        action_feature_name (Optional[str], optional): The name of the action feature. Defaults to ``None``.
        max_num_candidates (Optional[int], optional): The maximum number of candidates. Defaults to 0.
        num_generated_batches (int, optional): The number of batches to generate. Defaults to 1.
        num_tasks (int, optional): The number of tasks. Defaults to 0.
        num_batches (bool, optional): The total number of batches to iterate over. Defaults to ``None``.

    """

    def __init__(
        self,
        batch_size: int,
        feature_configs: List[
            FeatureConfig
        ],  # we need feature config for random generation
        raw_hist_sid_names: List[str],
        raw_cand_sid_names: List[str],
        combined_history_feature_name: str,
        combined_candidate_feature_name: str,
        contextual_feature_names: List[str] = [],
        num_generated_batches=1,
        num_batches: Optional[int] = None,
    ):
        super().__init__()
        self.num_batches: int = cast(
            int, num_batches if num_batches is not None else sys.maxsize
        )
        self._cached_batched: List[GPTSIDBatch] = []
        self._num_generated_batches = num_generated_batches
        kwargs = dict(
            batch_size=batch_size,
            feature_configs=feature_configs,
            raw_hist_sid_names=raw_hist_sid_names,
            raw_cand_sid_names=raw_cand_sid_names,
            combined_history_feature_name=combined_history_feature_name,
            combined_candidate_feature_name=combined_candidate_feature_name,
            contextual_feature_names=contextual_feature_names,
            device=torch.cpu.current_device(),
        )
        for _ in range(self._num_generated_batches):
            self._cached_batched.append(GPTSIDBatch.random(**kwargs))
        self._iloc = 0

    def __iter__(self) -> Iterator[GPTSIDBatch]:
        """
        Returns an iterator over the cached batches, cycling through them.

        Returns:
            Union[RankingBatch, RetrievalBatch]: The next batch in the cycle.
        """
        for _ in range(len(self)):
            yield self._cached_batched[self._iloc]
            self._iloc = (self._iloc + 1) % self._num_generated_batches

    def __len__(self) -> int:
        """
        Get the number of batches.

        Returns:
            int: The number of batches.
        """
        return self.num_batches

    @classmethod
    def get_dataset(
        cls,
        batch_size: int,
        feature_configs: List[FeatureConfig],
        raw_hist_sid_names: List[str],
        raw_cand_sid_names: List[str],
        combined_history_feature_name: str,
        combined_candidate_feature_name: str,
        contextual_feature_names: List[str] = [],
        num_generated_batches: int = 1,
        num_batches: Optional[int] = None,
    ):
        return cls(
            batch_size=batch_size,
            feature_configs=feature_configs,
            raw_hist_sid_names=raw_hist_sid_names,
            raw_cand_sid_names=raw_cand_sid_names,
            combined_history_feature_name=combined_history_feature_name,
            combined_candidate_feature_name=combined_candidate_feature_name,
            contextual_feature_names=contextual_feature_names,
            num_generated_batches=num_generated_batches,
            num_batches=num_batches,
        )
