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

from .hstu_batch import FeatureConfig, HSTUBatch


class HSTURandomDataset(IterableDataset[HSTUBatch]):
    """
    A dummy sequence dataset for generating batches of data.

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
        feature_configs: List[FeatureConfig],
        item_feature_name: str,
        contextual_feature_names: List[str] = [],
        action_feature_name: Optional[str] = None,
        max_num_candidates: int = 0,
        num_generated_batches=1,
        num_tasks: int = 0,
        num_batches: Optional[int] = None,
    ):
        super().__init__()
        self.num_batches: int = cast(
            int, num_batches if num_batches is not None else sys.maxsize
        )
        self._cached_batched: List[HSTUBatch] = []
        self._num_generated_batches = num_generated_batches
        kwargs = dict(
            batch_size=batch_size,
            feature_configs=feature_configs,
            item_feature_name=item_feature_name,
            contextual_feature_names=contextual_feature_names,
            action_feature_name=action_feature_name,
            max_num_candidates=max_num_candidates,
            device=torch.cpu.current_device(),
        )
        for _ in range(self._num_generated_batches):
            if num_tasks > 0:
                self._cached_batched.append(
                    HSTUBatch.random(num_tasks=num_tasks, **kwargs)
                )
            else:
                self._cached_batched.append(HSTUBatch.random(**kwargs))
        self._iloc = 0

    def __iter__(self) -> Iterator[HSTUBatch]:
        """
        Returns an iterator over the cached batches, cycling through them.

        Returns:
            HSTUBatch: The next batch in the cycle.
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
