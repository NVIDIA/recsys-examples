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
from typing import Dict, Iterator, List, Tuple

import torch
from torch.utils.data.dataset import IterableDataset
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

from .hstu_batch import FeatureConfig, HSTUBatch


class RandomInferenceDataset(
    IterableDataset[Tuple[HSTUBatch, torch.Tensor, torch.Tensor]]
):
    """
    A random generator for the inference batches

    Args:
        feature_configs (List[FeatureConfig]): The feature configs.
        item_feature_name (str): The item feature name.
        contextual_feature_names (List[str]): The list for contextual features.
        action_feature_name (str): The action feature name.
        max_num_users (int): The maximum user numbers.
        max_batch_size (int): The maximum batch size.
        max_history_length (int): The maximum history length for item in request per user.
                                  The length of action sequence in request is the same.
        max_num_candidates (int): The maximum candidates number.
        max_incremental_seqlen (int): The maximum incremental length of HISTORY
                                      item AND action sequence.
        max_num_cached_batches (int, optional): The number of batches to generate. Defaults to 1.
        full_mode (bool): The flag for full batch mode.
    """

    def __init__(
        self,
        feature_configs: List[FeatureConfig],
        item_feature_name: str,
        contextual_feature_names: List[str] = [],
        action_feature_name: str = "",
        max_num_users: int = 1,
        max_batch_size: int = 32,
        max_history_length: int = 4096,
        max_num_candidates: int = 200,
        max_incremental_seqlen: int = 64,
        max_num_cached_batches: int = 1,
        full_mode: bool = False,
    ):
        super().__init__()

        self._item_fea_name = item_feature_name
        self._action_fea_name = action_feature_name
        self._contextual_fea_names = contextual_feature_names
        self._fea_name_to_max_seqlen = dict()
        self._max_item_id = 0
        self._max_action_id = 0
        for fc in feature_configs:
            for fea_name, fea_max_id in zip(fc.feature_names, fc.max_item_ids):
                self._fea_name_to_max_seqlen[fea_name] = fc.max_sequence_length
                if fea_name == self._item_fea_name:
                    self._max_item_id = fea_max_id
                elif fea_name == self._action_fea_name:
                    self._max_action_id = fea_max_id

        self._max_num_users = min(max_num_users, 2**16)
        self._max_batch_size = max_batch_size
        self._max_hist_len = max_history_length
        self._max_num_candidates = max_num_candidates
        self._max_incr_fea_len = max(max_incremental_seqlen, 1)
        self._num_generated_batches = max(max_num_cached_batches, 1)

        self._full_mode = full_mode

        self._item_history: Dict[int, torch.Tensor] = dict()
        self._action_history: Dict[int, torch.Tensor] = dict()

        num_cached_batches = 0
        self._cached_batch = list()
        for seqlen_idx in range(
            max_incremental_seqlen, self._max_hist_len, max_incremental_seqlen
        ):
            for idx in range(0, self._max_num_users, self._max_batch_size):
                if self._full_mode:
                    user_ids = list(
                        range(idx, min(self._max_num_users, idx + self._max_batch_size))
                    )
                else:
                    user_ids = torch.randint(
                        self._max_num_users, (self._max_batch_size,)
                    ).tolist()
                    user_ids = list(set(user_ids))

                batch_size = len(user_ids)

                item_seq = list()
                action_seq = list()
                for uid in user_ids:
                    if uid not in self._item_history or uid not in self._action_history:
                        self._item_history[uid] = torch.randint(
                            self._max_item_id + 1,
                            (self._max_hist_len + self._max_num_candidates,),
                        )
                        self._action_history[uid] = torch.randint(
                            self._max_action_id + 1,
                            (self._max_hist_len + self._max_num_candidates,),
                        )

                    item_seq.append(
                        self._item_history[uid][: seqlen_idx + self._max_num_candidates]
                    )
                    action_seq.append(self._action_history[uid][:seqlen_idx])
                features = KeyedJaggedTensor.from_jt_dict(
                    {
                        self._item_fea_name: JaggedTensor.from_dense(item_seq),
                        self._action_fea_name: JaggedTensor.from_dense(action_seq),
                    }
                )

                if self._full_mode:
                    num_candidates = torch.full((batch_size,), self._max_num_candidates)
                else:
                    num_candidates = torch.randint(
                        low=1, high=self._max_num_candidates + 1, size=(batch_size,)
                    )

                total_history_lengths = torch.full((batch_size,), seqlen_idx * 2)

                batch = HSTUBatch(
                    features=features,
                    batch_size=batch_size,
                    feature_to_max_seqlen=self._fea_name_to_max_seqlen,
                    contextual_feature_names=self._contextual_fea_names,
                    item_feature_name=self._item_fea_name,
                    action_feature_name=self._action_fea_name,
                    max_num_candidates=self._max_num_candidates,
                    num_candidates=num_candidates,
                ).to(device=torch.cuda.current_device())
                self._cached_batch.append(
                    tuple([batch, torch.tensor(user_ids).long(), total_history_lengths])
                )
                num_cached_batches += 1
                if num_cached_batches >= self._num_generated_batches:
                    break

        self._num_generated_batches = len(self._cached_batch)
        self._max_num_batches = self._num_generated_batches
        self._iloc = 0

    def __iter__(self) -> Iterator[Tuple[HSTUBatch, torch.Tensor, torch.Tensor]]:
        """
        Returns an iterator over the cached batches, cycling through them.

        Returns:
            Tuple[HSTUBatch, torch.Tensor, torch.Tensor]: The next (batch, user_ids, history_lens) in the cycle.
        """
        for _ in range(len(self)):
            yield self._cached_batch[self._iloc]
            self._iloc = (self._iloc + 1) % self._num_generated_batches

    def __len__(self) -> int:
        """
        Get the number of batches.

        Returns:
            int: The number of batches.
        """
        return self._max_num_batches
