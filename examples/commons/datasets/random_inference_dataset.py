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
import random
from typing import Dict, List, Optional

import torch
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

from .hstu_batch import FeatureConfig, HSTUBatch


class RandomInferenceDataGenerator:
    """
    A random generator for the inference batches

    Args:
        feature_configs (List[FeatureConfig]): The feature configs.
        item_feature_name (str): The item feature name.
        contextual_feature_names (List[str]): The list for contextual features.
        action_feature_name (str): The action feature name.
        max_num_users (int): The maximum user numbers.
        max_batch_size (int): The maximum batch size.
        max_seqlen (int): The maximum sequence length (with candidates) for item
                          in request per user. The length of action sequence in
                          request the same with that of HISTORY item sequence.
        max_num_candidates (int): The maximum candidates number.
        max_incremental_seqlen (int): The maximum incremental length of HISTORY
                                      item AND action sequence.
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
        max_seqlen: int = 4096,
        max_num_candidates: int = 200,
        max_incremental_seqlen: int = 64,
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
        self._max_hist_len = max_seqlen - max_num_candidates
        self._max_incr_fea_len = max(max_incremental_seqlen, 1)
        self._max_num_candidates = max_num_candidates

        self._full_mode = full_mode

        self._item_history: Dict[int, torch.Tensor] = dict()
        self._action_history: Dict[int, torch.Tensor] = dict()

    def get_inference_batch_user_ids(self) -> Optional[torch.Tensor]:
        if self._full_mode:
            batch_size = self._max_batch_size
            user_ids = list(range(self._max_batch_size))
        else:
            batch_size = random.randint(1, self._max_batch_size)
            user_ids = torch.randint(self._max_num_users, (batch_size,)).tolist()
            user_ids = list(set(user_ids))

        user_ids = torch.tensor(
            [
                uid
                for uid in user_ids
                if uid not in self._item_history
                or len(self._item_history[uid]) < self._max_hist_len
            ]
        ).long()
        if self._full_mode and len(user_ids) == 0:
            batch_size = self._max_batch_size
            user_ids = list(
                range(
                    self._max_batch_size,
                    min(self._max_batch_size * 2, self._max_num_users),
                )
            )
            user_ids = torch.tensor(user_ids).long()
        return user_ids if len(user_ids) > 0 else None

    def get_random_inference_batch(
        self, user_ids, truncate_start_positions
    ) -> Optional[HSTUBatch]:
        batch_size = len(user_ids)
        if batch_size == 0:
            return None
        user_ids = user_ids.tolist()
        item_hists = [
            self._item_history[uid] if uid in self._item_history else torch.tensor([])
            for uid in user_ids
        ]
        action_hists = [
            self._action_history[uid]
            if uid in self._action_history
            else torch.tensor([])
            for uid in user_ids
        ]

        lengths = torch.tensor([len(hist_seq) for hist_seq in item_hists]).long()
        incr_lengths = torch.randint(
            low=1, high=self._max_incr_fea_len + 1, size=(batch_size,)
        )
        new_lengths = torch.clamp(lengths + incr_lengths, max=self._max_hist_len).long()
        incr_lengths = new_lengths - lengths

        num_candidates = torch.randint(
            low=1, high=self._max_num_candidates + 1, size=(batch_size,)
        )
        if self._full_mode:
            incr_lengths = torch.full((batch_size,), self._max_incr_fea_len)
            new_lengths = torch.clamp(
                lengths + incr_lengths, max=self._max_hist_len
            ).long()
            incr_lengths = new_lengths - lengths
            num_candidates = torch.full((batch_size,), self._max_num_candidates)

        # Caveats: truncate_start_positions is for interleaved item-action sequence
        item_start_positions = (truncate_start_positions / 2).to(torch.int32)
        action_start_positions = (truncate_start_positions / 2).to(torch.int32)

        item_seq = list()
        action_seq = list()
        for idx, uid in enumerate(user_ids):
            self._item_history[uid] = torch.cat(
                [
                    item_hists[idx],
                    torch.randint(self._max_item_id + 1, (incr_lengths[idx],)),
                ],
                dim=0,
            ).long()
            self._action_history[uid] = torch.cat(
                [
                    action_hists[idx],
                    torch.randint(self._max_action_id + 1, (incr_lengths[idx],)),
                ],
                dim=0,
            ).long()

            item_history = torch.cat(
                [
                    self._item_history[uid][item_start_positions[idx] :],
                    torch.randint(self._max_item_id + 1, (num_candidates[idx].item(),)),
                ],
                dim=0,
            )
            item_seq.append(item_history)
            action_seq.append(self._action_history[uid][action_start_positions[idx] :])

        features = KeyedJaggedTensor.from_jt_dict(
            {
                self._item_fea_name: JaggedTensor.from_dense(item_seq),
                self._action_fea_name: JaggedTensor.from_dense(action_seq),
            }
        )

        return HSTUBatch(
            features=features,
            batch_size=batch_size,
            feature_to_max_seqlen=self._fea_name_to_max_seqlen,
            contextual_feature_names=self._contextual_fea_names,
            item_feature_name=self._item_fea_name,
            action_feature_name=self._action_fea_name,
            max_num_candidates=self._max_num_candidates,
            num_candidates=num_candidates,
        )
