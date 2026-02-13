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
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from commons.datasets.hstu_batch import HSTUBatch
from torch.utils.data.dataset import IterableDataset
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def load_seq(x: str):
    if isinstance(x, str):
        y = json.loads(x)
    else:
        y = x
    return y


def maybe_truncate_seq(
    y: List[int],
    max_seq_len: int,
) -> List[int]:
    y_len = len(y)
    if y_len > max_seq_len:
        y = y[:max_seq_len]
    return y


class InferenceDataset(IterableDataset[HSTUBatch]):
    """
    HSTUSequenceDataset is an iterable dataset designed for distributed recommendation systems.
    It handles loading, shuffling, and batching of sequence data for training models.

    Args:
        seq_logs_file (str): Path to the sequence logs file.
        batch_size (int): The batch size.
        max_seqlen (int): The maximum sequence length.
        item_feature_name (str): The name of the item feature.
        contextual_feature_names (list[str], optional): List of contextual feature names. Defaults to [].
        action_feature_name (str, optional): The name of the action feature. Defaults to None.
        max_num_candidates (int, optional): The maximum number of candidate items. Defaults to 0.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        shuffle (bool): Whether to shuffle the data.
        random_seed (int): The random seed for shuffling.
        is_train_dataset (bool): Whether this dataset is for training.
        nrows (int, optional): The number of rows to read from the file. Defaults to None, meaning all rows are read.
    """

    def __init__(
        self,
        seq_logs_file: str,
        batch_logs_file: str,
        batch_size: int,
        max_seqlen: int,
        item_feature_name: str,
        contextual_feature_names: List[str],
        action_feature_name: str,
        max_num_candidates: int = 0,
        *,
        item_vocab_size: int,
        userid_name: str,
        date_name: str,
        sequence_endptr_name: str,
        timestamp_names: List[str],
        random_seed: int = 0,
        seq_nrows: Optional[int] = None,
        batch_nrows: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._device = torch.cuda.current_device()

        if seq_nrows is None or batch_nrows is None:
            self._seq_logs_frame = pd.read_csv(seq_logs_file, delimiter=",")
            self._batch_logs_frame = pd.read_csv(batch_logs_file, delimiter=",")
        else:
            self._seq_logs_frame = pd.read_csv(
                seq_logs_file, delimiter=",", nrows=seq_nrows
            )
            self._batch_logs_frame = pd.read_csv(
                batch_logs_file, delimiter=",", nrows=batch_nrows
            )

        self._batch_logs_frame.sort_values(by=timestamp_names, inplace=True)
        len(self._batch_logs_frame)

        self._num_samples = len(self._batch_logs_frame)
        self._max_seqlen = max_seqlen

        self._batch_size = batch_size
        self._random_seed = random_seed

        self._contextual_feature_names = contextual_feature_names
        if self._max_seqlen <= len(self._contextual_feature_names):
            raise ValueError(
                f"max_seqlen is too small. should > {len(self._contextual_feature_names)}"
            )
        self._item_feature_name = item_feature_name
        self._action_feature_name = action_feature_name
        self._max_num_candidates = max_num_candidates
        self._item_vocab_size = item_vocab_size
        self._userid_name = userid_name
        self._date_name = date_name
        self._seq_end_name = sequence_endptr_name

        self._sample_ids = np.arange(self._num_samples)

    # We do batching in our own
    def __len__(self) -> int:
        return math.ceil(self._num_samples / self._batch_size)

    def __iter__(self) -> Iterator[HSTUBatch]:
        for i in range(len(self)):
            batch_start = i * self._batch_size
            batch_end = min(
                (i + 1) * self._batch_size,
                len(self._sample_ids),
            )
            sample_ids = self._sample_ids[batch_start:batch_end]
            user_ids: List[int] = []
            dates: List[int] = []
            seq_endptrs: List[int] = []
            for sample_id in sample_ids:
                seq_endptr = self._batch_logs_frame.iloc[sample_id][self._seq_end_name]
                if seq_endptr > self._max_seqlen:
                    continue
                user_ids.append(
                    self._batch_logs_frame.iloc[sample_id][self._userid_name]
                )
                dates.append(self._batch_logs_frame.iloc[sample_id][self._date_name])
                seq_endptrs.append(seq_endptr)

            last_date = dates[0]
            final_user_ids: List[int] = []
            final_dates: List[int] = []
            final_seq_endptrs: List[int] = []
            for uid, date, endp in zip(user_ids, dates, seq_endptrs):
                if date != last_date:
                    continue
                if uid not in final_user_ids:
                    final_user_ids.append(uid)
                    final_dates.append(date)
                    final_seq_endptrs.append(endp)
                else:
                    idx = final_user_ids.index(uid)
                    final_seq_endptrs[idx] = max(final_seq_endptrs[idx], endp)
            yield (
                torch.tensor(final_user_ids),
                torch.tensor(final_dates),
                torch.tensor(final_seq_endptrs),
            )

    def get_input_batch(
        self,
        user_ids,
        dates,
        sequence_endptrs,
        sequence_startptrs,
        with_contextual_features=False,
        with_ranking_labels=False,
    ):
        contextual_features: Dict[str, List[int]] = defaultdict(list)
        contextual_features_seqlen: Dict[str, List[int]] = defaultdict(list)
        item_features: List[int] = []
        item_features_seqlen: List[int] = []
        action_features: List[int] = []
        action_features_seqlen: List[int] = []
        num_candidates: List[int] = []
        labels: List[int] = []

        packed_user_ids: List[int] = []

        if len(user_ids) == 0:
            return None

        sequence_endptrs = torch.clip(sequence_endptrs, 0, self._max_seqlen)
        for idx in range(len(user_ids)):
            uid = user_ids[idx].item()
            date = dates[idx].item()
            end_pos = sequence_endptrs[idx].item()  # history_end_pos
            start_pos = sequence_startptrs[idx].item()

            data = self._seq_logs_frame[
                (self._seq_logs_frame[self._userid_name] == uid)
                & (self._seq_logs_frame[self._date_name] == date)
            ]
            data = data.iloc[0]
            if with_contextual_features:
                for contextual_feature_name in self._contextual_feature_names:
                    contextual_features[contextual_feature_name].append(
                        data[contextual_feature_name]
                    )
                    contextual_features_seqlen[contextual_feature_name].append(1)

            item_seq = load_seq(data[self._item_feature_name])[start_pos:end_pos]
            action_seq = load_seq(data[self._action_feature_name])[start_pos:end_pos]
            num_candidate = 0
            if self._max_num_candidates > 0:
                # randomly generated candidates
                if not with_ranking_labels:
                    # num_candidate = (torch.randint(self._max_num_candidates) + 1).item()
                    num_candidate = self._max_num_candidates
                    candidate_seq = torch.randint(
                        self._item_vocab_size, (num_candidate,)
                    ).tolist()

                # extract candidates from following sequences
                else:
                    all_seqs = self._seq_logs_frame[
                        (self._seq_logs_frame[self._userid_name] == uid)
                        & (self._seq_logs_frame[self._date_name] >= date)
                    ]
                    candidate_seq = sum(
                        [
                            load_seq(all_seqs.iloc[idx][self._item_feature_name])
                            for idx in range(len(all_seqs))
                        ],
                        start=[],
                    )[end_pos : end_pos + self._max_num_candidates]
                    num_candidate = len(candidate_seq)
                    label_seq = sum(
                        [
                            load_seq(all_seqs.iloc[idx][self._action_feature_name])
                            for idx in range(len(all_seqs))
                        ],
                        start=[],
                    )[end_pos : end_pos + self._max_num_candidates]

                all_item_seq = item_seq + candidate_seq

            item_features.extend(all_item_seq)
            item_features_seqlen.append(len(all_item_seq))
            num_candidates.append(num_candidate)
            if with_ranking_labels:
                labels.extend(label_seq)

            action_features.extend(action_seq)
            action_features_seqlen.append(len(action_seq))

            packed_user_ids.append(uid)

        if len(packed_user_ids) == 0:
            return None

        feature_to_max_seqlen = {}
        for name in self._contextual_feature_names:
            feature_to_max_seqlen[name] = max(
                contextual_features_seqlen[name], default=0
            )

        ### Currently use clipped maxlen. check how this impacts the hstu results
        feature_to_max_seqlen[self._item_feature_name] = max(item_features_seqlen)
        feature_to_max_seqlen[self._action_feature_name] = max(action_features_seqlen)

        if with_contextual_features:
            contextual_features_tensor = torch.tensor(
                [contextual_features[name] for name in self._contextual_feature_names],
            ).view(-1)
            contextual_features_lengths_tensor = torch.tensor(
                [
                    contextual_features_seqlen[name]
                    for name in self._contextual_feature_names
                ]
            ).view(-1)
        else:
            contextual_features_tensor = torch.empty((0,), dtype=torch.int64)
            contextual_features_lengths_tensor = torch.tensor(
                [0 for name in self._contextual_feature_names]
            ).view(-1)
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=self._contextual_feature_names
            + [self._item_feature_name, self._action_feature_name],
            values=torch.concat(
                [
                    contextual_features_tensor.to(device=self._device),
                    torch.tensor(item_features, device=self._device),
                    torch.tensor(action_features, device=self._device),
                ]
            ).long(),
            lengths=torch.concat(
                [
                    contextual_features_lengths_tensor.to(device=self._device),
                    torch.tensor(item_features_seqlen, device=self._device),
                    torch.tensor(action_features_seqlen, device=self._device),
                ]
            ).long(),
        )
        labels = torch.tensor(labels, dtype=torch.int64, device=self._device)
        batch_kwargs = dict(
            features=features,
            batch_size=len(user_ids),
            feature_to_max_seqlen=feature_to_max_seqlen,
            contextual_feature_names=self._contextual_feature_names,
            item_feature_name=self._item_feature_name,
            action_feature_name=self._action_feature_name,
            max_num_candidates=self._max_num_candidates,
            num_candidates=torch.tensor(num_candidates, device=self._device)
            if self._max_num_candidates > 0
            else None,
        )
        labels = labels if with_ranking_labels else None
        return HSTUBatch(labels=labels, **batch_kwargs)
