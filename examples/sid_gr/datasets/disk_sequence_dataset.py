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

import math
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import IterableDataset
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from .gpt_sid_batch import GPTSIDBatch


class DiskSequenceDataset(IterableDataset[GPTSIDBatch]):
    """
    DiskSequenceDataset is an iterable dataset designed for sid-gr
    """

    def __init__(
        self,
        raw_sequence_data_path: str,
        batch_size: int,  # local batch size
        max_history_length: int,  # history seqlen
        raw_sequence_feature_name: str,  # 'sequence_data'
        num_hierarchies: int,
        codebook_sizes: List[int],
        output_history_sid_feature_name: Optional[str] = None,
        output_candidate_sid_feature_name: Optional[str] = None,
        max_candidate_length: int = 1,  # candidate seqlen
        contextual_feature_names: List[str] = [],
        item_id_to_sid_mapping_tensor_path: Optional[str] = None,
        *,
        rank: int,
        world_size: int,
        shuffle: bool,
        random_seed: int,
        is_train_dataset: bool,
        deduplicate_data_across_hierarchy: bool = True,
        deduplicate_label_across_hierarchy: bool = False,
        sort_by_user_id: bool = True,  # for debugging purpose
    ):
        # items and timestamps are nested
        super().__init__()
        if output_history_sid_feature_name is None:
            output_history_sid_feature_name = f"hist_sids{num_hierarchies}d"
        if output_candidate_sid_feature_name is None:
            output_candidate_sid_feature_name = f"cand_sids{num_hierarchies}d"
        self._device = torch.cpu.current_device()
        raw_sequence_data = pd.read_parquet(raw_sequence_data_path)
        if sort_by_user_id:
            raw_sequence_data = raw_sequence_data.sort_values(by="user_id")
        if "sequence_length" not in raw_sequence_data.columns:
            raw_sequence_data["sequence_length"] = raw_sequence_data[
                raw_sequence_feature_name
            ].apply(len)
        assert (
            max_candidate_length <= 1
        ), "max_candidate_length should be less than or equal to 1 for now"

        # clamp the sequence length to 1 + max_candidate_length (at least 1 history item)
        raw_sequence_data = raw_sequence_data[
            raw_sequence_data["sequence_length"] >= 1 + max_candidate_length
        ]  # at least 2 items in the sequence

        # truncate the sequence to the total sequence length
        # note that max_history_length + max_candidate_length is the total sequence length
        raw_sequence_data[raw_sequence_feature_name] = raw_sequence_data[
            raw_sequence_feature_name
        ].apply(
            lambda x: x[: max_history_length + max_candidate_length]
            if isinstance(x, (list, np.ndarray))
            else x
        )
        raw_sequence_data["sequence_length"] = raw_sequence_data[
            "sequence_length"
        ].clip(upper=max_history_length + max_candidate_length)
        self._feature_to_max_seqlen = {
            output_history_sid_feature_name: max_history_length * num_hierarchies,
            output_candidate_sid_feature_name: max_candidate_length * num_hierarchies,
        }
        try:
            self.item_id_to_sid_mapping_tensor = torch.load(
                item_id_to_sid_mapping_tensor_path
            )

            if not isinstance(self.item_id_to_sid_mapping_tensor, torch.Tensor):
                raise TypeError("item_id_to_sid_mapping_tensor should be a tensor")
            assert (
                self.item_id_to_sid_mapping_tensor.dim() == 2
            ), "item_id_to_sid_mapping_tensor should be a 2D tensor"
            (
                mappping_num_hierarchies,
                num_items,
            ) = self.item_id_to_sid_mapping_tensor.shape
            assert (
                mappping_num_hierarchies == num_hierarchies
            ), "item_id_to_sid_mapping_tensor should have the same number of rows as num_hierarchies"
        except:
            raise RuntimeError("Failed to load item_id_to_sid_mapping_tensor")

        self._raw_sequence_data = raw_sequence_data
        self._num_samples = raw_sequence_data.shape[0]

        self._raw_sequence_feature_name = raw_sequence_feature_name
        self._output_history_sid_feature_name = output_history_sid_feature_name
        self._output_candidate_sid_feature_name = output_candidate_sid_feature_name
        self._num_hierarchies = num_hierarchies

        self._max_candidate_length = max_candidate_length
        self._batch_size = batch_size
        self._global_batch_size = batch_size * world_size
        self._is_train_dataset = is_train_dataset
        self._rank = rank
        self._world_size = world_size
        self._sample_ids = np.arange(self._num_samples)
        codebook_offsets = torch.tensor(
            np.cumsum([0] + codebook_sizes[:-1]), device=self._device
        )
        # dedup data and label offsets
        self.data_codebook_offsets = (
            codebook_offsets
            if deduplicate_data_across_hierarchy
            else torch.zeros(self._num_hierarchies, device=self._device)
        )
        self.label_codebook_offsets = (
            codebook_offsets
            if deduplicate_label_across_hierarchy
            else torch.zeros(self._num_hierarchies, device=self._device)
        )
        # TODO: Add shuffle and random seed

    def __iter__(self) -> Iterator[GPTSIDBatch]:
        for i in range(len(self)):
            local_batch_start = (
                i * self._global_batch_size + self._rank * self._batch_size
            )
            local_batch_end = min(
                i * self._global_batch_size + (self._rank + 1) * self._batch_size,
                len(self._sample_ids),
            )
            actual_batch_size = local_batch_end - local_batch_start
            sample_ids = self._sample_ids[local_batch_start:local_batch_end]
            sequence_data = self._raw_sequence_data.iloc[sample_ids]
            # split history and candidate
            # [1,2,| 3]      => [1,2], [3]
            # [1,2,3,4, | 5] => [1,2,3,4], [5]
            # [1,2, |3]      => [1,2], [3]
            # candidate might be empty, so we need to handle it separately
            history_item_ids = torch.tensor(
                sequence_data[self._raw_sequence_feature_name]
                .apply(
                    lambda x: x[: -self._max_candidate_length]
                    if self._max_candidate_length > 0
                    else x
                )
                .explode()
                .to_numpy()
                .astype(np.int64),
                device=self._device,
            )
            candidate_item_ids = (
                torch.tensor(
                    sequence_data[self._raw_sequence_feature_name]
                    .apply(lambda x: x[-self._max_candidate_length :])
                    .explode()
                    .to_numpy()
                    .astype(np.int64),
                    device=self._device,
                )
                if self._max_candidate_length > 0
                else None
            )
            user_id = torch.tensor(
                sequence_data["user_id"].to_numpy().astype(np.int64),
                device=self._device,
            )
            # add offset to the sids to avoid duplicate sids across hierarchy
            # [T, num_hierarchies]
            history_sids = torch.index_select(
                self.item_id_to_sid_mapping_tensor, dim=1, index=history_item_ids
            ).transpose(0, 1).contiguous() + self.data_codebook_offsets.unsqueeze(0)
            # labels are the candidate sids but starting from 0.
            candidate_sids = (
                (
                    torch.index_select(
                        self.item_id_to_sid_mapping_tensor,
                        dim=1,
                        index=candidate_item_ids,
                    )
                    .transpose(0, 1)
                    .contiguous()
                )
                if self._max_candidate_length > 0
                else None
            )

            if self._max_candidate_length > 0:
                labels = candidate_sids + self.label_codebook_offsets.unsqueeze(0)
            else:
                # we need to remove the starting sids for each sequence.
                # TODO@junzhang, to optimize the redundant df operations and sid transformations.
                label_item_ids = torch.tensor(
                    sequence_data[self._raw_sequence_feature_name]
                    .apply(lambda x: x[1:])
                    .explode()
                    .to_numpy()
                    .astype(np.int64),
                    device=self._device,
                )
                labels = (
                    torch.index_select(
                        self.item_id_to_sid_mapping_tensor, dim=1, index=label_item_ids
                    )
                    .transpose(0, 1)
                    .contiguous()
                ) + self.label_codebook_offsets.unsqueeze(0)

            candidate_sids = (
                candidate_sids + self.data_codebook_offsets.unsqueeze(0)
                if self._max_candidate_length > 0
                else None
            )
            # 'sequence length' is the total length
            history_lengths = (
                torch.tensor(
                    sequence_data["sequence_length"].to_numpy().astype(np.int64)
                    - self._max_candidate_length,
                    device=self._device,
                    dtype=torch.int64,
                )
                * self._num_hierarchies
            )
            candidate_lengths = (
                torch.ones(actual_batch_size, device=self._device, dtype=torch.int64)
                * self._max_candidate_length
                * self._num_hierarchies
            )

            def pad_tensor(padding_length: int, tensor: torch.Tensor) -> torch.Tensor:
                if padding_length == 0:
                    return tensor
                return torch.nn.functional.pad(
                    tensor, (0, padding_length), "constant", 0
                )

            history_lengths = pad_tensor(
                self._batch_size - actual_batch_size, history_lengths
            )
            candidate_lengths = pad_tensor(
                self._batch_size - actual_batch_size, candidate_lengths
            )

            batch_kwargs = dict(
                features=KeyedJaggedTensor.from_lengths_sync(
                    keys=[
                        self._output_history_sid_feature_name,
                        self._output_candidate_sid_feature_name,
                    ],
                    values=torch.cat([history_sids.view(-1), candidate_sids.view(-1)])
                    if self._max_candidate_length > 0
                    else history_sids.view(-1),
                    lengths=torch.cat([history_lengths, candidate_lengths]),
                ),
                batch_size=self._batch_size,
                feature_to_max_seqlen=self._feature_to_max_seqlen,
                _num_hierarchies=self._num_hierarchies,
                history_feature_name=self._output_history_sid_feature_name,
                candidate_feature_name=self._output_candidate_sid_feature_name,
                labels=labels,  # for eval, we need label to calculate metrics.
                user_id=user_id,
                actual_batch_size=actual_batch_size,
            )
            yield GPTSIDBatch(**batch_kwargs)

    def __len__(self) -> int:
        return math.ceil(self._num_samples / self._global_batch_size)

    @classmethod
    def get_dataset(
        cls,
        raw_sequence_data_path: str,
        item_id_to_sid_mapping_tensor_path: str,
        batch_size: int,
        max_history_length: int,
        max_candidate_length: int,
        raw_sequence_feature_name: str,
        num_hierarchies: int,
        codebook_sizes: List[int],
        rank: int,
        world_size: int,
        shuffle: bool,
        random_seed: int,
        is_train_dataset: bool,
        deduplicate_data_across_hierarchy: bool,
        deduplicate_label_across_hierarchy: bool,
        output_history_sid_feature_name: str,
        output_candidate_sid_feature_name: str,
    ) -> "DiskSequenceDataset":
        return cls(
            raw_sequence_data_path=raw_sequence_data_path,
            item_id_to_sid_mapping_tensor_path=item_id_to_sid_mapping_tensor_path,
            batch_size=batch_size,
            max_history_length=max_history_length,
            max_candidate_length=max_candidate_length,
            raw_sequence_feature_name=raw_sequence_feature_name,
            num_hierarchies=num_hierarchies,
            codebook_sizes=codebook_sizes,
            output_history_sid_feature_name=output_history_sid_feature_name,
            output_candidate_sid_feature_name=output_candidate_sid_feature_name,
            rank=rank,
            world_size=world_size,
            shuffle=shuffle,
            random_seed=random_seed,
            is_train_dataset=is_train_dataset,
            deduplicate_data_across_hierarchy=deduplicate_data_across_hierarchy,
            deduplicate_label_across_hierarchy=deduplicate_label_across_hierarchy,
        )
