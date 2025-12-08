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

from typing import Iterator, List, Optional

import pandas as pd
from torch.utils.data.dataset import IterableDataset

from .gpt_sid_batch import GPTSIDBatch


def _transform_item_id_to_sid(
    raw_sequence_df: pd.DataFrame,
    item_to_sid_mapping: pd.DataFrame,
    item_id_column_name: str,
    sid_column_names: List[str],
) -> pd.DataFrame:
    df_merged = raw_sequence_df.merge(
        item_to_sid_mapping, on=item_id_column_name, how="left"
    )
    assert (
        df_merged[sid_column_names].isnull().sum() == 0
    ), "all item_ids should be mapped to sids"
    return df_merged


class DiskSequenceDataset(IterableDataset[GPTSIDBatch]):
    """
    DiskSequenceDataset is an iterable dataset designed for sid-gr
    """

    def __init__(
        self,
        raw_sequence_data_path: str,
        batch_size: int,
        max_seqlen: int,
        raw_sequence_feature_name: str,
        contextual_feature_names: List[str],
        raw_sequence_is_sid: bool,
        num_hierarchies: int,
        output_history_sid_feature_name: str,
        output_candidate_sid_feature_name: str,
        item_to_sid_mapping_path: Optional[str] = None,
        *,
        rank: int,
        world_size: int,
        shuffle: bool,
        random_seed: int,
        is_train_dataset: bool,
        nrows: Optional[int] = None,
    ):
        # items and timestamps are nested
        # user_id,item_ids,timestamps
        # 1, [128,122,134], [100,200,300]
        raw_sequence_data = pd.read_parquet(raw_sequence_data_path)
        if not raw_sequence_is_sid:
            assert (
                item_to_sid_mapping_path is not None
            ), "item_id_to_sid_mapping_path is required when raw_sequence_is_sid is False"
            # in item_to_sid_mapping_path, sids are not nested so that we can validate the hierarchy with the number of columns
            # item,sid0,sid1,sid2,sid3
            # 1,11,23,24,25
            self.item_id_to_sid_mapping = pd.read_parquet(
                item_to_sid_mapping_path
            ).fillna(0)
            assert (
                len(self.item_id_to_sid_mapping.columns) == num_hierarchies
            ), "item_id_to_sid_mapping should have the same number of columns as num_hierarchies"
        else:
            self.item_id_to_sid_mapping = None

        raw_sequence_data[contextual_feature_names]
        # we need to split the raw sequence data into history and candidate sequences
        raw_sequence_data[raw_sequence_feature_name]

    def __iter__(self) -> Iterator[GPTSIDBatch]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self.raw_sequence_data)

    def __getitem__(self, index: int) -> GPTSIDBatch:
        pass

    #   pass

    # @classmethod
    # def get_dataset(cls,
    #     dataset_name: str,
    #     sequence_features_data_path: str,
    #     max_seqlen: int,
    #     batch_size: int,
    #     rank: int,
    #     world_size: int,
    #     shuffle: bool,
    #     random_seed: int,
    # ):

    #   return cls(
    #     dataset_name=dataset_name,
    #     sequence_features_data_path=sequence_features_data_path,
    #     max_seqlen=max_seqlen,
    #     batch_size=batch_size,
    #     rank=rank,
    #     world_size=world_size,
    #     shuffle=shuffle,
    #     random_seed=random_seed,
    #   )
