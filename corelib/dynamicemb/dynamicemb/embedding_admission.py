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


from typing import Optional

import torch
from dynamicemb.scored_hashtable import (
    ScoreArg,
    ScorePolicy,
    ScoreSpec,
    get_scored_table,
)
from dynamicemb.types import Counter, MemoryType


class KVCounter(Counter):
    """
    Interface of a counter table which maps a key to a counter.
    """

    def __init__(
        self,
        capacity: int,
        bucket_capacity: Optional[int] = 128,
        key_type: Optional[torch.dtype] = torch.int64,
        device: torch.device = None,
    ):
        self.score_name_ = "counter"
        self.score_specs_ = [
            ScoreSpec(name=self.score_name_, policy=ScorePolicy.ACCUMULATE)
        ]
        self.score_args_ = [ScoreArg(name=self.score_name_, is_return=True)]

        self.table_ = get_scored_table(
            capacity, bucket_capacity, key_type, self.score_specs_, device
        )

    def add(
        self, keys: torch.Tensor, frequencies: torch.Tensor, inplace: bool
    ) -> torch.Tensor:
        """
        Add keys with frequencies to the `Counter` and get accumulated counter of each key.
        For not existed keys, the frequencies will be assigned directly.
        For existing keys, the frequencies will be accumulated.

        Args:
            keys (torch.Tensor): The input keys, should be unique keys.
            frequencies (torch.Tensor): The input frequencies, serve as initial or incremental values of frequencies' states.
            inplace: If true then store the accumulated_frequencies to counter.

        Returns:
            accumulated_frequencies (torch.Tensor): the frequencies' state in the `Counter` for the input keys.
        """
        assert inplace == True, "Only support inplace=True"
        self.score_args_[0].value = frequencies

        self.table_.insert(keys, self.score_args_)
        return frequencies

    def erase(self, keys) -> None:
        """
        Erase keys form the `Counter`.

        Args:
            keys (torch.Tensor): The input keys to be erased.
        """
        self.table_.erase(keys)

    def memory_usage(self, mem_type=MemoryType.DEVICE) -> int:
        """
        Get the consumption of a specific memory type.

        Args:
            mem_type (MemoryType): the specific memory type, default to MemoryType.DEVICE.
        """
        return self.table_.memory_usage(mem_type)

    def load(self, key_file, counter_file) -> None:
        """
        Load keys and frequencies from input file path.

        Args:
            key_file (str): the file path of keys.
            counter_file (str): the file path of frequencies.
        """
        self.table_.load(key_file, {self.score_name_: counter_file})

    def dump(self, key_file, counter_file) -> None:
        """
        Dump keys and frequencies to output file path.

        Args:
            key_file (str): the file path of keys.
            counter_file (str): the file path of frequencies.
        """
        self.table_.dump(key_file, {self.score_name_: counter_file})
