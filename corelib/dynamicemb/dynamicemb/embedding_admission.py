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


import torch
from dynamicemb.types import Counter, MemoryType


class KVCounter(Counter):
    """
    Interface of a counter table which maps a key to a counter.
    """

    def __init__(self):
        super().__init__()
        self._counter_dict = {}

    def add(self, keys: torch.Tensor, counters: torch.Tensor) -> torch.Tensor:
        """
        Add keys with counters to the `Counter` and get accumulated counter of each key.
        For not existed keys, the counters will be assigned directly.
        For existing keys, the counters will be accumulated.

        Args:
            keys (torch.Tensor): The input keys, should be unique keys.
            counters (torch.Tensor): The input counters, serve as initial or incremental values of counters' states.

        Returns:
            accumulated_counters (torch.Tensor): the counters' state in the `Counter` for the input keys.
        """
        # accumulated_counters: torch.Tensor
        keys_cpu = keys.cpu().tolist()
        counters_cpu = counters.cpu().tolist()

        accumulated = []
        for k, c in zip(keys_cpu, counters_cpu):
            self._counter_dict[k] = self._counter_dict.get(k, 0) + c
            accumulated.append(self._counter_dict[k])

        return torch.tensor(accumulated, device=keys.device, dtype=counters.dtype)

    def erase(self, keys) -> None:
        """
        Erase keys form the `Counter`.

        Args:
            keys (torch.Tensor): The input keys to be erased.
        """
        keys_cpu = keys.cpu().tolist()
        for k in keys_cpu:
            self._counter_dict.pop(k, None)

    def memory_usage(self, mem_type=MemoryType.DEVICE) -> int:
        """
        Get the consumption of a specific memory type.

        Args:
            mem_type (MemoryType): the specific memory type, default to MemoryType.DEVICE.
        """
        return 0

    def load(self, key_file, counter_file) -> None:
        """
        Load keys and counters from input file path.

        Args:
            key_file (str): the file path of keys.
            counter_file (str): the file path of counters.
        """
        return None

    def dump(self, key_file, counter_file) -> None:
        """
        Dump keys and counters to output file path.

        Args:
            key_file (str): the file path of keys.
            counter_file (str): the file path of counters.
        """
        return None
