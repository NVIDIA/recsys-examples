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


from typing import List, Optional

import torch
from dynamicemb.initializer import create_initializer_from_args
from dynamicemb.scored_hashtable import (
    ScoreArg,
    ScorePolicy,
    ScoreSpec,
    get_scored_table,
)
from dynamicemb.types import (
    AdmissionStrategy,
    Counter,
    DynamicEmbInitializerArgs,
    MemoryType,
)


class KVCounter:
    """Per-table counter configuration.

    Users specify one ``KVCounter`` per logical table. At runtime, the
    framework wraps a list of them into a single ``MultiTableKVCounter``
    that manages a fused multi-table scored hash table.
    """

    def __init__(
        self,
        capacity: int,
        bucket_capacity: int = 1024,
        key_type: torch.dtype = torch.int64,
    ):
        self.capacity = capacity
        self.bucket_capacity = bucket_capacity
        self.key_type = key_type


class MultiTableKVCounter(Counter):
    """Multi-table counter backed by a single fused ``ScoredHashTable``.

    Accepts a list of per-table ``KVCounter`` configs and creates one hash
    table whose capacity list maps to the individual counters.
    """

    def __init__(
        self,
        kv_counters: List[KVCounter],
        device: torch.device,
    ):
        if not kv_counters:
            raise ValueError("kv_counters must be non-empty")

        capacities = [kv.capacity for kv in kv_counters]
        self.score_name_ = "counter"
        self.score_specs_ = [
            ScoreSpec(name=self.score_name_, policy=ScorePolicy.ACCUMULATE)
        ]
        self.score_arg_ = ScoreArg(name=self.score_name_)
        self.table_ = get_scored_table(
            capacities,
            kv_counters[0].bucket_capacity,
            kv_counters[0].key_type,
            self.score_specs_,
            device,
        )

    def add(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        frequencies: torch.Tensor,
        founds: torch.Tensor,
    ) -> torch.Tensor:
        if not (
            keys.shape == table_ids.shape == frequencies.shape == founds.shape
        ):
            raise ValueError(
                "keys, table_ids, frequencies, and founds must have the same shape"
            )
        if founds.dtype != torch.bool:
            raise TypeError(f"founds must be torch.bool, got {founds.dtype}")

        self.score_arg_.value = frequencies
        scores_out = torch.empty(keys.numel(), dtype=torch.int64, device=keys.device)
        self.table_.insert(
            keys,
            table_ids,
            self.score_arg_,
            score_out=scores_out,
            mask=~founds,
        )
        return scores_out

    def erase(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if not (keys.shape == table_ids.shape == mask.shape):
            raise ValueError("keys, table_ids, and mask must have the same shape")
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be torch.bool, got {mask.dtype}")
        self.table_.erase(keys, table_ids, mask=mask)

    def memory_usage(self, mem_type=MemoryType.DEVICE) -> int:
        return self.table_.memory_usage(mem_type)

    def load(self, key_file, counter_file, table_id: int) -> None:
        self.table_.load(key_file, {self.score_name_: counter_file}, table_id=table_id)

    def dump(self, key_file, counter_file, table_id: int) -> None:
        self.table_.dump(key_file, {self.score_name_: counter_file}, table_id=table_id)


class FrequencyAdmissionStrategy(AdmissionStrategy):
    """
    Frequency-based admission strategy.
    Only admits keys whose frequency (score) meets or exceeds a threshold.

    Parameters
    ----------
    threshold : int
        Minimum frequency threshold for admission. Keys with frequency >= threshold
        will be admitted into the embedding table.
    initializer_args: Optional[DynamicEmbInitializerArgs]
        Initializer arguments which determine how to initialize the embedding if the key is not admitted.
    """

    def __init__(
        self,
        threshold: int,
        initializer_args: Optional[DynamicEmbInitializerArgs] = None,
    ):
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")

        self.threshold = threshold
        self.initializer_args = initializer_args
        self._non_admit_initializer = None

    def admit(
        self,
        keys: torch.Tensor,
        frequencies: torch.Tensor,
        founds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Admit keys with frequencies >= threshold.

        Parameters
        ----------
        keys : torch.Tensor
            Keys to evaluate (shape: [N])
        frequencies : torch.Tensor
            Frequency counts for each key (shape: [N])
        founds : torch.Tensor
            Boolean mask identifying keys already found in cache or storage.

        Returns
        -------
        torch.Tensor
            Boolean mask (shape: [N]) where True indicates admission
        """
        if not (keys.shape == frequencies.shape == founds.shape):
            raise ValueError(
                "keys, frequencies, and founds must have the same shape"
            )
        if founds.dtype != torch.bool:
            raise TypeError(f"founds must be torch.bool, got {founds.dtype}")

        # Found keys need no admission decision. Keep them false so callers can
        # consume this as a sparse action mask without compacting first.
        admit_mask = (~founds) & (frequencies >= self.threshold)
        return admit_mask

    def initialize_non_admitted_embeddings(
        self,
        buffer: torch.Tensor,
        mask: torch.Tensor,
    ) -> bool:
        """
        Initialize the embeddings for the keys that are not admitted.

        Returns:
            bool: True if the embeddings are initialized, False otherwise.
        """
        if self.initializer_args is None:
            return False
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be torch.bool, got {mask.dtype}")
        if mask.dim() != 1 or mask.numel() != buffer.shape[0]:
            raise ValueError(
                "mask must be one-dimensional and match the first buffer dimension"
            )
        if self._non_admit_initializer is None:
            self._non_admit_initializer = create_initializer_from_args(
                self.initializer_args
            )
        self._non_admit_initializer(
            buffer,
            mask,
            None,
        )
        return True
