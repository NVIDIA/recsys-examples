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

import abc
import enum
from dataclasses import dataclass
from typing import Generic, Iterator, Optional, Tuple, TypeVar

import numpy as np
import torch


@enum.unique
class MemoryType(enum.Enum):
    DEVICE = "device"  # memory allocated using cudaMalloc/cudaMallocAsync
    MANAGED = "managed"  # memory allocated using cudaMallocManaged
    PINNED_HOST = "pinned_host"  # memory allocated using cudaHostAlloc/cudaMallocHost
    HOST = "host"  # system memory allocated using e.g. malloc.


class DynamicEmbInitializerMode(enum.Enum):
    """
    Enumeration for different modes of initializing dynamic embedding vector values.

    Attributes
    ----------
    NORMAL : str
        Normal Distribution.
    UNIFORM : str
        Uniform distribution of random values.
    CONSTANT : str
        All dynamic embedding vector values are a given constant.
    DEBUG : str
        Debug value generation mode for testing.
    """

    NORMAL = "normal"
    TRUNCATED_NORMAL = "truncated_normal"
    UNIFORM = "uniform"
    CONSTANT = "constant"
    DEBUG = "debug"


@dataclass
class DynamicEmbInitializerArgs:
    """
    Arguments for initializing dynamic embedding vector values.

    Attributes
    ----------
    mode : DynamicEmbInitializerMode
        The mode of initialization, one of the DynamicEmbInitializerMode values.
    mean : float, optional
        The mean value for (truncated) normal distributions. Defaults to 0.0.
    std_dev : float, optional
        The standard deviation for (truncated) normal distributions. Defaults to 1.0.
    lower : float, optional
        The lower bound for uniform/truncated_normal distribution. Defaults to 0.0.
    upper : float, optional
        The upper bound for uniform/truncated_normal distribution. Defaults to 1.0.
    value : float, optional
        The constant value for constant initialization. Defaults to 0.0.
    """

    mode: DynamicEmbInitializerMode = DynamicEmbInitializerMode.UNIFORM
    mean: float = 0.0
    std_dev: float = 1.0
    lower: float = None
    upper: float = None
    value: float = 0.0

    def __eq__(self, other):
        if not isinstance(other, DynamicEmbInitializerArgs):
            return NotImplementedError
        if self.mode == DynamicEmbInitializerMode.NORMAL:
            return self.mean == other.mean and self.std_dev == other.std_dev
        elif self.mode == DynamicEmbInitializerMode.TRUNCATED_NORMAL:
            return (
                self.mean == other.mean
                and self.std_dev == other.std_dev
                and self.lower == other.lower
                and self.upper == other.upper
            )
        elif self.mode == DynamicEmbInitializerMode.UNIFORM:
            return self.lower == other.lower and self.upper == other.upper
        elif self.mode == DynamicEmbInitializerMode.CONSTANT:
            return self.value == other.value
        return True

    def __ne__(self, other):
        if not isinstance(other, DynamicEmbInitializerArgs):
            return NotImplementedError
        return not (self == other)


KEY_TYPE = torch.int64
EMBEDDING_TYPE = torch.float32
SCORE_TYPE = torch.int64
OPT_STATE_TYPE = torch.float32
COUNTER_TYPE = torch.int64

torch_dtype_to_np_dtype = {
    torch.uint64: np.uint64,
    torch.int64: np.int64,
    torch.float32: np.float32,
}


OptionsT = TypeVar("OptionsT")
OptimizerT = TypeVar("OptimizerT")


class CopyMode(enum.Enum):
    """Copy mode for load_from_flat / store_to_flat.

    EMBEDDING -- 1-region copy: copies only the embedding portion per row,
                 padded to max_emb_dim. Output: [N, max_emb_dim].
    VALUE     -- 2-region padded copy: emb padded to max_emb_dim, then opt
                 states padded to (max_value_dim - max_emb_dim).
                 Output: [N, max_value_dim].
                 values[:, :max_emb_dim] gives embeddings,
                 values[:, max_emb_dim:] gives optimizer states.
    """

    EMBEDDING = "embedding"
    VALUE = "value"


# make it standalone to avoid recursive references.
class Storage(abc.ABC, Generic[OptionsT, OptimizerT]):
    @abc.abstractmethod
    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        num_missing: int
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        missing_scores: torch.Tensor
        founds: torch.Tensor
        output_scores: torch.Tensor
        values: torch.Tensor
        return (
            num_missing,
            missing_keys,
            missing_indices,
            missing_scores,
            founds,
            output_scores,
            values,
        )

    @abc.abstractmethod
    def insert(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        preserve_existing: bool = False,
    ) -> None:
        pass

    @abc.abstractmethod
    def dump(
        self,
        table_id: int,
        meta_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        timestamp: int,
    ) -> None:
        pass

    @abc.abstractmethod
    def load(
        self,
        table_id: int,
        meta_file_path: str,
        emb_file_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool,
        timestamp: int,
    ) -> None:
        pass

    @abc.abstractmethod
    def embedding_dtype(
        self,
    ) -> torch.dtype:
        pass

    @abc.abstractmethod
    def embedding_dim(self, table_id: int) -> int:
        pass

    @abc.abstractmethod
    def value_dim(self, table_id: int) -> int:
        pass

    @abc.abstractmethod
    def max_embedding_dim(self) -> int:
        pass

    @abc.abstractmethod
    def max_value_dim(self) -> int:
        pass

    @abc.abstractmethod
    def init_optimizer_state(
        self,
    ) -> float:
        pass

    @abc.abstractmethod
    def export_keys_values(
        self,
        device: torch.device,
        batch_size: int = 65536,
        table_id: int = 0,
    ) -> Iterator[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]
    ]:
        pass


class Cache(abc.ABC):
    @abc.abstractmethod
    def lookup(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lookup with overflow fallback.

        Returns:
            score_out: Output scores.
            founds: Boolean tensor indicating which keys were found.
            indices: Slot indices (``-1`` for keys not found).
        """
        ...

    @abc.abstractmethod
    def insert_and_evict(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Insert with counter-aware eviction and overflow fallback.

        Returns:
            indices, num_evicted, evicted_keys, evicted_indices,
            evicted_scores, evicted_table_ids.
        """
        ...

    @abc.abstractmethod
    def reset(
        self,
    ) -> None:
        pass

    @abc.abstractmethod
    def cache_metrics(
        self,
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def set_record_cache_metrics(self, record: bool) -> None:
        pass


class Counter(abc.ABC):
    """
    Interface of a counter table which maps a key to a counter.
    Supports multi-table via table_ids parameter.
    """

    @abc.abstractmethod
    def add(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add keys with frequencies to the `Counter` and get accumulated counter of each key.

        Args:
            keys (torch.Tensor): The input keys, should be unique keys.
            table_ids (torch.Tensor): The table id for each key.
            frequencies (torch.Tensor): The input frequencies.

        Returns:
            accumulated_frequencies (torch.Tensor): the frequencies' state for the input keys.
        """
        ...

    @abc.abstractmethod
    def erase(self, keys: torch.Tensor, table_ids: torch.Tensor) -> None:
        """
        Erase keys from the `Counter`.

        Args:
            keys (torch.Tensor): The input keys to be erased.
            table_ids (torch.Tensor): The table id for each key.
        """

    @abc.abstractmethod
    def memory_usage(self, mem_type=MemoryType.DEVICE) -> int:
        """
        Get the consumption of a specific memory type.

        Args:
            mem_type (MemoryType): the specific memory type, default to MemoryType.DEVICE.
        """

    @abc.abstractmethod
    def load(self, key_file, counter_file, table_id: int) -> None:
        """
        Load keys and frequencies from input file path.

        Args:
            key_file (str): the file path of keys.
            counter_file (str): the file path of frequencies.
            table_id (int): the logical table to load into.
        """

    @abc.abstractmethod
    def dump(self, key_file, counter_file, table_id: int) -> None:
        """
        Dump keys and frequencies to output file path.

        Args:
            key_file (str): the file path of keys.
            counter_file (str): the file path of frequencies.
            table_id (int): the logical table to dump from.
        """


class AdmissionStrategy(abc.ABC):
    @abc.abstractmethod
    def admit(
        self,
        keys: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        """
        Admit keys with frequencies >= threshold.
        """

    @abc.abstractmethod
    def initialize_non_admitted_embeddings(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """
        Initialize the embeddings for the keys that are not admitted.
        """
