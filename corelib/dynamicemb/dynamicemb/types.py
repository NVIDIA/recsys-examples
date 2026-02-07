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
from typing import Generic, Optional, Tuple, TypeVar

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


TableOptionType = TypeVar("TableOptionType")
OptimizerInterface = TypeVar("OptimizerInterface")

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


# make it standalone to avoid recursive references.
class Storage(abc.ABC, Generic[TableOptionType, OptimizerInterface]):
    @abc.abstractmethod
    def __init__(
        self,
        options: TableOptionType,
        optimizer: OptimizerInterface,
    ):
        pass

    @abc.abstractmethod
    def find(
        self,
        unique_keys: torch.Tensor,
        unique_vals: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: torch.Tensor
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        missing_scores: torch.Tensor
        return num_missing, missing_keys, missing_indices, missing_scores

    @abc.abstractmethod
    def find_embeddings(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: int
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        missing_scores: torch.Tensor
        return num_missing, missing_keys, missing_indices, missing_scores

    @abc.abstractmethod
    def insert(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def update(
        self, keys: torch.Tensor, grads: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: torch.Tensor
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def enable_update(self) -> bool:
        ...

    @abc.abstractmethod
    def dump(
        self,
        meta_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
    ) -> None:
        pass

    @abc.abstractmethod
    def load(
        self,
        meta_file_path: str,
        emb_file_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool,
    ) -> None:
        pass

    @abc.abstractmethod
    def embedding_dtype(
        self,
    ) -> torch.dtype:
        pass

    @abc.abstractmethod
    def embedding_dim(
        self,
    ) -> int:
        pass

    @abc.abstractmethod
    def value_dim(
        self,
    ) -> int:
        pass

    @abc.abstractmethod
    def init_optimizer_state(
        self,
    ) -> float:
        pass


class Cache(abc.ABC):
    @abc.abstractmethod
    def find(
        self,
        unique_keys: torch.Tensor,
        unique_vals: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: int
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        missing_scores: torch.Tensor
        return num_missing, missing_keys, missing_indices, missing_scores

    @abc.abstractmethod
    def find_embeddings(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: int
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        missing_scores: torch.Tensor
        return num_missing, missing_keys, missing_indices, missing_scores

    @abc.abstractmethod
    def find_missed_keys(
        self,
        unique_keys: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: int
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        missing_scores: torch.Tensor
        return num_missing, missing_keys, missing_indices, missing_scores

    @abc.abstractmethod
    def insert_and_evict(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_evicted: int
        evicted_keys: torch.Tensor
        evicted_values: torch.Tensor
        evicted_scores: torch.Tensor
        return num_evicted, evicted_keys, evicted_values, evicted_scores

    @abc.abstractmethod
    def update(
        self, keys: torch.Tensor, grads: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        num_missing: int
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def flush(self, storage: Storage) -> None:
        pass

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
    """

    @abc.abstractmethod
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
        accumulated_frequencies: torch.Tensor
        return accumulated_frequencies

    @abc.abstractmethod
    def erase(self, keys) -> None:
        """
        Erase keys form the `Counter`.

        Args:
            keys (torch.Tensor): The input keys to be erased.
        """

    @abc.abstractmethod
    def memory_usage(self, mem_type=MemoryType.DEVICE) -> int:
        """
        Get the consumption of a specific memory type.

        Args:
            mem_type (MemoryType): the specific memory type, default to MemoryType.DEVICE.
        """

    @abc.abstractmethod
    def load(self, key_file, counter_file) -> None:
        """
        Load keys and frequencies from input file path.

        Args:
            key_file (str): the file path of keys.
            counter_file (str): the file path of frequencies.
        """

    @abc.abstractmethod
    def dump(self, key_file, counter_file) -> None:
        """
        Dump keys and frequencies to output file path.

        Args:
            key_file (str): the file path of keys.
            counter_file (str): the file path of frequencies.
        """

    @abc.abstractmethod
    def create(self, device: torch.device) -> "Counter":
        """
        Create the counter table on the specified device.
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
