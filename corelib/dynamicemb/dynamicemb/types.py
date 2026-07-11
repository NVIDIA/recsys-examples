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
from typing import TYPE_CHECKING, Any, Generic, Iterator, List, Optional, Tuple, TypeVar

import numpy as np
import torch

if TYPE_CHECKING:
    from dynamicemb.dynamicemb_config import DynamicEmbTableOptions
    from dynamicemb.extendable_tensor import ExtendableBuffer
    from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
    from dynamicemb.scored_hashtable import ScoreSpec
    from dynamicemb_extensions import EvictStrategy


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
DEMB_TABLE_ALIGN_SIZE = 16

# Per-bucket row alignment for hashtable backends (same as :data:`DEMB_TABLE_ALIGN_SIZE`).
BUCKET_ALIGNMENT: int = DEMB_TABLE_ALIGN_SIZE

# Sentinel ``bucket_capacity``: treat the whole per-rank table as one bucket; see
# :func:`dynamicemb.dynamicemb_config.get_sharded_table_capacity` (per-rank row count).
MAX_BUCKET_CAPACITY: int = 2**63 - 1

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


# ---------------------------------------------------------------------------
# DynamicEmbTableState – shared state dataclass
# ---------------------------------------------------------------------------


@dataclass
class DynamicEmbTableState:
    options_list: List["DynamicEmbTableOptions"]
    num_tables: int
    device: torch.device
    score_policy: "ScoreSpec"
    evict_strategy: "EvictStrategy"
    key_index_map: Any
    capacity: int
    tables: List["ExtendableBuffer"]
    # Per-table value buffer base pointers on ``device``; refreshed on init and expand.
    table_ptrs_dev: torch.Tensor
    table_emb_dims: torch.Tensor
    # Persistent host tensor counterpart. ``table_emb_dims_cpu`` remains the
    # Python integer list used for individual dimension lookups.
    table_emb_dims_host: torch.Tensor
    table_value_dims: torch.Tensor
    table_emb_dims_cpu: List[int]
    table_value_dims_cpu: List[int]
    max_emb_dim: int
    emb_dim: int
    value_dim: int
    emb_dtype: torch.dtype
    all_dims_vec4: bool
    optimizer: "BaseDynamicEmbeddingOptimizer"
    initial_optim_state: float
    threads_in_wave: int
    # Computed from device properties during state construction; avoids a
    # device-property query in the value-exchange hot path.
    exchange_target_grid_size: int
    score: Optional[int] = None
    training: bool = False
    # Overflow region fields (per-table, only set when overflow is enabled)
    overflow_caps: Optional[List[int]] = None
    # NO_EVICTION: per-table auto-increment index used as insert score (internal only).
    # no_eviction_next_index: CPU pinned tensor (num_tables,); no_eviction_next_index_dev: same on state.device.
    no_eviction_next_index: Optional[torch.Tensor] = None
    no_eviction_next_index_dev: Optional[torch.Tensor] = None
    # Estimated per-table size (last_collected + accumulated unique since collection);
    # CPU tensor of shape (num_tables,), used to avoid key_index_map.size() when not needed.
    estimated_table_sizes: Optional[torch.Tensor] = None
    collect_table_sizes_flag: bool = False


@dataclass
class CacheFindOrInsertResult:
    """Device-resident metadata produced by fused cache find-or-insert.

    Every eviction tensor is aligned with the input keys. ``evicted_mask``
    selects positions whose provisional cache insertion displaced an old row;
    values at unselected positions are unspecified.
    """

    indices: torch.Tensor
    founds: torch.Tensor
    evicted_keys: torch.Tensor
    evicted_indices: torch.Tensor
    evicted_scores: torch.Tensor
    evicted_table_ids: torch.Tensor
    evicted_mask: torch.Tensor


@dataclass
class CacheExchangeRequest:
    """Full-layout cache/storage exchange request.

    ``cache_state`` is the shared, concretely typed cache value-buffer
    descriptor used by built-in and external storage implementations. The
    eviction tensors use the same layout as ``keys`` and are selected by
    ``evicted_mask``. The request must be consumed on the same CUDA stream as
    the cache ``find_or_insert`` that produced it; published provisional rows
    are not a cross-stream readiness signal.
    """

    cache_state: DynamicEmbTableState
    keys: torch.Tensor
    table_ids: torch.Tensor
    scores: Optional[torch.Tensor]
    cache_founds: torch.Tensor
    cache_indices: torch.Tensor
    evicted_keys: torch.Tensor
    evicted_indices: torch.Tensor
    evicted_scores: torch.Tensor
    evicted_table_ids: torch.Tensor
    evicted_mask: torch.Tensor


@dataclass
class CacheExchangeResult:
    """Full-layout storage lookup metadata after exchange.

    ``direct_storage_rows`` and ``direct_storage_slots`` are ``-1`` for rows
    materialized in cache. A non-negative pair identifies a storage hit whose
    provisional cache placement was rolled back after backing storage could
    not preserve its eviction victim. The storage reference remains acquired
    so forward/backward can consume that row directly without a host sync or
    an embedding staging buffer. ``direct_storage_table_ptrs`` names the
    backing value buffers for those physical rows and must have the same table
    layout and physical row widths as ``CacheExchangeRequest.cache_state``.
    """

    founds: torch.Tensor
    storage_founds: torch.Tensor
    storage_scores: torch.Tensor
    direct_storage_rows: torch.Tensor
    direct_storage_slots: torch.Tensor
    direct_storage_table_ptrs: torch.Tensor


# make it standalone to avoid recursive references.
class Storage(abc.ABC, Generic[OptionsT, OptimizerT]):
    def exchange(self, request: CacheExchangeRequest) -> CacheExchangeResult:
        """Exchange cache misses and displaced rows with backing storage.

        Every storage implementation configured behind a cache must override
        this method. There is deliberately no orchestration fallback to
        ``find`` plus ``insert`` in the prefetch path. If cache placement can
        fail, the backend must return addressable direct rows and retain their
        ownership until :meth:`release_cache_exchange_refs` is called.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement exchange() for caching mode"
        )

    def release_cache_exchange_refs(
        self,
        cache: "Cache",
        cache_slots: torch.Tensor,
        storage_slots: torch.Tensor,
        table_ids: torch.Tensor,
    ) -> None:
        """Release aligned cache/direct-storage refs after optimizer update.

        Backends configured behind a cache must implement this together with
        :meth:`exchange`. Built-in storage releases both layouts in one kernel;
        opaque backends may use their own reference layout.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            "release_cache_exchange_refs() for caching mode"
        )

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
    def embedding_dims(self, on_device: bool = False) -> torch.Tensor:
        """Per-table embedding dimensions, indexed by table id.

        Returns an int64 tensor of shape ``(num_tables,)``. When ``on_device`` is
        True the tensor lives on the storage's CUDA device (e.g. to pass to a
        kernel); otherwise it is on CPU. For multi-tier storage the dims come
        from the tier that produces the value buffer in :meth:`find`.
        """

    @abc.abstractmethod
    def all_dims_vec4(self) -> bool:
        """Whether every per-table embedding dim and value dim is a multiple of 4.

        When True the vectorized (vec4) optimizer / load / store kernels are safe
        for all rows; otherwise some per-table dim is misaligned and the scalar
        kernels must be used, so a vec4 store does not run past a row's
        optimizer state and corrupt the next state region.
        """

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
    def decrement_counter(
        self,
        slot_indices: torch.Tensor,
        table_ids: torch.Tensor,
        *,
        fallback_slot_indices: Optional[torch.Tensor] = None,
        fallback_storage: Optional[Storage] = None,
    ) -> None:
        """Release aligned cache refs and optional storage fallbacks."""
        ...

    @abc.abstractmethod
    def find_or_insert(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> CacheFindOrInsertResult:
        """Find existing keys or provision cache slots in one operation."""
        ...

    @abc.abstractmethod
    def reclaim(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        slot_indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Release provisional slots selected by ``mask`` for immediate reuse."""
        ...

    @abc.abstractmethod
    def update_scores(
        self,
        keys: torch.Tensor,
        slot_indices: torch.Tensor,
        table_ids: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Replace scores at known slots still owned by the expected keys."""
        ...

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
        founds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add missing keys with frequencies to the `Counter` and get accumulated
        counters in the original input layout.

        Args:
            keys (torch.Tensor): The input keys, should be unique keys.
            table_ids (torch.Tensor): The table id for each key.
            frequencies (torch.Tensor): The input frequencies.
            founds (torch.Tensor): Full-length boolean mask. Positions where
                ``founds`` is true are skipped.

        Returns:
            accumulated_frequencies (torch.Tensor): Full-length accumulated
                frequencies. Skipped positions are zero.
        """
        ...

    @abc.abstractmethod
    def erase(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """
        Erase selected keys from the `Counter` without compacting the inputs.

        Args:
            keys (torch.Tensor): The input keys to be erased.
            table_ids (torch.Tensor): The table id for each key.
            mask (torch.Tensor): Full-length boolean mask. Positions where
                ``mask`` is true are erased.
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
        founds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return a full-length admission mask for positions not already found.
        """

    @abc.abstractmethod
    def initialize_non_admitted_embeddings(
        self,
        buffer: torch.Tensor,
        mask: torch.Tensor,
    ) -> bool:
        """
        Initialize rows selected by a full-length boolean mask and return
        whether an initializer was configured.
        """
