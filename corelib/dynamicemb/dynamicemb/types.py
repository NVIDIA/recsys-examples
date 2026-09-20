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
from typing import Generic, Iterator, List, Optional, Tuple, TypeVar

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


# What an unbounded UNIFORM falls back to when there is no table to derive
# bounds from. A table's own resolve to +/-sqrt(1 / num_embeddings) in the
# planner; everywhere else -- a strategy's initializer, or an args object that
# never reached the planner -- these are what it gets.
DEFAULT_UNIFORM_LOWER: float = 0.0
DEFAULT_UNIFORM_UPPER: float = 1.0


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

    def get_grouped_key(self):
        """What has to match for two tables to share one initializer.

        Only the mode. The parameters are resolved per table by
        :class:`~dynamicemb.initializer.MultiTableInitializer`, so tables that
        differ in them still fuse -- and they differ without being asked to,
        since an unbounded UNIFORM resolves against each table's row count.
        """
        return self.mode

    def __eq__(self, other):
        # Only the fields the mode actually reads take part: a CONSTANT's mean
        # or a NORMAL's bounds are never used, so two args that differ there
        # still initialize identically.
        if not isinstance(other, DynamicEmbInitializerArgs):
            return NotImplemented
        if self.mode != other.mode:
            return False
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
        # DEBUG takes no parameters, so same mode is all there is to compare.
        return True


def group_key_of(obj: Optional[object]):
    """Ask an object what has to match for its tables to be fused.

    Objects that answer decide their own granularity, which is how a strategy,
    a counter or an initializer lets tables differ in whatever it resolves per
    table. Anything else stands for itself, so two tables group only when
    handed the very same object -- conservative, and what happened before any
    of them had a say.
    """
    if obj is None:
        return None
    get_grouped_key = getattr(obj, "get_grouped_key", None)
    return get_grouped_key() if callable(get_grouped_key) else obj


KEY_TYPE = torch.int64
SCORE_TYPE = torch.int64
# Checkpoint value precision for a checkpoint whose meta JSON does not say --
# i.e. every one written before the dump recorded ``embedding_dtype`` /
# ``optim_state_dtype``, which were all fp32. A dump now writes the table's own
# ``emb_dtype``; these are the compatibility fallback, not the format.
EMBEDDING_TYPE = torch.float32
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


@dataclass
class ReplayStats:
    """What a ``replay_increment`` did to one table.

    Replay is all-or-nothing for the keys this rank owns: each of them is written
    back at the slot and value row it occupied in the source table, or the call
    raises. ``skipped`` counts the rest of the delta -- the keys another rank
    owns, which this rank deliberately ignores.

    Attributes
    ----------
    upserted : int
        Keys written back at their source slot.
    erased : int
        Keys removed before the upsert, counted as *actually removed* -- a delta
        can name keys this replica never held. Counts only the delta's
        ``erased_keys``; an eviction is reproduced by overwriting its slot and
        costs no removal.
    skipped : int
        Delta keys this rank does not own (row-wise sharding) and so ignored.
        Always 0 on a single rank, where the whole delta is this rank's.
    """

    upserted: int = 0
    erased: int = 0
    skipped: int = 0

    def merge(self, other: "ReplayStats") -> "ReplayStats":
        """Accumulate *other* into self (used to fold per-batch / per-tier runs)."""
        self.upserted += other.upserted
        self.erased += other.erased
        self.skipped += other.skipped
        return self


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
        """The scalar every optimizer-state element starts at.

        For callers that need the value itself. To initialize a buffer, go
        through the optimizer's ``reset_optimizer_states`` instead: a single
        scalar cannot describe an optimizer whose state regions start at
        different values.
        """
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
    """How a table's admission is configured.

    This is inert: it allocates nothing, touches no device, and may be handed
    to as many tables as you like. It does not decide anything either -- a
    module turns the configurations of the tables it fuses into the one thing
    that does, by calling :meth:`create_admitter`.
    """

    @classmethod
    @abc.abstractmethod
    def create_admitter(
        cls,
        table_strategies: List["AdmissionStrategy"],
        device: torch.device,
    ) -> "MultiTableAdmitter":
        """The admitter these tables share, with its device state allocated.

        Called once, by the module, with one configuration per table it fuses.
        This is where anything on the device comes into being: a counter's hash
        table, an initializer's per-table parameters.
        """

    @classmethod
    def one_configuration(
        cls, table_strategies: List["AdmissionStrategy"]
    ) -> "AdmissionStrategy":
        """The single configuration these tables agree on.

        They were grouped on :meth:`get_grouped_key`, so they are already
        interchangeable and the first stands for all; this says so rather than
        take element zero and leave the reader to wonder.
        """
        keys = {strategy.get_grouped_key() for strategy in table_strategies}
        if len(keys) != 1:
            raise ValueError(
                f"Tables of one module must agree on their admission strategy, "
                f"got {len(keys)} different ones: {keys}"
            )
        return table_strategies[0]

    def get_grouped_key(self):
        """What has to match for two tables to share one fused module.

        The tables of a fused module share a single admitter, so this decides
        which configurations are interchangeable. The default is the instance
        itself: only the very same object groups, which is how strategies
        behaved before they had a say. Override it to let equal but separately
        constructed configurations share a module -- return everything that
        changes the decision, and for anything resolved per table return only
        what the tables must agree on.
        """
        return id(self)


class MultiTableAdmitter(abc.ABC):
    """What a fused module runs: one admitter over all of its tables.

    Built by :meth:`AdmissionStrategy.create_admitter`, and the only half that
    holds anything -- whatever deciding takes, it owns.
    """

    @abc.abstractmethod
    def admit(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        frequencies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Which of these missing keys may enter the table.

        Args:
            keys (torch.Tensor): The keys to decide on, deduplicated.
            table_ids (torch.Tensor): The table each key belongs to. One module
                spans several, so a decision may be made per table.
            frequencies (Optional[torch.Tensor]): How often each key occurred in
                *this batch*, where the module counts occurrences at all; None
                means treat each key as one occurrence. An increment, not a
                running total -- any total is the admitter's own to keep.

        Returns:
            torch.Tensor: Boolean mask over `keys`, True where admitted.
        """

    def state(self) -> Optional[Counter]:
        """Persistent state the framework has to carry, or None.

        Whatever an admitter accumulates across steps lives on the device, has
        to be reported in memory accounting, and has to survive a checkpoint.
        Return it here and the framework does all three; keeping it private
        would only mean it is none of those.
        """
        return None

    @property
    def non_admitted_initializer(self):
        """What writes the rows this admitter rejects, or None for the table's.

        A rejected key still takes part in the forward, so its row has to be
        written by something. Returning a ``MultiTableInitializer`` here hands
        that job to it; None leaves those rows to the table's own initializer.
        The module does the calling, so buffer layout stays out of an
        admitter's business, and which of the two writes them is settled once,
        when the admitter is built, and not renegotiated on every batch.
        """
        return None
