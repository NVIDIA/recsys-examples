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
    """Where the rows of the tables live: their keys, scores and values.

    ``find`` and ``insert`` are what the lookup step of a batch (prefetch, forward and
    backward) calls. They take per-key arrays of a *capacity* the host knows -- ``N``,
    the number of input keys -- and ``num_valid``: a one-element int64 CUDA tensor, how
    many of the leading entries of those arrays are live. The number of unique keys of
    a batch is only known to the device, and the step does not read it back.
    ``num_valid=None`` means that every entry is live, an exact-length call, which is
    what the explicit APIs and tests make. An implementation

    * reads, writes and returns only the first ``num_valid`` entries of a per-key
      array. What lies past them is undefined -- it may be uninitialized memory, not a
      valid key or table id -- so it must never be read or used as an index;
    * returns its per-key results at the capacity of its input, and a count of its own
      (the ``num_missing`` of :meth:`find`) as a one-element int64 CUDA tensor, never
      as a Python int, also when ``num_valid`` is ``None``;
    * returns arrays that the counted ops of ``dynamicemb_extensions`` take
      (``flagged_compact``, ``counted_gather``, ``counted_scatter``,
      ``counted_scatter_fill``), and they check what they get: CUDA tensors on the
      device of its inputs, 1-D and contiguous (a 2-D one has contiguous rows), with
      8-byte elements for keys and scores, int64 for table ids and indices, bool for
      flags;
    * does not wait for the device: no ``.item()``, ``.cpu()`` or ``.tolist()`` of a
      device tensor, and no torch op whose output size depends on device data
      (``nonzero``, ``masked_select``, boolean-mask indexing, ``unique``). The counted
      ops above are what to build on instead.

    A store that cannot work from the device -- one that keeps its rows in host memory,
    say -- may read the count once, ``n = capacity if num_valid is None else
    int(num_valid.item())``. That read waits for the device: the step is not free of
    host syncs there, which is the cost of that store.

    :class:`Cache`, :class:`Counter` and :class:`MultiTableAdmitter` follow the same
    rules. The other methods take no arrays of a step: they are accessors, or explicit
    APIs (dump, load, export, ...) called with exact-length data, and may wait.
    """

    @abc.abstractmethod
    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
        key_scores: Optional[torch.Tensor] = None,
        *,
        num_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Look up the live keys: which of them the store holds, and their values.

        Args:
            unique_keys (torch.Tensor): ``[N]`` keys. A training step passes the unique
                keys of the batch, unique within each table; the evaluation of a table
                that is not pooled passes its raw ids, repeats included, so answer
                every entry, a repeat as any other.
            table_ids (torch.Tensor): ``[N]`` int64, the table each key belongs to.
            copy_mode (CopyMode): which columns of a row ``values`` holds: the
                embedding only (``EMBEDDING``) or the whole row, the embedding and its
                optimizer states (``VALUE``).
            lfu_accumulated_frequency (Optional[torch.Tensor]): ``[N]`` how many times
                the batch had each key, for a store that ranks its keys by frequency.
            key_scores (Optional[torch.Tensor]): ``[N]`` the score the input gave each
                key, for a store that ranks its keys by the score of the input;
                ``None`` for a table that takes no score per key.
            num_valid (Optional[torch.Tensor]): how many leading entries of the
                ``[N]`` arrays are live; ``None``: all of them.

        Returns:
            A tuple of eight. Of the ``[N]`` arrays only the first ``num_valid``
            entries are defined, and of the ``missing_*`` ones only the first
            ``num_missing``.

            - num_missing (torch.Tensor): one-element int64 CUDA tensor, how many of
              the live keys were not found.
            - missing_keys (torch.Tensor): ``[N]`` the keys that were not found, in the
              order of the input.
            - missing_indices (torch.Tensor): ``[N]`` int64, where those keys are in
              ``unique_keys``.
            - missing_table_ids (torch.Tensor): ``[N]`` int64, their table ids.
            - missing_scores (Optional[torch.Tensor]): ``[N]`` the score each of them
              is to be inserted with (what the store ranks by), or ``None`` when there
              is none for the caller to carry.
            - founds (torch.Tensor): ``[N]`` bool, True where the key was found.
            - output_scores (torch.Tensor): ``[N]`` int64, the score a found key has
              after this lookup.
            - values (torch.Tensor): ``[N, max_dim]`` of ``embedding_dtype()``, the rows
              of the keys that were found: ``max_dim`` is ``max_embedding_dim()`` for
              ``CopyMode.EMBEDDING`` and ``max_value_dim()`` for ``CopyMode.VALUE``,
              and the columns are laid out as :class:`CopyMode` says, each region
              padded whatever a table's own ``embedding_dim``. The row of a key that
              was not found is not written. The step writes into this tensor (it
              initializes the rows of the new keys, resets their optimizer states and,
              in the backward, applies the gradients to the rows), so it has to be a
              tensor of its own, not a view of the store's memory.
        """

    @abc.abstractmethod
    def insert(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        preserve_existing: bool = False,
        *,
        num_valid: Optional[torch.Tensor] = None,
    ) -> None:
        """Insert the live keys with their values, or refresh the rows they have.

        Args:
            keys (torch.Tensor): ``[N]`` keys, unique within each table.
            table_ids (torch.Tensor): ``[N]`` int64, the table each key belongs to.
            values (torch.Tensor): ``[N, max_value_dim]`` the whole row of each key,
                its embedding and its optimizer states, laid out as
                ``CopyMode.VALUE`` says. It is the caller's, and its rows past
                ``num_valid`` are undefined: copy the rows you keep (a view would keep
                the whole array alive).
            scores (Optional[torch.Tensor]): ``[N]`` the score to insert each key with,
                as :meth:`find` handed it out in ``missing_scores``; ``None``: the
                store's own default.
            preserve_existing (bool): the keys are ones the store already holds and
                this writes their updated rows back: it keeps the scores they have
                instead of assigning new ones. A key it does not hold may be ignored.
            num_valid (Optional[torch.Tensor]): how many leading entries of the
                ``[N]`` arrays are live; ``None``: all of them.
        """

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
        shard_index: Optional[int] = None,
        num_shards: Optional[int] = None,
    ) -> None:
        """Read one checkpoint shard of *table_id* back into this store.

        Args:
            shard_index (Optional[int]): this shard's position among the ranks
                that hold the table, and *num_shards* how many there are: the
                world for a row-wise table, one node for a table-row-wise one.
                They are the table's own geometry rather than the global rank
                and world size, which a table-row-wise table does not share.
                An implementation that re-splits a checkpoint across ranks --
                keeping the keys this shard owns and dropping the rest --
                filters by these. Left ``None`` they fall back to the global
                rank and world size, which is the row-wise answer, so a store
                that only ever serves row-wise tables may ignore them.
            num_shards (Optional[int]): see *shard_index*.
        """

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
    """A cache of rows in front of a :class:`Storage`, addressed by slot.

    The module builds its cache itself, a ``DynamicEmbCache``, and another one cannot
    be plugged in: the prefetch also uses what only that class has, its
    ``increment_counter`` and ``decrement_counter``, which pin and release a slot, and
    its table state. This interface is the contract of the two calls that take the
    arrays of a step. ``lookup`` and ``insert_and_evict`` follow the rules of
    :class:`Storage` for them: per-key arrays of a capacity ``N`` and ``num_valid``,
    the number of live leading entries (``None``: all of them); results at the
    capacity of the input, counts as one-element int64 CUDA tensors; no wait for the
    device (recording the cache metrics is the exception, see ``cache_metrics``).
    """

    @abc.abstractmethod
    def lookup(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
        *,
        num_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lookup with overflow fallback.

        Args:
            unique_keys (torch.Tensor): ``[N]`` keys. A training step passes the unique
                keys of the batch, unique within each table; the evaluation of a table
                that is not pooled passes its raw ids, repeats included, so answer
                every entry, a repeat as any other.
            table_ids (torch.Tensor): ``[N]`` int64, the table each key belongs to.
            lfu_accumulated_frequency (Optional[torch.Tensor]): ``[N]`` how many times
                the batch had each key.
            num_valid (Optional[torch.Tensor]): how many leading entries of the
                ``[N]`` arrays are live; ``None``: all of them.

        Returns:
            Three ``[N]`` arrays, of which only the first ``num_valid`` entries are
            defined.

            - score_out (torch.Tensor): int64, output scores.
            - founds (torch.Tensor): bool, True for the keys that were found.
            - indices (torch.Tensor): int64, slot indices (``-1`` for keys not found).
        """

    @abc.abstractmethod
    def insert_and_evict(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        *,
        num_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Insert with counter-aware eviction and overflow fallback.

        Args:
            keys (torch.Tensor): ``[N]`` keys, unique within each table.
            table_ids (torch.Tensor): ``[N]`` int64, the table each key belongs to.
            scores (Optional[torch.Tensor]): ``[N]`` the score to insert each key with;
                ``None``: the cache's own ranking.
            num_valid (Optional[torch.Tensor]): how many leading entries of the
                ``[N]`` arrays are live; ``None``: all of them.

        Returns:
            A tuple of six. Of ``indices`` only the first ``num_valid`` entries are
            defined, and of the ``evicted_*`` arrays only the first ``num_evicted``.

            - indices (torch.Tensor): ``[N]`` int64, the slot each key has now
              (``-1`` for a key that could not be placed).
            - num_evicted (torch.Tensor): one-element int64 CUDA tensor, how many
              entries the ``evicted_*`` arrays hold.
            - evicted_keys (torch.Tensor): ``[N]`` the keys that left the cache to
              make room, and any key that could not be placed at all (as itself). In
              no defined order, not that of the input: entry ``j`` of the four
              ``evicted_*`` arrays describes one key.
            - evicted_indices (torch.Tensor): ``[N]`` int64, the slot each of them
              held: the row to write back before the new key's value overwrites it.
              Negative for a key that could not be placed, which has no row.
            - evicted_scores (torch.Tensor): ``[N]`` int64, their scores.
            - evicted_table_ids (torch.Tensor): ``[N]`` int64, their table ids.
        """

    @abc.abstractmethod
    def reset(
        self,
    ) -> None:
        pass

    @property
    @abc.abstractmethod
    def cache_metrics(self) -> Optional[torch.Tensor]:
        """The numbers of the last lookup and the last insert, in a host tensor: the
        keys of the lookup, how many of them were found, the keys of the insert and
        how many it evicted. ``None`` while they are not recorded (see
        :meth:`set_record_cache_metrics`). Recording them is what makes ``lookup``
        and ``insert_and_evict`` read the device; this property does not."""

    @abc.abstractmethod
    def set_record_cache_metrics(self, record: bool) -> None:
        pass


class Counter(abc.ABC):
    """
    Interface of a counter table which maps a key to a counter.
    Supports multi-table via table_ids parameter.

    ``add`` and ``erase`` are called with the arrays of a step: per-key arrays of a
    capacity ``N`` and ``num_valid``, the number of live leading entries, under the
    rules of :class:`Storage` (what lies past ``num_valid`` is undefined and never
    read; no wait for the device).
    """

    @abc.abstractmethod
    def add(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        frequencies: torch.Tensor,
        *,
        num_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add keys with frequencies to the `Counter` and get accumulated counter of each key.

        Args:
            keys (torch.Tensor): The input keys, ``[N]``, should be unique keys.
            table_ids (torch.Tensor): The table id for each key, ``[N]``.
            frequencies (torch.Tensor): The input frequencies, ``[N]``.
            num_valid (Optional[torch.Tensor]): One-element int64 CUDA tensor, how
                many leading entries of the three arrays are live; ``None``: all.

        Returns:
            accumulated_frequencies (torch.Tensor): the frequencies' state for the
                input keys, ``[N]`` int64; the entries past ``num_valid`` are undefined.
        """
        ...

    @abc.abstractmethod
    def erase(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        *,
        num_valid: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Erase keys from the `Counter`.

        Args:
            keys (torch.Tensor): The input keys to be erased, ``[N]``.
            table_ids (torch.Tensor): The table id for each key, ``[N]``.
            num_valid (Optional[torch.Tensor]): One-element int64 CUDA tensor, how
                many leading entries of ``keys`` and ``table_ids`` are live; the others
                are not erased and not read. ``None``: all.
        """

    @abc.abstractmethod
    def memory_usage(self, mem_type=MemoryType.DEVICE) -> int:
        """
        Get the consumption of a specific memory type.

        Args:
            mem_type (MemoryType): the specific memory type, default to MemoryType.DEVICE.
        """

    @abc.abstractmethod
    def load(
        self,
        key_file,
        counter_file,
        table_id: int,
        dist_type: str,
        shard_index: Optional[int] = None,
        num_shards: Optional[int] = None,
    ) -> None:
        """
        Load keys and frequencies from input file path.

        Args:
            key_file (str): the file path of keys.
            counter_file (str): the file path of frequencies.
            table_id (int): the logical table to load into.
            dist_type (str): the key -> rank rule the counter's table is
                sharded by, so an implementation keeps only the keys this rank
                owns. Required rather than optional: a counter belongs to a
                sharded table, so its rule is always known, and an
                implementation that guesses one drops the keys it guesses
                wrong about without saying so.
            shard_index (Optional[int]): this shard's position among the ranks
                that hold the counter's table, and *num_shards* how many there
                are. Same meaning as on :meth:`Storage.load`, including the
                fall back to the global rank and world size when ``None``.
            num_shards (Optional[int]): see *shard_index*.
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
        *,
        num_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Which of these missing keys may enter the table.

        The arrays are those of a step: per-key arrays of a capacity ``N`` and
        ``num_valid``, the number of live leading entries, under the rules of
        :class:`Storage`.

        Args:
            keys (torch.Tensor): The keys to decide on, deduplicated, ``[N]``.
            table_ids (torch.Tensor): The table each key belongs to, ``[N]``. One
                module spans several, so a decision may be made per table.
            frequencies (Optional[torch.Tensor]): How often each key occurred in
                *this batch*, ``[N]``, where the module counts occurrences at all;
                None means treat each key as one occurrence. An increment, not a
                running total -- any total is the admitter's own to keep.
            num_valid (Optional[torch.Tensor]): One-element int64 CUDA tensor, how
                many leading entries of the arrays are live; ``None``: all of them.
                What lies past them is undefined: it must not be read or used as an
                index, and the call must not wait for the device.

        Returns:
            torch.Tensor: Boolean mask over `keys`, ``[N]``, True where admitted. Only
            its first ``num_valid`` entries are read; the others may hold anything.
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

        It is called with the arrays of a step, under the rules of :class:`Storage`:
        ``initializer(buffer, keys, table_ids, indices, num_valid=count)``, where
        ``indices`` selects the rows to write, at the capacity of the batch, and only
        its first ``count`` entries are live. An initializer of your own has to take
        ``num_valid`` and leave the other entries of ``indices`` alone, as the
        ``__call__`` of ``MultiTableInitializer`` (``dynamicemb/initializer.py``)
        says.
        """
        return None
