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
import warnings
from dataclasses import dataclass
from typing import Any, Generic, Iterator, List, NamedTuple, Optional, Tuple, TypeVar

import numpy as np
import torch
from dynamicemb import sync_free as sf


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


class MaskedFind(NamedTuple):
    """What a storage's ``lookup_masked`` found, for the ``load_masked`` that follows.

    Nothing is compacted and no count is read back: every field is as long as
    the keys that were looked up, and a lane whose key is the reserved all-ones
    key (``sync_free.mask_keys``) is not found and owns no row, so a caller leaves a key
    out of a lookup, and out of the load that follows it, by masking the key.
    """

    founds: torch.Tensor  # bool[N]
    # The full-length score tensor the lookups used (``None`` for an LRU table and
    # for a const lookup). A later insert, or the lookup of a second tier, takes
    # it as is: the kernels read ``score_input[i]`` for every lane, masked ones
    # included, so it must never be compacted.
    score_in: Optional[torch.Tensor]
    scores: Optional[torch.Tensor]  # int64[N]; None for a storage that keeps none
    # Opaque token for ``load_masked``: the value rows (a tensor, ``-1`` where
    # there is none), or one such tensor per tier for a ``HybridStorage``.
    rows: Any


class PaddedFind(NamedTuple):
    """What :meth:`Storage.find_masked` returns."""

    founds: torch.Tensor  # bool[N]; False for every lane that is not a live key
    # ``[N, width]``, ``width`` that of the ``CopyMode``; the rows of the lanes that
    # were not found are unspecified.
    values: torch.Tensor


def has_padded_contract(obj) -> bool:
    """Whether the class of ``obj`` (or ``obj`` itself, a class) declared
    ``accepts_empty_keys = True`` in its own body.

    An inherited declaration does not count: a subclass that overrides anything the
    padded code relies on has not made the promise, so it has to make it itself.
    """
    cls = obj if isinstance(obj, type) else type(obj)
    return bool(cls.__dict__.get("accepts_empty_keys", False))


def as_padded(storage):
    """What the padded lookup code talks to: ``storage`` itself when its class
    declared the padded contract, else an :class:`OldContractStorage` over it.

    Works for any object that has the old ``find`` / ``insert``, whether it
    inherits :class:`Storage` or not.
    """
    return storage if has_padded_contract(storage) else OldContractStorage(storage)


def _check_padded_find(
    storage, keys: torch.Tensor, found: PaddedFind, width: int
) -> None:
    """The host facts of what a padded ``find_masked`` returned, none of which needs a read
    of the device: a result that does not have them would be lined up with the wrong rows,
    silently, by the code that loads it."""
    name = type(storage).__name__
    founds, values = found
    n = keys.numel()
    if not (
        isinstance(founds, torch.Tensor)
        and founds.dtype == torch.bool
        and tuple(founds.shape) == (n,)
        and founds.device == keys.device
    ):
        raise ValueError(
            f"{name}.find_masked must return founds as a bool tensor with one entry per key "
            f"([{n}], on {keys.device}), got {_describe(founds)}."
        )
    if not (
        isinstance(values, torch.Tensor)
        and values.dim() == 2
        and values.size(0) == n
        and values.size(1) >= width
        and values.device == keys.device
    ):
        raise ValueError(
            f"{name}.find_masked must return values as a tensor of [{n}, at least {width}] "
            f"on {keys.device}, got {_describe(values)}."
        )


def _describe(t) -> str:
    if not isinstance(t, torch.Tensor):
        return type(t).__name__
    return f"{t.dtype} {list(t.shape)} on {t.device}"


def _live_rows(keys: torch.Tensor, *aligned: Optional[torch.Tensor], count=None):
    """``(count, positions, [dense_keys, *dense_aligned])`` for the live rows of ``keys``.

    THE read of every old-contract adapter: a class whose methods take exact-length
    tensors needs the number of live rows on the host, and that number exists only on
    the device. It is one ``sync_free.read_now`` per call, placed after everything that
    fed the arrays was enqueued; a caller that has the count already hands it in
    (``count``) and nothing is read. ``positions`` (int64, ascending) says where each
    dense row sits among the ``N`` rows; ``aligned`` arrays may be None (they stay
    None) or 2-D (rows are selected). A call with no live row returns ``(0, None, [])``.
    """
    if keys.numel() == 0:
        return 0, None, []
    live = sf.valid_keys(keys)
    if count is None:
        count = int(sf.read_now(live.sum())[0])
    if count == 0:
        return 0, None, []
    rows = torch.arange(keys.numel(), dtype=torch.int64, device=keys.device)
    positions, dense_keys, *dense = sf.compact_by_count(
        live, count, rows, keys, *aligned
    )
    return count, positions, [dense_keys, *dense]


def _scatter_rows(
    size: int, positions: torch.Tensor, where: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    """A ``[size]`` tensor of ``rows``' dtype, zero but at ``positions[where]``, where it
    holds ``rows``. ``where`` indexes the dense array that ``positions`` maps to the
    ``size`` rows."""
    out = torch.zeros(size, dtype=rows.dtype, device=positions.device)
    index = positions[where.to(device=positions.device, dtype=torch.int64)]
    rows = rows.to(positions.device)
    if out.dtype == torch.uint64:  # no advanced indexing for uint64
        out.view(torch.int64)[index] = rows.view(torch.int64)
    else:
        out[index] = rows
    return out


class _DenseFind(NamedTuple):
    """The opaque token of an old-contract lookup (``MaskedFind.rows``)."""

    positions: Optional[torch.Tensor]  # where the dense rows sit among the N rows
    found: Optional[torch.Tensor]  # bool[count]
    values: Optional[torch.Tensor]  # [count, width]
    # The host count of missing keys the class returned from ``find`` (a host int by
    # its contract); 0 when no row was live, so nothing was asked.
    num_missing: int = 0


class OldContractStorage:
    """A view of a storage whose class did not declare ``accepts_empty_keys``: the
    padded methods run the class's old, host-sized ``find`` / ``insert`` on the live
    rows after one read of how many there are, and scatter the answer back.

    That read is by the old contract (see :class:`Storage`): a method that takes
    tensors of exactly the length of the live rows needs the length on the host. A
    class that implements the padded contract is never wrapped and costs nothing.
    Every other attribute is forwarded to the storage, so ``embedding_dims``,
    ``max_value_dim``, ``_state`` ... work on the view.
    """

    accepts_empty_keys = True  # the view itself speaks the padded contract

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):  # only for what the view does not define
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def lookup_masked(
        self,
        unique_keys,
        table_ids,
        lfu_accumulated_frequency=None,
        key_scores=None,
        copy_mode=None,
        *,
        live_count=None,
    ) -> MaskedFind:
        """``find`` of the live rows, as a :class:`MaskedFind`. ``live_count`` is the
        number of live keys when the caller knows it (nothing is then read)."""
        copy_mode = CopyMode.VALUE if copy_mode is None else copy_mode
        device = unique_keys.device
        found = torch.zeros(unique_keys.numel(), dtype=torch.bool, device=device)
        count, positions, dense = _live_rows(
            unique_keys,
            table_ids,
            lfu_accumulated_frequency,
            key_scores,
            count=live_count,
        )
        token, score_in = _DenseFind(None, None, None, 0), None
        if count:
            d_keys, d_ids, d_freq, d_scores = dense
            # Only what there is: a store written before per-key scores has no such
            # parameter, and the legacy calls did not pass an empty frequency either.
            kwargs = {}
            if d_freq is not None:
                kwargs["lfu_accumulated_frequency"] = d_freq
            if d_scores is not None:
                kwargs["key_scores"] = d_scores
            result = self.inner.find(d_keys, d_ids, copy_mode=copy_mode, **kwargs)
            d_found = result[5].to(device=device, dtype=torch.bool)
            found[positions] = d_found  # CUDA index, CUDA value: no sync
            token = _DenseFind(positions, d_found, result[7].to(device), int(result[0]))
            # What the class says a later insert of its missing keys takes as scores:
            # its own ``missing_scores``, N-aligned (None when it returned none).
            if result[4] is not None and result[4].numel() > 0:
                score_in = _scatter_rows(
                    unique_keys.numel(), positions, result[2], result[4]
                )
        return MaskedFind(founds=found, score_in=score_in, scores=None, rows=token)

    def nothing_missed(self, found: MaskedFind) -> bool:
        """Whether the lookup that made ``found`` is known to have found every live key:
        the class returned a host count of the keys it lacked (zero here), so a caller
        may skip what would only handle the misses without reading anything."""
        return found.rows.num_missing == 0

    def load_masked(self, found, table_ids, copy_mode, output=None):
        """Write the rows ``found`` fetched into ``output`` (``[N, width]``; the lanes
        that were not found keep what they hold) or into a new buffer."""
        token = found.rows
        width = (
            self.inner.max_embedding_dim()
            if copy_mode == CopyMode.EMBEDDING
            else self.inner.max_value_dim()
        )
        if output is None:
            output = torch.empty(
                found.founds.numel(),
                width,
                dtype=self.inner.embedding_dtype(),
                device=found.founds.device,
            )
        if token.positions is not None:  # the token was built from a count read
            assert token.values.size(1) >= width, (
                "load_masked asks for more columns than the lookup fetched: "
                f"{width} > {token.values.size(1)}"
            )
            current = output[token.positions]
            output[token.positions] = torch.where(
                token.found.unsqueeze(1),
                token.values[:, :width].to(output.dtype),
                current,
            )
        return output

    def find_masked(
        self,
        keys,
        table_ids,
        copy_mode,
        lfu_accumulated_frequency=None,
        key_scores=None,
    ) -> PaddedFind:
        found = self.lookup_masked(
            keys, table_ids, lfu_accumulated_frequency, key_scores, copy_mode
        )
        return PaddedFind(found.founds, self.load_masked(found, table_ids, copy_mode))

    def insert_masked(
        self, keys, table_ids, values, scores=None, preserve_existing=False
    ) -> None:
        """``insert`` of the live rows; nothing is called when there are none."""
        count, _, dense = _live_rows(keys, table_ids, values, scores)
        if count == 0:
            return
        d_keys, d_ids, d_values, d_scores = dense
        # As the old call sites did: the scores only when there are some, the flag
        # only when it is set, so that a store that predates them keeps working.
        args = (d_keys, d_ids, d_values) + (() if d_scores is None else (d_scores,))
        if preserve_existing:
            self.inner.insert(*args, preserve_existing=True)
        else:
            self.inner.insert(*args)


# make it standalone to avoid recursive references.
class Storage(abc.ABC, Generic[OptionsT, OptimizerT]):
    """A tier that holds keys and their value rows (what ``external_storage`` is).

    **Two contracts.** ``find`` and ``insert`` are the host-sized contract: tensors of
    exactly the length of the keys that matter, and ``find`` returns a host count of
    the missing keys. DynamicEmb serves a class that implements only these by reading,
    once per call, how many of the rows it has are real (one blocking device-to-host
    read per call), compacting the tensors to that length and calling ``find`` /
    ``insert`` (:class:`OldContractStorage`).

    The padded contract, ``find_masked`` and ``insert_masked``, takes tensors of
    the length ``N`` of the whole batch whose rows may be inert (the reserved key), has
    no count anywhere and must not wait for the device. A class that implements both
    and declares ``accepts_empty_keys = True`` **in its own class body** is called
    through them, and DynamicEmb reads nothing from the device on its behalf. The
    attribute is read from the class's own ``__dict__``: a subclass that overrides
    anything has not made the promise and has to make it itself.

    Rules of the padded contract:

    * ``N`` is ``keys.numel()``. A *live* row has a key other than the reserved
      all-ones key (``-1`` as int64, ``0xFF..FF`` as uint64); every other row is
      *inert*, whatever its table id (``-1`` on the padding past the real rows, or a
      real table id where DynamicEmb merely hides a key). Do not read the table id or
      the values row of an inert row except through a mask. ``find_masked`` reports
      ``founds[i] = False`` for it and leaves ``values[i]`` unspecified;
      ``insert_masked`` stores nothing for it.
    * There is no count: nothing takes, returns or branches on a number of live rows.
    * All tensors are CUDA tensors on the storage's device, produced on the current
      stream. Enqueue your work on the current stream, do not synchronize it and do not
      read a device value on the host. DynamicEmb reuses its argument buffers after the
      call returns, so copy what you keep. The ``values`` that ``find_masked`` returns
      is a tensor DynamicEmb may write into (the initializer writes the rows of keys
      found nowhere), so it must not be a view of data you read again.
    * Live ``(table_id, key)`` pairs are unique within one ``insert_masked`` call (they
      come from the dedup); an evaluation of a sequence layout may send repeated keys
      to ``find_masked``. Keys ``0xFF..FC`` to ``0xFF..FE`` are never valid keys and
      are never sent.
    * ``values`` have the padded layout of :class:`CopyMode`: ``[N, max_embedding_dim]``
      for ``EMBEDDING`` and ``[N, max_value_dim]`` for ``VALUE`` (the embedding padded
      to ``max_embedding_dim``, then the optimizer state), in ``embedding_dtype()``.
    * The calls of one batch of a cache's backing storage arrive in this order:
      ``find_masked`` / ``lookup_masked`` (the keys the storage lacks), then
      ``insert_masked`` (what the storage evicted for room), then ``insert_masked``
      (what the cache evicted that the storage no longer has).

    The in-tree :class:`~dynamicemb.key_value_table.DynamicEmbStorage` and
    :class:`~dynamicemb.key_value_table.HybridStorage` are the reference
    implementations.
    """

    @abc.abstractmethod
    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
        key_scores: Optional[torch.Tensor] = None,
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

    # ---- the padded contract (optional; see the class docstring) ------------------

    # A class sets it to True in ITS OWN body after implementing ``find_masked`` and
    # ``insert_masked``; an inherited value does not count (``has_padded_contract``).
    accepts_empty_keys: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        own = cls.__dict__
        defined = [m for m in ("find_masked", "insert_masked") if m in own]
        if defined and not own.get("accepts_empty_keys", False):
            warnings.warn(
                f"{cls.__name__} defines {' and '.join(defined)} but not "
                "accepts_empty_keys = True in its class body, so DynamicEmb will not "
                "call them: it serves the class through find / insert, with one "
                "blocking read per call.",
                UserWarning,
                stacklevel=2,
            )

    def find_masked(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
        key_scores: Optional[torch.Tensor] = None,
    ) -> PaddedFind:
        """Padded ``find``: look up every live row of ``keys``.

        ``founds[i]`` says whether live row ``i`` is stored and ``values[i]`` is its
        row (``copy_mode`` width); an inert row is not found and its values row is
        unspecified. ``lfu_accumulated_frequency`` and ``key_scores`` (int64[N] or
        None) are what ``find`` took, aligned with ``keys``: a store that keeps
        frequencies or scores may use them, others ignore them. No count is returned
        or needed and the call must not synchronize.

        The default serves a class that did not declare the padded contract through
        its ``find``, after one blocking read of the number of live rows.
        """
        return OldContractStorage(self).find_masked(
            keys, table_ids, copy_mode, lfu_accumulated_frequency, key_scores
        )

    def insert_masked(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        preserve_existing: bool = False,
    ) -> None:
        """Padded ``insert``: upsert the live rows (``values[i]`` is ``[max_value_dim]``,
        aligned with ``keys``; ``scores`` int64[N] or None). Inert rows store nothing.

        ``scores`` is what the ``missing_scores`` of ``find`` would have been: the
        batch's frequency or score of the key, where the table has one. With
        ``preserve_existing=True`` the call refreshes the rows of keys the store
        already has and changes no score; what it does for a key the store does not
        hold is unspecified (DynamicEmb sends only keys it has just looked up or
        inserted).

        The default serves a class that did not declare the padded contract through
        its ``insert``, after one blocking read of the number of live rows.
        """
        OldContractStorage(self).insert_masked(
            keys, table_ids, values, scores, preserve_existing
        )

    def lookup_masked(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
        key_scores: Optional[torch.Tensor] = None,
        copy_mode: CopyMode = CopyMode.VALUE,
    ) -> MaskedFind:
        """The lookup half of the protocol the padded lookup code uses: look the keys
        up now, :meth:`load_masked` the rows later into a buffer of its choosing.

        The default is :meth:`find_masked` (the rows are fetched here and kept in
        ``MaskedFind.rows``); ``score_in`` is the batch's frequency (or score) tensor
        a later :meth:`insert_masked` takes as ``scores``.
        """
        found = PaddedFind(
            *self.find_masked(
                unique_keys,
                table_ids,
                copy_mode,
                lfu_accumulated_frequency,
                key_scores,
            )
        )
        _check_padded_find(
            self,
            unique_keys,
            found,
            self.max_embedding_dim()
            if copy_mode == CopyMode.EMBEDDING
            else self.max_value_dim(),
        )
        score_in = (
            lfu_accumulated_frequency
            if lfu_accumulated_frequency is not None
            else key_scores
        )
        return MaskedFind(
            founds=found.founds, score_in=score_in, scores=None, rows=found.values
        )

    def load_masked(
        self,
        found: MaskedFind,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Write the rows ``found`` fetched into ``output`` (``[N, width]``; the lanes
        that were not found keep what they hold) or return them as a new buffer."""
        width = (
            self.max_embedding_dim()
            if copy_mode == CopyMode.EMBEDDING
            else self.max_value_dim()
        )
        values = found.rows[:, :width]
        if output is None:
            return values
        output.copy_(
            torch.where(found.founds.unsqueeze(1), values.to(output.dtype), output)
        )
        return output


class Cache(abc.ABC):
    # No code path takes a cache other than ``DynamicEmbCache`` (they read its table
    # state); a subclass of it opts in with ``accepts_empty_keys = True`` in its own
    # body, as a storage does (see :class:`Storage`).
    accepts_empty_keys: bool = False

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

        ``num_evicted`` is a host ``int`` and each of the four ``evicted_*``
        tensors holds exactly that many entries: this is the contract of every
        implementation. A key the cache refuses (its bucket and overflow are all
        pinned) is reported as an evicted entry whose index is ``-(i + 1)`` for
        its position ``i`` in ``keys``, a value row that was never in the cache.

        The in-tree :class:`~dynamicemb.key_value_table.DynamicEmbCache` also
        takes the keyword-only argument ``async_count=True``, which a user cache
        does not have to implement (it is passed only to that class). Then the
        count is not read: ``num_evicted`` stays an ``int64[1]`` tensor on the
        device and the evicted tensors are as long as ``keys``, their entries
        past ``num_evicted`` inert (the reserved all-ones key, index and table
        id ``-1``).
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

    Two class attributes say how the module may call :meth:`admit`. Both
    default to the cautious answer, so an admitter that sets neither is called
    exactly as it always was.

    ``accepts_empty_keys``: True lets the module call ``admit`` with arrays as
    long as the whole batch of unique keys, rather than only the missing keys,
    so that it never has to ask the device how many are missing. Positions
    that are not missing keys hold the reserved all-ones key (``-1`` for int64
    keys, ``0xFFFFFFFFFFFFFFFF`` for uint64 ones; no table stores it), and
    positions past the end of the batch also have the table id ``-1``. What
    ``frequencies`` holds at them is unspecified. The module ignores the answer
    for those positions. Set it on your own class only if ``admit`` copes with
    all of that, does not count those positions as keys, and returns a mask as
    long as ``keys``. The module may then mark the rows its initializers must
    leave alone with a negative index, which only dynamicemb's own initializers
    are written for, so an admitter whose ``non_admitted_initializer`` is
    anything else is still called the old way. The attribute is read from the
    class's own ``__dict__``: a subclass of an admitter that opted in has to
    opt in itself, so that overriding ``admit`` never inherits a promise the
    new code did not make.

    ``uses_rng``: True for an admitter whose ``admit`` draws random numbers.
    The module then never lets timing decide whether to call it for a batch,
    so a seeded run consumes the same draws every time.
    """

    accepts_empty_keys: bool = False
    uses_rng: bool = False

    @abc.abstractmethod
    def admit(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        frequencies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Which of these missing keys may enter the table.

        Args:
            keys (torch.Tensor): The keys to decide on, deduplicated. With
                ``accepts_empty_keys`` also the reserved all-ones key at the
                positions that are not missing keys; see the class docstring.
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
