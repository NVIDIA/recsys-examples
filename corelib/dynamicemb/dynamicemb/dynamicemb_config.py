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

import enum
import math
import os
import warnings
from dataclasses import dataclass, field, replace
from math import sqrt
from typing import Any, Dict, Optional, Tuple

import torch
from dynamicemb.types import (
    BUCKET_ALIGNMENT,
    DEMB_TABLE_ALIGN_SIZE,
    MAX_BUCKET_CAPACITY,
    AdmissionStrategy,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    Storage,
)
from dynamicemb_extensions import DynamicEmbDataType, EvictStrategy
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.modules.embedding_configs import BaseEmbeddingConfig
from torchrec.types import DataType

DEFAULT_INDEX_TYPE = torch.int64
DYNAMICEMB_CSTM_SCORE_CHECK = "DYNAMICEMB_CSTM_SCORE_CHECK"
BATCH_SIZE_PER_DUMP = 65536
# Must match ``MappingEmbeddingGenerator`` mod in ``debug_init`` (initializer.cu).
DEBUG_EMB_INITIALIZER_MOD = 100_000
# Default hashtable bucket width in rows; keep in sync with
# :class:`DynamicEmbTableOptions` and :func:`get_table_value_bytes`.
DEFAULT_BUCKET_CAPACITY = 128


def warning_for_cstm_score() -> None:
    env = os.getenv(DYNAMICEMB_CSTM_SCORE_CHECK)
    if env is not None and env == "0":
        return False
    return True


DynamicEmbKernel = "DynamicEmb"


@enum.unique
class DynamicEmbCheckMode(enum.IntEnum):
    """
    Enumeration for different modes of checking dynamic embedding's insertion behaviors.
    DynamicEmb uses a hashtable as the backend. If the embedding table capacity is small and the number of indices in a single lookup is large,
    it is easy for too many indices to be allocated to the same hash table bucket in one lookup, resulting in the inability to insert indices into the hashtable.
    DynamicEmb resolves this issue by setting the lookup results of indices that cannot be inserted to 0.
    Fortunately, in a hashtable with a large capacity, such insertion failures are very rare and almost never occur.
    This issue is more frequent in hashtables with small capacities, which can affect training accuracy.
    Therefore, we do not recommend using dynamic embedding tables for very small embedding tables.

    To prevent this behavior from affecting training without user awareness, DynamicEmb provides a safe check mode.
    Users can set whether to enable safe check when configuring DynamicEmbTableOptions.
    Enabling safe check will add some overhead, but it can provide insights into whether the hash table frequently fails to insert indices.
    If the number of insertion failures is high and the proportion of affected indices is large,
    it is recommended to either increase the dynamic embedding capacity or avoid using dynamic embedding tables for small embedding tables.

    Attributes
    ----------
    ERROR : int
        When there are indices that can't be inserted successfully:
            This mode will throw a runtime error indicating how many indices failed to insert.
            The program will crash.
    WARNING : int
        When there are indices that can't be inserted successfully:
            This mode will give a warning about how many indices failed to insert.
            The program will continue. For uninserted indices, their embeddings' values will be set to 0.0.
    IGNORE : int
        Don't check whether insertion is successful or not, therefore it doesn't bring additional checking overhead.
        For uninserted indices, their embeddings' values will be set to 0.0 silently.
    """

    ERROR = 0
    WARNING = 1
    IGNORE = 2


class DynamicEmbPoolingMode(enum.IntEnum):
    SUM = 0
    MEAN = 1
    NONE = 2


@enum.unique
class DynamicEmbEvictStrategy(enum.Enum):
    LRU = EvictStrategy.KLru
    LFU = EvictStrategy.KLfu
    EPOCH_LRU = EvictStrategy.KEpochLru
    EPOCH_LFU = EvictStrategy.KEpochLfu
    CUSTOMIZED = EvictStrategy.KCustomized


class DynamicEmbScoreStrategy(enum.IntEnum):
    """
    Enumeration for different modes to set index-embedding's score.
    The index-embedding pair with smaller scores will be more likely to be evicted from the embedding table when the table is full.

    dynamicemb allows configuring scores by table.
    For a table, the scores in the subsequent forward passes are larger than those in the previous ones for modes TIMESTAMP and STEP.
    Users can also provide customized score(mode CUSTOMIZED) for each table's forward pass.
    Attributes
    ----------
    TIMESTAMP:
        In a forward pass, embedding table's scores will be set to global nanosecond timer of device, and due to the timing of GPU scheduling,
          different scores may have slight differences.
        Users must not set scores under TIMESTAMP mode.
    STEP:
        Each embedding table has a member `step` which will increment for every forward pass.
        All scores in each forward pass are the same which is step's value.
        Users must not set scores under STEP mode.
    CUSTOMIZED:
        Each embedding table's score are managed by users.
        Users have to set the score before every forward pass using `set_score` interface.
    LFU:
        If there are not enough slots inside the bucket to store new keys, the least used key in the bucket will be evicted.
    NO_EVICTION:
        The table’s capacity doubles whenever there are not enough slots for new keys, and this continues until available memory is exhausted.
        When the memory resources are insufficient, there will be a warning message, and training can continue but the accuracy of eviction cannot be guaranteed.
    """

    TIMESTAMP = 0
    STEP = 1
    CUSTOMIZED = 2
    LFU = 3
    NO_EVICTION = 4


@dataclass
class DynamicEmbTableOptions:
    """
    Encapsulates the configuration options for dynamic embedding table.

    This class includes parameters that control the behavior and performance of the embedding lookup module, specifically tailored for dynamic embeddings.
    `get_grouped_key` will return fields used to group dynamic embedding tables.

    Fields listed first (through ``device_id``) are often filled by the planner or runtime rather than
    being the main user configuration knobs. Score handling for the hash table follows score
    policies and kernels, not a separate score-dtype field on this dataclass.

    Parameters
    ----------
    embedding_dtype : Optional[torch.dtype], optional
        Data (weight) type of dynamic embedding table.
    dim : Optional[int], optional
        Value vector dimension. With ``DynamicEmbeddingShardingPlanner``, ``_prepare_dynemb_table_options``
        sets it from ``BaseEmbeddingConfig.embedding_dim``. The embedding kernel only warns if it
        differs from the sharded ``local_cols`` (see ``_get_dynamicemb_options_per_table``).
    max_capacity : Optional[int], optional
        Per-shard maximum table rows on one GPU. With ``DynamicEmbeddingShardingPlanner``,
        ``_prepare_dynemb_table_options`` sets ``max_capacity`` to
        ``num_buckets * bucket_capacity`` from :func:`get_sharded_table_shape`.
        If ``init_capacity`` is unset it becomes ``max_capacity``; if set and aligned,
        it is clamped to at most ``max_capacity``.
        The embedding kernel checks consistency with TorchREC shard metadata (see
        ``_get_dynamicemb_options_per_table``).
    evict_strategy : DynamicEmbEvictStrategy
        Strategy used for evicting entries when the table exceeds its capacity.
        Default is :attr:`DynamicEmbEvictStrategy.LRU`.
    local_hbm_for_values : int
        High-bandwidth memory allocated for local values, in bytes. Default is 0.
        With ``DynamicEmbeddingShardingPlanner``, this is set to
        ``ceil(global_hbm_for_values / world_size)`` per rank.
    device_id : Optional[int], optional
        CUDA device index.
    training: bool
        Flag to indicate dynamic embedding tables is working on training mode or evaluation mode, default to `True`.
        If in training mode. **dynamicemb** stores embeddings and optimizer states together in the underlying key-value table. e.g.
        ```python
        key:torch.int64
        value = torch.concat(embedding, opt_states, dim=1)
        ```
        Therefore, if `training=True` the module allocates memory for optimizer states; the per-row state size follows the ``optimizer`` entry in ``fused_params`` (FBGEMM ``EmbOptimType``), as used by :class:`~dynamicemb.batched_dynamicemb_tables.BatchedDynamicEmbeddingTablesV2`.
    initializer_args : DynamicEmbInitializerArgs
        Arguments for initializing dynamic embedding vector values when training, and default using uniform distribution.
        For ``UNIFORM`` and ``TRUNCATED_NORMAL``, ``lower`` and ``upper`` default to
        ``±1/sqrt(N)`` where ``N`` is ``EmbeddingConfig.num_embeddings``.
    eval_initializer_args: DynamicEmbInitializerArgs
        The initializer args for evaluation mode, and will return torch.zeros(...) as embedding by default if index/sparse feature is missing.
    caching: bool
        Flag to indicate dynamic embedding tables is working on caching mode, default to `False`.
        When the device memory on a single GPU is insufficient to accommodate a single shard of the dynamic embedding table,
            dynamicemb supports the mixed use of device memory and host memory(pinned memory).
        But by default, the values of the entire table are concatenated with device memory and host memory.
        This means that the storage location of one embedding is determined by `hash_function(key)`, and mapping to device memory will bring better lookup performance.
        However, sparse features in training are often with temporal locality.
        In order to store hot keys in device memory, dynamicemb creates two table instances,
            whose values are stored in device memory and host memory respectively, and store hot keys on the GPU table priorily.
        If the GPU table is full, the evicted keys will be inserted into the host table.
        If the host table is also full, the key will be evicted(all the eviction is based on the score per key).
        The original intention of eviction is based on this insight: features that only appear once should not occupy memory(even host memory) for a long time.
        In short:
            set **`caching=True`** will create a GPU table and a host table, and make GPU table serves as a cache;
            set **`caching=False`** will create a hybrid table which use GPU and host memory in a concatenated way to store value.
            All keys and other meta data are always stored on GPU for both cases.
    init_capacity : Optional[int], optional
        The initial capacity of the table. If not set, it defaults to max_capacity after sharding.
        If `init_capacity` is provided, it will serve as the initial table capacity on a single GPU.
        With :class:`~dynamicemb.planner.planner.DynamicEmbeddingShardingPlanner`, it is rounded up
        to a multiple of the effective ``bucket_capacity`` in ``_prepare_dynemb_table_options``,
        then capped at ``max_capacity`` if the aligned value is larger.
        As the `load_factor` of the table increases, its capacity will gradually double (rehash) until it reaches `max_capacity`.
        Rehash will be done implicitly.
        Note: This is the setting for a single table at each rank.
    max_load_factor : float
        The maximum load factor before rehashing occurs. Default is 0.5.
        In NO_EVICTION mode, this option is ignored; the implementation uses
        a fixed effective max load factor of 0.5 for the key_index_map.
    score_strategy(DynamicEmbScoreStrategy):
        dynamicemb gives each key-value pair a score to represent its importance.
        Once there is insufficient space, the key-value pair will be evicted based on the score.
        The `score_strategy` is used to configure how to set the scores for keys in each batch.
        Default to DynamicEmbScoreStrategy.TIMESTAMP.
        For the multi-GPUs scenario of model parallelism, every rank's score_strategy should keep the same for one table,
            as they are the same table, but stored on different ranks.
    bucket_capacity : int
        Capacity of each bucket in the hash table, and default is 128 (using 1024 when the table serves as cache).
        A key will only be mapped to one bucket.
        When the bucket is full, the key with the smallest score in the bucket will be evicted, and its slot will be used to store a new key.
        The larger the bucket capacity, the more accurate the score based eviction will be, but it will also result in performance loss.
    safe_check_mode : DynamicEmbCheckMode
        Used to check if all keys in the current batch have been successfully inserted into the table.
        Should dynamic embedding table insert safe check be enabled? By default, it is disabled.
        Please refer to the API documentation for DynamicEmbCheckMode for more information.
    global_hbm_for_values : int
        Total GPU memory allocated to store embedding + optimizer states, in bytes. Default is 0.
        It has different meanings under `caching=True` and  `caching=False`.
            When `caching=False`, it decides how much GPU memory is in the total memory to store value in a single hybrid table.
            When `caching=True`, it decides the table capacity of the GPU table.
    external_storage: Storage
        The external storage/ParamterServer which inherits the interface of Storage, and can be configured per table.
        If not provided, will using DynamicEmbeddingTable as the Storage.
    index_type : Optional[torch.dtype], optional
        Index type of sparse features, will be set to DEFAULT_INDEX_TYPE(torch.int64) by default.
    admit_strategy : Optional[AdmissionStrategy], optional
        Admission strategy for controlling which keys are allowed to enter the embedding table.
        If provided, only keys that meet the strategy's criteria will be inserted into the table.
        Keys that don't meet the criteria will still be initialized and used in the forward pass,
        but won't be stored in the table. Default is None (all keys are admitted).
    admission_counter : Optional[Counter], optional
        Counter for tracking the number of keys that have been admitted to the embedding table.
        If provided, the counter will be used to track the number of keys that have been admitted to the embedding table.
        Default is None (no counter is used).
    Notes
    -----
    The ``DynamicEmb_APIs.md`` file in the ``dynamicemb`` package mirrors this class and related planner
    behavior (e.g. :class:`~dynamicemb.planner.planner.DynamicEmbeddingShardingPlanner`).
    """

    embedding_dtype: Optional[torch.dtype] = None
    dim: Optional[int] = None
    max_capacity: Optional[int] = None
    evict_strategy: DynamicEmbEvictStrategy = DynamicEmbEvictStrategy.LRU
    local_hbm_for_values: int = 0  # in bytes
    device_id: Optional[int] = None

    training: bool = True
    initializer_args: DynamicEmbInitializerArgs = field(
        default_factory=DynamicEmbInitializerArgs
    )
    eval_initializer_args: DynamicEmbInitializerArgs = field(
        default_factory=lambda: DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.CONSTANT,
            value=0.0,
        )
    )
    caching: bool = False
    init_capacity: Optional[
        int
    ] = None  # if not set then set to max_capcacity after sharded
    max_load_factor: float = 0.5  # max load factor before rehash(double capacity)
    score_strategy: DynamicEmbScoreStrategy = DynamicEmbScoreStrategy.TIMESTAMP
    bucket_capacity: int = DEFAULT_BUCKET_CAPACITY
    safe_check_mode: DynamicEmbCheckMode = DynamicEmbCheckMode.IGNORE
    global_hbm_for_values: int = 0  # in bytes
    external_storage: Storage = None
    index_type: Optional[torch.dtype] = None
    admit_strategy: Optional[AdmissionStrategy] = None
    admission_counter: Optional[Any] = None

    def __post_init__(self):
        assert (
            self.eval_initializer_args.mode == DynamicEmbInitializerMode.CONSTANT
        ), "eval_initializer_args must be constant initialization"

    def __eq__(self, other):
        if not isinstance(other, DynamicEmbTableOptions):
            return NotImplementedError
        self_group_keys = self.get_grouped_key()
        other_group_keys = other.get_grouped_key()
        return self_group_keys == other_group_keys

    def __ne__(self, other):
        if not isinstance(other, DynamicEmbTableOptions):
            return NotImplementedError
        return not (self == other)

    def get_grouped_key(self):
        grouped_key = {}
        grouped_key["training"] = self.training
        grouped_key["caching"] = self.caching
        grouped_key["external_storage"] = self.external_storage
        grouped_key["index_type"] = self.index_type
        grouped_key["score_strategy"] = self.score_strategy
        grouped_key["admit_strategy"] = self.admit_strategy
        return grouped_key

    def __hash__(self):
        group_keys = self.get_grouped_key()
        return hash(tuple(group_keys.items()))


def data_type_to_dyn_emb(data_type: DataType) -> DynamicEmbDataType:
    if data_type.value == DataType.FP32.value:
        return DynamicEmbDataType.Float32
    elif data_type.value == DataType.FP16.value:
        return DynamicEmbDataType.Float16
    elif data_type.value == DataType.BF16.value:
        return DynamicEmbDataType.BFloat16
    elif data_type.value == DataType.INT64.value:
        return DynamicEmbDataType.Int64
    elif data_type.value == DataType.INT32.value:
        return DynamicEmbDataType.Int32
    elif data_type.value == DataType.INT8.value:
        return DynamicEmbDataType.Int8
    elif data_type.value == DataType.UINT8.value:
        return DynamicEmbDataType.UInt8
    else:
        raise ValueError(
            f"DataType {data_type} cannot be converted to DynamicEmbDataType"
        )


def data_type_to_dtype(data_type: DataType) -> torch.dtype:
    if data_type.value == DataType.FP32.value:
        return torch.float32
    elif data_type.value == DataType.FP16.value:
        return torch.float16
    elif data_type.value == DataType.BF16.value:
        return torch.bfloat16
    elif data_type.value == DataType.INT64.value:
        return torch.int64
    elif data_type.value == DataType.INT32.value:
        return torch.int32
    elif data_type.value == DataType.INT8.value:
        return torch.int8
    elif data_type.value == DataType.UINT8.value:
        return torch.uint8
    else:
        raise ValueError(f"DataType {data_type} cannot be converted to torch.dtype")


def dyn_emb_to_torch(data_type: DynamicEmbDataType) -> torch.dtype:
    if data_type == DynamicEmbDataType.Float32:
        return torch.float32
    elif data_type == DynamicEmbDataType.BFloat16:
        return torch.bfloat16
    elif data_type == DynamicEmbDataType.Float16:
        return torch.float16
    elif data_type == DynamicEmbDataType.Int64:
        return torch.int64
    elif data_type == DynamicEmbDataType.UInt64:
        return torch.uint64
    elif data_type == DynamicEmbDataType.Int32:
        return torch.int32
    elif data_type == DynamicEmbDataType.UInt32:
        return torch.uint32
    elif data_type == DynamicEmbDataType.Size_t:
        return torch.int64  # Size_t to int64
    else:
        raise ValueError(f"Unsupported DynamicEmbDataType: {data_type}")


def torch_to_dyn_emb(torch_dtype: torch.dtype) -> DynamicEmbDataType:
    if torch_dtype == torch.float32:
        return DynamicEmbDataType.Float32
    elif torch_dtype == torch.bfloat16:
        return DynamicEmbDataType.BFloat16
    elif torch_dtype == torch.float16:
        return DynamicEmbDataType.Float16
    elif torch_dtype == torch.int64:
        return DynamicEmbDataType.Int64
    elif torch_dtype == torch.uint64:
        return DynamicEmbDataType.UInt64
    elif torch_dtype == torch.int32:
        return DynamicEmbDataType.Int32
    elif torch_dtype == torch.uint32:
        return DynamicEmbDataType.UInt32
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def dtype_to_bytes(dtype: torch.dtype) -> int:
    dtype_size_map = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.int8: 1,
        torch.uint8: 1,
        torch.int16: 2,
        torch.uint16: 2,
        torch.int32: 4,
        torch.uint32: 4,
        torch.int64: 8,
        torch.uint64: 8,
        torch.bool: 1,
    }
    if dtype not in dtype_size_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_size_map[dtype]


def string_to_evict_strategy(strategy_str: str) -> EvictStrategy:
    if strategy_str == "KLru":
        return EvictStrategy.KLru
    elif strategy_str == "KLfu":
        return EvictStrategy.KLfu
    elif strategy_str == "KEpochLru":
        return EvictStrategy.KEpochLru
    elif strategy_str == "KEpochLfu":
        return EvictStrategy.KEpochLfu
    elif strategy_str == "KCustomized":
        return EvictStrategy.KCustomized
    else:
        raise ValueError(f"Invalid EvictStrategy string: {strategy_str}")


DTYPE_NUM_BYTES: Dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}


def get_optimizer_state_dim(
    optimizer_type: EmbOptimType,
    dim: int,
    dtype: Optional[torch.dtype] = None,
) -> int:
    """Optimizer state elements per row (same rules as FBGEMM fused table value layout).

    ``dtype`` is only required for ``EXACT_ROWWISE_ADAGRAD`` (fixed 16-byte rowwise state
    in embedding dtype units). Callers that know the embedding dtype may pass it for any
    optimizer; it is ignored except for rowwise Adagrad.
    """
    if optimizer_type == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
        if dtype is None:
            raise ValueError(
                "dtype is required when optimizer_type is EmbOptimType.EXACT_ROWWISE_ADAGRAD."
            )
        return 16 // DTYPE_NUM_BYTES[dtype]
    if optimizer_type == EmbOptimType.ADAM:
        return dim * 2
    if optimizer_type == EmbOptimType.EXACT_ADAGRAD:
        return dim
    return 0


def complete_initializer_args(
    initializer_args: DynamicEmbInitializerArgs,
    *,
    embedding_config: Optional[BaseEmbeddingConfig] = None,
) -> DynamicEmbInitializerArgs:
    """Complete missing UNIFORM ``lower`` / ``upper`` from ``embedding_config`` (or defaults).

    Does not mutate ``initializer_args``. If defaults are applied, returns a new
    :class:`~dynamicemb.types.DynamicEmbInitializerArgs`; otherwise returns the
    same instance unchanged.

    Parameters
    ----------
    initializer_args
        Requested initializer; only ``UNIFORM`` mode may get ``lower`` / ``upper`` filled.
    embedding_config
        Used to set default bounds as ``±sqrt(1 / num_embeddings)`` when the
        corresponding bound is ``None``. If omitted, defaults are ``0.0`` and ``1.0``.
    """
    if initializer_args.mode != DynamicEmbInitializerMode.UNIFORM:
        return initializer_args

    needs_lower = initializer_args.lower is None
    needs_upper = initializer_args.upper is None
    if not needs_lower and not needs_upper:
        return initializer_args

    if embedding_config is not None:
        scale = sqrt(1.0 / float(embedding_config.num_embeddings))
        default_lower = -scale
        default_upper = scale
    else:
        default_lower = 0.0
        default_upper = 1.0

    return replace(
        initializer_args,
        lower=default_lower if needs_lower else initializer_args.lower,
        upper=default_upper if needs_upper else initializer_args.upper,
    )


def get_constraint_capacity(
    memory_bytes,
    dtype,
    dim,
    optimizer_type: EmbOptimType,
    bucket_capacity,
) -> int:
    byte_consume_per_vector = (
        dim + get_optimizer_state_dim(optimizer_type, dim, dtype)
    ) * dtype_to_bytes(dtype)
    bucket_size_in_bytes = bucket_capacity * byte_consume_per_vector
    # If reserved HBM is less than one bucket, round up to one bucket
    if memory_bytes < bucket_size_in_bytes:
        warnings.warn(
            f"Reserved HBM ({memory_bytes} bytes) is less than one bucket "
            f"({bucket_size_in_bytes} bytes). Rounding up to one bucket.",
            UserWarning,
        )
        memory_bytes = bucket_size_in_bytes
    capacity = memory_bytes // byte_consume_per_vector  # at least one bucket
    return (capacity // bucket_capacity) * bucket_capacity


def align_to_table_size(n: int, alignment: int = DEMB_TABLE_ALIGN_SIZE) -> int:
    """Round up n to a multiple of ``alignment``.

    Non-positive values are treated as 0 and rounded up to ``alignment``
    to avoid zero capacity in planners/tables.
    """
    n = int(n)
    if n <= 0:
        return alignment
    return (n + alignment - 1) // alignment * alignment


def get_sharded_table_shape(
    embedding_config: BaseEmbeddingConfig,
    world_size: int,
    bucket_capacity: int,
) -> Tuple[int, int]:
    """Per-rank hashtable bucket layout: number of buckets and bucket width (rows per bucket).

    Per-rank row capacity is ``num_buckets * bucket_capacity``. Same rules as the embedding
    sharding planner when filling ``max_capacity`` (via that product) and ``bucket_capacity``.

    Parameters
    ----------
    embedding_config
        TorchREC embedding table config (e.g. :class:`~torchrec.modules.embedding_configs.EmbeddingConfig`);
        uses :attr:`~torchrec.modules.embedding_configs.BaseEmbeddingConfig.num_embeddings`.
    world_size
        Number of ranks (must be positive).
    bucket_capacity
        Requested bucket size in rows. If set to :data:`dynamicemb.types.MAX_BUCKET_CAPACITY`
        (``2**63 - 1``), returns ``(1, effective_bucket_rows)`` where the bucket spans the
        full per-rank table (aligned to :data:`BUCKET_ALIGNMENT`). Otherwise ``bucket_capacity``
        must be a positive multiple of :data:`BUCKET_ALIGNMENT`.

    Returns
    -------
    tuple[int, int]
        ``(num_buckets, bucket_capacity)`` for one rank. The first value is **not** per-rank table
        capacity in rows; callers must use ``num_buckets * bucket_capacity`` whenever they need
        row capacity (e.g. to match per-rank ``max_capacity`` set by the planner).
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    num_global = int(embedding_config.num_embeddings)
    shard_rows = math.ceil(num_global / world_size)

    if bucket_capacity == MAX_BUCKET_CAPACITY:
        effective_bucket = align_to_table_size(shard_rows, BUCKET_ALIGNMENT)
        return 1, effective_bucket

    if bucket_capacity <= 0:
        raise ValueError(
            f"bucket_capacity must be positive when not MAX_BUCKET_CAPACITY, "
            f"got {bucket_capacity}"
        )
    if bucket_capacity % BUCKET_ALIGNMENT != 0:
        raise ValueError(
            f"bucket_capacity ({bucket_capacity}) must be a multiple of "
            f"BUCKET_ALIGNMENT ({BUCKET_ALIGNMENT}) when not using MAX_BUCKET_CAPACITY."
        )

    table_capacity = align_to_table_size(shard_rows, bucket_capacity)
    num_buckets = table_capacity // bucket_capacity
    return num_buckets, bucket_capacity


def get_table_value_bytes(
    embedding_config: BaseEmbeddingConfig,
    optimizer_type: EmbOptimType,
    world_size: int,
    bucket_capacity: int = DEFAULT_BUCKET_CAPACITY,
) -> int:
    """Return how many bytes one DynamicEmb table needs for stored values across all ranks.

    This counts embedding plus optimizer-state storage. Per-rank rows are
    ``num_buckets * bucket_capacity`` from :func:`get_sharded_table_shape`; multiply by
    ``world_size`` for total rows, then by ``element_size * (dim + optimizer_state_dim)`` per row.
    The result uses the same rules as table construction with the sharding planner.

    Parameters
    ----------
    embedding_config
        Table shape and dtype from TorchREC (``num_embeddings``, ``embedding_dim``, ``data_type``).
    optimizer_type
        FBGEMM ``EmbOptimType``; see :func:`get_optimizer_state_dim` for per-optimizer state size.
    world_size
        Number of ranks, as in distributed planning.
    bucket_capacity
        Hashtable bucket width in rows, consistent with :func:`get_sharded_table_shape`
        and ``DynamicEmbTableOptions.bucket_capacity`` (default
        :data:`DEFAULT_BUCKET_CAPACITY`; including the ``MAX_BUCKET_CAPACITY``
        sentinel when applicable).
    """
    num_buckets, bucket_cap = get_sharded_table_shape(
        embedding_config, world_size, bucket_capacity
    )
    table_capacity_per_rank = num_buckets * bucket_cap
    total_rows = table_capacity_per_rank * world_size
    dim = embedding_config.embedding_dim
    torch_dtype = data_type_to_dtype(embedding_config.data_type)
    element_size = dtype_to_bytes(torch_dtype)
    optimizer_state_dim = get_optimizer_state_dim(optimizer_type, dim, torch_dtype)
    return int(element_size * (dim + optimizer_state_dim) * total_rows)


def _next_power_of_2(n):
    # Handle the case where n is 0
    if n == 0:
        return 1

    # If n is already a power of 2, return n
    if (n & (n - 1)) == 0:
        return n

    # Find the next power of 2
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32  # This line is necessary for 64-bit integers
    return n + 1
