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
import os
import warnings
from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, Optional

import torch
from dynamicemb.types import (
    AdmissionStrategy,
    Counter,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    Storage,
)
from dynamicemb_extensions import DynamicEmbDataType, EvictStrategy, OptimizerType
from torchrec.modules.embedding_configs import BaseEmbeddingConfig
from torchrec.types import DataType

DEFAULT_INDEX_TYPE = torch.int64
DYNAMICEMB_CSTM_SCORE_CHECK = "DYNAMICEMB_CSTM_SCORE_CHECK"
BATCH_SIZE_PER_DUMP = 65536


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
    """

    TIMESTAMP = 0
    STEP = 1
    CUSTOMIZED = 2
    LFU = 3


# TODO: jiashuy, reorganize it
@dataclass
class _ContextOptions:
    """
    Parameters in InternalConfigs is not configurable when using dynamicemb by DistributedModelParallel.
    Internal Configurations that including three parts:

    1. Fixed
        score_type : torch.dtype
            Score represents how important an embedding item is. This specifies the type of the score.

    2. Inferred from the context: from params already defined in torchrec, or from the runtime env, e.g. world size.
        embedding_dtype : Optional[torch.dtype], optional
            Data (weight) type of dynamic embedding table.
        dim : Optional[int], optional
            The dimensionality of the value vectors. Default is -1, indicating it should be set explicitly.
        max_capacity : Optional[int], optional
                The maximum capacity of the shard of the embedding table on a single GPU. Automatically set in the shared planner.
                It is not configurable, but it's important for the total memory consumption.
                It will be automatically inferred from EmbeddingConfig.num_embeddings and the world size, rounded up to a power of 2，
                    and minimized to the size of bucket capacity of the hash table.
                If init_capacity is set, max_capacity will not be smaller than init_capacity.
        evict_strategy : DynamicEmbEvictStrategy
            Strategy used for evicting entries when the table exceeds its capacity. Default is DynamicEmbEvictStrategy.LRU.
        local_hbm_for_values : int
            High-bandwidth memory allocated for local values, in bytes. Default is 0.
        num_aligned_embedding_per_rank: int
                Number of aligned embedding per rank when the `num_embeddings` does not meet our alignment requirements, default to None.
        device_id : Optional[int], optional
                CUDA device index.
        optimizer_type: OptimizerType, used internally to determine how much memory optimizer states will consume,
            and default to `OptimizerType.Null`.

    3. Will be removed in the feature.
        block_size : int
            The size of blocks used during operations. Default is 128.
        io_block_size : int
            The size of input/output blocks during data transfer operations. Default is 1024.
        io_by_cpu : bool
            Flag indicating whether to use CPU for handling IO operations. Default is False.
        use_constant_memory : bool
            Flag to indicate if constant memory should be utilized. Default is False.
        reserved_key_start_bit : int
            Bit offset for reserved keys in the key space. Default is 0.
        num_of_buckets_per_alloc : int
            Number of buckets allocated per memory allocation request. Default is 1.

    """

    # Fixed:
    score_type: torch.dtype = torch.uint64

    # Inferred from the context.
    embedding_dtype: Optional[torch.dtype] = None
    dim: Optional[int] = None
    max_capacity: Optional[int] = None
    evict_strategy: DynamicEmbEvictStrategy = DynamicEmbEvictStrategy.LRU
    local_hbm_for_values: int = 0  # in bytes
    num_aligned_embedding_per_rank: int = None
    device_id: Optional[int] = None
    optimizer_type: OptimizerType = OptimizerType.Null

    # Will be removed in the future, please ignore them
    block_size: int = 128
    io_block_size: int = 1024
    io_by_cpu: bool = False  # use cpu to deal with the value copy.
    use_constant_memory: bool = False
    reserved_key_start_bit: int = 0
    num_of_buckets_per_alloc: int = 1


@dataclass
class DynamicEmbTableOptions(_ContextOptions):
    """
    Encapsulates the configuration options for dynamic embedding table.

    This class include parameters that control the behavior and performance of embedding lookup module, specifically tailored for dynamic embeddings.
    `get_grouped_key` will return fields used to group dynamic embedding tables.

    Parameters
    ----------
    training: bool
        Flag to indicate dynamic embedding tables is working on training mode or evaluation mode, default to `True`.
        If in training mode. **dyanmicemb** store embeddings and optimizer states together in the underlying key-value table. e.g.
        ```python
        key:torch.int64
        value = torch.concat(embedding, opt_states, dim=1)
        ```
        Therefore, if `training=True` dynamicemb will allocate memory for optimizer states whose consumption is decided by the `optimizer_type` which got from `fused_params`.
    initializer_args : DynamicEmbInitializerArgs
        Arguments for initializing dynamic embedding vector values when training, and default using uniform distribution.
        For `UNIFORM` and `TRUNCATED_NORMAL`, the `lower` and `upper` will set to $\pm {1 \over \sqrt{EmbeddingConfig.num\_embeddings}}$.
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
        If set, it will be rounded up to the power of 2.
        As the `load_factor` of the table increases, its capacity will gradually double (rehash) until it reaches `max_capacity`.
        Rehash will be done implicitly.
        Note: This is the setting for a single table at each rank.
    max_load_factor : float
        The maximum load factor before rehashing occurs. Default is 0.5.
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
    For detailed descriptions and additional context on each parameter, please refer to the documentation in this repository.
    """

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
    bucket_capacity: int = 128
    safe_check_mode: DynamicEmbCheckMode = DynamicEmbCheckMode.IGNORE
    global_hbm_for_values: int = 0  # in bytes
    external_storage: Storage = None
    index_type: Optional[torch.dtype] = None
    admit_strategy: Optional[AdmissionStrategy] = None
    admission_counter: Optional[Counter] = None

    def __post_init__(self):
        assert (
            self.eval_initializer_args.mode == DynamicEmbInitializerMode.CONSTANT
        ), "eval_initializer_args must be constant initialization"

        if self.init_capacity is not None:
            target_init_capacity = _next_power_of_2(self.init_capacity)
            if self.init_capacity != target_init_capacity:
                warnings.warn(
                    f"init_capacity is changed to {target_init_capacity} from {self.init_capacity}"
                )
                self.init_capacity = target_init_capacity

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


def get_optimizer_state_dim(optimizer_type, dim, dtype):
    if optimizer_type == OptimizerType.RowWiseAdaGrad:
        return 16 // DTYPE_NUM_BYTES[dtype]
    elif optimizer_type == OptimizerType.Adam:
        return dim * 2
    elif optimizer_type == OptimizerType.AdaGrad:
        return dim
    else:
        return 0


# TODO: sync with table
def validate_initializer_args(
    initializer_args: DynamicEmbInitializerArgs, eb_config: BaseEmbeddingConfig = None
) -> None:
    if initializer_args.mode == DynamicEmbInitializerMode.UNIFORM:
        default_lower = -sqrt(1 / eb_config.num_embeddings) if eb_config else 0.0
        default_upper = sqrt(1 / eb_config.num_embeddings) if eb_config else 1.0
        if initializer_args.lower is None:
            initializer_args.lower = default_lower
        if initializer_args.upper is None:
            initializer_args.upper = default_upper


def get_constraint_capacity(
    memory_bytes,
    dtype,
    dim,
    optimizer_type,
    bucket_capacity,
) -> int:
    byte_consume_per_vector = (
        dim + get_optimizer_state_dim(optimizer_type, dim, dtype)
    ) * dtype_to_bytes(dtype)
    bucket_size_in_bytes = bucket_capacity * byte_consume_per_vector
    assert (
        memory_bytes >= bucket_size_in_bytes
    ), f"reserved HBM bytes {memory_bytes} on rank {torch.distributed.get_rank()} is less than one bucket {bucket_size_in_bytes}"
    capacity = (
        memory_bytes // byte_consume_per_vector
    )  # maybe zero, we need at least one bucket
    return (capacity // bucket_capacity) * bucket_capacity


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
