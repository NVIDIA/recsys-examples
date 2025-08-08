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
from dataclasses import dataclass, field, fields
from math import sqrt
from typing import Dict, Optional

import torch
from dynamicemb_extensions import (
    DynamicEmbDataType,
    DynamicEmbTable,
    EvictStrategy,
    InitializerArgs,
    OptimizerType,
)
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

    def as_ctype(self) -> InitializerArgs:
        return InitializerArgs(
            self.mode.value,
            self.mean,
            self.std_dev,
            self.lower if self.lower else 0.0,
            self.upper if self.upper else 1.0,
            self.value,
        )


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


# Configs used as keys to group HKV variables(considering kernel behaviors, result type).
@dataclass
class GroupedHKVConfig:
    index_type: Optional[torch.dtype] = None
    embedding_dtype: Optional[torch.dtype] = None
    score_type: torch.dtype = torch.uint64
    device_id: Optional[int] = None
    optimizer_type: OptimizerType = OptimizerType.Null


# HKV configs can't be inferred by context.
@dataclass
class HKVConfig(GroupedHKVConfig):
    # Inferred from the context.
    dim: Optional[int] = None
    max_capacity: Optional[int] = None
    # Configured by the user.
    global_hbm_for_values: int = 0  # in bytes
    evict_strategy: DynamicEmbEvictStrategy = DynamicEmbEvictStrategy.LRU
    bucket_capacity: int = 128
    safe_check_mode: DynamicEmbCheckMode = DynamicEmbCheckMode.IGNORE
    # Used internally
    local_hbm_for_values: int = 0  # in bytes
    init_capacity: Optional[
        int
    ] = None  # if not set then set to max_capcacity after sharded
    max_load_factor: float = 0.5  # max load factor before rehash
    block_size: int = 128
    io_block_size: int = 1024
    io_by_cpu: bool = False  # use cpu to deal with the value copy.
    use_constant_memory: bool = False
    reserved_key_start_bit: int = 0
    num_of_buckets_per_alloc: int = 1


@dataclass
class DynamicEmbTableOptions(HKVConfig):
    """
    Encapsulates the configuration options for dynamic embedding tables.

    This class extends HKVConfig to include parameters that control the behavior and performance of
    hierarchical key-value storage systems, specifically tailored for dynamic embeddings in
    recommender systems. The options provided here allow users to customize their embedding tables
    according to their specific requirements.

    Including options:
        1. HKVConfig: explicitly defined by users.
            - Common configs for tables can be fused into a group.
            - Uncommon configs for each table in a group.
            - Some of HKVConfig can be inferred by context (index_type, embedding_dtype, dim, max_capacity, device_id, etc.)
        2. Initializer args.

    Parameters
    ----------
    index_type : Optional[torch.dtype], optional
        Index type of sparse features, will be set to DEFAULT_INDEX_TYPE(torch.int64) by default.
    embedding_dtype : Optional[torch.dtype], optional
        Data (weight) type of dynamic embedding table.
    score_type : torch.dtype
        Score represents how important an embedding item is. This specifies the type of the score.
    device_id : Optional[int], optional
        CUDA device index.

    optimizer_type: OptimizerType
        Optimizer type used to create HKV table, because different optimizers bring different states consume.
        It only used internally, and default to `OptimizerType.Null`.

    dim : Optional[int], optional
        The dimensionality of the value vectors. Default is -1, indicating it should be set explicitly.
    max_capacity : Optional[int], optional
        The maximum capacity of the embedding table. Automatically set in the shared planner.
    init_capacity : Optional[int], optional
        The initial capacity of the table. If not set, it defaults to max_capacity after sharding.
    max_load_factor : float
        The maximum load factor before rehashing occurs. Default is 0.5.
    global_hbm_for_values : int
        Total high-bandwidth memory allocated for entire embedding values, in bytes. Default is 0.
    local_hbm_for_values : int
        High-bandwidth memory allocated for local values, in bytes. Default is 0.
    evict_strategy : DynamicEmbEvictStrategy
        Strategy used for evicting entries when the table exceeds its capacity. Default is DynamicEmbEvictStrategy.LRU.
        At present, only DynamicEmbEvictStrategy.LRU and DynamicEmbEvictStrategy.LFU are available.
    bucket_capacity : int
        The number of entries each bucket can hold. Default is 128.
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
    initializer_args : DynamicEmbInitializerArgs
        Arguments for initializing dynamic embedding vector values.
        Default is uniform distribution, and absolute values of upper and lower bound are sqrt(1 / eb_config.num_embeddings).
    eval_initializer_args: DynamicEmbInitializerArgs
        The initializer args for evaluation mode.
        Default is constant initialization with value 0.0.
    score_strategy(DynamicEmbScoreStrategy):
        The strategy to set the score for each indices in forward and backward per table.
        Default to DynamicEmbScoreStrategy.TIMESTAMP.
        For the multi-GPUs scenario of model parallelism, every rank's score_strategy should keep the same for one table,
            as they are the same table, but stored on different ranks.
    safe_check_mode : DynamicEmbCheckMode
        Should dynamic embedding table insert safe check be enabled? By default, it is disabled.
        Please refer to the API documentation for DynamicEmbCheckMode for more information.
    training: bool
        Flag to indicate dynamic embedding tables is working on training mode or evaluation mode, default to `True`.
    caching: bool
        Flag to indicate dynamic embedding tables is working on caching mode, which will support to prefetch embeddings
        from host memory to HBM if existed, default to `False`.
    num_aligned_embedding_per_rank: int
        Number of aligned embedding per rank when the `num_embeddings` does not meet our alignment requirements, default to None.

    Notes
    -----
    For detailed descriptions and additional context on each parameter, please refer to the documentation at
    https://github.com/NVIDIA-Merlin/HierarchicalKV.
    """

    initializer_args: DynamicEmbInitializerArgs = field(
        default_factory=DynamicEmbInitializerArgs
    )
    eval_initializer_args: DynamicEmbInitializerArgs = field(
        default_factory=lambda: DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.CONSTANT,
            value=0.0,
        )
    )
    score_strategy: DynamicEmbScoreStrategy = DynamicEmbScoreStrategy.TIMESTAMP
    training: bool = True
    caching: bool = False
    num_aligned_embedding_per_rank: int = None

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
        grouped_key = {f.name: getattr(self, f.name) for f in fields(GroupedHKVConfig)}
        grouped_key["training"] = self.training
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


def create_dynamicemb_table(table_options: DynamicEmbTableOptions) -> DynamicEmbTable:
    if not table_options.training:
        table_options.optimizer_type = OptimizerType.Null
    return DynamicEmbTable(
        torch_to_dyn_emb(table_options.index_type),
        torch_to_dyn_emb(table_options.embedding_dtype),
        table_options.evict_strategy.value,
        table_options.dim,
        table_options.init_capacity,
        table_options.max_capacity,
        table_options.local_hbm_for_values,
        table_options.bucket_capacity,
        table_options.max_load_factor,
        table_options.block_size,
        table_options.io_block_size,
        table_options.device_id,
        table_options.io_by_cpu,
        table_options.use_constant_memory,
        table_options.reserved_key_start_bit,
        table_options.num_of_buckets_per_alloc,
        table_options.initializer_args.as_ctype(),
        table_options.safe_check_mode.value,
        table_options.optimizer_type,
    )


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
    byte_consume = (
        dim + get_optimizer_state_dim(optimizer_type, dim, dtype)
    ) * dtype_to_bytes(dtype)
    capacity = memory_bytes // byte_consume
    return (capacity // bucket_capacity) * bucket_capacity
