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
import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from itertools import accumulate
from typing import List, Optional, Tuple, cast

import torch  # usort:skip
import torch.distributed as dist
from dynamicemb.batched_dynamicemb_function import (
    DynamicEmbeddingFunction,
    dynamicemb_prefetch,
)
from dynamicemb.dynamicemb_config import *
from dynamicemb.initializer import *
from dynamicemb.key_value_table import Cache, DynamicEmbeddingTable, Storage
from dynamicemb.optimizer import *
from dynamicemb.utils import tabulate
from dynamicemb_extensions import OptimizerType, device_timestamp
from torch import Tensor, nn  # usort:skip


@enum.unique
class SparseType(enum.Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    BF16 = "bf16"


class BoundsCheckMode(enum.IntEnum):
    # Raise an exception (CPU) or device-side assert (CUDA)
    FATAL = 0
    # Log the first out-of-bounds instance per kernel, and set to zero.
    WARNING = 1
    # Set to zero.
    IGNORE = 2
    # No bounds checks.
    NONE = 3


class WeightDecayMode(enum.IntEnum):
    NONE = 0
    L2 = 1
    DECOUPLE = 2
    COUNTER = 3
    COWCLIP = 4


class CounterWeightDecayMode(enum.IntEnum):
    NONE = 0
    L2 = 1
    DECOUPLE = 2


class LearningRateMode(enum.IntEnum):
    EQUAL = -1
    TAIL_ID_LR_INCREASE = 0
    TAIL_ID_LR_DECREASE = 1
    COUNTER_SGD = 2


class GradSumDecay(enum.IntEnum):
    NO_DECAY = -1
    CTR_DECAY = 0


@dataclass
class TailIdThreshold:
    val: float = 0
    is_ratio: bool = False


@dataclass
class CounterBasedRegularizationDefinition:
    counter_weight_decay_mode: CounterWeightDecayMode = CounterWeightDecayMode.NONE
    counter_halflife: int = -1
    adjustment_iter: int = -1
    adjustment_ub: float = 1.0
    learning_rate_mode: LearningRateMode = LearningRateMode.EQUAL
    grad_sum_decay: GradSumDecay = GradSumDecay.NO_DECAY
    tail_id_threshold: TailIdThreshold = field(default_factory=TailIdThreshold)
    max_counter_update_freq: int = 1000


@dataclass
class CowClipDefinition:
    counter_weight_decay_mode: CounterWeightDecayMode = CounterWeightDecayMode.NONE
    counter_halflife: int = -1
    weight_norm_coefficient: float = 0.0
    lower_bound: float = 0.0


def encode_meta_json_file_path(root_path: str, table_name: str) -> str:
    return os.path.join(root_path, f"{table_name}_opt_args.json")


def encode_checkpoint_file_path(
    root_path: str, table_name: str, rank: int, world_size: int, item: str
) -> str:
    assert item in ["keys", "values", "scores", "opt_values"]
    return os.path.join(
        root_path, f"{table_name}_emb_{item}.rank_{rank}.world_size_{world_size}"
    )


def encode_counter_checkpoint_file_path(
    root_path: str, table_name: str, rank: int, world_size: int, item: str
) -> str:
    assert item in ["keys", "frequencies"]
    return os.path.join(
        root_path, f"{table_name}_counter_{item}.rank_{rank}.world_size_{world_size}"
    )


def find_files(root_path: str, table_name: str, suffix: str) -> Tuple[List[str], int]:
    suffix_to_encode_file_path_func = {
        "emb_keys": partial(encode_checkpoint_file_path, item="keys"),
        "emb_values": partial(encode_checkpoint_file_path, item="values"),
        "emb_scores": partial(encode_checkpoint_file_path, item="scores"),
        "opt_values": partial(encode_checkpoint_file_path, item="opt_values"),
        "counter_keys": partial(encode_counter_checkpoint_file_path, item="keys"),
        "counter_frequencies": partial(
            encode_counter_checkpoint_file_path, item="frequencies"
        ),
    }
    if suffix not in suffix_to_encode_file_path_func:
        raise RuntimeError(f"Invalid suffix: {suffix}")
    encode_file_path_func = suffix_to_encode_file_path_func[suffix]

    import glob

    # v2 version
    files = glob.glob(encode_file_path_func(root_path, table_name, "*", "*"))
    if len(files) == 0:
        return [], 0
    files = sorted(files)
    world_size = int(files[0].split(".")[-1].split("_")[-1])
    if len(files) != world_size:
        raise RuntimeError(
            f"Checkpoints is corrupted. Found {len(files)} under path {root_path} for table {table_name}, but the number of checkpointed world size is {world_size}."
        )

    for i in range(world_size):
        expected_file_path = encode_file_path_func(root_path, table_name, i, world_size)
        if expected_file_path not in set(files):
            raise RuntimeError(
                f"Checkpoints is corrupted. Expected file path {expected_file_path} for table {table_name}, but it is not found."
            )

    return files, len(files)


def get_loading_files(
    root_path: str,
    name: str,
    rank: int,
    world_size: int,
) -> Tuple[List[str], List[str], List[str], List[str], int, int]:
    if not os.path.exists(root_path):
        raise RuntimeError(f"can't find path to load, path:", root_path)

    key_files, num_key_files = find_files(root_path, name, "emb_keys")
    value_files, num_value_files = find_files(root_path, name, "emb_values")
    score_files, num_score_files = find_files(root_path, name, "emb_scores")
    opt_files, num_opt_files = find_files(root_path, name, "opt_values")

    if num_key_files != num_value_files:
        assert (
            num_key_files > 0
        ), "No key files found under path {root_path} for table {name}"
        raise RuntimeError(
            f"The number of key files under path {root_path} for table {name} does not match the number of value files."
        )

    counter_key_files, num_counter_key_files = find_files(
        root_path, name, "counter_keys"
    )
    counter_freq_files, num_counter_freq_files = find_files(
        root_path, name, "counter_frequencies"
    )

    if num_counter_key_files != num_counter_freq_files:
        raise RuntimeError(
            f"The number of key files of admission counter under path {root_path} for table {name} does not match the number of frequency files({num_counter_key_files}/{num_counter_freq_files})."
        )

    if num_counter_key_files > 0 and num_counter_key_files != num_key_files:
        raise RuntimeError(
            f"The number of key files under path {root_path} for table {name} does not match the number of keys files of admission counter({num_key_files}/{num_counter_key_files})."
        )

    if world_size == num_key_files:
        return (
            [encode_checkpoint_file_path(root_path, name, rank, world_size, "keys")],
            [encode_checkpoint_file_path(root_path, name, rank, world_size, "values")],
            [encode_checkpoint_file_path(root_path, name, rank, world_size, "scores")]
            if num_score_files == num_key_files
            else [],
            [
                encode_checkpoint_file_path(
                    root_path, name, rank, world_size, "opt_values"
                )
            ]
            if num_opt_files == num_key_files
            else [],
            [
                encode_counter_checkpoint_file_path(
                    root_path, name, rank, world_size, "keys"
                )
            ]
            if num_counter_key_files == num_key_files
            else [],
            [
                encode_counter_checkpoint_file_path(
                    root_path, name, rank, world_size, "frequencies"
                )
            ]
            if num_counter_freq_files == num_key_files
            else [],
        )
    # TODO: support skipping files.
    return (
        key_files,
        value_files,
        score_files,
        opt_files,
        counter_key_files,
        counter_freq_files,
    )


def _print_memory_consume(
    table_names, dynamicemb_options, optimizer, device_id
) -> None:
    subtitle = [
        "",
        "total",
        "embedding",
        "optim_state",
        "total",
        "embedding",
        "optim_state",
        "total",
        "embedding",
        "optim_state",
    ]
    table_consume = []
    table_consume.append(subtitle)

    def _get_optimizer_state_dim(optimizer_type, dim, element_size):
        if optimizer_type == OptimizerType.RowWiseAdaGrad:
            return 16 // element_size
        elif optimizer_type == OptimizerType.Adam:
            return dim * 2
        elif optimizer_type == OptimizerType.AdaGrad:
            return dim
        else:
            return 0

    DTYPE_NUM_BYTES: Dict[torch.dtype, int] = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
    }

    def MB_(x) -> int:
        return x // (1024 * 1024)

    def KB_(x) -> int:
        return x // (1024)

    F = None

    for table_name, table_option in zip(table_names, dynamicemb_options):
        element_size = DTYPE_NUM_BYTES[table_option.embedding_dtype]
        emb_dim = table_option.dim
        if optimizer is not None:
            optim_state_dim = optimizer.get_state_dim(emb_dim)
        else:
            optim_state_dim = _get_optimizer_state_dim(
                table_option.optimizer_type, emb_dim, element_size
            )
        total_dim = emb_dim + optim_state_dim
        total_memory = table_option.max_capacity * element_size * total_dim
        if F is None:
            if total_memory // (1024 * 1024) != 0:
                F = MB_
            else:
                F = KB_
        local_hbm_for_values = min(table_option.local_hbm_for_values, total_memory)
        local_dram_for_values = total_memory - local_hbm_for_values
        table_consume.append(
            [
                table_name,
                F(total_memory),
                F(table_option.max_capacity * element_size * emb_dim),
                F(table_option.max_capacity * element_size * optim_state_dim),
                F(local_hbm_for_values),
                F(int(local_hbm_for_values * emb_dim // total_dim)),
                F(int(local_hbm_for_values * optim_state_dim // total_dim)),
                F(local_dram_for_values),
                F(int(local_dram_for_values * emb_dim // total_dim)),
                F(int(local_dram_for_values * optim_state_dim // total_dim)),
            ]
        )
    unit = "MB" if F == MB_ else "KB"
    title = [
        "table name",
        "",
        f"memory({unit})",
        "",
        "",
        f"hbm({unit})/cuda:{device_id}",
        "",
        "",
        f"dram({unit})",
        "",
    ]
    output = "\n\n" + tabulate(table_consume, title, sub_headers=True)
    print(output)


class BatchedDynamicEmbeddingTablesV2(nn.Module):
    """
    Dynamic Embedding uses a GPU-optimized scored hash table backend.
    Looks up one or more dynamic embedding tables. The module is application for training.

    Its optional to fuse the optimizer with the backward operator by parameter *update_grads_explicitly*.
    """

    optimizer_args: OptimizerArgs

    def __init__(
        self,
        table_options: List[DynamicEmbTableOptions],
        table_names: Optional[List[str]] = None,
        feature_table_map: Optional[List[int]] = None,  # [T]
        use_index_dedup: bool = False,
        prefetch_pipeline: bool = False,  #  we set the arg name same as FBGEMM TBE to align with it
        pooling_mode: DynamicEmbPoolingMode = DynamicEmbPoolingMode.SUM,
        output_dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        enforce_hbm: bool = False,  # place all weights/momentums in HBM when using cache
        bounds_check_mode: BoundsCheckMode = BoundsCheckMode.WARNING,
        optimizer: EmbOptimType = EmbOptimType.SGD,
        # General Optimizer args
        stochastic_rounding: bool = True,
        gradient_clipping: bool = False,
        max_gradient: float = 1.0,
        max_norm: float = 0.0,
        learning_rate: float = 0.01,
        # used by EXACT_ADAGRAD, EXACT_ROWWISE_ADAGRAD, EXACT_ROWWISE_WEIGHTED_ADAGRAD, LAMB, and ADAM only
        # NOTE that default is different from nn.optim.Adagrad default of 1e-10
        eps: float = 1.0e-8,
        # used by EXACT_ADAGRAD, EXACT_ROWWISE_ADAGRAD, and EXACT_ROWWISE_WEIGHTED_ADAGRAD only
        initial_accumulator_value: float = 0.0,
        momentum: float = 0.9,  # used by LARS-SGD
        # EXACT_ADAGRAD, SGD, EXACT_SGD do not support weight decay
        # LAMB, ADAM, PARTIAL_ROWWISE_ADAM, PARTIAL_ROWWISE_LAMB, LARS_SGD support decoupled weight decay
        # EXACT_ROWWISE_WEIGHTED_ADAGRAD supports L2 weight decay
        # EXACT_ROWWISE_ADAGRAD support both L2 and decoupled weight decay (via weight_decay_mode)
        weight_decay: float = 0.0,
        weight_decay_mode: WeightDecayMode = WeightDecayMode.NONE,
        eta: float = 0.001,  # used by LARS-SGD,
        beta1: float = 0.9,  # used by LAMB and ADAM
        beta2: float = 0.999,  # used by LAMB and ADAM
        counter_based_regularization: Optional[
            CounterBasedRegularizationDefinition
        ] = None,  # used by Rowwise Adagrad
        cowclip_regularization: Optional[
            CowClipDefinition
        ] = None,  # used by Rowwise Adagrad
        # TO align with FBGEMM TBE
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        assert len(table_options) >= 1
        table_option = table_options[0]
        for other_option in table_options:
            assert (
                table_option == other_option
            ), "All tables must match in grouped keys."
        self._dynamicemb_options = table_options
        self.initializer_args = table_option.initializer_args
        self.index_type = table_option.index_type
        self.embedding_dtype = table_option.embedding_dtype
        self.output_dtype = output_dtype
        self.pooling_mode = pooling_mode
        self.use_index_dedup = use_index_dedup
        self._enable_prefetch = prefetch_pipeline
        self.prefetch_stream = None
        self.num_prefetch_ahead = 0
        self._table_names = table_names
        self.bounds_check_mode_int: int = bounds_check_mode.value
        self._create_score()
        self._admit_strategy = self._dynamicemb_options[0].admit_strategy
        self._evict_strategy = self._dynamicemb_options[0].evict_strategy.value
        if device is not None:
            self.device_id = int(str(device)[-1])
        else:
            assert torch.cuda.is_available(), "No available CUDA device."
            self.device_id = torch.cuda.current_device()

        if table_option.device_id is None:
            for option in self._dynamicemb_options:
                option.device_id = self.device_id
        self.dims: List[int] = [option.dim for option in self._dynamicemb_options]
        # Sequence mode requires uniform embedding dim because the output is
        # [N, D].  Pooling mode supports mixed dims natively via D_offsets.
        if pooling_mode == DynamicEmbPoolingMode.NONE:
            assert all(d == self.dims[0] for d in self.dims), (
                f"Sequence mode requires uniform embedding dim, got {set(self.dims)}. "
                "Tables with different dims are automatically split into separate "
                "BatchedDynamicEmbeddingTablesV2 instances by the planner."
            )

        # physical table number.
        T_ = len(self._dynamicemb_options)
        assert T_ > 0
        self.feature_table_map: List[int] = (
            feature_table_map if feature_table_map is not None else list(range(T_))
        )
        # logical table number.
        T = len(self.feature_table_map)
        assert T_ <= T
        table_has_feature = [False] * T_
        for t in self.feature_table_map:
            table_has_feature[t] = True
        assert all(table_has_feature), "Each table must have at least one feature!"

        feature_dims = [self.dims[t] for t in self.feature_table_map]
        D_offsets = [0] + list(accumulate(feature_dims))
        self.total_D: int = D_offsets[-1]
        self.max_D: int = max(self.dims)

        # Per-feature cumulative dimension offsets, registered on GPU for use
        # by multi-dim pooling kernels.  Only needed when tables have mixed
        # embedding dimensions; uniform-dim pooling uses a simpler path.
        if self.max_D > min(self.dims):
            self.register_buffer(
                "D_offsets_t",
                torch.tensor(
                    D_offsets,
                    device=torch.device(self.device_id),
                    dtype=torch.int32,
                ),
            )
        else:
            self.register_buffer("D_offsets_t", None)

        self.feature_num = len(self.feature_table_map)
        # TODO:deal with shuffeld feature_table_map
        self.table_offsets_in_feature: List[int] = []
        old_table_id = -1
        for idx, table_id in enumerate(self.feature_table_map):
            if table_id != old_table_id:
                self.table_offsets_in_feature.append(idx)
                old_table_id = table_id
        self.table_offsets_in_feature.append(self.feature_num)
        self.feature_offsets = torch.tensor(
            self.table_offsets_in_feature,
            device=torch.device(self.device_id),
            dtype=torch.int64,
        )

        for option in self._dynamicemb_options:
            if option.init_capacity is None:
                option.init_capacity = option.max_capacity

        self._optimizer: BaseDynamicEmbeddingOptimizer = self._create_optimizer(
            optimizer,
            stochastic_rounding,
            gradient_clipping,
            max_gradient,
            max_norm,
            learning_rate,
            eps,
            initial_accumulator_value,
            beta1,
            beta2,
            weight_decay,
            eta,
            momentum,
            weight_decay_mode,
            counter_based_regularization,
            cowclip_regularization,
        )
        self._storage_externel = table_option.external_storage is not None
        self._create_cache_storage()
        self._initializers = []
        self._eval_initializers = []
        self._create_initializers()

        self._admission_counter = self._create_admission_counters(table_options)

        # TODO:1->10
        self._empty_tensor = nn.Parameter(
            torch.empty(
                10,
                requires_grad=True,
                device=torch.device(self.device_id),
                dtype=self.embedding_dtype,
            )
        )

    def _create_cache_storage(self) -> None:
        self._storages: List[Storage] = []
        self._caches: List[Cache] = []
        self._caching = self._dynamicemb_options[0].caching

        for option in self._dynamicemb_options:
            if option.training and option.optimizer_type == OptimizerType.Null:
                option.optimizer_type = convert_optimizer_type(self._optimizer_type)
            elif not option.training and option.optimizer_type != OptimizerType.Null:
                option.optimizer_type = OptimizerType.Null
                warnings.warn(
                    "Set OptimizerType to Null as not on training mode.", UserWarning
                )

            if option.caching and option.training:
                cache_option = deepcopy(option)
                cache_option.bucket_capacity = 1024
                capacity = get_constraint_capacity(
                    option.local_hbm_for_values,
                    option.embedding_dtype,
                    option.dim,
                    option.optimizer_type,
                    cache_option.bucket_capacity,
                )
                if capacity == 0:
                    raise ValueError(
                        "Can't use caching mode as the reserved HBM size is too small."
                    )

                cache_option.max_capacity = capacity
                cache_option.init_capacity = capacity
                self._caches.append(
                    DynamicEmbeddingTable(cache_option, self._optimizer)
                )

                storage_option = deepcopy(option)
                storage_option.local_hbm_for_values = 0
                PS = storage_option.external_storage
                self._storages.append(
                    PS(storage_option, self._optimizer)
                    if PS
                    else DynamicEmbeddingTable(storage_option, self._optimizer)
                )
            else:
                self._caches.append(None)
                self._storages.append(DynamicEmbeddingTable(option, self._optimizer))

        _print_memory_consume(
            self._table_names, self._dynamicemb_options, self._optimizer, self.device_id
        )

    def _create_initializers(self) -> None:
        for option in self._dynamicemb_options:
            initializer = create_initializer_from_args(option.initializer_args)
            self._initializers.append(initializer)
            eval_initializer = create_initializer_from_args(
                option.eval_initializer_args
            )
            self._eval_initializers.append(eval_initializer)

    def _create_admission_counters(
        self, table_options: List[DynamicEmbTableOptions]
    ) -> List[Optional["Counter"]]:
        """
        Create admission counters for each table.
        """
        counters = []
        device = torch.device(self.device_id)

        for option in table_options:
            counter = option.admission_counter

            if counter is None:
                counters.append(None)
            else:
                counters.append(counter.create(device=device))

        return counters

    def _create_optimizer(
        self,
        optimizer_type: EmbOptimType,
        stochastic_rounding: bool,
        gradient_clipping: bool,
        max_gradient: float,
        max_norm: float,
        learning_rate: float,
        eps: float,
        initial_accumulator_value: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        eta: float,
        momentum: float,
        weight_decay_mode: WeightDecayMode,
        counter_based_regularization: Optional[CounterBasedRegularizationDefinition],
        cowclip_regularization: Optional[CowClipDefinition],
    ) -> BaseDynamicEmbeddingOptimizer:
        self._optimizer_type = optimizer_type
        self.stochastic_rounding = stochastic_rounding

        self.weight_decay_mode = weight_decay_mode
        if (weight_decay_mode == WeightDecayMode.COUNTER) != (
            counter_based_regularization is not None
        ):
            raise AssertionError(
                "Need to set weight_decay_mode=WeightDecayMode.COUNTER together with valid counter_based_regularization"
            )
        if (weight_decay_mode == WeightDecayMode.COWCLIP) != (
            cowclip_regularization is not None
        ):
            raise AssertionError(
                "Need to set weight_decay_mode=WeightDecayMode.COWCLIP together with valid cowclip_regularization"
            )

        self._used_rowwise_adagrad_with_counter: bool = (
            optimizer_type == EmbOptimType.EXACT_ROWWISE_ADAGRAD
            and (
                weight_decay_mode in (WeightDecayMode.COUNTER, WeightDecayMode.COWCLIP)
            )
        )

        if counter_based_regularization is None:
            counter_based_regularization = CounterBasedRegularizationDefinition()
        if cowclip_regularization is None:
            cowclip_regularization = CowClipDefinition()
        self._max_counter_update_freq: int = -1
        # Extract parameters from CounterBasedRegularizationDefinition or CowClipDefinition
        # which are passed as entries for OptimizerArgs
        if self._used_rowwise_adagrad_with_counter:
            if self.weight_decay_mode == WeightDecayMode.COUNTER:
                self._max_counter_update_freq = (
                    counter_based_regularization.max_counter_update_freq
                )
                opt_arg_weight_decay_mode = (
                    counter_based_regularization.counter_weight_decay_mode
                )
                counter_halflife = counter_based_regularization.counter_halflife
            else:
                opt_arg_weight_decay_mode = (
                    cowclip_regularization.counter_weight_decay_mode
                )
                counter_halflife = cowclip_regularization.counter_halflife
        else:
            opt_arg_weight_decay_mode = weight_decay_mode
            # Default: -1, no decay applied, as a placeholder for OptimizerArgs
            # which should not be effective when CounterBasedRegularizationDefinition
            # and CowClipDefinition are not used
            counter_halflife = -1

        optimizer_args = OptimizerArgs(
            stochastic_rounding=stochastic_rounding,
            gradient_clipping=gradient_clipping,
            max_gradient=max_gradient,
            max_norm=max_norm,
            learning_rate=learning_rate,
            eps=eps,
            initial_accumulator_value=initial_accumulator_value,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            weight_decay_mode=opt_arg_weight_decay_mode.value,
            eta=eta,
            momentum=momentum,
            counter_halflife=counter_halflife,
            adjustment_iter=counter_based_regularization.adjustment_iter,
            adjustment_ub=counter_based_regularization.adjustment_ub,
            learning_rate_mode=counter_based_regularization.learning_rate_mode.value,
            grad_sum_decay=counter_based_regularization.grad_sum_decay.value,
            tail_id_threshold=counter_based_regularization.tail_id_threshold.val,
            is_tail_id_thresh_ratio=int(
                counter_based_regularization.tail_id_threshold.is_ratio
            ),
            total_hash_size=0,
            weight_norm_coefficient=cowclip_regularization.weight_norm_coefficient,
            lower_bound=cowclip_regularization.lower_bound,
            regularization_mode=weight_decay_mode.value,
        )
        self._optimizer_args = optimizer_args

        if optimizer_type == EmbOptimType.SGD:
            optimizer = SGDDynamicEmbeddingOptimizer(
                optimizer_args,
            )
        elif optimizer_type == EmbOptimType.EXACT_SGD:
            optimizer = SGDDynamicEmbeddingOptimizer(
                optimizer_args,
            )
        elif optimizer_type == EmbOptimType.ADAM:
            optimizer = AdamDynamicEmbeddingOptimizer(
                optimizer_args,
            )
        elif optimizer_type == EmbOptimType.EXACT_ADAGRAD:
            optimizer = AdaGradDynamicEmbeddingOptimizer(
                optimizer_args,
            )
        elif optimizer_type == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
            optimizer = RowWiseAdaGradDynamicEmbeddingOptimizer(
                optimizer_args,
                self.embedding_dtype,
            )
        else:
            raise ValueError(
                f"Not supported optimizer type ,optimizer type = {optimizer_type} {type(optimizer_type)} {optimizer_type.value}."
            )
        return optimizer

    def split_embedding_weights(self) -> List[Tensor]:
        """
        Returns a list of weights, split by table
        """
        splits = []
        for t, _ in enumerate(self._dynamicemb_options):
            splits.append(
                torch.empty(
                    (1, 1), device=torch.device("cuda"), dtype=self.embedding_dtype
                )
            )
        return splits

    def flush(self) -> None:
        self.num_prefetch_ahead = 0
        if self._caching:
            for cache, storage in zip(self._caches, self._storages):
                cache.flush(storage)

    def reset_cache_states(self) -> None:
        if self._caching:
            for cache in self._caches:
                cache.reset()

    @property
    def table_names(self) -> List[str]:
        return self._table_names

    @property
    def optimizer(
        self,
    ) -> BaseDynamicEmbeddingOptimizer:
        return self._optimizer

    @property
    def tables(self) -> List[DynamicEmbeddingTable]:
        # if use external PS, the users should not get the DynamicEmbeddingTables
        # if self._storage_externel:
        #     raise RuntimeError(
        #         "Should not get the internal tables when using external storage."
        #     )
        return self._storages

    @property
    def caches(self) -> List[Cache]:
        return self._caches

    def set_record_cache_metrics(self, record: bool) -> None:
        for cache in self._caches:
            cache.set_record_cache_metrics(record)

    def set_learning_rate(self, lr: float) -> None:
        self._optimizer.set_learning_rate(lr)

    @property
    def enable_prefetch(
        self,
    ) -> None:
        return self._enable_prefetch

    @enable_prefetch.setter
    def enable_prefetch(self, value: bool):
        self._enable_prefetch = value
        self.num_prefetch_ahead = 0

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
        # 2D tensor of batch size for each rank and feature.
        # Shape (number of features, number of ranks)
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
        total_unique_indices: Optional[int] = None,
    ) -> List[Tensor]:
        if self._enable_prefetch:
            self.num_prefetch_ahead -= 1

        if indices.dtype != self.index_type:
            indices = indices.to(self.index_type)

        if any([not o.training for o in self._dynamicemb_options]) and self.training:
            raise RuntimeError(
                "BatchedDynamicEmbeddingTables does not support training when some tables are in eval mode."
            )

        scores = []
        # if self.training:
        for table_name in self._table_names:
            if table_name not in self._scores.keys():
                raise RuntimeError(
                    f"Must set score for table '{table_name}' whose score_strategy is customized."
                )
            scores.append(self._scores[table_name])

        for i, cache in enumerate(self._caches):
            if isinstance(cache, DynamicEmbeddingTable):
                table = cast(DynamicEmbeddingTable, cache)
                table.score_update = True
                table.set_score(self._scores[self.table_names[i]])
        for i, storage in enumerate(self._storages):
            if isinstance(storage, DynamicEmbeddingTable):
                table = cast(DynamicEmbeddingTable, storage)
                # if not training and not caching, we don't need to update score.
                table.score_update = self.training or self._caching
                table.set_score(self._scores[self.table_names[i]])

        feature_batch_size = offsets.numel() - 1
        assert feature_batch_size > 0, "feature_batch_size must be greater than 0"
        assert (
            feature_batch_size % self.feature_num == 0
        ), "feature_batch_size must be divisible by feature_num"
        batch_size = (
            feature_batch_size // self.feature_num if self.feature_num > 0 else 0
        )

        res = DynamicEmbeddingFunction.apply(
            indices,
            offsets,
            self._caches,
            self._storages,
            self.feature_offsets,
            self.output_dtype,
            self._initializers if self.training else self._eval_initializers,
            self._optimizer,
            self._enable_prefetch,
            self.training,
            self._admit_strategy,
            self._evict_strategy,
            per_sample_weights,  # Pass frequency counters as weights
            self._admission_counter,
            self.pooling_mode,
            self.total_D,
            batch_size,
            self.dims,
            self.max_D,
            self.D_offsets_t,
            self._empty_tensor,
        )
        for cache in self._caches:
            if isinstance(cache, DynamicEmbeddingTable):
                table = cast(DynamicEmbeddingTable, cache)
                table.score_update = False
        for storage in self._storages:
            if isinstance(storage, DynamicEmbeddingTable):
                table = cast(DynamicEmbeddingTable, storage)
                table.score_update = False

        # We have to update cache's core in eval mode.
        if self.training or self._caching:
            self._update_score()

        return res

    def prefetch(
        self,
        indices: Tensor,
        offsets: Tensor,
        forward_stream: Optional[torch.cuda.Stream] = None,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
    ) -> None:
        assert self._enable_prefetch, "Prefetch is not enabled."
        if not self._caching:
            logging.warning("Caching is not enabled, prefetch will do nothing.")
        if self.prefetch_stream is None and forward_stream is not None:
            # Set the prefetch stream to the current stream
            self.prefetch_stream = torch.cuda.current_stream()
            assert (
                self.prefetch_stream != forward_stream
            ), "prefetch_stream and forward_stream should not be the same stream"

            current_stream = torch.cuda.current_stream()
            # Record tensors on the current stream
            indices.record_stream(current_stream)
            offsets.record_stream(current_stream)

        if self._enable_prefetch:
            self.num_prefetch_ahead += 1
            assert self.num_prefetch_ahead >= 1, "Prefetch context mismatches."

        prefetch_scores = self._get_prefetch_score()

        for i, cache in enumerate(self._caches):
            if isinstance(cache, DynamicEmbeddingTable):
                table = cast(DynamicEmbeddingTable, cache)
                table.score_update = True
                table.set_score(prefetch_scores[i])
        for i, storage in enumerate(self._storages):
            if isinstance(storage, DynamicEmbeddingTable):
                table = cast(DynamicEmbeddingTable, storage)
                table.score_update = True
                table.set_score(prefetch_scores[i])

        dynamicemb_prefetch(
            indices,
            offsets,
            self._caches,
            self._storages,
            self.feature_offsets,
            self._initializers if self.training else self._eval_initializers,
            self.training,
            forward_stream,
        )

        for cache in self._caches:
            if isinstance(cache, DynamicEmbeddingTable):
                table = cast(DynamicEmbeddingTable, cache)
                table.score_update = False
        for storage in self._storages:
            if isinstance(storage, DynamicEmbeddingTable):
                table = cast(DynamicEmbeddingTable, storage)
                table.score_update = False

    def set_score(
        self,
        named_score: Dict[str, int],
    ) -> None:
        table_names: List[str] = named_score.keys()
        table_scores: List[int] = named_score.values()
        for table_name, table_score in zip(table_names, table_scores):
            if not isinstance(table_score, int):
                raise ValueError(
                    f"Table's score is expect to int but got {type(table_score)}"
                )
            if table_score == 0:
                raise ValueError(f"Can't set table's score to 0.")
            index = self._table_names.index(table_name)
            assert (
                self._dynamicemb_options[index].score_strategy
                == DynamicEmbScoreStrategy.CUSTOMIZED
            ), "Can only set score for table whose score_strategy is DynamicEmbScoreStrategy.CUSTOMIZED."

            if table_name in self._scores and self._scores[table_name] > table_score:
                if warning_for_cstm_score():
                    warnings.warn(
                        f"New set score is less than the old one for table '{table_name}': {table_score} < {self._scores[table_name]}",
                        UserWarning,
                    )
            self._scores[table_name] = table_score

    def get_score(self) -> Dict[str, int]:
        return self._scores.copy()

    def _create_score(self):
        self._scores: Dict[str, int] = {}
        for table_name, option in zip(self._table_names, self._dynamicemb_options):
            if option.score_strategy == DynamicEmbScoreStrategy.TIMESTAMP:
                option.evict_strategy = DynamicEmbEvictStrategy.LRU
                self._scores[table_name] = device_timestamp()
            elif option.score_strategy == DynamicEmbScoreStrategy.STEP:
                option.evict_strategy = DynamicEmbEvictStrategy.CUSTOMIZED
                self._scores[table_name] = 1
            elif option.score_strategy == DynamicEmbScoreStrategy.CUSTOMIZED:
                option.evict_strategy = DynamicEmbEvictStrategy.CUSTOMIZED
            elif option.score_strategy == DynamicEmbScoreStrategy.LFU:
                option.evict_strategy = DynamicEmbEvictStrategy.LFU
                self._scores[table_name] = 1

    def _update_score(self):
        for table_name, option in zip(self._table_names, self._dynamicemb_options):
            old_score = self._scores[table_name]
            if option.score_strategy == DynamicEmbScoreStrategy.TIMESTAMP:
                new_score = device_timestamp()
                if new_score < old_score:
                    warnings.warn(
                        f"Table '{table_name}' 's score({new_score}) is less than old one({old_score}).",
                        UserWarning,
                    )
                self._scores[table_name] = new_score
            elif option.score_strategy == DynamicEmbScoreStrategy.STEP:
                max_uint64 = (2**64) - 1
                new_score = old_score + 1
                if new_score > max_uint64:
                    warnings.warn(
                        f"Table '{table_name}' 's score({new_score}) is out of range, reset to 0.",
                        UserWarning,
                    )
                    self._scores[table_name] = 0
                else:
                    self._scores[table_name] = new_score
            elif option.score_strategy == DynamicEmbScoreStrategy.LFU:
                self._scores[table_name] = 1

    def _get_prefetch_score(
        self,
    ):
        ret_scores = []
        for table_name, option in zip(self._table_names, self._dynamicemb_options):
            cur_score = self._scores[table_name]
            if (
                self.enable_prefetch
                and option.score_strategy == DynamicEmbScoreStrategy.STEP
            ):
                max_uint64 = (2**64) - 1
                new_score = cur_score + self.num_prefetch_ahead - 1
                if new_score > max_uint64:
                    warnings.warn(
                        f"Table '{table_name}' 's score({new_score}) is out of range, reset to 0.",
                        UserWarning,
                    )
                    new_score = 0
            else:
                new_score = cur_score

            ret_scores.append(new_score)
        return ret_scores

    def dump(
        self,
        save_dir: str,
        optim: bool = False,
        counter: bool = False,
        table_names: Optional[List[str]] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> None:
        if table_names is None:
            table_names = self._table_names

        if pg is None:
            assert dist.is_initialized(), "Distributed is not initialized."
            pg = dist.group.WORLD
        rank = dist.get_rank(group=pg)
        world_size = dist.get_world_size(group=pg)

        self.flush()
        for table_name, storage, counter_table in zip(
            self._table_names, self._storages, self._admission_counter
        ):
            if table_name not in set(table_names):
                continue

            meta_file_path = encode_meta_json_file_path(save_dir, table_name)
            emb_key_path = encode_checkpoint_file_path(
                save_dir, table_name, rank, world_size, "keys"
            )
            emb_value_path = encode_checkpoint_file_path(
                save_dir, table_name, rank, world_size, "values"
            )
            emb_score_path = encode_checkpoint_file_path(
                save_dir, table_name, rank, world_size, "scores"
            )
            opt_value_path = encode_checkpoint_file_path(
                save_dir, table_name, rank, world_size, "opt_values"
            )

            if isinstance(storage, DynamicEmbeddingTable) and not storage._use_score:
                dist.barrier()  # sync global timestamp
                cast(DynamicEmbeddingTable, storage).update_timestamp()

            current_score = (
                self._scores.get(table_name, None) if hasattr(self, "_scores") else None
            )

            storage.dump(
                meta_file_path,
                emb_key_path,
                emb_value_path,
                emb_score_path,
                opt_value_path,
                include_optim=optim,
                include_meta=(rank == 0),
                current_score=current_score,
            )

            if not counter:
                continue

            counter_key_path = encode_counter_checkpoint_file_path(
                save_dir, table_name, rank, world_size, "keys"
            )
            counter_frequency_path = encode_counter_checkpoint_file_path(
                save_dir, table_name, rank, world_size, "frequencies"
            )

            if counter_table is not None:
                counter_table.dump(counter_key_path, counter_frequency_path)
            else:
                warnings.warn(
                    f"Counter table is none and will not dump it for table: {table_name}"
                )

    def load(
        self,
        save_dir: str,
        optim: bool = False,
        counter: bool = False,
        table_names: Optional[List[str]] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ):
        if table_names is None:
            table_names = self._table_names

        if pg is None and not dist.is_initialized():  # for inference load
            rank = 0
            world_size = 1
        else:
            rank = dist.get_rank(group=pg)
            world_size = dist.get_world_size(group=pg)

        for table_name, storage, counter_table in zip(
            self._table_names, self._storages, self._admission_counter
        ):
            if table_name not in set(table_names):
                continue
            (
                emb_key_files,
                emb_value_files,
                emb_score_files,
                opt_value_files,
                counter_key_files,
                counter_frequency_files,
            ) = get_loading_files(
                save_dir,
                table_name,
                rank=rank,
                world_size=world_size,
            )
            meta_json_file = encode_meta_json_file_path(save_dir, table_name)

            if isinstance(storage, DynamicEmbeddingTable) and not storage._use_score:
                dist.barrier()  # sync global timestamp
                cast(DynamicEmbeddingTable, storage).update_timestamp()
            num_key_files = len(emb_key_files)
            for i in range(num_key_files):
                loaded_score = storage.load(
                    meta_json_file,
                    emb_key_files[i],
                    emb_value_files[i],
                    emb_score_files[i] if len(emb_score_files) > 0 else None,
                    opt_value_files[i] if len(opt_value_files) > 0 else None,
                    include_optim=optim,
                )
                if loaded_score is not None and table_name in self._scores:
                    self._scores[table_name] = loaded_score

            if not counter:
                continue
            if counter_table is None:
                warnings.warn(
                    f"Counter table is none and will not load for table: {table_name}"
                )
                continue
            num_counter_key_files = len(counter_key_files)
            for i in range(num_counter_key_files):
                counter_table.load(counter_key_files[i], counter_frequency_files[i])

    def export_keys_values(
        self, table_name: str, device: torch.device, batch_size: int = 65536
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        keys_list = []
        values_list = []
        self.flush()
        for dynamic_table_name, dynamic_table in zip(self.table_names, self.tables):
            assert isinstance(
                dynamic_table, DynamicEmbeddingTable
            ), "Only DynamicEmbeddingTable is supported for batched export keys and values"
            if table_name != dynamic_table_name:
                continue

            local_max_rows = dynamic_table.size()
            accumulated_counts = 0

            for keys, embeddings, _, _ in dynamic_table.export_keys_values(
                device, batch_size
            ):
                keys_list.append(keys)
                values_list.append(embeddings)
                accumulated_counts += keys.numel()

            if local_max_rows != accumulated_counts:
                raise ValueError(
                    f"Rank {dist.get_rank()} has accumulated count {accumulated_counts} which is different from expected {local_max_rows}, "
                    f"difference: {accumulated_counts - local_max_rows}"
                )
        return torch.cat(keys_list), torch.cat(values_list, dim=0)

    def incremental_dump(
        self,
        named_thresholds: Dict[str, int] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> Tuple[Dict[str, Tuple[Tensor, Tensor]], Dict[str, int]]:
        table_names: List[str] = named_thresholds.keys()
        table_thresholds: List[int] = named_thresholds.values()
        ret_tensors: Dict[str, Tuple[Tensor, Tensor]] = {}
        ret_scores: Dict[str, int] = {}

        for table_name, threshold in zip(table_names, table_thresholds):
            index = self._table_names.index(table_name)

            storage = self._storages[index]
            if not isinstance(storage, DynamicEmbeddingTable):
                raise RuntimeError(
                    "Only DynamicEmbeddingTable support incremental dump."
                )

            cache = self._caches[index]
            if cache is not None:
                cache.flush(storage)

            key, value = storage.incremental_dump(threshold, pg)

            ret_tensors[table_name] = (key, value)
            ret_scores[table_name] = self._scores[table_name]

        return ret_tensors, ret_scores
