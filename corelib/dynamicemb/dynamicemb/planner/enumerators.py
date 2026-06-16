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

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.constants import POOLING_FACTOR
from torchrec.distributed.planner.enumerators import (
    GUARDED_COMPUTE_KERNELS,
    EmbeddingEnumerator,
    get_partition_by_type,
)
from torchrec.distributed.planner.types import (
    Shard,
    ShardEstimator,
    ShardingOption,
    Topology,
)
from torchrec.distributed.planner.utils import sharder_name
from torchrec.distributed.sharding_plan import (
    _calculate_cw_shard_sizes_and_offsets,
    _calculate_uneven_rw_shard_sizes_and_offsets,
)
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheParams,
    KeyValueParams,
    ModuleSharder,
    ShardingType,
)
from torchrec.modules.embedding_configs import DataType
from torchrec.modules.embedding_tower import EmbeddingTower, EmbeddingTowerCollection

from .planner import DynamicEmbParameterConstraints

logger: logging.Logger = logging.getLogger(__name__)

BATCH_SIZE: int = 512


def _extract_constraints_for_param(
    constraints: Optional[Dict[str, DynamicEmbParameterConstraints]], name: str
) -> Tuple[
    bool,
    List[float],
    Optional[int],
    Optional[CacheParams],
    Optional[bool],
    Optional[bool],
    Optional[BoundsCheckMode],
    Optional[List[str]],
    Optional[DataType],
    Optional[str],
    Optional[KeyValueParams],
]:
    use_dynamicemb = False
    input_lengths = [POOLING_FACTOR]
    col_wise_shard_dim = None
    cache_params = None
    enforce_hbm = None
    stochastic_rounding = None
    bounds_check_mode = None
    feature_names = None
    output_dtype = None
    device_group = None
    key_value_params = None

    if constraints and constraints.get(name):
        use_dynamicemb = constraints[name].use_dynamicemb
        input_lengths = constraints[name].pooling_factors
        col_wise_shard_dim = constraints[name].min_partition
        cache_params = constraints[name].cache_params
        enforce_hbm = constraints[name].enforce_hbm
        stochastic_rounding = constraints[name].stochastic_rounding
        bounds_check_mode = constraints[name].bounds_check_mode
        feature_names = constraints[name].feature_names
        output_dtype = constraints[name].output_dtype
        device_group = constraints[name].device_group
        key_value_params = constraints[name].key_value_params

    return (
        use_dynamicemb,
        input_lengths,
        col_wise_shard_dim,
        cache_params,
        enforce_hbm,
        stochastic_rounding,
        bounds_check_mode,
        feature_names,
        output_dtype,
        device_group,
        key_value_params,
    )


def _calculate_rw_shard_sizes_and_offsets(
    hash_size: int, num_devices: int, columns: int
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Sets prefix of shard_sizes to be `math.ceil(hash_size/num_devices)`.

    For example if hash_size = 10, num_devices = 4, we will allocate the rows as 3,3,3,1
    (rather than 3,3,2,2).
    This is due to implementation in RW sharding that sets block_size_lists to be ceil.
    The balanced way is harder to support on GPU.
    For more details see https://fb.quip.com/xbgbAchCTOL0

    Also consider the example of hash_size = 5, num_devices = 4. The expected rows per
    rank is [2,2,1,0].
    """

    block_size: int = math.ceil(hash_size / num_devices)
    last_rank: int = hash_size // block_size
    last_block_size: int = hash_size - block_size * last_rank
    shard_sizes: List[List[int]] = []

    for rank in range(num_devices):
        if rank < last_rank:
            local_row: int = block_size
        elif rank == last_rank:
            local_row: int = last_block_size
        else:
            local_row: int = 0
        shard_sizes.append([local_row, columns])
    shard_offsets = [[0, 0]]

    for i in range(num_devices - 1):
        shard_offsets.append([shard_sizes[i][0] + shard_offsets[i][0], 0])

    return shard_sizes, shard_offsets


def calculate_shard_sizes_and_offsets(
    tensor: torch.Tensor,
    world_size: int,
    local_world_size: int,
    sharding_type: str,
    use_dynamicemb: bool,
    col_wise_shard_dim: Optional[int] = None,
    device_memory_sizes: Optional[List[int]] = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Calculates sizes and offsets for tensor sharded according to provided sharding type.

    Args:
        tensor (torch.Tensor): tensor to be sharded.
        world_size (int): total number of devices in topology.
        local_world_size (int): total number of devices in host group topology.
        sharding_type (str): provided ShardingType value.
        col_wise_shard_dim (Optional[int]): dimension for column wise sharding split.

    Returns:
        Tuple[List[List[int]], List[List[int]]]: shard sizes, represented as a list of the dimensions of the sharded tensor on each device, and shard offsets, represented as a list of coordinates of placement on each device.

    Raises:
        ValueError: If `sharding_type` is not a valid ShardingType.
    """

    if use_dynamicemb:
        sizes = [[1, 1]] * world_size
        offsets = [[i, 0] for i in range(world_size)]
        return sizes, offsets

    (rows, columns) = tensor.shape

    if sharding_type == ShardingType.DATA_PARALLEL.value:
        return [[rows, columns]] * world_size, [[0, 0]] * world_size
    elif sharding_type == ShardingType.TABLE_WISE.value:
        return [[rows, columns]], [[0, 0]]
    elif sharding_type == ShardingType.ROW_WISE.value:
        return (
            _calculate_rw_shard_sizes_and_offsets(rows, world_size, columns)
            if not device_memory_sizes
            else _calculate_uneven_rw_shard_sizes_and_offsets(
                rows, world_size, columns, device_memory_sizes
            )
        )
    elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
        return _calculate_rw_shard_sizes_and_offsets(rows, local_world_size, columns)
    elif (
        sharding_type == ShardingType.COLUMN_WISE.value
        or sharding_type == ShardingType.TABLE_COLUMN_WISE.value
    ):
        return _calculate_cw_shard_sizes_and_offsets(columns, rows, col_wise_shard_dim)

    raise ValueError(
        f"Unrecognized or unsupported sharding type provided: {sharding_type}"
    )


class DynamicEmbeddingEnumerator(EmbeddingEnumerator):
    def __init__(
        self,
        topology: Topology,
        batch_size: Optional[int] = BATCH_SIZE,
        # TODO:check the input type is DynamicEmbParameterConstraints or ParameterConstraints
        constraints: Optional[Dict[str, DynamicEmbParameterConstraints]] = None,
        estimator: Optional[Union[ShardEstimator, List[ShardEstimator]]] = None,
        use_exact_enumerate_order: Optional[bool] = False,
    ) -> None:
        """
        DynamicEmbeddingEnumerator extends the EmbeddingEnumerator to handle dynamic embedding tables.

        Parameters
        ----------
        topology : Topology
            The topology of the GPU and Host memory.
        batch_size : Optional[int], optional
            The batch size for training. Defaults to BATCH_SIZE.
            The creation and usage are consistent with the same types in TorchREC.
        constraints : Optional[Dict[str, DynamicEmbParameterConstraints]], optional
            A dictionary of constraints for the parameters. Defaults to None.
        estimator : Optional[Union[ShardEstimator, List[ShardEstimator]]], optional
            An estimator or a list of estimators for estimating shard sizes. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        use_exact_enumerate_order (bool): whether to enumerate shardable parameters in the exact name_children enumeration order
        """
        super().__init__(
            topology, batch_size, constraints, estimator, use_exact_enumerate_order
        )
        self._constraints = constraints

    def enumerate(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> List[ShardingOption]:
        self._sharder_map = {
            sharder_name(sharder.module_type): sharder for sharder in sharders
        }
        sharding_options: List[ShardingOption] = []

        named_modules_queue = [("", module)]
        while named_modules_queue:
            if not self._use_exact_enumerate_order:
                child_path, child_module = named_modules_queue.pop()
            else:
                child_path, child_module = named_modules_queue.pop(0)
            sharder_key = sharder_name(type(child_module))
            sharder = self._sharder_map.get(sharder_key, None)
            if not sharder:
                for n, m in child_module.named_children():
                    if child_path != "":
                        named_modules_queue.append((child_path + "." + n, m))
                    else:
                        named_modules_queue.append((n, m))
                continue

            # Determine the pooling state for all sharding_options using this
            # (child_module, child_path). With this optimization, we change enumerate()
            # from being O(N^2) with respect to the number of tables to O(N). The
            # previous quadratic behavior is because in populate_estimates() invoked below, each
            # sharding_option needs to determine its pooling state, which is does via
            # an expensive O(N) walk through the list of embedding tables. With this
            # change sharding_option.is_pooled becomes O(1).
            is_pooled = ShardingOption.module_pooled(child_module, child_path)

            for name, param in sharder.shardable_parameters(child_module).items():
                (
                    use_dynamicemb,
                    input_lengths,
                    col_wise_shard_dim,
                    cache_params,
                    enforce_hbm,
                    stochastic_rounding,
                    bounds_check_mode,
                    feature_names,
                    output_dtype,
                    device_group,
                    key_value_params,
                ) = _extract_constraints_for_param(self._constraints, name)

                # skip for other device groups
                if device_group and device_group != self._compute_device:
                    continue

                sharding_options_per_table: List[ShardingOption] = []

                for sharding_type in self._filter_sharding_types(
                    name, sharder.sharding_types(self._compute_device), use_dynamicemb
                ):
                    for compute_kernel in self._filter_compute_kernels(
                        name,
                        sharder.compute_kernels(sharding_type, self._compute_device),
                        sharding_type,
                        use_dynamicemb,
                    ):
                        (
                            shard_sizes,
                            shard_offsets,
                        ) = calculate_shard_sizes_and_offsets(
                            tensor=param,
                            world_size=self._world_size,
                            local_world_size=self._local_world_size,
                            sharding_type=sharding_type,
                            use_dynamicemb=use_dynamicemb,
                            col_wise_shard_dim=col_wise_shard_dim,
                            device_memory_sizes=self._device_memory_sizes,
                        )
                        dependency = None
                        if isinstance(child_module, EmbeddingTower):
                            dependency = child_path
                        elif isinstance(child_module, EmbeddingTowerCollection):
                            raise RuntimeError("please revisit this logic")
                            # tower_index = _get_tower_index(name, child_module)
                            # dependency = child_path + ".tower_" + str(tower_index)
                        sharding_options_per_table.append(
                            ShardingOption(
                                name=name,
                                tensor=param,
                                module=(child_path, child_module),
                                input_lengths=input_lengths,
                                batch_size=self._batch_size,
                                compute_kernel=compute_kernel,
                                sharding_type=sharding_type,
                                partition_by=get_partition_by_type(sharding_type),
                                shards=[
                                    Shard(size=size, offset=offset)
                                    for size, offset in zip(shard_sizes, shard_offsets)
                                ],
                                cache_params=cache_params,
                                enforce_hbm=enforce_hbm,
                                stochastic_rounding=stochastic_rounding,
                                bounds_check_mode=bounds_check_mode,
                                dependency=dependency,
                                is_pooled=is_pooled,
                                feature_names=feature_names,
                                output_dtype=output_dtype,
                                key_value_params=key_value_params,
                            )
                        )
                if not sharding_options_per_table:
                    raise RuntimeError(
                        "No available sharding type and compute kernel combination "
                        f"after applying user provided constraints for {name}. "
                        f"Module: {sharder_key}, sharder: {sharder.__class__.__name__}, compute device: {self._compute_device}. "
                        f"To debug, search above for warning logs about no available sharding types/compute kernels for table: {name}"
                    )

                sharding_options.extend(sharding_options_per_table)

        self.populate_estimates(sharding_options)

        return sharding_options

    def _filter_sharding_types(
        self, name: str, allowed_sharding_types: List[str], use_dynamicemb: bool
    ) -> List[str]:
        if use_dynamicemb:
            return [ShardingType.ROW_WISE.value]
        if not self._constraints or not self._constraints.get(name):
            return allowed_sharding_types
        constraints: DynamicEmbParameterConstraints = self._constraints[name]
        if not constraints.sharding_types:
            return allowed_sharding_types
        constrained_sharding_types: List[str] = constraints.sharding_types

        filtered_sharding_types = list(
            set(constrained_sharding_types) & set(allowed_sharding_types)
        )

        if not filtered_sharding_types:
            logger.warn(
                "No available sharding types after applying user provided "
                f"constraints for {name}. Constrained sharding types: "
                f"{constrained_sharding_types}, allowed sharding types: "
                f"{allowed_sharding_types}, filtered sharding types: "
                f"{filtered_sharding_types}. Please check if the constrained "
                "sharding types are too restrictive, if the sharder allows the "
                "sharding types, or if non-strings are passed in."
            )
        return filtered_sharding_types

    def _filter_compute_kernels(
        self,
        name: str,
        allowed_compute_kernels: List[str],
        sharding_type: str,
        use_dynamicemb: bool,
    ) -> List[str]:
        if use_dynamicemb:
            return [EmbeddingComputeKernel.FUSED.value]

        # setup constrained_compute_kernels
        if (
            self._constraints
            and self._constraints.get(name)
            and self._constraints[name].compute_kernels
        ):
            # pyre-ignore
            constrained_compute_kernels: List[str] = self._constraints[
                name
            ].compute_kernels
        else:
            constrained_compute_kernels: List[str] = [
                compute_kernel.value
                for compute_kernel in EmbeddingComputeKernel
                if compute_kernel not in GUARDED_COMPUTE_KERNELS
            ]

        # setup filtered_compute_kernels
        filtered_compute_kernels = list(
            set(constrained_compute_kernels) & set(allowed_compute_kernels)
        )

        # special rules
        if EmbeddingComputeKernel.DENSE.value in filtered_compute_kernels:
            if (
                EmbeddingComputeKernel.FUSED.value in filtered_compute_kernels
            ):  # always false for data_parallel
                filtered_compute_kernels.remove(EmbeddingComputeKernel.DENSE.value)

        if not filtered_compute_kernels:
            logger.warn(
                "No available compute kernels after applying user provided "
                f"constraints for {name}. Constrained compute kernels: "
                f"{constrained_compute_kernels}, allowed compute kernels: "
                f"{allowed_compute_kernels}, filtered compute kernels: "
                f"{filtered_compute_kernels}, sharding type: {sharding_type}. Please check if the constrained "
                "compute kernels are too restrictive, if the sharder allows the "
                "compute kernels, or if non-strings are passed in."
            )
        return filtered_compute_kernels
