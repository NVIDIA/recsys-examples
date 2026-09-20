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

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.enumerators import (
    EmbeddingEnumerator,
    _extract_constraints_for_param,
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
    calculate_shard_sizes_and_offsets as _calculate_shard_sizes_and_offsets,
)
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_tower import EmbeddingTower, EmbeddingTowerCollection

from .planners import DynamicEmbParameterConstraints

BATCH_SIZE: int = 512


def calculate_shard_sizes_and_offsets(
    tensor: torch.Tensor,
    world_size: int,
    local_world_size: int,
    sharding_type: str,
    use_dynamicemb: bool,
    col_wise_shard_dim: Optional[int] = None,
    device_memory_sizes: Optional[List[int]] = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    """TorchRec's, plus the one shape it has no way to describe.

    A DynamicEmb table has no rows to divide. Its ``num_embeddings`` is a
    nominal figure, the storage is a hash table sized by
    ``DynamicEmbTableOptions``, and which rank holds a key is decided by
    ``dist_type`` rather than by any row range. So there is nothing for the
    row-wise arithmetic to compute, and the 1x1 shard per rank stands in for
    it -- one entry per rank so the tensor still looks sharded to the planner,
    sized so it is costed as the near-nothing it is. ``get_state_dict`` in
    batched_dynamicemb_compute_kernel.py writes the same shape for the same
    reason.

    Every other table goes to TorchRec unchanged.
    """
    if use_dynamicemb:
        sizes = [[1, 1]] * world_size
        offsets = [[i, 0] for i in range(world_size)]
        return sizes, offsets

    return _calculate_shard_sizes_and_offsets(
        tensor=tensor,
        world_size=world_size,
        local_world_size=local_world_size,
        sharding_type=sharding_type,
        col_wise_shard_dim=col_wise_shard_dim,
        device_memory_sizes=device_memory_sizes,
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
                    name, sharder.sharding_types(self._compute_device), sharder_key
                ):
                    for compute_kernel in self._filter_compute_kernels(
                        name,
                        sharder.compute_kernels(sharding_type, self._compute_device),
                        sharding_type,
                    ):
                        (
                            shard_sizes,
                            shard_offsets,
                        ) = calculate_shard_sizes_and_offsets(
                            tensor=param,
                            world_size=self._world_size,
                            local_world_size=self._local_world_size,
                            sharding_type=sharding_type,
                            use_dynamicemb=self._use_dynamicemb(name),
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

    def _use_dynamicemb(self, name: str) -> bool:
        """Whether *name* is a DynamicEmb table.

        Read off the constraint rather than passed in: the two hooks below are
        the only things that need it, TorchRec already hands them the table's
        name, and threading it down instead would mean forking ``enumerate``
        to carry it -- which is what this used to do.
        """
        constraint = self._constraints.get(name) if self._constraints else None
        return bool(getattr(constraint, "use_dynamicemb", False))

    def _filter_sharding_types(
        self, name: str, allowed_sharding_types: List[str], sharder_key: str = ""
    ) -> List[str]:
        # A DynamicEmb table is row-wise and nothing else, so the rest of the
        # search space is not applicable rather than merely unattractive.
        if self._use_dynamicemb(name):
            return [ShardingType.ROW_WISE.value]
        return super()._filter_sharding_types(
            name, allowed_sharding_types, sharder_key
        )

    def _filter_compute_kernels(
        self,
        name: str,
        allowed_compute_kernels: List[str],
        sharding_type: str,
    ) -> List[str]:
        # FUSED is a placeholder that keeps the table in the search space;
        # DynamicEmbeddingShardingPlanner replaces the whole ParameterSharding,
        # CUSTOMIZED_KERNEL included, once planning is done.
        if self._use_dynamicemb(name):
            return [EmbeddingComputeKernel.FUSED.value]
        return super()._filter_compute_kernels(
            name, allowed_compute_kernels, sharding_type
        )
