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

from typing import Dict, List, Optional, Union

from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.types import (
    Shard,
    ShardEstimator,
    ShardingOption,
    Topology,
)
from torchrec.distributed.types import ModuleSharder, ShardingType

from .planners import DynamicEmbParameterConstraints

BATCH_SIZE: int = 512


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
        """TorchRec's search space, with the DynamicEmb tables' shards replaced.

        A DynamicEmb table has no rows to divide: its ``num_embeddings`` is
        nominal, its storage is a hash table sized by ``DynamicEmbTableOptions``,
        and ``DynamicEmbeddingShardingPlanner`` replaces its ParameterSharding
        outright once planning is done. The planner still has to see it -- the
        plan is keyed by module, and dropping the table drops the module with it
        -- so it is sized as the near-nothing it costs the planner: one 1x1
        shard per rank.

        Done after the fact rather than inside the walk, because the sizing
        TorchRec calls is a module function with no seam to override. The walk
        itself is where everything else lives -- memoization, the sharder data
        map the estimators index into, ZCH bucket sizes, weight stashing -- and
        a copy of it here is a copy that stops receiving all of that.

        Re-estimating is not optional: ``super().enumerate`` costed these
        options from their real row counts before this rewrote them.
        """
        sharding_options = super().enumerate(module, sharders)
        dynamicemb_options = [
            option for option in sharding_options if self._use_dynamicemb(option.name)
        ]
        for option in dynamicemb_options:
            option.shards = [
                Shard(size=[1, 1], offset=[rank, 0]) for rank in range(self._world_size)
            ]
        if dynamicemb_options:
            self.populate_estimates(dynamicemb_options)
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
        return super()._filter_sharding_types(name, allowed_sharding_types, sharder_key)

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
