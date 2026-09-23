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
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.types import (
    ShardEstimator,
    ShardingOption,
    Topology,
)
from torchrec.distributed.types import ModuleSharder

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

    def _use_dynamicemb(self, name: str) -> bool:
        """Whether *name* is a DynamicEmb table.

        Read off the constraint rather than passed in: TorchRec hands the hooks
        the table's name, and the constraint is reachable from it.
        """
        constraint = self._constraints.get(name) if self._constraints else None
        return bool(getattr(constraint, "use_dynamicemb", False))

    def enumerate(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> List[ShardingOption]:
        """TorchRec's search space, with the DynamicEmb tables taken out of it.

        They are not TorchRec's to place. A DynamicEmb table's capacity comes
        from ``DynamicEmbTableOptions`` rather than from ``num_embeddings``,
        which is nominal for a hash table, so anything the estimators cost here
        is a number about a table that does not exist. Its sharding and its
        ranks are decided by ``DynamicEmbeddingShardingPlanner``, which writes
        the ParameterSharding itself and takes what the tables spend out of the
        Topology -- so the TorchRec planner plans the remaining tables into what
        is left.

        Leaving them in would mean carrying a placeholder through the search
        space with no way to make it honest: sized truthfully it would be
        partitioned on numbers we do not believe, and sized at nothing it would
        be a ghost that costs nothing and can be placed anywhere.
        """
        return [
            option
            for option in super().enumerate(module, sharders)
            if not self._use_dynamicemb(option.name)
        ]
