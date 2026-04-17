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

import math
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Union

import torch
from torch import distributed as dist
from torch import nn
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import EmbeddingShardingPlanner, ParameterConstraints
from torchrec.distributed.planner.types import (
    Enumerator,
    Partitioner,
    PerfModel,
    Proposer,
    Stats,
    StorageReservation,
    Topology,
)
from torchrec.distributed.sharding_plan import placement
from torchrec.distributed.types import (
    EnumerableShardingSpec,
    ModuleSharder,
    ParameterSharding,
    ShardingPlan,
    ShardingType,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import BaseEmbeddingConfig, data_type_to_dtype

from ..dynamicemb_config import (
    DEFAULT_INDEX_TYPE,
    DynamicEmbKernel,
    DynamicEmbTableOptions,
    _sharded_table_bucket_layout,
    align_to_table_size,
    complete_initializer_args,
)

HBM_CAP: int = 32 * 1024 * 1024 * 1024
DDR_CAP: int = 128 * 1024 * 1024 * 1024
GB: int = 1024 * 1024 * 1024 * 1024


@dataclass
class DynamicEmbParameterConstraints(ParameterConstraints):
    """
    DynamicEmb-specific parameter constraints that extend ParameterConstraints.

    Attributes
    ----------
    use_dynamicemb : Optional[bool]
        A flag indicating whether to use DynamicEmb storage. Defaults to False.
    dynamicemb_options : Optional[DynamicEmbTableOptions]
        Configuration for the dynamic embedding table, including initializer args. The initialization method for the parameters.
        Common choices include "uniform", "normal", etc. Defaults to "uniform".
    """

    use_dynamicemb: Optional[bool] = False
    dynamicemb_options: Optional[DynamicEmbTableOptions] = field(
        default_factory=DynamicEmbTableOptions
    )


@dataclass
class DynamicEmbParameterSharding(ParameterSharding):
    """
    DynamicEmb-specific parameter constraints that extend ParameterSharding.
    """

    # provide a default value for compute_kernel
    compute_kernel: str = EmbeddingComputeKernel.CUSTOMIZED_KERNEL

    # introduced fields by DynamicEmbParameterSharding
    customized_compute_kernel: Optional[str] = DynamicEmbKernel
    dist_type: str = "continuous"
    dynamicemb_options: Optional[DynamicEmbTableOptions] = field(
        default_factory=DynamicEmbTableOptions
    )

    def get_additional_fused_params(self):
        all_fields = {
            f.name: getattr(self, f.name) for f in fields(DynamicEmbParameterSharding)
        }
        parameter_sharding_fields = {
            f.name: getattr(self, f.name) for f in fields(ParameterSharding)
        }
        return {
            k: v for k, v in all_fields.items() if k not in parameter_sharding_fields
        }

    @staticmethod
    def pop_additional_fused_params(fused_params: Dict[str, Any]) -> None:
        """Remove DynamicEmb-only keys from ``GroupedEmbeddingConfig.fused_params`` before
        :class:`~dynamicemb.batched_dynamicemb_tables.BatchedDynamicEmbeddingTablesV2`.

        These entries are used for planning / per-table options and are not valid ``**fused_params``
        for that module.
        """
        for f in (
            "customized_compute_kernel",
            "dist_type",
            "dynamicemb_options",
        ):
            fused_params.pop(f, None)


def _prepare_dynemb_table_options(
    constraints: Dict[str, DynamicEmbParameterConstraints],
    eb_configs: List[BaseEmbeddingConfig],
):
    """Check ``constraints`` ↔ ``eb_configs`` naming, then fill per-table DynamicEmb options.

    For each DynamicEmb table: ``complete_initializer_args``, then
    ``_sharded_table_bucket_layout`` (sets effective ``bucket_capacity`` and per-rank
    ``max_capacity``), then ``local_hbm_for_values`` from
    ``global_hbm_for_values`` and world size; then default ``index_type`` / ``embedding_dtype``,
    align ``init_capacity`` to the effective ``bucket_capacity`` when set; if aligned
    ``init_capacity`` exceeds ``max_capacity``, clamp it to ``max_capacity``; if ``init_capacity``
    was unset, set it to ``max_capacity``; set ``dim`` from ``BaseEmbeddingConfig.embedding_dim``.
    """
    world_size = dist.get_world_size()
    if constraints is None or eb_configs is None:
        raise ValueError("Constraints and eb_configs must not be None")

    # Extract names from eb_configs
    config_names = [config.name for config in eb_configs]

    # Check if each BaseEmbeddingConfig's name matches the keys in the constraints dictionary
    for config_name in config_names:
        if config_name not in constraints:
            raise ValueError(
                f"Config name '{config_name}' does not match any key in constraints"
            )

    # Verify that each BaseEmbeddingConfig name is unique
    if len(set(config_names)) != len(config_names):
        raise ValueError("Config names must be unique")

    # Ensure that all constraints keys have corresponding BaseEmbeddingConfig with matching name
    if set(config_names) != set(constraints.keys()):
        raise ValueError(
            "Not all constraint keys have matching BaseEmbeddingConfig names"
        )

    for i, config_name in enumerate(config_names):
        tmp_constraint = constraints[config_name]
        if not tmp_constraint.use_dynamicemb:
            continue

        tmp_config = eb_configs[i]
        opts = tmp_constraint.dynamicemb_options

        opts.initializer_args = complete_initializer_args(
            opts.initializer_args,
            embedding_config=tmp_config,
        )
        num_buckets, effective_bucket_capacity = _sharded_table_bucket_layout(
            tmp_config,
            world_size,
            opts.bucket_capacity,
        )
        opts.bucket_capacity = effective_bucket_capacity
        aligned_per_rank_rows = num_buckets * effective_bucket_capacity
        opts.max_capacity = aligned_per_rank_rows
        opts.local_hbm_for_values = math.ceil(opts.global_hbm_for_values / world_size)

        if opts.init_capacity is not None:
            aligned_init = align_to_table_size(
                opts.init_capacity, effective_bucket_capacity
            )
            if aligned_init != opts.init_capacity:
                warnings.warn(
                    f"init_capacity is aligned to {aligned_init} from {opts.init_capacity} "
                    f"(bucket_capacity={effective_bucket_capacity})",
                    UserWarning,
                )
            opts.init_capacity = aligned_init
            if opts.init_capacity > opts.max_capacity:
                warnings.warn(
                    f"init_capacity {opts.init_capacity} exceeds max_capacity {opts.max_capacity}; "
                    f"clamping init_capacity to max_capacity",
                    UserWarning,
                )
                opts.init_capacity = opts.max_capacity
        else:
            opts.init_capacity = opts.max_capacity

        opts.dim = tmp_config.embedding_dim

        if opts.index_type is None:
            opts.index_type = DEFAULT_INDEX_TYPE
        if opts.embedding_dtype is None:
            opts.embedding_dtype = data_type_to_dtype(tmp_config.data_type)


class DynamicEmbeddingShardingPlanner:
    def __init__(
        self,
        eb_configs: List[BaseEmbeddingConfig],
        topology: Optional[Topology] = None,
        batch_size: Optional[int] = None,
        enumerator: Optional[Enumerator] = None,
        storage_reservation: Optional[StorageReservation] = None,
        proposer: Optional[Union[Proposer, List[Proposer]]] = None,
        partitioner: Optional[Partitioner] = None,
        performance_model: Optional[PerfModel] = None,
        stats: Optional[Union[Stats, List[Stats]]] = None,
        constraints: Optional[Dict[str, DynamicEmbParameterConstraints]] = None,
        debug: bool = True,
    ):
        """
        DynamicEmbeddingShardingPlanner wraps the API of EmbeddingShardingPlanner from the Torchrec repo,
        giving it the ability to plan dynamic embedding tables. The only difference from EmbeddingShardingPlanner
        is that DynamicEmbeddingShardingPlanner has an additional parameter `eb_configs`, which is a list of
        TorchREC BaseEmbeddingConfig. Per-rank table options are filled in ``_prepare_dynemb_table_options``
        (initializer bounds, sharded table capacity via ``_sharded_table_bucket_layout``, and per-rank HBM budget).

        Parameters
        ----------
        eb_configs : List[BaseEmbeddingConfig]
            A list of TorchREC BaseEmbeddingConfig in the TorchREC model
        topology : Optional[Topology], optional
            The topology of GPU and Host memory. If None, a default topology will be created. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
            Note: The memory budget does not include the consumption of dynamicemb.
        batch_size : Optional[int], optional
            The batch size for training. Defaults to None, will set 512 in Planner.
        enumerator : Optional[Enumerator], optional
            An enumerator for sharding. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        storage_reservation : Optional[StorageReservation], optional
            Storage reservation details. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        proposer : Optional[Union[Proposer, List[Proposer]]], optional
            A proposer or a list of proposers for proposing sharding plans. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        partitioner : Optional[Partitioner], optional
            A partitioner for partitioning the embedding tables. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        performance_model : Optional[PerfModel], optional
            A performance model for evaluating sharding plans. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        stats : Optional[Union[Stats, List[Stats]]], optional
            Statistics or a list of statistics for the sharding process. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        constraints : Optional[Dict[str, DynamicEmbParameterConstraints]], optional
            A dictionary of constraints for every TorchREC embedding table and Dynamic embedding table. Defaults to None.
        debug : bool, optional
            A flag indicating whether to enable debug mode. Defaults to True.
        """

        super(DynamicEmbeddingShardingPlanner, self).__init__()

        _prepare_dynemb_table_options(constraints, eb_configs)

        dyn_emb_table_consts = {
            key: constraint
            for key, constraint in constraints.items()
            if constraint.use_dynamicemb
        }
        torchrec_tables_consts = {
            key: constraint
            for key, constraint in constraints.items()
            if not constraint.use_dynamicemb
        }
        dyn_emb_table_eb_configs = {
            config.name: config
            for config in eb_configs
            if constraints.get(config.name) and constraints[config.name].use_dynamicemb
        }

        if topology is None:
            warnings.warn(
                "No topology provided. This may lead to planner raise OOM (Out of Memory) errors, "
                "as the planner might not have enough information to optimize memory usage. "
                "Consider providing a TorchREC topology to avoid potential issues.",
                RuntimeWarning,
            )

            topology = Topology(
                local_world_size=get_local_size(),
                world_size=dist.get_world_size(),
                compute_device="cuda" if torch.cuda.is_available() else "cpu",
                hbm_cap=HBM_CAP,
                ddr_cap=DDR_CAP,
            )
        self._torchrec_planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            proposer=proposer,
            partitioner=partitioner,
            performance_model=performance_model,
            stats=stats,
            constraints=torchrec_tables_consts,
            debug=debug,
        )
        # generate DynamicEmb table's plan
        compute_device = topology.compute_device
        local_size = topology.local_world_size
        world_size = dist.get_world_size()

        self._dyn_emb_plan = {}
        for dyn_emb_name, dynamicemb_constraint in dyn_emb_table_consts.items():
            opts = dynamicemb_constraint.dynamicemb_options
            num_embeddings_per_rank = opts.max_capacity
            embedding_dim = dyn_emb_table_eb_configs[dyn_emb_name].embedding_dim

            tmp_para_sharding = DynamicEmbParameterSharding(
                sharding_spec=(
                    EnumerableShardingSpec(
                        [
                            ShardMetadata(
                                shard_sizes=[num_embeddings_per_rank, embedding_dim],
                                # TODO:0 is we don't have column-wise sharding now
                                shard_offsets=[num_embeddings_per_rank * i, 0],
                                placement=placement(compute_device, i, local_size),
                            )
                            for i in range(world_size)
                        ]
                    )
                ),
                sharding_type=ShardingType.ROW_WISE.value,
                # compute_kernel=EmbeddingComputeKernel.DynamicEmb.value,
                ranks=[i for i in range(world_size)],
                compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
                customized_compute_kernel=DynamicEmbKernel,
                dist_type="hash_roundrobin",
                dynamicemb_options=opts,
            )
            self._dyn_emb_plan[dyn_emb_name] = tmp_para_sharding

    def collective_plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        pg: Optional[dist.ProcessGroup] = dist.GroupMember.WORLD,
    ) -> ShardingPlan:
        """
        Generate a collective sharding plan.

        Parameters
        ----------
        module : nn.Module
            The PyTorch module to be sharded.
        sharders : List[ModuleSharder[nn.Module]]
            A list of module sharders.
        pg : Optional[dist.ProcessGroup], optional
            The process group for distributed training. Defaults to dist.GroupMember.WORLD.

        Returns
        -------
        ShardingPlan
            The generated sharding plan.
        """
        torchrec_plan = self._torchrec_planner.collective_plan(module, sharders, pg)
        dyn_emb_names = self._dyn_emb_plan.keys()
        for dyn_emb_name in dyn_emb_names:
            for (
                torchrec_module_name,
                torchrec_module_plan,
            ) in torchrec_plan.plan.items():
                for table_name, table_plan in torchrec_module_plan.items():
                    if dyn_emb_name == table_name:
                        torchrec_module_plan[table_name] = self._dyn_emb_plan[
                            table_name
                        ]

        return torchrec_plan
