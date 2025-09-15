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
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchrec
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
from torchrec.modules.embedding_configs import BaseEmbeddingConfig

from ..batched_dynamicemb_compute_kernel import (
    BatchedDynamicEmbedding,
    BatchedDynamicEmbeddingBag,
)
from ..dynamicemb_config import (
    DynamicEmbKernel,
    DynamicEmbTableOptions,
    validate_initializer_args,
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
        Including HKV Configs and Initializer Args. The initialization method for the parameters.
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

    compute_kernel: str = EmbeddingComputeKernel.CUSTOMIZED_KERNEL
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


# def _get_safe_local_capacity(local_capacity, bucket_capacity):
#     LOCAL_BATCH_SIZE: int = 8192
#     factor: int = -1
#     if bucket_capacity == local_capacity:
#         factor = 1
#     elif bucket_capacity >= 128:
#         factor = 2
#     elif bucket_capacity == 64:
#         factor = 4
#     elif bucket_capacity == 32:
#         factor = 8
#     elif bucket_capacity == 16:
#         factor = 16
#     else:
#         raise ValueError("Bucket capacity is too small!")
#     hkv_cap_min = LOCAL_BATCH_SIZE * factor
#     if local_capacity < hkv_cap_min:
#         local_capacity = hkv_cap_min
#     return local_capacity


def _validate_configs(
    constraints: Dict[str, DynamicEmbParameterConstraints],
    eb_configs: List[BaseEmbeddingConfig],
):
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
        validate_initializer_args(
            tmp_constraint.dynamicemb_options.initializer_args, tmp_config
        )
        # modify num_embeddings per rank to power of 2
        num_aligned_embedding_per_rank = int(
            _next_power_of_2(math.ceil(tmp_config.num_embeddings / world_size))
        )
        if (
            num_aligned_embedding_per_rank
            < constraints[config_name].dynamicemb_options.bucket_capacity
        ):
            num_aligned_embedding_per_rank = constraints[
                config_name
            ].dynamicemb_options.bucket_capacity

        # num_aligned_embedding_per_rank = _get_safe_local_capacity(num_aligned_embedding_per_rank, tmp_constraint.dynamicemb_options.bucket_capacity)
        if tmp_config.num_embeddings != int(
            num_aligned_embedding_per_rank * world_size
        ):
            tmp_constraint.dynamicemb_options.num_aligned_embedding_per_rank = (
                num_aligned_embedding_per_rank
            )


def _dyn_emb_table_size_per_rank(
    dyn_emb_table_config: BaseEmbeddingConfig, world_size: int
) -> Tuple[int, int]:
    num_embeddings = dyn_emb_table_config.num_embeddings
    # TODO: align num_embedding automatic , maybe need user input , and we don't align it.
    num_embeddings_per_rank = int(
        _next_power_of_2(math.ceil(num_embeddings / world_size))
    )

    embedding_dim = dyn_emb_table_config.embedding_dim
    return embedding_dim, num_embeddings_per_rank


def _dyn_emb_table_hbm_size_per_rank(
    dyn_emb_table_const: DynamicEmbParameterConstraints, world_size: int
) -> int:
    HBM_memory_in_byte_per_rank = math.ceil(
        dyn_emb_table_const.dynamicemb_options.global_hbm_for_values / world_size
    )
    return HBM_memory_in_byte_per_rank


def _reserve_storage_for_dyn_emb(
    topology: Topology,
    dyn_emb_table_const: Dict[str, DynamicEmbParameterConstraints],
    dyn_emb_table_configs: Dict[str, BaseEmbeddingConfig],
) -> None:
    world_size = dist.get_world_size()
    for name, constraint in dyn_emb_table_const.items():
        tmp_table_config = dyn_emb_table_configs[name]

        embedding_dim, num_embeddings_per_rank = _dyn_emb_table_size_per_rank(
            tmp_table_config, world_size
        )
        # TODO: should HBM align with the power of 2
        HBM_memory_in_bytes_per_rank = _dyn_emb_table_hbm_size_per_rank(
            constraint, world_size
        )

        constraint.dynamicemb_options.local_hbm_for_values = (
            HBM_memory_in_bytes_per_rank
        )


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
        TorchREC BaseEmbeddingConfig. This is because the dynamic embedding table needs to re-plan the number of
        embedding vectors on each rank to align with the power of 2.

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
        _validate_configs(constraints, eb_configs)

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
        _reserve_storage_for_dyn_emb(
            topology, dyn_emb_table_consts, dyn_emb_table_eb_configs
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
            tmp_table_config = dyn_emb_table_eb_configs[dyn_emb_name]

            is_pooled = isinstance(tmp_table_config, torchrec.EmbeddingBagConfig)
            compute_kernel = (
                BatchedDynamicEmbeddingBag if is_pooled else BatchedDynamicEmbedding
            )

            embedding_dim, num_embeddings_per_rank = _dyn_emb_table_size_per_rank(
                tmp_table_config, world_size
            )

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
                dist_type="roundrobin",
                dynamicemb_options=dynamicemb_constraint.dynamicemb_options,
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
