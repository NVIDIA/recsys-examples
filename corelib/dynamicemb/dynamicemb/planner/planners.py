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

import copy
import math
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Set, Tuple

from torch import distributed as dist
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import EmbeddingShardingPlanner, ParameterConstraints
from torchrec.distributed.planner.types import StorageReservation, Topology
from torchrec.distributed.collective_utils import invoke_on_rank_and_broadcast_result
from torchrec.distributed.planner.utils import sharder_name
from torchrec.distributed.sharding_plan import placement
from torchrec.distributed.types import (
    EmbeddingModuleShardingPlan,
    EnumerableShardingSpec,
    ModuleSharder,
    ParameterSharding,
    ShardingPlan,
    ShardingType,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import BaseEmbeddingConfig, data_type_to_dtype

from .plan import (
    module_without_tables,
    optimizer_types,
    per_rank_storage,
    topology_minus,
)
from ..dynamicemb_config import (
    DEFAULT_INDEX_TYPE,
    DynamicEmbTableOptions,
    _sharded_table_bucket_layout,
    align_to_table_size,
    complete_initializer_args,
)


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
    """A ``ParameterSharding`` that carries one DynamicEmb table's options across
    the TorchRec boundary.

    ``compute_kernel`` has to be ``CUSTOMIZED_KERNEL``: that value is what makes
    TorchRec call :meth:`get_additional_fused_params`
    (``torchrec/distributed/utils.py:518``) and skip its own ``state_dict``
    handling. It is an inherited field with no default, so every construction
    states it.

    ``dynamicemb_options`` is the only field added here. Anything a DynamicEmb
    table needs downstream belongs inside it rather than beside it -- a second
    copy on this dataclass is one more thing that can disagree with the options
    object the same plan already carries.
    """

    dynamicemb_options: DynamicEmbTableOptions = field(
        default_factory=DynamicEmbTableOptions
    )

    @classmethod
    def _additional_field_names(cls) -> Set[str]:
        """The fields this class adds on top of ``ParameterSharding``.

        A set difference rather than a written-out list, so that fields added
        upstream are excluded on their own and fields added here are picked up
        by both directions below without a second edit.
        """
        return {f.name for f in fields(cls)} - {
            f.name for f in fields(ParameterSharding)
        }

    def get_additional_fused_params(self) -> Dict[str, Any]:
        """TorchRec's hook for a customized kernel's extra per-table params.

        Called from ``torchrec/distributed/utils.py:518-526`` -- by ``hasattr``,
        not by type -- and only when ``compute_kernel == CUSTOMIZED_KERNEL``.
        The result is merged into ``GroupedEmbeddingConfig.fused_params``.
        """
        return {name: getattr(self, name) for name in self._additional_field_names()}

    @staticmethod
    def pop_additional_fused_params(fused_params: Dict[str, Any]) -> None:
        """Undo :meth:`get_additional_fused_params` before
        :class:`~dynamicemb.batched_dynamicemb_tables.BatchedDynamicEmbeddingTablesV2`.

        These entries are for planning and per-table options; they are not valid
        ``**fused_params`` for that module.
        """
        for name in DynamicEmbParameterSharding._additional_field_names():
            fused_params.pop(name, None)


def _prepare_dynemb_table_options(
    constraints: Dict[str, DynamicEmbParameterConstraints],
    eb_configs: List[BaseEmbeddingConfig],
):
    """Check ``constraints`` ↔ ``eb_configs`` naming, then fill per-table DynamicEmb options.

    For each DynamicEmb table: ``complete_initializer_args`` -- the only place
    that still knows ``num_embeddings``, which an unbounded UNIFORM needs, since
    ``max_capacity`` becomes the per-rank row count a few lines down -- then
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


def _table_configs(
    module: nn.Module,
    sharders: List[ModuleSharder[nn.Module]],
) -> List[BaseEmbeddingConfig]:
    """Every shardable table's config, found the way the planner finds tables.

    This is what the ``eb_configs`` argument used to carry. Reading it off the
    module instead means the caller states their tables once, where they build
    the collection, rather than again when they build the planner -- and that
    the two cannot disagree.
    """
    sharder_map = {sharder_name(sharder.module_type): sharder for sharder in sharders}
    configs: List[BaseEmbeddingConfig] = []
    queue: List[nn.Module] = [module]
    while queue:
        node = queue.pop()
        if sharder_name(type(node)) in sharder_map:
            for accessor in ("embedding_bag_configs", "embedding_configs"):
                if hasattr(node, accessor):
                    configs.extend(getattr(node, accessor)())
                    break
            continue
        queue.extend(child for _, child in node.named_children())
    return configs


class DynamicEmbeddingShardingPlanner(EmbeddingShardingPlanner):
    def __init__(
        self,
        topology: Optional[Topology] = None,
        constraints: Optional[Dict[str, DynamicEmbParameterConstraints]] = None,
        storage_reservation: Optional[StorageReservation] = None,
        **kwargs: Any,
    ):
        """A TorchRec planner that also plans DynamicEmb tables.

        Takes what :class:`~torchrec.distributed.planner.planners.EmbeddingShardingPlanner`
        takes, and nothing else. ``constraints`` is the only argument that means
        anything more here: a :class:`DynamicEmbParameterConstraints` with
        ``use_dynamicemb=True`` marks a table as DynamicEmb's, and those tables
        are settled by :meth:`_settle` rather than handed to TorchRec.

        Arguments other than the three below are forwarded to TorchRec
        unchanged, so ``callbacks``, ``timeout_seconds``, ``plan_loader`` and
        whatever else it grows work here too.

        Parameters
        ----------
        topology : Optional[Topology], optional
            The topology of GPU and Host memory. If None, a default topology will be created. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
            State what the machine has: what the DynamicEmb tables cost is taken
            out of it at plan time, and TorchRec is given the difference.
        constraints : Optional[Dict[str, DynamicEmbParameterConstraints]], optional
            A dictionary of constraints for every TorchREC embedding table and Dynamic embedding table. Defaults to None.
            Per-table DynamicEmb options are filled in here at plan time
            (initializer bounds, sharded capacity via ``_sharded_table_bucket_layout``,
            and the per-rank HBM budget), so the options a caller passes are
            completed in place rather than copied.
        storage_reservation : Optional[StorageReservation], optional
            Storage reservation details. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
            It is for what TorchRec reserves for -- dense modules and the input
            KJT -- and sees a Topology the DynamicEmb tables are already out of.
        **kwargs
            Forwarded to ``EmbeddingShardingPlanner``: ``batch_size``,
            ``enumerator``, ``proposer``, ``partitioner``, ``performance_model``,
            ``stats``, ``debug``, and the rest.
        """
        self._constraints: Dict[str, DynamicEmbParameterConstraints] = constraints or {}
        self._dyn_emb_plan: Dict[str, DynamicEmbParameterSharding] = {}
        self._settled = False

        if topology is None:
            # TorchRec builds one from `None`, and builds it better than this
            # used to: it passes `pod_size`, which sizes the high-bandwidth
            # interconnect domain and is what tells a perf model that crossing
            # a node costs more than staying inside one. Warn anyway -- the
            # capacities it falls back to are constants, 32GB of HBM and 128GB
            # of host, and a DynamicEmb table is usually planned against a
            # machine that is nothing like that.
            warnings.warn(
                "No topology provided, so TorchRec's default capacities are "
                "used: 32GB of HBM and 128GB of host memory per rank. Those are "
                "constants rather than a measurement, and the DynamicEmb tables "
                "are sized against them. Pass a Topology describing the machine.",
                RuntimeWarning,
            )

        super().__init__(
            topology=topology,
            constraints={
                name: constraint
                for name, constraint in self._constraints.items()
                if not constraint.use_dynamicemb
            },
            storage_reservation=storage_reservation,
            **kwargs,
        )

    def _settle(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> None:
        """Decide the DynamicEmb tables, from the module's own table configs.

        Their capacity comes from ``DynamicEmbTableOptions`` and their placement
        from the key -> rank rule, so this is arithmetic over the constraints
        rather than a search, and it does not need TorchRec to have planned
        anything yet.

        It runs on every rank, outside the broadcast in :meth:`collective_plan`,
        for two reasons. The options it fills in stay on the rank that filled
        them -- they are the half of a DynamicEmb table that does not cross the
        wire -- and :meth:`_attach_options` needs them everywhere, while
        :meth:`_decide` runs on rank 0 alone. What does need one rank's
        authority is the placement, and that is what the broadcast covers.

        Once per planner: the options are filled in place, and a second pass
        would re-align already aligned capacities and warn about it.
        """
        if self._settled:
            return

        configs = {
            config.name: config
            for config in _table_configs(module, sharders)
            if self._constraints.get(config.name)
            and self._constraints[config.name].use_dynamicemb
        }
        _prepare_dynemb_table_options(self._constraints, list(configs.values()))

        compute_device = self._topology.compute_device
        local_size = self._topology.local_world_size
        world_size = dist.get_world_size()

        for name, config in configs.items():
            opts = self._constraints[name].dynamicemb_options
            rows = opts.max_capacity
            self._dyn_emb_plan[name] = DynamicEmbParameterSharding(
                sharding_spec=EnumerableShardingSpec(
                    [
                        ShardMetadata(
                            shard_sizes=[rows, config.embedding_dim],
                            # TODO:0 is we don't have column-wise sharding now
                            shard_offsets=[rows * i, 0],
                            placement=placement(compute_device, i, local_size),
                        )
                        for i in range(world_size)
                    ]
                ),
                sharding_type=ShardingType.ROW_WISE.value,
                ranks=list(range(world_size)),
                compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
                dynamicemb_options=opts,
            )

        self._settled = True

    def plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> ShardingPlan:
        """This rank's plan for both halves of the model.

        Parameters
        ----------
        module : nn.Module
            The PyTorch module to be sharded.
        sharders : List[ModuleSharder[nn.Module]]
            A list of module sharders.

        Returns
        -------
        ShardingPlan
            One plan covering the DynamicEmb tables and the TorchRec ones.
        """
        self._settle(module, sharders)
        return self._attach_options(self._decide(module, sharders))

    def collective_plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        pg: Optional[dist.ProcessGroup] = dist.GroupMember.WORLD,
    ) -> ShardingPlan:
        """One rank decides and the rest are told, as TorchRec's own planner does.

        Deciding on every rank and trusting them to agree would be correct only
        as long as every rank was handed identical constraints and an identical
        topology, which nothing checks. Rank 0 deciding makes the question moot.

        What crosses the wire is the decisions -- sharding type, ranks, shard
        sizes and offsets and placements. The ``DynamicEmbTableOptions`` are
        reattached from this rank's own constraints afterwards, because some of
        what they hold is rank-local by nature: ``external_storage`` is a live
        handle to a store this process opened, and ``score_function`` is a
        Python callable. Sending rank 0's copy of either would be wrong even
        where pickle allows it.

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
            The generated sharding plan, identical on every rank.
        """
        # Every rank settles: the options it fills in are the half of a
        # DynamicEmb table that stays local, and _attach_options needs them
        # everywhere. Only the placement needs one rank's authority.
        self._settle(module, sharders)
        return self._attach_options(
            invoke_on_rank_and_broadcast_result(pg, 0, self._decide, module, sharders)
        )

    def _decide(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> ShardingPlan:
        """Both halves of the plan, with the per-table options left off.

        In order: the DynamicEmb tables are already settled (``__init__`` did
        that -- their capacity comes from their options and their placement from
        the key -> rank rule, so there is nothing to search); what they cost
        comes out of the Topology; TorchRec plans what is left of the model
        inside what is left of the memory; the two halves are put back together.

        The Topology is swapped rather than passed, because the enumerator and
        the estimators were built against ``self._topology`` in ``__init__``.
        They read it for bandwidths and device type, not for the budget -- the
        budget is the Topology that ``reserve`` returns, which is derived from
        this one -- so the swap reaches the decision that depends on it.

        ``_insert_dynamicemb_plan`` gets the caller's module, not the reduced
        one: the reduced one no longer admits to having these tables, which is
        the whole point of it, so it cannot say what path they live under.
        """
        table_names = set(self._dyn_emb_plan)
        reduced_module = module_without_tables(module, table_names, sharders)
        spent = per_rank_storage(
            self._dyn_emb_plan,
            optimizer_types(module, sharders),
            len(self._topology.devices),
        )

        whole_topology = self._topology
        try:
            self._topology = topology_minus(whole_topology, spent)
            torchrec_plan = super().plan(reduced_module, sharders)
        finally:
            self._topology = whole_topology

        self._insert_dynamicemb_plan(torchrec_plan, module, sharders)
        return torchrec_plan

    def _attach_options(self, plan: ShardingPlan) -> ShardingPlan:
        """Put this rank's DynamicEmbTableOptions back on the plan.

        See :meth:`collective_plan` for why they were taken off. The plan's own
        shard metadata is checked against them when the kernel is built
        (``_get_dynamicemb_options_per_table``), so a rank whose options
        disagree with the decisions it was sent does not go unnoticed.
        """
        for module_plan in plan.plan.values():
            for table_name, parameter_sharding in module_plan.items():
                settled = self._dyn_emb_plan.get(table_name)
                if settled is not None:
                    parameter_sharding.dynamicemb_options = settled.dynamicemb_options
        return plan

    @staticmethod
    def _for_the_plan(
        settled: DynamicEmbParameterSharding,
    ) -> DynamicEmbParameterSharding:
        """A copy of a settled table's ParameterSharding, fit to hand out.

        Two things the original must not be exposed to. The options come off
        because they are reattached per rank after the broadcast
        (:meth:`collective_plan`), and stripping the original would empty the
        planner's own record of the table. The shard metadata is deep-copied
        because TorchRec writes to it: ``replace_placement_with_meta_device``
        rewrites every placement in place when DMP's device is ``meta``
        (``distributed/embeddingbag.py:995``), which would otherwise reach back
        into the planner and leave a second `plan` call handing out metadata the
        first DMP had already rewritten.
        """
        handed_out = copy.copy(settled)
        handed_out.sharding_spec = copy.deepcopy(settled.sharding_spec)
        handed_out.dynamicemb_options = None
        return handed_out

    def _insert_dynamicemb_plan(
        self,
        torchrec_plan: ShardingPlan,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> None:
        """Add the DynamicEmb tables' ParameterShardings to a plan built without them.

        The enumerator leaves them out of the search space -- their capacity and
        their placement are settled here, not searched for -- so nothing in
        ``torchrec_plan`` describes them, and a module holding only DynamicEmb
        tables is missing from it entirely. ``DistributedModelParallel`` shards a
        module only if the plan has an entry for its path, silently leaving it
        alone otherwise, so the entries have to be put there rather than
        overwritten.

        Which means finding the path each table lives under, which the planner
        would have discovered on our behalf. Walking for it repeats the
        enumerator's descent, in the same order and by the same rule -- a module
        a sharder claims is a leaf, anything else is recursed into -- because
        the paths have to be the ones the planner would have produced.
        """
        if not self._dyn_emb_plan:
            return

        sharder_map = {
            sharder_name(sharder.module_type): sharder for sharder in sharders
        }
        queue: List[Tuple[str, nn.Module]] = [("", module)]
        placed: Set[str] = set()
        while queue:
            path, child_module = queue.pop()
            sharder = sharder_map.get(sharder_name(type(child_module)))
            if sharder is None:
                for name, child in child_module.named_children():
                    queue.append((f"{path}.{name}" if path else name, child))
                continue
            for table_name in sharder.shardable_parameters(child_module):
                parameter_sharding = self._dyn_emb_plan.get(table_name)
                if parameter_sharding is None:
                    continue
                module_plan = torchrec_plan.plan.setdefault(
                    path, EmbeddingModuleShardingPlan()
                )
                module_plan[table_name] = self._for_the_plan(parameter_sharding)
                placed.add(table_name)

        missing = set(self._dyn_emb_plan) - placed
        if missing:
            raise RuntimeError(
                f"No sharder claims a module holding the DynamicEmb tables "
                f"{sorted(missing)}, so they cannot be placed in the plan. "
                "Pass a DynamicEmb sharder for every collection that has one."
            )
