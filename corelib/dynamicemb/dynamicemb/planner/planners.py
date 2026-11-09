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
from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from torch import distributed as dist
from torch import nn
from torchrec.distributed.collective_utils import invoke_on_rank_and_broadcast_result
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import EmbeddingShardingPlanner, ParameterConstraints
from torchrec.distributed.planner.types import StorageReservation, Topology
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

from ..dynamicemb_config import (
    DEFAULT_INDEX_TYPE,
    DynamicEmbTableOptions,
    _sharded_table_bucket_layout,
    align_to_table_size,
    complete_initializer_args,
    get_group_value_bytes_by_tier,
    normalize_caching_against_hbm_budget,
)
from .placement import BalancedNodePlacer, NodePlacer, TableToPlace
from .plan import (
    TableKey,
    fusion_key,
    make_storage,
    module_without_tables,
    optimizer_types,
    per_rank_storage,
    sharded_modules,
    table_fanout,
    table_ranks,
    topology_minus,
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
    def with_options(
        cls,
        decisions: ParameterSharding,
        options: DynamicEmbTableOptions,
    ) -> "DynamicEmbParameterSharding":
        """This type, built from TorchRec's and one rank's options.

        A DynamicEmb table reaches the plan as a plain ``ParameterSharding``
        -- decisions are all that crosses the wire -- and becomes a DynamicEmb
        one here, on the rank that has the options to complete it. It is only
        ever built this way, whole: :meth:`get_additional_fused_params` reads
        ``dynamicemb_options``, so an instance carrying ``None`` there would
        hand TorchRec a ``None`` for the table's whole configuration, and the
        annotation on that field says it cannot happen.

        The inherited fields are copied by :func:`~dataclasses.fields` rather
        than named, so a field added upstream travels without an edit here --
        the same reason :meth:`_additional_field_names` is a set difference.

        The shard metadata is deep-copied rather than shared, because TorchRec
        writes to it: ``replace_placement_with_meta_device`` rewrites every
        placement in place when DMP's device is ``meta``
        (``distributed/embeddingbag.py:995``). On rank 0 the plan still holds
        the planner's own record at this point -- a broadcast returns the
        leader its own object, not a round trip
        (``collective_utils.py:65-67``) -- so sharing it would let the first
        DMP rewrite what a second `plan` call hands out.
        """
        built = cls(
            **{f.name: getattr(decisions, f.name) for f in fields(ParameterSharding)},
            dynamicemb_options=options,
        )
        built.sharding_spec = copy.deepcopy(decisions.sharding_spec)
        return built

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


# EmbeddingBagCollection exposes `embedding_bag_configs`, EmbeddingCollection
# `embedding_configs`. Both return the very list of configs they were built
# with (`modules/embedding_modules.py`), so a config a caller is holding is
# the object found here, and `is` identifies it.
_CONFIG_ACCESSORS = ("embedding_bag_configs", "embedding_configs")


def _collections(module: nn.Module) -> Iterator[Tuple[str, List[BaseEmbeddingConfig]]]:
    """Every table-holding module under ``module``, with the path to it.

    By the accessors rather than by a sharder, because
    :func:`get_constraint_key` is called while the caller is still assembling
    the constraints and has no sharder list yet. The paths still agree with
    :func:`~dynamicemb.planner.plan.sharded_modules`: that walk stops at a
    module a sharder claims, and no collection contains another, so a
    collection is a leaf either way and `named_modules` reaches it by the same
    dotted path. The root is ``""`` in both.

    Nothing inside a collection answers to these accessors -- the ModuleDict
    below it holds plain ``nn.EmbeddingBag``s -- so descending past one finds
    nothing and costs only the walk.
    """
    for path, node in module.named_modules():
        for accessor in _CONFIG_ACCESSORS:
            if hasattr(node, accessor):
                yield path, list(getattr(node, accessor)())
                break


def get_constraint_key(
    config: BaseEmbeddingConfig,
    module: Optional[nn.Module] = None,
) -> str:
    """The key to register ``config``'s constraint under.

    Table names are unique in most models, and there the answer is the table's
    own name -- which is what every existing ``constraints`` dict already uses,
    and why calling this changes nothing for a model that does not duplicate a
    name. Where two collections *do* declare one name, the name alone names
    both, and a DynamicEmb table needs a key that names one: its
    ``DynamicEmbTableOptions`` are written into at plan time, so two tables
    cannot share the object a single key reaches.

    ``module`` is the root the planner will be given. Pass it and the key
    carries the path from that root to the collection holding ``config``;
    leave it out and the key is the bare table name. Passing the collection
    itself is the same thing said differently -- its path from itself is empty
    -- so a caller with one collection and no enclosing model gets bare names
    without a special case.

    The answer depends only on the arguments, never on whether some other table
    happens to share a name. A key that changed because an unrelated collection
    grew a table would be a key written down somewhere else and now wrong.

    ``config`` is matched by identity, not by name, because by name there is
    nothing to tell the two apart. The collections return the config objects
    they were constructed from, so this is the object the caller built.

    Parameters
    ----------
    config : BaseEmbeddingConfig
        The table's config -- the ``EmbeddingConfig`` or ``EmbeddingBagConfig``
        object passed to the collection that holds it.
    module : Optional[nn.Module]
        The model, or the collection itself. ``None`` returns ``config.name``.

    Returns
    -------
    str
        The key for ``constraints``.

        Not for ``get_planner``'s ``table_row_wise_embedding_table_names``,
        ``data_parallel_embedding_table_names`` or ``dynamicemb_options_dict``:
        those are keyed by the bare table name, since ``get_planner`` takes a
        flat ``eb_configs`` with no module to take a path from. A qualified key
        matches none of them, and ``get_planner`` says so rather than planning
        the table some other way. A model that needs a qualified key builds its
        constraints and the planner directly.

    Raises
    ------
    ValueError
        If no collection under ``module`` holds this config object, or if more
        than one does -- two collections built from one config object are two
        tables with one identity, and there is no key that means either.

    Examples
    --------
    >>> ebc_a = EmbeddingBagCollection(tables=[shared_a, item_id])
    >>> ebc_b = EmbeddingBagCollection(tables=[shared_b])
    >>> model = Model(ebc_a, ebc_b)              # model.sparse.ebc_a, ...
    >>> get_constraint_key(shared_a, model)
    'sparse.ebc_a.shared'
    >>> get_constraint_key(shared_b, model)
    'sparse.ebc_b.shared'
    >>> get_constraint_key(item_id, ebc_a)
    'item_id'
    """
    if module is None:
        return config.name

    holders = [
        path
        for path, configs in _collections(module)
        if any(held is config for held in configs)
    ]

    if not holders:
        known = sorted(
            f"{path}.{held.name}" if path else held.name
            for path, configs in _collections(module)
            for held in configs
        )
        raise ValueError(
            f"No collection under this module holds the config for table "
            f"{config.name!r}. It is matched by identity, so a config that was "
            "copied, rebuilt, or passed to a different model will not be found "
            f"even where the name matches. This module's tables are {known}."
        )

    if len(holders) > 1:
        raise ValueError(
            f"The config for table {config.name!r} is the same object in more "
            f"than one collection ({', '.join(sorted(holders))}), so it has no "
            "one key. Those are two tables -- each collection is sharded on its "
            "own and builds its own storage -- so give each one its own config "
            "object, and they can then be constrained separately."
        )

    path = holders[0]
    return f"{path}.{config.name}" if path else config.name


def _pooling_of(config: BaseEmbeddingConfig) -> Optional[object]:
    """How a table pools, or ``None`` when it does not pool at all.

    ``EmbeddingBagConfig`` carries ``pooling``; ``EmbeddingConfig`` -- a
    sequence table in an ``EmbeddingCollection`` -- has no such field. TorchRec
    groups on pooling either way, so the answer only has to be consistent:
    every sequence table reports ``None`` and so agrees with the others, and
    differs from any pooled table.
    """
    return getattr(config, "pooling", None)


def _sharding_type_of(constraint: DynamicEmbParameterConstraints) -> str:
    """The sharding type a DynamicEmb table is declared with.

    Read off ``ParameterConstraints.sharding_types``, the field TorchRec already
    has for this, rather than a DynamicEmb-only one. A DynamicEmb table is not
    searched over, so the list has to name exactly one.
    """
    declared = constraint.sharding_types
    if not declared:
        return ShardingType.ROW_WISE.value
    if len(declared) != 1:
        raise ValueError(
            f"A DynamicEmb table is placed, not searched for, so it needs exactly "
            f"one sharding type; got {declared}."
        )
    return declared[0]


def resolve_constraints(
    paths_and_configs: List[Tuple[str, BaseEmbeddingConfig]],
    constraints: Dict[str, DynamicEmbParameterConstraints],
) -> Dict[TableKey, DynamicEmbParameterConstraints]:
    """Which constraint applies to each table occurrence, and whether it can.

    Each occurrence asks for its keys in turn -- qualified
    (``collection.name``) before bare (``name``) -- and takes the first that
    `constraints` has. That order is the whole of the addressing rule: a bare
    name still reaches a table that nothing more specific names, which is why
    a model whose names are unique needs no change, and a qualified key
    overrides it for one collection.

    Three things are refused here, all of them cases where a constraint would
    otherwise be quietly ignored:

    * A key no occurrence answers to. The model is one of the two statements
      of what tables exist and `constraints` is the other; a key matching
      nothing is a typo, and a typo that reads as "plan this table with
      TorchRec's defaults" is the kind that is found in a profile.
    * A qualified key on a table that is not DynamicEmb's. TorchRec's planner
      is keyed by table name alone (``planner/enumerators.py``), so it would
      never look the key up, and the constraint would do nothing.
    * One ``DynamicEmbTableOptions`` object reached by two occurrences. This
      is the case a bare name creates when two collections declare it:
      `_prepare_dynemb_table_options` writes each table's geometry into its
      options, so one object cannot describe both, and the second pass would
      read back what the first wrote. The message names
      :func:`get_constraint_key`, which is the way out.

    A table no key reaches is not an error and is simply absent from the
    result: it is a table planned with TorchRec's defaults.
    """
    resolved: Dict[TableKey, DynamicEmbParameterConstraints] = {}
    answered: Set[str] = set()
    misdirected: Dict[str, str] = {}

    for path, config in paths_and_configs:
        key = TableKey(path, config.name)
        for candidate in key.constraint_keys:
            constraint = constraints.get(candidate)
            if constraint is None:
                continue
            # Marked even when a more specific key already won, so a bare name
            # that one collection overrides is not then reported as a typo.
            answered.add(candidate)
            if key in resolved:
                continue
            resolved[key] = constraint
            if candidate != key.name and not constraint.use_dynamicemb:
                misdirected[candidate] = str(key)

    unknown = set(constraints) - answered
    if unknown:
        known = sorted(str(TableKey(path, c.name)) for path, c in paths_and_configs)
        raise ValueError(
            f"Constraints name tables the model does not have: {sorted(unknown)}. "
            f"The model's shardable tables are {known}. A table may be named by "
            "its own name, or by `collection.name` -- see `get_constraint_key`."
        )

    if misdirected:
        detail = "; ".join(
            f"{key!r} for {name}" for key, name in sorted(misdirected.items())
        )
        raise ValueError(
            f"These constraints are keyed by collection and table but belong to "
            f"tables TorchRec plans, not DynamicEmb: {detail}. TorchRec's planner "
            "looks constraints up by table name alone, so a qualified key would "
            "never be found and the constraint would be silently dropped. Key "
            "them by the table name, or set use_dynamicemb=True."
        )

    sharing: Dict[int, List[TableKey]] = defaultdict(list)
    for key, constraint in resolved.items():
        if constraint.use_dynamicemb:
            sharing[id(constraint.dynamicemb_options)].append(key)
    shared = sorted(
        (sorted(str(k) for k in keys) for keys in sharing.values() if len(keys) > 1)
    )
    if shared:
        detail = "; ".join(" and ".join(keys) for keys in shared)
        raise ValueError(
            f"These DynamicEmb tables share one DynamicEmbTableOptions object: "
            f"{detail}. Each collection is sharded on its own and builds its own "
            "storage, so those are separate tables, and the options carry each "
            "table's own dim, capacity and dtype -- written in at plan time, so "
            "one object cannot describe both. Give each its own constraint, "
            "keyed by `get_constraint_key(config, model)`."
        )

    return resolved


def _prepare_dynemb_table_options(
    table_constraints: Dict[TableKey, DynamicEmbParameterConstraints],
    configs: Dict[TableKey, BaseEmbeddingConfig],
    world_size: int,
    local_size: int,
):
    """Fill in each DynamicEmb table's options from the config it was declared with.

    One entry per occurrence, keyed by :class:`TableKey`, and
    :func:`resolve_constraints` has already refused the case where two of them
    would reach one options object -- which matters because this is not
    idempotent: it reads back ``bucket_capacity``, ``max_capacity`` and
    ``init_capacity`` after writing them, so a second pass over one object
    would align already-aligned capacities and warn about it.

    ``world_size`` and ``local_size`` are passed rather than read from ``dist``
    so that the capacity a table is sized for and the ranks it is placed on come
    from one statement of the machine, the planner's Topology.

    For each DynamicEmb table: ``complete_initializer_args`` -- the only place
    that still knows ``num_embeddings``, which an unbounded UNIFORM needs, since
    ``max_capacity`` becomes the per-rank row count a few lines down -- then
    ``_sharded_table_bucket_layout`` (sets effective ``bucket_capacity`` and per-rank
    ``max_capacity``), then ``local_hbm_for_values`` from
    ``global_hbm_for_values`` and the table's own ``fanout`` -- the world under
    row-wise, one node under table-row-wise, not the world either way; then
    a table whose budget resolved to zero loses ``caching=True`` (with a
    warning), before anything groups or costs on that field; then default
    ``index_type`` / ``embedding_dtype``,
    align ``init_capacity`` to the effective ``bucket_capacity`` when set; if aligned
    ``init_capacity`` exceeds ``max_capacity``, clamp it to ``max_capacity``; if ``init_capacity``
    was unset, set it to ``max_capacity``; set ``dim`` from ``BaseEmbeddingConfig.embedding_dim``.
    """
    for key, tmp_constraint in table_constraints.items():
        # A table the constraints do not mention is TorchRec's to plan, and is
        # not in `table_constraints` at all; one they mention without
        # use_dynamicemb is TorchRec's too.
        if not tmp_constraint.use_dynamicemb:
            continue

        tmp_config = configs[key]
        opts = tmp_constraint.dynamicemb_options

        opts.initializer_args = complete_initializer_args(
            opts.initializer_args,
            embedding_config=tmp_config,
        )
        # The divisor is how many pieces the table's rows are split into, which
        # is the world under row-wise and one node under table-row-wise -- so it
        # is per table, not global (§4.2). It does not depend on *which* node,
        # which is why a table can be sized before it is placed.
        fanout = table_fanout(_sharding_type_of(tmp_constraint), world_size, local_size)
        num_buckets, effective_bucket_capacity = _sharded_table_bucket_layout(
            tmp_config,
            fanout,
            opts.bucket_capacity,
        )
        opts.bucket_capacity = effective_bucket_capacity
        aligned_per_rank_rows = num_buckets * effective_bucket_capacity
        opts.max_capacity = aligned_per_rank_rows
        opts.local_hbm_for_values = math.ceil(opts.global_hbm_for_values / fanout)
        # A cache is an HBM tier: a table whose budget resolved to zero cannot
        # have one, and this is the last moment before the options are grouped
        # and costed -- both read `caching` to decide a layout.
        normalize_caching_against_hbm_budget(opts, label=str(key))

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
) -> List[Tuple[str, BaseEmbeddingConfig]]:
    """Every shardable table's config, paired with the module it came out of.

    This is what the ``eb_configs`` argument used to carry. Reading it off the
    module instead means the caller states their tables once, where they build
    the collection, rather than again when they build the planner -- and that
    the two cannot disagree.

    The owning module travels with the config because TorchRec shards each
    shardable module on its own: one ``ShardedEmbedding*Collection`` per
    collection, each with its own ``EmbeddingSharding``s, so tables from two
    collections never reach the same ``group_tables`` call and the runtimes
    they build cannot share HBM. The walk is the only place that knows which
    collection a table came from; flattening it away here is what let
    :func:`~dynamicemb.planner.plan.per_rank_storage` pool budgets across
    collections and under-state the host tier.

    The identity is the **path** to the module rather than its ``id()``: every
    rank runs this walk independently and has to reach the same grouping, and
    object addresses do not agree across processes. The root itself is ``""``,
    which is as sound a key as any other -- a sharder may match the root.

    The paths come from :func:`~dynamicemb.planner.plan.sharded_modules`, which
    is also where `_insert_dynamicemb_plan` gets the ones it files the plan
    under, so the collection a table is recorded against and the path its
    ParameterSharding lands on are one answer rather than two.
    """
    configs: List[Tuple[str, BaseEmbeddingConfig]] = []
    for path, node, _ in sharded_modules(module, sharders):
        for accessor in ("embedding_bag_configs", "embedding_configs"):
            if hasattr(node, accessor):
                configs.extend((path, c) for c in getattr(node, accessor)())
                break
    return configs


class DynamicEmbeddingShardingPlanner(EmbeddingShardingPlanner):
    def __init__(
        self,
        topology: Optional[Topology] = None,
        constraints: Optional[Dict[str, DynamicEmbParameterConstraints]] = None,
        storage_reservation: Optional[StorageReservation] = None,
        node_placer: Optional[NodePlacer] = None,
        **kwargs: Any,
    ):
        """A TorchRec planner that also plans DynamicEmb tables.

        Takes what :class:`~torchrec.distributed.planner.planners.EmbeddingShardingPlanner`
        takes, and nothing else. ``constraints`` is the only argument that means
        anything more here: a :class:`DynamicEmbParameterConstraints` with
        ``use_dynamicemb=True`` marks a table as DynamicEmb's, and those tables
        are planned by :meth:`_plan_dynamicemb` rather than handed to TorchRec.

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
            Keyed by table name, which is what TorchRec is keyed by and what
            every model whose names are unique can keep using. A DynamicEmb
            table in a model where two collections declare one name needs a key
            that names one of them, ``collection.name``; :func:`get_constraint_key`
            returns it, and returns the bare name where that is the answer.
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
        all_constraints: Dict[str, DynamicEmbParameterConstraints] = constraints or {}
        # Decisions only. The options a table was planned with stay on its
        # constraint, which is where they came from -- a second handle on the
        # plan record is one more thing that can drift from it, and is what
        # forced the plan to be stripped before the broadcast and patched
        # after. See `_attach_options`.
        self._dyn_emb_plan: Dict[TableKey, ParameterSharding] = {}
        self._planned_dynamicemb = False
        self._table_poolings: Dict[TableKey, object] = {}
        # Which constraint each table occurrence resolved to, from
        # `resolve_constraints`. The options live on the constraint, so this is
        # also how a table reaches its own options -- and why two occurrences
        # of one name can hold different ones.
        self._resolved_constraints: Dict[TableKey, DynamicEmbParameterConstraints] = {}
        self._reduced_topology: Optional[Topology] = None
        self._node_placer: NodePlacer = node_placer or BalancedNodePlacer()

        super().__init__(
            topology=topology,
            constraints={
                name: constraint
                for name, constraint in all_constraints.items()
                if not constraint.use_dynamicemb
            },
            storage_reservation=storage_reservation,
            **kwargs,
        )
        # After super().__init__: TorchRec assigns self._constraints itself and would discard the DynamicEmb entries.
        self._constraints: Dict[str, DynamicEmbParameterConstraints] = all_constraints

    def _plan_dynamicemb(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> None:
        """Step 1: plan the DynamicEmb tables, from the module's own table configs.

        Their capacity comes from ``DynamicEmbTableOptions`` and their placement
        from the key -> rank rule, so this is arithmetic over the constraints
        rather than a search, and it does not need TorchRec to have planned
        anything yet.

        It runs on every rank, outside the broadcast in :meth:`collective_plan`,
        for two reasons. The options it fills in stay on the rank that filled
        them -- they are the half of a DynamicEmb table that does not cross the
        wire -- and :meth:`_attach_options` needs them everywhere, while
        :meth:`_plan_torchrec` runs on rank 0 alone. What does need one rank's
        authority is the placement, and that is what the broadcast covers.

        It ends by deducting what these tables cost from the Topology, which is
        also where overcommitment is refused -- here rather than in
        :meth:`_plan_torchrec` so that every rank refuses, not just rank 0.
        See the comment at that line.

        Once per planner: the options are filled in place, and a second pass
        would re-align already aligned capacities and warn about it. The
        reduced Topology is kept for the same reason, and `self._topology`
        itself is left alone -- it is what the machine has, and every `plan`
        call has to start from that.
        """
        if self._planned_dynamicemb:
            return

        # One source for both, so what a table's capacity is divided by and what
        # its shards are placed on cannot disagree. The Topology is the planner's
        # statement of the machine; `dist` would be a second opinion (F11).
        world_size = self._topology.world_size
        local_size = self._topology.local_world_size

        # Every table, not just the DynamicEmb ones: resolving the constraints
        # checks that they and the model describe the same set of tables, which
        # is only a check if it sees both sides whole.
        paths_and_configs = _table_configs(module, sharders)
        by_key = {
            TableKey(path, config.name): config for path, config in paths_and_configs
        }
        # Each occurrence's own constraint, so a name declared in two
        # collections can hold two sets of options -- and so a constraint that
        # would reach neither, or both, is refused rather than ignored.
        self._resolved_constraints = resolve_constraints(
            paths_and_configs, self._constraints
        )
        _prepare_dynemb_table_options(
            self._resolved_constraints, by_key, world_size, local_size
        )

        dynamicemb_configs = {
            key: config
            for key, config in by_key.items()
            if key in self._resolved_constraints
            and self._resolved_constraints[key].use_dynamicemb
        }

        nodes = self._choose_nodes(
            dynamicemb_configs, module, sharders, world_size, local_size
        )
        compute_device = self._topology.compute_device

        for key, config in dynamicemb_configs.items():
            constraint = self._resolved_constraints[key]
            opts = constraint.dynamicemb_options
            sharding_type = _sharding_type_of(constraint)
            rows = opts.max_capacity

            if sharding_type != ShardingType.ROW_WISE.value:
                # Written back rather than only used here: downstream reads the
                # options, not the plan, so a table the NodePlacer placed would
                # otherwise look unplaced. Every rank ran the same placement, so
                # every rank writes the same thing.
                #
                # Only here. A row-wise table keeps whatever the caller set, so
                # that `table_ranks` can refuse a node_index on one -- writing
                # None over it first would turn a refusal into a silent
                # correction, which is the opposite of what that check is for.
                opts.node_index = nodes[key]

            # Row-wise is every rank; table-row-wise is one node's ranks. The
            # shard metadata follows, so there are `local_size` entries for a
            # TRW table rather than `world_size` (§4.1).
            ranks = table_ranks(sharding_type, opts.node_index, world_size, local_size)

            self._dyn_emb_plan[key] = ParameterSharding(
                sharding_spec=EnumerableShardingSpec(
                    [
                        ShardMetadata(
                            shard_sizes=[rows, config.embedding_dim],
                            # TODO:0 is we don't have column-wise sharding now
                            shard_offsets=[rows * i, 0],
                            placement=placement(compute_device, rank, local_size),
                        )
                        for i, rank in enumerate(ranks)
                    ]
                ),
                sharding_type=sharding_type,
                ranks=ranks,
                compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
            )

        # What TorchRec groups on that DynamicEmbTableOptions does not carry.
        # Recorded here because `_plan_torchrec` costs the plan later and has
        # only the ParameterShardings, which do not mention pooling.
        self._table_poolings = {
            key: _pooling_of(config) for key, config in dynamicemb_configs.items()
        }

        # What they take out of this rank's budget, and whether a rank has any
        # left. Done here rather than in `_plan_torchrec` because that runs on
        # rank 0 alone, inside the broadcast: a refusal raised there unwinds
        # rank 0 and leaves every other rank waiting on a broadcast that will
        # never be sent, so the one place that catches overcommitment would
        # report it as a hang. Everything it reads is on every rank by now, so
        # every rank can refuse together.
        self._reduced_topology = topology_minus(
            self._topology,
            per_rank_storage(
                self._dyn_emb_plan,
                self._dyn_emb_options,
                optimizer_types(module, sharders),
                len(self._topology.devices),
                self._table_poolings,
            ),
        )
        self._planned_dynamicemb = True

    @property
    def _dyn_emb_options(self) -> Dict[TableKey, DynamicEmbTableOptions]:
        """Each planned table's options, from the one place they live.

        The plan record holds decisions; this reads the other half off the
        constraints the decisions were made from, so the two cannot drift.
        """
        return {
            key: self._resolved_constraints[key].dynamicemb_options
            for key in self._dyn_emb_plan
        }

    def _choose_nodes(
        self,
        dynamicemb_configs: Dict[TableKey, BaseEmbeddingConfig],
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        world_size: int,
        local_size: int,
    ) -> Dict[TableKey, Optional[int]]:
        """Which node each table-row-wise table lands on.

        Row-wise tables get ``None``: they are on every rank, so there is no
        node to name, and `table_ranks` refuses one.

        The work goes to a :class:`~dynamicemb.planner.placement.NodePlacer`,
        which the caller can replace. What this does is prepare its inputs --
        what each table costs a rank, and what the row-wise tables have already
        taken from every rank, so the placer is fitting into the room that will
        actually be there rather than into an empty machine.

        `node_index` is passed through as a pin rather than acted on here, so a
        caller who has named some nodes and left others to the placer gets both,
        and the placer can fit the rest around what it was told.
        """
        optimizers = optimizer_types(module, sharders)
        by_type: Dict[str, List[TableKey]] = {}
        for key in dynamicemb_configs:
            by_type.setdefault(
                _sharding_type_of(self._resolved_constraints[key]), []
            ).append(key)

        trw = by_type.get(ShardingType.TABLE_ROW_WISE.value, [])
        if not trw:
            return {}

        def group_member(key: TableKey):
            return (
                self._resolved_constraints[key].dynamicemb_options,
                optimizers.get(key),
            )

        # The row-wise tables are on every rank, so they shift no choice between
        # nodes -- but they do decide whether a table-row-wise one still fits.
        # Costed in groups, and separately from the table-row-wise ones: TorchRec
        # splits tables by sharding type into different `EmbeddingSharding`s
        # before `group_tables` runs inside each (`embeddingbag.py:267`,
        # `rw_sharding.py:178`), so a row-wise table never fuses with a
        # table-row-wise one however alike their options.
        committed = [make_storage() for _ in range(world_size)]
        row_wise_groups: Dict[object, list] = defaultdict(list)
        for key in by_type.get(ShardingType.ROW_WISE.value, []):
            row_wise_groups[
                fusion_key(
                    ShardingType.ROW_WISE.value,
                    self._resolved_constraints[key].dynamicemb_options,
                    optimizers.get(key),
                    _pooling_of(dynamicemb_configs[key]),
                    key.collection,
                )
            ].append(group_member(key))
        for members in row_wise_groups.values():
            hbm, ddr = get_group_value_bytes_by_tier(members)
            cost = make_storage(hbm=hbm, ddr=ddr)
            committed = [spent + cost for spent in committed]

        # The tables are handed over whole rather than priced: what one costs a
        # node depends on which of the others land there with it, and that is
        # the decision the placer is making.
        return self._node_placer.place(
            [
                TableToPlace(
                    key=key,
                    options=self._resolved_constraints[key].dynamicemb_options,
                    optimizer_type=optimizers.get(key),
                    pooling=_pooling_of(dynamicemb_configs[key]),
                    pinned_to=self._resolved_constraints[
                        key
                    ].dynamicemb_options.node_index,
                )
                for key in sorted(trw)
            ],
            self._topology,
            committed,
        )

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
        self._plan_dynamicemb(module, sharders)
        return self._attach_options(self._plan_torchrec(module, sharders))

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
        sizes and offsets and placements -- as a plain ``ParameterSharding``,
        which is the type that holds exactly those and nothing else. The
        ``DynamicEmbTableOptions`` do not travel, because some of what they
        hold is rank-local by nature: ``external_storage`` is a live handle to
        a store this process opened, and ``score_function`` is a Python
        callable. Sending rank 0's copy of either would be wrong even where
        pickle allows it.

        That works out because ``_dyn_emb_plan`` records the decisions and
        nothing else -- the options a table was planned with stay on its
        constraint -- so :meth:`_insert_dynamicemb_plan` puts its own record
        straight into the plan and there is nothing to take off it.
        :meth:`_attach_options` then turns each of those entries into a
        ``DynamicEmbParameterSharding`` with this rank's options. The plan is
        built up on arrival rather than stripped before departure, and no
        instance of that class exists without its options at any point.

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
        # Step 1 runs on every rank: the options it fills in are the half of a
        # DynamicEmb table that stays local, and _attach_options needs them
        # everywhere. Only the placement needs one rank's authority.
        self._plan_dynamicemb(module, sharders)
        return self._attach_options(
            invoke_on_rank_and_broadcast_result(
                pg, 0, self._plan_torchrec, module, sharders
            )
        )

    def _plan_torchrec(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> ShardingPlan:
        """Steps 2 to 4: take up the reduced budget, plan the rest, assemble.

        By the time this runs, :meth:`_plan_dynamicemb` has produced a
        ParameterSharding for each DynamicEmb table and has already taken what
        they cost out of the Topology, per rank, so:

        2. this planner's Topology becomes that reduced one;
        3. TorchRec plans the remaining tables inside what is left, on a copy of
           the module those tables have been taken out of;
        4. both halves are assembled into one ShardingPlan.

        The deduction itself is step 1's, not this method's, because this one
        runs on rank 0 alone: it can refuse, and a refusal raised in here would
        reach no other rank. See :meth:`_plan_dynamicemb`.

        The Topology is assigned rather than passed, because the enumerator and
        the estimators were built against ``self._topology`` in ``__init__`` and
        keep that object. Assigning still reaches the decision that depends on
        capacity: ``reserve`` is called with ``self._topology`` at plan time
        (``planner/planners.py:832``) and what it returns is the budget the
        partitioner is given. What those components kept, they kept as scalars
        -- ``compute_device``, ``world_size``, ``intra_group_size`` -- which
        this deduction does not change, so an in-place edit would reach no
        further. Design doc §7.3 lists them, and the two it does not reach.

        It is put back afterwards: the caller's Topology says what the machine
        has, and a second `plan` call has to start from that again rather than
        from a machine that already looks spent.

        Step 4 walks the caller's module, not the reduced one: the reduced one
        no longer admits to having these tables, which is the whole point of it,
        so it cannot say what path they live under.

        The per-table options are left off what goes on the plan; see
        :meth:`collective_plan`.
        """
        # 2. the budget with the DynamicEmb tables taken out, from step 1
        assert self._reduced_topology is not None, "_plan_dynamicemb has not run"
        whole_topology = self._topology
        try:
            self._topology = self._reduced_topology

            # 3. TorchRec plans the tables that are left, in the memory that is left
            reduced_module = module_without_tables(
                module, set(self._dyn_emb_plan), sharders
            )
            torchrec_plan = super().plan(reduced_module, sharders)
        finally:
            self._topology = whole_topology

        # 4. assemble
        self._insert_dynamicemb_plan(torchrec_plan, module, sharders)
        return torchrec_plan

    def _attach_options(self, plan: ShardingPlan) -> ShardingPlan:
        """Complete each DynamicEmb table's entry with this rank's options.

        The entries arrive as plain ``ParameterSharding``s -- decisions only,
        which is all :meth:`collective_plan` sends -- and leave as
        ``DynamicEmbParameterSharding``s. The entry is replaced rather than
        written into, because what is being added is the half of the table the
        wire cannot carry, not a correction to what it did.

        The plan's own shard metadata is checked against the options when the
        kernel is built (``_get_dynamicemb_options_per_table``), so a rank
        whose options disagree with the decisions it was sent does not go
        unnoticed.
        """
        options = self._dyn_emb_options
        for path, module_plan in plan.plan.items():
            for table_name, decisions in list(module_plan.items()):
                # By the path the plan filed it under, which is the collection
                # the table was planned in. Matching on the name alone would
                # hand one collection's options to a namesake in another.
                table_options = options.get(TableKey(path, table_name))
                if table_options is not None:
                    module_plan[table_name] = DynamicEmbParameterSharding.with_options(
                        decisions, table_options
                    )
        return plan

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
        would have discovered on our behalf. :func:`sharded_modules` makes that
        descent, by the enumerator's own rule -- a module a sharder claims is a
        leaf, anything else is recursed into -- because the paths have to be the
        ones the planner would have produced. The same function gives
        `_table_configs` the collection it records a table against, so the two
        cannot drift: a path only this one knew about would file a plan under a
        name `DistributedModelParallel` never looks for, and it leaves modules
        its plan does not name alone rather than complaining.

        The planner's own records go in as they are. They hold the decisions
        and nothing else, which is exactly what the plan carries and all that
        crosses the wire; :meth:`_attach_options` completes them with each
        rank's options, and copies the shard metadata in doing so.
        """
        if not self._dyn_emb_plan:
            return

        placed: Set[TableKey] = set()
        for path, child_module, sharder in sharded_modules(module, sharders):
            for table_name in sharder.shardable_parameters(child_module):
                key = TableKey(path, table_name)
                parameter_sharding = self._dyn_emb_plan.get(key)
                if parameter_sharding is None:
                    continue
                module_plan = torchrec_plan.plan.setdefault(
                    path, EmbeddingModuleShardingPlan()
                )
                module_plan[table_name] = parameter_sharding
                placed.add(key)

        missing = set(self._dyn_emb_plan) - placed
        if missing:
            raise RuntimeError(
                f"No sharder claims a module holding the DynamicEmb tables "
                f"{sorted(str(key) for key in missing)}, so they cannot be "
                f"placed in the plan. "
                "Pass a DynamicEmb sharder for every collection that has one."
            )
