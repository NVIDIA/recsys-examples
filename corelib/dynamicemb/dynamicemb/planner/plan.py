# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""What the DynamicEmb tables cost, and what that leaves for TorchRec.

DynamicEmb settles its own tables before TorchRec plans anything: their capacity
comes from ``DynamicEmbTableOptions`` and their placement from the key -> rank
rule, so there is nothing for a proposer to propose. TorchRec is then given a
:class:`~torchrec.distributed.planner.types.Topology` with that cost already
taken out, and plans the remaining tables inside what is left.

Handing TorchRec a smaller Topology, rather than a StorageReservation that
subtracts on its behalf, keeps the caller's own ``storage_reservation`` doing
exactly what the caller expects -- it sees an ordinary Topology and applies its
own policy to it. The alternative would have to wrap it, which means deciding
who subtracts first and breaking the ``isinstance`` checks TorchRec's stats and
error messages use (``planner/planners.py:1138``, ``planner/stats.py:1109``).
"""

import copy
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torch import nn
from torchrec.distributed.planner.types import (
    PlannerError,
    PlannerErrorType,
    Storage,
    Topology,
)
from torchrec.distributed.planner.utils import storage_repr_in_gb
from torchrec.distributed.planner.utils import sharder_name
from torchrec.distributed.utils import optimizer_type_to_emb_opt_type
from torchrec.distributed.types import (
    EnumerableShardingSpec,
    ModuleSharder,
    ParameterSharding,
    ShardingType,
    ShardMetadata,
)

from ..dynamicemb_config import get_group_value_bytes_by_tier

_STORAGE_HAS_SSD = "ssd" in Storage.__dataclass_fields__


def make_storage(hbm: int = 0, ddr: int = 0, ssd: int = 0) -> Storage:
    """A Storage on any TorchRec: the ssd field only exists from 1.5.0."""
    if _STORAGE_HAS_SSD:
        return Storage(hbm=hbm, ddr=ddr, ssd=ssd)
    return Storage(hbm=hbm, ddr=ddr)


def storage_ssd(storage: Storage) -> int:
    """``storage.ssd`` where the field exists, 0 where it does not.

    The companion to :func:`make_storage`: that one covers building a Storage
    on TorchRec before 1.5.0, this one covers reading the field back. Both are
    needed -- the first alone leaves `.ssd` to raise AttributeError.
    """
    return storage.ssd if _STORAGE_HAS_SSD else 0


__all__ = [
    "optimizer_types",
    "shard_rank",
    "per_rank_storage",
    "topology_minus",
    "make_storage",
    "fusion_key",
    "module_without_tables",
    "storage_ssd",
    "table_fanout",
    "table_layout",
]


def shard_rank(shard: ShardMetadata) -> int:
    """The rank a shard is placed on.

    ``placement`` is either the ``"rank:{r}/{device}"`` string that
    ``torchrec.distributed.sharding_plan.placement`` builds or the
    ``_remote_device`` it gets parsed into -- TorchRec itself handles both
    (``distributed/embeddingbag.py:184-185``), so this does too.
    """
    placement = shard.placement
    if placement is None:
        raise ValueError(f"Shard {shard} has no placement, so its rank is unknown.")
    if isinstance(placement, str):
        return int(placement.split("/", 1)[0].removeprefix("rank:"))
    return placement.rank()


def optimizer_types(
    module: nn.Module,
    sharders: List[ModuleSharder[nn.Module]],
) -> Dict[str, Optional[OptimType]]:
    """The optimizer each table trains with, by table name, from either convention.

    TorchRec lets a model say this two ways, and DynamicEmb's own tests use both.
    `apply_optimizer_in_backward` puts the class on the parameter, which is where
    TorchRec's estimator reads it (`shard_estimators.py`) and what
    `optimizer_type_to_emb_opt_type` exists to convert; a sharder's `fused_params`
    carries the EmbOptimType directly. The parameter wins where both are present,
    matching `merge_fused_params`.

    A table with neither is reported as ``None``: the caller reserves its rows
    and nothing for optimizer state, which keeps the reservation a floor rather
    than a guess.
    """
    from_sharders: Optional[OptimType] = None
    for sharder in sharders:
        fused_params = getattr(sharder, "fused_params", None) or {}
        from_sharders = fused_params.get("optimizer", from_sharders)

    per_table: Dict[str, Optional[OptimType]] = {}
    for param_name, param in module.named_parameters():
        parts = param_name.split(".")
        if len(parts) < 2:
            continue
        # `embeddings.<table>.weight` for an EC, `embedding_bags.<table>.weight`
        # for an EBC, either of them arbitrarily nested.
        table_name = parts[-2]
        optimizer_classes = getattr(param, "_optimizer_classes", None)
        if optimizer_classes:
            try:
                per_table[table_name] = optimizer_type_to_emb_opt_type(
                    optimizer_classes[0]
                )
                continue
            except ValueError:
                # An optimizer TorchRec has no EmbOptimType for -- DynamicEmb's
                # own FTRL is one -- so fall through to what the sharder says.
                pass
        per_table[table_name] = from_sharders
    return per_table


def fusion_key(
    sharding_type: str,
    options: "object",
    optimizer_type: Optional[OptimType] = None,
    pooling: "object" = None,
) -> Tuple:
    """What two tables must agree on before they can be costed as one.

    TorchRec fuses tables within a sharding that match on its grouping key
    (``embedding_sharding.py:609``): data type, pooling, a dimension bucket,
    and the fused params -- inside which ``DynamicEmbTableOptions`` rides, with
    its own key, and the optimizer beside it. This reproduces the part of that
    we can know without reaching into TorchRec's internals.

    What is here: the **sharding type**, since tables are split by it into
    separate ``EmbeddingSharding``s before grouping runs inside each
    (``embeddingbag.py:267``, ``rw_sharding.py:178``); the **options**, which
    hash on their own grouping key; the **optimizer**, since its state is part
    of a row and TorchRec keys on it through the fused params; the **data
    type**, which ``DynamicEmbTableOptions.get_grouped_key`` leaves out; and
    the **pooling**, likewise not DynamicEmb's to know but cheap to pass.

    What is not: TorchRec's dimension bucket, and the rest of the fused params.
    Those need `_get_grouping_fused_params`, the bucketer and
    `_prefetch_and_cached` -- upstream internals that would have to be tracked
    as they change, which §4.9 is about not doing.

    So this stays **coarser** than the truth, and coarser is the direction that
    errs safely: merging tables the runtime keeps apart moves
    ``min(sum total, sum budget)`` up, over-stating HBM. A plan can be refused
    that would have fit; none is accepted that will not. Every attribute added
    here narrows that gap.
    """
    return (
        sharding_type,
        options,
        optimizer_type,
        getattr(options, "embedding_dtype", None),
        pooling,
    )


def per_rank_storage(
    parameter_shardings: Dict[str, ParameterSharding],
    optimizer_types: Dict[str, Optional[OptimType]],
    world_size: int,
    poolings: Optional[Dict[str, object]] = None,
) -> List[Storage]:
    """What the DynamicEmb tables take from each rank, as one Storage per rank.

    Per rank rather than one figure repeated, because a table need not be on
    every rank: under ``TABLE_ROW_WISE`` it is on one node's ranks and on no
    others, so a single per-rank number would charge every rank for a table most
    of them do not hold.

    The bytes come from the plan's own shard metadata rather than from the
    constraints the plan was built from, so that what is subtracted here and what
    is handed to the runtime are the same decision read twice, not the same
    arithmetic done twice.

    **Tables are costed in groups, because the runtime fuses them.** A group's
    HBM budget is the sum of its members', spent on the sum of their sizes, so a
    table whose budget exceeds its size lends the remainder to its groupmates.
    Costing each table alone gives ``sum(min(total, budget))``, which is at most
    ``min(sum total, sum budget)`` -- it under-states HBM, and that is the
    direction that lets a plan through which the machine will not hold.

    The grouping used here is the sharding type paired with
    ``DynamicEmbTableOptions``'s own key
    (:meth:`~dynamicemb.dynamicemb_config.DynamicEmbTableOptions.get_grouped_key`).
    The sharding type belongs in it because TorchRec splits tables by it into
    separate ``EmbeddingSharding``s before ``group_tables`` runs inside each
    (``embeddingbag.py:267``, ``rw_sharding.py:178``), so a row-wise table never
    fuses with a table-row-wise one however alike their options.

    It is still **coarser** than the runtime's, which also keys on data type,
    pooling, a dimension bucket and the fused params
    (``embedding_sharding.py:609``). Reproducing that exactly would mean
    reimplementing `_get_grouping_fused_params`, the dimension bucketer and
    `_prefetch_and_cached` here, and then owning them as TorchRec changes them
    -- the kind of fork §4.9 exists to avoid.

    Coarser is the safe inexactness. Merging tables the runtime would keep apart
    only moves ``min(sum total, sum budget)`` up, so this over-states HBM rather
    than under-stating it: a plan may be refused that would have fit, and none
    is accepted that will not.

    ``optimizer_types`` maps table name -> ``OptimType``; a table missing from it
    is counted without optimizer state, which is a floor rather than a guess (see
    :func:`~dynamicemb.dynamicemb_config.get_group_value_bytes_by_tier`).
    """
    totals = [make_storage() for _ in range(world_size)]

    # rank -> the tables that rank holds, bucketed by what the runtime fuses on.
    # A dict keyed by the options themselves: DynamicEmbTableOptions hashes and
    # compares on its grouping key, so two tables land together exactly when
    # they would be fused.
    per_rank_groups: List[Dict[object, List[Tuple[object, Optional[OptimType]]]]] = [
        defaultdict(list) for _ in range(world_size)
    ]

    for name, parameter_sharding in parameter_shardings.items():
        options = getattr(parameter_sharding, "dynamicemb_options", None)
        if options is None:
            raise ValueError(
                f"Table {name!r} has no DynamicEmbTableOptions on its "
                "ParameterSharding, so its footprint cannot be sized."
            )

        spec = parameter_sharding.sharding_spec
        if not isinstance(spec, EnumerableShardingSpec) or not spec.shards:
            raise ValueError(
                f"Table {name!r} has no EnumerableShardingSpec, so which ranks "
                "hold it is unknown."
            )

        # Options carry max_capacity and local_hbm_for_values already divided by
        # the table's fan-out, so every shard of a table weighs the same and
        # only the set of ranks differs.
        member = (options, optimizer_types.get(name))
        key = fusion_key(
            parameter_sharding.sharding_type,
            options,
            optimizer_types.get(name),
            (poolings or {}).get(name),
        )
        for shard in spec.shards:
            rank = shard_rank(shard)
            if not 0 <= rank < world_size:
                raise ValueError(
                    f"Table {name!r} places a shard on rank {rank}, which is "
                    f"outside the world of {world_size}."
                )
            per_rank_groups[rank][key].append(member)

    for rank, groups in enumerate(per_rank_groups):
        for members in groups.values():
            hbm, ddr = get_group_value_bytes_by_tier(members)
            totals[rank] += make_storage(hbm=hbm, ddr=ddr)

    return totals


def topology_minus(topology: Topology, per_rank: List[Storage]) -> Topology:
    """``topology`` with each rank's DynamicEmb footprint taken out of it.

    Returns a copy: the caller's Topology is an input they may reuse, and
    TorchRec's planner keeps a reference to whatever it is given.

    **Refuses a rank the DynamicEmb tables have already overdrawn**, rather than
    clamping to zero or passing the negative on. This is the one place where
    what DynamicEmb spends meets what the machine has, so it is where every
    route to overcommitting it converges -- a pinned table on a node too small
    for it, a placement the placer accepted on numbers it costs per table, or
    row-wise tables that simply add up to more than a rank holds.

    Passing it on was the earlier behaviour, on the reasoning that TorchRec's
    own reservation reports the shortfall. It does not: it tests
    ``devices[0].storage.hbm < 0`` and no other rank
    (``planner/storage_reservations.py:509``), so an overdraft anywhere but
    rank 0 survives it, and the check belongs only to
    ``HeuristicalStorageReservation`` -- a caller who passes a different one
    loses even that. Clamping would be worse: zero reads as "no room, plan
    accordingly" and the plan would come back looking feasible.
    """
    if len(per_rank) != len(topology.devices):
        raise ValueError(
            f"per_rank has {len(per_rank)} entries but the topology has "
            f"{len(topology.devices)} devices."
        )

    reduced = copy.deepcopy(topology)
    overdrawn: List[Tuple[int, Storage, Storage]] = []
    for device in reduced.devices:
        spent = per_rank[device.rank]
        before = device.storage
        device.storage = before - spent
        if device.storage.hbm < 0 or device.storage.ddr < 0:
            overdrawn.append((device.rank, before, spent))

    if overdrawn:
        worst = min(overdrawn, key=lambda o: min(o[1].hbm - o[2].hbm, 0))
        rank, before, spent = worst
        raise PlannerError(
            error_type=PlannerErrorType.INSUFFICIENT_STORAGE,
            message=(
                f"The DynamicEmb tables do not fit on "
                f"{len(overdrawn)} of {len(reduced.devices)} ranks, leaving "
                "nothing for TorchRec to plan into. Rank "
                f"{rank} has {storage_repr_in_gb(before)} and they take "
                f"{storage_repr_in_gb(spent)}. Lower global_hbm_for_values, "
                "shard the largest tables row-wise so they spread over every "
                "rank instead of one node, or state a Topology that matches "
                "the machine."
            ),
        )
    return reduced


# EmbeddingBagCollection keeps its tables in `embedding_bags`, EmbeddingCollection
# in `embeddings`. Both are nn.ModuleDicts keyed by table name, which is what
# `shardable_parameters` reads.
_TABLE_CONTAINERS = ("embedding_bags", "embeddings")


def _collection_without(collection: nn.Module, table_names: Set[str]) -> nn.Module:
    """A copy of one collection with the named tables gone from its ModuleDict.

    The ModuleDict is pruned rather than the collection rebuilt from a subset of
    its configs, because rebuilding needs constructor arguments that are not all
    recoverable -- ``EmbeddingCollection`` takes ``need_indices``,
    ``use_gather_select`` and ``use_gather_select_per_sharding``, none of which
    have public accessors -- and because it would drop the subclass a caller may
    have passed, which `sharder_name(type(module))` matches on.

    ``_embedding_bag_configs`` and ``_feature_names`` are deliberately left
    whole. Planning reads the tables through ``shardable_parameters``, which goes
    through the ModuleDict, so the enumerator no longer sees them; the storage
    reservation sizes the input KJT from the feature names, and the KJT really
    does carry the DynamicEmb features at runtime, so leaving them counted is the
    accurate answer rather than a leftover.
    """
    for attribute in _TABLE_CONTAINERS:
        tables = getattr(collection, attribute, None)
        if isinstance(tables, nn.ModuleDict):
            break
    else:
        raise ValueError(
            f"{type(collection).__name__} holds its tables in neither "
            f"{' nor '.join(_TABLE_CONTAINERS)}, so the DynamicEmb ones cannot "
            "be taken out of it for TorchRec to plan the rest."
        )

    kept = nn.ModuleDict()
    for name, table in tables.items():
        if name not in table_names:
            kept[name] = table

    reduced = copy.copy(collection)
    # copy.copy shares __dict__, so _modules is the same dict object until it is
    # replaced; without this the assignment below would reach into the caller's
    # module.
    reduced._modules = dict(collection._modules)
    setattr(reduced, attribute, kept)
    return reduced


def module_without_tables(
    module: nn.Module,
    table_names: Set[str],
    sharders: List[ModuleSharder[nn.Module]],
) -> nn.Module:
    """``module`` as TorchRec should plan it: without the DynamicEmb tables.

    TorchRec's enumerator walks the module it is given and enumerates every table
    a sharder claims, so tables DynamicEmb has already placed have to be out of
    that module rather than filtered out of the result. Filtering afterwards
    still runs the estimators over them, which sizes shards from a capacity that
    is not what a DynamicEmb table means by ``num_embeddings``.

    Only the modules on the path to a pruned collection are copied; everything
    else is shared, and the module the caller passes is not touched. The tree
    keeps its shape, so the paths in the returned plan are the paths the caller's
    module has -- which is what `ShardingPlan` is keyed by.

    A collection whose tables are all DynamicEmb becomes an empty one rather than
    disappearing: `shardable_parameters` then yields nothing and the enumerator
    moves on, while removing it outright would change the tree the paths come
    from.
    """
    sharder_map = {sharder_name(sharder.module_type): sharder for sharder in sharders}

    def prune(node: nn.Module) -> nn.Module:
        sharder = sharder_map.get(sharder_name(type(node)))
        if sharder is not None:
            owned = set(sharder.shardable_parameters(node)) & table_names
            return _collection_without(node, owned) if owned else node

        replacements = {
            name: pruned
            for name, child in node.named_children()
            if (pruned := prune(child)) is not child
        }
        if not replacements:
            return node

        copied = copy.copy(node)
        copied._modules = dict(node._modules)
        for name, child in replacements.items():
            setattr(copied, name, child)
        return copied

    return prune(module)


def table_fanout(sharding_type: str, world_size: int, local_size: int) -> int:
    """How many pieces a table's rows are split into.

    ``world_size`` for row-wise, which sits on every rank; ``local_size`` for
    table-row-wise, which sits on one node. This is what a table's per-rank
    capacity and HBM budget are divided by, and what the bucketizer distributes
    keys over.

    It does not depend on *which* node a table-row-wise table lands on, only on
    how big a node is -- so capacity can be settled before placement is, and an
    automatic placer can size a table before it decides where to put it.

    Raises:
        ValueError: an unsupported sharding type, or a world that does not
            divide into whole nodes.
    """
    if sharding_type == ShardingType.ROW_WISE.value:
        return world_size
    if sharding_type != ShardingType.TABLE_ROW_WISE.value:
        raise ValueError(
            f"DynamicEmb tables support {ShardingType.ROW_WISE.value} and "
            f"{ShardingType.TABLE_ROW_WISE.value}, not {sharding_type!r}."
        )
    if local_size <= 0 or world_size % local_size:
        raise ValueError(
            f"A world of {world_size} does not divide into nodes of {local_size}, "
            f"so {ShardingType.TABLE_ROW_WISE.value} has no node to place a table on."
        )
    return local_size


def table_layout(
    sharding_type: str,
    host_index: Optional[int],
    world_size: int,
    local_size: int,
) -> List[int]:
    """The ranks a DynamicEmb table's shards sit on, in shard order.

    Row-wise puts a shard on every rank, so the answer is every rank and
    ``host_index`` means nothing -- there is no node to name when the table is on
    all of them. Table-row-wise puts the table on one node, so the answer is that
    node's ranks and ``host_index`` says which node.

    The length is :func:`table_fanout`.

    Raises:
        ValueError: whatever :func:`table_fanout` raises, or a ``host_index``
            that is missing, out of range, or set on a row-wise table.
    """
    fanout = table_fanout(sharding_type, world_size, local_size)

    if sharding_type == ShardingType.ROW_WISE.value:
        if host_index is not None:
            raise ValueError(
                f"host_index={host_index} was set on a {sharding_type} table. It "
                "says which node holds a table, and a row-wise table is on every "
                "rank of every node."
            )
        return list(range(world_size))

    num_nodes = world_size // fanout
    if host_index is None:
        raise ValueError(
            f"A {sharding_type} table needs a host_index saying which of the "
            f"{num_nodes} nodes holds it. Set DynamicEmbTableOptions.host_index, "
            "or leave it unset and let the planner's host placer choose."
        )
    if not 0 <= host_index < num_nodes:
        raise ValueError(
            f"host_index={host_index} is outside the {num_nodes} nodes of a world "
            f"of {world_size} with {local_size} ranks each."
        )
    base = host_index * fanout
    return list(range(base, base + fanout))
