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
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Set, Tuple

from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torch import nn
from torchrec.distributed.planner.types import (
    PlannerError,
    PlannerErrorType,
    Storage,
    Topology,
)
from torchrec.distributed.planner.utils import sharder_name, storage_repr_in_gb
from torchrec.distributed.types import (
    EnumerableShardingSpec,
    ModuleSharder,
    ParameterSharding,
    ShardingType,
    ShardMetadata,
)
from torchrec.distributed.utils import optimizer_type_to_emb_opt_type

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
    "TableKey",
    "optimizer_types",
    "sharded_modules",
    "shard_rank",
    "per_rank_storage",
    "topology_minus",
    "make_storage",
    "fusion_key",
    "module_without_tables",
    "storage_ssd",
    "table_fanout",
    "table_ranks",
]


@dataclass(frozen=True, order=True)
class TableKey:
    """Which table, as TorchRec identifies one: a collection and a name.

    TorchRec names a sharding option ``module_path + "." + name``
    (``planner/types.py``) and keys a ``ShardingPlan`` by ``plan[path][name]``,
    because each shardable module is sharded on its own. A name alone is
    therefore not an identity: two collections may each declare ``"item_id"``,
    and those are two tables -- separate storage, separate optimizer state,
    and neither sees the other's updates.

    DynamicEmb keyed everything by name until this type existed, which made a
    name's second occurrence share the first one's ``DynamicEmbTableOptions``.
    Those carry geometry, written into them at plan time -- ``dim``,
    ``max_capacity``, ``embedding_dtype`` -- so one object cannot describe two
    tables, and the planner refused the model outright rather than mis-size
    them. Keying by the pair is what lets each occurrence hold its own
    options, and `get_constraint_key` is how a caller addresses one.

    ``collection`` is the **path** to the module, not its ``id()``: every rank
    runs the walk on its own and has to reach the same key, and object
    addresses do not agree across processes. The root's path is ``""``, which
    is as sound a collection as any other -- a sharder may claim the root.
    """

    collection: str
    name: str

    def __str__(self) -> str:
        """``collection.name``, or the bare name at the root.

        The form a caller writes in ``constraints`` and the form error messages
        report, so what is said back is what can be typed in.
        """
        return f"{self.collection}.{self.name}" if self.collection else self.name

    @property
    def constraint_keys(self) -> Tuple[str, ...]:
        """The keys this table answers to in ``constraints``, most specific first.

        Qualified before bare, so a constraint naming this one collection wins
        over one naming the table wherever it appears. That order is what makes
        a bare name usable as the default it has always been while a single
        collection overrides it -- and it is why a model that never duplicates
        a name needs no change: its bare keys still match, there being nothing
        more specific to lose to.

        At the root the two forms coincide, and there is one.
        """
        qualified = str(self)
        return (qualified,) if qualified == self.name else (qualified, self.name)


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


def _parameter_optimizer(param: nn.Parameter) -> Optional[OptimType]:
    """What `apply_optimizer_in_backward` put on a parameter, if anything.

    ``None`` both when nothing was applied and when what was applied has no
    ``EmbOptimType`` -- DynamicEmb's own FTRL is one -- so that the caller falls
    back to what the sharder says rather than to a guess.
    """
    optimizer_classes = getattr(param, "_optimizer_classes", None)
    if not optimizer_classes:
        return None
    try:
        return optimizer_type_to_emb_opt_type(optimizer_classes[0])
    except ValueError:
        return None


def sharded_modules(
    module: nn.Module,
    sharders: List[ModuleSharder[nn.Module]],
) -> Iterator[Tuple[str, nn.Module, ModuleSharder[nn.Module]]]:
    """Every module a sharder claims, with the path ``ShardingPlan`` keys it by.

    The descent the planner and the enumerator both make: a module some sharder
    claims is a leaf, and anything else is recursed into
    (``planner/enumerators.py``). Written once because several callers make it
    and two of them have to agree about the paths it produces --
    :func:`~dynamicemb.planner.planners._table_configs`, which records which
    collection a table came out of, and ``_insert_dynamicemb_plan``, which puts
    the table's ParameterSharding under that path. A plan filed under a path
    `DistributedModelParallel` does not look for is not an error there: it
    shards the modules its plan names and leaves the rest alone, so two walks
    that drifted apart would show up as a collection that silently stayed
    unsharded. One walk cannot drift from itself.

    A generator, so a caller that only wants the modules does not pay for the
    tables, and ``module_without_tables`` is not here: it rebuilds the tree
    from the bottom up rather than reading it, which is a different shape.
    """
    sharder_map = {sharder_name(sharder.module_type): sharder for sharder in sharders}
    queue: List[Tuple[str, nn.Module]] = [("", module)]
    while queue:
        path, node = queue.pop()
        sharder = sharder_map.get(sharder_name(type(node)))
        if sharder is not None:
            yield path, node, sharder
            continue
        # `child_name` is the attribute the parent holds the child under, not
        # a table name -- which is what `name` means nearly everywhere else
        # here. The root's path is "", where a dotted join would leave a
        # leading dot.
        queue.extend(
            (f"{path}.{child_name}" if path else child_name, child)
            for child_name, child in node.named_children()
        )


def optimizer_types(
    module: nn.Module,
    sharders: List[ModuleSharder[nn.Module]],
) -> Dict[TableKey, Optional[OptimType]]:
    """The optimizer each table trains with, by table, from either convention.

    TorchRec lets a model say this two ways, and DynamicEmb's own tests use both.
    `apply_optimizer_in_backward` puts the class on the parameter, which is where
    TorchRec's estimator reads it (`shard_estimators.py`) and what
    `optimizer_type_to_emb_opt_type` exists to convert; a sharder's `fused_params`
    carries the EmbOptimType directly. The parameter wins where both are present,
    matching `merge_fused_params`.

    ``fused_params`` is read off the sharder that claims the table's collection,
    not off the sharder list as a whole. A model may be sharded by several --
    an EmbeddingBagCollection and an EmbeddingCollection each have their own --
    and they need not agree on the optimizer, so flattening them to one value
    would hand some tables another module's answer. TorchRec resolves it the
    same way, by `module_type_key` (`shard_estimators.py:182`).

    Tables are found through `shardable_parameters`, the same accessor
    :func:`module_without_tables` prunes by, so a table that is planned is a
    table that appears here.

    Keyed by :class:`TableKey` rather than by name, since the walk is standing
    in the collection already and two collections may each declare a name --
    and they need not have been given the same optimizer.

    A table with neither convention is reported as ``None``: the caller reserves
    its rows and nothing for optimizer state, which keeps the reservation a
    floor rather than a guess.
    """
    per_table: Dict[TableKey, Optional[OptimType]] = {}
    for path, node, sharder in sharded_modules(module, sharders):
        from_sharder = (getattr(sharder, "fused_params", None) or {}).get("optimizer")
        for table_name, param in sharder.shardable_parameters(node).items():
            applied = _parameter_optimizer(param)
            per_table[TableKey(path, table_name)] = (
                from_sharder if applied is None else applied
            )
    return per_table


def fusion_key(
    sharding_type: str,
    options: "object",
    optimizer_type: Optional[OptimType] = None,
    pooling: "object" = None,
    # New components go after this line, never before it: the callers pass the
    # ones above positionally, so a parameter inserted ahead of them would be
    # handed someone else's argument and quietly coarsen the key instead of
    # narrowing it.
    collection: "object" = None,
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
    type**, which ``DynamicEmbTableOptions.get_grouped_key`` leaves out; the
    **pooling**, likewise not DynamicEmb's to know but cheap to pass; and the
    **owning collection**, since each shardable module is sharded on its own
    -- one ``ShardedEmbedding*Collection`` per collection, each with its own
    ``EmbeddingSharding``s -- so two tables in different collections never
    reach the same ``group_tables`` call however alike everything else about
    them is, and build separate runtimes that cannot lend each other HBM.

    What is not: TorchRec's dimension bucket, and the rest of the fused params.
    Those need `_get_grouping_fused_params`, the bucketer and
    `_prefetch_and_cached` -- upstream internals that would have to be tracked
    as they change, which §4.9 is about not doing.

    So this stays **coarser** than the truth, and coarseness is not safe in both
    directions. Merging tables the runtime keeps apart moves
    ``min(sum total, sum budget)`` up, because ``sum min <= min sum``:

    * **HBM is over-stated**, which only ever refuses a layout that would have
      fit. Nothing is accepted that HBM will not hold.
    * **The host tier is under-stated** by exactly as much, since it is
      ``total - hbm`` and ``total`` does not move. That direction *can* accept
      a layout the runtime will not hold: the planner believes the rows stay in
      HBM, the runtime keeps the tables apart, and the ones that no longer fit
      a budget go to a host tier that was never costed for them.

    The same key is what ``per_rank_storage`` deducts the tables with, so the
    under-statement reaches the Topology handed to TorchRec as well -- the two
    agree, they are simply both short. Every attribute added here narrows that
    gap, which is why one is added rather than left out whenever we can know it
    without reaching into TorchRec's internals. See
    ``test_tables_that_do_not_fuse_do_not_pool_their_budgets``, which is this
    paragraph as a test: drop any one attribute and a layout the host tier
    cannot hold stops being refused.
    """
    return (
        sharding_type,
        options,
        optimizer_type,
        getattr(options, "embedding_dtype", None),
        pooling,
        collection,
    )


def per_rank_storage(
    parameter_shardings: Dict["TableKey", ParameterSharding],
    options: Dict["TableKey", "DynamicEmbTableOptions"],
    optimizer_types: Dict["TableKey", Optional[OptimType]],
    world_size: int,
    poolings: Optional[Dict["TableKey", object]] = None,
) -> List[Storage]:
    """What the DynamicEmb tables take from each rank, as one Storage per rank.

    Per rank rather than one figure repeated, because a table need not be on
    every rank: under ``TABLE_ROW_WISE`` it is on one node's ranks and on no
    others, so a single per-rank number would charge every rank for a table most
    of them do not hold.

    The bytes come from the plan's own shard metadata rather than from the
    constraints the plan was built from, so that what is subtracted here and what
    is handed to the runtime are the same decision read twice, not the same
    arithmetic done twice. The *options* are passed in beside the plan rather
    than read off it: a ParameterSharding records decisions, and the options a
    table was planned with live with the constraints, in one place.

    **Tables are costed in groups, because the runtime fuses them.** A group
    whose summed size fits its summed HBM budget is held entirely in HBM, so
    there a table whose budget exceeds its size covers its groupmates (a group
    that does not fit splits each table on its own budget, and lends nothing).
    Costing each table alone gives ``sum(min(total, budget))``, which is at most
    ``min(sum total, sum budget)`` -- it under-states HBM, and that is the
    direction that lets a plan through which the machine will not hold.

    The grouping used here is :func:`fusion_key`, the same function the placer
    buckets by, so the two cannot disagree about what fuses.

    It is still **coarser** than the runtime's, which also keys on a dimension
    bucket and the rest of the fused params (``embedding_sharding.py:609``).
    Reproducing that exactly would mean reimplementing
    `_get_grouping_fused_params`, the dimension bucketer and
    `_prefetch_and_cached` here, and then owning them as TorchRec changes them
    -- the kind of fork §4.9 exists to avoid. What that coarseness costs, in
    both directions, is :func:`fusion_key`'s docstring.

    Coarser is not safe in both directions, and this docstring used to say that
    it was. Merging tables the runtime keeps apart moves
    ``min(sum total, sum budget)`` up, which over-states HBM -- that half only
    ever refuses a layout that would have fit. But the host tier is ``total -
    hbm`` and ``total`` does not move, so the same merge under-states the host
    tier by exactly as much, and *that* half accepts layouts the machine will
    not hold. Every component :func:`fusion_key` carries narrows the gap; see
    its docstring for the full statement.

    Every map here is keyed by :class:`TableKey`, so a name declared in two
    collections arrives as two entries and is charged for twice -- which is
    what the runtime does, building one table per collection. The collection
    is also in the grouping key, so the two never pool budgets.

    ``optimizer_types`` maps table -> ``OptimType``; a table missing from it
    is counted without optimizer state, which is a floor rather than a guess (see
    :func:`~dynamicemb.dynamicemb_config.get_group_value_bytes_by_tier`).

    ``poolings`` carries what TorchRec groups on and neither the plan record
    nor the options know. A table missing from it is grouped as ``None``, which
    pools it with the others that are missing, so it is passed whole or not at
    all.
    """
    totals = [make_storage() for _ in range(world_size)]

    # rank -> the tables that rank holds, bucketed by what the runtime fuses on.
    # A dict keyed by the options themselves: DynamicEmbTableOptions hashes and
    # compares on its grouping key, so two tables land together exactly when
    # they would be fused.
    per_rank_groups: List[Dict[object, List[Tuple[object, Optional[OptimType]]]]] = [
        defaultdict(list) for _ in range(world_size)
    ]

    for key, parameter_sharding in parameter_shardings.items():
        table_options = options.get(key)
        if table_options is None:
            raise ValueError(
                f"Table {str(key)!r} is planned but has no DynamicEmbTableOptions, "
                "so its footprint cannot be sized."
            )

        spec = parameter_sharding.sharding_spec
        if not isinstance(spec, EnumerableShardingSpec) or not spec.shards:
            raise ValueError(
                f"Table {str(key)!r} has no EnumerableShardingSpec, so which ranks "
                "hold it is unknown."
            )

        # Options carry max_capacity and local_hbm_for_values already divided by
        # the table's fan-out, so every shard of a table weighs the same and
        # only the set of ranks differs.
        member = (table_options, optimizer_types.get(key))
        # The collection is in the key, so a name declared twice groups twice
        # -- once per runtime that will exist -- without this loop knowing it
        # is a duplicate.
        group = fusion_key(
            parameter_sharding.sharding_type,
            table_options,
            optimizer_types.get(key),
            (poolings or {}).get(key),
            key.collection,
        )
        for shard in spec.shards:
            rank = shard_rank(shard)
            if not 0 <= rank < world_size:
                raise ValueError(
                    f"Table {str(key)!r} places a shard on rank {rank}, which is "
                    f"outside the world of {world_size}."
                )
            per_rank_groups[rank][group].append(member)

    for rank, groups in enumerate(per_rank_groups):
        for members in groups.values():
            hbm, ddr = get_group_value_bytes_by_tier(members)
            totals[rank] += make_storage(hbm=hbm, ddr=ddr)

    return totals


def _negative_tiers(storage: Storage) -> List[str]:
    """Which tiers of a Storage have gone below zero, in order.

    ``ssd`` through :func:`storage_ssd`, since the field only exists from
    TorchRec 1.5.0 and reads as zero -- never negative -- before that.
    """
    over = [tier for tier in ("hbm", "ddr") if getattr(storage, tier) < 0]
    if storage_ssd(storage) < 0:
        over.append("ssd")
    return over


# What to do about a tier a rank has overdrawn. HBM and host pull against each
# other -- `global_hbm_for_values` is the line between them, so moving it to
# fix one tier spends the other -- which is why the advice cannot be one list.
_TIER_REMEDIES = {
    "hbm": (
        "lower global_hbm_for_values, which leaves fewer rows in HBM and "
        "spills the rest to the host tier"
    ),
    "ddr": (
        "raise global_hbm_for_values, which keeps more rows in HBM instead, "
        "or state a ddr_cap that matches the host memory a rank really has "
        "(TorchRec defaults it to 128 GB per rank)"
    ),
    "ssd": "lower the tables' capacity, or state an ssd_cap that matches the machine",
}


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

    Every tier is checked, not only HBM: ``Storage.__sub__`` subtracts all
    three, so a guard that read one would pass the others on silently. Nothing
    charges ``ssd`` today, which is what makes the check cheap to add now and
    hard to notice missing later.
    """
    if len(per_rank) != len(topology.devices):
        raise ValueError(
            f"per_rank has {len(per_rank)} entries but the topology has "
            f"{len(topology.devices)} devices."
        )

    reduced = copy.deepcopy(topology)
    overdrawn: List[Tuple[int, Storage, Storage, List[str]]] = []
    for device in reduced.devices:
        spent = per_rank[device.rank]
        before = device.storage
        device.storage = before - spent
        tiers = _negative_tiers(device.storage)
        if tiers:
            overdrawn.append((device.rank, before, spent, tiers))

    if overdrawn:
        worst = min(overdrawn, key=lambda o: min(o[1].hbm - o[2].hbm, 0))
        rank, before, spent, tiers = worst
        raise PlannerError(
            error_type=PlannerErrorType.INSUFFICIENT_STORAGE,
            message=(
                f"The DynamicEmb tables do not fit on "
                f"{len(overdrawn)} of {len(reduced.devices)} ranks, leaving "
                "nothing for TorchRec to plan into. Rank "
                f"{rank} has {storage_repr_in_gb(before)} and they take "
                f"{storage_repr_in_gb(spent)}, which overdraws "
                f"{' and '.join(tiers)}. To fix it: "
                + "; ".join(_TIER_REMEDIES[tier] for tier in tiers)
                + ". Either way you can also shard the largest tables row-wise "
                "so they spread over every rank instead of one node, or state "
                "a Topology that matches the machine."
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
    # `copy.copy` gives the new module its own `__dict__` -- `nn.Module.__getstate__`
    # returns `self.__dict__.copy()` -- but the values in it are shared, so
    # `_modules` is the same dict object as the caller's until it is replaced.
    # Without this line the assignment below would reach into their module.
    reduced._modules = dict(collection._modules)
    setattr(reduced, attribute, kept)
    return reduced


def module_without_tables(
    module: nn.Module,
    tables: Set["TableKey"],
    sharders: List[ModuleSharder[nn.Module]],
) -> nn.Module:
    """``module`` as TorchRec should plan it: without the DynamicEmb tables.

    ``tables`` names them by :class:`TableKey`, since a name DynamicEmb planned
    in one collection may be TorchRec's in another, and only the one that was
    planned here may be taken away.

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

    def prune(path: str, node: nn.Module) -> nn.Module:
        sharder = sharder_map.get(sharder_name(type(node)))
        if sharder is not None:
            # By `TableKey`, so a name DynamicEmb planned in one collection is
            # taken out of that one and left in the other -- which is the whole
            # point of keying by the pair. The path is carried down the
            # recursion for this alone.
            owned = {
                table_name
                for table_name in sharder.shardable_parameters(node)
                if TableKey(path, table_name) in tables
            }
            return _collection_without(node, owned) if owned else node

        replacements = {
            child_name: pruned
            for child_name, child in node.named_children()
            if (pruned := prune(f"{path}.{child_name}" if path else child_name, child))
            is not child
        }
        if not replacements:
            return node

        copied = copy.copy(node)
        # Same shared-`_modules` reason as `_collection_without`.
        copied._modules = dict(node._modules)
        for child_name, child in replacements.items():
            setattr(copied, child_name, child)
        return copied

    return prune("", module)


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


def table_ranks(
    sharding_type: str,
    node_index: Optional[int],
    world_size: int,
    local_size: int,
) -> List[int]:
    """The ranks a DynamicEmb table's shards sit on, in shard order.

    Row-wise puts a shard on every rank, so the answer is every rank and
    ``node_index`` means nothing -- there is no node to name when the table is on
    all of them. Table-row-wise puts the table on one node, so the answer is that
    node's ranks and ``node_index`` says which node.

    The length is :func:`table_fanout`.

    Raises:
        ValueError: whatever :func:`table_fanout` raises, or a ``node_index``
            that is missing, out of range, or set on a row-wise table.
    """
    fanout = table_fanout(sharding_type, world_size, local_size)

    if sharding_type == ShardingType.ROW_WISE.value:
        if node_index is not None:
            raise ValueError(
                f"node_index={node_index} was set on a {sharding_type} table. It "
                "says which node holds a table, and a row-wise table is on every "
                "rank of every node."
            )
        return list(range(world_size))

    num_nodes = world_size // fanout
    if node_index is None:
        raise ValueError(
            f"A {sharding_type} table needs a node_index saying which of the "
            f"{num_nodes} nodes holds it. Set DynamicEmbTableOptions.node_index, "
            "or leave it unset and let the planner's node placer choose."
        )
    if not 0 <= node_index < num_nodes:
        raise ValueError(
            f"node_index={node_index} is outside the {num_nodes} nodes of a world "
            f"of {world_size} with {local_size} ranks each."
        )
    base = node_index * fanout
    return list(range(base, base + fanout))
