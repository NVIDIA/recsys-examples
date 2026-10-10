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

"""Which node a table-row-wise DynamicEmb table lands on.

A row-wise table needs no such decision -- it is on every rank. A table-row-wise
one lives on a single node, and something has to say which. This is that
something, kept apart from the planner so the rule can be replaced without
touching how a plan is assembled.

The choice is a bin-packing problem and not a search: a DynamicEmb table's size
is declared rather than estimated, so what a node will owe is known before
anything is placed there. What is *not* known is what TorchRec will later want
to put on the same node, which is why the default leaves room rather than
filling nodes in order.
"""

import abc
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Tuple

from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torchrec.distributed.planner.types import Storage, Topology
from torchrec.distributed.types import ShardingType

from ..dynamicemb_config import (
    DynamicEmbTableOptions,
    get_group_value_bytes_by_tier,
)

from .plan import fusion_key, make_storage, storage_ssd

__all__ = ["TableToPlace", "HostPlacer", "BalancedHostPlacer"]


@dataclass(frozen=True)
class TableToPlace:
    """One table-row-wise table awaiting a node.

    It carries what the table *is* rather than what it costs, because what it
    costs depends on what it is placed with: the runtime fuses compatible
    tables on a rank and spends their pooled HBM budget on their pooled size,
    so a table's price is not fixed until its neighbours are known.

    Attributes
    ----------
    name : str
        The table's name, and the tie-break when two tables weigh the same.
    options : DynamicEmbTableOptions
        The table's own options. These are both its size and -- through
        ``get_grouped_key`` -- which other tables it fuses with.
    optimizer_type : Optional[OptimType]
        What it trains with, since optimizer state is part of a row. ``None``
        counts the rows without it, a floor rather than a guess.
    pooling : object
        The table's ``PoolingType``. Not DynamicEmb's to know -- it is on the
        TorchRec config -- but TorchRec groups on it, so two tables that differ
        here do not pool their budgets.
    pinned : Optional[int]
        The node the caller named through ``DynamicEmbTableOptions.host_index``.
        A placer must honour it, and must still check that it fits.
    """

    name: str
    options: "DynamicEmbTableOptions"
    optimizer_type: Optional[OptimType] = None
    pooling: object = None
    pinned: Optional[int] = None

    @property
    def fuses_with(self) -> Tuple:
        """What this table must match for its budget to be pooled with another's.

        :func:`~dynamicemb.planner.plan.fusion_key`, so the placer and
        `per_rank_storage` cannot disagree about which tables share a TBE --
        and a table-row-wise table is always that, since a pin or a placement
        decides only the node.
        """
        return fusion_key(
            ShardingType.TABLE_ROW_WISE.value,
            self.options,
            self.optimizer_type,
            self.pooling,
        )

    @property
    def member(self) -> Tuple["DynamicEmbTableOptions", Optional[OptimType]]:
        """The pair :func:`get_group_value_bytes_by_tier` costs."""
        return (self.options, self.optimizer_type)

    def alone(self) -> Storage:
        """What this table costs a rank with nothing to pool with.

        Only for ordering: the largest-first sort needs a number before any
        placement exists. What a table actually costs a node is the difference
        it makes to that node's grouped total.
        """
        hbm, ddr = get_group_value_bytes_by_tier([self.member])
        return make_storage(hbm=hbm, ddr=ddr)


class HostPlacer(abc.ABC):
    """Chooses a node for each table-row-wise table."""

    @abc.abstractmethod
    def place(
        self,
        tables: List[TableToPlace],
        topology: Topology,
        committed: List[Storage],
    ) -> Dict[str, int]:
        """Return ``{table name: host index}``, covering every table given.

        Parameters
        ----------
        tables : List[TableToPlace]
            The table-row-wise tables, pinned and unpinned alike.
        topology : Topology
            The machine as the planner was told it. ``local_world_size`` gives
            the node size, and each device's ``storage`` the room on it.
        committed : List[Storage]
            What is already spent on each rank -- the row-wise DynamicEmb tables,
            which are on every rank and so shift no decision, and any table this
            placer has no say over.

        Implementations must be deterministic: this runs on every rank, and two
        ranks that disagree would shard the same table onto different nodes.
        """
        ...


class BalancedHostPlacer(HostPlacer):
    """Largest table first, onto the node with the most room left.

    The usual first-fit-decreasing, with "fits" measured against the *tightest*
    rank of a node rather than the node's total, because a table-row-wise table
    charges every rank of its node the same and so is limited by the worst of
    them.

    Choosing the emptiest node rather than the first that fits is deliberate.
    TorchRec plans its own tables afterwards, into what this leaves behind, and
    it has a node decision of its own to make for its TABLE_ROW_WISE and
    GRID_SHARD tables. Filling nodes in order would hand it a machine with
    nothing left on the low-numbered ones.

    Ties break on the table name and then the lower node index, so every rank
    reaches the same answer.
    """

    def place(
        self,
        tables: List[TableToPlace],
        topology: Topology,
        committed: List[Storage],
    ) -> Dict[str, int]:
        local_size = topology.local_world_size
        world_size = len(topology.devices)
        if local_size <= 0 or world_size % local_size:
            raise ValueError(
                f"A world of {world_size} does not divide into nodes of "
                f"{local_size}, so there is no node to place a table on."
            )
        num_nodes = world_size // local_size

        # What each node has spare before any of these tables: its tightest
        # rank, since a table-row-wise table charges every rank of its node the
        # same and so is limited by the worst of them.
        def capacity(node: int) -> Storage:
            ranks = range(node * local_size, (node + 1) * local_size)
            free = [topology.devices[r].storage - committed[r] for r in ranks]
            return make_storage(
                hbm=min(f.hbm for f in free),
                ddr=min(f.ddr for f in free),
                ssd=min(storage_ssd(f) for f in free),
            )

        room = [capacity(n) for n in range(num_nodes)]

        # What each node has been given, bucketed by what the runtime fuses on.
        # Costing a node means costing each bucket, because a bucket's members
        # pool their HBM budgets -- so what a table adds to a node depends on
        # what is already there, and cannot be decided once per table.
        held: List[Dict[Hashable, List[Tuple[object, Optional[OptimType]]]]] = [
            defaultdict(list) for _ in range(num_nodes)
        ]

        def cost_of(node: int) -> Storage:
            total = make_storage(hbm=0, ddr=0)
            for members in held[node].values():
                hbm, ddr = get_group_value_bytes_by_tier(members)
                total = total + make_storage(hbm=hbm, ddr=ddr)
            return total

        def added_by(node: int, table: TableToPlace) -> Storage:
            """What this table would add to this node, after pooling.

            The difference the table makes, not its price alone: placed beside
            a groupmate whose budget it can share, it may add less than it
            would cost on an empty node -- or more, if its own spare budget
            then goes to them.
            """
            before = cost_of(node)
            held[node][table.fuses_with].append(table.member)
            after = cost_of(node)
            held[node][table.fuses_with].pop()
            return after - before

        def give(node: int, table: TableToPlace) -> None:
            held[node][table.fuses_with].append(table.member)

        def spare(node: int) -> Storage:
            left = room[node] - cost_of(node)
            return left

        chosen: Dict[str, int] = {}

        # Pinned tables first: they are not choices, and what they take has to
        # be gone from the room before anything is fitted around them.
        for table in tables:
            if table.pinned is None:
                continue
            if not 0 <= table.pinned < num_nodes:
                raise ValueError(
                    f"Table {table.name!r} is pinned to node {table.pinned}, "
                    f"which is outside the {num_nodes} nodes of this topology."
                )
            # Checked the same as a table this placer chose for. A pin says
            # which node, not that the node will hold it, and charging it
            # unchecked leaves the budget negative on that node's ranks --
            # which TorchRec's own reservation looks for on `devices[0]` alone
            # (`storage_reservations.py:509`), so a pin to any other node would
            # go unnoticed and the plan would be returned.
            needs, left = added_by(table.pinned, table), spare(table.pinned)
            if not needs.fits_in(left):
                raise ValueError(
                    f"Table {table.name!r} is pinned to node {table.pinned}, "
                    f"which cannot hold it. It adds {needs} to each of that "
                    f"node's {local_size} ranks, which have {left} left. Pin "
                    "it elsewhere, leave host_index unset and let the placer "
                    "choose, or give it row-wise sharding."
                )
            chosen[table.name] = table.pinned
            give(table.pinned, table)

        # Then the rest, biggest first. "Biggest" is what a table weighs on its
        # own, which is the only size there is before anything is placed.
        unpinned = sorted(
            (t for t in tables if t.pinned is None),
            key=lambda t: (-t.alone().hbm, -t.alone().ddr, t.name),
        )
        for table in unpinned:
            # Emptiest node that can hold it, measured after what each would
            # charge for this table -- so a node holding a groupmate with budget
            # to spare can win on that account, which is what the runtime will
            # do anyway.
            def rank_node(n: int) -> Tuple[bool, int, int, int]:
                needs, left = added_by(n, table), spare(n)
                after = left - needs
                # Whether it fits comes first. Ranking on room alone compares
                # the tiers one after another, so a node with headroom in HBM
                # would win over one that fits in both -- with its host tier
                # already overdrawn.
                return (needs.fits_in(left), after.hbm, after.ddr, -n)

            node = max(range(num_nodes), key=rank_node)
            needs, left = added_by(node, table), spare(node)
            if not needs.fits_in(left):
                raise ValueError(
                    f"No node can hold the table-row-wise table {table.name!r}. "
                    f"It adds {needs} to each of a node's {local_size} ranks, "
                    f"and the emptiest node has {left} left on its tightest "
                    "rank. Give it row-wise sharding, lower its capacity, or "
                    "pin the tables so they pack differently."
                )
            chosen[table.name] = node
            give(node, table)

        return chosen
