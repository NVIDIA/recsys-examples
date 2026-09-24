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
from typing import Dict, List, Optional

from torchrec.distributed.planner.types import Storage, Topology

__all__ = ["TableToPlace", "HostPlacer", "BalancedHostPlacer"]


@dataclass(frozen=True)
class TableToPlace:
    """One table-row-wise table awaiting a node.

    Attributes
    ----------
    name : str
        The table's name, and the tie-break when two tables cost the same.
    cost : Storage
        What one rank of this table takes. Every rank of the chosen node pays
        it, since table-row-wise splits the rows evenly across the node.
    pinned : Optional[int]
        The node the caller named through ``DynamicEmbTableOptions.host_index``.
        A placer must honour it.
    """

    name: str
    cost: Storage
    pinned: Optional[int] = None


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

        # Room on each rank, then the tightest rank of each node: what one more
        # per-rank cost has to fit into.
        free = [device.storage - committed[device.rank] for device in topology.devices]

        def tightest(node: int) -> Storage:
            ranks = range(node * local_size, (node + 1) * local_size)
            return Storage(
                hbm=min(free[r].hbm for r in ranks),
                ddr=min(free[r].ddr for r in ranks),
                ssd=min(free[r].ssd for r in ranks),
            )

        def charge(node: int, cost: Storage) -> None:
            for r in range(node * local_size, (node + 1) * local_size):
                free[r] = free[r] - cost

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
            chosen[table.name] = table.pinned
            charge(table.pinned, table.cost)

        # Then the rest, biggest first.
        unpinned = sorted(
            (t for t in tables if t.pinned is None),
            key=lambda t: (-t.cost.hbm, -t.cost.ddr, t.name),
        )
        for table in unpinned:
            # `max` keeps the first of equal keys, and range() is ascending, so
            # a tie goes to the lower node index.
            node = max(
                range(num_nodes),
                key=lambda n: (tightest(n).hbm, tightest(n).ddr, -n),
            )
            room = tightest(node)
            if not table.cost.fits_in(room):
                raise ValueError(
                    f"No node can hold the table-row-wise table {table.name!r}. "
                    f"It takes {table.cost} on each of a node's {local_size} "
                    f"ranks, and the emptiest node has {room} left on its "
                    "tightest rank. Give it row-wise sharding, lower its "
                    "capacity, or pin the tables so they pack differently."
                )
            chosen[table.name] = node
            charge(node, table.cost)

        return chosen
