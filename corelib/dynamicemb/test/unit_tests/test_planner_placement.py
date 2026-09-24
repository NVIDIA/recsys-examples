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

"""Which node a table-row-wise table lands on.

Single process, no GPU and no process group: the choice is arithmetic over
declared sizes, which is why it is a component of its own.
"""

import pytest
from torchrec.distributed.planner.types import Storage, Topology

from dynamicemb.planner.placement import BalancedHostPlacer, TableToPlace

GB = 1024**3
LOCAL, NODES = 4, 2
WORLD = LOCAL * NODES


def _topology(hbm: int = 80 * GB, ddr: int = 128 * GB) -> Topology:
    return Topology(
        world_size=WORLD,
        local_world_size=LOCAL,
        compute_device="cuda",
        hbm_cap=hbm,
        ddr_cap=ddr,
    )


def _nothing_spent():
    return [Storage(hbm=0, ddr=0, ssd=0) for _ in range(WORLD)]


def _table(name, hbm_gb, pinned=None):
    return TableToPlace(
        name=name, cost=Storage(hbm=hbm_gb * GB, ddr=0, ssd=0), pinned=pinned
    )


def test_largest_first_onto_the_emptiest_node():
    chosen = BalancedHostPlacer().place(
        [_table("a", 30), _table("b", 20), _table("c", 10)],
        _topology(),
        _nothing_spent(),
    )
    # a is biggest and goes first; both nodes are empty so it takes the lower.
    # b then sees node 1 emptier, and c follows b only because node 1 still has
    # more room than node 0 does after a.
    assert chosen == {"a": 0, "b": 1, "c": 1}


def test_a_pin_is_honoured_and_others_fit_around_it():
    chosen = BalancedHostPlacer().place(
        [_table("a", 30, pinned=1), _table("b", 20)],
        _topology(),
        _nothing_spent(),
    )
    assert chosen["a"] == 1
    assert chosen["b"] == 0


def test_row_wise_tables_do_not_shift_the_choice_but_do_take_room():
    """They are on every rank, so they cost each node the same. What they change
    is whether a table-row-wise table still fits."""
    everywhere = [Storage(hbm=40 * GB, ddr=0, ssd=0) for _ in range(WORLD)]
    chosen = BalancedHostPlacer().place(
        [_table("a", 10), _table("b", 10)], _topology(), everywhere
    )
    assert chosen == {"a": 0, "b": 1}

    with pytest.raises(ValueError, match="No node can hold"):
        BalancedHostPlacer().place([_table("big", 50)], _topology(), everywhere)


def test_a_node_is_charged_on_every_one_of_its_ranks():
    """Table-row-wise splits rows across the node, so each rank pays the cost --
    it is the tightest rank, not the node total, that has to fit."""
    placer = BalancedHostPlacer()
    chosen = placer.place([_table("a", 60)], _topology(hbm=80 * GB), _nothing_spent())
    assert chosen == {"a": 0}
    # node 0 now has 20GB per rank, so a 30GB table can only go to node 1
    spent = [
        Storage(hbm=60 * GB if r < LOCAL else 0, ddr=0, ssd=0) for r in range(WORLD)
    ]
    assert placer.place([_table("b", 30)], _topology(hbm=80 * GB), spent) == {"b": 1}


def test_one_crowded_rank_rules_out_its_whole_node():
    uneven = [
        Storage(hbm=(70 * GB if r == 3 else 0), ddr=0, ssd=0) for r in range(WORLD)
    ]
    chosen = BalancedHostPlacer().place([_table("a", 20)], _topology(), uneven)
    assert chosen == {"a": 1}


def test_a_table_that_fits_nowhere_says_so():
    with pytest.raises(ValueError, match="No node can hold"):
        BalancedHostPlacer().place([_table("a", 200)], _topology(), _nothing_spent())


def test_a_pin_outside_the_topology_is_an_error():
    with pytest.raises(ValueError, match="outside the 2 nodes"):
        BalancedHostPlacer().place(
            [_table("a", 1, pinned=5)], _topology(), _nothing_spent()
        )


def test_the_answer_does_not_depend_on_the_order_it_was_asked():
    """It runs on every rank; two ranks that disagreed would shard one table
    onto two nodes."""
    tables = [_table("a", 30), _table("b", 30), _table("c", 10), _table("d", 10)]
    first = BalancedHostPlacer().place(tables, _topology(), _nothing_spent())
    second = BalancedHostPlacer().place(
        list(reversed(tables)), _topology(), _nothing_spent()
    )
    assert first == second


def test_equal_tables_spread_rather_than_stack():
    chosen = BalancedHostPlacer().place(
        [_table("a", 10), _table("b", 10), _table("c", 10), _table("d", 10)],
        _topology(),
        _nothing_spent(),
    )
    assert sorted(chosen.values()) == [0, 0, 1, 1]
