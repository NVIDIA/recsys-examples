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
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torchrec.distributed.planner.types import Topology
from torchrec.modules.embedding_configs import PoolingType

from dynamicemb.planner.plan import make_storage

from dynamicemb.dynamicemb_config import DynamicEmbTableOptions
from dynamicemb.planner.placement import BalancedHostPlacer, TableToPlace

GB = 1024**3
DIM = 16
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
    return [make_storage(hbm=0, ddr=0, ssd=0) for _ in range(WORLD)]


def _table(name, size_gb, pinned=None, budget_gb=None):
    """A table weighing `size_gb` per rank.

    Its HBM budget defaults to its size, so it sits entirely in HBM and
    `_table(name, N)` costs N GB of it -- which is what the tests that predate
    pooling assume. Pass `budget_gb` to make a table that spills to the host,
    or one with budget to spare for a groupmate.
    """
    rows = size_gb * GB // (DIM * 4) or 1
    budget = size_gb if budget_gb is None else budget_gb
    return TableToPlace(
        name=name,
        options=DynamicEmbTableOptions(
            max_capacity=rows,
            dim=DIM,
            embedding_dtype=torch.float32,
            local_hbm_for_values=budget * GB,
        ),
        pinned=pinned,
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
    everywhere = [make_storage(hbm=40 * GB, ddr=0, ssd=0) for _ in range(WORLD)]
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
        make_storage(hbm=60 * GB if r < LOCAL else 0, ddr=0, ssd=0)
        for r in range(WORLD)
    ]
    assert placer.place([_table("b", 30)], _topology(hbm=80 * GB), spent) == {"b": 1}


def test_one_crowded_rank_rules_out_its_whole_node():
    uneven = [
        make_storage(hbm=(70 * GB if r == 3 else 0), ddr=0, ssd=0) for r in range(WORLD)
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


def test_a_pin_to_a_node_that_cannot_hold_it_is_refused():
    """A pin says which node, not that the node will hold it.

    Charging it unchecked leaves that node's ranks negative, and TorchRec's own
    reservation looks for that on `devices[0]` alone -- so a pin to any node but
    the first would be returned as a plan.
    """
    with pytest.raises(ValueError, match="cannot hold it"):
        BalancedHostPlacer().place(
            [_table("a", 200, pinned=1)], _topology(), _nothing_spent()
        )


def test_a_pin_is_refused_against_room_already_taken():
    """The check is against what is left, not against an empty machine: an
    earlier pin to the same node counts."""
    placer = BalancedHostPlacer()
    tables = [_table("first", 60, pinned=0), _table("second", 40, pinned=0)]
    with pytest.raises(ValueError, match="'second'"):
        placer.place(tables, _topology(hbm=80 * GB), _nothing_spent())


def test_a_pin_that_fits_is_still_honoured():
    chosen = BalancedHostPlacer().place(
        [_table("a", 70, pinned=1)], _topology(hbm=80 * GB), _nothing_spent()
    )
    assert chosen == {"a": 1}


def test_a_groupmate_with_spare_budget_makes_room():
    """What a table costs a node depends on what is already there.

    Two tables that fuse pool their HBM budgets and spend them on their pooled
    size, so one with budget to spare keeps the other out of host memory. Priced
    alone the second needs 14 GB of host and is refused; after pooling it needs
    none and fits. That is the review's second case -- host over-stated, a
    layout refused that the runtime would have held.
    """
    # Plenty of HBM, almost no host memory: the host figure is what decides.
    topo = _topology(hbm=80 * GB, ddr=4 * GB)
    generous = _table("generous", 4, budget_gb=20)  # 4 GB of a 20 GB budget
    needy = _table("needy", 16, budget_gb=2)  # 16 GB with 2 GB allowed

    assert needy.alone().ddr == 14 * GB, "priced alone it spills 14 GB to host"
    assert generous.alone().ddr == 0

    chosen = BalancedHostPlacer().place([generous, needy], topo, _nothing_spent())
    assert chosen["generous"] == chosen["needy"], (
        "they fuse, so together they fit in the pooled 22 GB budget and need no "
        "host memory -- which only shows if the placer prices them together"
    )


def test_tables_that_do_not_fuse_are_not_pooled():
    """Different grouping keys mean different TBEs, so no budget is shared."""
    topo = _topology(hbm=80 * GB, ddr=4 * GB)
    generous = _table("generous", 4, budget_gb=20)
    needy = TableToPlace(
        name="needy",
        options=DynamicEmbTableOptions(
            max_capacity=16 * GB // (DIM * 4),
            dim=DIM,
            embedding_dtype=torch.float32,
            local_hbm_for_values=2 * GB,
            # part of the grouping key, so these two never share a TBE
            dist_type="hash_roundrobin",
        ),
    )
    assert generous.options != needy.options

    with pytest.raises(ValueError, match="No node can hold"):
        BalancedHostPlacer().place([generous, needy], topo, _nothing_spent())


def test_a_different_optimizer_is_a_different_group():
    """TorchRec keys on the optimizer through the fused params, so two tables
    that differ there never share a TBE and never pool their budgets."""
    topo = _topology(hbm=80 * GB, ddr=4 * GB)
    generous = _table("generous", 4, budget_gb=20)
    needy = _table("needy", 16, budget_gb=2)
    needy = TableToPlace(
        name=needy.name,
        options=needy.options,
        optimizer_type=OptimType.ADAM,  # generous has None
        pooling=needy.pooling,
    )
    assert generous.fuses_with != needy.fuses_with

    with pytest.raises(ValueError, match="No node can hold"):
        BalancedHostPlacer().place([generous, needy], topo, _nothing_spent())


def test_a_different_pooling_is_a_different_group():
    """Pooling is TorchRec's to group on and DynamicEmbTableOptions does not
    carry it, so it has to be passed in for the two not to be pooled."""
    topo = _topology(hbm=80 * GB, ddr=4 * GB)
    generous = _table("generous", 4, budget_gb=20)
    needy = _table("needy", 16, budget_gb=2)
    needy = TableToPlace(
        name=needy.name,
        options=needy.options,
        pooling=PoolingType.SUM,  # generous has None
    )
    assert generous.fuses_with != needy.fuses_with

    with pytest.raises(ValueError, match="No node can hold"):
        BalancedHostPlacer().place([generous, needy], topo, _nothing_spent())


def test_a_different_dtype_is_a_different_group():
    """data_type is TorchRec's grouping key and not part of
    DynamicEmbTableOptions', so it is read off the options separately."""
    generous = _table("generous", 4, budget_gb=20)
    needy = _table("needy", 16, budget_gb=2)
    halved = DynamicEmbTableOptions(
        max_capacity=needy.options.max_capacity,
        dim=DIM,
        embedding_dtype=torch.float16,
        local_hbm_for_values=2 * GB,
    )
    assert halved == needy.options, "the DynamicEmb key alone does not separate them"
    needy = TableToPlace(name="needy", options=halved)
    assert generous.fuses_with != needy.fuses_with, "but the fusion key does"
