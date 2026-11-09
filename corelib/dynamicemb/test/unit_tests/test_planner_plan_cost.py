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

"""What the DynamicEmb tables cost each rank, and what TorchRec is left with.

Single process, no GPU and no process group: these are the arithmetic half of
planning, and the point of having them as functions over a plan is that they can
be checked without one.
"""

import copy
from dataclasses import fields
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
import torchrec
from dynamicemb.dynamicemb_config import (
    DynamicEmbTableOptions,
    get_group_value_bytes_by_tier,
    get_local_value_bytes_by_tier,
)
from dynamicemb.planner import plan
from dynamicemb.planner.plan import (
    _STORAGE_HAS_SSD,
    TableKey,
    make_storage,
    module_without_tables,
    optimizer_types,
    per_rank_storage,
    shard_rank,
    table_fanout,
    table_ranks,
    topology_minus,
)
from dynamicemb.planner.planners import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
    DynamicEmbParameterSharding,
    get_constraint_key,
)
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.types import PlannerError, Topology
from torchrec.distributed.types import (
    EnumerableShardingSpec,
    ParameterSharding,
    ShardingPlan,
    ShardingType,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig

WORLD_SIZE = 4
LOCAL_SIZE = 2
DIM = 16
ROWS = 1024
GB = 1024**3
# What one rank of the machine under test has, as `_topology` builds it.
HBM_PER_RANK = 80 * GB
DDR_PER_RANK = 128 * GB


def _options(rows: int = ROWS, hbm_budget: int = 0) -> DynamicEmbTableOptions:
    return DynamicEmbTableOptions(
        max_capacity=rows,
        dim=DIM,
        embedding_dtype=torch.float32,
        local_hbm_for_values=hbm_budget,
    )


class _Planned(NamedTuple):
    """A table as the planner keeps it: decisions on the plan record, and the
    options it was planned with beside them rather than on it."""

    sharding: ParameterSharding
    options: DynamicEmbTableOptions


def _planned_table(ranks, rows: int = ROWS, hbm_budget: int = 0) -> _Planned:
    """One shard of `rows` rows on each of `ranks`, and the options beside it."""
    return _Planned(
        ParameterSharding(
            sharding_type=ShardingType.ROW_WISE.value,
            compute_kernel="customized_kernel",
            ranks=list(ranks),
            sharding_spec=EnumerableShardingSpec(
                [
                    ShardMetadata(
                        shard_sizes=[rows, DIM],
                        shard_offsets=[rows * i, 0],
                        placement=f"rank:{rank}/cuda:{rank % LOCAL_SIZE}",
                    )
                    for i, rank in enumerate(ranks)
                ]
            ),
        ),
        _options(rows, hbm_budget),
    )


def _key(name, collection="sparse.ebc"):
    """A `TableKey` from a bare name, for the tests that use one collection.

    `per_rank_storage` is keyed by the pair, since a name need not be unique.
    Where a test says nothing about collections there is only one, and this
    puts them all in it -- `sparse.ebc`, the path `_Model` holds its collection
    at, so the string means the same thing everywhere in this file. A test that
    is about collections builds its `TableKey`s itself.
    """
    return TableKey(collection, name)


def _spent(
    planned,
    optimizer_types,
    world_size: int = WORLD_SIZE,
    poolings=None,
):
    """`per_rank_storage` over a {name: _Planned} mapping.

    The plan and the options go in as two arguments because that is how the
    planner holds them -- the plan record carries decisions, and the options
    live with the constraints they came from.
    """
    return per_rank_storage(
        {_key(name): p.sharding for name, p in planned.items()},
        {_key(name): p.options for name, p in planned.items()},
        optimizer_types={_key(name): o for name, o in optimizer_types.items()},
        world_size=world_size,
        poolings=None
        if poolings is None
        else {_key(name): p for name, p in poolings.items()},
    )


def _topology(hbm: int = HBM_PER_RANK, ddr: int = DDR_PER_RANK) -> Topology:
    return Topology(
        world_size=WORLD_SIZE,
        local_world_size=LOCAL_SIZE,
        compute_device="cuda",
        hbm_cap=hbm,
        ddr_cap=ddr,
    )


def test_shard_rank_reads_both_placement_forms():
    """Placement is a _remote_device or the string it was built from.

    ``ShardMetadata.__post_init__`` parses a string argument, so the normal
    construction gives the first form. The string form is set afterwards to
    cover the case TorchRec itself still guards for
    (``distributed/embeddingbag.py:184-185``).
    """
    shard = ShardMetadata(
        shard_offsets=[0, 0], shard_sizes=[ROWS, DIM], placement="rank:3/cuda:1"
    )
    # __post_init__ parsed the string into a _remote_device.
    assert not isinstance(shard.placement, str)
    assert shard_rank(shard) == 3

    # A later assignment does not go through __post_init__, so it stays a str --
    # the form TorchRec itself still guards for.
    shard.placement = "rank:3/cuda:1"
    assert isinstance(shard.placement, str)
    assert shard_rank(shard) == 3


class TestPerRankStorage:
    """What each rank spends on the DynamicEmb tables planned onto it."""

    def test_row_wise_table_charges_every_rank_equally(self):
        per_rank = _spent(
            {"t": _planned_table(range(WORLD_SIZE))}, optimizer_types={"t": None}
        )
        assert len(per_rank) == WORLD_SIZE
        values = ROWS * DIM * 4  # float32, no optimizer state
        assert all(s == make_storage(hbm=0, ddr=values, ssd=0) for s in per_rank)

    def test_a_table_on_one_node_charges_only_that_node(self):
        """The reason this is per rank and not one repeated figure.

        Under TABLE_ROW_WISE a table lives on one node's ranks. Charging every rank
        for it would reserve four nodes' worth of room for one node's table.
        """
        per_rank = _spent(
            {"t": _planned_table([2, 3])},  # the second node only
            optimizer_types={"t": None},
        )
        values = ROWS * DIM * 4
        assert per_rank[0] == make_storage()
        assert per_rank[1] == make_storage()
        assert per_rank[2] == make_storage(hbm=0, ddr=values, ssd=0)
        assert per_rank[3] == make_storage(hbm=0, ddr=values, ssd=0)

    def test_optimizer_state_lands_entirely_in_ddr(self):
        """local_hbm_for_values is a fixed budget, so the optimizer's extra bytes
        cannot come out of HBM -- they all spill to the host tier."""
        hbm_budget = ROWS * DIM * 4  # exactly the values, nothing spare
        without = _spent(
            {"t": _planned_table(range(WORLD_SIZE), hbm_budget=hbm_budget)},
            optimizer_types={"t": None},
        )
        with_adam = _spent(
            {"t": _planned_table(range(WORLD_SIZE), hbm_budget=hbm_budget)},
            optimizer_types={"t": OptimType.ADAM},
        )

        assert without[0].ddr == 0
        assert with_adam[0].hbm == without[0].hbm == hbm_budget
        assert with_adam[0].ddr > 0

    def test_a_shard_outside_the_world_is_an_error(self):
        with pytest.raises(ValueError, match="outside the world"):
            _spent({"t": _planned_table([WORLD_SIZE])}, optimizer_types={"t": None})


class TestTopologyMinus:
    """Taking what the DynamicEmb tables cost out of the machine.

    Per rank, into a copy, and refusing rather than clamping -- TorchRec's own
    reservation only ever looks at ``devices[0]``.
    """

    def test_topology_minus_subtracts_per_rank_and_copies(self):
        topology = _topology()
        per_rank = [
            make_storage(hbm=i * GB, ddr=2 * i * GB, ssd=0) for i in range(WORLD_SIZE)
        ]
        before = copy.deepcopy([d.storage for d in topology.devices])

        reduced = topology_minus(topology, per_rank)

        for i, device in enumerate(reduced.devices):
            assert device.storage.hbm == HBM_PER_RANK - i * GB
            assert device.storage.ddr == DDR_PER_RANK - 2 * i * GB
        # the caller's topology is an input they may reuse
        assert [d.storage for d in topology.devices] == before

    def test_topology_minus_refuses_an_overdrawn_rank(self):
        """Neither clamped nor passed on. Clamping reads as "no room, plan
        accordingly" and the plan comes back looking feasible; passing it on relies
        on TorchRec's reservation, which tests `devices[0]` and no other rank."""
        topology = _topology(hbm=1 * GB)
        with pytest.raises(PlannerError, match="do not fit"):
            topology_minus(topology, [make_storage(hbm=4 * GB)] * WORLD_SIZE)

    def test_an_overdraft_away_from_rank_zero_is_still_caught(self):
        """The case TorchRec's own check misses: rank 0 is fine, a later rank is
        not. A table-row-wise table pinned to any node but the first lands here."""
        spent = [make_storage(hbm=0)] * (WORLD_SIZE - 1) + [make_storage(hbm=200 * GB)]
        with pytest.raises(PlannerError, match=f"Rank {WORLD_SIZE - 1} has"):
            topology_minus(_topology(), spent)

    def test_an_exact_fit_is_not_an_overdraft(self):
        topology = _topology(hbm=4 * GB, ddr=8 * GB)
        reduced = topology_minus(
            topology, [make_storage(hbm=4 * GB, ddr=8 * GB)] * WORLD_SIZE
        )
        assert reduced.devices[0].storage.hbm == 0
        assert reduced.devices[0].storage.ddr == 0

    def test_negative_tiers_reads_ssd_where_the_field_exists(self, monkeypatch):
        """The ssd half of the guard, on a TorchRec whose Storage has no ssd.

        `storage_ssd` reads the flag at call time, so forcing it exercises the
        branch that the container this usually runs in cannot reach. Without this
        the ssd case would be skipped everywhere and never execute at all."""
        monkeypatch.setattr(plan, "_STORAGE_HAS_SSD", True)

        assert plan._negative_tiers(SimpleNamespace(hbm=1, ddr=1, ssd=-1)) == ["ssd"]
        assert plan._negative_tiers(SimpleNamespace(hbm=-1, ddr=1, ssd=-1)) == [
            "hbm",
            "ssd",
        ]
        assert plan._negative_tiers(SimpleNamespace(hbm=1, ddr=1, ssd=0)) == []

    @pytest.mark.skipif(
        not _STORAGE_HAS_SSD, reason="Storage.ssd exists from TorchRec 1.5.0"
    )
    def test_an_ssd_overdraft_is_caught(self):
        """`Storage.__sub__` subtracts all three tiers. A guard reading only hbm
        and ddr would hand the planner a negative ssd without a word.

        `ssd_cap=0` is stated rather than left out: TorchRec defaults it to 2 TB
        (`planner/constants.py:27`), which no plausible per-rank cost overdraws."""
        topology = Topology(
            world_size=WORLD_SIZE,
            local_world_size=LOCAL_SIZE,
            compute_device="cuda",
            hbm_cap=HBM_PER_RANK,
            ddr_cap=DDR_PER_RANK,
            ssd_cap=0,
        )
        assert topology.devices[0].storage.ssd == 0, "nothing to draw on"

        with pytest.raises(PlannerError, match="overdraws ssd"):
            topology_minus(topology, [make_storage(ssd=1 * GB)] * WORLD_SIZE)

    def test_topology_minus_rejects_a_length_mismatch(self):
        with pytest.raises(ValueError, match="entries but the topology has"):
            topology_minus(_topology(), [make_storage()] * (WORLD_SIZE - 1))


class TestTheBroadcastSeam:
    """Decisions travel, options do not.

    Rank 0 broadcasts a plan of plain ``ParameterSharding``s; every rank then
    completes its own entries from the options it kept, because
    ``external_storage`` is a live handle and ``score_function`` a callable.
    """

    def test_the_plan_record_carries_decisions_and_nothing_else(self):
        """`_insert_dynamicemb_plan` puts its own record straight into the plan, so
        the record has to be exactly what the plan carries. Options on it would
        ride the broadcast, where `external_storage` is a live handle this process
        opened and `score_function` is a Python callable."""
        planned = _planned_table(range(WORLD_SIZE))

        assert type(planned.sharding) is ParameterSharding
        assert not hasattr(planned.sharding, "dynamicemb_options")

    def test_the_options_are_added_on_arrival(self):
        """The other half: a rank completes the decisions it was sent with the
        options it kept. Every inherited field has to survive, which is why
        `with_options` copies them through `fields(ParameterSharding)` rather than
        naming them."""
        planned = _planned_table(range(WORLD_SIZE))

        completed = DynamicEmbParameterSharding.with_options(
            planned.sharding, planned.options
        )

        assert isinstance(completed, DynamicEmbParameterSharding)
        assert completed.dynamicemb_options is planned.options
        for f in fields(ParameterSharding):
            if f.name == "sharding_spec":
                continue  # copied, not shared -- the next test is about that
            assert getattr(completed, f.name) == getattr(
                planned.sharding, f.name
            ), f.name

        # What TorchRec will ask it for, once the kernel is built.
        assert completed.get_additional_fused_params() == {
            "dynamicemb_options": planned.options
        }

    def test_the_completed_entry_does_not_share_metadata_with_the_record(self):
        """The one field the loop above cannot check, because `==` cannot tell a
        copy from the original.

        DMP rewrites placements in place when its device is `meta`
        (`embeddingbag.py:183-188`), and on rank 0 the plan still holds the
        planner's own record -- a broadcast hands the leader back its own object
        rather than a round trip (`collective_utils.py:65-67`). Sharing the spec
        would let the first DMP rewrite what a second `plan` call hands out, on
        rank 0 and nowhere else.
        """
        planned = _planned_table(range(WORLD_SIZE))

        completed = DynamicEmbParameterSharding.with_options(
            planned.sharding, planned.options
        )

        # Two steps: `is not` alone passes for a shallow copy, which still shares
        # the `shards` list and the ShardMetadata objects DMP actually writes to.
        assert completed.sharding_spec is not planned.sharding.sharding_spec
        completed.sharding_spec.shards[0].placement = "rank:7/cuda:7"
        assert (
            str(planned.sharding.sharding_spec.shards[0].placement) != "rank:7/cuda:7"
        ), "a write through the completed entry reached the planner's own record"


# --- where the refusal happens ---------------------------------------------


def _dynamicemb_constraints(*names, hbm_budget=GB, **options_kwargs):
    """Row-wise DynamicEmb constraints, one per key, all of one budget.

    The keys are whatever `constraints` accepts -- a bare table name or a
    `collection.name` -- since that is the thing under test in several of these.
    """
    return {
        name: DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            use_dynamicemb=True,
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=hbm_budget, **options_kwargs
            ),
        )
        for name in names
    }


def _dynamicemb_planner(topology: Topology, table: str, hbm_budget: int):
    """A planner whose only table is a DynamicEmb one of the given HBM budget."""
    return DynamicEmbeddingShardingPlanner(
        topology=topology,
        constraints=_dynamicemb_constraints(table, hbm_budget=hbm_budget),
    )


class TestWhereTheRefusalHappens:
    """Overcommitment is refused in step 1, which every rank runs.

    Raised inside the broadcast it would unwind rank 0 alone and leave the
    others waiting for a plan that is never sent, so the one check that
    catches it would surface as a hang.
    """

    def test_an_overdraft_is_refused_before_the_broadcast(self):
        """`_plan_torchrec` runs on rank 0 alone, inside the broadcast in
        `collective_plan`. A refusal raised in there unwinds rank 0 and leaves
        every other rank waiting on a broadcast that will never be sent, so the
        one check that catches overcommitment would surface as a hang.

        `_plan_dynamicemb` runs on every rank, so that is where it belongs. This
        asserts it by calling step 1 alone -- `_plan_torchrec` is never reached."""
        # Row-wise over WORLD_SIZE ranks: 2**26 rows of DIM fp32 each, so 4 GiB a
        # rank, against a budget that allows it and a rank that has 1 GB.
        module = _ebc(table_names=["big"], rows=2**28)
        sharders = [EmbeddingBagCollectionSharder()]
        planner = _dynamicemb_planner(_topology(hbm=1 * GB), "big", 64 * GB)

        with pytest.raises(PlannerError, match="do not fit"):
            planner._plan_dynamicemb(module, sharders)

    def test_a_plan_that_fits_leaves_the_caller_topology_whole(self):
        """Step 1 deducts into a Topology of its own. `self._topology` still says
        what the machine has, because a second `plan` call has to start there."""
        topology = _topology()
        module = _ebc(table_names=["small"])
        planner = _dynamicemb_planner(topology, "small", 1 * GB)

        planner._plan_dynamicemb(module, [EmbeddingBagCollectionSharder()])

        assert planner._reduced_topology is not None
        assert planner._topology.devices[0].storage.hbm == HBM_PER_RANK
        assert planner._reduced_topology.devices[0].storage.hbm < HBM_PER_RANK


# --- module_without_tables -------------------------------------------------


def _ebc(table_names, device=torch.device("meta"), rows: int = ROWS):
    return torchrec.EmbeddingBagCollection(
        tables=[
            EmbeddingBagConfig(
                name=name,
                embedding_dim=DIM,
                num_embeddings=rows,
                feature_names=[f"f_{name}"],
            )
            for name in table_names
        ],
        device=device,
    )


class _Model(torch.nn.Module):
    def __init__(self, ebc):
        super().__init__()
        self.sparse = torch.nn.Module()
        self.sparse.ebc = ebc
        self.dense = torch.nn.Linear(DIM, DIM, device=torch.device("meta"))


def _plan_dynamicemb(module, constraints, sharders=None):
    """Run the planner's step 1 over `module` and hand back the planner.

    Afterwards `_dyn_emb_plan`, `_dyn_emb_options` and `_reduced_topology` are
    filled in. Nothing of TorchRec's has run and there is no `ShardingPlan`
    yet; `_insert_dynamicemb_plan` is what turns this into one.

    `sharders` defaults to the pooled collection's, which is what every table
    here is in. A test with a sequence collection passes its own.
    """
    planner = DynamicEmbeddingShardingPlanner(
        topology=_topology(), constraints=constraints
    )
    planner._plan_dynamicemb(module, sharders or [EmbeddingBagCollectionSharder()])
    return planner


class TestModuleWithoutTables:
    """The copy TorchRec is given to plan: the DynamicEmb tables gone, every
    path the plan is keyed by still where it was."""

    def test_named_tables_are_gone_from_the_copy(self):
        model = _Model(_ebc(table_names=["a", "b", "c"]))
        reduced = module_without_tables(
            model,
            {TableKey("sparse.ebc", "a"), TableKey("sparse.ebc", "c")},
            [EmbeddingBagCollectionSharder()],
        )

        assert set(reduced.sparse.ebc.embedding_bags.keys()) == {"b"}
        # the caller's module is untouched
        assert set(model.sparse.ebc.embedding_bags.keys()) == {"a", "b", "c"}

    def test_the_tree_keeps_its_shape_so_plan_paths_still_match(self):
        model = _Model(_ebc(table_names=["a", "b"]))
        reduced = module_without_tables(
            model, {TableKey("sparse.ebc", "a")}, [EmbeddingBagCollectionSharder()]
        )

        assert [n for n, _ in reduced.named_children()] == [
            n for n, _ in model.named_children()
        ]
        assert type(reduced.sparse.ebc) is type(model.sparse.ebc)
        # only the path to the pruned collection is copied; the rest is shared
        assert reduced.dense is model.dense

    def test_a_collection_of_only_dynamicemb_tables_becomes_empty(self):
        """It stays in the tree. Removing it would change the paths the plan is
        keyed by, and an empty collection simply yields no shardable parameters."""
        model = _Model(_ebc(table_names=["a", "b"]))
        reduced = module_without_tables(
            model,
            {TableKey("sparse.ebc", "a"), TableKey("sparse.ebc", "b")},
            [EmbeddingBagCollectionSharder()],
        )

        assert len(reduced.sparse.ebc.embedding_bags) == 0
        assert [EmbeddingBagCollectionSharder()][0].shardable_parameters(
            reduced.sparse.ebc
        ) == {}

    def test_nothing_is_copied_when_nothing_is_removed(self):
        model = _Model(_ebc(table_names=["a"]))
        assert (
            module_without_tables(model, set(), [EmbeddingBagCollectionSharder()])
            is model
        )
        assert (
            module_without_tables(
                model,
                {TableKey("sparse.ebc", "absent")},
                [EmbeddingBagCollectionSharder()],
            )
            is model
        )


# --- table_ranks: which ranks hold a table, and what its rows divide by ----

# A second, roomier machine, because table_ranks is about nodes and the one
# above has only two ranks in each. Named apart from WORLD_SIZE/LOCAL_SIZE
# rather than one underscore away from them.
WORLD_8, LOCAL_4 = 8, 4  # two nodes of four
RW = ShardingType.ROW_WISE.value
TRW = ShardingType.TABLE_ROW_WISE.value


class TestTableRanks:
    """Which ranks hold a table, and how many ways its rows are split."""

    def test_row_wise_is_every_rank(self):
        assert table_ranks(RW, None, WORLD_8, LOCAL_4) == list(range(WORLD_8))

    def test_table_row_wise_is_one_node(self):
        assert table_ranks(TRW, 0, WORLD_8, LOCAL_4) == [0, 1, 2, 3]
        assert table_ranks(TRW, 1, WORLD_8, LOCAL_4) == [4, 5, 6, 7]

    @pytest.mark.parametrize(
        "sharding_type, node_index, world, local, message",
        [
            # Not ignored: a row-wise table is on every rank, so naming a node for
            # it means the caller expected something the plan will not do.
            # `_plan_dynamicemb` leans on this -- it writes the placer's answer back
            # into the options for table-row-wise tables only, because clearing a
            # row-wise table's `node_index` to None first would turn the refusal
            # into a silent correction.
            (RW, 0, WORLD_8, LOCAL_4, "on every rank of every node"),
            # The mirror: one node has to be named, and there is no default worth
            # guessing.
            (TRW, None, WORLD_8, LOCAL_4, "needs a node_index"),
            (TRW, 2, WORLD_8, LOCAL_4, "outside the 2 nodes"),
            # A world that is not a whole number of nodes has no node to place on.
            (TRW, 0, 7, LOCAL_4, "does not divide into nodes"),
            # DynamicEmb plans these two sharding types and no others.
            (
                ShardingType.COLUMN_WISE.value,
                None,
                WORLD_8,
                LOCAL_4,
                "not 'column_wise'",
            ),
        ],
        ids=[
            "node_index_on_row_wise",
            "table_row_wise_without_node_index",
            "node_index_past_the_last_node",
            "world_is_not_whole_nodes",
            "unsupported_sharding_type",
        ],
    )
    def test_table_ranks_refuses(
        self, sharding_type, node_index, world, local, message
    ):
        with pytest.raises(ValueError, match=message):
            table_ranks(sharding_type, node_index, world, local)

    def test_a_table_row_wise_table_only_costs_its_own_node(self):
        """§4.1 and §4.2 meeting §2: the placement decides who pays, and
        per_rank_storage reads it off the shard metadata."""
        ranks = table_ranks(TRW, 1, WORLD_8, LOCAL_4)
        spent = _spent(
            {"t": _planned_table(ranks)},
            optimizer_types={"t": None},
            world_size=WORLD_8,
        )

        # No HBM budget and no optimizer state, so the whole shard is host bytes.
        a_shard = make_storage(hbm=0, ddr=ROWS * DIM * 4, ssd=0)
        for rank in range(WORLD_8):
            expected = a_shard if rank in ranks else make_storage()
            assert spent[rank] == expected, f"rank {rank}"

    def test_a_row_wise_table_is_split_the_same_way_it_always_was(self):
        """`table_fanout` replaced a bare `dist.get_world_size()`. For row-wise it
        has to be that same number, or every existing table resizes.

        The node size is varied to say that row-wise does not read it at all."""
        for world, local in [(8, 4), (2, 2), (1, 1), (16, 8), (4, 1)]:
            assert table_fanout(RW, world, local) == world

    def test_a_table_row_wise_table_is_split_over_one_node(self):
        """The mirror: a node's worth of ranks, whichever node it turns out to be.

        Not depending on *which* node is what lets a table's capacity and HBM
        budget be settled before the placer has decided where to put it."""
        for world, local in [(8, 4), (2, 2), (1, 1), (16, 8), (4, 1)]:
            assert table_fanout(TRW, world, local) == local

    @pytest.mark.parametrize(
        "sharding_type, world, local, message",
        [
            # DynamicEmb plans row-wise and table-row-wise and no others.
            (ShardingType.COLUMN_WISE.value, WORLD_8, LOCAL_4, "not 'column_wise'"),
            # A world that is not a whole number of nodes has no node to split
            # over -- and neither does a node of no ranks.
            (TRW, 7, LOCAL_4, "does not divide into nodes"),
            (TRW, WORLD_8, 0, "does not divide into nodes"),
        ],
        ids=[
            "unsupported_sharding_type",
            "world_is_not_whole_nodes",
            "node_of_no_ranks",
        ],
    )
    def test_table_fanout_refuses(self, sharding_type, world, local, message):
        with pytest.raises(ValueError, match=message):
            table_fanout(sharding_type, world, local)


class TestGroupedTiers:
    """What a set of tables that share a TBE costs a rank.

    The runtime sums the group's sizes and the group's HBM budgets to choose the
    layout, so a group that fits its summed budget is held in HBM whole and a
    table with budget to spare covers the others. Costing each table alone
    gives ``sum(min(total, budget))``, at most ``min(sum total, sum budget)``.
    A group that does not fit splits each table on its own budget instead.
    """

    def test_a_spare_budget_covers_the_group_when_the_group_fits(self):
        """The group's rows fit its summed budget, so the runtime holds all of
        them in HBM -- the table with no budget included. Costing each alone
        under-states HBM, and that is the direction that lets through a plan
        the machine will not hold."""
        rows_bytes = ROWS * DIM * 4
        small = _options(hbm_budget=5 * rows_bytes)  # needs 1x, allowed 5x
        large = _options(rows=4 * ROWS, hbm_budget=0)  # needs 4x, allowed none

        alone = [get_group_value_bytes_by_tier([(o, None)]) for o in (small, large)]
        per_table_hbm = sum(h for h, _ in alone)
        per_table_ddr = sum(d for _, d in alone)

        grouped_hbm, grouped_ddr = get_group_value_bytes_by_tier(
            [(small, None), (large, None)]
        )

        assert per_table_hbm < grouped_hbm, "per-table must under-state HBM here"
        assert per_table_ddr > grouped_ddr
        assert (grouped_hbm, grouped_ddr) == (5 * rows_bytes, 0)

    @pytest.mark.parametrize("caching", [False, True])
    def test_a_spare_budget_is_not_lent_when_the_group_does_not_fit(self, caching):
        """5x of rows against 2x of budget: the runtime splits each table on its
        own budget, so the small table is in HBM whole, the large one is all on
        the host, and the 1x the small table had spare is not spent. Charging
        the summed budget would put that 1x in HBM and leave it off the host."""
        rows_bytes = ROWS * DIM * 4
        small = _options(hbm_budget=2 * rows_bytes)  # needs 1x, allowed 2x
        large = _options(rows=4 * ROWS, hbm_budget=0)  # needs 4x, allowed none
        small.caching = large.caching = caching

        grouped = get_group_value_bytes_by_tier([(small, None), (large, None)])

        # Caching keeps every row on the host behind the cache.
        assert grouped == (rows_bytes, 5 * rows_bytes if caching else 4 * rows_bytes)
        if not caching:
            alone = [get_group_value_bytes_by_tier([(o, None)]) for o in (small, large)]
            assert grouped == (sum(h for h, _ in alone), sum(d for _, d in alone))

    def test_budgets_within_their_tables_are_spent_as_given(self):
        """The case a shared scale got wrong: 10% and 90% of two equal tables.
        No budget exceeds its table, so nothing is spare and the group spends
        exactly its summed budget, whichever table it belongs to."""
        rows_bytes = ROWS * DIM * 4
        a = _options(hbm_budget=rows_bytes // 10)
        b = _options(hbm_budget=rows_bytes * 9 // 10)
        hbm, ddr = get_group_value_bytes_by_tier([(a, None), (b, None)])
        assert hbm == a.local_hbm_for_values + b.local_hbm_for_values
        assert ddr == 2 * rows_bytes - hbm

    def test_a_group_of_one_is_the_single_table_answer(self):
        o = _options(hbm_budget=ROWS * DIM * 4 // 2)
        assert get_group_value_bytes_by_tier(
            [(o, None)]
        ) == get_local_value_bytes_by_tier(o, None)

    def test_an_unsized_table_reports_its_budget(self):
        """No dim or capacity means nothing to size from, so it fills its own
        budget rather than counting as free."""
        bare = DynamicEmbTableOptions(local_hbm_for_values=7)
        assert get_group_value_bytes_by_tier([(bare, None)]) == (7, 0)

    def test_per_rank_storage_groups_what_shares_a_rank(self):
        """Two tables with the same grouping key on the same ranks are one group,
        so the spare budget of one covers the other."""
        rows_bytes = ROWS * DIM * 4
        # 5x of budget for 5x of rows: the group fits, so it is held whole.
        small = _options(hbm_budget=5 * rows_bytes)
        large = _options(rows=4 * ROWS, hbm_budget=0)

        def sharding(options):
            ranks = list(range(WORLD_SIZE))
            return ParameterSharding(
                sharding_type=ShardingType.ROW_WISE.value,
                compute_kernel="customized_kernel",
                ranks=ranks,
                sharding_spec=EnumerableShardingSpec(
                    [
                        ShardMetadata(
                            shard_sizes=[options.max_capacity, DIM],
                            shard_offsets=[options.max_capacity * i, 0],
                            placement=f"rank:{r}/cuda:{r % LOCAL_SIZE}",
                        )
                        for i, r in enumerate(ranks)
                    ]
                ),
            )

        spent = per_rank_storage(
            {_key("small"): sharding(small), _key("large"): sharding(large)},
            {_key("small"): small, _key("large"): large},
            optimizer_types={_key("small"): None, _key("large"): None},
            world_size=WORLD_SIZE,
        )
        grouped = get_group_value_bytes_by_tier([(small, None), (large, None)])
        assert all(s == make_storage(hbm=grouped[0], ddr=grouped[1]) for s in spent)


# --- caching with no HBM budget ---------------------------------------------


class TestCachingWithoutHbmBudget:
    """`caching=True` asks for an HBM tier; a table with no HBM budget cannot have one.

    The runtime would build it a one-bucket cache whose finite overflow cannot
    absorb a batch's admitted keys, so the planner takes caching off before
    grouping or the cost model reads the field.
    """

    def test_zero_budget_caching_is_normalized_off_before_grouping(self):
        """Both tables ask for caching=True; only the funded one keeps it, and
        the two no longer share a grouped key, so they cannot be fused into one
        module. The zero-budget one is costed as the host-only table it will
        actually be."""
        module = _Model(_ebc(table_names=["zero", "funded"]))
        with pytest.warns(UserWarning, match="local_hbm_for_values of zero"):
            planner = _plan_dynamicemb(
                module,
                {
                    **_dynamicemb_constraints("zero", hbm_budget=0, caching=True),
                    **_dynamicemb_constraints(
                        "funded", hbm_budget=2 * GB, caching=True
                    ),
                },
            )
        options = planner._dyn_emb_options
        zero = options[TableKey("sparse.ebc", "zero")]
        funded = options[TableKey("sparse.ebc", "funded")]
        assert zero.caching is False
        assert funded.caching is True
        assert zero.get_grouped_key() != funded.get_grouped_key()
        # ... and costed as the host-only table it will actually be: no HBM,
        # its whole (per-rank) size on the host tier.
        assert get_group_value_bytes_by_tier([(zero, None)]) == (
            0,
            zero.max_capacity * zero.dim * 4,
        )


# --- optimizer_types: which optimizer each table trains with ----------------


def _ec(table_names, device=torch.device("meta")):
    return torchrec.EmbeddingCollection(
        tables=[
            EmbeddingConfig(
                name=name,
                embedding_dim=DIM,
                num_embeddings=ROWS,
                feature_names=[f"f_{name}"],
            )
            for name in table_names
        ],
        device=device,
    )


class _MixedModel(torch.nn.Module):
    """One pooled collection and one sequence collection, as a real model has.

    On CPU rather than meta because `apply_optimizer_in_backward` registers
    hooks on the parameters.
    """

    def __init__(self):
        super().__init__()
        self.pooled = _ebc(table_names=["p0", "p1"], device=torch.device("cpu"))
        self.sequence = _ec(table_names=["s0"], device=torch.device("cpu"))


def _mixed_sharders(pooled_optimizer=None, sequence_optimizer=None):
    return [
        EmbeddingBagCollectionSharder(
            fused_params=None
            if pooled_optimizer is None
            else {"optimizer": pooled_optimizer}
        ),
        EmbeddingCollectionSharder(
            fused_params=None
            if sequence_optimizer is None
            else {"optimizer": sequence_optimizer}
        ),
    ]


class TestOptimizerTypes:
    """Where a table's optimizer comes from, when the model is sharded by more
    than one sharder and they need not agree."""

    @pytest.fixture
    def model(self):
        return _MixedModel()

    def test_each_collection_reads_its_own_sharders_optimizer(self, model):
        """A table follows the sharder that claims it.

        Flattening the list to a single value -- taking the last sharder that
        names an optimizer -- would hand every table one module's answer, and
        the one it picked would depend on the order the caller happened to pass
        them in.
        """
        sharders = _mixed_sharders(
            pooled_optimizer=OptimType.ADAM,
            sequence_optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
        )

        assert optimizer_types(model, sharders) == {
            TableKey("pooled", "p0"): OptimType.ADAM,
            TableKey("pooled", "p1"): OptimType.ADAM,
            TableKey("sequence", "s0"): OptimType.EXACT_ROWWISE_ADAGRAD,
        }

        # and the answer does not depend on the order they were passed in
        assert optimizer_types(model, list(reversed(sharders))) == optimizer_types(
            model, sharders
        )

    def test_a_parameter_optimizer_wins_over_its_sharder(self, model):
        apply_optimizer_in_backward(
            torch.optim.SGD, model.pooled.parameters(), {"lr": 0.1}
        )

        assert optimizer_types(
            model, _mixed_sharders(pooled_optimizer=OptimType.ADAM)
        ) == {
            TableKey("pooled", "p0"): OptimType.EXACT_SGD,
            TableKey("pooled", "p1"): OptimType.EXACT_SGD,
            TableKey("sequence", "s0"): None,
        }

    def test_an_optimizer_torchrec_cannot_name_falls_back_to_the_sharder(self, model):
        """DynamicEmb's own FTRL is one of these: applied to the parameter, but
        with no EmbOptimType to convert to. The sharder's answer is better than
        none."""
        apply_optimizer_in_backward(
            torch.optim.RMSprop, model.pooled.parameters(), {"lr": 0.1}
        )

        assert optimizer_types(
            model, _mixed_sharders(pooled_optimizer=OptimType.ADAM)
        ) == {
            TableKey("pooled", "p0"): OptimType.ADAM,
            TableKey("pooled", "p1"): OptimType.ADAM,
            TableKey("sequence", "s0"): None,
        }

    def test_a_table_no_sharder_claims_is_not_reported(self, model):
        """`shardable_parameters` is the same accessor `module_without_tables`
        prunes by, so what is planned and what is costed are the same set."""
        assert set(optimizer_types(model, [EmbeddingBagCollectionSharder()])) == {
            TableKey("pooled", "p0"),
            TableKey("pooled", "p1"),
        }


# --- costing: two collections are two runtimes ------------------------------


class TestCollectionsDoNotPoolBudgets:
    """Two collections are costed apart, however alike their tables are."""

    def _row_wise_sharding(self, options):
        ranks = list(range(WORLD_SIZE))
        return ParameterSharding(
            sharding_type=ShardingType.ROW_WISE.value,
            compute_kernel="customized_kernel",
            ranks=ranks,
            sharding_spec=EnumerableShardingSpec(
                [
                    ShardMetadata(
                        shard_sizes=[options.max_capacity, DIM],
                        shard_offsets=[options.max_capacity * i, 0],
                        placement=f"rank:{r}/cuda:{r % LOCAL_SIZE}",
                    )
                    for i, r in enumerate(ranks)
                ]
            ),
        )

    def test_spare_budget_does_not_cross_a_collection_boundary(self):
        """The mirror of `test_per_rank_storage_groups_what_shares_a_rank`: the
        same two tables, told they came from different collections, are priced
        on their own and the one with no budget shows up on the host tier.

        Pooled they report no host memory at all, which is the unsafe
        direction -- the planner would approve a model whose second runtime
        asks for host memory nobody costed.
        """
        rows_bytes = ROWS * DIM * 4
        # Enough for both, so pooled they fit and need no host memory.
        generous = _options(hbm_budget=5 * rows_bytes)
        needy = _options(rows=4 * ROWS, hbm_budget=0)

        def records(generous_key, needy_key):
            return (
                {
                    generous_key: self._row_wise_sharding(generous),
                    needy_key: self._row_wise_sharding(needy),
                },
                {generous_key: generous, needy_key: needy},
                {generous_key: None, needy_key: None},
            )

        together = records(_key("generous"), _key("needy"))
        apart = records(TableKey("ebc_a", "generous"), TableKey("ebc_b", "needy"))

        pooled = per_rank_storage(
            together[0],
            together[1],
            optimizer_types=together[2],
            world_size=WORLD_SIZE,
        )
        split = per_rank_storage(
            apart[0], apart[1], optimizer_types=apart[2], world_size=WORLD_SIZE
        )

        alone = [get_group_value_bytes_by_tier([(o, None)]) for o in (generous, needy)]
        assert all(
            s
            == make_storage(hbm=sum(h for h, _ in alone), ddr=sum(d for _, d in alone))
            for s in split
        )
        assert all(s.ddr > 0 for s in split), (
            "`needy` has no budget of its own, so unpooled it has to spill -- "
            "which is exactly what pooling across collections hid"
        )
        assert all(
            p.ddr < s.ddr for p, s in zip(pooled, split)
        ), "pooling under-states the host tier, and by more than nothing"


class _TwoCollections(torch.nn.Module):
    """Two collections, told separately which tables to hold."""

    def __init__(self, left_names, right_names, right_rows=ROWS):
        super().__init__()
        self.left = _ebc(table_names=left_names, device=torch.device("meta"))
        self.right = _ebc(
            table_names=right_names, device=torch.device("meta"), rows=right_rows
        )


class TestGetConstraintKey:
    """The key a caller registers a table's constraint under.

    The answer depends only on the arguments: a key that moved because an
    unrelated collection grew a table would be a key written down elsewhere and
    now silently wrong.
    """

    def test_without_a_module_it_is_the_table_name(self):
        config = _ebc(table_names=["t"]).embedding_bag_configs()[0]
        assert get_constraint_key(config) == "t"

    def test_a_collection_passed_as_the_root_gives_the_bare_name(self):
        """Its path from itself is empty, so a caller holding one collection and
        no enclosing model needs no special case."""
        ebc = _ebc(table_names=["t"])
        assert get_constraint_key(ebc.embedding_bag_configs()[0], ebc) == "t"

    def test_a_module_gives_the_path_to_the_collection(self):
        """`l` is qualified too, though nothing collides with it: this is not
        "qualify only what is ambiguous". The answer is a function of the
        arguments alone, so a key does not move when an unrelated collection
        later grows a table of the same name."""
        module = _TwoCollections(left_names=["shared", "l"], right_names=["shared"])
        assert [
            get_constraint_key(c, module) for c in module.left.embedding_bag_configs()
        ] == ["left.shared", "left.l"]
        assert [
            get_constraint_key(c, module) for c in module.right.embedding_bag_configs()
        ] == ["right.shared"]

    def test_the_config_is_matched_by_identity_not_by_name(self):
        """By name there is nothing to tell two declarations apart, which is the
        whole problem this solves."""
        module = _TwoCollections(left_names=["shared"], right_names=["shared"])
        stranger = _ebc(table_names=["shared"]).embedding_bag_configs()[0]
        with pytest.raises(ValueError, match="matched by identity"):
            get_constraint_key(stranger, module)

    def test_one_config_object_in_two_collections_has_no_key(self):
        """Two collections built from one config object are two tables with one
        identity, and no key means either."""
        config = EmbeddingBagConfig(
            name="shared", embedding_dim=DIM, num_embeddings=ROWS, feature_names=["f"]
        )

        class _Shared(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.left = torchrec.EmbeddingBagCollection(
                    tables=[config], device=torch.device("meta")
                )
                self.right = torchrec.EmbeddingBagCollection(
                    tables=[config], device=torch.device("meta")
                )

        with pytest.raises(ValueError, match="more than one collection"):
            get_constraint_key(config, _Shared())


class TestResolveConstraints:
    """Which constraint each table occurrence answers to.

    Qualified before bare, and nothing silently unused: a constraint that would
    reach no table, or that TorchRec would never look up, is refused rather
    than ignored.

    Driven through the planner rather than by calling `resolve_constraints`,
    because what a caller writes is a `constraints` dict and what they get back
    is a planned table or an error. Which function inside reaches that answer
    is ours to move.

    That a bare name reaches its table at all is not tested here: every other
    test in this file keys its constraints that way, so the compatibility this
    rests on is what the rest of the file is written in.
    """

    def test_a_qualified_key_wins_over_a_bare_one(self):
        """Which is what makes a bare name usable as a default that one
        collection then overrides.

        That the bare key is overridden here and still not reported as naming
        no table is part of the same assertion: every key a table answers to is
        marked, not only the one that won, so reaching these asserts at all
        means the override did not read as a typo.
        """
        module = _TwoCollections(left_names=["shared"], right_names=["shared"])
        constraints = {
            **_dynamicemb_constraints("shared", hbm_budget=GB),
            **_dynamicemb_constraints("right.shared", hbm_budget=2 * GB),
        }
        options = _plan_dynamicemb(module, constraints)._dyn_emb_options
        # By identity: the table was planned with the options off the constraint
        # that won it, not with a copy that happens to match.
        assert (
            options[TableKey("left", "shared")]
            is constraints["shared"].dynamicemb_options
        )
        assert (
            options[TableKey("right", "shared")]
            is constraints["right.shared"].dynamicemb_options
        )

    def test_a_key_no_table_answers_to_is_refused(self):
        """Silently ignoring it reads as "plan this table with TorchRec's
        defaults", which is found in a profile rather than in an error."""
        module = _TwoCollections(left_names=["l"], right_names=["r"])
        with pytest.raises(ValueError, match="does not have"):
            _plan_dynamicemb(module, _dynamicemb_constraints("typo"))

    def test_a_qualified_key_on_a_torchrec_table_is_refused(self):
        """TorchRec's planner is keyed by table name alone, so it would never
        look the key up and the constraint would do nothing."""
        module = _TwoCollections(left_names=["l"], right_names=["r"])
        with pytest.raises(ValueError, match="TorchRec plans"):
            _plan_dynamicemb(
                module,
                {"left.l": DynamicEmbParameterConstraints(use_dynamicemb=False)},
            )

    def test_two_tables_sharing_one_options_object_are_refused(self):
        """What a bare name does when two collections declare it. The options
        carry each table's own geometry, written in at plan time, so one object
        cannot describe both -- and the message names the way out."""
        module = _TwoCollections(left_names=["shared"], right_names=["shared"])
        with pytest.raises(ValueError, match="get_constraint_key"):
            _plan_dynamicemb(module, _dynamicemb_constraints("shared"))


class TestATableInMoreThanOneCollection:
    """A name two collections declare is two tables, costed and configured as two.

    TorchRec reads it the same way: it enumerates per module path and
    identifies a sharding option by `path + "." + name`, so it costs both
    copies and does not refuse the model. `DynamicEmbDump` agrees from the
    other end, writing each collection into its own directory. What the planner
    used to do was refuse it outright -- for every table, DynamicEmb's and
    TorchRec's alike -- on a uniqueness check in
    `_prepare_dynemb_table_options`.
    """

    def test_each_copy_holds_its_own_options(self):
        """The thing that was not expressible before: two tables under one name,
        each sized and budgeted on its own."""
        module = _TwoCollections(left_names=["shared"], right_names=["shared"])
        planner = _plan_dynamicemb(
            module,
            {
                **_dynamicemb_constraints("left.shared", hbm_budget=GB),
                **_dynamicemb_constraints("right.shared", hbm_budget=2 * GB),
            },
        )
        options = planner._dyn_emb_options
        assert set(options) == {
            TableKey("left", "shared"),
            TableKey("right", "shared"),
        }
        assert (
            options[TableKey("right", "shared")].global_hbm_for_values
            == 2 * options[TableKey("left", "shared")].global_hbm_for_values
        )

    def test_the_copies_may_differ_in_shape(self):
        """One options object holds one dim, capacity and dtype, and two
        declarations of different shapes used to be refused for that reason.
        Each occurrence now has its own, so each is sized from its own config.
        """
        module = _TwoCollections(
            left_names=["shared"], right_names=["shared"], right_rows=ROWS * 4
        )
        planner = _plan_dynamicemb(
            module, _dynamicemb_constraints("left.shared", "right.shared")
        )
        options = planner._dyn_emb_options
        assert (
            options[TableKey("right", "shared")].max_capacity
            > options[TableKey("left", "shared")].max_capacity
        )

    def test_the_planner_costs_both_copies(self):
        """The reservation has to charge for each runtime the model builds.
        Charging once leaves a whole table's HBM asked for and never reserved.
        """
        shared = _TwoCollections(left_names=["shared"], right_names=["shared"])
        alone = _TwoCollections(left_names=["shared"], right_names=["other"])

        def reduced(module, constraints):
            planner = _plan_dynamicemb(module, constraints)
            assert planner._reduced_topology is not None
            return planner._reduced_topology.devices[0].storage.hbm

        # One name in two collections costs what two names in two collections
        # cost, because it is the same two runtimes with the same two budgets.
        spent_twice = HBM_PER_RANK - reduced(
            shared, _dynamicemb_constraints("left.shared", "right.shared")
        )
        spent_apart = HBM_PER_RANK - reduced(
            alone, _dynamicemb_constraints("shared", "other")
        )
        assert spent_twice > 0
        assert spent_twice == spent_apart

    def test_each_copy_is_filed_under_its_own_collection(self):
        module = _TwoCollections(left_names=["shared"], right_names=["shared"])
        planner = _plan_dynamicemb(
            module, _dynamicemb_constraints("left.shared", "right.shared")
        )
        plan = ShardingPlan({})
        planner._insert_dynamicemb_plan(plan, module, [EmbeddingBagCollectionSharder()])
        assert set(plan.plan) == {"left", "right"}
        assert "shared" in plan.plan["left"] and "shared" in plan.plan["right"]

    def test_only_the_copy_that_is_dynamicembs_is_taken_from_torchrec(self):
        """One name, DynamicEmb's in one collection and TorchRec's in the other.
        Pruning by name alone would take both away and leave TorchRec a table it
        was told to plan and can no longer see."""
        module = _TwoCollections(left_names=["shared"], right_names=["shared"])
        planner = _plan_dynamicemb(module, _dynamicemb_constraints("left.shared"))
        reduced = module_without_tables(
            module, set(planner._dyn_emb_plan), [EmbeddingBagCollectionSharder()]
        )
        assert set(reduced.left.embedding_bags) == set()
        assert set(reduced.right.embedding_bags) == {"shared"}


# --- the collection path: recorded once, filed once -------------------------


class TestCollectionPathsAgree:
    """The walk the planner makes, written once.

    `_table_configs` records which collection a table came out of and
    `_insert_dynamicemb_plan` files that table's ParameterSharding under the
    collection's path. They have to produce the same paths, and nothing made
    them -- each built the path with its own copy of
    `f"{path}.{name}" if path else name`.
    """

    def test_the_plan_lands_where_the_collection_was_recorded(self):
        """The invariant the shared walk exists for. A ParameterSharding filed
        under a path `DistributedModelParallel` does not look for is not an
        error there -- it shards the modules its plan names and leaves the rest
        alone -- so two walks that disagreed would surface as a collection that
        silently stayed unsharded."""
        module = _Model(_ebc(table_names=["t"]))
        planner = _plan_dynamicemb(module, _dynamicemb_constraints("t"))

        plan = ShardingPlan({})
        planner._insert_dynamicemb_plan(plan, module, [EmbeddingBagCollectionSharder()])

        assert set(plan.plan) == {"sparse.ebc"}
        assert {key.collection for key in planner._dyn_emb_plan} == {"sparse.ebc"}
        assert "t" in plan.plan["sparse.ebc"]
