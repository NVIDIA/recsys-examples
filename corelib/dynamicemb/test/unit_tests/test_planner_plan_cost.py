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

import pytest
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torchrec.distributed.planner.types import Storage, Topology
from torchrec.distributed.types import (
    EnumerableShardingSpec,
    ShardingType,
    ShardMetadata,
)

from dynamicemb.dynamicemb_config import DynamicEmbTableOptions
from dynamicemb.planner.plan import per_rank_storage, shard_rank, topology_minus
from dynamicemb.planner.planners import DynamicEmbParameterSharding

WORLD_SIZE = 4
LOCAL_SIZE = 2
DIM = 16
ROWS = 1024
GB = 1024**3


def _options(rows: int = ROWS, hbm_budget: int = 0) -> DynamicEmbTableOptions:
    return DynamicEmbTableOptions(
        max_capacity=rows,
        dim=DIM,
        embedding_dtype=torch.float32,
        local_hbm_for_values=hbm_budget,
    )


def _sharding(ranks, rows: int = ROWS, hbm_budget: int = 0):
    """A ParameterSharding placing one shard of `rows` rows on each of `ranks`."""
    return DynamicEmbParameterSharding(
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
        dynamicemb_options=_options(rows, hbm_budget),
    )


def _topology(hbm: int = 80 * GB, ddr: int = 128 * GB) -> Topology:
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
    assert shard_rank(shard) == 3

    shard.placement = "rank:3/cuda:1"
    assert shard_rank(shard) == 3


def test_row_wise_table_charges_every_rank_equally():
    per_rank = per_rank_storage(
        {"t": _sharding(range(WORLD_SIZE))},
        optimizer_types={"t": None},
        world_size=WORLD_SIZE,
    )
    assert len(per_rank) == WORLD_SIZE
    values = ROWS * DIM * 4  # float32, no optimizer state
    assert all(s == Storage(hbm=0, ddr=values, ssd=0) for s in per_rank)


def test_a_table_on_one_node_charges_only_that_node():
    """The reason this is per rank and not one repeated figure.

    Under TABLE_ROW_WISE a table lives on one node's ranks. Charging every rank
    for it would reserve four nodes' worth of room for one node's table.
    """
    per_rank = per_rank_storage(
        {"t": _sharding([2, 3])},  # the second node only
        optimizer_types={"t": None},
        world_size=WORLD_SIZE,
    )
    values = ROWS * DIM * 4
    assert per_rank[0] == Storage(0, 0, 0)
    assert per_rank[1] == Storage(0, 0, 0)
    assert per_rank[2] == Storage(hbm=0, ddr=values, ssd=0)
    assert per_rank[3] == Storage(hbm=0, ddr=values, ssd=0)


def test_tables_on_a_rank_add_up():
    per_rank = per_rank_storage(
        {"a": _sharding(range(WORLD_SIZE)), "b": _sharding([0])},
        optimizer_types={"a": None, "b": None},
        world_size=WORLD_SIZE,
    )
    values = ROWS * DIM * 4
    assert per_rank[0].ddr == 2 * values
    assert per_rank[1].ddr == values


def test_optimizer_state_lands_entirely_in_ddr():
    """local_hbm_for_values is a fixed budget, so the optimizer's extra bytes
    cannot come out of HBM -- they all spill to the host tier."""
    hbm_budget = ROWS * DIM * 4  # exactly the values, nothing spare
    without = per_rank_storage(
        {"t": _sharding(range(WORLD_SIZE), hbm_budget=hbm_budget)},
        optimizer_types={"t": None},
        world_size=WORLD_SIZE,
    )
    with_adam = per_rank_storage(
        {"t": _sharding(range(WORLD_SIZE), hbm_budget=hbm_budget)},
        optimizer_types={"t": OptimType.ADAM},
        world_size=WORLD_SIZE,
    )

    assert without[0].ddr == 0
    assert with_adam[0].hbm == without[0].hbm == hbm_budget
    assert with_adam[0].ddr > 0


def test_a_shard_outside_the_world_is_an_error():
    with pytest.raises(ValueError, match="outside the world"):
        per_rank_storage(
            {"t": _sharding([WORLD_SIZE])},
            optimizer_types={"t": None},
            world_size=WORLD_SIZE,
        )


def test_topology_minus_subtracts_per_rank_and_copies():
    topology = _topology()
    per_rank = [Storage(hbm=i * GB, ddr=2 * i * GB, ssd=0) for i in range(WORLD_SIZE)]
    before = copy.deepcopy([d.storage for d in topology.devices])

    reduced = topology_minus(topology, per_rank)

    for i, device in enumerate(reduced.devices):
        assert device.storage.hbm == 80 * GB - i * GB
        assert device.storage.ddr == 128 * GB - 2 * i * GB
    # the caller's topology is an input they may reuse
    assert [d.storage for d in topology.devices] == before


def test_topology_minus_does_not_clamp():
    """Zero means "no room"; negative means "already oversubscribed". Only the
    second is true here, and the planner's error path reports the shortfall."""
    topology = _topology(hbm=1 * GB)
    reduced = topology_minus(topology, [Storage(hbm=4 * GB, ddr=0, ssd=0)] * WORLD_SIZE)
    assert reduced.devices[0].storage.hbm == -3 * GB


def test_topology_minus_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="entries but the topology has"):
        topology_minus(_topology(), [Storage(0, 0, 0)] * (WORLD_SIZE - 1))
