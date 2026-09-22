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
from typing import Dict, List, Optional

from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torchrec.distributed.planner.types import Storage, Topology
from torchrec.distributed.types import (
    EnumerableShardingSpec,
    ParameterSharding,
    ShardMetadata,
)

from ..dynamicemb_config import get_local_value_bytes_by_tier

__all__ = ["shard_rank", "per_rank_storage", "topology_minus"]


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


def per_rank_storage(
    parameter_shardings: Dict[str, ParameterSharding],
    optimizer_types: Dict[str, Optional[OptimType]],
    world_size: int,
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

    ``optimizer_types`` maps table name -> ``OptimType``; a table missing from it
    is counted without optimizer state, which is a floor rather than a guess (see
    :func:`~dynamicemb.dynamicemb_config.get_local_value_bytes_by_tier`).
    """
    totals = [Storage(hbm=0, ddr=0, ssd=0) for _ in range(world_size)]

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

        # get_local_value_bytes_by_tier answers for *one* rank's share, which is
        # what a shard is. Options carry max_capacity and local_hbm_for_values
        # already divided by the table's fan-out, so every shard of a table
        # weighs the same and only the set of ranks differs.
        hbm, ddr = get_local_value_bytes_by_tier(options, optimizer_types.get(name))
        share = Storage(hbm=hbm, ddr=ddr, ssd=0)

        for shard in spec.shards:
            rank = shard_rank(shard)
            if not 0 <= rank < world_size:
                raise ValueError(
                    f"Table {name!r} places a shard on rank {rank}, which is "
                    f"outside the world of {world_size}."
                )
            totals[rank] += share

    return totals


def topology_minus(topology: Topology, per_rank: List[Storage]) -> Topology:
    """``topology`` with each rank's DynamicEmb footprint taken out of it.

    Returns a copy: the caller's Topology is an input they may reuse, and
    TorchRec's planner keeps a reference to whatever it is given.

    A rank left with negative storage is not clamped to zero. Zero reads as "no
    room, plan accordingly" while negative reads as "already oversubscribed", and
    only the second is true; the planner's own error path reports the shortfall,
    which a clamp would hide.
    """
    if len(per_rank) != len(topology.devices):
        raise ValueError(
            f"per_rank has {len(per_rank)} entries but the topology has "
            f"{len(topology.devices)} devices."
        )

    reduced = copy.deepcopy(topology)
    for device in reduced.devices:
        device.storage -= per_rank[device.rank]
    return reduced
