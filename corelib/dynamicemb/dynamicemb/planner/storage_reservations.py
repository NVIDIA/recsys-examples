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

"""Tell the planner about the HBM the DynamicEmb tables are going to take."""

from typing import Dict, List, Optional

from torch import nn
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    PlannerError,
    PlannerErrorType,
    Storage,
    Topology,
)
from torchrec.distributed.planner.utils import storage_repr_in_gb
from torchrec.distributed.types import ModuleSharder
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torchrec.distributed.utils import optimizer_type_to_emb_opt_type

from ..dynamicemb_config import (
    DynamicEmbTableOptions,
    get_local_value_bytes_by_tier,
)


def _optimizer_types(
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


class DynamicEmbStorageReservation(HeuristicalStorageReservation):
    """TorchRec's heuristic, plus the HBM the DynamicEmb tables will occupy.

    A DynamicEmb table is deliberately free in the planner's search space: it
    gets one 1x1 shard per rank because the planner does not choose its
    placement -- ``DynamicEmbeddingShardingPlanner`` writes that
    ParameterSharding itself. The table still takes HBM, though, and until this
    existed nothing took it out of the budget. The planner was handed the whole
    device and gave back five percent, and then packed TorchRec tables into
    memory the DynamicEmb tables had already claimed. What used to happen at
    run time as an OOM now happens here as a PlannerError.

    A ``StorageReservation`` is where TorchRec puts this: it reserves storage
    for "non-sharded parts of the model", and from the planner's side that is
    exactly what these tables are -- spent, but not searched over.

    Both tiers, because a DynamicEmb table spends both: whatever does not fit in
    its HBM budget lives on the host, and under ``caching`` the host tier holds
    every row while HBM only caches them. TorchRec's own budget has a ``ddr``
    column that ``fits_in`` checks and ``fused_uvm``-family tables draw on, so
    leaving the host side out would repeat the HBM mistake one tier down.

    **What it reserves is a floor, not the total.** It counts the value rows and
    the optimizer state sharing them. The key map, the slot metadata and the
    score columns are on top of it, as is anything the table allocates while
    growing. A table whose optimizer this cannot determine is counted without
    optimizer state rather than guessed at. Reserving the floor is strictly
    better than reserving nothing; do not read a plan that fits as a promise
    that it runs.

    Read at ``reserve`` time rather than at construction, because
    ``_prepare_dynemb_table_options`` fills ``local_hbm_for_values`` in the
    planner's constructor -- after ``get_planner`` builds the Topology, and well
    before ``plan`` calls this.

    Args:
        dynamicemb_options: the per-table options, by table name. The same
            objects the constraints hold, so the figures are whatever the
            planner settled on.
        percentage: passed through to ``HeuristicalStorageReservation``.
    """

    def __init__(
        self,
        dynamicemb_options: Dict[str, DynamicEmbTableOptions],
        percentage: float = 0.05,
    ) -> None:
        super().__init__(percentage=percentage)
        self._dynamicemb_options = dynamicemb_options
        self._dynamicemb_storage: Optional[Storage] = None


    def reserve(
        self,
        topology: Topology,
        batch_size: int,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Topology:
        reserved_topology = super().reserve(
            topology=topology,
            batch_size=batch_size,
            module=module,
            sharders=sharders,
            constraints=constraints,
        )

        optimizer_types = _optimizer_types(module, sharders)
        hbm_per_rank = 0
        ddr_per_rank = 0
        for name, options in self._dynamicemb_options.items():
            hbm, ddr = get_local_value_bytes_by_tier(
                options, optimizer_types.get(name)
            )
            hbm_per_rank += hbm
            ddr_per_rank += ddr
        if hbm_per_rank <= 0 and ddr_per_rank <= 0:
            return reserved_topology

        self._dynamicemb_storage = Storage(hbm=hbm_per_rank, ddr=ddr_per_rank)
        for device in reserved_topology.devices:
            device.storage -= self._dynamicemb_storage

        if (
            reserved_topology.devices[0].storage.hbm < 0
            or reserved_topology.devices[0].storage.ddr < 0
        ):
            raise PlannerError(
                error_type=PlannerErrorType.INSUFFICIENT_STORAGE,
                message=(
                    "The DynamicEmb tables leave no room for the TorchRec ones. "
                    f"Per rank: the device has "
                    f"{storage_repr_in_gb(topology.devices[0].storage)}, the "
                    f"DynamicEmb tables take "
                    f"{storage_repr_in_gb(self._dynamicemb_storage)}, and what "
                    "TorchRec's own reservation (percentage, dense modules, KJT "
                    "buffers) leaves on top of that is "
                    f"{storage_repr_in_gb(reserved_topology.devices[0].storage)}.\n"
                    "Lowering global_hbm_for_values moves rows from HBM to the "
                    "host tier, which trades one column for the other rather "
                    "than freeing both; if it is ddr that ran out, the tables "
                    "need fewer rows, not a different tier."
                ),
            )
        return reserved_topology
