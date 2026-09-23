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

# pyre-strict
from typing import Dict, List, Set

import torch
import torch.distributed as dist
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner import Topology
from torchrec.distributed.types import BoundsCheckMode, ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig

from ..dynamicemb_config import DynamicEmbTableOptions
from .enumerators import DynamicEmbeddingEnumerator
from .planners import (
    DynamicEmbeddingShardingPlanner as DynamicEmbeddingShardingPlanner,
)
from .planners import DynamicEmbParameterConstraints

# refer to https://github.com/pytorch/torchrec/blob/76a0826c6aec07c347f492aed2d4adf25cbdc3d9/torchrec/distributed/embedding_types.py#L75-L91
# compute_kernel is somehow coupled with sharding_type.
_pipeline_type_to_model_parallel_allowed_compute_kernels = {
    "prefetch": ["fused_uvm_caching"],
    "native": ["fused", "fused_uvm"],
    "none": [],  # none does not constrain the compute kernels
}
_pipeline_type_to_data_parallel_allowed_compute_kernels = {
    "prefetch": ["dense"],
    "native": ["dense"],
    "none": [],
}
_sharding_type_to_allowed_compute_kernels = {
    "data_parallel": _pipeline_type_to_data_parallel_allowed_compute_kernels,
    "model_parallel": _pipeline_type_to_model_parallel_allowed_compute_kernels,
}


def get_planner(
    eb_configs: List[EmbeddingConfig],
    data_parallel_embedding_table_names: Set[str],
    dynamicemb_options_dict: Dict[str, DynamicEmbTableOptions],
    device: torch.device,
    pipeline_type: str = "none",
    # Host memory of one node, not one rank: Topology keeps ddr per rank and
    # this is divided by the local world size below. A terabyte is a
    # conservative figure for the nodes this runs on rather than a measurement
    # of any of them -- a caller who knows their machine should say so.
    ddr_cap: int = 1024 * 1024 * 1024 * 1024,
    intra_host_bw: int = 450e9,  # Nvlink bandwidth
    inter_host_bw: int = 25e9,  # NIC bandwidth
):
    constraints = {}
    for config in eb_configs:
        if config.name in data_parallel_embedding_table_names:
            compute_kernel_type = _sharding_type_to_allowed_compute_kernels[
                "data_parallel"
            ][pipeline_type]
            constraint = DynamicEmbParameterConstraints(
                sharding_types=[
                    ShardingType.DATA_PARALLEL.value,
                ],
                bounds_check_mode=BoundsCheckMode.NONE,
                use_dynamicemb=False,
                compute_kernels=compute_kernel_type,
            )
        elif config.name in dynamicemb_options_dict:
            # No compute_kernels: a DynamicEmb table's kernel is not something
            # to search over. DynamicEmbeddingEnumerator._filter_compute_kernels
            # pins it to one placeholder so the table stays in the search space,
            # and DynamicEmbeddingShardingPlanner then replaces the whole
            # ParameterSharding, CUSTOMIZED_KERNEL included.
            dynamicemb_options = dynamicemb_options_dict[config.name]
            constraint = DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                bounds_check_mode=BoundsCheckMode.NONE,  # dynamic embedding has no bounding!
                enforce_hbm=True,
                use_dynamicemb=True,
                dynamicemb_options=dynamicemb_options,
            )
        else:
            compute_kernel_type = _sharding_type_to_allowed_compute_kernels[
                "model_parallel"
            ][pipeline_type]
            # TODO: save and load does not support table-wise sharding, disable them for now
            constraint = DynamicEmbParameterConstraints(
                sharding_types=[
                    ShardingType.ROW_WISE.value,
                    # ShardingType.TABLE_WISE.value,
                    # ShardingType.TABLE_ROW_WISE.value,
                ],
                bounds_check_mode=BoundsCheckMode.NONE,
                use_dynamicemb=False,
                compute_kernels=compute_kernel_type,
            )
        constraints.update({config.name: constraint})
    hbm_cap = torch.cuda.get_device_properties(0).total_memory

    # Topology stores ddr per rank -- `[ddr_cap] * world_size`, replicated, not
    # divided. `ddr_cap` is a node's host memory, shared by the ranks on it, so
    # handing it over as-is tells the planner every rank owns the whole node's
    # RAM: eight times too much on an eight-GPU node. Divide it here, which is
    # right while the ranks of a node spend it evenly -- they do under row-wise,
    # where each holds a slice of every table.
    local_world_size = get_local_size()
    topology = Topology(
        local_world_size=local_world_size,
        world_size=dist.get_world_size(),
        compute_device=device.type,
        hbm_cap=hbm_cap,
        ddr_cap=ddr_cap // local_world_size,
        intra_host_bw=intra_host_bw,
        inter_host_bw=inter_host_bw,
    )
    enumerator = DynamicEmbeddingEnumerator(
        topology=topology,
        constraints=constraints,
    )
    return DynamicEmbeddingShardingPlanner(
        topology=topology,
        constraints=constraints,
        enumerator=enumerator,
        # No storage_reservation: the planner takes the DynamicEmb tables out of
        # the Topology itself, and TorchRec's default reservation then covers
        # what it is for -- the dense modules and the input KJT.
    )
