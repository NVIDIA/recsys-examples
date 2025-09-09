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

# import our own finalize model grads
from dynamicemb.dynamicemb_config import DynamicEmbTableOptions
from dynamicemb.planner import DynamicEmbeddingEnumerator
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner as DynamicEmbeddingShardingPlanner,
)
from dynamicemb.planner import DynamicEmbParameterConstraints
from torch import distributed as dist
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embedding_types import ShardingType

# from torchrec.distributed import ModuleShardingPlan
from torchrec.distributed.planner import Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.types import BoundsCheckMode, ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig

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
    ddr_cap: int = 512 * 1024 * 1024 * 1024,  # Assume a Node have 512GB memory
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
            # TODO add dynamic embedding compute kernels
            compute_kernel_type = []
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

    topology = Topology(
        local_world_size=get_local_size(),
        world_size=dist.get_world_size(),
        compute_device=device.type,
        hbm_cap=hbm_cap,
        ddr_cap=ddr_cap,  # For HVK  , if we need to put embedding vector into Host memory , it is important set ddr capacity
        intra_host_bw=intra_host_bw,
        inter_host_bw=inter_host_bw,
    )
    enumerator = DynamicEmbeddingEnumerator(
        topology=topology,
        constraints=constraints,
    )
    return DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=topology,
        constraints=constraints,
        enumerator=enumerator,
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
    )
