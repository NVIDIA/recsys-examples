# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Distributed, model-level test for ``dynamicemb.pop_evicted_keys``.

Launch with torchrun, e.g. ``torchrun --nproc_per_node=2 -m pytest THIS_FILE -s``.

A small row-wise sharded DynamicEmb table is trained with full forward+backward
steps to force last-tier eviction across steps (the ref-counter releases the
previous step's keys on backward). Then the model-level API is checked:
  * ``pg=None``  -> each rank returns only its LOCAL shard's evicted keys, which
    are disjoint across ranks (row-wise sharding hashes a key to one owner);
  * ``pg`` given -> all_gather within pg, so every rank sees the same global
    union.
Clearing is rank-local (read-and-clear).

The planner / Platform helpers are self-contained copies of the ones in
``incremental_dump/test_distributed_dynamicemb.py`` so this retain-evicted-keys
test suite lives entirely under one folder.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torchrec
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EvictedItemMode,
)
from dynamicemb.incremental_dump import pop_evicted_keys
from dynamicemb.planner import (
    DynamicEmbeddingEnumerator,
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingBagCollectionSharder
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.types import BoundsCheckMode, ShardingType
from torchrec.modules.embedding_configs import PoolingType


class Platform:
    def __init__(self, device):
        device_id = device.index
        gpu_name = torch.cuda.get_device_name(device_id)
        if "A100" in gpu_name:
            self.intra_host_bw = 300e9
            self.inter_host_bw = 25e9
            self.hbm_cap = 80 * 1024 * 1024 * 1024
        elif "H100" in gpu_name:
            self.intra_host_bw = 450e9
            self.inter_host_bw = 25e9
            self.hbm_cap = 80 * 1024 * 1024 * 1024
        elif "H200" in gpu_name:
            self.intra_host_bw = 450e9
            self.inter_host_bw = 450e9
            self.hbm_cap = 140 * 1024 * 1024 * 1024
        else:
            raise RuntimeError(f"Not plan for {gpu_name}")


def _get_planner(table_name, eb_config, score_strategy, batch_size, device):
    const = DynamicEmbParameterConstraints(
        sharding_types=[ShardingType.ROW_WISE.value],
        pooling_factors=[1],
        num_poolings=[1],
        enforce_hbm=True,
        bounds_check_mode=BoundsCheckMode.NONE,
        use_dynamicemb=True,
        dynamicemb_options=DynamicEmbTableOptions(
            global_hbm_for_values=1024**3,
            score_strategy=score_strategy,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.DEBUG,
            ),
            evicted_item_mode=EvictedItemMode.RETAIN_KEY,
        ),
    )
    platform = Platform(device)
    topology = Topology(
        local_world_size=torchrec.distributed.comm.get_local_size(),
        world_size=dist.get_world_size(),
        compute_device=device.type,
        hbm_cap=platform.hbm_cap,
        # One node's host memory, not one rank's: Topology replicates
        # whatever it is given to every device instead of dividing it.
        ddr_cap=1024 * 1024 * 1024 * 1024 // torchrec.distributed.comm.get_local_size(),
        intra_host_bw=platform.intra_host_bw,
        inter_host_bw=platform.inter_host_bw,
    )
    enumerator = DynamicEmbeddingEnumerator(
        topology=topology, constraints={table_name: const}
    )
    return DynamicEmbeddingShardingPlanner(
        topology=topology,
        constraints={table_name: const},
        batch_size=batch_size,
        enumerator=enumerator,
        debug=True,
    )


@pytest.fixture
def optimizer_kwargs():
    return {
        "optimizer": EmbOptimType.ADAM,
        "learning_rate": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0,
        "eps": 0.001,
    }


@pytest.fixture(scope="session")
def backend_session():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    yield
    dist.destroy_process_group()


def test_pop_evicted_keys_distributed(backend_session, optimizer_kwargs):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    table_name, feature_name, dim = "t_0", "f_0", 8
    eb_config = torchrec.EmbeddingBagConfig(
        name=table_name,
        embedding_dim=dim,
        num_embeddings=1024,  # small -> per-rank capacity small -> forced eviction
        feature_names=[feature_name],
        pooling=PoolingType.SUM,
    )
    ebc = torchrec.EmbeddingBagCollection(
        device=torch.device("meta"), tables=[eb_config]
    )

    planner = _get_planner(
        table_name,
        eb_config,
        (DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU),
        batch_size=256,
        device=device,
    )
    sharder = DynamicEmbeddingBagCollectionSharder(fused_params=optimizer_kwargs)
    plan = planner.collective_plan(ebc, [sharder], dist.GroupMember.WORLD)
    model = DistributedModelParallel(
        module=ebc, device=device, sharders=[sharder], plan=plan
    )

    def _train(steps, base):
        for step in range(steps):
            off = base + local_rank * (10**12) + step * 1000
            vals = torch.arange(off + 1, off + 257, dtype=torch.int64, device=device)
            kjt = torchrec.KeyedJaggedTensor(
                keys=[feature_name],
                values=vals,
                lengths=torch.ones(256, dtype=torch.int64, device=device),
            )
            model(kjt).values().sum().backward()
        torch.cuda.synchronize()

    def _keys(result):
        return result.get("model", {}).get(table_name, torch.empty(0, device=device))

    # ---- Round 1: pg=None -> local, disjoint across ranks ----
    _train(steps=40, base=0)
    local = _keys(pop_evicted_keys(model))
    assert local.numel() > 0, "training must force last-tier eviction on this rank"
    local_set = set(local.tolist())
    assert len(local_set) == local.numel(), "local keys must be unique"

    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_set)
    for r in range(world_size):
        for q in range(r + 1, world_size):
            assert gathered[r].isdisjoint(
                gathered[q]
            ), "row-wise sharding -> each evicted key belongs to exactly one rank"

    # ---- Round 2: pg given -> global union, identical on every rank ----
    _train(steps=40, base=10**9)
    glob = _keys(pop_evicted_keys(model, pg=dist.group.WORLD))
    glob_set = set(glob.tolist())
    assert glob.numel() > 0 and len(glob_set) == glob.numel()
    gathered_g = [None] * world_size
    dist.all_gather_object(gathered_g, glob_set)
    for g in gathered_g:
        assert g == gathered_g[0], "pg aggregation -> every rank sees the same union"

    # ---- Read-and-clear: a pop with no new training is empty ----
    assert _keys(pop_evicted_keys(model)).numel() == 0
