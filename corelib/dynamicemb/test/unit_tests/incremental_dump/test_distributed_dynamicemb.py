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

import os
import random
from typing import Dict, List

import pytest
import torch
import torch.distributed as dist
import torchrec
from dynamicemb import (
    BATCH_SIZE_PER_DUMP,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamicemb.incremental_dump import get_score, incremental_dump, set_score
from dynamicemb.planner import (
    DynamicEmbeddingEnumerator,
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import (
    DynamicEmbeddingBagCollectionSharder,
    DynamicEmbeddingCollectionSharder,
)
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed.comm import intra_and_cross_node_pg
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.types import BoundsCheckMode, ShardingType
from torchrec.modules.embedding_configs import BaseEmbeddingConfig, PoolingType


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


class CustomizedScore:
    def __init__(self, table_names: List[int]):
        self.table_names_ = table_names
        self.steps_: Dict[str, int] = {table_name: 1 for table_name in table_names}

    def get(self, table_name: str):
        assert table_name in self.table_names_
        ret = self.steps_[table_name]
        self.steps_[table_name] += 1
        return ret


def random_indices(batch, min_index, max_index):
    result = set({})
    while len(result) < batch:
        result.add(random.randint(min_index, max_index))
    return result


class Platform:
    def __init__(self, device):
        device_id = device.index
        gpu_name = torch.cuda.get_device_name(device_id)
        if "A100" in gpu_name:
            self.platform = "a100"
            self.intra_host_bw = 300e9
            self.inter_host_bw = 25e9
            self.hbm_cap = 80 * 1024 * 1024 * 1024
        elif "H100" in gpu_name:
            self.platform = "h100"
            self.intra_host_bw = 450e9
            self.inter_host_bw = 25e9  # TODO: need check
            self.hbm_cap = 80 * 1024 * 1024 * 1024
        elif "H200" in gpu_name:
            self.platform = "h200"
            self.intra_host_bw = 450e9
            self.inter_host_bw = 450e9
            self.hbm_cap = 140 * 1024 * 1024 * 1024
        else:
            raise RuntimeError(f"Not plan for {gpu_name}")


def get_planner(
    table_names: List[str],
    eb_configs: List[BaseEmbeddingConfig],
    use_dynamicembs: List[bool],
    score_strategies: List[DynamicEmbScoreStrategy],
    batch_size: int,
    multi_hot_sizes: List[int],
    device,
):
    dict_const = {}
    for i in range(len(table_names)):
        const = DynamicEmbParameterConstraints(
            sharding_types=[
                ShardingType.ROW_WISE.value,
            ],
            pooling_factors=[multi_hot_sizes[i]],
            num_poolings=[1],
            enforce_hbm=True,
            bounds_check_mode=BoundsCheckMode.NONE,
            use_dynamicemb=use_dynamicembs[i],
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=1024**3,
                score_strategy=score_strategies[i],
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.DEBUG,
                ),
            ),
        )
        dict_const[table_names[i]] = const

    platform = Platform(device)
    topology = Topology(
        local_world_size=torchrec.distributed.comm.get_local_size(),
        world_size=dist.get_world_size(),
        compute_device=device.type,
        hbm_cap=platform.hbm_cap,
        ddr_cap=1024 * 1024 * 1024 * 1024,
        intra_host_bw=platform.intra_host_bw,
        inter_host_bw=platform.inter_host_bw,
    )
    enumerator = DynamicEmbeddingEnumerator(
        topology=topology,
        constraints=dict_const,
    )

    return DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=topology,
        constraints=dict_const,
        batch_size=batch_size,
        enumerator=enumerator,
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        debug=True,
    )


def generate_sparse_feature(
    feature_names: List[str],
    multi_hot_sizes: List[int],
    local_batch_size: int,
    unique_indices_list: List[set],
    use_dynamicembs: List[bool],
    num_embeddings: List[int],
):
    feature_num = len(feature_names)
    feature_batch = feature_num * local_batch_size

    indices = []
    lengths = []

    for i in range(feature_batch):
        f = i // local_batch_size
        cur_bag_size = random.randint(0, multi_hot_sizes[f])
        cur_bag = set({})
        while len(cur_bag) < cur_bag_size:
            if use_dynamicembs[f]:
                cur_bag.add(random.randint(0, (1 << 63) - 1))
            else:
                cur_bag.add(random.randint(0, num_embeddings[f] - 1))

        unique_indices_list[f].update(cur_bag)
        indices.extend(list(cur_bag))
        lengths.append(cur_bag_size)

    return torchrec.KeyedJaggedTensor(
        keys=feature_names,
        values=torch.tensor(indices, dtype=torch.int64).cuda(),
        lengths=torch.tensor(lengths, dtype=torch.int64).cuda(),
    )


@pytest.fixture
def optimizer_kwargs():
    optimizer_kwargs_ = {
        "optimizer": EmbOptimType.ADAM,
        "learning_rate": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0,
        "eps": 0.001,
    }
    return optimizer_kwargs_


@pytest.fixture(scope="session")
def backend_session():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    yield
    # dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "table_num, num_embeddings, use_dynamicembs, score_strategies, multi_hot_sizes",
    [
        pytest.param(
            1,
            [BATCH_SIZE_PER_DUMP * 8],
            [True],
            [DynamicEmbScoreStrategy.TIMESTAMP],
            [10],
        ),
        pytest.param(
            4,
            [BATCH_SIZE_PER_DUMP * 8] * 4,
            [True] * 4,
            [
                DynamicEmbScoreStrategy.STEP,
                DynamicEmbScoreStrategy.TIMESTAMP,
                DynamicEmbScoreStrategy.CUSTOMIZED,
                DynamicEmbScoreStrategy.STEP,
            ],
            [10] * 4,
        ),
        pytest.param(
            4,
            [BATCH_SIZE_PER_DUMP * 8] * 4,
            [False, False, False, False],
            [None, None, None, None],
            [10] * 4,
        ),
        pytest.param(
            4,
            [BATCH_SIZE_PER_DUMP * 8] * 4,
            [True, False, True, False],
            [
                DynamicEmbScoreStrategy.STEP,
                None,
                DynamicEmbScoreStrategy.CUSTOMIZED,
                None,
            ],
            [10] * 4,
        ),
        pytest.param(
            4,
            [BATCH_SIZE_PER_DUMP * 8] * 4,
            [True] * 4,
            [DynamicEmbScoreStrategy.CUSTOMIZED] * 4,
            [10] * 4,
        ),
    ],
)
@pytest.mark.parametrize(
    "is_pooled, pooling_mode",
    [
        (True, PoolingType.SUM),
        (False, None),
        (False, None),
    ],
)
@pytest.mark.parametrize("local_batch", [128])
@pytest.mark.parametrize("num_iteration", [12])
@pytest.mark.parametrize("dump_interval", [3])
@pytest.mark.parametrize("dim", [8])
def test_incremental_dump_api(
    request,
    table_num,
    num_embeddings,
    use_dynamicembs,
    score_strategies,
    multi_hot_sizes,
    is_pooled,
    pooling_mode,
    local_batch,
    num_iteration,
    dump_interval,
    dim,
    optimizer_kwargs,
    backend_session,
):
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    table_names = [f"t_{t}" for t in range(table_num)]
    feature_names = [f"f_{t}" for t in range(table_num)]

    if is_pooled:
        eb_configs = [
            torchrec.EmbeddingBagConfig(
                name=table_names[feature_idx],
                embedding_dim=dim,
                num_embeddings=num_embeddings[feature_idx],
                feature_names=[feature_names[feature_idx]],
                pooling=pooling_mode,
            )
            for feature_idx in range(table_num)
        ]
        ebc = torchrec.EmbeddingBagCollection(
            device=torch.device("meta"),
            tables=eb_configs,
        )
    else:
        eb_configs = [
            torchrec.EmbeddingConfig(
                name=table_names[feature_idx],
                embedding_dim=dim,
                num_embeddings=num_embeddings[feature_idx],
                feature_names=[feature_names[feature_idx]],
            )
            for feature_idx in range(table_num)
        ]
        ebc = torchrec.EmbeddingCollection(
            device=torch.device("meta"),
            tables=eb_configs,
        )

    print("EmbeddingCollection:", ebc)
    planner = get_planner(
        table_names,
        eb_configs,
        use_dynamicembs,
        score_strategies,
        local_batch,
        multi_hot_sizes,
        device,
    )

    if is_pooled:
        sharder = DynamicEmbeddingBagCollectionSharder(fused_params=optimizer_kwargs)
    else:
        sharder = DynamicEmbeddingCollectionSharder(
            fused_params=optimizer_kwargs, use_index_dedup=False
        )

    plan = planner.collective_plan(ebc, [sharder], dist.GroupMember.WORLD)
    print("Plan:", plan)

    model = DistributedModelParallel(
        module=ebc,
        device=device,
        # pyre-ignore
        sharders=[sharder],
        plan=plan,
    )

    customized_scores = CustomizedScore(table_names)
    ret: Dict[str, Dict[str, int]] = get_score(model)
    prefix_path = "model"

    if ret is None:
        return
    else:
        assert len(ret) == 1 and prefix_path in ret
    undump_score: Dict[str, int] = ret[prefix_path]

    scores_to_set: Dict[str, int] = {}
    for i in range(table_num):
        if score_strategies[i] == DynamicEmbScoreStrategy.CUSTOMIZED:
            scores_to_set[table_names[i]] = customized_scores.get(table_names[i])
    all_customized = (
        True
        if score_strategies == [DynamicEmbScoreStrategy.CUSTOMIZED] * table_num
        else False
    )
    param_scores = (
        scores_to_set[table_names[0]]
        if all_customized
        else {prefix_path: scores_to_set}
    )
    set_score(model, param_scores)
    undump_score.update(scores_to_set)

    for i in range(0, num_iteration):
        unique_indices = [set({}) for _ in table_names]
        for j in range(dump_interval):
            scores_to_set: Dict[str, int] = {}
            for i in range(table_num):
                if score_strategies[i] == DynamicEmbScoreStrategy.CUSTOMIZED:
                    scores_to_set[table_names[i]] = customized_scores.get(
                        table_names[i]
                    )
            set_score(model, {prefix_path: scores_to_set})
            sparse_feature = generate_sparse_feature(
                feature_names,
                multi_hot_sizes,
                local_batch,
                unique_indices,
                use_dynamicembs,
                num_embeddings,
            )
            ret = model(sparse_feature)  # => this is awaitable
            kt = ret.values()  # wait

        print("Dump score=", undump_score)
        param_scores = (
            undump_score[table_names[0]]
            if all_customized
            else {prefix_path: undump_score}
        )
        ret_tensors, ret_scores = incremental_dump(
            model, param_scores, intra_and_cross_node_pg()[0]
        )
        undump_score = ret_scores[prefix_path]
        for i, (table_name, indices) in enumerate(zip(table_names, unique_indices)):
            if use_dynamicembs[i]:
                dump_keys = ret_tensors[prefix_path][table_name][0]
                dump_vals = ret_tensors[prefix_path][table_name][1]
                dumped_indices = set(dump_keys.tolist())
                assert indices.issubset(dumped_indices)

                ############### Test the dumped embeddings ###############
                dump_keys = dump_keys % 100000
                dump_vals = dump_vals.to(dump_keys.dtype)
                assert torch.all(dump_keys.unsqueeze(1).expand(-1, dim) == dump_vals)
