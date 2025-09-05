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

import random
from typing import Dict, List

import pytest
import torch
from dynamicemb import (
    BATCH_SIZE_PER_DUMP,
    DynamicEmbCheckMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


class CustomizedScore:
    def __init__(self, table_names: List[int]):
        self.table_names_ = table_names
        self.step_ = 1

    def get(self):
        ret: Dict[str, int] = {}
        for table_name in self.table_names_:
            ret[table_name] = self.step_

        self.step_ += 1
        return ret


def random_indices(batch, min_index, max_index):
    result = set({})
    while len(result) < batch:
        result.add(random.randint(min_index, max_index))
    return result


def generate_sparse_feature(feature_num, batch, multi_hot_size):
    indices = []
    lengths = []

    for i in range(feature_num * batch):
        cur_bag_size = random.randint(0, multi_hot_size)
        cur_bag = set({})
        while len(cur_bag) < cur_bag_size:
            index = random.randint(0, (1 << 63) - 1)
            cur_bag.add(index)

        indices.extend(list(cur_bag))
        lengths.append(cur_bag_size)

    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)

    return indices, lengths, offsets


@pytest.mark.parametrize(
    "score_strategy",
    [
        DynamicEmbScoreStrategy.TIMESTAMP,
        DynamicEmbScoreStrategy.STEP,
        DynamicEmbScoreStrategy.CUSTOMIZED,
    ],
)
@pytest.mark.parametrize(
    "pooling_mode", [DynamicEmbPoolingMode.SUM, DynamicEmbPoolingMode.NONE]
)
@pytest.mark.parametrize(
    "table_num, features_per_table, num_embeddings, dim",
    [
        pytest.param(
            1,
            [1],
            [BATCH_SIZE_PER_DUMP * 8],
            8,  # large capacity to test dump in loops.
            id="No eviction",
        ),
    ],
)
@pytest.mark.parametrize("bucket_capacity, local_batch", [(128, 128)])
@pytest.mark.parametrize("num_iteration", [12])  # [1024])
@pytest.mark.parametrize("dump_interval", [3])  # ][64])
@pytest.mark.parametrize("fixed_hot_size", [1])
@pytest.mark.parametrize("caching", [True, False])
def test_without_eviction(
    request,
    current_device,
    score_strategy,
    pooling_mode,
    table_num,
    features_per_table,
    num_embeddings,
    dim,
    bucket_capacity,
    local_batch,
    num_iteration,
    dump_interval,
    fixed_hot_size,
    caching,
):
    print(f"\n{request.node.name}")
    options_list = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=current_device,
            dim=dim,
            max_capacity=num_embeddings[i],
            bucket_capacity=bucket_capacity,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=score_strategy,
            num_of_buckets_per_alloc=num_embeddings[i] // bucket_capacity,
            caching=caching,
        )
        for i in range(table_num)
    ]

    table_names = [f"t_{i}" for i in range(table_num)]

    model = BatchedDynamicEmbeddingTablesV2(
        table_options=options_list,
        output_dtype=torch.float32,
        table_names=table_names,
        feature_table_map=[
            i
            for i, feature_num in enumerate(features_per_table)
            for _ in range(feature_num)
        ],
        pooling_mode=pooling_mode,
        use_index_dedup=False,
    )

    device = torch.device(f"cuda:{current_device}")

    def _generate_sparse_feature(batch, unique_indices_list, device):
        ret_indices = []
        ret_lengths = [1] * (len(unique_indices_list) * batch)
        for i, unique_indices in enumerate(unique_indices_list):
            for j in range(batch):
                index = random.randint(0, (1 << 63) - 1)
                ret_indices.append(index)
                unique_indices.add(index)

        offsets = [0]
        for length in ret_lengths:
            offsets.append(offsets[-1] + length)

        indices_ = torch.tensor(ret_indices, dtype=torch.int64, device=device)
        offsets_ = torch.tensor(offsets, dtype=torch.int64, device=device)
        return indices_, offsets_

    if score_strategy != DynamicEmbScoreStrategy.CUSTOMIZED:
        undump_score = model.get_score()
        print("Init score=", undump_score)
        for i in range(0, num_iteration, dump_interval):
            unique_indices = [set({}) for _ in table_names]
            for j in range(dump_interval):
                indices, offsets = _generate_sparse_feature(
                    local_batch, unique_indices, device
                )
                print(f"Iteration({i + j})=", model.get_score())
                model(indices, offsets)

            print("Dump score=", undump_score)
            ret_tensors, undump_score = model.incremental_dump(undump_score)
            for table_name, indices in zip(table_names, unique_indices):
                dumped_indices = set(ret_tensors[table_name][0].tolist())
                # must match
                assert len(dumped_indices) == len(indices)
                assert dumped_indices == indices
    else:
        customized_score = CustomizedScore(table_names)
        undump_score = customized_score.get()
        model.set_score(undump_score)
        print("Init score=", undump_score)
        for i in range(0, num_iteration, dump_interval):
            unique_indices = [set({}) for _ in table_names]
            for j in range(dump_interval):
                indices, offsets = _generate_sparse_feature(
                    local_batch, unique_indices, device
                )
                print(f"Iteration({i + j})=", model.get_score())
                model(indices, offsets)
                model.set_score(customized_score.get())

            print("Dump score=", undump_score)
            ret_tensors, undump_score = model.incremental_dump(undump_score)
            for table_name, indices in zip(table_names, unique_indices):
                dumped_indices = set(ret_tensors[table_name][0].tolist())
                # must match
                assert len(dumped_indices) == len(indices)
                assert dumped_indices == indices
