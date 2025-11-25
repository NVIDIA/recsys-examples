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
import shutil

import pytest
import torch
import torch.distributed as dist
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import InsertResult, ScorePolicy

score_step = 0


def get_scores(score_policy, keys):
    batch = keys.numel()
    device = keys.device

    global score_step

    score_step += 1

    if score_policy == ScorePolicy.ASSIGN:
        return torch.empty(batch, dtype=torch.uint64, device=device).fill_(score_step)
    elif score_policy == ScorePolicy.ACCUMULATE:
        return torch.ones(batch, dtype=torch.uint64, device=device)
    else:
        return torch.zeros(batch, dtype=torch.uint64, device=device)


@pytest.fixture(scope="session")
def backend_session():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.barrier()

    yield

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("key_type", [torch.int64, torch.uint64])
@pytest.mark.parametrize("bucket_capacity", [128])
@pytest.mark.parametrize("num_buckets", [2047, 8192])
@pytest.mark.parametrize("batch_size", [128, 65536])
@pytest.mark.parametrize(
    "score_policy",
    [ScorePolicy.ASSIGN, ScorePolicy.ACCUMULATE, ScorePolicy.GLOBAL_TIMER],
)
def test_table_dump_load(
    key_type,
    num_buckets,
    bucket_capacity,
    batch_size,
    score_policy,
    backend_session,
):
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    batch_size = batch_size * world_size

    table = get_scored_table(
        capacity=num_buckets * bucket_capacity,
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score1", policy=score_policy)],
    )

    offset = 0
    max_step = 20
    step = 0
    while step < max_step:
        keys = torch.randperm(batch_size, device=device, dtype=torch.int64) + offset

        masks = keys % world_size == local_rank
        keys = keys[masks]
        keys = keys.to(key_type)
        batch_ = keys.numel()

        score_args = [
            ScoreArg(
                name="score1", value=get_scores(score_policy, keys), is_return=True
            )
        ]

        insert_results = torch.empty(
            batch_, dtype=table.result_type, device=device
        ).fill_(InsertResult.INIT.value)
        indices = torch.zeros(batch_, dtype=table.index_type, device=device)

        table.insert(keys, score_args, indices, insert_results)

        # not assign or busy
        assert (
            (insert_results == InsertResult.INSERT.value)
            | (insert_results == InsertResult.EVICT.value)
        ).all()

        offset += batch_size
        step += 1

    key_file = f"keys_rank{local_rank}"
    score_file = f"score1_rank{local_rank}"

    shutil.rmtree(key_file, ignore_errors=True)
    shutil.rmtree(score_file, ignore_errors=True)

    table.dump(key_file, {"score1": score_file})

    load_table = get_scored_table(
        capacity=num_buckets * bucket_capacity,
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score1", policy=score_policy)],
    )

    load_table.load(key_file, {"score1": score_file})

    assert table.size() == load_table.size()

    offset = 0
    max_step = 20
    step = 0

    num_total_keys = 0
    while step < max_step:
        keys = torch.arange(0, batch_size, 1, device=device, dtype=torch.int64) + offset

        masks = keys % world_size == local_rank
        keys = keys[masks]
        keys = keys.to(key_type)
        batch_ = keys.numel()

        score_args0 = [
            ScoreArg(
                name="score1",
                value=get_scores(score_policy, keys),
                policy=ScorePolicy.CONST,
                is_return=True,
            )
        ]

        score_args1 = [
            ScoreArg(
                name="score1",
                value=get_scores(score_policy, keys),
                policy=ScorePolicy.CONST,
                is_return=True,
            )
        ]

        founds0 = torch.empty(batch_, dtype=torch.bool, device=device)
        founds1 = torch.empty(batch_, dtype=torch.bool, device=device)

        table.lookup(keys, score_args0, founds0, None)

        load_table.lookup(keys, score_args1, founds1)

        assert torch.equal(founds0, founds1)
        num_total_keys += founds0.sum()

        scores0 = score_args0[0].value.to(torch.int64)[founds0]
        scores1 = score_args1[0].value.to(torch.int64)[founds1]

        if table.score_specs[0].policy == ScorePolicy.GLOBAL_TIMER:
            # same machine
            scores_bias = scores1 - scores0
            if scores_bias.numel() > 0:
                assert (scores_bias == scores_bias[0]).all()
        else:
            assert torch.equal(scores0, scores1)

        offset += batch_size
        step += 1

    assert num_total_keys == load_table.size()
