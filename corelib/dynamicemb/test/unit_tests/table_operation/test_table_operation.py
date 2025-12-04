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
from typing import List

import pytest
import torch
import torch.distributed as dist
import torchrec
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import InsertResult, ScorePolicy, table_partition


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


def random_indices(batch, min_index, max_index):
    result = set({})
    while len(result) < batch:
        result.add(random.randint(min_index, max_index))
    return result


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
    yield
    # dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("key_type", [torch.int64, torch.uint64])
@pytest.mark.parametrize("digest_type", [torch.uint8])
@pytest.mark.parametrize("score_type", [torch.uint64])
@pytest.mark.parametrize("bucket_capacity", [128, 1024])
@pytest.mark.parametrize("num_buckets", [1, 13, 1024])
def test_table_partition(
    key_type,
    digest_type,
    score_type,
    bucket_capacity,
    num_buckets,
):
    print("--------------------------------------------------------")
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()

    dtypes = [key_type, digest_type, score_type]
    dtypes_byte = [dtype.itemsize for dtype in dtypes]
    storage = torch.empty(
        sum(dtypes_byte) * bucket_capacity * num_buckets,
        dtype=torch.uint8,
        device=device,
    )

    keys, digests, scores = table_partition(
        storage,
        dtypes,
        bucket_capacity,
        num_buckets,
    )

    # dtype
    assert keys.dtype == key_type
    assert digests.dtype == digest_type
    assert scores.dtype == score_type

    # size
    assert keys.size() == (num_buckets, bucket_capacity)
    assert digests.size() == (num_buckets, bucket_capacity)
    assert scores.size() == (num_buckets, bucket_capacity)

    # stride
    bucket_bytes = sum(dtypes_byte) * bucket_capacity
    assert keys.stride() == (bucket_bytes // key_type.itemsize, 1)
    assert digests.stride() == (bucket_bytes // digest_type.itemsize, 1)
    assert scores.stride() == (bucket_bytes // score_type.itemsize, 1)

    # no overlap
    ascend_keys = (
        torch.arange(0, num_buckets * bucket_capacity, dtype=torch.int64, device=device)
        .view(num_buckets, bucket_capacity)
        .to(key_type)
    )
    zero_digests = torch.zeros(
        num_buckets * bucket_capacity, dtype=digest_type, device=device
    ).view(num_buckets, bucket_capacity)
    descend_scores = (
        torch.arange(
            num_buckets * bucket_capacity - 1, -1, -1, dtype=torch.int64, device=device
        )
        .view(num_buckets, bucket_capacity)
        .to(score_type)
    )
    keys[:] = ascend_keys
    digests[:] = zero_digests
    scores[:] = descend_scores
    assert torch.equal(keys, ascend_keys)
    assert torch.equal(digests, zero_digests)
    assert torch.equal(scores, descend_scores)

    table = get_scored_table(
        capacity=num_buckets * bucket_capacity - 1,  # corner case
        bucket_capacity=bucket_capacity - 1,  # corner case
        key_type=key_type,
        score_specs=[ScoreSpec(name="score1", policy=ScorePolicy.CONST)],
    )

    assert table.capacity() == num_buckets * bucket_capacity
    assert table.key_type == key_type
    assert len(table.score_specs) == 1

    print(
        "Table partition passed: table capacity and bucket capacity rounded as expected."
    )
    print("Table partition passed: sizes, strides and dtype all matched.")
    print(
        "Table partition passed: there was no overlap across keys, digests and scores in memory address."
    )


@pytest.mark.parametrize("key_type", [torch.int64, torch.uint64])
@pytest.mark.parametrize("bucket_capacity", [128, 1024])
@pytest.mark.parametrize("num_buckets", [13, 512])
@pytest.mark.parametrize("batch_size", [1, 32, 128])
@pytest.mark.parametrize(
    "score_policy",
    [ScorePolicy.ASSIGN, ScorePolicy.ACCUMULATE, ScorePolicy.GLOBAL_TIMER],
)
def test_table_basic(
    key_type,
    num_buckets,
    bucket_capacity,
    batch_size,
    score_policy,
):
    print("--------------------------------------------------------")
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()

    table = get_scored_table(
        capacity=num_buckets * bucket_capacity,
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score1", policy=score_policy)],
    )

    keys = torch.randperm(batch_size, device=device, dtype=torch.int64).to(key_type)

    score_args = [
        ScoreArg(name="score1", value=get_scores(score_policy, keys), is_return=True)
    ]
    score_copy_0 = score_args[0].value.clone()
    insert_results = torch.empty(batch_size, dtype=table.result_type, device=device)
    indices = torch.empty(batch_size, dtype=table.index_type, device=device)

    table.insert(keys, score_args, indices, insert_results)

    assert insert_results.eq(InsertResult.INSERT.value).all()

    score_args_reinsert = [
        ScoreArg(name="score1", value=get_scores(score_policy, keys), is_return=True)
    ]
    score_copy_1 = score_args_reinsert[0].value.clone()
    insert_results = torch.zeros(batch_size, dtype=table.result_type, device=device)
    indices_reinsert = torch.zeros(batch_size, dtype=table.index_type, device=device)

    table.insert(keys, score_args_reinsert, indices_reinsert, insert_results)

    assert insert_results.eq(InsertResult.ASSIGN.value).all()
    assert torch.equal(indices, indices_reinsert)

    score_args_lookup = [
        ScoreArg(
            name="score1",
            value=get_scores(score_policy, keys),
            policy=ScorePolicy.CONST,
            is_return=True,
        )
    ]
    founds = torch.empty(batch_size, dtype=torch.bool, device=device).fill_(False)
    indices_lookup = torch.empty(
        batch_size, dtype=table.index_type, device=device
    ).fill_(-1)

    table.lookup(keys, score_args_lookup, founds, indices_lookup)

    assert founds.all()
    assert torch.equal(indices_lookup, indices)

    if table.score_specs[0].policy == ScorePolicy.ASSIGN:
        assert torch.equal(score_args_lookup[0].value, score_args_reinsert[0].value)
    elif table.score_specs[0].policy == ScorePolicy.ACCUMULATE:
        assert torch.equal(
            score_args_lookup[0].value.to(torch.int64),
            score_copy_0.to(torch.int64) + score_copy_1.to(torch.int64),
        )
    else:
        assert torch.equal(score_args_lookup[0].value, score_args_reinsert[0].value)
        assert (
            score_args[0].value.to(torch.int64)
            < score_args_reinsert[0].value.to(torch.int64)
        ).all()

    table.erase(keys)
    table.lookup(keys, score_args_lookup, founds, indices_lookup)
    assert not founds.any()

    max_num_reclaim = keys.numel()
    accum_num_reclaim = 0

    print(
        "Basic table operation(insert, lookup, erase) passed during the filling stage."
    )

    offset = batch_size
    max_step = 20
    step = 1
    while table.size() < table.capacity() and step < max_step:
        keys = (
            torch.randperm(bucket_capacity, device=device, dtype=torch.int64) + offset
        )
        keys = keys.to(key_type)

        score_args = [
            ScoreArg(
                name="score1", value=get_scores(score_policy, keys), is_return=True
            )
        ]

        insert_results = torch.empty(
            bucket_capacity, dtype=table.result_type, device=device
        ).fill_(InsertResult.INIT.value)
        indices = torch.zeros(bucket_capacity, dtype=table.index_type, device=device)

        table.insert(keys, score_args, indices, insert_results)

        num_inserted = (insert_results == InsertResult.INSERT.value).sum()
        num_reclaimed = (insert_results == InsertResult.RECLAIM.value).sum()
        num_eviction = (insert_results == InsertResult.EVICT.value).sum()
        num_assign = (insert_results == InsertResult.ASSIGN.value).sum()

        assert keys.numel() == num_inserted + num_reclaimed + num_eviction
        assert num_assign == 0

        accum_num_reclaim += num_reclaimed

        print(
            f"Table insert passed when load factor({table.load_factor():.3f}) with : insert({num_inserted}), reclaim({num_reclaimed}), evict({num_eviction})"
        )

        offset += bucket_capacity
        step += 1

    if table.size() == table.capacity():
        assert (
            accum_num_reclaim == max_num_reclaim
        ), f"Occupyied({accum_num_reclaim}/{max_num_reclaim}) reclaimed slots when table is full."

        keys = torch.randperm(batch_size, device=device, dtype=torch.int64) + offset
        keys = keys.to(key_type)

        score_args = [
            ScoreArg(
                name="score1", value=get_scores(score_policy, keys), is_return=False
            )
        ]

        insert_results = torch.empty(
            batch_size, dtype=table.result_type, device=device
        ).fill_(InsertResult.INIT.value)
        indices = torch.zeros(batch_size, dtype=table.index_type, device=device)

        table.insert(keys, score_args, indices, insert_results)

        # only eviction
        assert (insert_results == InsertResult.EVICT.value).all()

        founds.fill_(True)
        table.erase(keys)
        table.lookup(keys, score_args, founds, indices_lookup)
        assert not founds.any()

        indices_reinsert = torch.empty(
            batch_size, dtype=table.index_type, device=device
        ).fill_(-1)
        table.insert(keys, score_args, indices_reinsert, insert_results)

        assert (insert_results == InsertResult.RECLAIM.value).all()

        assert torch.equal(
            torch.sort(indices).values, torch.sort(indices_reinsert).values
        )

        print("Table operation(insert, erase, lookup) passed when table is full.")


@pytest.mark.parametrize("key_type", [torch.int64])
@pytest.mark.parametrize("bucket_capacity", [128, 1024])
@pytest.mark.parametrize("num_buckets", [8192])
@pytest.mark.parametrize("batch_size", [65536, 1048576])
@pytest.mark.parametrize(
    "score_policy",
    [ScorePolicy.ASSIGN, ScorePolicy.ACCUMULATE, ScorePolicy.GLOBAL_TIMER],
)
def test_table_evict(
    key_type,
    num_buckets,
    bucket_capacity,
    batch_size,
    score_policy,
):
    print("--------------------------------------------------------")
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()

    table = get_scored_table(
        capacity=num_buckets * bucket_capacity,
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score1", policy=score_policy)],
    )

    score_args = [ScoreArg(name="score1", is_return=True)]
    score_args_lookup = [
        ScoreArg(
            name="score1",
            policy=ScorePolicy.CONST,
            is_return=True,
        )
    ]

    offset = 0

    while table.size() < table.capacity():
        keys = torch.randperm(batch_size, device=device, dtype=torch.int64) + offset
        offset += batch_size
        keys = keys.to(key_type)

        score_args[0].value = get_scores(score_policy, keys)
        score_args_lookup[0].value = torch.zeros(
            batch_size, dtype=torch.uint64, device=device
        )

        insert_results = torch.empty(
            batch_size, dtype=table.result_type, device=device
        ).fill_(InsertResult.INIT.value)

        indices = torch.zeros(batch_size, dtype=table.index_type, device=device)

        (
            num_evicted,
            evicted_keys,
            evicted_indices,
            evicted_scores,
        ) = table.insert_and_evict(keys, score_args, indices, insert_results)
        evicted_scores = evicted_scores[0]

        founds = torch.empty(batch_size, dtype=torch.bool, device=device).fill_(False)
        indices_lookup = torch.empty(
            batch_size, dtype=table.index_type, device=device
        ).fill_(-1)

        table.lookup(keys, score_args_lookup, founds, indices_lookup)

        num_existed = founds.sum()

        num_inserted = (insert_results == InsertResult.INSERT.value).sum()
        num_reclaim = (insert_results == InsertResult.RECLAIM.value).sum()
        num_assign = (insert_results == InsertResult.ASSIGN.value).sum()
        num_inserted_by_eviction = (insert_results == InsertResult.EVICT.value).sum()
        num_insert_failed = (insert_results == InsertResult.BUSY.value).sum()

        assert (
            num_reclaim == 0
        ), f"There is no erase operation, but got {num_reclaim} reclaimed slots when insert."
        assert (
            num_assign == 0
        ), f"There is no duplicated keys, but got {num_assign} duplicated keys when insert."

        assert batch_size == num_inserted + num_inserted_by_eviction + num_insert_failed
        assert num_existed == num_inserted + num_inserted_by_eviction
        assert num_evicted == num_inserted_by_eviction + num_insert_failed

        assert torch.equal(indices, indices_lookup)

        if table.score_specs[0].policy == ScorePolicy.ASSIGN:
            assert torch.equal(
                score_args_lookup[0].value.to(torch.int64)[founds],
                score_args[0].value.to(torch.int64)[founds],
            )
            global score_step
            assert (
                score_args_lookup[0].value.to(torch.int64)[founds] == score_step
            ).all()
        elif table.score_specs[0].policy == ScorePolicy.ACCUMULATE:
            assert (score_args_lookup[0].value.to(torch.int64)[founds] == 1).all()
        else:
            assert torch.equal(
                score_args_lookup[0].value.to(torch.int64)[founds],
                score_args[0].value.to(torch.int64)[founds],
            )

        print(
            f"Table insert_and_evict passed when load factor:({table.load_factor():.3f}) with: insert({num_inserted}), evict({num_inserted_by_eviction}), failed({num_insert_failed})"
        )
