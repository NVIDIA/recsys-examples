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
import shutil

import pytest
import torch
import torch.distributed as dist
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import (
    InsertResult,
    ScorePolicy,
    device_timestamp,
    table_count_matched,
)
from ordered_set import OrderedSet

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


def generate_files_for_accumulate(
    batch_size: int,
    # rank: int,
    # world_size: int,
    seed: int = 42,
):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    total_keys = OrderedSet()
    # select_keys = OrderedSet()
    while len(total_keys) < batch_size:
        x = random.randint(0, (1 << 63) - 1)
        total_keys.add(x)

        # if x % world_size == rank:
        #     select_keys.add(x)

    keys = torch.tensor(list(total_keys), dtype=torch.int64).cuda()
    scores = torch.ones_like(keys)
    return keys, scores


@pytest.mark.parametrize("key_type", [torch.int64])
@pytest.mark.parametrize("bucket_capacity", [128, 1024])
@pytest.mark.parametrize("num_buckets", [8192])
@pytest.mark.parametrize("batch_size", [128 * 4096])
@pytest.mark.parametrize(
    "score_policy",
    [ScorePolicy.ACCUMULATE],
)
def test_table_load(
    key_type,
    bucket_capacity,
    num_buckets,
    batch_size,
    score_policy,
    backend_session,
):
    print("--------------------------------------------------------")
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.cuda.current_device()

    table = get_scored_table(
        capacity=num_buckets * bucket_capacity,
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score1", policy=score_policy)],
    )

    score_args_lookup = [
        ScoreArg(
            name="score1",
            policy=ScorePolicy.CONST,
            is_return=True,
        )
    ]

    key_file = "debug_keys"
    score_file = "debug_scores"
    keys, scores = generate_files_for_accumulate(batch_size)

    if rank == 0:
        fkey = open(key_file, "wb")
        fscore = open(score_file, "wb")
        fkey.write(keys.cpu().numpy().tobytes())
        fscore.write(scores.cpu().numpy().tobytes())
        fkey.close()
        fscore.close()

    dist.barrier()

    table.load(key_file, {"score1": score_file})

    masks = keys % world_size == rank
    selected_keys = keys[masks]

    assert table.size() == selected_keys.numel()

    founds = torch.empty(selected_keys.numel(), dtype=torch.bool, device=device).fill_(
        False
    )
    score_args_lookup[0].value = torch.zeros(
        selected_keys.numel(), dtype=torch.uint64, device=device
    )

    table.lookup(selected_keys, score_args_lookup, founds)

    assert founds.sum() == selected_keys.numel()
    assert torch.equal(
        score_args_lookup[0].value, torch.ones_like(selected_keys).to(torch.uint64)
    )

    print(
        f"Table load passed when world size={world_size} and bucket capacity={bucket_capacity})"
    )


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
        device=device,
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


def table_num_matched(table, threshold) -> int:
    d_num_matched = torch.zeros(1, dtype=torch.int64, device=table.device)
    table_count_matched(
        table.table_storage_,
        table.fileds_type_,
        table.bucket_capacity_,
        [threshold],
        d_num_matched,
    )
    return d_num_matched.cpu().item()


@pytest.mark.parametrize(
    "bucket_capacity, batch_size, num_buckets, num_iteration, dump_interval",
    [
        pytest.param(
            128, 128, 4096, 8192, 1024, id="Never evict keys from current batch"
        ),
        pytest.param(128, 65536, 4096, 32, 8),
        pytest.param(
            128, 1024 + 13, 8, 12, 3, id="Always evict keys from current batch"
        ),
        pytest.param(
            128, 1024, 32, 32, 4, id="Always evict keys from last dump_interval"
        ),
        pytest.param(512, 512, 1024, 2048, 256, id="Different bucket capacity"),
    ],
)
@pytest.mark.parametrize(
    "score_policy",
    [ScorePolicy.ASSIGN, ScorePolicy.ACCUMULATE, ScorePolicy.GLOBAL_TIMER],
)
def test_table_incremental_dump(
    request,
    bucket_capacity,
    batch_size,
    num_buckets,
    num_iteration,
    dump_interval,
    score_policy,
    backend_session,
):
    print(f"\n{request.node.name}")

    assert torch.cuda.is_available()
    device = torch.cuda.current_device()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    global_batch = batch_size * world_size

    table = get_scored_table(
        capacity=num_buckets * bucket_capacity,
        bucket_capacity=bucket_capacity,
        key_type=torch.int64,
        score_specs=[ScoreSpec(name="score1", policy=score_policy)],
        device=device,
    )

    # insert once: the first insert will insert all keys, and no eviction(batch = bucket_capacity)
    keys = torch.randperm(
        bucket_capacity * world_size, device=device, dtype=torch.int64
    )
    masks = keys % world_size == local_rank
    keys = keys[masks]
    score_args = [
        ScoreArg(name="score1", value=get_scores(score_policy, keys), is_return=True)
    ]
    insert_results = torch.empty(
        bucket_capacity, dtype=table.result_type, device=device
    ).fill_(InsertResult.INIT.value)

    table.insert(keys, score_args, None, insert_results)

    # check 1: count_matched works well
    assert table_num_matched(table, 0) == bucket_capacity

    # check 2: table.incremental_dump is consistent with count_matched
    dumped_keys, dumped_named_scores = table.incremental_dump({"score1": 0})
    assert dumped_keys.numel() == bucket_capacity * world_size
    assert set(dumped_keys[(dumped_keys % world_size == local_rank)].tolist()) == set(
        keys.tolist()
    )
    assert (insert_results == InsertResult.INSERT.value).sum() == bucket_capacity

    # insert iteratively
    offset = bucket_capacity * world_size

    for i in range(0, num_iteration, dump_interval):
        if score_policy == ScorePolicy.GLOBAL_TIMER:
            undumped_score = device_timestamp()
        elif score_policy == ScorePolicy.ASSIGN:
            global score_step
            undumped_score = score_step + 1
        else:
            undumped_score = 1

        # pre-check: using uninsert score will dump nothing.
        if score_policy != ScorePolicy.ACCUMULATE:
            dumped_keys, _ = table.incremental_dump({"score1": undumped_score})
            assert dumped_keys.numel() == 0

        interval_keys = torch.arange(
            offset,
            offset + dump_interval * global_batch,
            device=device,
            dtype=torch.int64,
        )
        interval_existed_in_table = torch.empty(
            dump_interval * global_batch, device=device, dtype=torch.bool
        ).fill_(False)

        # select local keys
        interval_existed_in_table[local_rank::world_size] = True
        interval_offset = offset
        assert (
            interval_offset % world_size == 0
        ), "Global keys and mask assumption mismatched."

        num_remain = dump_interval * batch_size

        for j in range(dump_interval):
            keys = (
                torch.randperm(global_batch, device=device, dtype=torch.int64) + offset
            )
            masks = keys % world_size == local_rank
            keys = keys[masks]

            offset += global_batch
            score_args[0].value = get_scores(score_policy, keys)
            insert_results = torch.empty(
                batch_size, dtype=table.result_type, device=device
            ).fill_(InsertResult.INIT.value)

            old_size = table.size()
            num_evict, evict_keys, _, evict_named_scores = table.insert_and_evict(
                keys, score_args, None, insert_results
            )
            new_size = table.size()

            num_inserted = (insert_results == InsertResult.INSERT.value).sum()
            num_reclaim = (insert_results == InsertResult.RECLAIM.value).sum()
            num_assign = (insert_results == InsertResult.ASSIGN.value).sum()
            num_inserted_by_eviction = (
                insert_results == InsertResult.EVICT.value
            ).sum()
            num_insert_failed = (insert_results == InsertResult.BUSY.value).sum()

            assert num_assign == 0
            assert num_reclaim == 0

            assert new_size - old_size == num_inserted
            assert num_inserted_by_eviction + num_insert_failed == num_evict
            assert (
                num_inserted + num_inserted_by_eviction + num_insert_failed
                == batch_size
            )

            # evicted keys are not in the table anymore, including insert failed keys.
            evict_keys_interval_mask = (evict_keys - interval_offset) >= 0
            evict_keys_interval_mask = (
                evict_keys[evict_keys_interval_mask] - interval_offset
            )
            interval_existed_in_table[evict_keys_interval_mask] = False

            num_remain -= evict_keys_interval_mask.numel()

        # check 3: incremental dump as expected
        keys, named_scores = table.incremental_dump({"score1": undumped_score})
        masks = keys % world_size == local_rank
        keys = keys.to(device)[masks]
        scores = named_scores["score1"].to(device)[masks]
        if score_policy != ScorePolicy.ACCUMULATE:
            assert ((scores - undumped_score) >= 0).all()
            assert keys.numel() == num_remain
            ascend_dumped_keys, _ = torch.sort(keys)
            ascend_interval_keys, _ = torch.sort(
                interval_keys[interval_existed_in_table]
            )
            assert torch.equal(ascend_dumped_keys, ascend_interval_keys)
        else:
            assert keys.numel() == table.size()
            assert torch.isin(interval_keys[interval_existed_in_table], keys).all()

        # check 4: using min_score to count will get table's size
        dumped_keys, _ = table.incremental_dump({"score1": 0})
        assert (
            dumped_keys[dumped_keys % world_size == local_rank].numel() == table.size()
        )

        # log
        load_factor = table.size() / table.capacity()
        print(
            f"Rank {local_rank}: load factor={load_factor:.3f}, there are {interval_existed_in_table.sum()} keys existed in the table for the last interval {dump_interval*batch_size}"
        )
