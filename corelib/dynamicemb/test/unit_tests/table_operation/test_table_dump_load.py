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
from dynamicemb_extensions import (
    InsertResult,
    ScorePolicy,
    device_timestamp,
    table_count_matched,
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
@pytest.mark.parametrize(
    "capacity_multipliers",
    [
        pytest.param([1], id="1table"),
        pytest.param([1, 2], id="2tables"),
        pytest.param([1, 2, 3, 4], id="4tables"),
        pytest.param([2, 0], id="2tables-zero-last"),
        pytest.param([0, 3], id="2tables-zero-first"),
        pytest.param([2, 0, 3, 0], id="4tables-mixed"),
    ],
)
def test_table_dump_load(
    key_type,
    num_buckets,
    bucket_capacity,
    batch_size,
    score_policy,
    capacity_multipliers,
    backend_session,
):
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    batch_size = batch_size * world_size

    num_tables = len(capacity_multipliers)
    capacity = [m * num_buckets * bucket_capacity for m in capacity_multipliers]

    table = get_scored_table(
        capacity=capacity,
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score1", policy=score_policy)],
        device=device,
    )

    max_step = 20
    inserted_batches = []
    offset = 0
    for step in range(max_step):
        keys = torch.randperm(batch_size, device=device, dtype=torch.int64) + offset

        masks = keys % world_size == local_rank
        keys = keys[masks]
        tid = torch.randint(num_tables, (keys.numel(),), device=device)
        keys = keys.to(key_type)
        batch_ = keys.numel()

        inserted_batches.append((keys.clone(), tid.clone()))

        score_arg = ScoreArg(name="score1", value=get_scores(score_policy, keys))

        insert_results = torch.empty(
            batch_, dtype=table.result_type, device=device
        ).fill_(InsertResult.INIT.value)

        table.insert(keys, tid, score_arg, insert_results)

        assert (
            (insert_results == InsertResult.INSERT.value)
            | (insert_results == InsertResult.EVICT.value)
            | (insert_results == InsertResult.ILLEGAL.value)
        ).all()

        offset += batch_size

    for t in range(num_tables):
        if capacity_multipliers[t] == 0:
            assert table.size(table_id=t) == 0

    key_files = []
    score_files = []
    for t in range(num_tables):
        kf = f"keys_rank{local_rank}_table{t}"
        sf = f"score1_rank{local_rank}_table{t}"
        shutil.rmtree(kf, ignore_errors=True)
        shutil.rmtree(sf, ignore_errors=True)
        table.dump(kf, {"score1": sf}, table_id=t)
        key_files.append(kf)
        score_files.append(sf)

    load_table = get_scored_table(
        capacity=capacity,
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score1", policy=score_policy)],
    )

    for t in range(num_tables):
        load_table.load(key_files[t], {"score1": score_files[t]}, table_id=t)

    for t in range(num_tables):
        assert table.size(table_id=t) == load_table.size(table_id=t)
    assert table.size() == load_table.size()

    num_total_keys = 0
    for keys, tid in inserted_batches:
        score_arg0 = ScoreArg(
            name="score1",
            value=get_scores(score_policy, keys),
            policy=ScorePolicy.CONST,
        )

        score_arg1 = ScoreArg(
            name="score1",
            value=get_scores(score_policy, keys),
            policy=ScorePolicy.CONST,
        )

        score_out0, founds0, _ = table.lookup(keys, tid, score_arg0)
        score_out1, founds1, _ = load_table.lookup(keys, tid, score_arg1)

        assert torch.equal(founds0, founds1)
        num_total_keys += founds0.sum()

        scores0 = score_out0[founds0]
        scores1 = score_out1[founds1]

        if table.score_specs[0].policy == ScorePolicy.GLOBAL_TIMER:
            scores_bias = scores1 - scores0
            found_tids = tid[founds0]
            for t in range(num_tables):
                mask = found_tids == t
                if mask.any():
                    table_bias = scores_bias[mask]
                    assert (table_bias == table_bias[0]).all()
        else:
            assert torch.equal(scores0, scores1)

    assert num_total_keys == load_table.size()


def table_num_matched(table, threshold) -> int:
    d_num_matched = table_count_matched(
        table.table_storage_,
        table.fileds_type_[0],
        table.bucket_capacity_,
        threshold,
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
@pytest.mark.parametrize(
    "capacity_multipliers",
    [
        pytest.param([1], id="1table"),
        pytest.param([1, 2], id="2tables"),
        pytest.param([2, 0], id="2tables-zero-last"),
        pytest.param([2, 0, 3, 0], id="4tables-mixed"),
    ],
)
def test_table_incremental_dump(
    request,
    bucket_capacity,
    batch_size,
    num_buckets,
    num_iteration,
    dump_interval,
    score_policy,
    capacity_multipliers,
    backend_session,
):
    print(f"\n{request.node.name}")

    assert torch.cuda.is_available()
    device = torch.cuda.current_device()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    global_batch = batch_size * world_size

    num_tables = len(capacity_multipliers)
    capacity = [m * num_buckets * bucket_capacity for m in capacity_multipliers]

    table = get_scored_table(
        capacity=capacity,
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
    score_arg = ScoreArg(name="score1", value=get_scores(score_policy, keys))
    insert_results = torch.empty(
        bucket_capacity, dtype=table.result_type, device=device
    ).fill_(InsertResult.INIT.value)

    tid = torch.randint(num_tables, (keys.numel(),), device=device)
    table.insert(keys, tid, score_arg, insert_results)

    num_illegal = (insert_results == InsertResult.ILLEGAL.value).sum().item()
    num_initial_inserted = bucket_capacity - num_illegal

    # check 1: count_matched works well
    assert table_num_matched(table, 0) == num_initial_inserted

    # check 2: table.incremental_dump is consistent with count_matched
    dumped_keys, dumped_named_scores, _ = table.incremental_dump({"score1": 0})
    assert dumped_keys.numel() == num_initial_inserted
    inserted_mask = insert_results == InsertResult.INSERT.value
    assert set(dumped_keys[(dumped_keys % world_size == local_rank)].tolist()) == set(
        keys[inserted_mask].tolist()
    )
    assert (insert_results == InsertResult.INSERT.value).sum() == num_initial_inserted

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
            dumped_keys, _, _ = table.incremental_dump({"score1": undumped_score})
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
            tid = torch.randint(num_tables, (keys.numel(),), device=device)

            offset += global_batch
            score_arg.value = get_scores(score_policy, keys)
            insert_results = torch.empty(
                batch_size, dtype=table.result_type, device=device
            ).fill_(InsertResult.INIT.value)

            old_size = table.size()
            _, num_evict, evict_keys, _, evict_named_scores, _ = table.insert_and_evict(
                keys, tid, score_arg, insert_results
            )
            new_size = table.size()

            num_inserted = (insert_results == InsertResult.INSERT.value).sum()
            num_reclaim = (insert_results == InsertResult.RECLAIM.value).sum()
            num_assign = (insert_results == InsertResult.ASSIGN.value).sum()
            num_inserted_by_eviction = (
                insert_results == InsertResult.EVICT.value
            ).sum()
            num_insert_failed = (insert_results == InsertResult.BUSY.value).sum()
            num_illegal = (insert_results == InsertResult.ILLEGAL.value).sum()

            assert num_assign == 0
            assert num_reclaim == 0

            assert new_size - old_size == num_inserted
            assert num_inserted_by_eviction + num_insert_failed == num_evict
            assert (
                num_inserted
                + num_inserted_by_eviction
                + num_insert_failed
                + num_illegal
                == batch_size
            )

            # evicted keys are not in the table anymore, including insert failed keys.
            evict_keys_interval_mask = (evict_keys - interval_offset) >= 0
            evict_keys_interval_mask = (
                evict_keys[evict_keys_interval_mask] - interval_offset
            )
            interval_existed_in_table[evict_keys_interval_mask] = False
            num_remain -= evict_keys_interval_mask.numel()

            # keys targeting zero-capacity tables were never inserted
            illegal_keys = keys[insert_results == InsertResult.ILLEGAL.value]
            illegal_interval_mask = (illegal_keys - interval_offset) >= 0
            illegal_in_interval = illegal_keys[illegal_interval_mask] - interval_offset
            interval_existed_in_table[illegal_in_interval] = False
            num_remain -= illegal_in_interval.numel()

        # check 3: incremental dump as expected
        keys, named_scores, _ = table.incremental_dump({"score1": undumped_score})
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
        dumped_keys, _, _ = table.incremental_dump({"score1": 0})
        assert (
            dumped_keys[dumped_keys % world_size == local_rank].numel() == table.size()
        )

        # log
        load_factor = table.size() / table.capacity()
        print(
            f"Rank {local_rank}: load factor={load_factor:.3f}, there are {interval_existed_in_table.sum()} keys existed in the table for the last interval {dump_interval*batch_size}"
        )
