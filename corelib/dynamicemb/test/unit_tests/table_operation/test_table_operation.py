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
from dynamicemb_extensions import (
    InsertResult,
    ScorePolicy,
    bucketize_keys,
    table_partition,
)


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
        capacity=[num_buckets * bucket_capacity - 1],  # corner case
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
@pytest.mark.parametrize("batch_size", [1, 32, 128, 1024])
@pytest.mark.parametrize(
    "capacity_multipliers",
    [
        pytest.param([1], id="1table"),
        pytest.param([1, 2], id="2tables"),
        pytest.param([2, 0, 3], id="3tables-with-zero"),
        pytest.param([1, 2, 3, 4], id="4tables"),
    ],
)
def test_bucketize_keys(
    key_type,
    bucket_capacity,
    batch_size,
    capacity_multipliers,
):
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()

    num_tables = len(capacity_multipliers)
    num_buckets_per_table = [m * 13 for m in capacity_multipliers]
    table_bucket_offsets = torch.tensor(
        [0] + [sum(num_buckets_per_table[: i + 1]) for i in range(num_tables)],
        dtype=torch.int64,
        device=device,
    )
    keys = torch.randint(
        1, batch_size * 10, (batch_size,), device=device, dtype=torch.int64
    ).to(key_type)
    table_ids = torch.randint(
        num_tables, (batch_size,), dtype=torch.int64, device=device
    )

    bkt_keys, offsets, inverse = bucketize_keys(
        keys, table_ids, table_bucket_offsets, bucket_capacity
    )

    assert bkt_keys.numel() == batch_size
    assert inverse.numel() == batch_size

    # inverse should be a valid permutation: bkt_keys == keys[inverse]
    assert torch.equal(bkt_keys.to(torch.int64), keys.to(torch.int64)[inverse])

    # offsets should be monotonically non-decreasing
    assert (offsets[1:] >= offsets[:-1]).all()

    # The last offset value should equal the batch size
    assert offsets[-1].item() == batch_size

    # offsets defines contiguous segments; verify segment sizes are positive
    num_active_buckets = offsets.numel() - 1
    for b in range(num_active_buckets):
        start = offsets[b].item()
        end = offsets[b + 1].item()
        assert end > start, f"Active bucket {b} has empty segment [{start}, {end})"

    # Randomly permute input and verify the sorted output is identical
    perm = torch.randperm(batch_size, device=device)
    perm_keys = keys.to(torch.int64)[perm].to(key_type)
    perm_table_ids = table_ids[perm]

    bkt_keys2, offsets2, inverse2 = bucketize_keys(
        perm_keys, perm_table_ids, table_bucket_offsets, bucket_capacity
    )

    # The sorted keys and offsets should be the same regardless of input order
    assert torch.equal(bkt_keys.to(torch.int64), bkt_keys2.to(torch.int64))
    assert torch.equal(offsets, offsets2)

    # inverse2 should map from sorted positions back into the permuted input
    assert torch.equal(bkt_keys2.to(torch.int64), perm_keys.to(torch.int64)[inverse2])


@pytest.mark.parametrize("key_type", [torch.int64, torch.uint64])
@pytest.mark.parametrize("bucket_capacity", [128, 1024])
@pytest.mark.parametrize("num_buckets", [13, 512])
@pytest.mark.parametrize("batch_size", [1, 32, 128])
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
def test_table_basic(
    key_type,
    num_buckets,
    bucket_capacity,
    batch_size,
    score_policy,
    capacity_multipliers,
):
    print("--------------------------------------------------------")
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()

    num_tables = len(capacity_multipliers)
    capacity = [m * num_buckets * bucket_capacity for m in capacity_multipliers]

    table = get_scored_table(
        capacity=capacity,
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score1", policy=score_policy)],
    )

    # Verify construction metadata
    assert table.num_tables_ == num_tables
    for t in range(num_tables):
        if capacity_multipliers[t] == 0:
            assert table.capacity(table_id=t) == 0

    keys_per_table = batch_size // num_tables
    if keys_per_table > 0 and num_tables > 1:
        base_keys = torch.randperm(keys_per_table, device=device, dtype=torch.int64)
        keys = base_keys.repeat(num_tables)
        table_ids = torch.arange(
            num_tables, device=device, dtype=torch.int64
        ).repeat_interleave(keys_per_table)
        perm = torch.randperm(keys.numel(), device=device)
        keys = keys[perm].to(key_type)
        table_ids = table_ids[perm]
    else:
        keys = torch.randperm(batch_size, device=device, dtype=torch.int64).to(key_type)
        table_ids = torch.randint(
            num_tables, (batch_size,), dtype=torch.int64, device=device
        )

    score_arg = ScoreArg(name="score1", value=get_scores(score_policy, keys))
    score_copy_0 = score_arg.value.clone()
    insert_results = torch.empty(batch_size, dtype=table.result_type, device=device)

    indices = table.insert(keys, table_ids, score_arg, insert_results)

    assert (
        (insert_results == InsertResult.INSERT.value)
        | (insert_results == InsertResult.ILLEGAL.value)
    ).all()

    valid_mask = insert_results != InsertResult.ILLEGAL.value
    has_zero_capacity = any(m == 0 for m in capacity_multipliers)
    if not has_zero_capacity:
        assert valid_mask.sum().item() == batch_size

    # Re-insert same keys with same table_ids
    score_arg_reinsert = ScoreArg(name="score1", value=get_scores(score_policy, keys))
    score_copy_1 = score_arg_reinsert.value.clone()
    insert_results = torch.zeros(batch_size, dtype=table.result_type, device=device)

    scores_reinsert = torch.empty(keys.numel(), dtype=torch.int64, device=keys.device)
    indices_reinsert = table.insert(
        keys, table_ids, score_arg_reinsert, insert_results, score_out=scores_reinsert
    )

    assert (
        (insert_results == InsertResult.ASSIGN.value)
        | (insert_results == InsertResult.ILLEGAL.value)
    ).all()
    assert torch.equal(indices[valid_mask], indices_reinsert[valid_mask])

    score_arg_lookup = ScoreArg(
        name="score1",
        value=get_scores(score_policy, keys),
        policy=ScorePolicy.CONST,
    )
    score_out, founds, indices_lookup = table.lookup(keys, table_ids, score_arg_lookup)

    assert founds[valid_mask].all()
    if (~valid_mask).any():
        assert not founds[~valid_mask].any()
    assert torch.equal(indices_lookup[valid_mask], indices[valid_mask])

    if table.score_specs[0].policy == ScorePolicy.ASSIGN:
        assert torch.equal(
            score_out[valid_mask],
            score_arg_reinsert.value.to(torch.int64)[valid_mask],
        )
    elif table.score_specs[0].policy == ScorePolicy.ACCUMULATE:
        assert torch.equal(
            score_out[valid_mask],
            (
                score_copy_0.to(torch.int64)[valid_mask]
                + score_copy_1.to(torch.int64)[valid_mask]
            ),
        )
    else:
        assert torch.equal(score_out[valid_mask], scores_reinsert[valid_mask])
        assert (
            score_arg.value.to(torch.int64)[valid_mask] < scores_reinsert[valid_mask]
        ).all()

    # Verify cross-table isolation: same key in different tables is found independently.
    # With local indices, the same key in different tables may map to the same
    # local index, so we only check that lookup in the wrong table still finds
    # the key (it was inserted into every table).
    if num_tables > 1 and valid_mask.any():
        wrong_table_ids = (table_ids + 1) % num_tables
        _, wrong_founds, wrong_indices = table.lookup(
            keys, wrong_table_ids, score_arg_lookup
        )
        both_valid = valid_mask & wrong_founds
        if both_valid.any():
            assert (
                wrong_indices[both_valid] >= 0
            ).all(), "Cross-table lookup should return valid local indices"

    table.erase(keys, table_ids)
    _, founds, _ = table.lookup(keys, table_ids, score_arg_lookup)
    assert not founds.any()

    num_valid = valid_mask.sum().item()
    max_num_reclaim = num_valid
    accum_num_reclaim = 0

    print(
        "Basic table operation(insert, lookup, erase) passed during the filling stage."
    )
    torch.cuda.synchronize()

    offset = 0
    max_step = 20
    fill_batch = max(bucket_capacity, table.capacity() // (max_step // 2) + 1)
    step = 1
    while table.size() < table.capacity() and step < max_step:
        keys = torch.randperm(fill_batch, device=device, dtype=torch.int64) + offset
        keys = keys.to(key_type)
        tid = torch.randint(
            num_tables, (keys.numel(),), dtype=torch.int64, device=device
        )

        score_arg = ScoreArg(name="score1", value=get_scores(score_policy, keys))

        insert_results = torch.empty(
            keys.numel(), dtype=table.result_type, device=device
        ).fill_(InsertResult.INIT.value)

        indices = table.insert(keys, tid, score_arg, insert_results)

        num_inserted = (insert_results == InsertResult.INSERT.value).sum()
        num_reclaimed = (insert_results == InsertResult.RECLAIM.value).sum()
        num_eviction = (insert_results == InsertResult.EVICT.value).sum()
        num_assign = (insert_results == InsertResult.ASSIGN.value).sum()
        num_illegal = (insert_results == InsertResult.ILLEGAL.value).sum()

        assert (
            keys.numel() == num_inserted + num_reclaimed + num_eviction + num_illegal
        ), (
            f"keys.numel() = {keys.numel()}, num_inserted = {num_inserted}, "
            f"num_reclaimed = {num_reclaimed}, num_eviction = {num_eviction}, "
            f"num_illegal = {num_illegal}"
        )
        assert num_assign == 0, f"num_assign = {num_assign}"

        accum_num_reclaim += num_reclaimed

        print(
            f"Table insert passed when load factor({table.load_factor():.3f}) with: "
            f"insert({num_inserted}), reclaim({num_reclaimed}), evict({num_eviction}), "
            f"illegal({num_illegal})"
        )

        offset += fill_batch
        step += 1

    if table.size() == table.capacity():
        # Reclaim count check only valid for single-table (with multi-table,
        # random table_ids mean fill keys may not land in the same table as
        # the erased slots)
        if num_tables == 1:
            assert (
                accum_num_reclaim == max_num_reclaim
            ), f"Occupied({accum_num_reclaim}/{max_num_reclaim}) reclaimed slots when table is full."

        keys = torch.randperm(batch_size, device=device, dtype=torch.int64) + offset
        keys = keys.to(key_type)
        tid = torch.randint(
            num_tables, (keys.numel(),), dtype=torch.int64, device=device
        )

        score_arg = ScoreArg(name="score1", value=get_scores(score_policy, keys))

        insert_results = torch.empty(
            batch_size, dtype=table.result_type, device=device
        ).fill_(InsertResult.INIT.value)

        indices = table.insert(keys, tid, score_arg, insert_results)

        assert (
            (insert_results == InsertResult.EVICT.value)
            | (insert_results == InsertResult.ILLEGAL.value)
        ).all()

        table.erase(keys, tid)
        _, founds, _ = table.lookup(keys, tid, score_arg)
        assert not founds.any()

        insert_results2 = torch.empty(
            batch_size, dtype=table.result_type, device=device
        ).fill_(InsertResult.INIT.value)
        indices_reinsert = table.insert(keys, tid, score_arg, insert_results2)

        assert (
            (insert_results2 == InsertResult.RECLAIM.value)
            | (insert_results2 == InsertResult.ILLEGAL.value)
        ).all()

        reclaim_mask = insert_results2 == InsertResult.RECLAIM.value
        assert torch.equal(
            torch.sort(indices[reclaim_mask]).values,
            torch.sort(indices_reinsert[reclaim_mask]).values,
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
def test_table_evict(
    key_type,
    num_buckets,
    bucket_capacity,
    batch_size,
    score_policy,
    capacity_multipliers,
):
    print("--------------------------------------------------------")
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()

    num_tables = len(capacity_multipliers)
    capacity = [m * num_buckets * bucket_capacity for m in capacity_multipliers]

    table = get_scored_table(
        capacity=capacity,
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score1", policy=score_policy)],
    )

    score_arg = ScoreArg(name="score1")
    score_arg_lookup = ScoreArg(
        name="score1",
        policy=ScorePolicy.CONST,
    )

    offset = 0

    while table.size() < table.capacity():
        keys = torch.randperm(batch_size, device=device, dtype=torch.int64) + offset
        offset += batch_size
        keys = keys.to(key_type)
        table_ids = torch.randint(
            num_tables, (batch_size,), dtype=torch.int64, device=device
        )

        score_arg.value = get_scores(score_policy, keys)
        score_arg_lookup.value = torch.zeros(
            batch_size, dtype=torch.uint64, device=device
        )

        insert_results = torch.empty(
            batch_size, dtype=table.result_type, device=device
        ).fill_(InsertResult.INIT.value)

        insert_scores_out = torch.empty(
            keys.numel(), dtype=torch.int64, device=keys.device
        )
        (
            indices,
            num_evicted,
            evicted_keys,
            evicted_indices,
            evicted_scores,
            evicted_table_ids,
        ) = table.insert_and_evict(
            keys, table_ids, score_arg, insert_results, score_out=insert_scores_out
        )

        score_out, founds, indices_lookup = table.lookup(
            keys, table_ids, score_arg_lookup
        )

        num_existed = founds.sum()

        num_inserted = (insert_results == InsertResult.INSERT.value).sum()
        num_reclaim = (insert_results == InsertResult.RECLAIM.value).sum()
        num_assign = (insert_results == InsertResult.ASSIGN.value).sum()
        num_inserted_by_eviction = (insert_results == InsertResult.EVICT.value).sum()
        num_insert_failed = (insert_results == InsertResult.BUSY.value).sum()
        num_illegal = (insert_results == InsertResult.ILLEGAL.value).sum()

        assert (
            num_reclaim == 0
        ), f"There is no erase operation, but got {num_reclaim} reclaimed slots when insert."
        assert (
            num_assign == 0
        ), f"There is no duplicated keys, but got {num_assign} duplicated keys when insert."

        assert (
            batch_size
            == num_inserted + num_inserted_by_eviction + num_insert_failed + num_illegal
        )
        assert num_existed == num_inserted + num_inserted_by_eviction
        assert num_evicted == num_inserted_by_eviction + num_insert_failed

        assert torch.equal(indices[founds], indices_lookup[founds])

        if table.score_specs[0].policy == ScorePolicy.ASSIGN:
            assert torch.equal(
                score_out[founds],
                insert_scores_out[founds],
            )
            global score_step
            assert (score_out[founds] == score_step).all()
        elif table.score_specs[0].policy == ScorePolicy.ACCUMULATE:
            assert (score_out[founds] == 1).all()
        else:
            assert torch.equal(
                score_out[founds],
                insert_scores_out[founds],
            )

        print(
            f"Table insert_and_evict passed when load factor:({table.load_factor():.3f}) with: "
            f"insert({num_inserted}), evict({num_inserted_by_eviction}), "
            f"failed({num_insert_failed}), illegal({num_illegal})"
        )


@pytest.mark.parametrize("key_type", [torch.int64])
@pytest.mark.parametrize(
    "table_config",
    [
        pytest.param(
            {"num_buckets": [1], "bucket_capacity": 128},
            id="1table_1bkt_cap128",
        ),
        pytest.param(
            {"num_buckets": [1, 3], "bucket_capacity": 128},
            id="2tables_asym_cap128",
        ),
        pytest.param(
            {"num_buckets": [1, 2, 4], "bucket_capacity": 64},
            id="3tables_mixed_cap64",
        ),
    ],
)
def test_overflow_with_counter(key_type, table_config):
    """Unified test: construction, fill, counter lock, overflow insertion,
    lookup, counter release, eviction, and reset."""
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()

    num_buckets_list = table_config["num_buckets"]
    bucket_capacity = table_config["bucket_capacity"]
    num_tables = len(num_buckets_list)
    capacity = [nb * bucket_capacity for nb in num_buckets_list]
    total_main_cap = sum(capacity)
    ovf_cap_per_table = 3 * bucket_capacity
    num_rounds = 10

    # ------------------------------------------------------------------
    # Phase 1: Construction verification
    # ------------------------------------------------------------------
    table = get_scored_table(
        capacity=capacity,
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score1", policy=ScorePolicy.ASSIGN)],
        enable_overflow=True,
    )

    assert table.enable_overflow_ is True
    assert table.overflow_bucket_capacity_ == ovf_cap_per_table

    expected_total = total_main_cap + ovf_cap_per_table * num_tables
    assert table.capacity() == expected_total
    assert table._ref_counter is not None
    assert table._ref_counter.numel() == expected_total
    assert (table._ref_counter == 0).all()

    for t in range(num_tables):
        assert table.main_capacity(table_id=t) == capacity[t]
        assert table.capacity(table_id=t) == capacity[t] + ovf_cap_per_table

    print(f"Phase 1 passed: {num_tables} tables, bucket_capacity={bucket_capacity}")

    # ------------------------------------------------------------------
    # Phase 2: Fill main table using counter + insert (no overflow)
    # ------------------------------------------------------------------
    key_offset = 1
    fill_keys_list = []
    fill_tids_list = []
    fill_indices_list = []

    batch_per_table = total_main_cap
    max_iters = 100

    for _iter in range(max_iters):
        all_full = all(table.size(table_id=t) == capacity[t] for t in range(num_tables))
        if all_full:
            break

        iter_keys, iter_tids = [], []
        for t in range(num_tables):
            k = torch.arange(
                key_offset,
                key_offset + batch_per_table,
                device=device,
                dtype=torch.int64,
            ).to(key_type)
            iter_keys.append(k)
            iter_tids.append(
                torch.full((batch_per_table,), t, dtype=torch.int64, device=device)
            )
            key_offset += batch_per_table

        batch_keys = torch.cat(iter_keys)
        batch_tids = torch.cat(iter_tids)
        batch_n = batch_keys.numel()
        batch_scores = ScoreArg(
            name="score1",
            value=torch.full((batch_n,), 100, dtype=torch.uint64, device=device),
        )
        batch_results = torch.empty(batch_n, dtype=table.result_type, device=device)

        indices = table.insert(batch_keys, batch_tids, batch_scores, batch_results)

        success_mask = (
            (batch_results == InsertResult.INSERT.value)
            | (batch_results == InsertResult.RECLAIM.value)
            | (batch_results == InsertResult.ASSIGN.value)
        )

        if not success_mask.any():
            break

        succ_indices = indices[success_mask]
        table.increment_counter(succ_indices)
        fill_keys_list.append(batch_keys[success_mask])
        fill_tids_list.append(batch_tids[success_mask])
        fill_indices_list.append(succ_indices)

    for t in range(num_tables):
        assert (
            table.size(table_id=t) == capacity[t]
        ), f"Table {t}: expected size {capacity[t]}, got {table.size(table_id=t)}"

    fill_keys = torch.cat(fill_keys_list)
    fill_tids = torch.cat(fill_tids_list)
    fill_indices = torch.cat(fill_indices_list)

    per_table_main_caps = torch.tensor(capacity, dtype=torch.int64, device=device)

    print(f"Phase 2 passed: {fill_keys.numel()}/{total_main_cap} keys in main")

    # ------------------------------------------------------------------
    # Phase 3: Verify counters on filled slots (already locked in Phase 2)
    # ------------------------------------------------------------------
    counter_cpu = table._ref_counter.cpu()
    for idx in fill_indices.cpu().tolist():
        assert (
            counter_cpu[idx].item() >= 1
        ), f"Counter at slot {idx} should be >= 1 after increment"

    print("Phase 3 passed: all filled slots locked")

    def to_counter_indices(indices, tids):
        per_key_main = per_table_main_caps[tids]
        is_ovf = indices >= per_key_main
        return torch.where(
            is_ovf,
            total_main_cap + (indices - per_key_main),
            indices,
        )

    # ------------------------------------------------------------------
    # Phase 4: Iterative overflow insertion (10 rounds)
    # ------------------------------------------------------------------
    ovf_keys_per_table = ovf_cap_per_table
    batch_per_table = ovf_keys_per_table // num_rounds

    ovf_key_pools = []
    for t in range(num_tables):
        k = torch.arange(
            key_offset,
            key_offset + ovf_keys_per_table,
            device=device,
            dtype=torch.int64,
        ).to(key_type)
        ovf_key_pools.append(k)
        key_offset += ovf_keys_per_table

    all_ovf_indices = []
    all_ovf_keys = []
    all_ovf_tids = []
    all_ovf_results = []

    for r in range(num_rounds):
        round_keys_list = []
        round_tids_list = []
        for t in range(num_tables):
            start = r * batch_per_table
            end = start + batch_per_table
            round_keys_list.append(ovf_key_pools[t][start:end])
            round_tids_list.append(
                torch.full((batch_per_table,), t, dtype=torch.int64, device=device)
            )
        round_keys = torch.cat(round_keys_list)
        round_tids = torch.cat(round_tids_list)
        round_batch = round_keys.numel()

        round_scores = ScoreArg(
            name="score1",
            value=torch.full((round_batch,), 1, dtype=torch.uint64, device=device),
        )
        round_results = torch.empty(round_batch, dtype=table.result_type, device=device)

        round_indices, _, *_ = table.insert_and_evict_with_counter_and_overflow(
            round_keys,
            round_tids,
            round_scores,
            round_results,
        )

        success_mask = (
            (round_results == InsertResult.INSERT.value)
            | (round_results == InsertResult.EVICT.value)
            | (round_results == InsertResult.ASSIGN.value)
        )

        per_key_main = per_table_main_caps[round_tids]
        ovf_mask = round_indices >= per_key_main

        successful_indices = round_indices[success_mask]
        if successful_indices.numel() > 0:
            counter_idx = to_counter_indices(
                successful_indices, round_tids[success_mask]
            )
            table.increment_counter(counter_idx)

        all_ovf_indices.append(round_indices)
        all_ovf_keys.append(round_keys)
        all_ovf_tids.append(round_tids)
        all_ovf_results.append(round_results)

        round_ovf_count = ovf_mask[success_mask].sum().item()
        print(
            f"  Round {r}: {success_mask.sum().item()}/{round_batch} succeeded, "
            f"{round_ovf_count} in overflow"
        )

    print("Phase 4 passed: iterative overflow insertion complete")

    # ------------------------------------------------------------------
    # Phase 5: Lookup all inserted keys
    # ------------------------------------------------------------------
    all_ovf_keys_cat = torch.cat(all_ovf_keys)
    all_ovf_tids_cat = torch.cat(all_ovf_tids)
    all_ovf_results_cat = torch.cat(all_ovf_results)
    all_ovf_indices_cat = torch.cat(all_ovf_indices)
    ovf_success_mask = (
        (all_ovf_results_cat == InsertResult.INSERT.value)
        | (all_ovf_results_cat == InsertResult.EVICT.value)
        | (all_ovf_results_cat == InsertResult.ASSIGN.value)
    )

    per_key_main_p4 = per_table_main_caps[all_ovf_tids_cat]
    p4_in_ovf = (
        all_ovf_indices_cat[ovf_success_mask] >= per_key_main_p4[ovf_success_mask]
    )
    assert p4_in_ovf.any(), "At least some Phase 4 keys should land in overflow"

    lookup_score = ScoreArg(
        name="score1",
        value=torch.zeros(total_main_cap, dtype=torch.uint64, device=device),
        policy=ScorePolicy.CONST,
    )
    _, fill_founds, fill_lookup_indices = table.lookup_with_overflow(
        fill_keys,
        fill_tids,
        lookup_score,
    )
    assert fill_founds.all(), "All fill keys should still be found"
    assert torch.equal(
        fill_indices, fill_lookup_indices
    ), "Fill lookup indices should match original insert indices"

    ovf_success_keys = all_ovf_keys_cat[ovf_success_mask]
    ovf_success_tids = all_ovf_tids_cat[ovf_success_mask]
    ovf_success_indices = all_ovf_indices_cat[ovf_success_mask]

    if ovf_success_keys.numel() > 0:
        ovf_lookup_score = ScoreArg(
            name="score1",
            value=torch.zeros(
                ovf_success_keys.numel(), dtype=torch.uint64, device=device
            ),
            policy=ScorePolicy.CONST,
        )
        _, ovf_founds, ovf_lookup_indices = table.lookup_with_overflow(
            ovf_success_keys,
            ovf_success_tids,
            ovf_lookup_score,
        )
        assert (
            ovf_founds.all()
        ), "All successfully inserted overflow keys should be found"
        assert torch.equal(ovf_success_indices, ovf_lookup_indices)

    ghost_keys = torch.arange(
        key_offset, key_offset + 50, device=device, dtype=torch.int64
    ).to(key_type)
    ghost_tids = torch.zeros(50, dtype=torch.int64, device=device)
    ghost_score = ScoreArg(
        name="score1",
        value=torch.zeros(50, dtype=torch.uint64, device=device),
        policy=ScorePolicy.CONST,
    )
    _, ghost_founds, _ = table.lookup_with_overflow(
        ghost_keys,
        ghost_tids,
        ghost_score,
    )
    assert not ghost_founds.any(), "Never-inserted keys should not be found"

    print("Phase 5 passed: all lookups verified")

    # ------------------------------------------------------------------
    # Phase 6: Decrement counters + verify eviction is possible
    # ------------------------------------------------------------------
    table.decrement_counter(fill_indices)

    ovf_successful_all = []
    for r_idx, r_tids, r_results in zip(all_ovf_indices, all_ovf_tids, all_ovf_results):
        r_success = (
            (r_results == InsertResult.INSERT.value)
            | (r_results == InsertResult.EVICT.value)
            | (r_results == InsertResult.ASSIGN.value)
        )
        if r_success.any():
            ovf_successful_all.append(
                to_counter_indices(r_idx[r_success], r_tids[r_success])
            )
    if ovf_successful_all:
        table.decrement_counter(torch.cat(ovf_successful_all))

    assert (
        table._ref_counter == 0
    ).all(), "All counters should be 0 after decrementing"

    evict_batch = min(32, total_main_cap)
    evict_keys = torch.arange(
        key_offset, key_offset + evict_batch, device=device, dtype=torch.int64
    ).to(key_type)
    key_offset += evict_batch
    evict_tids = torch.zeros(evict_batch, dtype=torch.int64, device=device)
    evict_scores = ScoreArg(
        name="score1",
        value=torch.full((evict_batch,), 200, dtype=torch.uint64, device=device),
    )
    evict_results = torch.empty(evict_batch, dtype=table.result_type, device=device)
    _, num_evicted, evicted_keys, *_ = table.insert_and_evict_with_counter_and_overflow(
        evict_keys,
        evict_tids,
        evict_scores,
        evict_results,
    )

    has_evict = (evict_results == InsertResult.EVICT.value).any().item()
    print(f"Phase 6: evictions={num_evicted}, " f"has_evict_result={has_evict}")
    assert (
        num_evicted > 0 or has_evict
    ), "With counters at 0 and higher scores, some eviction should occur"

    print("Phase 6 passed: counter release and eviction verified")

    # ------------------------------------------------------------------
    # Phase 7: Reset
    # ------------------------------------------------------------------
    table.reset()
    for t in range(num_tables):
        assert table.size(table_id=t) == 0, f"Table {t} should have size 0 after reset"
    assert (table._ref_counter == 0).all(), "Counters should be 0 after reset"
    assert (
        table.overflow_bucket_sizes == 0
    ).all(), "Overflow bucket sizes should be 0 after reset"

    print("Phase 7 passed: reset verified")
    print("test_overflow_with_counter PASSED")
