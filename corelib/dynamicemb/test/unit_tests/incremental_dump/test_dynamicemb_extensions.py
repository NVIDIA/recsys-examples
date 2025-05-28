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
from dataclasses import astuple, dataclass

import pytest
import torch
from dynamicemb import DynamicEmbCheckMode, DynamicEmbInitializerArgs, dyn_emb_to_torch
from dynamicemb_extensions import (
    DynamicEmbDataType,
    DynamicEmbTable,
    EvictStrategy,
    InitializerArgs,
    OptimizerType,
    count_matched,
    device_timestamp,
    dyn_emb_rows,
    export_batch_matched,
    find,
    find_or_insert,
    insert_and_evict,
)


@dataclass
class ExtensionsTableOption:
    key_type: DynamicEmbDataType = DynamicEmbDataType.Int64
    value_type: DynamicEmbDataType = DynamicEmbDataType.Float32
    evict_strategy: EvictStrategy = EvictStrategy.KLru
    dim: int = 16
    init_capacity: int = 512 * 1024
    max_capacity: int = 512 * 1024
    local_hbm_for_values: int = 1024**3
    bucket_capacity: int = 128
    max_load_factor: float = 0.6
    block_size: int = 128
    io_block_size: int = 1024
    device_id: int = -1
    io_by_cpu: bool = False
    use_constant_memory: bool = False
    reserved_key_start_bit: int = 0
    num_of_buckets_par_alloc: int = 1
    initializer_args: InitializerArgs = DynamicEmbInitializerArgs().as_ctype()
    safe_check_mode: int = DynamicEmbCheckMode.IGNORE.value
    optimizer_type: OptimizerType = OptimizerType.Null


class ScoreAdaptor:
    def __init__(
        self, evict_strategy: EvictStrategy, dtype: torch.dtype, device: torch.device
    ):
        self.evict_strategy_ = evict_strategy
        self.dtype_ = dtype
        self.device_ = device

        min_step = 0
        self.min_score_: int = (
            device_timestamp() if evict_strategy == EvictStrategy.KLru else min_step
        )
        self.step_: int = min_step + 1

    def min_score(self):
        return self.min_score_

    def next_score(self) -> int:
        next_score = self.step_
        return (
            device_timestamp()
            if self.evict_strategy_ == EvictStrategy.KLru
            else next_score
        )

    def score(self):
        if self.evict_strategy_ == EvictStrategy.KLru:
            return None
        score = self.step_
        self.step_ += 1
        return score


def random_indices(batch, min_index, max_index):
    result = set({})
    while len(result) < batch:
        result.add(random.randint(min_index, max_index))
    return result


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


@pytest.fixture(name="ext_option")
def extentions_table_option(current_device):
    ext_table_option = ExtensionsTableOption()
    ext_table_option.device_id = current_device
    return ext_table_option


@pytest.fixture
def score_type():
    """
    It's safe to convert from uint64 to int64 for score:
      1. Under DynamicEmbScoreStrategy.TIMESTAMP mode, score is a device timestamp in nanosecond,
        and it takes nearly 300 years since GPU startup to make the highest bit to 1.
      2. Under DynamicEmbScoreStrategy.STEP mode, it will take longer,
        because score increase for 1 insertion and not for 1 nanosecond.
    """
    return torch.uint64


@pytest.fixture
def counter_dtype():
    return torch.uint64


@pytest.mark.parametrize(
    "evict_strategy", [EvictStrategy.KLru, EvictStrategy.KCustomized]
)
@pytest.mark.parametrize(
    "bucket_capacity, batch, capacity, num_iteration, dump_interval",
    [
        pytest.param(
            128, 128, 512 * 1024, 8192, 1024, id="Never evict keys from current batch"
        ),
        pytest.param(128, 65536, 512 * 1024, 32, 8),
        pytest.param(
            128, 1024 + 13, 1024, 12, 3, id="Always evict keys from current batch"
        ),
        pytest.param(
            128, 1024, 4 * 1024, 32, 4, id="Always evict keys from last dump_interval"
        ),
        pytest.param(512, 512, 512 * 1024, 2048, 256, id="Different bucket capacity"),
    ],
)
def test_dynamicemb_extensions(
    request,
    ext_option,
    score_type,
    counter_dtype,
    evict_strategy,
    bucket_capacity,
    batch,
    capacity,
    num_iteration,
    dump_interval,
):
    print(f"\n{request.node.name}")
    # init
    ext_option.init_capacity = capacity
    ext_option.max_capacity = capacity
    ext_option.bucket_capacity = bucket_capacity
    ext_option.evict_strategy = evict_strategy
    ext_option.num_of_buckets_par_alloc = capacity // bucket_capacity

    assert ext_option.dim * ext_option.max_capacity <= ext_option.local_hbm_for_values

    table = DynamicEmbTable(*astuple(ext_option))
    device = torch.device(f"cuda:{ext_option.device_id}")
    score_adaptor = ScoreAdaptor(evict_strategy, score_type, device)
    init_score: int = score_adaptor.min_score()

    # insert once: the first find_or_insert will insert all keys, and no eviction(batch = bucket_capacity)
    keys_set = random_indices(bucket_capacity, 0, (1 << 63) - 1)
    keys = torch.tensor(
        list(keys_set), dtype=dyn_emb_to_torch(ext_option.key_type), device=device
    )
    values = torch.empty(
        bucket_capacity,
        ext_option.dim,
        dtype=dyn_emb_to_torch(ext_option.value_type),
        device=device,
    )
    score = score_adaptor.score()
    find_or_insert(table, bucket_capacity, keys, values, score)

    # check 1: count_matched works well
    d_num_matched = torch.zeros(1, dtype=counter_dtype, device=device)
    count_matched(table, init_score, d_num_matched)
    num_matched = d_num_matched.cpu().item()
    assert num_matched == bucket_capacity

    # check 2: export_batch_matched is consistent with count_matched
    dump_keys = torch.empty(num_matched, dtype=keys.dtype, device=device)
    dump_vals = torch.empty(
        num_matched, ext_option.dim, dtype=values.dtype, device=device
    )
    d_num_matched.fill_(0)
    export_batch_matched(
        table, init_score, table.capacity(), 0, d_num_matched, dump_keys, dump_vals
    )
    assert num_matched == d_num_matched.cpu().item()
    assert keys_set == set(dump_keys.cpu().tolist())

    # insert iteratively
    last_score, last_insert, last_evict = init_score, bucket_capacity, 0
    undump_score, undump_insert, undump_evict = (score_adaptor.next_score(), 0, 0)
    for i in range(0, num_iteration, dump_interval):
        for j in range(dump_interval):
            keys_set = random_indices(batch, 0, (1 << 63) - 1)
            keys = torch.tensor(
                list(keys_set),
                dtype=dyn_emb_to_torch(ext_option.key_type),
                device=device,
            )
            values = torch.empty(
                batch,
                ext_option.dim,
                dtype=dyn_emb_to_torch(ext_option.value_type),
                device=device,
            )
            score = score_adaptor.score()
            founds = torch.full([batch], True, dtype=torch.bool, device=device)

            find(table, batch, keys, values, founds)
            undump_insert += torch.sum(founds == False).item()

            evict_keys = torch.empty_like(keys)
            evict_vals = torch.empty_like(values)
            evict_scores = torch.empty(batch, dtype=score_type, device=device)
            evict_counter = torch.zeros(1, dtype=counter_dtype, device=device)
            old_size = dyn_emb_rows(table)
            insert_and_evict(
                table,
                batch,
                keys,
                values,
                score,
                evict_keys,
                evict_vals,
                evict_scores,
                evict_counter,
            )
            new_size = dyn_emb_rows(table)
            num_evict = evict_counter.cpu().item()
            assert new_size - old_size == torch.sum(founds == False).item() - num_evict

            # convert to torch.int64 to compare, check highest bit not 1.
            mask = torch.tensor([2**63], dtype=score_type, device="cpu")
            assert ((evict_scores[:num_evict].cpu() & mask) == 0).all()
            valid_evict_scores = evict_scores[:num_evict].to(
                dtype=torch.int64, device="cpu"
            )
            last_evict += torch.sum(
                (valid_evict_scores >= last_score) & (valid_evict_scores < undump_score)
            ).item()
            undump_evict += torch.sum(valid_evict_scores >= undump_score).item()

        # check 3: using smaller score to count will get more than larger one
        d_num_matched.fill_(0)
        count_matched(table, last_score, d_num_matched)
        num_dump_more = d_num_matched.cpu().item()
        d_num_matched.fill_(0)
        count_matched(table, undump_score, d_num_matched)
        num_dump_less = d_num_matched.cpu().item()
        assert num_dump_more - num_dump_less == last_insert - last_evict

        # log
        load_factor = dyn_emb_rows(table) / table.capacity()
        print(
            f"Load factor={load_factor:.3f}, num_dump_more={num_dump_more}, num_dump_less={num_dump_less}, \
          last_insert={last_insert}, last_evict={last_evict}, \
          undumped_insert={undump_insert}, undumped_evict={undump_evict}"
        )

        # check 4: using min_score to count will get table's size
        d_num_matched.fill_(0)
        count_matched(table, init_score, d_num_matched)
        assert dyn_emb_rows(table) == d_num_matched.cpu().item()

        # update
        last_score, last_insert, last_evict = (
            undump_score,
            undump_insert,
            undump_evict,
        )
        undump_score, undump_insert, undump_evict = (score_adaptor.next_score(), 0, 0)

        # check 5: using uninsert score will dump nothing.
        d_num_matched.fill_(0)
        count_matched(table, undump_score, d_num_matched)
        assert d_num_matched.cpu().item() == 0
