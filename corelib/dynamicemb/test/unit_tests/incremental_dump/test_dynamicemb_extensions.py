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
from typing import Dict, List, Optional, Set, Tuple

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
    dyn_emb_capacity,
    dyn_emb_cols,
    dyn_emb_rows,
    export_batch,
    export_batch_matched,
    find,
    find_or_insert,
    find_pointers_with_scores,
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
        if evict_strategy == EvictStrategy.KLru:
            self.min_score_: int = device_timestamp()
        elif evict_strategy == EvictStrategy.KLfu:
            # LFU mode: use monotonically increasing score
            self.min_score_: int = min_step
        else:
            self.min_score_: int = min_step
        self.step_: int = min_step + 1

    def min_score(self):
        return self.min_score_

    #   add LFU here same to customized
    def next_score(self) -> int:
        if self.evict_strategy_ == EvictStrategy.KLru:
            return device_timestamp()
        else:
            # LFU and other modes: return monotonically increasing score
            next_score = self.step_
            self.step_ += 1
            return next_score

    # add LFU here
    def score(self):
        if self.evict_strategy_ == EvictStrategy.KLru:
            return None
        elif self.evict_strategy_ == EvictStrategy.KLfu:
            # score = self.step_
            # self.step_ += 1
            # return score
            # LFU should give merlin hashtable frequency increment 1
            return 1
        else:
            score = self.step_
            self.step_ += 1
            return score


class LFUSimulator:
    """Host-side LFU policy simulator for comparison with HKV table results"""

    def __init__(self, initial_capacity: int, dim: int):
        self.capacity = initial_capacity
        self.dim = dim

        # key -> (value, frequency, insertion_order)
        self.table: Dict[int, Tuple[torch.Tensor, int, int]] = {}
        self.insertion_counter = 0

    def size(self) -> int:
        return len(self.table)

    def find_or_insert(
        self, keys: torch.Tensor, values: torch.Tensor, score: int = 1
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Simulate find_or_insert operation
        Returns: (found_mask, evicted_keys)
        """
        batch_size = keys.size(0)
        found_mask = torch.zeros(batch_size, dtype=torch.bool)
        evicted_keys = []

        # Calculate number of new keys to insert
        new_keys_count = 0
        for key in keys:
            if key.item() not in self.table:
                new_keys_count += 1

        for i, key in enumerate(keys):
            key_item = key.item()
            self.insertion_counter += 1

            if key_item in self.table:
                # Key exists, update frequency and access time
                old_value, old_freq, _ = self.table[key_item]
                new_freq = old_freq + score
                self.table[key_item] = (
                    old_value.clone(),
                    new_freq,
                    self.insertion_counter,
                )
                values[i] = old_value
                found_mask[i] = True
            else:
                # Key doesn't exist, need to insert
                if len(self.table) < self.capacity:
                    # Has space, insert directly
                    self.table[key_item] = (
                        values[i].clone(),
                        score,
                        self.insertion_counter,
                    )
                else:
                    # Need to evict, find the lowest frequency key
                    min_freq = float("inf")
                    min_key = None
                    min_time = float("inf")

                    for k, (v, freq, time) in self.table.items():
                        # LFU policy: evict lowest frequency first, if same frequency then evict earliest accessed
                        if freq < min_freq or (freq == min_freq and time < min_time):
                            min_freq = freq
                            min_key = k
                            min_time = time

                    # Evict the least frequently used key
                    if min_key is not None:
                        del self.table[min_key]
                        evicted_keys.append(min_key)

                    # Insert new key
                    self.table[key_item] = (
                        values[i].clone(),
                        score,
                        self.insertion_counter,
                    )

                found_mask[i] = False

        return found_mask, evicted_keys

    def find(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Simulate find operation, don't update frequency
        """
        batch_size = keys.size(0)
        found_mask = torch.zeros(batch_size, dtype=torch.bool)

        for i, key in enumerate(keys):
            key_item = key.item()
            if key_item in self.table:
                value, freq, time = self.table[key_item]
                values[i] = value
                found_mask[i] = True
            else:
                found_mask[i] = False

        return found_mask

    def insert_and_evict(
        self, keys: torch.Tensor, values: torch.Tensor, score: int = 1
    ) -> Tuple[List[int], int]:
        """
        Simulate insert_and_evict operation
        Returns: (evicted_keys, num_evicted)
        """
        evicted_keys = []
        evicted_scores = []

        for i, key in enumerate(keys):
            key_item = key.item()
            self.insertion_counter += 1

            if key_item in self.table:
                # Key exists, update frequency
                old_value, old_freq, _ = self.table[key_item]
                new_freq = old_freq + score
                self.table[key_item] = (
                    values[i].clone(),
                    new_freq,
                    self.insertion_counter,
                )
            else:
                # Key doesn't exist, need to insert
                if len(self.table) == self.capacity:
                    # Need to evict
                    min_freq = float("inf")
                    min_key = None
                    min_time = float("inf")

                    for k, (v, freq, time) in self.table.items():
                        if freq < min_freq or (freq == min_freq and time < min_time):
                            min_freq = freq
                            min_key = k
                            min_time = time

                    if min_key is not None:
                        del self.table[min_key]
                        evicted_keys.append(min_key)
                        evicted_scores.append(min_freq)

                # Insert new key
                self.table[key_item] = (
                    values[i].clone(),
                    score,
                    self.insertion_counter,
                )

        return evicted_keys, evicted_scores, len(evicted_keys)

    def erase(self, keys: Set[int]):
        for key in keys:
            assert key in self.table, "Key is not in the table, can't be erased."

            del self.table[key]

    def get_keys_frequency(self, keys: Set[int]):
        scores = []

        for key in keys:
            assert key in self.table, "Key is not in the table, can't get its score."
            scores.append(self.table[key][1])
        return scores

    def get_keys_by_score_threshold(self, min_score: int) -> Set[int]:
        """Get all keys with frequency >= min_score"""
        return {k for k, (v, freq, time) in self.table.items() if freq >= min_score}

    def get_all_keys(self) -> Set[int]:
        """Get all keys"""
        return set(self.table.keys())

    def get_key_frequency(self, key: int) -> Optional[int]:
        """Get frequency of specified key"""
        if key in self.table:
            return self.table[key][1]
        return None


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
    "evict_strategy",
    [EvictStrategy.KLru, EvictStrategy.KCustomized],
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
    # import pdb; pdb.set_trace()
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
            # After insert_and_evict: table_size_change = new_insertions - evictions
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


def export_all_keys_values_scores(table, device):
    """
    Export all keys, values and scores from HKV table
    Returns: (keys, values, scores) all on CPU
    """
    capacity = dyn_emb_capacity(table)
    current_size = dyn_emb_rows(table)
    dim = dyn_emb_cols(table)  # Get actual dimension from table

    if current_size == 0:
        return (
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dim, dtype=torch.float32),
            torch.empty(0, dtype=torch.uint64),
        )

    # Prepare output tensors
    key_dtype = torch.int64  # key type
    value_dtype = torch.float32  # value type
    score_dtype = torch.uint64  # score type

    batch_size = min(65536, capacity)

    all_keys = []
    all_values = []
    all_scores = []

    offset = 0
    while offset < capacity:
        # Prepare batch tensors
        keys = torch.empty(batch_size, dtype=key_dtype, device=device)
        values = torch.empty(batch_size * dim, dtype=value_dtype, device=device)
        scores = torch.empty(batch_size, dtype=score_dtype, device=device)
        d_counter = torch.zeros(1, dtype=torch.uint64, device=device)

        # Call export_batch
        export_batch(table, batch_size, offset, d_counter, keys, values, scores)

        # Get actual returned count
        actual_count = d_counter.cpu().item()

        if actual_count > 0:
            # Keep only valid data
            valid_keys = keys[:actual_count].cpu()
            valid_values = values[: actual_count * dim].view(actual_count, dim).cpu()
            valid_scores = scores[:actual_count].cpu()

            all_keys.append(valid_keys)
            all_values.append(valid_values)
            all_scores.append(valid_scores)

        offset += batch_size

        # If this batch returns no data, we've finished traversing
        if actual_count == 0:
            break

    if all_keys:
        return torch.cat(all_keys), torch.cat(all_values), torch.cat(all_scores)
    else:
        return (
            torch.empty(0, dtype=key_dtype),
            torch.empty(0, dim, dtype=value_dtype),
            torch.empty(0, dtype=score_dtype),
        )


def find_hkv_score(table, key: int, device: torch.device) -> Tuple[bool, int]:
    founds = torch.empty(1, dtype=torch.bool, device=device).fill_(False)
    keys = torch.tensor([key], dtype=torch.int64, device=device)
    values = torch.empty(1, dtype=torch.int64, device=device)
    scores = torch.empty(1, dtype=torch.uint64, device=device).fill_(0)
    find_pointers_with_scores(table, 1, keys, values, founds, scores)
    return founds[0].item(), scores[0].item()


def simulator_insert_and_evict(lfu_simulator, key, score, ext_option):
    random_val = torch.randn(
        1,
        ext_option.dim,
        dtype=dyn_emb_to_torch(ext_option.value_type),
    )
    _, _, num_evict = lfu_simulator.insert_and_evict(
        torch.tensor([key], dtype=torch.int64),
        random_val,
        score,
    )
    return num_evict


@pytest.mark.parametrize("evict_strategy", [EvictStrategy.KLfu])
@pytest.mark.parametrize(
    "bucket_capacity, batch, capacity, num_iteration",
    [
        pytest.param(128, 3, 512, 5, id="Small scale LFU vs HKV"),
        pytest.param(
            128, 1024 + 13, 1024, 12, id="Always evict keys from current batch"
        ),
        pytest.param(
            128, 1024 + 13, 2048, 32, id="Always evict keys from current batch"
        ),
        pytest.param(
            128, 1024, 4 * 1024, 32, id="Always evict keys from last dump_interval"
        ),
    ],
)
def test_dynamicemb_extensions_lfu(
    request,
    ext_option,
    score_type,
    counter_dtype,
    evict_strategy,
    bucket_capacity,
    batch,
    capacity,
    num_iteration,
):
    print(f"\n{request.node.name} - LFU Algorithm Consistency Test")
    print(f"Testing high-level LFU algorithm behavior between HKV table and simulator")

    # Initialize configuration
    ext_option.init_capacity = capacity
    ext_option.max_capacity = capacity
    # Use capacity as bucket_capacity, because when num_buckets > 1, HKV table behavior cannot be simulated
    ext_option.bucket_capacity = capacity
    ext_option.evict_strategy = evict_strategy
    ext_option.num_of_buckets_par_alloc = capacity // ext_option.bucket_capacity

    assert ext_option.dim * ext_option.max_capacity <= ext_option.local_hbm_for_values

    # Initialize HKV table
    table = DynamicEmbTable(*astuple(ext_option))
    device = torch.device(f"cuda:{ext_option.device_id}")
    score_adaptor = ScoreAdaptor(evict_strategy, score_type, device)

    # Initialize LFU simulator
    lfu_simulator = LFUSimulator(capacity, ext_option.dim)

    print(f"Capacity: {capacity}, Batch: {batch}, Total operations: {num_iteration}")
    print(
        f"Testing LFU insertion/eviction count consistency between HKV table and simulator"
    )

    # Step-by-step testing, verify insertion and eviction counts for each operation
    total_hkv_evictions = 0
    total_sim_evictions = 0

    comparison_interval = max(1, num_iteration // 10)

    for iteration in range(num_iteration):
        keys_set = random_indices(batch, 0, (1 << 31) - 1)
        keys = torch.tensor(
            list(keys_set),
            dtype=dyn_emb_to_torch(ext_option.key_type),
            device=device,
        )
        values = torch.randn(
            batch,
            ext_option.dim,
            dtype=dyn_emb_to_torch(ext_option.value_type),
            device=device,
        )
        score = score_adaptor.score()  # LFU returns 1

        # Record size before operation
        hkv_size_before = dyn_emb_rows(table)
        sim_size_before = lfu_simulator.size()

        found_values = torch.empty(
            batch,
            ext_option.dim,
            dtype=dyn_emb_to_torch(ext_option.value_type),
            device=device,
        )
        founds = torch.full([batch], False, dtype=torch.bool, device=device)

        find(table, batch, keys, found_values, founds)

        # HKV table operation: insert_and_evict
        evict_keys = torch.empty_like(keys)
        evict_vals = torch.empty_like(values)
        evict_scores = torch.empty(batch, dtype=torch.uint64, device=device)
        evict_counter = torch.zeros(1, dtype=torch.uint64, device=device)

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

        hkv_evicted_count = evict_counter.cpu().item()

        # LFU simulator operation: insert_and_evict
        sim_keys = keys.cpu()
        sim_values = values.cpu()
        (
            sim_evicted_keys,
            sim_eviceted_scores,
            sim_evicted_count,
        ) = lfu_simulator.insert_and_evict(sim_keys, sim_values, score)

        """
        If HKV needs to evict m keys, but n keys meet the criteria, there are at most C^m_n possibilities (this is a combinatorial problem...). 
        The p keys with the smallest scores must be evicted (0 <= p <= m).

        During concurrent insertion, a key that should have been found in the current batch might be evicted by another thread, causing it to be re-inserted, and the score count reset. 
        However, sequential insertion in the Simulator might cause the score to accumulate, and vice versa (scores originally evictted in the Simulator might accumulate in HKV).

        After each insertion, based on the results of insertions in HKV and the Simulator, we can determine whether the evictment is reasonable and calibrate the Simulator's keys to match those in HKV, using this as the basis for the next insertion's judgment.
        """

        print(
            f"Iteration{iteration}: HKV evict {hkv_evicted_count} keys, Simulator evict {sim_evicted_count} keys."
        )
        # Check the eviction results whether reasonable or not
        if sim_evicted_count == 0:
            # case1: Simulator have enough empty slots, so HKV should also be.
            # case2: Keys all existed in Simulator, and it should be the same to HKV.
            # case3: Part of keys existed, and there are enough slots for un-existed keys, and it should be the same to HKV.
            # That's because HKV should not evict keys as there existed enough empty slots.
            assert (
                hkv_evicted_count == 0
            ), f"HKV Eviction should not to be happened as capacity old({hkv_size_before}), new({hkv_size_after}), batch({batch})"

        evict_keys = evict_keys[:hkv_evicted_count].cpu().tolist()
        evict_scores = evict_scores[:hkv_evicted_count].cpu().tolist()

        cur_batch_keys = keys.cpu().tolist()

        # If a key was evict by HKV but not evict by Simulator, will add to the dict, and vice versa
        # Keys in current batch will not append to the dict.
        ambiguous_key_score: Dict[int, int] = {}
        hkv_ambiguous_cnt = 0
        sim_ambiguous_cnt = 0

        for i, hkv_key in enumerate(evict_keys):
            hkv_score = evict_scores[i]
            # the key was evicted from both HKV table and Simulator table, or the key insert failed into both table.
            if hkv_key in sim_evicted_keys:
                # maybe hkv_key in cur_batch_keys, but if so they should still keep the same.
                index = sim_evicted_keys.index(hkv_key)
                sim_score = sim_eviceted_scores[index]
                assert (
                    hkv_score == sim_score
                ), f"HKV and Simulator both evict key{hkv_key}, but score is different: HKV({hkv_score}/Simulator({sim_score}))"
            else:
                if hkv_key not in cur_batch_keys:
                    assert hkv_key not in ambiguous_key_score
                    ambiguous_key_score[hkv_key] = hkv_score
                    hkv_ambiguous_cnt += 1

                # The hkv_key still existed in Simulator table.
                sim_score = lfu_simulator.get_key_frequency(hkv_key)
                assert (
                    sim_score is not None
                ), f"Key({hkv_key} is expected to exist in Simulator as not be evicted."

                if hkv_key in cur_batch_keys:
                    found, hkv_score_1 = find_hkv_score(table, hkv_key, device)

                    if found:
                        # case1: existed key in HKV was evicted by other thread, and reinsert succeeded.
                        assert (
                            hkv_score + hkv_score_1 == sim_score
                        ), "score_in_hkv + score_evict_by_hkv should equal to score_in_simulator, as simulator doesn't evict the key, so the score should be accumulated."

                        # Calibration two tables
                        lfu_simulator.erase([hkv_key])
                        num_evict = simulator_insert_and_evict(
                            lfu_simulator, hkv_key, hkv_score_1, ext_option
                        )
                        assert num_evict == 0
                    else:
                        # case2: HKV insert failed, but Simulator insert succeed.
                        assert hkv_score == sim_score, "Score should keep the same."
                        # Calibration two tables
                        lfu_simulator.erase([hkv_key])
                else:
                    assert hkv_score == sim_score, "Score should keep the same."
                    # Calibration two tables
                    lfu_simulator.erase([hkv_key])

        for i, sim_key in enumerate(sim_evicted_keys):
            sim_score = sim_eviceted_scores[i]
            if sim_key not in evict_keys:
                if sim_key not in cur_batch_keys:
                    assert sim_key not in ambiguous_key_score
                    ambiguous_key_score[sim_key] = sim_score
                    sim_ambiguous_cnt += 1

                found, hkv_score = find_hkv_score(table, sim_key, device)

                sim_score_1 = lfu_simulator.get_key_frequency(sim_key)

                if sim_score_1 is None:
                    assert hkv_score == sim_score

                    # Calibration two tables
                    num_evict = simulator_insert_and_evict(
                        lfu_simulator, sim_key, sim_score, ext_option
                    )
                    assert num_evict == 0
                else:
                    assert hkv_score == sim_score + sim_score_1
                    # Calibration two tables

                    lfu_simulator.erase([sim_key])
                    num_evict = simulator_insert_and_evict(
                        lfu_simulator, sim_key, hkv_score, ext_option
                    )
                    assert num_evict == 0

        ambiguous_scores = list(ambiguous_key_score.values())
        assert len(set(ambiguous_scores)) <= 1
        assert len(ambiguous_scores) == hkv_ambiguous_cnt + sim_ambiguous_cnt

        print(
            f"Iteration{iteration}: there are {len(ambiguous_scores)} ambiguous scores and has the same score: HKV({hkv_ambiguous_cnt})/Simulator({sim_ambiguous_cnt})."
        )

        # Record size after operation
        hkv_size_after = dyn_emb_rows(table)
        sim_size_after = lfu_simulator.size()
        assert hkv_size_after == sim_size_after

        # Calculate insertion and eviction counts
        hkv_net_change = hkv_size_after - hkv_size_before
        sim_net_change = sim_size_after - sim_size_before

        if hkv_net_change == 0:
            assert founds.sum() + hkv_evicted_count == batch
            print(f"Found: {founds.sum().item()}, evict{hkv_evicted_count}")
        print(
            f"HKV capacity: before{hkv_size_before}, after{hkv_size_after}, evicted{hkv_evicted_count}, change{hkv_net_change}"
        )
        print(
            f"LFUTable capacity: before{sim_size_before}, after{sim_size_after}, evicted{sim_evicted_count}, change{sim_net_change}"
        )

        # Verify eviction count consistency
        assert (
            hkv_evicted_count == sim_evicted_count
        ), f"Iteration {iteration}: Eviction count mismatch - HKV: {hkv_evicted_count}, Simulator: {sim_evicted_count}"

        # Verify net change consistency
        assert (
            hkv_net_change == sim_net_change
        ), f"Iteration {iteration}: Net change mismatch - HKV: {hkv_net_change}, Simulator: {sim_net_change}"

        # Record cumulative counts
        total_hkv_evictions += hkv_evicted_count
        total_sim_evictions += sim_evicted_count

        # Regular consistency checks
        if iteration % comparison_interval == 0 or iteration == num_iteration - 1:
            print(f"\n=== Status Check at iteration {iteration} ===")

            # Get all keys, values and scores from HKV table
            hkv_keys, hkv_values, hkv_scores = export_all_keys_values_scores(
                table, device
            )

            # Get all keys and frequencies from simulator
            sim_all_keys = lfu_simulator.get_all_keys()

            # Compare table sizes
            hkv_size = len(hkv_keys)
            sim_size = lfu_simulator.size()

            print(f"Table sizes - HKV: {hkv_size}, Simulator: {sim_size}")
            assert hkv_size == sim_size

            # Calculate size difference
            size_diff = abs(hkv_size - sim_size)
            size_diff_ratio = (
                size_diff / max(hkv_size, sim_size, 1)
                if max(hkv_size, sim_size) > 0
                else 0
            )

            print(f"Size difference: {size_diff} ({size_diff_ratio:.3%})")

            # Compare frequency consistency of common keys
            hkv_key_set = set(hkv_keys.tolist())
            common_keys = hkv_key_set & sim_all_keys

            print(
                f"Common keys: {len(common_keys)} out of {len(hkv_key_set | sim_all_keys)} total unique keys"
            )

            assert set(hkv_key_set) == set(sim_all_keys)

            if len(common_keys) > 0:
                # Check frequency consistency of common keys
                hkv_key_to_score = {}
                for i, key in enumerate(hkv_keys):
                    hkv_key_to_score[key.item()] = hkv_scores[i].item()

                frequency_mismatches = 0
                for key in common_keys:
                    hkv_freq = hkv_key_to_score[key]
                    sim_freq = lfu_simulator.get_key_frequency(key)
                    if sim_freq is None or hkv_freq != sim_freq:
                        frequency_mismatches += 1
                assert frequency_mismatches == 0

                frequency_consistency_rate = (
                    len(common_keys) - frequency_mismatches
                ) / len(common_keys)
                print(
                    f"Frequency consistency: {frequency_consistency_rate:.3f} ({len(common_keys) - frequency_mismatches}/{len(common_keys)})"
                )

            print(f"Cumulative HKV evictions: {total_hkv_evictions}")
            print(f"Cumulative Simulator evictions: {total_sim_evictions}")

    # Final check
    print(f"\n=== Final LFU Test Results ===")
    final_hkv_keys, final_hkv_values, final_hkv_scores = export_all_keys_values_scores(
        table, device
    )
    final_sim_size = lfu_simulator.size()
    final_hkv_size = len(final_hkv_keys)

    print(f"Final HKV size: {final_hkv_size}")
    print(f"Final Simulator size: {final_sim_size}")
    assert final_hkv_size == final_sim_size

    # Calculate final size difference
    final_size_diff = abs(final_hkv_size - final_sim_size)
    final_size_diff_ratio = (
        final_size_diff / max(final_hkv_size, final_sim_size, 1)
        if max(final_hkv_size, final_sim_size) > 0
        else 0
    )

    print(f"Final size difference: {final_size_diff} ({final_size_diff_ratio:.3%})")

    # Final frequency consistency check
    if final_hkv_size > 0:
        final_hkv_key_set = set(final_hkv_keys.tolist())
        final_sim_all_keys = lfu_simulator.get_all_keys()
        final_common_keys = final_hkv_key_set & final_sim_all_keys

        assert set(final_hkv_key_set) == set(final_sim_all_keys)

        print(
            f"Final common keys: {len(final_common_keys)} out of {len(final_hkv_key_set | final_sim_all_keys)} total unique keys"
        )

        if len(final_common_keys) > 0:
            # Check frequency consistency of common keys
            final_hkv_key_to_score = {}
            for i, key in enumerate(final_hkv_keys):
                final_hkv_key_to_score[key.item()] = final_hkv_scores[i].item()

            final_frequency_mismatches = 0
            for key in final_common_keys:
                hkv_freq = final_hkv_key_to_score[key]
                sim_freq = lfu_simulator.get_key_frequency(key)
                if sim_freq is None or hkv_freq != sim_freq:
                    final_frequency_mismatches += 1

            assert final_frequency_mismatches == 0

            final_frequency_consistency_rate = (
                len(final_common_keys) - final_frequency_mismatches
            ) / len(final_common_keys)
            print(
                f"Final frequency consistency: {final_frequency_consistency_rate:.3f}"
            )

    print(f"\n Summary:")
    print(f"   Total operations: {num_iteration}")
    print(f"   Total HKV evictions: {total_hkv_evictions}")
    print(f"   Total simulator evictions: {total_sim_evictions}")
    print(
        f"   Passed: All eviction count and net change assertions passed - LFU insertion/eviction logic is consistent"
    )
