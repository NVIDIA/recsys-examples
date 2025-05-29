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
from typing import Dict, Set, Tuple, List, Optional
from typing import Dict, Set, Tuple, List, Optional

import pytest
import torch
from dynamicemb import DynamicEmbCheckMode, DynamicEmbInitializerArgs, dyn_emb_to_torch
from dynamicemb_extensions import (
    DynamicEmbDataType,
    DynamicEmbTable,
    EvictStrategy,
    InitializerArgs,
    count_matched,
    device_timestamp,
    dyn_emb_rows,
    dyn_emb_capacity,
    dyn_emb_cols,
    export_batch,
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
            # LFU 模式：使用单调递增的分数
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
            # LFU 和其他模式：返回单调递增的分数
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
            # LFU 应给merlin hashtable频率增量1 
            return 1 
        else:
            score = self.step_
            self.step_ += 1
            return score


class LFUSimulator:
    """Host端LFU策略模拟器，用于与HKV table结果对比"""
    def __init__(self, initial_capacity: int, dim: int):
        self.capacity = initial_capacity
        self.dim = dim
        
        # key -> (value, frequency, insertion_order)
        self.table: Dict[int, Tuple[torch.Tensor, int, int]] = {}
        self.insertion_counter = 0
        
    def size(self) -> int:
        return len(self.table)
    
    def find_or_insert(self, keys: torch.Tensor, values: torch.Tensor, score: int = 1) -> Tuple[torch.Tensor, List[int]]:
        """
        模拟find_or_insert操作
        返回: (found_mask, evicted_keys)
        """
        batch_size = keys.size(0)
        found_mask = torch.zeros(batch_size, dtype=torch.bool)
        evicted_keys = []
        
        # 计算需要插入的新keys数量
        new_keys_count = 0
        for key in keys:
            if key.item() not in self.table:
                new_keys_count += 1
        
        
        for i, key in enumerate(keys):
            key_item = key.item()
            self.insertion_counter += 1
            
            if key_item in self.table:
                # Key已存在，更新频率和访问时间
                old_value, old_freq, _ = self.table[key_item]
                new_freq = old_freq + score
                self.table[key_item] = (old_value.clone(), new_freq, self.insertion_counter)
                values[i] = old_value
                found_mask[i] = True
            else:
                # Key不存在，需要插入
                if len(self.table) < self.capacity:
                    # 有空间，直接插入
                    self.table[key_item] = (values[i].clone(), score, self.insertion_counter)
                else:
                    # 需要驱逐，找到频率最低的key
                    min_freq = float('inf')
                    min_key = None
                    min_time = float('inf')
                    
                    for k, (v, freq, time) in self.table.items():
                        # LFU策略：优先驱逐频率最低的，频率相同时驱逐最早访问的
                        if freq < min_freq or (freq == min_freq and time < min_time):
                            min_freq = freq
                            min_key = k
                            min_time = time
                    
                    # 驱逐最少使用的key
                    if min_key is not None:
                        del self.table[min_key]
                        evicted_keys.append(min_key)
                    
                    # 插入新key
                    self.table[key_item] = (values[i].clone(), score, self.insertion_counter)
                
                found_mask[i] = False
        
        return found_mask, evicted_keys
    
    def find(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        模拟find操作，不更新频率
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
    
    def insert_and_evict(self, keys: torch.Tensor, values: torch.Tensor, score: int = 1) -> Tuple[List[int], int]:
        """
        模拟insert_and_evict操作
        返回: (evicted_keys, num_evicted)
        """
        evicted_keys = []
        
        for i, key in enumerate(keys):
            key_item = key.item()
            self.insertion_counter += 1
            
            if key_item in self.table:
                # Key已存在，更新频率
                old_value, old_freq, _ = self.table[key_item]
                new_freq = old_freq + score
                self.table[key_item] = (values[i].clone(), new_freq, self.insertion_counter)
            else:
                # Key不存在，需要插入
                if len(self.table) >= self.capacity:
                    # 需要驱逐
                    min_freq = float('inf')
                    min_key = None
                    min_time = float('inf')
                    
                    for k, (v, freq, time) in self.table.items():
                        if freq < min_freq or (freq == min_freq and time < min_time):
                            min_freq = freq
                            min_key = k
                            min_time = time
                    
                    if min_key is not None:
                        del self.table[min_key]
                        evicted_keys.append(min_key)
                
                # 插入新key
                self.table[key_item] = (values[i].clone(), score, self.insertion_counter)
        
        return evicted_keys, len(evicted_keys)
    
    def get_keys_by_score_threshold(self, min_score: int) -> Set[int]:
        """获取频率大于等于min_score的所有keys"""
        return {k for k, (v, freq, time) in self.table.items() if freq >= min_score}
    
    def get_all_keys(self) -> Set[int]:
        """获取所有keys"""
        return set(self.table.keys())
    
    def get_key_frequency(self, key: int) -> Optional[int]:
        """获取指定key的频率"""
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
    "evict_strategy", [EvictStrategy.KLru, EvictStrategy.KCustomized, EvictStrategy.KLfu]
)
# @pytest.mark.parametrize(
#     "evict_strategy", [EvictStrategy.KLfu]
# )
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
            # 插入驱逐后的tablesize - 原本的table szize == 新增的值减去驱逐的值 
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
    导出HKV table中所有的keys、values和scores
    返回: (keys, values, scores) 所有都在CPU上
    """
    capacity = dyn_emb_capacity(table)
    current_size = dyn_emb_rows(table)
    dim = dyn_emb_cols(table)  # 从table获取实际维度
    
    if current_size == 0:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.uint64)
    
    key_dtype = torch.int64  # key类型
    value_dtype = torch.float32  # value类型
    score_dtype = torch.uint64  # score类型
    
    batch_size = min(65536, capacity)
    
    all_keys = []
    all_values = []
    all_scores = []
    
    offset = 0
    while offset < capacity:
        # 准备batch tensors
        keys = torch.empty(batch_size, dtype=key_dtype, device=device)
        values = torch.empty(batch_size * dim, dtype=value_dtype, device=device)
        scores = torch.empty(batch_size, dtype=score_dtype, device=device)
        d_counter = torch.zeros(1, dtype=torch.uint64, device=device)
        
        # 调用export_batch
        export_batch(table, batch_size, offset, d_counter, keys, values, scores)
        
        # 获取实际返回的数量
        actual_count = d_counter.cpu().item()
        
        if actual_count > 0:
            # 只保留有效的数据
            valid_keys = keys[:actual_count].cpu()
            valid_values = values[:actual_count * dim].view(actual_count, dim).cpu()
            valid_scores = scores[:actual_count].cpu()
            
            all_keys.append(valid_keys)
            all_values.append(valid_values)
            all_scores.append(valid_scores)
        
        offset += batch_size
        
        # 如果这个batch没有返回任何数据，说明已经遍历完了
        if actual_count == 0:
            break
    
    if all_keys:
        return torch.cat(all_keys), torch.cat(all_values), torch.cat(all_scores)
    else:
        return torch.empty(0, dtype=key_dtype), torch.empty(0, dim, dtype=value_dtype), torch.empty(0, dtype=score_dtype)

@pytest.mark.parametrize(
    "evict_strategy", [EvictStrategy.KLfu]
)
@pytest.mark.parametrize(
    "bucket_capacity, batch, capacity, num_iteration, dump_interval",
    [
        pytest.param(128, 3, 512, 5, 1, id="Small scale LFU vs HKV"),
        # pytest.param(
        #     128, 128, 512 * 1024, 8192, 1024, id="Never evict keys from current batch"
        # ),
        # pytest.param(128, 65536, 512 * 1024, 32, 8),
        pytest.param(
            128, 1024 + 13, 1024, 12, 3, id="Always evict keys from current batch"
        ),
        pytest.param(
            128, 1024 + 13, 2048, 32, 3, id="Always evict keys from current batch"
        ),
        # pytest.param(
        #     128, 1024, 4 * 1024, 32, 4, id="Always evict keys from last dump_interval"
        # ),
        # pytest.param(512, 512, 512 * 1024, 2048, 256, id="Different bucket capacity"),
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
    dump_interval,
):
    print(f"\n{request.node.name} - LFU Algorithm Consistency Test")
    print(f"Testing high-level LFU algorithm behavior between HKV table and simulator")
    
    # 初始化配置
    ext_option.init_capacity = capacity
    ext_option.max_capacity = capacity
    # 使用capacity作为bucket_capacity，因为buckets的数量大于1的时候，HKVtable的行为不可模拟
    ext_option.bucket_capacity = capacity 
    ext_option.evict_strategy = evict_strategy
    ext_option.num_of_buckets_par_alloc = capacity //  ext_option.bucket_capacity

    assert ext_option.dim * ext_option.max_capacity <= ext_option.local_hbm_for_values

    # 初始化HKV table
    table = DynamicEmbTable(*astuple(ext_option))
    device = torch.device(f"cuda:{ext_option.device_id}")
    score_adaptor = ScoreAdaptor(evict_strategy, score_type, device)

    # 初始化LFU模拟器
    lfu_simulator = LFUSimulator(capacity, ext_option.dim)
    
    print(f"Capacity: {capacity}, Batch: {batch}, Total operations: {num_iteration}")
    print(f"Testing LFU insertion/eviction count consistency between HKV table and simulator")

    # 逐步测试，验证每次操作的插入和驱逐数量
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
        score = score_adaptor.score()  # LFU返回1
        
        # 记录操作前的大小
        hkv_size_before = dyn_emb_rows(table)
        sim_size_before = lfu_simulator.size()
        
        # HKV table操作：insert_and_evict
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
        
        # LFU模拟器操作：insert_and_evict
        sim_keys = keys.cpu()
        sim_values = values.cpu()
        sim_evicted_keys, sim_evicted_count = lfu_simulator.insert_and_evict(sim_keys, sim_values, score)
        
        # 记录操作后的大小
        hkv_size_after = dyn_emb_rows(table)
        sim_size_after = lfu_simulator.size()
        
        # 计算插入和驱逐数量
        hkv_net_change = hkv_size_after - hkv_size_before
        sim_net_change = sim_size_after - sim_size_before
        

        # 验证驱逐数量一致性
        assert hkv_evicted_count == sim_evicted_count, f"Iteration {iteration}: Eviction count mismatch - HKV: {hkv_evicted_count}, Simulator: {sim_evicted_count}"
        
        # 验证net change一致性
        assert hkv_net_change == sim_net_change, f"Iteration {iteration}: Net change mismatch - HKV: {hkv_net_change}, Simulator: {sim_net_change}"
        
        # 记录累计数量
        total_hkv_evictions += hkv_evicted_count
        total_sim_evictions += sim_evicted_count
        
        # 定期进行一致性检查（不强制要求通过）
        if iteration % comparison_interval == 0 or iteration == num_iteration - 1:
            print(f"\n=== Status Check at iteration {iteration} ===")
            
            # 获取HKV table中所有键值和scores
            hkv_keys, hkv_values, hkv_scores = export_all_keys_values_scores(table, device)
            
            # 获取模拟器中所有键值和频率
            sim_all_keys = lfu_simulator.get_all_keys()
            
            # 比较table大小
            hkv_size = len(hkv_keys)
            sim_size = lfu_simulator.size()
            
            print(f"Table sizes - HKV: {hkv_size}, Simulator: {sim_size}")
            
            # 计算大小差异（不assert）
            size_diff = abs(hkv_size - sim_size)
            size_diff_ratio = size_diff / max(hkv_size, sim_size, 1) if max(hkv_size, sim_size) > 0 else 0
            
            print(f"Size difference: {size_diff} ({size_diff_ratio:.3%})")
            
            # 比较共同键的频率一致性（不assert）
            hkv_key_set = set(hkv_keys.tolist())
            common_keys = hkv_key_set & sim_all_keys
            
            print(f"Common keys: {len(common_keys)} out of {len(hkv_key_set | sim_all_keys)} total unique keys")
            
            if len(common_keys) > 0:
                # 检查共同键的频率一致性
                hkv_key_to_score = {}
                for i, key in enumerate(hkv_keys):
                    hkv_key_to_score[key.item()] = hkv_scores[i].item()
                
                frequency_mismatches = 0
                for key in common_keys:
                    hkv_freq = hkv_key_to_score[key]
                    sim_freq = lfu_simulator.get_key_frequency(key)
                    if sim_freq is None or hkv_freq != sim_freq:
                        frequency_mismatches += 1
                
                frequency_consistency_rate = (len(common_keys) - frequency_mismatches) / len(common_keys)
                print(f"Frequency consistency: {frequency_consistency_rate:.3f} ({len(common_keys) - frequency_mismatches}/{len(common_keys)})")
            
            print(f"Cumulative HKV evictions: {total_hkv_evictions}")
            print(f"Cumulative Simulator evictions: {total_sim_evictions}")
    
    # 最终检查
    print(f"\n=== Final LFU Test Results ===")
    final_hkv_keys, final_hkv_values, final_hkv_scores = export_all_keys_values_scores(table, device)
    final_sim_size = lfu_simulator.size()
    final_hkv_size = len(final_hkv_keys)
    
    print(f"Final HKV size: {final_hkv_size}")
    print(f"Final Simulator size: {final_sim_size}")
    
    # 计算最终大小差异（仅报告）
    final_size_diff = abs(final_hkv_size - final_sim_size)
    final_size_diff_ratio = final_size_diff / max(final_hkv_size, final_sim_size, 1) if max(final_hkv_size, final_sim_size) > 0 else 0
    
    print(f"Final size difference: {final_size_diff} ({final_size_diff_ratio:.3%})")
    
    # 最终频率一致性检查（仅报告）
    if final_hkv_size > 0:
        final_hkv_key_set = set(final_hkv_keys.tolist())
        final_sim_all_keys = lfu_simulator.get_all_keys()
        final_common_keys = final_hkv_key_set & final_sim_all_keys
        
        print(f"Final common keys: {len(final_common_keys)} out of {len(final_hkv_key_set | final_sim_all_keys)} total unique keys")
        
        if len(final_common_keys) > 0:
            # 检查共同键的频率一致性
            final_hkv_key_to_score = {}
            for i, key in enumerate(final_hkv_keys):
                final_hkv_key_to_score[key.item()] = final_hkv_scores[i].item()
            
            final_frequency_mismatches = 0
            for key in final_common_keys:
                hkv_freq = final_hkv_key_to_score[key]
                sim_freq = lfu_simulator.get_key_frequency(key)
                if sim_freq is None or hkv_freq != sim_freq:
                    final_frequency_mismatches += 1
            
            final_frequency_consistency_rate = (len(final_common_keys) - final_frequency_mismatches) / len(final_common_keys)
            print(f"Final frequency consistency: {final_frequency_consistency_rate:.3f}")
    
    print(f"\n Summary:")
    print(f"   Total operations: {num_iteration}")
    print(f"   Total HKV evictions: {total_hkv_evictions}")
    print(f"   Total simulator evictions: {total_sim_evictions}")
    print(f"   Passed: All eviction count and net change assertions passed - LFU insertion/eviction logic is consistent")
    

if __name__ == "__main__":
    test_dynamicemb_extensions(evict_strategy=EvictStrategy.KLfu, bucket_capacity=128, batch=128, capacity=512 * 1024, num_iteration=8192, dump_interval=1024)