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

"""Table-level tests for the LruLfu score policy (2 AoS score words per key:
word 0 = last-access timestamp, word 1 = frequency; eviction ranks by frequency,
the timestamp serves time-based incremental dump). Scores are read back via the
gather_score_blocks primitive so the CUDA kernels are directly observable."""

import pytest
import torch
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import InsertResult, ScorePolicy


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


def _lru_lfu_table(capacity, bucket_capacity=128, key_type=torch.int64):
    return get_scored_table(
        capacity=[capacity],
        bucket_capacity=bucket_capacity,
        key_type=key_type,
        score_specs=[
            ScoreSpec(
                name="frequency", policy=ScorePolicy.LRU_LFU, is_reduction=True
            )
        ],
    )


def _insert(table, keys, tids, freq, policy=ScorePolicy.LRU_LFU):
    n = keys.numel()
    score_out = torch.empty(n, dtype=torch.int64, device=keys.device)
    insert_results = torch.empty(
        n, dtype=table.result_type, device=keys.device
    ).fill_(InsertResult.INIT.value)
    indices = table.insert(
        keys,
        tids,
        ScoreArg(name="frequency", value=freq, policy=policy),
        insert_results,
        score_out=score_out,
    )
    return indices, score_out


def test_lru_lfu_num_scores():
    """LruLfu occupies two physical score words per key."""
    table = _lru_lfu_table(4096)
    assert table.num_scores_ == 2


def test_lru_lfu_frequency_accumulation(current_device):
    """word 1 accumulates the frequency; word 0 is a (monotonic) timestamp; and a
    CONST lookup returns the frequency (word 1), not the timestamp (word 0)."""
    device = torch.cuda.current_device()
    table = _lru_lfu_table(4096)
    n = 100
    keys = torch.arange(1000, 1000 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    freq = torch.ones(n, dtype=torch.uint64, device=device)

    indices, score_out = _insert(table, keys, tids, freq)
    assert torch.all(score_out == 1), "frequency after first insert should be 1"

    blocks = table.gather_score_blocks(0, indices)  # [n, 2] = (timestamp, freq)
    assert tuple(blocks.shape) == (n, 2)
    ts0 = blocks[:, 0].clone()
    assert torch.all(blocks[:, 1] == 1), "word 1 (frequency) should be 1"
    assert torch.all(ts0 > 0), "word 0 (timestamp) should be a device timer value"

    # Each LruLfu lookup is an access: accumulate frequency and re-stamp time.
    out2, founds, _ = table.lookup(
        keys, tids, ScoreArg(name="frequency", value=freq, policy=ScorePolicy.LRU_LFU)
    )
    assert torch.all(founds)
    assert torch.all(out2 == 2), "frequency after one lookup should be 2"

    out3, _, _ = table.lookup(
        keys, tids, ScoreArg(name="frequency", value=freq, policy=ScorePolicy.LRU_LFU)
    )
    assert torch.all(out3 == 3)

    # CONST (eval) lookup must return the reduction score == frequency (word 1),
    # NOT word 0 (the timestamp).
    out_const, _, _ = table.lookup(
        keys, tids, ScoreArg(name="frequency", value=None, policy=ScorePolicy.CONST)
    )
    assert torch.all(
        out_const == 3
    ), "CONST lookup must return frequency (word 1), not the timestamp"

    blocks2 = table.gather_score_blocks(0, indices)
    assert torch.all(blocks2[:, 1] == 3)
    assert torch.all(
        blocks2[:, 0] >= ts0
    ), "timestamp (word 0) must be non-decreasing after further accesses"


def test_lru_lfu_eviction_by_frequency(current_device):
    """Eviction ranks by frequency (word 1): frequently-accessed keys survive even
    though newly inserted keys have more recent timestamps. Exercises the 2-score
    reduce() path."""
    device = torch.cuda.current_device()
    bc = 128
    table = _lru_lfu_table(bc, bucket_capacity=bc)  # single bucket: all keys compete

    n = bc
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    ones = torch.ones(n, dtype=torch.uint64, device=device)
    _insert(table, keys, tids, ones)

    # Boost the frequency of the first half ("hot") via repeated accesses; the
    # second half ("cold") stays at frequency 1.
    hot = keys[: n // 2]
    hot_tids = tids[: n // 2]
    hot_freq = torch.ones(n // 2, dtype=torch.uint64, device=device)
    for _ in range(10):
        table.lookup(
            hot,
            hot_tids,
            ScoreArg(name="frequency", value=hot_freq, policy=ScorePolicy.LRU_LFU),
        )

    # Insert a bucket's worth of brand-new keys (frequency 1, newest timestamps)
    # to force eviction.
    new_keys = torch.arange(100000, 100000 + n, dtype=torch.int64, device=device)
    new_tids = torch.zeros(n, dtype=torch.int64, device=device)
    new_freq = torch.ones(n, dtype=torch.uint64, device=device)
    ir = torch.empty(n, dtype=table.result_type, device=device).fill_(
        InsertResult.INIT.value
    )
    _, num_evicted, evicted_keys, _, _, _ = table.insert_and_evict(
        new_keys,
        new_tids,
        ScoreArg(name="frequency", value=new_freq, policy=ScorePolicy.LRU_LFU),
        ir,
    )

    assert num_evicted > 0, "eviction should have occurred in the full bucket"
    evicted_set = set(int(k) for k in evicted_keys.tolist())
    assert evicted_set.isdisjoint(
        set(int(k) for k in hot.tolist())
    ), "high-frequency (hot) keys must not be evicted by LFU"

    # The hot keys must still be present.
    _, founds, _ = table.lookup(
        hot, hot_tids, ScoreArg(name="frequency", value=None, policy=ScorePolicy.CONST)
    )
    assert torch.all(founds), "high-frequency keys must survive LFU eviction"


def test_lru_lfu_gather_scatter_roundtrip(current_device):
    """gather_score_blocks -> scatter_score_blocks -> gather_score_blocks preserves
    both score words (used by dump/load)."""
    device = torch.cuda.current_device()
    n = 50
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    freq = torch.arange(1, 1 + n, dtype=torch.uint64, device=device)  # varied

    src = _lru_lfu_table(4096)
    idx_src, _ = _insert(src, keys, tids, freq)
    blocks = src.gather_score_blocks(0, idx_src)

    dst = _lru_lfu_table(4096)
    # Place keys without touching scores (CONST), then restore the full block.
    idx_dst, _ = _insert(dst, keys, tids, None, policy=ScorePolicy.CONST)
    dst.scatter_score_blocks(0, idx_dst, blocks)
    assert torch.equal(
        dst.gather_score_blocks(0, idx_dst), blocks
    ), "scatter then gather must round-trip both score words"


def test_lru_lfu_copy_score_blocks_roundtrip(current_device):
    """copy_score_blocks_from copies all score words between tables of different
    capacity (the rehash primitive)."""
    device = torch.cuda.current_device()
    n = 50
    keys = torch.arange(1, 1 + n, dtype=torch.int64, device=device)
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    freq = torch.arange(1, 1 + n, dtype=torch.uint64, device=device)

    src = _lru_lfu_table(4096)
    idx_src, _ = _insert(src, keys, tids, freq)
    blocks = src.gather_score_blocks(0, idx_src)

    # Rehash target: a larger table (keys re-hash into different buckets).
    dst = _lru_lfu_table(8192)
    idx_dst, _ = _insert(dst, keys, tids, None, policy=ScorePolicy.CONST)
    dst.copy_score_blocks_from(src, 0, idx_src, idx_dst)
    assert torch.equal(
        dst.gather_score_blocks(0, idx_dst), blocks
    ), "copy_score_blocks_from must preserve both score words across tables"
