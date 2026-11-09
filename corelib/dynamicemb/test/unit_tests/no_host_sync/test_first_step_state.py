# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""What a module builds before its first step, so that the step does not.

The probe measures after a warm-up, so a one-time cost that the first step pays is
invisible to it. ``read_now`` kept its pinned landing buffer and event per thread and
device and made them on the first read, a pinned allocation in the middle of a step; a
module now makes them when it is built (``sync_free.reserve_read_scratch``), on the thread
that builds it. (The default LruLfu evict module is loaded into the table's device when the
table is built too, ``score_jit.load_lex_module``; that is a driver-side cache a test
cannot see from Python, and it is checked on a GPU by running ``cache_storage_lrulfu``'s
first step under nsys.)
"""
import threading

import pytest

try:
    import torch
except ImportError:
    pytest.skip("needs torch", allow_module_level=True)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _in_a_fresh_thread(fn):
    """Run ``fn`` on a thread that has made no pinned read buffer, and return its result
    (the thread-local scratch of ``sync_free`` is per thread)."""
    out = {}

    def run():
        try:
            out["result"] = fn()
        except BaseException as error:  # reported by the caller, not lost in the thread
            out["error"] = error

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if "error" in out:
        raise out["error"]
    return out["result"]


@cuda_only
def test_reserve_read_scratch_makes_the_buffer_once_and_read_now_reuses_it(sf):
    device = torch.device("cuda", torch.cuda.current_device())

    def work():
        torch.cuda.set_device(device)
        assert not getattr(sf._TLS, "scratch", None), "a new thread has no buffer"
        sf.reserve_read_scratch(device)
        buf, event = sf._TLS.scratch[device]
        (read,) = sf.read_now(torch.arange(4, device=device))
        sf.reserve_read_scratch(device)
        again, same_event = sf._TLS.scratch[device]
        return (
            read.tolist(),
            buf.is_pinned(),
            buf.data_ptr() == again.data_ptr(),
            event is same_event,
        )

    values, pinned, same_buffer, same_event = _in_a_fresh_thread(work)
    assert values == [0, 1, 2, 3]
    assert pinned and same_buffer and same_event


@cuda_only
def test_a_module_reserves_the_read_scratch_when_it_is_built(sf):
    from dynamicemb import DynamicEmbPoolingMode, DynamicEmbTableOptions, EmbOptimType
    from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2

    device = torch.device("cuda", torch.cuda.current_device())

    def build():
        torch.cuda.set_device(device)
        assert not getattr(sf._TLS, "scratch", None), "a new thread has no buffer"
        options = DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=device.index,
            dim=8,
            max_capacity=1 << 12,
            local_hbm_for_values=1024**3,
        )
        BatchedDynamicEmbeddingTablesV2(
            table_options=[options],
            table_names=["t_0"],
            feature_table_map=[0],
            pooling_mode=DynamicEmbPoolingMode.SUM,
            optimizer=EmbOptimType.SGD,
            learning_rate=0.0,
            stochastic_rounding=False,
        )
        return device in sf._TLS.scratch

    assert _in_a_fresh_thread(build), "the module did not make the read buffer"


def _spy_on_load_lex_module(monkeypatch):
    """The bytes the allocator holds each time a table loads the default evict module."""
    from dynamicemb.jit import score_jit

    seen = []
    real = score_jit.load_lex_module

    def spy():
        seen.append(torch.cuda.memory_allocated())
        real()

    monkeypatch.setattr(score_jit, "load_lex_module", spy)
    return seen


def _table(policy):
    from dynamicemb.scored_hashtable import ScoreSpec, get_scored_table
    from dynamicemb_extensions import ScorePolicy

    return get_scored_table(
        capacity=[256],
        bucket_capacity=128,
        key_type=torch.int64,
        score_specs=[
            ScoreSpec(
                name="score", policy=getattr(ScorePolicy, policy), is_reduction=True
            )
        ],
    )


@cuda_only
def test_a_table_loads_the_default_evict_module_after_its_own_allocations(
    sf, monkeypatch
):
    """Loading the module into the context does nothing before the device has one, and
    the first CUDA call of a thread can be the table's own allocation: the table loads it
    last, when its storage is there (the first insert of a step would otherwise be the one
    that loads it)."""
    seen = _spy_on_load_lex_module(monkeypatch)
    before = torch.cuda.memory_allocated()
    table = _table("LRU_LFU")
    assert (
        len(seen) == 1
    ), "the compound score is the one that evicts through the module"
    assert seen[0] - before >= table.storage_bytes_, "the storage came after the load"


@cuda_only
def test_a_table_with_a_single_score_word_does_not_load_the_evict_module(
    sf, monkeypatch
):
    seen = _spy_on_load_lex_module(monkeypatch)
    _table("ASSIGN")
    assert seen == []
