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
import itertools
import math
from typing import List

import torch
from configs import (
    InferenceEmbeddingConfig,
    KVCacheMetadata,
    RankingConfig,
    copy_kvcache_metadata,
    get_inference_hstu_config,
    get_kvcache_config,
)
from datasets.utils import FeatureConfig
from modules.inference_dense_module import InferenceDenseModule, copy_jagged_metadata
from modules.jagged_data import JaggedData

_item_fea_name = "item_feat"
_item_vocab_size = 10000
_action_fea_name = "act_feat"
_action_vocab_size = 128
_context_fea_names: List[str] = []
_context_emb_size = 1000
_hstu_cudagraph_configs = {
    "batch_size": [1, 2, 4, 8, 16],
    "length_per_sequence": [384, 512, 768, 1280, 2304, 4352],
}


def benchmark_model(
    # model config
    embedding_dim,
    num_layers,
    num_heads,
    head_dim,
    dtype,
    use_cudagraph,
    # dataset config
    max_sequence_length,
    max_num_candidates,
    num_contextual_features,
    max_contextual_seqlen,
    # inference config
    max_batch_size,
    # kvcache config
    page_size,
    num_pages,
    offload_chunksize,
):
    feature_configs = [
        FeatureConfig(
            feature_names=[_item_fea_name, _action_fea_name],
            max_item_ids=[_item_vocab_size - 1, _action_vocab_size - 1],
            max_sequence_length=max_sequence_length,
            is_jagged=False,
        ),
    ]
    total_max_seqlen = sum(
        [fc.max_sequence_length * len(fc.feature_names) for fc in feature_configs]
    )

    hstu_config = get_inference_hstu_config(
        hidden_size=embedding_dim,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=total_max_seqlen,
    )

    pages_in_primary_pool = (
        math.ceil(math.ceil(max_batch_size * total_max_seqlen / page_size) / 10240)
        * 10240
    )
    pages_in_primary_pool = max(num_pages, pages_in_primary_pool)
    reserved_pages = max_batch_size * total_max_seqlen / page_size
    total_pages = pages_in_primary_pool + reserved_pages
    cache_page_gmem = 2 * page_size * num_heads * head_dim
    kvcache_gmem = total_pages * cache_page_gmem * num_layers
    kvcache_gmem *= 4 if dtype == torch.float32 else 2
    kvcache_gmem /= 1024.0**3
    print("[[KVCache]] allocated {0} GiB".format(kvcache_gmem))

    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=pages_in_primary_pool,
        page_size=page_size,
        offload_chunksize=offload_chunksize,
    )
    emb_configs = [
        InferenceEmbeddingConfig(
            feature_names=_context_fea_names + [_item_fea_name]
            if max_contextual_seqlen > 0
            else [_item_fea_name],
            table_name="item",
            vocab_size=_item_vocab_size,
            dim=embedding_dim,
            use_dynamicemb=True,
        ),
        InferenceEmbeddingConfig(
            feature_names=[_action_fea_name],
            table_name="t_" + _action_fea_name,
            vocab_size=_action_vocab_size,
            dim=embedding_dim,
            use_dynamicemb=False,
        ),
    ]
    num_tasks = 3
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=[128, 10, 1],
        num_tasks=num_tasks,
    )
    bench_model = InferenceDenseModule(
        hstu_config=hstu_config,
        kvcache_config=kv_cache_config,
        task_config=task_config,
        use_cudagraph=use_cudagraph,
        cudagraph_configs=_hstu_cudagraph_configs,
    )
    if dtype == torch.bfloat16:
        bench_model.bfloat16()
    bench_model.eval()

    return bench_model


def test_input(
    num_input_sets,
    # benchmark config
    batch_size,
    total_history_length,
    new_history_length,
    num_targets,  # total_history_length + num_targets <= max_seqlen (already doubled)
    # global config
    max_seqlen,
    max_num_candidates,
    contextual_max_seqlen,
    num_layers,
    embedding_dim,
    page_size,
    kvcache_num_pages,
    dtype,
):
    # jd const
    seq_len = new_history_length + num_targets

    # kvcache const
    num_pages = math.ceil(total_history_length / page_size)
    batch_pages = batch_size * num_pages
    last_page_len = total_history_length % page_size
    last_page_len = 32 if last_page_len == 0 else last_page_len

    seqlen = torch.full((batch_size,), seq_len)
    seqlen_offsets = torch.arange(batch_size + 1) * seq_len
    num_candidates = torch.full((batch_size,), num_targets)
    num_candidates_offsets = torch.arange(batch_size + 1) * num_targets

    kvcache_page_indptr = torch.arange(batch_size + 1) * num_pages
    kvcache_last_page_len = torch.full((batch_size,), last_page_len)
    batch_indices = (
        torch.tile(torch.arange(batch_size), dims=(new_history_length, 1))
        .T.detach()
        .clone()
        .contiguous()
        .view(-1)
    )
    positions = torch.tile(
        torch.arange(total_history_length - new_history_length, total_history_length),
        dims=(batch_size,),
    )
    new_history_nnz = batch_size * new_history_length
    new_history_nnz_cuda = torch.tensor([new_history_nnz])
    kv_seqlen_offsets = torch.arange(batch_size + 1) * (
        total_history_length + num_targets
    )

    num_input_sets = max(num_input_sets, 2)
    input_lists = []
    for i in range(num_input_sets):
        hidden_states = torch.randn(
            (batch_size * seq_len, embedding_dim), dtype=dtype
        ).cuda()
        jagged_metadata = JaggedData(
            values=hidden_states,
            seqlen=seqlen.int().cuda(),
            seqlen_offsets=seqlen_offsets.int().cuda(),
            max_seqlen=max_seqlen,
            max_num_candidates=max_num_candidates,
            num_candidates=num_candidates.int().cuda(),
            num_candidates_offsets=num_candidates_offsets.int().cuda(),
            contextual_max_seqlen=0,
            contextual_seqlen=None,
            contextual_seqlen_offsets=None,
            has_interleaved_action=True,
        )
        # use input offset to force reading different kvcache pages
        kvcache_page_ids = torch.randperm(batch_pages) + i * batch_pages
        kvcache_page_ids = kvcache_page_ids % kvcache_num_pages
        kvcache_metadata = KVCacheMetadata(
            kv_indices=kvcache_page_ids.int().cuda(),
            kv_indptr=kvcache_page_indptr.int().cuda(),
            kv_last_page_len=kvcache_last_page_len.int().cuda(),
            batch_indices=batch_indices.int().cuda(),
            position=positions.int().cuda(),
            new_history_nnz=new_history_nnz,
            new_history_nnz_cuda=new_history_nnz_cuda.int().cuda(),
            total_history_offsets=kv_seqlen_offsets.int().cuda(),
            onload_history_kv_events=[torch.cuda.Event() for _ in range(num_layers)],
        )
        input_lists.append(
            (
                hidden_states,
                jagged_metadata,
                kvcache_metadata,
            )
        )

    return input_lists


def run_single_bench(
    model,
    out_buffer,
    # benchmark config
    batch_size,
    total_history_length,
    new_history_length,
    num_targets,
    use_cudagraph,
):
    num_input_sets = 2
    input_list = test_input(
        num_input_sets,
        # benchmark config
        batch_size,
        total_history_length,
        new_history_length,
        num_targets,  # total_history_length + num_targets <= max_seqlen (already doubled)
        # global config
        model._gpu_kv_cache_manager.max_seq_len,
        num_targets,
        0,
        model._hstu_config.num_layers,
        model._embedding_dim,
        model._gpu_kv_cache_manager.page_size,
        model._gpu_kv_cache_manager.num_cache_pages,
        model._hidden_states.dtype,
    )

    num_warumps = 10
    num_iterations = 10

    if use_cudagraph:
        total_tokens = batch_size * (new_history_length + num_targets)

        ts_start, ts_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        torch.cuda.synchronize()
        for i in range(num_warumps):
            hidden_states, jd_metadata, kvcache_metadata = input_list[
                i % num_input_sets
            ]
            model._hidden_states[:total_tokens, ...].copy_(
                jd_metadata.values, non_blocking=True
            )
            copy_jagged_metadata(model._jagged_metadata, jd_metadata)
            copy_kvcache_metadata(model._kvcache_metadata, kvcache_metadata)
            model._hstu_block.predict(
                batch_size,
                total_tokens,
                model._hidden_states,
                model._jagged_metadata,
                model._kvcache_metadata,
            )
        torch.cuda.synchronize()

        ts_start.record()
        for i in range(num_iterations):
            hidden_states, jd_metadata, kvcache_metadata = input_list[
                i % num_input_sets
            ]
            model._hidden_states[:total_tokens, ...].copy_(
                jd_metadata.values, non_blocking=True
            )
            copy_jagged_metadata(model._jagged_metadata, jd_metadata)
            copy_kvcache_metadata(model._kvcache_metadata, kvcache_metadata)
            model._hstu_block.predict(
                batch_size,
                total_tokens,
                model._hidden_states,
                model._jagged_metadata,
                model._kvcache_metadata,
            )
        ts_end.record()
        torch.cuda.synchronize()
        time1 = ts_start.elapsed_time(ts_end)
        if time1 < 1000:
            num_iterations = num_iterations * math.ceil(1000 / time1)
            torch.cuda.synchronize()
            ts_start.record()
            for i in range(num_iterations):
                hidden_states, jd_metadata, kvcache_metadata = input_list[
                    i % num_input_sets
                ]
                model._hidden_states[:total_tokens, ...].copy_(
                    jd_metadata.values, non_blocking=True
                )
                copy_jagged_metadata(model._jagged_metadata, jd_metadata)
                copy_kvcache_metadata(model._kvcache_metadata, kvcache_metadata)
                out_data = model._hstu_block.predict(
                    batch_size,
                    total_tokens,
                    model._hidden_states,
                    model._jagged_metadata,
                    model._kvcache_metadata,
                )
                out_buffer[:total_tokens, ...].copy_(out_data, non_blocking=True)
            ts_end.record()
            torch.cuda.synchronize()

        print("time(ms)", ts_start.elapsed_time(ts_end) / num_iterations)
    else:
        total_tokens = batch_size * (new_history_length + num_targets)
        for _, jd_metadata, kvcache_metadata in input_list:
            kvcache_metadata
            kvcache_metadata.onload_history_kv_buffer = (
                model._kvcache_metadata.onload_history_kv_buffer[:]
            )
            kvcache_metadata.onload_history_kv_events = (
                model._kvcache_metadata.onload_history_kv_events[:]
            )
            kvcache_metadata.kv_cache_table = model._kvcache_metadata.kv_cache_table[:]

        ts_start, ts_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        torch.cuda.synchronize()
        for i in range(num_warumps):
            hidden_states, jd_metadata, kvcache_metadata = input_list[
                i % num_input_sets
            ]
            model._hstu_block.predict(
                batch_size,
                total_tokens,
                hidden_states,
                jd_metadata,
                kvcache_metadata,
            )
        torch.cuda.synchronize()

        ts_start.record()
        for i in range(num_iterations):
            hidden_states, jd_metadata, kvcache_metadata = input_list[
                i % num_input_sets
            ]
            model._hstu_block.predict(
                batch_size,
                total_tokens,
                hidden_states,
                jd_metadata,
                kvcache_metadata,
            )
        ts_end.record()
        torch.cuda.synchronize()
        time1 = ts_start.elapsed_time(ts_end)
        if time1 < 1000:
            num_iterations = num_iterations * math.ceil(1000 / time1)
            torch.cuda.synchronize()
            ts_start.record()
            for i in range(num_iterations):
                hidden_states, jd_metadata, kvcache_metadata = input_list[
                    i % num_input_sets
                ]
                model._hstu_block.predict(
                    batch_size,
                    total_tokens,
                    hidden_states,
                    jd_metadata,
                    kvcache_metadata,
                )
            ts_end.record()
            torch.cuda.synchronize()

        print("time(ms)", ts_start.elapsed_time(ts_end) / num_iterations)


def run_benchmark():
    kwargs = {
        # model config
        "embedding_dim": 1024,
        "num_layers": 8,
        "num_heads": 4,
        "head_dim": 256,
        "dtype": torch.bfloat16,
        "use_cudagraph": True,
        # dataset config
        "max_sequence_length": 4096,
        "max_num_candidates": 256,
        "num_contextual_features": 0,
        "max_contextual_seqlen": 0,
        # inference config
        "max_batch_size": 16,
        # kvcache config
        "page_size": 32,
        "num_pages": 10240,
        "offload_chunksize": 1024,
    }
    print()

    with torch.inference_mode():
        model = benchmark_model(**kwargs)
        out_buffer = torch.empty_like(model._hidden_states)

        print(
            "Testcase: (batch_size, total_history_length, new_history_length, num_targets)"
        )
        for (
            batch_size,
            total_history_length,
            new_history_length,
            num_targets,
        ) in itertools.product(
            # batch_size
            [1, 2, 4, 8],
            # total_history_length
            [128, 256, 512, 1024, 2048, 4096],
            # new_history_length
            [128, 256, 512, 1024, 2048, 4096],
            # num_targets
            [256],
        ):
            # skips
            if new_history_length > total_history_length:
                continue
            if (
                new_history_length + num_targets
                > model._gpu_kv_cache_manager.max_seq_len
            ):
                print("too large input length:", new_history_length + num_targets)
                continue
            if batch_size > kwargs["max_batch_size"]:
                break

            print(
                "test case",
                (
                    batch_size,
                    total_history_length,
                    new_history_length,
                    num_targets,
                ),
            )

            run_single_bench(
                model,
                out_buffer,
                batch_size,
                total_history_length,
                new_history_length,
                num_targets,
                kwargs["use_cudagraph"],
            )
            print()


if __name__ == "__main__":
    run_benchmark()
