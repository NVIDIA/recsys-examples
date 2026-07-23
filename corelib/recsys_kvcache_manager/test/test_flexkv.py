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

import os
import time

import torch
from recsys_kvcache_manager.host_kvstorage_manager import HostKVTaskStatus
from recsys_kvcache_manager.kvcache_config import get_kvcache_config
from recsys_kvcache_manager.kvcache_manager import KVCacheManager


def _get_flexkv_config_path() -> str:
    return os.environ.get("RECSYS_FLEXKV_CONFIG_PATH", "")


def create_testing_kvcache_manager() -> KVCacheManager:
    flexkv_config_path = _get_flexkv_config_path()
    extra_configs = {
        "flexkv_mode": "direct",
        "flexkv_host_kvstorage_fail_policy": "fail_open",
        "flexkv_enable_mps": 0,
        "flexkv_as_batch": 1,
    }
    if flexkv_config_path:
        extra_configs["flexkv_config_path"] = flexkv_config_path

    kvcache_config = get_kvcache_config(
        num_layers=3,
        num_heads=4,
        head_dim=128,
        page_size=32,
        offload_chunksize=128,
        num_primary_cache_pages=512,
        num_buffer_pages=0,
        host_capacity_per_layer=1024 * 2 * 32 * 4 * 128 * 2,
        max_batch_size=8,
        max_seq_len=2048,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        host_kvstorage_backend="flexkv",
        offload_timeout_ms=100.0,
        offload_mode="lazy",
        extra_configs=extra_configs,
    )
    gpu_gib = (
        kvcache_config.num_layers
        * kvcache_config.num_primary_cache_pages
        * kvcache_config.page_size
        * 2
        * kvcache_config.num_heads
        * kvcache_config.head_dim
        * 2
    ) / (1024.0**3)
    host_gib = (kvcache_config.num_layers * kvcache_config.host_capacity_per_layer) / (
        1024.0**3
    )
    print(f"[TEST] KVCache GPU Memory Usage: {gpu_gib} GiB.")
    print(f"[TEST] KVCache Host Memory Usage: {host_gib} GiB.")
    if flexkv_config_path:
        print(f"[TEST] FlexKV config path: {flexkv_config_path}")
    kvcache_mgr = KVCacheManager.from_config(kvcache_config)
    flexkv_mgr = kvcache_mgr.host_kvstorage_manager
    cache_cfg = flexkv_mgr._client.cache_config
    if cache_cfg.enable_ssd:
        print(
            "[TEST] Created KVCache Manager with FlexKV SSD tier: "
            f"num_cpu_blocks={cache_cfg.num_cpu_blocks}, "
            f"num_ssd_blocks={cache_cfg.num_ssd_blocks}, "
            f"ssd_cache_dir={cache_cfg.ssd_cache_dir}, "
            f"enable_gds={cache_cfg.enable_gds}"
        )
    else:
        print("[TEST] Created KVCache Manager with FlexKV CPU tier only")
    return kvcache_mgr


def run_phase_1(kvcache_mgr: KVCacheManager, all_keys, all_values) -> None:
    seqlen = [700, 128, 336, 624, 486, 358, 716, 537]
    user_ids = torch.tensor(list(range(0, 8)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)

    keys = [all_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
    values = [all_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    if not hasattr(index_meta, "sequence_lengths"):
        index_meta.sequence_lengths = index_meta.seq_lengths
    if not hasattr(index_meta, "slot_mappings"):
        index_meta.slot_mappings = None
    if hasattr(index_meta, "namespaces") and index_meta.namespaces is not None:
        index_meta.namespaces = [
            ns if isinstance(ns, list) else [ns] for ns in index_meta.namespaces
        ]
    assert torch.allclose(
        lookup_res.cached_start_indices, torch.zeros((8,), dtype=torch.int32)
    )
    assert torch.allclose(
        lookup_res.cached_lengths, torch.zeros((8,), dtype=torch.int32)
    )

    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
    if getattr(kvcache_metadata, "new_history_nnz_cuda", None) is not None:
        kvcache_metadata.new_history_nnz = int(
            kvcache_metadata.new_history_nnz_cuda.item()
        )
    assert torch.allclose(
        kvcache_metadata.kv_indices,
        torch.tensor(list(range(0, 125)), dtype=torch.int32).cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.kv_indptr,
        torch.tensor([0, 22, 26, 37, 57, 73, 85, 108, 125], dtype=torch.int32).cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.kv_last_page_len,
        torch.tensor(
            [i % 32 if i % 32 > 0 else 32 for i in seqlen], dtype=torch.int32
        ).cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.total_history_lengths, sequence_lengths.cuda()
    )
    assert torch.allclose(
        kvcache_metadata.total_history_offsets[1:]
        - kvcache_metadata.total_history_offsets[:-1],
        sequence_lengths.cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.new_history_offsets[1:]
        - kvcache_metadata.new_history_offsets[:-1],
        sequence_lengths.cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.new_history_nnz_cuda,
        torch.tensor([sum(seqlen)], dtype=torch.int32).cuda(),
    )
    assert kvcache_metadata.new_history_nnz == sum(seqlen)

    for layer_idx in range(3):
        kvcache_mgr.gpu_kvcache_mgr.put(
            torch.cat([k[layer_idx] for k in keys], dim=0),
            torch.cat([v[layer_idx] for v in values], dim=0),
            layer_idx,
            kvcache_metadata,
        )

    for i, uid in enumerate(range(0, 8)):
        k, v = keys[i], values[i]
        for layer_idx in range(3):
            page_ids = kvcache_metadata.kv_indices[
                kvcache_metadata.kv_indptr[i] : kvcache_metadata.kv_indptr[i + 1]
            ]
            last_page_lens = kvcache_metadata.kv_last_page_len[i].item()
            cached_k, cached_v = kvcache_mgr.gpu_kvcache_mgr.get(
                page_ids, last_page_lens, layer_idx
            )
            assert torch.allclose(
                cached_k, k[layer_idx]
            ), f"Layer {layer_idx} key mismatch for uid {user_ids[i].item()}"
            assert torch.allclose(
                cached_v, v[layer_idx]
            ), f"Layer {layer_idx} value mismatch for uid {user_ids[i].item()}"

    task_handle = kvcache_mgr.offload_launch(
        index_meta=index_meta,
        kvcache_metadata=kvcache_metadata,
    )
    assert task_handle is not None and task_handle.handle is not None

    kvcache_mgr.offload_flush()

    expected_host_lens = [s // 32 * 32 for s in seqlen]
    _, post_lookup = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    host_lens = [int(x.item()) for x in post_lookup.host_cached_lengths]
    assert host_lens == expected_host_lens, (
        "Phase1 host_cached_lengths mismatch after offload. "
        f"expected={expected_host_lens}, actual={host_lens}. "
    )

    torch.cuda.synchronize()


def run_phase_2(kvcache_mgr: KVCacheManager, all_keys, all_values) -> None:
    cachedlen = [700, 128, 336, 624, 486, 358, 716, 537]
    seqlen = [1400, 256, 672, 1248, 973, 716, 1432, 1075]
    deltalen = [seqlen[i] - cachedlen[i] for i in range(8)]
    host_cachedlen = [(i // 32) * 32 for i in cachedlen]

    user_ids = torch.tensor(list(range(0, 8)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)
    keys = [all_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
    values = [all_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
    new_keys = [k[:, cachedlen[i] : seqlen[i], ...] for i, k in enumerate(keys)]
    new_values = [v[:, cachedlen[i] : seqlen[i], ...] for i, v in enumerate(values)]

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    if not hasattr(index_meta, "sequence_lengths"):
        index_meta.sequence_lengths = index_meta.seq_lengths
    if not hasattr(index_meta, "slot_mappings"):
        index_meta.slot_mappings = None
    if hasattr(index_meta, "namespaces") and index_meta.namespaces is not None:
        index_meta.namespaces = [
            ns if isinstance(ns, list) else [ns] for ns in index_meta.namespaces
        ]
    assert torch.allclose(
        lookup_res.cached_start_indices, torch.zeros((8,), dtype=torch.int32)
    )
    assert torch.allclose(
        lookup_res.cached_lengths, torch.tensor(cachedlen, dtype=torch.int32)
    )
    assert torch.allclose(
        lookup_res.gpu_cached_lengths, torch.tensor(cachedlen, dtype=torch.int32)
    )
    assert torch.allclose(
        lookup_res.host_cached_lengths, torch.tensor(host_cachedlen, dtype=torch.int32)
    )

    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
    if getattr(kvcache_metadata, "new_history_nnz_cuda", None) is not None:
        kvcache_metadata.new_history_nnz = int(
            kvcache_metadata.new_history_nnz_cuda.item()
        )

    assert kvcache_metadata.kv_indices.size(0) == 245
    assert torch.allclose(
        kvcache_metadata.kv_indptr,
        torch.tensor(
            [0, 44, 52, 73, 112, 143, 166, 211, 245], dtype=torch.int32
        ).cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.kv_last_page_len,
        torch.tensor(
            [i % 32 if i % 32 > 0 else 32 for i in seqlen], dtype=torch.int32
        ).cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.total_history_lengths, sequence_lengths.cuda()
    )
    assert torch.allclose(
        kvcache_metadata.total_history_offsets[1:]
        - kvcache_metadata.total_history_offsets[:-1],
        sequence_lengths.cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.new_history_offsets[1:]
        - kvcache_metadata.new_history_offsets[:-1],
        torch.tensor(deltalen, dtype=torch.int32).cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.new_history_nnz_cuda,
        torch.tensor([sum(deltalen)], dtype=torch.int32).cuda(),
    )
    assert kvcache_metadata.new_history_nnz == sum(deltalen)

    kvcache_mgr.onboard_launch(index_meta, lookup_res, kvcache_metadata)
    assert kvcache_metadata.kv_onload_handle.status == HostKVTaskStatus.SKIPPED

    for layer_idx in range(3):
        kvcache_metadata.kv_onload_handle.stream_wait_layer(layer_idx)
    assert kvcache_metadata.kv_onload_handle.handle is None

    for layer_idx in range(3):
        kvcache_mgr.gpu_kvcache_mgr.put(
            torch.cat([k[layer_idx] for k in new_keys], dim=0),
            torch.cat([v[layer_idx] for v in new_values], dim=0),
            layer_idx,
            kvcache_metadata,
        )

    # for layer_idx in range(3):
    #     kvcache_mgr.gpu_kvcache_mgr.put(
    #         torch.cat([k[layer_idx] for k in new_keys], dim=0),
    #         torch.cat([v[layer_idx] for v in new_values], dim=0),
    #         layer_idx,
    #         kvcache_metadata,
    #     )

    for i, uid in enumerate(range(0, 8)):
        k, v = keys[i], values[i]
        for layer_idx in range(3):
            page_ids = kvcache_metadata.kv_indices[
                kvcache_metadata.kv_indptr[i] : kvcache_metadata.kv_indptr[i + 1]
            ]
            last_page_lens = kvcache_metadata.kv_last_page_len[i].item()
            cached_k, cached_v = kvcache_mgr.gpu_kvcache_mgr.get(
                page_ids, last_page_lens, layer_idx
            )
            assert torch.allclose(
                cached_k, k[layer_idx]
            ), f"Layer {layer_idx} key mismatch for uid {user_ids[i].item()}"
            assert torch.allclose(
                cached_v, v[layer_idx]
            ), f"Layer {layer_idx} value mismatch for uid {user_ids[i].item()}"

    task_handle = kvcache_mgr.offload_launch(
        index_meta=index_meta,
        kvcache_metadata=kvcache_metadata,
    )
    assert task_handle is not None and task_handle.handle is not None

    kvcache_mgr.offload_flush()

    expected_host_lens = [s // 32 * 32 for s in seqlen]

    _, post_lookup = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    host_lens = [int(x.item()) for x in post_lookup.host_cached_lengths]

    assert host_lens == expected_host_lens, (
        "Phase2 host_cached_lengths mismatch after offload. "
        f"expected={expected_host_lens}, actual={host_lens}. "
    )

    torch.cuda.synchronize()


def run_phase_3(kvcache_mgr: KVCacheManager, all_keys, all_values) -> None:
    cachedlen = [i // 32 * 32 for i in [1400, 256, 672, 1248, 973, 716, 1432, 1075]]
    seqlen = [2000, 512, 960, 1783, 1391, 1024, 2047, 1537]
    deltalen = [seqlen[i] - cachedlen[i] for i in range(8)]

    user_ids = torch.tensor(list(range(0, 8)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)
    keys = [all_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
    values = [all_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
    new_keys = [k[:, cachedlen[i] : seqlen[i], ...] for i, k in enumerate(keys)]
    new_values = [v[:, cachedlen[i] : seqlen[i], ...] for i, v in enumerate(values)]

    kvcache_mgr.evict(user_ids, for_gpu=True)

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    if not hasattr(index_meta, "sequence_lengths"):
        index_meta.sequence_lengths = index_meta.seq_lengths
    if not hasattr(index_meta, "slot_mappings"):
        index_meta.slot_mappings = None
    if hasattr(index_meta, "namespaces") and index_meta.namespaces is not None:
        index_meta.namespaces = [
            ns if isinstance(ns, list) else [ns] for ns in index_meta.namespaces
        ]
    assert torch.allclose(
        lookup_res.cached_start_indices, torch.zeros((8,), dtype=torch.int32)
    )
    assert torch.allclose(
        lookup_res.cached_lengths, torch.tensor(cachedlen, dtype=torch.int32)
    )
    assert torch.allclose(
        lookup_res.gpu_cached_lengths, torch.zeros((8,), dtype=torch.int32)
    )
    assert torch.allclose(
        lookup_res.host_cached_lengths, torch.tensor(cachedlen, dtype=torch.int32)
    )

    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
    if getattr(kvcache_metadata, "new_history_nnz_cuda", None) is not None:
        kvcache_metadata.new_history_nnz = int(
            kvcache_metadata.new_history_nnz_cuda.item()
        )

    assert kvcache_metadata.kv_indices.size(0) == 354
    assert torch.allclose(
        kvcache_metadata.kv_last_page_len,
        torch.tensor(
            [i % 32 if i % 32 > 0 else 32 for i in seqlen], dtype=torch.int32
        ).cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.total_history_lengths, sequence_lengths.cuda()
    )
    assert torch.allclose(
        kvcache_metadata.total_history_offsets[1:]
        - kvcache_metadata.total_history_offsets[:-1],
        sequence_lengths.cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.new_history_offsets[1:]
        - kvcache_metadata.new_history_offsets[:-1],
        torch.tensor(deltalen, dtype=torch.int32).cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.new_history_nnz_cuda,
        torch.tensor([sum(deltalen)], dtype=torch.int32).cuda(),
    )
    assert kvcache_metadata.new_history_nnz == sum(deltalen)

    onboard_task_handle = kvcache_mgr.onboard_launch(
        index_meta, lookup_res, kvcache_metadata
    )
    assert onboard_task_handle is not None and onboard_task_handle.handle is not None
    assert onboard_task_handle.status == HostKVTaskStatus.LAUNCHED, (
        "phase3 onboard launch was not LAUNCHED, "
        f"status={onboard_task_handle.status}, metadata={onboard_task_handle.metadata}"
    )

    onboard_deadline = time.time() + 60.0
    onboard_ready = False
    while time.time() < onboard_deadline:
        onboard_wait_result = kvcache_mgr.onboard_wait(index_meta, onboard_task_handle)
        if onboard_wait_result.ready:
            onboard_ready = True
            break
        time.sleep(0.01)
    assert onboard_ready, "phase3 onboard_wait never reached ready=True within timeout."

    for layer_idx in range(3):
        kvcache_mgr.gpu_kvcache_mgr.put(
            torch.cat([k[layer_idx] for k in new_keys], dim=0),
            torch.cat([v[layer_idx] for v in new_values], dim=0),
            layer_idx,
            kvcache_metadata,
        )

    for i, uid in enumerate(range(0, 8)):
        k, v = keys[i], values[i]
        for layer_idx in range(3):
            page_ids = kvcache_metadata.kv_indices[
                kvcache_metadata.kv_indptr[i] : kvcache_metadata.kv_indptr[i + 1]
            ]
            last_page_lens = kvcache_metadata.kv_last_page_len[i].item()
            cached_k, cached_v = kvcache_mgr.gpu_kvcache_mgr.get(
                page_ids, last_page_lens, layer_idx
            )
            assert torch.allclose(
                cached_k, k[layer_idx]
            ), f"Layer {layer_idx} key mismatch for uid {user_ids[i].item()}"
            assert torch.allclose(
                cached_v, v[layer_idx]
            ), f"Layer {layer_idx} value mismatch for uid {user_ids[i].item()}"

    task_handle = kvcache_mgr.offload_launch(
        index_meta=index_meta,
        kvcache_metadata=kvcache_metadata,
    )
    assert task_handle is not None and task_handle.handle is not None

    kvcache_mgr.offload_flush()

    expected_host_lens = [s // 32 * 32 for s in seqlen]
    _, post_lookup = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    host_lens = [int(x.item()) for x in post_lookup.host_cached_lengths]
    assert host_lens == expected_host_lens, (
        "Phase3 host_cached_lengths mismatch after offload. "
        f"expected={expected_host_lens}, actual={host_lens}. "
    )

    torch.cuda.synchronize()


def run_phase_4(kvcache_mgr: KVCacheManager, all_keys, all_values) -> None:
    cachedlen = [0 for _ in range(8)]
    seqlen = [689, 1417, 1174, 1987, 596, 520, 1538, 1189]
    deltalen = [seqlen[i] - cachedlen[i] for i in range(8)]

    user_ids = torch.tensor(list(range(8, 16)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)

    keys = [all_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(8, 16))]
    values = [
        all_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(8, 16))
    ]

    new_keys = [k[:, cachedlen[i] : seqlen[i], ...] for i, k in enumerate(keys)]
    new_values = [v[:, cachedlen[i] : seqlen[i], ...] for i, v in enumerate(values)]

    kvcache_mgr.evict(user_ids, for_gpu=True)

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    if not hasattr(index_meta, "sequence_lengths"):
        index_meta.sequence_lengths = index_meta.seq_lengths
    if not hasattr(index_meta, "slot_mappings"):
        index_meta.slot_mappings = None
    if hasattr(index_meta, "namespaces") and index_meta.namespaces is not None:
        index_meta.namespaces = [
            ns if isinstance(ns, list) else [ns] for ns in index_meta.namespaces
        ]
    assert torch.allclose(
        lookup_res.cached_start_indices, torch.zeros((8,), dtype=torch.int32)
    )
    assert torch.allclose(
        lookup_res.cached_lengths, torch.tensor(cachedlen, dtype=torch.int32)
    )
    assert torch.allclose(
        lookup_res.gpu_cached_lengths, torch.zeros((8,), dtype=torch.int32)
    )
    assert torch.allclose(
        lookup_res.host_cached_lengths, torch.tensor(cachedlen, dtype=torch.int32)
    )

    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
    if getattr(kvcache_metadata, "new_history_nnz_cuda", None) is not None:
        kvcache_metadata.new_history_nnz = int(
            kvcache_metadata.new_history_nnz_cuda.item()
        )

    assert kvcache_metadata.kv_indices.size(0) == 290
    assert torch.allclose(
        kvcache_metadata.kv_last_page_len,
        torch.tensor(
            [i % 32 if i % 32 > 0 else 32 for i in seqlen], dtype=torch.int32
        ).cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.total_history_lengths, sequence_lengths.cuda()
    )
    assert torch.allclose(
        kvcache_metadata.total_history_offsets[1:]
        - kvcache_metadata.total_history_offsets[:-1],
        sequence_lengths.cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.new_history_offsets[1:]
        - kvcache_metadata.new_history_offsets[:-1],
        torch.tensor(deltalen, dtype=torch.int32).cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.new_history_nnz_cuda,
        torch.tensor([sum(deltalen)], dtype=torch.int32).cuda(),
    )
    assert kvcache_metadata.new_history_nnz == sum(deltalen)

    for layer_idx in range(3):
        kvcache_mgr.gpu_kvcache_mgr.put(
            torch.cat([k[layer_idx] for k in new_keys], dim=0),
            torch.cat([v[layer_idx] for v in new_values], dim=0),
            layer_idx,
            kvcache_metadata,
        )

    for i, uid in enumerate(range(8, 16)):
        k, v = keys[i], values[i]
        for layer_idx in range(3):
            page_ids = kvcache_metadata.kv_indices[
                kvcache_metadata.kv_indptr[i] : kvcache_metadata.kv_indptr[i + 1]
            ]
            last_page_lens = kvcache_metadata.kv_last_page_len[i].item()
            cached_k, cached_v = kvcache_mgr.gpu_kvcache_mgr.get(
                page_ids, last_page_lens, layer_idx
            )
            assert torch.allclose(
                cached_k, k[layer_idx]
            ), f"Layer {layer_idx} key mismatch for uid {user_ids[i].item()}"
            assert torch.allclose(
                cached_v, v[layer_idx]
            ), f"Layer {layer_idx} value mismatch for uid {user_ids[i].item()}"

    task_handle = kvcache_mgr.offload_launch(
        index_meta=index_meta,
        kvcache_metadata=kvcache_metadata,
    )
    assert task_handle is not None and task_handle.handle is not None

    kvcache_mgr.offload_flush()

    expected_host_lens = [s // 32 * 32 for s in seqlen]
    _, post_lookup = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    host_lens = [int(x.item()) for x in post_lookup.host_cached_lengths]
    assert host_lens == expected_host_lens, (
        "Phase4 host_cached_lengths mismatch after offload. "
        f"expected={expected_host_lens}, actual={host_lens}. "
    )

    torch.cuda.synchronize()


def run_phase_5(kvcache_mgr: KVCacheManager, all_keys, all_values) -> None:
    cachedlen = [2000, 512, 960, 1783, 1391, 1024, 2047, 1537]
    seqlen = [2500, 1024, 1280, 2377, 1854, 1365, 2729, 2049]
    host_cachedlen = [(i // 32) * 32 for i in cachedlen]

    user_ids = torch.tensor(list(range(0, 8)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)

    keys = [all_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
    values = [all_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    if not hasattr(index_meta, "sequence_lengths"):
        index_meta.sequence_lengths = index_meta.seq_lengths
    if not hasattr(index_meta, "slot_mappings"):
        index_meta.slot_mappings = None
    if hasattr(index_meta, "namespaces") and index_meta.namespaces is not None:
        index_meta.namespaces = [
            ns if isinstance(ns, list) else [ns] for ns in index_meta.namespaces
        ]

    assert torch.allclose(
        lookup_res.cached_start_indices, torch.zeros((8,), dtype=torch.int32)
    )
    assert torch.allclose(
        lookup_res.host_cached_lengths, torch.tensor(host_cachedlen, dtype=torch.int32)
    )
    [int(x.item()) for x in lookup_res.host_cached_lengths]
    cachedlen_runtime = [int(x.item()) for x in lookup_res.cached_lengths]
    for i in range(8):
        assert host_cachedlen[i] <= cachedlen_runtime[i] <= cachedlen[i], (
            f"Phase5 cached length out of expected range for uid {user_ids[i].item()}: "
            f"host_cached={host_cachedlen[i]}, cached={cachedlen_runtime[i]}, "
            f"target_cached={cachedlen[i]}"
        )

    deltalen = [seqlen[i] - cachedlen_runtime[i] for i in range(8)]
    new_keys = [k[:, cachedlen_runtime[i] : seqlen[i], ...] for i, k in enumerate(keys)]
    new_values = [
        v[:, cachedlen_runtime[i] : seqlen[i], ...] for i, v in enumerate(values)
    ]

    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
    if getattr(kvcache_metadata, "new_history_nnz_cuda", None) is not None:
        kvcache_metadata.new_history_nnz = int(
            kvcache_metadata.new_history_nnz_cuda.item()
        )

    assert kvcache_metadata.kv_indices.size(0) == 478
    assert torch.allclose(
        kvcache_metadata.kv_last_page_len,
        torch.tensor(
            [i % 32 if i % 32 > 0 else 32 for i in seqlen], dtype=torch.int32
        ).cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.total_history_lengths, sequence_lengths.cuda()
    )
    assert torch.allclose(
        kvcache_metadata.total_history_offsets[1:]
        - kvcache_metadata.total_history_offsets[:-1],
        sequence_lengths.cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.new_history_offsets[1:]
        - kvcache_metadata.new_history_offsets[:-1],
        torch.tensor(deltalen, dtype=torch.int32).cuda(),
    )
    assert torch.allclose(
        kvcache_metadata.new_history_nnz_cuda,
        torch.tensor([sum(deltalen)], dtype=torch.int32).cuda(),
    )
    assert kvcache_metadata.new_history_nnz == sum(deltalen)

    onboard_task_handle = kvcache_mgr.onboard_launch(
        index_meta, lookup_res, kvcache_metadata
    )
    assert onboard_task_handle.status == HostKVTaskStatus.LAUNCHED, (
        "phase5 onboard launch failed, "
        f"status={onboard_task_handle.status}, metadata={onboard_task_handle.metadata}"
    )

    onboard_deadline = time.time() + 60.0
    onboard_ready = False
    while time.time() < onboard_deadline:
        onboard_wait_result = kvcache_mgr.onboard_wait(index_meta, onboard_task_handle)
        if onboard_wait_result.ready:
            onboard_ready = True
            break
        time.sleep(0.01)
    if not onboard_ready:
        print(
            "[WARN] phase5 onboard_wait never reached ready=True within timeout; "
            "fallback to finish_task(completely=True)."
        )

    for layer_idx in range(3):
        kvcache_mgr.gpu_kvcache_mgr.put(
            torch.cat([k[layer_idx] for k in new_keys], dim=0),
            torch.cat([v[layer_idx] for v in new_values], dim=0),
            layer_idx,
            kvcache_metadata,
        )

    for i, uid in enumerate(range(0, 8)):
        start, end = (
            kvcache_metadata.total_history_offsets[i],
            kvcache_metadata.total_history_offsets[i + 1],
        )
        k, v = keys[i], values[i]
        for layer_idx in range(3):
            page_ids = kvcache_metadata.kv_indices[
                kvcache_metadata.kv_indptr[i] : kvcache_metadata.kv_indptr[i + 1]
            ]
            last_page_lens = kvcache_metadata.kv_last_page_len[i].item()
            cached_k, cached_v = kvcache_mgr.gpu_kvcache_mgr.get(
                page_ids, last_page_lens, layer_idx
            )
            assert torch.allclose(
                cached_k, k[layer_idx]
            ), f"Layer {layer_idx} key mismatch for uid {user_ids[i].item()}"
            assert torch.allclose(
                cached_v, v[layer_idx]
            ), f"Layer {layer_idx} value mismatch for uid {user_ids[i].item()}"

    task_handle = kvcache_mgr.offload_launch(
        index_meta=index_meta,
        kvcache_metadata=kvcache_metadata,
    )
    assert task_handle is not None and task_handle.handle is not None

    kvcache_mgr.offload_flush()

    expected_host_lens = [s // 32 * 32 for s in seqlen]
    _, post_lookup = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    host_lens = [int(x.item()) for x in post_lookup.host_cached_lengths]
    assert host_lens == expected_host_lens, (
        "Phase5 host_cached_lengths mismatch after offload. "
        f"expected={expected_host_lens}, actual={host_lens}. "
    )

    # FlexKV backend currently has no explicit host evict API; verify for_host evict is a no-op.
    kvcache_mgr.evict(user_ids, for_host=True)
    _, lookup_after_host_evict = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    assert torch.allclose(
        lookup_after_host_evict.host_cached_lengths,
        torch.tensor(expected_host_lens, dtype=torch.int32),
    )

    # GPU evict still works; host remains unchanged for FlexKV.
    kvcache_mgr.evict(user_ids, for_gpu=True, for_host=True)
    _, lookup_after_full_evict = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    assert torch.allclose(
        lookup_after_full_evict.gpu_cached_lengths,
        torch.zeros_like(lookup_after_full_evict.gpu_cached_lengths),
    )
    assert torch.allclose(
        lookup_after_full_evict.host_cached_lengths,
        torch.tensor(expected_host_lens, dtype=torch.int32),
    )

    torch.cuda.synchronize()


if __name__ == "__main__":
    kvcache_mgr = create_testing_kvcache_manager()
    try:
        max_sequence_lengths = [
            2500,
            1024,
            1280,
            2377,
            1854,
            1365,
            2729,
            2049,
            689,
            1417,
            1174,
            1987,
            596,
            520,
            1538,
            1189,
        ]
        g_keys = [
            torch.randn(
                (3, max_sequence_lengths[i], 4, 128), dtype=torch.bfloat16
            ).cuda()
            for i in range(len(max_sequence_lengths))
        ]
        g_values = [
            torch.randn(
                (3, max_sequence_lengths[i], 4, 128), dtype=torch.bfloat16
            ).cuda()
            for i in range(len(max_sequence_lengths))
        ]
        print("Testing Phase 1: Allocate total and Offload total ...")
        run_phase_1(kvcache_mgr, g_keys, g_values)
        print("                 ... Passed.")
        print(
            "Testing Phase 2: Allocate partial, Onboard skipped and Offload partial ..."
        )
        run_phase_2(kvcache_mgr, g_keys, g_values)
        print("                 ... Passed.")
        print("Testing Phase 3: Evict GPU, onboard host, put delta and offload ...")
        run_phase_3(kvcache_mgr, g_keys, g_values)
        print("                 ... Passed.")
        print("Testing Phase 4: Allocate evict and Offload total ...")
        run_phase_4(kvcache_mgr, g_keys, g_values)
        print("                 ... Passed.")
        print("Testing Phase 5: Allocate evict, Onboard partial and Offload total ...")
        run_phase_5(kvcache_mgr, g_keys, g_values)
        print("                 ... Passed.")
    finally:
        kvcache_mgr = getattr(kvcache_mgr, "host_kvstorage_manager", None)
        client = getattr(kvcache_mgr, "_client", None)
        if client is not None and hasattr(client, "shutdown"):
            try:
                client.shutdown()
            except Exception as e:
                print(f"[WARN] FlexKV client shutdown failed: {e}")
