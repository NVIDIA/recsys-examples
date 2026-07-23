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

import torch
from recsys_kvcache_manager.host_kvstorage_manager import HostKVTaskStatus
from recsys_kvcache_manager.kvcache_config import get_kvcache_config
from recsys_kvcache_manager.kvcache_manager import KVCacheManager


def create_testing_kvcache_manager():
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
        host_kvstorage_backend="native",
        offload_timeout_ms=100.0,
        offload_mode="lazy",
    )
    print(
        f"[TEST] KVCache GPU Memory Usage: {\
        (kvcache_config.num_layers * \
        kvcache_config.num_primary_cache_pages * \
        kvcache_config.page_size * \
        2 * kvcache_config.num_heads * \
        kvcache_config.head_dim * 2) / (1024. ** 3) \
        } GiB."
    )
    print(
        f"[TEST] KVCache Host Memory Usage: {\
        kvcache_config.num_layers * \
        kvcache_config.host_capacity_per_layer / (1024. ** 3) \
        } GiB."
    )

    kvcache_mgr = KVCacheManager.from_config(kvcache_config)
    print(f"[TEST] Created KVCache Manager")
    return kvcache_mgr


if __name__ == "__main__":
    kvcache_mgr = create_testing_kvcache_manager()

    # Testing Order:
    # A
    # 1. Allocate total     u0 ~ 8  [700, 128, 336, 624, 486, 358, 716, 537]            # 125 pags
    # 2. Offload total
    # B
    # 3. Allocate partial   u0 ~ 8  [1400, 256, 672, 1248, 973, 716, 1432, 1075]        # 245 pages
    # 4. Offload partial
    # 5. [DEBUG] Evict gpu

    # C
    # 6. Allocate total     u0 ~ 8  [2000, 512, 960, 1783, 1391, 1024, 2047, 1537]      # 354 pages
    # 7. Onboard total
    # 8. Offload partial

    # D
    # 9. Allocate evict     u8 ~ 16 [689, 1417, 1174, 1987,  596,  520, 1538, 1189]     # 290 pages
    # 10. Offload total

    # E
    # 11. Allocate evict    u0 ~ 8  [2500, 1024, 1280, 2377, 1854, 1365, 2729, 2049]    # 478 pages
    # 12. Onboard partial
    # 13. Offload evict
    # 14. Evict host
    # 15. Evict total

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
        torch.randn((3, max_sequence_lengths[i], 4, 128), dtype=torch.bfloat16).cuda()
        for i in range(len(max_sequence_lengths))
    ]
    g_values = [
        torch.randn((3, max_sequence_lengths[i], 4, 128), dtype=torch.bfloat16).cuda()
        for i in range(len(max_sequence_lengths))
    ]

    def run_phase_1():
        seqlen = [700, 128, 336, 624, 486, 358, 716, 537]

        user_ids = torch.tensor(list(range(0, 8)), dtype=torch.int64)
        sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)

        keys = [g_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
        values = [
            g_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))
        ]

        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
        assert torch.allclose(
            lookup_res.cached_start_indices, torch.zeros((8,), dtype=torch.int32)
        )
        assert torch.allclose(
            lookup_res.cached_lengths, torch.zeros((8,), dtype=torch.int32)
        )

        kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
        assert torch.allclose(
            kvcache_metadata.kv_indices,
            torch.tensor(list(range(0, 125)), dtype=torch.int32).cuda(),
        )
        assert torch.allclose(
            kvcache_metadata.kv_indptr,
            torch.tensor(
                [0, 22, 26, 37, 57, 73, 85, 108, 125], dtype=torch.int32
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
            sequence_lengths.cuda(),
        )

        assert torch.allclose(
            kvcache_metadata.new_history_nnz_cuda,
            torch.tensor(
                [
                    sum(seqlen),
                ],
                dtype=torch.int32,
            ).cuda(),
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

        kvcache_mgr.offload_launch(index_meta)
        kvcache_mgr.offload_flush()

        host_len = [seqlen[i] // 32 * 32 for i in range(8)]
        for i, uid in enumerate(range(0, 8)):
            k, v = keys[i][:, : host_len[i], ...], values[i][:, : host_len[i], ...]

            kvdata = kvcache_mgr.host_kvstorage_manager.impl_.get_kvdata_tensor(
                [
                    uid,
                ],
                False,
            )[0]
            cached_k, cached_v = kvdata.unbind(dim=2)
            cached_k = cached_k.reshape(
                cached_k.size(0), -1, cached_k.size(3), cached_k.size(4)
            )
            cached_v = cached_v.reshape(
                cached_v.size(0), -1, cached_v.size(3), cached_v.size(4)
            )
            assert torch.allclose(
                cached_k, k.cpu()
            ), f"Key mismatch for uid {user_ids[i].item()}"
            assert torch.allclose(
                cached_v, v.cpu()
            ), f"Value mismatch for uid {user_ids[i].item()}"

        torch.cuda.synchronize()

    def run_phase_2():
        cachedlen = [700, 128, 336, 624, 486, 358, 716, 537]
        seqlen = [1400, 256, 672, 1248, 973, 716, 1432, 1075]
        deltalen = [seqlen[i] - cachedlen[i] for i in range(8)]

        user_ids = torch.tensor(list(range(0, 8)), dtype=torch.int64)
        sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)

        keys = [g_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
        values = [
            g_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))
        ]

        new_keys = [k[:, cachedlen[i] : seqlen[i], ...] for i, k in enumerate(keys)]
        new_values = [v[:, cachedlen[i] : seqlen[i], ...] for i, v in enumerate(values)]

        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
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
            lookup_res.host_cached_lengths,
            torch.tensor([(i // 32) * 32 for i in cachedlen], dtype=torch.int32),
        )

        kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
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
            torch.tensor(
                [
                    sum(deltalen),
                ],
                dtype=torch.int32,
            ).cuda(),
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

        kvcache_mgr.offload_launch(index_meta)
        kvcache_mgr.offload_flush()

        host_len = [seqlen[i] // 32 * 32 for i in range(8)]
        for i, uid in enumerate(range(0, 8)):
            k, v = keys[i][:, : host_len[i], ...], values[i][:, : host_len[i], ...]

            kvdata = kvcache_mgr.host_kvstorage_manager.impl_.get_kvdata_tensor(
                [
                    uid,
                ],
                False,
            )[0]
            cached_k, cached_v = kvdata.unbind(dim=2)
            cached_k = cached_k.reshape(
                cached_k.size(0), -1, cached_k.size(3), cached_k.size(4)
            )
            cached_v = cached_v.reshape(
                cached_v.size(0), -1, cached_v.size(3), cached_v.size(4)
            )
            assert torch.allclose(
                cached_k, k.cpu()
            ), f"Key mismatch for uid {user_ids[i].item()}"
            assert torch.allclose(
                cached_v, v.cpu()
            ), f"Value mismatch for uid {user_ids[i].item()}"

        torch.cuda.synchronize()

    def run_phase_3():
        cachedlen = [i // 32 * 32 for i in [1400, 256, 672, 1248, 973, 716, 1432, 1075]]
        seqlen = [2000, 512, 960, 1783, 1391, 1024, 2047, 1537]
        deltalen = [seqlen[i] - cachedlen[i] for i in range(8)]

        user_ids = torch.tensor(list(range(0, 8)), dtype=torch.int64)
        sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)

        keys = [g_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
        values = [
            g_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))
        ]

        new_keys = [k[:, cachedlen[i] : seqlen[i], ...] for i, k in enumerate(keys)]
        new_values = [v[:, cachedlen[i] : seqlen[i], ...] for i, v in enumerate(values)]

        kvcache_mgr.evict(user_ids, for_gpu=True)

        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
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
            torch.tensor(
                [
                    sum(deltalen),
                ],
                dtype=torch.int32,
            ).cuda(),
        )
        assert kvcache_metadata.new_history_nnz == sum(deltalen)

        kvcache_mgr.onboard_launch(index_meta, lookup_res, kvcache_metadata)
        for layer_idx in range(3):
            kvcache_metadata.kv_onload_handle.stream_wait_layer(layer_idx)

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

        kvcache_mgr.offload_launch(index_meta)
        kvcache_mgr.offload_flush()

        host_len = [seqlen[i] // 32 * 32 for i in range(8)]
        for i, uid in enumerate(range(0, 8)):
            k, v = keys[i][:, : host_len[i], ...], values[i][:, : host_len[i], ...]

            kvdata = kvcache_mgr.host_kvstorage_manager.impl_.get_kvdata_tensor(
                [
                    uid,
                ],
                False,
            )[0]
            cached_k, cached_v = kvdata.unbind(dim=2)
            cached_k = cached_k.reshape(
                cached_k.size(0), -1, cached_k.size(3), cached_k.size(4)
            )
            cached_v = cached_v.reshape(
                cached_v.size(0), -1, cached_v.size(3), cached_v.size(4)
            )
            assert torch.allclose(
                cached_k, k.cpu()
            ), f"Key mismatch for uid {user_ids[i].item()}"
            assert torch.allclose(
                cached_v, v.cpu()
            ), f"Value mismatch for uid {user_ids[i].item()}"

        torch.cuda.synchronize()

    def run_phase_4():
        cachedlen = [0 for _ in range(8)]
        seqlen = [689, 1417, 1174, 1987, 596, 520, 1538, 1189]
        deltalen = [seqlen[i] - cachedlen[i] for i in range(8)]

        user_ids = torch.tensor(list(range(8, 16)), dtype=torch.int64)
        sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)

        keys = [g_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(8, 16))]
        values = [
            g_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(8, 16))
        ]

        new_keys = [k[:, cachedlen[i] : seqlen[i], ...] for i, k in enumerate(keys)]
        new_values = [v[:, cachedlen[i] : seqlen[i], ...] for i, v in enumerate(values)]

        kvcache_mgr.evict(user_ids, for_gpu=True)

        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
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
            torch.tensor(
                [
                    sum(deltalen),
                ],
                dtype=torch.int32,
            ).cuda(),
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

        kvcache_mgr.offload_launch(index_meta)
        kvcache_mgr.offload_flush()

        host_len = [seqlen[i] // 32 * 32 for i in range(8)]
        for i, uid in enumerate(range(8, 16)):
            k, v = keys[i][:, : host_len[i], ...], values[i][:, : host_len[i], ...]

            kvdata = kvcache_mgr.host_kvstorage_manager.impl_.get_kvdata_tensor(
                [
                    uid,
                ],
                False,
            )[0]
            cached_k, cached_v = kvdata.unbind(dim=2)
            cached_k = cached_k.reshape(
                cached_k.size(0), -1, cached_k.size(3), cached_k.size(4)
            )
            cached_v = cached_v.reshape(
                cached_v.size(0), -1, cached_v.size(3), cached_v.size(4)
            )
            assert torch.allclose(
                cached_k, k.cpu()
            ), f"Key mismatch for uid {user_ids[i].item()}"
            assert torch.allclose(
                cached_v, v.cpu()
            ), f"Value mismatch for uid {user_ids[i].item()}"

        torch.cuda.synchronize()

    def run_phase_5():
        cachedlen = [2000, 512, 960, 1783, 1391, 1024, 2047, 1537]
        seqlen = [2500, 1024, 1280, 2377, 1854, 1365, 2729, 2049]
        deltalen = [seqlen[i] - cachedlen[i] for i in range(8)]

        user_ids = torch.tensor(list(range(0, 8)), dtype=torch.int64)
        sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)

        keys = [g_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
        values = [
            g_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))
        ]

        new_keys = [k[:, cachedlen[i] : seqlen[i], ...] for i, k in enumerate(keys)]
        new_values = [v[:, cachedlen[i] : seqlen[i], ...] for i, v in enumerate(values)]

        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
        assert torch.allclose(
            lookup_res.cached_start_indices, torch.zeros((8,), dtype=torch.int32)
        )
        assert torch.allclose(
            lookup_res.cached_lengths, torch.tensor(cachedlen, dtype=torch.int32)
        )
        assert torch.allclose(
            lookup_res.gpu_cached_start_indices,
            torch.tensor([1984, 0, 0, 1760, 0, 0, 0, 0], dtype=torch.int32),
        )
        assert torch.allclose(
            lookup_res.gpu_cached_lengths,
            torch.tensor([16, 0, 0, 23, 1391, 1024, 2047, 1537], dtype=torch.int32),
        )
        assert torch.allclose(
            lookup_res.host_cached_lengths,
            torch.tensor(
                [1984, 512, 960, 1760, 1376, 1024, 2016, 1536], dtype=torch.int32
            ),
        )

        kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)

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
            torch.tensor(
                [
                    sum(deltalen),
                ],
                dtype=torch.int32,
            ).cuda(),
        )
        assert kvcache_metadata.new_history_nnz == sum(deltalen)

        kvcache_mgr.onboard_launch(index_meta, lookup_res, kvcache_metadata)
        for layer_idx in range(3):
            kvcache_metadata.kv_onload_handle.stream_wait_layer(layer_idx)

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

        kvcache_mgr.offload_launch(index_meta)
        kvcache_mgr.offload_flush()

        host_len = [seqlen[i] // 32 * 32 for i in range(8)]
        for i, uid in enumerate(range(0, 8)):
            k, v = keys[i][:, : host_len[i], ...], values[i][:, : host_len[i], ...]

            kvdata = kvcache_mgr.host_kvstorage_manager.impl_.get_kvdata_tensor(
                [
                    uid,
                ],
                False,
            )[0]
            cached_k, cached_v = kvdata.unbind(dim=2)
            cached_k = cached_k.reshape(
                cached_k.size(0), -1, cached_k.size(3), cached_k.size(4)
            )
            cached_v = cached_v.reshape(
                cached_v.size(0), -1, cached_v.size(3), cached_v.size(4)
            )
            assert torch.allclose(
                cached_k, k.cpu()
            ), f"Key mismatch for uid {user_ids[i].item()}"
            assert torch.allclose(
                cached_v, v.cpu()
            ), f"Value mismatch for uid {user_ids[i].item()}"

        kvcache_mgr.evict(user_ids, for_host=True)
        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
        assert torch.allclose(
            lookup_res.host_cached_lengths,
            torch.zeros_like(lookup_res.host_cached_lengths),
        )

        kvcache_mgr.evict(user_ids, for_gpu=True, for_host=True)
        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
        assert torch.allclose(
            lookup_res.gpu_cached_lengths,
            torch.zeros_like(lookup_res.host_cached_lengths),
        )

        torch.cuda.synchronize()

    print(f"Testing Phase 1: Allocate total and Offload total ...")
    run_phase_1()
    print(f"                 ... Passed.")

    print(f"Testing Phase 2: Allocate partial, Onboard skipped and Offload partial ...")
    run_phase_2()
    print(f"                 ... Passed.")

    print(f"Testing Phase 3: Allocate partial, Onboard total and Offload partial ...")
    run_phase_3()
    print(f"                 ... Passed.")

    print(f"Testing Phase 4: Allocate evict and Offload total ...")
    run_phase_4()
    print(f"                 ... Passed.")

    print(f"Testing Phase 5: Allocate evict, Onboard partial and Offload total ...")
    run_phase_5()
    print(f"                 ... Passed.")
