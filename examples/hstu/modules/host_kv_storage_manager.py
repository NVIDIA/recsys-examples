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
from typing import List, Optional, Tuple

import tensorrt_llm
import torch
from configs import InferenceHSTUConfig, KVCacheConfig

KVCacheManagerImpl = tensorrt_llm.bindings.internal.batch_manager.KVCacheManager
KvCacheConfigCpp = tensorrt_llm.bindings.KvCacheConfig
DataType = tensorrt_llm.bindings.DataType


class HSTUHostKVStorageImpl:
    def __init__(self):
        pass

    def get_user_kvdata_info(self, user_id: int):
        pass

    def append_kvdata(
        self, user_id: int, start_pos: int, length: int, kv_data: List[torch.Tensor]
    ):
        pass

    def get_kv_data(
        self, user_id: int, length: int, layer_idx: int, output_buffer: torch.Tensor
    ) -> torch.Tensor:
        pass

    def evict_all_kvdata(self):
        pass


class DummyHSTUHostKVStorageImpl(HSTUHostKVStorageImpl):
    def __init__(self, num_layers, page_size, offload_chunksize):
        super(HSTUHostKVStorageImpl, self).__init__()
        self._num_layers = num_layers
        self._page_size = page_size
        self._offload_chunksize = offload_chunksize

        self.sequence_start_pos = dict()
        self.sequence_length = dict()
        self.kv_data_storage = [dict() for _ in range(self._num_layers)]

    def get_user_kvdata_info(self, user_id: int):
        seq_start_pos = self.sequence_start_pos.get(user_id, 0)
        seq_length = self.sequence_length.get(user_id, 0)
        return (seq_start_pos, seq_length)

    def append_kvdata(
        self, user_id: int, start_pos: int, length: int, kv_data: List[torch.Tensor]
    ):
        old_start_pos, old_length = self.get_user_kvdata_info(user_id)
        # assert old_start_pos + old_length == start_pos, \
        #     "{0} new kvdata starting position is {1}, unmatching current {2} ~ {3}".format(
        #         user_id, start_pos, old_start_pos, old_start_pos + old_length)
        # assert self._num_layers == len(
        #     kv_data), "the given kv_data has wrong number of layers"

        if user_id not in self.sequence_start_pos:
            self.sequence_start_pos[user_id] = start_pos
            self.sequence_length[user_id] = 0
        self.sequence_length[user_id] += length

        for layer_idx in range(self._num_layers):
            storage = self.kv_data_storage[layer_idx]
            if user_id not in storage:
                storage[user_id] = list()
            storage[user_id].append(kv_data[layer_idx])

    def get_kv_data(
        self, user_id: int, length: int, layer_idx: int, output_buffer: torch.Tensor
    ) -> torch.Tensor:
        kv_data_list = []
        current_length = 0
        for data_chunk in self.kv_data_storage[layer_idx][user_id]:
            if data_chunk.shape[0] * self._page_size + current_length > length:
                slice_length = (length - current_length) // self._page_size
                kv_data_list.append(data_chunk[:slice_length, ...])
                break
            else:
                kv_data_list.append(data_chunk)
            current_length += kv_data_list[-1].shape[0] * self._page_size
        # assert sum([t.shape[0] for t in kv_data_list]) == output_buffer.shape[0]
        return torch.cat(kv_data_list, dim=0, out=output_buffer)

    def evict_all_kvdata(self):
        self.sequence_start_pos.clear()
        self.sequence_length.clear()
        for layer_idx in range(self._num_layers):
            self.kv_data_storage[layer_idx].clear()


class HSTUHostKVStorageManager:
    def __init__(
        self, hstu_config: InferenceHSTUConfig, kv_cache_config: KVCacheConfig
    ) -> None:
        self.num_layers = hstu_config.num_layers
        self.head_dim = hstu_config.head_dim
        self.num_heads = hstu_config.num_heads
        self.page_size = kv_cache_config.page_size
        self.num_cache_pages = kv_cache_config.blocks_in_primary_pool

        self.max_seq_len = hstu_config.max_seq_len
        if kv_cache_config.max_attention_window is None:
            self.max_attention_window = hstu_config.max_seq_len
        else:
            self.max_attention_window = max(kv_cache_config.max_attention_window)

        self.offload_chunksize = kv_cache_config.offload_chunksize
        self.max_batch_size = hstu_config.max_batch_size

        self.impl: HSTUHostKVStorageImpl = DummyHSTUHostKVStorageImpl(
            self.num_layers, self.page_size, self.offload_chunksize
        )

        self.kv_cache_dtype = (
            torch.bfloat16
            if hstu_config.bf16
            else torch.float16
            if hstu_config.fp16
            else torch.float32
        )
        self.static_kvdata_buffer_ = torch.empty(
            (
                self.num_layers,
                (self.max_batch_size * self.max_seq_len) // self.page_size,
                2,
                self.page_size,
                self.num_heads,
                self.head_dim,
            ),
            dtype=self.kv_cache_dtype,
            device=torch.device("cpu"),
            pin_memory=True,
        ).uniform_(-0.05, 0.05)

    def fetch_kv_data(
        self, user_id: int, length: int, layer_idx: int, output_buffer: torch.Tensor
    ):
        self.impl.get_kv_data(user_id, length, layer_idx, output_buffer)

    def get_user_kvdata_info(self, user_id: int) -> Tuple[int, int]:
        return self.impl.get_user_kvdata_info(user_id)

    def lookup_kvdata(
        self,
        user_ids: torch.Tensor,
        cached_start_pos: torch.Tensor,
        cached_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int]]:
        onload_history_seq_length = torch.zeros_like(
            user_ids, dtype=torch.int32, device=torch.device("cpu")
        )
        onload_history_seqlen_offsets = torch.zeros(
            (user_ids.shape[0] + 1,), dtype=torch.int32, device=torch.device("cpu")
        )
        for idx in range(user_ids.shape[0]):
            user_id = user_ids[idx].item()
            cache_len = cached_lengths[idx].item()
            onload_history_seq_length[idx] = self.get_onload_history_seqlen(
                user_id, cached_start_pos[idx].item() if cache_len > 0 else -1
            )
        torch.cumsum(
            onload_history_seq_length, 0, out=onload_history_seqlen_offsets[1:]
        )
        onload_length = onload_history_seqlen_offsets[-1].item()
        if onload_length == 0:
            return (onload_length, None, None)

        # copy lookup results into buffer (compress into one)
        for idx in range(user_ids.shape[0]):
            user_id = user_ids[idx].item()
            length = onload_history_seq_length[idx].item()
            if length == 0:
                continue

            start_page_idx = onload_history_seqlen_offsets[idx].item() // self.page_size
            end_page_idx = (
                onload_history_seqlen_offsets[idx + 1].item() // self.page_size
            )
            for layer_idx in range(self.num_layers):
                self.fetch_kv_data(
                    user_id,
                    length,
                    layer_idx,
                    self.static_kvdata_buffer_[
                        layer_idx, start_page_idx:end_page_idx, ...
                    ],
                )

        onload_kv_page_ids = torch.arange(
            start=self.num_cache_pages,
            end=onload_length // self.page_size + self.num_cache_pages,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        onload_kv_page_indptr = (onload_history_seqlen_offsets / self.page_size).to(
            dtype=torch.int32, device=torch.cuda.current_device()
        )

        return onload_length, onload_kv_page_ids, onload_kv_page_indptr

    def append_kvdata(
        self,
        offloaded_kv_data: List[torch.Tensor],  # (total_num_pages, *single_page_shape)
        user_ids: torch.Tensor,
        offload_start_pos: torch.Tensor,
        offload_page_indptr: torch.Tensor,
    ):
        for idx in range(len(user_ids)):
            uid = user_ids[idx].item()
            start_pos = offload_start_pos[idx].item()
            page_start = offload_page_indptr[idx].item()
            page_end = offload_page_indptr[idx + 1].item()
            length = (page_end - page_start) * self.page_size
            if length == 0:
                continue

            self.impl.append_kvdata(
                uid,
                start_pos,
                length,
                [
                    offloaded_kv_data[layer_idx][page_start:page_end, ...]
                    .detach()
                    .clone()
                    for layer_idx in range(self.num_layers)
                ],
            )

    def evict_all_kvdata(self):
        self.impl.evict_all_kvdata()

    def get_lookup_buffer(self) -> torch.Tensor:
        return self.static_kvdata_buffer_

    def get_onload_history_seqlen(self, user_id: int, cached_start_pos: int) -> int:
        (offloaded_start_pos, offloaded_length) = self.get_user_kvdata_info(user_id)
        return (
            min(offloaded_length, cached_start_pos - offloaded_start_pos)
            if cached_start_pos > -1
            else offloaded_length
        )
