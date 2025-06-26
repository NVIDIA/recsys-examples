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
from typing import Optional, Tuple

import torch
from commons.utils.nvtx_op import output_nvtx_hook
from configs import (
    HSTUConfig, 
    RankingConfig, 
    KVCacheConfig, 
    KVCacheMetadata, 
    get_kvcache_metadata_buffer,
    copy_kvcache_metadata
)
from dataset.utils import RankingBatch
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from model.base_model import BaseModel
from modules.embedding import ShardedEmbedding
from modules.gpu_kv_cache_manager import HSTUGpuKVCacheManager
from modules.host_kv_storage_manager import HSTUHostKVStorageManager
from modules.hstu_block import HSTUBlock
from modules.jagged_data import JaggedData
from modules.metrics import get_multi_event_metric_module
from modules.mlp import MLP
from modules.multi_task_loss_module import MultiTaskLossModule
from ops.triton_ops.triton_jagged import triton_concat_2D_jagged
import os
import numpy as np


def get_jagged_metadata_buffer(max_batch_size, max_seq_len):
    int_dtype = torch.int32
    device = torch.cuda.current_device()

    default_num_candidates = max_seq_len // 2

    return JaggedData(
        values=None,
        # hidden states
        max_seqlen=max_seq_len,
        seqlen=torch.full((max_batch_size, ), max_seq_len, dtype=int_dtype, device=device),
        seqlen_offsets=torch.arange(end=max_batch_size+1, dtype=int_dtype, device=device) * max_seq_len,

        # candidates (included in hidden states)
        max_num_candidates=default_num_candidates,
        num_candidates=torch.full((max_batch_size, ), default_num_candidates, dtype=int_dtype, device=device),
        num_candidates_offsets=torch.arange(end=max_batch_size+1, dtype=int_dtype, device=device) * default_num_candidates,

        # contextual features
        contextual_max_seqlen=0,
        contextual_seqlen=None,
        contextual_seqlen_offsets=None,

        has_interleaved_action=True,
    )

def copy_jagged_metadata(dst_metadata, src_metata):
    bs = src_metata.seqlen.shape[0]
    dst_metadata.max_seqlen = src_metata.max_seqlen
    dst_metadata.seqlen[:bs].copy_(src_metata.seqlen[:bs], non_blocking=True)
    dst_metadata.seqlen_offsets[:bs+1].copy_(src_metata.seqlen_offsets[:bs+1], non_blocking=True)
    dst_metadata.max_num_candidates = src_metata.max_num_candidates
    dst_metadata.num_candidates[:bs].copy_(src_metata.num_candidates[:bs], non_blocking=True)
    dst_metadata.num_candidates_offsets[:bs+1].copy_(src_metata.num_candidates_offsets[:bs+1], non_blocking=True)


class RankingGRInferenceModel(torch.nn.Module):
    """
    A class representing the ranking model inference.

    Args:
        hstu_config (HSTUConfig): The HSTU configuration.
        kvcache_config (KVCacheConfig): The HSTU KV cache configuration.
        task_config (RankingConfig): The ranking task configuration.
        ddp_config (Optional[DistributedDataParallelConfig]): The distributed data parallel configuration. If not provided, will use default value.
    """

    def __init__(
        self,
        hstu_config: HSTUConfig,
        kvcache_config: KVCacheConfig,
        task_config: RankingConfig,
        ddp_config: Optional[DistributedDataParallelConfig] = None,
        use_cudagraph = False,
        # gpu_kv_cache_impl = None,
        # host_kv_storage_impl = None,
    ):
        super().__init__()
        self._tp_size = parallel_state.get_tensor_model_parallel_world_size()
        assert (
            self._tp_size == 1
        ), "RankingGRInfer does not support tensor model parallel for now"
        self._device = torch.cuda.current_device()
        self._hstu_config = hstu_config
        self._task_config = task_config
        self._ddp_config = ddp_config

        self._embedding_dim = hstu_config.hidden_size
        for ebc_config in task_config.embedding_configs:
            assert (
                ebc_config.dim == self._embedding_dim
            ), "hstu layer hidden size should equal to embedding dim"

        self._logit_dim_list = [
            layer_sizes[-1] for layer_sizes in task_config.prediction_head_arch
        ]
        self._embedding_collection = ShardedEmbedding(task_config.embedding_configs)
        # temporary using a non-sharing GPU embeding
        self._embedding_collection.to_empty(device=self._device)


        self._gpu_kv_cache_manager = HSTUGpuKVCacheManager(
            hstu_config, kvcache_config)
        self._host_kv_storage_manager = HSTUHostKVStorageManager(
            hstu_config, kvcache_config)

        self._hstu_block = HSTUBlock(hstu_config, inference_mode=True, kvcache_config=kvcache_config)
        self._dense_module = MLP(
            self._embedding_dim,
            task_config.prediction_head_arch[0],
            task_config.prediction_head_act_type,
            task_config.prediction_head_bias,
            device=self._device,
        )

        self._hstu_block = self._hstu_block.cuda()
        self._dense_module = self._dense_module.cuda()

        dtype = torch.bfloat16 if hstu_config.bf16 else torch.float16 if hstu_config.fp16 else torch.float32
        device = torch.cuda.current_device()

        max_batch_size = kvcache_config.max_batch_size
        max_seq_len = kvcache_config.max_seq_len
        hidden_dim = hstu_config.hidden_size

        self._hidden_states = torch.randn((max_batch_size * max_seq_len, hidden_dim), 
            dtype=dtype, device=device)
        self._jagged_metadata = get_jagged_metadata_buffer(max_batch_size, max_seq_len)
        self._kvcache_metadata = get_kvcache_metadata_buffer(hstu_config, kvcache_config)
        self._kvcache_metadata.kv_cache_table = [ 
            self._gpu_kv_cache_manager.get_buffers(layer_idx) for layer_idx in range(hstu_config.num_layers) ]

        if use_cudagraph:
            self._hstu_block.set_cudagraph(
                max_batch_size, 
                max_seq_len, 
                self._hidden_states,
                self._jagged_metadata,
                self._kvcache_metadata)

    def bfloat16(self):
        """
        Convert the model to use bfloat16 precision. Only affects the dense module.

        Returns:
            RankingGR: The model with bfloat16 precision.
        """
        self._hstu_block.bfloat16()
        self._dense_module.bfloat16()
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the dense module.

        Returns:
            RankingGR: The model with half precision.
        """
        self._hstu_block.half()
        self._dense_module.half()
        return self
    
    def lookup_cached_lengths(self, user_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (gpu_cached_lengths, gpu_offloaded_legnths) = self._gpu_kv_cache_manager.lookup(user_ids)
        host_lengths = self._host_kv_storage_manger.lookup(user_ids)
        return (gpu_cached_lengths, gpu_offloaded_legnths, host_lengths)
    
    def prepare_kv_cache(self, batch: RankingBatch, user_ids: torch.Tensor) -> Tuple[torch.Tensor, KVCacheMetadata]:
        self._gpu_kv_cache_manager.allocate(batch, user_ids)
        kv_cache_metadata = self._gpu_kv_cache_manager.get_cache_page_metadata(batch, user_ids)
        (onload_kv_page_ids, onload_kv_page_indptr, onload_length) = self._host_kv_storage_manager.lookup(user_ids, torch.zeros_like(user_ids))

        if onload_kv_page_indptr[-1].item() > 0:
            kv_page_ids = triton_concat_2D_jagged(
                    max_seq_len=onload_kv_page_indptr[-1] + kv_cache_metadata.kv_indices[-1],
                    values_a=onload_kv_page_ids.view(-1, 1),
                    values_b=kv_cache_metadata.kv_indices.view(-1, 1),
                    offsets_a=onload_kv_page_indptr.to(torch.int64),
                    offsets_b=kv_cache_metadata.kv_indptr.to(torch.int64),
                )
            kv_cache_metadata.kv_indices = kv_page_ids.view(-1)
            kv_cache_metadata.kv_indptr = onload_kv_page_indptr + kv_cache_metadata.kv_indptr

        copy_kvcache_metadata(self._kvcache_metadata, kv_cache_metadata)
        self._gpu_kv_cache_manager.onboard(self._host_kv_storage_manager.get_lookup_buffer(), onload_length, self._kvcache_metadata)
    
    def finalize_kv_cache(self, user_ids: torch.Tensor):
        self._gpu_kv_cache_manager.evict(user_ids)
        return
    
    def forward(
        self,
        batch: RankingBatch,
        user_ids: torch.Tensor,
    ):  
        with torch.inference_mode():
            jagged_data = self._hstu_block.hstu_preprocess(
                embeddings=self._embedding_collection(batch.features),
                batch=batch,
            )
            self.prepare_kv_cache(batch, user_ids)
            copy_jagged_metadata(self._jagged_metadata, jagged_data)
            hstu_output = self._hstu_block.predict(batch.batch_size, jagged_data.seqlen_offsets[-1].item(), jagged_data.values, self._jagged_metadata, self._kvcache_metadata)
            self.finalize_kv_cache(user_ids)
            jagged_item_logit = self._dense_module(hstu_output.values)
            
        return jagged_item_logit