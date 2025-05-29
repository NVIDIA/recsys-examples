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
from configs import HSTUConfig, RankingConfig, KVCacheConfig, KVCacheMetadata
from dataset.utils import RankingBatch
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.transformer.module import Float16Module
from model.base_model import BaseModel
from modules.embedding import ShardedEmbedding
from modules.gpu_kv_cache_manager import HSTUGpuKVCacheManager
from modules.hstu_block import HSTUBlock
from modules.jagged_module import JaggedData
from modules.metrics import get_multi_event_metric_module
from modules.mlp import MLP
from modules.multi_task_loss_module import MultiTaskLossModule
from modules.multi_task_over_arch import MultiTaskOverArch


class RankingGRInfer(torch.nn.Module):
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


        self._gpu_kv_cache_manager = HSTUGpuKVCacheManager(
            hstu_config, kvcache_config)
        # self._host_kv_storage_manager = HSTUHostKvStorageManager(host_kv_storage_impl)

        self._hstu_block = HSTUBlock(hstu_config, inference_mode=True)
        self._dense_module = torch.nn.Sequential(
            MultiTaskOverArch(
                [
                    MLP(
                        self._embedding_dim,
                        layer_sizes,
                        has_bias,
                        head_act_type,
                        device=self._device,
                    )
                    for layer_sizes, head_act_type, has_bias in zip(
                        task_config.prediction_head_arch,
                        task_config.prediction_head_act_type,
                        task_config.prediction_head_bias,  # type: ignore[arg-type]
                    )
                ]
            ),
        )

        self._hstu_block = self._hstu_block.cuda()
        self._dense_module = self._dense_module.cuda()
        # TODO, add ddp optimizer flag
        if hstu_config.bf16 or hstu_config.fp16:
            self._dense_module = Float16Module(hstu_config, self._dense_module)

    def bfloat16(self):
        """
        Convert the model to use bfloat16 precision. Only affects the dense module.

        Returns:
            RankingGR: The model with bfloat16 precision.
        """
        self._dense_module.bfloat16()
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the dense module.

        Returns:
            RankingGR: The model with half precision.
        """
        self._dense_module.half()
        return self
    
    def lookup_cached_lengths(self, user_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (gpu_cached_lengths, gpu_offloaded_legnths) = self._gpu_kv_cache_manager.lookup(user_ids)
        host_lengths = self._host_kv_storage_manger.lookup(user_ids)
        return (gpu_cached_lengths, gpu_offloaded_legnths, host_lengths)
    
    def prepare_kv_cache(self, jd: JaggedData, user_ids: torch.Tensor) -> Tuple[torch.Tensor, KVCacheMetadata]:
        self._gpu_kv_cache_manager.allocate(jd, user_ids)
        kv_cache_metadata = self._gpu_kv_cache_manager.get_cache_page_metadata(jd, user_ids)
        
        # host_kv = self._host_kv_storage_manger.lookup(batch.user_ids)
        # #^ cpu jagged tensor pinned memory
        # host_kv_page_ids, host_kv_page_indptr, events_list = self._gpu_kv_cache_manager.onboard(host_kv_cpu)
        # #^ gpu tensor,        ^ gpu tensor,            
        
        # kv_cache_page_ids = triton_concat_2D_jagged(host_kv_page_ids, kv_cache_page_ids, host_kv_page_indptr, kv_cache_page_indptr)
        # kv_cache_page_indptr = host_kv_page_indptr + kv_cache_page_indptr
        
        return (self._gpu_kv_cache_manager, kv_cache_metadata)
    
    def finalize_kv_cache(self, user_ids: torch.Tensor):
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
            kv_cache_handler = self.prepare_kv_cache(jagged_data, user_ids)
            hstu_output = self._hstu_block.predict(jagged_data, kv_cache_handler)
            self.finalize_kv_cache(batch)
            jagged_item_logit = self._dense_module(hstu_output).values

        return jagged_item_logit
