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
import os
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from commons.datasets.hstu_batch import HSTUBatch
from configs import (
    InferenceHSTUConfig,
    KVCacheConfig,
    KVCacheMetadata,
    RankingConfig,
    copy_kvcache_metadata,
    get_kvcache_metadata_buffer
)
from modules.hstu_block_inference import HSTUBlockInference
from modules.jagged_data import JaggedData
from modules.mlp import MLP
from ops.triton_ops.triton_jagged import triton_concat_2D_jagged
from torchrec.sparse.jagged_tensor import JaggedTensor
import paged_kvcache_ops


def get_jagged_metadata_buffer(max_batch_size, max_seq_len, contextual_max_seqlen):
    int_dtype = torch.int32
    device = torch.cuda.current_device()
    default_num_candidates = max_seq_len // 2
    return JaggedData(
        values=None,
        # hidden states
        max_seqlen=max_seq_len,
        seqlen=torch.full(
            (max_batch_size,), max_seq_len, dtype=int_dtype, device=device
        ),
        seqlen_offsets=torch.arange(
            end=max_batch_size + 1, dtype=int_dtype, device=device
        )
        * max_seq_len,
        # candidates (included in hidden states)
        max_num_candidates=default_num_candidates,
        num_candidates=torch.full(
            (max_batch_size,), default_num_candidates, dtype=int_dtype, device=device
        ),
        num_candidates_offsets=torch.arange(
            end=max_batch_size + 1, dtype=int_dtype, device=device
        )
        * default_num_candidates,
        # contextual features
        contextual_max_seqlen=contextual_max_seqlen,
        contextual_seqlen=torch.full(
            (max_batch_size,), 0, dtype=int_dtype, device=device
        )
        if contextual_max_seqlen > 0
        else None,
        contextual_seqlen_offsets=torch.full(
            (max_batch_size + 1,), 0, dtype=int_dtype, device=device
        )
        if contextual_max_seqlen > 0
        else None,
        has_interleaved_action=True,
        scaling_seqlen=-1,
    )


def copy_jagged_metadata(dst_metadata, src_metata):
    def copy_tensor(dst, src):
        dst[: src.shape[0], ...].copy_(src, non_blocking=True)
        dst[src.shape[0] :, ...] = 0

    def copy_offsets(dst, src):
        dst[: src.shape[0], ...].copy_(src, non_blocking=True)
        dst[src.shape[0] :, ...] = src[-1, ...]

    bs = src_metata.seqlen.shape[0]
    dst_metadata.max_seqlen = src_metata.max_seqlen
    copy_tensor(dst_metadata.seqlen, src_metata.seqlen[:bs])
    copy_offsets(dst_metadata.seqlen_offsets, src_metata.seqlen_offsets[: bs + 1])
    dst_metadata.max_num_candidates = src_metata.max_num_candidates
    copy_tensor(dst_metadata.num_candidates, src_metata.num_candidates[:bs])
    copy_offsets(
        dst_metadata.num_candidates_offsets, src_metata.num_candidates_offsets[: bs + 1]
    )
    dst_metadata.contextual_max_seqlen = src_metata.contextual_max_seqlen
    if src_metata.contextual_max_seqlen > 0:
        copy_tensor(dst_metadata.contextual_seqlen, src_metata.contextual_seqlen[:bs])
        copy_offsets(
            dst_metadata.contextual_seqlen_offsets,
            src_metata.contextual_seqlen_offsets[: bs + 1],
        )
    dst_metadata.scaling_seqlen = src_metata.scaling_seqlen


class InferenceDenseModule(torch.nn.Module):
    """
    A class representing the ranking model inference.

    Args:
        hstu_config (InferenceHSTUConfig): The HSTU configuration.
        task_config (RankingConfig): The ranking task configuration.
    """

    def __init__(
        self,
        hstu_config: InferenceHSTUConfig,
        kvcache_config: KVCacheConfig,
        task_config: RankingConfig,
        use_cudagraph=False,
        cudagraph_configs=None,
    ):
        super().__init__()
        self._device = torch.cuda.current_device()
        self._hstu_config = hstu_config
        self._task_config = task_config

        self._embedding_dim = hstu_config.hidden_size
        for ebc_config in task_config.embedding_configs:
            assert (
                ebc_config.dim == self._embedding_dim
            ), "hstu layer hidden size should equal to embedding dim"

        self._hstu_block = HSTUBlockInference(hstu_config, kvcache_config)
        self._mlp = MLP(
            self._embedding_dim,
            task_config.prediction_head_arch,
            task_config.prediction_head_act_type,
            task_config.prediction_head_bias,
            device=self._device,
        )

        self._hstu_block = self._hstu_block.cuda()
        self._mlp = self._mlp.cuda()

        dtype = (
            torch.bfloat16
            if hstu_config.bf16
            else torch.float16
            if hstu_config.fp16
            else torch.float32
        )

        max_batch_size = hstu_config.max_batch_size
        max_seq_len = hstu_config.max_seq_len
        hidden_dim = hstu_config.hidden_size

        from commons.ops.triton_ops.common import (
            set_static_max_seq_lens,
            set_use_runtime_max_seq_len,
        )
        set_use_runtime_max_seq_len(False)
        set_static_max_seq_lens(max_seq_len, max_seq_len)

        self._hidden_states = torch.randn(
            (max_batch_size * max_seq_len, hidden_dim), dtype=dtype, device=self._device
        )
        self._jagged_metadata = get_jagged_metadata_buffer(
            max_batch_size, max_seq_len, hstu_config.contextual_max_seqlen
        )

        self._use_kvcache = False
        if kvcache_config is not None:
            self._use_kvcache = True
            from modules.async_kvcache_manager import AsyncHSTUKVCacheManager

            if kvcache_config.max_queued_offload_tokens is None:
                kvcache_config.max_queued_offload_tokens = 4 * hstu_config.max_batch_size * hstu_config.max_seq_len
            self.async_kvcache = AsyncHSTUKVCacheManager(
                hstu_config.num_layers,
                hstu_config.num_heads,
                hstu_config.head_dim,
                kvcache_config.page_size,
                kvcache_config.blocks_in_primary_pool,
                math.ceil(hstu_config.max_batch_size * hstu_config.max_seq_len / kvcache_config.page_size),
                0,
                kvcache_config.offload_chunksize,
                -1,
                hstu_config.max_seq_len,
                hstu_config.max_batch_size,
                kvcache_config.max_queued_offload_tokens,
                kvcache_config.num_onload_buffer_chunks,
                kvcache_config.num_offload_buffer_chunks,
                kvcache_config.num_memcpy_workers,
                kvcache_config.enable_nvcomp,
            )

        # TODO(junyiq): Add cudagraph optimization for the MLP as well.
        self.use_cudagraph = use_cudagraph
        if use_cudagraph:
            self._kvcache_metadata = None
            if self._use_kvcache:
                self._kvcache_metadata = get_kvcache_metadata_buffer(
                    hstu_config, kvcache_config
                )
                self._kvcache_metadata.kv_cache_table = self.async_kvcache.cache_table_list
                self._kvcache_metadata.kv_onload_handle = paged_kvcache_ops.KVOnloadHandle()
                self._kvcache_metadata.kv_offload_handle = paged_kvcache_ops.KVOffloadHandle()
            self._hstu_block.set_cudagraph(
                max_batch_size,
                max_seq_len,
                self._hidden_states,
                self._jagged_metadata,
                self._kvcache_metadata,
                cudagraph_configs=cudagraph_configs,
            )

    def bfloat16(self):
        """
        Convert the model to use bfloat16 precision. Only affects the dense module.

        Returns:
            RankingGR: The model with bfloat16 precision.
        """
        self._hstu_block.bfloat16()
        self._mlp.bfloat16()
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the dense module.

        Returns:
            RankingGR: The model with half precision.
        """
        self._hstu_block.half()
        self._mlp.half()
        return self

    def get_num_class(self):
        return self._task_config.prediction_head_arch[-1]

    def get_num_tasks(self):
        return self._task_config.num_tasks

    def get_metric_types(self):
        return self._task_config.eval_metrics

    def load_checkpoint(self, checkpoint_dir):
        if checkpoint_dir is None:
            return

        model_state_dict_path = os.path.join(
            checkpoint_dir, "torch_module", "model.0.pth"
        )
        model_state_dict = torch.load(model_state_dict_path)["model_state_dict"]
        self.load_state_dict(model_state_dict, strict=False)

    def load_state_dict(self, model_state_dict, *args, **kwargs):
        new_state_dict = {}
        for k in model_state_dict:
            if (
                k.startswith(
                    "_embedding_collection._data_parallel_embedding_collection.embeddings."
                )
                or "_model_parallel_embedding_collection" in k
            ):
                continue

            is_transposed = False
            if k.endswith("_linear_uvqk_weight"):
                newk = k.removesuffix("_linear_uvqk_weight") + "_linear_uvqk.weight"
                is_transposed = True
            elif k.endswith("_linear_uvqk_bias"):
                newk = k.removesuffix("_linear_uvqk_bias") + "_linear_uvqk.bias"
            elif k.endswith("_linear_proj_weight"):
                newk = k.removesuffix("_linear_proj_weight") + "_linear_proj.weight"
                is_transposed = True
            else:
                newk = k
            new_state_dict[newk] = (
                model_state_dict[k] if not is_transposed else model_state_dict[k].T
            )

        unloaded_modules = super().load_state_dict(new_state_dict, *args, **kwargs)
        for hstu_layer in self._hstu_block._attention_layers:
            hstu_layer._linear_uvqk_weight.copy_(hstu_layer._linear_uvqk.weight.T)
            hstu_layer._linear_proj_weight.copy_(hstu_layer._linear_proj.weight.T)

        assert unloaded_modules.missing_keys == []
        assert unloaded_modules.unexpected_keys == []

    def forward(
        self,
        batch: HSTUBatch,
        embeddings: Dict[str, JaggedTensor],
        user_ids: torch.Tensor,
        total_history_lengths: torch.Tensor,
        prepare_kvcache_result: List,
    ):
        with torch.inference_mode():
            (
                old_cached_lengths,
                num_history_tokens,
                offload_uids_buffer,
                metadata_host_buffer,
                metadata_gpu_buffer, # returned static
                kvcache_metadata_fut,
                onload_fut,
            ) = prepare_kvcache_result

            jagged_data = self._hstu_block._preprocessor(
                embeddings=embeddings,
                batch=batch,
                seq_start_position=old_cached_lengths.cuda(),
            )

            kvcache_metadata = self.async_kvcache.prepare_kvcache_wait(
                onload_fut,
                kvcache_metadata_fut,
                batch.batch_size,
                num_history_tokens,
                self.async_kvcache.static_page_ids_gpu_buffer,
                self.async_kvcache.static_offload_page_ids_gpu_buffer,
                offload_uids_buffer,
                metadata_host_buffer,
                metadata_gpu_buffer, # returned static
                self.async_kvcache.static_onload_handle,
            )
            self.async_kvcache.offload_kvcache(kvcache_metadata)
            kvcache_metadata.total_history_offsets += jagged_data.num_candidates_offsets
            kvcache_metadata.total_history_lengths += jagged_data.num_candidates
            kvcache_metadata.max_seqlen += jagged_data.max_num_candidates

            num_tokens = batch.features.values().shape[0]
            if self.use_cudagraph:
                self._hidden_states[:num_tokens, ...].copy_(
                    jagged_data.values, non_blocking=True
                )
                copy_jagged_metadata(self._jagged_metadata, jagged_data)
                copy_kvcache_metadata(self._kvcache_metadata, kvcache_metadata)

                hstu_output = self._hstu_block.predict(
                    batch.batch_size,
                    num_tokens,
                    self._hidden_states,
                    self._jagged_metadata,
                    kvcache_metadata,
                )
                jagged_data.values = hstu_output
            else:
                hstu_output = self._hstu_block.predict(
                    batch.batch_size,
                    num_tokens,
                    jagged_data.values,
                    jagged_data,
                    kvcache_metadata,
                )
                jagged_data.values = hstu_output

            jagged_data = self._hstu_block._postprocessor(jagged_data)
            jagged_item_logit = self._mlp(jagged_data.values)

        return jagged_item_logit
    
    def forward_nokvcache(
        self,
        batch: HSTUBatch,
        embeddings: Dict[str, JaggedTensor],
    ):
        with torch.inference_mode():
            jagged_data = self._hstu_block._preprocessor(
                embeddings=embeddings,
                batch=batch,
            )

            num_tokens = batch.features.values().shape[0]
            hstu_output = self._hstu_block.predict(
                batch.batch_size,
                num_tokens,
                jagged_data.values,
                jagged_data,
                None,
            )
            jagged_data.values = hstu_output
            jagged_data = self._hstu_block._postprocessor(jagged_data)
            jagged_item_logit = self._mlp(jagged_data.values)

        return jagged_item_logit


def get_inference_dense_model(
    hstu_config: InferenceHSTUConfig,
    kvcache_config: KVCacheConfig,
    task_config: RankingConfig,
    use_cudagraph=False,
    cudagraph_configs=None,
):
    return InferenceDenseModule(
        hstu_config,
        kvcache_config,
        task_config,
        use_cudagraph,
        cudagraph_configs,
    )
