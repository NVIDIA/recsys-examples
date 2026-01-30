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
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from configs import (
    InferenceHSTUConfig,
    KVCacheConfig,
    KVCacheMetadata,
    RankingConfig,
    copy_kvcache_metadata,
    get_kvcache_metadata_buffer,
)
from dataset.utils import Batch
from modules.hstu_block_inference import HSTUBlockInference
from modules.inference_embedding import InferenceEmbedding
from modules.jagged_data import JaggedData
from modules.mlp import MLP
from ops.triton_ops.triton_jagged import triton_concat_2D_jagged
from modules.async_kvcache_manager import AsyncHSTUKVCacheManager
import math

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


class InferenceRankingGR(torch.nn.Module):
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

        self._embedding_collection = InferenceEmbedding(task_config.embedding_configs)

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
        device = torch.cuda.current_device()

        max_batch_size = kvcache_config.max_batch_size
        max_seq_len = kvcache_config.max_seq_len
        hidden_dim = hstu_config.hidden_size

        self._hidden_states = torch.randn(
            (max_batch_size * max_seq_len, hidden_dim), dtype=dtype, device=device
        )
        self._jagged_metadata = get_jagged_metadata_buffer(
            max_batch_size, max_seq_len, hstu_config.contextual_max_seqlen
        )

        if kvcache_config.max_queued_offload_tokens is None:
            kvcache_config.max_queued_offload_tokens = 4 * kvcache_config.max_batch_size * kvcache_config.max_seq_len
        self.async_kvcache = AsyncHSTUKVCacheManager(
            hstu_config.num_layers,
            hstu_config.num_heads,
            hstu_config.head_dim,
            kvcache_config.page_size,
            kvcache_config.blocks_in_primary_pool,
            math.ceil(kvcache_config.max_batch_size * kvcache_config.max_seq_len / kvcache_config.page_size),
            0,
            kvcache_config.offload_chunksize,
            -1,
            kvcache_config.max_seq_len,
            kvcache_config.max_batch_size,
            kvcache_config.max_queued_offload_tokens,
            kvcache_config.num_onload_buffer_chunks,
            kvcache_config.num_offload_buffer_chunks,
            kvcache_config.num_memcpy_workers,
            kvcache_config.enable_nvcomp,
        )

        from ops.triton_ops.common import set_use_runtime_max_seq_len, set_static_max_seq_lens
        set_use_runtime_max_seq_len(False)
        set_static_max_seq_lens(max_seq_len, max_seq_len)
        # from ops.triton_ops.triton_position import triton_add_position_embeddings
        # triton_add_position_embeddings(
        #     jagged=self._hidden_states,
        #     jagged_offsets=torch.tensor[], #seq_offsets,
        #     high_inds=high_inds,
        #     max_seq_len=max_seq_len,
        #     dense=self._position_embeddings_weight,
        #     scale=alpha,
        #     ind_offsets=ind_offsets)

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

    def load_checkpoint(self, checkpoint_dir):
        embedding_table_dir = os.path.join(
            checkpoint_dir,
            "dynamicemb_module",
            "model._embedding_collection._model_parallel_embedding_collection",
        )
        dynamic_tables = (
            self._embedding_collection._dynamic_embedding_collection._embedding_tables
        )

        try:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "0000"
            dist.init_process_group(world_size=1, rank=0)
            for idx, table_name in enumerate(dynamic_tables.table_names):
                dynamic_tables.load(
                    embedding_table_dir, optim=False, table_names=[table_name]
                )
            dist.destroy_process_group()
            os.environ.pop("MASTER_ADDR")
            os.environ.pop("MASTER_PORT")
        except ValueError as e:
            warnings.warn(
                f"FAILED TO LOAD dynamic embedding tables failed due to ValueError:\n\t{e}\n\n"
                "Please check if the checkpoint is version 1. The loading of this old version is disabled."
            )

        model_state_dict_path = os.path.join(
            checkpoint_dir, "torch_module", "model.0.pth"
        )
        model_state_dict = torch.load(model_state_dict_path)["model_state_dict"]
        self.load_state_dict(model_state_dict, strict=False)

    def load_state_dict(self, model_state_dict, *args, **kwargs):
        new_state_dict = {}
        for k in model_state_dict:
            if k.startswith(
                "_embedding_collection._data_parallel_embedding_collection.embeddings."
            ):
                emb_table_names = k.split(".")[-1].removesuffix("_weights").split("/")
                old_emb_table_weights = model_state_dict[k].view(
                    -1, self._embedding_dim
                )
                weight_offset = 0
                # TODO(junyiq): Use a more flexible way to skip contextual features.
                for name in emb_table_names:
                    for emb_config in self._task_config.embedding_configs:
                        if name == emb_config.table_name:
                            emb_table_size = emb_config.vocab_size
                            if self._hstu_config.contextual_max_seqlen == 0:
                                weight_offset = (
                                    old_emb_table_weights.shape[0] - emb_table_size
                                )
                            break
                    else:
                        if self._hstu_config.contextual_max_seqlen == 0:
                            print(
                                f"No embedding config found for {name}. Skipped as disabled contextual features."
                            )
                            continue
                        raise Exception("No embedding config found for " + name)
                    newk = (
                        "_embedding_collection._static_embedding_collection.embeddings."
                        + name
                        + ".weight"
                    )
                    new_state_dict[newk] = old_emb_table_weights[
                        weight_offset : weight_offset + emb_table_size
                    ]
                    weight_offset += emb_table_size
                continue
            elif "_model_parallel_embedding_collection" in k:
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

        assert unloaded_modules.missing_keys == [
            "_embedding_collection._dynamic_embedding_collection._embedding_tables._empty_tensor"
        ]
        if self._hstu_config.contextual_max_seqlen != 0:
            assert unloaded_modules.unexpected_keys == []

    def clear_kv_cache(self):
        self._gpu_kv_cache_manager.evict_all()
        self._host_kv_storage_manager.evict_all_kvdata()

    def forward(
        self,
        batch: Batch,
        user_ids: torch.Tensor,
        total_history_lengths: torch.Tensor,
    ):
        with torch.inference_mode():
            user_ids_list = user_ids.tolist()

            prepare_kvcache_result = self.async_kvcache.prepare_kvcache_async(
                batch.batch_size,
                user_ids_list,
                total_history_lengths.tolist(),
                self.async_kvcache.static_page_ids_gpu_buffer,
                self.async_kvcache.static_offload_page_ids_gpu_buffer,
                self.async_kvcache.static_onload_handle,
            )

            (
                old_cached_lengths,
                num_history_tokens,
                offload_uids_buffer,
                metadata_host_buffer,
                metadata_gpu_buffer,
                kvcache_metadata_fut,
                onload_fut,
            ) = prepare_kvcache_result
            old_cached_lengths = torch.tensor(old_cached_lengths, dtype=torch.int32)

            striped_batch = self.async_kvcache.strip_cached_tokens(
                batch, old_cached_lengths,
            )

            torch.cuda.nvtx.range_push("HSTU embedding")
            embeddings = self._embedding_collection(striped_batch.features)
            torch.cuda.nvtx.range_pop()

            jagged_data = self._hstu_block._preprocessor(
                embeddings=embeddings,
                batch=striped_batch,
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
                metadata_gpu_buffer,
                self.async_kvcache.static_onload_handle,
            )
            self.async_kvcache.offload_kvcache(kvcache_metadata)

            kvcache_metadata.total_history_offsets += jagged_data.num_candidates_offsets
            kvcache_metadata.total_history_lengths += jagged_data.num_candidates
            kvcache_metadata.max_seqlen += jagged_data.max_num_candidates

            num_tokens = striped_batch.features.values().shape[0]
            hstu_output = self._hstu_block.predict(
                striped_batch.batch_size,
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
        batch: Batch,
    ):
        with torch.inference_mode():

            embeddings = self._embedding_collection(batch.features)
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

