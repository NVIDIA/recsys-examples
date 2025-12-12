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
from typing import Dict, List, Optional, Tuple, Union

import torch
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
from modules.jagged_data import JaggedData
from modules.mlp import MLP
from ops.triton_ops.triton_jagged import triton_concat_2D_jagged
from torchrec.sparse.jagged_tensor import JaggedTensor


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

        self._use_kvcache = False
        if kvcache_config is not None:
            from modules.gpu_kv_cache_manager import HSTUGpuKVCacheManager
            from modules.host_kv_storage_manager import HSTUHostKVStorageManager

            self._use_kvcache = True
            self._gpu_kv_cache_manager = HSTUGpuKVCacheManager(
                hstu_config, kvcache_config
            )
            self._host_kv_storage_manager = HSTUHostKVStorageManager(
                hstu_config, kvcache_config
            )

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

        max_batch_size = hstu_config.max_batch_size
        max_seq_len = hstu_config.max_seq_len
        hidden_dim = hstu_config.hidden_size

        self._hidden_states = torch.randn(
            (max_batch_size * max_seq_len, hidden_dim), dtype=dtype, device=device
        )
        self._jagged_metadata = get_jagged_metadata_buffer(
            max_batch_size, max_seq_len, hstu_config.contextual_max_seqlen
        )

        if self._use_kvcache:
            self._kvcache_metadata = get_kvcache_metadata_buffer(
                hstu_config, kvcache_config
            )
            self._offload_states = None
            self._kvcache_metadata.onload_history_kv_buffer = [
                self._gpu_kv_cache_manager.get_onload_buffers(layer_idx)
                for layer_idx in range(hstu_config.num_layers)
            ]
            self._kvcache_metadata.onload_history_kv_events = [
                torch.cuda.Event() for _ in range(hstu_config.num_layers)
            ]
            self._kvcache_metadata.kv_cache_table = [
                self._gpu_kv_cache_manager.get_kvcache_table(layer_idx)
                for layer_idx in range(hstu_config.num_layers)
            ]

        # TODO(junyiq): Add cudagraph optimization for the MLP as well.
        self.use_cudagraph = use_cudagraph
        if use_cudagraph:
            self._hstu_block.set_cudagraph(
                max_batch_size,
                max_seq_len,
                self._hidden_states,
                self._jagged_metadata,
                self._kvcache_metadata if self._use_kvcache else None,
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

    def get_user_kvdata_info(
        self,
        user_ids: Union[List[int], torch.Tensor],
        allow_bubble: bool = False,
        dbg_print: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kvdata_start_pos = list()
        kvdata_lengths = list()
        for idx in range(len(user_ids)):
            uid = int(user_ids[idx])
            host_sp, host_len = self._host_kv_storage_manager.get_user_kvdata_info(uid)
            gpu_sp, gpu_len = self._gpu_kv_cache_manager.get_user_kvdata_info(uid)
            sp = host_sp if gpu_sp == -1 or gpu_len == 0 else min(host_sp, gpu_sp)
            length = (
                host_len
                if gpu_sp == -1 or gpu_len == 0
                else (gpu_sp + gpu_len - host_sp)
            )
            if gpu_sp > host_sp + host_len and not allow_bubble:
                warnings.warn(
                    f"KVdata missing between host storage and gpu kvcache for user {uid}"
                )
                length = host_len
            kvdata_start_pos.append(sp)
            kvdata_lengths.append(length)
        return (
            torch.tensor(kvdata_start_pos, dtype=torch.int32),
            torch.tensor(kvdata_lengths, dtype=torch.int32),
        )

    def prepare_kv_cache(
        self, batch: Batch, user_ids: torch.Tensor, user_start_pos: torch.Tensor
    ) -> KVCacheMetadata:
        batch_size = user_ids.shape[0]
        new_history_lengths = (
            torch.sum(batch.features.lengths().view(-1, batch_size), 0).view(-1)
            - batch.num_candidates
        )
        (
            cached_start_pos,
            cached_lengths,
        ) = self._gpu_kv_cache_manager.get_batch_kvdata_info(user_ids)

        self._gpu_kv_cache_manager.allocate(
            user_ids, user_start_pos, new_history_lengths
        )
        kv_cache_metadata = self._gpu_kv_cache_manager.get_cache_metadata(user_ids)
        append_metadata = self._gpu_kv_cache_manager.get_append_metadata(
            new_history_lengths, kv_cache_metadata.total_history_lengths
        )
        for _field_name in [
            "batch_indices",
            "position",
            "new_history_nnz",
            "new_history_nnz_cuda",
        ]:
            setattr(
                kv_cache_metadata, _field_name, getattr(append_metadata, _field_name)
            )

        kv_cache_metadata.onload_history_kv_buffer = (
            self._kvcache_metadata.onload_history_kv_buffer[:]
        )
        kv_cache_metadata.onload_history_kv_events = (
            self._kvcache_metadata.onload_history_kv_events[:]
        )
        kv_cache_metadata.kv_cache_table = self._kvcache_metadata.kv_cache_table[:]
        (
            onload_length,
            onload_kv_page_ids,
            onload_kv_page_indptr,
        ) = self._host_kv_storage_manager.lookup_kvdata(
            user_ids, cached_start_pos, cached_lengths
        )
        if onload_length > 0:
            kv_page_ids = triton_concat_2D_jagged(
                max_seq_len=onload_kv_page_indptr[-1] + kv_cache_metadata.kv_indptr[-1],
                values_a=onload_kv_page_ids.view(-1, 1),
                values_b=kv_cache_metadata.kv_indices.view(-1, 1),
                offsets_a=onload_kv_page_indptr.to(torch.int64),
                offsets_b=kv_cache_metadata.kv_indptr.to(torch.int64),
            )
            kv_cache_metadata.kv_indices = kv_page_ids.view(-1)
            kv_cache_metadata.kv_indptr = (
                onload_kv_page_indptr + kv_cache_metadata.kv_indptr
            )
            self._gpu_kv_cache_manager.onload(
                self._host_kv_storage_manager.get_lookup_buffer(),
                onload_length,
                self._kvcache_metadata if self.use_cudagraph else kv_cache_metadata,
            )

        # cudagraph preparation
        if self.use_cudagraph:
            copy_kvcache_metadata(self._kvcache_metadata, kv_cache_metadata)
        # assert max(kv_cache_metadata.kv_indices.tolist()) < self._kvcache_metadata.kv_cache_table[0].shape[0]

        return kv_cache_metadata

    def finalize_kv_cache(self, user_ids: torch.Tensor, **kwargs):
        pass

    def clear_kv_cache(self):
        self._gpu_kv_cache_manager.evict_all()
        self._host_kv_storage_manager.evict_all_kvdata()

    def offload_kv_cache(
        self, user_ids: torch.Tensor, kvcache_metadata: KVCacheMetadata
    ):
        offload_results = self.offload_kv_cache_async(user_ids, kvcache_metadata)
        if offload_results is not None:
            self.offload_kv_cache_wait(offload_results)

    def offload_kv_cache_async(
        self, user_ids: torch.Tensor, kvcache_metadata: KVCacheMetadata
    ):
        host_kvdata_start_pos, host_kvdata_lengths = zip(
            *[
                self._host_kv_storage_manager.get_user_kvdata_info(int(user_ids[idx]))
                for idx in range(len(user_ids))
            ]
        )
        host_kvdata_start_pos = torch.tensor(host_kvdata_start_pos, dtype=torch.int32)
        host_kvdata_lengths = torch.tensor(host_kvdata_lengths, dtype=torch.int32)

        offload_results = self._gpu_kv_cache_manager.offload_async(
            user_ids, host_kvdata_start_pos, host_kvdata_lengths, kvcache_metadata
        )
        return offload_results

    def offload_kv_cache_wait(
        self,
        offload_results: Optional[
            Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ):
        if offload_results is not None:
            self._gpu_kv_cache_manager.offload_wait()
            self._host_kv_storage_manager.append_kvdata(*offload_results)

    def kvcache_prelogue(
        self,
        batch,
        user_ids: torch.Tensor = None,
        user_start_pos: torch.Tensor = None,
    ):
        user_start_pos_cuda = None
        kvcache_metadata = None
        if self._use_kvcache:
            assert user_ids is not None and user_start_pos is not None
            user_start_pos_cuda = user_start_pos.to(
                device=torch.cuda.current_device(), non_blocking=True
            )
            kvcache_metadata = self.prepare_kv_cache(batch, user_ids, user_start_pos)

        return user_start_pos_cuda, kvcache_metadata

    def forward(
        self,
        batch: Batch,
        embeddings: Dict[str, JaggedTensor],
        user_ids: torch.Tensor = None,
        user_start_pos_cuda: torch.Tensor = None,
        kvcache_metadata=None,
    ):
        with torch.inference_mode():
            if self._use_kvcache:
                assert user_start_pos_cuda is not None
                assert kvcache_metadata is not None

            jagged_data = self._hstu_block._preprocessor(
                embeddings=embeddings,
                batch=batch,
                seq_start_position=user_start_pos_cuda if self._use_kvcache else None,
            )

            num_tokens = batch.features.values().shape[0]
            if self.use_cudagraph:
                self._hidden_states[:num_tokens, ...].copy_(
                    jagged_data.values, non_blocking=True
                )
                copy_jagged_metadata(self._jagged_metadata, jagged_data)
                if self._use_kvcache:
                    copy_kvcache_metadata(self._kvcache_metadata, kvcache_metadata)
                    self._kvcache_metadata.total_history_offsets += (
                        self._jagged_metadata.num_candidates_offsets
                    )

                hstu_output = self._hstu_block.predict(
                    batch.batch_size,
                    num_tokens,
                    self._hidden_states,
                    self._jagged_metadata,
                    self._kvcache_metadata if self._use_kvcache else None,
                )
                jagged_data.values = hstu_output
            else:
                if self._use_kvcache:
                    kvcache_metadata.total_history_offsets += (
                        jagged_data.num_candidates_offsets
                    )

                hstu_output = self._hstu_block.predict(
                    batch.batch_size,
                    num_tokens,
                    jagged_data.values,
                    jagged_data,
                    kvcache_metadata if self._use_kvcache else None,
                )
                jagged_data.values = hstu_output

            if self._use_kvcache:
                self._gpu_kv_cache_manager._offload_start_event.record(
                    torch.cuda.current_stream()
                )

            jagged_data = self._hstu_block._postprocessor(jagged_data)
            jagged_item_logit = self._mlp(jagged_data.values)

            if self._use_kvcache:
                self._offload_states = self.offload_kv_cache_async(
                    user_ids, kvcache_metadata
                )
                self.offload_kv_cache_wait(self._offload_states)
                self.finalize_kv_cache(user_ids)

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
